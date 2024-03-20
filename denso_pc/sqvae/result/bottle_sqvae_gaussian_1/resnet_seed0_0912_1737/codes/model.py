import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import FrEIA.framework as Ff
import FrEIA.modules as Fm

from quantizer import GaussianVectorQuantizer, VmfVectorQuantizer
import networks.mnist as net_mnist
import networks.fashion_mnist as net_fashionmnist
import networks.cifar10 as net_cifar10
import networks.celeba as net_celeba
import networks.celebamask_hq as net_celebamask_hq
import networks.mvtecad as net_mvtecad
import constants as const


def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding=kernel_size//2),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)



class SQVAE(nn.Module):
    def __init__(self, cfgs, flgs):
        super(SQVAE, self).__init__()
        # Data space
        dataset = cfgs.dataset.name
        self.dim_x = cfgs.dataset.dim_x
        self.dataset = cfgs.dataset.name

        # Encoder/decoder
        self.param_var_q = cfgs.model.param_var_q
        self.encoder = eval("net_{}.EncoderVq_{}".format(dataset.lower(), cfgs.network.name))(
            cfgs.quantization.dim_dict, cfgs.network, flgs.bn, flgs.var_q)
        self.decoder = eval("net_{}.DecoderVq_{}".format(dataset.lower(), cfgs.network.name))(
            cfgs.quantization.dim_dict, cfgs.network, flgs.bn)
        self.apply(weights_init)

        # Codebook
        self.size_dict = cfgs.quantization.size_dict
        self.dim_dict = cfgs.quantization.dim_dict
        self.log_param_q_scalar = nn.Parameter(torch.tensor(cfgs.model.log_param_q_init))
        if self.param_var_q == "vmf":
            self.quantizer = VmfVectorQuantizer(
                self.size_dict, self.dim_dict, cfgs.quantization.temperature.init)
        else:
            self.quantizer = GaussianVectorQuantizer(
                self.size_dict, self.dim_dict, cfgs.quantization.temperature.init, self.param_var_q)
        
    
    def forward(self, x, flg_train=False, flg_quant_det=True):
        # Encoding
        if self.param_var_q == "vmf":
            z_from_encoder = F.normalize(self.encoder(x), p=2.0, dim=1)
            self.param_q = (self.log_param_q_scalar.exp() + torch.tensor([1.0], device="cuda"))
        else:
            if self.param_var_q == "gaussian_1":
                z_from_encoder = self.encoder(x)
                log_var_q = torch.tensor([0.0], device="cuda")
            else:
                z_from_encoder, log_var = self.encoder(x)
                if self.param_var_q == "gaussian_2":
                    log_var_q = log_var.mean(dim=(1,2,3), keepdim=True)
                elif self.param_var_q == "gaussian_3":
                    log_var_q = log_var.mean(dim=1, keepdim=True)
                elif self.param_var_q == "gaussian_4":
                    log_var_q = log_var
                else:
                    raise Exception("Undefined param_var_q")
            self.param_q = (log_var_q.exp() + self.log_param_q_scalar.exp())
        
        # Quantization
        z_quantized, loss_latent, perplexity = self.quantizer(z_from_encoder, self.param_q, flg_train, flg_quant_det)
        latents = dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized)

        # Decoding
        x_reconst = self.decoder(z_quantized)

        # Loss
        loss = self._calc_loss(x_reconst, x, loss_latent)
        loss["perplexity"] = perplexity
        
        return x_reconst, latents, loss
    
    def _calc_loss(self):
        raise NotImplementedError()
    

class GaussianSQVAE(SQVAE):
    def __init__(self, cfgs, flgs):
        super(GaussianSQVAE, self).__init__(cfgs, flgs)
        self.flg_arelbo = flgs.arelbo # Use MLE for optimization of decoder variance
        if not self.flg_arelbo:
            self.logvar_x = nn.Parameter(torch.tensor(np.log(0.1)))
    
    def _calc_loss(self, x_reconst, x, loss_latent):
        bs = x.shape[0]
        # Reconstruction loss
        mse = F.mse_loss(x_reconst, x, reduction="sum") / bs
        if self.flg_arelbo:
            # "Preventing Posterior Collapse Induced by Oversmoothing in Gaussian VAE"
            # https://arxiv.org/abs/2102.08663
            loss_reconst = self.dim_x * torch.log(mse) / 2
        else:
            loss_reconst = mse / (2*self.logvar_x.exp()) + self.dim_x * self.logvar_x / 2
        # Entire loss
        loss_all = loss_reconst + loss_latent
        loss = dict(all=loss_all, mse=mse)

        return loss 

class GaussianSQVAE_FastFlow(nn.Module):
    def __init__(self, cfgs, flgs):
        super(GaussianSQVAE_FastFlow, self).__init__()
        # Data space
        dataset = cfgs.dataset.name
        self.dim_x = cfgs.dataset.dim_x
        self.dataset = cfgs.dataset.name
        input_size = cfgs.dataset.shape[1:]

        # Encoder/decoder
        self.encoder = eval("net_{}.EncoderVq_{}".format(dataset.lower(), cfgs.network.name))(
            cfgs.quantization.dim_dict, cfgs.network, flgs.bn, flgs.var_q)
        self.decoder = eval("net_{}.DecoderVq_{}".format(dataset.lower(), cfgs.network.name))(
            cfgs.quantization.dim_dict, cfgs.network, flgs.bn)
        self.apply(weights_init)

        # Codebook
        self.size_dict = cfgs.quantization.size_dict
        self.dim_dict = cfgs.quantization.dim_dict
        self.log_param_q_scalar = nn.Parameter(torch.tensor(cfgs.model.log_param_q_init))
        self.quantizer = GaussianVectorQuantizer(self.size_dict, self.dim_dict, cfgs.quantization.temperature.init)
        
        channels = [cfgs.quantization.dim_dict]
        scales = [4]
        # for transformers, use their pretrained norm w/o grad
        # for resnets, self.norms are trainable LayerNorm
        self.norms = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.norms.append(
                nn.LayerNorm(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    elementwise_affine=True,
                )
            )

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=False,
                    hidden_ratio=1.0,
                    flow_steps=8,
                )
            )
        
    
    def forward(self, x, flg_train=False, flg_quant_det=True):
        # Encoding

        z_from_encoder = self.encoder(x)
        log_var_q = torch.tensor([0.0], device="cuda")
        self.param_q = (log_var_q.exp() + self.log_param_q_scalar.exp())
        
        #Normalizing Flow
        z_from_encoder = self.norms[0](z_from_encoder)
        z_from_encoder, log_jac_dets = self.nf_flows[0](z_from_encoder)
        jac_loss = -log_jac_dets.mean()


        # Quantization
        z_quantized, loss_latent, perplexity = self.quantizer(z_from_encoder, self.param_q, flg_train, flg_quant_det)
        latents = dict(z_from_encoder=z_from_encoder, z_to_decoder=z_quantized)

        # Decoding
        x_reconst = self.decoder(z_quantized)

        # Loss
        loss = self._calc_loss(x_reconst, x, loss_latent, jac_loss)
        loss["perplexity"] = perplexity
        loss["jac_loss"] = jac_loss
        
        return x_reconst, latents, loss
    
    def _calc_loss(self, x_reconst, x, loss_latent, jac_loss):
        bs = x.shape[0]
        # Reconstruction loss
        mse = F.mse_loss(x_reconst, x, reduction="sum") / bs
        # "Preventing Posterior Collapse Induced by Oversmoothing in Gaussian VAE"
        # https://arxiv.org/abs/2102.08663
        loss_reconst = self.dim_x * torch.log(mse) / 2

        # Entire loss
        loss_all = loss_reconst + loss_latent + jac_loss
        loss = dict(all=loss_all, mse=mse)

        return loss 



class VmfSQVAE(SQVAE):
    def __init__(self, cfgs, flgs):
        super(VmfSQVAE, self).__init__(cfgs, flgs)
        self.log_kappa_inv = nn.Parameter(torch.tensor([cfgs.model.log_kappa_inv]))
        self.__m = np.ceil(cfgs.network.num_class / 2)
        self.n_interval = cfgs.network.num_class - 1

    def _calc_loss(self, x_reconst, x, loss_latent):
        x_shape = x.shape
        # Reconstruction loss
        x = x.view(-1, 1)
        x_reconst_viewed = (x_reconst.permute(0, 2, 3, 1).contiguous()
                            .view(-1, int(self.__m * 2)) )
        x_reconst_normed = F.normalize(x_reconst_viewed, p=2.0, dim=-1)
        x_one_hot = (F.one_hot(x.to(torch.int).long(), num_classes = int(self.__m * 2))
                    .type_as(x))[:,0,:]
        x_reconst_selected = (x_one_hot * x_reconst_normed).sum(-1).view(x_shape)
        kappa_inv = self.log_kappa_inv.exp().add(1e-9)
        loss_reconst = (- 1./kappa_inv * x_reconst_selected.sum((1,2)).mean()
                        - self.dim_x * self._log_normalization(kappa_inv))
        # Entire loss
        loss_all = loss_reconst + loss_latent
        idx_estimated = torch.argmax(x_reconst_normed, dim=-1, keepdim=True)
        acc = torch.isclose(x.to(int), idx_estimated).sum() / idx_estimated.numel()
        loss = dict(all=loss_all, acc=acc)

        return loss

    def _log_normalization(self, kappa_inv):
        coeff = (
            - (self.__m - 1) * kappa_inv.log()
            - 1./kappa_inv 
            - torch.log(self.__m - 1, 1./kappa_inv)
        )

        return coeff

