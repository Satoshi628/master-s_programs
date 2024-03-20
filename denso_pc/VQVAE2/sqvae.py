import math
import timm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import distributed as dist_fn
from torch.distributions import Categorical

import constants as const


def calc_distance(z_continuous, codebook, dim_dict):
    z_continuous_flat = z_continuous.view(-1, dim_dict)
    distances = (torch.sum(z_continuous_flat**2, dim=1, keepdim=True) 
                + torch.sum(codebook**2, dim=1)
                - 2 * torch.matmul(z_continuous_flat, codebook.t()))

    return distances

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, inputs):
        out = self.conv(inputs)
        out += inputs

        return out

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1)
        flatten = inputs.reshape(-1, self.dim)
        dist = (
            flatten.pow(2).sum(1, keepdim=True)
            - 2 * flatten @ self.embed
            + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*inputs.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            dist_fn.all_reduce(embed_onehot_sum)
            dist_fn.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        # diff = 0.5*(quantize.detach() - inputs).pow(2).sum(dim=(-1,-2,-3)).mean()
        diff = 0.5*(quantize.detach() - inputs).pow(2).mean()
        quantize = inputs + (quantize - inputs).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


class GaussianVectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=1.0):
        super(GaussianVectorQuantizer, self).__init__()
        self.codebook = nn.Parameter(torch.randn(size_dict, dim_dict))
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
    
    def forward(self, z_from_encoder, var_q, flg_quant_det=True):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        precision_q = 1. / torch.clamp(var_q, min=1e-10)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, self.codebook, 0.5 * precision_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if self.training:
            encodings = F.gumbel_softmax(logit, tau=self.temperature, dim=-1)
            indices = torch.argmax(encodings, dim=1).view(bs, width, height)
            z_quantized = torch.mm(encodings, self.codebook).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=z_from_encoder.device)
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
                indices = indices.view(bs, width, height)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(self.codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, self.codebook).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()
        
        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, 0.5 * precision_q).mean()
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, indices, perplexity

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, weight):        
        distances = weight * calc_distance(z_from_encoder, codebook, self.dim_dict)
        return distances
    
    def set_temperature(self, value):
        self.temperature = value
    
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum((x1-x2)**2 * weight, dim=(1,2,3))


class VmfVectorQuantizer(nn.Module):
    def __init__(self, size_dict, dim_dict, temperature=1.0):
        super(VmfVectorQuantizer, self).__init__()
        self.codebook = nn.Parameter(torch.randn(size_dict, dim_dict))
        self.size_dict = size_dict
        self.dim_dict = dim_dict
        self.temperature = temperature
    
    def forward(self, z_from_encoder, kappa_q, flg_train=True, flg_quant_det=False):
        bs, dim_z, width, height = z_from_encoder.shape
        z_from_encoder_permuted = z_from_encoder.permute(0, 2, 3, 1).contiguous()
        codebook_norm = F.normalize(self.codebook, p=2.0, dim=1)

        logit = -self._calc_distance_bw_enc_codes(z_from_encoder_permuted, codebook_norm, kappa_q)
        probabilities = torch.softmax(logit, dim=-1)
        log_probabilities = torch.log_softmax(logit, dim=-1)
        
        # Quantization
        if flg_train:
            encodings = F.gumbel_softmax(logit, tau=self.temperature, dim=-1)
            z_quantized = torch.mm(encodings, codebook_norm).view(bs, width, height, dim_z)
            avg_probs = torch.mean(probabilities.detach(), dim=0)
        else:
            if flg_quant_det:
                indices = torch.argmax(logit, dim=1).unsqueeze(1)
                encodings_hard = torch.zeros(indices.shape[0], self.size_dict, device=z_from_encoder.device)
                encodings_hard.scatter_(1, indices, 1)
                avg_probs = torch.mean(encodings_hard, dim=0)
            else:
                dist = Categorical(probabilities)
                indices = dist.sample().view(bs, width, height)
                encodings_hard = F.one_hot(indices, num_classes=self.size_dict).type_as(self.codebook)
                avg_probs = torch.mean(probabilities, dim=0)
            z_quantized = torch.matmul(encodings_hard, codebook_norm).view(bs, width, height, dim_z)
        z_to_decoder = z_quantized.permute(0, 3, 1, 2).contiguous()

        # Latent loss
        kld_discrete = torch.sum(probabilities * log_probabilities, dim=(0,1)) / bs
        kld_continuous = self._calc_distance_bw_enc_dec(z_from_encoder, z_to_decoder, kappa_q).mean()        
        loss = kld_discrete + kld_continuous
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-7)))

        return z_to_decoder, loss, perplexity

    def _calc_distance_bw_enc_codes(self, z_from_encoder, codebook, kappa_q):
        z_from_encoder_flat = z_from_encoder.view(-1, self.dim_dict)
        distances = -kappa_q * torch.matmul(z_from_encoder_flat, codebook.t())

        return distances
    
    def set_temperature(self, value):
        self.temperature = value
    
    def _calc_distance_bw_enc_dec(self, x1, x2, weight):
        return torch.sum(x1 * (x1-x2) * weight, dim=(1,2,3))


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, inputs):
        return self.blocks(inputs)



class Gaussian_SQVAE(nn.Module):
    def __init__(
            self,
            backbone_name,
            input_size,
            n_embed=512,
            log_param_q_init=math.log(20.),
            temp_decay=1e-5,

        ):
        super(Gaussian_SQVAE, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if isinstance(input_size, int):
            input_size = [input_size, input_size]

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm

        # for param in self.feature_extractor.parameters():
        #     param.requires_grad = False

        self.input_size = input_size

        #VQ
        self.vq_list = nn.Sequential(*[GaussianVectorQuantizer(n_embed, c_dim) for c_dim in channels])
        
        # self.Decoder_list = [Decoder(in_channel, out_channel, channel, n_res_block=2, n_res_channel=32, stride=4) for _ in range(int(math.log2(scales[-1])))]
        self.Decoder_list = nn.Sequential(Decoder(1024, 512, 1024, n_res_block=2, n_res_channel=1024, stride=2),
                            Decoder(1024, 256, 512, n_res_block=2, n_res_channel=512, stride=2),
                            Decoder(512, 256, 256, n_res_block=2, n_res_channel=256, stride=2),
                            Decoder(256, 3, 128, n_res_block=2, n_res_channel=128, stride=2))
        
        self.log_param_q_scalar = nn.Parameter(torch.full([3], log_param_q_init))
        self.temp_decay = temp_decay

    def forward(self, inputs):
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(inputs)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(inputs + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size[0] // 16, self.input_size[1] // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(inputs)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size[0] // 16, self.input_size[1] // 16)
            features = [x]
        else:
            features = self.feature_extractor(inputs)

        quants, loss, quant_indices, perplexitys = self._quantize(features)
        ret = {"quantize_loss": loss}
        ret["quant_indices"] = quant_indices
        ret["perplexity"] = perplexitys

        feature = self.Decoder_list[0](quants[-1])
        feature = torch.cat([feature, quants[-2]], dim=1)

        feature = self.Decoder_list[1](feature)
        feature = torch.cat([feature, quants[-3]], dim=1)
        
        feature = self.Decoder_list[2](feature)

        reconstract_img = self.Decoder_list[3](feature)
        ret["images"] = reconstract_img
        ret["reconstraction_loss"] = self._calc_loss(reconstract_img, inputs)
        
        if self.training:
            for vq in self.vq_list:
                vq.temperature *= math.exp(-self.temp_decay)
        
        return ret

    def _quantize(self, outputs_list):
        outputs = []
        quant_indices = []
        perplexitys = []
        loss = 0
        for idx, output in enumerate(outputs_list):
            output = F.normalize(output, p=2.0, dim=1)
            param_q = (1 + self.log_param_q_scalar[idx].exp())
            quant_t, diff_t, id_t, perplexity = self.vq_list[idx](output, param_q)
            outputs.append(quant_t)
            quant_indices.append(id_t)
            perplexitys.append(perplexity)
            loss = loss + diff_t

        return outputs, loss, quant_indices, perplexitys


    def _calc_loss(self, x_reconst, x):
        # Reconstruction loss
        mse = F.mse_loss(x_reconst, x)

        # "Preventing Posterior Collapse Induced by Oversmoothing in Gaussian VAE"
        # https://arxiv.org/abs/2102.08663
        loss_reconst = np.prod(x.shape[1:]) * torch.log(mse) / 2

        return loss_reconst

class VMF_SQVAE(nn.Module):
    def __init__(
            self,
            backbone_name,
            input_size,
            n_embed=512,
            classes=10,
            log_param_q_init=math.log(0.05),
            log_kappa_inv=math.log(0.01)
        ):
        super(VMF_SQVAE, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        if isinstance(input_size, int):
            input_size = [input_size, input_size]

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # for transformers, use their pretrained norm w/o grad
            # for resnets, self.norms are trainable LayerNorm

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.input_size = input_size

        #VQ
        self.vq_list = nn.Sequential(*[VmfVectorQuantizer(n_embed, c_dim) for c_dim in channels])
        # self.Decoder_list = [Decoder(in_channel, out_channel, channel, n_res_block=2, n_res_channel=32, stride=4) for _ in range(int(math.log2(scales[-1])))]
        self.Decoder_list = nn.Sequential(Decoder(1024, 512, 1024, n_res_block=2, n_res_channel=1024, stride=2),
                            Decoder(1024, 256, 512, n_res_block=2, n_res_channel=512, stride=2),
                            Decoder(512, 256, 256, n_res_block=2, n_res_channel=256, stride=2),
                            Decoder(256, 3, 128, n_res_block=2, n_res_channel=128, stride=2))
        
        self.log_param_q_init = log_param_q_init
        self.log_kappa_inv = nn.Parameter(torch.tensor([log_kappa_inv]))
        self.__m = np.ceil(classes / 2)
        self.n_interval = classes - 1

    def forward(self, inputs):
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(inputs)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(inputs + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size[0] // 16, self.input_size[1] // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(inputs)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size[0] // 16, self.input_size[1] // 16)
            features = [x]
        else:
            features = self.feature_extractor(inputs)

        quants, loss, quant_indices = self._quantize(features)
        ret = {"quantize_loss":loss}
        ret["quant_indices"] = quant_indices

        feature = self.Decoder_list[0](quants[-1])
        feature = torch.cat([feature, quants[-2]], dim=1)

        feature = self.Decoder_list[1](feature)
        feature = torch.cat([feature, quants[-3]], dim=1)
        
        feature = self.Decoder_list[2](feature)

        reconstract_img = self.Decoder_list[3](feature)
        ret["images"] = reconstract_img
        ret["reconstraction_loss"] = self.MSE(reconstract_img, inputs)
        return ret

    def _quantize(self, outputs_list):
        outputs = []
        quant_indices = []
        loss = 0
        for idx, output in enumerate(outputs_list):
            output = F.normalize(output, p=2.0, dim=1)
            param_q = (1 + self.log_param_q_scalar[idx].exp())
            quant_t, quant_loss, id_t = self.vq_list[idx](output, param_q)
            outputs.append(quant_t)
            quant_indices.append(id_t)
            loss = loss + quant_loss

        return outputs, loss, quant_indices

    
    def _calc_loss(self, x_reconst, x, loss_latent):
        x_shape = x.shape
        
        # Reconstruction loss
        x = x.view(-1, 1)
        x_reconst_viewed = (x_reconst.permute(0, 2, 3, 1).contiguous().view(-1, int(self.__m * 2)) )
        x_reconst_normed = F.normalize(x_reconst_viewed, p=2.0, dim=-1)

        x_one_hot = (F.one_hot(x.to(torch.int).long(), num_classes = int(self.__m * 2)).type_as(x))[:,0,:]
        x_reconst_selected = (x_one_hot * x_reconst_normed).sum(-1).view(x_shape)

        kappa_inv = self.log_kappa_inv.exp().add(1e-9)
        loss_reconst = (- 1./kappa_inv * x_reconst_selected.sum((1,2)).mean() - np.prod(x.shape[1:]) * self._log_normalization(kappa_inv))
        
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
                - torch.log(ive(self.__m - 1, 1./kappa_inv))
            )
            return coeff


if __name__ == "__main__":
    model = Gaussian_SQVAE(backbone_name="wide_resnet50_2",
                        input_size=[256, 256],
                        n_embed=512,
                        log_param_q_init=math.log(20.)).cuda()

    inputs = torch.rand([1,3,256,256]).cuda()
    ret = model(inputs)
    print(ret["quantize_loss"])
    print(ret["images"].shape)
    print(ret["quant_indices"][0].shape)
    print(ret["perplexity"])
