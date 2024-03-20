import math
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import distributed as dist_fn

from quantizer import GaussianVectorQuantizer, VmfVectorQuantizer
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

        diff = 0.5*(quantize.detach() - inputs).pow(2).sum(dim=(-1,-2,-3)).mean()
        # diff = 0.5*(quantize.detach() - inputs).pow(2).mean()
        quantize = inputs + (quantize - inputs).detach()

        return quantize.permute(0, 3, 1, 2), diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))




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



class VQ_FastFlow(nn.Module):
    def __init__(
            self,
            backbone_name,
            flow_steps,
            input_size,
            conv3x3_only=False,
            hidden_ratio=1.0,
            n_embed=512
        ):
        super(VQ_FastFlow, self).__init__()
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
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

        #VQ
        self.vq_list = nn.ModuleList()
        for c in channels:
            self.vq_list.append(Quantize(c, n_embed))
        
        # self.Decoder_list = [Decoder(in_channel, out_channel, channel, n_res_block=2, n_res_channel=32, stride=4) for _ in range(int(math.log2(scales[-1])))]
        self.Decoder_list = nn.Sequential(Decoder(1024, 512, 1024, n_res_block=2, n_res_channel=1024, stride=2),
                            Decoder(1024, 256, 512, n_res_block=2, n_res_channel=512, stride=2),
                            Decoder(512, 256, 256, n_res_block=2, n_res_channel=256, stride=2),
                            Decoder(256, 3, 128, n_res_block=2, n_res_channel=128, stride=2))
        self.MSE = nn.MSELoss()

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
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        outputs = []
        loss = 0
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            outputs.append(output)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            # loss += -log_jac_dets.mean()
        ret = {"jac_dets_loss": loss}
        
        quants, loss, quant_indices = self._quantize(outputs)
        ret["quantize_loss"] = loss
        ret["quant_indices"] = quant_indices

        if self.training or True:
            past_feature = None                
            feature = self.Decoder_list[0](quants[-1])
            feature = torch.cat([feature, quants[-2]], dim=1)

            feature = self.Decoder_list[1](feature)
            feature = torch.cat([feature, quants[-3]], dim=1)
            
            feature = self.Decoder_list[2](feature)

            reconstract_img = self.Decoder_list[3](feature)
            # reconstract_img = self.Decoder_list[4](feature)
            ret["images"] = reconstract_img
            ret["reconstraction_loss"] = self.MSE(reconstract_img, inputs)


        if not self.training:
            anomaly_map_list = []
            for output, quant in zip(outputs, quants):
                log_prob = -torch.mean((output - quant)**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size[0], self.input_size[1]],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map
        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                output = output.mean(dim=1, keepdim=True)
                a_map = F.interpolate(
                    output,
                    size=[self.input_size[0], self.input_size[1]],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.cat(anomaly_map_list, dim=1)
            anomaly_map_list = anomaly_map_list.mean(dim=1)
            ret["output_map"] = anomaly_map_list
        return ret

    def _quantize(self, outputs_list):
        outputs = []
        quant_indices = []
        loss = 0
        for idx, output in enumerate(outputs_list):
            quant_t, diff_t, id_t = self.vq_list[idx](output)
            outputs.append(quant_t)
            quant_indices.append(id_t)
            loss = loss + diff_t

        return outputs, loss, quant_indices

class SQ_FastFlow(nn.Module):
    def __init__(
            self,
            backbone_name,
            flow_steps,
            input_size,
            conv3x3_only=False,
            hidden_ratio=1.0,
            n_embed=512
        ):
        super(SQ_FastFlow, self).__init__()
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
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm(
                        [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                        elementwise_affine=True,
                    )
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, int(input_size[0] / scale), int(input_size[1] / scale)],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

        #VQ
        # self.quantizer = nn.Sequential(*[Quantize(c_dim, n_embed) for c_dim in channels])
        self.quantizer = nn.ModuleList()
        for c in channels:
            self.quantizer.append(GaussianVectorQuantizer(n_embed, c, 1.0))
        self.log_param_q_scalar = nn.Parameter(torch.full([len(channels)], 2.995732273553991)) # log(20.0)
        
        # self.Decoder_list = [Decoder(in_channel, out_channel, channel, n_res_block=2, n_res_channel=32, stride=4) for _ in range(int(math.log2(scales[-1])))]
        self.Decoder_list = nn.Sequential(Decoder(1024, 512, 1024, n_res_block=2, n_res_channel=1024, stride=2),
                            Decoder(1024, 256, 512, n_res_block=2, n_res_channel=512, stride=2),
                            Decoder(512, 256, 256, n_res_block=2, n_res_channel=256, stride=2),
                            Decoder(256, 3, 128, n_res_block=2, n_res_channel=128, stride=2))
        self.MSE = nn.MSELoss()

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
            features = [self.norms[i](feature) for i, feature in enumerate(features)]

        outputs = []
        loss = 0
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            # log_jac_dets = log_jac_dets / math.prod(feature.shape)
            outputs.append(output)
            loss += -log_jac_dets.mean()
        ret = {"jac_dets_loss": loss}
        
        
        param_q = (1 + self.log_param_q_scalar.exp())

        quants, loss, quant_indices = self._quantize(outputs, param_q)
        ret["quantize_loss"] = loss
        ret["quant_indices"] = quant_indices

        if self.training or True:
            past_feature = None                
            feature = self.Decoder_list[0](quants[-1])
            feature = torch.cat([feature, quants[-2]], dim=1)

            feature = self.Decoder_list[1](feature)
            feature = torch.cat([feature, quants[-3]], dim=1)
            
            feature = self.Decoder_list[2](feature)

            reconstract_img = self.Decoder_list[3](feature)
            # reconstract_img = self.Decoder_list[4](feature)
            ret["images"] = reconstract_img
            ret["reconstraction_loss"] = self.MSE(reconstract_img, inputs)


        if not self.training:
            anomaly_map_list = []
            for output, quant in zip(outputs, quants):
                log_prob = -torch.mean((output - quant)**2, dim=1, keepdim=True) * 0.5
                prob = torch.exp(log_prob)
                a_map = F.interpolate(
                    -prob,
                    size=[self.input_size[0], self.input_size[1]],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
            anomaly_map = torch.mean(anomaly_map_list, dim=-1)
            ret["anomaly_map"] = anomaly_map
        if not self.training:
            anomaly_map_list = []
            for output in outputs:
                output = output.mean(dim=1, keepdim=True)
                a_map = F.interpolate(
                    output,
                    size=[self.input_size[0], self.input_size[1]],
                    mode="bilinear",
                    align_corners=False,
                )
                anomaly_map_list.append(a_map)
            anomaly_map_list = torch.cat(anomaly_map_list, dim=1)
            anomaly_map_list = anomaly_map_list.mean(dim=1)
            ret["output_map"] = anomaly_map_list
        return ret

    def _quantize(self, outputs_list, param_q):
        outputs = []
        quant_indices = []
        loss = 0
        flg_train = flg_quant_det = self.training
        for idx, output in enumerate(outputs_list):
            quant_t, diff_t, id_t = self.quantizer[idx](output, param_q[idx], flg_train, flg_quant_det)
            outputs.append(quant_t)
            quant_indices.append(id_t)
            loss = loss + diff_t

        return outputs, loss, quant_indices


if __name__ == "__main__":
    model = FastFlow(backbone_name="wide_resnet50_2",
                        flow_steps=8,
                        input_size=[256, 256],).cuda()

    inputs = torch.rand([1,3,256,256]).cuda()
    ret = model(inputs)
    print(ret["quantize_loss"])
    print(ret["images"].shape)
    print(ret["quant_indices"][0].shape)
