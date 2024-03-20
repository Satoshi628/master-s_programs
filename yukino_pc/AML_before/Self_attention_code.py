import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

   # 初期設定
    def __init__(self, ch):
        super(SelfAttention, self).__init__()

        # Q, K, V  1×1convしたQ, K, V
        self.conv_q = nn.Conv2d(ch, ch // 8, 1, 1, 0)
        self.bn_q = nn.BatchNorm2d(ch // 8)

        self.conv_k = nn.Conv2d(ch, ch // 8, 1, 1, 0) 
        self.bn_k = nn.BatchNorm2d(ch // 8)

        self.conv_v = nn.Conv2d(ch, ch // 2, 1, 1, 0)
        self.bn_v = nn.BatchNorm2d(ch // 2)

        self.conv_point = nn.Conv2d(ch // 2, ch, 1, 1, 0)
        self.bn_p = nn.BatchNorm2d(ch)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.upsamp = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)

        #self.gamma = nn.Parameter(torch.zeros(1, 1))
        self.gamma = nn.Parameter(torch.zeros(1))
        #self.sigma_ratio = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, memory):

        # -------------------- point wise attention --------------------

        batch, ch, height, width = x.shape  # b:Batch Size, dk:ハイパーパラメータ, W*H:画像の大きさ

        # ---------- Q ----------
        query = self.bn_q(self.conv_q(memory))  # Qを1×1conv

        #print("memory", memory.size())
        # sys.exit()
        # print("Q",query.size())
        # input()

        # reshape -> サイズ指定  permute -> 次元の入れ替え
        # (B, dk, W, H) -> (B, W*H, dk)
        query = torch.reshape(
            query, (batch, ch // 8, height * width)).permute([0, 2, 1])

        #print("Q", query.size())
        # input()

        # ---------- K ----------
        key = self.bn_k(self.conv_k(memory))  # Kを1×1conv

        # scale down
        key = self.pool(key)
        #print("k", key.size())
        # sys.exit()

        # reshape -> サイズ指定
        # (B, dk, W, H) -> (B, dk, W*H)
        key = torch.reshape(key, (batch, ch // 8, height * width // 4))
        #print("k", key.size())
        # sys.exit()

        # QとKの行列積の計算
        attention = torch.bmm(query, key)  # (B, W*H, W*H)

        attention = F.softmax(attention, dim=2)
        # print(attention.shape)
        # sys.exit()

        # ---------- V ----------
        v = self.bn_v(self.conv_v(memory))  # Vを1×1conv

        # scale down
        # torch.Size([8, 64, 64, 64])
        v = self.pool(v)

        # reshape -> サイズ指定  permute -> 次元の入れ替え
        # (B, dv, W, H) -> (B, W*H, dv)
        v = torch.reshape(v, (batch, ch // 2, height *
                          width // 4)).permute([0, 2, 1])
        # print(v.shape)
        # sys.exit()

        # softmax(attention)とVの行列積の計算
        attn_g = torch.bmm(attention, v)  # (B, W*H, dv)
        # print(attn_g.shape)  # torch.Size([8, 16384, 64])
        # sys.exit()

        # 次元数を戻す 3次元 -> 4次元 # (B, dv,H,W )

        attn_g = attn_g.permute([0, 2, 1])
        attn_g = torch.reshape(attn_g, (batch, ch // 2, height, width))
        #print("attn_g:", attn_g.size())
        # sys.exit()

        #print("inputs.size:", inputs.size())
        #print("z.size:", z.size())
        # input()
        attn_g = self.bn_p(self.conv_point(attn_g))
        # attn_g: torch.Size([8, 128, 128, 128])
        ##print("z.size:", z.size())
        # input()
        check = self.gamma
        attn_g = check*attn_g
        # print(check)
        # print(attn_g.shape)
        attn_g = x + attn_g

        return attn_g


"""/opt/conda/lib/python3.7/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /opt/conda/conda-bld/pytorch_1623448265233/work/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)

/opt/conda/lib/python3.7/site-packages/torch/nn/functional.py:3613: UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details.
  "See the documentation of nn.Upsample for details.".format(mode)
Epoch  1/400  LossG=1.0902  LossD=0.3619 train_miou=14.12%  val_miou=23.04%                                                                                                                                        
  0%|                                                                                                                                                                                       | 0/12 [00:00<?, ?it/s][W accumulate_grad.h:185] Warning: grad and param do not obey the gradient layout contract. This is not an error, but may impair performance.
grad.sizes() = [1024, 512, 1, 1], strides() = [512, 1, 1, 1]
param.sizes() = [1024, 512, 1, 1], strides() = [512, 1, 512, 512] (function operator())"""


"""        # -------------------- channel attention --------------------
        # b:Batch Size, inputs_channels:チャンネル, W*H:画像の大きさ
        b, inputs_channels, W, H = inputs.size()

        # (B, C, W, H) -> (B, C, W*H)
        c1 = inputs.reshape(b, inputs_channels, W * H)
        c2 = c1.permute([0, 2, 1])  # (B, C, W*H) -> (B, W*H, C)

        # c1とc2の行列積を計算
        c3 = torch.bmm(c1, c2)
        c3 = F.softmax((c3 / (W * H)**0.5), dim=2)

        v4 = c1

        # c3とv4の行列積を計算
        v5 = torch.bmm(c3, v4)
        v5 = v5.reshape(b, inputs_channels, W, H)  # (B, C, W*H) -> (B, C, W, H)
        v5 = v5 + inputs

        # point wise attentonとchannel attentionをsum fusion
        # z:point wise attention, v5:channel attention
        v6 = torch.cat((z, v5), dim=1)
        v6 = self.conv_channel(v6)

        return v6, attention, c3"""
