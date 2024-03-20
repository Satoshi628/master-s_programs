import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):

   # 初期設定
    def __init__(self, inputs_channels, dk, dv, memory_channels=None):
        super(SelfAttention, self).__init__()
        if memory_channels is None:
            memory_channels = inputs_channels

        # Q, K, V  1×1convしたQ, K, V
        self.conv_q = nn.Conv2d(memory_channels, dk, kernel_size=1)
        self.conv_k = nn.Conv2d(memory_channels, dk, kernel_size=1)
        self.conv_v = nn.Conv2d(memory_channels, dv, kernel_size=1)
        self.conv_point = nn.Conv2d(dv, inputs_channels, kernel_size=1)
        self.conv_channel = nn.Conv2d(
            inputs_channels*2, inputs_channels, kernel_size=1)
        self.dk = dk
        self.dv = dv
        self.downsamp = nn.AvgPool2d(4)
        self.upsamp = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, inputs, memory):

        # -------------------- point wise attention --------------------

        b, dk, W, H = memory.size()  # b:Batch Size, dk:ハイパーパラメータ, W*H:画像の大きさ

        # ---------- Q ----------
        q = self.conv_q(memory)  # Qを1×1conv

        # scale down
        q = self.downsamp(q)

        #print("memory",memory.size())
        #print("Q",q.size())
        #input()

        # view -> サイズ指定  permute -> 次元の入れ替え
        # (B, dk, W, H) -> (B, W*H, dk)
        q = q.view(b, self.dk, W * H//16).permute([0, 2, 1])

        # ---------- K ----------
        k = self.conv_k(memory)  # Kを1×1conv

        # scale down
        k = self.downsamp(k)

        # view -> サイズ指定
        # (B, dk, W, H) -> (B, dk, W*H)
        k = k.view(b, self.dk, W * H//16).permute([0, 1, 2])

        # ---------- V ----------
        v = self.conv_v(memory)  # Vを1×1conv

        # scale down
        v = self.downsamp(v)

        # view -> サイズ指定  permute -> 次元の入れ替え
        # (B, dv, W, H) -> (B, W*H, dv)
        v = v.view(b, self.dv, W * H//16).permute([0, 2, 1])

        # QとKの行列積の計算
        attention = torch.bmm(q, k)  # (B, W*H, W*H)
        root_dk = (self.dk)**0.5  # √dk
        attention = F.softmax((attention / root_dk), dim=2)

        # softmax(attention/√dk)とVの行列積の計算
        z = torch.bmm(attention, v)  # (B, W*H, dv)

        # 次元数を戻す 3次元 -> 4次元 # (B, dv,H,W )
        z = z.permute([0, 2, 1]).view(b, self.dv, W // 4, H // 4)
        z = self.upsamp(z)
        #print("inputs.size:", inputs.size())
        #print("z.size:", z.size())
        #input()
        z = self.conv_point(z)
        #print("inputs.size:", inputs.size())
        #print("z.size:", z.size())
        #input()
        z = z + inputs

        # -------------------- channel attention --------------------
        # b:Batch Size, inputs_channels:チャンネル, W*H:画像の大きさ
        b, inputs_channels, W, H = inputs.size()

        # (B, C, W, H) -> (B, C, W*H)
        c1 = inputs.view(b, inputs_channels, W * H)
        c2 = c1.permute([0, 2, 1])  # (B, C, W*H) -> (B, W*H, C)

        # c1とc2の行列積を計算
        c3 = torch.bmm(c1, c2)
        c3 = F.softmax((c3 / (W * H)**0.5), dim=2)

        v4 = c1

        # c3とv4の行列積を計算
        v5 = torch.bmm(c3, v4)
        v5 = v5.view(b, inputs_channels, W, H)  # (B, C, W*H) -> (B, C, W, H)
        v5 = v5 + inputs

        # point wise attentonとchannel attentionをsum fusion
        # z:point wise attention, v5:channel attention
        v6 = torch.cat((z, v5), dim=1)
        v6 = self.conv_channel(v6)

        return v6, attention, c3
