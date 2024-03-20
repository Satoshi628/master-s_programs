import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
	# 初期設定
	def __init__(self, in_channels, momery_channels, scale, d_k=None):
		super().__init__()
		if d_k is None:
			self.d_k = momery_channels
		else:
			self.d_k = d_k

		# Q, K, V  1×1convしたQ, K, V
		self.conv_qkv = nn.Conv2d(momery_channels, self.d_k * 3, kernel_size=1)

		self.conv = nn.Sequential(
			nn.Conv2d(self.d_k, in_channels, kernel_size=1),
			nn.BatchNorm2d(in_channels),
		)

		self.downpool = nn.MaxPool2d(scale)
		self.upsample = nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True)

		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, x, memory):
		# qkv.size() => [batch, d_k * 3, H, W]
		qkv = self.conv_qkv(memory)
		# qkv.size() => [batch, d_k * 3, H/scale, W/scale]
		qkv = self.downpool(qkv)
		# 3つに分割,それぞれ[batch,d_k,H/scale, W/scale]
		q, k, v = qkv.chunk(3, dim=1)

		batch, channle, H, W = v.size()

		# q,k,vの形を成型
		q = q.flatten(-2, -1).permute(0, 2, 1)  # [batch, H*W/scale^2, d_k]
		k = k.flatten(-2, -1)					# [batch, d_k, H*W/scale^2]
		v = v.flatten(-2, -1).permute(0, 2, 1)  # [batch, H*W/scale^2, d_k]

		# Attention Map計算
		attention = torch.bmm(q, k)  # [batch, H*W/scale^2, H*W/scale^2]
		root_dk = (self.d_k)**0.5
		# [batch, H*W/scale^2, H*W/scale^2]
		attention = F.softmax(attention / root_dk, dim=-1)

		S = torch.bmm(attention, v)  # [batch, H*W/scale^2, d_k]

		#形を整形 [batch,d_k,H/scale,W/scale]
		S = S.transpose(1, 2).view(batch, channle, H, W)

		# upsample
		S = self.upsample(S)

		out = self.conv(S)

		out = x + self.gamma * out

		return out



class SelfAttention_kv_downsample(nn.Module):
	# 初期設定
	def __init__(self, in_channels, momery_channels, scale, d_k=None):
		super().__init__()
		if d_k is None:
			self.d_k = momery_channels
		else:
			self.d_k = d_k

		# Q, K, V  1×1convしたQ, K, V
		self.conv_qkv = nn.Conv2d(momery_channels, self.d_k * 3, kernel_size=1)

		self.conv = nn.Sequential(
			nn.Conv2d(self.d_k, in_channels, kernel_size=1),
			nn.BatchNorm2d(in_channels),
		)

		self.downpool = nn.MaxPool2d(scale)

		self.gamma = nn.Parameter(torch.zeros(1))

	def forward(self, x, memory):
		# qkv.size() => [batch, d_k * 3, H, W]
		qkv = self.conv_qkv(memory)
		# 3つに分割,それぞれ[batch,d_k,H/scale, W/scale]
		q, k, v = qkv.chunk(3, dim=1)

		# k.size() => [batch, d_k, H/scale, W/scale]
		k = self.downpool(k)
		# v.size() => [batch, d_k, H/scale, W/scale]
		v = self.downpool(v)

		batch, channle, H, W = q.size()

		# q,k,vの形を成型
		q = q.flatten(-2, -1).permute(0, 2, 1)  # [batch, H*W, d_k]
		k = k.flatten(-2, -1)					# [batch, d_k, H*W/scale^2]
		v = v.flatten(-2, -1).permute(0, 2, 1)  # [batch, H*W/scale^2, d_k]

		# Attention Map計算
		attention = torch.bmm(q, k)  # [batch, H*W/, H*W/scale^2]
		root_dk = (self.d_k)**0.5
		# [batch, H*W, H*W/scale^2]
		attention = F.softmax(attention / root_dk, dim=-1)

		S = torch.bmm(attention, v)  # [batch, H*W, d_k]

		#形を整形 [batch,d_k,H,W]
		S = S.transpose(1, 2).view(batch, channle, H, W)

		out = self.conv(S)

		out = x + self.gamma * out

		return out