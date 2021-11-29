import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class CONV1d_FusionBlock(nn.Module):
	def __init__(self, in_channels, n_segment, n_div):
		super(CONV1d_FusionBlock, self).__init__()

		self.n_div = n_div
		self.fold = in_channels // n_div
		self.n_segment = n_segment

		self.temporal_conv = nn.Conv3d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=(3, 1, 1),
		                               padding=(1, 0, 0), stride=1, bias=True)

		# self.temporal_bn = nn.BatchNorm3d(2 * self.fold)
		#
		# self.relu = nn.ReLU(inplace=True)

		nn.init.constant_(self.temporal_conv.weight, 0)
		nn.init.constant_(self.temporal_conv.bias, 0)

	def forward(self, x):
		'''
		:param x: (nt, c, h, w)
		:return:(nt, c, h, w)
		'''

		# Reshaping to tensor of size [batch, frames, channels, H, W]
		nt, c, h, w = x.size()
		n_batch = nt // self.n_segment

		x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
		out_part = x[:, :2 * self.fold]

		out_part = self.temporal_conv(out_part)  # n, 2*fold, t, h, w
		# out_part = self.temporal_bn(out_part)
		# out_part = self.relu(out_part)

		out = torch.zeros_like(x)
		out[:, :2 * self.fold] = out_part
		out[:, 2 * self.fold:] = x[:, 2 * self.fold:]

		out = out.transpose(1, 2).contiguous().view(nt, c, h, w)

		return out


class CONV3d_FusionBlock(nn.Module):
	def __init__(self, in_channels, n_segment, n_div):
		super(CONV3d_FusionBlock, self).__init__()

		self.n_div = n_div
		self.fold = in_channels // n_div
		self.n_segment = n_segment

		self.temporal_conv = nn.Conv3d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=(3, 3, 3),
		                               padding=(1, 1, 1), stride=1, bias=False)

		self.temporal_bn = nn.BatchNorm3d(2 * self.fold)

		self.relu = nn.ReLU(inplace=True)

		nn.init.constant_(self.temporal_conv.weight, 0)
		# nn.init.constant_(self.temporal_conv.bias, 0)

	def forward(self, x):
		'''
		:param x: (nt, c, h, w)
		:return:(nt, c, h, w)
		'''

		# Reshaping to tensor of size [batch, frames, channels, H, W]
		nt, c, h, w = x.size()
		n_batch = nt // self.n_segment

		x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
		out_part = x[:, :2 * self.fold]

		out_part = self.temporal_conv(out_part)  # n, 2*fold, t, h, w
		out_part = self.temporal_bn(out_part)
		out_part = self.relu(out_part)

		out = torch.zeros_like(x)
		out[:, :2 * self.fold] = out_part
		out[:, 2 * self.fold:] = x[:, 2 * self.fold:]

		out = out.transpose(1, 2).contiguous().view(nt, c, h, w)

		return out


class CONV1d_Channel_FusionBlock(nn.Module):
	def __init__(self, in_channels, n_segment, n_div):
		super(CONV1d_Channel_FusionBlock, self).__init__()

		self.n_div = n_div
		self.fold = in_channels // n_div
		self.n_segment = n_segment

		self.temporal_conv = nn.Conv3d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=(3, 1, 1),
		                               padding=(1, 0, 0), stride=1, bias=True, groups=2 * self.fold)

		self.temporal_bn = nn.BatchNorm3d(2 * self.fold)
		#
		# self.relu = nn.ReLU(inplace=True)

		nn.init.constant_(self.temporal_conv.weight, 0)
		nn.init.constant_(self.temporal_conv.bias, 0)

	def forward(self, x):
		'''
		:param x: (nt, c, h, w)
		:return:(nt, c, h, w)
		'''

		# Reshaping to tensor of size [batch, frames, channels, H, W]
		nt, c, h, w = x.size()
		n_batch = nt // self.n_segment

		x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
		out_part = x[:, :2 * self.fold]

		out_part = self.temporal_conv(out_part)  # n, 2*fold, t, h, w
		out_part = self.temporal_bn(out_part)
		# out_part = self.relu(out_part)

		out = torch.zeros_like(x)
		out[:, :2 * self.fold] = out_part
		out[:, 2 * self.fold:] = x[:, 2 * self.fold:]

		out = out.transpose(1, 2).contiguous().view(nt, c, h, w)

		return out

class SpatialReinforcement(nn.Module):
	def __init__(self, planes, n_segment=8):
		super(SpatialReinforcement, self).__init__()
		self.n_segment = n_segment

		self.reinf_conv = nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1)

	def forward(self, x):
		n_batch, c, t, h, w = x.size()

		x_mean = x.mean(dim=2)  # B x C x H x W
		x_mean = self.reinf_conv(x_mean)

		x_mean = x_mean.repeat((self.n_segment, 1, 1, 1, 1)).view(n_batch, c, self.n_segment, h, w).contiguous()
		x = x - x_mean

		return x


class MEModule(nn.Module):
	""" Motion exciation module
	:param reduction=16
	:param n_segment=8/16

	"""
	def __init__(self, channel, reduction=16, n_segment=8):
		super(MEModule, self).__init__()
		self.channel = channel
		self.reduction = reduction
		self.n_segment = n_segment
		self.conv1 = nn.Conv2d(
			in_channels=self.channel,
			out_channels=self.channel // self.reduction,
			kernel_size=1,
			bias=False)
		self.bn1 = nn.BatchNorm2d(num_features=self.channel // self.reduction)
		self.conv2 = nn.Conv2d(
			in_channels=self.channel // self.reduction,
			out_channels=self.channel // self.reduction,
			kernel_size=3,
			padding=1,
			groups=channel // self.reduction,
			bias=False)
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
		self.sigmoid = nn.Sigmoid()
		self.pad = (0, 0, 0, 0, 0, 0, 0, 1)
		self.conv3 = nn.Conv2d(
			in_channels=self.channel // self.reduction,
			out_channels=self.channel,
			kernel_size=1,
			bias=False)
		self.bn3 = nn.BatchNorm2d(num_features=self.channel)
		self.identity = nn.Identity()

	def forward(self, x):
		nt, c, h, w = x.size()
		bottleneck = self.conv1(x)  # nt, c//r, h, w
		bottleneck = self.bn1(bottleneck)  # nt, c//r, h, w

		# t feature
		reshape_bottleneck = bottleneck.view((-1, self.n_segment) + bottleneck.size()[1:])  # n, t, c//r, h, w
		t_fea, __ = reshape_bottleneck.split([self.n_segment - 1, 1], dim=1)  # n, t-1, c//r, h, w

		# apply transformation conv to t+1 feature
		conv_bottleneck = self.conv2(bottleneck)  # nt, c//r, h, w

		# reshape fea: n, t, c//r, h, w
		reshape_conv_bottleneck = conv_bottleneck.view((-1, self.n_segment) + conv_bottleneck.size()[1:])
		__, tPlusone_fea = reshape_conv_bottleneck.split([1, self.n_segment - 1], dim=1)  # n, t-1, c//r, h, w

		# motion fea = t+1_fea - t_fea
		# pad the last timestamp
		diff_fea = tPlusone_fea - t_fea  # n, t-1, c//r, h, w

		# pad = (0,0,0,0,0,0,0,1)
		diff_fea_pluszero = F.pad(diff_fea, self.pad, mode="constant", value=0)  # n, t, c//r, h, w
		diff_fea_pluszero = diff_fea_pluszero.view((-1,) + diff_fea_pluszero.size()[2:])  # nt, c//r, h, w
		y = self.avg_pool(diff_fea_pluszero)  # nt, c//r, 1, 1
		y = self.conv3(y)  # nt, c, 1, 1
		y = self.bn3(y)  # nt, c, 1, 1
		y = self.sigmoid(y)  # nt, c, 1, 1
		y = y - 0.5
		output = x + x * y.expand_as(x)

		return output


class SREncoding(nn.Module):
	def __init__(self, in_channels, n_segment, n_div=8):
		super(SREncoding, self).__init__()
		self.n_segment = n_segment
		self.scene_encoding_conv = nn.Conv3d(in_channels, in_channels // n_div, kernel_size=(1, 1, 1))
		self.bg_T_encoding_conv = nn.Conv3d(in_channels, in_channels // n_div, kernel_size=(3, 1, 1), padding=(1, 0, 0))
		self.bg_S_encoding_conv = nn.Conv2d(in_channels // n_div, in_channels // n_div, kernel_size=(3, 3), padding=(1, 1))

		self.pooling = nn.AdaptiveAvgPool2d((1, 1))
		self.sigmoid = nn.Sigmoid()
		self.att_encoding_conv = nn.Conv2d(in_channels // n_div, in_channels, kernel_size=(1, 1))

	def forward(self, x):
		nt, c, h, w = x.size()
		n = nt // self.n_segment
		x_3d = x.view((n, c, self.n_segment, h, w)).contiguous()

		scene = self.scene_encoding_conv(x_3d)
		background = self.bg_T_encoding_conv(x_3d)
		background = background.mean(dim=2)  # B x C x H x W
		background = self.bg_S_encoding_conv(background)

		background = background.repeat((self.n_segment, 1, 1, 1, 1)).view(n, -1, self.n_segment, h, w).contiguous()
		forground = scene - background
		forground = forground.view((nt, -1, h, w)).contiguous()
		forground = self.pooling(forground)
		forground = self.att_encoding_conv(forground)

		x = x * forground
		return x

class Emulated_two_stream(nn.Module):
	def __init__(self, in_channels, n_segment=8, n_div=8):
		super(Emulated_two_stream, self).__init__()
		self.planes = in_channels
		self.n_segment = n_segment
		self.fold = in_channels // n_div

		self.temporal_conv = nn.Conv3d(in_channels=2 * self.fold, out_channels=2 * self.fold, kernel_size=(3, 1, 1),
		                               padding=(1, 0, 0), stride=1, bias=True, groups=2 * self.fold)

		self.temporal_bn = nn.BatchNorm3d(2 * self.fold)

		nn.init.constant_(self.temporal_conv.weight, 0)
		nn.init.constant_(self.temporal_conv.bias, 0)

		self.SR = SpatialReinforcement(planes=(2 * self.fold), n_segment=n_segment)
		self.fusion_conv = nn.Conv3d(4 * self.fold, 2 * self.fold, kernel_size=(1, 1, 1))

	def forward(self, x):
		# (n x t) x c x h x w
		nt, c, h, w = x.size()
		n_batch = nt // self.n_segment

		x = x.view(n_batch, self.n_segment, c, h, w).transpose(1, 2)  # n, c, t, h, w
		foreground_part = x[:, :2 * self.fold]
		background_part = x[:, 2 * self.fold:]

		temporal_enhance_part = self.temporal_conv(foreground_part)  # n, 2*fold, t, h, w
		temporal_enhance_part = self.temporal_bn(temporal_enhance_part)
		# out_part = self.relu(out_part)

		spatial_enhance_part = self.SR(foreground_part)

		foreground_part = torch.cat((temporal_enhance_part, spatial_enhance_part), dim=1)
		foreground_part = self.fusion_conv(foreground_part)

		out = torch.zeros_like(x)
		out[:, :2 * self.fold] = foreground_part
		out[:, 2 * self.fold:] = background_part

		out = out.transpose(1, 2).contiguous().view(nt, c, h, w)

		return out

class FEX(nn.Module):
	def __init__(self, in_channels, n_segment=8, n_div=8, is_first_block=False):
		super(FEX, self).__init__()
		self.planes = in_channels
		self.n_segment = n_segment
		self.fold = in_channels // n_div

		# self.attention_model = SREncoding(in_channels=in_channels, n_segment=n_segment, n_div=16)
		self.ST_model = Emulated_two_stream(in_channels=in_channels, n_segment=n_segment, n_div=n_div)

		self.is_first_block = is_first_block

	def forward(self, x):
		# if self.is_first_block:
		# 	x = self.attention_model(x)
		x = self.ST_model(x)

		return x

