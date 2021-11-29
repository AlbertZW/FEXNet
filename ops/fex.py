import torch
import torch.nn as nn
import torch.nn.functional as F

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

class FE(nn.Module):
	def __init__(self, in_channels, n_segment, n_div=8):
		super(FE, self).__init__()
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

class SS(nn.Module):
	def __init__(self, in_channels, n_segment=8, n_div=8):
		super(SS, self).__init__()
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

		# self.attention_model = FE(in_channels=in_channels, n_segment=n_segment, n_div=16)
		self.ST_model = SS(in_channels=in_channels, n_segment=n_segment, n_div=n_div)

		self.is_first_block = is_first_block

	def forward(self, x):
		# if self.is_first_block:
		# 	x = self.attention_model(x)
		x = self.ST_model(x)

		return x

