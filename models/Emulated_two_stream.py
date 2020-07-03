import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from models.non_local import NONLocalBlock3D

class TemporalInteraction(nn.Module):
	def __init__(self, n_div=8, conv_core=3):
		super(TemporalInteraction, self).__init__()
		self.n_div = n_div
		self.conv_core = conv_core

		self.interaction_conv = nn.Conv2d(conv_core, 1, kernel_size=(1, 1))

	def forward(self, x):
		n_batch, c, t, h, w = x.size()
		# nt, c, h, w = x.size()
		# n_batch = nt // self.n_div
		# x = x.view(n_batch, self.n_div, c, h, w)
		# x = x.permute(0, 2, 1, 3, 4).contiguous()
		x = x.view(n_batch * c, self.n_div, h, w)

		zero_padding = torch.zeros_like(x[:, 0, :, :].unsqueeze(1))
		x = torch.cat((zero_padding, x), dim=1)
		x = torch.cat((x, zero_padding), dim=1)
		for i in range(self.n_div):
			if i == 0:
				x_fused = self.interaction_conv(x[:, i: i + 3, :, :])
			else:
				x_fused = torch.cat((x_fused, self.interaction_conv(x[:, i: i + 3, :, :])), dim=1)
		x = x_fused

		x = x.view((n_batch, -1, self.n_div) + x.size()[-2:])
		x = x.permute(0, 2, 1, 3, 4).contiguous()
		x = x.view((n_batch * t, -1) + x.size()[-2:])

		return x

class SpatialReinforcement(nn.Module):
	def __init__(self, planes, n_div=8):
		super(SpatialReinforcement, self).__init__()
		self.n_div = n_div

		self.reinf_conv = nn.Conv2d(planes, planes, kernel_size=(3, 3), padding=1)

	def forward(self, x):
		nt, c, h, w = x.size()
		n_batch = nt // self.n_div
		x = x.view(n_batch, self.n_div, c, h, w)

		x_mean = x.mean(dim=1)  # B x C x H x W
		x_mean = self.reinf_conv(x_mean)

		for i in range(self.n_div):
			x[:, i, :, :, :] = x[:, i, :, :, :] - x_mean

		x = x.view(nt, c, h, w)
		return x

class Emulated_two_stream(nn.Module):
	def __init__(self, planes, n_div=8):
		super(Emulated_two_stream, self).__init__()
		self.planes = planes
		self.n_div = n_div

		self.non_local = NONLocalBlock3D(in_channels=planes, sub_sample=True, bn_layer=True)
		self.TIM = TemporalInteraction(n_div=n_div, conv_core=3)

		self.right_block = SpatialReinforcement(planes=planes, n_div=n_div)
		# self.right_block = Emulated_OFF(planes=planes, n_div=n_div)

		self.fusion_conv = nn.Conv2d(2 * planes, planes, kernel_size=(1, 1))

	def forward(self, x):
		nt, c, h, w = x.size()
		n_batch = nt // self.n_div
		left_x = x.view(n_batch, self.n_div, c, h, w)
		# b x n x c x h x w
		left_x = left_x.permute(0, 2, 1, 3, 4).contiguous()
		left_x = self.non_local(left_x)
		# b x c x n x h x w

		# left_x = left_x.permute(0, 2, 1, 3, 4).contiguous()
		# left_x = left_x.view(n_batch * self.n_div, c, h, w)
		# (b x n) x c x h x w

		left_x = self.TIM(left_x)

		right_x = self.right_block(x)

		x = torch.cat((left_x, right_x), dim=1)  # channel concat.
		x = self.fusion_conv(x)

		return x

class E2S_BottleNeck(nn.Module):
	def __init__(self, bottleNeck, planes, n_div=8):
		super(E2S_BottleNeck, self).__init__()
		self.bottleNeck = bottleNeck
		self.planes = planes
		self.n_div = n_div
		self.E2S_module = Emulated_two_stream(planes, n_div)

	def forward(self, x):
		identity = x

		out = self.bottleNeck.conv1(x)
		out = self.bottleNeck.bn1(out)
		out = F.relu(out)

		# Emulated two-stream
		out = self.E2S_module(out)

		out = self.bottleNeck.conv2(out)
		out = self.bottleNeck.bn2(out)
		out = F.relu(out)

		out = self.bottleNeck.conv3(out)
		out = self.bottleNeck.bn3(out)

		if self.bottleNeck.downsample is not None:
			identity = self.bottleNeck.downsample(x)

		out = out + identity
		out = F.relu(out)

		return out


def construct_E2S_Net(net, n_div=8):
	if isinstance(net, torchvision.models.ResNet):
		bottleNeck_dims = [64, 128, 256, 512]

		def add_on_E2S(stage, planes):
			blocks = list(stage.children())
			print('=> Processing stage with {} blocks residual'.format(len(blocks)))
			for i, b in enumerate(blocks):
				blocks[i] = E2S_BottleNeck(b, planes=planes, n_div=n_div)
			return nn.Sequential(*blocks)

		# net.layer1 = add_on_E2S(net.layer1, bottleNeck_dims[0])
		# net.layer2 = add_on_E2S(net.layer2, bottleNeck_dims[1])
		# net.layer3 = add_on_E2S(net.layer3, bottleNeck_dims[2])
		net.layer4 = add_on_E2S(net.layer4, bottleNeck_dims[3])

	else:
		raise NotImplementedError(net)

