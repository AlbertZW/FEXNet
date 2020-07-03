import numpy as np
from torch import nn

from processing_data.transforms import *
from models.basic_ops import ConsensusModule
from torch.nn.init import normal_, constant_
from collections import OrderedDict

from models.Emulated_two_stream import construct_E2S_Net

class E2S_Net(nn.Module):
	def __init__(self, num_class, num_segments, base_model='resnet50',
	             modality='RGB', new_length=None, dropout=0.8, crop_num=1):

		super(E2S_Net, self).__init__()
		self.category_num = num_class
		self.num_segments = num_segments
		self.modality = modality
		self.dropout = dropout
		self.crop_num = crop_num

		consensus_type = 'avg'
		self.consensus = ConsensusModule(consensus_type)

		self.fc_lr5 = False

		self._enable_pbn = True

		if new_length is None:
			self.new_length = 1 if modality == "RGB" else 5
		else:
			self.new_length = new_length

		self._prepare_base_model(base_model)

		if self.modality == 'Flow':
			print("Converting the ImageNet model to a flow init model")
			self.base_model = self._construct_flow_model(self.base_model)
			print("Done. Flow model ready...")

		self.feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features

		setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
		self.new_fc = nn.Linear(self.feature_dim, num_class)

		std = 0.001
		if hasattr(self.new_fc, 'weight'):
			normal_(self.new_fc.weight, 0, std)
			constant_(self.new_fc.bias, 0)


	def _prepare_base_model(self, base_model):
		if 'resnet' in base_model:
			self.base_model = getattr(torchvision.models, base_model)(True)
			construct_E2S_Net(self.base_model, n_div=self.num_segments)

			self.base_model.last_layer_name = 'fc'
			self.input_size = 224
			self.input_mean = [0.485, 0.456, 0.406]
			self.input_std = [0.229, 0.224, 0.225]

			self.base_model.avgpool = nn.AdaptiveAvgPool2d(1)

		elif base_model == 'BNInception':
			import tf_model_zoo
			self.base_model = getattr(tf_model_zoo, base_model)()
			self.base_model.last_layer_name = 'fc'
			self.input_size = 224
			self.input_mean = [104, 117, 128]
			self.input_std = [1]

		elif 'inception' in base_model:
			import tf_model_zoo
			self.base_model = getattr(tf_model_zoo, base_model)()
			self.base_model.last_layer_name = 'classif'
			self.input_size = 299
			self.input_mean = [0.5]
			self.input_std = [0.5]
		else:
			raise ValueError('Unknown spatial model: {}'.format(base_model))

	def train(self, mode=True):
		"""
		Override the default train() to freeze the BN parameters
		:return:
		"""
		super(E2S_Net, self).train(mode)
		count = 0
		if self._enable_pbn and mode:
			print("Freezing BatchNorm2D except the first one.")
			for m in self.base_model.modules():
				if isinstance(m, nn.BatchNorm2d):
					count += 1
					if count >= (2 if self._enable_pbn else 1):
						m.eval()
						# shutdown update in frozen mode
						m.weight.requires_grad = False
						m.bias.requires_grad = False
			# for m in self.temporal_model.modules():
			# 	if isinstance(m, nn.BatchNorm2d):
			# 		count += 1
			# 		if count >= (2 if self._enable_pbn else 1):
			# 			m.eval()
			# 			# shutdown update in frozen mode
			# 			m.weight.requires_grad = False
			# 			m.bias.requires_grad = False

	def get_optim_policies(self):
		first_conv_weight = []
		first_conv_bias = []
		normal_weight = []
		normal_bias = []
		lr5_weight = []
		lr10_bias = []
		bn = []
		custom_ops = []

		conv_cnt = 0
		bn_cnt = 0
		for m in self.modules():
			if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
				ps = list(m.parameters())
				conv_cnt += 1
				if conv_cnt == 1:
					first_conv_weight.append(ps[0])
					if len(ps) == 2:
						first_conv_bias.append(ps[1])
				else:
					normal_weight.append(ps[0])
					if len(ps) == 2:
						normal_bias.append(ps[1])
			elif isinstance(m, torch.nn.Linear):
				ps = list(m.parameters())
				if self.fc_lr5:
					lr5_weight.append(ps[0])
				else:
					normal_weight.append(ps[0])
				if len(ps) == 2:
					if self.fc_lr5:
						lr10_bias.append(ps[1])
					else:
						normal_bias.append(ps[1])

			elif isinstance(m, torch.nn.BatchNorm2d):
				bn_cnt += 1
				# later BN's are frozen
				if not self._enable_pbn or bn_cnt == 1:
					bn.extend(list(m.parameters()))
			elif isinstance(m, torch.nn.BatchNorm3d):
				bn_cnt += 1
				# later BN's are frozen
				if not self._enable_pbn or bn_cnt == 1:
					bn.extend(list(m.parameters()))
			elif len(m._modules) == 0:
				if len(list(m.parameters())) > 0:
					raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

		return [
			{'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
			 'name': "first_conv_weight"},
			{'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
			 'name': "first_conv_bias"},
			{'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
			 'name': "normal_weight"},
			{'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
			 'name': "normal_bias"},
			{'params': bn, 'lr_mult': 1, 'decay_mult': 0,
			 'name': "BN scale/shift"},
			{'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
			 'name': "custom_ops"},
			# for fc
			{'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
			 'name': "lr5_weight"},
			{'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
			 'name': "lr10_bias"},
		]

	def forward(self, input):
		sample_len = (3 if self.modality == "RGB" else 2) * self.new_length

		if self.modality == 'RGBDiff':
			sample_len = 3 * self.new_length
			input = self._get_diff(input)

		base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
		base_out = self.new_fc(base_out)
		base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])
		output = self.consensus(base_out)
		return output.squeeze(1)

	def _construct_flow_model(self, base_model):
		# modify the convolution layers
		# Torch models are usually defined in a hierarchical way.
		# nn.modules.children() return all sub modules in a DFS manner
		modules = list(self.base_model.modules())
		first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
		conv_layer = modules[first_conv_idx]
		container = modules[first_conv_idx - 1]

		# modify parameters, assume the first blob contains the convolution kernels
		params = [x.clone() for x in conv_layer.parameters()]
		kernel_size = params[0].size()
		new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
		new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

		new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
		                     conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
		                     bias=True if len(params) == 2 else False)
		new_conv.weight.data = new_kernels
		if len(params) == 2:
			new_conv.bias.data = params[1].data  # add bias if neccessary
		layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

		# replace the first convlution layer
		setattr(container, layer_name, new_conv)
		return base_model

	@property
	def crop_size(self):
		return self.input_size

	@property
	def scale_size(self):
		return self.input_size * 256 // 224


	def get_augmentation(self, type='spatial', flip=True):
		if self.modality == 'RGB':
			if flip:
				return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
				                                       GroupRandomHorizontalFlip(is_flow=False)])
			else:
				print('#' * 20, 'NO FLIP!!!')
				return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66])])
		elif self.modality == 'Flow':
			return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
			                                       GroupRandomHorizontalFlip(is_flow=True)])
