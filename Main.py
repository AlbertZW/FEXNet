from argSet import *
from processing_data.dataset import DataSet
from processing_data.transforms import *

import abc
import os
import shutil
import numpy as np
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

class MainFramework:
	def __init__(self, args):
		self.args = args

		self.datasourceConfig = DatasourceConfig(self.args.dataset, self.args.modality)
		self.classifier_filename = 'ckpt.pretrained_classif.pth.tar'
		self.classifier_path = self.datasourceConfig.save_model_path

		self.start_epoch = 0
		self.best_prec1 = 0
		self.epoch_train_losses = []
		self.epoch_train_scores = []
		self.epoch_test_losses = []
		self.epoch_test_scores = []
		# self.snapshot_pref = '_'.join((args.dataset, args.arch))

		self.processing_model = None

	@abc.abstractmethod
	def _load_model(self):
		pass

	@abc.abstractmethod
	def _apply_pretrained_model(self):
		pass

	@staticmethod
	def _load_checkpoint(model, save_model_path, file_name, load_pretrained_backbone=False):
		resume_path = os.path.join(save_model_path, file_name)
		loading_status = False
		start_epoch = 0
		best_prec1 = 0

		if os.path.isfile(resume_path):
			loading_status = True
			print(("=> loading checkpoint '{}'".format(resume_path)))
			checkpoint = torch.load(resume_path)
			checkpoint_dict = checkpoint['state_dict']

			if load_pretrained_backbone:
				model_dict = model.state_dict()

				regarded_dict = {}
				for k, v in checkpoint_dict.items():
					for curr_k in model_dict.keys():
						if k == curr_k and v.size() == model_dict[curr_k].size():
							regarded_dict[k] = v

				model_dict.update(regarded_dict)
				model.load_state_dict(model_dict)
			else:
				start_epoch = checkpoint['epoch']
				best_prec1 = checkpoint['best_prec1']
				model.load_state_dict(checkpoint_dict)
			print(("=> loaded checkpoint '{}'".format(file_name)))
		else:
			print(("=> no checkpoint found at '{}'".format(resume_path)))

		return loading_status, start_epoch, best_prec1


	def _load_data(self, train, crop_size, scale_size, train_augmentation, normalize, data_length):
		if train:
			dataset = DataSet(self.datasourceConfig.data_path,
						os.path.join(self.datasourceConfig.list_folder, self.args.train_list),
						num_segments=self.args.num_segments,
						new_length=data_length,
						modality=self.args.modality,
						image_tmpl=self.datasourceConfig.frame_format if self.args.modality in ["RGB", "RGBDiff"] else self.args.flow_prefix + "{}_{:05d}.jpg",
						transform=torchvision.transforms.Compose([
						  train_augmentation,
						  Stack(roll=(self.args.arch in ['BNInception', 'InceptionV3'])),
						  ToTorchFormatTensor(div=(self.args.arch not in ['BNInception', 'InceptionV3'])),
						  normalize,
						]))
			dataloader = torch.utils.data.DataLoader(dataset=dataset,
								batch_size=self.args.train_batch_size,
								shuffle=True,
								num_workers=self.args.workers, pin_memory=True)

		else:
			dataset = DataSet(self.datasourceConfig.data_path,
						os.path.join(self.datasourceConfig.list_folder, self.args.val_list),
						num_segments=self.args.num_segments,
						new_length=data_length,
						modality=self.args.modality,
						image_tmpl=self.datasourceConfig.frame_format if self.args.modality in ["RGB", "RGBDiff"] else self.args.flow_prefix + "{}_{:05d}.jpg",
						random_shift=False,
						transform=torchvision.transforms.Compose([
							GroupScale(int(scale_size)),
							GroupCenterCrop(crop_size),
							Stack(roll=self.args.arch == 'BNInception'),
							ToTorchFormatTensor(div=self.args.arch != 'BNInception'),
							normalize,
						]))
			dataloader = torch.utils.data.DataLoader(dataset=dataset,
								batch_size=self.args.val_batch_size,
								shuffle=False,
								num_workers=self.args.workers, pin_memory=True)

		return dataloader


	def _set_dataloader(self, crop_size, scale_size, train_augmentation, normalize, data_length):
		self.train_loader = self._load_data(train=True,
		                                    crop_size=crop_size, scale_size=scale_size,
		                                    train_augmentation=train_augmentation,
		                                    normalize=normalize, data_length=data_length,
		                                    )
		self.val_loader = self._load_data(train=False,
		                                  crop_size=crop_size, scale_size=scale_size,
		                                  train_augmentation=train_augmentation,
		                                  normalize=normalize, data_length=data_length,
		                                  )


	def _resume_training(self, cuda_model):
		# filename = '_'.join((self.snapshot_pref, self.args.modality.lower(), 'checkpoint.pth.tar'))
		filename = '_'.join((self.args.modality.lower(), 'checkpoint.pth.tar'))
		resume_path = os.path.join(self.datasourceConfig.save_model_path, filename)
		loading_status, start_epoch, best_prec1 = self._load_checkpoint(cuda_model, resume_path, filename)
		if loading_status:
			self.start_epoch = start_epoch
			self.best_prec1 = best_prec1
			self.epoch_train_losses = np.load(os.path.join(self.datasourceConfig.save_model_path, 'training_losses.npy')).tolist()
			self.epoch_train_scores = np.load(os.path.join(self.datasourceConfig.save_model_path, 'training_scores.npy')).tolist()
			self.epoch_test_losses = np.load(os.path.join(self.datasourceConfig.save_model_path, 'test_loss.npy')).tolist()
			self.epoch_test_scores = np.load(os.path.join(self.datasourceConfig.save_model_path, 'test_score.npy')).tolist()

	def _set_loss_function(self):
		# define loss function (criterion) and optimizer
		if self.args.loss_type == 'nll':
			criterion = torch.nn.CrossEntropyLoss().cuda()
		else:
			raise ValueError("Unknown loss type")
		return criterion

	def _set_optimizer(self):
		policies = self.processing_model.get_optim_policies()

		for group in policies:
			print(('group: {} has {} params, lr_mult: {}, decay_mult: {}'.format(
				group['name'], len(group['params']), group['lr_mult'], group['decay_mult'])))

		optimizer = torch.optim.SGD(policies,
		                            self.args.lr,
		                            momentum=self.args.momentum,
		                            weight_decay=self.args.weight_decay)

		return optimizer

	@staticmethod
	def _adjust_learning_rate(optimizer, epoch, arg_lr, lr_steps, arg_weight_decay):
		"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
		decay = 0.1 ** (sum(epoch >= np.array(lr_steps)))
		lr = arg_lr * decay
		decay = arg_weight_decay
		for param_group in optimizer.param_groups:
			param_group['lr'] = lr * param_group['lr_mult']
			param_group['weight_decay'] = decay * param_group['decay_mult']

	@staticmethod
	def accuracy(output, target, topk=(1,)):
		"""Computes the precision@k for the specified values of k"""
		maxk = max(topk)
		batch_size = target.size(0)

		_, pred = output.topk(maxk, 1, True, True)
		pred = pred.t()
		correct = pred.eq(target.view(1, -1).expand_as(pred))

		res = []
		for k in topk:
			correct_k = correct[:k].view(-1).float().sum(0)
			res.append(correct_k.mul_(100.0 / batch_size))
		return res

	@abc.abstractmethod
	def train(self, train_loader, model, criterion, optimizer, epoch, clip_grad, print_freq):
		pass

	@abc.abstractmethod
	def validate(self, val_loader, model, criterion, iter, print_freq):
		pass

	@staticmethod
	def save_checkpoint(state, is_best, modality, save_model_path, epoch=0, filename='checkpoint.pth.tar'):
		filename = '_'.join((modality.lower(), filename))
		file_path = os.path.join(save_model_path, filename)
		torch.save(state, file_path)
		if is_best:
			best_name = '_'.join((modality.lower(), 'model_best.pth.tar'))
			best_path = os.path.join(save_model_path, best_name)
			shutil.copyfile(file_path, best_path)

	def main(self):
		# base_model (backbone) only refers to the 2D CNN, RL backbone is always set to ResNet18
		self._load_model()

		processing_model = torch.nn.DataParallel(self.processing_model, device_ids=self.args.gpus).cuda()

		if self.args.resume:
			self._resume_training(processing_model)

		if self.args.applyPretrainedModel:
			self._apply_pretrained_model()

		cudnn.benchmark = True

		crop_size = self.processing_model.crop_size
		scale_size = self.processing_model.scale_size
		# input_mean = self.processing_model.input_mean
		# input_std = self.processing_model.input_std

		train_augmentation = self.processing_model.get_augmentation(
			flip=False if 'something' in self.args.dataset else True, type='spatial')

		# Data loading code
		normalize = IdentityTransform()

		if self.args.modality == 'RGB':
			data_length = 1
		elif self.args.modality in ['Flow', 'RGBDiff']:
			data_length = 5

		self._set_dataloader(crop_size=crop_size, scale_size=scale_size,
		                     train_augmentation=train_augmentation,
		                     normalize=normalize, data_length=data_length)

		criterion = self._set_loss_function()
		optimizer = self._set_optimizer()

		if self.args.evaluate:
			# filename = '_'.join((self.snapshot_pref, self.args.modality.lower(), 'model_best.pth.tar'))
			filename = '_'.join((self.args.modality.lower(), 'model_best.pth.tar'))
			self._load_checkpoint(self.processing_model, self.datasourceConfig.save_model_path, filename)

			self.validate(self.val_loader, processing_model, criterion, 0, self.args.print_freq)
			return

		for epoch in range(self.start_epoch, self.args.epochs):
			MainFramework._adjust_learning_rate(optimizer, epoch, self.args.lr, self.args.lr_steps, self.args.weight_decay)

			# train for one epoch
			train_accu, train_loss = self.train(self.train_loader, processing_model, criterion, optimizer, epoch,
												self.args.clip_gradient, self.args.print_freq)

			self.epoch_train_losses.append(train_loss)
			self.epoch_train_scores.append(train_accu)

			tmp_training_loss_path = os.path.join(self.datasourceConfig.save_model_path, 'tmp_training_losses.npy')
			tmp_training_score_path = os.path.join(self.datasourceConfig.save_model_path, 'tmp_training_scores.npy')
			np.save(tmp_training_loss_path, np.array(self.epoch_train_losses))
			np.save(tmp_training_score_path, np.array(self.epoch_train_scores))

			# evaluate on validation set
			if (epoch + 1) % self.args.eval_freq == 0 or epoch == self.args.epochs - 1:
				prec1, val_loss = self.validate(self.val_loader, processing_model, criterion, (epoch + 1) * len(self.train_loader),
		                                   self.args.print_freq)

				self.epoch_test_losses.append(val_loss)
				self.epoch_test_scores.append(prec1)
				np.save(os.path.join(self.datasourceConfig.save_model_path, 'test_loss.npy'), np.array(self.epoch_test_losses))
				np.save(os.path.join(self.datasourceConfig.save_model_path, 'test_score.npy'), np.array(self.epoch_test_scores))

				training_loss_path = os.path.join(self.datasourceConfig.save_model_path, 'training_losses.npy')
				training_score_path = os.path.join(self.datasourceConfig.save_model_path, 'training_scores.npy')
				shutil.copyfile(tmp_training_loss_path, training_loss_path)
				shutil.copyfile(tmp_training_score_path, training_score_path)

				# remember best prec@1 and save checkpoint
				is_best = prec1 > self.best_prec1
				if is_best:
					self.best_prec1 = prec1
				MainFramework.save_checkpoint({
					'epoch': epoch + 1,
					'arch': self.args.arch,
					'state_dict': processing_model.state_dict(),
					'best_prec1': self.best_prec1,
				#}, is_best, self.snapshot_pref, self.args.modality, self.datasourceConfig.save_model_path, epoch)
				}, is_best, self.args.modality, self.datasourceConfig.save_model_path, epoch)



if __name__ == "__main__":
	global_args = argSet(
		############################################
		# the actual parameters for somethingV1 implementation
		# dataset='somethingV1',
		# segment_num=8,
		# dropout=0.8,
		# backbone_arch='resnet50',
		# # backbone_arch='BNInception',
		# gpus=[0, 1],
		# epochs=60,
		# train_batch_size=32,
		# val_batch_size=128,
		# lr_steps=[30, 50],
		###########################################
		dataset='UCF101',
		# dataset='Kinetics400',
		segment_num=8,
		dropout=0.8,
		arch='resnet50',
		# arch='BNInception',
		gpus=[0],
		epochs=60,
		train_batch_size=2,
		val_batch_size=16,
		lr_steps=[30, 50],
		# resume=True,
		# evaluate=True,

		applyPretrainedModel=False,
	)

	import E2EMain
	mainframework = E2EMain.E2EImplementation(global_args)

	mainframework.main()

