import os
import time
import shutil
import torch
import torchvision
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.nn.utils import clip_grad_norm

from processing_data.dataset import DataSet
from processing_data.transforms import *

import Main
from models.E2S_Net import E2S_Net

# test 2D CNN (temporally TSM)
class E2EImplementation(Main.MainFramework):
	def __init__(self, args):
		super(E2EImplementation, self).__init__(args)

	def _load_model(self):
		self.processing_model = E2S_Net(
			self.datasourceConfig.num_class,
			self.args.num_segments,
			base_model=self.args.arch,
			modality=self.args.modality,
			dropout=self.args.dropout
		)

	def _apply_pretrained_model(self):
		self._load_checkpoint(self.processing_model, self.classifier_path, self.classifier_filename)

	def train(self, train_loader, model, criterion, optimizer, epoch, clip_grad, print_freq):
		batch_time = Main.AverageMeter()
		data_time = Main.AverageMeter()
		losses = Main.AverageMeter()
		top1 = Main.AverageMeter()
		top5 = Main.AverageMeter()

		# switch to train mode
		model.train()

		end = time.time()
		for i, (input, target) in enumerate(train_loader):
			# measure data loading time
			data_time.update(time.time() - end)

			target = target.cuda()
			input_var = torch.autograd.Variable(input)
			target_var = torch.autograd.Variable(target)

			# compute output
			output = model(input_var)
			loss = criterion(output, target_var)

			# measure accuracy and record loss
			prec1, prec5 = Main.MainFramework.accuracy(output.data, target, topk=(1, 5))
			losses.update(loss.item(), input.size(0))
			top1.update(prec1.item(), input.size(0))
			top5.update(prec5.item(), input.size(0))

			# compute gradient and do SGD step
			optimizer.zero_grad()

			loss.backward()

			if clip_grad is not None:
				total_norm = clip_grad_norm(model.parameters(), clip_grad)
			# if total_norm > args.clip_gradient:
			#     print("clipping gradient: {} with coef {}".format(total_norm, args.clip_gradient / total_norm))

			optimizer.step()

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % print_freq == 0:
				print(('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
				       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				       'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					epoch, i, len(train_loader), batch_time=batch_time,
					data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr'])))

		return top1.avg, losses.avg

	def validate(self, val_loader, model, criterion, iter, print_freq):
		batch_time = Main.AverageMeter()
		losses = Main.AverageMeter()
		top1 = Main.AverageMeter()
		top5 = Main.AverageMeter()

		# switch to evaluate mode
		model.eval()

		end = time.time()
		for i, (input, target) in enumerate(val_loader):
			target = target.cuda(async=True)
			input_var = torch.autograd.Variable(input, volatile=True)
			target_var = torch.autograd.Variable(target, volatile=True)

			# compute output
			with torch.no_grad():
				output = model(input_var)
				loss = criterion(output, target_var)

			# measure accuracy and record loss
			prec1, prec5 = Main.MainFramework.accuracy(output.data, target, topk=(1, 5))

			losses.update(loss.item(), input.size(0))
			top1.update(prec1.item(), input.size(0))
			top5.update(prec5.item(), input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()

			if i % print_freq == 0:
				print(('Test: [{0}/{1}]\t'
				       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				       'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				       'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				       'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
					i, len(val_loader), batch_time=batch_time, loss=losses,
					top1=top1, top5=top5)))

		print(('Testing Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
		       .format(top1=top1, top5=top5, loss=losses)))

		return top1.avg, losses.avg