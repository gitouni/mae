# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
import os
from typing import Iterable, Union

import torch
import numpy as np
from PIL import Image
from torchvision.utils import make_grid
import util.misc as misc
import util.lr_sched as lr_sched
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from models_mae import MaskedAutoencoderViT
from adap_weight import aw_loss
from util.misc import NativeScalerWithGradNormCount as NativeScaler

def tensor2img(tensor:torch.Tensor, out_type=np.uint8, mean_value:Union[float, Iterable]=IMAGENET_DEFAULT_MEAN, std_value:Union[float, Iterable]=IMAGENET_DEFAULT_STD):
	'''
	Converts a torch Tensor into an image Numpy array
	Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
	Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
	'''
	n_dim = tensor.dim()
	if isinstance(mean_value, Iterable):
		mean = np.array(mean_value)[None, None, :]  # (1,1,3)
		std = np.array(std_value)[None, None, :]  # (1,1,3)
	else:
		mean = mean_value
		std = std_value
	if n_dim == 4:
		n_img = len(tensor)
		img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).cpu().detach().numpy()
		img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
	elif n_dim == 3:
		img_np = tensor.cpu().detach().numpy()
		img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
	elif n_dim == 2:
		img_np = tensor.cpu().detach().numpy()
	else:
		raise TypeError('Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
	if out_type == np.uint8:
		img_np = (img_np * std) + mean
		img_np = (np.clip(img_np, 0, 1) * 255).round()
		# Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
	return img_np.astype(out_type).squeeze()


def train_one_epoch(model: MaskedAutoencoderViT,
					data_loader: Iterable, optimizer: torch.optim.Optimizer,
					device: torch.device, epoch: int, train_dict:dict, loss_scaler:NativeScaler,
					log_writer=None,
					args=None):
	assert train_dict['method'] in args.masking_method, "train_dict['method'] must be in {}".format(args.masking_method)
	model.train(True)
	metric_logger = misc.MetricLogger(delimiter="  ")
	metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
	header = 'Epoch: [{}]'.format(epoch)
	print_freq = args.print_freq

	accum_iter = args.accum_iter

	optimizer.zero_grad()

	if log_writer is not None:
		print('log_dir: {}'.format(log_writer.log_dir))

	for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
		# we use a per iteration (instead of per epoch) lr scheduler
		if data_iter_step % accum_iter == 0:
			lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

		if train_dict['method'] == 'random_masking':
			mask_ratio = train_dict['mask_ratio']
			samples = samples.to(device)
			if not model.use_gan:
				loss, *_ = model(samples, mask_ratio=mask_ratio)
			else:
				mae_loss, _, _, disc_loss, _, _ = model(samples, mask_ratio=mask_ratio) # mae_loss, pred, mask, disc_loss, adv_loss, currupt_img
				adaptive_weight = args.disc_weight
				loss = mae_loss + adaptive_weight * disc_loss
		elif train_dict['method'] == 'fixed_masking':
			img, mask = samples
			img = img.to(device)
			mask = mask.to(device)
			if not model.use_gan:
				loss, *_ = model.fixed_forward(img, mask)
			else:
				mae_loss, _, _, disc_loss, _, _ = model.fixed_forward(img, mask) # mae_loss, pred, mask, disc_loss, adv_loss, currupt_img
				adaptive_weight = args.disc_weight
				loss = mae_loss + adaptive_weight * disc_loss

		loss_value = loss.item()

		if not math.isfinite(loss_value):
			print("Loss is {}, stopping training".format(loss_value))
			sys.exit(1)

		loss /= accum_iter
		loss_scaler(loss, optimizer, parameters=model.parameters(),
					update_grad=(data_iter_step + 1) % accum_iter == 0, retain_graph=args.use_gan)
		if (data_iter_step + 1) % accum_iter == 0:
			optimizer.zero_grad()
		if model.use_gan:
			mae_loss, _, mask, disc_loss, _, _ = model(samples.to(device), mask_ratio=args.mask_ratio)
			disc_loss_value = disc_loss.item()
			mae_loss_value = mae_loss.item()

			if not math.isfinite(disc_loss_value):
				print("Loss is {}, stopping training".format(disc_loss_value))
				sys.exit(1)

			disc_loss = disc_loss/accum_iter
			mae_loss = mae_loss/accum_iter

			loss_scaler(disc_loss, optimizer, parameters=model.parameters(),
						update_grad=(data_iter_step + 1) % accum_iter == 0, retain_graph=True)
			if (data_iter_step + 1) % accum_iter == 0:
				optimizer.zero_grad()
		torch.cuda.synchronize()

		metric_logger.update(loss=loss_value)
		if model.use_gan:
			metric_logger.update(disc_loss=disc_loss_value)
			metric_logger.update(mae_loss=mae_loss_value)
			disc_value_reduce = misc.all_reduce_mean(disc_loss_value)
			mae_value_reduce = misc.all_reduce_mean(mae_loss_value)
		lr = optimizer.param_groups[0]["lr"]
		metric_logger.update(lr=lr)

		loss_value_reduce = misc.all_reduce_mean(loss_value)
		if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
			""" We use epoch_1000x as the x-axis in tensorboard.
			This calibrates different curves when batch size changes.
			"""
			epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
			log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
			log_writer.add_scalar('lr', lr, epoch_1000x)
			if model.use_gan:
				log_writer.add_scalar('train_disc_loss', disc_value_reduce, epoch_1000x)
				log_writer.add_scalar('train_mae_loss', mae_value_reduce, epoch_1000x)


	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def val_one_epoch(model: MaskedAutoencoderViT,
					data_loader: Iterable,
					device: torch.device, epoch: int, val_dict:dict,
					log_writer=None,
					args=None):
	assert val_dict['method'] in args.masking_method, "val_dict['method'] must be in {}".format(args.masking_method)
	model.eval()
	metric_logger = misc.MetricLogger(delimiter="  ")
	header = 'Epoch: [{}]'.format(epoch)
	print_freq = args.print_freq

	accum_iter = args.accum_iter

	if log_writer is not None:
		print('log_dir: {}'.format(log_writer.log_dir))
	if hasattr(args, 'save_dir'):
		save_flag = True
		curr_save_dir = os.path.join(args.save_dir, "%03d"%epoch)
		gt_save_dir = os.path.join(curr_save_dir, 'gt')
		pred_save_dir = os.path.join(curr_save_dir, 'pred')
		mask_save_dir = os.path.join(curr_save_dir, 'mask')
		os.makedirs(curr_save_dir,exist_ok=True)
		os.makedirs(gt_save_dir, exist_ok=True)
		os.makedirs(pred_save_dir, exist_ok=True)
		os.makedirs(mask_save_dir, exist_ok=True)
		print("val save_dir:{}".format(curr_save_dir))
		save_cnt = 0
	else:
		save_flag = False
	for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
		# we use a per iteration (instead of per epoch) lr scheduler
		if val_dict['method'] == 'random_masking':
			mask_ratio = val_dict['mask_ratio']
			img:torch.Tensor = samples.to(device)
			loss, pred, mask = model(samples, mask_ratio=mask_ratio, use_gan=False)
		elif val_dict['method'] == 'fixed_masking':
			img, mask = samples
			img:torch.Tensor = img.to(device)
			mask:torch.Tensor = mask.to(device)
			loss, pred, mask = model.fixed_forward(img, mask, use_gan=False)
		if save_flag and (data_iter_step % print_freq == 0):
			pred = model.unpatchify(pred) # (N, H, W)
			mask = mask[...,None].repeat(1, 1, model.patch_size[0]*model.patch_size[1])  # (N, L) -> (N, L, D)
			mask = model.unpatchify(mask, in_chans=1) # (N, H, W)
			Image.fromarray(tensor2img(img)).save(os.path.join(gt_save_dir, '%04d.%s'%(save_cnt, args.save_fmt)))
			Image.fromarray(tensor2img(pred * mask + img * (1-mask))).save(os.path.join(pred_save_dir, '%04d.%s'%(save_cnt, args.save_fmt)))
			Image.fromarray(tensor2img(mask, mean_value=0., std_value=1.)).save(os.path.join(mask_save_dir, '%04d.%s'%(save_cnt, args.save_fmt)))
			save_cnt += 1
		
		loss_value = loss.item()
		
		torch.cuda.synchronize()
		metric_logger.update(loss=loss_value)
		loss_value_reduce = misc.all_reduce_mean(loss_value)
		if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
			""" We use epoch_1000x as the x-axis in tensorboard.
			This calibrates different curves when batch size changes.
			"""
			epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
			log_writer.add_scalar('val_loss', loss_value_reduce, epoch_1000x)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def test_one_epoch(model: MaskedAutoencoderViT,
					data_loader: Iterable,
					device: torch.device, test_dict:dict,
					log_writer=None,
					args=None):
	assert test_dict['method'] in args.masking_method, "test_dict['method'] must be in {}".format(args.masking_method)
	model.eval()
	metric_logger = misc.MetricLogger(delimiter="  ")
	print_freq = 1
	header = 'Testing: '
	if log_writer is not None:
		print('log_dir: {}'.format(log_writer.log_dir))
	if hasattr(args, 'save_dir'):
		save_flag = True
		curr_save_dir = args.save_dir
		os.makedirs(curr_save_dir,exist_ok=True)
		print("val save_dir:{}".format(curr_save_dir))
		save_cnt = 0
	else:
		save_flag = False
	for data_iter_step, (samples) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
		# we use a per iteration (instead of per epoch) lr scheduler
		if test_dict['method'] == 'random_masking':
			mask_ratio = test_dict['mask_ratio']
			img:torch.Tensor = samples.to(device)
			loss, pred, mask = model(samples, mask_ratio=mask_ratio, use_gan=False)
		elif test_dict['method'] == 'fixed_masking':
			img, raw_mask = samples
			img:torch.Tensor = img.to(device)
			mask:torch.Tensor = mask.to(device)
			loss, pred, _ = model.fixed_forward(img, mask, use_gan=False)
		if save_flag and (data_iter_step % print_freq == 0):
			pred = model.unpatchify(pred) # (N, H, W)
			if test_dict['method'] == 'random_masking':
				mask = mask[...,None].repeat(1, 1, model.patch_size[0]*model.patch_size[1])  # (N, L) -> (N, L, D)
				mask = model.unpatchify(mask, in_chans=1) # (N, H, W)
			elif test_dict['method'] == 'fixed_masking':
				mask = raw_mask  # avoid over-masking due to patchifying
			Image.fromarray(tensor2img(pred * mask + img * (1-mask))).save(os.path.join(curr_save_dir, '%04d.%s'%(save_cnt, args.save_fmt)))
			save_cnt += 1
		
		loss_value = loss.item()
		
		torch.cuda.synchronize()
		metric_logger.update(loss=loss_value)

	# gather the stats from all processes
	metric_logger.synchronize_between_processes()
	print("Averaged stats:", metric_logger)
	return {k: meter.global_avg for k, meter in metric_logger.meters.items()}