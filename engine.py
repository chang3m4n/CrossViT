# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Train and eval functions used in main.py

Mostly copy-paste from https://github.com/facebookresearch/deit/blob/main/engine.py
"""

import math
from typing import Iterable, Optional
import torch

from timm.data import Mixup
from timm.utils import accuracy
from einops import rearrange

import utils


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None,
                    world_size: int = 1, distributed: bool = False, amp=True,
                    finetune=False
                    ):
    if finetune:
        model.train(not finetune)
    else:
        model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20  # 增加打印频率以提供更频繁的进度更新

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        batch_size = targets.size(0)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast(enabled=amp):
            outputs = model(samples)
            loss = criterion(outputs, targets)
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                raise ValueError("Loss is {}, stopping training".format(loss_value))

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order

            if amp:
                loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)
            else:
                loss.backward(create_graph=is_second_order)
                if max_norm is not None and max_norm != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                optimizer.step()

        # 移除同步以加速训练，单GPU环境下通常不需要

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # 单GPU环境下不需要进程间同步
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, world_size=1, distributed=False, amp=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # 移除收集结果的列表，直接计算指标

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        # compute output
        with torch.cuda.amp.autocast(enabled=amp):
            output = model(images)

        # 单GPU环境下直接计算准确率
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_size = target.shape[0]
        loss = criterion(output, target)
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(acc5=acc5.item())
    
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def concat_all_gather(tensor):
    # 保留此函数以保持代码兼容性，但在单GPU环境下它不会被调用
    return tensor
