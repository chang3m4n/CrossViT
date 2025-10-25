# Copyright IBM All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Misc functions, including distributed helpers.

Mostly copy-paste from https://github.com/facebookresearch/deit/blob/main/utils.py
"""

import io
import os
import time
from collections import defaultdict, deque
import datetime

import torch
import torch.distributed as dist


import io
import os
import time
from collections import defaultdict, deque
import datetime
import tempfile
import logging

import torch
import torch.distributed as dist
from fvcore.common.checkpoint import Checkpointer


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """在单GPU环境下不需要同步，保留函数以保持兼容性"""
        pass  # 单GPU环境下不需要聚合统计信息

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    torch.save(checkpoint, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)


def load_checkpoint(model, state_dict, mode=None):
    """
    直接加载模型权重，避免使用临时文件，并处理分类头层不匹配问题
    """
    # 检查state_dict是否已经是一个包含'model'键的字典
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    
    # 提取模型当前的参数名
    model_dict = model.state_dict()
    
    # 创建一个新的state_dict，忽略分类头层(head)的参数不匹配
    new_state_dict = {}
    skipped_head_params = []
    
    for k, v in state_dict.items():
        # 检查参数名是否包含'head'，如果是则跳过或处理
        if 'head' in k:
            # 检查当前模型是否有相同的键
            if k in model_dict:
                # 检查形状是否匹配
                if v.shape == model_dict[k].shape:
                    new_state_dict[k] = v
                else:
                    skipped_head_params.append(k)
            else:
                skipped_head_params.append(k)
        else:
            # 对于非head层的参数，直接添加（如果存在于模型中）
            if k in model_dict:
                new_state_dict[k] = v
    
    # 记录跳过的head参数
    if skipped_head_params:
        print(f"Warning: Skipped mismatched head parameters: {skipped_head_params}")
        print(f"Model expects {model_dict['head.0.weight'].shape[0]} classes, but checkpoint has {state_dict['head.0.weight'].shape[0]} classes")
    
    # 更新模型的state_dict，只使用匹配的参数
    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)
    
    print("Successfully loaded model weights (excluding mismatched head parameters)")
    print(f"Loaded {len(new_state_dict)} parameters out of {len(state_dict)} total in checkpoint")

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    """单GPU环境下总是返回False"""
    return False


def get_world_size():
    """单GPU环境下总是返回1"""
    return 1


def get_rank():
    """单GPU环境下总是返回0"""
    return 0


def is_main_process():
    """单GPU环境下总是为主进程"""
    return True


def save_on_master(*args, **kwargs):
    """单GPU环境下直接保存模型"""
    torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # 强制禁用分布式模式，适应单GPU环境
    print('Not using distributed mode (single GPU)')
    args.distributed = False
    args.rank = 0
    args.world_size = 1
    args.gpu = 0
    return
