import torch
from einops import rearrange
from typing import Tuple
import copy
import math

def create_optim(name, parameters ,lr):
    if name == 'Adam':
        optimizer = torch.optim.Adam(parameters, lr=lr)
    elif name == 'Adamax':
        optimizer = torch.optim.Adamax(parameters, lr=lr)
    elif name == 'SGD':
        optimizer = torch.optim.SGD(parameters, lr=lr)
    else:
        raise NotImplemented
    return optimizer

def create_lr_scheduler(optimizer, lr_scheduler_opt):
    lr_scheduler_opt = copy.deepcopy(lr_scheduler_opt)
    lr_scheduler_name = lr_scheduler_opt.pop('name')
    if lr_scheduler_name == 'MultiStepLR':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'CyclicLR':
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,**lr_scheduler_opt)
    elif lr_scheduler_name == 'none':
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[100000000000])
    else:
        raise NotImplementedError
    return lr_scheduler

def create_flattened_coords(coords_shape:Tuple) -> torch.Tensor:
    minimum = -1
    maximum = 1
    coords = torch.stack(torch.meshgrid(
        torch.linspace(minimum, maximum, coords_shape[0]),
        torch.linspace(minimum, maximum, coords_shape[1]),
        torch.linspace(minimum, maximum, coords_shape[2]), indexing='ij'),
    axis=-1)
    flattened_coords = rearrange(coords,'d h w c -> (d h w) c')
    return flattened_coords
class PointSampler:
    def __init__(self, data: torch.Tensor, max_level:int, batch_size: int, epochs:int, device:str='cpu') -> None:
        self.batch_size = int(batch_size/8**max_level)
        assert self.batch_size>512 and self.batch_size<2097152, "Batch size error"
        # self.batch_size = int(batch_size)
        self.epochs = epochs
        self.device = device
        assert data.shape[0]%2**max_level==0 and data.shape[1]%2**max_level==0 and data.shape[1]%2**max_level==0, f"{data.shape} can't be devided by 2^{max_level}"
        self.shape = (data.shape[0]//2**max_level, data.shape[1]//2**max_level, data.shape[2]//2**max_level)
        self.coords = create_flattened_coords(self.shape).to(device)
        self.pop_size = self.shape[0]*self.shape[1]*self.shape[2]
        self.evaled_epochs = []
    
    def judge_eval(self, eval_epoch):
        if self.epochs_count%eval_epoch==0 and self.epochs_count!=0 and not (self.epochs_count in self.evaled_epochs):
            self.evaled_epochs.append(self.epochs_count)
            return True
        elif self.index>=self.pop_size and self.epochs_count>=self.epochs-1:
            self.epochs_count = self.epochs
            return True
        else:
            return False

    def __len__(self):
        return self.epochs*math.ceil(self.pop_size/self.batch_size)

    def __iter__(self):
        self.index = 0
        self.epochs_count = 0
        return self

    def __next__(self):
        if self.index < self.pop_size:
            # sampled_idxs = self.index
            sampled_idxs = torch.randint(0, self.pop_size, (self.batch_size,))
            sampled_coords = self.coords[sampled_idxs, :]
            sampled_coords = sampled_coords.to(self.device)
            self.index += self.batch_size
            return sampled_idxs, sampled_coords
        elif self.epochs_count < self.epochs-1:
            self.epochs_count += 1
            self.index = 0
            return self.__next__()
        else:
            raise StopIteration