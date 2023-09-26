import sys
import os
from os.path import join as opj
from os.path import dirname as opd
from typing import Dict, Union
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import random

timestamp = time.strftime("_%Y_%m%d_%H%M%S")

class MyLogger():
    def __init__(self, project_name:str, stdlog:bool=True, tensorboard:bool=True):
        self.project_dir = opj('outputs', project_name)
        self.stdlog = stdlog
        self.tensorboard = tensorboard
        # if os.path.exists(self.project_dir):
        self.project_dir += timestamp
        temp_name = self.project_dir
        for i in range(10):
            if not os.path.exists(temp_name):
                break
            temp_name = self.project_dir + '-' + str(i)
        self.project_dir = temp_name
        self.logdir = self.project_dir
        self.logger_dict:Dict[str, Union[SummaryWriter]] = {}
        if tensorboard:
            self.tensorboard_init()
        else:
            os.makedirs(self.project_dir, exist_ok=True)
        if stdlog:
            self.stdlog_init()
        self.dir_init()

    def stdlog_init(self):
        stderr_handler=open(opj(self.logdir,'stderr.log'), 'w')
        sys.stderr=stderr_handler
        
    def tensorboard_init(self,):
        self.tblogger = SummaryWriter(self.logdir)
        self.logger_dict['tblogger']=self.tblogger
    
    def dir_init(self,):
        self.compressed_dir = opj(self.project_dir, 'compressed')
        self.decompressed_dir = opj(self.project_dir, 'decompressed')
        self.script_dir = opj(self.project_dir, 'script')
        self.info_dir = opj(self.project_dir, 'info')
        os.mkdir(self.compressed_dir)
        os.mkdir(self.decompressed_dir)
        os.mkdir(self.script_dir)
        os.mkdir(self.info_dir)

    def log_metrics(self, metrics_dict: Dict[str, float], iters):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'csvlogger':
                self.logger_dict[logger_name].log_metrics(metrics_dict, iters)
                self.logger_dict[logger_name].save()
            elif logger_name == 'clearml_logger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].report_scalar(k, k, metrics_dict[k], iters)
            elif logger_name == 'tblogger':
                for k in metrics_dict.keys():
                    self.logger_dict[logger_name].add_scalar(k, metrics_dict[k], iters)

    def close(self):
        for logger_name in self.logger_dict.keys():
            if logger_name == 'tblogger':
                self.logger_dict[logger_name].close()

def reproduc(opt):
    """Make experiments reproducible
    """
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    torch.backends.cudnn.benchmark = opt['benchmark']
    torch.backends.cudnn.deterministic = opt['deterministic']