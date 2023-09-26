import os
import torch
import numpy as np
import struct
from omegaconf import OmegaConf
from utils.Network import MLP
from utils.OctTree import OctTreeMLP

def save_model(model:MLP, model_path:str):
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    for i in range(len(model.net)):
        layer = model.net[i]

        weight = layer[0].weight.detach().cpu().numpy()
        weight = np.array(weight).reshape(-1)
        weight_path = os.path.join(model_path, f'{i}-W')
        with open(weight_path, 'wb') as data_file:
            data_file.write(struct.pack('f'*len(weight), *weight))
        
        bias = layer[0].bias.detach().cpu().numpy()
        bias_path = os.path.join(model_path, f'{i}-B')        
        with open(bias_path, 'wb') as data_file:
            data_file.write(struct.pack('f'*len(bias), *bias))

def load_model(model_path, hyper):
    model = MLP(**hyper)
    for i in range(len(model.net)):
        layer = model.net[i]

        weight_shape = layer[0].weight.shape
        weight_path = os.path.join(model_path, f'{i}-W')
        with open(weight_path, 'rb') as data_file:
            data = np.array(struct.unpack('f'*weight_shape[0]*weight_shape[1], data_file.read())).astype(np.float32)
            data = np.reshape(data, (weight_shape[0], weight_shape[1]))
        with torch.no_grad():
            model.net[i][0].weight.data = torch.tensor(data)

        bias_shape = layer[0].bias.shape
        bias_path = os.path.join(model_path, f'{i}-B') 
        with open(bias_path, 'rb') as data_file:
            data = np.array(struct.unpack('f'*bias_shape[0], data_file.read())).astype(np.float32)
        with torch.no_grad():
            model.net[i][0].bias.data = torch.tensor(data)
    return model

def save_tree_models(tree_mlp:OctTreeMLP, model_dir:str):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    for node in tree_mlp.node_list:
        model = node.net
        model_path = os.path.join(model_dir, f'{node.level}-{node.di}-{node.hi}-{node.wi}')
        save_model(model=model, model_path=model_path)
    opt_path = os.path.join(model_dir, 'opt.yaml')
    OmegaConf.save(tree_mlp.opt, opt_path)

def load_tree_models(model_dir:str):
    opt_path = os.path.join(model_dir, 'opt.yaml')
    opt = OmegaConf.load(opt_path)
    tree_mlp = OctTreeMLP(opt)
    for node in tree_mlp.node_list:
        hyper = node.net.hyper
        model_path = os.path.join(model_dir, f'{node.level}-{node.di}-{node.hi}-{node.wi}')
        model = load_model(model_path=model_path, hyper=hyper)
        node.net = model
    return tree_mlp