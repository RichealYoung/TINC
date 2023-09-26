import torch
from torch import nn
import numpy as np

def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # print(f'sine_init: num_input-{num_input}')
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # print(f'first_layer_sine_init: num_input-{num_input}')
            m.weight.uniform_(-1 / num_input, 1 / num_input)
class Sine(nn.Module):
    def __init__(self, w0=30):
        super().__init__()
        self.w0 = w0

    def forward(self, input):
        return torch.sin(self.w0*input)

def ActInit(net, act):
    if act == 'Sine':
        net.apply(sine_init)
        net[0].apply(first_layer_sine_init)
        print('Param init...')
    else:
        pass

def Activation(act, w0=30):
    if act == 'Sine':
        act_fun = Sine(w0)
    elif act == 'ReLU':
        act_fun = nn.ReLU()
    elif 'LeakyReLU' in act:
        negative_slope = float(act[9:])
        act_fun = nn.LeakyReLU(negative_slope)
    elif act == 'Sigmoid':
        act_fun = nn.Sigmoid()
    elif act == 'Tanh':
        act_fun = nn.Tanh()
    return act_fun
class MLP(nn.Module):
    def __init__(self, input, output, hidden:int=64, layer:int=3, act:str='Sine', output_act:bool=True, w0=30, **kwargs):
        super().__init__()
        self.hyper = {'input':input, 'output':output, 'hidden':hidden, 'layer':layer, 'act':act, 'output_act':output_act, 'w0':w0}
        self.net=[]
        act_fun = Activation(act)
        if layer == 1:
            if output_act == False:
                self.net.append(nn.Sequential(nn.Linear(input, output)))
            else:
                self.net.append(nn.Sequential(nn.Linear(input, output), act_fun))
        else:
            self.net.append(nn.Sequential(nn.Linear(input, hidden), Activation(act, w0=w0)))
            for i in range(layer-2):
                self.net.append(nn.Sequential(nn.Linear(hidden, hidden), act_fun))
            if output_act == False:
                self.net.append(nn.Sequential(nn.Linear(hidden, output)))
            else:
                self.net.append(nn.Sequential(nn.Linear(hidden, output), act_fun))
        self.net=nn.Sequential(*self.net)
        ActInit(self.net, act)

    def forward(self, coords):
        output = self.net(coords)
        return output