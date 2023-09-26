from itertools import product
from typing import List
from omegaconf import OmegaConf
import omegaconf.dictconfig
import omegaconf.listconfig
import time

timestamp = time.strftime("_%Y_%m%d_%H%M%S")

def omegaconf2list(opt,prefix='',sep = '.'):
    notation_list = []
    for k, v in opt.items():
        k = str(k)
        if isinstance(v, omegaconf.listconfig.ListConfig):
            notation_list.append("{}{}={}".format(prefix,k,v))
        elif isinstance(v,(float,str,int,)):
            notation_list.append("{}{}={}".format(prefix,k,v))
        elif v is None:
            notation_list.append("{}{}=~".format(prefix,k,))
        elif isinstance(v, omegaconf.dictconfig.DictConfig):
            nested_flat_list = omegaconf2list(v, prefix + k + sep,sep=sep)
            if nested_flat_list:
                notation_list.extend(nested_flat_list)
        else:
            raise NotImplementedError
    return notation_list

def omegaconf2dotlist(opt,prefix='',):
    return omegaconf2list(opt,prefix,sep='.')

def dict2dotlist_list(optdict):
    if 'PRODUCT' in optdict.keys():
        dotlist_list = PRODUCT(optdict['PRODUCT'])
    elif 'CONCAT' in optdict.keys():
        dotlist_list = CONCAT(optdict['CONCAT'])
    else:
        # is a plain opt
        dotlist = []
        for k,v in optdict.items():
            dotlist.append('{}={}'.format(k,v))
        dotlist_list = [dotlist]
    return dotlist_list

def PRODUCT(optlist):
    dotlist_list_list = []
    for opt in optlist:
        dotlist_list_list.append(dict2dotlist_list(opt))
    dottuple_list = list(product(*dotlist_list_list))
    dotlist_list = []
    for dottuple in dottuple_list:
        dotlist = []
        for dl in dottuple:
            dotlist.extend(dl)
        dotlist_list.append(dotlist)
    return dotlist_list

def CONCAT(optlist):
    dotlist_list = []
    for opt in optlist:
        dotlist_list.extend(dict2dotlist_list(opt))
    return dotlist_list

def omegaconf2dict(ofDict):
    Dict={}
    if not 'config' in str(type(ofDict)):
        return ofDict
    for key in ofDict.keys():
        Dict[key] = omegaconf2dict(ofDict[key])
    return Dict