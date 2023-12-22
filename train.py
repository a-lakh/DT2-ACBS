# coding=utf-8
import gc
import os
import sys
import time
import torch
import torch.nn as nn
import collections
import numpy as np
from tqdm import tqdm

from lib.config import cfg

from dataloaders.obj_vg_dataset import VGDataset_vispos as obj_dataset
from dataloaders.pred_vg_dataset import VGDataset_vispos as pred_traindataset
from tools.prototypical_batch_sampler import PrototypicalBatchSampler

from tools.long_tail_KD import longtail_gate_res101_SGCLS_VLS_cat as my_model
from utils.parser_util import get_parser
from utils.recall_util import *

from tools.PredCls import train_PredCls
from tools.SGCls import train_SGCls

def init_seed(opt):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)

def init_net(opt):
    '''
    Initialize the ProtoNet
    '''
    model = my_model()
    return model

def init_dataset(opt, mode):
    dataset = pred_traindataset(cfg, split=mode, num_val_im=0)
    n_classes = len(np.unique(dataset.preds))
    
    return dataset

def init_sampler_obj(opt, labels, itr):
    classes_per_it = opt.obj_classes_per_it
    num_samples = opt.obj_num_samples

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=itr)

def init_sampler(opt, labels, itr):
    classes_per_it = opt.pred_classes_per_it
    num_samples = opt.pred_num_samples

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=itr)

def init_dataloader_obj(opt, mode, batch_size=256):
    if(mode == "train"):
        itr = opt.iterations_train
    else:
        itr = opt.iterations_val

    dataset = obj_dataset(cfg, split=mode, num_val_im=0)
    sampler = init_sampler_obj(opt, dataset.preds, itr)
    class_list = np.unique(dataset.preds)-1

    if opt.stage == 1:
        #Normal Training
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

    else:
        #For Balanaced Sampling
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    return dataloader

def init_dataloader(opt, mode, batch_size=256):
    if(mode == "train"):
        itr = opt.iterations_train
    else:
        itr = opt.iterations_val

    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.preds, itr)
    class_list = np.unique(dataset.preds)-1

    if opt.stage == 1:
        #Normal Training
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

    else:
        #For Balanaced Sampling
        dataloader = torch.utils.data.DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    return dataloader, class_list

def init_optim(opt, model):
    '''
    Initialize optimizer
    '''
    return torch.optim.Adam(params=model.parameters(),
                            lr=opt.learning_rate)


def init_lr_scheduler(opt, optim):
    '''
    Initialize the learning rate scheduler
    '''
    return torch.optim.lr_scheduler.StepLR(optimizer=optim,
                                           gamma=opt.lr_scheduler_gamma,
                                           step_size=opt.lr_scheduler_step)


def cvt2normal_state(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = collections.OrderedDict()
    for name, param in state_dict.items():
        name = name.replace('module.', '')  # remove `module.`
        new_state_dict[name] = param
    return new_state_dict

def main():
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()
    if not os.path.exists(options.experiment_root):
        os.makedirs(options.experiment_root)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    model = init_net(options)

    ### Config ###
    if(options.stage==1):
        options.alpha = 0
        options.beta = 0
        options.type = 0

    if(options.train_task==1):
        TASK = 'PredCls'
    else:
        TASK = 'SGCls'

    print("Starting "+TASK+" Training!")
    if(options.stage==1):
        try:
            model.load_state_dict(cvt2normal_state(torch.load('./pretrained_model/resnet101_VG.pth')))	#default weights for stage-1
        except:
            print("ERROR: resnet101_VG weights not found in pretrained_model directory!")
    else:
        if(options.init_weight_path is not None):
            model.load_state_dict(torch.load(options.init_weight_path), stage=options.stage)
        else:
            try:
                model.load_state_dict(cvt2normal_state(torch.load(os.path.join(options.experiment_root, TASK+'_stage_01_best_model.pth'))), stage=options.stage)
            except:
                print("ERROR: stage-1 weights not found in experiment directory & no external stage-1 weights have been provided!")
                raise
                
    model.freeze_layers(stage=options.stage)

    if torch.cuda.is_available() and options.cuda:
        # use multiple gpus
        n_gpu = torch.cuda.device_count()
        print("Let's use", n_gpu, "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=range(n_gpu)).cuda()

    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    if(options.train_task == 1):
        t0 = time.time()
        tr_dataloader, class_list = init_dataloader(options, 'train', batch_size=options.batch_size)
        t1 = time.time()
        print("trainset load time:",t1-t0)

        t0 = time.time()
        val_dataloader, class_list = init_dataloader(options, 'val', batch_size=options.batch_size)
        t1 = time.time()
        print("valset load time:",t1-t0)

        res = train_PredCls(
                 opt=options,
                 tr_dataloader=tr_dataloader,
                 val_dataloader=val_dataloader,
                 model=model,
                 optim=optim,
                 lr_scheduler=lr_scheduler)

    elif(options.train_task == 2):
        t0 = time.time()
        tr_dataloader1, class_list = init_dataloader(options, 'train', batch_size=options.batch_size)
        tr_dataloader2 = init_dataloader_obj(options, 'train', batch_size=options.batch_size)
        t1 = time.time()
        print("trainset load time:",t1-t0)

        t0 = time.time()
        val_dataloader, class_list = init_dataloader(options, 'val', batch_size=options.batch_size)
        t1 = time.time()
        print("valset load time:",t1-t0)

        res = train_SGCls(
                 opt=options,
                 tr_dataloader1=tr_dataloader1,
                 tr_dataloader2=tr_dataloader2,
                 val_dataloader=val_dataloader,
                 model=model,
                 optim=optim,
                 lr_scheduler=lr_scheduler)

if __name__ == '__main__':
    main()
