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

from dataloaders.pred_vg_dataset import VGDataset_bbox_vispos as pred_testdataset
from tools.prototypical_batch_sampler import PrototypicalBatchSampler

from tools.long_tail_KD import longtail_gate_res101_SGCLS_VLS_cat as my_model
from utils.parser_util import get_parser_test
from utils.recall_util import *

from tools.PredCls import test_PredCls
from tools.SGCls import test_SGCls

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

def test_dataloader(opt, mode, load_opt, batch_size=256):

    dataset = pred_testdataset(cfg, split=mode, num_val_im=0, mode=load_opt)
    class_list = np.unique(dataset.clslist)-1

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=4)

    return dataloader, class_list

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
    Initialize everything and test
    '''
    options = get_parser_test().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    model = init_net(options)

    if(options.test_task==1):
        TASK = 'PredCls'
    elif(options.test_task==2):
        TASK = 'SGCls'
    else:
        TASK = 'SGDet'

    if(options.init_weight_path is not None):
        model.load_state_dict(cvt2normal_state(torch.load(options.init_weight_path)), stage=3)
    else:
        try:
            model.load_state_dict(cvt2normal_state(torch.load(os.path.join(options.experiment_root, TASK+'_stage_02_best_model.pth'))), stage=3)
        except:
            print("ERROR: stage-2 weights not found in experiment directory & no external stage-2 weights have been provided!")
            raise

    # Evaluation is done with stage=3
    model.freeze_layers(stage=3)

    if torch.cuda.is_available() and options.cuda:
        # use multiple gpus
        n_gpu = torch.cuda.device_count()
        print("Let's use", n_gpu, "GPUs!")
        model = torch.nn.DataParallel(model, device_ids=range(n_gpu)).cuda()


    if(options.test_task == 1):
        print("Evaluating PredCls Model")
        t0 = time.time()
        tst_dataloader, class_list = test_dataloader(options, 'test', "test", batch_size=options.batch_size)
        t1 = time.time()
        print("testset load time:",t1-t0)
        with torch.no_grad():
            test_PredCls(opt=options,
                 test_dataloader=tst_dataloader,
                 class_list=class_list,
                 model=model)
    elif(options.test_task == 2):
        print("Evaluating SGCls Model")
        t0 = time.time()
        tst_dataloader, class_list = test_dataloader(options, 'test', "test", batch_size=options.batch_size)
        t1 = time.time()
        print("testset load time:",t1-t0)
        with torch.no_grad():
            test_SGCls(opt=options,
                 test_dataloader=tst_dataloader,
                 class_list=class_list,
                 model=model)
    elif(options.test_task == 3):
        print("Evaluating SGDet Model")
        #### Will be Updated Soon!
    else:
        print("Pick the a valid Task for Testing {1,2,3}")

if __name__ == '__main__':
    main()
