# coding=utf-8
import gc
import sys
import os
import time
import torch
import collections
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from utils.recall_util import *

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def train_PredCls(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model for PredCls
    '''

    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    try:
        model.module.freeze_layers(stage=opt.stage) ###
    except:
        model.freeze_layers(stage=opt.stage) ###


    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    best_state = None

    criterion = nn.CrossEntropyLoss()

    if val_dataloader is not None:
        best_model_path = os.path.join(opt.experiment_root, 'PredCls_stage_0'+str(opt.stage)+'_best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'PredCls_stage_0'+str(opt.stage)+'_last_model.pth')

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        try:
            model.module.train()
            model.module.disable_bn2(stage=opt.stage) ###
        except:
            model.train()
            model.disable_bn2(stage=opt.stage) ###

        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, lbl, y, idx = batch
            y -= 1
            lbl -= 1

            subj, obj, spa_feature = x
            subj, obj, spa_feature, lbl, y = subj.to(device), obj.to(device), spa_feature.to(device), lbl.to(device), y.to(device)
            model_output = model(subj, obj, spa_feature, lbl)

            _, predicted = torch.max(model_output.data, 1)
            acc = (predicted == y).float().mean().item()*100

            loss = criterion(model_output, y)
            loss.backward()
            optim.step()

            train_loss.append(loss.detach().item())
            train_acc.append(acc)

        avg_loss = np.mean(train_loss[-len(tr_iter):])
        avg_acc = np.mean(train_acc[-len(tr_iter):])

        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc), file=sys.stderr)
        lr_scheduler.step()

        if val_dataloader is None:
            continue

        with torch.no_grad():
            val_iter = iter(val_dataloader)
            try:
                model.module.eval()
            except:
                model.eval()

            for batch in val_iter:
                x, lbl, y, idx = batch
                y -= 1
                lbl -= 1

                subj, obj, spa_feature = x
                subj, obj, spa_feature, lbl, y = subj.to(device), obj.to(device), spa_feature.to(device), lbl.to(device), y.to(device)
                model_output = model(subj, obj, spa_feature, lbl)

                _, predicted = torch.max(model_output.data, 1)
                acc = (predicted == y).float().mean().item()*100

                loss = criterion(model_output, y)
                
                val_loss.append(loss.item())
                val_acc.append(acc)

        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            save_list_to_file(os.path.join(opt.experiment_root,
                                           name + '.txt'), locals()[name])

        avg_loss = np.mean(val_loss[-len(val_iter):])
        avg_acc = np.mean(val_acc[-len(val_iter):])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)

        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_loss, avg_acc, postfix), file=sys.stderr)
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)


    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc

def test_PredCls(opt, test_dataloader, class_list, model):
    ''' 
    Test the model trained with PredCls
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    try:
        model.module.eval()
    except:
        model.eval()

    avg_acc = list()
    for epoch in range(1):
        test_iter = iter(test_dataloader)

        pred_list = {}
        gnd_list = {}

        valid_rel = 0 
        avg_acc = 0 
        t0 = time.time()
        for batch in tqdm(test_iter):
            x, lbl, y, idx = batch
            y -= 1
            lbl -= 1

            subj, obj, spa_feature = x
            subj, obj, spa_feature, lbl, y = subj.to(device), obj.to(device), spa_feature.to(device), lbl.to(device), y.to(device)

            model_output = model(subj, obj, spa_feature, lbl)

            _, acc, num, pred_list, gnd_list = eval_result_PredCls(model_output, y, idx, pred_list=pred_list, gnd_list=gnd_list, mode="test")

            valid_rel += num
            avg_acc += acc.item()
            t0 = time.time()

    avg_acc = avg_acc/valid_rel

    K_Array = [20,50,100]
    cls_recallK = list()
    mean_recallK = list()
    recallK = list()
    for k in K_Array:
        mRk, cls_recall = get_mRecall(k, pred_list, gnd_list, class_list)
        Rk = get_Recall(k, pred_list, gnd_list, class_list)
        mean_recallK.append(mRk)
        recallK.append(Rk)
        cls_recallK.append(cls_recall)

    print('Test Acc: {}'.format(avg_acc))
    print('Test mean Recall: {}'.format(mean_recallK))
    print('Test Recall: {}'.format(recallK))
    print('Class Recalls: {}'.format(cls_recallK))

    return avg_acc
