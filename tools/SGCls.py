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

def train_SGCls(opt, tr_dataloader1, tr_dataloader2, model, optim, lr_scheduler, val_dataloader=None):
    '''
    Train the model for SGCls with the ACBS algorithm
    '''

    device = 'cuda' if torch.cuda.is_available() and opt.cuda else 'cpu'

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    val_obj_acc = []

    best_acc = 0
    best_state = None

    train_o_acc = []
    train_o_loss = []
    train_p_loss = []
    KD_loss = []
    val_obj_acc = []

    criterion = nn.CrossEntropyLoss()

    if val_dataloader is not None:
        best_model_path = os.path.join(opt.experiment_root, 'SGCls_stage_0'+str(opt.stage)+'_best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'SGCls_stage_0'+str(opt.stage)+'_last_model.pth')

    obj_based_wt = None
    predicate_based_wt = None

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter1 = iter(tr_dataloader1)
        tr_iter2 = iter(tr_dataloader2)

        try:
            model.module.train()
            model.module.disable_bn2(stage=opt.stage) ###
        except:
            model.train()
            model.disable_bn2(stage=opt.stage) ###

        for batch in tqdm(tr_iter1):

            optim.zero_grad()
            x, lbl, y, idx = batch
            y -= 1
            lbl -= 1

            subj, obj, spa_feature = x
            subj, obj, spa_feature, lbl, y = subj.to(device), obj.to(device), spa_feature.to(device), lbl.to(device), y.to(device)
            if(opt.type==0):
                model_output = model(subj, obj, spa_feature, lbl)
            else:
                lbl_score, lbl_pred, model_output = model(subj, obj, spa_feature, None)

            _, predicted = torch.max(model_output.data, 1)
            acc = (predicted == y).float().mean().item()*100

            pred_loss = criterion(model_output, y)

            obj_x = torch.cat((subj,obj), axis=0)
            obj_y = lbl.reshape(-1)
            del subj, obj; gc.collect()

            predsampled_model_output = model(None,obj_x,None,None,KD=1)

            _, obj_predicted = torch.max(predsampled_model_output.data, 1)
            obj_acc = (obj_predicted == obj_y).float().mean().item()*100

            obj_loss = criterion(predsampled_model_output, obj_y)
            loss = pred_loss + opt.beta * obj_loss

            loss.backward()
            optim.step()

            train_loss.append(loss.detach().item())
            train_acc.append(acc)
            train_o_acc.append(obj_acc)
            train_p_loss.append(pred_loss.detach().item())
            train_o_loss.append(obj_loss.detach().item())

        ###################
        predicate_based_wt = torch.cat((model.module.predsampled_lbl_out.bias.detach().reshape((150,1)), model.module.predsampled_lbl_out.weight.detach()), dim=1)
        avg_loss = np.mean(train_loss[-len(tr_iter1):])
        avg_acc = np.mean(train_acc[-len(tr_iter1):])
        avg_o_acc = np.mean(train_o_acc[-len(tr_iter1):])
        avg_p_loss = np.mean(train_p_loss[-len(tr_iter1):])
        avg_o_loss = np.mean(train_o_loss[-len(tr_iter1):])
 
        print('pred Avg Train Loss: {}, Avg Train Acc: {}, Avg Train (obj) Acc:{}, Avg Train (obj) Loss:{}, Total Loss:{}'.format(avg_p_loss, avg_acc, avg_o_acc, avg_o_loss, avg_loss), file=sys.stderr)

        for batch in tqdm(tr_iter2):
            optim.zero_grad()
            x, y = batch
            y -= 1

            x, y = x.to(device), y.to(device)

            if predicate_based_wt is not None:
                model_output, predsampled_model_output = model(None,x,None,None, KD=2)
                #teacher_model_output = predsampled_object_classifier
            else:
                model_output = model(None,x,None,None)    

            _, predicted = torch.max(model_output.data, 1)

            try:
                acc = (predicted == y).float().mean().item()*100
            except:
                acc = 100 #counter check for the case when it is empty

            loss = criterion(model_output, y)
            
            ##### use KD #######
            if (not opt.non_soft_label_kd) and (predicate_based_wt is not None):  ### soft label
                current_logits = model_output
                teacher_logits = predsampled_model_output    
                soft_log_probs = torch.nn.functional.log_softmax(current_logits / opt.temperature, dim=1)
                soft_targets = torch.nn.functional.softmax(teacher_logits / opt.temperature, dim=1)
                distil_loss = torch.nn.functional.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean')
                distil_loss = distil_loss * opt.temperature ** 2 ## scale loss
            else: ## weight regularizer
                object_based_wt = torch.cat((model.module.lbl_out.bias.reshape((150,1)), model.module.lbl_out.weight), dim=1)
                if predicate_based_wt is None:
                    distil_loss = 0
                else:
                    distil_loss = torch.norm(object_based_wt - predicate_based_wt)
                    
            loss = loss + opt.alpha * distil_loss
            ############

            loss.backward()
            optim.step()

            train_loss.append(loss.detach().item())
            train_acc.append(acc)
            if predicate_based_wt is None:
                KD_loss.append(distil_loss)
            else:
                KD_loss.append(distil_loss.detach().item())

        avg_loss = np.mean(train_loss[-len(tr_iter2):])
        avg_acc = np.mean(train_acc[-len(tr_iter2):])
        avg_kd_loss = np.mean(KD_loss[-len(tr_iter2):])

        print('obj Avg Train Loss: {}, Avg Train Acc: {}, Avg KD Loss: {}'.format(avg_loss, avg_acc, avg_kd_loss), file=sys.stderr)

        lr_scheduler.step()

        if val_dataloader is None:
            continue

        val_iter = iter(val_dataloader)
        with torch.no_grad():
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

                lbl_score, lbl_pred, model_output = model(subj, obj, spa_feature, None)

                _, predicted = torch.max(model_output.data, 1)
                pred_acc = (predicted == y).float()
                lbl_acc = (lbl_pred == lbl).float()
                acc = (lbl_acc[:,0]*lbl_acc[:,1]*pred_acc).mean().item()	# accuracy is computed as each triplet being predicted correctly

                obj_acc = lbl_acc.mean().item()

                loss = criterion(model_output, y)

                val_loss.append(loss.item())
                val_acc.append(acc)
                val_obj_acc.append(obj_acc)

        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            save_list_to_file(os.path.join(opt.experiment_root,
                                           name + '.txt'), locals()[name])

        avg_loss = np.mean(val_loss[-len(val_iter):])
        avg_acc = np.mean(val_acc[-len(val_iter):])
        avg_obj_acc = np.mean(val_obj_acc[-len(val_iter):])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(best_acc)

        print('Avg Val Obj Acc: {}, Avg Val Loss: {}, Avg Val Acc: {}{}'.format(avg_obj_acc, avg_loss, avg_acc, postfix), file=sys.stderr)
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()

    torch.save(model.state_dict(), last_model_path)

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc

def test_SGCls(opt, test_dataloader, class_list, model):
    ''' 
    Test the model trained for SGCls 
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'

    avg_acc = list()
    for epoch in range(1):
        test_iter = iter(test_dataloader)

        pred_list = {}
        gnd_list = {}

        valid_rel = 0 
        avg_acc = 0 

        try:
            model.module.eval()
        except:
            model.eval()


        for batch in tqdm(test_iter):
            x, lbl, y, idx = batch
            y -= 1
            lbl -= 1

            subj, obj, spa_feature = x
            subj, obj, spa_feature, lbl, y = subj.to(device), obj.to(device), spa_feature.to(device), lbl.to(device), y.to(device)

            lbl_score, lbl_pred, model_output = model(subj, obj, spa_feature, None)

            _, acc, num, pred_list, gnd_list = eval_result_SGCls(model_output, y, lbl, lbl_score, lbl_pred, idx, pred_list=pred_list, gnd_list=gnd_list, mode="test")

            valid_rel += num
            avg_acc += acc.item()

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
