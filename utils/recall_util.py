import torch
import numpy as np
from torch.nn import functional as F
from torch.nn.modules import Module


def get_Recall(k, pred_list, gnd_list, class_list):

    Recalls = list()
    for ki in pred_list.keys():
        pred_list[ki] = sorted(pred_list[ki], key=lambda tup: tup[0], reverse=True)
    
    num_cls_imgs = 0
    cls_recall = 0

    for ki in gnd_list.keys():

        img_pred_list = [a_tuple[1] for a_tuple in pred_list[ki]]
        if(len(img_pred_list)>=k):
            topk_pred_list = img_pred_list[:int(k)]
        else:
            topk_pred_list = img_pred_list
                
        cls_idx = np.where(np.asarray(gnd_list[ki]) >= 0)[0]
        match_idx = np.where(np.asarray(topk_pred_list) >= 0)[0]

        if(len(cls_idx)>0):
            num_cls_imgs += 1
            cls_recall += float(min(len(match_idx),len(cls_idx))) / float(len(cls_idx))
        
    if(num_cls_imgs>0):
        cls_recall /= num_cls_imgs

    return cls_recall

def get_mRecall(k, pred_list, gnd_list, class_list):

    Recalls = list()
    for ki in pred_list.keys():
        pred_list[ki] = sorted(pred_list[ki], key=lambda tup: tup[0], reverse=True)
    
    for cls in class_list:
        num_cls_imgs = 0
        cls_recall = 0

        for ki in gnd_list.keys():

            img_pred_list = [a_tuple[1] for a_tuple in pred_list[ki]]
            if(len(img_pred_list)>=k):
                topk_pred_list = img_pred_list[:int(k)]
            else:
                topk_pred_list = img_pred_list
                
            cls_idx = np.where(np.asarray(gnd_list[ki]) == int(cls))[0]
            match_idx = np.where(np.asarray(topk_pred_list) == int(cls))[0]

            if(len(cls_idx)>0):
                num_cls_imgs += 1
                cls_recall += float(min(len(match_idx),len(cls_idx))) / float(len(cls_idx))
        
        if(num_cls_imgs>0):
            cls_recall /= num_cls_imgs

        Recalls.append(cls_recall)
    return np.mean(Recalls), Recalls

def eval_result_PredCls(input, target, idx, pred_list=None, gnd_list=None, mode="train"):
    if type(idx) is np.ndarray:
        idx = torch.from_numpy(idx)

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    idx_cpu = idx.to('cpu')

    if(mode=='train'):
        _, y_hat = torch.max(input_cpu.data, 1)
        acc_val = y_hat.eq(target_cpu).float().mean()
        loss_val = -1
    else:
        y_prob, y_hat = torch.max(input_cpu.data, 1)
        valid_idx = np.where(target_cpu != -1)[0]

        num = valid_idx.shape[0]
        acc_val = y_hat.eq(target_cpu).float()[valid_idx].sum()
        loss_val = -1

        if(gnd_list!=None):
            for it in range(list(idx_cpu.size())[0]):
                img_idx = idx_cpu[it].detach().numpy()

                if str(img_idx) not in pred_list.keys():
                    pred_list[str(img_idx)]=list()
                if str(img_idx) not in gnd_list.keys():
                    gnd_list[str(img_idx)]=list()

                if(int(target_cpu[it].detach().numpy())!=-1):
                    gnd_list[str(img_idx)].append(int(target_cpu[it].detach().numpy()))

                if(int(target_cpu[it].detach().numpy())==y_hat[it].detach().numpy()):
                    pred_list[str(img_idx)].append((float(y_prob[it].detach().numpy()), y_hat[it].detach().numpy()))
                else:
                    pred_list[str(img_idx)].append((float(y_prob[it].detach().numpy()), -1))
                
    if(gnd_list==None):
        return loss_val, acc_val, num 
    else:
        return loss_val, acc_val, num, pred_list, gnd_list

def eval_result_SGCls(input, target, lbl, lbl_score, lbl_pred, idx, pred_list=None, gnd_list=None, mode="train"):
    if type(idx) is np.ndarray:
        idx = torch.from_numpy(idx)

    lbl_pred_cpu = lbl_pred.to('cpu')
    lbl_cpu = lbl.to('cpu')
    lbl_score_cpu = lbl_score.to('cpu')

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    idx_cpu = idx.to('cpu')

    if(mode=='train'):
        _, y_hat = torch.max(input_cpu.data, 1)
        acc_val = y_hat.eq(target_cpu).float().mean()
        loss_val = -1
    else:
        y_prob, y_hat = torch.max(input_cpu.data, 1)
        valid_idx = np.where(target_cpu != -1)[0]

        num = valid_idx.shape[0]
        acc_val = y_hat.eq(target_cpu).float()[valid_idx].sum()
        loss_val = -1

        if(gnd_list!=None):
            for it in range(list(idx_cpu.size())[0]):
                img_idx = idx_cpu[it].detach().numpy()

                if str(img_idx) not in pred_list.keys():
                    pred_list[str(img_idx)]=list()
                if str(img_idx) not in gnd_list.keys():
                    gnd_list[str(img_idx)]=list()

                if(int(target_cpu[it].detach().numpy())!=-1):
                    gnd_list[str(img_idx)].append(int(target_cpu[it].detach().numpy()))
                
                if(torch.sum(lbl_cpu[it]==lbl_pred_cpu[it]) and int(target_cpu[it].detach().numpy())==y_hat[it].detach().numpy()):
                    pred_list[str(img_idx)].append((float(y_prob[it].detach().numpy()), y_hat[it].detach().numpy()))
                else:
                    pred_list[str(img_idx)].append((float(y_prob[it].detach().numpy()), -1))

    if(gnd_list==None):
        return loss_val, acc_val, num 
    else:
        return loss_val, acc_val, num, pred_list, gnd_list

def eval_result_all_prob_SGCls(input, target, lbl, lbl_score, lbl_pred, idx, pred_list=None, gnd_list=None, mode="train"):
    if type(idx) is np.ndarray:
        idx = torch.from_numpy(idx)

    lbl_pred_cpu = lbl_pred.to('cpu')
    lbl_cpu = lbl.to('cpu')
    lbl_score_cpu = lbl_score.to('cpu')

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    idx_cpu = idx.to('cpu')

    if(mode=='train'):
        _, y_hat = torch.max(input_cpu.data, 1)
        acc_val = y_hat.eq(target_cpu).float().mean()
        loss_val = -1
    else:
        y_prob, y_hat = torch.max(input_cpu.data, 1)
        valid_idx = np.where(target_cpu != -1)[0]

        num = valid_idx.shape[0]
        acc_val = y_hat.eq(target_cpu).float()[valid_idx].sum()
        loss_val = -1

        if(gnd_list!=None):
            for it in range(list(idx_cpu.size())[0]):
                img_idx = idx_cpu[it].detach().numpy()

                if str(img_idx) not in pred_list.keys():
                    pred_list[str(img_idx)]=list()
                if str(img_idx) not in gnd_list.keys():
                    gnd_list[str(img_idx)]=list()

                if(int(target_cpu[it].detach().numpy())!=-1):
                    gnd_list[str(img_idx)].append(int(target_cpu[it].detach().numpy()))

                if(torch.sum(lbl_cpu[it]==lbl_pred_cpu[it]) and int(target_cpu[it].detach().numpy())==y_hat[it].detach().numpy()):
                    pred_list[str(img_idx)].append((float(y_prob[it].detach().numpy()*lbl_score[it][1].to('cpu').detach().numpy()*lbl_score[it][0].to('cpu').detach().numpy()), y_hat[it].detach().numpy()))
                else:
                    pred_list[str(img_idx)].append((float(y_prob[it].detach().numpy()*lbl_score[it][1].to('cpu').detach().numpy()*lbl_score[it][0].to('cpu').detach().numpy()), -1))
                
    if(gnd_list==None):
        return loss_val, acc_val, num 
    else:
        return loss_val, acc_val, num, pred_list, gnd_list

