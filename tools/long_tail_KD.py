import torch
import copy

import torch.nn as nn
import torchvision as tv

class longtail_gate_res101_SGCLS_VLS_cat(nn.Module):
    '''
    Model as described in the reference paper,
    source: https://github.com/jakesnell/prototypical-networks/blob/f0c48808e496989d01db59f86d4449d7aee9ab0c/protonets/models/few_shot.py#L62-L84
    '''
    def __init__(self):
        super(longtail_gate_res101_SGCLS_VLS_cat, self).__init__()

        self.resnet = tv.models.resnet101(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 256)

        self.predicate_block = copy.deepcopy(nn.Sequential(*list(self.resnet.children())[-3:]))
        self.predicate_block[2] = nn.Linear(num_ftrs, 128)

        for child in list(self.resnet.children())[:-4]: # freeze layer1-layer2
            for param in child.parameters():
                param.requires_grad = False

        self.lbl_out = nn.Linear(256, 150)
        self.predsampled_lbl_out = nn.Linear(256, 150)

        self.emb = nn.Embedding(150, 128)

        self.norm = nn.Linear(8, 8)
        self.linear0 = nn.Linear(520, 256)
        self.linear1 = nn.Linear(256, 128)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(p=0.5)
        self.bn = nn.BatchNorm1d(256, affine=False)

        self.out= nn.Linear(128, 50)

    def forward(self, subj, obj, spa_feature, lbl, KD=0):
        if(subj==None):
            if(KD==1):    
                return self.forward_predsampled_obj(obj)   
            elif(KD==2):    
                return self.forward_both_obj(obj)  
            else:
                return self.forward_obj(obj)    
        elif(lbl==None):
            return self.forward_all(subj, obj, spa_feature)
        else:
            return self.forward_pred(subj, obj, spa_feature, lbl)

    def forward_obj(self, obj):

        # Common Block
        obj = self.resnet.conv1(obj)
        obj = self.resnet.bn1(obj)
        obj = self.resnet.relu(obj)
        obj = self.resnet.maxpool(obj)
        obj = self.resnet.layer1(obj)
        obj = self.resnet.layer2(obj)
        obj = self.resnet.layer3(obj)

        # object lbl model
        lbl_obj = self.resnet.layer4(obj)
        lbl_obj = self.resnet.avgpool(lbl_obj)
        lbl_obj = lbl_obj.view(lbl_obj.size(0), -1)
        lbl_obj = self.resnet.fc(lbl_obj)
        lbl_obj = self.relu(lbl_obj)
        lbl_obj = self.lbl_out(lbl_obj)

        return lbl_obj

    def forward_both_obj(self, obj):

        # Common Block
        obj = self.resnet.conv1(obj)
        obj = self.resnet.bn1(obj)
        obj = self.resnet.relu(obj)
        obj = self.resnet.maxpool(obj)
        obj = self.resnet.layer1(obj)
        obj = self.resnet.layer2(obj)
        obj = self.resnet.layer3(obj)

        # object lbl model
        lbl_obj = self.resnet.layer4(obj)
        lbl_obj = self.resnet.avgpool(lbl_obj)
        lbl_obj = lbl_obj.view(lbl_obj.size(0), -1)
        lbl_obj = self.resnet.fc(lbl_obj)
        lbl_obj = self.relu(lbl_obj)

        predsampled_lbl_obj = self.predsampled_lbl_out(lbl_obj)
        lbl_obj = self.lbl_out(lbl_obj)
        
        return lbl_obj, predsampled_lbl_obj    

    def forward_predsampled_obj(self, obj):

        # Common Block
        obj = self.resnet.conv1(obj)
        obj = self.resnet.bn1(obj)
        obj = self.resnet.relu(obj)
        obj = self.resnet.maxpool(obj)
        obj = self.resnet.layer1(obj)
        obj = self.resnet.layer2(obj)
        obj = self.resnet.layer3(obj)

        # object lbl model
        lbl_obj = self.resnet.layer4(obj)
        lbl_obj = self.resnet.avgpool(lbl_obj)
        lbl_obj = lbl_obj.view(lbl_obj.size(0), -1)
        lbl_obj = self.resnet.fc(lbl_obj)
        lbl_obj = self.relu(lbl_obj)
        lbl_obj = self.predsampled_lbl_out(lbl_obj)

        return lbl_obj 

    def forward_pred(self, subj, obj, spa_feature, lbl):

        # Spatial Feature
        spa_feature = self.norm(spa_feature.float())

        # Common Block
        subj = self.resnet.conv1(subj)
        subj = self.resnet.bn1(subj)
        subj = self.resnet.relu(subj)
        subj = self.resnet.maxpool(subj)
        subj = self.resnet.layer1(subj)
        subj = self.resnet.layer2(subj)
        subj = self.resnet.layer3(subj)

        obj = self.resnet.conv1(obj)
        obj = self.resnet.bn1(obj)
        obj = self.resnet.relu(obj)
        obj = self.resnet.maxpool(obj)
        obj = self.resnet.layer1(obj)
        obj = self.resnet.layer2(obj)
        obj = self.resnet.layer3(obj)

        # LBL Feature
        pred_lbl = self.drop(self.emb(lbl))

        # predicate model
        predicate_subj = self.predicate_block[:2](subj)
        predicate_subj = predicate_subj.view(predicate_subj.size(0), -1)
        predicate_subj = self.predicate_block[2](predicate_subj)
        predicate_subj = predicate_subj.view(predicate_subj.size(0), -1).float()
        predicate_subj = torch.cat((predicate_subj, pred_lbl[:, 0, :].float()), dim=1)

        predicate_obj = self.predicate_block[:2](obj)
        predicate_obj = predicate_obj.view(predicate_obj.size(0), -1)
        predicate_obj = self.predicate_block[2](predicate_obj)
        predicate_obj = predicate_obj.view(predicate_obj.size(0), -1)
        predicate_obj = torch.cat((predicate_obj, pred_lbl[:, 1, :].float()), dim=1)

        x = torch.cat((predicate_subj, predicate_obj, spa_feature), dim=1)
        x = self.relu(self.bn(self.linear0(x)))
        x = self.tanh(self.linear1(x))

        return self.out(x)

    def forward_all(self, subj, obj, spa_feature):
        # Spatial Feature
        spa_feature = self.norm(spa_feature.float())

        # Common Block
        subj = self.resnet.conv1(subj)
        subj = self.resnet.bn1(subj)
        subj = self.resnet.relu(subj)
        subj = self.resnet.maxpool(subj)
        subj = self.resnet.layer1(subj)
        subj = self.resnet.layer2(subj)
        subj = self.resnet.layer3(subj)

        obj = self.resnet.conv1(obj)
        obj = self.resnet.bn1(obj)
        obj = self.resnet.relu(obj)
        obj = self.resnet.maxpool(obj)
        obj = self.resnet.layer1(obj)
        obj = self.resnet.layer2(obj)
        obj = self.resnet.layer3(obj)

        # object lbl model
        lbl_subj = self.resnet.layer4(subj)
        lbl_subj = self.resnet.avgpool(lbl_subj)
        lbl_subj = lbl_subj.view(lbl_subj.size(0), -1)
        lbl_subj = self.resnet.fc(lbl_subj)
        lbl_subj = self.relu(lbl_subj)
        lbl_subj = self.lbl_out(lbl_subj)

        lbl_obj = self.resnet.layer4(obj)
        lbl_obj = self.resnet.avgpool(lbl_obj)
        lbl_obj = lbl_obj.view(lbl_obj.size(0), -1)
        lbl_obj = self.resnet.fc(lbl_obj)
        lbl_obj = self.relu(lbl_obj)
        lbl_obj = self.lbl_out(lbl_obj)

        score_subj, lbl_subj = torch.max(torch.softmax(lbl_subj, dim=1), dim=1)
        score_obj, lbl_obj = torch.max(torch.softmax(lbl_obj, dim=1), dim=1)

        final_lbl = torch.cat((lbl_subj.view(-1, 1), lbl_obj.view(-1, 1)), dim=1)
        score_lbl = torch.cat((score_subj.view(-1, 1), score_obj.view(-1, 1)), dim=1)

        # LBL Feature
        pred_lbl = self.drop(self.emb(final_lbl))

        # predicate model
        predicate_subj = self.predicate_block[:2](subj)
        predicate_subj = predicate_subj.view(predicate_subj.size(0), -1)
        predicate_subj = self.predicate_block[2](predicate_subj)
        predicate_subj = predicate_subj.view(predicate_subj.size(0), -1).float()
        predicate_subj = torch.cat((predicate_subj, pred_lbl[:, 0, :].float()), dim=1)

        predicate_obj = self.predicate_block[:2](obj)
        predicate_obj = predicate_obj.view(predicate_obj.size(0), -1)
        predicate_obj = self.predicate_block[2](predicate_obj)
        predicate_obj = predicate_obj.view(predicate_obj.size(0), -1)
        predicate_obj = torch.cat((predicate_obj, pred_lbl[:, 1, :].float()), dim=1)

        x = torch.cat((predicate_subj, predicate_obj, spa_feature), dim=1)
        x = self.relu(self.bn(self.linear0(x)))
        x = self.tanh(self.linear1(x))

        return score_lbl, final_lbl, self.out(x)

    def load_state_dict(self, state_dict, stage=1):
        own_state = self.state_dict()
        for name, param in state_dict.items():

            if stage == 1:
                # load resnet weights for the object branch
                own_name = 'resnet.' + name
                if name not in own_name or 'fc' in name:
                    print('Skip loading parameter {}'.format(own_name))
                else:
                    own_state[own_name].copy_(param)

                # load resnet weights for the predicate branch
                if 'layer4' in name:
                    own_name = name.replace('layer4', 'predicate_block.0')
                    own_state[own_name].copy_(param)

            elif stage == 2:
                if 'out' not in name:
                    own_state[name].copy_(param)
                else:
                    print('Skip loading parameter {}'.format(name))

            else: # testing
                own_state[name].copy_(param)

    def load_state_dict_init(self, pred_dict, obj_dict):

        own_state = self.state_dict()

        # load obj model
        print('Loading object model')
        for name, param in obj_dict.items():
            own_name = name.replace('out', 'lbl_out')
            if own_name in own_state:
                own_state[own_name].copy_(param)

        # load predicate model
        print('Loading predicate model')
        for name, param in pred_dict.items():
            own_name = name.replace('resnet.layer4', 'predicate_block.0').replace('resnet.fc', 'predicate_block.2')
            if own_name in own_state and 'layer' not in own_name:
                own_state[own_name].copy_(param)

    def unfreeze_obj(self):
        # freeze all layers except for the last layer (out)
        for name, param in self.named_parameters():
            if 'lbl_out' in name or 'layer4' in name:
                param.requires_grad = True

    def freeze_obj(self):
        # freeze all layers except for the last layer (out)
        for name, param in self.named_parameters():
            if 'lbl_out' in name or 'layer4' in name:
                param.requires_grad = False

    def freeze_layers(self, stage=1):

        # freeze layer1-2 in resnet
        for child in list(self.resnet.children())[:-4]: # training layer3+fc
            for param in child.parameters():
                param.requires_grad = False

        if stage == 2: # balanced sampling
            # freeze all layers except for the last layer (out)
            for name, param in self.named_parameters():
               if 'out' not in name:
                    param.requires_grad = False
               else:
                    print('{} is not freezed'.format(name))

    def disable_bn2(self, stage=1):

        # disable batchnorm2d in all conv layers
        for module in self.modules():
            # print(module)
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()

        # enable batchnorm in layer3, layer4 and predicate_block
        if stage == 1: # regular sampling
            for name, param in self.named_parameters():
                if 'layer4' in name or 'predicate_block' in name or 'layer3' in name:
                    param.requires_grad = True
            self.resnet.layer3.train()
            self.resnet.layer4.train()
            self.predicate_block.train()
