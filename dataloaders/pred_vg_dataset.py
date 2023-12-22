import os
import h5py, json
import numpy as np
from PIL import Image
from collections import defaultdict
from itertools import permutations

import torch
import torchvision as tv

from lib.vg_hdf5 import vg_hdf5, load_graphs
from utils.bbox_util import bbox_iou

class VGDataset_vispos(vg_hdf5):
    """
    inherit from vg_hdf5, build dataset for few-shot learning using word vector
    form of item: (x, y)
                  x - concatenation of word vector of subject and object
                  y - label of the relation
    """
    def __init__(self, cfg, split="train_frequent", transforms=None, num_im=-1, num_val_im=5000,
            filter_duplicate_rels=True, filter_non_overlap=True, filter_empty_rels=True, mode="train"):
        assert split in ["train", "train_frequent", "train_few_shot", "val", "val_frequent", "val_few_shot", "test", "test_frequent", "test_few_shot"], \
            "split must be one of [train, train_frequent, train_few_shot, val, val_frequent, val_few_shot, test, test_frequent, test_few_shot]"
        assert num_im >= -1, "the number of samples must be >= 0"
        assert mode in ["train", "test"]

        # split = 'train' if split == 'test' else 'test'
        self.data_dir = cfg.DATASET.PATH
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize((256, 256)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

        self.split = split
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and 'train' in self.split

        self.roidb_file = os.path.join(self.data_dir, "VG-SGG.h5")
        self.image_file = os.path.join(self.data_dir, "imdb_1024.h5")

        # read in dataset from a h5 file and a dict (json) file
        assert os.path.exists(self.data_dir), \
            "cannot find folder {}, please download the visual genome data into this folder".format(self.data_dir)
        self.im_h5 = h5py.File(self.image_file, 'r')
        self.info = json.load(open(os.path.join(self.data_dir, "VG-SGG-dicts.json"), 'r'))
        self.im_refs = self.im_h5['images'] # image data reference
        im_scale = self.im_refs.shape[2]

        # add background class
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                               self.class_to_ind[k])
        # cfg.ind_to_class = self.ind_to_classes

        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])
        # cfg.ind_to_predicate = self.ind_to_predicates

        self.split_mask, self.image_index, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships, self.predicates = load_graphs(
                self.roidb_file, self.image_file,
                self.split, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                filter_non_overlap=filter_non_overlap and 'train' in self.split,
            )

        self.json_category_id_to_contiguous_id = self.class_to_ind

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.img_id = []
        self.subj_id = []
        self.obj_id = []
        self.preds = []
        self.clslist = []
        pair_sets = {}

        for id in range(len(self.image_index)):
            obj_relation_triplets = self.relationships[id].copy()
            num_obj = len(self.gt_boxes[id])
            all_relations = list(permutations(range(num_obj),2))

            if self.filter_duplicate_rels:
                # Filter out dupes!
                assert 'train' in self.split
                old_size = obj_relation_triplets.shape[0]
                all_rel_sets = defaultdict(list)
                for (o0, o1, r) in obj_relation_triplets:
                    all_rel_sets[(o0, o1)].append(r)
                    pair_sets[(o0, o1)]=1
                obj_relation_triplets = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
                obj_relation_triplets = np.array(obj_relation_triplets)

            for i in range(obj_relation_triplets.shape[0]):
                if obj_relation_triplets[i][2] == 15:
                    continue
                self.img_id.append(id)
                self.subj_id.append(obj_relation_triplets[i][0])
                self.obj_id.append(obj_relation_triplets[i][1])
                self.preds.append(obj_relation_triplets[i][2])
                self.clslist.append(obj_relation_triplets[i][2])

                try:
                    all_relations.remove((obj_relation_triplets[i][0],obj_relation_triplets[i][1])) #remove valid relation
                except:
                    try:
                        all_relations.remove((obj_relation_triplets[i][1],obj_relation_triplets[i][0])) #remove valid relation
                    except:
                        pass

            if(mode=='test'):
                for pair in all_relations:
                    self.img_id.append(id)
                    self.subj_id.append(pair[0])
                    self.obj_id.append(pair[1])
                    self.preds.append(-1)

    def __len__(self):
        return len(self.img_id)
    

    def __getitem__(self, index):
        # get image
        id = self.img_id[index]
        img_idx = self.img_id[index]
        img = Image.fromarray(self._im_getter(img_idx))

        # get object bounding boxes, labels and relations
        subj_box = self.gt_boxes[img_idx][self.subj_id[index]].copy()
        subj_label = self.gt_classes[img_idx][self.subj_id[index]].copy()
        obj_box = self.gt_boxes[img_idx][self.obj_id[index]].copy()
        obj_label = self.gt_classes[img_idx][self.obj_id[index]].copy()

        subj_pos, obj_pos = tuple(subj_box), tuple(obj_box)

        subj = self.transforms(img.crop(subj_pos))
        obj = self.transforms(img.crop(obj_pos))

        subj_x1, subj_y1, subj_x2, subj_y2 = subj_pos
        subj_h, subj_w = subj_y2 - subj_y1, subj_x2 - subj_x1
        obj_x1, obj_y1, obj_x2, obj_y2 = obj_pos
        obj_h, obj_w = obj_y2 - obj_y1, obj_x2 - obj_x1

        spa = torch.tensor([
            (subj_x1 - obj_x1) / subj_w, (subj_y1 - obj_y1) / subj_h,
            (subj_x2 - obj_x2) / subj_w, (subj_y2 - obj_y2) / subj_h,
            obj_h / subj_h, obj_w / subj_w, 
            (obj_w * obj_h) / (subj_w * subj_h), 
            (obj_w + obj_h) / (subj_w + subj_h),
        ], dtype=torch.float32)
            
        lbl = torch.tensor([subj_label, obj_label, ],dtype=torch.int64)

        pred = self.preds[index]
        id = self.img_id[index]

        return (subj, obj, spa), lbl, pred, id 

class VGDataset_bbox_vispos(vg_hdf5):
    """
    inherit from vg_hdf5, build dataset for few-shot learning using word vector
    form of item: (x, y)
                  x - concatenation of word vector of subject and object
                  y - label of the relation
    """
    def __init__(self, cfg, split="train_frequent", transforms=None, num_im=-1, num_val_im=5000,
            filter_duplicate_rels=True, filter_non_overlap=True, filter_empty_rels=True, mode="train"):
        assert split in ["train", "train_frequent", "train_few_shot", "val", "val_frequent", "val_few_shot", "test", "test_frequent", "test_few_shot"], \
            "split must be one of [train, train_frequent, train_few_shot, val, val_frequent, val_few_shot, test, test_frequent, test_few_shot]"
        assert num_im >= -1, "the number of samples must be >= 0"
        assert mode in ["train", "test"]

        # split = 'train' if split == 'test' else 'test'
        self.data_dir = cfg.DATASET.PATH
        self.transforms = tv.transforms.Compose([
            tv.transforms.Resize((256, 256)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
        ])

        self.split = split
        self.filter_non_overlap = filter_non_overlap
        self.filter_duplicate_rels = filter_duplicate_rels and 'train' in self.split

        self.roidb_file = os.path.join(self.data_dir, "VG-SGG.h5")
        self.image_file = os.path.join(self.data_dir, "imdb_1024.h5")

        # read in dataset from a h5 file and a dict (json) file
        assert os.path.exists(self.data_dir), \
            "cannot find folder {}, please download the visual genome data into this folder".format(self.data_dir)
        self.im_h5 = h5py.File(self.image_file, 'r')
        self.info = json.load(open(os.path.join(self.data_dir, "VG-SGG-dicts.json"), 'r'))
        self.im_refs = self.im_h5['images'] # image data reference
        im_scale = self.im_refs.shape[2]

        # add background class
        self.info['label_to_idx']['__background__'] = 0
        self.class_to_ind = self.info['label_to_idx']
        self.ind_to_classes = sorted(self.class_to_ind, key=lambda k:
                               self.class_to_ind[k])
        # cfg.ind_to_class = self.ind_to_classes

        self.predicate_to_ind = self.info['predicate_to_idx']
        self.predicate_to_ind['__background__'] = 0
        self.ind_to_predicates = sorted(self.predicate_to_ind, key=lambda k:
                                  self.predicate_to_ind[k])
        # cfg.ind_to_predicate = self.ind_to_predicates

        self.split_mask, self.image_index, self.im_sizes, self.gt_boxes, self.gt_classes, self.relationships, self.predicates = load_graphs(
                self.roidb_file, self.image_file,
                self.split, num_im, num_val_im=num_val_im,
                filter_empty_rels=filter_empty_rels,
                filter_non_overlap=filter_non_overlap and 'train' in self.split,
            )

        self.json_category_id_to_contiguous_id = self.class_to_ind

        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()
        }

        self.img_id = []
        self.subj_id = []
        self.obj_id = []
        self.preds = []
        self.clslist = []
        pair_sets = {}

        for id in range(len(self.image_index)):
            obj_relation_triplets = self.relationships[id].copy()
            num_obj = len(self.gt_boxes[id])
            all_relations = list(permutations(range(num_obj),2))

            if self.filter_duplicate_rels:
                # Filter out dupes!
                assert 'train' in self.split
                old_size = obj_relation_triplets.shape[0]
                all_rel_sets = defaultdict(list)
                for (o0, o1, r) in obj_relation_triplets:
                    all_rel_sets[(o0, o1)].append(r)
                    pair_sets[(o0, o1)]=1
                obj_relation_triplets = [(k[0], k[1], np.random.choice(v)) for k,v in all_rel_sets.items()]
                obj_relation_triplets = np.array(obj_relation_triplets)

            for i in range(obj_relation_triplets.shape[0]):
                self.img_id.append(id)
                self.subj_id.append(obj_relation_triplets[i][0])
                self.obj_id.append(obj_relation_triplets[i][1])
                self.preds.append(obj_relation_triplets[i][2])
                self.clslist.append(obj_relation_triplets[i][2])

                try:
                    all_relations.remove((obj_relation_triplets[i][0],obj_relation_triplets[i][1])) #remove valid relation
                except:
                    try:
                        all_relations.remove((obj_relation_triplets[i][1],obj_relation_triplets[i][0])) #remove valid relation
                    except:
                        pass

            if(mode=='test'):
                for pair in all_relations:
                    sub_box = self.gt_boxes[id][pair[0]]
                    ob_box = self.gt_boxes[id][pair[1]]
                    
                    iou = bbox_iou(ob_box, sub_box)
                    if(iou==0): continue

                    self.img_id.append(id)
                    self.subj_id.append(pair[0])
                    self.obj_id.append(pair[1])
                    self.preds.append(-1)

    def __len__(self):
        return len(self.img_id)
    

    def __getitem__(self, index):
        # get image
        id = self.img_id[index]
        img_idx = self.img_id[index]
        img = Image.fromarray(self._im_getter(img_idx))

        # get object bounding boxes, labels and relations
        subj_box = self.gt_boxes[img_idx][self.subj_id[index]].copy()
        subj_label = self.gt_classes[img_idx][self.subj_id[index]].copy()
        obj_box = self.gt_boxes[img_idx][self.obj_id[index]].copy()
        obj_label = self.gt_classes[img_idx][self.obj_id[index]].copy()

        subj_pos, obj_pos = tuple(subj_box), tuple(obj_box)

        subj = self.transforms(img.crop(subj_pos))
        obj = self.transforms(img.crop(obj_pos))

        subj_x1, subj_y1, subj_x2, subj_y2 = subj_pos
        subj_h, subj_w = subj_y2 - subj_y1, subj_x2 - subj_x1
        obj_x1, obj_y1, obj_x2, obj_y2 = obj_pos
        obj_h, obj_w = obj_y2 - obj_y1, obj_x2 - obj_x1

        spa = torch.tensor([
            (subj_x1 - obj_x1) / subj_w, (subj_y1 - obj_y1) / subj_h,
            (subj_x2 - obj_x2) / subj_w, (subj_y2 - obj_y2) / subj_h,
            obj_h / subj_h, obj_w / subj_w, 
            (obj_w * obj_h) / (subj_w * subj_h), 
            (obj_w + obj_h) / (subj_w + subj_h),
        ], dtype=torch.float32)
            
        lbl = torch.tensor([subj_label, obj_label, ],dtype=torch.int64)

        pred = self.preds[index]
        id = self.img_id[index]

        return (subj, obj, spa), lbl, pred, id 
