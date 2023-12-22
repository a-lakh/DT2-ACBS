import os
import h5py, json
import numpy as np
from PIL import Image
from collections import defaultdict

import torch
import torchvision as tv

from lib.vg_hdf5 import vg_hdf5, load_graphs

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
        self.obj_id = []
        self.preds = []

        for id in range(len(self.image_index)):
            for i in range(len(self.gt_boxes[id])):
                self.img_id.append(id)
                self.obj_id.append(i)
                self.preds.append(self.gt_classes[id][i])

    def __len__(self):
        return len(self.img_id)

    def __getitem__(self, index):
        # get image
        img_idx = self.img_id[index]
        img = Image.fromarray(self._im_getter(img_idx))

        # get object bounding boxes, labels and relations
        obj_box = self.gt_boxes[img_idx][self.obj_id[index]].copy()

        obj_pos = tuple(obj_box)
        obj = self.transforms(img.crop(obj_pos))

        pred = self.gt_classes[img_idx][self.obj_id[index]].copy()

        return obj, pred
