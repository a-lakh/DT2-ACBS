from yacs.config import CfgNode as CN

cfg = CN()

""""======================================="""
cfg.DATASET = CN()
cfg.DATASET.NAME = "vg"
cfg.DATASET.MODE = "benchmark"                    # dataset mode, benchmark | 1600-400-400 | 2500-600-400, etc
cfg.DATASET.PATH = "./datasets/vg"
cfg.DATASET.LOADER = 'object'                     # which kind of data loader to use, object | object+attribute | object+attribute+relationship
""""======================================="""
