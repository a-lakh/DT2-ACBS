## DATASET
The following is adapted from [Danfei Xu](https://github.com/danfeiX/scene-graph-TF-release/blob/master/data_tools/README.md). You will require follwing files aprt from actual dataset.

```
datasets/vg/imdb_1024.h5
datasets/vg/bbox_distribution.npy
datasets/vg/proposals.h5
datasets/vg/VG-SGG-dicts.json
datasets/vg/VG-SGG.h5
```

### Download:
1. Download the VG images [part1 (9 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) [part2 (5 Gb)](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). Extract these images to the file `VG_100K/`.
2. Scene graph database: [VG-SGG.h5](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5)
3. Scene graph database metadata: [VG-SGG-dicts.json](http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json)
4. RoI proposals: [proposals.h5](http://svl.stanford.edu/projects/scene-graph/dataset/proposals.h5)
5. RoI distribution: [bbox_distribution.npy](http://svl.stanford.edu/projects/scene-graph/dataset/bbox_distribution.npy)

Place all of the above into once single folder called in `datasets/vg/`. If you want to use other directory, please link it in `cfg.DATASET.PATH` of `lib/config.py`.