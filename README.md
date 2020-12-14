# Kakao brain: Homework
* paper : https://arixv.org/abs/2011.10043



## TODO

- [x] Synchronized batch-norm
- [x] LARS optimizer
- [x] Matrix Optimization (Fast training)
- [ ] PixContrast Loss



## Requirements

* Dockerfile : TBD

  

## Unsupervised Training 

```python
python train.py --multiprocessing-distributed --world_size=1 --rank=0 --train_path='$datapath' --batch_size=1024 
```



## Models

* ResNet-50 : 

  

## Transfer learning

### VOC
* Training Epochs: ~23 epochs (24K iter)
* Image size : [480,800] in train, 800 at inference.
* Backbone : R50-C4



1. Convert a pre-trained PixPro model to detectron2's format:

   ```bash
   python downstream/convert-pretrain-to-detectron2.py '$input.pth.tar' pixpro_voc.pkl
   ```

   

2. Set a path

   ```bash
   export DETECTRON2_DATASETS=/data/opensets/voc/VOCdevkit
   ```

   

3. VOC training

   ```bash
   python downstream/train_downstream.py --config-file downstream/configs/pascal_voc_R_50_C4_24k.yaml --num-gpus 4 MODEL.WEIGHTS downstream/pixpro_voc.pkl
   ```

   

   

## Evaluation





## Results

| pretrain                       | AP50 | AP   | AP75 |
| ------------------------------ | ---- | ---- | ---- |
| ImageNet-1M, PixPro, 100epochs |      |      |      |

