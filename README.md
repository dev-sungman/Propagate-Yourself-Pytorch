# Kakao brain: Homework
* paper : https://arixv.org/abs/2011.10043



## TODO

- [ ] Synchronized batchnorm
- [ ] LARS optimizer
- [ ] PixContrast Loss
- [ ] Matix optimzation



## Requirements

* Dockerfile : TBD

  

## Unsupervised Training (Propagate Yourself)

This implementation only supports multi-gpu, DistributedDataParallel training.

```python
python train.py --multiprocessing-distributed --world_size=1 --rank=0 --train_path='$datapath' --batch_size=1024 
```



## Models

* ResNet-50 : 

  

## Transfer learning

### VOC
* Training Epochs: ~23 epochs
* Backbone : R50-C4

### COCO





## Evaluation



## Results



