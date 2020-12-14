# Kakao brain: Homework
* paper : https://arixv.org/abs/2011.10043



## TODO

- [x] Synchronized batchnorm
- [x] LARS optimizer
- [ ] PixContrast Loss
- [x] Matix optimzation



## Requirements

* Dockerfile : TBD

  

## Unsupervised Training (Propagate Yourself)

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



## Evaluation



## Results



