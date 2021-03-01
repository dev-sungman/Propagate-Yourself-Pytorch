# Propagate Yourself
* Paper : https://arixv.org/abs/2011.10043
* Pytorch 1.7.0, cuda 10.1
* Used 4 GPUS (V100) for training.



### TODO:

- [ ] Fix pixcontrast modules 

# Environment Settings

* **CLI**

  ```bash
  git clone https://github.com/Sungman-Cho/Propagate-Yourself-Pytorch.git
  source install_packages.sh
  ```



## Unsupervised Training (ImageNet-1K)

* PixPro training

  ```bash
  python train.py --multiprocessing-distributed --batch_size=512 --loss=pixpro
  ```


* PixContrast training

  ```bash
  python train.py --multiprocessing-distributed --batch_size=512 --loss=pixcontrast
	```




## Transfer learning

### before downstream training 

* Make your current directory `downstream`. 

* Convert a trained PixPro model to detectron2's format:

  ```bash
  python convert-pretrain-to-detectron2.py '$your_checkpoint.pth.tar' pixpro.pkl
  ```

* Convert a trained Pixcontrast model to detectorn2's format:

  ```bash
  python convert-pretrain-to-detectron2.py '$your_checkpoint.pth.tar' pixcontrast.pkl
  ```




### VOC

* Training Epochs: 24K iter

* Image size : [480,800] in train, 800 at inference.

* Backbone : R50-C4

* Training

  ```bash
  # baseline training
  source train_voc_base.sh
  
  # pixpro training
  source train_voc_pixpro.sh
  
  # pixcontrast training
  source train_voc_pixcontrast.sh
  ```



### COCO

* Followed 1x settings ([detectron2](https://github.com/facebookresearch/detectron2/blob/master/MODEL_ZOO.md)) 

* Backbone : R50-C4

* Training

  ```bash
  # baseline training
  source train_coco_base.sh
  
  # pixpro training
  source train_coco_pixpro.sh
  
  # pixcontrast training
  source train_coco_pixcontrast.sh
  ```



## Results

### VOC

|              pretrain               |    AP     |   AP50    |   AP75    |
| :---------------------------------: | :-------: | :-------: | :-------: |
|              baseline               |   53.26   | **81.17** |   58.65   |
|   ImageNet-1M, PixPro, 100epochs    | **54.11** |   79.75   | **59.73** |
| ImageNet-1M, PixContrast, 100epochs |   52.23   |   78.01   |   57.85   |

