# Kakao brain: Homework
* paper : https://arixv.org/abs/2011.10043



## Requirements

* Using shared image (brain cloud) : Pixpro_pytorch (To be updated) 

* Using pip 

  ```bash
  pip install -r requirements.txt
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

* Make your current directory downstream. 

* Convert a pre-trained PixPro model to detectron2's format:

  ```bash
  python convert-pretrain-to-detectron2.py '$input.pth.tar' pixpro.pkl
  ```

  

### VOC

* Training Epochs: 24K iter

* Image size : [480,800] in train, 800 at inference.

* Backbone : R50-C4

* training

  ```bash
  source train_voc.sh
  ```



### COCO

* Training Epochs: 24K iter

* Image size : [640,800] in train, 800 at inference.

* Backbone : R50-C4

* Training

  ```bash
  source train_coco.sh
  ```

  

## Results

|              pretrain               | Downstream |  AP   | AP50  | AP75  |
| :---------------------------------: | :--------: | :---: | :---: | :---: |
|   ImageNet-1M, PixPro, 100epochs    |    VOC     | 54.11 | 79.75 | 59.73 |
| ImageNet-1M, PixContrast, 100epochs |    VOC     | 52.23 | 78.01 | 57.85 |
|   ImageNet-1M, PixPro, 100epochs    |    COCO    |       |       |       |

