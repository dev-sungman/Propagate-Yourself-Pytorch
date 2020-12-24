export DETECTRON2_DATASETS=/data/private
python train_coco.py --config-file configs/coco_R_50_C4_1x.yaml --num-gpus 4 MODEL.WEIGHTS pixpro.pkl
