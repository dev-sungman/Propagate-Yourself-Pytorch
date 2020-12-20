export DETECTRON2_DATASETS=/data/opensets/voc/VOCdevkit
python train_voc.py --config-file configs/pascal_voc_R_50_C4_24k.yaml --num-gpus 4 MODEL.WEIGHTS pixpro_voc.pkl
