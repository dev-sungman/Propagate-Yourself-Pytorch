import os
import sys
import time
import pathlib
from datetime import datetime

from config import parse_arguments
from datasets import PixProDataset
from models.resnet import resnet50
from models.pixpro import PixPro

from tensorboardX import SummaryWriter
#from torchlars import LARS

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


def train(args, loader, model, device, writer, optimizer, criterion, log_dir, checkpoint_dir):
    for images, pos, flips in loader:
        x_base, x_moment, y = model(images[0], images[1], pos[0], pos[1], flips[0], flips[1])
        # Compute Pixel Contrastive
        # Compute PixPro
        
        ### FOR DEBUGGING!!!
        bm, mm = model._get_feature_position_matrix(pos[0], pos[1], (7,7))
        inter_rect = model._get_intersection_rect(pos[0], pos[1])
        model.draw_for_debug(pos[0], pos[1], inter_rect, images[0], images[1], bm, mm)

        raise



def main(args):
    print('[*] PixPro Pytorch')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[*] device: ', device)
    
    # path setting
    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M'))
    folder_name = '{}_{}'.format(today, args.msg)

    log_dir = os.path.join(args.log_dir, folder_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name)
    #pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    #pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    print('[*] log directory: ', log_dir)
    print('[*] checkpoint directory: ', checkpoint_dir)
    
    # log file
    #f = open(os.path.join(log_dir, 'arguments.txt'), 'w')
    #f.write(str(args))
    #f.close()
    
    dataset = PixProDataset(root=args.train_path)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    loader = DataLoader(dataset, batch_size=1, shuffle=(train_sampler is None),
                    num_workers=4, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    print('[*] build model ...')
    model = PixPro(
                encoder=resnet50, 
                dim1 = args.pcl_dim_1, 
                dim2 = args.pcl_dim_2, 
                momentum = args.encoder_momentum,
                threshold = args.threshold,
                temperature = args.T,
                sharpness = args.sharpness ,
                num_linear = args.num_linear,
                )
    
    lr_scale = args.lr_base * args.batch_size/256
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, weight_decay=args.weight_decay)
    #base_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = LARS(optimizer=base_optimizer, eps=1e-8)
    
    #writer = SummaryWriter(log_dir)
    writer = None
    criterion = None
    
    print('[*] start training ...')
    for epoch in range(args.epochs):
        train(args, loader, model, device, writer, 
                optimizer, criterion, log_dir, checkpoint_dir)
        raise
if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
