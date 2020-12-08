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

import torch
import torch.nn as nn
import torchvision

def train(args, loader, model, device, writer, optimizer, criterion, log_dir, checkpoint_dir):
    
    #for (i1,i2), (p1,p2), (f1,f2) in loader:
        



def main(args)
    print('[*] Propagate Yourself-pytorch'
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
    f = open(os.path.join(log_dir, 'arguments.txt'), 'w')
    #f.write(str(args))
    f.close()

    dataset = PixProDataset(root=args.train_path)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None

    loader = DataLoader(dataset, batch_size=args.batch_size,
                    num_workers=4, pin_memory=True, sampler=train_sampler,
                    shuffle=True, drop_last=True)
    
    print('[*] build model ...')
    model = PixPro(
                encoder=resnet50, 
                dim_1 = args.pcl_dim_1, 
                dim_2 = args.pcl_dim_2, 
                momentum = args.encoder_momentum, 
                temperature = args.,
                sharpness = args.sharpness ,
                num_linear = args.num_linear,
                )
     
    for epoch in range(args.epochs):
        train(args, loader, model, device, writer, 
                optimizer, criterion, log_dir, checkpoint_dir)

if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
