import os
import sys
import time
import pathlib
from datetime import datetime

from config import parse_arguments
from datasets import VOCdatasets, COCOdatasets
from models.resnet import resnet50

from tensorboardX import SummaryWriter

import torch
import torch.nn as nn
import torchvision

def main(args):
    print('[*] Propagate Yourself-pytorch'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('[*] device: ', device)
    
    # path setting
    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M'))
    folder_name = '{}_{}'.format(today, args.msg)

    log_dir = os.path.join(args.log_dir, folder_name)
    checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(log_dir).mkdir(parents=True, exist_ok=True)
    print('[*] log directory: ', log_dir)
    print('[*] checkpoint directory: ', checkpoint_dir)
    
    # log file
    f = open(os.path.join(log_dir, 'arguments.txt'), 'w')
    f.write(str(args))
    f.close()
    

    print('[*] prepare datasets & dataloader ...')
    train_datasets = VOCdatasets()
    test_datasets = VOCdatasets()

    train_loader = torch.utils.data.DataLoader(train_datasets, 
                    batch_size=args.batch_size, 
                    num_workers = args.workers,
                    pin_memory=True,
                    shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(test_datasets,
                    batch_size=args.batch_size,
                    num_workers = args.workers,
                    pin_memory=True,
                    shuffle=True)
    
    print('[*] build model ...')


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
