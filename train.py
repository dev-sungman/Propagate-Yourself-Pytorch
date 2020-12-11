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
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
import torchvision
from torch.utils.data import DataLoader

import random


def main(args):
    print('[*] PixPro Pytorch')
    
    # path setting
    today = str(datetime.today()).split(' ')[0] + '_' + str(time.strftime('%H%M'))
    folder_name = '{}_{}'.format(today, args.msg)

    args.log_dir = os.path.join(args.log_dir, folder_name)
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, folder_name)
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    print('[*] log directory: ', args.log_dir)
    print('[*] checkpoint directory: ', args.checkpoint_dir)
    
    # log file
    #f = open(os.path.join(args.log_dir, 'arguments.txt'), 'w')
    #f.write(str(args))
    #f.close()
    
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    
    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()
    
    if args.multiprocessing_distributed:
        args.world_size = ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))

    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu
    
    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    
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

    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)

            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node -1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError('only DDP is supported.')
    else:
        raise NotImplementedError('only DDP is supported.')

    
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base, weight_decay=args.weight_decay)
    #base_optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    #optimizer = LARS(optimizer=base_optimizer, eps=1e-8)
    
    if args.resume:
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    dataset = PixProDataset(root=args.train_path)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = None
    
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                    num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)
    
    scaled_lr = args.lr_base * args.batch_size/256
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 300)    
    #writer = SummaryWriter(log_dir)
    writer = None
    
    _iter = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train(args, loader, model, writer, optimizer, _iter)
        scheduler.step()

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            save_name = '{}.pth.tar'.format(_iter)
            save_name = os.path.join(checkpoint_dir, save_name)
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, save_name)

        _iter += 1


def train(args, loader, model, writer, optimizer, overall_iter):
    running_loss = 0
    for _iter, (data[0], data[1]) in enumerate(loader):
        images[0], pos[0], flips[0] = data[0]
        images[1], pos[1], flips[1] = data[1]

        images[0], images[1] = images[0].cuda(args.gpu, non_blocking=True), images[1].cuda(args.gpu, non_blocking=True)
        print(images[0].shape)
        x_base, x_moment, y, pixpro_loss = model(images[0], images[1], pos[0], pos[1], flips[0], flips[1])
        
        overall_loss = pixpro_loss
        running_loss += overall_loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (_iter % args.print_freq == 0) & (_iter != 0):
            print('Overall iter: {:5d}, Loss: {:5f}'.format(overall_iter, running_loss/_iter))
            #writer.add_scalar('train loss', running_loss/_iter, overall_iter)

        ### FOR DEBUGGING (visualize) !!!
        #bm, mm = model._get_feature_position_matrix(pos[0], pos[1], (7,7))
        #inter_rect = model._get_intersection_rect(pos[0], pos[1])
        #model.draw_for_debug(pos[0], pos[1], inter_rect, images[0], images[1], bm, mm)


if __name__ == '__main__':
    argv = parse_arguments(sys.argv[1:])
    main(argv)
