import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    ##### Base settings
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--msg', type=str, default='')
    parser.add_argument('--print_freq', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    
    ##### Distributed Training
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:10001')
    parser.add_argument('--dist-backend', type=str, default='nccl')
    parser.add_argument('--multiprocessing-distributed', default=False, action='store_true')
    parser.add_argument('--gpu', type=int, default=None)

    ##### Training Parameter
    # image path
    parser.add_argument('--train_path', type=str, default='/data/opensets/imagenet-pytorch/train')
    parser.add_argument('--valid_path', type=str, default=None)
    # input image size
    parser.add_argument('--image_size', type=int, default=224)
    # num epochs
    parser.add_argument('--epochs', type=int, default=100)
    # batch size
    parser.add_argument('--batch_size', type=int, default=1024)
    # initial learning 
    parser.add_argument('--lr_base', type=float, default=1.0)
    # weight decay
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    # loss weights (for combining with instance contrast)
    parser.add_argument('--inst_weight', type=int, default=1)
    parser.add_argument('--workers', type=int, default=48)
    
    ##### Encoder + Projection
    # temperature
    parser.add_argument('--T', type=float, default=0.3)
    # encoder momentum. it will increased to 1.
    parser.add_argument('--encoder_momentum', type=float, default=0.99)

    # embedding size
    parser.add_argument('--pcl_dim_1', type=int, default=2048)
    parser.add_argument('--pcl_dim_2', type=int, default=256)
    
    ##### Pixel contrast
    # pixel contrast threshold
    parser.add_argument('--threshold', type=float, default=0.7) 


    ##### Pixel Propagation Module
    # sharpness
    parser.add_argument('--sharpness', type=int, default=2)
    # number of linear layer
    parser.add_argument('--num_linear', type=int, default=1)

    return parser.parse_args()






