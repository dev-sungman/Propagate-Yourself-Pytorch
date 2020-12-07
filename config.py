import argparse

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default=None)
    
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='runs')
    parser.add_argument('--msg', type=str, default=None)

    ##### Hyperparameter
    parser.add_argument('--dataset', type=str, default=None)
    # pixel contrast threshold
    parser.add_argument('--pc_tresh', type=float, default=0.7) 
    # temperature
    parser.add_argument('--temper', type=float, default=0.3)
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
    
    # encoder momentum. it will increased to 1.
    parser.add_argument('--encoder_momentum', type=float, default=0.99)

    # embedding size
    parser.add_argument('--pcl_dim_1', type=int, default=2048)
    parser.add_argument('--pcl_dim_2', type=int, default=256)

    # loss weights
    parser.add_argument('--inst_weight', type=int, default=1)

    return parser.parse_args()






