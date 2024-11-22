import argparse
import logging
import os
import random
import importlib

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from trainer import trainer_custom  # Changed to import the trainer for CustomDataset
from config import get_config

from model.swin_deeplab import SwinDeepLab

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='C:\\Users\\nazia\\Lab\\Dataset\\CropandWeed\\train', help='root dir for data')  # Updated for CustomDataset
parser.add_argument('--config_file', type=str, default='swin_224_7_a', help='config file name w/o suffix')
parser.add_argument('--dataset', type=str, default='CustomDataset', help='dataset name')  # Changed to CustomDataset
parser.add_argument('--num_classes', type=int, default=3, help='number of output classes')  # Set num_classes to 2
parser.add_argument('--output_dir', type=str, default='.', help='output directory')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum iterations for training')
parser.add_argument('--max_epochs', type=int, default=150, help='maximum epochs to train')
parser.add_argument('--batch_size', type=int, default=8, help='batch size per GPU')
parser.add_argument('--n_gpu', type=int, default=1, help='total number of GPUs')
parser.add_argument('--deterministic', type=int, default=1, help='use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='learning rate for the model')
parser.add_argument('--img_size', type=int, default=224, help='input image size')
parser.add_argument('--seed', type=int, default=1234, help='random seed for reproducibility')
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs.", default=None, nargs='+')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true', help="use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision level')
parser.add_argument('--tag', help='experiment tag')
parser.add_argument('--eval', action='store_true', help='perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='test throughput only')
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')

parser.add_argument('--val_path', type=str, default='C:\\Users\\nazia\\Lab\\Dataset\\CropandWeed\\val', help='path for validation data')

args = parser.parse_args()
config = get_config(args)

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Dataset and configuration setup remain the same
    dataset_config = {
        'CustomDataset': {
            'root_path': args.root_path,
            'val_path': args.val_path,  # Include validation path here
            'num_classes': args.num_classes,
        },
    }

    args.num_classes = dataset_config[args.dataset]['num_classes']
    args.root_path = dataset_config[args.dataset]['root_path']
    args.val_path = dataset_config[args.dataset]['val_path']  # Pass validation path

    # Adjust learning rate if batch size changes
    if args.batch_size != 24 and args.batch_size % 6 == 0:
        args.base_lr *= args.batch_size / 24


    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load model configuration
    model_config = importlib.import_module(f'model.configs.{args.config_file}')
    net = SwinDeepLab(
        model_config.EncoderConfig, 
        model_config.ASPPConfig, 
        model_config.DecoderConfig
    ).cuda()
    
    # Load pretrained model if specified in configuration
    if model_config.EncoderConfig.encoder_name == 'swin' and model_config.EncoderConfig.load_pretrained:
        net.encoder.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    if model_config.ASPPConfig.aspp_name == 'swin' and model_config.ASPPConfig.load_pretrained:
        net.aspp.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained:
        net.decoder.load_from('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')
    if model_config.DecoderConfig.decoder_name == 'swin' and model_config.DecoderConfig.load_pretrained and model_config.DecoderConfig.extended_load:
        net.decoder.load_from_extended('./pretrained_ckpt/swin_tiny_patch4_window7_224.pth')

    # Use trainer_custom for CustomDataset
    trainer = {'CustomDataset': trainer_custom}  # Change to use trainer_custom for CustomDataset
    trainer[args.dataset](args, net, args.output_dir, val_path=args.val_path)  # Pass val_path