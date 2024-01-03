from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from torch.autograd import Variable

import os, sys, errno
import argparse
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import open3d as o3d
import copy

from utils import post_process_depth, flip_lr
from networks.NewCRFDepth import NewCRFDepth


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='NDDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='nddepth')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07', default='large07')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')

if sys.argv.__len__() == 2:
    arg_filename_with_prefix = '@' + sys.argv[1]
    args = parser.parse_args([arg_filename_with_prefix])
else:
    args = parser.parse_args()

if args.dataset == 'kitti' or args.dataset == 'nyu':
    from dataloaders.dataloader import NewDataLoader


model_dir = os.path.dirname(args.checkpoint_path)
sys.path.append(model_dir)


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def test(params):
    """Test function."""
    args.mode = 'test'
    dataloader = NewDataLoader(args, 'test')
    
    model = NewCRFDepth(version='large07', inv_depth=False, max_depth=args.max_depth)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.cuda()

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    num_test_samples = get_num_lines(args.filenames_file)

    with open(args.filenames_file) as f:
        lines = f.readlines()

    print('now testing {} files with {}'.format(num_test_samples, args.checkpoint_path))

    pred_depths = []
    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            inv_K = Variable(sample['inv_K'].cuda())
            inv_K_p = Variable(sample['inv_K_p'].cuda())
            b, _, h, w = image.shape
           
            # Predict
            epoch = 5
            depth1_list, uncer1_list, depth2_list, uncer2_list, _, _ = model(image, inv_K, epoch)
            depth_est = 0.5 * (depth1_list[-1] + depth2_list[-1])

            post_process = True
            if post_process:
                image_flipped = flip_lr(image)
                depth1_list_flipped, uncer1_list_flipped, depth2_list_flipped, uncer2_list_flipped, _, _ = model(image_flipped, inv_K, epoch)
                depth_est_flipped = 0.5 * (depth1_list_flipped[-1] + depth2_list_flipped[-1])
                depth_est = post_process_depth(depth_est, depth_est_flipped)

            pred_depth = depth_est.cpu().numpy().squeeze()

            if args.do_kb_crop:
                height, width = 352, 1216
                top_margin = int(height - 352)
                left_margin = int((width - 1216) / 2)
                pred_depth_uncropped = np.zeros((height, width), dtype=np.float32)
                pred_depth_uncropped[top_margin:top_margin + 352, left_margin:left_margin + 1216] = pred_depth
                pred_depth = pred_depth_uncropped
            
            pred_depths.append(pred_depth)

    elapsed_time = time.time() - start_time
    print('Elapesed time: %s' % str(elapsed_time))
    print('Done.')
    
    save_name = 'result_' + args.model_name
    
    print('Saving result pngs..')
    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    for s in tqdm(range(num_test_samples)):
        if args.dataset == 'kitti':
            filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            
        pred_depth = pred_depths[s]

        if args.dataset == 'kitti':
            pred_depth_scaled = pred_depth * 256.0
      
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
           
    return



if __name__ == '__main__':
    test(args)
