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

from utils import post_process_depth, flip_lr, D_to_cloud, inv_normalize
from networks.NewCRFDepth import NewCRFDepth


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='NDDepth PyTorch implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--model_name', type=str, help='model name', default='nddepth')
parser.add_argument('--encoder', type=str, help='type of encoder, base07, large07, tiny07', default='large07')
parser.add_argument('--data_path', type=str, help='path to the data', required=True)
parser.add_argument('--filenames_file', type=str, help='path to the filenames text file', required=True)
parser.add_argument('--input_height', type=int, help='input height', default=480)
parser.add_argument('--input_width', type=int, help='input width', default=640)
parser.add_argument('--max_depth', type=float, help='maximum depth in estimation', default=10)
parser.add_argument('--checkpoint_path', type=str, help='path to a specific checkpoint to load', default='')
parser.add_argument('--dataset', type=str, help='dataset to train on', default='nyu')
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')
parser.add_argument('--pred_clouds', help='if set, pred cloud points', action='store_true')
parser.add_argument('--save_viz', help='if set, save visulization of the outputs', action='store_true')

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
    pred_clouds = []
    start_time = time.time()
    with torch.no_grad():
        for _, sample in enumerate(tqdm(dataloader.data)):
            image = Variable(sample['image'].cuda())
            inv_K = Variable(sample['inv_K'].cuda())
            inv_K_p = Variable(sample['inv_K_p'].cuda())
            b, _, h, w = image.shape
            depth_to_cloud = D_to_cloud(b, h, w).cuda()
           
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
            
            if args.pred_clouds:
                if args.dataset == 'nyu':
                    color = inv_normalize(image[0, :, :, :]).permute(1, 2, 0)[45:472, 43:608, :].reshape(-1, 3).cpu().numpy()
                    points = depth_to_cloud(depth_est, inv_K_p).reshape(1, h, w, 3)[:, 45:472, 43:608, :].reshape(1, -1, 3)
                    points = points.cpu().numpy().squeeze()
                else:
                    color = inv_normalize(image[0, :, :, :]).permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
                    points = depth_to_cloud(depth_est, inv_K_p)
                    points = points.cpu().numpy().squeeze()
                pc = o3d.geometry.PointCloud()
                pc.points = o3d.utility.Vector3dVector(points)
                pc.colors = o3d.utility.Vector3dVector(color)

                pred_clouds.append(pc)

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
    
    save_name = 'models/result_' + args.model_name
    
    print('Saving result pngs..')
    if not os.path.exists(save_name):
        try:
            os.mkdir(save_name)
            os.mkdir(save_name + '/raw')
            os.mkdir(save_name + '/cmap')
            os.mkdir(save_name + '/rgb')
            os.mkdir(save_name + '/gt')
            os.mkdir(save_name + '/cloud')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    
    for s in tqdm(range(num_test_samples)):
        if args.dataset == 'kitti':
            filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
        elif args.dataset == 'kittipred':
            filename_pred_png = save_name + '/raw/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_cmap_png = save_name + '/cmap/' + lines[s].split()[0].split('/')[-1].replace('.jpg', '.png')
            filename_image_png = save_name + '/rgb/' + lines[s].split()[0].split('/')[-1]
        else:
            scene_name = lines[s].split()[0].split('/')[0]
            filename_pred_png = save_name + '/raw/' + scene_name + '_' + lines[s].split()[0].split('/')[1].replace(
                '.jpg', '.png')
            filename_pred_ply = save_name + '/cloud/' + scene_name + '_' + lines[s].split()[0].split('/')[1][:-4] + '_' + 'PADepth' + '.ply'
            filename_cmap_png = save_name + '/cmap/' + scene_name + '_' + lines[s].split()[0].split('/rgb_')[1].replace(
                '.jpg', '.png')
            filename_gt_png = save_name + '/gt/' + scene_name + '_' + lines[s].split()[0].split('/rgb_')[1].replace(
                '.jpg', '_gt.png')
            filename_image_png = save_name + '/rgb/' + scene_name + '_' + lines[s].split()[0].split('/rgb_')[1]
        
        rgb_path = os.path.join(args.data_path, './' + lines[s].split()[0])
        image = cv2.imread(rgb_path)
        if args.dataset == 'nyu':
            gt_path = os.path.join(args.data_path, './' + lines[s].split()[1])
            gt = cv2.imread(gt_path, -1).astype(np.float32) / 1000.0  # Visualization purpose only
            gt[gt == 0] = np.amax(gt)
        
        pred_depth = pred_depths[s]

        if args.dataset == 'kitti' or args.dataset == 'kittipred':
            pred_depth_scaled = pred_depth * 256.0
        else:
            pred_depth_scaled = pred_depth * 1000.0
        
        pred_depth_scaled = pred_depth_scaled.astype(np.uint16)
        cv2.imwrite(filename_pred_png, pred_depth_scaled, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if args.save_viz:
            cv2.imwrite(filename_image_png, image[10:-1 - 9, 10:-1 - 9, :])
            if args.dataset == 'nyu':
                plt.imsave(filename_gt_png, (10 - gt) / 10, cmap='jet')
                pred_depth_cropped = pred_depth[10:-1 - 9, 10:-1 - 9]
                plt.imsave(filename_cmap_png, (10 - pred_depth) / 10, cmap='jet')
            else:
                plt.imsave(filename_cmap_png, np.log10(pred_depth), cmap='magma')

        if args.pred_clouds:
            pred_cloud = pred_clouds[s]
            o3d.io.write_point_cloud(filename_pred_ply, pred_cloud)
        
    return


if __name__ == '__main__':
    test(args)
