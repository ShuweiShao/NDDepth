import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Sampler
from torchvision import transforms
import matplotlib.pyplot as plt
import os, sys
import numpy as np
import math
import torch
from skimage.segmentation import all_felzenszwalb as felz_seg

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


def block_print():
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    sys.stdout = sys.__stdout__


def get_num_lines(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    return len(lines)


def colorize(value, vmin=None, vmax=None, cmap='Greys'):
    value = value.cpu().numpy()[:, :, :]
    value = np.log10(value)

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value*0.

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)

    img = value[:, :, :3]

    return img.transpose((2, 0, 1))


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


eval_metrics = ['silog', 'abs_rel', 'log10', 'rms', 'irms', 'sq_rel', 'log_rms', 'd1', 'd2', 'd3']


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, sq_rel, log_rms, d1, d2, d3]


def compute_errors_kb(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    gt_inv = 1.0 / (1e-3 * gt + 1e-8)
    pred_inv = 1.0 / (1e-3 * pred + 1e-8)

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    irms = (gt_inv - pred_inv) ** 2
    irms = np.sqrt(irms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt) * 100
    sq_rel = np.mean(((gt - pred) / gt) ** 2) * 100

    err = np.log(pred) - np.log(gt)
    silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)

    return [silog, abs_rel, log10, rms, irms, sq_rel, log_rms, d1, d2, d3]


class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0


def flip_lr(image):
    """
    Flip image horizontally

    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Image to be flipped

    Returns
    -------
    image_flipped : torch.Tensor [B,3,H,W]
        Flipped image
    """
    assert image.dim() == 4, 'You need to provide a [B,C,H,W] image to flip'
    return torch.flip(image, [3])


def fuse_inv_depth(inv_depth, inv_depth_hat, method='mean'):
    """
    Fuse inverse depth and flipped inverse depth maps

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_hat : torch.Tensor [B,1,H,W]
        Flipped inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    fused_inv_depth : torch.Tensor [B,1,H,W]
        Fused inverse depth map
    """
    if method == 'mean':
        return 0.5 * (inv_depth + inv_depth_hat)
    elif method == 'max':
        return torch.max(inv_depth, inv_depth_hat)
    elif method == 'min':
        return torch.min(inv_depth, inv_depth_hat)
    else:
        raise ValueError('Unknown post-process method {}'.format(method))


def post_process_depth(depth, depth_flipped, method='mean'):
    """
    Post-process an inverse and flipped inverse depth map

    Parameters
    ----------
    inv_depth : torch.Tensor [B,1,H,W]
        Inverse depth map
    inv_depth_flipped : torch.Tensor [B,1,H,W]
        Inverse depth map produced from a flipped image
    method : str
        Method that will be used to fuse the inverse depth maps

    Returns
    -------
    inv_depth_pp : torch.Tensor [B,1,H,W]
        Post-processed inverse depth map
    """
    B, C, H, W = depth.shape
    inv_depth_hat = flip_lr(depth_flipped)
    inv_depth_fused = fuse_inv_depth(depth, inv_depth_hat, method=method)
    xs = torch.linspace(0., 1., W, device=depth.device,
                        dtype=depth.dtype).repeat(B, C, H, 1)
    mask = 1.0 - torch.clamp(20. * (xs - 0.05), 0., 1.)
    mask_hat = flip_lr(mask)
    return mask_hat * depth + mask * inv_depth_hat + \
           (1.0 - mask - mask_hat) * inv_depth_fused


class DistributedSamplerNoEvenlyDivisible(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        num_samples = int(math.floor(len(self.dataset) * 1.0 / self.num_replicas))
        rest = len(self.dataset) - num_samples * self.num_replicas
        if self.rank < rest:
            num_samples += 1
        self.num_samples = num_samples
        self.total_size = len(dataset)
        # self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        # assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class D_to_cloud(nn.Module):
    """Layer to transform depth into point cloud
    """
    def __init__(self, batch_size, height, width):
        super(D_to_cloud, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False) # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points

        return cam_points.permute(0, 2, 1)


class DN_to_distance(nn.Module):
    """Layer to transform depth and normal into distance
    """
    def __init__(self, batch_size, height, width):
        super(DN_to_distance, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False) # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, depth, norm_normal, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        distance = (norm_normal.view(self.batch_size, 3, -1) * cam_points).sum(1, keepdim=True)
        distance = distance.reshape(self.batch_size, 1, self.height, self.width)
        return distance.abs()

class DN_to_depth(nn.Module):
    """Layer to transform distance and normal into depth
    """
    def __init__(self, batch_size, height, width):
        super(DN_to_depth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32) # 2, H, W    
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords), requires_grad=False) # 2, H, W  

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False) # B, 1, H, W

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0) # 1, 2, L
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1) # B, 2, L
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1), requires_grad=False) # B, 3, L

    def forward(self, norm_normal, distance, inv_K):
        normalized_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        normalized_points = normalized_points.reshape(self.batch_size, 3, self.height, self.width)
        normal_points = (norm_normal * normalized_points).sum(1, keepdim=True)
        depth = distance / (normal_points + 1e-7)
        return depth.abs()
    
def colormap(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    _DEPTH_COLORMAP = plt.get_cmap('jet', 256)  # for plotting
    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis[0,:,:,:]

def colormap_magma(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    _DEPTH_COLORMAP = plt.get_cmap('magma', 256)  # for plotting
    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis[0,:,:,:]

def colormap_viridis(inputs, normalize=True, torch_transpose=True):
    if isinstance(inputs, torch.Tensor):
        inputs = inputs.detach().cpu().numpy()
    _DEPTH_COLORMAP = plt.get_cmap('viridis', 256)  # for plotting
    vis = inputs
    if normalize:
        ma = float(vis.max())
        mi = float(vis.min())
        d = ma - mi if ma != mi else 1e5
        vis = (vis - mi) / d

    if vis.ndim == 4:
        vis = vis.transpose([0, 2, 3, 1])
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, 0, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 3:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[:, :, :, :3]
        if torch_transpose:
            vis = vis.transpose(0, 3, 1, 2)
    elif vis.ndim == 2:
        vis = _DEPTH_COLORMAP(vis)
        vis = vis[..., :3]
        if torch_transpose:
            vis = vis.transpose(2, 0, 1)

    return vis[0,:,:,:]

def normalize(a):
    return (a - a.min())/(a.max() - a.min() + 1e-8)

def normalize_image(x):
    """Rescale image pixels to span range [0, 1]
    """
    ma = float(x.max().cpu().data)
    mi = float(x.min().cpu().data)
    d = ma - mi if ma != mi else 1e5
    return (x - mi) / d

def compute_seg(rgb, aligned_norm, D):
        """
        inputs:
            rgb                 b, 3, H, W
            aligned_norm        b, 3, H, W
            D                   b, H, W

        outputs:
            segment                b, 1, H, W
            planar mask        b, 1, H, W
        """
        b, _, h, w  = rgb.shape
        device = rgb.device

        # compute cost
        pdist = nn.PairwiseDistance(p=2)

        rgb_down = pdist(rgb[:, :, 1:], rgb[:, :, :-1])
        rgb_right = pdist(rgb[:, :, :, 1:], rgb[:, :, :, :-1])
        
        rgb_down = torch.stack([normalize(rgb_down[i]) for i in range(b)])
        rgb_right = torch.stack([normalize(rgb_right[i]) for i in range(b)])

        D_down = abs(D[:, 1:] - D[:, :-1])
        D_right = abs(D[:, :, 1:] - D[:, :, :-1])
        norm_down = pdist(aligned_norm[:, :, 1:], aligned_norm[:, :, :-1])
        norm_right = pdist(aligned_norm[:, :, :, 1:], aligned_norm[:, :, :, :-1])

        D_down = torch.stack([normalize(D_down[i]) for i in range(b)])
        norm_down = torch.stack([normalize(norm_down[i]) for i in range(b)])

        D_right = torch.stack([normalize(D_right[i]) for i in range(b)])
        norm_right = torch.stack([normalize(norm_right[i]) for i in range(b)])

        normD_down = D_down + norm_down
        normD_right = D_right + norm_right

        normD_down = torch.stack([normalize(normD_down[i]) for i in range(b)])
        normD_right = torch.stack([normalize(normD_right[i]) for i in range(b)])

        # get max from (rgb, normD)
        # cost_down = torch.stack([rgb_down, normD_down])
        # cost_right = torch.stack([rgb_right, normD_right])
        # cost_down, _ = torch.max(cost_down, 0)
        # cost_right, _ = torch.max(cost_right, 0)
        cost_down = normD_down
        cost_right = normD_right
        # cost_down = rgb_down
        # cost_right = rgb_right

        # get dissimilarity map visualization
        dst = cost_down[:,  :,  : -1] + cost_right[ :, :-1, :]
        
        # felz_seg
        cost_down_np = cost_down.detach().cpu().numpy()
        cost_right_np = cost_right.detach().cpu().numpy()
        segment = torch.stack([torch.from_numpy(felz_seg(normalize(cost_down_np[i]), normalize(cost_right_np[i]), 0, 0, h, w, scale=2, min_size=200)).to(device) for i in range(b)])
        segment += 1
        segment = segment.unsqueeze(1)
        
        # generate mask for segment with area larger than 200
        max_num = segment.max().item() + 1

        area = torch.zeros((b, max_num)).to(device)
        area.scatter_add_(1, segment.view(b, -1), torch.ones((b, 1, h, w)).view(b, -1).to(device))

        planar_area_thresh = 200
        valid_mask = (area > planar_area_thresh).float()
        planar_mask = torch.gather(valid_mask, 1, segment.view(b, -1))
        planar_mask = planar_mask.view(b, 1, h, w)

        planar_mask[:, :, :8, :] = 0
        planar_mask[:, :, -8:, :] = 0
        planar_mask[:, :, :, :8] = 0
        planar_mask[:, :, :, -8:] = 0

        return segment, planar_mask, dst.unsqueeze(1)

def get_smooth_ND(normal, distance, planar_mask):
    
    """Computes the smoothness loss for normal and distance
    """
    grad_normal_x = torch.mean(torch.abs(normal[:, :, :, :-1] - normal[:, :, :, 1:]), 1, keepdim=True)
    grad_normal_y = torch.mean(torch.abs(normal[:, :, :-1, :] - normal[:, :, 1:, :]), 1, keepdim=True)

    grad_distance_x = torch.abs(distance[:, :, :, :-1] - distance[:, :, :, 1:])
    grad_distance_y = torch.abs(distance[:, :, :-1, :] - distance[:, :, 1:, :])

    planar_mask_x = planar_mask[:, :, :, :-1]
    planar_mask_y = planar_mask[:, :, :-1, :]
    
    loss_grad_normal = (grad_normal_x * planar_mask_x).sum() / planar_mask_x.sum() + (grad_normal_y * planar_mask_y).sum() / planar_mask_y.sum()
    loss_grad_distance = (grad_distance_x * planar_mask_x).sum() / planar_mask_x.sum() + (grad_distance_y * planar_mask_y).sum() / planar_mask_y.sum()
    
    return loss_grad_normal, loss_grad_distance







        
