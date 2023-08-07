import numpy as np
import torch
from skimage import measure
import torchvision
import trimesh
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
import json
from pyhocon import ConfigFactory
from tqdm import tqdm

import sys
sys.path.append("../code")
import utils.plots as plts
import utils.general as utils
from utils import rend_util
from model.network_rico import sdf_to_alpha, alpha_to_w, cdf_Phi_s
from model.loss import compute_scale_and_shift


def get_plot_data(model_input, model_outputs, pose, rgb_gt, normal_gt, depth_gt, semantic_gt):
    batch_size, num_samples, _ = rgb_gt.shape

    rgb_eval = model_outputs['rgb_values'].reshape(batch_size, num_samples, 3)
    normal_map = model_outputs['normal_map'].reshape(batch_size, num_samples, 3)
    normal_map = (normal_map + 1.) / 2.
    
    depth_map = model_outputs['depth_values'].reshape(batch_size, num_samples)
    depth_gt = depth_gt.to(depth_map.device)
    scale, shift = compute_scale_and_shift(depth_map[..., None], depth_gt, depth_gt > 0.)
    depth_map = depth_map * scale + shift
    
    # save point cloud
    depth = depth_map.reshape(1, 1, 384, 384)
    # pred_points = get_point_cloud(depth, model_input, model_outputs)

    gt_depth = depth_gt.reshape(1, 1, 384, 384)
    # gt_points = get_point_cloud(gt_depth, model_input, model_outputs)

    # semantic map
    semantic_map = model_outputs['semantic_values'].argmax(dim=-1).reshape(batch_size, num_samples, 1)
    # in label mapping, 0 is bg idx and 0
    # for instance, first fg is 3 and 1
    # so when using argmax, the output will be label_mapping idx if correct
    
    plot_data = {
        'rgb_gt': rgb_gt,
        'normal_gt': (normal_gt + 1.)/ 2.,
        'depth_gt': depth_gt,
        'pose': pose,
        'rgb_eval': rgb_eval,
        'normal_map': normal_map,
        'depth_map': depth_map,
        # "pred_points": pred_points,
        # "gt_points": gt_points,
        "semantic_map": semantic_map,
        "semantic_gt": semantic_gt,
    }

    return plot_data

def get_sdf_vals_edit(pts, model, idx, edit_param, edit_type):
    with torch.no_grad():
        sdf_original = model.implicit_network.forward(pts)[:,:model.implicit_network.d_out]  # [N_pts, K]

        if edit_type == 'translate':
            edit_pts = pts - edit_param
        
        sdf_edit = model.implicit_network.forward(edit_pts)[:,:model.implicit_network.d_out]  # [N_pts, K]

        sdf_original[:, idx] = sdf_original[:, idx] * 0. + sdf_edit[:, idx]

        sdf = sdf_original

    sdf = -model.implicit_network.pool(-sdf.unsqueeze(1)).squeeze(-1) # get the minium value of sdf if bound apply before min
    return sdf

def neus_sample_edit(cam_loc, ray_dirs, model, idx, edit_param, edit_type):
    device = cam_loc.device
    perturb = False
    _, far = model.near_far_from_cube(cam_loc, ray_dirs, bound=model.scene_bounding_sphere)
    near = model.near * torch.ones(ray_dirs.shape[0], 1).cuda()

    _t = torch.linspace(0, 1, model.N_samples).float().to(device)
    z_vals = near * (1 - _t) + far * _t

    with torch.no_grad():
        _z = z_vals     # [N, 64]

        # follow the objsdf setting and use min sdf for sample
        _pts = cam_loc.unsqueeze(-2) + _z.unsqueeze(-1) * ray_dirs.unsqueeze(-2)
        N_rays, N_steps = _pts.shape[0], _pts.shape[1]
        
        _sdf = get_sdf_vals_edit(_pts.reshape(-1, 3), model, idx, edit_param, edit_type)
        
        _sdf = _sdf.reshape(N_rays, N_steps)

        for i in range(model.N_upsample_iters):
            prev_sdf, next_sdf = _sdf[..., :-1], _sdf[..., 1:]
            prev_z_vals, next_z_vals = _z[..., :-1], _z[..., 1:]
            mid_sdf = (prev_sdf + next_sdf) * 0.5
            dot_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
            prev_dot_val = torch.cat([torch.zeros_like(dot_val[..., :1], device=device), dot_val[..., :-1]], dim=-1)
            dot_val = torch.stack([prev_dot_val, dot_val], dim=-1)  
            dot_val, _ = torch.min(dot_val, dim=-1, keepdim=False)
            dot_val = dot_val.clamp(-10.0, 0.0)
            
            dist = (next_z_vals - prev_z_vals)
            prev_esti_sdf = mid_sdf - dot_val * dist * 0.5
            next_esti_sdf = mid_sdf + dot_val * dist * 0.5
            
            prev_cdf = cdf_Phi_s(prev_esti_sdf, 64 * (2**i))
            next_cdf = cdf_Phi_s(next_esti_sdf, 64 * (2**i))
            alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
            _w = alpha_to_w(alpha)
            z_fine = rend_util.sample_pdf(_z, _w, model.N_samples_extra // model.N_upsample_iters, det=not perturb)
            _z = torch.cat([_z, z_fine], dim=-1)
            
            _pts_fine = cam_loc.unsqueeze(-2) + z_fine.unsqueeze(-1) * ray_dirs.unsqueeze(-2)
            N_rays, N_steps_fine = _pts_fine.shape[0], _pts_fine.shape[1]

            sdf_fine = get_sdf_vals_edit(_pts_fine.reshape(-1, 3), model, idx, edit_param, edit_type)

            sdf_fine = sdf_fine.reshape(N_rays, N_steps_fine)
            _sdf = torch.cat([_sdf, sdf_fine], dim=-1)
            _z, z_sort_indices = torch.sort(_z, dim=-1)
            
            _sdf = torch.gather(_sdf, 1, z_sort_indices)
        
        z_all = _z
    
    return z_all

def get_sdf_vals_and_sdfs_edit(pts, model, idx, edit_param, edit_type):
    with torch.no_grad():
        sdf_original = model.implicit_network.forward(pts)[:,:model.implicit_network.d_out]  # [N_pts, K]

        if edit_type == 'translate':
            edit_pts = pts - edit_param
        
        sdf_edit = model.implicit_network.forward(edit_pts)[:,:model.implicit_network.d_out]  # [N_pts, K]

        sdf_original[:, idx] = sdf_original[:, idx] * 0. + sdf_edit[:, idx]

        sdf = sdf_original
        
    sdf_all = sdf
    sdf = -model.implicit_network.pool(-sdf.unsqueeze(1)).squeeze(-1)
    return sdf, sdf_all

def get_outputs_edit(points, model, idx, edit_param, edit_type):
    points.requires_grad_(True)

    # directly use the original geometry feature vector
    # fuse sdf together
    # then compute semantic, gradient, sdf

    original_output = model.implicit_network.forward(points)
    sdf_original = original_output[:,:model.implicit_network.d_out]
    feature_vectors = original_output[:,model.implicit_network.d_out:]

    if edit_type == 'translate':
        edit_pts = points - edit_param
        edit_output = model.implicit_network.forward(edit_pts)
        sdf_edit = edit_output[:, :model.implicit_network.d_out]
    
    sdf_raw = sdf_original
    sdf_raw[:, idx] = sdf_original[:, idx] * 0. + sdf_edit[:, idx]

    sigmoid_value = model.implicit_network.sigmoid
    semantic = sigmoid_value * torch.sigmoid(-sigmoid_value * sdf_raw)

    sdf = -model.implicit_network.pool(-sdf_raw.unsqueeze(1)).squeeze(-1)

    d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
    gradients = torch.autograd.grad(
        outputs=sdf,
        inputs=points,
        grad_outputs=d_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    
    return sdf, feature_vectors, gradients, semantic, sdf_raw


def render_edit(model, input, indices, idx=0, edit_param=[0., 0., 0.], edit_type='translate'):
    '''
    Currently only support one object
    if edit_type == 'translate', then edit_param is [dx, dy, dz]
    if edit_type == 'rotate', then edit_param is []: TODO
    just use neus
    '''
    assert idx > 0
    edit_param = torch.tensor(edit_param).cuda()

    intrinsics = input["intrinsics"].cuda()
    uv = input["uv"].cuda()
    pose = input["pose"].cuda()

    ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
    # we should use unnormalized ray direction for depth
    ray_dirs_tmp, _ = rend_util.get_camera_params(uv, torch.eye(4).to(pose.device)[None], intrinsics)
    depth_scale = ray_dirs_tmp[0, :, 2:]  # [N, 1]

    batch_size, num_pixels, _ = ray_dirs.shape
    cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
    ray_dirs = ray_dirs.reshape(-1, 3)

    '''
    Sample points with edited forward
    '''
    z_vals = neus_sample_edit(cam_loc, ray_dirs, model, idx, edit_param, edit_type)
    
    N_samples_tmp = z_vals.shape[1]

    points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)  # [N_rays, N_samples_tmp, 3]
    points_flat_tmp = points.reshape(-1, 3)

    sdf_tmp, sdf_all_tmp = get_sdf_vals_and_sdfs_edit(points_flat_tmp, model, idx, edit_param, edit_type)
    sdf_tmp = sdf_tmp.reshape(-1, N_samples_tmp)
    s_value = model.get_s_value()

    cdf, opacity_alpha = sdf_to_alpha(sdf_tmp, s_value)     # [N_rays, N_samples_tmp-1]

    sdf_all_tmp = sdf_all_tmp.reshape(-1, N_samples_tmp, model.num_semantic)

    z_mid_vals = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
    N_samples = z_mid_vals.shape[1]

    points_mid = cam_loc.unsqueeze(1) + z_mid_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)  # [N_rays, N_samples, 3]
    points_flat = points_mid.reshape(-1, 3)

    dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
    dirs_flat = dirs.reshape(-1, 3)
        
    sdf, feature_vectors, gradients, semantic, sdf_raw = get_outputs_edit(points_flat, model, idx, edit_param, edit_type)

    # here the rgb output might be wrong
    rgb_flat = model.rendering_network(points_flat, gradients, dirs_flat, feature_vectors, indices)
    rgb = rgb_flat.reshape(-1, N_samples, 3)

    semantic = semantic.reshape(-1, N_samples, model.num_semantic)

    weights = alpha_to_w(opacity_alpha)

    rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
    semantic_values = torch.sum(weights.unsqueeze(-1)*semantic, 1)
    raw_depth_values = torch.sum(weights * z_mid_vals, 1, keepdims=True) / (weights.sum(dim=1, keepdims=True) +1e-8)
    depth_values = depth_scale * raw_depth_values

    output = {
            'rgb_values': rgb_values,
            'semantic_values': semantic_values,
            'depth_values': depth_values,
        }
    
    # compute normal map
    normals = gradients / (gradients.norm(2, -1, keepdim=True) + 1e-6)
    normals = normals.reshape(-1, N_samples, 3)
    normal_map = torch.sum(weights.unsqueeze(-1) * normals, 1)
    
    # transform to local coordinate system
    rot = pose[0, :3, :3].permute(1, 0).contiguous()
    normal_map = rot @ normal_map.permute(1, 0)
    normal_map = normal_map.permute(1, 0).contiguous()
    
    output['normal_map'] = normal_map

    return output


edit_idx = 1
edit_param = [0., 0., 0.]
edit_type = 'translate'

exp_name = 'RICO_synthetic_1'
scan_id = int(exp_name[-1])

exp_path = os.path.join('../exps/', exp_name)
timestamp = os.listdir(exp_path)[-1]  # use the latest if not other need
exp_path = os.path.join(exp_path, timestamp)

conf = ConfigFactory.parse_file(os.path.join(exp_path, 'runconf.conf'))
dataset_conf = conf.get_config('dataset')
dataset_conf['scan_id'] = scan_id
conf_model = conf.get_config('model')

train_dataset = utils.get_class(conf.get_string('train.dataset_class'))(**dataset_conf)
plot_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=conf.get_int('plot.plot_nimgs'),
    shuffle=False,
    collate_fn=train_dataset.collate_fn)

model = utils.get_class(conf.get_string('train.model_class'))(conf=conf_model)

if torch.cuda.is_available():
    model.cuda()

ckpt_path = os.path.join(exp_path, 'checkpoints/ModelParameters', 'latest.pth')
ckpt = torch.load(ckpt_path)
print(ckpt['epoch'])

# model.load_state_dict(ckpt['model_state_dict'])
# load in a non-DDP fashion
model.load_state_dict({k.replace('module.',''): v for k,v in ckpt['model_state_dict'].items()})
os.makedirs('./tmp_edit', exist_ok=True)

model.eval()

data_idx = 75
vis_data = plot_dataloader.dataset[data_idx]

indices, model_input, ground_truth = vis_data
indices = torch.tensor([indices])
print(indices)
for k, v in model_input.items():
    model_input[k] = v.unsqueeze(0)
for k, v in ground_truth.items():
    ground_truth[k] = v.unsqueeze(0)

model_input["intrinsics"] = model_input["intrinsics"].cuda()
model_input["uv"] = model_input["uv"].cuda()
model_input['pose'] = model_input['pose'].cuda()

split = utils.split_input(model_input, 384*384, n_pixels=128)
res = []

for s in tqdm(split):
    # out = model(s, indices)
    out = render_edit(model, s, indices, edit_idx, edit_param, edit_type)
    d = {'rgb_values': out['rgb_values'].detach(),
            'normal_map': out['normal_map'].detach(),
            'depth_values': out['depth_values'].detach(),
            'semantic_values': out['semantic_values'].detach()}
    if 'rgb_un_values' in out:
        d['rgb_un_values'] = out['rgb_un_values'].detach()
    res.append(d)

batch_size = ground_truth['rgb'].shape[0]
model_outputs = utils.merge_output(res, 384*384, batch_size)
plot_data = get_plot_data(model_input, model_outputs, model_input['pose'], ground_truth['rgb'], ground_truth['normal'], ground_truth['depth'], ground_truth['instance_mask'])

plot_conf = conf.get_config('plot')
plot_conf['obj_boxes'] = None
plts.plot_rico(
    None,
    indices,
    plot_data,
    './tmp_edit/',
    ckpt['epoch'],
    [384, 384],
    plot_mesh = False,
    **plot_conf
)