from collections import OrderedDict
import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree
import trimesh
import torch
import glob
import os
import pyrender
import os
import cv2
import json
from tqdm import tqdm
from pathlib import Path

os.environ['PYOPENGL_PLATFORM'] = 'egl'

def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


def evaluate(mesh_pred, mesh_trgt, obj_type='bg', threshold=.05, down_sample=.02):
    pcd_trgt = o3d.geometry.PointCloud()
    pcd_pred = o3d.geometry.PointCloud()
    
    trgt_pts = mesh_trgt.vertices[:, :3]
    pred_pts = mesh_pred.vertices[:, :3]

    if obj_type == 'obj':
        pts_mask = pred_pts[:, 2] < -0.9
        pred_pts = pred_pts[pts_mask]
    
    pcd_trgt.points = o3d.utility.Vector3dVector(trgt_pts)
    pcd_pred.points = o3d.utility.Vector3dVector(pred_pts)

    if down_sample:
        pcd_pred = pcd_pred.voxel_down_sample(down_sample)
        pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

    verts_pred = np.asarray(pcd_pred.points)
    verts_trgt = np.asarray(pcd_trgt.points)

    dist1 = nn_correspondance(verts_pred, verts_trgt)
    dist2 = nn_correspondance(verts_trgt, verts_pred)

    precision = np.mean((dist2 < threshold).astype('float'))
    recal = np.mean((dist1 < threshold).astype('float'))
    fscore = 2 * precision * recal / (precision + recal)
    chamfer = (np.mean(dist2) + np.mean(dist1)) / 2
    metrics = {
        'Acc': np.mean(dist2),
        'Comp': np.mean(dist1),
        'Chamfer': chamfer,
        'Prec': precision,
        'Recal': recal,
        'F-score': fscore,
    }
    return metrics

# hard-coded image size
H, W = 384, 384

def average_dicts(dicts):
    # input is a list of dict
    # all the dict have same keys
    dict_num = len(dicts)
    keys = dicts[0].keys()
    ret = {}

    for k in keys:
        values = [x[k] for x in dicts]
        value = np.array(values).mean()
        ret[k] = value
    
    return ret


root_dir = "../exps/"
exp_name = "RICO_synthetic"
out_dir = "evaluation/" + exp_name
Path(out_dir).mkdir(parents=True, exist_ok=True)


scenes = {
    1: 'scene1',
    2: 'scene2',
    3: 'scene3',
    4: 'scene4',
    5: 'scene5',
}

all_obj_results = []
all_obj_results_dict = OrderedDict()

for k, v in scenes.items():
    
    cur_exp = f"{exp_name}_{k}"
    cur_root = os.path.join(root_dir, cur_exp)
    if not os.path.isdir(cur_root):
        continue
    # use last timestamps
    dirs = sorted(os.listdir(cur_root))
    cur_root = os.path.join(cur_root, dirs[-1])

    files = list(filter(os.path.isfile, glob.glob(os.path.join(cur_root, "plots/*.ply"))))
    
    # evalute the meshes for obj and bg, the first is bg and last is all
    files.sort(key=lambda x:os.path.getmtime(x))

    cam_file = f"../data/syn_data/scene{k}/cameras.npz"
    scale_mat = np.load(cam_file)['scale_mat_0']

    ply_files = files[1: -1]
    # print(ply_files)

    cnt = 1
    obj_results = []
    obj_results_dict = OrderedDict()
    for ply_file in ply_files:

        mesh = trimesh.load(ply_file)
        mesh.vertices = (scale_mat[:3, :3] @ mesh.vertices.T + scale_mat[:3, 3:]).T
        
        gt_mesh = os.path.join(f"../data/syn_data/scene{k}/GT_mesh", f"object{cnt}.ply")
        
        gt_mesh = trimesh.load(gt_mesh)
        
        metrics = evaluate(mesh, gt_mesh, 'obj')
        obj_results.append(metrics)
        obj_results_dict[cnt] = metrics

        cnt += 1
    
    obj_results = average_dicts(obj_results)
    all_obj_results.append(obj_results)
    all_obj_results_dict[k] = obj_results_dict

# the average result print
all_obj_results = average_dicts(all_obj_results)
print('objects:')
print(all_obj_results)

# all the result save
obj_json_str = json.dumps(all_obj_results_dict, indent=4)
obj_json_file = os.path.join('evaluation', exp_name + '_obj.json')

with open(obj_json_file, 'w') as json_file:
    json_file.write(obj_json_str)
json_file.close()