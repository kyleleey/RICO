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

def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose

# hard-coded image size
H, W = 384, 384

# load pose
def load_poses(scan_id, object_id):
    pose_path = os.path.join(f'../data/syn_data/scene{scan_id}', 'cameras.npz')

    camera_dict = np.load(pose_path)
    len_pose = len(camera_dict.files) // 2

    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(len_pose)]
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(len_pose)]
    P = world_mats[0] @ scale_mats[0]
    P = P[:3, :4]
    intrinsics, pose = load_K_Rt_from_P(None, P)

    poses = []
    cnt = 0

    masks_path = os.path.join(f'../data/syn_data/scene{scan_id}', 'instance_mask')
    mask_files = sorted(os.listdir(masks_path))

    id_json = os.path.join(f'../data/syn_data/scene{scan_id}', 'instance_id.json')
    with open(id_json, 'r') as f:
        id_data = json.load(f)
    f.close()

    if object_id > 0: # the valid object id
        obj_idx = id_data[f'obj_{object_id-1}']
    else:
        obj_idx = -1 # invalid id, maybe for bg, however we load all the poses in this situation

    for scale_mat, world_mat in zip(scale_mats, world_mats):
        # first check if object is in this pose's corresponding image
        mask = cv2.imread(os.path.join(masks_path, mask_files[cnt]))
        mask = np.array(mask)
        mask = np.unique(mask)
        
        if obj_idx == -1:
            orig_pose = world_mat
            pose = np.linalg.inv(orig_pose) @ intrinsics
            poses.append(np.array(pose))

        elif obj_idx in mask:
            orig_pose = world_mat
            pose = np.linalg.inv(orig_pose) @ intrinsics
            poses.append(np.array(pose))

        cnt += 1
    
    poses = np.array(poses)
    print(poses.shape)
    return poses, intrinsics


class Renderer():
    def __init__(self, height=480, width=640):
        self.renderer = pyrender.OffscreenRenderer(width, height)
        self.scene = pyrender.Scene()
        self.render_flags = pyrender.RenderFlags.SKIP_CULL_FACES

    def __call__(self, height, width, intrinsics, pose, mesh, need_flag=True):
        self.renderer.viewport_height = height
        self.renderer.viewport_width = width
        self.scene.clear()
        self.scene.add(mesh)
        cam = pyrender.IntrinsicsCamera(cx=intrinsics[0, 2], cy=intrinsics[1, 2],
                                        fx=intrinsics[0, 0], fy=intrinsics[1, 1])
        self.scene.add(cam, pose=self.fix_pose(pose))
        if need_flag:
            return self.renderer.render(self.scene, self.render_flags)
        else:
            return self.renderer.render(self.scene)  # , self.render_flags)

    def fix_pose(self, pose):
        # 3D Rotation about the x-axis.
        t = np.pi
        c = np.cos(t)
        s = np.sin(t)
        R = np.array([[1, 0, 0],
                      [0, c, -s],
                      [0, s, c]])
        axis_transform = np.eye(4)
        axis_transform[:3, :3] = R
        return pose @ axis_transform

    def mesh_opengl(self, mesh):
        return pyrender.Mesh.from_trimesh(mesh)

    def delete(self):
        self.renderer.delete()
        

def refuse_depth(mesh, poses, K, need_flag=False, scan_id=-1):
    renderer = Renderer()
    mesh_opengl = renderer.mesh_opengl(mesh)

    depths = []

    for pose in tqdm(poses):
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        _, depth_pred = renderer(H, W, intrinsic, pose, mesh_opengl, need_flag=need_flag)
        depths.append(depth_pred)
    
    return depths


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

all_bg_results = []
all_bg_results_dict = OrderedDict()

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
    
    bg_file = files[0]
    print(bg_file)
    bg_mesh = trimesh.load(bg_file)

    cam_file = f"../data/syn_data/scene{k}/cameras.npz"
    scale_mat = np.load(cam_file)['scale_mat_0']
    bg_mesh.vertices = (scale_mat[:3, :3] @ bg_mesh.vertices.T + scale_mat[:3, 3:]).T

    poses, K = load_poses(k, -1)
    K = K[:3, :3]
    bg_mesh_depth = refuse_depth(bg_mesh, poses, K, scan_id=k)
    
    gt_mesh = os.path.join(f"../data/syn_data/scene{k}/GT_mesh", f"background.ply")
    gt_mesh = trimesh.load(gt_mesh)

    gt_mesh.vertex_normals = -gt_mesh.vertex_normals

    gt_mesh_depth = refuse_depth(gt_mesh, poses, K, need_flag=True, scan_id=k)

    masks_path = os.path.join(f'../data/syn_data/scene{k}', 'instance_mask')
    mask_files = sorted(os.listdir(masks_path))
    masks = [cv2.imread(os.path.join(masks_path, x)) for x in mask_files]

    depth_errors = []
    for gt_depth, pred_depth, seg_mask in zip(gt_mesh_depth, bg_mesh_depth, masks):
        seg = seg_mask
        seg = np.array(seg)
        seg = seg[:, :, 0] > 0  # obj regions

        gtd = gt_depth[seg]
        prd = pred_depth[seg]

        mse = (np.square(gtd - prd)).mean(axis=0)
        depth_errors.append(mse)
    depth_errors = np.array(depth_errors)
    metrics = {'bg_depth_error': depth_errors.mean().astype(float)}

    all_bg_results.append(metrics)
    all_bg_results_dict[k] = metrics

# the average result print
all_bg_results = average_dicts(all_bg_results)
print('background:')
print(all_bg_results)
# all the result save
bg_json_str = json.dumps(all_bg_results_dict, indent=4)
bg_json_file = os.path.join('evaluation', exp_name + '_bg_depth.json')

with open(bg_json_file, 'w') as json_file:
    json_file.write(bg_json_str)
json_file.close()