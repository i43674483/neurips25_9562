import os

import imageio
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset


class SpectrumDataset(Dataset):

    def __init__(self, args, datadir, mode="test"):
        super().__init__()
        self.datadir = datadir

        self.mode = mode
        scene_id = args.scene_id

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        self.tx_pos_dir = os.path.join(datadir, 'tx_pos.csv')
        tx_pos = pd.read_csv(self.tx_pos_dir).values         
        tx_pos = torch.tensor(tx_pos, dtype=torch.float32)

        self.gateway_pos_dir = os.path.join(datadir, 'gateway_info.yml')

        self.spectrum_dir = os.path.join(datadir, 'spectrum')

        self.spt_names = sorted([f for f in os.listdir(self.spectrum_dir) if f.endswith('.png')])

        example_spt = imageio.imread(os.path.join(self.spectrum_dir, self.spt_names[0]))

        self.n_elevation, self.n_azimuth = example_spt.shape
        self.rays_per_spectrum = self.n_elevation * self.n_azimuth

        num_locations = tx_pos.shape[0]
        interval = 8

        train_index = os.path.join(datadir, args['train_index_path'])
        test_index = os.path.join(datadir, args['test_index_path'])

        i_train = np.loadtxt(train_index, dtype=int) -1
        i_test = np.loadtxt(test_index, dtype=int) - 1
            
        if self.mode == "train":
            i_render = i_train
        else:
            i_render = i_test

        self.num_source_views = args.num_source_views

        rgb_files = [os.path.join(self.spectrum_dir, one_spec_name) for one_spec_name in self.spt_names]

        with open(self.gateway_pos_dir) as f:
            gateway_info = yaml.safe_load(f)
            gateway_pos = gateway_info['gateway1']['position']  
            gateway_orientation = gateway_info['gateway1']['orientation']

        rotation_matrix = torch.from_numpy(Rotation.from_quat(gateway_orientation).as_matrix()).float()

        intrinsics = torch.eye(4)                            
        
        intrinsics[:3, 3] = torch.tensor(gateway_pos)        
        intrinsics = intrinsics.repeat(num_locations, 1, 1)  

        c2w_matrices = torch.zeros((num_locations, 4, 4))
        c2w_matrices[:, :3, :3] = rotation_matrix

        homogeneous_translations = torch.cat((tx_pos, torch.ones((num_locations, 1))), dim=1)
        c2w_matrices[:, :4, 3] = homogeneous_translations

        c2w_matrices[:, 3, 3] = 1.0

        self.train_intrinsics.append(intrinsics[i_train])
        self.train_poses.append(c2w_matrices[i_train])

        self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist()) 

        num_render = len(i_render)

        self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())  

        self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])  

        self.render_poses.extend([c2w_mat for c2w_mat in c2w_matrices[i_render]])


        near_depth = float(args.near)
        far_depth  = float(args.far)

        self.render_depth_range.extend([[near_depth, far_depth]] * num_render) 

        self.render_train_set_ids.extend([scene_id] * num_render)


    def __len__(self):
        return (len(self.render_rgb_files) * self.rays_per_spectrum if self.mode == "train" else len(self.render_rgb_files))

    
    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)

        rgb_file = self.render_rgb_files[idx]

        rgb = imageio.imread(rgb_file).astype(np.float32)[:, :, np.newaxis] / 255.0
        img_size = rgb.shape[: 2]

        render_pose = self.render_poses[idx]

        intrinsics = self.render_intrinsics[idx]  # torch.Size([4, 4])

        depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]

        train_rgb_files = self.train_rgb_files[train_set_id] 

        train_poses = self.train_poses[train_set_id]

        train_intrinsics = self.train_intrinsics[train_set_id]

        num_select = self.num_source_views 

        render_3d = render_pose[:3, 3]
        train_3d = train_poses[:, :3, 3]
        distances = torch.norm(train_3d - render_3d, dim=1)

        N = num_select * 2
        nearest_pose_ids = distances.argsort()[: N]

        nearest_pose_ids = np.random.choice(nearest_pose_ids, num_select, replace=False)

        img_size = rgb.shape[: 2]
        camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(np.float32)

        
        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32)[:, :, np.newaxis] / 255.0  # (90, 360, 1)
            train_pose = train_poses[id]              # torch.Size([4, 4])
            train_intrinsics_ = train_intrinsics[id]  # torch.Size([4, 4])

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[: 2]   # (90, 360)

            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(), train_pose.flatten())).astype(np.float32)  # (34,)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)        # (32, 90, 360, 1)
        src_cameras = np.stack(src_cameras, axis=0)  # (32, 34)

        depth_range = torch.tensor([depth_range[0] * 1.0, depth_range[1] * 1.0], dtype=torch.float64)

        return {
            "rgb": torch.from_numpy(rgb),                 
            "camera": torch.from_numpy(camera),            
            "rgb_path": rgb_file,                          
            "src_rgbs": torch.from_numpy(src_rgbs),       
            "src_cameras": torch.from_numpy(src_cameras),  
            "depth_range": depth_range,            
        }



