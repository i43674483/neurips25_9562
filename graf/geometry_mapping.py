import torch
import torch.nn.functional as F


class Mapper:
    def __init__(self):

        self.device = torch.device("cuda:0")

    def inbound(self, pixel_locations, h, w):

        return (
            (pixel_locations[..., 0] <= w - 1.0)  # 
            & (pixel_locations[..., 0] >= 0)
            & (pixel_locations[..., 1] <= h - 1.0)
            & (pixel_locations[..., 1] >= 0)
        )


    def normalize(self, pixel_locations, h, w): 
        resize_factor = torch.tensor([w - 1.0, h - 1.0]).to(pixel_locations.device)[None, None, :]
        
        normalized_pixel_locations = (
            2 * pixel_locations / resize_factor - 1.0
        )

        return normalized_pixel_locations


    def compute_projections(self, xyz, train_cameras):

        original_shape = xyz.shape[: 2] 

        xyz = xyz.reshape(-1, 3)         

        num_views = len(train_cameras)   
        
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)
        
        
        """
        1 0 0 tx
        0 1 0 ty
        0 0 1 tz
        0 0 0 1
        """
        train_poses_xyz = train_cameras[:, -3:]

        """
        1 0 0 0  0
        0 1 0 0  0
        0 0 1 0  0
        tx ty tz 1

        """
        train_poses = torch.zeros(train_intrinsics.size(), device=train_intrinsics.device)
        train_poses[:, :3, :3] = torch.eye(3)
        train_poses[:, 3, :3] = train_poses_xyz
        train_poses[:, 3, 3] = 1
        
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1) 
        
        projections = train_intrinsics.bmm(torch.inverse(train_poses)).bmm(
            xyz_h.t()[None, ...].repeat(num_views, 1, 1)
        )

        projections = projections.permute(0, 2, 1)

        pixel_locations = projections[..., :2] / torch.clamp(
            projections[..., 2:3], min=1e-8
        ) 

        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)

        mask = projections[..., 2] > 0 
        
        return pixel_locations.reshape((num_views,) + original_shape + (2,)), mask.reshape(
            (num_views,) + original_shape)


    def compute_angle(self, xyz, query_camera, train_cameras):
        
        original_shape = xyz.shape[: 2]  
        xyz = xyz.reshape(-1, 3)

        c2w_train_poses = train_cameras[:, 18: 34].reshape(-1, 4, 4) 
        num_views = len(c2w_train_poses)                            

        c2w_query_pose = (query_camera[18: 34].reshape(-1, 4, 4).repeat(num_views, 1, 1))
        
        ray2tar_pose = c2w_query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)

        # Normalize the ray to have a unit length. This makes sure the ray is a direction vector.
        ray2tar_pose /= torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6 

        ray2train_pose = c2w_train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)

        ray2train_pose /= torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6

        ray_diff = ray2tar_pose - ray2train_pose  # torch.Size([32, 32768, 3])

        # Calculate the magnitude (norm) of this difference vector.
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)  

        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)  # torch.Size([32, 32768, 3])

        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True) 

        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)  # torch.Size([32, 32768, 4])

        ray_diff = ray_diff.reshape((num_views,) + original_shape + (4,))  # torch.Size([32, 512, 64, 4])

        return ray_diff


    def compute(self, xyz, query_camera, train_imgs, train_cameras, featmaps):

        train_imgs = train_imgs.squeeze(0)        

        train_cameras = train_cameras.squeeze(0)  
        query_camera = query_camera.squeeze(0)   

        rgb_feat_sampled = featmaps               

        ray_diff = self.compute_angle(xyz, query_camera, train_cameras)  
        ray_diff = ray_diff.permute(1, 2, 0, 3) 

        mask_size = xyz.size()[:2] + (train_imgs.size()[0], 1)

        mask = torch.zeros(mask_size, device=xyz.device)

        return rgb_feat_sampled, ray_diff, mask


