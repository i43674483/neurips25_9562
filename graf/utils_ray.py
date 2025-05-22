import numpy as np
import torch


class RayGenerator(object):
    def __init__(self, data, device, render_stride=1):
        super().__init__()
        
        self.render_stride = render_stride
        self.rgb = data["rgb"] if "rgb" in data.keys() else None
        self.camera = data["camera"]       # torch.Size([1, 34])
        self.rgb_path = data["rgb_path"]
        self.depth_range = data["depth_range"]
        self.device = device
        W, H, self.intrinsics, self.c2w_mat = parse_anttnea_params(self.camera)
        self.batch_size = len(self.camera) 

        self.H = int(H[0])  # tensor([90.])
        self.W = int(W[0])  # tensor([360.])

        self.n_elevation = self.H
        self.n_azimuth = self.W
        self.rays_per_spectrum = self.H * self.W 

        self.rays_o, self.rays_d, self.tx_o = self.gen_rays_spectrum(self.intrinsics, self.c2w_mat)

        if self.rgb is not None: 
            self.rgb = self.rgb.reshape(-1, 1) 

        if "src_rgbs" in data.keys():
            self.src_rgbs = data["src_rgbs"]
        else:
            self.src_rgbs = None

        if "src_cameras" in data.keys():
            self.src_cameras = data["src_cameras"]

        else:
            self.src_cameras = None


    def gen_rays_spectrum(self, intrinsics, c2w_mat):
        """
        generate sample rays origin at gateway with resolution given by spectrum

        Parameters
        ----------
        azimuth : int. The number of azimuth angles
        elevation : int. The number of elevation angles

        Returns
        -------
        r_o : tensor. [n_rays, 3]. The origin of rays
        r_d : tensor. [n_rays, 3]. The direction of rays, unit vector

        self.intrinsics: gateway location
        self.c2w_mat: TX location and gateway rotation
        self.rays_o, self.rays_d = self.gen_rays_spectrum(self.intrinsics, self.c2w_mat)
        """

        azimuth = torch.linspace(1, 360, self.n_azimuth) / 180 * np.pi  # torch.Size([360])
        elevation = torch.linspace(1, 90, self.n_elevation) / 180 * np.pi  # torch.Size([90])

        azimuth = torch.tile(azimuth, (self.n_elevation,))
        
        elevation = torch.repeat_interleave(elevation, self.n_azimuth) 

        x = 1 * torch.cos(elevation) * torch.cos(azimuth)  # torch.Size([32400]), [n_azimuth * n_elevation], i.e., [n_rays] element-wise
        y = 1 * torch.cos(elevation) * torch.sin(azimuth)  # torch.Size([32400])
        z = 1 * torch.sin(elevation)                       # torch.Size([32400])

        r_d = torch.stack([x, y, z], dim=0)
        
        R = c2w_mat[0, :3, :3]
        
        r_d_w = R @ r_d 

        gateway_pos = intrinsics[0, :3, 3]

        r_o = torch.tile(gateway_pos, (self.rays_per_spectrum,)).reshape(-1, 3) 

        tx_pos_render = c2w_mat[0, :3, 3]
        tx_pos = torch.tile(tx_pos_render, (self.rays_per_spectrum,)).reshape(-1, 3)

        return r_o, r_d_w.T, tx_pos


    def get_all(self):

        ret = {
            "ray_o": self.rays_o.cuda(),                                                       
            "ray_d": self.rays_d.cuda(),                                                     
            "depth_range": self.depth_range.cuda(),                                           
            "camera": self.camera.cuda(),                                                    
            "rgb": self.rgb.cuda() if self.rgb is not None else None,                         
            "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,         
            "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
            "tx_o": self.tx_o.cuda(),                                                        
        }

        return ret


    def sample_random_pixel(self, N_rand):

        random_state = np.random.RandomState(1994)
        
        select_inds = random_state.choice(self.H * self.W, size=(N_rand,), replace=False)

        return select_inds


    def random_sample(self, N_rand):

        select_inds = self.sample_random_pixel(N_rand) 

        rays_o = self.rays_o[select_inds]  # torch.Size([512, 3])
        rays_d = self.rays_d[select_inds]  # torch.Size([512, 3])
        tx_o =   self.tx_o[select_inds]    # torch.Size([512, 3])

        if self.rgb is not None:
            rgb = self.rgb[select_inds]
            
        else:
            rgb = None

        ret = {
            "ray_o": rays_o.cuda(),
            "ray_d": rays_d.cuda(), 
            "camera": self.camera.cuda(),
            "depth_range": self.depth_range.cuda(),
            "rgb": rgb.cuda() if rgb is not None else None,
            "src_rgbs": self.src_rgbs.cuda() if self.src_rgbs is not None else None,
            "src_cameras": self.src_cameras.cuda() if self.src_cameras is not None else None,
            "selected_inds": select_inds,                                                    
            "tx_o": tx_o.cuda(),                                                            
        }
        
        return ret


def parse_anttnea_params(params):  
    H = params[:, 0]
    W = params[:, 1]
    intrinsics = params[:, 2:18].reshape((-1, 4, 4))  # torch.Size([1, 4, 4])
    c2w = params[:, 18:34].reshape((-1, 4, 4))        # torch.Size([1, 4, 4])

    return W, H, intrinsics, c2w

