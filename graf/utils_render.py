import torch
from collections import OrderedDict


def sample_along_camera_ray(ray_o, ray_d, depth_range, N_samples, inv_uniform=False, det=False):
    """
    :param ray_o: origin of the ray in scene coordinate system; tensor of shape [N_rays, 3]
    :param ray_d: homogeneous ray direction vectors in scene coordinate system; tensor of shape [N_rays, 3]
    :param depth_range: [near_depth, far_depth]  # torch.Size([1, 2])
    :param inv_uniform: if True, uniformly sampling inverse depth
    :param det: if True, will perform deterministic sampling
    :return: tensor of shape [N_rays, N_samples, 3]
    """

    near_depth_value = depth_range[0, 0]  
    far_depth_value = depth_range[0, 1]   
    assert near_depth_value > 0 and far_depth_value > 0 and far_depth_value > near_depth_value

    near_depth = near_depth_value * torch.ones_like(ray_d[..., 0])
    far_depth = far_depth_value * torch.ones_like(ray_d[..., 0])

    if inv_uniform:
        start = 1.0 / near_depth                           
        step = (1.0 / far_depth - start) / (N_samples - 1)  

        inv_z_vals = torch.stack([start + i * step for i in range(N_samples)], dim=1)  # torch.Size([512, 64])

        z_vals = 1.0 / inv_z_vals

    else:
        start = near_depth
        step = (far_depth - near_depth) / (N_samples - 1)

        z_vals = torch.stack([start + i * step for i in range(N_samples)], dim=1)


    if not det:  
        mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])      # torch.Size([512, 63])

        upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)  # torch.Size([512, 64])

        lower = torch.cat([z_vals[:, 0:1], mids], dim=-1)  # torch.Size([512, 64])

        t_rand = torch.rand_like(z_vals)  # torch.Size([512, 64])

        z_vals = lower + (upper - lower) * t_rand  # [N_rays, N_samples], torch.Size([512, 64])


    ray_d = ray_d.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3], torch.Size([512, 64, 3])
    ray_o = ray_o.unsqueeze(1).repeat(1, N_samples, 1)  # [N_rays, N_samples, 3], torch.Size([512, 64, 3])

    pts = z_vals.unsqueeze(2) * ray_d + ray_o           # [N_rays, N_samples, 3], torch.Size([512, 64, 3])
    return pts, z_vals


def render_rays(
    ray_batch,
    model,
    featmaps,
    projector,
    N_samples,
    inv_uniform=False,
    det=False
):

    ret = {"outputs_coarse": None, "outputs_fine": None}
    ray_o, ray_d, tx_o = ray_batch["ray_o"], ray_batch["ray_d"], ray_batch["tx_o"]

    pts, z_vals = sample_along_camera_ray(
        ray_o=ray_o,
        ray_d=ray_d,
        depth_range=ray_batch["depth_range"],
        N_samples=N_samples,
        inv_uniform=inv_uniform,
        det=det,
    )

    N_rays, N_samples = pts.shape[: 2]  # N_rays: 512, N_samples: 64
    
    rgb_feat, ray_diff, mask = projector.compute(
        pts,                       # torch.Size([512, 64, 3])
        ray_batch["camera"],       # torch.Size([1, 34])
        ray_batch["src_rgbs"],     # torch.Size([1, 32, 90, 360, 1])
        ray_batch["src_cameras"],  # torch.Size([1, 32, 34])
        featmaps=featmaps[0],      # torch.Size([32, 32, 24, 92])
    )  # [N_rays, N_samples, N_views, x]

    rgb = model.net_coarse(rgb_feat, ray_diff, mask, pts, ray_d, tx_o)  # torch.Size([512, 1]), torch.Size([512, 65])

    ret["outputs_coarse"] = {"rgb": rgb}

    return ret
