import torch
from collections import OrderedDict
from graf.utils_render import render_rays


def spectrum_render_one(
    ray_sampler,
    ray_batch,
    model,
    projector,
    chunk_size,
    N_samples,
    inv_uniform=False,
    det=False,
    featmaps=None,
):


    all_ret = OrderedDict([("outputs_coarse", OrderedDict()), ("outputs_fine", OrderedDict())])
    N_rays = ray_batch["ray_o"].shape[0]    # 32400

    for i in range(0, N_rays, chunk_size):
        chunk = OrderedDict()

        for k in ray_batch: 
            if k in ["camera", "depth_range", "src_rgbs", "src_cameras"]:
                chunk[k] = ray_batch[k]
            elif ray_batch[k] is not None:  # ['ray_o', 'ray_d', 'rgb', 'tx_o']
                chunk[k] = ray_batch[k][i : i + chunk_size]
            else:
                chunk[k] = None
            
        ret = render_rays(
            chunk,
            model,
            featmaps,
            projector=projector,
            N_samples=N_samples,
            inv_uniform=inv_uniform,
            det=det
        )

        if i == 0:
            for k in ret["outputs_coarse"]:  # dict_keys(['rgb', 'weights', 'depth'])
                if ret["outputs_coarse"][k] is not None:
                    all_ret["outputs_coarse"][k] = []

        for k in ret["outputs_coarse"]:  # dict_keys(['rgb', 'weights', 'depth'])
            if ret["outputs_coarse"][k] is not None:
                all_ret["outputs_coarse"][k].append(ret["outputs_coarse"][k].cpu())

    for k in all_ret["outputs_coarse"]:
        if k == "random_sigma":
            continue

        tmp = torch.cat(all_ret["outputs_coarse"][k], dim=0).reshape((ray_sampler.H, ray_sampler.W, -1))

        # all_ret["outputs_coarse"][k] = tmp.squeeze()
        all_ret["outputs_coarse"][k] = tmp


    return all_ret
