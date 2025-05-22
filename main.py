

import os
import torch
import yaml
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader

import numpy as np
from box import Box
import skimage.metrics

from graf.graf_model import GRAFWrapper
from graf.geometry_mapping import Mapper
from graf.loss_function import LossFunction
from graf.utils_metric import img2mse, mse2psnr, img_HWC2CHW, colorize, cycle, img2psnr
from graf.utils_ray import RayGenerator
from graf.utils_render import render_rays
from graf.spectrum_render import spectrum_render_one
from utils.data_painter import paint_spectrum_compare
from graf.spectrum_dataset import SpectrumDataset


def train(args):
    random_seed_num = 1994

    device = torch.device("cuda:0")

    print(f"\nTRAIN MODE\n")

    ### Data path  
    basedir_sym = args.basedir                    
    basedir = os.path.realpath(basedir_sym)  

    data_dir_suffix = args.datadir_suffix             
    datadir = os.path.join(basedir, data_dir_suffix)

    ### Log path
    log_suffix = args.logdir_suffix
    rootdir = os.path.join(basedir, log_suffix)

    out_folder = os.path.join(rootdir, args.expname)
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    model_out_folder = os.path.join(out_folder, "ckpts")
    if not os.path.exists(model_out_folder):
        os.makedirs(model_out_folder)

    # save the args and config files
    f_path = os.path.join(out_folder, "config.yml")
    with open(f_path, "w") as file:
        for key, value in args.items():
            file.write(f"{key}: {value}\n")

    train_dataset = SpectrumDataset(args, datadir, args.mode)
    train_sampler = None

    def seed_worker(_):
        np.random.seed(random_seed_num)

    loader_config = {
        "batch_size": 1,
        "worker_init_fn": seed_worker,
        "num_workers": 8,
        "pin_memory": True,
        "sampler": train_sampler,
        "shuffle": train_sampler is None
    }

    train_data_loader = DataLoader(
        dataset=train_dataset,
        **loader_config
    )

    model = GRAFWrapper(args, model_out_folder)

    loss_function = LossFunction()
    log_values = dict()
    map_coordinates = Mapper()

    iteration_count = model.start_iteration + 1
    epoch = 0
    np.random.seed(random_seed_num)

    while iteration_count < model.start_iteration + args.n_iters + 1:
        for train_data in train_data_loader:

            ray_sampler = RayGenerator(train_data, device)

            num_src_pixels = train_data["src_rgbs"][0].shape[0]
            scaling_factor = args.num_source_views / num_src_pixels
            num_sampled_rays = int(args.N_rand * scaling_factor)

            random_ray_batch = ray_sampler.random_sample(num_sampled_rays)

            feature_intput = random_ray_batch["src_rgbs"].squeeze(0).permute(0, 3, 1, 2)
            feature_intput = feature_intput.repeat(1, 3, 1, 1)

            featmaps = model.feature_net(feature_intput)

            ret = render_rays(
                            ray_batch=random_ray_batch,
                            model=model,
                            featmaps=featmaps,
                            projector=map_coordinates,
                            N_samples=args.N_samples,
                            inv_uniform=args.inv_uniform,
                            det=args.det)
            
            model.optimizer.zero_grad()
            loss, log_values = loss_function(ret["outputs_coarse"], random_ray_batch, log_values)

            loss.backward()
            log_values["loss"] = loss.item()
            model.optimizer.step()
            model.scheduler.step()

            log_values["lr"] = model.scheduler.get_last_lr()[0]

            if iteration_count % args.i_weights == 0:

                print("Saving checkpoints at {} to {}...".format(iteration_count, model_out_folder))
                fpath = os.path.join(model_out_folder, "model_{:06d}.pth".format(iteration_count))
                model.save_model(fpath)

            if iteration_count % args.i_print == 0:
                msg = "Step {:>8}     --------------     Loss {:.6f}".format(
                    iteration_count, log_values["loss"]
                )
                print(msg)

            iteration_count += 1
            if iteration_count > model.start_iteration + args.n_iters + 1:
                break

        epoch += 1


def test(args):

    device = torch.device("cuda:0")

    print(f"\nTEST MODE\n")

    basedir_sym = args.basedir                    
    basedir = os.path.realpath(basedir_sym)

    data_dir_suffix = args.datadir_suffix             
    datadir = os.path.join(basedir, data_dir_suffix)  
    
    log_suffix = args.logdir_suffix 
    rootdir = os.path.join(basedir, log_suffix)  

    out_folder = os.path.join(rootdir, args.expname)     

    model_out_folder = os.path.join(out_folder, "ckpts")

    predict_dir_test = os.path.join(out_folder, "predictions")
    if not os.path.exists(predict_dir_test):
        os.makedirs(predict_dir_test)

    test_dataset = SpectrumDataset(args, datadir, args.mode)
    test_loader = DataLoader(test_dataset, batch_size=1)

    model = GRAFWrapper(args, model_out_folder)

    map_coordinates = Mapper()

    save_img_idx = 0
    for val_data in test_loader:

        tmp_ray_sampler = RayGenerator(val_data, device, render_stride=args.render_stride)

        H, W = tmp_ray_sampler.H, tmp_ray_sampler.W
        gt_img = tmp_ray_sampler.rgb.reshape(H, W, 1)

        psnr_value = synthesize_one_spectrum(
            save_img_idx,
            args,
            model,
            tmp_ray_sampler,
            map_coordinates,
            gt_img, 
            out_folder=predict_dir_test,
        )

        print(
            "\nTest sample index: {:>4}/{:<4} | PSNR: {:.3f}\n".format(
                save_img_idx + 1, len(test_dataset), psnr_value)
        )

        save_img_idx += 1

        torch.cuda.empty_cache()


@torch.no_grad()
def synthesize_one_spectrum(
    test_sample_idx,
    args,
    model,
    ray_sampler,
    map_coordinates,
    gt_img, 
    out_folder="",
    ):

    model.switch_to_eval()
    with torch.no_grad():

        random_ray_batch = ray_sampler.get_all()
        if model.feature_net is not None: 
            featmaps = model.feature_net(random_ray_batch["src_rgbs"].squeeze(0).repeat(1, 1, 1, 3).permute(0, 3, 1, 2))
        else:
            featmaps = [None, None]

        ret = spectrum_render_one(
            ray_sampler=ray_sampler,
            ray_batch=random_ray_batch,
            model=model,
            projector=map_coordinates,
            chunk_size=args.chunk_size,
            N_samples=args.N_samples,
            inv_uniform=args.inv_uniform,
            det=True,
            featmaps=featmaps,
        )


    average_im = ray_sampler.src_rgbs.cpu().mean(dim=(0, 1))

    rgb_gt = img_HWC2CHW(gt_img)
    average_im = img_HWC2CHW(average_im)

    rgb_pred = img_HWC2CHW(ret["outputs_coarse"]["rgb"].detach().cpu()) 

    rgb_gt = rgb_gt[0].numpy()
    rgb_pred = rgb_pred[0].numpy()

    filename = os.path.join(out_folder, f'{test_sample_idx}.png')

    paint_spectrum_compare(rgb_pred, rgb_gt, filename)

    psnr_value = skimage.metrics.peak_signal_noise_ratio(rgb_pred, rgb_gt, data_range=1)

    return psnr_value


if __name__ == '__main__':

    ## Loadiing configurations ##
    script_dir = os.path.dirname(os.path.realpath(__file__))
    print("\n\nThe script_path: {}\n\n".format(script_dir))

    config_file_name = "spectrum.yml"
    config_path = os.path.join(script_dir, "configs", config_file_name)

    with open(config_path) as f_path:
        kwargs = yaml.safe_load(f_path)

    kwargs = Box(kwargs)

    torch.cuda.set_device(kwargs.local_rank)
    
    if kwargs.mode == "train":

        train(kwargs)

    else:

        test(kwargs)



