import torch
import os
from graf.graf_neural_network import RadianceFieldTransformer
from graf.feature_network import ResUNet
from box import Box


def de_parallel(model):
    return model.module if hasattr(model, "module") else model


class GRAFWrapper(object):
    def __init__(self, args, out_folder):
        self.args = args

        device = torch.device("cuda:0")

        embed_size = 3 + 3 * 2 * 10

        self.net_coarse = RadianceFieldTransformer(
            args=args,
            in_channels=self.args.coarse_feat_dim,
            pos_dim=embed_size,
            view_dim=embed_size,
            tx_dim=embed_size,
            return_attention=args.N_importance > 0
        ).to(device)

        self.feature_net = ResUNet(
            coarse_out_ch=self.args.coarse_feat_dim,  
            fine_out_ch=self.args.fine_feat_dim,     
            single_net=self.args.single_net,
        ).to(device)

        learnable_params = list(self.net_coarse.parameters())
        learnable_params += list(self.feature_net.parameters())

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.net_coarse.parameters()},
                {"params": self.feature_net.parameters(), "lr": args.lrate_feature},
            ],
            lr=args.lrate_gnt,)

        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=args.lrate_decay_steps, gamma=args.lrate_decay_factor)

        self.start_iteration = self.load_from_ckpt(out_folder)


    def switch_to_eval(self):

        self.net_coarse.eval()
        self.feature_net.eval()


    def switch_to_train(self):
        self.net_coarse.train()
        self.feature_net.train()


    def save_model(self, filename):
        to_save = {
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "net_coarse": de_parallel(self.net_coarse).state_dict(),
            "feature_net": de_parallel(self.feature_net).state_dict(),
        }

        torch.save(to_save, filename)


    def load_model(self, filename, load_opt=True, load_scheduler=True):
        to_load = torch.load(filename)

        if load_opt:
            self.optimizer.load_state_dict(to_load["optimizer"])
        if load_scheduler:
            self.scheduler.load_state_dict(to_load["scheduler"])

        self.net_coarse.load_state_dict(to_load["net_coarse"])
        self.feature_net.load_state_dict(to_load["feature_net"])


    def load_from_ckpt(self, out_folder, force_latest_ckpt=False):

        ckpts = []
        if os.path.exists(out_folder):
            ckpts = [os.path.join(out_folder, f) for f in sorted(os.listdir(out_folder)) if f.endswith(".pth")]

        if self.args.ckpt_path is not None and not force_latest_ckpt:
            if os.path.isfile(self.args.ckpt_path):   # load the specified ckpt
                ckpts = [self.args.ckpt_path]

        if len(ckpts) > 0 and not self.args.no_reload:
            fpath = ckpts[-1]
            self.load_model(fpath, True, True)
            step = int(fpath[-10: -4])
            print("Reloading from {}, starting at step={}".format(fpath, step))
        else:
            print("No ckpts found, training from scratch...")
            step = 0

        return step



