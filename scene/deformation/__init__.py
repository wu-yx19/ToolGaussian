import torch
import torch.nn as nn
import torch.nn.init as init
from .hexplane import HexPlaneField

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)

class DeformationNet(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.D = args.defor_depth # depth
        self.W = args.net_width
        self.no_grid = args.no_grid

        self.hexplane = HexPlaneField(args.bounds, args.kplanes_config, args.multires)

        self.set_feature_net()
        self.set_deform_net()
        self.apply(initialize_weights) # go through all layers and initialize weights

    def set_feature_net(self):
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(self.hexplane.feat_dim, self.W)]
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)

    def set_deform_net(self):
        self.pos_deform, self.scales_deform, self.rotations_deform, self.opacity_deform = \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3)),\
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4)), \
            nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))


    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, opacity_emb = None, time_emb=None):

        # embedding -> (grid ->) feature_net -> feature
        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:
            grid_feature = self.hexplane(rays_pts_emb[:,:3], time_emb[:,:1]) # get rid of potential embeddings other than original
            h = grid_feature
        feature = self.feature_out(h)

        # feature -> deformation -> output
        if self.args.no_dx:
            pts = rays_pts_emb[:, :3]
        else:
            dx = self.pos_deform(feature)
            pts = rays_pts_emb[:, :3] + dx

        if self.args.no_ds:
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(feature)
            scales = scales_emb[:,:3] + ds

        if self.args.no_dr:
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(feature)
            rotations = rotations_emb[:,:4] + dr

        if self.args.no_do:
            opacity = opacity_emb[:,:1]
        else:
            do = self.opacity_deform(feature)
            opacity = opacity_emb[:,:1] + do

        return pts, scales, rotations, opacity

    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list

    def get_grid_parameters(self):
        return list(self.hexplane.parameters())
