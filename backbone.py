import torch
import torch.nn as nn
from IPython import embed

from geotransformer.modules.kpconv import ConvBlock, ResidualBlock, UnaryBlock, LastUnaryBlock, nearest_upsample
from utils.riconv_utils import RIConv2SetAbstraction, compute_norms

class RIKPConvFPN(nn.Module):
    def __init__(self, nsample, in_channel, input_dim, output_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm):
        super(RIKPConvFPN, self).__init__()
        
        self.encoder0 = RIConv2SetAbstraction(nsample, in_channel= 0+in_channel, mlp=[input_dim])

        self.encoder1_1 = ConvBlock(input_dim, init_dim, kernel_size, init_radius, init_sigma, group_norm)
        self.encoder1_2 = ResidualBlock(init_dim, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm)

        self.encoder2_1 = ResidualBlock(
            init_dim * 2, init_dim * 2, kernel_size, init_radius, init_sigma, group_norm, strided=True
        )
        self.encoder2_2 = ResidualBlock(
            init_dim * 2, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )
        self.encoder2_3 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm
        )

        self.encoder3_1 = ResidualBlock(
            init_dim * 4, init_dim * 4, kernel_size, init_radius * 2, init_sigma * 2, group_norm, strided=True
        )
        self.encoder3_2 = ResidualBlock(
            init_dim * 4, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )
        self.encoder3_3 = ResidualBlock(
            init_dim * 8, init_dim * 8, kernel_size, init_radius * 4, init_sigma * 4, group_norm
        )

        self.decoder2 = UnaryBlock(init_dim * 12, init_dim * 4, group_norm)
        self.decoder1 = LastUnaryBlock(init_dim * 6, output_dim)
        self.radius_normal = init_radius

    def forward(self, points_list, neighbors_list, subsampling_list, upsampling_list):
        feats_list = []

        points_f = points_list[0].unsqueeze(0)
        norms_f = compute_norms(points_f)
        feats_s0, group_idx = self.encoder0(points_f, norms_f, None)
        feats_s0 = feats_s0.squeeze(0)
        group_idx = group_idx.squeeze(0)

        feats_s1 = self.encoder1_1(feats_s0, points_list[0].detach(), points_list[0].detach(), neighbors_list[0].detach())
        feats_s1 = self.encoder1_2(feats_s1, points_list[0].detach(), points_list[0].detach(), neighbors_list[0].detach())

        feats_s2 = self.encoder2_1(feats_s1, points_list[1].detach(), points_list[0].detach(), subsampling_list[0].detach())
        feats_s2 = self.encoder2_2(feats_s2, points_list[1].detach(), points_list[1].detach(), neighbors_list[1].detach())
        feats_s2 = self.encoder2_3(feats_s2, points_list[1].detach(), points_list[1].detach(), neighbors_list[1].detach())

        feats_s3 = self.encoder3_1(feats_s2, points_list[2].detach(), points_list[1].detach(), subsampling_list[1].detach())
        feats_s3 = self.encoder3_2(feats_s3, points_list[2].detach(), points_list[2].detach(), neighbors_list[2].detach())
        feats_s3 = self.encoder3_3(feats_s3, points_list[2].detach(), points_list[2].detach(), neighbors_list[2].detach())

        latent_s3 = feats_s3
        feats_list.append(feats_s3)

        latent_s2 = nearest_upsample(latent_s3, upsampling_list[1])
        latent_s2 = torch.cat([latent_s2, feats_s2], dim=1)
        latent_s2 = self.decoder2(latent_s2)

        feats_list.append(feats_s0)

        latent_s1 = nearest_upsample(latent_s2, upsampling_list[0])
        latent_s1 = torch.cat([latent_s1, feats_s1], dim=1)
        latent_s1 = self.decoder1(latent_s1)
        feats_list.append(latent_s1)

        feats_list.reverse()

        return feats_list, group_idx