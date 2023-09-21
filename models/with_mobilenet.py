import torch
from torch import nn

from modules.conv import conv, conv_dw, conv_dw_no_bn


class Cpm(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels),
            conv_dw_no_bn(out_channels, out_channels)
        )
        self.conv = conv(out_channels, out_channels, bn=False)

    def forward(self, x):
        x = self.align(x)
        x = self.conv(x + self.trunk(x))
        return x


class InitialStage(nn.Module):
    def __init__(self, num_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False),
            conv(num_channels, num_channels, bn=False)
        )
        self.heatmaps = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
            conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class RefinementStageBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
        self.trunk = nn.Sequential(
            conv(out_channels, out_channels),
            conv(out_channels, out_channels, dilation=2, padding=2)
        )

    def forward(self, x):
        initial_features = self.initial(x)
        trunk_features = self.trunk(initial_features)
        return initial_features + trunk_features


class RefinementStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_heatmaps, num_pafs):
        super().__init__()
        self.trunk = nn.Sequential(
            RefinementStageBlock(in_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels),
            RefinementStageBlock(out_channels, out_channels)
        )
        self.heatmaps = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
        )
        self.pafs = nn.Sequential(
            conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
            conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
        )

    def forward(self, x):
        trunk_features = self.trunk(x)
        heatmaps = self.heatmaps(trunk_features)
        pafs = self.pafs(trunk_features)
        return [heatmaps, pafs]


class PoseEstimationWithMobileNet(nn.Module):
    def __init__(self, num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
        super().__init__()
        self.model = nn.Sequential(
            conv(     3,  32, stride=2, bias=False),
            conv_dw( 32,  64),
            conv_dw( 64, 128, stride=2),
            conv_dw(128, 128),
            conv_dw(128, 256, stride=2),
            conv_dw(256, 256),
            conv_dw(256, 512),  # conv4_2
            conv_dw(512, 512, dilation=2, padding=2),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512),
            conv_dw(512, 512)   # conv5_5
        )
        self.cpm = Cpm(512, num_channels)

        self.initial_stage = InitialStage(num_channels, num_heatmaps, num_pafs)
        self.refinement_stages = nn.ModuleList()
        for idx in range(num_refinement_stages):
            self.refinement_stages.append(RefinementStage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                          num_heatmaps, num_pafs))

    def forward(self, x):
        backbone_features = self.model(x)
        backbone_features = self.cpm(backbone_features)

        stages_output = self.initial_stage(backbone_features)
        for refinement_stage in self.refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output

class PoseClassificationWithMobileNet(nn.Module):
    def __init__(self, pretrained, num_heatmaps=19, num_pafs=38, height=256):
        super().__init__()

        self.num_heatmaps = num_heatmaps
        self.num_pafs = num_pafs
        self.height = height

        self.pretrained = pretrained
        for param in self.pretrained.parameters():
            param.requires_grad = False

        ######## first try
        # self.classifier = nn.Linear((height + num_heatmaps + num_pafs) * 16**2, 1)
        # self.classifier = nn.Sequential(
        #     nn.Linear((height + num_heatmaps) * 16**2, 16),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),

        #     nn.Linear(16, 1),
        # )

        ######### secend try
        # self.hm_feature_extractor = nn.Sequential(
        #     nn.Conv2d((num_heatmaps), 32, 5, 1, 2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),

        #     nn.Conv2d(32, 32, 5, 2, 2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),

        #     nn.Conv2d(32, 32, 5, 1, 2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),

        #     nn.Conv2d(32, 32, 5, 2, 2),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),

        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(),

        #     nn.Linear(32, 1)
        # )

        ######## third try
        self.classifier = nn.Sequential(
            nn.Linear(self.num_heatmaps * 2, self.num_heatmaps),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(self.num_heatmaps, 1)
        )
        

    def forward(self, x):
        batch_size = x.size(0)

        with torch.no_grad():
            stages_output = self.pretrained(x)
        
        heatmap_feature = stages_output[-2]
        pafs_feature = stages_output[-1]
        # print(heatmap_feature.shape)
        hf_max_indices = heatmap_feature.view(batch_size, self.num_heatmaps, -1).argmax(dim=-1)
        # print(hf_max_indices.shape)
        hf_h_coords, hf_w_coords = hf_max_indices // 32, hf_max_indices % 32
        # print(hf_h_coords.shape, hf_w_coords.shape)
        coordinates = torch.stack([hf_h_coords, hf_w_coords], dim=-1).float()
        # print(coordinates.shape)
        subtracted_coords = coordinates - coordinates[:, 0].unsqueeze(1)
        # print(subtracted_coords.shape)
        norms = torch.norm(subtracted_coords, dim=2, keepdim=True)
        normalized_coords = subtracted_coords / (norms + 1e-10) # (batch_size, key_points, 2(coordinates))
        # print(normalized_coords.shape)

        output = self.classifier(normalized_coords.view(batch_size, -1))
        # print(output.shape)

        return output, stages_output