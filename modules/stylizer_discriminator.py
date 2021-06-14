from torch import nn
import torch.nn.functional as F
from modules.util import kp2gaussian, DownBlock2d, SameBlock2d
import torch


class StylizerDiscrim(nn.Module):
    """
    Discriminator serving to update dense motion generator.
    """

    def __init__(self, num_channels=3, block_expansion=32, num_blocks=3, max_features=512,
                 sn=False, use_kp=False, num_kp=10, kp_variance=0.01, **kwargs):
        super(StylizerDiscrim, self).__init__()
        
        # first 维持输入和输出维度相同，channel数量不同
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        # 声明下采样层
        down_blocks = []
        for i in range(num_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # 注：此处才是用列表声明网络层的正确用法
        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(min(max_features, block_expansion * (2 ** num_blocks)), out_channels=1, kernel_size=1)

    def forward(self, x, kp=None):
        # save the feature maps, convenient for visualization
        feature_maps = []
        out = x
        for down_block in self.down_blocks:
            feature_maps.append(down_block(out))
            out = feature_maps[-1]
        prediction_map = self.conv(out)

        return feature_maps, prediction_map
