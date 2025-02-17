from torch import nn
import torch.nn.functional as F
from modules.util import DownBlock2d, SameBlock2d


class StylizerDiscrim(nn.Module):
    """
    Discriminator serving to update dense motion generator.
    """

    def __init__(self, num_channels=3, block_expansion=32, num_blocks=3, max_features=512):
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

    def forward(self, x):
        # save the feature maps, convenient for visualization
        feature_maps = []
        x = self.first(x)
        for down_block in self.down_blocks:
            x = down_block(x)
            feature_maps.append(x)
            x = feature_maps[-1]
        prediction_map = self.conv(x)

        return feature_maps, prediction_map
