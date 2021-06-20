import torch
from torch import nn
import torch.nn.functional as F
from modules.util import ResBlock2d, SameBlock2d, UpBlock2d, DownBlock2d


class StylizerGenerator(nn.Module):
    """
    Generator that given original dense motion field generates a new stylized motion field trained on target
    stylized facial expression dataset.
    """
    # 本模块负责魔改运动场进行风格化
    # 输入了预测的运动场信息，输出重新生成的运动场信息
    # dense motion 在此模块进行修改

    def __init__(self, num_channels, block_expansion, max_features, num_down_blocks, num_bottleneck_blocks):
        super(StylizerGenerator, self).__init__()

        # first 维持输入和输出维度相同，channel数量不同
        self.first = SameBlock2d(num_channels, block_expansion, kernel_size=(7, 7), padding=(3, 3))

        # 声明下采样层
        down_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # 注：此处才是用列表声明网络层的正确用法
        # 储存时会通过储存对应self的名字所对应的内容
        self.down_blocks = nn.ModuleList(down_blocks)

        # 声明上采样层
        up_blocks = []
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_down_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features, kernel_size=(3, 3), padding=(1, 1)))
        # 注：此处用列表声明模块同上
        self.up_blocks = nn.ModuleList(up_blocks)

        # 声明生成器中的瓶颈层
        self.bottleneck = torch.nn.Sequential()
        in_features = min(max_features, block_expansion * (2 ** num_down_blocks))
        for i in range(num_bottleneck_blocks):
            self.bottleneck.add_module('r' + str(i), ResBlock2d(in_features, kernel_size=(3, 3), padding=(1, 1)))

        # 声明最后的输出层
        self.final = nn.Conv2d(block_expansion, num_channels, kernel_size=(7, 7), padding=(3, 3))
        self.num_channels = num_channels

    def forward(self, dense_motion):
        output_dict = {}
        # Encoding (downsampling) part
        # 此处是对运动场进行下采样
        out = self.first(dense_motion)
        for i in range(len(self.down_blocks)):
            out = self.down_blocks[i](out)

        # Bottleneck part
        out = self.bottleneck(out)
        # decoding part
        for i in range(len(self.up_blocks)):
            out = self.up_blocks[i](out)
        out = self.final(out)
        out = F.sigmoid(out)

        output_dict["prediction"] = out

        return output_dict
