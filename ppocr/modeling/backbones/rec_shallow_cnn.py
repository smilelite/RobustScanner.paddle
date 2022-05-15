# copyright (c) 2021 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This code is refer from: 
https://github.com/open-mmlab/mmocr/blob/main/mmocr/models/textrecog/backbones/shallow_cnn.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from paddle import ParamAttr
import paddle.nn as nn
import paddle.nn.functional as F

__all__ = ["ShallowCNN"]


class ConvModule(nn.Layer):
    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias_attr=False,
                 ):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2D(in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            weight_attr=nn.initializer.KaimingNormal(),
            bias_attr=bias_attr)
        self.bn = nn.BatchNorm2D(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out


class ShallowCNN(nn.Layer):
    """Implement Shallow CNN block for SATRN.

    SATRN: `On Recognizing Texts of Arbitrary Shapes with 2D Self-Attention
    <https://arxiv.org/pdf/1910.04396.pdf>`_.

    Args:
        base_channels (int): Number of channels of input image tensor
            :math:`D_i`.
        hidden_dim (int): Size of hidden layers of the model :math:`D_m`.
    """

    def __init__(self,
                 in_channels=3,
                 out_channels=512):
        super(ShallowCNN, self).__init__()

        self.conv1 = ConvModule(
            in_channels,
            out_channels // 2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.conv2 = ConvModule(
            out_channels // 2,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias_attr=False)
        self.pool = nn.MaxPool2D(kernel_size=2, stride=2, padding=0)
        self.out_channels = out_channels
        
    def forward(self, x):
        """
        Args:
            x (Tensor): Input image feature :math:`(N, D_i, H, W)`.

        Returns:
            Tensor: A tensor of shape :math:`(N, D_m, H/4, W/4)`.
        """

        x = self.conv1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.pool(x)

        return x
