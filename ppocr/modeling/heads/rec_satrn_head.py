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

import math
import paddle
import copy
from paddle import nn
import paddle.nn.functional as F
from paddle.nn import LayerList
from paddle.nn.initializer import XavierNormal as xavier_uniform_
import numpy as np
from paddle.nn.initializer import Constant as constant_
from paddle.nn.initializer import XavierNormal as xavier_normal_

zeros_ = constant_(value=0.)
ones_ = constant_(value=1.)

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

class BaseDecoder(nn.Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def forward_train(self, feat, out_enc, targets, img_metas):
        raise NotImplementedError

    def forward_test(self, feat, out_enc, img_metas):
        raise NotImplementedError

    def forward(self,
                feat,
                out_enc,
                label=None,
                img_metas=None,
                train_mode=True):
        self.train_mode = train_mode

        if train_mode:
            return self.forward_train(feat, out_enc, label, img_metas)
        return self.forward_test(feat, out_enc, img_metas)

class PositionalEncoding(nn.Layer):
    """Fixed positional encoding with sine and cosine functions."""

    def __init__(self, d_hid=512, n_position=200, dropout=0):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Not a parameter
        # Position table of shape (1, n_position, d_hid)
        self.register_buffer(
            'position_table',
            self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = paddle.to_tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        # denominator = denominator.view(1, -1)
        denominator = paddle.reshape(denominator, (1, -1))
        pos_tensor = paddle.arange(0, n_position, dtype=paddle.float32).unsqueeze(1)
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = paddle.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = paddle.cos(sinusoid_table[:, 1::2])

        return sinusoid_table.unsqueeze(0)

    def forward(self, x):
        """
        Args:
            x (Tensor): Tensor of shape (batch_size, pos_len, d_hid, ...)
        """
        x = x + self.position_table[:, :x.size(1)].clone().detach()
        return self.dropout(x)

class Adaptive2DPositionalEncoding(nn.Layer):
    """Implement Adaptive 2D positional encoder for SATRN, see
      `SATRN <https://arxiv.org/abs/1910.04396>`_
      Modified from https://github.com/Media-Smart/vedastr
      Licensed under the Apache License, Version 2.0 (the "License");
    Args:
        d_hid (int): Dimensions of hidden layer.
        n_height (int): Max height of the 2D feature output.
        n_width (int): Max width of the 2D feature output.
        dropout (int): Size of hidden layers of the model.
    """

    def __init__(self,
                 d_hid=512,
                 n_height=100,
                 n_width=100,
                 dropout=0.1):
        super(Adaptive2DPositionalEncoding, self).__init__()
        h_position_encoder = self._get_sinusoid_encoding_table(n_height, d_hid)
        h_position_encoder = paddle.transpose(h_position_encoder, (0, 1))
        h_position_encoder = paddle.reshape(h_position_encoder, (1, d_hid, n_height, 1))

        w_position_encoder = self._get_sinusoid_encoding_table(n_width, d_hid)
        w_position_encoder = paddle.transpose(w_position_encoder, (0, 1))
        w_position_encoder = paddle.reshape(w_position_encoder, (1, d_hid, 1, n_width))

        self.register_buffer('h_position_encoder', h_position_encoder)
        self.register_buffer('w_position_encoder', w_position_encoder)

        self.h_scale = self.scale_factor_generate(d_hid)
        self.w_scale = self.scale_factor_generate(d_hid)
        self.pool = nn.AdaptiveAvgPool2D(1)
        self.dropout = nn.Dropout(p=dropout)

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """Sinusoid position encoding table."""
        denominator = paddle.to_tensor([
            1.0 / np.power(10000, 2 * (hid_j // 2) / d_hid)
            for hid_j in range(d_hid)
        ])
        denominator = paddle.reshape(denominator, (1, -1))
        pos_tensor = paddle.arange(0, n_position, dtype=paddle.float32).unsqueeze(1)
        sinusoid_table = pos_tensor * denominator
        sinusoid_table[:, 0::2] = paddle.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = paddle.cos(sinusoid_table[:, 1::2])

        return sinusoid_table

    def scale_factor_generate(self, d_hid):
        scale_factor = nn.Sequential(
            nn.Conv2D(d_hid, d_hid, kernel_size=1), nn.ReLU(),
            nn.Conv2D(d_hid, d_hid, kernel_size=1), nn.Sigmoid())

        return scale_factor

    def forward(self, x):
        b, c, h, w = x.shape

        avg_pool = self.pool(x)

        h_pos_encoding = \
            self.h_scale(avg_pool) * self.h_position_encoder[:, :, :h, :]
        w_pos_encoding = \
            self.w_scale(avg_pool) * self.w_position_encoder[:, :, :, :w]

        out = x + h_pos_encoding + w_pos_encoding

        out = self.dropout(out)

        return out

class PositionwiseFeedForward(nn.Layer):
    """Two-layer feed-forward module.

    Args:
        d_in (int): The dimension of the input for feedforward
            network model.
        d_hid (int): The dimension of the feedforward
            network model.
        dropout (float): Dropout layer on feedforward output.
    """

    def __init__(self, d_in, d_hid, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x

class LocalityAwareFeedforward(nn.Layer):
    """Locality-aware feedforward layer in SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_
    """

    def __init__(self,
                 d_in,
                 d_hid,
                 dropout=0.1):
        super(LocalityAwareFeedforward, self).__init__()
        self.conv1 = ConvModule(
            d_in,
            d_hid,
            kernel_size=1,
            padding=0,
            bias_attr=False)

        self.depthwise_conv = ConvModule(
            d_hid,
            d_hid,
            kernel_size=3,
            padding=1,
            bias_attr=False,
            groups=d_hid)

        self.conv2 = ConvModule(
            d_hid,
            d_in,
            kernel_size=1,
            padding=0,
            bias_attr=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.depthwise_conv(x)
        x = self.conv2(x)

        return x

def masked_fill(x, mask, value):
    y = paddle.full(x.shape, value, x.dtype)
    return paddle.where(mask, y, x)

class ScaledDotProductAttention(nn.Layer):
    """Scaled Dot-Product Attention Module. This code is adopted from
    https://github.com/jadore801120/attention-is-all-you-need-pytorch.

    Args:
        temperature (float): The scale factor for softmax input.
        attn_dropout (float): Dropout layer on attn_output_weights.
    """

    def __init__(self, temperature, attn_dropout=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        k = paddle.transpose(k, (0, 1, 3, 2))
        attn = paddle.matmul(q / self.temperature, k)

        if mask is not None:
            attn = masked_fill(attn, mask == 0, float('-inf')) 

        attn = self.dropout(F.softmax(attn, axis=-1))
        output = paddle.matmul(attn, v)

        return output, attn

class MultiHeadAttention(nn.Layer):
    """Multi-Head Attention module.

    Args:
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
    """

    def __init__(self,
                 n_head=8,
                 d_model=512,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.dim_k = n_head * d_k
        self.dim_v = n_head * d_v

        self.linear_q = nn.Linear(self.dim_k, self.dim_k, bias_attr=qkv_bias)
        self.linear_k = nn.Linear(self.dim_k, self.dim_k, bias_attr=qkv_bias)
        self.linear_v = nn.Linear(self.dim_v, self.dim_v, bias_attr=qkv_bias)

        self.attention = ScaledDotProductAttention(d_k**0.5, dropout)

        self.fc = nn.Linear(self.dim_v, d_model, bias_attr=qkv_bias)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, len_q, _ = q.shape
        _, len_k, _ = k.shape

        q = self.linear_q(q)
        k = self.linear_k(k)
        v = self.linear_v(v)

        q = paddle.reshape(q, (batch_size, len_q, self.n_head, self.d_k))
        k = paddle.reshape(k, (batch_size, len_k, self.n_head, self.d_k))
        v = paddle.reshape(v, (batch_size, len_k, self.n_head, self.d_v))

        q = paddle.transpose(q, (0, 2, 1))
        k = paddle.transpose(k, (0, 2, 1))
        v = paddle.transpose(v, (0, 2, 1))

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)

        attn_out, _ = self.attention(q, k, v, mask=mask)

        attn_out = paddle.transpose(attn_out, (0, 2, 1))
        attn_out = paddle.reshape(attn_out, (batch_size, len_q, self.dim_v))

        attn_out = self.fc(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out

class SatrnEncoderLayer(nn.Layer):
    """"""
    def __init__(self,
                 d_model=512,
                 d_inner=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False):
        super(SatrnEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
                n_head=n_head,
                d_model=d_model, 
                d_k=d_k,
                d_v=d_v,
                dropout=0.1,
                qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = LocalityAwareFeedforward(
            d_model, d_inner, dropout=dropout)

    def forward(self, x, h, w, mask=None):
        n, hw, c = x.shape
        residual = x
        x = self.norm1(x)
        x = residual + self.attn(x, x, x, mask)
        residual = x
        x = self.norm2(x)
        x = x.transpose(1, 2).contiguous().view(n, c, h, w)
        x = self.feed_forward(x)
        x = x.view(n, c, hw).transpose(1, 2)
        x = residual + x
        return x

class SatrnEncoder(nn.Layer):
    """Implement encoder for SATRN, see `SATRN.

    <https://arxiv.org/abs/1910.04396>`_.

    Args:
        n_layers (int): Number of attention layers.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        d_inner (int): Hidden dimension of feedforward layers.
        dropout (float): Dropout rate.
        init_cfg (dict or list[dict], optional): Initialization configs.
    """

    def __init__(self,
                 n_layers=12,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 n_position=100,
                 d_inner=256,
                 dropout=0.1,
                 **kwargs):
        super(SatrnEncoder, self).__init__()
        self.d_model = d_model
        self.position_enc = Adaptive2DPositionalEncoding(
            d_hid=d_model,
            n_height=n_position,
            n_width=n_position,
            dropout=dropout)
        self.layer_stack = LayerList([
            SatrnEncoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, feat, img_metas=None):
        """
        Args:
            feat (Tensor): Feature tensor of shape :math:`(N, D_m, H, W)`.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: A tensor of shape :math:`(N, T, D_m)`.
        """
        if img_metas is not None:
            assert len(img_metas[0]) == feat.shape[0]

        valid_ratios = None
        if img_metas is not None:
            valid_ratios = img_metas[-1]

        feat += self.position_enc(feat)
        n, c, h, w = feat.shape

        mask = None
        if valid_ratios is not None:
            mask = paddle.zeros(shape=[n, c, h, w], dtype='bool')
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(w, math.ceil(w * valid_ratio))
                if valid_width < w:
                    mask[i, :, :, :valid_width] = True
            mask = paddle.reshape(mask, (n, c, h * w))
        feat = paddle.reshape(feat, (n, c, h * w))
        output = paddle.transpose(feat, (0, 2, 1))
        for enc_layer in self.layer_stack:
            output = enc_layer(output, h, w, mask)
        output = self.layer_norm(output)

        return output

class TFDecoderLayer(nn.Layer):
    """Transformer Decoder Layer.

    Args:
        d_model (int): The number of expected features
            in the decoder inputs (default=512).
        d_inner (int): The dimension of the feedforward
            network model (default=256).
        n_head (int): The number of heads in the
            multiheadattention models (default=8).
        d_k (int): Total number of features in key.
        d_v (int): Total number of features in value.
        dropout (float): Dropout layer on attn_output_weights.
        qkv_bias (bool): Add bias in projection layer. Default: False.
        act_cfg (dict): Activation cfg for feedforward module.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'enc_dec_attn',
            'norm', 'ffn', 'norm') or ('norm', 'self_attn', 'norm',
            'enc_dec_attn', 'norm', 'ffn').
            Default：None.
    """

    def __init__(self,
                 d_model=512,
                 d_inner=256,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 dropout=0.1,
                 qkv_bias=False,
                 operation_order=None):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.self_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        self.enc_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout, qkv_bias=qkv_bias)

        self.mlp = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)

        self.operation_order = operation_order
        if self.operation_order is None:
            self.operation_order = ('norm', 'self_attn', 'norm',
                                    'enc_dec_attn', 'norm', 'ffn')
        assert self.operation_order in [
            ('norm', 'self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn'),
            ('self_attn', 'norm', 'enc_dec_attn', 'norm', 'ffn', 'norm')
        ]

    def forward(self,
                dec_input,
                enc_output,
                self_attn_mask=None,
                dec_enc_attn_mask=None):
        if self.operation_order == ('self_attn', 'norm', 'enc_dec_attn',
                                    'norm', 'ffn', 'norm'):
            dec_attn_out = self.self_attn(dec_input, dec_input, dec_input,
                                          self_attn_mask)
            dec_attn_out += dec_input
            dec_attn_out = self.norm1(dec_attn_out)

            enc_dec_attn_out = self.enc_attn(dec_attn_out, enc_output,
                                             enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out
            enc_dec_attn_out = self.norm2(enc_dec_attn_out)

            mlp_out = self.mlp(enc_dec_attn_out)
            mlp_out += enc_dec_attn_out
            mlp_out = self.norm3(mlp_out)
        elif self.operation_order == ('norm', 'self_attn', 'norm',
                                      'enc_dec_attn', 'norm', 'ffn'):
            dec_input_norm = self.norm1(dec_input)
            dec_attn_out = self.self_attn(dec_input_norm, dec_input_norm,
                                          dec_input_norm, self_attn_mask)
            dec_attn_out += dec_input

            enc_dec_attn_in = self.norm2(dec_attn_out)
            enc_dec_attn_out = self.enc_attn(enc_dec_attn_in, enc_output,
                                             enc_output, dec_enc_attn_mask)
            enc_dec_attn_out += dec_attn_out

            mlp_out = self.mlp(self.norm3(enc_dec_attn_out))
            mlp_out += enc_dec_attn_out

        return mlp_out

class TFDecoder(BaseDecoder):
    """Transformer Decoder block with self attention mechanism.

    Args:
        n_layers (int): Number of attention layers.
        d_embedding (int): Language embedding dimension.
        n_head (int): Number of parallel attention heads.
        d_k (int): Dimension of the key vector.
        d_v (int): Dimension of the value vector.
        d_model (int): Dimension :math:`D_m` of the input from previous model.
        d_inner (int): Hidden dimension of feedforward layers.
        n_position (int): Length of the positional encoding vector. Must be
            greater than ``max_seq_len``.
        dropout (float): Dropout rate.
        num_classes (int): Number of output classes :math:`C`.
        max_seq_len (int): Maximum output sequence length :math:`T`.
        start_idx (int): The index of `<SOS>`.
        padding_idx (int): The index of `<PAD>`.
        init_cfg (dict or list[dict], optional): Initialization configs.

    Warning:
        This decoder will not predict the final class which is assumed to be
        `<PAD>`. Therefore, its output size is always :math:`C - 1`. `<PAD>`
        is also ignored by loss as specified in
        :obj:`mmocr.models.textrecog.recognizer.EncodeDecodeRecognizer`.
    """

    def __init__(self,
                 n_layers=6,
                 d_embedding=512,
                 n_head=8,
                 d_k=64,
                 d_v=64,
                 d_model=512,
                 d_inner=256,
                 n_position=200,
                 dropout=0.1,
                 num_classes=93,
                 max_seq_len=40,
                 start_idx=1,
                 padding_idx=92,
                 **kwargs):
        super().__init__()

        self.padding_idx = padding_idx
        self.start_idx = start_idx
        self.max_seq_len = max_seq_len

        self.trg_word_emb = nn.Embedding(
            num_classes, d_embedding, padding_idx=padding_idx)

        self.position_enc = PositionalEncoding(
            d_embedding, n_position=n_position)
        self.dropout = nn.Dropout(p=dropout)

        self.layer_stack = LayerList([
            TFDecoderLayer(
                d_model, d_inner, n_head, d_k, d_v, dropout=dropout, **kwargs)
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(d_model, epsilon=1e-6)

        pred_num_class = num_classes - 1  # ignore padding_idx
        self.classifier = nn.Linear(d_model, pred_num_class)

    @staticmethod
    def get_pad_mask(seq, pad_idx):

        return (seq != pad_idx).unsqueeze(-2)

    @staticmethod
    def get_subsequent_mask(seq):
        """For masking out the subsequent info."""
        len_s = seq.size(1)
        subsequent_mask = 1 - paddle.triu(
            paddle.ones((len_s, len_s), dtype=bool), diagonal=1)
        subsequent_mask = subsequent_mask.unsqueeze(0)

        return subsequent_mask

    def _attention(self, trg_seq, src, src_mask=None):
        trg_embedding = self.trg_word_emb(trg_seq)
        trg_pos_encoded = self.position_enc(trg_embedding)
        tgt = self.dropout(trg_pos_encoded)

        trg_mask = self.get_pad_mask(
            trg_seq,
            pad_idx=self.padding_idx) & self.get_subsequent_mask(trg_seq)
        output = tgt
        for dec_layer in self.layer_stack:
            output = dec_layer(
                output,
                src,
                self_attn_mask=trg_mask,
                dec_enc_attn_mask=src_mask)
        output = self.layer_norm(output)

        return output

    def _get_mask(self, logit, img_metas):
        valid_ratios = None
        if img_metas is not None:
            valid_ratios = img_metas[-1]
        N, T, d = logit.shape
        mask = None
        if valid_ratios is not None:
            mask = paddle.zeros(shape=[N, T, d], dtype='bool')
            for i, valid_ratio in enumerate(valid_ratios):
                valid_width = min(T, math.ceil(T * valid_ratio))
                mask[i, :valid_width] = True

        return mask

    def forward_train(self, feat, out_enc, targets_dict, img_metas):
        r"""
        Args:
            feat (None): Unused.
            out_enc (Tensor): Encoder output of shape :math:`(N, T, D_m)`
                where :math:`D_m` is ``d_model``.
            targets_dict (dict): A dict with the key ``padded_targets``, a
                tensor of shape :math:`(N, T)`. Each element is the index of a
                character.
            img_metas (dict): A dict that contains meta information of input
                images. Preferably with the key ``valid_ratio``.

        Returns:
            Tensor: The raw logit tensor. Shape :math:`(N, T, C)`.
        """
        src_mask = self._get_mask(out_enc, img_metas)
        targets = targets_dict['padded_targets'].to(out_enc.device)
        attn_output = self._attention(targets, out_enc, src_mask=src_mask)
        outputs = self.classifier(attn_output)

        return outputs

    def forward_test(self, feat, out_enc, img_metas):
        src_mask = self._get_mask(out_enc, img_metas)
        N = out_enc.size(0)
        init_target_seq = paddle.full((N, self.max_seq_len + 1),
                                     self.padding_idx,
                                     dtype='int64')
        # bsz * seq_len
        init_target_seq[:, 0] = self.start_idx

        outputs = []
        for step in range(0, self.max_seq_len):
            decoder_output = self._attention(
                init_target_seq, out_enc, src_mask=src_mask)
            # bsz * seq_len * C
            step_result = F.softmax(
                self.classifier(decoder_output[:, step, :]), axis=-1)
            # bsz * num_classes
            outputs.append(step_result)
            step_max_index = paddle.argmax(step_result, axis=-1)
            init_target_seq[:, step + 1] = step_max_index

        outputs = paddle.stack(outputs, axis=1)

        return outputs

class STARNHead(nn.Layer):
    def __init__(self,
                 encoder_layers=6,
                 encoder_heads=8,
                 encoder_dk=32,
                 encoder_dv=32,
                 encoder_dmodel=256,
                 encoder_dinner=256 * 4,
                 encoder_dropout=0.1,
                 decoder_layers=6,
                 decoder_emb=256,
                 decoder_heads=8,
                 decoder_dk=32,
                 decoder_dv=32,
                 decoder_dmodel=256,
                 decoder_dinner=256 * 4,
                 decoder_dropout=0.1,
                 num_classes=93,
                 start_idx=91,
                 padding_idx=92,
                 **kwargs):
        super(STARNHead, self).__init__()

        # encoder module
        self.encoder = SatrnEncoder(
            n_layers=encoder_layers,
            n_head=encoder_heads,
            d_k=encoder_dk,
            d_v=encoder_dv,
            d_model=encoder_dmodel,
            n_position=100,
            d_inner=encoder_dinner,
            dropout=encoder_dropout,)

        # decoder module
        self.decoder = TFDecoder(
            n_layers=decoder_layers,
            d_embedding=decoder_emb,
            n_head=decoder_heads,
            d_k=decoder_dk,
            d_v=decoder_dv,
            d_model=decoder_dmodel,
            d_inner=decoder_dinner,
            n_position=200,
            dropout=decoder_dropout,
            num_classes=num_classes,
            max_seq_len=40,
            start_idx=start_idx,
            padding_idx=padding_idx,)

    def forward(self, feat, targets=None):
        '''
        img_metas: [label, valid_ratio]
        '''
        out_enc = self.encoder(feat, targets)  # bsz c

        if self.training:
            label = targets[0]  # label
            label = paddle.to_tensor(label, dtype='int64')
            final_out = self.decoder(
                feat, out_enc, label, img_metas=targets)
        if not self.training:
            final_out = self.decoder(
                feat,
                out_enc,
                label=None,
                img_metas=targets,
                train_mode=False)
            # (bsz, seq_len, num_classes)

        return final_out

