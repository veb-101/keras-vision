from typing import Union

import keras.ops as kops  # type: ignore
from keras.layers import Layer, Conv2D, BatchNormalization, Activation, DepthwiseConv2D, Identity, ZeroPadding2D  # type: ignore

from .utils import make_divisible


class ConvLayer(Layer):
    def __init__(
        self,
        num_filters: int = 16,
        kernel_size: int = 3,
        strides: int = 2,
        use_activation: bool = True,
        use_bn: bool = True,
        use_bias: bool = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.use_bn = use_bn
        self.use_activation = use_activation
        self.use_bias = use_bias if use_bias is not None else (False if self.use_bn else True)

        if self.strides == 2:
            self.zero_pad = ZeroPadding2D(padding=(1, 1))
            conv_padding = "valid"
        else:
            self.zero_pad = Identity()
            conv_padding = "same"
        self.conv = Conv2D(filters=self.num_filters, kernel_size=self.kernel_size, strides=self.strides, padding=conv_padding, use_bias=self.use_bias)

        if self.use_bn:
            self.bn = BatchNormalization(epsilon=1e-05, momentum=0.1)

        if self.use_activation:
            self.activation = Activation("swish")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, **kwargs):
        x = self.zero_pad(x)
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)

        if self.use_activation:
            x = self.activation(x)

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_filters": self.num_filters,
                "kernel_size": self.kernel_size,
                "strides": self.strides,
                "use_bias": self.use_bias,
                "use_activation": self.use_activation,
                "use_bn": self.use_bn,
            }
        )
        return config


# Code taken from: https://github.com/veb-101/Training-Mobilenets-From-Scratch/blob/main/mobilenet_v2.py
class InvertedResidualBlock(Layer):
    def __init__(
        self,
        in_channels: int = 32,
        out_channels: int = 64,
        depthwise_stride: int = 1,
        expansion_factor: Union[int, float] = 2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Input Parameters

        self.num_in_channels = in_channels
        self.out_channels = out_channels
        self.depthwise_stride = depthwise_stride
        self.expansion_factor = expansion_factor

        num_out_channels = int(make_divisible(self.out_channels, divisor=8))
        expansion_channels = int(make_divisible(self.expansion_factor * self.num_in_channels))

        # Layer Attributes
        apply_expansion = expansion_channels > self.num_in_channels
        self.residual_connection = True if (self.num_in_channels == num_out_channels) and (self.depthwise_stride == 1) else False

        # Layers
        if apply_expansion:
            self.expansion_conv_block = ConvLayer(num_filters=expansion_channels, kernel_size=1, strides=1, use_activation=True, use_bn=True)
        else:
            self.expansion_conv_block = Identity()

        self.depthwise_conv_zero_pad = ZeroPadding2D(padding=(1, 1))
        self.depthwise_conv = DepthwiseConv2D(kernel_size=3, strides=self.depthwise_stride, padding="valid", use_bias=False)
        self.bn = BatchNormalization(epsilon=1e-05, momentum=0.1)
        self.activation = Activation("swish")
        self.out_conv_block = ConvLayer(num_filters=num_out_channels, kernel_size=1, strides=1, use_activation=False, use_bn=True)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, data, **kwargs):
        out = self.expansion_conv_block(data)
        out = self.depthwise_conv_zero_pad(out)
        out = self.depthwise_conv(out)
        out = self.bn(out)
        out = self.activation(out)
        out = self.out_conv_block(out)

        if self.residual_connection:
            return out + data

        return out

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.num_in_channels,
                "out_channels": self.out_channels,
                "depthwise_stride": self.depthwise_stride,
                "expansion_factor": self.expansion_factor,
            }
        )
        return config
