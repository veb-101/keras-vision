# import os
import gc
from typing import Tuple

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import ops as kops
from keras import layers as keras_layer
from .squeeze_excite import SEBlock

__all__ = ["ReparamLargeKernelConv"]


_DEPTHWISE_CONV = "depthwise"
_CONV = "conv2d"


class ReparamLargeKernelConv(keras_layer.Layer):
    """Building Block of RepLKNet

    This class defines overparameterized large kernel conv block
    introduced in `RepLKNet <https://arxiv.org/abs/2203.06717>`

    Reference: https://github.com/DingXiaoH/RepLKNet-pytorch
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        groups: int,
        small_kernel: int,
        inference_mode: bool = False,
        use_se: bool = False,
        activation: str | None = None,
        **kwargs,
    ) -> None:
        """Construct a ReparamLargeKernelConv module.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size of the large kernel conv branch.
            stride: Stride size. Default: 1
            groups: Group number. Default: 1
            small_kernel: Kernel size of small kernel conv branch.
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
            activation: Activation function. Default: ``gelu``
        """
        super().__init__(**kwargs)
        self.stride = stride
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation

        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        self.padding = self.kernel_size // 2

        self.use_se = use_se
        self.inference_mode = inference_mode

        self.se_layer_name = "se_layer"
        self.activation_layer_name = "activation_layer"

        # Conv type to use.
        self.conv_type = _DEPTHWISE_CONV if self.groups == self.in_channels else _CONV
        if self.conv_type == _DEPTHWISE_CONV:
            # When out_channels is a multiple of in_channels, the conv kernel's shape is (kH,kW, in_channel, depth_multiplier)
            # As it's a calculation that will be used many times, we make it as part of the initialization to make it more verbose.
            self.depth_multiplier = self.out_channels // self.in_channels

        if self.inference_mode:
            self._set_reparam_layers()
        else:
            self.lkb_origin = self._conv_bn(kernel_size=self.kernel_size, padding=self.padding, name=f"{self.name}_lkb_origin")
            if self.small_kernel is not None:
                assert self.small_kernel <= self.kernel_size, "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = self._conv_bn(kernel_size=self.small_kernel, padding=self.small_kernel // 2, name=f"{self.name}_small_conv")

        self.squeeze_excite = (
            SEBlock(self.out_channels, rd_ratio=0.25, name=self.se_layer_name) if self.use_se else keras_layer.Identity(name=self.se_layer_name)
        )

        self.activation_layer = (
            keras_layer.Activation(self.activation, name=self.activation_layer_name)
            if self.activation
            else keras_layer.Identity(name=self.activation_layer_name)
        )

    def _set_reparam_layers(self):
        self.lkb_reparam_zero_pad = keras_layer.ZeroPadding2D(padding=self.padding) if self.padding else None

        if self.conv_type == _CONV:
            self.lkb_reparam = keras_layer.Conv2D(
                filters=self.out_channels,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding="valid",
                groups=self.groups,
                use_bias=True,
            )
        else:
            self.lkb_reparam = keras_layer.DepthwiseConv2D(
                depth_multiplier=self.depth_multiplier,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding="valid",
                use_bias=True,
            )

    def _conv_bn(self, kernel_size: int, padding: str, name: str):
        """Helper method to construct conv-batchnorm layers.

        Args:
            kernel_size: Size of the convolution kernel.
            padding: Zero-padding size.

        Returns:
            Conv-BN module.
        """

        conv_bn_mod = keras.Sequential(name=name)

        if padding:
            conv_bn_mod.add(keras_layer.ZeroPadding2D(padding=padding))

        if self.conv_type == _CONV:
            conv_layer = keras_layer.Conv2D(
                filters=self.out_channels,
                kernel_size=kernel_size,
                strides=self.stride,
                padding="valid",
                groups=self.groups,
                use_bias=False,
            )
        else:
            conv_layer = keras_layer.DepthwiseConv2D(
                depth_multiplier=self.depth_multiplier,
                kernel_size=kernel_size,
                strides=self.stride,
                padding="valid",
                use_bias=False,
            )

        conv_bn_mod.add(conv_layer)
        conv_bn_mod.add(keras_layer.BatchNormalization(epsilon=1e-5))

        return conv_bn_mod

    def call(self, x):
        """Apply forward pass."""
        if self.inference_mode:
            out = self.lkb_reparam(self.lkb_reparam_zero_pad(x))
        else:
            out = self.lkb_origin(x)
            if hasattr(self, "small_conv"):
                out = out + self.small_conv(x)

        out = self.squeeze_excite(out)
        out = self.activation_layer(out)
        return out

    def reparameterize(self) -> None:
        """
        Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.
        """
        if self.inference_mode:
            return

        kernel, bias = self._get_kernel_bias()

        self.lkb_reparam.build((None, None, None, self.in_channels))
        self.lkb_reparam.set_weights([kernel, bias])

        self.__delattr__("lkb_origin")
        if hasattr(self, "small_conv"):
            self.__delattr__("small_conv")

        # * Removing layers and their weights that were part of the original model.
        # I can do this because I've provide specific names to those layers.
        layers_idx_to_remove = [idx for idx, layer in enumerate(self._layers) if self.name in layer.name]

        for i in reversed(layers_idx_to_remove):
            del self._layers[i]

        gc.collect()

        self.inference_mode = True
        # self.trainable = False

    def _get_kernel_bias(self) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepLKNet-pytorch

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        eq_k, eq_b = self._fuse_bn_tensor(self.lkb_origin)

        if hasattr(self, "small_conv"):
            small_k, small_b = self._fuse_bn_tensor(self.small_conv)
            eq_b += small_b

            pad = (self.kernel_size - self.small_kernel) // 2
            # To pad height and width only, we set padding on the first two dimensions:
            paddings = [
                (pad, pad),  # Pad kernel height
                (pad, pad),  # Pad kernel width
                (0, 0),  # No padding for channel dims
                (0, 0),
            ]
            eq_k = kops.pad(small_k, paddings, mode="constant")
        return eq_k, eq_b

    def _fuse_bn_tensor(self, branch: keras.Sequential) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Method to fuse batchnorm layer with conv layer.

        Args:
            branch: A convolutional layer with.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        _branch_len = len(branch.layers)  # As padding layer can be there in the branch.
        conv_params = branch.get_layer(index=_branch_len - 2).get_weights()  # [kernel, bias]
        bn_params = branch.get_layer(index=_branch_len - 1).get_weights()  # [gamma, beta, moving_mean, moving_variance]

        kernel = conv_params[0]
        running_mean = bn_params[2]
        running_var = bn_params[3]
        gamma = bn_params[0]
        beta = bn_params[1]
        eps = branch.layers[-1].epsilon

        std = kops.sqrt(running_var + eps)

        if self.conv_type == _CONV:
            _kernel_multiplier_reshape_dims = (1, 1, 1, -1)

        elif self.conv_type == _DEPTHWISE_CONV:
            _kernel_multiplier_reshape_dims = (1, 1, -1, self.depth_multiplier)

        t = kops.divide(gamma, std)
        t = kops.reshape(t, _kernel_multiplier_reshape_dims)

        # print(kernel.shape, t.shape)
        return kops.multiply(kernel, t), kops.multiply(beta - running_mean, kops.divide(gamma, std))

    def build(self, input_shape):
        self._set_reparam_layers()
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "out_channels": self.out_channels,
                "kernel_size": self.kernel_size,
                "stride": self.stride,
                "groups": self.groups,
                "small_kernel": self.small_kernel,
                "inference_mode": self.inference_mode,
                "use_se": self.use_se,
                "activation": self.activation,
            }
        )

        return config


def reparameterize_model(model: keras.Model | keras.Sequential) -> keras.Model | keras.Sequential:
    """Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    Args:
        model: MobileOne model in train mode.

    Returns:
        MobileOne model in inference mode.
    """
    for module in model.layers:
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


if __name__ == "__main__":
    import numpy as np

    in_channels = 16
    out_channels = in_channels * 3
    groups = in_channels
    patch_size = 3
    stride = 2
    inp = np.random.randn(1, 16, 16, in_channels)

    layer = ReparamLargeKernelConv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=patch_size,
        stride=stride,
        groups=groups,
        small_kernel=3,
    )
    layer(inp)

    model = keras.Sequential()
    model.add(keras.Input(shape=(None, None, in_channels)))
    model.add(layer)

    model.summary()

    print(model.layers[0]._layers[0].get_layer(index=1).get_weights()[0].shape)

    # # print(model.layers[1]._layers)
    # # print(model.layers[0]._layers[-1].weights)

    # # print(model.layers[1]._layers)
    # # # print(model.layers[1]._layers[-1].weights)
    # model = reparameterize_model(model)
    # model.summary()

    # print(model.layers[0]._layers)
    # # # print(model.layers[1]._layers)
