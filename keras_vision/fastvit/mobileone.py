# import os
import gc
from typing import Union, Tuple

# import tensorflow as tf

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import ops as kops
from keras import layers as keras_layer

from .squeeze_excite import SEBlock

__all__ = ["MobileOneBlock", "reparameterize_model"]


_DEPTHWISE_CONV = "depthwise"
_CONV = "conv2d"


class MobileOneBlock(keras_layer.Layer):
    """MobileOne building block.

    This block has a multi-branched architecture at train-time
    and plain-CNN style architecture at inference time
    For more details, please refer to our paper:
    `An Improved One millisecond Mobile Backbone` -
    https://arxiv.org/pdf/2206.04040.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        inference_mode: bool = False,
        use_se: bool = False,
        use_act: bool = True,
        use_scale_branch: bool = True,
        num_conv_branches: int = 1,
        activation: str = "gelu",
        **kwargs,
    ) -> None:
        """Construct a MobileOneBlock module.

        Args:
            in_channels: Number of channels in the input.
            out_channels: Number of channels produced by the block.
            kernel_size: Size of the convolution kernel.
            stride: Stride size.
            padding: Zero-padding size.
            dilation: Kernel dilation factor.
            groups: Group number.
            inference_mode: If True, instantiates model in inference mode.
            use_se: Whether to use SE-ReLU activations.
            use_act: Whether to use activation. Default: ``True``
            use_scale_branch: Whether to use scale branch. Default: ``True``
            num_conv_branches: Number of linear conv branches.
        """
        super().__init__(**kwargs)
        self.inference_mode = inference_mode
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_conv_branches = num_conv_branches
        self.activation = activation
        self.padding = padding
        self.use_scale_branch = use_scale_branch
        self.use_se = use_se
        self.use_act = use_act

        #####################################################
        self.se_layer_name = "se_layer"
        self.activation_layer_name = "activation_layer"
        self.rbr_skip_bn_branch_name = "rbr_skip_bn"
        self.rbr_conv_branch_name = "rbr_conv_branch"
        self.rbr_scale_conv_branch_name = "rbr_scale_conv_branch"

        #####################################################

        # name_prefix = kwargs["name"] if kwargs.get("name", None) else "mob"

        # Check if SE-ReLU is requested
        self.se = SEBlock(in_channels=self.out_channels, name=self.se_layer_name) if self.use_se else keras_layer.Identity(name=self.se_layer_name)
        self.activation_layer = (
            keras_layer.Activation(self.activation, name=self.activation_layer_name) if self.use_act else keras_layer.Identity(name=self.activation_layer_name)
        )

        # Conv type to use.
        self.conv_type = _DEPTHWISE_CONV if self.groups == self.in_channels else _CONV
        if self.conv_type == _DEPTHWISE_CONV:
            # When out_channels is a multiple of in_channels, the conv kernel's shape is (kH,kW, in_channel, depth_multiplier)
            # As it's a calculation that will be used many times, we make it as part of the initialization to make it more verbose.
            self.depth_multiplier = self.out_channels // self.in_channels

        if self.inference_mode:
            self._set_reparam_layers()
        else:
            # Re-parameterizable skip connection
            # identity branch - only contains BN
            if self.out_channels == self.in_channels and self.stride == 1:
                self.rbr_skip = keras_layer.BatchNormalization(epsilon=1e-5, name=f"{self.name}_{self.rbr_skip_bn_branch_name}")
            else:
                self.rbr_skip = None

            # Re-parameterizable conv branches
            # nxn conv-bn branch - can be many
            if self.num_conv_branches > 0:
                for i in range(self.num_conv_branches):
                    setattr(
                        self,
                        f"{self.name}_{self.rbr_conv_branch_name}_{i}",
                        self._conv_bn(
                            kernel_size=self.kernel_size,
                            padding=self.padding,
                            name=f"{self.name}_{self.rbr_conv_branch_name}_{i}",
                        ),
                    )

            else:
                self.rbr_conv = None

            # Re-parameterizable scale branch
            # 1x1 conv branch
            self.rbr_scale = None
            if (kernel_size > 1) and self.use_scale_branch:
                self.rbr_scale = self._conv_bn(
                    kernel_size=1,
                    padding=0,
                    name=f"{self.name}_{self.rbr_scale_conv_branch_name}",
                )

    def _set_reparam_layers(self):
        self.zero_pad_infer = keras_layer.ZeroPadding2D(padding=(1, 1)) if self.padding == 1 else None
        if self.conv_type == _CONV:
            self.reparam_conv = keras_layer.Conv2D(
                filters=self.out_channels,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding="valid",
                dilation_rate=self.dilation,
                groups=self.groups,
                use_bias=True,
            )
        else:
            self.reparam_conv = keras_layer.DepthwiseConv2D(
                depth_multiplier=self.depth_multiplier,
                kernel_size=self.kernel_size,
                strides=self.stride,
                padding="valid",
                dilation_rate=self.dilation,
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

        if padding == 1:
            conv_bn_mod.add(keras_layer.ZeroPadding2D(padding=(1, 1)))

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
        # Inference mode forward pass.
        if self.inference_mode:
            if self.zero_pad_infer is None:
                return self.activation_layer(self.se(self.reparam_conv(x)))
            else:
                return self.activation_layer(self.se(self.reparam_conv(self.zero_pad_infer(x))))

        # Multi-branched train-time forward pass.
        # Skip branch output
        identity_out = 0.0
        if self.rbr_skip is not None:
            identity_out = self.rbr_skip(x)

        # Scale branch output
        scale_out = 0.0
        if self.rbr_scale is not None:
            scale_out = self.rbr_scale(x)

        # Other branches
        out = scale_out + identity_out
        if self.num_conv_branches > 0:
            for ix in range(self.num_conv_branches):
                out = out + getattr(self, f"{self.name}_{self.rbr_conv_branch_name}_{ix}")(x)

        out = self.activation_layer(self.se(out))
        # tf.print(self.name, kops.sum(out))
        return out

    def reparameterize(self):
        """Following works like `RepVGG: Making VGG-style ConvNets Great Again` -
        https://arxiv.org/pdf/2101.03697.pdf. We re-parameterize multi-branched
        architecture used at training time to obtain a plain CNN-like structure
        for inference.

        Here, to remove layers from the graph I'm using the 'name' provided to each layer during layer creation.
        """

        if self.inference_mode:
            return

        kernel, bias = self._get_kernel_bias()

        self.reparam_conv.build((None, None, None, self.in_channels))
        self.reparam_conv.set_weights([kernel, bias])

        if self.num_conv_branches > 0:
            for ix in range(self.num_conv_branches):
                self.__delattr__(f"{self.name}_{self.rbr_conv_branch_name}_{ix}")

        self.__delattr__("rbr_scale")
        if hasattr(self, "rbr_skip"):
            self.__delattr__("rbr_skip")

        # * Removing layers and their weights that were part of the original model.
        # * I can do this because I've provide specific names to those layers.
        layers_idx_to_remove = [idx for idx, layer in enumerate(self._layers) if self.name in layer.name]

        for i in reversed(layers_idx_to_remove):
            del self._layers[i]

        gc.collect()

        self.inference_mode = True
        # self.trainable = False

    def _get_kernel_bias(self):
        """Method to obtain re-parameterized kernel and bias.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L83

        Returns:
            Tuple of (kernel, bias) after fusing branches.
        """
        # Get weights and bias of scale branch
        kernel_scale = 0.0
        bias_scale = 0.0

        if self.rbr_scale is not None:
            kernel_scale, bias_scale = self._fuse_bn_tensor(self.rbr_scale)
            # Pad scale branch containing 1x1 Conv-BN to match rbr_conv branch's kernel size.
            # Pad scale branch kernel to match conv branch kernel size.
            pad = self.kernel_size // 2
            # To pad height and width only, we set padding on the first two dimensions:
            paddings = [
                (pad, pad),  # Pad kernel height
                (pad, pad),  # Pad kernel width
                (0, 0),  # No padding for channel dims
                (0, 0),
            ]
            kernel_scale = kops.pad(kernel_scale, paddings, mode="constant")

        # Get weights and bias of skip branch
        kernel_identity = 0.0
        bias_identity = 0.0
        if self.rbr_skip is not None:
            kernel_identity, bias_identity = self._fuse_bn_tensor(self.rbr_skip)

        # Get weights and bias of conv branches
        kernel_conv = 0.0
        bias_conv = 0.0
        if self.num_conv_branches > 0:
            for ix in range(self.num_conv_branches):
                _kernel, _bias = self._fuse_bn_tensor(getattr(self, f"{self.name}_{self.rbr_conv_branch_name}_{ix}"))
                kernel_conv += _kernel
                bias_conv += _bias

        kernel_final = kernel_conv + kernel_scale + kernel_identity
        bias_final = bias_conv + bias_scale + bias_identity
        return kernel_final, bias_final

    def _fuse_bn_tensor(self, branch: Union[keras.Sequential, keras_layer.BatchNormalization]) -> Tuple[keras.KerasTensor, keras.KerasTensor]:
        """Method to fuse batchnorm layer with preceeding conv layer.
        Reference: https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py#L95

        Args:
            branch: Sequence of ops to be fused.

        Returns:
            Tuple of (kernel, bias) after fusing batchnorm.
        """
        # Goal is to: First make batch-norm params the same shape as the conv-kernel either before merging the weights.

        if isinstance(branch, keras.Sequential):
            _branch_len = len(branch.layers)
            conv_params = branch.get_layer(index=_branch_len - 2).get_weights()  # [kernel, bias]
            bn_params = branch.get_layer(index=_branch_len - 1).get_weights()  # [gamma, beta, moving_mean, moving_variance]

            kernel = conv_params[0]
            running_mean = bn_params[2]
            running_var = bn_params[3]
            gamma = bn_params[0]
            beta = bn_params[1]
            eps = branch.layers[-1].epsilon

        else:
            assert isinstance(branch, keras_layer.BatchNormalization)

            # Resize batch-normalization weights for merging

            bn_params = branch.get_weights()  # [gamma, beta, moving_mean, moving_variance]
            _branch_weights_dtype = bn_params[0].dtype

            if not hasattr(self, "id_tensor"):
                """
                Conv
                TF - kH, kW, inC, outC
                PT - outC, inC, kH, kW

                Depthwise
                TF - kH, kW, outC, depth_multiplier (outC/inC) 
                PT - outC, inC, kH, kW

                We are primarily using channel last format.
                """
                input_dim = self.in_channels // self.groups
                _kernel_size = self.kernel_size // 2

                if self.conv_type == _CONV:
                    _kernel_shape = (
                        self.kernel_size,
                        self.kernel_size,
                        input_dim,
                        self.in_channels,
                    )
                    kernel_value = kops.zeros(_kernel_shape, dtype=_branch_weights_dtype)

                elif self.conv_type == _DEPTHWISE_CONV:
                    _kernel_shape = (
                        self.kernel_size,
                        self.kernel_size,
                        self.in_channels,
                        input_dim,
                    )
                    kernel_value = kops.zeros(_kernel_shape, dtype=_branch_weights_dtype)

                # Prepare indices and values for scatter operation
                indices = []
                updates = []

                for i in range(self.in_channels):
                    if self.conv_type == _CONV:
                        indices.append([_kernel_size, _kernel_size, i % input_dim, i])  # Center position
                        updates.append(1.0)
                    elif self.conv_type == _DEPTHWISE_CONV:
                        indices.append([_kernel_size, _kernel_size, i, i % input_dim])  # Center position
                        updates.append(1.0)

                kernel_value = kops.scatter_update(kernel_value, indices, updates)
                self.id_tensor = kernel_value

            kernel = self.id_tensor
            running_mean = bn_params[2]
            running_var = bn_params[3]
            gamma = bn_params[0]
            beta = bn_params[1]
            eps = branch.epsilon

        std = kops.sqrt(running_var + eps)

        if self.conv_type == _CONV:
            _kernel_multiplier_reshape_dims = (1, 1, 1, -1)
        elif self.conv_type == _DEPTHWISE_CONV:
            _kernel_multiplier_reshape_dims = (1, 1, -1, self.depth_multiplier)

        t = kops.divide(gamma, std)
        t = kops.reshape(t, _kernel_multiplier_reshape_dims)
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
                "padding": self.padding,
                "dilation": self.dilation,
                "groups": self.groups,
                "inference_mode": self.inference_mode,
                "use_se": self.use_se,
                "use_act": self.use_act,
                "use_scale_branch": self.use_scale_branch,
                "num_conv_branches": self.num_conv_branches,
                "activation": self.activation,
            }
        )

        return config


def reparameterize_model(
    model: keras.Model | keras.Sequential,
) -> keras.Model | keras.Sequential:
    """Method returns a model where a multi-branched structure
        used in training is re-parameterized into a single branch
        for inference.

    Args:
        model: MobileOne model in train mode.

    Returns:
        MobileOne model in inference mode.
    """
    # # Avoid editing original graph
    # model = copy.deepcopy(model)

    for module in model.layers:
        if hasattr(module, "reparameterize"):
            module.reparameterize()
    return model


if __name__ == "__main__":
    in_channels = 384
    # se_block = SEBlock(in_channels=in_channels)

    import numpy as np

    inp = np.random.randn(1, 16, 16, in_channels)

    # out = se_block(inp)
    # print(out.shape)

    layer = MobileOneBlock(
        in_channels=in_channels,
        out_channels=int(in_channels * 2.0),
        kernel_size=3,
        stride=1,
        padding=1,
        groups=in_channels,
        inference_mode=False,
        use_se=True,
        num_conv_branches=1,
        name="mob_1",
    )

    layer(inp)

    model = keras.Sequential()
    model.add(keras.Input(shape=(None, None, in_channels)))
    model.add(layer)

    # model.add(
    #     MobileOneBlock(
    #         in_channels=in_channels,
    #         out_channels=in_channels,
    #         kernel_size=3,
    #         stride=1,
    #         padding=1,
    #         groups=1,
    #         use_se=True,
    #         num_conv_branches=2,
    #         name="mob_2",
    #     )
    # )

    # model.summary(expand_nested=True, show_trainable=True)
    # for i in model.variables:
    #     # print(dir(i))
    #     print(i.path)
    #     # break

    # # for i in model._get_variable_map():
    # print(model._get_variable_map().keys())
    # for i in model._get_variable_map().keys():
    #     print(i)

    # model.save_weights("test_variables.weights.h5")

    # # ============================================================
    # print("---------------------")

    # layer = MobileOneBlock(
    #     in_channels=in_channels,
    #     out_channels=int(in_channels * 2.0),
    #     kernel_size=3,
    #     stride=1,
    #     padding=1,
    #     groups=1,
    #     inference_mode=False,
    #     use_se=True,
    #     num_conv_branches=1,
    #     name="mob_1",
    # )

    # layer(inp)

    # model = keras.Sequential()
    # model.add(keras.Input(shape=(None, None, in_channels)))
    # model.add(layer)

    # for i in model._get_variable_map().keys():
    #     print(i)
    # model.load_weights("test_variables.weights.h5")
    # # print(model.layers[1]._layers)
    # # print(model.layers[0]._layers[-1].weights)

    # # print(model.layers[1]._layers)
    # # print(model.layers[1]._layers[-1].weights)

    model = reparameterize_model(model)
    model.summary(expand_nested=True, show_trainable=True)

    # # print(model.layers[0]._layers)
    # # # # print(model.layers[1]._layers)
