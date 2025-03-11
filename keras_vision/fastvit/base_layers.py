# import os

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "torch"

import gc

# import tensorflow as tf
import keras
from keras import ops as kops
from keras import layers as keras_layer
from typing import List

from .mobileone import MobileOneBlock
from .replknet import ReparamLargeKernelConv
from .multihead_self_attention import MHSA
from .conv_ffn import ConvFFN


def convolutional_stem(in_channels: int, out_channels: int, inference_mode: bool = False, name="convolutional_stem") -> keras.Sequential:
    """Build convolutional stem with MobileOne blocks.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        inference_mode: Flag to instantiate model in inference mode. Default: ``False``

    Returns:
        keras.Sequential object with stem elements.
    """
    conv_stem = keras.Sequential(
        layers=(
            MobileOneBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
                name=f"{name}_mobileone_1",
            ),
            MobileOneBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                groups=out_channels,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
                name=f"{name}_mobileone_2",
            ),
            MobileOneBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=inference_mode,
                use_se=False,
                num_conv_branches=1,
                name=f"{name}_mobileone_3",
            ),
        ),
        name=name,
    )
    # conv_stem.build((None, None, None, in_channels))

    return conv_stem


class PatchEmbed(keras_layer.Layer):
    """Convolutional patch embedding layer."""

    def __init__(
        self,
        patch_size: int,
        stride: int,
        in_channels: int,
        embed_dim: int,
        inference_mode: bool = False,
        **kwargs,
    ) -> None:
        """Build patch embedding layer.

        Args:
            patch_size: Patch size for embedding computation.
            stride: Stride for convolutional embedding layer.
            in_channels: Number of channels of input tensor.
            embed_dim: Number of embedding dimensions.
            inference_mode: Flag to instantiate model in inference mode. Default: ``False``
        """
        super().__init__(**kwargs)

        self.patch_size = patch_size
        self.stride = stride
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.inference_mode = inference_mode

        layers = list()
        layers.append(
            ReparamLargeKernelConv(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.stride,
                groups=self.in_channels,
                small_kernel=3,
                inference_mode=self.inference_mode,
                name=f"{self.name}_rlkc",
            )
        )
        layers.append(
            MobileOneBlock(
                in_channels=self.embed_dim,
                out_channels=self.embed_dim,
                kernel_size=1,
                stride=1,
                padding=0,
                groups=1,
                inference_mode=self.inference_mode,
                use_se=False,
                num_conv_branches=1,
                name=f"{self.name}_mobileone",
            )
        )
        self.proj = keras.Sequential(layers=layers, name=f"{self.name}_proj")

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        x = self.proj(x)
        return x

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config["patch_size"] = self.patch_size
        config["stride"] = self.stride
        config["in_channels"] = self.in_channels
        config["embed_dim"] = self.embed_dim
        config["inference_mode"] = self.inference_mode
        return config


class LayerScale(keras_layer.Layer):
    def __init__(
        self,
        layer_scale_init_value=1e-6,
        shape=(1, 1, 1),
        _requires_grad=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layer_scale_init_value = layer_scale_init_value
        self.shape = shape
        self._requires_grad = _requires_grad

    def build(self, input_shape):
        # Create the trainable weight variable in build()
        self.layer_scale = self.add_weight(
            shape=self.shape,
            initializer=keras.initializers.Constant(self.layer_scale_init_value),
            trainable=self._requires_grad,
            name="layer_scale",
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        return inputs * self.layer_scale

    def get_config(self):
        config = super().get_config()
        config["layer_scale_init_value"] = self.layer_scale_init_value
        config["shape"] = self.shape
        config["_requires_grad"] = self._requires_grad
        return config


class RepMixer(keras_layer.Layer):
    """Reparameterizable token mixer.

    For more details, please refer to our paper:
    `FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization <https://arxiv.org/pdf/2303.14189.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        **kwargs,
    ):
        """Build RepMixer Module.

        Args:
            dim: Input feature map dimension. :math:`C_{in}` from an expected input of size :math:`(B, C_{in}, H, W)`.
            kernel_size: Kernel size for spatial mixing. Default: 3
            use_layer_scale: If True, learnable layer scale is used. Default: ``True``
            layer_scale_init_value: Initial value for layer scale. Default: 1e-5
            inference_mode: If True, instantiates model in inference mode. Default: ``False``
        """
        super().__init__(**kwargs)
        self.dim = dim
        self.kernel_size = kernel_size
        self.inference_mode = inference_mode
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value

        if self.inference_mode:
            self._set_reparam_layers()
        else:
            if use_layer_scale:
                self.layer_scale_layer = LayerScale(
                    layer_scale_init_value=self.layer_scale_init_value,
                    shape=(1, 1, self.dim),
                    _requires_grad=True,
                    name=f"{self.name}_layer_scale",
                )

            self.norm = MobileOneBlock(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                groups=self.dim,
                use_act=False,
                use_scale_branch=False,
                num_conv_branches=0,
                name=f"{self.name}_mobileone_1",
            )
            self.mixer = MobileOneBlock(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                groups=self.dim,
                use_act=False,
                name=f"{self.name}_mobileone_2",
            )

    def build(self, input_shape):
        self._set_reparam_layers()
        super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        if self.inference_mode:
            x = self.reparam_conv_zero_pad(x)
            x = self.reparam_conv(x)
            return x
        else:
            if self.use_layer_scale:
                x = x + self.layer_scale_layer(self.mixer(x) - self.norm(x))
            else:
                x = x + self.mixer(x) - self.norm(x)
            # tf.print(self.name, kops.sum(x))
            return x

    def _set_reparam_layers(self):
        self.reparam_conv_zero_pad = keras_layer.ZeroPadding2D(padding=self.kernel_size // 2, name="reparam_conv_pad")

        self.reparam_conv = keras_layer.DepthwiseConv2D(
            kernel_size=self.kernel_size,
            strides=1,
            padding="valid",
            use_bias=True,
            # name="reparam_conv",
        )

    def reparameterize(self) -> None:
        """Reparameterize mixer and norm into a single
        convolutional layer for efficient inference.
        """
        # self.trainable = False
        if self.inference_mode:
            return

        self.mixer.reparameterize()
        self.norm.reparameterize()

        mixer_reparam_weights = self.mixer.reparam_conv.get_weights()
        norm_reparam_weights = self.mixer.reparam_conv.get_weights()

        if self.use_layer_scale:
            layer_scale_weights = self.layer_scale_layer.get_weights()[0]

            # self.layer_scale.trainable = False
            kernel = self.mixer.id_tensor + kops.expand_dims(layer_scale_weights, axis=-1) * (mixer_reparam_weights[0] - norm_reparam_weights[0])
            bias = keras.ops.squeeze(layer_scale_weights) * (mixer_reparam_weights[1] - norm_reparam_weights[1])
        else:
            kernel = self.mixer.id_tensor + mixer_reparam_weights[0] - norm_reparam_weights[0]
            bias = mixer_reparam_weights[1] - norm_reparam_weights[1]

        self.reparam_conv.build((None, None, None, self.dim))
        self.reparam_conv.set_weights([kernel, bias])

        print(self._layers)
        self.__delattr__("mixer")
        self.__delattr__("norm")

        if self.use_layer_scale:
            self.__delattr__("layer_scale_layer")

        # layers_idx_to_remove = [idx for idx, layer in enumerate(self._layers) if "reparam_conv" not in layer.name]
        layers_idx_to_remove = [idx for idx, layer in enumerate(self._layers) if self.name in layer.name]

        for i in reversed(layers_idx_to_remove):
            del self._layers[i]
        print(self._layers)

        gc.collect()
        self.inference_mode = True
        # self.trainable = False

    def get_config(self):
        config = super().get_config()
        config["dim"] = self.dim
        config["layer_scale_init_value"] = self.layer_scale_init_value
        config["kernel_size"] = self.kernel_size
        config["inference_mode"] = self.inference_mode
        config["use_layer_scale"] = self.use_layer_scale

        return config


class DropPath(keras_layer.Layer):
    def __init__(self, drop_prob=0.0, scale_by_keep=True, **kwargs):
        """
        Args:
            drop_prob (float): probability of dropping the path.
            scale_by_keep (bool): whether to scale the output by 1/(1 - drop_prob) when dropping.
        """
        super().__init__(**kwargs)
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def call(self, x: keras.KerasTensor, **kwargs) -> keras.KerasTensor:
        # If not training or drop_prob is 0, simply return the input.
        if not kwargs.get("training", False) or self.drop_prob == 0.0:
            return x

        keep_prob = 1.0 - self.drop_prob
        # Create a binary mask with shape (batch, 1, 1, ..., 1) matching x's rank.
        # This ensures each sample in the batch is dropped (or not) independently.
        shape = (kops.shape(x)[0],) + (1,) * (kops.ndim(x) - 1)

        # Generate random numbers and threshold to obtain a mask (True where kept).
        random_tensor = keras.random.uniform(shape, dtype=x.dtype) < keep_prob
        random_tensor = kops.cast(random_tensor, kops.dtype(x))

        # Optionally scale the kept outputs by 1/keep_prob.
        if self.scale_by_keep:
            random_tensor /= keep_prob

        return x * random_tensor

    def get_config(self):
        config = super().get_config()
        config["drop_prob"] = self.drop_prob
        config["scale_by_keep"] = self.scale_by_keep
        return config


class RepMixerBlock(keras_layer.Layer):
    """Implementation of Metaformer block with RepMixer as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        mlp_ratio: float = 4.0,
        act_layer: str = "gelu",
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        inference_mode: bool = False,
        **kwargs,
    ):
        """Build RepMixer Block.

        Args:
            dim: Number of embedding dimensions.
            kernel_size: Kernel size for repmixer. Default: 3
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``"gelu"``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """

        super().__init__(**kwargs)
        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(mlp_ratio)

        self.dim = dim
        self.kernel_size = kernel_size
        self.mlp_ratio = mlp_ratio
        self.act_layer = act_layer
        self.drop = drop
        self.drop_path = drop_path
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value
        self.inference_mode = inference_mode

        # Layer Scale
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_layer = LayerScale(
                layer_scale_init_value=self.layer_scale_init_value,
                shape=(1, 1, self.dim),
                _requires_grad=True,
                name=f"{self.name}_layer_scale",
            )

        self.token_mixer = RepMixer(
            dim=self.dim,
            kernel_size=self.kernel_size,
            use_layer_scale=self.use_layer_scale,
            layer_scale_init_value=self.layer_scale_init_value,
            inference_mode=self.inference_mode,
            name=f"{self.name}_repmixer",
        )

        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=self.dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=self.act_layer,
            drop=self.drop,
            name=f"{self.name}_convffn",
        )

        # Drop Path
        self.drop_path_layer = DropPath(self.drop_path) if drop_path > 0.0 else keras_layer.Identity()

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        if self.use_layer_scale:
            x = self.token_mixer(x)
            x = x + self.drop_path_layer(self.layer_scale_layer(self.convffn(x)))
        else:
            x = self.token_mixer(x)
            x = x + self.drop_path_layer(self.convffn(x))

        # tf.print(self.name, kops.sum(x))
        return x

    def get_config(self):
        config = super().get_config()

        config["dim"] = self.dim
        config["kernel_size"] = self.kernel_size
        config["mlp_ratio"] = self.mlp_ratio
        config["act_layer"] = self.act_layer
        config["drop"] = self.drop
        config["drop_path"] = self.drop_path
        config["use_layer_scale"] = self.use_layer_scale
        config["layer_scale_init_value"] = self.layer_scale_init_value
        config["inference_mode"] = self.inference_mode
        return config


class AttentionBlock(keras_layer.Layer):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        act_layer: str = "gelu",
        norm_layer: str = "BatchNormalization",
        drop: float = 0.0,
        drop_path: float = 0.0,
        use_layer_scale: bool = True,
        layer_scale_init_value: float = 1e-5,
        **kwargs,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``gelu``
            norm_layer: Normalization layer. Default: ``BatchNormalization``
            drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            use_layer_scale: Flag to turn on layer scale. Default: ``True``
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__(**kwargs)

        assert mlp_ratio > 0, "MLP ratio should be greater than 0, found: {}".format(self.mlp_ratio)
        self.dim = dim
        self.mlp_ratio = mlp_ratio
        self.act_layer = act_layer
        self.norm_layer = norm_layer
        self.drop = drop
        self.drop_path_prob = drop_path
        self.use_layer_scale = use_layer_scale
        self.layer_scale_init_value = layer_scale_init_value

        # Layer Scale
        if use_layer_scale:
            self.layer_scale_1 = LayerScale(
                layer_scale_init_value=self.layer_scale_init_value,
                shape=(1, 1, self.dim),
                _requires_grad=True,
                name=f"{self.name}_layer_scale_1",
            )
            self.layer_scale_2 = LayerScale(
                layer_scale_init_value=self.layer_scale_init_value,
                shape=(1, 1, self.dim),
                _requires_grad=True,
                name=f"{self.name}_layer_scale_2",
            )

        if self.norm_layer == "BatchNormalization":
            self.norm = keras_layer.BatchNormalization()

        self.token_mixer = MHSA(dim=self.dim, name=f"{self.name}_MHSA")

        mlp_hidden_dim = int(self.dim * self.mlp_ratio)
        self.convffn = ConvFFN(
            in_channels=self.dim,
            hidden_channels=mlp_hidden_dim,
            act_layer=self.act_layer,
            drop=self.drop,
            name=f"{self.name}_convffn",
        )

        # Drop path
        self.drop_path = DropPath(drop_prob=self.drop_path_prob) if self.drop_path_prob > 0.0 else keras_layer.Identity()

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1(self.token_mixer(self.norm(x))))
            x = x + self.drop_path(self.layer_scale_2(self.convffn(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm(x)))
            x = x + self.drop_path(self.convffn(x))
        return x

    def get_config(self):
        config = super().get_config()

        config["dim"] = self.dim
        config["mlp_ratio"] = self.mlp_ratio
        config["act_layer"] = self.act_layer
        config["norm_layer"] = self.norm_layer
        config["drop"] = self.drop
        config["drop_path"] = self.drop_path_prob
        config["use_layer_scale"] = self.use_layer_scale
        config["layer_scale_init_value"] = self.layer_scale_init_value
        return config


def basic_blocks(
    dim: int,
    block_index: int,
    num_blocks: List[int],
    token_mixer_type: str,
    kernel_size: int = 3,
    mlp_ratio: float = 4.0,
    act_layer: str = "gelu",
    norm_layer: str = "BatchNormalization",
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    use_layer_scale: bool = True,
    layer_scale_init_value: float = 1e-5,
    inference_mode=False,
    name="basic_block",
) -> keras.Sequential:
    """Build FastViT blocks within a stage.

    Args:
        dim: Number of embedding dimensions.
        block_index: block index.
        num_blocks: List containing number of blocks per stage.
        token_mixer_type: Token mixer type.
        kernel_size: Kernel size for repmixer.
        mlp_ratio: MLP expansion ratio.
        act_layer: Activation layer.
        norm_layer: Normalization layer.
        drop_rate: Dropout rate.
        drop_path_rate: Drop path rate.
        use_layer_scale: Flag to turn on layer scale regularization.
        layer_scale_init_value: Layer scale value at initialization.
        inference_mode: Flag to instantiate block in inference mode.

    Returns:
        keras.Sequential object of all the blocks within the stage.
    """
    blocks = []
    for block_idx in range(num_blocks[block_index]):
        block_dpr = drop_path_rate * (block_idx + sum(num_blocks[:block_index])) / (sum(num_blocks) - 1)
        if token_mixer_type == "repmixer":
            blocks.append(
                RepMixerBlock(
                    dim=dim,
                    kernel_size=kernel_size,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    inference_mode=inference_mode,
                    name=f"{name}_idx_{block_idx}_repmixer",
                )
            )
        elif token_mixer_type == "attention":
            blocks.append(
                AttentionBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=block_dpr,
                    use_layer_scale=use_layer_scale,
                    layer_scale_init_value=layer_scale_init_value,
                    name=f"{name}_idx_{block_idx}_attention",
                )
            )
        else:
            raise ValueError(f"Token mixer type: {token_mixer_type} not supported")
    blocks = keras.Sequential(layers=blocks, name=name)

    return blocks


if __name__ == "__main__":
    import numpy as np

    in_channels = 32

    inp = np.random.randn(1, 32, 32, in_channels)

    # layer = RepMixer(dim=in_channels)
    layer = convolutional_stem(in_channels=32, out_channels=32)

    model = keras.Sequential()
    model.add(keras.Input(shape=(None, None, in_channels)))
    model.add(layer)

    model.summary()
    print(len(model.variables))
