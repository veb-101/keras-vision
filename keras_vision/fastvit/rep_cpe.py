# import os

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "torch"

import gc
from typing import Union, Tuple
import keras
from keras import ops as kops
from keras import layers as keras_layer


class RepCPE(keras_layer.Layer):
    """Implementation of conditional positional encoding.

    For more details refer to paper:
    `Conditional Positional Encodings for Vision Transformers <https://arxiv.org/pdf/2102.10882.pdf>`_

    In our implementation, we can reparameterize this module to eliminate a skip connection.
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int = 768,
        spatial_shape: Union[int, Tuple[int, int]] = (7, 7),
        inference_mode=False,
        **kwargs,
    ) -> None:
        """Build reparameterizable conditional positional encoding

        Args:
            in_channels: Number of input channels.
            embed_dim: Number of embedding dimensions. Default: 768
            spatial_shape: Spatial shape of kernel for positional encoding. Default: (7, 7)
            inference_mode: Flag to instantiate block in inference mode. Default: ``False``
        """
        super().__init__(**kwargs)
        if isinstance(spatial_shape, int):
            spatial_shape = tuple([spatial_shape] * 2)
        assert isinstance(spatial_shape, Tuple), f'"spatial_shape" must by a sequence or int, get {type(spatial_shape)} instead.'
        assert len(spatial_shape) == 2, f'Length of "spatial_shape" should be 2, got {len(spatial_shape)} instead.'

        self.spatial_shape = spatial_shape
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.inference_mode = inference_mode
        self.groups = embed_dim

        if self.inference_mode:
            self._set_reparam_layers()
        else:
            self.padding = keras_layer.ZeroPadding2D(padding=int(self.spatial_shape[0] // 2))
            self.pe = keras_layer.Conv2D(
                filters=self.embed_dim,
                kernel_size=self.spatial_shape,
                strides=1,
                padding="valid",
                use_bias=True,
                groups=self.embed_dim,
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
            x = self.pe(self.padding(x)) + x
            return x

    def _set_reparam_layers(self):
        self.reparam_conv_zero_pad = keras_layer.ZeroPadding2D(padding=int(self.spatial_shape[0] // 2), name="reparam_conv_pad")

        self.reparam_conv = keras_layer.Conv2D(
            filters=self.embed_dim,
            kernel_size=self.spatial_shape,
            strides=1,
            padding="valid",
            use_bias=True,
            groups=self.embed_dim,
            name="reparam_conv",
        )

    def reparameterize(self) -> None:
        if self.inference_mode:
            return

        # Build equivalent Id tensor
        self.reparam_conv.build((None, None, None, self.in_channels))

        conv_params = self.reparam_conv.get_weights()
        weights_dtype = conv_params[0].dtype

        input_dim = self.in_channels // self.groups

        kernel_value = kops.zeros(
            (self.spatial_shape[0], self.spatial_shape[1], input_dim, self.in_channels),
            dtype=weights_dtype,
        )

        # Prepare indices and values for scatter operation
        indices = []
        updates = []

        for i in range(self.in_channels):
            indices.append([self.spatial_shape[0] // 2, self.spatial_shape[1] // 2, i % input_dim, i])  # Center position
            updates.append(1.0)

        kernel_value = kops.scatter_update(kernel_value, indices, updates)
        id_tensor = kernel_value

        # Reparameterize Id tensor and conv

        w_final = kops.add(id_tensor, conv_params[0])
        b_final = conv_params[1]
        self.reparam_conv.set_weights([w_final, b_final])

        for idx, layer in enumerate(self._layers):
            print(layer.name)
        layers_idx_to_remove = [idx for idx, layer in enumerate(self._layers) if "reparam_conv" not in layer.name]
        for i in reversed(layers_idx_to_remove):
            del self._layers[i]

        self.__delattr__("pe")

        gc.collect()
        self.inference_mode = True
        # self.trainable = False

    def get_config(self):
        config = super().get_config()

        config["in_channels"] = self.in_channels
        config["embed_dim"] = self.embed_dim
        config["spatial_shape"] = self.spatial_shape
        config["inference_mode"] = self.inference_mode
        return config


if __name__ == "__main__":
    import numpy as np

    in_channels = embed_dim = 768

    inp = np.random.randn(1, 32, 32, in_channels)

    layer = RepCPE(in_channels=in_channels)
    # layer(inp)  # Required for

    model = keras.Sequential()
    model.add(keras.Input(shape=(None, None, in_channels)))
    model.add(layer)

    model.summary()

    # * Need to initialize parameters before reparameterization.
    layer.reparameterize()
    model = keras.Sequential()
    model.add(keras.Input(shape=(None, None, in_channels)))
    model.add(layer)
    model.summary()
