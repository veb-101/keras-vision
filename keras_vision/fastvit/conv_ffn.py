# import os

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers as keras_layer
from typing import Optional

# import tensorflow as tf
# from keras import ops as kops


class ConvFFN(keras.Layer):
    """Convolutional FFN Module."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        act_layer: str = "gelu",
        drop: float = 0.0,
        **kwargs,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_channels: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_channels: Number of output channels. Default: None
            act_layer: str. Default: ``gelu``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.hidden_channels = hidden_channels or in_channels
        self.act_layer = act_layer
        self.dropout_val = drop

        self.depth_multiplier = self.out_channels // self.in_channels

        self.conv_zero_pad = keras_layer.ZeroPadding2D(padding=3)
        self.conv_depthwise = keras_layer.DepthwiseConv2D(depth_multiplier=self.depth_multiplier, kernel_size=7, use_bias=False)
        self.conv_bn = keras_layer.BatchNormalization(epsilon=1e-5)

        kernel_initializer = keras.initializers.TruncatedNormal(stddev=0.02)
        bias_initializer = keras.initializers.Zeros()

        self.fc1 = keras_layer.Conv2D(
            filters=self.hidden_channels,
            kernel_size=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.act = keras_layer.Activation(self.act_layer)
        self.fc2 = keras_layer.Conv2D(
            filters=self.out_channels,
            kernel_size=1,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )
        self.drop = keras_layer.Dropout(self.dropout_val)

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        x = self.conv_zero_pad(x)
        x = self.conv_depthwise(x)
        x = self.conv_bn(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.drop(x)
        x = self.fc2(x)
        # tf.print(self.name, kops.sum(x))
        # tf.print("--")
        return x

    def build(self, input_shape):
        super().build(input_shape)

    def get_config(self):
        config = super().get_config()
        config["in_channels"] = self.in_channels
        config["out_channels"] = self.out_channels
        config["hidden_channels"] = self.hidden_channels
        config["act_layer"] = self.act_layer
        config["dropout_val"] = self.dropout_val
        return config


if __name__ == "__main__":
    in_channels = 64
    hidden_channels = 64
    out_channels = in_channels
    conv_stem = ConvFFN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
    )

    model = keras.Sequential()
    model.add(keras.Input(shape=(None, None, in_channels)))
    model.add(conv_stem)

    model.summary()
