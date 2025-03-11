# import os

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "torch"

from keras import layers as keras_layer

__all__ = ["SEBlock"]


class SEBlock(keras_layer.Layer):
    """Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625, **kwargs) -> None:
        """Construct a Squeeze and Excite Module.

        Args:
            in_channels: Number of input channels.
            rd_ratio: Input channel reduction ratio.
        """
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.rd_ratio = rd_ratio

        self.reduce = keras_layer.Conv2D(
            filters=int(self.in_channels * self.rd_ratio),
            kernel_size=1,
            strides=(1, 1),
            padding="valid",
            use_bias=True,
        )

        self.expand = keras_layer.Conv2D(
            filters=self.in_channels,
            kernel_size=1,
            strides=(1, 1),
            padding="valid",
            use_bias=True,
        )
        self.global_average_pool = keras_layer.GlobalAveragePooling2D(keepdims=True)  # to maintain (h, w) dims
        self.relu = keras_layer.Activation("relu")
        self.sigmoid = keras_layer.Activation("sigmoid")

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs):
        """Apply forward pass."""

        x = self.global_average_pool(inputs)
        # b, h, w, c = kops.shape(inputs)
        # x = kops.average_pool()
        x = self.reduce(x)
        x = self.relu(x)
        x = self.expand(x)
        x = self.sigmoid(x)
        return inputs * x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "in_channels": self.in_channels,
                "rd_ratio": self.rd_ratio,
            }
        )

        return config


if __name__ == "__main__":
    in_channels = 16
    se_block = SEBlock(in_channels=in_channels)

    import numpy as np

    inp = np.random.randn(1, 16, 16, in_channels)

    out = se_block(inp)
    print(out.shape)
