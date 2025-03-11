# import os

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import ops as kops
from keras import layers as keras_layer


class MHSA(keras.Layer):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """

    def __init__(
        self,
        dim: int,
        head_dim: int = 32,
        qkv_bias: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        **kwargs,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__(**kwargs)
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.dim = dim  # 512
        self.head_dim = head_dim  # 32
        self.num_heads = self.dim // head_dim  # 512 // 32 = 16
        self.scale = head_dim**-0.5
        self.projection_dim = dim // self.num_heads  # 512 // 16 = 32
        self.qkv_bias = qkv_bias
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop

        self.qkv = keras_layer.Dense(units=self.dim * 3, use_bias=self.qkv_bias)
        self.attn_drop_layer = keras_layer.Dropout(attn_drop)
        self.proj = keras_layer.Dense(units=self.dim)
        self.proj_drop_layer = keras_layer.Dropout(proj_drop)

    def build(self, input_shape):
        # You can perform setup tasks that depend on the input shape here
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, projection_dim)
        x = kops.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        # Transpose to shape (batch_size, num_heads, seq_len, projection_dim)
        return kops.transpose(x, axes=(0, 2, 1, 3))

    def call(self, x: keras.KerasTensor) -> keras.KerasTensor:
        shape = kops.shape(x)
        B, H, W, C = shape

        N = H * W

        if len(shape) == 4:
            x = kops.reshape(x, (B, -1, C))  # (B, N, C)

        qkv = self.qkv(x)
        qkv = kops.reshape(qkv, (B, N, 3, self.num_heads, self.head_dim))

        qkv = kops.transpose(qkv, axes=(0, 2, 1, 3, 4))
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = self.split_heads(q, B)
        k = self.split_heads(k, B)
        v = self.split_heads(v, B)

        q *= self.scale

        # Attention mechanism
        attn_logits = kops.matmul(q, kops.transpose(k, axes=(0, 1, 3, 2)))
        attn = kops.softmax(attn_logits)
        attn = self.attn_drop_layer(attn)

        weighted_avg = kops.matmul(attn, v)
        weighted_avg = kops.transpose(weighted_avg, axes=(0, 2, 1, 3))
        weighted_avg = kops.reshape(weighted_avg, (B, N, C))

        # Output projection
        output = self.proj(weighted_avg)
        output = self.proj_drop_layer(output)

        if len(shape) == 4:
            output = kops.reshape(output, (B, H, W, C))

        # print(output.shape)
        return output

    def get_config(self):
        config = super().get_config()
        config["dim"] = self.dim
        config["head_dim"] = self.head_dim
        config["qkv_bias"] = self.qkv_bias
        config["attn_drop"] = self.attn_drop
        config["proj_drop"] = self.proj_drop
        return config


if __name__ == "__main__":
    import numpy as np

    in_channels = 512

    inp = np.random.randn(1, 32, 32, in_channels)

    layer = MHSA(dim=in_channels)

    model = keras.Sequential()
    model.add(keras.Input(shape=(None, None, in_channels)))
    model.add(layer)

    model.summary()
