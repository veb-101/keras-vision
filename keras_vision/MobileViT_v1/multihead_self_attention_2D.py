import keras
import keras.ops as kops
from keras.layers import Layer, Dropout
from keras.layers import EinsumDense


class MultiHeadSelfAttentionEinSum2D(Layer):
    def __init__(
        self,
        num_heads: int = 2,
        embedding_dim: int = 64,
        projection_dim: int = None,
        qkv_bias: bool = True,
        attention_drop: float = 0.2,
        linear_drop: float = 0.2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim if projection_dim else self.embedding_dim // self.num_heads
        self.scale = self.projection_dim**0.5
        self.qkv_bias = qkv_bias
        self.attention_drop = attention_drop
        self.linear_drop = linear_drop

        self.use_attention_drop = self.attention_drop > 0.0
        self.use_linear_drop = self.linear_drop > 0.0

        if self.use_attention_drop:
            self.attn_dropout = Dropout(self.attention_drop)

        if self.use_linear_drop:
            self.linear_dropout = Dropout(self.linear_drop)

        ##### Notations (wrt paper) #####
        # B/b = batch
        # P/p = patch_size
        # N/n = number of patches
        # D/d = embedding_dim
        # H/h = num_heads
        # E/e = projection_dim

        # New Shape: (B, P, N, D) * (D, H, E * 3) --> (B, P, H, N, E * 3)
        self.W_QKV = EinsumDense(
            "bpnd,dhe->bphne",
            output_shape=[None, self.num_heads, None, 3 * self.projection_dim],
            bias_axes="he" if self.qkv_bias else None,
        )

        # Shape: (B, P, H, N, E) * (E, H, D) --> (B, P, N, D)
        self.Wo = EinsumDense("bphne,ehd->bpnd", output_shape=[None, None, embedding_dim], bias_axes="d" if self.qkv_bias else None)

    def call(self, inputs):
        """
        In the attention matrix, dot product among patches only occur among pixels at similar positions.
        So the first pixel in each patch, only attends to the pixel at similar position in other patches.
        For visual reference, check the diagram on page 5 and 17 in the paper.
        """

        # Inputs Shape --> (B, P, N, D)

        output_tensor = self.W_QKV(inputs)  # Shape: (B, P, H, N, E * 3)

        # Shape: (B, P, H, N, E) * 3
        q, k, v = kops.split(output_tensor, indices_or_sections=3, axis=-1)

        # Shape: (B, P, H, N, E) * (B, P, H, N, E) --> (B, P, H, N, N)
        attention_matrix = kops.einsum("...ij, ...kj-> ...ik", q, k)

        # Shape: (B, P, H, N, N)
        attention_matrix = kops.nn.softmax(kops.divide(attention_matrix, self.scale), axis=-1)

        if self.use_attention_drop:
            attention_matrix = self.attn_dropout(attention_matrix)

        # Shape: (B, P, H, N, N) * (B, P, H, N, E) --> (B, P, H, N, E)
        weighted_values = kops.einsum("...ij,...jk->...ik", attention_matrix, v)

        final = self.Wo(weighted_values)  # Shape: (B, P, H, N, E) * (E, H, D) --> (B, P, N, D)

        if self.use_linear_drop:
            final = self.linear_dropout(final)

        return final

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embedding_dim": self.embedding_dim,
                "projection_dim": self.projection_dim,
                "qkv_bias": self.qkv_bias,
                "attention_drop": self.attention_drop,
                "linear_drop": self.linear_drop,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


if __name__ == "__main__":
    batch_dims = 1
    P = 4
    N = 256
    embedding_dim = 64

    num_heads = 4
    projection_dim = embedding_dim // num_heads  # 8
    use_bias = True

    proj_dim = embedding_dim // num_heads

    print("Einsum")
    lal = MultiHeadSelfAttentionEinSum2D(
        num_heads=num_heads,
        embedding_dim=embedding_dim,
        projection_dim=proj_dim,
        qkv_bias=use_bias,
        name="mhsa",
    )

    inputs = keras.random.normal((batch_dims, P, N, embedding_dim))
    _ = lal(inputs)

    model = keras.Sequential()

    model.add(keras.layers.InputLayer(shape=(None, None, embedding_dim)))
    model.add(lal)

    # model.summary()
    print(model.count_params())

    print("Tensorflow default")
    layer = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=proj_dim, use_bias=use_bias)
    _ = layer(inputs, inputs)

    inputs = keras.Input(shape=(None, None, embedding_dim))
    output_tensor = layer(inputs, inputs)
    model = keras.Model(inputs, output_tensor)

    print(model.count_params())
    # model.summary()
