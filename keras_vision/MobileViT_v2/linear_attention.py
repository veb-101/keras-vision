import keras.ops as kops
from keras.layers import Layer, Dropout
from .base_layers import ConvLayer


class LinearSelfAttention(Layer):
    def __init__(
        self,
        embedding_dim: int = 64,
        qkv_bias: bool = True,
        attention_drop: float = 0.0,
        linear_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        ##### Notations (wrt paper) #####
        # B/b = batch
        # P/p = patch_size
        # N/n = number of patches
        # D/d = embedding_dim

        # self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.qkv_bias = qkv_bias
        self.attention_drop = attention_drop
        self.linear_drop = linear_drop

        # # Shape: (B, P, N, D) * (D, 1 + (2 * D)) --> (B, P, N, 1 + (2 * D))
        self.W_QKV = ConvLayer(
            num_filters=1 + (2 * self.embedding_dim),
            kernel_size=1,
            use_bn=False,
            use_bias=self.qkv_bias,
            strides=1,
            use_activation=False,
        )
        self.attn_dropout = Dropout(self.attention_drop)

        self.Wo = ConvLayer(
            num_filters=self.embedding_dim,
            kernel_size=1,
            strides=1,
            use_bn=False,
            use_bias=True,
            use_activation=False,
        )

        self.linear_dropout = Dropout(self.linear_drop)

        # Shape: (B, P, N, D) * (D, D) --> (B, P, N, D)

    def build(self, input_shape):
        # You can perform setup tasks that depend on the input shape here
        super().build(input_shape)

    def call(self, inputs, training=False):
        # Inputs Shape --> (B, P, N, D)
        qkv = self.W_QKV(inputs, training=training)  # Shape: (B, P, N, (1 + 2*D))

        # Split qkv into query, key, and value
        # Project to query, key and value
        # Query --> [B, P, N, 1]
        # value, key --> [B, P, N, d]

        # q = kops.expand_dims(qkv[:, :, :, 0], -1)
        # k = qkv[:, :, :, 1 : self.embedding_dim + 1]
        # v = qkv[:, :, :, self.embedding_dim + 1 :]
        # print(q.shape, k.shape, v.shape)

        q, k, v = kops.split(qkv, indices_or_sections=[1, 1 + self.embedding_dim], axis=-1)
        # print("q", q.shape)
        # print("k", k.shape)
        # print("v", v.shape)

        # Apply softmax along N dimension
        context_scores = kops.softmax(q, axis=-2)

        context_scores = self.attn_dropout(context_scores, training=training)

        # Compute context vector
        # [B, P, N, d] x [B, P, N, 1] -> [B, P, N, d]
        context_vector = k * context_scores

        # [B, P, N, d] --> [B, P, 1, d]
        context_vector = kops.sum(context_vector, axis=-2, keepdims=True)
        # print("context_vector", context_vector.shape)

        # Combine context vector with values
        # [B, P, N, d] * [B, P, 1, d] --> [B, P, N, d]
        # [B, P, 1, d] ---expand---> [B, P, N, d]
        v = kops.relu(v)
        updated_values = kops.einsum("...nd, ...kd->...nd", v, context_vector)

        # Shape: (B, P, N, D) * (D, ) --> (B, P, N, D)
        final = self.Wo(updated_values, training=training)

        final = self.linear_dropout(final, training=training)
        return final

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embedding_dim": self.embedding_dim,
                "qkv_bias": self.qkv_bias,
                "attention_drop": self.attention_drop,
                "linear_drop": self.linear_drop,
            }
        )
        return config


# if __name__ == "__main__":

#     batch_dims = 1
#     P = 4
#     N = 256
#     embedding_dim = 64
#     use_bias = True

#     lal = LinearSelfAttention(
#         embedding_dim=embedding_dim,
#         qkv_bias=use_bias,
#         name="LSA",
#     )
