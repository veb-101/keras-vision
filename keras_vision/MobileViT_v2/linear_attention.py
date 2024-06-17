import keras.ops as kops
from keras.layers import Layer, Dropout, EinsumDense


class LinearSelfAttention(Layer):
    def __init__(
        self,
        embedding_dim: int = 64,
        qkv_bias: bool = True,
        attention_drop: float = 0.2,
        linear_drop: float = 0.2,
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

        # Shape: (B, P, N, D) * (D, 1 + (2 * D)) --> (B, P, N, 1 + (2 * D))
        self.W_QKV = EinsumDense(
            "bpnd,de->bpne",
            output_shape=[None, None, 1 + (2 * self.embedding_dim)],
            bias_axes="e" if self.qkv_bias else None,
        )
        self.attn_dropout = Dropout(self.attention_drop)

        self.Wo = EinsumDense("bpnd,de->bpne", output_shape=[None, None, self.embedding_dim], bias_axes="e" if self.qkv_bias else None)

        self.linear_dropout = Dropout(self.linear_drop)

        # Shape: (B, P, N, D) * (D, D) --> (B, P, N, D)

    def build(self, input_shape):
        # You can perform setup tasks that depend on the input shape here
        super().build(input_shape)

    def call(self, inputs):
        """
        From paper repo:

        "For MobileViTv2, we unfold the feature map [B, C, H, W] into [B, C, P, N] where P is the number of pixels
        in a patch and N is the number of patches. Because channel is the first dimension in this unfolded tensor,
        we use point-wise convolution (instead of a linear layer). This avoids a transpose operation (which may be
        expensive on resource-constrained devices) that may be required to convert the unfolded tensor from
        channel-first to channel-last format in case of a linear layer."

        In TensorFlow as channel C is already the last dimension, I'm using the EinsumDense layer.
        """

        # Inputs Shape --> (B, P, N, D)
        qkv = self.W_QKV(inputs)  # Shape: (B, P, N, (1 + 2*D))

        # q = kops.expand_dims(qkv[:, :, :, 0], -1)
        # k = qkv[:, :, :, 1 : self.embedding_dim + 1]
        # v = qkv[:, :, :, self.embedding_dim + 1 :]
        # print(q.shape, k.shape, v.shape)

        q, k, v = kops.split(qkv, indices_or_sections=[1, 1 + self.embedding_dim], axis=-1)

        v = kops.relu(v)

        # Apply softmax along N dimension
        context_scores = kops.nn.softmax(q, axis=2)

        context_scores = self.attn_dropout(context_scores)

        # Compute context vector
        # [B, P, N, d] x [B, P, N, 1] -> [B, P, N, d]
        context_vector = k * context_scores

        # [B, P, N, d] --> [B, P, 1, d]
        context_vector = kops.sum(context_vector, axis=2, keepdims=True)

        # Combine context vector with values
        # [B, P, N, d] * [B, P, 1, d] --> [B, P, N, d]
        # [B, P, 1, d] ---expand---> [B, P, N, d]
        updated_values = kops.einsum("...nd, ...kd->...nd", v, context_vector)

        # Shape: (B, P, N, D) * (D, ) --> (B, P, N, D)
        final = self.Wo(updated_values)

        final = self.linear_dropout(final)

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

#     inputs = tf.random.normal((batch_dims, P, N, embedding_dim))
#     _ = lal(inputs)

#     model = tf.keras.Sequential()

#     model.add(tf.keras.layers.InputLayer(input_shape=(None, None, embedding_dim)))
#     model.add(lal)

#     # model.summary()
#     print(model.count_params())
