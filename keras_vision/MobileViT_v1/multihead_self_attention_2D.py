import keras.ops as kops # type: ignore
from keras.layers import Layer, Dropout, Dense, Softmax # type: ignore


class MultiHeadSelfAttention(Layer):
    def __init__(
        self,
        num_heads: int = 2,
        embedding_dim: int = 64,
        projection_dim: int = None,
        qkv_bias: bool = True,
        attention_drop: float = 0.2,
        linear_drop: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim if projection_dim else embedding_dim // num_heads
        self.qkv_bias = qkv_bias
        self.scale = self.projection_dim**-0.5

        self.qkv = Dense(3 * self.num_heads * self.projection_dim, use_bias=qkv_bias)
        self.proj = Dense(embedding_dim, use_bias=qkv_bias)
        self.attn_dropout = Dropout(attention_drop)
        self.linear_dropout = Dropout(linear_drop)
        self.softmax = Softmax()

    def build(self, input_shape):
        # You can perform setup tasks that depend on the input shape here
        super().build(input_shape)

    def split_heads(self, x, batch_size):
        # Split the last dimension into (num_heads, projection_dim)
        x = kops.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        # Transpose to shape (batch_size, num_heads, seq_len, projection_dim)
        return kops.transpose(x, axes=(0, 2, 1, 3))

    def call(self, x, training=False):
        batch_size = kops.shape(x)[0]

        # Project and reshape to (batch_size, seq_len, 3, num_heads, projection_dim)
        qkv = self.qkv(x)
        qkv = kops.reshape(qkv, (batch_size, -1, 3, self.num_heads, self.projection_dim))
        qkv = kops.transpose(qkv, axes=(0, 2, 1, 3, 4))
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        q *= self.scale

        # Attention mechanism
        attn_logits = kops.matmul(q, kops.transpose(k, axes=(0, 1, 3, 2)))
        attn = self.softmax(attn_logits)
        attn = self.attn_dropout(attn)

        weighted_avg = kops.matmul(attn, v)
        weighted_avg = kops.transpose(weighted_avg, axes=(0, 2, 1, 3))
        weighted_avg = kops.reshape(weighted_avg, (batch_size, -1, self.num_heads * self.projection_dim))

        # Output projection
        output = self.proj(weighted_avg)
        output = self.linear_dropout(output)

        return output
