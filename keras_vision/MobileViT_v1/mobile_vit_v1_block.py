import keras
import keras.ops as kops  # type: ignore
from keras.layers import Layer, Dropout, Dense, LayerNormalization, Concatenate  # type: ignore

from .base_layers import ConvLayer
from .multihead_self_attention_2D import MultiHeadSelfAttention as MHSA
from math import ceil


class Transformer(Layer):
    def __init__(
        self,
        num_heads: int = 4,
        embedding_dim: int = 90,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        linear_drop: float = 0.1,
        attention_drop: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.embedding_dim = embedding_dim
        self.qkv_bias = qkv_bias
        self.mlp_ratio = mlp_ratio
        self.dropout = dropout
        self.linear_drop = linear_drop
        self.attention_drop = attention_drop

        self.norm_1 = LayerNormalization(epsilon=1e-5)

        self.attn = MHSA(
            num_heads=self.num_heads,
            embedding_dim=self.embedding_dim,
            qkv_bias=self.qkv_bias,
            attention_drop=self.attention_drop,
            linear_drop=self.linear_drop,
        )
        self.norm_2 = LayerNormalization(epsilon=1e-5)

        hidden_features = int(self.embedding_dim * self.mlp_ratio)

        self.mlp_block_0 = Dense(hidden_features, activation="swish")
        self.mlp_block_1 = Dropout(self.dropout)
        self.mlp_block_2 = Dense(self.embedding_dim)
        self.mlp_block_3 = Dropout(self.linear_drop)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x, training=False):
        x = x + self.attn(self.norm_1(x))

        mlp_block_out = self.mlp_block_0(self.norm_2(x))
        mlp_block_out = self.mlp_block_1(mlp_block_out)
        mlp_block_out = self.mlp_block_2(mlp_block_out)
        mlp_block_out = self.mlp_block_3(mlp_block_out)

        x = x + mlp_block_out

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_heads": self.num_heads,
                "embedding_dim": self.embedding_dim,
                "qkv_bias": self.qkv_bias,
                "mlp_ratio": self.mlp_ratio,
                "dropout": self.dropout,
                "linear_drop": self.linear_drop,
                "attention_drop": self.attention_drop,
            }
        )
        return config


class MobileViT_v1_Block(Layer):
    def __init__(
        self,
        out_filters: int = 64,
        embedding_dim: int = 90,
        patch_size: int = 2,
        transformer_repeats: int = 2,
        num_heads: int = 4,
        linear_drop: float = 0.1,
        attention_drop: float = 0.0,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.out_filters = out_filters
        self.embedding_dim = embedding_dim
        self.patch_size = patch_size
        self.transformer_repeats = transformer_repeats
        self.num_heads = num_heads
        self.dropout = dropout
        self.attention_drop = attention_drop
        self.linear_drop = linear_drop

        self.patch_size_h, self.patch_size_w = patch_size if isinstance(self.patch_size, tuple) else (self.patch_size, self.patch_size)
        self.patch_area = self.patch_size_h * self.patch_size_w

        # # local_feature_extractor 1 and 2
        self.local_rep_layer_1 = ConvLayer(num_filters=self.out_filters, kernel_size=3, strides=1, use_bn=True, use_activation=True)
        self.local_rep_layer_2 = ConvLayer(num_filters=self.embedding_dim, kernel_size=1, strides=1, use_bn=False, use_activation=False, use_bias=False)

        self.transformer_layers = [
            Transformer(
                embedding_dim=self.embedding_dim,
                num_heads=self.num_heads,
                dropout=self.dropout,
                attention_drop=self.attention_drop,
                linear_drop=self.linear_drop,
            )
            for _ in range(self.transformer_repeats)
        ]

        self.transformer_layer_norm = LayerNormalization(epsilon=1e-5)

        # Fusion blocks
        self.local_features_3 = ConvLayer(num_filters=self.out_filters, kernel_size=1, strides=1, use_bn=True, use_activation=True)
        self.concat = Concatenate(axis=-1)
        self.fuse_local_global = ConvLayer(num_filters=self.out_filters, kernel_size=3, strides=1, use_bn=True, use_activation=True)

    def build(self, input_shape):
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        shape = input_shape
        return (shape[0], shape[1], shape[2], self.out_filters)

    def call(self, x, training=False):
        # Local Representation
        local_representation = self.local_rep_layer_1(x)
        local_representation = self.local_rep_layer_2(local_representation)

        # Transformer as Convolution Steps
        # --------------------------------
        # # Unfolding
        unfolded, info_dict = self.unfolding(local_representation)

        # # Infomation sharing/mixing --> global representation
        for layer in self.transformer_layers:
            unfolded = layer(unfolded)

        global_representation = self.transformer_layer_norm(unfolded)

        # # Folding
        folded = self.folding(global_representation, info_dict=info_dict)
        # --------------------------------

        # Fusion
        local_mix = self.local_features_3(folded)
        fusion = self.concat([x, local_mix])
        fusion = self.fuse_local_global(fusion)

        return fusion

    def unfolding(self, x):
        """
        ### Notations (wrt paper) ###
            B/b = batch
            P/p = patch_size
            N/n = number of patches
            D/d = embedding_dim

        H, W
        [                            [
            [1, 2, 3, 4],     Goal      [1, 3, 9, 11],
            [5, 6, 7, 8],     ====>     [2, 4, 10, 12],
            [9, 10, 11, 12],            [5, 7, 13, 15],
            [13, 14, 15, 16],           [6, 8, 14, 16]
        ]                            ]
        """

        # Initially convert channel-last to channel-first for processing
        shape = kops.shape(x)
        batch_size, orig_h, orig_w, D = shape[0], shape[1], shape[2], shape[3]
        
        # orig_h, orig_w, D = x.shape[1], x.shape[2], x.shape[3]       

        h_ceil = ceil(orig_h / self.patch_size_h)
        w_ceil = ceil(orig_w / self.patch_size_w)

        new_h = h_ceil * self.patch_size_h
        new_w = w_ceil * self.patch_size_w

        # Condition to decide if resizing is necessary
        resize_required = (new_h != orig_h) or (new_w != orig_w)

        if resize_required:
            x = kops.image.resize(x, (new_h, new_w))
            num_patches_h = new_h // self.patch_size_h
            num_patches_w = new_w // self.patch_size_w
        else:
            num_patches_h = orig_h // self.patch_size_h
            num_patches_w = orig_w // self.patch_size_w

        num_patches = num_patches_h * num_patches_w

        # [B, H, W, D] --> [B*nh, ph, nw, pw*D]
        reshaped_fm = kops.reshape(x, (batch_size * num_patches_h, self.patch_size_h, num_patches_w, self.patch_size_w * D))

        # [B * n_h, p_h, n_w, p_w*D] --> [B * n_h, n_w, p_h, p_w * D]
        transposed_fm = kops.transpose(reshaped_fm, axes=[0, 2, 1, 3])

        # [B * n_h, n_w, p_h, p_w * D] --> [B, N, P, D] where P = p_h * p_w and N = n_h * n_w
        reshaped_fm = kops.reshape(transposed_fm, (batch_size, num_patches, self.patch_area, D))

        # [B, N, P, D] --> [B, P, N, D]
        transposed_fm = kops.transpose(reshaped_fm, axes=[0, 2, 1, 3])

        # [B, P, N, D] -> [BP, N, D]
        patches = kops.reshape(transposed_fm, [batch_size * self.patch_area, num_patches, D])

        info_dict = {
            "batch_size": batch_size,
            "orig_size": (orig_h, orig_w),
            "resize": resize_required,
            "num_patches_h": num_patches_h,
            "num_patches_w": num_patches_w,
            "total_patches": num_patches,
        }

        return patches, info_dict

    def folding(self, x, info_dict):
        """
        ### Notations (wrt paper) ###
            B/b = batch
            P/p = patch_size
            N/n = number of patches
            D/d = embedding_dim
        """

        # Get shape parameters for further processing
        shape = kops.shape(x)
        D = shape[2]
        batch_size = info_dict["batch_size"]
        num_patches = info_dict["total_patches"]
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]
        resize_required = info_dict["resize"]

        # Reshape to [BP, N, D] -> [B, P, N, D]
        x = kops.reshape(x, [batch_size, self.patch_area, num_patches, D])

        # [B, P, N D] --> [B, N, P, D]
        x = kops.transpose(x, (0, 2, 1, 3))

        # [B, N, P, D] --> [B *n_h, n_w, p_h, p_w * D]
        x = kops.reshape(x, (batch_size * num_patch_h, num_patch_w, self.patch_size_h, self.patch_size_w * D))

        # [B *n_h, n_w, p_h, p_w * D] --> [B * n_h, p_h, n_w, p_w * D]
        x = kops.transpose(x, (0, 2, 1, 3))

        # [B * n_h, p_h, n_w, p_w * D] --> [B, n_h * p_h, n_w, p_w, D] --> [B, H, W, C]
        x = kops.reshape(x, (batch_size, num_patch_h * self.patch_size_h, num_patch_w * self.patch_size_w, D))

        if resize_required:
            x = kops.image.resize(x, info_dict["orig_size"])

        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "out_filters": self.out_filters,
                "embedding_dim": self.embedding_dim,
                "patch_size": self.patch_size,
                "transformer_repeats": self.transformer_repeats,
                "num_heads": self.num_heads,
                "dropout": self.dropout,
                "attention_drop": self.attention_drop,
                "linear_drop": self.linear_drop,
            }
        )
        return config


if __name__ == "__main__":
    batch = 1
    H = W = 32
    C = 48
    P = 2 * 2
    L = 2
    embedding_dim = 64

    mvitblk = MobileViT_v1_Block(
        out_filters=C,
        embedding_dim=embedding_dim,
        patch_size=P,
        transformer_repeats=L,
        attention_drop=0.0,
        linear_drop=0.0,
    )

    inputs = keras.random.normal((batch, H, W, C))

    out = mvitblk(inputs)
    print("inputs.shape", inputs.shape)
    print("out.shape", out.shape)
    for i in mvitblk.get_weights():
        print(i.shape)
    print()
