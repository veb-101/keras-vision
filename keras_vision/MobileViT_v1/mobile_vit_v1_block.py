from typing import Union

import keras
import keras.ops as kops
from keras.layers import Layer, Dropout, Dense, LayerNormalization, Concatenate

from .base_layers import ConvLayer
from .multihead_self_attention_2D import MultiHeadSelfAttention as MHSA


class Transformer(Layer):
    def __init__(
        self,
        num_heads: int = 4,
        embedding_dim: int = 90,
        qkv_bias: bool = True,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        linear_drop: float = 0.0,
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
            linear_drop=dropout,
        )
        self.norm_2 = LayerNormalization(epsilon=1e-5)

        hidden_features = int(self.embedding_dim * self.mlp_ratio)

        self.mlp_block_0 = Dense(hidden_features, activation="swish")
        self.mlp_block_1 = Dropout(self.linear_drop)
        self.mlp_block_2 = Dense(embedding_dim)
        self.mlp_block_3 = Dropout(dropout)

    def call(self, x):
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

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# # https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/modules/mobilevit_blockops.py#L186
# def unfolding(
#     x,
#     B: int = 1,
#     D: int = 144,
#     patch_h: int = 2,
#     patch_w: int = 2,
#     num_patches_h: int = 10,
#     num_patches_w: int = 10,
# ):
#     """
#     ### Notations (wrt paper) ###
#         B/b = batch
#         P/p = patch_size
#         N/n = number of patches
#         D/d = embedding_dim

#     H, W
#     [                            [
#         [1, 2, 3, 4],     Goal      [1, 3, 9, 11],
#         [5, 6, 7, 8],     ====>     [2, 4, 10, 12],
#         [9, 10, 11, 12],            [5, 7, 13, 15],
#         [13, 14, 15, 16],           [6, 8, 14, 16]
#     ]                            ]
# #     """
#     # print("B", B)
#     # [B, H, W, D] --> [B*nh, ph, nw, pw*D]
#     reshaped_fm = kops.reshape(x, (B * num_patches_h, patch_h, num_patches_w, patch_w * D))

#     # [B*nh, ph, nw, pw*D] --> [B*nh, nw, ph, pw*D]
#     transposed_fm = kops.transpose(reshaped_fm, axes=[0, 2, 1, 3])

#     # [B*nh, nw, ph, pw*D] --> [B, N, P, D]
#     reshaped_fm = kops.reshape(transposed_fm, (B, num_patches_h * num_patches_w, patch_h * patch_w, D))

#     # [B, N, P, D] --> [B, P, N, D]
#     transposed_fm = kops.transpose(reshaped_fm, axes=[0, 2, 1, 3])

#     return transposed_fm


class MobileViT_v1_Block(Layer):
    def __init__(
        self,
        out_filters: int = 64,
        embedding_dim: int = 90,
        patch_size: Union[int, tuple] = 2,
        transformer_repeats: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_drop: float = 0.0,
        linear_drop: float = 0.0,
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
        self.patch_size_h, self.patch_size_w = kops.cast(self.patch_size_h, dtype="int32"), kops.cast(self.patch_size_w, dtype="int32")

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

    def call(self, x):

        fmH, fmW = kops.shape(x)[1], kops.shape(x)[2]

        local_representation = self.local_rep_layer_1(x)
        local_representation = self.local_rep_layer_2(local_representation)
        out_channels = local_representation.shape[-1]

        # Transformer as Convolution Steps
        # --------------------------------
        # # Unfolding

        unfolded, info_dict = self.unfolding(local_representation)

        # # Infomation sharing/mixing --> global representation
        for layer in self.transformer_layers:
            unfolded = layer(unfolded)

        global_representation = self.transformer_layer_norm(unfolded)

        # #Folding
        folded = self.folding(global_representation, info_dict=info_dict, outH=fmH, outW=fmW, outC=out_channels)

        # Fusion
        local_mix = self.local_features_3(folded)
        fusion = self.concat([x, local_mix])
        fusion = self.fuse_local_global(fusion)

        return fusion

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

    # https://github.com/apple/ml-cvnets/blob/84d992f413e52c0468f86d23196efd9dad885e6f/cvnets/modules/mobilevit_blockops.py#L186
    def unfolding(self, feature_map):
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
        shape = kops.shape(feature_map)
        batch_size, orig_h, orig_w, in_channels = shape[0], shape[1], shape[2], shape[3]
        feature_map = kops.transpose(feature_map, [0, 3, 1, 2])  # [B, H, W, C] -> [B, C, H, W]

        patch_area = self.patch_size_w * self.patch_size_h

        orig_h, orig_w = kops.cast(orig_h, dtype="int32"), kops.cast(orig_w, dtype="int32")

        h_ceil = kops.ceil(orig_h / self.patch_size_h)
        w_ceil = kops.ceil(orig_w / self.patch_size_w)

        new_h = kops.cast(h_ceil * kops.cast(self.patch_size_h, dtype=h_ceil.dtype), dtype="int32")
        new_w = kops.cast(w_ceil * kops.cast(self.patch_size_w, dtype=h_ceil.dtype), dtype="int32")

        # Condition to decide if resizing is necessary
        resize_required = kops.logical_or(kops.not_equal(new_w, orig_w), kops.not_equal(new_h, orig_h))
        feature_map = kops.cond(
            resize_required,
            true_fn=lambda: kops.image.resize(feature_map, [new_h, new_w], data_format="channels_first"),
            false_fn=lambda: feature_map,
        )

        num_patch_h = new_h // self.patch_size_h
        num_patch_w = new_w // self.patch_size_w
        num_patches = num_patch_h * num_patch_w

        # Handle dynamic shape multiplication
        dynamic_shape_mul = kops.prod([batch_size, in_channels * num_patch_h])

        # Reshape and transpose to create patches
        reshaped_fm = kops.reshape(feature_map, [dynamic_shape_mul, self.patch_size_h, num_patch_w, self.patch_size_w])
        transposed_fm = kops.transpose(reshaped_fm, [0, 2, 1, 3])
        reshaped_fm = kops.reshape(transposed_fm, [batch_size, in_channels, num_patches, patch_area])
        transposed_fm = kops.transpose(reshaped_fm, [0, 3, 2, 1])
        patches = kops.reshape(transposed_fm, [batch_size * patch_area, num_patches, in_channels])

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": resize_required,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
            "patch_area": patch_area,
        }

        return patches, info_dict

    def folding(self, patches, info_dict, outH, outW, outC):
        # Ensure the input patches tensor has the correct dimensions
        assert len(patches.shape) == 3, f"Tensor should be of shape BPxNxC. Got: {patches.shape}"

        # Reshape to [B, P, N, C]
        patches = kops.reshape(patches, [info_dict["batch_size"], info_dict["patch_area"], info_dict["total_patches"], -1])

        # Get shape parameters for further processing
        shape = kops.shape(patches)
        batch_size = shape[0]
        channels = shape[3]

        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # Transpose dimensions [B, P, N, C] --> [B, C, N, P]
        patches = kops.transpose(patches, [0, 3, 2, 1])

        # Calculate total elements dynamically
        num_total_elements = batch_size * channels * num_patch_h

        # Reshape to match the size of the feature map before splitting into patches
        # [B, C, N, P] --> [B*C*n_h, n_w, p_h, p_w]
        feature_map = kops.reshape(patches, [num_total_elements, num_patch_w, self.patch_size_h, self.patch_size_w])

        # Transpose to switch width and height axes [B*C*n_h, n_w, p_h, p_w] --> [B*C*n_h, p_h, n_w, p_w]
        feature_map = kops.transpose(feature_map, [0, 2, 1, 3])

        # Reshape back to the original image dimensions [B*C*n_h, p_h, n_w, p_w] --> [B, C, H, W]
        # Reshape back to [B, C, H, W]
        new_height = num_patch_h * self.patch_size_h
        new_width = num_patch_w * self.patch_size_w
        feature_map = kops.reshape(feature_map, [batch_size, -1, new_height, new_width])

        # Conditional resizing using kops.cond
        feature_map = kops.cond(
            info_dict["interpolate"],
            lambda: kops.image.resize(feature_map, info_dict["orig_size"], data_format="channels_first"),
            lambda: feature_map,
        )

        feature_map = kops.transpose(feature_map, [0, 2, 3, 1])
        feature_map = kops.reshape(feature_map, (batch_size, outH, outW, outC))

        return feature_map

    @classmethod
    def from_config(cls, config):
        return cls(**config)


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
