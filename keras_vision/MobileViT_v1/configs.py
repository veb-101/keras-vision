from typing import Optional
from dataclasses import dataclass


@dataclass
class Config_MobileViT_v1_S:
    block_1_1_dims: int = 16
    block_1_2_dims: int = 32

    block_2_1_dims: int = 64
    block_2_2_dims: int = 64
    block_2_3_dims: int = 64

    block_3_1_dims: int = 96
    block_3_2_dims: int = 96

    block_4_1_dims: int = 128
    block_4_2_dims: int = 128

    block_5_1_dims: int = 160
    block_5_2_dims: int = 160

    final_conv_dims: int = 640

    tf_block_3_dims: int = 144
    tf_block_4_dims: int = 192
    tf_block_5_dims: int = 240

    tf_block_3_repeats: int = 2
    tf_block_4_repeats: int = 4
    tf_block_5_repeats: int = 3

    depthwise_expansion_factor: int = 4


@dataclass
class Config_MobileViT_v1_XS:
    block_1_1_dims: int = 16
    block_1_2_dims: int = 32

    block_2_1_dims: int = 48
    block_2_2_dims: int = 48
    block_2_3_dims: int = 48

    block_3_1_dims: int = 64
    block_3_2_dims: int = 64

    block_4_1_dims: int = 80
    block_4_2_dims: int = 80

    block_5_1_dims: int = 96
    block_5_2_dims: int = 96

    final_conv_dims: int = 384

    tf_block_3_dims: int = 96
    tf_block_4_dims: int = 120
    tf_block_5_dims: int = 144

    tf_block_3_repeats: int = 2
    tf_block_4_repeats: int = 4
    tf_block_5_repeats: int = 3

    depthwise_expansion_factor: int = 4


@dataclass
class Config_MobileViT_v1_XXS:
    block_1_1_dims: int = 16
    block_1_2_dims: int = 16

    block_2_1_dims: int = 24
    block_2_2_dims: int = 24
    block_2_3_dims: int = 24

    block_3_1_dims: int = 48
    block_3_2_dims: int = 48

    block_4_1_dims: int = 64
    block_4_2_dims: int = 64

    block_5_1_dims: int = 80
    block_5_2_dims: int = 80

    final_conv_dims: int = 320

    tf_block_3_dims: int = 64
    tf_block_4_dims: int = 80
    tf_block_5_dims: int = 96

    tf_block_3_repeats: int = 2
    tf_block_4_repeats: int = 4
    tf_block_5_repeats: int = 3

    depthwise_expansion_factor: int = 2


def get_mobile_vit_v1_configs(model_type: str = "S", updates: Optional[dict] = None):
    if model_type == "S":
        base_config = Config_MobileViT_v1_S
    elif model_type == "XS":
        base_config = Config_MobileViT_v1_XS
    else:
        base_config = Config_MobileViT_v1_XXS

    if updates:
        return base_config(**updates)

    return base_config
