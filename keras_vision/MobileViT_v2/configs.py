from typing import Optional
from dataclasses import dataclass
from .utils import bound_fn, make_divisible


@dataclass
class Config_MobileViT_v2:
    block_1_1_dims: int = 32
    block_1_2_dims: int = 64

    block_2_1_dims: int = 128
    block_2_2_dims: int = 128

    block_3_1_dims: int = 256
    block_3_2_dims: int = 256

    block_4_1_dims: int = 384
    block_4_2_dims: int = 384

    block_5_1_dims: int = 512
    block_5_2_dims: int = 512

    tf_block_3_dims: int = 128
    tf_block_4_dims: int = 192
    tf_block_5_dims: int = 256

    tf_block_3_repeats: int = 2
    tf_block_4_repeats: int = 4
    tf_block_5_repeats: int = 3

    depthwise_expansion_factor: int = 2


def get_mobile_vit_v2_configs(width_multiplier: float, updates: Optional[dict]):

    if updates:
        configs = Config_MobileViT_v2(**updates)
    else:
        configs = Config_MobileViT_v2()

    configs.block_1_1_dims = bound_fn(min_val=16, max_val=64, value=configs.block_1_1_dims * width_multiplier)
    configs.block_1_1_dims = int(make_divisible(configs.block_1_1_dims, divisor=8, min_value=16))

    configs.block_1_2_dims = int(make_divisible(configs.block_1_2_dims * width_multiplier, divisor=16))

    configs.block_2_1_dims = int(make_divisible(configs.block_2_1_dims * width_multiplier, divisor=8))
    configs.block_2_2_dims = int(make_divisible(configs.block_2_2_dims * width_multiplier, divisor=8))

    configs.block_3_1_dims = int(make_divisible(configs.block_3_1_dims * width_multiplier, divisor=8))
    configs.block_3_2_dims = int(make_divisible(configs.block_3_2_dims * width_multiplier, divisor=8))

    configs.block_4_1_dims = int(make_divisible(configs.block_4_1_dims * width_multiplier, divisor=8))
    configs.block_4_2_dims = int(make_divisible(configs.block_4_2_dims * width_multiplier, divisor=8))

    configs.block_5_1_dims = int(make_divisible(configs.block_5_1_dims * width_multiplier, divisor=8))
    configs.block_5_2_dims = int(make_divisible(configs.block_5_2_dims * width_multiplier, divisor=8))

    configs.tf_block_3_dims = int(make_divisible(configs.tf_block_3_dims * width_multiplier, divisor=8))
    configs.tf_block_4_dims = int(make_divisible(configs.tf_block_4_dims * width_multiplier, divisor=8))
    configs.tf_block_5_dims = int(make_divisible(configs.tf_block_5_dims * width_multiplier, divisor=8))

    return configs
