import warnings
from typing import Optional
import numpy as np
from keras import Model, Input
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.utils import get_file


from .configs import get_mobile_vit_v2_configs
from .base_layers import ConvLayer, InvertedResidualBlock
from .mobile_vit_v2_block import MobileViT_v2_Block


def MobileViT_v2(
    configs,
    linear_drop: float = 0.1,
    attention_drop: float = 0.0,
    dropout: float = 0.0,
    num_classes: int | None = 1000,
    input_shape: tuple[int, int, int] = (256, 256, 3),
    model_name: str = "MobileViT-v3-1.0",
):
    """
    Arguments
    --------

        configs: A dataclass instance with model information such as per layer output channels, transformer embedding dimensions, transformer repeats, IR expansion factor

        num_classes: (int)   Number of output classes

        input_shape: (tuple) Input shape -> H, W, C

        model_type: (str)   Model to create

        linear_drop: (float) Dropout rate for Dense layers

        attention_drop: (float) Dropout rate for the attention matrix

    """

    input_layer = Input(shape=input_shape)

    # Block 1
    out = ConvLayer(
        num_filters=configs.block_1_1_dims,
        kernel_size=3,
        strides=2,
        name="block-1-Conv",
    )(input_layer)

    out = InvertedResidualBlock(
        in_channels=configs.block_1_1_dims,
        out_channels=configs.block_1_2_dims,
        depthwise_stride=1,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-1-IR2",
    )(out)

    # Block 2
    out = InvertedResidualBlock(
        in_channels=configs.block_1_2_dims,
        out_channels=configs.block_2_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-2-IR1",
    )(out)

    out = InvertedResidualBlock(
        in_channels=configs.block_2_1_dims,
        out_channels=configs.block_2_2_dims,
        depthwise_stride=1,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-2-IR2",
    )(out)

    # # ========================================================
    # # According to paper, there should be one more InvertedResidualBlock, but it not present in the final code.

    # out_b2_3 = InvertedResidualBlock(
    #     in_channels=configs.block_2_2_dims,
    #     out_channels=configs.block_2_3_dims,
    #     depthwise_stride=1,
    #     expansion_factor=configs.depthwise_expansion_factor,
    #     name="block-2-IR3",
    # )(out)

    # out = out + out_b2_3
    # # ========================================================

    # Block 3
    out = InvertedResidualBlock(
        in_channels=configs.block_2_2_dims,
        out_channels=configs.block_3_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-3-IR1",
    )(out)

    out = MobileViT_v2_Block(
        out_filters=configs.block_3_2_dims,
        embedding_dim=configs.tf_block_3_dims,
        transformer_repeats=configs.tf_block_3_repeats,
        name="MobileViTBlock-1",
        dropout=dropout,
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 4
    out = InvertedResidualBlock(
        in_channels=configs.block_3_2_dims,
        out_channels=configs.block_4_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-4-IR1",
    )(out)

    out = MobileViT_v2_Block(
        out_filters=configs.block_4_2_dims,
        embedding_dim=configs.tf_block_4_dims,
        transformer_repeats=configs.tf_block_4_repeats,
        name="MobileViTBlock-2",
        dropout=dropout,
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    # Block 5
    out = InvertedResidualBlock(
        in_channels=configs.block_4_2_dims,
        out_channels=configs.block_5_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-5-IR1",
    )(out)

    out = MobileViT_v2_Block(
        out_filters=configs.block_5_2_dims,
        embedding_dim=configs.tf_block_5_dims,
        transformer_repeats=configs.tf_block_5_repeats,
        name="MobileViTBlock-3",
        dropout=dropout,
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    if num_classes:
        # Output layer
        out = GlobalAveragePooling2D()(out)

        if linear_drop > 0.0:
            out = Dropout(rate=dropout)(out)

        out = Dense(units=num_classes)(out)

    model = Model(inputs=input_layer, outputs=out, name=model_name)

    return model


def build_MobileViT_v2(
    width_multiplier: float = 1.0,
    num_classes: int = 1000,
    input_shape: tuple = (256, 256, 3),
    include_top: bool = True,  # Whether to include the classification layer in the model
    pretrained: bool = False,  # Whether to load pretrained weights
    cache_dir: Optional[str] = None,  # Local cache directory for weights
    updates: Optional[dict] = None,
    **kwargs,
):
    """
    Create MobileViT-v2 Classification models

    Arguments
    --------
        width_multiplier: (int, float) manipulate number of channels.
                            Default: 1.0 --> Refers to the base model.

        num_classes: (int)   Number of output classes

        input_shape: (tuple) Input shape -> H, W, C

        updates: (dict) a key-value pair indicating the changes to be made to the base model.

    Additional arguments:
    ---------------------

        linear_drop: (float) Dropout rate for Dense layers

        attention_drop: (float) Dropout rate for the attention matrix

    """

    updated_configs = get_mobile_vit_v2_configs(width_multiplier, updates=updates)

    # Build the base model
    model = MobileViT_v2(
        configs=updated_configs,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name=f"MobileViT-v2-{width_multiplier}",
        **kwargs,
    )

    # Initialize parameters of MobileViT block.
    if None in input_shape:
        dummy_input_shape = (1, 256, 256, 3)
    else:
        dummy_input_shape = (1,) + input_shape

    model(np.random.randn(*dummy_input_shape), training=False)

    # if pretrained:
    #     weights_path = get_file(
    #         fname=f"keras_MobileVIT_v1_model_{model_type}.weights.h5",
    #         origin=WEIGHTS_URL.format(version=VERSION, model_type=model_type),
    #         cache_subdir="models",
    #         hash_algorithm="auto",
    #         extract=False,
    #         archive_format="auto",
    #         cache_dir=cache_dir,
    #     )

    #     with warnings.catch_warnings():
    #         # Ignore UserWarnings within this block
    #         warnings.simplefilter("ignore", UserWarning)
    #         model.load_weights(weights_path, skip_mismatch=True)

    return model


if __name__ == "__main__":

    model = build_MobileViT_v2(
        width_multiplier=0.75,
        input_shape=(None, None, 3),
        num_classes=1000,
        linear_drop=0.0,
        attention_drop=0.0,
    )

    print(f"{model.name} num. parametes: {model.count_params()}")
    # _ = model(tf.random.uniform((1, 256, 256, 3)), training=False)
    # model.summary(positions=[0.33, 0.64, 0.75, 1.0])
    # model.save(f"{model.name}", include_optimizer=False)

    # Refer to Config_MobileViT_v2 class to see all customizable modules available.
    # updates = {
    #     "block_3_1_dims": 256,
    #     "block_3_2_dims": 384,
    #     "tf_block_3_dims": 164,
    #     "tf_block_3_repeats": 3,
    # }

    # model = build_MobileViT_v2(
    #     width_multiplier=0.75,
    #     updates=updates,
    #     linear_drop=0.0,
    #     attention_drop=0.0,
    # )

    # model.summary(positions=[0.33, 0.64, 0.75, 1.0])
    # print(f"{model.name} num. parametes: {model.count_params()}")
    # model.save(f"{model.name}", include_optimizer=False)
