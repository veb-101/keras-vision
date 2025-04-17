import warnings
from typing import Optional

import numpy as np
from keras import Model, Input
from keras import layers as keras_layer
from keras import utils

from .configs import get_mobile_vit_v1_configs
from .base_layers import ConvLayer, InvertedResidualBlock

from .mobile_vit_v1_block import MobileViT_v1_Block

WEIGHTS_URL = (
    r"https://huggingface.co/veb-101/Keras-3-apple-mobilevit/resolve/main/keras-3-mobilevit-v1-weights/keras_MobileVIT_v1_model_{model_type}.weights.h5"
)


def MobileViT_v1(
    configs,
    linear_drop: float = 0.1,
    attention_drop: float = 0.0,
    dropout: float = 0.0,
    num_classes: int | None = 1000,
    classifier_head_activation: str = "linear",
    input_shape: tuple[int, int, int] = (256, 256, 3),
    model_name: str = "MobileViT_v1-S",
):
    """
    Build the MobileViT-v1 model architecture.

    Args:
        configs (dataclass): A dataclass instance containing model information such as per-layer output channels, transformer embedding dimensions, transformer repeats, and IR expansion factor.

        linear_drop (float): Dropout rate for the Dense layers. Default is 0.1

        attention_drop (float): Dropout rate for the attention matrix. Default is 0.0

        dropout (float): Dropout rate to be applied between different layers. Default is 0.0

        num_classes (int): The number of output classes for the classification task. If None, no classification layer is added. Default is 1000.

        classifier_head_activation (str): Activation function to use after the final dense layer in classification head. Default: "linear". Other options include: "softmax", "sigmoid", etc.

        input_shape (tuple): The shape of the input data in the format (height, width, channels). Default is (256, 256, 3).

        model_name (str): The name of the model. Default is "MobileViT-v1-S".

    Returns:
        keras.Model: The constructed MobileViT-v1 model instance.

    Example
    -------
    >>> configs = get_mobile_vit_v1_configs(model_type="S")
    >>> model = MobileViT_v1(
            configs=configs,
            linear_drop=0.1,
            attention_drop=0.1,
            dropout=0.2,
            num_classes=1000,
            input_shape=(256, 256, 3),
            model_name=f"MobileViT_v1-S",
        )
    >>> model.summary()
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

    out = InvertedResidualBlock(
        in_channels=configs.block_2_2_dims,
        out_channels=configs.block_2_3_dims,
        depthwise_stride=1,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-2-IR3",
    )(out)

    # Block 3
    out = InvertedResidualBlock(
        in_channels=configs.block_2_2_dims,
        out_channels=configs.block_3_1_dims,
        depthwise_stride=2,
        expansion_factor=configs.depthwise_expansion_factor,
        name="block-3-IR1",
    )(out)

    out = MobileViT_v1_Block(
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

    out = MobileViT_v1_Block(
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

    out = MobileViT_v1_Block(
        out_filters=configs.block_5_2_dims,
        embedding_dim=configs.tf_block_5_dims,
        transformer_repeats=configs.tf_block_5_repeats,
        name="MobileViTBlock-3",
        dropout=dropout,
        attention_drop=attention_drop,
        linear_drop=linear_drop,
    )(out)

    out = ConvLayer(num_filters=configs.final_conv_dims, kernel_size=1, strides=1, name="final_conv")(out)

    if num_classes:
        # Output layer
        out = keras_layer.GlobalAveragePooling2D()(out)

        if linear_drop > 0.0:
            out = keras_layer.Dropout(rate=dropout)(out)

        out = keras_layer.Dense(units=num_classes, activation=classifier_head_activation)(out)
        out = keras_layer.Activation(activation=classifier_head_activation, name=f"{classifier_head_activation}")(out)

    model = Model(inputs=input_layer, outputs=out, name=model_name)

    return model


def build_MobileViT_v1(
    model_type: str = "S",
    num_classes: int = 1000,
    classifier_head_activation: str = "linear",
    input_shape: tuple = (256, 256, 3),
    include_top: bool = True,  # Whether to include the classification layer in the model
    pretrained: bool = False,  # Whether to load pretrained weights
    cache_dir: Optional[str] = None,  # Local cache directory for weights
    updates: Optional[dict] = None,
    **kwargs,
):
    """
    Build a MobileViT-v1 classification model.

    Args:
        model_type (str): MobileViT version to create. Options: S, XS, XXS

        num_classes (int): Number of output classes

        classifier_head_activation (str): Activation function to use after the final dense layer in classification head. Default: "linear". Other options include: "softmax", "sigmoid", etc.

        input_shape (tuple): Input shape -> H, W, C

        include_top (bool): Whether to include the classification layers

        pretrained (bool): Whether to load pretrained weights

        cache_dir (str): Local directory to cache the downloaded weights

        updates (dict): a key-value pair indicating the changes to be made to the base model.

    Optional Args
    ---------
        linear_drop (float): Dropout rate used for MHSA output and Transformer Dense block
        attention_drop (float): Dropout rate for the attention matrix
        dropout (float): Additional Dropout rate used in Transformer Dense block.

    Returns:
        keras.Model: The constructed MobileViT-v1 model instance.

    Example
    -------
    >>> model = build_MobileViT_v1(
            model_type="S",
            num_classes=1000,
            input_shape=(256, 256, 3),
            include_top=True,
            pretrained=True,
        )
    >>> model.summary()
    """

    model_type = model_type.upper()
    if model_type not in ("S", "XS", "XXS"):
        raise ValueError("Bad Input. 'model_type' should be one of ['S', 'XS', 'XXS']")

    updated_configs = get_mobile_vit_v1_configs(model_type, updates=updates)

    # Build the base model
    model = MobileViT_v1(
        configs=updated_configs,
        num_classes=num_classes if include_top else None,
        classifier_head_activation=classifier_head_activation,
        input_shape=input_shape,
        model_name=f"MobileViT_v1-{model_type}",
        **kwargs,
    )

    # Initialize parameters of MobileViT block.
    if None in input_shape:
        dummy_input_shape = (1, 256, 256, 3)
    else:
        dummy_input_shape = (1,) + input_shape

    model(np.random.randn(*dummy_input_shape), training=False)

    if pretrained:
        weights_path = utils.get_file(
            fname=f"keras_MobileVIT_v1_model_{model_type}.weights.h5",
            origin=WEIGHTS_URL.format(model_type=model_type),
            cache_subdir="models",
            hash_algorithm="auto",
            extract=False,
            archive_format="auto",
            cache_dir=cache_dir,
        )

        # with warnings.catch_warnings():
        #     # Ignore UserWarnings within this block
        #     warnings.simplefilter("ignore", UserWarning)
        model.load_weights(weights_path, skip_mismatch=True)

    return model


if __name__ == "__main__":
    model = build_MobileViT_v1(
        model_type="S",  # "XS", "XXS"
        input_shape=(256, 256, 3),  # (None, None, 3)
        num_classes=1000,
        linear_drop=0.0,
        attention_drop=0.0,
    )

    print(f"{model.name} num. parametes: {model.count_params()}")
    # model.summary(positions=[0.33, 0.64, 0.75, 1.0])
    # model.save(f"{model.name}", include_optimizer=False)

    # # Refer to BaseConfigs class to see all customizable modules available.
    # updates = {
    #     "block_3_1_dims": 256,
    #     "block_3_2_dims": 384,
    #     "tf_block_3_dims": 164,
    #     "tf_block_3_repeats": 3,
    # }

    # model = build_MobileViT_v1(
    #     model_type="XXS",
    #     updates=updates,
    #     linear_drop=0.0,
    #     attention_drop=0.0,
    # )

    # # model.summary(positions=[0.33, 0.64, 0.75, 1.0])
    # print(f"{model.name} num. parametes: {model.count_params()}")
