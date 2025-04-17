import warnings
from typing import Optional
import numpy as np
from keras import Model, Input
from keras import layers as keras_layer
from keras import utils


from .configs import get_mobile_vit_v2_configs
from .base_layers import ConvLayer, InvertedResidualBlock
from .mobile_vit_v2_block import MobileViT_v2_Block

WEIGHTS_URL = r"https://huggingface.co/veb-101/Keras-3-apple-mobilevit/resolve/main/keras-3-mobilevit-v2-weights/{file_name}"


def MobileViT_v2(
    configs,
    linear_drop: float = 0.0,
    attention_drop: float = 0.0,
    dropout: float = 0.0,
    num_classes: int | None = 1000,
    classifier_head_activation: str = "linear",  # linear
    input_shape: tuple[int, int, int] = (256, 256, 3),
    model_name: str = "MobileViT-v2-1.0",
):
    """
    Build the MobileViT-v2 model architecture.

    Args:
        configs (dataclass): A dataclass instance containing model information such as per-layer output channels, transformer embedding dimensions, transformer repeats, and IR expansion factor.

        linear_drop (float): Dropout rate for the Dense layers. Default is 0.0.

        attention_drop (float): Dropout rate for the attention matrix. Default is 0.0.

        dropout (float): Dropout rate to be applied between different layers. Default is 0.0.

        num_classes (int): The number of output classes for the classification task. If None, no classification layer is added. Default is 1000.

        classifier_head_activation (str): Activation function to use after the final dense layer in classification head. Default: "linear". Other options include: "softmax", "sigmoid", etc.

        input_shape (tuple): The shape of the input data in the format (height, width, channels). Default is (256, 256, 3).

        model_name (str): The name of the model. Default is "MobileViT-v2-1.0".

    Returns:
        keras.Model: The constructed MobileViT-v2 model instance.

    Example
    -------
    >>> configs = get_mobile_vit_v2_configs(width_multiplier=1.0)
    >>> model = MobileViT_v2(
           configs=configs,
           linear_drop=0.1,
           attention_drop=0.1,
           dropout=0.2,
           num_classes=1000,
           input_shape=(256, 256, 3),
           model_name="MobileViT-v2-1.0"
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
        out = keras_layer.GlobalAveragePooling2D()(out)

        if linear_drop > 0.0:
            out = keras_layer.Dropout(rate=dropout)(out)

        out = keras_layer.Dense(units=num_classes)(out)
        out = keras_layer.Activation(activation=classifier_head_activation, name=f"{classifier_head_activation}")(out)

    model = Model(inputs=input_layer, outputs=out, name=model_name)

    return model


def build_MobileViT_v2(
    width_multiplier: float = 1.0,
    num_classes: int = 1000,
    classifier_head_activation: str = "linear",
    input_shape: tuple = (256, 256, 3),
    include_top: bool = True,  # Whether to include the classification layer in the model
    pretrained: bool = False,  # Whether to load pretrained weights
    cache_dir: Optional[str] = None,  # Local cache directory for weights
    updates: Optional[dict] = None,
    pretrained_weight_name: str = "keras_mobilevitv2-im1k-256-0.5.weights.h5",
    **kwargs,
):
    """
    Build a MobileViT-v2 classification model.

    Args:
        width_multiplier (float): A multiplier for the width (number of channels) of the model layers. Default is 1.0, which corresponds to the base model.

        num_classes (int): Number of output classes

        classifier_head_activation (str): Activation function to use after the final dense layer in classification head. Default: "linear". Other options include: "softmax", "sigmoid", etc.

        input_shape (tuple): Input shape -> H, W, C

        include_top (bool): Whether to include the classification layers

        pretrained (bool): Whether to load pretrained weights

        cache_dir (str): Local directory to cache the downloaded weights

        updates (dict): a key-value pair indicating the changes to be made to the base model.

        pretrained_weight_name (str): The name of the file containing the pretrained weights. Default is "keras_mobilevitv2-im1k-256-0.5.weights.h5".

    Optional Args
    ---------
        linear_drop (float): Dropout rate for Dense layers
        attention_drop (float): Dropout rate for the attention matrix
        dropout (float): Dropout rate for in-between different layers

    Returns:
        keras.Model: The constructed MobileViT-v2 model instance.

    Example
    -------
    >>> model = build_MobileViT_v2(
            width_multiplier=1.0,
            num_classes=1000,
            input_shape=(256, 256, 3),
            include_top=True,
            pretrained=True,
        )
    >>> model.summary()
    """

    updated_configs = get_mobile_vit_v2_configs(width_multiplier, updates=updates)

    # Build the base model
    model = MobileViT_v2(
        configs=updated_configs,
        num_classes=num_classes if include_top else None,
        classifier_head_activation=classifier_head_activation,
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

    if pretrained:
        weights_path = utils.get_file(
            fname=pretrained_weight_name,
            origin=WEIGHTS_URL.format(file_name=pretrained_weight_name),
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
    model = build_MobileViT_v2(
        width_multiplier=0.75,
        input_shape=(None, None, 3),
        num_classes=1000,
        linear_drop=0.0,
        attention_drop=0.0,
    )

    print(f"{model.name} num. parametes: {model.count_params()}")
