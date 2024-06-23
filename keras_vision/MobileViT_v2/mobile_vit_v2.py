import warnings
from typing import Optional
import numpy as np
from keras import Model, Input
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.utils import get_file


from .configs import get_mobile_vit_v2_configs
from .base_layers import ConvLayer, InvertedResidualBlock
from .mobile_vit_v2_block import MobileViT_v2_Block

WEIGHTS_RELEASE_TAG_VERSION = 0.5
WEIGHTS_URL = "https://github.com/veb-101/keras-vision/releases/download/v{weight_release_tag}/{file_name}"


def MobileViT_v2(
    configs,
    linear_drop: float = 0.0,
    attention_drop: float = 0.0,
    dropout: float = 0.0,
    num_classes: int | None = 1000,
    input_shape: tuple[int, int, int] = (256, 256, 3),
    model_name: str = "MobileViT-v2-1.0",
):
    """
    Build the MobileViT-v2 model architecture.

    Parameters
    ----------
    configs : object
        A dataclass instance containing model information such as per-layer output channels, transformer embedding dimensions, transformer repeats, and IR expansion factor.

    linear_drop : float, optional
        Dropout rate for the Dense layers. Default is 0.0.

    attention_drop : float, optional
        Dropout rate for the attention matrix. Default is 0.0.

    dropout : float, optional
        Dropout rate to be applied between different layers. Default is 0.0.

    num_classes : int, optional
        The number of output classes for the classification task. If None, no classification layer is added. Default is 1000.

    input_shape : tuple of int, optional
        The shape of the input data in the format (height, width, channels). Default is (256, 256, 3).

    model_name : str, optional
        The name of the model. Default is "MobileViT-v3-1.0".

    Returns
    -------
    model : keras.Model
        The constructed MobileViT-v2 model instance.

    Example
    -------
    >>> configs = get_mobile_vit_v2_configs(width_multiplier=1.0)
    >>> model = MobileViT_v2(
    >>>     configs=configs,
    >>>     linear_drop=0.1,
    >>>     attention_drop=0.1,
    >>>     dropout=0.2,
    >>>     num_classes=1000,
    >>>     input_shape=(256, 256, 3),
    >>>     model_name="MobileViT-v2-1.0"
    >>> )
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
    pretrained_weight_name: str = "keras_mobilevitv2-im1k-256-0.5.weights.h5",
    **kwargs,
):
    """
    Build a MobileViT-v2 classification model.

    Parameters
    ----------
    width_multiplier : float, optional
        A multiplier for the width (number of channels) of the model layers. Default is 1.0,
        which corresponds to the base model.

    num_classes : int, optional
        The number of output classes for the classification task. Default is 1000.

    input_shape : tuple, optional
        The shape of the input data in the format (height, width, channels). Default is (256, 256, 3).

    include_top : bool, optional
        Whether to include the fully-connected layer at the top of the network. Default is True.

    pretrained : bool, optional
        Whether to load pretrained weights for the model. Default is False.

    cache_dir : str, optional
        Directory to cache the pretrained weights. Default is None.

    updates : dict, optional
        A dictionary of updates to modify the base model configuration. Default is None.

    pretrained_weight_name : str, optional
        The name of the file containing the pretrained weights. Default is "keras_mobilevitv2-im1k-256-0.5.weights.h5".

    **kwargs : dict, optional
        Additional keyword arguments for model customization. These can include:
        - linear_drop : float, optional
            Dropout rate for Dense layers.
        - attention_drop : float, optional
            Dropout rate for the attention matrix.
        - dropout : float, optional
            Dropout rate for in-between different layers

    Returns
    -------
    model : keras.Model
        The constructed MobileViT-v2 model instance.

    Example
    -------
    >>> model = build_MobileViT_v2(
    >>>     width_multiplier=1.0,
    >>>     num_classes=1000,
    >>>     input_shape=(256, 256, 3),
    >>>     include_top=True,
    >>>     pretrained=True,
    >>>     cache_dir='/path/to/cache'
    >>> )
    >>> model.summary()
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

    if pretrained:
        weights_path = get_file(
            fname=pretrained_weight_name,
            origin=WEIGHTS_URL.format(weight_release_tag=WEIGHTS_RELEASE_TAG_VERSION, file_name=pretrained_weight_name),
            cache_subdir="models",
            hash_algorithm="auto",
            extract=False,
            archive_format="auto",
            cache_dir=cache_dir,
        )

        with warnings.catch_warnings():
            # Ignore UserWarnings within this block
            warnings.simplefilter("ignore", UserWarning)
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
