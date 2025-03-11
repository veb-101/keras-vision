import os

# os.environ["KERAS_BACKEND"] = "tensorflow"
# os.environ["KERAS_BACKEND"] = "jax"
# os.environ["KERAS_BACKEND"] = "torch"

import warnings
from functools import partial
from typing import Tuple, List, Optional
from collections.abc import Callable

import keras
from keras import layers as keras_layer
# from keras import ops as kops
# import tensorflow as tf

from .rep_cpe import RepCPE
from .mobileone import MobileOneBlock
from .base_layers import basic_blocks, PatchEmbed, convolutional_stem


__all__ = [
    "build_fastvit",
    "fastvit_t8",
    "fastvit_t12",
    "fastvit_s12",
    "fastvit_sa12",
    "fastvit_sa24",
    "fastvit_sa36",
    "fastvit_ma36",
    "reparameterize_model",
]

WEIGHTS_URL = r"https://huggingface.co/veb-101/apple-fastvit-Keras-3/resolve/main/{weight_file_name}"


"""
# Unused so far

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 256, 256),
        "pool_size": None,
        "crop_pct": 0.95,
        "interpolation": "bicubic",
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "fastvit_t": _cfg(crop_pct=0.9),
    "fastvit_s": _cfg(crop_pct=0.9),
    "fastvit_m": _cfg(crop_pct=0.95),
}
"""


def build_fastvit(
    layers: List[int],
    token_mixers: Tuple[str],
    embed_dims: List[int] = None,
    mlp_ratios: List[int] = None,
    downsamples: List[int] = None,
    repmixer_kernel_size: int = 3,
    norm_layer: str = "BatchNormalization",
    act_layer: str = "gelu",
    pos_embs: List[Callable[[int, int], keras_layer.Layer] | None] = None,
    down_patch_size=7,
    down_stride=2,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
    fork_feat=False,
    cls_ratio: float = 2.0,
    inference_mode: bool = False,
    input_shape: Tuple[int | None] = (None, None, 3),
    num_classes: int = 1000,
    model_name: str = "fastvit",
) -> None:
    if not fork_feat:
        num_classes = num_classes

    # Convolutional stem
    patch_embed = convolutional_stem(
        in_channels=3,
        out_channels=embed_dims[0],
        inference_mode=inference_mode,
        name="patch_embed_conv_stem",
    )

    # Build the main stages of the network architecture
    network = []
    for i in range(len(layers)):
        # Add position embeddings if requested
        if pos_embs and pos_embs[i] is not None:
            network.append(
                pos_embs[i](
                    in_channels=embed_dims[i],
                    embed_dim=embed_dims[i],
                    inference_mode=inference_mode,
                    name=f"rep_cpe_{i}",
                )
            )

        stage = basic_blocks(
            embed_dims[i],
            i,
            layers,
            token_mixer_type=token_mixers[i],
            kernel_size=repmixer_kernel_size,
            mlp_ratio=mlp_ratios[i],
            act_layer=act_layer,
            norm_layer=norm_layer,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            use_layer_scale=use_layer_scale,
            layer_scale_init_value=layer_scale_init_value,
            inference_mode=inference_mode,
            # name=f"fastvit_stage_{i + 1}_basic_block_{token_mixers[i]}",
            name=f"fastvit_stage_{i + 1}_basic_block",
        )
        network.append(stage)
        if i >= len(layers) - 1:
            break

        # Patch merging/downsampling between stages.
        if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
            network.append(
                PatchEmbed(
                    patch_size=down_patch_size,
                    stride=down_stride,
                    in_channels=embed_dims[i],
                    embed_dim=embed_dims[i + 1],
                    inference_mode=inference_mode,
                    name=f"fastvit_stage_{i + 1}_patch_embed",
                )
            )

    # For segmentation and detection, extract intermediate output
    if fork_feat:
        # add a norm layer for each output
        out_indices = [0, 2, 4, 6]

        norm_layers = dict()

        for i_emb, i_layer in enumerate(out_indices):
            if i_emb == 0 and os.environ.get("FORK_LAST3", None):
                """For RetinaNet, `start_level=1`. The first norm layer will not used.
                cmd: `FORK_LAST3=1 python -m torch.distributed.launch ...`
                """
                layer = keras_layer.Identity()
            else:
                if norm_layer == "BatchNormalization":
                    layer = keras_layer.BatchNormalization()
            layer_name = f"norm{i_layer}"
            norm_layers[layer_name] = layer

    else:
        # Classifier head
        gap = keras_layer.GlobalAveragePooling2D()
        conv_exp = MobileOneBlock(
            in_channels=embed_dims[-1],
            out_channels=int(embed_dims[-1] * cls_ratio),
            kernel_size=3,
            stride=1,
            padding=1,
            groups=embed_dims[-1],
            inference_mode=inference_mode,
            use_se=True,
            num_conv_branches=1,
            name="final_mobileone",
        )
        if isinstance(num_classes, int) and num_classes > 0:
            kernel_initializer = keras.initializers.TruncatedNormal(stddev=0.02)
            bias_initializer = keras.initializers.Zeros()
            head = keras_layer.Dense(num_classes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, name="head")
        else:
            head = keras_layer.Identity(name="head")

    ##############################=- BUILDING MODEL -=##############################
    inputs = keras.Input(shape=input_shape)

    # embeddings
    x = patch_embed(inputs)
    # model = keras.Model(inputs=inputs, outputs=x)

    # tokens
    outs = []
    for idx, block in enumerate(network):
        x = block(x)
        if fork_feat and idx in out_indices:
            # norm_layer = getattr(self, f"norm{idx}")
            x_out = norm_layers[f"norm{idx}"](x)
            outs.append(x_out)

    if fork_feat:
        # output the features of four stages for dense prediction
        model = keras.Model(inputs=inputs, outputs=outs)
        return model

    x = conv_exp(x)
    x = gap(x)
    cls_out = head(x)
    # Create and return the model
    model = keras.Model(inputs=inputs, outputs=cls_out, name=model_name)
    return model


def _load_weights(
    model: keras.Model,
    model_weights_file_base_name: str,
    inference_mode: bool = False,
    load_kd_weights: bool = False,
    cache_dir: Optional[str] = None,  # Local cache directory for weights
):
    WEIGHT_FILE_SUFFIX = ".weights.h5"
    if load_kd_weights:
        if inference_mode:
            WEIGHT_FILE_SUFFIX = f"_reparam_kd{WEIGHT_FILE_SUFFIX}"
        else:
            WEIGHT_FILE_SUFFIX = f"_kd{WEIGHT_FILE_SUFFIX}"
    else:
        if inference_mode:
            WEIGHT_FILE_SUFFIX = f"_reparam{WEIGHT_FILE_SUFFIX}"

    model_weights_name = f"{model_weights_file_base_name}{WEIGHT_FILE_SUFFIX}"
    model_weights_url = WEIGHTS_URL.format(weight_file_name=model_weights_name)
    # print("inference_mode", inference_mode, "load_kd_weights", load_kd_weights)
    # print("model_weights_name", model_weights_name)
    # print("model_weights_url", model_weights_url)

    weights_path = keras.utils.get_file(
        fname=model_weights_name,
        origin=model_weights_url,
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


def fastvit_t8(
    pretrained=False,
    inference_mode=False,
    num_classes=1000,
    include_top: bool = True,
    input_shape=(None, None, 3),
    load_kd_weights: bool = False,
    cache_dir: Optional[str] = None,  # Local cache directory for weights
) -> keras.Model:
    """
    Instantiate FastViT-T8 model variant or feature extractor with optional pretrained weights.

    Params:
        pretrained: (bool) Whether to load pretrained weights

        inference_mode: (bool) Whether to load fused fastvit model

        num_classes: (int) Number of output classes

        include_top: (bool) Whether to include the classification layers

        input_shape: (tuple) Input shape -> H, W, C

        load_kd_weights: (bool) Load knowledge distillation trained model weights if `pretrained=True`.

        cache_dir: (str) Local directory to cache the downloaded weights

    """

    layers = [2, 2, 4, 2]
    embed_dims = [48, 96, 192, 384]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")

    # Build the model
    model = build_fastvit(
        layers=layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name="fastvit_t8",
        inference_mode=inference_mode,
    )
    if pretrained:
        model = _load_weights(
            model=model,
            model_weights_file_base_name="keras_fastvit_t8",
            load_kd_weights=load_kd_weights,
            cache_dir=cache_dir,
            inference_mode=inference_mode,
        )

    return model


def fastvit_t12(
    pretrained=False,
    inference_mode=False,
    num_classes=1000,
    include_top: bool = True,
    input_shape=(None, None, 3),
    load_kd_weights: bool = False,
    cache_dir: Optional[str] = None,  # Local cache directory for weights
) -> keras.Model:
    """
    Instantiate FastViT-T12 model variant or feature extractor with optional pretrained weights.

    Params:
        pretrained: (bool) Whether to load pretrained weights

        inference_mode: (bool) Whether to load fused fastvit model

        num_classes: (int) Number of output classes

        include_top: (bool) Whether to include the classification layers

        input_shape: (tuple) Input shape -> H, W, C

        load_kd_weights: (bool) Load knowledge distillation trained model weights if `pretrained=True`.

        cache_dir: (str) Local directory to cache the downloaded weights

    """

    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [3, 3, 3, 3]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")

    # Build the model
    model = build_fastvit(
        layers=layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name="fastvit_t12",
        inference_mode=inference_mode,
    )

    if pretrained:
        model = _load_weights(
            model=model,
            model_weights_file_base_name="keras_fastvit_t12",
            load_kd_weights=load_kd_weights,
            cache_dir=cache_dir,
            inference_mode=inference_mode,
        )
    return model


def fastvit_s12(
    pretrained=False,
    inference_mode=False,
    num_classes=1000,
    include_top: bool = True,
    input_shape=(None, None, 3),
    load_kd_weights: bool = False,
    cache_dir: Optional[str] = None,  # Local cache directory for weights
) -> keras.Model:
    """
    Instantiate FastViT-S12 model variant or feature extractor with optional pretrained weights.

    Params:
        pretrained: (bool) Whether to load pretrained weights

        inference_mode: (bool) Whether to load fused fastvit model

        num_classes: (int) Number of output classes

        include_top: (bool) Whether to include the classification layers

        input_shape: (tuple) Input shape -> H, W, C

        load_kd_weights: (bool) Load knowledge distillation trained model weights if `pretrained=True`.

        cache_dir: (str) Local directory to cache the downloaded weights

    """

    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    token_mixers = ("repmixer", "repmixer", "repmixer", "repmixer")

    # Build the model
    model = build_fastvit(
        layers=layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name="fastvit_t12",
        inference_mode=inference_mode,
    )
    if pretrained:
        model = _load_weights(
            model=model,
            model_weights_file_base_name="keras_fastvit_s12",
            load_kd_weights=load_kd_weights,
            cache_dir=cache_dir,
            inference_mode=inference_mode,
        )

    return model


def fastvit_sa12(
    pretrained=False,
    inference_mode=False,
    num_classes=1000,
    include_top: bool = True,
    input_shape=(None, None, 3),
    load_kd_weights: bool = False,
    cache_dir: Optional[str] = None,  # Local cache directory for weights
) -> keras.Model:
    """
    Instantiate FastViT-SA12 model variant or feature extractor with optional pretrained weights.

    Params:
        pretrained: (bool) Whether to load pretrained weights

        inference_mode: (bool) Whether to load fused fastvit model

        num_classes: (int) Number of output classes

        include_top: (bool) Whether to include the classification layers

        input_shape: (tuple) Input shape -> H, W, C

        load_kd_weights: (bool) Load knowledge distillation trained model weights if `pretrained=True`.

        cache_dir: (str) Local directory to cache the downloaded weights

    """

    layers = [2, 2, 6, 2]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")

    # Build the model
    model = build_fastvit(
        layers=layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name="fastvit_sa12",
        inference_mode=inference_mode,
    )

    if pretrained:
        model = _load_weights(
            model=model,
            model_weights_file_base_name="keras_fastvit_sa12",
            load_kd_weights=load_kd_weights,
            cache_dir=cache_dir,
            inference_mode=inference_mode,
        )

    return model


def fastvit_sa24(
    pretrained=False,
    inference_mode=False,
    num_classes=1000,
    include_top: bool = True,
    input_shape=(None, None, 3),
    load_kd_weights: bool = False,
    cache_dir: Optional[str] = None,  # Local cache directory for weights
) -> keras.Model:
    """
    Instantiate FastViT-SA24 model variant or feature extractor with optional pretrained weights.

    Params:
        pretrained: (bool) Whether to load pretrained weights

        inference_mode: (bool) Whether to load fused fastvit model

        num_classes: (int) Number of output classes

        include_top: (bool) Whether to include the classification layers

        input_shape: (tuple) Input shape -> H, W, C

        load_kd_weights: (bool) Load knowledge distillation trained model weights if `pretrained=True`.

        cache_dir: (str) Local directory to cache the downloaded weights

    """

    layers = [4, 4, 12, 4]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")

    # Build the model
    model = build_fastvit(
        layers=layers,
        token_mixers=token_mixers,
        embed_dims=embed_dims,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name="fastvit_sa24",
        inference_mode=inference_mode,
    )
    if pretrained:
        model = _load_weights(
            model=model,
            model_weights_file_base_name="keras_fastvit_sa24",
            load_kd_weights=load_kd_weights,
            cache_dir=cache_dir,
            inference_mode=inference_mode,
        )
    return model


def fastvit_sa36(
    pretrained=False,
    inference_mode=False,
    num_classes=1000,
    include_top: bool = True,
    input_shape=(None, None, 3),
    load_kd_weights: bool = False,
    cache_dir: Optional[str] = None,  # Local cache directory for weights
) -> keras.Model:
    """
    Instantiate FastViT-SA36 model variant or feature extractor with optional pretrained weights.

    Params:
        pretrained: (bool) Whether to load pretrained weights

        inference_mode: (bool) Whether to load fused fastvit model

        num_classes: (int) Number of output classes

        include_top: (bool) Whether to include the classification layers

        input_shape: (tuple) Input shape -> H, W, C

        load_kd_weights: (bool) Load knowledge distillation trained model weights if `pretrained=True`.

        cache_dir: (str) Local directory to cache the downloaded weights

    """

    layers = [6, 6, 18, 6]
    embed_dims = [64, 128, 256, 512]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")

    # Build the model
    model = build_fastvit(
        layers=layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name="fastvit_sa36",
        inference_mode=inference_mode,
    )
    if pretrained:
        model = _load_weights(
            model=model,
            model_weights_file_base_name="keras_fastvit_sa36",
            load_kd_weights=load_kd_weights,
            cache_dir=cache_dir,
            inference_mode=inference_mode,
        )

    return model


def fastvit_ma36(
    pretrained=False,
    inference_mode=False,
    num_classes=1000,
    include_top: bool = True,
    input_shape=(None, None, 3),
    load_kd_weights: bool = False,
    cache_dir: Optional[str] = None,  # Local cache directory for weights
) -> keras.Model:
    """
    Instantiate FastViT-MA36 model variant or feature extractor with optional pretrained weights.

    Params:
        pretrained: (bool) Whether to load pretrained weights

        inference_mode: (bool) Whether to load fused fastvit model

        num_classes: (int) Number of output classes

        include_top: (bool) Whether to include the classification layers

        input_shape: (tuple) Input shape -> H, W, C

        load_kd_weights: (bool) Load knowledge distillation trained model weights if `pretrained=True`.

        cache_dir: (str) Local directory to cache the downloaded weights

    """

    layers = [6, 6, 18, 6]
    embed_dims = [76, 152, 304, 608]
    mlp_ratios = [4, 4, 4, 4]
    downsamples = [True, True, True, True]
    pos_embs = [None, None, None, partial(RepCPE, spatial_shape=(7, 7))]
    token_mixers = ("repmixer", "repmixer", "repmixer", "attention")

    # Build the model
    model = build_fastvit(
        layers=layers,
        embed_dims=embed_dims,
        token_mixers=token_mixers,
        pos_embs=pos_embs,
        mlp_ratios=mlp_ratios,
        downsamples=downsamples,
        layer_scale_init_value=1e-6,
        num_classes=num_classes if include_top else None,
        input_shape=input_shape,
        model_name="fastvit_ma36",
        inference_mode=inference_mode,
    )

    if pretrained:
        model = _load_weights(
            model=model,
            model_weights_file_base_name="keras_fastvit_ma36",
            load_kd_weights=load_kd_weights,
            cache_dir=cache_dir,
            inference_mode=inference_mode,
        )
    return model


def recursively_call_reparameterize(layer):
    """
    Recursively calls 'reparameterize' on `layer` if it exists, and
    does the same for any sub-layers if they exist.

    """

    if hasattr(layer, "reparameterize"):
        # print(layer.name)
        layer.reparameterize()

    # If this layer has sub-layers, recurse on each of them.
    if hasattr(layer, "layers"):
        for sublayer in layer.layers:
            recursively_call_reparameterize(sublayer)
    elif hasattr(layer, "_layers"):
        for sublayer in layer._layers:
            recursively_call_reparameterize(sublayer)


def reparameterize_model(model):
    """
    Call this once on the top-level Keras model to traverse
    and call 'reparameterize' on all necessary sub-layers.
    """
    for layer in model.layers:
        # print("outer layer", layer.name)
        recursively_call_reparameterize(layer)

    return model


if __name__ == "__main__":
    import numpy as np

    image_test = np.random.randn(1, 256, 256, 3)

    model = fastvit_ma36(inference_mode=False, pretrained=True, num_classes=10)
    model.summary()

    # model = fastvit_ma36(inference_mode=True, pretrained=True, num_classes=10)
    # model.summary()

    # model = fastvit_ma36(inference_mode=False, pretrained=True, num_classes=10, load_kd_weights=True)
    # model.summary()

    # model = fastvit_ma36(inference_mode=True, pretrained=True, num_classes=10, load_kd_weights=True)
    # model.summary()

    # model_out = model.predict(image_test)
    # print(kops.round(kops.sum(model_out), decimals=4))

    # cloned_model = keras.models.clone_model(model)
    # cloned_model.set_weights(model.get_weights())
    # cloned_model = reparameterize_model(cloned_model)
    # cloned_model.summary()
    # cloned_model_out = cloned_model.predict(image_test)

    # print(kops.round(kops.sum(cloned_model_out), decimals=4))
