import os

from functools import partial
from typing import Tuple

import keras
from keras import layers as keras_layer


from .rep_cpe import RepCPE
from .mobileone import MobileOneBlock
from .base_layers import basic_blocks, PatchEmbed, convolutional_stem

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def build_fastvit(
    layers,
    token_mixers: Tuple[str, ...],
    embed_dims=None,
    mlp_ratios=None,
    downsamples=None,
    repmixer_kernel_size=3,
    norm_layer: str = "BatchNormalization",
    act_layer: str = "gelu",
    num_classes=1000,
    pos_embs=None,
    down_patch_size=7,
    down_stride=2,
    drop_rate=0.0,
    drop_path_rate=0.0,
    use_layer_scale=True,
    layer_scale_init_value=1e-5,
    fork_feat=False,
    init_cfg=None,
    pretrained=None,
    cls_ratio=2.0,
    inference_mode=False,
    input_shape=(None, None, 3),
) -> None:
    # Define the input layer

    if not fork_feat:
        num_classes = num_classes

    # Convolutional stem
    patch_embed = convolutional_stem(
        in_channels=3,
        out_channels=embed_dims[0],
        inference_mode=inference_mode,
    )

    # Build the main stages of the network architecture
    network = []
    for i in range(len(layers)):
        # Add position embeddings if requested
        if pos_embs and pos_embs[i] is not None:
            network.append(pos_embs[i](embed_dims[i], embed_dims[i], inference_mode=inference_mode))

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
        )
        if num_classes > 0:
            kernel_initializer = keras.initializers.TruncatedNormal(stddev=0.02)
            bias_initializer = keras.initializers.Zeros()
            head = keras_layer.Dense(num_classes, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        else:
            head = keras_layer.Identity()

    ##############################=- BUILDING MODEL -=##############################
    inputs = keras.Input(shape=input_shape)

    # embeddings
    x = patch_embed(inputs)

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
    model = keras.Model(inputs=inputs, outputs=cls_out)
    return model


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


def fastvit_t8(pretrained=False, **kwargs) -> keras.Model:
    """Instantiate FastViT-T8 model variant."""
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
        input_shape=(None, None, 3),
    )

    return model


def fastvit_t12(pretrained=False, **kwargs) -> keras.Model:
    """Instantiate FastViT-T12 model variant."""
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
        input_shape=(None, None, 3),
    )

    return model


def fastvit_s12(pretrained=False, **kwargs) -> keras.Model:
    """Instantiate FastViT-S12 model variant."""
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
    )

    return model


def fastvit_sa12(pretrained=False, **kwargs) -> keras.Model:
    """Instantiate FastViT-SA12 model variant."""
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
    )

    return model


def fastvit_sa24(pretrained=False, **kwargs) -> keras.Model:
    """Instantiate FastViT-SA24 model variant."""
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
    )

    return model


def fastvit_sa36(pretrained=False, **kwargs) -> keras.Model:
    """Instantiate FastViT-SA36 model variant."""
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
    )

    return model


def fastvit_ma36(pretrained=False, **kwargs) -> keras.Model:
    """Instantiate FastViT-MA36 model variant."""
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
    )

    return model


def recursively_call_reparameterize(layer):
    """
    Recursively calls 'reparameterize' on `layer` if it exists, and
    does the same for any sub-layers if they exist.

    """
    if hasattr(layer, "layers"):
        print(layer.layers)
    elif hasattr(layer, "_layers"):
        print(layer._layers)

    if hasattr(layer, "reparameterize"):
        print(layer.name)
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
        print("outer layer", layer.name)
        recursively_call_reparameterize(layer)

    return model


if __name__ == "__main__":
    import numpy as np

    model = fastvit_t8()
    model.summary(expand_nested=True)

    cloned_model = keras.models.clone_model(model)
    cloned_model.set_weights(model.get_weights())
    cloned_model = reparameterize_model(cloned_model)

    # # model = fastvit_ma36()
    cloned_model.summary()
    # out = model(np.random.randn(2, 256, 256, 3))
    # print(out.shape)
