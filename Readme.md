[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/keras-vision)](https://www.python.org/)  [![PyPI version](https://badge.fury.io/py/keras-vision.svg)](https://badge.fury.io/py/keras-vision) [![Keras](https://img.shields.io/badge/Keras%203.x-%23D00000.svg?logo=Keras&logoColor=white)](https://github.com/keras-team/keras/releases) ![PyPI - Downloads](https://img.shields.io/pypi/dm/keras-vision?style=plastic&logo=Keras&logoColor=red&link=https%3A%2F%2Fpypi.org%2Fproject%2Fkeras-vision%2F)

Porting all models from everywhere to Keras to leverage multi-backend support.

Cause why not?🤷🏻‍♂️

# Table of Contents

- [Table of Contents](#table-of-contents)
  - [Progress](#progress)
  - [Updates](#updates)
  - [Quick Setup](#quick-setup)
    - [Stable PyPi Package](#stable-pypi-package)
    - [Latest Git Updates](#latest-git-updates)
  - [Models Supported](#models-supported)

## Progress

- Working on adding MobileNet V4

## Updates

1. [2025-03-12] Finished adding **FastViT** image classification model by Apple added. The model weights are available at [url](https://huggingface.co/veb-101/apple-fastvit-Keras-3/tree/main).
2. [2025-02-22] Working (v1) FastViT classification code added.
3. [2024-06-24] Released MobileViT v2 - All Image Classification variants.
4. [2024-05-15] Fixed MobileViT v1 - Now works will all 3 backends. 🎉🎉
5. [2024-05-04] Converted MobileViT to Keras 3 and released weights of all 3 variants.
   1. Jax backend currently not working, I'm working on a fix.
   2. Release: <https://github.com/veb-101/keras-vision/releases/tag/v0.4>

## Quick Setup

### Stable PyPi Package

```bash
pip install -U keras-vision
```

### OR

### Latest Git Updates

```bash
pip install git+https://github.com/veb-101/keras-vision.git
```

## Models Supported

<table>
   <thead>
      <tr>
         <th style="text-align:center">
            <strong># No.</strong>
         </th>
         <th style="text-align:center">
            <strong>Models</strong>
         </th>
         <th style="text-align:center">
            <strong>Paper</strong>
         </th>
         <th style="text-align:center">
            <strong>Additional Materials</strong>
         </th>
         <th style="text-align:center">
            <strong>Example notebook</strong>
         </th>
         <th style="text-align:center">
            <strong>Weights URL</strong>
         </th>
      </tr>
   </thead>
   <tbody>
      <tr>
         <td style="text-align:center">1</td>
         <td style="text-align:center">
            <a href="https://github.com/veb-101/keras-vision/blob/main/keras_vision/MobileViT_v1/mobile_vit_v1.py">MobileViT-V1</a>
         </td>
         <td style="text-align:center">
            <a href="https://arxiv.org/abs/2110.02178">MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer</a>
         </td>
         <td style="text-align:center">
            <a href="https://learnopencv.com/mobilevit-keras-3/">Blogpost: Building MobileViT In Keras 3</a>
         </td>
         <td style="text-align:center">
            <a href="https://colab.research.google.com/github/veb-101/keras-vision/blob/main/examples/mobile_vit_v1.ipynb">Colab link</a>
         </td>
         <td style="text-align:center">
            <a href="https://github.com/veb-101/keras-vision/releases/tag/v0.4">Releases v0.4</a>
         </td>
      <tr>
         <td style="text-align:center">2</td>
         <td style="text-align:center">
            <a href="https://github.com/veb-101/keras-vision/blob/main/keras_vision/MobileViT_v2/mobile_vit_v2.py">MobileViT-V2</a>
         </td>
         <td style="text-align:center">
            <a href="https://arxiv.org/abs/2206.02680">Separable Self-attention for Mobile Vision Transformers</a>
         </td>
         <td style="text-align:center">
            --
         </td>
         <td style="text-align:center">
            <a href="https://colab.research.google.com/github/veb-101/keras-vision/blob/main/examples/mobile_vit_v2.ipynb">Colab link</a>
         </td>
         <td style="text-align:center">
            <a href="https://github.com/veb-101/keras-vision/releases/tag/v0.5">Releases v0.5</a>
         </td>
      </tr>
      <tr>
         <td style="text-align:center">2</td>
         <td style="text-align:center">
            <a href="https://github.com/veb-101/keras-vision/blob/main/keras_vision/fastvit/fastvit.py">FastViT</a>
         </td>
         <td style="text-align:center">
            <a href="https://arxiv.org/abs/2206.02680">FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization</a>
         </td>
         <td style="text-align:center">
            --
         </td>
         <td style="text-align:center">
            <a href="https://colab.research.google.com/github/veb-101/keras-vision/blob/main/examples/fastvit.ipynb">Colab link</a>
         </td>
         <td style="text-align:center">
            <a href="https://huggingface.co/veb-101/apple-fastvit-Keras-3/tree/main">🤗</a>
         </td>
      </tr>
   </tbody>
</table>
