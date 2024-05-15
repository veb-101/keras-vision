[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/keras-vision)](https://www.python.org/)  [![PyPI version](https://badge.fury.io/py/keras-vision.svg)](https://badge.fury.io/py/keras-vision) [![Keras](https://img.shields.io/badge/Keras%203.x-%23D00000.svg?logo=Keras&logoColor=white)](https://github.com/keras-team/keras/releases) ![PyPI - Downloads](https://img.shields.io/pypi/dm/keras-vision?style=plastic&logo=Keras&logoColor=red&link=https%3A%2F%2Fpypi.org%2Fproject%2Fkeras-vision%2F)


Porting all models from everywhere to Keras to leverage multi-backend support.

Cause why not?🤷🏻‍♂️

# Table of Contents

1. [Updates](#updates)
2. [Quick Setup](#quick-setup)
   - [Stable PyPi Package](#stable-pypi-package)
   - [Latest Git Updates](#latest-git-updates)
3. [Models Supported](#models-supported)


## Updates

1. [2024-05-15] Fixed MobileViT v1 - Now works will all 3 backends. 🎉🎉
2. [2024-05-04] Converted MobileViT to Keras 3 and released weights of all 3 variants.
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
      </tr>
   </tbody>
</table>
