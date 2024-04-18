from typing import Union, Optional


# https://www.tensorflow.org/guide/mixed_precision#ensuring_gpu_tensor_cores_are_used
def make_divisible(v: Union[int, float], divisor: Optional[Union[int, float]] = 8, min_value: Optional[Union[int, float]] = None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
