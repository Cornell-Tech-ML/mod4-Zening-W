from typing import Tuple

from . import operators
from .autodiff import Context
from .fast_ops import FastOps
from .tensor import Tensor
from .tensor_functions import Function, rand, tensor


# List of functions in this file:
# - avgpool2d: Tiled average pooling 2D
# - argmax: Compute the argmax as a 1-hot tensor
# - Max: New Function for max operator
# - max: Apply max reduction
# - softmax: Compute the softmax as a tensor
# - logsoftmax: Compute the log of the softmax as a tensor - See https://en.wikipedia.org/wiki/LogSumExp#log-sum-exp_trick_for_log-domain_calculations
# - maxpool2d: Tiled max pooling 2D
# - dropout: Dropout positions based on random noise, include an argument to turn off


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """Reshape an image tensor for 2D pooling

    Args:
    ----
        input: batch x channel x height x width
        kernel: height x width of pooling

    Returns:
    -------
        Tensor of size batch x channel x new_height x new_width x (kernel_height * kernel_width) as well as the new_height and new_width value.

    """
    batch, channel, height, width = input.shape
    kh, kw = kernel
    assert height % kh == 0
    assert width % kw == 0
    new_height = height // kh
    new_width = width // kw
    # Reshape the input to split the height and width into patch blocks
    # (batch, channel, height, width)
    # -> (batch, channel, new_height, kh, new_width, kw)
    input = input.contiguous()
    reshaped = input.view(batch, channel, new_height, kh, new_width, kw)

    # Rearrange the dimensions to group the kh and kw together
    # (batch, channel, new_height, new_width, kh, kw)
    # -> (batch, channel, new_height, new_width, kh*kw)
    tiled = reshaped.permute(0, 1, 2, 4, 3, 5).contiguous()
    tiled = tiled.view(batch, channel, new_height, new_width, kh * kw)

    return tiled, new_height, new_width


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D average pooling over an input tensor."""
    # Unpack input dimensions and kernel size
    batch, channels, _, _ = input.shape
    kh, kw = kernel

    # Use the 'tile' function to reshape input into patches of size (kh * kw)
    # 'tiled_input' will have shape: (batch, channel, new_height, new_width, kh*kw)
    tiled_input, out_height, out_width = tile(input, kernel)

    # Compute the mean across the last dimension, which corresponds to the kernel elements
    # Resulting shape: (batch, channel, out_height, out_width)
    averaged = tiled_input.mean(dim=-1).contiguous()

    # Reshape is unnecessary here because mean has already collapsed the last dimension.
    # However, if we explicitly want to ensure the final shape, we can do:
    averaged = averaged.view(batch, channels, out_height, out_width)

    return averaged


fast_max = FastOps.reduce(operators.max, -float("inf"))


def argmax(input: Tensor, dim: int) -> Tensor:
    """Returns a boolean mask indicating the positions of the maximum values along the specified dimension of the input tensor."""
    # Compute the maximum values along the specified dimension.
    max_values = fast_max(input, dim)

    # Compare each element in `input` with the corresponding max value.
    # This produces a boolean mask that is True where input == max_values, False elsewhere.
    return input == max_values


class Max(Function):
    @staticmethod
    def forward(ctx: Context, input_tensor: Tensor, dimension: Tensor) -> Tensor:
        """Forward pass of the max operation."""
        # Extract the dimension value from the dimension tensor
        dim_val = int(dimension.item())

        # Save necessary variables for backward computation
        ctx.save_for_backward(input_tensor, dim_val)

        # Compute and return the maximum along the given dimension
        return fast_max(input_tensor, dim_val)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Backward pass of the max operation."""
        # Retrieve saved values
        input_tensor, dim_val = ctx.saved_values

        # Identify positions of the max values along the specified dimension
        max_positions = argmax(input_tensor, dim_val)

        # Distribute grad_output only to those max positions
        grad_input = max_positions * grad_output

        return grad_input, 0.0


def max(input: Tensor, dim: int) -> Tensor:
    """Apply max reduction on one dimension"""
    return Max.apply(input, tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """Compute the softmax along a specified dimension of the input tensor."""
    # Identify the maximum value along the specified dimension to stabilize numerics
    max_along_dim = max(input, dim)

    # Shift the input by subtracting the max, then exponentiate
    shifted = input - max_along_dim
    exponentials = shifted.exp()

    # Compute the sum of exponentials along the same dimension
    exp_sum = exponentials.sum(dim)

    # Divide each element by the sum of exponentials to get probabilities
    probabilities = exponentials / exp_sum

    return probabilities


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """Compute the log of the softmax along a specified dimension."""
    # Identify max values along the dimension to stabilize the exponentials
    max_values = max(input, dim)

    # Shift input by subtracting max_values to avoid overflow
    shifted = input - max_values
    # Compute the log-sum-exp term
    lse = (shifted.exp().sum(dim)).log() + max_values

    # Finally, logsoftmax is input minus the log-sum-exp
    return input - lse


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Apply 2D max pooling over an input tensor."""
    # Unpack input shape and the kernel size
    batch, channels, _, _ = input.shape
    kh, kw = kernel

    # Use tile to reshape input into patches
    # tiled: (batch, channel, new_height, new_width, kh*kw)
    tiled, new_height, new_width = tile(input, (kh, kw))

    # Compute maximum along the last dimension (the kernel dimension)
    max_pooled = max(tiled, dim=4).contiguous()

    # Reshape back to (batch, channel, new_height, new_width)
    result = max_pooled.view(batch, channels, new_height, new_width)
    return result


def dropout(input: Tensor, p: float = 0.5, ignore: bool = False) -> Tensor:
    """Randomly drops out elements of the input tensor with a specified probability."""
    if ignore or p <= 0.0:
        return input
    # If dropout probability is 1, drop all elements and return a tensor of zeros
    if p >= 1.0:
        return input.zeros(input.shape)

    # Generate a random mask where elements are kept with probability (1 - p)
    mask = rand(input.shape) > p

    # Apply the mask to the input and return the result
    return input * mask
