import torch
import torch.nn.functional as F
from typing import Tuple, Optional, Literal
import numpy as np
from torch.utils._pytree import tree_flatten, tree_unflatten
from typing import overload, Callable, Iterable, List, TypeVar, Any, Literal, Sequence, Optional
from functools import partial
import math
import scipy.linalg
#https://github.com/i404788/s5-pytorch

"""
Jax-Pytorch ported functions, mostly interfaces are kept the same but unsupported features are removed:
* Jax-Keyed RNGs are sampled from global RNG
* Canonical/Named shapes/dtypes/etc are now regular shapes,dtypes
"""

T = TypeVar("T")
T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")


@overload
def safe_map(f: Callable[[T1], T], __arg1: Iterable[T1]) -> List[T]: ...


@overload
def safe_map(f: Callable[[T1, T2], T], __arg1: Iterable[T1], __arg2: Iterable[T2]) -> List[T]: ...


@overload
def safe_map(f: Callable[[T1, T2, T3], T], __arg1: Iterable[T1], __arg2: Iterable[T2], __arg3: Iterable[T3]) -> List[T]: ...


@overload
def safe_map(f: Callable[..., T], __arg1: Iterable[Any], __arg2: Iterable[Any], __arg3: Iterable[Any], __arg4: Iterable[Any], *args) -> List[T]: ...


def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))

def make_HiPPO(N):
    """ Create a HiPPO-LegS matrix.
        From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
        Args:
            N (int32): state size
        Returns:
            N x N HiPPO LegS matrix
    """
    P = np.sqrt(1 + 2 * np.arange(N))
    A = P[:, np.newaxis] * P[np.newaxis, :]
    A = np.tril(A) - np.diag(np.arange(N))
    return -A

def make_NPLR_HiPPO(N):
    """
    Makes components needed for NPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        N (int32): state size
    Returns:
        N x N HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B
    """
    # Make -HiPPO
    hippo = make_HiPPO(N)

    # Add in a rank 1 term. Makes it Normal.
    P = np.sqrt(np.arange(N) + 0.5)

    # HiPPO also specifies the B matrix
    B = np.sqrt(2 * np.arange(N) + 1.0)
    return hippo, P, B

def make_DPLR_HiPPO(N):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        N:
    Returns:
        eigenvalues Lambda, low-rank term P, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation
    """
    A, P, B = make_NPLR_HiPPO(N)

    S = A + P[:, np.newaxis] * P[np.newaxis, :]

    S_diag = np.diagonal(S)
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    # Diagonalize S to V \Lambda V^*
    Lambda_imag, V = np.linalg.eigh(S * -1j)

    P = V.conj().T @ P
    B_orig = B
    B = V.conj().T @ B
    return Lambda_real + 1j * Lambda_imag, P, B, V, B_orig

def make_Normal_S(N):
    nhippo = make_HiPPO(N)
    # Add in a rank 1 term. Makes it Normal.
    p = 0.5 * np.sqrt(2 * np.arange(1, N + 1) + 1.0)
    q = 2 * p
    S = nhippo + p[:, np.newaxis] * q[np.newaxis, :]
    return S


def make_Normal_HiPPO(N, B=1):
    """Create a normal approximation to HiPPO-LegS matrix.
    For HiPPO matrix A, A=S+pqT is normal plus low-rank for
    a certain normal matrix S and low rank terms p and q.
    We are going to approximate the HiPPO matrix with the normal matrix S.
    Note we use original numpy instead of jax.numpy first to use the
    onp.linalg.eig function. This is because Jax's linalg.eig function does not run
    on GPU for non-symmetric matrices. This creates tracing issues.
    So we instead use onp.linalg eig and then cast to a jax array
    (since we only have to do this once in the beginning to initialize).
    Args:
        N (int32): state size
        B (int32): diagonal blocks
    Returns:
        Lambda (complex64): eigenvalues of S (N,)
        V      (complex64): eigenvectors of S (N,N)
    """

    assert N % B == 0, "N must divide blocks"
    S = (make_Normal_S(N // B),) * B
    S = scipy.linalg.block_diag(*S)

    # Diagonalize S to V \Lambda V^*
    Lambda, V = np.linalg.eig(S)

    # Convert to jax array
    return torch.tensor(Lambda), torch.tensor(V)


def log_step_initializer(dt_min=0.001, dt_max=0.1):
    """ Initialize the learnable timescale Delta by sampling
         uniformly between dt_min and dt_max.
         Args:
             dt_min (float32): minimum value
             dt_max (float32): maximum value
         Returns:
             init function
     """
    def init(shape):
        """ Init function
             Args:
                 key: jax random key
                 shape tuple: desired shape
             Returns:
                 sampled log_step (float32)
         """
        return uniform(shape, minval=np.log(dt_min), maxval=np.log(dt_max))
        # return torch.rand(shape) * (np.log(dt_max) - np.log(dt_min)) + np.log(dt_min)

    return init


def init_log_steps(H, dt_min, dt_max):
    """ Initialize an array of learnable timescale parameters
         Args:
             key: jax random key
             input: tuple containing the array shape H and
                    dt_min and dt_max
         Returns:
             initialized array of timescales (float32): (H,)
     """
    log_steps = []
    for i in range(H):
        log_step = log_step_initializer(dt_min=dt_min, dt_max=dt_max)(shape=(1,))
        log_steps.append(log_step)

    return torch.tensor(log_steps)


def init_VinvB(init_fun, Vinv):
    """ Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             shape (tuple): desired shape  (P,H)
             Vinv: (complex64)     the inverse eigenvectors used for initialization
         Returns:
             B_tilde (complex64) of shape (P,H,2)
     """
    def init(shape, dtype):
        B = init_fun(shape, dtype)
        VinvB = Vinv @ B.type(Vinv.dtype)
        VinvB_real = VinvB.real
        VinvB_imag = VinvB.imag
        return torch.cat((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)
    return init


def trunc_standard_normal(shape):
    """ Sample C with a truncated normal distribution with standard deviation 1.
         Args:
             key: jax random key
             shape (tuple): desired shape, of length 3, (H,P,_)
         Returns:
             sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
     """
    H, P, _ = shape
    Cs = []
    for i in range(H):
        C = lecun_normal()(shape=(1, P, 2))
        Cs.append(C)
    return torch.tensor(Cs)[:, 0]


def init_CV(init_fun, shape, V) -> torch.Tensor:
    """ Initialize C_tilde=CV. First sample C. Then compute CV.
        Note we will parameterize this with two different matrices for complex
        numbers.
         Args:
             init_fun:  the initialization function to use, e.g. lecun_normal()
             shape (tuple): desired shape  (H,P)
             V: (complex64)     the eigenvectors used for initialization
         Returns:
             C_tilde (complex64) of shape (H,P,2)
     """
    C_ = init_fun(shape + (2,))
    C = C_[..., 0] + 1j * C_[..., 1]
    CV = C @ V
    return CV


def init_columnwise_B(shape, dtype):
    """Initialize B matrix in columnwise fashion.
    We will sample each column of B from a lecun_normal distribution.
    This gives a different fan-in size then if we sample the entire
    matrix B at once. We found this approach to be helpful for PathX
    It appears to be related to the point in
    https://arxiv.org/abs/2206.12037 regarding the initialization of
    the C matrix in S4, so potentially more important for the
    C initialization than for B.
     Args:
         key: jax random key
         shape (tuple): desired shape, either of length 3, (P,H,_), or
                      of length 2 (N,H) depending on if the function is called
                      from the low-rank factorization initialization or a dense
                      initialization
     Returns:
         sampled B matrix (float32), either of shape (H,P) or
          shape (H,P,2) (for complex parameterization)
    """
    shape = shape[:2] + ((2,) if len(shape) == 3 else ())
    lecun = variance_scaling(0.5 if len(shape) == 3 else 1.0, fan_in_axes=(0,))
    return lecun(shape, dtype)


def init_columnwise_VinvB(init_fun, Vinv):
    """Same function as above, but with transpose applied to prevent shape mismatch
    when using the columnwise initialization. In general this is unnecessary
    and will be removed in future versions, but is left for now consistency with
    certain random seeds until we rerun experiments."""

    def init(shape, dtype):
        B = init_fun(shape[:2], dtype)
        VinvB = Vinv @ B
        VinvB_real = VinvB.real
        VinvB_imag = VinvB.imag
        return torch.cat((VinvB_real[..., None], VinvB_imag[..., None]), axis=-1)

    return init


def init_rowwise_C(shape, dtype):
    """Initialize C matrix in rowwise fashion. Analogous to init_columnwise_B function above.
    We will sample each row of C from a lecun_normal distribution.
    This gives a different fan-in size then if we sample the entire
    matrix B at once. We found this approach to be helpful for PathX.
    It appears to be related to the point in
    https://arxiv.org/abs/2206.12037 regarding the initialization of
    the C matrix in S4.
     Args:
         shape (tuple): desired shape, of length 3, (H,P,_)
     Returns:
         sampled C matrix (float32) of shape (H,P,2) (for complex parameterization)
    """
    shape = shape[:2] + ((2,) if len(shape) == 3 else ())
    lecun = variance_scaling(0.5, fan_in_axes=(0,))
    return lecun(shape, dtype)

def combine(tree, operator, a_flat, b_flat):
    # Lower `fn` to operate on flattened sequences of elems.
    a = tree_unflatten(a_flat, tree)
    b = tree_unflatten(b_flat, tree)
    c = operator(a, b)
    c_flat, _ = tree_flatten(c)
    return c_flat


def _scan(tree, operator, elems, axis: int):
    """Perform scan on `elems`."""
    num_elems = elems[0].shape[axis]

    if num_elems < 2:
        return elems

    # Combine adjacent pairs of elements.
    reduced_elems = combine(tree, operator,
                            [torch.ops.aten.slice(elem, axis, 0, -1, 2) for elem in elems],
                            [torch.ops.aten.slice(elem, axis, 1, None, 2) for elem in elems])

    # Recursively compute scan for partially reduced tensors.
    odd_elems = _scan(tree, operator, reduced_elems, axis)

    if num_elems % 2 == 0:
        even_elems = combine(tree, operator,
                             [torch.ops.aten.slice(e, axis, 0, -1) for e in odd_elems],
                             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])
    else:
        even_elems = combine(tree, operator,
                             odd_elems,
                             [torch.ops.aten.slice(e, axis, 2, None, 2) for e in elems])

    # The first element of a scan is the same as the first element
    # of the original `elems`.
    even_elems = [
        torch.cat([torch.ops.aten.slice(elem, axis, 0, 1), result], dim=axis)
        if result.shape.numel() > 0 and elem.shape[axis] > 0 else
        result if result.shape.numel() > 0 else
        torch.ops.aten.slice(elem, axis, 0, 1)  # Jax allows/ignores concat with 0-dim, Pytorch does not
        for (elem, result) in zip(elems, even_elems)]

    return list(safe_map(partial(_interleave, axis=axis), even_elems, odd_elems))

# Pytorch impl. of jax.lax.associative_scan


def associative_scan(operator: Callable, elems, axis: int = 0, reverse: bool = False):
    # if not callable(operator):
    #     raise TypeError("lax.associative_scan: fn argument should be callable.")
    elems_flat, tree = tree_flatten(elems)

    if reverse:
        elems_flat = [torch.flip(elem, [axis]) for elem in elems_flat]

    assert axis >= 0 or axis < elems_flat[0].ndim, "Axis should be within bounds of input"
    num_elems = int(elems_flat[0].shape[axis])
    if not all(int(elem.shape[axis]) == num_elems for elem in elems_flat[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems_flat]))

    scans = _scan(tree, operator, elems_flat, axis)

    if reverse:
        scans = [torch.flip(scanned, [axis]) for scanned in scans]

    return tree_unflatten(scans, tree)


# def _interleave(a, b, axis):
#     assert a.shape[axis] == b.shape[axis] or a.shape[axis] == b.shape[axis] + 1
#     if b_trunc := (a.shape[axis] == b.shape[axis] + 1):
#         pad = [0, 0] * b.ndim
#         pad[(b.ndim-axis-1)*2+1] = 1 # +1=always end of dim, pad-order is reversed so start is at end
#         b = torch.nn.functional.pad(b, pad)

#     keys = list('ijklmnop')[:a.ndim]  # Get enough keys for each dim
#     expr = 't ' + ' '.join(keys) + ' -> '

#     keys[axis] = f'({keys[axis]} t)'  # Interleave along desired axis
#     expr += ' '.join(keys)
#     # for example 't i j -> (i t) j'
#     out: torch.Tensor = rearrange([a, b], expr)
#     if b_trunc:
#         out = out[slice_along_axis(0, b.shape[axis]+a.shape[axis]-1, axis=axis)]
#     return out

# @torch.jit.script
def _interleave(a, b, axis: int):
    # https://stackoverflow.com/questions/60869537/how-can-i-interleave-5-pytorch-tensors
    b_trunc = (a.shape[axis] == b.shape[axis] + 1)
    if b_trunc:
        pad = [0, 0] * b.ndim
        pad[(b.ndim-axis-1)*2+1] = 1  # +1=always end of dim, pad-order is reversed so start is at end
        b = torch.nn.functional.pad(b, pad)

    stacked = torch.stack([a, b], dim=axis+1)
    interleaved = torch.flatten(stacked, start_dim=axis, end_dim=axis+1)
    if b_trunc:
        # TODO: find torch alternative for slice_along axis for torch.jit.script to work
        interleaved = torch.ops.aten.slice(interleaved, axis, 0, b.shape[axis]+a.shape[axis]-1)
    return interleaved


def test_interleave():
    x, y = torch.randn(1, 32, 32), torch.randn(1, 32, 32)
    v = _interleave(x, y, axis=1)
    assert v.shape == (1, 64, 32)
    assert (v[:, 0] == x[:, 0]).all()
    assert (v[:, 1] == y[:, 0]).all()
    assert (v[:, 2] == x[:, 1]).all()
    assert (v[:, 3] == y[:, 1]).all()
    assert (v[:, 4] == x[:, 2]).all()

    v = _interleave(x, y, axis=2)
    assert v.shape == (1, 32, 64)
    assert (v[..., 0] == x[..., 0]).all()
    assert (v[..., 1] == y[..., 0]).all()
    assert (v[..., 2] == x[..., 1]).all()
    assert (v[..., 3] == y[..., 1]).all()
    assert (v[..., 4] == x[..., 2]).all()

    x, y = torch.randn(1, 24, 24), torch.randn(1, 24, 24)
    assert _interleave(x, y, axis=1).shape == (1, 48, 24)
    assert _interleave(x, y, axis=2).shape == (1, 24, 48)

    x, y = torch.randn(3, 96), torch.randn(2, 96)
    v = _interleave(x, y, axis=0)
    assert v.shape == (5, 96)
    assert (v[0] == x[0]).all()
    assert (v[1] == y[0]).all()
    assert (v[2] == x[1]).all()
    assert (v[3] == y[1]).all()
    assert (v[4] == x[2]).all()
    print('Interleave working as expected!')


def _compute_fans(shape, fan_in_axes=None):
    """Computes the number of input and output units for a weight shape."""
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in, fan_out = shape
    else:
        if fan_in_axes is not None:
            # Compute fan-in using user-specified fan-in axes.
            fan_in = np.prod([shape[i] for i in fan_in_axes])
            fan_out = np.prod([s for i, s in enumerate(shape)
                              if i not in fan_in_axes])
        else:
            # If no axes specified, assume convolution kernels (2D, 3D, or more.)
            # kernel_shape: (..., input_depth, depth)
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
    return fan_in, fan_out


def uniform(shape, dtype=torch.float, minval=0., maxval=1.0, device=None):
    src = torch.rand(shape, dtype=dtype, device=device)
    if minval == 0 and maxval == 1.:
        return src
    else:
        return (src * (maxval - minval)) + minval


def _complex_uniform(shape: Sequence[int],
                     dtype, device=None) -> torch.Tensor:
    """
    Sample uniform random values within a disk on the complex plane,
    with zero mean and unit variance.
    """
    r = torch.sqrt(2 * torch.rand(shape, dtype=dtype, device=device))
    theta = 2 * torch.pi * torch.rand(shape, dtype=dtype, device=device)
    return r * torch.exp(1j * theta)


def complex_as_float_dtype(dtype):
    match dtype:
        case torch.complex32:
            return torch.float32  # NOTE: complexe32 is not wel supported yet
        case torch.complex64:
            return torch.float32
        case torch.complex128:
            return torch.float64
        case _:
            return dtype


def _complex_truncated_normal(upper: float,
                              shape: Sequence[int],
                              dtype, device=None) -> torch.Tensor:
    """
    Sample random values from a centered normal distribution on the complex plane,
    whose modulus is truncated to `upper`, and the variance before the truncation
    is one.
    """
    real_dtype = torch.tensor(0, dtype=dtype).real.dtype
    t = ((1 - torch.exp(torch.tensor(-(upper ** 2), dtype=dtype, device=device)))
         * torch.rand(shape, dtype=real_dtype, device=device).type(dtype))
    r = torch.sqrt(-torch.log(1 - t))
    theta = 2 * torch.pi * torch.rand(shape, dtype=real_dtype, device=device).type(dtype)
    return r * torch.exp(1j * theta)


def _truncated_normal(lower, upper, shape, dtype=torch.float):
    if shape is None:
        shape = torch.broadcast_shapes(np.shape(lower), np.shape(upper))

    sqrt2 = math.sqrt(2)
    a = math.erf(lower / sqrt2)
    b = math.erf(upper / sqrt2)

    # a<u<b
    u = uniform(shape, dtype, minval=a, maxval=b)
    out = sqrt2 * torch.erfinv(u)
    # Clamp the value to the open interval (lower, upper) to make sure that
    # rounding (or if we chose `a` for `u`) doesn't push us outside of the range.
    with torch.no_grad():
        return torch.clip(
            out,
            torch.nextafter(torch.tensor(lower), torch.tensor(np.inf, dtype=dtype)),
            torch.nextafter(torch.tensor(upper), torch.tensor(-np.inf, dtype=dtype)))


def variance_scaling(scale: float,
                     mode: Literal["fan_in", "fan_out", "fan_avg"] = 'fan_in',
                     distribution: Literal["truncated_normal", "normal", "uniform"] = 'truncated_normal',
                     fan_in_axes: Optional[Sequence[int]] = None,
                     dtype=torch.float):
    def init(shape: Sequence[float],
             dtype=dtype,
             device=None):
        fan_in, fan_out = _compute_fans(shape, fan_in_axes)
        match mode:
            case 'fan_in':
                denom = max(1, fan_in)
            case 'fan_out':
                denom = max(1, fan_out)
            case 'fan_avg':
                denom = max(1, (fan_in + fan_out) / 2)
            case _:
                raise ValueError(f"invalid mode for variance scaling initializer: {mode}")

        variance = scale/denom
        match distribution:
            case 'normal':
                return torch.normal(0, np.sqrt(variance), shape, dtype=dtype, device=device)
            case 'uniform':
                if dtype.is_complex:
                    return _complex_uniform(shape, dtype=dtype, device=device) * np.sqrt(variance)
                else:
                    return uniform(shape, dtype=dtype, device=device, minval=-1, maxval=1.0) * np.sqrt(3 * variance)
            case 'truncated_normal':
                if dtype.is_complex:
                    stddev = np.sqrt(variance) * 0.95311164380491208
                    return _complex_truncated_normal(2, shape, dtype=dtype, device=device) * stddev
                else:
                    stddev = np.sqrt(variance) * 0.87962566103423978
                    return _truncated_normal(-2., 2., shape, dtype=dtype) * stddev
            case _:
                raise ValueError(f"invalid distribution for variance scaling initializer: {distribution}")

    return init


def lecun_normal(fan_in_axes=None, dtype=torch.float):
    """Builds a Lecun normal initializer.

    A `Lecun normal initializer`_ is a specialization of
    :func:`jax.nn.initializers.variance_scaling` where ``scale = 1.0``,
    ``mode="fan_in"``, and ``distribution="truncated_normal"``.

    Args:
    in_axis: axis or sequence of axes of the input dimension in the weights
      array.
    out_axis: axis or sequence of axes of the output dimension in the weights
      array.
    batch_axis: axis or sequence of axes in the weight array that should be
      ignored.
    dtype: the dtype of the weights.

    Returns:
    An initializer.

    Example:

    >>> import jax, jax.numpy as jnp
    >>> initializer = jax.nn.initializers.lecun_normal()
    >>> initializer(jax.random.PRNGKey(42), (2, 3), jnp.float32)  # doctest: +SKIP
    Array([[ 0.46700746,  0.8414632 ,  0.8518669 ],
         [-0.61677957, -0.67402434,  0.09683388]], dtype=float32)

    .. _Lecun normal initializer: https://arxiv.org/abs/1706.02515
    """
    return variance_scaling(1.0, "fan_in", "truncated_normal", fan_in_axes=fan_in_axes, dtype=dtype)


def test_variance_scaling():
    v = variance_scaling(1.0, distribution='normal')
    n_f32 = v((1, 10000), dtype=torch.float)
    assert np.isclose(n_f32.std().item(), 1.0, rtol=0.015, atol=0.015), f'std for f32 normal[0,1.0] is {n_f32.std()} != 1.0'
    del n_f32
    # NOTE: this is used in the original as `complex_normal` (but with stddev=0.5**0.5)
    n_c64 = v((1, 10000), dtype=torch.complex64)
    assert np.isclose(n_c64.std().item(), 1.0, rtol=0.015, atol=0.015), f'std for c64 normal[0,1.0] is {n_c64.std()} != 1.0'
    del n_c64

    # Truncated normal
    v = variance_scaling(1.0, distribution='truncated_normal')
    tn_f32 = v((1, 10000), dtype=torch.float)
    assert np.isclose(tn_f32.std().item(), 0.775, rtol=0.015, atol=0.015), f'std for f32 truncated normal[0,1.0] is {tn_f32.std()} != 0.775'
    del tn_f32

    # NOTE: this is used in the original (both trunc_standard_normal & lecun_normal it seems),
    # seems that they are using the fan-in/out feature to 'hide the low variance initialization'
    # The actual std observed is np.sqrt(2/shape[1]/(2*shape[0])); shape[2] has no impact
    v = variance_scaling(1.0, distribution='truncated_normal')
    tn_f32 = v((1, 10000, 2), dtype=torch.float)
    tn_c32 = torch.complex(tn_f32[..., 0], tn_f32[..., 1])
    expected_std = np.sqrt(2/tn_f32.shape[1]/(2*tn_f32.shape[0]))
    print(tn_c32.shape)
    assert np.isclose(tn_c32.std().item(), expected_std, rtol=0.015, atol=0.015), f'std for f32 truncated normal[0,1.0] is {tn_c32.std()} != {expected_std}'
    del tn_f32
    del tn_c32



@torch.jit.script
def binary_operator(q_i: Tuple[torch.Tensor, torch.Tensor], q_j: Tuple[torch.Tensor, torch.Tensor]):
    """Binary operator for parallel scan of linear recurrence. Assumes a diagonal matrix A.
    Args:
        q_i: tuple containing A_i and Bu_i at position i       (P,), (P,)
        q_j: tuple containing A_j and Bu_j at position j       (P,), (P,)
    Returns:
        new element ( A_out, Bu_out )
    """
    A_i, Bu_i = q_i
    A_j, Bu_j = q_j
    # return A_j * A_i, A_j * Bu_i + Bu_j
    return A_j * A_i, torch.addcmul(Bu_j, A_j, Bu_i)


def apply_ssm(Lambda_bars: torch.Tensor, B_bars, C_tilde, D, input_sequence, state=None, bidir: bool = False):
    cinput_sequence = input_sequence.type(Lambda_bars.dtype)  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Zero-pad for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    if state is not None:
        # Bu_elements = torch.cat(((state).unsqueeze(0), Bu_elements), dim=0)
        # Lambda_bars = torch.cat((torch.ones_like(state.unsqueeze(0)), Lambda_bars), dim=0)
        # Manually compute first step (Lambda_bar=1 so no change)
        Bu_elements[0] = Bu_elements[0]  + state * Lambda_bars[0]

    _, xs = associative_scan(binary_operator, (Lambda_bars, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(binary_operator, (Lambda_bars, Bu_elements), reverse=True)
        xs = torch.cat((xs, xs2), axis=-1)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du, xs[-1] #torch.stack((_[-1], xs[-1]))


def apply_ssm_liquid(Lambda_bars, B_bars, C_tilde, D, input_sequence, state=None, bidir: bool = False):
    """Liquid time constant SSM \u00e1 la dynamical systems given in Eq. 8 of
    https://arxiv.org/abs/2209.12951"""
    cinput_sequence = input_sequence.type(Lambda_bars.dtype)  # Cast to correct complex type

    if B_bars.ndim == 3:
        # Dynamic timesteps (significantly more expensive)
        Bu_elements = torch.vmap(lambda B_bar, u: B_bar @ u)(B_bars, cinput_sequence)
    else:
        # Static timesteps
        Bu_elements = torch.vmap(lambda u: B_bars @ u)(cinput_sequence)

    if Lambda_bars.ndim == 1:  # Zero-pad for associative_scan
        Lambda_bars = Lambda_bars.tile(input_sequence.shape[0], 1)

    if state is not None:
        # Manually compute first step (Lambda_bar=1 so no change)
        Bu_elements[0] = Bu_elements[0]  + state * Lambda_bars[0]

    _, xs = associative_scan(binary_operator, (Lambda_bars + Bu_elements, Bu_elements))

    if bidir:
        _, xs2 = associative_scan(binary_operator, (Lambda_bars, Bu_elements), reverse=True)
        xs = torch.cat((xs, xs2), axis=-1)

    Du = torch.vmap(lambda u: D * u)(input_sequence)
    return torch.vmap(lambda x: (C_tilde @ x).real)(xs) + Du, xs[-1]


# Discretization functions
def discretize_bilinear(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using bilinear transform method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device)
    BL = 1 / (Identity - (Delta / 2.0) * Lambda)
    Lambda_bar = BL * (Identity + (Delta / 2.0) * Lambda)
    B_bar = (BL * Delta)[..., None] * B_tilde
    return Lambda_bar, B_bar


def discretize_zoh(Lambda, B_tilde, Delta):
    """Discretize a diagonalized, continuous-time linear SSM
    using zero-order hold method.
    Args:
        Lambda (complex64): diagonal state matrix              (P,)
        B_tilde (complex64): input matrix                      (P, H)
        Delta (float32): discretization step sizes             (P,)
    Returns:
        discretized Lambda_bar (complex64), B_bar (complex64)  (P,), (P,H)
    """
    Identity = torch.ones(Lambda.shape[0], device=Lambda.device) # (replaced by -1)
    Lambda_bar = torch.exp(Lambda * Delta)
    B_bar = (1 / Lambda * (Lambda_bar - Identity))[..., None] * B_tilde
    return Lambda_bar, B_bar


def as_complex(t: torch.Tensor, dtype=torch.complex64):
    assert t.shape[-1] == 2, "as_complex can only be done on tensors with shape=(...,2)"
    nt = torch.complex(t[..., 0], t[..., 1])
    if nt.dtype != dtype:
        nt = nt.type(dtype)
    return nt


Initialization = Literal['dense_columns', 'dense', 'factorized']


class S5SSM(torch.nn.Module):
    def __init__(self, lambdaInit: torch.Tensor,
                 V: torch.Tensor, Vinv: torch.Tensor, h: int, p: int,
                 dt_min: float,
                 dt_max: float,
                 liquid: bool = False,
                 factor_rank: Optional[int] = None,
                 discretization: Literal['zoh', 'bilinear'] = 'zoh',
                 bcInit: Initialization = 'factorized',
                 degree: int = 1,
                 bidir: bool = False):
        """The S5 SSM
        Args:
            lambdaInit  (complex64): Initial diagonal state matrix       (P,)
            V           (complex64): Eigenvectors used for init          (P,P)
            Vinv        (complex64): Inverse eigenvectors used for init  (P,P)
            h           (int32):     Number of features of input seq
            p           (int32):     state size
            k           (int32):     rank of low-rank factorization (if used)
            bcInit      (string):    Specifies How B and C are initialized
                        Options: [factorized: low-rank factorization,
                                dense: dense matrix drawn from Lecun_normal]
                                dense_columns: dense matrix where the columns
                                of B and the rows of C are each drawn from Lecun_normal
                                separately (i.e. different fan-in then the dense option).
                                We found this initialization to be helpful for Pathx.
            discretization: (string) Specifies discretization method
                            options: [zoh: zero-order hold method,
                                    bilinear: bilinear transform]
            liquid:         (bool): use liquid_ssm from LiquidS4
            dt_min:      (float32): minimum value to draw timescale values from when
                                    initializing log_step
            dt_max:      (float32): maximum value to draw timescale values from when
                                    initializing log_step
            step_scale:  (float32): allows for changing the step size, e.g. after training
                                    on a different resolution for the speech commands benchmark
        """
        super().__init__()
        self.Lambda = torch.nn.Parameter(lambdaInit)
        self.degree = degree
        self.liquid = liquid
        self.bcInit = bcInit
        self.bidir = bidir
        # TODO:
        # if self.clip_eigs:
        #    self.Lambda = np.clip(self.Lambda_re, None, -1e-4) + 1j * self.Lambda_im

        # the P-dim of C can needs to be 2P for bidir
        cp = p
        if self.bidir:
            cp *= 2

        match bcInit:
            case 'complex_normal':
                self.C = torch.nn.Parameter(torch.normal(0, 0.5 ** 0.5, (h, cp), dtype=torch.complex64))
                self.B = torch.nn.Parameter(init_VinvB(lecun_normal(), Vinv)((p, h), torch.float))
            case 'dense_columns' | 'dense':
                if bcInit == "dense_columns":
                    B_eigen_init = init_columnwise_VinvB
                    B_init = init_columnwise_B
                    C_init = init_rowwise_C
                elif bcInit == "dense":
                    B_eigen_init = init_VinvB
                    B_init = C_init = lecun_normal()
                # TODO: make init_*VinvB all a the same interface
                self.B = torch.nn.Parameter(B_eigen_init(B_init, Vinv)((p, h), torch.float))
                if self.bidir:
                    C = torch.cat([init_CV(C_init, (h, p), V), init_CV(C_init, (h, p), V)], axis=-1)
                else:
                    C = init_CV(C_init, (h, p), V)
                self.C = torch.nn.Parameter(C)
            case _:
                raise NotImplementedError(f"BC_init method {bcInit} not implemented")

        # Initialize feedthrough (D) matrix
        self.D = torch.nn.Parameter(torch.rand(h,))
        self.log_step = torch.nn.Parameter(init_log_steps(p, dt_min, dt_max))
        match discretization:
            case 'zoh':
                self.discretize = discretize_zoh
            case 'bilinear':
                self.discretize = discretize_bilinear
            case _:
                raise ValueError(f'Unknown discretization {discretization}')

    def initial_state(self, batch_size: Optional[int]):
        batch_shape = (batch_size,) if batch_size is not None else ()
        return torch.zeros((*batch_shape, self.C_tilde.shape[-2]))

    def get_BC_tilde(self):
        match self.bcInit:
            case 'dense_columns' | 'dense' | 'complex_normal':
                B_tilde = as_complex(self.B)
                C_tilde = self.C
            case 'factorized':
                B_tilde = self.BP @ self.BH.T
                C_tilde = self.CH.T @ self.CP
        return B_tilde, C_tilde

    def forward_rnn(self, signal, prev_state, step_scale: float | torch.Tensor = 1.0):
        assert not self.bidir, "Can't use bidirectional when manually stepping"
        B_tilde, C_tilde = self.get_BC_tilde()
        step = step_scale * torch.exp(self.log_step)
        Lambda_bar, B_bar = self.discretize(self.Lambda, B_tilde, step)
        if self.degree != 1:
            assert (B_bar.shape[-2] == B_bar.shape[-1]), "higher-order input operators must be full-rank"
            B_bar **= self.degree

        # https://arxiv.org/abs/2209.12951v1, Eq. 9
        Bu = B_bar @ signal.type(B_bar.dtype)
        if self.liquid:
            Lambda_bar += Bu
        # https://arxiv.org/abs/2208.04933v2, Eq. 2
        x = Lambda_bar * prev_state + Bu
        y = (C_tilde @ x + self.D * signal).real
        return y, x

    def forward(self, signal, step_scale: float | torch.Tensor = 1.0, state=None, return_state=False):
        B_tilde, C_tilde = self.get_BC_tilde()

        if not torch.is_tensor(step_scale) or step_scale.ndim == 0:
            step = step_scale * torch.exp(self.log_step)
        else:
            # TODO: This is very expensive due to individual steps being multiplied by B_tilde in self.discretize
            step = step_scale[:, None] * torch.exp(self.log_step)

        # print(f'{self.Lambda.shape=} {B_tilde.shape=} {step.shape=}')
        # Lambda_bars, B_bars = torch.vmap(lambda s: self.discretize(self.Lambda, B_tilde, s))(step)
        # print(Lambda_bars.shape, B_bars.shape)
        Lambda_bars, B_bars = self.discretize(self.Lambda, B_tilde, step)
        if self.degree != 1:
            assert (B_bars.shape[-2] == B_bars.shape[-1]), "higher-order input operators must be full-rank"
            B_bars **= self.degree

        assert not (self.bidir and (state is not None)), "injecting state is not compatible with bidirectional S5"

        forward = apply_ssm_liquid if self.liquid else apply_ssm
        out, state = forward(Lambda_bars, B_bars, C_tilde, self.D, signal, state=state, bidir=self.bidir)
        # NOTE: technically it could work in a limited sense; taking the first and last element
        #   but that wouldn't be equivalent to running bidir on full sequences.
        #  It would be more like a circular S5 where you keep splicing the new signal into it;
        #   we leave implementing/testing this as an exercise to the reader
        assert not (self.bidir and return_state), "return_state does not work with bidirectional S5"
        if return_state:
            return out, state
        return out


class S5(torch.nn.Module):
    def __init__(self,
                 width: int,
                 state_width: Optional[int] = None,
                 factor_rank: Optional[int] = None,
                 block_count: int = 1,
                 dt_min: float = 0.001,
                 dt_max: float = 0.1,
                 liquid: bool = False,
                 degree: int = 1,
                 bidir: bool = False,
                 bcInit: Optional[Initialization] = None):
        super().__init__()
        state_width = state_width or width
        assert state_width % block_count == 0, "block_count should be a factor of state_width"

        block_size = state_width // block_count
        Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)
        Vinv = V.conj().T
        Lambda, B, V, B_orig, Vinv = map(lambda v: torch.tensor(v, dtype=torch.complex64), (Lambda, B, V, B_orig, Vinv))
        if block_count > 1:
            Lambda = Lambda[:block_size]
            V = V[:, :block_size]
            Lambda = (Lambda * torch.ones((block_count, block_size))).ravel()
            V = torch.block_diag(*([V] * block_count))
            Vinv = torch.block_diag(*([Vinv] * block_count))

        assert bool(factor_rank) != bool(bcInit != 'factorized'), "Can't have `bcInit != factorized` and `factor_rank` defined"
        bc_init = "factorized" if factor_rank is not None else (bcInit or "dense")
        self.width = width
        self.seq = S5SSM(
            Lambda,
            V,
            Vinv,
            width,
            state_width,
            dt_min,
            dt_max,
            factor_rank=factor_rank,
            bcInit=bc_init,
            liquid=liquid,
            degree=degree,
            bidir=bidir
        )

    def initial_state(self, batch_size: Optional[int] = None):
        return self.seq.initial_state(batch_size)

    def forward(self, signal, step_scale: float | torch.Tensor = 1.0, state=None, return_state=False):
        # NOTE: step_scale can be float | Tensor[batch] | Tensor[batch, seq]
        if not torch.is_tensor(step_scale):
            # Duplicate across batchdim
            step_scale = torch.ones(signal.shape[0], device=signal.device) * step_scale

        if state is None:
            return torch.vmap(lambda s, ss: self.seq(s, step_scale=ss, return_state=return_state))(signal, step_scale)
        else:
            return torch.vmap(lambda s, ss, _state: self.seq(s, step_scale=ss, state=_state, return_state=return_state))(signal, step_scale, state)


class GEGLU(torch.nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
