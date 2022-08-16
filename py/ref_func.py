"""
    Spherical Harmonics Generation
    Adopted from official repo: https://github.com/google-research/multinerf
    Converted to PyTorch (CUDA)
"""

import numpy as np
import torch

def generalized_binomial_coeff(a, k):
  """Compute generalized binomial coefficients."""
  return np.prod(a - np.arange(k)) / np.math.factorial(k)

def assoc_legendre_coeff(l, m, k):
  """Compute associated Legendre polynomial coefficients.

  Returns the coefficient of the cos^k(theta)*sin^m(theta) term in the
  (l, m)th associated Legendre polynomial, P_l^m(cos(theta)).

  Args:
    l: associated Legendre polynomial degree.
    m: associated Legendre polynomial order.
    k: power of cos(theta).

  Returns:
    A float, the coefficient of the term corresponding to the inputs.
  """
  return ((-1)**m * 2**l * np.math.factorial(l) / np.math.factorial(k) /
          np.math.factorial(l - k - m) *
          generalized_binomial_coeff(0.5 * (l + k + m - 1.0), l))

def sph_harm_coeff(l, m, k):
  """Compute spherical harmonic coefficients."""
  return (np.sqrt(
      (2.0 * l + 1.0) * np.math.factorial(l - m) /
      (4.0 * np.pi * np.math.factorial(l + m))) * assoc_legendre_coeff(l, m, k))

def get_ml_array(deg_view):
  """Create a list with all pairs of (l, m) values to use in the encoding."""
  ml_list = []
  for i in range(deg_view):
    l = 2**i
    # Only use nonnegative m values, later splitting real and imaginary parts.
    for m in range(l + 1):
      ml_list.append((m, l))

  # Convert list into a numpy array.
  ml_array = np.array(ml_list).T
  return ml_array

def generate_ide_fn(deg_view):
  """Generate integrated directional encoding (IDE) function.

  This function returns a function that computes the integrated directional
  encoding from Equations 6-8 of arxiv.org/abs/2112.03907.

  Args:
    deg_view: number of spherical harmonics degrees to use.

  Returns:
    A function for evaluating integrated directional encoding.

  Raises:
    ValueError: if deg_view is larger than 5.
  """
  if deg_view > 5:
    raise ValueError('Only deg_view of at most 5 is numerically stable.')

  ml_array = get_ml_array(deg_view)
  l_max = 2**(deg_view - 1)

  # Create a matrix corresponding to ml_array holding all coefficients, which,
  # when multiplied (from the right) by the z coordinate Vandermonde matrix,
  # results in the z component of the encoding.
  mat = torch.zeros(l_max + 1, ml_array.shape[1]).cuda()
  for i, (m, l) in enumerate(ml_array.T):
    for k in range(l - m + 1):
      mat[k, i] = sph_harm_coeff(l, m, k)

  def integrated_dir_enc_fn(xyz, kappa_inv):
    """Function returning integrated directional encoding (IDE).

    Args:
      xyz: [..., 3] array of Cartesian coordinates of directions to evaluate at.
      kappa_inv: [..., 1] reciprocal of the concentration parameter of the von
        Mises-Fisher distribution.

    Returns:
      An array with the resulting IDE.
    """
    x = xyz[..., 0:1]
    y = xyz[..., 1:2]
    z = xyz[..., 2:3]

    ml_array_cu = torch.from_numpy(ml_array).cuda()
    # Compute z Vandermonde matrix.
    vmz = torch.cat([z**i for i in range(mat.shape[0])], dim=-1)
    # Compute x+iy Vandermonde matrix.
    vmxy = torch.cat([(x + 1j * y)**m for m in ml_array_cu[0, :]], dim=-1)
    # Get spherical harmonics.
    sph_harms = vmxy * (vmz @ mat)

    # Apply attenuation function using the von Mises-Fisher distribution
    # concentration parameter, kappa.
    sigma = 0.5 * ml_array_cu[1, :] * (ml_array_cu[1, :] + 1)
    ide = sph_harms * torch.exp(-sigma * kappa_inv)
    # Split into real and imaginary parts and return
    return torch.cat([torch.real(ide), torch.imag(ide)], dim=-1)

  return integrated_dir_enc_fn

if __name__ == "__main__":
  view_dirs = torch.normal(0, 1, (256, 64, 3)).cuda()
  view_dirs = view_dirs / torch.norm(view_dirs, dim=-1, keepdim=True)
  rho = torch.rand(256, 64, 1).cuda() * 0.95 + 0.05
  ide_func = generate_ide_fn(4)
  result = ide_func(view_dirs, rho)
  print(result.shape)
