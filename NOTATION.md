# Notation

The conventions of the math in Specflow's glossary, ADRs, issues and docstrings. Symbols are the usual ones of fluid dynamics, and any other is defined where it is used.

## Typography

- Math is written in TeX, and vectors are not bold.
- A symbol names the field, not its representation: Python names carry the space (`_spectral`, `_physical`).

## Indices

- $i$ and $j$ run over the $d$ directions, and a repeated index is summed.
- A component may be labelled by its coordinate instead, as in $u_x$.
- $\partial_j = \partial / \partial x_j$ and $\nabla^2 = \partial_j \partial_j$.
- The square of a vector is its squared norm, and $|\cdot|$ is a magnitude.

## Wavenumbers

- Wavenumbers are dimensional: $\kappa = 2 \pi n / L$, with $n$ the integer wavenumber.
- $\hat f(\kappa)$ is a Fourier coefficient of $f$, normalized by the mean: $\hat f(\kappa) = \langle f \, e^{-i \kappa_j x_j} \rangle$, so that $f = \sum_n \hat f(\kappa) \, e^{i \kappa_j x_j}$ and $\langle |f|^2 \rangle = \sum_n |\hat f(\kappa)|^2$. A hat appears only in a formula about the coefficients.

## Averages

- $\langle \cdot \rangle$ is the mean over the periodic domain, and $\max$ the maximum over the mesh points.
- $\|f\| = \langle |f|^2 \rangle^{1/2}$.
