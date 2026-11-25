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
- A prime marks a fluctuation, $f' = f - \langle f \rangle$, and never a root mean square, which is written $f_\text{rms}$.

## Spectra

- Shell $m$ holds the wavenumbers with $\mathrm{round}(|n|) = m$, centred on $\kappa_m = m \, \Delta\kappa$ with $\Delta\kappa = 2 \pi / L$; shell 0 holds the mean alone.
- A spectrum is a density in $\kappa$: the spectrum of a quantity whose mean is $\sum_n Q(n)$ is $\frac{1}{\Delta\kappa} \sum_{\mathrm{round}(|n|) = m} Q(n)$ at $\kappa_m$, so that its sum over the shells times $\Delta\kappa$ is the mean.
- A flux at a shell is the rate at which a quantity leaves the ball of shells $0, \dots, m$, through its edge at $(m + \tfrac12) \, \Delta\kappa$; it is positive outward.
