# Notation

The symbols of Specflow, in the terms of `GLOSSARY.md`. Every symbol in the glossary, the ADRs, the issues and the docstrings is defined here.

## Conventions

Math is written in TeX. Repeated indices are summed, $\nabla^2 = \partial_j \partial_j$, and the square of a vector is its squared norm. Vectors are not bold, and a field's space is in its name, never in its symbol.

## Symbols

- $d$, $N$, $L$: the dimension, the number of points per direction and the length of the periodic domain.
- $x, y, z$ or $x_i$: the coordinates; $\Delta x = L/N$: the mesh spacing.
- $n$: an integer wavenumber; $\kappa = 2 \pi n / L$: the wavenumber, with components $\kappa_i$ and magnitude $|\kappa|$; $\kappa_0 = 2 \pi / L$.
- $t$, $\Delta t$: the time and the time step; $C$: the CFL number.
- $u$: the velocity; $\omega$: the vorticity; $\theta$: a scalar; $q$: any variable.
- $S_{ij} = (\partial_i u_j + \partial_j u_i) / 2$: the strain rate.
- $\nu$: the viscosity; $\alpha$: a diffusivity.
- $\langle \cdot \rangle$: the mean over the periodic domain.
