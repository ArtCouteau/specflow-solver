# Specflow

Specflow simulates flows in a periodic domain with a pseudo-spectral method: variables evolve as Fourier coefficients, and nonlinear products are formed at the mesh points.

## Domain

**Periodic domain**:
The region $[0, L)^d$, with $d = 1$, 2 or 3, periodic in every direction.
_Avoid_: triply periodic domain, box

**Mesh**:
The discretization of the periodic domain: $N$ equispaced points per direction, $N$ even, and the matching wavenumbers.
_Avoid_: grid

## Representation

**Physical space**:
The representation of a field by its values at the mesh points.
_Avoid_: real space, grid space

**Spectral space**:
The representation of a field by its Fourier coefficients on the mesh wavenumbers. Variables are always held in spectral space.
_Avoid_: Fourier space, wavenumber space

## Evolution

**State**:
The variables at one instant, each identified by its name.
_Avoid_: solution, solution variable

**Variable**:
A named field of the state, evolved by exactly one equation (e.g. velocity).
_Avoid_: unknown, prognostic field

**Equation**:
The evolution law of one variable: a sum of terms, optionally followed by a projection.
_Avoid_: model, system

**Term**:
One additive piece of an equation's tendency: advection, diffusion, stress closure or forcing.

**Linear term**:
A term that is linear in its equation's variable and diagonal in spectral space, fully described by one multiplier per wavenumber (e.g. diffusion).

**Nonlinear term**:
Any term that is not a linear term, evaluated from the state (advection, stress closures, forcing). Forcing is one, although it does not depend on the state.
_Avoid_: explicit term

**Term state**:
The internal state of a term, such as forcing coefficients and their random generator. It belongs to the term, not to the state.
_Avoid_: auxiliary state

**Tendency**:
The time derivative of a variable, as a function of the state and the time.
_Avoid_: RHS, right-hand side

**Linear operator**:
The part of an equation's tendency formed by its linear terms: one multiplier per wavenumber, such as $-\nu \kappa^2$.
_Avoid_: stiff part, implicit part

**Contribution**:
The part of an equation's tendency produced by one term, with the equation's projection applied.

**Projection**:
The removal of the gradient part of a vector field; it enforces incompressibility in place of a pressure term.
_Avoid_: pressure term, pressure solve

## Terms

**Advection**:
Transport of a variable by the velocity, such as $u_j \partial_j u_i$ or $u_j \partial_j \theta$. The only dealiased term.
_Avoid_: convection

**Diffusion**:
The linear term $\alpha \nabla^2 q$ with a constant diffusivity $\alpha$ (the viscosity $\nu$ for velocity).
_Avoid_: viscous term, dissipation

**Stress closure**:
A term whose stress tensor is modeled from the local velocity gradient in physical space, such as Smagorinsky or a generalized Newtonian viscosity. The only Fourier-smoothed term.
_Avoid_: LES term, viscosity model

**Forcing**:
A term that injects energy at prescribed wavenumbers, independently of the flow, such as Ornstein–Uhlenbeck forcing.

## Aliasing

**Dealiasing**:
The exact removal of aliasing errors from a quadratic product.
_Avoid_: filtering, smoothing

**Fourier smoothing**:
A smooth damping of the highest wavenumbers, used where a product is not quadratic and cannot be dealiased exactly.
_Avoid_: dealiasing, spectral filter

## Analysis

**Diagnostic**:
A quantity derived from the state for monitoring or analysis, such as kinetic energy, dissipation, enstrophy or divergence.
_Avoid_: statistic, observable, quantity

**Dissipation**:
The rate at which kinetic energy is lost to viscosity, $2 \nu \langle S_{ij} S_{ij} \rangle$, with $S_{ij}$ the strain rate. A diagnostic, not a term.
_Avoid_: diffusion

**Budget**:
The split of the rate of change of a quadratic quantity, such as kinetic energy, into the contributions of each term.
_Avoid_: balance

## Runs

**Simulation**:
The time integration of a set of equations from an initial state over a time interval, together with its outputs.
_Avoid_: solver

**Snapshot**:
The state and the term states saved at one output time; a restart starts from a snapshot.
_Avoid_: dump, checkpoint, restart file

## Parallelism

**Decomposition**:
The split of the mesh across ranks; the serial decomposition gives the whole mesh to rank 0.

**Rank**:
One process of a distributed simulation, holding its part of the decomposition; ranks are numbered from 0. Never the order of a tensor.
_Avoid_: process, worker

## Notation

Math is written in TeX. Repeated indices are summed, $\nabla^2 = \partial_j \partial_j$, and the square of a vector is its squared norm. Vectors are not bold, and a field's space is in its name, never in its symbol.

- $d$, $N$, $L$: the dimension, the number of points per direction and the length of the periodic domain.
- $x, y, z$ or $x_i$: the coordinates; $\Delta x = L/N$: the mesh spacing.
- $n$: an integer wavenumber; $\kappa = 2 \pi n / L$: the wavenumber, with components $\kappa_i$ and magnitude $|\kappa|$; $\kappa_0 = 2 \pi / L$.
- $t$, $\Delta t$: the time and the time step; $C$: the CFL number.
- $u$: the velocity; $\omega$: the vorticity; $\theta$: a scalar; $q$: any variable.
- $\nu$: the viscosity; $\alpha$: a diffusivity.
- $\langle \cdot \rangle$: the mean over the periodic domain.
