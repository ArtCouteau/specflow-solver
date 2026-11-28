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
The internal state of a term: its history and, for a random term, its random generator. It belongs to the term, not to the state.
_Avoid_: auxiliary state

**History**:
The previous values a term carries from one step to the next, such as the forcing coefficients. Part of the term state.

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
A quantity derived from one state for monitoring or analysis: a scalar such as kinetic energy, dissipation, enstrophy or divergence, or a function such as a spectrum, a PDF or a structure function.
_Avoid_: observable, quantity

**Fluctuation**:
A field minus its mean over the periodic domain.
_Avoid_: perturbation, fluctuating part

**Statistic**:
A diagnostic accumulated over the samples of a window, such as a time-averaged spectrum or a PDF built from many states.
_Avoid_: diagnostic, average

**Window**:
The time interval whose samples a statistic accumulates, taken in a statistically steady state.
_Avoid_: averaging period

**Dissipation**:
The rate at which kinetic energy is lost to viscosity, $2 \nu \langle S_{ij} S_{ij} \rangle$, with $S_{ij}$ the strain rate. A diagnostic, not a term.
_Avoid_: diffusion

**Injection rate**:
The rate at which forcing adds kinetic energy, $\langle u_i f_i \rangle$. A diagnostic, not a term.
_Avoid_: power input, energy input

**Budget**:
The split of the rate of change of a quadratic quantity, such as kinetic energy, into the contributions of each term.
_Avoid_: balance

**Budget spectrum**:
One term's contribution to the rate of change of the energy spectrum, shell by shell.
_Avoid_: transfer spectrum, energy transfer

**Flux**:
The rate at which advection carries energy out of the ball of shells below a wavenumber.
_Avoid_: cumulative transfer

## Runs

**Simulation**:
The time integration of a set of equations from an initial state over a time interval, together with its outputs.
_Avoid_: solver

**Snapshot**:
The state, the term states and the diagnostics meant to be averaged, saved at one output time; a restart starts from a snapshot.
_Avoid_: dump, checkpoint, restart file

## Parallelism

**Decomposition**:
The split of the mesh across ranks; the serial decomposition gives the whole mesh to rank 0.

**Rank**:
One process of a distributed simulation, holding its part of the decomposition; ranks are numbered from 0. Never the order of a tensor.
_Avoid_: process, worker
