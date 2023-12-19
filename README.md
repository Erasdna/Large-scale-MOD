[toc]

# Large-scale-MOD
Semester project in the ANCHP chair at EPFL supervised by Prof. Daniel Kressner and Margherita Guido

## How to use:

1. Install Julia (for example from [here](https://julialang.org/downloads/))
2. Open the repository in your terminal. Type `julia`, this opens the REPL
3. In the REPL type `Ctrl` + `]` this opens package mode 
4. In package mode type `instantiate`. All the packages required by this project will now be installed in a local environment
5. Type `activate .` to activate this current environment

While this environment is active you can run code from the project within it. `Examples/Full_example.jl` provides an example with all the methods and row sampling. 

You can also run Julia as a script by writing:
`julia --project=<path-to-repo> <Location-of-file-to-be-ran>`
## Overview of the code

The folder `src` contains the main components of the code:
- `Example_problems.jl`: Contains an example problem taken from [here](https://arxiv.org/abs/2309.02156)
- `LinearSystem.jl`: Contains the `updateLinearSystem` function which updates the linear system related to the Elliptic PDE problem we solve
- `Problem.jl`: Contains the `EllipticPDE` and `DifferentialOperators2D`. 
    - The `EllipticPDE` structs holds information about the PDE to be solved. 
    - `DifferentialOperators2D` manages the discrete differential operators
- `RandomizedLeastSquares.jl`: Contains methods that allow for subsampling the rows of the least squares problem
    - The most important such method is `UniformLS` which uniformly samples the rows of a matrix and the rhs of the equation.
- `ReductionStrategies.jl`: Contains the various order reduction, initial strategies we implement (Nyström, Randomized Range Finder, Randomized SVD and POD).
- `Solver.jl`: Contains the `solve` function, which takes a `Problem` and solves it iteratively with a given timestep

## Project description

We consider a linear problem that evolves in time:
$$
\mathbf{A}(t_i)\mathbf{x} = \mathbf{b}(t_i), \quad \mathbf{A}(t_i) \in \mathbb{R}^{n\times n}, \mathbf{b}(t_i)\in \mathbb{R}^{n}
$$

In particular we are interested in a case where $\mathbf{A}(t_i)$ is large, sparse and not necessarily symmetric and therefore has to be solved using GMRES. In this project we consider such problems arising when solving PDEs numerically and we implement an example of a simple Elliptic PDE. The main goal of the project will be to speedup the solution of this problem by reducing the number of GMRES iterations necessary to converge to a good solution. We will do this be generating initial guesses for GMRES

### The initial guess approach

The initial guess approach involves three stages:
1. Generate a reduced basis $\mathbf{Q}\in\mathbb{R}^{n\times m}$ from a history matrix $\mathbf{X}$ of the $M$ previous solutions
**Assumption:** $m\leq M << n$
2. Solve the reduced problem $\mathbf{s}^* =\text{argmin}_{s\in \mathbb{R}^{m}} ||\mathbf{A}(t_i)\mathbf{X} s - \mathbf{b}(t_i)||_2$
3. Run GMRES with starting vector $\mathbf{x}_0=\mathbf{X}\mathbf{s}^*$

**Simply said**: We use the previous solution to create a smaller, more friendly space in which we can solve our problem (order reduction). Then we solve this smaller, much cheaper problem and hope that we get very close to the GMRES solution.

#### Strategy 0: Baseline

There two typical baselines when doing initial guess generation:

1. Use the previous state. This is a "free" initial guess, but it may be far away from the next state.
2. Use the Principal Orthogonal Decomposition (POD). This involves two stages:
    1. Perform a SVD of $\mathbf{X}$: $\text{SVD}(\mathbf{X}) = [\mathbf{\Psi},\mathbf{\Sigma}, \mathbf{\Phi}]$
    2. Truncate $\mathbf{\Psi}$ to the first $m$ columns

The POD is the optimal $m$-rank approximation of $\mathbf{X}$ and will therefore give the best GMRES reduction. The other methods we implement will try  to improve on this by introducing randomness.

#### Strategy 1: Randomized Range Finder

**Goal**: Find a matrix $\mathbf{Q}$ that approximates well the range of $\mathbf{X}$. 

This method involves three steps:
1. Sample a sketching matrix $\Omega \in \mathbb{R}^{M\times m}$
2. Compute $\mathbf{X}\Omega$
3. Compute QR of $\mathbf{X}\Omega = [\mathbf{Q},R]$

Where $\Omega$ is random Gaussian matrix. This method reduces the size of the matrix on which QR is performed through random sketching. This is the main method implemented [here](https://arxiv.org/abs/2309.02156)

#### Strategy 2: Randomized SVD

**Goal**: Compute a cheap SVD of $\mathbf{X}$

The Randomized Range Finder is the first step of the [Randomized SVD](https://arxiv.org/abs/0909.4061)

Full method:
1. Obtain $\hat{Q}$ from the Randomized Range Finder
2. Compute $\hat{Q}^\top \mathbf{X}$
3. Compute the SVD of $\hat{Q}^\top \mathbf{X} = [\hat{U}, \Sigma, V]$
4. Set $\mathbf{Q}=\hat{Q}\hat{U}$

This method requires extra steps compared with the Randomized Range Finder, but may result in a better basis. In practice, it turns out that the Randomized Range Finder is generally sufficient.

#### Strategy 3: Generalized Nyström 

**Goal**: Obtain a left side projection matrix from the [Generalized Nyström approximation](https://arxiv.org/abs/2009.11392)

The Generalized Nyström is given by: $\mathbf{X}\Omega_1(\Omega_2^\top \mathbf{X}\Omega_1)^{\dagger}\Omega_2^\top \mathbf{X}$ 
- Where $\Omega_1 \in \mathbb{R}^{M \times k}$ and  $\Omega_2 \in \mathbb{R}^{n \times (k+p)}$ are random matrices

This can be rewritten as a left and right side projector:
$$
\mathcal{P}_{\mathbf{X}\Omega_1,\Omega_2}\mathbf{X}\mathcal{P}_{\Omega_1,\mathbf{X}^\top \Omega_2} = 
\mathbf{X}\Omega_1(\Omega_2^\top \mathbf{X} \Omega_1)^{\dagger}\Omega_2^\top \mathbf{X} \Omega_1 (\Omega_2^\top \mathbf{X} \Omega_1)^{\dagger} \Omega_2^\top \mathbf{X}
$$

And set $\mathbf{Q}=\mathbf{X}\Omega_1(\Omega_2^\top \mathbf{X} \Omega_1)^{\dagger}$

We implement the method in the following steps using $k+p=m$ and $p\in [0,k/2]$:
1. Sample $\Omega_1 \in \mathbb{R}^{M \times k}$ and  $\Omega_2 \in \mathbb{R}^{n \times (k+p)}$
2. Compute $\mathbf{X}\Omega_1$
3. Compute $\Omega_2^\top\mathbf{X}\Omega_1$
4. Compute $\mathbf{Q} = \mathbf{X}\Omega_1 (\Omega_2^\top\mathbf{X}\Omega_1)^\dagger$

This method will be cheaper than the Randomized Range Finder and Randomized SVD, but the error guarantees are not as strong.

### Example problem

As an example problem we use the one from [this paper.](https://arxiv.org/abs/2309.02156)

$$
\begin{align*}
\nabla \cdot (a(\mathbf{x},t) \nabla f(\mathbf{x},t)) &= g(\mathbf{x},t) & \forall \mathbf{x} \in \Omega \\
f(\mathbf{x},t) &= 0 & \forall \mathbf{x} \in \partial\Omega \nonumber \\
& & \Omega \subset [0,1]^2
\end{align*}
$$
We use the following function for $a$ and the exact solution $f$
$$
\begin{align*}
    a(\mathbf{x},t) &= e^{-(x - 0.5)^2 - (y - 0.5)^2}\cos(tx) + 2.1 \\
    f(x,y,t) &= \sin(4\pi y)\sin(4\pi x)\left(1+\sin(15\pi x t)\sin(3 \pi y t)e^{-(x-0.5)^2 - (y-0.5)^2 -0.25^2)}\right) \\
\end{align*}
$$
