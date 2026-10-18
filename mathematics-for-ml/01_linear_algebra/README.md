# **Linear Algebra Roadmap**

## **01 Scalars and Basics**

- Scalar as a single number; scalar as a Python float, a 0-d tensor and a 1x1 matrix, and how they differ

- Number sets: naturals, integers, rationals, reals; notation `R`, $R^n$, $R^{m x n}$

- Arithmetic laws: commutative, associative, distributive; identity and inverse elements; where they hold for scalars and fail later for matrices

- Absolute value, sign, inequalities, the triangle inequality in one dimension

- Exponent and root rules; `exp` and `log` as inverses; $log(ab) = log a + log b$; change of base

- Why logs matter in ML: products of probabilities become sums, underflow is avoided, log-likelihood

- Summation and product notation; splitting, shifting and reindexing sums

- Functions: domain, range, composition, inverse, monotonicity; piecewise functions such as ReLU and clipping

- Scalar functions used throughout ML: sigmoid, tanh, softplus, ReLU; their shapes and where they saturate (derivatives come in the calculus folders)

- `max`, `min`, `argmax`, clipping

- Notation used in all later notes: lowercase for scalars, bold lowercase for vectors, uppercase for matrices; subscripts versus superscripts; batch-index conventions

- Floating-point scalars: float32 versus float64, machine epsilon, why $0.1 + 0.2 != 0.3$, overflow in $exp(710)$, $log(0)$, rounding

- Scalar multiplication as scaling (preview of vectors)

- Compare floats with a tolerance (`math.isclose`), never with $==$

- Demonstrate overflow and underflow of `exp`, and the fixes `log1p` and `expm1`

- Plot sigmoid, tanh, softplus and ReLU and mark the saturated regions







## **02 Vectors**



Marked done. Tick each bullet against your existing notes and add any that are missing.



- Vector as an array, as a point, and as a direction; why mixing these views causes errors

- Notation: column vector in $R^n$, components, indexing, row versus column vectors

- Vector addition and scalar multiplication; geometric meaning

- Linear combinations, span, linear independence, basis, dimension (formal treatment returns in 06)

- Dot product: algebraic form, geometric form $||u|| ||v|| cos(theta)$, symmetry, bilinearity

- Three readings of the dot product: projection, similarity, linear functional

- Cauchy-Schwarz inequality and the triangle inequality

- Norms: L1, L2, L-infinity, general `p`-norm; unit vectors and normalization

- Angle between vectors; cosine similarity; orthogonal vectors
- Complex vectors and conjugate transpose (Hermitian transpose); when complex-valued vectors matter in signal processing and spectral methods

- Projection of one vector onto another, and the residual

- Distance measures: Euclidean, Manhattan, cosine distance, and when each is appropriate

- Hyperplanes and half-spaces; signed distance from a point to a hyperplane

- Outer product of two vectors and the rank-1 matrix it produces

- Vectors as data: feature vectors, embeddings, parameter vectors, gradients

- High-dimensional geometry:

  - Random Gaussian vectors have norm about $sqrt(d)$ with relative spread of order $1/sqrt(d)$

  - Their cosine similarity is about $N(0, 1/d)$, so they are nearly orthogonal

  - Distance concentration: nearest and farthest neighbours become hard to tell apart on random data

  - Hubness: a few points become the nearest neighbour of many others

  - Real embeddings are anisotropic, so cosines between unrelated items are often well above zero

- Skip: cross product (3D only), complex vectors

- Implement vector operations, norms, dot product, cosine similarity and projection without `np.dot` on the first pass

- Plot cosine histograms for random vectors at `d` in {2, 16, 128, 1024, 4096} and check that the spread tracks $1/sqrt(d)$

- Compare cosine and dot-product retrieval on a real embedding set and find a query where they disagree







## **03 Matrices**



Marked done. Tick each bullet against your existing notes and add any that are missing.



- Matrix as a table of numbers and as a linear map; why both views are needed

- Notation and indexing: $a_ij$, rows and columns, shape `m x n`, slicing

- Row vectors and column vectors as special matrices

- Matrix as a collection of column vectors and as a collection of row vectors

- First look at the linear-map view: the columns of `A` are the images of the standard basis vectors

- Special matrices: zero, identity, diagonal, triangular, symmetric, skew-symmetric, permutation, idempotent, sparse

- Orthogonal matrices (definition only, $Q^T Q = I$; properties come in 07)
- Hermitian matrices and unitary matrices (recognition and core properties; optional unless using complex-valued models)

- Transpose: definition, symmetric means $A = A^T$, any square matrix splits into symmetric plus skew-symmetric parts

- Block matrices: how to partition and read them (block multiplication is in 04)

- Shape conventions in ML: data matrix `X` is `[N, d]`, weights `W` are $[d_in, d_out]$, PyTorch `nn.Linear` stores $[d_out, d_in]$

- Matrices that appear everywhere: design matrix, covariance matrix, Gram matrix, adjacency matrix, an image as a matrix

- Memory layout: row-major versus column-major, and why it affects speed

- Creating, reshaping and inspecting matrices in NumPy: `shape`, `ndim`, `dtype`, `reshape`, `T`

- Implement a small matrix class with indexing, transpose and a symmetric check

- Build the symmetric and skew-symmetric decomposition of a random square matrix and verify it

- Compute a covariance matrix and a Gram matrix from a small data matrix and confirm both are symmetric







## **04 Matrix Operations**



- Addition and scalar multiplication of matrices, and their properties

- Matrix multiplication: definition, shape rule $(m x k)(k x n) = (m x n)$

- The four views of multiplication: dot-product entries, combinations of columns, combinations of rows, sum of outer products

- Associativity and distributivity hold; commutativity fails (give counterexamples)

- Transpose rules for sums and products, including $(AB)^T = B^T A^T$

- Trace, its linearity and cyclic property $tr(ABC) = tr(BCA)$

- Hadamard (elementwise) product versus matrix product

- Block matrices and block multiplication

- Matrix powers $A^k$, and how special matrices from 03 behave under products: diagonal times diagonal, triangular times triangular, orthogonal times orthogonal, permutation times permutation

- Matrix-vector product as a linear combination of columns

- FLOP count of a matmul: $2*m*n*k$
- Matrix multiplication cost versus memory traffic; why data movement can dominate runtime on CPUs and GPUs

- Row-major memory layout and why loop order changes speed

- Bias addition as broadcasting a rank-1 update

- Implement matmul with three loops, then with outer products, then block-wise, then compare timing with `A @ B`

- Verify $(AB)^T = B^T A^T$ and $tr(AB) = tr(BA)$ numerically on random matrices







## **05 Determinants and Inverses**



- Linear systems $Ax = b$: row picture and column picture

- Gaussian elimination, row echelon form, reduced row echelon form, pivots

- Three outcomes of a system: no solution, one solution, infinitely many; how pivots decide

- Elementary matrices and what each row operation does

- LU factorization, partial pivoting, $PA = LU$

- Forward and back substitution

- Inverse: definition, uniqueness, $(AB)^-1 = B^-1 A^-1$, $(A^T)^-1 = (A^-1)^T$

- Gauss-Jordan method for computing an inverse; the 2x2 closed-form inverse

- Why you solve $Ax = b$ instead of computing `inv(A) @ b`

- Determinant from its defining properties: multilinear, alternating, $det(I) = 1$

- Determinant of 2x2 and 3x3 by hand; cofactor expansion only for small sizes

- Geometric meaning: area and volume scaling, sign as orientation

- Rules: $det(AB) = det(A) det(B)$, $det(A^T) = det(A)$, $det(A^-1) = 1/det(A)$, triangular matrix gives product of diagonal

- Determinant is zero if and only if the matrix is singular

- Equivalent conditions for invertibility (rank, pivots, null space, determinant, eigenvalues)

- Computing determinant through LU; $log det$ and `slogdet` to avoid overflow

- Why determinant is a poor test for near-singularity (use singular values or condition number)

- Cramer's rule: know it exists, never use it

- Implement forward and back substitution, LU with pivoting, Gauss-Jordan inverse; verify against `scipy.linalg.lu`, `solve`, `inv`

- Where it appears: $log det$ in Gaussian densities and normalizing flows, singular covariance matrices







## **06 Vector Spaces**



- Vector space axioms (brief), subspaces, examples beyond $R^n$ (matrices, polynomials, functions)

- Linear combination, span, linear independence

- Basis, dimension, coordinates relative to a basis

- Change of basis between two bases

- The four fundamental subspaces: column space, null space, row space, left null space

- Dimensions of the four subspaces and the rank-nullity theorem

- Row rank equals column rank

- Rank facts: $rank(AB) <= min(rank(A), rank(B))$, full rank, rank-deficient matrices

- Solution set of $Ax = b$ as a particular solution plus the null space

- Affine subspaces

- Orthogonality of subspaces: row space is orthogonal to null space, column space is orthogonal to left null space

- Orthogonal complement

- Inner product spaces: inner product axioms, induced norm, Cauchy-Schwarz, triangle inequality

- Numerical rank: using a tolerance on singular values instead of exact zeros

- Where it appears: collinear features, collapsed embedding dimensions, why $dL/dW = X^T delta$ has rank at most the number of tokens in the batch

- Implement bases of all four subspaces (via SVD or QR), verify rank-nullity, build random matrices of a chosen rank







## **07 Linear Transformations**



- Definition of linearity and the matrix of a transformation

- Standard 2D and 3D transformations: rotation, scaling, shear, reflection, projection

- Composition as multiplication, inverse transformation

- Kernel and image as null space and column space

- Injective, surjective, isomorphism and how rank decides each

- Matrix of a transformation in a different basis; similarity $A = P B P^-1$ and what stays invariant

- Affine transformations and homogeneous coordinates

- Orthogonal matrices: $Q^T Q = I$, preserve length and angle, $det = +1$ (rotation) or `-1` (reflection)

- Application of rotations: RoPE rotates each 2D pair of a query or key by angle $m * theta$; since $R(a)^T R(b) = R(b - a)$, the dot product $(R_m q) . (R_n k)$ depends only on `n - m`

- Projection onto a vector and onto a subspace: $P = A(A^T A)^-1 A^T$, symmetric and idempotent, `I - P` projects onto the complement

- Application of projections: the LayerNorm backward pass contains $I - (1/d) 1 1^T - (1/d) x_hat x_hat^T$, which is exactly a projector (when $eps = 0$) removing the mean direction and the normalized-input direction; verify $P^2 = P$ and $P 1 = 0$ numerically

- Gram-Schmidt, its numerical instability, modified Gram-Schmidt

- QR factorization, Householder reflections (idea only)

- Least squares: normal equations, geometric picture (residual orthogonal to column space), solving through QR

- Normal equations square the condition number

- Pseudo-inverse and minimum-norm solutions

- Ridge regression as regularized least squares

- Implement a transformation visualizer (unit square and unit circle before and after)

- Implement Gram-Schmidt and QR from scratch

- Implement linear regression three ways: normal equations, QR, `lstsq`, on a well-conditioned and an ill-conditioned (Vandermonde) problem







## **08 Eigenvalues and Eigenvectors**



- Definition $Av = lambda v$ and the geometric meaning (directions that only get scaled)

- Eigenspaces

- Characteristic polynomial; compute 2x2 and 3x3 by hand

- Algebraic versus geometric multiplicity

- Sum of eigenvalues equals trace, product equals determinant

- Eigenvalues of triangular matrices, of $A^k$, $A^-1$, `A + cI`, and $A^T A$

- Diagonalization $A = V D V^-1$, when it exists, defective matrices

- Matrix powers through diagonalization; matrix exponential

- Complex eigenvalues as rotation

- Symmetric matrices: spectral theorem, real eigenvalues, orthogonal eigenvectors, $A = Q L Q^T$

- Positive (semi)definite matrices: definitions, quadratic forms, equivalent tests (eigenvalues, Gram matrix, Cholesky)

- Rayleigh quotient and its bounds

- Covariance matrices as symmetric PSD

- Softmax Jacobian $diag(p) - p p^T$ as a worked PSD example: symmetric, rank $n - 1$, the all-ones vector is in its null space (this is why softmax is shift-invariant); verify numerically

- Power iteration and its convergence rate $|lambda_2 / lambda_1|$; know inverse iteration and the QR algorithm exist

- Markov chains: stationary distribution as an eigenvector, Perron-Frobenius idea

- Dynamical systems and RNN stability through spectral radius

- Non-normal matrices: spectral radius below 1 does not rule out large transient growth

- Gershgorin discs (optional)

- Implement `eig` versus `eigh`, power iteration, eigendecomposition of a covariance matrix

- Implement Gaussian sampling through Cholesky and check the empirical covariance

- Experiment: recurrent matrix with spectral radius 0.9, 1.0, 1.1 and plot hidden-state and gradient norms







## **09 Matrix Decomposition**



- Why decompose: solving, sampling, compression, regression

- Cholesky $A = L L^T$ for positive definite matrices: solving systems, sampling, $log det$

- Recap of LU (05), QR (07) and eigendecomposition (08), and when to use each

- SVD $A = U S V^T$: definition, existence, geometry as rotate, stretch, rotate

- Deriving the SVD from the eigendecomposition of $A^T A$ and $A A^T$

- Full, thin and truncated SVD

- Link to the four subspaces: columns of `U` and `V` give bases for them

- Eckart-Young: best rank-`k` approximation; error $sigma_{k+1}$ in spectral norm, $sqrt(sum of sigma_i^2 for i > k)$ in Frobenius norm

- Choosing `k` from a singular-value plot

- PCA as the SVD of centered data, explained variance $sigma_i^2 / (n - 1)$, whitening, reconstruction

- Matrix norms: Frobenius, spectral, nuclear, induced norms

- Condition number $sigma_max / sigma_min$ and what it says about error amplification

- Pseudo-inverse and least squares through SVD; ridge as shrinkage $sigma^2 / (sigma^2 + lambda)$

- Effective rank: stable rank and spectral-entropy rank

- Measure, do not assume, low-rank structure: attention matrices and weight matrices often have fast-decaying singular values, but it varies by layer and model

- Compute the condition number of weight matrices layer by layer in a small trained model and compare layer types

- Low-rank structure in modern AI: LoRA $W + (alpha / r) B A$, `B` initialized to zero, trainable parameters $r(d_in + d_out)$ instead of $d_in * d_out$

- Embedding factorization, weight compression, low-rank KV compression (know it exists)

- Polar decomposition and its link to orthogonalized optimizer updates (advanced)

- Randomized SVD and non-negative matrix factorization (optional)

- Implement image compression by truncated SVD with an error-versus-rank plot

- Implement PCA from scratch and match `sklearn`

- Implement a `LoRALinear` layer in PyTorch with merge and unmerge

- Plot singular-value spectra of a small pretrained model's weights







## **10 Tensors**



- Scalar, vector, matrix, tensor: order, shape, axes

- Strides, views versus copies, contiguity; when `view` fails and `reshape` copies

- Reshape, transpose, permute, squeeze, unsqueeze

- Broadcasting rules and the bugs they hide

- Batched matrix multiplication, outer products, `tensordot`

- Einsum: notation, contraction, batching, ellipsis, reading and writing expressions

- Multi-head attention shape algebra: `[B, T, D] -> [B, T, H, D/H] -> [B, H, T, D/H]`, scores `[B, H, T, T]`

- Gather and scatter; embedding lookup as one-hot times a matrix

- Kronecker product (know it exists), tensor decompositions CP and Tucker (know they exist)

- Parameter, FLOP and memory counting for a transformer block
- Spectral norm and operator norm in deep learning; connections to amplification, Lipschitz bounds and stability (introductory)

- Implement multi-head attention with only `einsum`, with a causal mask, and match $F.scaled_dot_product_attention$

- Write a shape-assertion helper and use it at every layer boundary

- Practice reading code and predicting every intermediate shape without running it

- Check which axis softmax runs over in attention: the last axis (keys) is correct, the query axis still gives a plausible-looking loss curve

- Apply masks to the logits before softmax, using `-inf` or $masked_fill$, never after

- Predict how a `[B, 1, 1, T]` padding mask broadcasts against `[B, H, T, T]` scores, and what a wrongly shaped mask silently does instead of raising an error







## **11 Numerical Linear Algebra (new folder, add it)**



- Floating-point formats: fp32, fp16, bf16, fp8; machine epsilon, overflow, underflow

- Catastrophic cancellation and error growth in long sums; pairwise and Kahan summation

- Conditioning (a property of the problem) versus stability (a property of the algorithm)

- Error bound for solving $Ax = b$ in terms of the condition number

- Stable softmax and log-sum-exp; stable cross-entropy from logits

- Mixed precision basics: fp32 master weights, loss scaling for fp16, why bf16 usually needs none

- Avoiding explicit inverses

- Sparse matrices and conjugate gradient (optional)
- Iterative solvers beyond conjugate gradient: when direct factorization is too expensive; preconditioning as an idea (optional)

- FFT and circulant matrices (optional)

- Graph Laplacian and spectral clustering (optional)

- Implement a naive softmax that breaks, then fix it

- Sum 10^7 numbers in fp16, bf16, fp32, fp64 and compare errors

- Hilbert-matrix experiment: solution error versus condition number








---

## **12 Matrix Calculus for Machine Learning (add this folder)**

- Scalar, vector and matrix derivatives: distinguish gradients, Jacobians and Hessians by shape
- Derivatives of linear maps: \(y = Ax\), \(y = x^T A\), and affine layers \(y = Wx + b\)
- Gradient shape rules: gradients have the shape of the variable being differentiated
- Derivatives of dot products, quadratic forms \(x^T A x\), and squared error
- Frobenius inner product and matrix gradients
- Chain rule for compositions of vector-valued functions
- Jacobian-vector products (JVPs) and vector-Jacobian products (VJPs); why reverse-mode autodiff uses VJPs
- Backpropagation through a linear layer: derive gradients for \(W\), \(b\), and the input
- Batch dimensions and reductions: why gradients for shared weights sum over batch dimensions
- Gradients of least squares and ridge regression; connect to the solutions in folders 07 and 09
- Derive the softmax-cross-entropy gradient \(p-y\), connecting to folder 08 and the formula sheet
- Verify hand-derived gradients with finite differences and PyTorch autograd
- Implement a tiny two-layer network and compare analytical gradients, finite differences and autograd

## **Order and pacing**



| Step | Folder | Weeks |
|---|---|---|
| Done | 01 Scalars and Basics | revise only |
| Done | 02 Vectors | revise only |
| Done | 03 Matrices | revise only |
| 1 | 04 Matrix Operations | 1 |
| 2 | 05 Determinants and Inverses | 1.5 |
| 3 | 06 Vector Spaces | 1.5 |
| 4 | 07 Linear Transformations | 2 |
| 5 | 08 Eigenvalues and Eigenvectors | 2 |
| 6 | 09 Matrix Decomposition | 2.5 |
| 7 | 10 Tensors | 1 |
| 8 | 11 Numerical Linear Algebra | 1 |
| 9 | 12 Matrix Calculus for Machine Learning | 1 |

