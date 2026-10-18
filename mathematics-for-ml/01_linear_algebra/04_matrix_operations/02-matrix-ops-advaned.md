# **Matrix Operations for Deep Learning, Machine Learning, and Large Language Models**

## **0. Start Here — Beginner Primer**

**How these notes are organised.** Every section now has two layers:

- **Beginner Track:** One idea at a time, with small numbers and every step worked by hand. Read these first, in order, to build the foundations.
- **Advanced Track:** Proofs, LLM systems, and hardware analysis. Corrections to the original are listed in Appendix B; revisit this layer on a second pass.

All hand-worked numbers are verified by the script in Appendix A.

### **0.1 What a matrix is**

$$A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$$

- Shape is **rows × columns**, always in that order: this $A$ is $2\times3$.
- $A_{12}=2$ means row 1, column 2. Row 2 is $[4\ 5\ 6]$. Column 3 is $\begin{bmatrix}3\\6\end{bmatrix}$.
- A **vector** is a matrix with one column ($n\times1$) or one row ($1\times n$).

### **0.2 The dot product — the atom of everything**

$$[1,2,3]\cdot[4,5,6]=1\cdot4+2\cdot5+3\cdot6=4+10+18=32$$

Multiply matching entries, then add. Every matrix product in this module is just a table of dot products.

### **0.3 Where matrices live in a model**

| object | shape | example |
|---|---|---|
| a batch of token vectors $X$ | tokens × features | 5 tokens × 4096 features |
| a layer's weights $W$ | features in × features out | 4096 × 11008 |
| attention scores $S=QK^\top$ | tokens × tokens | 2048 × 2048 per head |
| one linear layer | $Y = XW + \mathbf{b}$ | one matmul plus one bias add |

### **0.4 Math → NumPy translation**

| math | NumPy | warning |
|---|---|---|
| $A+B$ | `A + B` | same shape required |
| $\alpha A$ | `alpha * A` | |
| $A\odot B$ (Hadamard) | `A * B` | **not** the matrix product |
| $AB$ | `A @ B` | inner dimensions must match |
| $A^\top$ | `A.T` | |
| $\mathrm{tr}(A)$ | `np.trace(A)` | square matrices only |
| shape | `A.shape` | print it constantly |

### **0.5 Glossary (read once, return when stuck)**

| term | plain meaning |
|---|---|
| **FLOP** | one floating-point add or multiply. "FLOP/s" is speed; "FLOPs" is a count. |
| **GEMM / GEMV** | general matrix–matrix / matrix–vector multiply. |
| **FMA** | fused multiply-add: $a\cdot b + c$ in one hardware step (counted as 2 FLOPs). |
| **batch** | how many examples or tokens you process at once. |
| **HBM / SRAM** | big-but-slow GPU memory / tiny-but-fast on-chip memory. |
| **memory-bound** | the chip waits for data; adding arithmetic speed would not help. |
| **compute-bound** | the chip's arithmetic units are the bottleneck. |
| **arithmetic intensity** | FLOPs done per byte moved. High = compute-bound, low = memory-bound. |
| **roofline / ridge point** | the intensity where a chip switches from memory-bound to compute-bound. |
| **KV cache** | stored keys and values of earlier tokens so decoding does not recompute them. |
| **epilogue** | extra work (bias, activation) fused onto the end of a GEMM kernel. |
| **contiguous** | neighbouring elements sit next to each other in memory. |

### **0.6 Reading order**

| sections | theme | why it matters |
|---|---|---|
| 1–3 | add, multiply, laws | the algebra every layer uses |
| 4–5 | transpose, trace | backpropagation and the loss function |
| 6–8 | Hadamard, blocks, powers, special matrices | gating, serving, structure that makes things cheap |
| 9–13 | FLOPs, bytes, roofline, layout, bias | why the same math runs fast or slow |
| 14–15 | measurements and reference numbers | evidence and cheat-sheet |

---

#### **add and scale by hand**

**Rule.** Matrices of the **same shape** add entry by entry. A scalar multiplies every entry.

Let $A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$ and $B=\begin{bmatrix}5&6\\7&8\end{bmatrix}$.

$$A+B=\begin{bmatrix}1+5&2+6\\3+7&4+8\end{bmatrix}=\begin{bmatrix}6&8\\10&12\end{bmatrix},\qquad 2A=\begin{bmatrix}2&4\\6&8\end{bmatrix}$$

$$3A-B=\begin{bmatrix}3&6\\9&12\end{bmatrix}-\begin{bmatrix}5&6\\7&8\end{bmatrix}=\begin{bmatrix}-2&0\\2&4\end{bmatrix}$$

**Check a law.** $(2+3)A=5A=\begin{bmatrix}5&10\\15&20\end{bmatrix}$, and $2A+3A=\begin{bmatrix}2&4\\6&8\end{bmatrix}+\begin{bmatrix}3&6\\9&12\end{bmatrix}=\begin{bmatrix}5&10\\15&20\end{bmatrix}$ 

**Shape rule.** A $2\times2$ plus a $2\times3$ is *undefined*: the third column has no partner.

**Gradient accumulation, $k=2$ micro-batches.** $g_1=\begin{bmatrix}2&4\\6&8\end{bmatrix}$, $g_2=\begin{bmatrix}4&0\\2&6\end{bmatrix}$:

$$\bar g=\tfrac12(g_1+g_2)=\tfrac12\begin{bmatrix}6&4\\8&14\end{bmatrix}=\begin{bmatrix}3&2\\4&7\end{bmatrix}$$

**One weight-decay step.** $\theta=\begin{bmatrix}1&-2\\0.5&4\end{bmatrix}$, $g=\begin{bmatrix}0.2&0.4\\-0.1&0\end{bmatrix}$, $\eta=0.1$, $\lambda=0.01$, so the shrink factor is $1-\eta\lambda=0.999$:

$$\theta'=0.999\,\theta-0.1\,g=\begin{bmatrix}0.999&-1.998\\0.4995&3.996\end{bmatrix}-\begin{bmatrix}0.02&0.04\\-0.01&0\end{bmatrix}=\begin{bmatrix}0.979&-2.038\\0.5095&3.996\end{bmatrix}$$

**Trap.** School algebra says $(a+b)^2=a^2+2ab+b^2$. For matrices that is **false** (computed in §3).

**Remember:** add = same shape, entrywise. Scale = every entry. Averaging gradients is just *add, then scale by $1/k$*.

## **1. Matrix Addition and Scalar Multiplication**

### **1.1 The laws and their practical consequences**

Addition and scalar multiplication are componentwise:

$$
A + B = \begin{bmatrix} a_{ij} + b_{ij} \end{bmatrix}, \qquad
\alpha A = \begin{bmatrix} \alpha\, a_{ij} \end{bmatrix}
$$

and they satisfy the eight ring laws, of which four do real work in deep learning:

$$
(\alpha + \beta)A = \alpha A + \beta A, \qquad
\alpha(A + B) = \alpha A + \alpha B
$$

$$
(\alpha\beta)A = \alpha(\beta A), \qquad
A + A = 2A
$$

Crucially, **scalar multiplication distributes over matrix addition, but matrix addition does not distribute over matrix multiplication in the way learners often assume.** $A(B + C) = AB + AC$ is true (see §3), and so is $(A+B)C = AC + BC$. But $A(B+C) \neq (A+B)(C+D)$ — there is no "matrix scalar" analogue of multiplying out.

**Worked example 1 — the residual stream.** A pre-norm transformer block computes

$$
\mathbf{y} = \mathbf{x} + F(\mathbf{x})
$$

where $\mathbf{x} \in \mathbb{R}^{n\times d}$ is the residual stream and $F$ is the block. In LLaMA's pre-norm arrangement, $F(\mathbf{x}) = \mathrm{Attn}(\mathrm{RMSNorm}(\mathbf{x}))$, and the attention output is projected back to $d$ dimensions, so the two summands genuinely share a shape and the addition is a genuine matrix addition. The add is not free: it is a full $n\times d$ read-modify-write of the residual stream, one of the reasons activations dominate training memory.

**Worked example 2 — gradient accumulation.** In data-parallel training with gradient accumulation over $k$ micro-batches, each micro-batch produces a gradient $g_i \in \mathbb{R}^{d_{\text{model}}\times d_{\text{model}}}$, and the optimizer step uses

$$
\bar{g} = \frac{1}{k}\sum_{i=1}^{k} g_i = \frac{1}{k}\big(g_1 + g_2 + \cdots + g_k\big)
$$

This is exactly scalar multiplication followed by repeated matrix addition, and it is why loss scaling and `loss / accumulation_steps` appear in every trainer: without the $\tfrac1k$, the effective learning rate scales with the number of micro-batches.

**Worked example 3 — weight decay.** Decoupled weight decay (as in AdamW) applies

$$
\theta \leftarrow (1 - \lambda)\,\theta - \eta\, g
$$

which is a scalar multiply of the weight matrix plus a scaled gradient. Decoupled decay is preferred over L2-regularisation (adding $\lambda\theta$ to the gradient) because in the latter the decay term gets mangled by Adam's per-parameter normalisation — the $\lambda\theta$ contribution becomes $\lambda\theta/(\sqrt{v}+\epsilon)$, whose magnitude depends on the gradient history rather than on the weight itself.

**Worked example 4 — averaging over ranks and over time.** In DDP and FSDP, gradients are all-reduced across ranks by summing and then dividing by the world size $W$:

$$
g_{\text{global}} = \frac{1}{W}\sum_{r=1}^{W} g^{(r)}
$$

The summation is a ring all-reduce (logarithmic in $W$ steps, and the *order* of summation varies by topology, which as §3.4 shows changes the last bits of the result). Stochastic Weight Averaging and EMA checkpoints average over time instead, combining $T$ snapshots as $\bar{\theta} = \frac{1}{T}\sum_{t=1}^{T}\theta^{(t)}$, with a running EMA of the form $\theta_t^{\text{EMA}} = \beta\theta_{t-1}^{\text{EMA}} + (1-\beta)\theta_t$ being the same operation with geometrically decaying weights. Both are a scalar multiply plus an add per step, and both are why the averaging order is worth thinking about before a training run is compared against another.

---

#### **the shape rule and one full product by hand**

**Rule.** $(m\times k)(k\times n)\to(m\times n)$. The two inner numbers must match and then disappear.

| product | shapes | result |
|---|---|---|
| $A B$ | $(2\times3)(3\times2)$ | $2\times2$  |
| $B A$ | $(3\times2)(2\times3)$ | $3\times3$  (different matrix!) |
| $A A$ | $(2\times3)(2\times3)$ |  inner dims 3 ≠ 2 |

**Worked product.** $A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$, $B=\begin{bmatrix}7&8\\9&10\\11&12\end{bmatrix}$. Each entry is a dot product of a row of $A$ with a column of $B$:

$$\begin{aligned}
C_{11}&=1\cdot7+2\cdot9+3\cdot11=7+18+33=58\\
C_{12}&=1\cdot8+2\cdot10+3\cdot12=8+20+36=64\\
C_{21}&=4\cdot7+5\cdot9+6\cdot11=28+45+66=139\\
C_{22}&=4\cdot8+5\cdot10+6\cdot12=32+50+72=154
\end{aligned}
\qquad C=\begin{bmatrix}58&64\\139&154\end{bmatrix}$$

**Counting work.** $m=2,n=2,k=3$: multiplications $mnk=12$, additions $mn(k-1)=8$, total $20=2mnk-mn$ (the notes round this to $2mnk=24$).

**The same $C$, read four ways** (this is §2.2 made concrete):

| reading | calculation | result |
|---|---|---|
| 1. dot products | the table above | $C$ |
| 2. column combos | col 1 of $C$ $=7\begin{bmatrix}1\\4\end{bmatrix}+9\begin{bmatrix}2\\5\end{bmatrix}+11\begin{bmatrix}3\\6\end{bmatrix}=\begin{bmatrix}58\\139\end{bmatrix}$; col 2 $=8\begin{bmatrix}1\\4\end{bmatrix}+10\begin{bmatrix}2\\5\end{bmatrix}+12\begin{bmatrix}3\\6\end{bmatrix}=\begin{bmatrix}64\\154\end{bmatrix}$ | $C$ |
| 3. row combos | row 1 of $C$ $=1[7\ 8]+2[9\ 10]+3[11\ 12]=[58\ \ 64]$; row 2 $=4[7\ 8]+5[9\ 10]+6[11\ 12]=[139\ \ 154]$ | $C$ |
| 4. outer products | $\begin{bmatrix}1\\4\end{bmatrix}[7\ 8]+\begin{bmatrix}2\\5\end{bmatrix}[9\ 10]+\begin{bmatrix}3\\6\end{bmatrix}[11\ 12]=\begin{bmatrix}7&8\\28&32\end{bmatrix}+\begin{bmatrix}18&20\\45&50\end{bmatrix}+\begin{bmatrix}33&36\\66&72\end{bmatrix}$ | $\begin{bmatrix}58&64\\139&154\end{bmatrix}$ |

**A linear layer by hand.** One token $x=[1,\,2]$ (2 features) through $W=\begin{bmatrix}1&0&-1\\2&1&0\end{bmatrix}$ (2 in → 3 out):

$$xW=[\,1\cdot1+2\cdot2,\ \ 1\cdot0+2\cdot1,\ \ 1\cdot(-1)+2\cdot0\,]=[5,\ 2,\ -1]$$

Add a second token $[3,-1]$ to make a batch $X=\begin{bmatrix}1&2\\3&-1\end{bmatrix}$. Row 2 of the output is $[3\cdot1+(-1)\cdot2,\ 3\cdot0+(-1)\cdot1,\ 3\cdot(-1)+0]=[1,-1,-3]$, so

$$XW=\begin{bmatrix}5&2&-1\\1&-1&-3\end{bmatrix}$$

Each token is processed independently — the batch is just more rows.

**Attention scores by hand.** $Q=\begin{bmatrix}1&0\\1&1\end{bmatrix}$, $K=\begin{bmatrix}1&2\\0&1\end{bmatrix}$, so $K^\top=\begin{bmatrix}1&0\\2&1\end{bmatrix}$ and

$$S=QK^\top=\begin{bmatrix}1\cdot1+0\cdot2&1\cdot0+0\cdot1\\1\cdot1+1\cdot2&1\cdot0+1\cdot1\end{bmatrix}=\begin{bmatrix}1&0\\3&1\end{bmatrix}$$

Entry $S_{ij}=q_i\cdot k_j$: "how much does token $i$'s query match token $j$'s key". Scale by $1/\sqrt{d_{\text{head}}}=1/\sqrt2$: row 1 becomes $[0.707,\ 0]$; softmax gives $\frac{[e^{0.707},\,e^{0}]}{e^{0.707}+e^{0}}=\frac{[2.028,\,1]}{3.028}=[0.670,\ 0.330]$.

**Remember:** shape check first ($m\times\mathbf{k}$ times $\mathbf{k}\times n$), then each output entry is one dot product. The four readings are one calculation grouped four ways.

## **2. Matrix Multiplication and the Shape Rule**

### **2.1 The shape rule**

For $A \in \mathbb{R}^{m\times k}$ and $B \in \mathbb{R}^{k\times n}$, the product $C = AB \in \mathbb{R}^{m\times n}$ is defined entrywise by contracting the shared index $k$:

$$
C_{ij} = \sum_{\ell=1}^{k} A_{i\ell}\, B_{\ell j}
$$

The rule is a bookkeeping rule, not a coincidence: the inner dimensions must agree because the summation index is contracted. In code, `A @ B` in NumPy/PyTorch enforces it for you; in a hand-written kernel it is your job.

**Worked example — a real transformer projection.** In LLaMA-2-7B, $d_{\text{model}} = 4096$, $n_{\text{heads}} = 32$, so $d_{\text{head}} = 128$, and $d_{\text{ff}} = 11008$. With a batch of $B$ sequences of $T$ tokens, the hidden state $X \in \mathbb{R}^{BT\times 4096}$ produces

$$
Q = XW^Q \in \mathbb{R}^{BT\times 4096}, \qquad
K = XV^K,\qquad V = XV^V
$$

where $W^Q \in \mathbb{R}^{4096\times 4096}$. The shape rule is exercised in the tightest way in the attention scores:

$$
S = QK^\top \in \mathbb{R}^{(B\cdot h)\times T\times T}, \qquad
S = \frac{QK^\top}{\sqrt{d_{\text{head}}}}
$$

with $Q \in \mathbb{R}^{(B h)\times T\times 128}$ and $K^\top \in \mathbb{R}^{(B h)\times 128\times T}$. Note that this is a *batched* product: 3-tensors contracted in their last two axes, with a batch axis. The shape rule extends unchanged to batches, which is why `torch.matmul` broadcasts over leading dimensions.

### **2.2 Four readings of the same product**

The formula $C_{ij} = \sum_\ell A_{i\ell}B_{\ell j}$ is one reading. There are at least four, all algebraically identical, all with the same operation count ($mnk$ multiplications and $mn(k-1)$ additions) and all differing in *dependency structure*. That difference is not cosmetic: it is what determines which kernel gets written. The taxonomy below follows the standard four-view presentation of the matrix product [32][33][34].

**Reading 1 — entry as an inner product.** $C_{ij} = \mathbf{a}_i^\top \mathbf{b}_j$, the dot product of row $i$ of $A$ with column $j$ of $B$. One output entry at a time. This is the reading used in proofs, and it is the reading a naive `for i / for j / for k` loop implements.

**Reading 2 — each column of $C$ is a linear combination of the columns of $A$.**

$$
C_{:,j} = A\,B_{:,j} = \sum_{\ell=1}^{k} B_{\ell j}\, A_{:,\ell}
$$

Read aloud: *apply $A$ to each column of $B$ separately, then stack the results.* This is the reading that makes a linear layer intuitive: the batch of token vectors is $B$, and $A$ is the layer. It is also the reading that makes batch-1 inference a matrix-vector product (§9).

**Reading 3 — each row of $C$ is a linear combination of the rows of $B$.**

$$
C_{i,:} = A_{i,:}\,B = \sum_{\ell=1}^{k} A_{i\ell}\, B_{\ell,:}
$$

This is the reading used by the *backward* pass of a linear layer (§4.2), and by attention heads, which mix across token positions.

**Reading 4 — $AB$ is a sum of rank-1 outer products.**

$$
AB = \sum_{\ell=1}^{k} A_{:,\ell}\, B_{\ell,:}
$$

Each term is an outer product of a column of $A$ with a row of $B$, and has rank 1. So $AB$ is a sum of $k$ rank-1 matrices, and $\mathrm{rank}(AB) \le \min(\mathrm{rank}(A), \mathrm{rank}(B))$ follows immediately. This is the reading that LAPACK and BLIS actually implement: their micro-kernels accumulate $C \mathrel{+}= A_{i\ell} B_{\ell j}$, one rank-1 update at a time.

The four readings are verified numerically on a $(128\times 256)(256\times 64)$ product in §14.3; all four agree with the reference `A @ B` to $9.2\times 10^{-14}$ or better.

### **2.3 Each reading in a running system**

- **Reading 2** is the `nn.Linear` in a transformer. `F.linear(X, W)` is exactly $XW$: each column of $X$ (one token's feature vector) is mapped independently. This is why a linear layer over a batch of $T$ tokens is *identical in cost* to $T$ separate matrix-vector products — and why the batch dimension is free in FLOP terms but not in memory terms.
- **Reading 4** is the BLIS/GotoBLAS2 micro-kernel. The published algorithm keeps a small $b_c \times b_c$ panel of $C$ in registers, and for each $k$ step performs a rank-1 update of that panel. The "outer-product" reading is not an academic curiosity; it is the inner loop of every BLAS on earth.
- **Reading 1** is the dot-product form used by attention scores $QK^\top$, where the $k$ being contracted is exactly the head dimension $d_{\text{head}}$.
- **Reading 3** is the gradient of a linear layer: $\nabla_X L = \delta W^\top$ expresses each row of the input-gradient as a combination of rows of $W^\top$.

**Reading 4 also explains Strassen.** Strassen's 2×2 scheme computes a product of 2×2 matrices with 7 multiplications instead of 8, by writing $AB$ as a sum of 7 outer products of *linear combinations* of the blocks of $A$ and $B$. Formally, the $\langle 2,2,2\rangle$ matrix-multiplication tensor has rank 7 rather than 8, and $R(\langle 2,2,2\rangle) \le 7$ is now known to be tight [1]. Practically it is almost never used for deep learning: the extra additions destroy numerical stability, and the constant factors mean it only pays off at sizes far beyond a single accelerator's memory [2][3].

---

#### **the three laws with real numbers**

Use $A=\begin{bmatrix}1&2\\3&4\end{bmatrix}$, $B=\begin{bmatrix}5&6\\7&8\end{bmatrix}$.

**Non-commutativity: $AB\ne BA$.**

$$AB=\begin{bmatrix}1\cdot5+2\cdot7&1\cdot6+2\cdot8\\3\cdot5+4\cdot7&3\cdot6+4\cdot8\end{bmatrix}=\begin{bmatrix}19&22\\43&50\end{bmatrix},\qquad
BA=\begin{bmatrix}5\cdot1+6\cdot3&5\cdot2+6\cdot4\\7\cdot1+8\cdot3&7\cdot2+8\cdot4\end{bmatrix}=\begin{bmatrix}23&34\\31&46\end{bmatrix}$$

**Why $(A+B)^2\neq A^2+2AB+B^2$.** Expanding honestly gives $A^2+AB+BA+B^2$.

$$A+B=\begin{bmatrix}6&8\\10&12\end{bmatrix}\ \Rightarrow\ (A+B)^2=\begin{bmatrix}36+80&48+96\\60+120&80+144\end{bmatrix}=\begin{bmatrix}116&144\\180&224\end{bmatrix}$$

With $A^2=\begin{bmatrix}7&10\\15&22\end{bmatrix}$ and $B^2=\begin{bmatrix}67&78\\91&106\end{bmatrix}$:

- $A^2+AB+BA+B^2=\begin{bmatrix}7+19+23+67&10+22+34+78\\15+43+31+91&22+50+46+106\end{bmatrix}=\begin{bmatrix}116&144\\180&224\end{bmatrix}$ 
- $A^2+2AB+B^2=\begin{bmatrix}112&132\\192&228\end{bmatrix}$  (wrong)

**Associativity: $(AB)v=A(Bv)$.** Take $v=\begin{bmatrix}1\\1\end{bmatrix}$.

$$(AB)v=\begin{bmatrix}19&22\\43&50\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix}=\begin{bmatrix}41\\93\end{bmatrix},\qquad
Bv=\begin{bmatrix}11\\15\end{bmatrix},\ \ A(Bv)=\begin{bmatrix}11+30\\33+60\end{bmatrix}=\begin{bmatrix}41\\93\end{bmatrix}\ $$

**Distributivity: $A(B+I)=AB+A$.**

$$A(B+I)=\begin{bmatrix}1&2\\3&4\end{bmatrix}\begin{bmatrix}6&6\\7&9\end{bmatrix}=\begin{bmatrix}20&24\\46&54\end{bmatrix},\qquad AB+A=\begin{bmatrix}19&22\\43&50\end{bmatrix}+\begin{bmatrix}1&2\\3&4\end{bmatrix}=\begin{bmatrix}20&24\\46&54\end{bmatrix}\ $$

**Associativity is a cost lever.** Same answer, very different work. $A$ is $10\times100$, $B$ is $100\times5$, $C$ is $5\times50$:

| order | step 1 | step 2 | total multiplications |
|---|---|---|---|
| $(AB)C$ | $10\cdot100\cdot5=5{,}000$ | $10\cdot5\cdot50=2{,}500$ | **7,500** |
| $A(BC)$ | $100\cdot5\cdot50=25{,}000$ | $10\cdot100\cdot50=50{,}000$ | **75,000** |

A 10× difference from parentheses alone.

**Floating point breaks associativity.** In Python: `(0.1 + 0.2) + 0.3` gives `0.6000000000000001`, but `0.1 + (0.2 + 0.3)` gives `0.6`. Same numbers, different grouping, different last bits.

**Remember:** you may *regroup* a product but never *reorder* it, and regrouping is free in algebra but not in cost or in the last floating-point bits.

## **3. Associativity, Distributivity, and Non-Commutativity**

### **3.1 The three laws**

For conformable matrices:

$$
(AB)C = A(BC) \qquad\text{(associativity)}
$$

$$
A(B + C) = AB + AC, \qquad (A + B)C = AC + BC \qquad\text{(distributivity)}
$$

$$
AB \neq BA \qquad\text{in general} \qquad\text{(failure of commutativity)}
$$

Associativity is what lets you parenthesise a deep product any way you like; it is the licence to reorder a neural network's matmuls. Distributivity is what lets you push gradients back through a sum (the chain rule for a residual connection). Commutativity fails, and that failure is a design constraint, not a curiosity.

### **3.2 A non-commutative counterexample at 4096 scale**

Textbook counterexamples use 2×2 matrices. Here is a real-scale one, measured on this machine: two $4096\times 4096$ matrices with standard initialisation statistics ($\mathcal{N}(0, 1/d)$), i.e. LLaMA-2-7B weight shapes.

```
max|A@B - B@A|   = 1.310443e-01
max|A@B|         = 8.438757e-02
relative gap     = 1.5529
```

The two products differ by *more* than the magnitude of the products themselves. Swapping the order of two weight matrices is not a rounding-level perturbation; it computes a different function.

Real anchors for non-commutativity:

- **Attention is not symmetric.** $\mathrm{softmax}(QK^\top/\sqrt d)$ and $\mathrm{softmax}(KQ^\top/\sqrt d)$ are different matrices (measured max difference $0.0199$ at a realistic logit scale, $n=512$, $d_h=128$). This is why you cannot swap query and key projections, and why "attention is a symmetric operator" is false.
- **Matrix absorption in DeepSeek-V2's MLA.** The up-projection $W^{UK}$ can be folded into the query projection $W^Q$ because $(R_m\mathbf{q})^\top(R_n\mathbf{k}) = \mathbf{q}^\top R_{n-m}\mathbf{k}$ and the rotations are orthogonal — but only if you apply the absorption in a specific order. The hardware analysis of MLA distinguishes a left-to-right ordering from a middle-product-first ordering, and these have *different* performance characteristics because they move different matrices across the memory bus [4]. Reordering a product of three matrices is associative; which reordering wins is a hardware question, and the answer differs per accelerator.
- **LoRA's factor order.** In $\Delta W = BA$ with $B \in \mathbb{R}^{d\times r}$, $A \in \mathbb{R}^{r\times k}$, the order $BA$ and $AB$ are not even conformable unless $r = d = k$. The low-rank structure is *order-specific* [5].

### **3.3 Associativity as a performance lever**

Because the product is associative, the parenthesisation of a chain of matmuls is free to choose — and the right choice can change the cost by orders of magnitude.

**Example A — einsum path optimisation.** `opt_einsum` treats "find the best order to contract this expression" as a search problem and reports that finding the *optimal* path is NP-hard, offering `optimal`, `dp` (dynamic programming), `branch` and `greedy` strategies [6]. Its dynamic-programming path finder explicitly begins by computing *all traces*, meaning summations over indices appearing in exactly one input, before decomposing the contraction graph [7]. Two things are worth noticing. A research-grade software library uses the trace as a structural primitive, and the library's `'limit'` objective minimises $\max(\text{flops}, \alpha\cdot\text{write})$, explicitly favouring paths with *higher arithmetic intensity* because "real contraction performance can be bound by memory speed." The roofline model of §11 is built into the cost model of mainstream tensor software.

**Example B — DeepSeek-V2's MLA absorption ordering.** As above: the same mathematical product, evaluated in different orders, has different measured performance because the orders have different memory-traffic profiles [4].

**Example C — LoRMA.** Low-Rank Multiplicative Adaptation reorders the matrix multiplications in the LoRA update, writing $W = \frac{\alpha}{r}(BA)W_0$ and explicitly re-ordering operations to address the computational complexity of the multiplicative update [8]. Associativity is what makes the reorder legal; the whole point of the paper is that a legal reorder can save compute.

---

#### **transpose by hand, and why backprop needs it**

**Transpose = flip rows and columns.** $A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}\ (2\times3)\ \Rightarrow\ A^\top=\begin{bmatrix}1&4\\2&5\\3&6\end{bmatrix}\ (3\times2)$.

**Check $(AB)^\top=B^\top A^\top$** using the §2 product, $C=AB=\begin{bmatrix}58&64\\139&154\end{bmatrix}$ so $C^\top=\begin{bmatrix}58&139\\64&154\end{bmatrix}$.

$B^\top=\begin{bmatrix}7&9&11\\8&10&12\end{bmatrix}$ $(2\times3)$ and $A^\top$ $(3\times2)$, so $B^\top A^\top$ is $2\times2$:

$$B^\top A^\top=\begin{bmatrix}7\cdot1+9\cdot2+11\cdot3&7\cdot4+9\cdot5+11\cdot6\\8\cdot1+10\cdot2+12\cdot3&8\cdot4+10\cdot5+12\cdot6\end{bmatrix}=\begin{bmatrix}58&139\\64&154\end{bmatrix}=C^\top\ $$

**Why the order must reverse.** $A^\top B^\top$ would be $(3\times2)(2\times3)=3\times3$ — the wrong shape entirely. Only $B^\top A^\top$ even fits.

**Backprop through one linear layer, fully by hand.** Forward: $y=xW$ with $x=[1,\,2]$ $(1\times2)$ and $W=\begin{bmatrix}1&0&-1\\2&1&0\end{bmatrix}$ $(2\times3)$, giving $y=[5,2,-1]$. Suppose the upstream gradient is $\delta=\partial L/\partial y=[1,\,0,\,-1]$ $(1\times3)$.

$$\frac{\partial L}{\partial x}=\delta W^\top=[1,0,-1]\begin{bmatrix}1&2\\0&1\\-1&0\end{bmatrix}=[1+1,\ 2+0]=[2,\ 2]\quad(1\times2)$$

$$\frac{\partial L}{\partial W}=x^\top\delta=\begin{bmatrix}1\\2\end{bmatrix}[1,0,-1]=\begin{bmatrix}1&0&-1\\2&0&-2\end{bmatrix}\quad(2\times3)$$

**Sanity check without any rule.** Take $L=\delta\cdot y=y_1-y_3=(x_1+2x_2)-(-x_1)=2x_1+2x_2$. So $\partial L/\partial x=[2,2]$ . And $\partial L/\partial W_{ij}=x_i\,\delta_j$, e.g. $W_{13}$: $1\cdot(-1)=-1$ .

**Shape bookkeeping to memorise:** each gradient has the **same shape as the thing it is a gradient of** — $\nabla x$ is $1\times2$, $\nabla W$ is $2\times3$. If your shapes don't match, you transposed the wrong thing.

**Remember:** $(AB)^\top=B^\top A^\top$; backward pass = two matmuls with transposed operands: $\delta W^\top$ for the input, $x^\top\delta$ for the weight.

## **4. Transpose Rules**

### **4.1 The rules and their proofs**

$$
(A + B)^\top = A^\top + B^\top, \qquad (\alpha A)^\top = \alpha A^\top, \qquad (A^\top)^\top = A
$$

$$
\boxed{(AB)^\top = B^\top A^\top}
$$

The last one is the one that matters, and it has a one-line proof from the definition:

$$
(AB)^\top_{ij} = (AB)_{ji} = \sum_{\ell=1}^{k} A_{j\ell}B_{\ell i} = \sum_{\ell=1}^{k} B_{\ell i}A_{j\ell} = (B^\top A^\top)_{ij}
$$

Note that the order reverses *and* the scalar multiplication is commutative inside the sum. If the underlying ring were non-commutative, the reversal would be a conjugation instead: $(AB)^\top = B^\top A^\top$ still holds, but products of complex matrices require $\overline{B}^\top$.

### **4.2 Backpropagation through a linear layer**

This is the reason the transpose rule exists in a deep-learning course. Let the layer be $\mathbf{y} = XW$ with $X \in \mathbb{R}^{m\times k}$, $W \in \mathbb{R}^{k\times n}$, and let $\delta = \partial L/\partial \mathbf{y} \in \mathbb{R}^{m\times n}$. Then

$$
\frac{\partial L}{\partial X} = \delta W^\top \in \mathbb{R}^{m\times k}, \qquad
\frac{\partial L}{\partial W} = X^\top \delta \in \mathbb{R}^{k\times n}
$$

Both gradients are *transposed products* — one of them reuses the forward weight matrix in transposed form, the other reuses the input in transposed form. This is why a framework's backward pass for a linear layer is a pair of GEMMs with swapped operands: forward is $(m\times k)(k\times n)$, backward-input is $(m\times n)(n\times k)$, backward-weight is $(k\times m)(m\times n)$.

This has a concrete memory consequence: the backward-input gradient needs $W^\top$, and the backward-weight gradient needs $X^\top$. The framework either materialises transposed copies (extra $kn$ and $mk$ elements of memory) or asks the GEMM library for the transposed layout directly. cuBLAS exposes the transpose flags as part of the operation spec, so "transpose for free" is a real and frequently-used option.

### **4.3 Transposed products and memory layout**

$(AB)^\top = B^\top A^\top$ is not only a proof technique; it is a *layout* decision with a performance consequence.

Attention computes $QK^\top$. With row-major storage (numpy, PyTorch default), $Q$ is stored row-major, so $QK^\top$ is an "NT" product: the left operand is row-major, the right operand is column-major. This is the layout BLAS libraries are tuned for, because the inner dimension is contiguous in the left operand and strided in the right, and the micro-kernel can vectorise over the $n$ axis.

The same transpose rule shows up in the "attention is not symmetric" fact from §3.2, and in the RoPE identity (§8.3), where $R_m^\top R_n = R_{n-m}$ holds precisely because the $2\times 2$ rotation blocks are orthogonal.

---

#### **trace by hand**

**Trace = sum of the diagonal.** $A=\begin{bmatrix}1&2\\3&4\end{bmatrix}\Rightarrow \mathrm{tr}(A)=1+4=5$; $B=\begin{bmatrix}5&6\\7&8\end{bmatrix}\Rightarrow\mathrm{tr}(B)=5+8=13$.

**Linearity.** $2A+3B=\begin{bmatrix}17&22\\27&32\end{bmatrix}$ so $\mathrm{tr}=17+32=49$, and $2\cdot5+3\cdot13=49$ .

**Cyclicity without commutativity** (using $AB,BA$ from §3):

$$\mathrm{tr}(AB)=19+50=69,\qquad \mathrm{tr}(BA)=23+46=69\ ,\qquad\text{but } AB=\begin{bmatrix}19&22\\43&50\end{bmatrix}\neq\begin{bmatrix}23&34\\31&46\end{bmatrix}=BA$$

Equal traces do **not** mean equal matrices.

**Cross-entropy as a trace.** Two examples, three classes. Predicted probabilities and one-hot labels (true classes are 0 and 2):

$$P=\begin{bmatrix}0.7&0.2&0.1\\0.1&0.3&0.6\end{bmatrix},\qquad Y=\begin{bmatrix}1&0&0\\0&0&1\end{bmatrix}$$

Row-wise: $\mathrm{NLL}=-(\ln0.7+\ln0.6)=0.3567+0.5108=0.8675$.
Trace form: $Y^\top\log P$ is $3\times3$ and its diagonal entries are $c=0:\ \ln0.7=-0.3567$, $c=1:\ 0$, $c=2:\ \ln0.6=-0.5108$. So $-\mathrm{tr}(Y^\top\log P)=0.3567+0+0.5108=0.8675$ . The one-hot $Y$ simply selects the log-probability of the right class.

**Attention concentration $\|P\|_F^2=\mathrm{tr}(P^\top P)$** ($n=2$ rows):

- one-hot (sharp): $P=I$, sum of squares $=1+1=2=n$
- uniform (diffuse): all entries $0.5$, sum of squares $=4\cdot0.25=1$

Sharp attention scores high ($=n$), diffuse scores low.

**Closed walks in a triangle graph.** $A=\begin{bmatrix}0&1&1\\1&0&1\\1&1&0\end{bmatrix}$, $A^2=\begin{bmatrix}2&1&1\\1&2&1\\1&1&2\end{bmatrix}$, so $\mathrm{tr}(A^2)=6$. That is every edge walked out-and-back from each end ($3\text{ edges}\times2$). Also $\mathrm{tr}(A^3)=6$: around the triangle, 3 starting vertices × 2 directions.

**Remember:** $\mathrm{tr}(AB)=\mathrm{tr}(BA)$ is about one number, not about the matrices; trace is a *very lossy* summary.

## **5. Trace, Linearity, and Cyclicity**

### **5.1 Linearity and the cyclic identity**

The trace of a square matrix is the sum of its diagonal [32]:

$$
\mathrm{tr}(A) = \sum_{i=1}^{n} A_{ii}
$$

It is linear in $A$:

$$
\mathrm{tr}(\alpha A + \beta B) = \alpha\,\mathrm{tr}(A) + \beta\,\mathrm{tr}(B)
$$

and it is **cyclic**:

$$
\boxed{\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)}
$$

The proof is a double sum with the indices swapped:

$$
\mathrm{tr}(AB) = \sum_i (AB)_{ii} = \sum_i \sum_j A_{ij}B_{ji} = \sum_j \sum_i B_{ji}A_{ij} = \sum_j (BA)_{jj} = \mathrm{tr}(BA)
$$

Note what this proof did *not* use: it never assumed $A$ and $B$ commute.

### **5.2 Cyclicity without commutativity**

This is the single most misread identity in the syllabus. The trace is cyclic, which means

$$
\mathrm{tr}(AB) = \mathrm{tr}(BA) \quad \text{for all square } A, B
$$

but this does **not** mean $AB = BA$. The two statements are about completely different objects: one is a single scalar (the sum of the diagonal), the other is an entire matrix.

Measured on $4096\times 4096$ matrices (LLaMA-2-7B shapes, float32):

```
max|A@B - B@A|     = 1.310443e-01     <- the matrices are wildly different
max|A@B|           = 8.438757e-02
tr(A@B)            = -0.19652960
tr(B@A)            = -0.19653012
|tr(A@B) - tr(B@A)| = 5.215e-07        <- the traces agree to float32 precision
```

The trace is a *very lossy* summary of a matrix. Two matrices that agree on their trace can differ in every other entry. This is why "the traces match" is never evidence that two products commute — it is evidence only that their diagonal sums match.

The cyclic identity with three or more factors, $\mathrm{tr}(ABC) = \mathrm{tr}(BCA)$, is verified in §14.3 on 400×400 matrices.

### **5.3 Traces in production models**

**Trace as the cross-entropy loss.** Let $Z \in \mathbb{R}^{n\times C}$ be the logits, $P = \mathrm{softmax}(Z)$ row-wise, and $Y \in \mathbb{R}^{n\times C}$ the one-hot label matrix ($Y_{ic} = 1$ iff $c$ is the label of example $i$). Then

$$
\mathrm{NLL} = -\sum_{i=1}^{n}\log P_{i,y_i} = -\sum_{i,c} Y_{ic}\log P_{ic} = -\mathrm{tr}\!\left(Y^\top \log P\right)
$$

The last step: $\mathrm{tr}(Y^\top \log P) = \sum_c (Y^\top \log P)_{cc} = \sum_c \sum_i Y_{ic}(\log P)_{ic}$. Verified numerically to $5.7\times 10^{-14}$ in §14.3. So the softmax classifier's loss is literally a trace of a matrix logarithm against a one-hot matrix — and soft-label regression is the same formula with a general label matrix.

**Trace as a concentration metric.** For a row-stochastic attention matrix $P$ (each row sums to 1):

$$
\|P\|_F^2 = \mathrm{tr}(P^\top P)
$$

This is a real attention-diagnostic: it equals $n$ exactly when $P$ is a permutation matrix (one-hot, maximally peaked attention) and is small when attention is diffuse. Measured: $2.72$ for a soft attention matrix, $256.00$ for the corresponding one-hot matrix, with $n = 256$.

The same distinction matters for the total mass: for a row-stochastic $P$, $\mathbf{1}^\top P\mathbf{1} = n$, whereas $\mathrm{tr}(P)$ is only the diagonal mass, measured at $1.83$ for a $64\times 10$ softmax and nowhere near $n$. Conflating the two is a common slip.

**Trace as a count of closed walks.** For an adjacency matrix $A$ of a graph, $\mathrm{tr}(A^k)$ counts the closed walks of length $k$ starting and ending at the same vertex. Measured on a 4-vertex graph: $\mathrm{tr}(A^2) = 10$, $\mathrm{tr}(A^3) = 12$, $\mathrm{tr}(A^4) = 50$. Random-walk structural encodings and graph positional encodings are built on exactly this fact, and so are spectral graph filters.

**Trace as a norm.** $\mathrm{tr}(A^\top A) = \|A\|_F^2$, and $\mathrm{tr}(A) = \sum_i A_{ii}$ for a parameter matrix is the sum of the diagonal — a number that changes sign when you negate a matrix but carries no useful magnitude information.

---

#### **Hadamard vs matrix product, and the gate in SwiGLU**

**Hadamard = multiply matching positions, never sum.** With the same $A,B$:

$$A\odot B=\begin{bmatrix}1\cdot5&2\cdot6\\3\cdot7&4\cdot8\end{bmatrix}=\begin{bmatrix}5&12\\21&32\end{bmatrix}\qquad\text{vs}\qquad AB=\begin{bmatrix}19&22\\43&50\end{bmatrix}$$

| | Hadamard $A\odot B$ | matrix product $AB$ |
|---|---|---|
| shapes needed | identical | inner dims match |
| work | $mn$ multiplications | $mnk$ multiplications |
| sums? | no | yes, over $k$ |
| NumPy | `A * B` | `A @ B` |

**SwiGLU for one token.** Suppose the gate path gives $[2,\,-1]$ and the up path gives $[3,\,4]$. SiLU is $z\cdot\sigma(z)$ with $\sigma(z)=1/(1+e^{-z})$:

$$\mathrm{SiLU}(2)=2\cdot0.8808=1.7616,\qquad \mathrm{SiLU}(-1)=-1\cdot0.2689=-0.2689$$

$$\mathrm{SiLU}(\text{gate})\odot\text{up}=[\,1.7616\cdot3,\ -0.2689\cdot4\,]=[\,5.285,\ -1.076\,]$$

The gate (SiLU path) decides how much of each up-path feature passes through. That filtered vector then goes through $W_d$.

**RMSNorm by hand.** $x=[3,4]$, $d=2$, gain $g=[1,2]$ (ignore $\epsilon$):

$$\text{rms}=\sqrt{\tfrac{3^2+4^2}{2}}=\sqrt{12.5}=3.5355,\qquad \frac{x}{\text{rms}}=[0.8485,\ 1.1314],\qquad \odot\,g=[0.8485,\ 2.2627]$$

**The broadcasting trap, small.** Square $X=\begin{bmatrix}1&2\\3&4\end{bmatrix}$ and $b=[10,20]$:

- `X + b` adds $b_j$ to column $j$ of every row: $\begin{bmatrix}11&22\\13&24\end{bmatrix}$ (per-feature)
- `X + b[:, None]` adds $b_i$ to every entry of row $i$: $\begin{bmatrix}11&12\\23&24\end{bmatrix}$ (per-token)

Both run without error on a square matrix, which is exactly why this bug is silent.

**Remember:** `*` pairs, `@` sums. Gating, scaling and masking use pairing; mixing features uses summing.

## **6. Hadamard Products and Matrix Products**

### **6.1 Two distinct contractions**

These are different operations with different shapes and different costs:

$$
(AB)_{ij} = \sum_{\ell=1}^{k} A_{i\ell}B_{\ell j} \qquad\text{(matrix product: contracts $k$, sums)}
$$

$$
(A \odot B)_{ij} = A_{ij}B_{ij} \qquad\text{(Hadamard product: pairs, never sums)}
$$

The matrix product is $O(mnk)$; the Hadamard product is $O(mn)$ and requires no contraction. The Hadamard product needs conformable shapes; the matrix product needs matching inner dimensions. They agree only in trivial cases.

### **6.2 Elementwise multiplication in modern transformers**

**Example A — SwiGLU, the most important one.** The feed-forward block of essentially every frontier open LLM (LLaMA, PaLM, Mistral, Qwen, DeepSeek, Gemma) is

$$
\mathrm{FFN}_{\text{SwiGLU}}(x) = \big[\,\mathrm{SiLU}(xW_g) \odot (xW_u)\,\big] W_d
$$

The $\odot$ is a genuine Hadamard product: the gate path and the up path have identical shape $(B T \times d_{\text{ff}})$ and are multiplied entry by entry. Shazeer's original formulation is $\mathrm{Swish}_\beta(xW + b)\otimes(xV + c)$, where the gate is $\mathrm{Swish}_\beta(z) = z\cdot\sigma(\beta z)$ with $\beta = 1$ in practice [10][11]. Because a gated unit needs two projections where a plain FFN needs one, the inner width is shrunk to $\tfrac{8}{3}d_{\text{model}}$ to keep the parameter count and compute budget matched to a vanilla $4d_{\text{model}}$ ReLU FFN [10].

**Example B — RMSNorm's per-channel scale.** RMSNorm computes

$$
\mathrm{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i} x_i^2 + \epsilon}} \odot g
$$

where $g \in \mathbb{R}^d$ is a learned per-channel gain. That $\odot$ is a multiplication by a *diagonal matrix* $\mathrm{diag}(g)$ — a special matrix from Module 03, applied by broadcast. This is why RMSNorm is cheap and why it is the norm choice in LLaMA-family models (which also drop biases entirely).

**Example C — elementwise composition of LoRA variants.** Batched LoRA (fLoRA) composes the adapter *elementwise* with the frozen weight rather than additively: $W_i = \Delta W_i \circ W_0$, with $\Delta W_i = B_iA_i$ still a matrix product [12]. This is a genuine Hadamard product of two conformable matrices sitting inside a parameter-efficient fine-tuning method.

**Example D — mixture-of-experts routing.** Top-$k$ expert routing multiplies the router's normalised scores elementwise into the combined expert outputs; the gating is elementwise, the expert transforms are matmuls.

### **6.3 The broadcasting failure mode**

NumPy uses `*` for elementwise and `@` for matrix product; PyTorch does the same. The footgun is not that these differ, but that **broadcasting silently aligns a 1-D vector with the *last* axis**. A bias vector of length $n$ is added to every row of an $m\times n$ matrix; if the vector is instead given a trailing axis (`b[:, None]`) it is added to every column. When the shapes happen to be legal both ways, nothing raises:

```python
Y = np.ones((4096, 11008))   # (tokens, ff)
b_ff  = np.ones(11008)       # per-feature bias, the correct one

Y + b_ff                     # OK: b_ff[j] added to column j of every row, shape (4096, 11008)
Y + np.ones(4096)            # ValueError: (4096,11008) vs (4096,) -- loud, good
Y + np.ones(4096)[:, None]   # LEGAL: adds a per-TOKEN value to every row -- silent if you meant per-feature
```

On a *square* activation (for example $d_{\text{model}}\times d_{\text{model}}$ with a $d_{\text{model}}$-long bias), `X + b` and `X + b[:, None]` are both legal and both return the same shape, so a transposed-layout mistake survives silently and only shows up as a small loss regression — the kind of failure that costs a day. The fix is to make the axis explicit and check it: `Y + b[None, :]`, plus `assert b.shape == Y.shape[-1:]`. This is the practical reason to know that "adding a bias" is really "adding a rank-1 matrix broadcast across rows" (§13).

---

#### **block multiplication by hand**

**Idea.** Cut matrices into blocks with matching cut lines, then multiply blocks as if they were numbers.

Split $A$ ($2\times4$) into two column blocks and $B$ ($4\times2$) into two row blocks:

$$A=[A_1\ |\ A_2]=\left[\begin{bmatrix}1&2\\5&6\end{bmatrix}\ \Big|\ \begin{bmatrix}3&4\\7&8\end{bmatrix}\right],\qquad B=\begin{bmatrix}B_1\\B_2\end{bmatrix}=\begin{bmatrix}\begin{bmatrix}1&0\\0&1\end{bmatrix}\\[4pt]\begin{bmatrix}2&1\\1&2\end{bmatrix}\end{bmatrix}$$

$$AB=A_1B_1+A_2B_2,\qquad A_1B_1=\begin{bmatrix}1&2\\5&6\end{bmatrix},\quad A_2B_2=\begin{bmatrix}3\cdot2+4\cdot1&3\cdot1+4\cdot2\\7\cdot2+8\cdot1&7\cdot1+8\cdot2\end{bmatrix}=\begin{bmatrix}10&11\\22&23\end{bmatrix}$$

$$AB=\begin{bmatrix}11&13\\27&29\end{bmatrix}$$

**Check against the unblocked product.** Row 1 of $A$ is $[1,2,3,4]$; column 1 of $B$ is $[1,0,2,1]^\top$: $1+0+6+4=11$ . Row 2 $[5,6,7,8]$ with column 2 $[0,1,1,2]^\top$: $0+6+7+16=29$ .

**Block-diagonal = independent small problems.** $\mathrm{diag}(A_1,A_2)\cdot\mathrm{diag}(B_1,B_2)=\mathrm{diag}(A_1B_1,\,A_2B_2)$. With $A_1=\begin{bmatrix}1&2\\3&4\end{bmatrix}$, $B_1=\begin{bmatrix}0&1\\1&0\end{bmatrix}$, $A_2=\begin{bmatrix}5&6\\7&8\end{bmatrix}$, $B_2=\begin{bmatrix}1&0\\0&2\end{bmatrix}$:

$$A_1B_1=\begin{bmatrix}2&1\\4&3\end{bmatrix},\qquad A_2B_2=\begin{bmatrix}5&12\\7&16\end{bmatrix}$$

**Cost saving.** Dense $4\times4$ times $4\times4$: $4^3=64$ multiplications. Two $2\times2$ blocks: $2\cdot2^3=16$. With $r$ equal blocks the saving is $r\times$.

**Remember:** blocks multiply like numbers if the cut lines match. Tiling (FlashAttention, GPU kernels) is this idea applied to memory.

## **7. Block Matrices and Block Multiplication**

### **7.1 The block partition rule**

A block matrix is an $m\times n$ matrix whose entries are themselves matrices. If $A$ is partitioned into blocks $A_{ij}$ of size $p\times q$ and $B$ into blocks $B_{jk}$ of size $q\times r$, then

$$
(AB)_{ij} = \sum_{k} A_{ik}B_{jk}
$$

— the *same* rule, with the block indices playing the role of the entry indices. The scalar rule is a special case where every block is $1\times 1$. Crucially, the blocks need not be square, and $A$ and $B$ can be partitioned differently in each axis.

For the special case of square block diagonal matrices, the product is block diagonal with matching blocks:

$$
\mathrm{diag}(A_1,\dots,A_r)\,\mathrm{diag}(B_1,\dots,B_r) = \mathrm{diag}(A_1B_1,\dots,A_rB_r)
$$

This is the algebraic fact behind a whole family of efficiency techniques: a block-diagonal matrix costs $\sum_i \mathrm{cost}(A_iB_i)$ instead of the dense $\mathrm{cost}(AB)$.

### **7.2 Block structure in serving systems**

**Example A — PagedAttention and the KV cache (vLLM).** vLLM stores each request's KV cache as a sequence of fixed-size blocks, with a *block table* mapping logical blocks to non-contiguous physical blocks, in direct analogy with virtual-memory paging. The paper reports near-zero waste in KV-cache memory and 2–4× throughput improvement over FasterTransformer and Orca at the same latency [13]. The block table is a *runtime data structure* that encodes a block-partitioned view of a matrix that is logically contiguous. vAttention later noted that PagedAttention has become the de facto standard for dynamic KV allocation in TensorRT-LLM, HuggingFace TGI, FlashInfer and LightLLM, at the cost of requiring attention kernels to be rewritten for non-contiguous access [14].

**Example B — block-sparse attention.** Longformer-style and native-sparse-attention schemes restrict attention to a block pattern rather than a dense $n\times n$ mask, turning the score matrix into a block matrix with many zero blocks. The block structure is not just a sparsity description; it is what makes the attention kernel's tile schedule legal.

**Example C — FlashAttention's tiles.** FlashAttention's $B_r \times B_c$ tiles of $Q$, $K$, $V$ are exactly a block partition, and the $O$ accumulator is a block of the output rather than a full $n\times d$ matrix. Tiling is the memory-hierarchy realisation of the block-matrix idea.

### **7.3 Block-diagonal approximations in optimization**

**K-FAC** approximates the Fisher Information Matrix of a deep network as block diagonal across layers, and then further factorises each layer's block as a Kronecker product:

$$
F \approx \mathrm{blockdiag}(F_1,\dots,F_L), \qquad F_\ell \approx A_{\ell-1}\otimes G_\ell
$$

where $A_{\ell-1} = \mathbb{E}[\bar{a}_\ell \bar{a}_\ell^\top]$ is the activation covariance and $G_\ell = \mathbb{E}[g_\ell g_\ell^\top]$ the pre-activation gradient covariance [15]. The block-diagonal step is what makes second-order optimisation tractable at all: the full FIM for a network with $n$ parameters is $n\times n$, while each K-FAC block is only $(d_{\text{in}}+1)\times(d_{\text{in}}+1)$ and $(d_{\text{out}})\times(d_{\text{out}})$.

**Monarch and Group-and-Shuffle** exploit the other direction: a *product* of block-diagonal matrices (optionally interleaved with permutations) is a hardware-efficient structured matrix. Monarch is parameterised as $M = PL^\top R$ with $L,R$ block-diagonal, which "leverages optimized batch-matrix-multiply (BMM) routines on GPUs, yielding up to 2× speedup compared to dense matrix multiply" [16]. The Group-and-Shuffle construction generalises Monarch to an alternating product of block-diagonal matrices and several permutations, giving dense orthogonal matrices with fewer parameters than BOFT [17].

**DeepSeek-V2's MLA** is a third pattern: the KV cache is not block-diagonal but *low-rank*, with a shared latent vector $c^{KV}_t = W^{DKV}h_t \in \mathbb{R}^{d_c}$ ($d_c \ll d_h n_h$) standing in for all $n_h$ head-specific key/value vectors, which the paper reports cuts the KV cache by 93.3% and boosts generation throughput by 5.76× over DeepSeek 67B [18]. The "absorb" operation then folds $W^{UK}$ into the query projection and $W^{UV}$ into the output projection so the latent can be attended to directly [18][19].

---

#### **powers, special matrices, and the softmax merge**

**Powers of a diagonal matrix act entrywise.** $\mathrm{diag}(2,3)^3=\mathrm{diag}(8,27)$.

**Powers via eigenvalues.** $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$ has eigenvectors $[1,1]^\top$ (eigenvalue 3) and $[1,-1]^\top$ (eigenvalue 1).

$$A^2=\begin{bmatrix}2\cdot2+1\cdot1&2\cdot1+1\cdot2\\1\cdot2+2\cdot1&1\cdot1+2\cdot2\end{bmatrix}=\begin{bmatrix}5&4\\4&5\end{bmatrix}$$

Check: $A^2[1,1]^\top=[9,9]^\top=9\cdot[1,1]^\top$, so eigenvalue $3^2=9$ . And $\mathrm{tr}(A^2)=10=9+1$ = sum of squared eigenvalues .

**A shear.** $U=\begin{bmatrix}1&1\\0&1\end{bmatrix}$: $U^2=\begin{bmatrix}1&2\\0&1\end{bmatrix}$, $U^3=\begin{bmatrix}1&3\\0&1\end{bmatrix}$ — the off-diagonal grows by 1 each power.

**A stochastic matrix is a diffusion step.** $P=\begin{bmatrix}0.9&0.1\\0.5&0.5\end{bmatrix}$ (rows sum to 1):

$$P^2=\begin{bmatrix}0.9\cdot0.9+0.1\cdot0.5&0.9\cdot0.1+0.1\cdot0.5\\0.5\cdot0.9+0.5\cdot0.5&0.5\cdot0.1+0.5\cdot0.5\end{bmatrix}=\begin{bmatrix}0.86&0.14\\0.70&0.30\end{bmatrix}$$

Rows still sum to 1. Entry $(2,1)=0.70$ is the probability of being in state 1 two steps after starting in state 2.

**Power iteration.** Repeatedly multiply by $A=\begin{bmatrix}2&1\\1&2\end{bmatrix}$ and rescale so the largest entry is 1, starting from $[1,0]$:

| step | $Av$ | rescaled | gap to $[1,1]$ |
|---|---|---|---|
| 1 | $[2,1]$ | $[1,\ 0.5]$ | 0.5 |
| 2 | $[2.5,2]$ | $[1,\ 0.8]$ | 0.2 |
| 3 | $[2.8,2.6]$ | $[1,\ 0.9286]$ | 0.071 |

It converges to the dominant eigenvector $[1,1]$, and the gap shrinks by about $\lambda_2/\lambda_1=1/3$ per step.

**Closure under multiplication** (the product of two special matrices is the same kind):

- Diagonal: $\mathrm{diag}(1,2,3)\,\mathrm{diag}(4,5,6)=\mathrm{diag}(4,10,18)$
- Upper-triangular: $\begin{bmatrix}1&2\\0&3\end{bmatrix}\begin{bmatrix}4&5\\0&6\end{bmatrix}=\begin{bmatrix}4&5+12\\0&18\end{bmatrix}=\begin{bmatrix}4&17\\0&18\end{bmatrix}$
- Permutation: $P=\begin{bmatrix}0&1&0\\0&0&1\\1&0&0\end{bmatrix}$ gives $P[a,b,c]^\top=[b,c,a]^\top$ — a pure relabelling, no arithmetic
- Orthogonal (90° rotation): $Q=\begin{bmatrix}0&-1\\1&0\end{bmatrix}$, $Q^\top Q=I$ and $Q^2=-I$, which is also orthogonal

**Causal mask by hand.** Raw scores in every row $[1,2,3]$. Token 1 may see only itself, token 2 the first two, token 3 all three (masked entries $=-\infty$, so $e^{-\infty}=0$):

| token | allowed scores | softmax |
|---|---|---|
| 1 | $[1]$ | $[1,\,0,\,0]$ |
| 2 | $[1,2]$ | $[0.269,\ 0.731,\ 0]$ |
| 3 | $[1,2,3]$ | $[0.090,\ 0.245,\ 0.665]$ |

Row 3: $e^1+e^2+e^3=2.718+7.389+20.086=30.193$, and $20.086/30.193=0.665$. The resulting weight matrix is lower-triangular.

**Online softmax merge (the FlashAttention trick) by hand.** Softmax denominator for scores $[1,2,3]$, computed in two blocks $[1,2]$ and $[3]$, tracking running max $m$ and running sum $\ell=\sum e^{x-m}$:

- Block 1: $m=2$, $\ell=e^{1-2}+e^{2-2}=0.3679+1=1.3679$
- Block 2: $\tilde m=3$, $\tilde\ell=e^{0}=1$
- Merge: $m'=3$, $\ell'=e^{2-3}\cdot1.3679+e^{3-3}\cdot1=0.5032+1=1.5032$
- Direct: $e^{-2}+e^{-1}+e^{0}=0.1353+0.3679+1=1.5032$ 

Same answer, but the full row never had to be in memory at once.

**Remember:** structure (diagonal, triangular, orthogonal, permutation) survives multiplication, and that is what makes it cheap.

## **8. Matrix Powers and Special-Matrix Closures**

### **8.1 Powers of a matrix**

$$
A^k = \underbrace{A \cdot A \cdots A}_{k \text{ factors}}, \qquad A^0 = I
$$

The algebraic facts that matter:

- If $A$ is diagonalisable, $A = PDP^{-1}$ with $D = \mathrm{diag}(\lambda_1,\dots,\lambda_n)$, then $A^k = PD^kP^{-1}$ and $D^k = \mathrm{diag}(\lambda_1^k,\dots,\lambda_n^k)$. Powers act on the eigenvalues.
- $\mathrm{tr}(A^k)$ is the sum of the $k$-th powers of the eigenvalues (counted with algebraic multiplicity) and, for a non-negative adjacency matrix, the number of closed walks of length $k$ (§5.3).
- $A^k$ for a stochastic matrix is a *diffusion operator*: the $(i,j)$ entry is the probability of a random walk of length $k$ from $i$ to $j$. PageRank is built on this, and so are the random-walk positional encodings used in graph transformers.

**Real anchors.** Power iteration for the leading eigenpair (PCA, spectral embedding, and the top-singular-direction analysis of LLM representations) repeatedly applies $A$; for a matrix with dominant eigenvalue $\lambda_1$, the normalised iterates converge to the corresponding eigenvector at a rate governed by $\lambda_2/\lambda_1$. In GNNs, $A^k$ is the $k$-hop neighbourhood aggregate, and $\mathrm{tr}(A^k)$ measures cycle structure. For a symmetric positive-definite matrix, the Cholesky factorisation $A = LL^\top$ with $L$ lower-triangular reduces a solve from $O(n^3)$ to $O(n^2)$, which is the standard route for Gaussian-process kernels and Kalman filtering — and it is again a *triangularity* argument, not a matrix-product argument.

### **8.2 Diagonal, triangular, orthogonal, and permutation closures**

Each special class from Module 03 is closed under matrix multiplication, and the closure property is what makes the class computationally cheap:

$$
\text{Diagonal} \times \text{Diagonal} = \text{Diagonal}
$$

$$
\text{Upper-triangular} \times \text{Upper-triangular} = \text{Upper-triangular}, \qquad
\text{Lower-triangular} \times \text{Lower-triangular} = \text{Lower-triangular}
$$

$$
\text{Orthogonal} \times \text{Orthogonal} = \text{Orthogonal} \quad (Q_1^\top Q_1 = Q_2^\top Q_2 = I \Rightarrow (Q_1Q_2)^\top(Q_1Q_2) = I)
$$

$$
\text{Permutation} \times \text{Permutation} = \text{Permutation}
$$

All four verified numerically in §14.3. The proofs are all one-liners, and each one is a *structural* statement about which entries can be non-zero or which inner products are constrained.

**Why the closures matter in deep learning:**

- *Diagonal × diagonal.* A diagonal matrix multiply is $O(n)$ and touches $n$ elements. Per-channel scaling (RMSNorm's gain, per-head biases, per-expert scaling factors, LoRA's $\alpha/r$) is a diagonal multiply in disguise, and a fused kernel can do it in one pass over the data.
- *Triangular × triangular.* The causal attention mask is lower-triangular. A triangular system solves in $O(n^2)$ instead of $O(n^3)$ by forward substitution, which is why triangular structure is exploited in Cholesky-based GP solvers, in banded linear algebra, and in the "triangular" structure exploited by specialised attention kernels. Note the *mask* is applied additively (set masked logits to $-\infty$) rather than by multiplication, but the structural principle is the same: the effective matrix is triangular.
- *Orthogonal × orthogonal.** Orthogonal Fine-Tuning (OFT) fine-tunes a pretrained weight $W^0 \in \mathbb{R}^{d\times n}$ via $y = (QW^0)^\top x$ with $Q$ a learnable orthogonal matrix, parameterised in block-diagonal form $Q = \mathrm{diag}(Q_1,\dots,Q_r)$ and built with the Cayley transform $Q_i = (I + K_i)(I-K_i)^{-1}$ for skew-symmetric $K_i$ [17]. Because $Q$ is orthogonal, it preserves the singular values of the weight: if $W^0 = U\Sigma V^\top$, then $W' = (QU)\Sigma V^\top$ is another valid SVD of the adapted weight, so only the left singular vectors move. The stated benefit is preservation of pairwise angles between neurons ("hyperspherical energy") and hence reduced catastrophic forgetting relative to LoRA.
- *Permutation × permutation.** A permutation matrix has exactly one $1$ per row and column, and multiplying by one is a *relabelling*. In geometric deep learning, the permutation group $S_n$ acts on a graph by $P$ acting on the adjacency matrix and features: $P^\top A P$ permutes rows and columns, and $\hat{x} = P^\top x$ reorders node features. The stability of GNNs under this action is the source of their permutation equivariance [20][21]. A permutation matrix is also orthogonal ($P^\top P = I$), so it is a special case of the orthogonal closure — and multiplying by one costs $O(n)$ because it is a gather, not arithmetic.

### **8.3 A closure inside the attention kernel**

The clearest place where a Module-03 special matrix operates on a Module-04 product is FlashAttention's online softmax. Each row of the attention score matrix is processed in blocks; the algorithm keeps a running maximum $m$, a running exponential sum $\ell$, and a running output accumulator $O$. When a new block arrives, the merge is

$$
m' = \max(m, \tilde m), \qquad
\ell' = e^{m-m'}\ell + e^{\tilde m - m'}\tilde\ell
$$

$$
O' = \frac{e^{m-m'}\ell\, O + e^{\tilde m-m'}\, \tilde P V_j}{\ell'}
$$

Now look at the $O$ update in isolation. Written without the division, the accumulator update is

$$
O \leftarrow \mathrm{diag}\!\big(e^{m-m'}\ell\big)\, O \;+\; \mathrm{diag}\!\big(e^{\tilde m-m'}\big)\, \tilde P V_j
$$

That is: **a diagonal matrix times the accumulator, plus a scaled rank-$\le T_r$ update**. The diagonal rescale is a *diagonal* matrix multiply (the closure of §8.2) applied to a product of a rank-deficient $P$ tile with a $V$ tile. The whole point is that the rescaling is $O(d)$ per row rather than $O(d^2)$, which is only possible because the scaling matrix is diagonal.

This is why FlashAttention is exact rather than approximate: the algebra above is a *rewriting* of the softmax, not a truncation of it. The $\frac{n^2}{2}$ entries of the score matrix are computed one tile at a time in SRAM and discarded, never written to HBM. The published result is 2–4× wall-clock speedup over optimised baselines with 10–20× memory saving, and the memory traffic drops from $O(n^2)$ to $O(n)$ [22][23].

#### **matrix × vector as a mix of columns**

**Rule.** $A\mathbf{x}$ = columns of $A$, each scaled by the matching entry of $\mathbf{x}$, then added.

$A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$, $\mathbf{x}=\begin{bmatrix}1\\0\\2\end{bmatrix}$:

$$A\mathbf{x}=1\begin{bmatrix}1\\4\end{bmatrix}+0\begin{bmatrix}2\\5\end{bmatrix}+2\begin{bmatrix}3\\6\end{bmatrix}=\begin{bmatrix}1+6\\4+12\end{bmatrix}=\begin{bmatrix}7\\16\end{bmatrix}$$

Row view agrees: $1+0+6=7$ and $4+0+12=16$ . Cost: $2mk=2\cdot2\cdot3=12$ FLOPs.

**Why decoding is memory-bound — in three lines.** One $4096\times4096$ FP16 weight matrix:

- bytes to read $=2\cdot4096^2=33{,}554{,}432$
- FLOPs for one token $=2\cdot4096^2=33{,}554{,}432$
- intensity $=1$ FLOP per byte

**Batching reuses the same read.** The weights are read once whether you serve 1 token or $m$ tokens, but FLOPs grow with $m$:

| tokens $m$ | FLOPs | bytes (≈ weights only) | intensity |
|---|---|---|---|
| 1 | $2\cdot4096^2$ | $2\cdot4096^2$ | ≈ 1 |
| 64 | $64\times$ | same | ≈ 64 |
| 153 | $153\times$ | same | ≈ 153 ← A100 ridge point |

So the rule of thumb for $m\ll n$: **intensity ≈ batch size**. That is the whole reason serving systems batch requests.

**Remember:** one token = matrix–vector = reads every weight once for almost no arithmetic.

## **9. Matrix-Vector Products as Column Combinations**

### **9.1 The identity**

Reading 2 from §2.2, specialised to a single vector: for $A \in \mathbb{R}^{m\times k}$ and $\mathbf{x} \in \mathbb{R}^k$,

$$
A\mathbf{x} = \sum_{j=1}^{k} x_j\, A_{:,j} = x_1 A_{:,1} + x_2 A_{:,2} + \cdots + x_k A_{:,k}
$$

The output is a linear combination of the *columns* of $A$, with the entries of $\mathbf{x}$ as the coefficients. Equivalently, $A\mathbf{x}$ is a weighted sum of the columns, so the "layers" metaphor is literally true: each column of $A$ is a direction in input space, and $\mathbf{x}$ says how much of each direction to use.

The cost is $2mk$ FLOPs — the same $2mnk$ formula with $n=1$.

### **9.2 Batch-1 decoding**

The single most important matrix-vector product in modern AI systems is the one that happens once per generated token.

Autoregressive decoding computes, for a single new token, $Y = xW$ for every weight matrix in the model, where $x \in \mathbb{R}^{d_{\text{model}}}$ is a *single* token's feature vector. That is a matrix-vector product, not a matrix-matrix product. Working the numbers for LLaMA-2-7B (parameter count computed from the published config in §14.3 and matching the published 6,738,415,616):

| quantity | value |
|---|---|
| parameters $N$ | 6,738,415,616 |
| weights read per token, FP16 | 13,476,831,232 B = 13.48 GB |
| forward FLOPs per token, $\approx 2N$ | 13.48 GFLOP |
| arithmetic intensity | **1.00 FLOP/byte** |

Compare that arithmetic intensity against the machine's balance. An A100 80GB SXM delivers 312 TFLOP/s of FP16 Tensor Core throughput over 2,039 GB/s of HBM2e bandwidth, a balance of **153 FLOP/byte** [25]; an H100 SXM delivers 989.4 TFLOP/s over 3.35 TB/s, a balance of **295 FLOP/byte** [25][37]. At an intensity of 1.00 FLOP/byte, the decode step is memory-bound by exactly the machine balance:

| accelerator | memory-bound time | compute-bound time | ratio |
|---|---|---|---|
| A100 80GB SXM | 6.610 ms/token | 0.043 ms/token | **153.0×** |
| H100 SXM | 4.023 ms/token | 0.014 ms/token | **295.2×** |

So on an A100, a 7B model in FP16 generates at best around 150 tokens per second, and *the entire gap to that number is wasted compute* — the FLOPs are there, the bandwidth is not. This is the quantitative reason serving systems care about batching: at batch 1 the arithmetic intensity is 1 FLOP/byte, and at batch 512 for a 4096×1024 linear layer NVIDIA measures 315 FLOP/byte [27]. The "tall and skinny" regime behaves the same way: for $M = N = 1$ the arithmetic intensity of a GEMM is $1/8$ FLOP/byte, and even at $M = N = 64$ it is only 8 FLOP/byte, below the V100's machine balance [30].

### **9.3 Parallel splits of a tall product**

The decode phase is bandwidth-bound partly because a single query against a long KV cache is one tall matrix-vector product — a serial dependency chain with no parallelism. FlashDecoding's answer is to *split* the KV cache into chunks and compute partial attentions in parallel, then combine the partial results with the online softmax. The DEFT paper states the motivation plainly: QKV partitioning parallelism "is usually limited by the batch size, which is much smaller than the number of streaming multiprocessors (SMs) on the GPU (108 for an A100), where the operation will only utilize a small portion of the GPU" [31].

Splitting a matrix-vector product into a matrix-matrix product is a concrete instance of "associativity and reordering are free algebraically but not for free in hardware" — and it is the reason KV-cache layout, chunk size, and split count are first-class tuning parameters in every serving engine.

---

#### **counting FLOPs and bytes by hand**

**FLOPs of a matmul.** $(m\times k)(k\times n)$: each of $mn$ outputs needs $k$ multiplications and $k-1$ additions.

| example | $m,n,k$ | FLOPs $\approx2mnk$ |
|---|---|---|
| the §2 product | 2, 2, 3 | $2\cdot12=24$ (exact: 20) |
| one LLaMA projection, 1 token | 1, 4096, 4096 | $2\cdot4096^2=33{,}554{,}432$ |
| square $4096^3$ | 4096, 4096, 4096 | $2\cdot4096^3=137{,}438{,}953{,}472\approx1.37\times10^{11}$ |

**Bytes and intensity for a square FP16 GEMM, $n=1024$.**

- bytes $=s(mk+kn+mn)=2\cdot3\cdot1024^2=6{,}291{,}456$
- FLOPs $=2\cdot1024^3=2{,}147{,}483{,}648$
- intensity $=2{,}147{,}483{,}648\,/\,6{,}291{,}456=341.3$ FLOP/byte $=n/3$ 

**Ridge point of an A100.** $312\times10^{12}\ \text{FLOP/s}\ \big/\ 2039\times10^{9}\ \text{B/s}=153$ FLOP/byte.

| $n$ (FP16 square GEMM) | intensity $n/3$ | vs ridge 153 | verdict |
|---|---|---|---|
| 64 | 21.3 | below | memory-bound |
| 1024 | 341.3 | above | compute-bound |

**Training rule of thumb.** Forward $\approx2N$ FLOPs per token, backward $\approx2\times$ forward, so $\approx6N$ per training token. For $N=7\times10^9$: $6\cdot7\times10^9=4.2\times10^{10}$ FLOPs per token.

**Remember:** FLOPs $=2mnk$; intensity = FLOPs ÷ bytes; compare intensity to the chip's ridge point.

## **10. FLOP Accounting for a Matrix Multiplication**

### **10.1 The $2mnk$ FLOP count**

For $C = AB$ with $A \in \mathbb{R}^{m\times k}$, $B \in \mathbb{R}^{k\times n}$:

- **Multiplications:** each of the $mn$ entries of $C$ requires $k$ multiplications, giving $mnk$.
- **Additions:** each entry is a sum of $k$ terms, requiring $k-1$ additions, giving $mn(k-1)$.

$$
\text{FLOPs} = mnk + mn(k-1) = 2mnk - mn \approx 2mnk
$$

A multiply-accumulate (FMA) instruction performs one multiplication and one addition in a single hardware operation; the convention in the ML systems literature is to count an FMA as two FLOPs, which is what gives the clean $2mnk$ [26]. For $n = m = k = 4096$ (a LLaMA-2-7B-sized square product), that is $2 \times 4096^3 = 1.37\times 10^{11}$ FLOPs.

### **10.2 Bytes moved and arithmetic intensity**

If each element occupies $s$ bytes, the compulsory traffic is

$$
\text{bytes} = s\big(mk + kn + mn\big)
$$

(reading $A$, reading $B$, writing $C$). The **arithmetic intensity** is

$$
I = \frac{\text{FLOPs}}{\text{bytes}} = \frac{2mnk}{s(mk + kn + mn)}
$$

For a square $n\times n$ product in FP16 ($s = 2$):

$$
I = \frac{2n^3}{2 \cdot 3n^2} = \frac{n}{3} \text{ FLOP/byte}
$$

and in FP32 ($s=4$), $I = n/6$ [26]. Two consequences follow immediately:

- **Arithmetic intensity grows linearly in $n$.** A $64\times 64$ FP16 GEMM has intensity $64/3 \approx 21.3$ FLOP/byte — well below the A100's 153 FLOP/byte ridge point, so it is memory-bound. A $1024\times 1024$ FP16 GEMM has intensity $\approx 341$ FLOP/byte and is compute-bound.
- **Precision changes the roofline position.** FP16 doubles the arithmetic intensity of the same shape relative to FP32, which is one reason Tensor Cores exist.

The ML Systems text also notes a subtlety that matters for bias handling: GEMM computes $C = \alpha AB + \beta C$, and the ideal $\beta = 0$ case reads $A$ and $B$ once and writes $C$ once, giving $n/3$ FLOP/byte. When $\beta \ne 0$, which is what happens when a bias is folded into the GEMM epilogue, the old $C$ must also be read, and the intensity drops to $n/4$ [26]. The bias is not free, and §13 makes this concrete.

### **10.3 Transformer-scale accounting**

Kaplan et al. give the per-token FLOP budget for a transformer in Table 1, separating the projection work from the context-dependent attention work:

| operation | parameters | FLOPs per token |
|---|---|---|
| embed | $(n_{\text{vocab}} + n_{\text{ctx}})d_{\text{model}}$ | $4d_{\text{model}}$ |
| attention: QKV | $n_{\text{layer}}d_{\text{model}}3d_{\text{attn}}$ | $2n_{\text{layer}}d_{\text{model}}3d_{\text{attn}}$ |
| attention: mask | — | $2n_{\text{layer}}n_{\text{ctx}}d_{\text{attn}}$ |
| attention: project | $n_{\text{layer}}d_{\text{attn}}d_{\text{model}}$ | $2n_{\text{layer}}d_{\text{attn}}d_{\text{model}}$ |
| feedforward | $n_{\text{layer}}2d_{\text{model}}d_{\text{ff}}$ | $2n_{\text{layer}}2d_{\text{model}}d_{\text{ff}}$ |
| de-embed | — | $2d_{\text{model}}n_{\text{vocab}}$ |
| **total (non-embedding)** | $N = 2d_{\text{model}}n_{\text{layer}}(2d_{\text{attn}} + d_{\text{ff}})$ | $C_{\text{forward}} = 2N + 2n_{\text{layer}}n_{\text{ctx}}d_{\text{attn}}$ |

and then, "accounting for the backwards pass (approximately twice the compute as the forwards pass), we define the estimated non-embedding compute as $C \approx 6N$ floating point operators per training token" [24]. The $6ND$ form ($D$ = training tokens) is the standard budget; the OpenLM reproduction notes that the *only* non-negligible source of error in the $6ND$ approximation is the attention layers [38].

A second, independent formulation from the ML Systems book gives multi-head self-attention as $B \times (4S^2 d_{\text{model}} + 8S d_{\text{model}}^2)$ FLOPs per forward pass [26] — the $S^2$ term is the quadratic attention interaction, the $S$ term the projections.

The ratio between the two tells you where the cost of a model lives. For a long-context run, the $2n_{\text{layer}}n_{\text{ctx}}d_{\text{attn}}$ term scales with context length while the rest does not, which is exactly why long-context training cost explodes: at $n_{\text{ctx}} = 2048$ the attention term is a small fraction of the total, and it grows linearly in context while everything else stays fixed [24].

---

#### **the roofline in one formula**

$$P=\min(\pi,\ \beta\cdot I)$$

$\pi$ = peak FLOP/s, $\beta$ = bandwidth in bytes/s, $I$ = your intensity. Performance is whichever ceiling you hit first.

**A100 with $I=1$** (batch-1 decode):

$$\beta I=2039\times10^{9}\cdot1=2.04\ \text{TFLOP/s}\quad\ll\quad\pi=312\ \text{TFLOP/s}$$

You get $2.04/312\approx0.65\%$ of the chip's peak. The arithmetic units are idle >99% of the time.

**ReLU is permanently memory-bound.** On $n$ FP16 elements: read $2n$ bytes, write $2n$ bytes, do $n$ comparisons.

$$I=\frac{n}{4n}=0.25\ \text{FLOP/byte}$$

**Why fusion pays.** Attention scores for one head at $n=2048$ tokens in FP16:

- $S$ has $2048^2=4{,}194{,}304$ entries $\times2$ B $=8{,}388{,}608$ B $\approx8.4$ MB
- Unfused: write $S$ to HBM, read it back for softmax = ≈ 16.8 MB of pure round-trip traffic per head
- Fused (FlashAttention): $S$ stays on-chip, so ≈ 0 MB of that traffic

Times 32 heads, one layer's $S$ is $32\times8.4$ MB $=0.268$ GB, matching the table below.

**Remember:** ask three questions of any operation: how many bytes, how many FLOPs, which intermediates can stay on-chip?

## **11. Compute Cost versus Memory Traffic**

### **11.1 The roofline model**

The roofline model bounds achievable performance $P$ by two hardware roofs: a peak compute throughput $\pi$ and a memory bandwidth $\beta$. With arithmetic intensity $I$ on the horizontal axis,

$$
P = \min\big(\pi,\; \beta \cdot I\big)
$$

The two roofs intersect at the **ridge point**

$$
I_{\text{ridge}} = \frac{\pi}{\beta}
$$

and an algorithm is compute-bound above it, memory-bound below it [28].

For the accelerators in this module:

| accelerator | FP16/BF16 peak | HBM bandwidth | ridge point | regime for a 7B decode step |
|---|---|---|---|---|
| A100 80GB SXM | 312 TFLOP/s | 2,039 GB/s | **153 FLOP/byte** | memory-bound (1.00 FLOP/byte) |
| A100 40GB SXM | 312 TFLOP/s | 1,555 GB/s | 201 FLOP/byte | memory-bound |
| H100 SXM | 989.4 TFLOP/s | 3,350 GB/s | **295 FLOP/byte** | memory-bound |

The A100 80GB SXM ridge of 153 FLOP/byte is independently confirmed by the ML Systems text, which states the ridge point as 153.0 FLOP/byte for that part [26] — a useful cross-check that the datasheet arithmetic and the published analysis agree.

The same NVIDIA documentation tabulates arithmetic intensities for common neural network operations under FP16 on a V100, and the table is the cleanest statement of the problem this module is about:

| operation | arithmetic intensity | usually limited by |
|---|---|---|
| Linear layer (4096 outputs, 1024 inputs, batch size 512) | 315 FLOPS/B | arithmetic |
| Linear layer (4096 outputs, 1024 inputs, batch size 1) | 1 FLOPS/B | memory |
| Max pooling with 3×3 window, unit stride | 2.25 FLOPS/B | memory |
| ReLU activation | 0.25 FLOPS/B | memory |
| Layer normalization | < 10 FLOPS/B | memory |

The same hardware does the same operation in two different regimes depending only on the batch size [27].

### **11.2 Binding regimes of the roofline**

Three practical regimes, and the diagnosis changes what you should optimise:

- **Large GEMM, large batch** (prefill, training). Intensity exceeds the ridge point; the kernel is compute-bound and Tensor Cores saturate. Optimise for FLOP efficiency: use FP16/BF16 Tensor Cores, pick a good tile shape, and *do not* waste time on fusion.
- **Batch-1 GEMV** (autoregressive decode). Intensity is 1 FLOP/byte, far below the ridge. The kernel is memory-bound and no amount of arithmetic optimisation helps. Optimise for bandwidth: keep weights in FP16, batch requests together to raise $m$, or reduce the number of bytes read (quantisation, sparsity).
- **Elementwise and normalisation** (ReLU, LayerNorm, RMSNorm, SwiGLU gating). Intensities of 0.25 to 10 FLOP/byte make these permanently memory-bound. The only way to speed them up is to avoid materialising intermediates — which is exactly what fusion does.

A fourth case sits between the third and the first: the attention *scores* matrix. For a batch of tokens, $S = QK^\top$ is a large GEMM (compute-bound), but the softmax applied to it is memory-bound, and the $PV$ product afterwards is another large GEMM. In an unfused implementation the $n\times n$ score matrix is written to HBM and read back for the softmax — that round trip is pure memory traffic, and it is the target of every fused attention kernel.

### **11.3 Fusion and avoided intermediate traffic**

The arithmetic cost of an unfused attention layer and a fused one is *identical* — the same $QK^\top$ and $PV$ products. The difference is entirely in traffic. Measured on LLaMA-2-7B shapes (32 heads, $d_{\text{head}}=128$, FP16):

| sequence length | score matrix $S$ ($h \times n \times n$) | $Q,K,V$ combined | ratio |
|---|---|---|---|
| 2,048 | 0.268 GB | 0.050 GB | 5.3× |
| 8,192 | 4.295 GB | 0.201 GB | 21.3× |
| 32,768 | 68.719 GB | 0.805 GB | 85.3× |

At 32k context the unfused attention intermediate is 68.7 GB — it does not fit in an 80 GB accelerator alongside the model. FlashAttention's contribution is precisely to keep $S$ and $P$ on-chip: the algorithm's HBM traffic drops from $O(n^2)$ to $O(n)$ [22], which is what makes long context physically possible rather than merely slow.

The general lesson generalises beyond attention. When you see a chain of operations, ask three questions: how many bytes does each stage move, what is the arithmetic intensity of each stage, and which intermediates could stay in on-chip memory? "Fuse the softmax into the GEMM" is not an optimisation trick; it is the removal of an $O(n^2)$ HBM round trip.

---

#### **why memory layout decides speed**

**Row-major offset.** A $3\times4$ matrix stored row by row. Element $(i,j)$ sits at offset $i\cdot4+j$ (0-indexed).

| | col 0 | col 1 | col 2 | col 3 |
|---|---|---|---|---|
| row 0 | 0 | 1 | 2 | 3 |
| row 1 | 4 | 5 | **6** | 7 |
| row 2 | 8 | 9 | 10 | 11 |

$(1,2)$ is at $1\cdot4+2=6$ .

- Walking **along a row** (4, 5, 6, 7): neighbours in memory, stride 1.
- Walking **down a column** (2, 6, 10): jumps of 4, stride $n$.

**Cache-line arithmetic.** A cache line is 64 bytes $=16$ float32 values. Walking a row loads one line per 16 elements (≈ 6% misses). Walking a column of a wide matrix touches a new line for almost every element (≈ 100% misses).

**Which loop order is friendly?** In $C_{ij}=\sum_kA_{ik}B_{kj}$ the innermost loop should move along a row of $B$ and $C$ (the $j$ index, i.e. IKJ), not down a column of $B$ (the $k$ index, i.e. IJK).

**Tile size by hand.** Three float32 tiles of side $t=32$: $3\cdot32^2\cdot4=12{,}288$ B $=12$ KiB, comfortably inside a 32 KiB L1 cache .

**Remember:** same arithmetic, different memory order, up to a several-fold time difference in compiled code.

## **12. Row-Major Layout and Loop Order**

### **12.1 Memory layout**

In row-major (C) order, the elements of row $i$ occupy contiguous addresses, and $A_{ij}$ sits at offset $i\cdot n + j$ (in elements). NumPy and PyTorch default to row-major; Fortran and MATLAB default to column-major. PyTorch exposes this as `tensor.is_contiguous()` and a `.t()` / `.contiguous()` pair, and the cost of that `.contiguous()` call is a full copy.

Row-major determines which access patterns are cheap. Walking a *row* is sequential and prefetch-friendly; walking a *column* strides by $n$ elements and touches one element per cache line. That single fact explains the entire performance ordering of the six loop permutations.

### **12.2 The six loop permutations**

The product $C = AB$ can be computed with the $i$, $j$, $k$ loops in any of six orders. All six are algebraically identical, and all six perform the same $mnk$ multiplications and $mn(k-1)$ additions. In compiled code they are not remotely equivalent in speed.

The classic analysis, from UT Austin's compiler course, counts cache misses under a small-cache model with $b$ elements per cache line ($b = 4$) and typical latencies of 10 cycles for an L2 hit and 70 for a miss [35]:

- **IJK** (i outer, j middle, k inner): the inner loop walks a *column* of $B$, which has poor spatial locality. Miss ratio $\to 0.25(b+1)/b = 0.3125$.
- **IKJ** (i outer, k middle, j inner): the inner loop walks a *row* of $B$, which has good locality. Total time $2N^3 + 70 \cdot 0.13 \cdot 4N^3 + 10 \cdot 0.87 \cdot 4N^3 = 73.2\,N^3$.
- **JKI**: $2N^3 + 70 \cdot 0.5 \cdot 4N^3 + 10 \cdot 0.5 \cdot 4N^3 = 162\,N^3$, a **2.2× slowdown** versus IKJ.
- **Blocked** with tile size $t$: miss ratio $0.75/(bt) = 0.001$ for $t=200$, giving $42.3\,N^3$, a **4× speedup** over JKI [35].

The headline number is that the *same mathematical operation*, reordering only the loops, varies by a factor of four in a compiled kernel. The "locality optimisation changed the order in which operations were done, not the number of operations" — the FLOP count is identical in all six [35].

Two caveats worth carrying. First, the same analysis notes the six permutations are equivalent "even modulo round-off error" in exact arithmetic, but §3.4 shows they are *not* equivalent in floating point. Second, and more practically, modern BLAS libraries do not expose loop order to the user at all: cuBLAS selects a kernel, tile shape, and split-K strategy automatically. Loop order matters when you write a kernel, when you choose a layout (row-major vs column-major operands), or when you write NumPy code that defeats the library's blocking by materialising intermediates.

### **12.3 Cache blocking and the working-set constraint**

Blocking partitions $A$, $B$, $C$ into tiles so that the working set

$$
\text{working set} \approx 3t^2 \cdot s
$$

($t$ = tile edge, $s$ = element size) fits in a *level-1* cache. For float32 and a 32 KiB L1: $3t^2 \cdot 4 \le 32768 \Rightarrow t \lesssim 52$, so $t = 32$ or $64$ is the right choice — not "whatever fits in L3". Blocking is a statement about L1, not about "cache" generically.

The cache-oblivious alternative removes the tuning parameter entirely: recursively split the largest dimension and recurse until the submatrices fit, giving $O(n^3/\sqrt{Z})$ cache misses where $Z$ is the cache size, versus $O(n^3/B)$ for the naive version and $O(n^2/L + n^3/L^{3/2})$ for a tuned blocked version with cache line size $L$ [36]. The catch, and it is a real one, is that cache-oblivious code has awkward control flow and worse constant factors than a hand-tuned kernel — which is why production BLAS libraries are all hand-tuned.

**A measured illustration of the trap.** Blocking is a compiled-code phenomenon, and it is worth showing why a Python/NumPy experiment can fail to demonstrate it. In §14.2, the naive outer-product formulation writes the *entire* $C$ matrix to memory once per $k$, while the blocked formulation writes only a tile, between 4× and 8× fewer element writes for the same arithmetic. Yet the blocked version was *slower* at every size tested, because the thousands of tiny `np.outer` calls it needs carry more per-call overhead than the traffic they save. Blocking pays only when the inner kernel is cheap enough that memory traffic dominates — which is precisely the regime of a compiled GEMM.

---

#### **bias as a rank-1 add**

Reuse $X=\begin{bmatrix}1&2\\3&-1\end{bmatrix}$ and $W=\begin{bmatrix}1&0&-1\\2&1&0\end{bmatrix}$ from §2, so $XW=\begin{bmatrix}5&2&-1\\1&-1&-3\end{bmatrix}$. Let the bias be $\mathbf{b}=[10,\,20,\,30]$.

Broadcast = add $\mathbf{b}$ to every row. As a matrix, that is the all-ones column times the bias row:

$$\mathbf{1}_2\mathbf{b}^\top=\begin{bmatrix}1\\1\end{bmatrix}[10,20,30]=\begin{bmatrix}10&20&30\\10&20&30\end{bmatrix}$$

$$Y=XW+\mathbf{1}\mathbf{b}^\top=\begin{bmatrix}5+10&2+20&-1+30\\1+10&-1+20&-3+30\end{bmatrix}=\begin{bmatrix}15&22&29\\11&19&27\end{bmatrix}$$

**Why "rank 1".** Every row of $\mathbf{1}\mathbf{b}^\top$ is the same row, so it contains only one independent direction.

**Bias is cheap in FLOPs, a little dear in bytes.** It adds $mn$ additions (tiny next to $2mnk$), but a GEMM with $\beta\neq0$ must also read the old $C$, dropping intensity from $n/3$ to $n/4$.

**Remember:** `X @ W + b` = matmul, then the same row added to every row. Always assert `b.shape == (X@W).shape[-1:]`.

## **13. Bias Addition as a Rank-1 Broadcast**

The linear layer of a transformer is

$$
\mathbf{y} = \mathbf{x}W + \mathbf{b}, \qquad \mathbf{x}\in\mathbb{R}^{1\times k},\; W \in \mathbb{R}^{k\times n},\; \mathbf{b}\in\mathbb{R}^{n}
$$

Applied to a batch of $m$ tokens, $X \in \mathbb{R}^{m\times k}$ and $Y \in \mathbb{R}^{m\times n}$, the bias is added to every row identically. Written as a matrix, that broadcast is an outer product:

$$
\boxed{Y = XW + \mathbf{1}_m\,\mathbf{b}^\top}
$$

and $\mathbf{1}_m\mathbf{b}^\top$ is a rank-1 matrix — the canonical rank-1 update. Numerically verified in §14.3: `X @ W + b` and `X @ W + ones @ b.reshape(1,-1)` agree bitwise, and the broadcast term has matrix rank exactly 1.

This is not a notation curiosity, it is the mechanism by which bias is implemented:

- **The GEMM epilogue.** BLAS GEMM is specified as $C = \alpha AB + \beta C$. A library computing $\alpha = 1$, $\beta = 1$ with $C$ pre-filled with the bias produces the whole linear layer in one call. PyTorch's `F.linear` dispatches to `addmm`, which does exactly this, and the bias never becomes a separate kernel or a separate tensor.
- **The intensity cost.** As noted in §10.2, the ideal $\beta = 0$ GEMM reads $A$ and $B$ once and writes $C$ once, giving $n/3$ FLOP/byte for a square FP16 GEMM. With $\beta \ne 0$, the previous $C$ must be read as well, and the intensity drops to $n/4$ — a 25% reduction in arithmetic intensity for free [26]. The bias is cheap in FLOPs and slightly expensive in bandwidth.
- **Bias-free architectures.** LLaMA and most modern decoder-only models drop biases entirely, in the FFN (SwiGLU's $b$ and $c$ terms are omitted in practice [10]) and, in LLaMA's case, in the projections too — relying on RMSNorm's learned per-channel gain to absorb the need for a bias shift [29]. BERT-style models, by contrast, keep biases everywhere. When comparing FLOP counts between model families, the bias terms are a small but non-zero part of the byte budget.
- **The axis trap.** As §6.3 showed, `X + b` and `X @ W + b` broadcast along different axes, and the mistake is silent because both shapes are legal.

---

#### **how to read the measurements**

**Predict before you look.** At $n=256$, float32: FLOPs $=2\cdot256^3=33{,}554{,}432$. At 102 GFLOP/s that is $33{,}554{,}432/102\times10^9\approx0.33$ ms — and the table shows $0.0003$ s . If a result disagrees with your back-of-envelope number by 10×, suspect the experiment, not the physics.

**Three takeaways in plain words.**

1. A tuned library (`A @ B`) beats hand-written loops by thousands of times — the arithmetic is identical, the memory schedule and vectorisation are not.
2. Blocking helps only when there is a cache problem to solve. At small sizes it just adds overhead.
3. Algebraic identities hold to *rounding error*, never to bit-exactness, so test with `np.allclose`, not `==`.

**Rounding scale to expect.** float64 relative error ≈ $10^{-16}$; float32 ≈ $10^{-7}$. A "difference" of $10^{-14}$ in float64 is noise.

**A first experiment you can run in 5 lines:**

```python
import numpy as np
A = np.random.randn(5, 7); B = np.random.randn(7, 3)
print(np.allclose((A @ B).T, B.T @ A.T))          # (AB)^T = B^T A^T
print(np.allclose(np.trace(A @ B @ A.T @ A),
                  np.trace(A @ A.T @ A @ B)))      # cyclic trace: tr(XYZ)=tr(ZXY)
```

**Remember:** timings test the machine, identities test the algebra, and the two are not the same experiment.

## **14. Implementation and Measurement**

The following code and numbers are first-party: generated on the machine described at the top of this module by `matrix_ops_bench.py` (delivered alongside these notes). Everything in §14.1–§14.4 is reproducible with `python3 matrix_ops_bench.py`.

### **14.1 Three matmul formulations**

The three implementations correspond to three of the readings in §2.2, and the code makes the correspondence literal.

**Three nested loops (Reading 1, the textbook form).**

```python
def matmul_ijk(A, B):
    """C[i][j] = sum_k A[i][k] * B[k][j].  Inner loop walks a COLUMN of B."""
    m, k = A.shape
    k2, n = B.shape
    C = np.zeros((m, n), dtype=A.dtype)
    for i in range(m):
        for j in range(n):
            s = 0.0
            for kk in range(k):
                s += A[i, kk] * B[kk, j]
            C[i, j] = s
    return C
```

**Outer products (Reading 4, the BLIS micro-kernel form).**

```python
def matmul_outer(A, B):
    """AB = sum_k A[:,k] (x) B[k,:]  -- k rank-1 updates, exactly as BLIS does it."""
    m, k = A.shape
    k2, n = B.shape
    C = np.zeros((m, n), dtype=A.dtype)
    for kk in range(k):
        C += np.outer(A[:, kk], B[kk, :])       # rank-1 update
    return C
```

**Blocked (the same algebra, a cache-resident working set).**

```python
def matmul_blocked_blas(A, B, block=128):
    """Cache-blocked: the working set A_tile + B_tile + C_tile stays in cache."""
    m, k = A.shape
    k2, n = B.shape
    C = np.zeros((m, n), dtype=A.dtype)
    for ii in range(0, m, block):
        i_end = min(ii + block, m)
        for kk in range(0, k, block):
            k_end = min(kk + block, k)
            A_blk = A[ii:i_end, kk:k_end]
            for jj in range(0, n, block):
                j_end = min(jj + block, n)
                C[ii:i_end, jj:j_end] += A_blk @ B[kk:k_end, jj:j_end]
    return C
```

### **14.2 Measured timings**

At $n = 256$, float32, row-major, on the 2-vCPU Xeon 8163 sandbox. The FLOP count is $2 \cdot 256^3 = 33{,}554{,}432$ and the compulsory traffic is 1,310,720 bytes, giving an ideal arithmetic intensity of 25.60 FLOP/byte.

| variant | time (s) | GFLOP/s | relative error | effective GB/s |
|---|---|---|---|---|
| ijk (textbook i-j-k) | 6.9991 | 0.00 | $3.51\times10^{-7}$ | 0.00 |
| ikj (row-friendly) | 7.6349 | 0.00 | $3.51\times10^{-7}$ | 0.00 |
| jki (column-friendly) | 7.5008 | 0.00 | $3.51\times10^{-7}$ | 0.00 |
| outer (View 4, rank-1) | 0.0140 | 2.40 | $3.51\times10^{-7}$ | 0.09 |
| blocked (tile 32, rank-1 inner) | 0.1325 | 0.25 | $3.51\times10^{-7}$ | 0.01 |
| **numpy `A @ B` (OpenBLAS)** | **0.0003** | **102.08** | $0.00$ | 3.99 |

Three things in that table deserve attention, and all three are about measurement rather than about algebra.

First, **the three loop permutations are indistinguishable** (7.00 s, 7.63 s, 7.50 s) even though the compiled-code analysis of §12.2 predicts a 2.2× spread between them. The reason is that in CPython every scalar access to a NumPy array is an interpreted operation costing far more than a cache miss. You are timing the interpreter, not the memory hierarchy. The 2.2× and 4× numbers in §12.2 are for compiled kernels, and the lesson for a Python learner is the opposite of "loop order doesn't matter": in Python you cannot observe it at all, which is precisely why you should call the library.

Second, **the blocked version was slower than the naive one** at $n = 256$ (0.13 s vs 0.014 s) — because at that size the entire problem ($3 \times 256^2 \times 4$ B = 768 KB) fits in the 1 MiB L2, so there was no locality problem for blocking to solve, only call overhead to add.

Third, **OpenBLAS is ~80,000× faster than the pure-Python loop and ~47× faster than the vectorised outer-product version**, and its 102 GFLOP/s is 45% of this machine's theoretical float32 peak for two AVX-512 FMA units at 2.5 GHz ($2 \times 16 \times 2 \times 2.5 \times 10^9 \approx 160$ GFLOP/s). That ratio is the value of a tuned library. It is not a better algorithm, since the arithmetic is the same $2mnk$, but a better memory schedule, better vectorisation, and FMA.

The blocking study, run at larger sizes with two inner kernels to separate the effects:

| $n$ | outer (s) | blocked, same inner op (s) | tiled GEMM (s) | tiled speedup |
|---|---|---|---|---|
| 256 | 0.0140 | 0.0114 | 0.0004 | 38.9× |
| 512 | 0.1398 | 0.1966 | 0.0033 | 44.2× |
| 1024 | 0.9685 | 1.5586 | 0.0473 | 20.1× |

The middle column is the honest negative result: with an identical inner operation, the blocked version writes 4–8× fewer elements to memory and is *still slower*, because NumPy's per-call overhead exceeds the traffic it saves. The right-hand column is what a real kernel does — tiled GEMM, with a compiled inner loop, wins by 20–44× over the naive outer-product formulation. The crossover is visible in a longer run of the same experiment: at $n = 3072$ (working set 188 MB, well beyond the 33 MiB L3) the blocked rank-1 version finally overtakes the naive one, $1.3\times$.

The OpenBLAS scaling run shows the same roofline story from the other side:

| $n$ | time (s) | GFLOP/s | ideal AI (FLOP/byte) | effective GB/s |
|---|---|---|---|---|
| 64 | 0.00001 | 79.32 | 6.40 | 12.39 |
| 128 | 0.00005 | 81.43 | 12.80 | 6.36 |
| 256 | 0.00032 | 105.13 | 25.60 | 4.11 |
| 512 | 0.00241 | 111.50 | 51.20 | 2.18 |
| 1024 | 0.02710 | 79.26 | 102.40 | 0.77 |
| 2048 | 0.27346 | 62.82 | 204.80 | 0.31 |

Arithmetic intensity rises linearly with $n$, exactly as $I = n/6$ predicts, and on this particular 2-vCPU shared machine throughput *peaks* at $n = 512$ and then degrades. Do not generalise that shape to a datacenter GPU: the degradation here reflects a two-core cloud VM with a single memory path, whereas an A100 sustains near-peak throughput far beyond $n = 2048$ by using 108 SMs with a deep register/SRAM hierarchy. The transferable lesson is the *intensity* column, not the GFLOP/s column. The crossover the table implies is instructive too: a float32 square GEMM reaches the A100's 153 FLOP/byte ridge at $n \approx 918$.

### **14.3 Numerical verification of the two identities**

**$(AB)^\top = B^\top A^\top$** on random matrices, with the maximum absolute discrepancy:

| $n$ | $\max\left|(AB)^\top - B^\top A^\top\right|$ | relative |
|---|---|---|
| 7 | $2.22\times 10^{-16}$ | $1.0\times 10^{-16}$ |
| 64 | $2.22\times 10^{-16}$ | $1.4\times 10^{-16}$ |
| 512 | $1.19\times 10^{-15}$ | $1.3\times 10^{-16}$ |
| 1000 | $2.84\times 10^{-15}$ | $2.8\times 10^{-16}$ |
| 4096 (float32) | $0.00$ | $0.00$ |

The identity holds to machine precision, and the discrepancy stays flat as $n$ grows — the error grows like $\sqrt{n}$ in the random-walk sense while the matrix entries grow like $\sqrt{n}$ too, so the *relative* error is constant. A rectangular case $(37\times 512)(512\times 91)$ was also checked, confirming the rule does not depend on squareness.

**$\mathrm{tr}(AB) = \mathrm{tr}(BA)$** on the same matrices, alongside the commutativity gap that the trace says nothing about:

| $n$ | $\left|\mathrm{tr}(AB) - \mathrm{tr}(BA)\right|$ | $\max|AB - BA|$ | ratio |
|---|---|---|---|
| 7 | $1.11\times 10^{-16}$ | $3.60$ | $3.2\times 10^{16}$ |
| 64 | $8.88\times 10^{-16}$ | $4.75$ | $5.3\times 10^{15}$ |
| 512 | $1.19\times 10^{-14}$ | $5.20$ | $4.4\times 10^{14}$ |
| 1000 | $2.27\times 10^{-14}$ | $5.32$ | $2.3\times 10^{14}$ |
| 4096 (float32) | $5.22\times 10^{-7}$ | $1.31\times 10^{-1}$ | $2.5\times 10^{6}$ |

The third column is the point of the whole table: the matrices differ by $O(1)$, by more than their own magnitude, while the traces agree to machine precision. The final row shows the same phenomenon at LLaMA-2-7B scale: $\max|AB - BA| = 0.131$ against $\max|AB| = 0.084$, a relative gap of 1.55, with the traces agreeing to $5.2\times 10^{-7}$ in float32.

The cyclic identity $\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)$ was checked on 400×400 matrices, with all three agreeing to $10^{-13}$.

**The four readings** on a $(128\times 256)(256\times 64)$ product:

| reading | max difference from `A @ B` |
|---|---|
| 1 — inner products | $9.24\times 10^{-14}$ |
| 2 — column combinations | $7.82\times 10^{-14}$ |
| 3 — row combinations | $7.11\times 10^{-14}$ |
| 4 — sum of outer products | $2.84\times 10^{-14}$ |

All four land within a factor of three of each other at machine precision, which is the concrete form of the claim that they are one operation rather than four.

**The trace identity for cross-entropy**, on a 64-example, 10-class softmax with logits scaled by 3:

```
NLL (row-wise)  = 297.226894154243
-tr(Y^T log P)  = 297.226894154243
|diff|          = 5.684e-14
```

**The special-matrix closures**, all verified as `True` in code: diagonal × diagonal is diagonal (with $\mathrm{diag}(D_1D_2) = [4, 10, 18]$); upper-triangular × upper-triangular is upper-triangular; lower-triangular × lower-triangular is lower-triangular; orthogonal × orthogonal is orthogonal; permutation × permutation is a permutation.

**LLaMA-2-7B parameter count**, computed from the published configuration [29] rather than quoted:

```
embedding                131,072,000
per layer (attention)     67,108,864
per layer (MLP)          135,266,304
per layer total          202,383,360
TOTAL PARAMETERS       6,738,415,616      (published value: 6,738,415,616 -> match)
```

The MLP line is $3 d_{\text{model}} d_{\text{ff}} = 3 \cdot 4096 \cdot 11008$, which is the SwiGLU structure of §6.2 showing up directly in the parameter budget: three matrices, not two.

### **14.4 Scope and transferability of the measurements**

- They are **not** GPU numbers. The 2-vCPU sandbox shares a memory path and has no Tensor Cores. Every GPU figure in this module comes from a vendor datasheet and is labelled as such.
- The pure-Python loop-order comparison is **not** a measurement of the loop-order effect. It is a measurement of interpreter overhead. The effect itself is documented analytically in §12.2 and realised in compiled kernels.
- The identity verifications **are** transferable: they test the algebra, not the hardware, and the errors are at the level of float32/float64 round-off.
- The LLaMA-2-7B arithmetic — parameter count, FLOPs per token, bytes per token, the roofline comparison — **is** transferable, because it is pure arithmetic over published hardware specifications. That is the computation worth internalising: the decode phase of a 7B model is memory-bound by a factor of exactly the machine balance, and no amount of arithmetic optimisation changes that.

---

#### **the numbers worth memorising first**

| # | fact | where it came from |
|---|---|---|
| 1 | matmul FLOPs $=2mnk$ | §10, worked in §2 (24 vs exact 20) |
| 2 | training FLOPs per token $\approx6N$ | §10 |
| 3 | square FP16 GEMM intensity $=n/3$ | §10 (341 at $n=1024$) |
| 4 | A100 ridge $=153$, H100 ridge $=295$ FLOP/byte | §11 ($312\text{T}/2.039\text{T}$) |
| 5 | batch-1 decode intensity ≈ 1 | §9 |

If you remember only these, you can already predict whether an operation will be compute-bound or memory-bound.

## **15. Reference Numbers**

A single table of the constants used above, for quick recall.

| constant | value | source |
|---|---|---|
| A100 80GB SXM, FP16 Tensor Core (dense) | 312 TFLOP/s | [25] |
| A100 80GB SXM, HBM2e bandwidth | 2,039 GB/s | [25] |
| A100 80GB SXM, machine balance (ridge) | 153 FLOP/byte | [25][26] |
| H100 SXM, FP16/BF16 Tensor Core (dense) | 989.4 TFLOP/s | [25] |
| H100 SXM, HBM3 bandwidth | 3,350 GB/s | [25] |
| H100 SXM, machine balance (ridge) | 295 FLOP/byte | [25] |
| HBM bandwidth on modern GPUs | 1–2 TB/s | [23] |
| On-chip SRAM bandwidth | 10–20 TB/s | [23] |
| SMs on an A100 | 108 | [31] |
| GEMM FLOPs, $M \times K \times N$ | $2MNK$ | [26] |
| Square FP16 GEMM intensity, $\beta = 0$ | $n/3$ FLOP/byte | [26] |
| Square FP16 GEMM intensity, $\beta \ne 0$ | $n/4$ FLOP/byte | [26] |
| Square FP32 GEMM intensity | $n/6$ FLOP/byte | [26] |
| Linear layer, batch 1, 4096 out / 1024 in | 1 FLOP/byte (memory-bound) | [27] |
| Linear layer, batch 512, same shape | 315 FLOP/byte (arithmetic-bound) | [27] |
| ReLU / LayerNorm intensity | 0.25 / < 10 FLOP/byte | [27] |
| Transformer forward FLOPs per token | $2N + 2n_{\text{layer}}n_{\text{ctx}}d_{\text{attn}}$ | [24] |
| Training FLOPs per token | $\approx 6N$; total $C \approx 6ND$ | [24] |
| MHSA forward FLOPs | $B(4S^2d_{\text{model}} + 8Sd_{\text{model}}^2)$ | [26] |
| LLaMA-2-7B parameters | 6,738,415,616 | [29], verified in §14.3 |
| LLaMA-2 config | $d{=}4096$, $L{=}32$, $h{=}32$, $d_{\text{ff}}{=}11008$, $V{=}32000$ | [29] |
| SwiGLU inner width | $\tfrac{8}{3}d_{\text{model}}$ | [10] |
| LoRA trainable parameters | $2Ld_{\text{model}}r$, $r \ll \min(d,k)$ | [5] |
| DeepSeek-V2 MLA KV-cache reduction | 93.3% | [18] |
| FlashAttention speedup / memory saving | 2–4× / 10–20× | [22] |
| vLLM PagedAttention throughput gain | 2–4× | [13] |
| Monarch dense-speedup | up to 2× | [16] |
| Compiled loop-order spread (IJK vs JKI) | 73.2 $N^3$ vs 162 $N^3$ (2.2×) | [35] |
| Compiled blocked-matmul time | 42.3 $N^3$ (4× over JKI) | [35] |
| Cache-oblivious matmul I/O complexity | $\Theta(n^3/\sqrt{Z})$ | [36] |
| Strassen rank of $\langle 2,2,2\rangle$ tensor | 7 | [1] |
| Best known asymptotic matmul bound (as of 2024) | $O(n^{2.371339})$ | [2] |

The two rows worth memorising are the ridge points, because every "this operation is unexpectedly slow" question resolves to a comparison against one of them.

---

## **16. Sources**

1. *StrassenNet: Neural Learning of Fast Matrix Multiplication Algorithms*, arXiv:2602.21797 — https://arxiv.org/html/2602.21797
2. *Practical aspects of fast matrix multiplication* — https://journals.pan.pl/Content/138658/PDF/acs-art04.pdf
3. *Analyzing Machine Learning Workloads Using a Detailed GPU Simulator* (arXiv:1811.08933) — https://arxiv.org/html/1811.08933v2
4. *Hardware-Centric Analysis of DeepSeek's Multi-Head Latent Attention* (arXiv:2506.02523) — https://arxiv.org/pdf/2506.02523
5. *LoRA: Low-Rank Adaptation of Large Language Models* (arXiv:2106.09685) — https://arxiv.org/html/2106.09685v1
6. *opt_einsum — Introduction (path finding)* — https://optimized-einsum.readthedocs.io/en/stable/path_finding.html
7. *opt_einsum — Dynamic Programming Path* — https://dgasmith.github.io/opt_einsum/paths/dp_path/
8. *LoRMA: Low-Rank Multiplicative Adaptation for LLMs* (arXiv:2506.07621) — https://arxiv.org/html/2506.07621v1
9. *PyTorch — Numerical accuracy* — https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html
10. *SwiGLU (AI Wiki)* — https://aiwiki.ai/wiki/swiglu
11. *Tile-Level Activation Overlap for Efficient LLM Inference* (arXiv:2607.02521) — https://arxiv.org/pdf/2607.02521
12. *Batched Low-Rank Adaptation of Foundation Models* (arXiv:2312.05677) — https://arxiv.org/html/2312.05677v3
13. *Efficient Memory Management for Large Language Model Serving with PagedAttention* (arXiv:2309.06180) — https://arxiv.org/pdf/2309.06180
14. *vAttention: Dynamic Memory Management for Serving LLMs without PagedAttention* (arXiv:2405.04437) — https://arxiv.org/html/2405.04437v1
15. *Block-Diagonal K-FAC: A Trade-off Between Curvature Information and Resource Efficiency* — https://opt-ml.org/papers/2025/paper35.pdf
16. *Monarch: Expressive Structured Matrices for Efficient and Accurate Training* (ICML 2022) — https://proceedings.mlr.press/v162/dao22a.html
17. *Group and Shuffle: Efficient Structured Orthogonal Parametrization* (arXiv:2406.10019) — https://arxiv.org/html/2406.10019v1
18. *DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model* (arXiv:2405.04434) — https://arxiv.org/html/2405.04434v5
19. *SnapMLA: Efficient Long-Context MLA Decoding* (arXiv:2602.10718) — https://arxiv.org/html/2602.10718v1
20. *Stability Properties of Graph Neural Networks* (arXiv:1905.04497) — https://arxiv.org/html/1905.04497v5
21. *A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems* — https://hal.science/hal-04409413/file/Hitchhiker's%20Guide.pdf
22. *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* (arXiv:2307.08691) — https://arxiv.org/html/2307.08691v1
23. *LLM Inference Handbook — FlashAttention* (Modular) — https://handbook.modular.com/kernel-optimization/flashattention/
24. *Scaling Laws for Neural Language Models* (arXiv:2001.08361) — https://ar5iv.labs.arxiv.org/html/2001.08361
25. *NVIDIA A100 Tensor Core GPU specifications* — https://www.nvidia.com/en-us/data-center/a100/
26. *Machine Learning Systems: Algorithm Foundations appendix* — https://mlsysbook.ai/vol1/backmatter/appendix_algorithm.html
27. *NVIDIA Deep Learning Performance — GPU Performance Background* — https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html
28. *ROCm — Understanding GPU performance (Roofline model)* — https://rocm.docs.amd.com/projects/HIP/en/docs-7.2.0/understand/performance_optimization.html
29. *Llama 2 model documentation* (Hugging Face Transformers) — https://huggingface.co/docs/transformers/model_doc/llama2
30. *Performance Engineering for a Tall & Skinny Matrix Multiplication on GPUs* (arXiv:1905.03136) — https://arxiv.org/pdf/1905.03136v1
31. *DEFT: Decoding with Flash Tree-Attention* (arXiv:2404.00242) — https://arxiv.org/pdf/2404.00242v3
32. *Linear Algebra Review and Reference* (Stanford CS229) — https://cs229.stanford.edu/summer2020/cs229-linalg.pdf
33. *Applied Linear Algebra — Lesson 11A: Matrix-Matrix Multiplication* (Anderson) — https://static1.squarespace.com/static/53bc694ee4b00ad4e0bd6d55/t/625d78d0495a0a2b81f76721/1650292944748/Anderson_Applied_Linear_Algebra_Lesson_11A_Matrix_matrix_Multiplication_draft_v20220329.pdf
34. *Matrices — outer product and the four views of multiplication* (George Rowlands) — https://www.georgerowlands.ch/garden/maths/linearAlgebra/matrices
35. *CS 380C: Advanced Topics in Compilers — loop order and cache behaviour* (UT Austin, P. Pingali) — https://www.cs.utexas.edu/~pingali/CS380C/2024/lectures/introduction.pdf
36. *Cache-Oblivious Algorithms* (TU Dresden) — https://os.inf.tu-dresden.de/Studium/ReadingGroupArchive/slides/2015/20150903-planeta-cache-oblivious.pdf
37. *Faster Than Flash: Exploiting Attention Sparsity for Efficient Long-Context Decoding* (arXiv:2609.00097) — https://arxiv.org/html/2609.00097v1
38. *Resolving Discrepancies in Compute-Optimal Scaling of Language Models* (arXiv:2406.19146) — https://arxiv.org/pdf/2406.19146v4

