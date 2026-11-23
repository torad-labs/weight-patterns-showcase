# Gradient competition in the archived directional update

The preceding native test found an adverse local first-order effect in the full-graph proposal. This diagnostic measures the signed projection gradient itself in the original graph and computes the smallest continuous adjustment that cancels the adverse component. It then applies the existing endpoint conversion to scratch chunks and reads the effective displacement. [Native predecessor](../directional-intervention-fixed-history/README.md) · [Declaration](declaration.json)

Let $F$ be the frozen full objective and $p$ the historical cell's signed projection. Let $g=\nabla p$ and $u=-\alpha\nabla F$ be the original continuous weight step. At the original state the cell lies inside its radial tolerance and outside its signed allowance. Its capped loss gradient is a positive scalar multiple of $g$, so either gradient gives the same linear boundary.

The nearest feasible step in the controller's Euclidean parameter metric is

$$
u'=u-\frac{\langle g,u\rangle}{\langle g,g\rangle}g.
$$

For this positive inner product, the projection gives $g^\mathsf{T}u'=0$. The adjustment has norm $\langle g,u\rangle/\|g\|$. The loss of global first-order progress is $\alpha\langle g,-\nabla F\rangle^2/\|g\|^2$, or the squared cosine fraction of the original global progress. These expressions use the same physical leaf coordinates, aliases, sparse embedding rows, and step length as the archived graph.

The rounded readout uses the exact BF16 and FP32 endpoint values computed by the original controller's read-only preview. It contracts their differences with the same gradients to forecast first-order movement. It contains no learned selector, new model installation, or new generation. Measurement source

## Continuous result

| Quantity | Measured value |
|---|---:|
| Signed target gradient norm $\|g\|$ | 35.60062884 |
| Historical capped cell loss gradient norm | $1.58762330\times10^{-5}$ |
| Signed gradient inner product with $-\nabla F$ | $+7.02181131\times10^{-3}$ |
| Signed gradient inner product with the proposed weight step $u$ | $+3.55503722\times10^{-5}$ |
| Cosine with the joint descent direction | $+0.01392911$ |
| Original weight step norm | $7.16907593\times10^{-5}$ |
| Minimum adjustment norm $\|u'-u\|$ | $9.98588320\times10^{-7}$ |
| Global first-order change, original step | $-1.01515268\times10^{-6}$ |
| Global first-order change, projected step | $-1.01495572\times10^{-6}$ |
| Global first-order improvement forgone | $1.96959969\times10^{-10}$, or 0.019402% |

The minimum adjustment is 1.3929% of the original step norm. The signed weight-gradient product reproduces the archived forward prediction to $8.84\times10^{-11}$ in projection units. Under gradient descent the selected cell's own loss reduces its signed projection, while the other loss terms taken together increase it. Their respective gradients have inner products $+0.000565204$ and $-0.007587015$ with $g$. Continuous output

## Native precision preview

| Displacement used in the linear forecast | Signed projection change | Global first-order loss change |
|---|---:|---:|
| Archived rounded step | $+3.60677714\times10^{-5}$ | $-8.25690266\times10^{-7}$ |
| Projected step after the direction's FP32 cast | $+3.12177\times10^{-12}$ | $-1.01495571\times10^{-6}$ |
| Projected step after exact stored-weight rounding | $-4.91336936\times10^{-6}$ | $-8.25464586\times10^{-7}$ |

The rounded projected step would change 31,552 native values, compared with 31,564 in the archived update. Its forecast retained 99.9727% of the original rounded step's global first-order descent and moved the signed target in the desired direction. These are first-order forecasts on the frozen historical state. The actual finite native effect and any response change remain unmeasured. [Results](result.json) · Per-tensor products · [MLflow run](http://127.0.0.1:5000/#/experiments/6/runs/28855399952348b9a7c465a98cfa2651)
