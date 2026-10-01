# DEG Scientific Contract 1.0

This benchmark uses PyDESeq2 0.4.12 with the supplied
`deg_inference.ObjectivePreservingInference` computation package. It is an
explicit estimator contract, not an attempt to reproduce undocumented historical
software settings. The package contains numerical primitives only. Implement
the analysis, biological input handling, result assembly, and enrichment yourself.

## Runtime and Public API

Use Python 3.10 and the six versions in `requirements.txt`, with transitive
versions constrained by `runtime_env/constraints.txt`. For example, from the
task directory, create an agent-owned environment and install:

```
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r input/requirements.txt -c input/runtime_env/constraints.txt
```

Set `PYTHONDONTWRITEBYTECODE=1` and include `input/runtime_env` on `PYTHONPATH`
when running your analysis. Do not modify the input package or installed
PyDESeq2. No package-file patching or hidden output lookup is needed.

Use the same `ObjectivePreservingInference(n_cpus=4)` instance as the
`inference=` argument of BOTH `DeseqDataSet` and `DeseqStats`. The 0.4.12
dataset API expresses the formula using `design_factors=["batch", "condition"]`
and `ref_level=["condition", "normal"]`; it does not accept the newer `design=`
keyword. Pass raw integer counts as samples by genes, ordered by metadata.
Keep all 17,498 genes and all eight samples. Batch and condition are categorical.
Do not prefilter counts, substitute normalized data, collapse genes, reorder
samples independently of metadata, or apply LFC shrinkage.

Dataset parameters: `fit_type="parametric"`, `min_mu=0.5`, `min_disp=1e-8`,
`max_disp=10.0`, `refit_cooks=True`, `min_replicates=7`, `beta_tol=1e-8`.
Call `dds.deseq2()`. Use `DeseqStats` with
`contrast=["condition", "tumor", "normal"]`, `alpha=0.05`,
`cooks_filter=True`, `independent_filter=True`, `lfc_null=0`,
`alt_hypothesis=None`; call `summary()`. No `lfc_shrink()`.
Normalization, parametric dispersion trend, prior estimation, dispersion outlier
selection, Cook refitting/filtering and independent filtering otherwise follow
the pinned PyDESeq2 implementation. The default median-of-ratios normalization
uses genes positive in every sample; it does not drop other genes from fitting.

## Numerical Estimator

For counts y, size factors s and design X, coefficient estimation minimizes
the negative-binomial negative log likelihood at
`mu_i = max(s_i * exp(X_i beta), 0.5)` plus `0.5e-6 * sum(beta**2)`, with
each natural-log coefficient in `[-30,30]`. The derivative includes the
clipping indicator. This is not the stock IRLS stopping rule. Positive-count
clipping regions are enumerated, impossible-to-improve regions are pruned by a
nonnegative saturated-likelihood bound, and zero-count terms use a convex
epigraph. SLSQP fits are refined by constrained stationarity equations and,
when needed, trust-constr or full eight-sample region enumeration.
Accepted coefficient fits require finite certificates, KKT residual at most
`1e-5` and constraint slack at least `-1e-7`; an uncertified fit raises an error.
The package is intentionally restricted to this eight-sample contract.

For fixed genewise means, dispersion fitting minimizes NB negative log
likelihood plus `0.5 * logdet(X.T @ diag(mu/(1+alpha*mu)) @ X)` (Cox-Reid).
MAP additionally uses `(log(alpha)-log(trend))**2/(2*prior_variance)`.
The prior must be positive and finite and is never dropped on fallback.
Use bounded scalar minimization over log dispersion `[log(1e-8),log(10)]`,
`xatol=1e-12`, with both endpoints compared explicitly. The NB gamma ratio is
evaluated stably at large inverse dispersion. These choices and all numerical
constants are implemented in the supplied source; do not retune them.

As required by the pinned inference interface, coefficient fitting returns
UNFLOORED means for subsequent pipeline stages. Hat weights use floored means
and a `1e-6` information ridge. The Wald test uses unfloored fitted means and
the pinned ridge-sandwich standard error, a two-sided zero-null normal test,
and log2 units for the reported effect and standard error. Independent filtering
uses the pinned 50-quantile rejection-curve rule at alpha 0.05, followed by BH
within the retained family. Missing adjusted probabilities remain missing.

The estimator does not sample or bootstrap, so no RNG seed is required.
Limit inference to at most four processes and set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`. CPU dispatch can cause small floating
point differences; no score epsilon or scoring tolerance is changed for this.

## Output and Enrichment

Report all genes with symbols from the supplied mapping. Use strict thresholds:
`log2FoldChange > 1` and `padj < .05` is upregulated;
`log2FoldChange < -1` and `padj < .05` is downregulated; all other rows,
including missing padj, are `no significant`. Numeric zero probabilities are
valid. Do not round probabilities or coefficients before classifying.

For each direction, deduplicate nonblank gene symbols in input order and query
the authentic `KEGG_2021_Human` library for `Human`. Use the normal Enrichr
service background, not a user-defined experimental background. In gseapy 1.1.5
the convenience wrapper has an HTTP default. Use its public class interface:

```python
from gseapy.enrichr import Enrichr

enrichment = Enrichr(gene_list=genes, gene_sets="KEGG_2021_Human",
                    organism="Human", background=None, outdir=None,
                    no_plot=True, cutoff=0.05)
enrichment.ENRICHR_URL = "https://maayanlab.cloud"
enrichment.set_organism()
enrichment.run()
```

Write every returned term, not only significant terms. Order each table by
`Adjusted P-value`, then `P-value`, then `Term`, ascending, retaining the
columns specified in `output_contract.json`. Preserve the server's numeric
values and overlaps. A failed network request is not an empty enrichment result.

## Method Sources

- https://pydeseq2.readthedocs.io/en/v0.4.12/api/docstrings/pydeseq2.dds.DeseqDataSet.html
- https://pydeseq2.readthedocs.io/en/v0.4.12/api/docstrings/pydeseq2.ds.DeseqStats.html
- https://github.com/scverse/PyDESeq2/tree/v0.4.12/pydeseq2
- https://doi.org/10.1186/s13059-014-0550-8
- https://github.com/zqfang/GSEApy/blob/v1.1.5/gseapy/enrichr.py
- https://maayanlab.cloud/Enrichr/help#api
