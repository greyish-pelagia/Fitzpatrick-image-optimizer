1. README overclaims compared with implementation.
   README.md:3 says the models restore medical imagery to “optimal diagnostic lighting” and “correct visual acquisition biases.” README.md:61 claims downstream diagnostic gains, but the repo does not show downstream
   classifier experiments, bias metrics by Fitzpatrick group, or clinical validation.
   Change: reframe as an experimental illumination-normalization project, clearly state limitations, and add per-skin-type metrics.
2. DeepLPF description does not match the code.
   README.md:52 says the model predicts 76 parameters for graduated, elliptical, and polynomial filters. But src/train_deeplpf.py:126 explicitly says the implementation uses differentiable proxies, and src/
   train_deeplpf.py:134 reduces many parameters to means or only uses the first few polynomial terms.
   Change: either implement the actual DeepLPF-style filters or rename this as a simplified parameter-conditioned residual filter.
3. Evaluation is not a credible ML evaluation yet.
   src/evaluate_deeplpf.py:48 samples from the same CSV used for training, and there is no train/val/test split. The published metrics in README.md:9 are therefore hard to trust.
   Change: add deterministic splits, holdout evaluation, fixed seeds, baseline comparisons, and metrics grouped by Fitzpatrick scale.
4. Missing images silently become black images.
   src/train_deeplpf.py:103 replaces missing or unreadable images with zero arrays. That hides data problems and can corrupt training/evaluation.
   Change: fail fast, filter invalid rows during dataset construction, and report skipped samples.
5. The “Retinex-Fuzzy-CNN” module is underspecified and partly nominal.
   src/train_retinex.py:238 predicts illumination and reflectance, but only R_adj and the raw image feed the output path; L is only regularized through TV loss at src/train_retinex.py:299. The “CDH” module is Sobel
   filters, not a real color-difference histogram.
   Change: either implement/justify the named methods rigorously or simplify naming to what the code actually does.
6. Repo is script-oriented, not package-oriented.
   Importing as a package fails: import src.train_deeplpf raises ModuleNotFoundError: No module named 'utils', because imports like src/train_deeplpf.py:15 assume direct script execution.
   Change: restructure into modules such as fitzpatrick_optimizer/models.py, datasets.py, metrics.py, train.py, infer.py, and use package-relative imports.
7. Public demo path is incomplete.
   The tracked repo includes CSVs but not images or weights. README.md:28 links only DeepLPF weights, while Retinex is WIP. Local ignored data is large: data/ is about 2.7 GB, models/ about 228 MB.
   Change: add a tiny public sample dataset, model-card-style weight links, checksums, and one command that runs inference end to end after clone.
8. Project metadata/tooling looks unfinished.
   pyproject.toml:4 still says Add your description here; there are no pytest/ruff/mypy/pre-commit configs; Python is pinned to 3.13 via pyproject.toml:6 and .python-version:1.
   Change: add quality tooling and consider supporting Python 3.11/3.12 unless 3.13 is required.

Priority Changes Before Publishing

1. Make the README honest and recruiter-friendly: problem, method, limitations, quickstart, results table, example outputs, reproducibility notes.
2. Add proper train/val/test splitting and publish metrics by Fitzpatrick scale, not just aggregate L1/MSE/PSNR/SSIM.
3. Add tests for preprocessing, dataset loading, model forward shapes, metrics, and CLI smoke paths.
4. Refactor scripts into importable modules with clean interfaces.
5. Add a minimal demo asset path: uv run ... should work on a tiny sample without requiring private local data.
6. Add a model card: dataset source, synthetic degradation assumptions, known limitations, fairness caveats, intended/non-intended use.
7. Remove or rewrite inflated wording like “diagnostic,” “perfectly preserve,” “elimination of bias,” unless backed by experiments.
8. Fix naming consistency: repo says Fitzpatric, dataset says Fitzpatrick; use Fitzpatrick consistently.
