# When to Trust Kernel Change?

Code for "When to Trust Kernel Change? A Predictive Validation Framework for Representation Diagnostics."

## Repository structure

```
experiments/    — Experiment and analysis scripts (Python)
figures/        — Figure generation scripts
```

## Experiments

All experiments use the shared metric library in `experiments/representational_metrics.py`. Key scripts:

- `exp_scaleup.py` — main synthetic experiments (3 targets × 6 widths × 10 seeds)
- `exp_hermite_sweep.py` — Hermite polynomial targets
- `exp_cnn_cifar.py` — CNN on CIFAR-10
- `exp_vit_mnist.py` — Vision Transformer on MNIST
- `exp_nlp_sst2.py` — Transformer on SST-2
- `exp_pretrained_rdep.py` — fine-tuning pre-trained models
- `exp_null_baseline.py` / `random_label_baseline.py` — shuffled-label baselines

## Requirements

Python 3.9+, PyTorch, numpy, scipy, matplotlib, joblib, scikit-learn
