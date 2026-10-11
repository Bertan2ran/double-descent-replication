# Sample-Wise Double Descent (ATDL Group 8)

Replication of sample-wise non-monotonicity on CIFAR-10 with a 5-layer CNN (k=24) under 20% label noise, based on Nakkiran et al. (2019)

---

## 1. Requirements

Install required dependencies:

```bash
pip install torch torchvision pandas matplotlib numpy
```

---

## 2. Reproduce Figures

Generate the report plots directly from the pre-computed CSV logs in `results/`:

```bash
python plot_sample_descent.py
```

---

## 3. Training Models

### Train a Single Run (e.g., n = 12,500)
```bash
python train.py --width 24 --samples 12500 --noise 0.20 --epochs 1000 --lr 0.0001 --batch_size 128 --out results/sample_n12500_1000ep.csv
```

### Run Full 10-Point Sample Sweep (n in [6250, 50000])
```bash
for SAMPLES in 6250 10000 12500 15000 20000 25000 30000 37500 45000 50000; do
    echo "Training sample size: n=${SAMPLES}"
    python train.py --width 24 --samples ${SAMPLES} --noise 0.20 --epochs 1000 --lr 0.0001 --batch_size 128 --out results/sample_n${SAMPLES}_1000ep.csv
done
```
