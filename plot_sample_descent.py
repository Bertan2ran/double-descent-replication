import glob
import pandas as pd
import matplotlib.pyplot as plt

def get_final_metrics(pattern):
    results = []
    for filepath in sorted(glob.glob(pattern)):
        df = pd.read_csv(filepath)
        final_row = df.iloc[-1]
        results.append({
            "samples": int(final_row["samples"]),
            "test_err": float(final_row["test_err"]),
            "train_err": float(final_row["train_err"])
        })
    return pd.DataFrame(results).sort_values("samples")

noisy_df = get_final_metrics("results/sample_n*_noisy.csv")
clean_df = get_final_metrics("results/sample_n*_clean.csv")

plt.figure(figsize=(6, 4))
if not noisy_df.empty:
    plt.plot(noisy_df["samples"], noisy_df["test_err"], marker="o", color="#d62728", label="Test Error (20% Noise)")
    plt.plot(noisy_df["samples"], noisy_df["train_err"], marker="x", linestyle="--", color="#d62728", alpha=0.5, label="Train Error (20% Noise)")

if not clean_df.empty:
    plt.plot(clean_df["samples"], clean_df["test_err"], marker="s", color="#1f77b4", label="Test Error (0% Clean)")

plt.xlabel("Number of Training Samples (n)")
plt.ylabel("Error")
plt.title("Sample-Wise Non-Monotonicity (CNN Width k=24)")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()
plt.savefig("results/figure3_sample_descent.pdf")
plt.show()