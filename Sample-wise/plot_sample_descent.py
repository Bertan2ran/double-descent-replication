import glob
import os
import matplotlib.pyplot as plt
import pandas as pd


def get_final_metrics(pattern):
    results = []
    for filepath in sorted(glob.glob(pattern)):
        df = pd.read_csv(filepath)
        # drop epochs where test_err was not evaluated
        eval_rows = df.dropna(subset=["test_err"])
        if eval_rows.empty:
            continue
        final_row = eval_rows.iloc[-1]
        results.append({
            "samples": int(final_row["samples"]),
            "test_err": float(final_row["test_err"]),
            "train_err": float(final_row["train_err"]),
        })
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("samples")


noisy_df = get_final_metrics("results/sample_n*_1000ep.csv")
clean_df = get_final_metrics("results/sample_n*_clean.csv")

os.makedirs("results", exist_ok=True)
plt.figure(figsize=(7, 4.5))

if not noisy_df.empty:
    plt.plot(
        noisy_df["samples"],
        noisy_df["test_err"],
        marker="o",
        linewidth=2,
        color="#d62728",
        label="Test Error (20% Noise)",
    )
    plt.plot(
        noisy_df["samples"],
        noisy_df["train_err"],
        marker="x",
        linestyle="--",
        color="#d62728",
        alpha=0.6,
        label="Train Error (20% Noise)",
    )

if not clean_df.empty:
    plt.plot(
        clean_df["samples"],
        clean_df["test_err"],
        marker="s",
        linewidth=2,
        color="#1f77b4",
        label="Test Error (0% Noise)",
    )

plt.xlabel("Number of Training Samples (n)", fontsize=11)
plt.ylabel("Error", fontsize=11)
plt.title("Sample-Wise Non-Monotonicity (CNN Width k=24)", fontsize=12)
plt.legend(frameon=True)
plt.grid(True, linestyle=":", alpha=0.6)
plt.tight_layout()

# both formats saving
plt.savefig("results/figure3_sample_descent_1000epoch.pdf")
plt.savefig("results/figure3_sample_descent1000epoch.png", dpi=300)
print("Plot successfully saved to results/figure3_sample_descent.png and .pdf")