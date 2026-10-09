import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('epoch_wise_results.csv')

# Set up the plot styled for an academic paper (ISBI format)
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(6, 4)) # Compact size for double-column layout

# Plot Train and Test Error
ax.plot(df['epoch'], df['test_err'], label='Test Error', color='#2ca02c', linewidth=1.5)
ax.plot(df['epoch'], df['train_err'], label='Train Error', color='#1f77b4', linewidth=1.5, linestyle='--')

# Format axes (Log scale for X as used in the paper)
ax.set_xscale('log')
ax.set_xlabel('Epochs', fontsize=10, fontweight='bold')
ax.set_ylabel('Error', fontsize=10, fontweight='bold')
ax.set_title('Epoch-Wise Double Descent (CNN k=128, 20% Noise)', fontsize=11, fontweight='bold')

# Limits and Legend
ax.set_ylim([0.0, 0.8])
ax.legend(loc='upper right', frameon=True, fontsize=9)
plt.tight_layout()

# Export as vector graphic
plt.savefig('epoch_wise_plot.pdf', format='pdf', bbox_inches='tight')
plt.show()