import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV files and strip spaces from column names
base_df = pd.read_csv('csv_results/fed_avg_vs_attacks/base(pattern)_unweighted_fedavg_cifar10_dirichlet(0.5)_random_multi-shot_v0.csv')
base_df.columns = base_df.columns.str.strip()

neurotoxin_df = pd.read_csv('csv_results/fed_avg_vs_attacks/neurotoxin(pattern)_unweighted_fedavg_cifar10_dirichlet(0.5)_random_multi-shot_v0.csv')
neurotoxin_df.columns = neurotoxin_df.columns.str.strip()

chameleon_df = pd.read_csv('csv_results/fed_avg_vs_attacks/chameleon(pattern)_unweighted_fedavg_cifar10_dirichlet(0.5)_random_multi-shot_v0.csv')
chameleon_df.columns = chameleon_df.columns.str.strip()

# Set style
sns.set_theme()
sns.set_palette("husl")

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle('Comparison of Different Pattern Attacks', fontsize=14)

# Plot clean test accuracy
ax1.plot(base_df['round'], base_df['test_clean_acc'], label='Base Pattern', marker='o', markersize=3)
ax1.plot(neurotoxin_df['round'], neurotoxin_df['test_clean_acc'], label='Neurotoxin Pattern', marker='o', markersize=3)
ax1.plot(chameleon_df['round'], chameleon_df['test_clean_acc'], label='Chameleon Pattern', marker='o', markersize=3)
ax1.set_title('Clean Test Accuracy')
ax1.set_xlabel('Round')
ax1.set_ylabel('Accuracy')
ax1.legend()
ax1.grid(True)

# Plot backdoor test accuracy
ax2.plot(base_df['round'], base_df['test_backdoor_acc'], label='Base Pattern', marker='o', markersize=3)
ax2.plot(neurotoxin_df['round'], neurotoxin_df['test_backdoor_acc'], label='Neurotoxin Pattern', marker='o', markersize=3)
ax2.plot(chameleon_df['round'], chameleon_df['test_backdoor_acc'], label='Chameleon Pattern', marker='o', markersize=3)
ax2.set_title('Backdoor Test Accuracy')
ax2.set_xlabel('Round')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)

# Adjust layout and save
plt.tight_layout()
plt.savefig('attack_comparison.png', dpi=300, bbox_inches='tight')
plt.close() 