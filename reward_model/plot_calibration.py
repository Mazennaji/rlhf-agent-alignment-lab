import json
import matplotlib.pyplot as plt

with open("reward_model/logs/calibration_history.json") as f:
    history = json.load(f)

fig, ax = plt.subplots(figsize=(8, 5))
for name, epochs in history.items():
    val_accs = [e["val_acc"] for e in epochs]
    ax.plot(range(1, len(val_accs) + 1), val_accs, label=name, alpha=0.7)

ax.set_xlabel("Epoch")
ax.set_ylabel("Validation Preference Accuracy")
ax.set_title("Reward Model Calibration Across Ensemble")
ax.legend()
fig.savefig("reward_model/logs/calibration_plot.png")
plt.show()