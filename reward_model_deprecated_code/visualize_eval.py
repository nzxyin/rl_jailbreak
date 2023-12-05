import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Raw data
raw_data = """
Model: deberta-v3-base, Checkpoint: 500, Average Softmax: 0.463748068464842
Model: deberta-v3-base, Checkpoint: 1000, Average Softmax: 0.4731802226590569
Model: deberta-v3-base, Checkpoint: 1500, Average Softmax: 0.4713162223164219
Model: deberta-v3-base, Checkpoint: 2000, Average Softmax: 0.4571221584154865
Model: deberta-v3-small, Checkpoint: 500, Average Softmax: 0.4795024718435007
Model: deberta-v3-small, Checkpoint: 1000, Average Softmax: 0.4664729868532509
Model: deberta-v3-small, Checkpoint: 1500, Average Softmax: 0.4582199113432367
Model: deberta-v3-small, Checkpoint: 2000, Average Softmax: 0.45068620573261947
Model: deberta-v3-xsmall, Checkpoint: 500, Average Softmax: 0.47630966614081727
Model: deberta-v3-xsmall, Checkpoint: 1000, Average Softmax: 0.4744240191817426
Model: deberta-v3-xsmall, Checkpoint: 1500, Average Softmax: 0.46466810869914227
Model: deberta-v3-xsmall, Checkpoint: 2000, Average Softmax: 0.4684397450805138
Model: roberta-base, Checkpoint: 500, Average Softmax: 0.014965838690553982
Model: roberta-base, Checkpoint: 1000, Average Softmax: 0.011981270394680655
Model: roberta-base, Checkpoint: 1500, Average Softmax: 0.01050897735523688
Model: roberta-base, Checkpoint: 2000, Average Softmax: 0.01020799154711438
"""

# Parsing raw data into a structured DataFrame
data_lines = raw_data.strip().split("\n")
data_parsed = [line.split(", ") for line in data_lines]

# Create a DataFrame
df = pd.DataFrame(data_parsed, columns=["Model", "Checkpoint", "Average Softmax"])

# Convert columns to appropriate data types
df["Checkpoint"] = df["Checkpoint"].str.split(": ").str[1].astype(int)
df["Average Softmax"] = df["Average Softmax"].str.split(": ").str[1].astype(float)

# Visualizing with seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(data=df, x='Checkpoint', y='Average Softmax', hue='Model', style='Model', marker="o", dashes=False)

plt.title('Eval Loss comparison among models')
plt.xlabel('Step')
plt.ylabel('Eval Loss')
plt.grid(True)
plt.legend(title='Models')
plt.show()
plt.savefig('./eval_loss_comparison.png')  # Save the figure
