import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# List of models you want to analyze
models = ["deberta-v3-base", "deberta-v3-small", "deberta-v3-xsmall", "ToxicityModel"]

# Initialize a list to hold all logs
all_logs = []

# Load all model logs
for model in models:
    file_path = f'./content/{model}/checkpoint-2000/trainer_state.json'  # Construct the file path

    # Load the current model's log data
    with open(file_path, 'r') as file:
        data = json.load(file)  # Convert file contents to a dictionary

    log_history = data['log_history']  # Extract 'step' and 'loss' information

    # Convert log history to a DataFrame and assign a new column with the current model's name
    df = pd.DataFrame(log_history)
    df['model'] = model  # Differentiate the logs per model
    if model == "ToxicityModel":
        df['model'] = "roberta-base"

    # Add this DataFrame to our list
    all_logs.append(df)

# Concatenate all our individual DataFrames into one
combined_df = pd.concat(all_logs, ignore_index=True)

# Create a plot with a line for each model's 'loss' over 'step'
plt.figure(figsize=(12, 8))

# 'hue' differentiates the lines based on the model names (i.e., the 'model' column in our DataFrame)
sns.lineplot(x='step', y='loss', hue='model', data=combined_df, marker='o', palette="viridis")

plt.title('Train Loss comparison among models')
plt.xlabel('Step')
plt.ylabel('Train Loss')
plt.grid(True)
plt.legend(title='Models')  # Add a legend to differentiate the models

# Show the plot
plt.show()

# If you want to save the figure, you need to do it before plt.show()
plt.savefig('./train_loss_comparison.png')  # Save the figure
