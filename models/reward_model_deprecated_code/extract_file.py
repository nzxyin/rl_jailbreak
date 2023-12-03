import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset
file_path = './english-00000-of-00001-21bd3a8cceb500f6.parquet'
data = pd.read_parquet(file_path)

# Split the data into 90% and 10%
train, test = train_test_split(data, test_size=0.1, random_state=42)  # random_state is optional, for reproducibility

print(len(train), len(test))
# Save to CSV
train.to_csv('toxic_aria_train_data.csv', index=False)  # Set index=False to avoid saving index column
test.to_csv('toxic_aria_test_data.csv', index=False)
