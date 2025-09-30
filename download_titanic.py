import pandas as pd
import os

os.makedirs("data", exist_ok=True)

# Direct links to CSVs (from GitHub mirrors)
train_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
test_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic_test.csv"

train = pd.read_csv(train_url)
test = pd.read_csv(test_url)

train.to_csv("data/train.csv", index=False)
test.to_csv("data/test.csv", index=False)

print("Downloaded train.csv and test.csv to data/")