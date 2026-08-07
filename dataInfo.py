import os

import pandas as pd

file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    #"CEAS_08.csv"      # replace with your actual filename
    #"Nazario.csv"
    #"Nigerian_Fraud.csv"
    "Enron.csv"
)

data = pd.read_csv(file_path)

print(data["label"].value_counts())
print(data[["subject", "body", "label"]].head())

# print(data.columns)
# print(data.head())
# print(data.info())
# print(data["Category"].value_counts())
# print(data.duplicated().sum())

# total_rows = len(data)
# duplicate_rows = data.duplicated().sum()
# unique_rows = total_rows - duplicate_rows

# summary = pd.DataFrame({
#     "Category": ["Unique Rows", "Duplicate Rows"],
#     "Count": [unique_rows, duplicate_rows],
#     "Percentage (%)": [
#         unique_rows / total_rows * 100,
#         duplicate_rows / total_rows * 100
#     ]
# })

# summary["Percentage (%)"] = summary["Percentage (%)"].round(2)

# print(summary)
