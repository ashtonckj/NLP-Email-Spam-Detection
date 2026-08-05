import pandas as pd
import os

file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "Nazario.csv"
)

data = pd.read_csv(file_path)

# Fill missing values
data["subject"] = data["subject"].fillna("")
data["body"] = data["body"].fillna("")

# Combine subject and body
data["Message"] = data["subject"] + " " + data["body"]

# Convert label to your dataset format
data["Category"] = "spam"

# Keep only required columns
data = data[["Category", "Message"]]

output_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "Nazario_converted.csv"
)

data.to_csv(output_path, index=False)