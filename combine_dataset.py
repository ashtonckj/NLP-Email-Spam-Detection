import pandas as pd
import os

ceas_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "clean_CEAS_08.csv"     
)

nazario_file_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "clean_Enron.csv"
)

ceas_data = pd.read_csv(ceas_file_path)
nazario_data = pd.read_csv(nazario_file_path)

combined = pd.concat([ceas_data, nazario_data], ignore_index=True)

output_path = os.path.join(
    os.path.dirname(__file__),
    "data",
    "Ceas08_Enron.csv"
)

combined.to_csv(output_path, index=False)
print("Done!")