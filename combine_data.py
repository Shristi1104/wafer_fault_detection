import pandas as pd

data = pd.read_csv("secom.data", delim_whitespace=True, header=None)
labels = pd.read_csv("secom_labels.data", delim_whitespace=True, header=None)

labels = labels.iloc[:, 0]

data.columns = [f"sensor_{i}" for i in range(data.shape[1])]

data["Pass/Fail"] = labels

data.to_csv("wafer_data.csv", index=False)

print("Done")