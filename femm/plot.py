import pandas as pd
import glob
from matplotlib import pyplot as plt


# Pattern for the file names
file_pattern = "plunger*.csv"

# Using glob to find all files matching the pattern
files = glob.glob(file_pattern)

# Dictionary to hold the data from each file
dfs = {}

for file_name in files:
    # Read each file into a DataFrame
    df = pd.read_csv(file_name, header=None, names=["Distance", file_name])

    # Set the first column as the index
    df.set_index("Distance", inplace=True)

    # Add the 'Force' column to the dictionary with the file name as the key
    dfs[file_name] = df[file_name]

# Combine all DataFrames into one, with each column named after the corresponding file
df = pd.DataFrame(dfs)

# Combine all DataFrames into one, with each column named after the corresponding file
# df = pd.concat(dfs)
print(df)
df.plot()
plt.show()
