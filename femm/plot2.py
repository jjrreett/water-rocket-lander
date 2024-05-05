# Re-running the code with pandas import and correction

import pandas as pd
import re
from matplotlib import pyplot as plt

# Raw data provided as a string
input_data = """
--> param_num_turns=26.10341662770836
--> debug using colil current 10
--> -0.5,-0.6367504859290586
--> -0.46875,-0.8335345721007491
--> -0.4375,-1.315662084068629
--> -0.40625,-1.147595766300942
--> -0.375,-1.317672565693375
--> -0.34375,-1.130558295325624
--> -0.3125,-1.145986249781705
--> -0.28125,-1.152641957603465
--> -0.25,-0.9429757186953858
--> -0.21875,-0.7273220148598374
--> -0.1875,0.01550681027470924
--> -0.15625,1.185770508760061
--> -0.125,1.80005250765961
--> -0.09375,2.515347273123685
--> -0.0625,3.340207758214667
--> -0.03125,3.543660029655423
--> 0,4.279377638975442
--> 0.03125,3.976225955869923
--> 0.0625,3.277035744655791
--> 0.09375,2.38029888483318
--> 0.125,1.583734687752017
--> 0.15625,0.3680754183675345
--> 0.1875,0.1662114893537872
--> 0.21875,-0.7056049903427969
--> 0.25,-0.8999714426652551
--> 0.28125,-1.107626762926091
--> 0.3125,-0.7096272104276885
--> 0.34375,-0.8022352266802154
--> 0.375,-0.8256155532352532
--> 0.40625,-0.7299268885698234
--> 0.4375,-0.7295588541422453
--> 0.46875,-0.5813603830260665
--> 0.5,-0.5706571601157623
"""

# Parse parameters and data from the input
params = {}
data_points = []

for line in input_data.strip().split("\n"):
    if "param_" in line:
        # Extracting parameter
        param_match = re.search(r"param_([^=]+)=([\d.]+)", line)
        if param_match:
            params[param_match.group(1)] = float(param_match.group(2))
    elif "," in line:
        # Extracting data points
        data_match = re.search(r"(-?[\d.]+),(-?[\d.-]+)", line)
        if data_match:
            data_points.append((float(data_match.group(1)), float(data_match.group(2))))

# Creating a DataFrame from the data points
df = pd.DataFrame(data_points, columns=["Distance", "Force"])
print(df)
df.plot(x="Distance")
plt.show()
