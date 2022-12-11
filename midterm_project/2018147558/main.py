import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import pandas as pd
from estimate import estimate
import warnings

# ... and any Python standard libraries.

if __name__ == "__main__":
    # Create .csv file that contains the estimation result of all five cases
    warnings.filterwarnings("ignore")
    result = pd.DataFrame({"case": [], "x": [], "y": [], "z": [], "beta": [], "sigma": []})
    for i in range(5):
        x, y, z, beta, sigma0 = estimate(i + 1)
        result.loc[i + 1] = [i + 1, x, y, z, beta, sigma0]
    result.to_csv("./estimation_result.csv")
