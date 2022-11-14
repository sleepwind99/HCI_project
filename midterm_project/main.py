import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import pandas as pd
from estimate import estimate

# ... and any Python standard libraries.

if __name__ == "__main__":
    # Create .csv file that contains the estimation result of all five cases
    for i in range(5):
        estimate(i)
    pass
