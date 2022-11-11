import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import pandas as pd
# ... and any Python standard libraries.

def estimate(case_number):
    # This will automatically load .csv file in dataset folder
    # Here is an example code

    # import pandas as pd
    data_trial = pd.read_csv(f"dataset/case{case_number}.csv")
    data_hp = pd.read_csv(f"dataset/case{case_number}_head_position.csv", header=None)

    # return (estimated secondary display position)
    # Note that return value should be iteratable
    # so that it can be converted into numpy.ndarray
    pass

if __name__ == "__main__":
    pass