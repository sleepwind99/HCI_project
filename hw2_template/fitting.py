"""
Humand and Computer Interfaces (2022 Fall)

Homework 2 Template code
Follow the homework specifications carefully.

DUE DATE: THE END OF 13 Nov, 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import pandas as pd

from drift_diffusion_simul import drift_diffusion_simulation

### You can define your own functions or constants here ... if you want.

tcue = []
tzone = []
p = []

def model(x, c1, c2, c3, c4):
    sigma1 = c2 * p
    sigma2 = c3 + (1 / (np.exp(c4 * tcue) - 1))
    mu = c1 * tzone
    sigma = ((sigma1 ** 2) * (sigma2 ** 2))/((sigma1 ** 2) + (sigma2 ** 2))
    result = (1 / (np.sqrt(2 * np.pi * sigma))) * np.exp(-(x - mu) ** 2 / (2 * sigma))
    return np.nan_to_num(result, np.float16)


# Problem 1
def mta_fitting(input_tech:bool, data_file_path = "../2018147558.csv"):
    """
    Moving-target acquisition fitting function

    data_file_path := string of path to the target data file.
    Refer to Homework 1 spec or given sample data for data format.

    input_tech := bool flag to select data of which input technique.
    If True, use button pressed case.
    If False, use button released case.
    """
    global tcue, tzone, p
    if(input_tech) : onkey = 1
    else : onkey = 0
    data = pd.read_csv(data_file_path)
    for i in range(1, 2):
        cond = data['cond'] == i
        key = data['key'] == onkey 
        test_data = data[cond & key].set_index('timestamp').reset_index()
        tcue = test_data['t_cue'].to_numpy()
        tzone = test_data['t_zone'].to_numpy()
        p = test_data['p'].to_numpy()
        y = test_data['success'].to_numpy()
        x = test_data['timestamp'].to_numpy()
        popt, _ = scipy.optimize.curve_fit(model, x, y, bounds=([-0.5, 0, 0, 150], [0.5, 0.5, 0.1, 350]))
        plt.plot(x, y, label="origianl")
        plt.plot(x, model(x, *popt), label="1")
        plt.show()


    y = data['success']

    # Complete this function ...

    ### The name of variables are free to set
    ### But the order of return shouldn't be revised.
    # return c1, c2, c3, c4, R2


# Problem 2
def fl_fitting(data_file_path="./dataset/pointing_task.csv"):
    """
    Fitts' law fitting function

    It is not necessary to change data_file_path.
    """

    # Complete this function ...

    ### The name of variables are free to set
    ### But the order of return shouldn't be revised.
    # return a, b, R2


# Problem 3
# Choose your answer by setting this variable
given_instruction_is = None     # 1 OR 2



# Problem 4
def dd_fitting(data_file_path="./dataset/choice_reaction.csv"):
    """
    Drift Diffusion model fitting fuction

    It is not necessary to change data_file_path.
    """

    # Complete this function ...

    ### The name of variables are free to set
    ### But the order of return shouldn't be revised.
    # return a, mu, Ter, likelihood


if __name__ == "__main__":
    mta_fitting(True)