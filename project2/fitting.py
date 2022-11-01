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


def model(x, c1, c2, c3, c4):
    x = np.array(x) / 1000
    sigma1 = c2 * x[:,3]
    sigma2 = c3 + (1 / (np.exp(c4 * x[:,1]) - 1))
    mu = c1 * x[:,2]
    sigma = (np.power(sigma1, 2) * np.power(sigma2, 2))/(np.power(sigma1, 2) + np.power(sigma2, 2))
    for i in range(0,8):
        if np.isnan(sigma[i]): sigma[i] = sigma1[i] ** 2
    result = (1 / np.sqrt(2 * np.pi * sigma)) * np.exp(-np.power(x[:,0] - mu, 2) / (2 * sigma))
    return 1 - result

# Problem 1
def mta_fitting(input_tech:bool, data_file_path = "../2018147558.csv"):
    x = []
    y = []
    if(input_tech) : onkey = 1
    else : onkey = 0
    data = pd.read_csv(data_file_path)
    key = data['key'] == onkey
    success = data['success'] == 1
    
    for i in range(1, 9):
        cond = data['cond'] == i
        y.append(1 - data[cond & key]['success'].mean())
        x.append(data[cond & key & success].loc[:,['timestamp','t_cue','t_zone','p']].mean().to_numpy())
    
    popt, _ = scipy.optimize.curve_fit(model, x, np.array(y), bounds=([-0.5, 0, 0, 150], [0.5, 0.5, 0.1, 350]))
    print(model(x,*popt))
    print(y)
    print(popt)
    plt.figure(figsize=(600/96, 600/96), dpi=96)
    plt.plot(y, model(x, *popt), 'ro')
    plt.plot([0, 0.5, 1], [0, 0.5, 1], label='x=y')
    if input_tech : plt.savefig('./scatterplot/mta_pressed.png', dpi = 96)
    else : plt.savefig('./scatterplot/mta_released.png', dpi = 96)
    

def p2model(x, a, b):
    return a + b * np.log(x[0]/x[1] + 1)

# Problem 2
def fl_fitting(data_file_path="./dataset/pointing_task.csv"):
    x = []
    y = []
    data = pd.read_csv(data_file_path)
    cond = data['success'] == 1
    x.append(data[cond].loc[:,['width', 'distance']].to_numpy())
    y.append(data[cond])
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