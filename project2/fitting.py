import numpy as np
import matplotlib.pyplot as plt
import scipy
from sklearn.metrics import r2_score
import pandas as pd

from drift_diffusion_simul import drift_diffusion_simulation

# You can define your own functions or constants here ... if you want.


def p1model(x, c1, c2, c3, c4):
    x = np.array(x) / 1000
    sigma1 = c2 * x[:, 3]
    sigma2 = c3 + (1 / (np.exp(c4 * x[:, 1]) - 1))
    mu = c1 * x[:, 2]
    sigma = (np.power(sigma1, 2) * np.power(sigma2, 2)) / \
        (np.power(sigma1, 2) + np.power(sigma2, 2))
    for i in range(0, 8):
        if np.isnan(sigma[i]):
            sigma[i] = sigma1[i] ** 2
    result = 0.5 * \
        (scipy.special.erf((x[:, 0] - mu)/np.sqrt(2*sigma)
                           ) + scipy.special.erf(mu/np.sqrt(2*sigma)))
    return 1 - result

# Problem 1


def mta_fitting(input_tech: bool, data_file_path="./dataset/mta_sample_data.csv"):
    x = []
    y = []
    if (input_tech):
        onkey = 1
    else:
        onkey = 0
    data = pd.read_csv(data_file_path)
    key = data['key'] == onkey
    success = data['success'] == 1

    for i in range(1, 9):
        cond = data['cond'] == i
        y.append(1 - data[cond & key]['success'].mean())
        x.append(data[cond & key & success].loc[:, [
                 'timestamp', 't_cue', 't_zone', 'p']].mean().to_numpy())

    popt, _ = scipy.optimize.curve_fit(p1model, x, np.array(
        y), bounds=([-0.6, 0, 0, 0], [0.6, 0.5, 0.1, 500]))
    plt.figure(figsize=(600/96, 600/96), dpi=96)
    plt.plot(y, p1model(x, *popt), 'ro')
    plt.plot([0, 0.5, 1], [0, 0.5, 1], label='x=y')
    if input_tech:
        plt.savefig('./scatterplot/mta_pressed.png', dpi=96)
    else:
        plt.savefig('./scatterplot/mta_released.png', dpi=96)
    R2 = r2_score(np.array(y), p1model(x, *popt))
    c1, c2, c3, c4 = popt
    return c1, c2, c3, c4, R2


def p2model(x, a, b):
    x = np.array(x)
    return a + b * np.log(x[:, 1]/x[:, 0] + 1)

# Problem 2


def fl_fitting(data_file_path="./dataset/pointing_task.csv"):
    x = []
    y = []
    data = pd.read_csv(data_file_path)
    widths = [data['width'] == 50, data['width'] == 36.7, data['width'] == 15]
    dists = [data['distance'] == 400, data['distance'] == 800]
    cond = data['success'] == 1
    for i in widths:
        for j in dists:
            x.append(data[i & j & cond].loc[:, [
                     'width', 'distance']].mean().to_numpy())
            y.append(data[i & j & cond]['time'].mean())
    popt, _ = scipy.optimize.curve_fit(p2model, x, np.array(y), p0=[0, 1])
    plt.figure(figsize=(600/96, 600/96), dpi=96)
    plt.plot(y, p2model(x, *popt), 'ro')
    plt.plot([0.3, 0.5, 0.7], [0.3, 0.5, 0.7], label='x=y')
    plt.savefig('./scatterplot/f1.png', dpi=96)
    R2 = r2_score(p2model(x, *popt), np.array(y))
    a, b = popt
    return a, b, R2


# Problem 3
# Choose your answer by setting this variable
given_instruction_is = 2    # 1 OR 2


def ecdf(x):
    y = np.cumsum(np.ones(x.shape[0]))
    return np.sort(x), y/y.shape[0]


def logLikelihood(xdata, x, y):
    res = (np.interp(xdata + 0.01, x, y) - np.interp(xdata, x, y))/0.01
    res[res == 0] = 0.0000000000001
    return np.sum(np.log(res))


def p3model(x, xdata):
    temp = np.array(drift_diffusion_simulation(x[0], x[1], x[2])).T
    xs, ys = ecdf(temp[temp[:, 1] == 1][:, 0])
    xe, ye = ecdf(temp[temp[:, 1] == 0][:, 0])
    res1 = logLikelihood(xdata[xdata[:, 1] == 1][:, 0], xs, ys)
    res2 = logLikelihood(xdata[xdata[:, 1] == 0][:, 0], xe, ye)
    result = res1 + res2
    print(result)
    return -result


# Problem 4
def dd_fitting(data_file_path="./dataset/choice_reaction.csv"):
    data = pd.read_csv(data_file_path).sort_values(
        by=['reaction_time']).to_numpy()

    res = scipy.optimize.minimize(p3model, [0.074, 0.31, 0.28], args=(
        data), method='Nelder-Mead', bounds=[(0.045, 0.075), (0.25, 0.32), (0.25, 0.35)])
    print(res)
    a, mu, Ter = res['x']
    xdata = np.array(drift_diffusion_simulation(a, mu, Ter)).T
    plt.figure(figsize=(600/96, 600/96), dpi=96)
    plt.hist((data[data[:,1] == 1][:,0], xdata[xdata[:,1] == 1][:,0]), bins=30, histtype='step', density=True)
    plt.savefig('./scatterplot/dd_correct.png', dpi=96)
    plt.clf()
    plt.hist((data[data[:,1] == 0][:,0], xdata[xdata[:,1] == 0][:,0]), bins=30, histtype='step', density=True)
    plt.savefig('./scatterplot/dd_error.png', dpi=96)

    return a, mu, Ter, -res['fun']


if __name__ == "__main__":
    # mta_fitting(True)
    # fl_fitting()
    # modify()
    dd_fitting()
