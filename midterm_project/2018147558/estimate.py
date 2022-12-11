import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
import pandas as pd
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm


# ... and any Python standard libraries.

epsilon = np.finfo("float").eps


def gaze_dist(i, t, sigma):
    i = np.array(i).reshape(2, 1)
    t = np.array(t).reshape(2, 1)
    sigmaMtx = np.identity(2) * np.power(sigma, 2)
    temp = (1 / (2 * np.pi * np.sqrt(np.power(sigma, 4)))) * np.exp(
        -0.5 * np.dot(np.dot((i - t).T, np.linalg.inv(sigmaMtx)), (i - t))
    )
    temp = temp[0][0]
    return temp


def isVaild(x, y):
    if -615 <= x and x <= 615:
        if -365 <= y and y <= -182.5:
            return False
    return True


def getRotation(origin, target, beta):
    origin, target = origin.reshape(3), target.reshape(3)
    cross = np.cross(origin, target)
    theta = np.arccos(np.dot(origin, target) / (norm(origin) * norm(target))) / beta
    rttObj = R.from_rotvec(cross * theta / norm(cross))
    return rttObj


def estimateModel(params, data_trial, data_hp):
    # params[0:3] : s
    # params[3] : beta
    # params[4] : sigma0
    if not isVaild(params[0], params[1]):
        return np.inf

    Evec = []
    interpos = []
    for rx, ry, rz in data_trial[["pose_Rx", "pose_Ry", "pose_Rz"]].to_numpy():
        Evec.append(R.from_euler("zyx", [rz, ry, rx]).apply(np.array([0, 0, -1])))
    Evec = np.array(Evec)
    Ovec = -data_hp.T
    s = np.array(params[0:3]).reshape(3, 1)
    transpos = R.align_vectors(np.array([0, 0, -1]).reshape(1, 3), (s - data_hp.T).reshape(1, 3))[0]
    for i in range(Evec.shape[0]):
        Gvec = getRotation(Ovec, Evec[i], params[3]).apply(Ovec.reshape(3))
        tempInter = planeInter(data_hp.T, Gvec, s, data_hp.T - s).reshape(3)
        interpos.append(
            [
                -transpos.apply(tempInter)[0] + (615 / 2),
                transpos.apply(tempInter)[1] + 365,
            ]
        )

    r = np.sqrt(np.sum(np.power(data_hp.T - s, 2)))
    t = data_trial[["mouseX", "mouseY"]].to_numpy()
    mmt = np.array([(t[:, 0] - 1920) * (615 / 1920), t[:, 1] * (365 / 1080)]).T
    likelihood = 0
    for i in range(len(interpos)):
        temp = gaze_dist(interpos[i], mmt[i], r * params[4])
        if temp == 0:
            likelihood += np.log(epsilon)
        else:
            likelihood += np.log(temp)
    return -likelihood


def estimate(case_number):

    # import pandas as pd
    data_trial = pd.read_csv(f"dataset/case{case_number}.csv")
    data_hp = pd.read_csv(f"dataset/case{case_number}_head_position.csv", header=None).to_numpy()

    res = scipy.optimize.dual_annealing(
        estimateModel,
        bounds=[(-800, 800), (-800, -182.5), (-300, 300), (0.125, 1), (epsilon, 2)],
        args=(data_trial[data_trial["isPrimary"] == 1], data_hp),
    )
    return np.array(res["x"])


# intersection function
def planeInter(p0, gaze, center, normal):
    line_dot = np.dot(normal.T, gaze)
    if abs(line_dot) > epsilon:
        dirvec = p0 - center
        factor = -np.dot(normal.T, dirvec) / line_dot
        return dirvec + factor * gaze.reshape(3, 1) + center
    else:
        return None
