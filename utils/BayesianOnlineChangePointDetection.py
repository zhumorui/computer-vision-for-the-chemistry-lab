import numpy as np
import matplotlib.pyplot as plt
import bocd

def change_point_detection(array,path):

    # normalize array
    # array = normalization(array)

    # Initialize object
    bc = bocd.BayesianOnlineChangePointDetection(bocd.ConstantHazard(300), bocd.StudentT(mu=0, kappa=1, alpha=1, beta=1))

    # Online estimation and get the maximum likelihood r_t at each time point
    rt_mle = np.empty(array.shape)
    for i, d in enumerate(array):
        bc.update(d)
        rt_mle[i] = bc.rt 


    # Plot data with estimated change points
    plt.plot(array, alpha=0.5, label="observation")
    index_changes = np.where(np.diff(rt_mle)<0)[0]
    plt.scatter(index_changes, array[index_changes], c='green', label="change point")
    plt.savefig(path)
    plt.close()
    

def normalization(array):
    _range = 100 - np.min(array)
    return (array - np.min(array)) / _range