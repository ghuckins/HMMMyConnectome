import numpy as np
import ssm
import os
import pandas as pd
import random
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn import svm
import math
import matplotlib.pyplot as plt
import pickle

root = "/Users/gracehuckins/Documents/HMMMyConnectome"

def import_all(num_networks):
    path = os.path.join(root,"results",f"data{num_networks}")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    data = []
    for file_name in os.listdir(path):
        data.append(np.loadtxt(os.path.join(path,file_name)))
    return data

def import_tuesthurs(num_networks):
    path = os.path.join(root,"results",f"data{num_networks}")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    tues_data = []
    thurs_data = []
    for filename in os.listdir(path):
        if filename.endswith('t.txt'):
            tues_data.append(np.loadtxt(os.path.join(path,filename)))
        if filename.endswith('r.txt'):
            thurs_data.append(np.loadtxt(os.path.join(path,filename)))
    return tues_data, thurs_data

def import_raw(num_networks):
    print("importing")
    path = os.path.join(root,"data")
    savepath = os.path.join(root,"results",f"data{num_networks}")

    metadata = pd.read_table(os.path.join(path, "trackingdata.txt")) #download trackingdata_goodscans
    metadata = metadata.set_index("subcode")

    for filename in os.listdir(path):
        if filename.startswith("sub"):
            day = metadata.loc[filename.replace(".txt","")]["day_of_week"]
            raw_data = np.loadtxt(os.path.join(path,filename))
            raw_data = gsr(raw_data)
            activities = zscore(getNetworkActivity(raw_data, num_networks), axis=0)

            if day == "2":
                np.savetxt(os.path.join(savepath, filename.replace(".txt","t.txt")), activities)
            if day == "4":
                np.savetxt(os.path.join(savepath, filename.replace(".txt","r.txt")), activities)
            else:
                np.savetxt(os.path.join(savepath, filename), activities)

#instead—numpy structured array to label columns
def getNetworkActivity(data, num_networks):
    parcellation = f"{num_networks}networks"
    parcels = pd.read_table(os.path.join(root,"data/parcel_data.txt"))[parcellation]
    roughnetworks = pd.unique(parcels)
    activities = []
    #assert that length of parcels and #features in data are the same
    for network in roughnetworks:
        if network.lower().startswith(parcellation):
            netactivity = np.average(
                data[:, (parcels == network)], axis=1
            ).reshape((-1, 1))
            activities.append(netactivity)
    return np.concatenate(activities, axis=1)

def gsr(data):
    gsignal = np.average(data, axis=1)
    gsignal = np.reshape(gsignal, (-1, 1))
    ginverse = np.linalg.inv(gsignal.T @ gsignal) @ gsignal.T
    return data - ginverse @ data

def get_transmats(data,latdim):
    obsdim = np.shape(data[0])[1]
    trans = []

    if not os.path.exists(os.path.join(root,"results",f"model{obsdim}",f"initialstate{latdim}")):
        get_model_params(obsdim,latdim)

    file = open(os.path.join(root,"results",f"model{obsdim}",f"observations{latdim}"), "rb")
    observations = pickle.load(file)
    file.close()
    file = open(os.path.join(root,"results",f"model{obsdim}",f"initialstate{latdim}"), "rb")
    initialstate = pickle.load(file)
    file.close()

    for item in data:
        hmm = ssm.HMM(latdim, obsdim, observations = 'diagonal_gaussian')
        hmm.observations = observations
        hmm.init_state_distn = initialstate
        hmm.fit(item, method="em", num_em_iters = 100, transonly=True, initialize=False)
        trans.append(hmm.transitions.transition_matrix.reshape(-1))

    return(trans)

def get_model_params(obsdim,latdim):
    '''
    calculates and saves the fit initial state and observation model for a given hmm on the full data

    Args:
        obsdim: the model output dimension—the number of networks being fit
        latdim: the model latent dimension

    Returns:
        nothing
    '''
    path = os.path.join(root,"results",f"model{obsdim}")
    if not os.path.exists(path):
        os.mkdir(path)
    model = ssm.HMM(latdim, obsdim, observations="diagonal_gaussian")
    data = import_all(obsdim)
    model.fit(data, method="em", num_em_iters=500, transonly=False, initialize=True)
    file = open(os.path.join(path,f"observations{latdim}"), "wb")
    pickle.dump(model.observations, file)
    file.close()
    file = open(os.path.join(path,f"initialstate{latdim}"), "wb")
    pickle.dump(model.init_state_distn, file)
    file.close()

def loocv(data1, data2, latdim1, latdim2):
    '''
    Performs LOO cross-validation by fitting separate HMMs to each dataset and classifying the left-out data
    based on which HMM assigns it a higher log likelihood

    Args:
        data1, data2: datasets to be classified
        latdim1: latent dimensionality for data1
        latdim2: latent dimensionality for data2

    Returns: Balanced accuracy of the classifier

    '''
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    assert np.shape(data2[0])[1] == obsdim, "Datasets must have the same number of features"
    length = min(len(data1), len(data2)) - 1
    correct1 = 0
    model2 = ssm.HMM(latdim2, obsdim, observations="diagonal_gaussian")
    model2.fit(data2[:length], method="em", num_em_iters=100)
    for i in range(len(data1)):
        model1 = ssm.HMM(latdim1, obsdim, observations="diagonal_gaussian")
        temp = data1.copy()
        temp.pop(i)
        random.shuffle(temp)
        model1.fit(temp[:length], method="em", num_em_iters=100)
        if model1.log_likelihood(data1[i]) > model2.log_likelihood(data1[i]):
            correct1 += 1
    correct2 = 0
    model1 = ssm.HMM(latdim1, obsdim, observations="diagonal_gaussian")
    model1.fit(data1[:length], method="em", num_em_iters=100)
    for i in range(len(data2)):
        model2 = ssm.HMM(latdim2, obsdim, observations="diagonal_gaussian")
        temp = data2.copy()
        temp.pop(i)
        random.shuffle(temp)
        model2.fit(temp[:length], method="em", num_em_iters=100)
        if model1.log_likelihood(data2[i]) < model2.log_likelihood(data2[i]):
            correct2 += 1
    return np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)])

def svmcv(data1, data2):

    zscored = zscore(np.concatenate((data1, data2)),axis=0)
    data1 = zscored[:len(data1)].tolist()
    data2 = zscored[len(data1):].tolist()

    length = min(len(data1), len(data2)) - 1
    y = np.concatenate((np.zeros(length), np.ones(length))).tolist()
    correct1 = 0
    correct2 = 0

    for i in range(len(data1)):
        temp1 = data1.copy()
        temp2 = data2.copy()
        temp1.pop(i)
        random.shuffle(temp1)
        random.shuffle(temp2)
        x = np.concatenate((temp1[:length], temp2[:length])).tolist()
        classifier = svm.SVC(kernel="linear")
        classifier.fit(x, y)
        if classifier.predict([data1[i]]) == 0:
            correct1 += 1

    for i in range(len(data2)):
        temp1 = data1.copy()
        temp2 = data2.copy()
        temp2.pop(i)
        random.shuffle(temp1)
        random.shuffle(temp2)
        x = np.concatenate([temp1[:length], temp2[:length]]).tolist()
        classifier = svm.SVC(kernel="linear")
        classifier.fit(x, y)
        if classifier.predict([data2[i]]) == 1:
            correct2 += 1

    return np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)])


def permtest(data1, data2, reps=50):
    length = math.floor(np.average([len(data1), len(data2)]))
    realacc = svmcv(data1,data2)
    permaccs = []

    for i in range(reps):
        alldatas = data1 + data2
        random.shuffle(alldatas)
        data1 = alldatas[:length]
        data2 = alldatas[length:2*length]
        acc = svmcv(data1,data2)
        permaccs.append(acc)

    plt.hist(permaccs)
    plt.axvline(x=realacc)
    plt.show()

def getstats(model, datas):
    '''
    takes a model and a dataset, returns avg occupancy time in each state across run, avg consecutive dwell time in each
    state and avg # of transitions in a run

    Args:
        model: hidden markov model trained to the data
        datas: data from which to extract statistics

    Returns:

    '''
    num_states = np.shape(hmm.observations.mus)[0]
    changes = []
    occs = []
    dwells = [[]]
    for i in range(num_states - 1):
        dwells.append([])
    for data in datas:
        state = model.most_likely_states(data)
        change = np.nonzero(np.diff(state))
        change = change[0]
        num_change = len(change)
        changes.append(num_change)
        occ = np.histogram(state, bins=num_states, range=(0, num_states))[0]
        occs.append(occ)
        for i in range(num_change):
            if i == 0:
                dwells[state[change[0]]].append(change[0])
            else:
                dwells[state[change[i]]].append(change[i] - change[i - 1])
    avgchanges = np.mean(changes)
    avgoccs = np.mean(occs, axis=0)
    avgdwells = []
    for item in dwells:
        avgdwells.append(np.mean(item))
    return avgoccs, avgdwells, avgchanges

def plot_hmm_ll(data, maxstates, folds):
    test_ll = np.zeros(maxstates)
    obsdim = np.shape(data[0])[1]
    hiddenstates = np.arange(1, maxstates + 1)
    kf = KFold(n_splits=folds)
    for train, test in kf.split(data):
        for num_states in hiddenstates:
            hmm = ssm.HMM(num_states, obsdim, observations="diagonal_gaussian")
            hmm.fit(data[train], method="em", num_em_iters=100)
            test_ll[num_states - 1] += hmm.log_likelihood(data[test])
    plt.plot(test_ll / folds)
    plt.show()

def plot_trans_matrix(mat1,mat2):
    np.fill_diagonal(mat1,0)
    np.fill_diagonal(mat2,0)
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(mat1)
    im = axs[1].imshow(mat2)
    axs[2].axis("off")
    plt.colorbar(im, ax=axs[2])
    plt.show()

def plot_emission_networks(mus1, mus2):
    num_states = np.shape(mus1)[0]
    obsdim = np.shape(mus1)[1]
    fig, axs = plt.subplots(2, num_states, subplot_kw=dict(projection="polar"))
    theta = np.arange(obsdim) / obsdim * 2 * math.pi
    i = 0
    while i < num_states:
        axs[0, i].set_ylim([-1.5, 1.5])
        axs[0, i].plot(theta, mus1[i, :])
        axs[0, i].set_xticklabels([])
        axs[0, i].set_yticklabels([])
        for t, r in zip(theta, mus1[0][i, :]):
            axs[0, i].annotate(str(round(t * 17 / (2 * math.pi) + 1)), xy=[t, r])
        axs[1, i].set_ylim([-1.5, 1.5])
        axs[1, i].plot(theta, mus2[i, :])
        axs[1, i].set_xticklabels([])
        axs[1, i].set_yticklabels([])
        for t, r in zip(theta, mus2[0][i, :]):
            axs[1, i].annotate(str(round(t * 17 / (2 * math.pi) + 1)), xy=[t, r])
        i += 1
    networkstring = (
        "1 - VisCent\n "
        "2 - VisPeri\n"
        "3 - SomMotA\n"
        "4 - SomMotB\n"
        "5 - DorsAttnA\n"
        "6 - DorsAttnB\n"
        "7 - SalVentAttnA\n"
        "8 - SalVentAttnB\n"
        "9 - LimbicB\n"
        "10 - LimbicA\n"
        "11 - ContA\n"
        "12 - ContB\n"
        "13 - ContC\n"
        "14 - DefaultA\n"
        "15 - DefaultB\n"
        "16 - DefaultC\n"
        "17 - TempPar"
    )
    plt.text(-100, 0, networkstring, fontsize=10)
    plt.show()


networks = 7
latdim = 14
tues, thurs = import_tuesthurs(network
tues_trans = get_transmats(tues,latdim)
thurs_trans = get_transmats(thurs,latdim)
permtest(tues_trans,thurs_trans)