import numpy as np
import os
import pandas as pd
import random
from scipy.stats import zscore
from sklearn.model_selection import KFold
from sklearn import svm
import math
import matplotlib.pyplot as plt

from dynamax.hidden_markov_model import DiagonalGaussianHMM
from dynamax.hidden_markov_model import LinearAutoregressiveHMM

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

def load_hcp(num_networks):
    dir = os.path.join(root,"results",f"hcpdata{num_networks}")
    if not os.path.exists(dir):
        import_raw_hcp(num_networks)
    data = {}
    for file in os.listdir(dir):
        file_key = file[3:9]
        if file_key in data.keys():
            data[file_key].append(np.loadtxt(os.path.join(dir,file)))
        else:
            data[file_key] = [np.loadtxt(os.path.join(dir,file))]
    return data

def getNetworkActivity(data, num_networks, hcp = False):
    if hcp:
        parcellation = f"Yeo_networks{num_networks}"
        path = os.path.join(root,"data/parcel_data_hcp.txt")
    else:
        parcellation = f"{num_networks}networks"
        path = os.path.join(root,"data/parcel_data.txt")
    parcels = pd.read_table(path)[parcellation]
    roughnetworks = pd.unique(parcels)
    if hcp:
        roughnetworks = np.delete(roughnetworks, np.where(roughnetworks=='No network found'))
    activities = []
    #assert that length of parcels and #features in data are the same
    for network in roughnetworks:
        if network.lower().startswith(parcellation) or hcp:
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

def get_transmats(data, latdim, together = False):

    if together:
        hmm, params, props = fit_model_transonly(data,latdim)
        trans = params.transitions.transition_matrix.reshape(-1)

    else:
        trans = []
        for item in data:
            hmm, params, props = fit_model_transonly(data,latdim)
            trans.append(params.transitions.transition_matrix.reshape(-1))

    return trans

def get_params_jax(obsdim, latdim, hcp = False):
    '''
    calculates and saves the fit initial state and observation model for a given hmm on the full data

    Args:
        obsdim: the model output dimension—the number of networks being fit
        latdim: the model latent dimension

    Returns:
        nothing
    '''
    if hcp:
        path = os.path.join(root,"results",f"hcpjaxmodel{obsdim}")
    else:
        path = os.path.join(root,"results",f"jaxmodel{obsdim}")
    if not os.path.exists(path):
        os.mkdir(path)
    hmm = DiagonalGaussianHMM(latdim, obsdim)
    if hcp:
        data = load_hcp(obsdim)
        data = [item for sublist in list(data.values()) for item in sublist]
    else:
        data = import_all(obsdim)
    params, props = hmm.initialize(method="kmeans", emissions=np.array(data))
    params, _ = hmm.fit_em(params, props, np.array(data), num_iters=1000)
    np.savetxt(os.path.join(path,f"probs{latdim}"), params.initial.probs)
    np.savetxt(os.path.join(path,f"means{latdim}"), params.emissions.means)
    np.savetxt(os.path.join(path,f"scale_diags{latdim}"), params.emissions.scale_diags)

def loocv(data1, data2, latdim):
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
    hmm1 = DiagonalGaussianHMM(latdim, obsdim)
    hmm2 = DiagonalGaussianHMM(latdim, obsdim)
    base_params1, props1 = hmm1.initialize(method="kmeans", emissions=np.array(data1[:length]))
    base_params2, props2 = hmm2.initialize(method="kmeans", emissions=np.array(data2[:length]))
    params2, _ = hmm2.fit_em(base_params2, props2, np.array(data2[:length]), num_iters=100, verbose=False)
    for i in range(len(data1)):
        temp = data1.copy()
        temp.pop(i)
        random.shuffle(temp)
        params1, _ = hmm1.fit_em(base_params1, props1, np.array(temp[:length]), num_iters=100, verbose=False)
        if hmm1.marginal_log_prob(params1, data1[i]) > hmm2.marginal_log_prob(params2, data1[i]):
            correct1 += 1
    correct2 = 0
    params1, _ = hmm1.fit_em(base_params1, props1, np.array(data1[:length]), num_iters=100, verbose=False)
    for i in range(len(data2)):
        temp = data2.copy()
        temp.pop(i)
        random.shuffle(temp)
        params2, _ = hmm2.fit_em(base_params2, props2, np.array(temp[:length]), num_iters=100, verbose=False)
        if hmm1.marginal_log_prob(params1, data2[i]) < hmm2.marginal_log_prob(params2, data2[i]):
            correct2 += 1
    print(correct1,correct2)
    return np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)])

def loocv_ar(data1, data2, latdim, lags):
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    assert np.shape(data2[0])[1] == obsdim, "Datasets must have the same number of features"
    length = min(len(data1), len(data2)) - 1
    correct1 = 0
    hmm = LinearAutoregressiveHMM(latdim, obsdim, num_lags=lags)
    base_params1, props1 = hmm.initialize(method="kmeans", emissions=np.array(data1[:length]))
    base_params2, props2 = hmm.initialize(method="kmeans", emissions=np.array(data2[:length]))
    inputs = []
    for datum in data2[:length]:
        inputs.append(hmm.compute_inputs(datum))
    params2, _ = hmm.fit_em(base_params2, props2, np.array(data2[:length]), inputs=np.array(inputs), num_iters=100, verbose = False)
    for i in range(len(data1)):
        temp = data1.copy()
        temp.pop(i)
        random.shuffle(temp)
        inputs = []
        for datum in temp[:length]:
            inputs.append(hmm.compute_inputs(datum))
        params1, _ = hmm.fit_em(base_params1, props1, np.array(temp[:length]), inputs=np.array(inputs), num_iters=100, verbose = False)
        inputs = hmm.compute_inputs(data1[i])
        if hmm.marginal_log_prob(params1, data1[i], inputs=inputs) > hmm.marginal_log_prob(params2, data1[i], inputs=inputs):
            correct1 += 1
    correct2 = 0
    inputs = []
    for datum in data1[:length]:
        inputs.append(hmm.compute_inputs(datum))
    params1, _ = hmm.fit_em(base_params1, props1, np.array(data1[:length]), inputs=np.array(inputs), num_iters=100, verbose = False)
    for i in range(len(data2)):
        temp = data2.copy()
        temp.pop(i)
        random.shuffle(temp)
        inputs = []
        for datum in temp[:length]:
            inputs.append(hmm.compute_inputs(datum))
        params2, _ = hmm.fit_em(base_params2, props2, np.array(temp[:length]), inputs=np.array(inputs), num_iters=100, verbose = False)
        inputs = hmm.compute_inputs(data2[i])
        if hmm.marginal_log_prob(params1, data2[i], inputs=inputs) < hmm.marginal_log_prob(params2, data2[i], inputs=inputs):
            correct2 += 1
    print(correct1,correct2)
    return np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)])

def loocvtrans(data1, data2, latdim):
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    assert np.shape(data2[0])[1] == obsdim, "Datasets must have the same number of features"
    length = min(len(data1), len(data2)) - 1
    correct1 = 0

    hmm2, params2, props2 = fit_model_transonly(data2[:length],latdim)
    for i in range(len(data1)):
        temp = data1.copy()
        temp.pop(i)
        random.shuffle(temp)
        hmm1, params1, props1 = fit_model_transonly(temp[:length],latdim)
        if hmm1.marginal_log_prob(params1, data1[i]) > hmm2.marginal_log_prob(params2, data1[i]):
            correct1 += 1
    correct2 = 0
    hmm1, params1, props1 = fit_model_transonly(data1[:length],latdim)
    for i in range(len(data2)):
        temp = data2.copy()
        temp.pop(i)
        random.shuffle(temp)
        hmm2, params2, props2 = fit_model_transonly(temp[:length],latdim)
        if hmm1.marginal_log_prob(params1, data2[i]) < hmm2.marginal_log_prob(params2, data2[i]):
            correct2 += 1
    print(correct1,correct2)
    return np.average([correct1/(correct1 + len(data2) - correct2), correct2/(correct2 + len(data1) - correct1)])

def fit_model_transonly(data, latdim, hcp = False):
    obsdim = np.shape(data[0])[1]

    if hcp:
        path = os.path.join(root,"results",f"hcpjaxmodel{obsdim}")
    else:
        path = os.path.join(root,"results",f"jaxmodel{obsdim}")

    if not os.path.exists(os.path.join(path,f"means{latdim}")):
        get_params_jax(obsdim, latdim, hcp)

    means = np.loadtxt(os.path.join(path,f"means{latdim}"))
    scale_diags = np.loadtxt(os.path.join(path,f"scale_diags{latdim}"))
    probs = np.loadtxt(os.path.join(path,f"probs{latdim}"))

    hmm = DiagonalGaussianHMM(latdim, obsdim)
    params, props = hmm.initialize(initial_probs=probs, emission_means=means, emission_scale_diags=scale_diags)

    props.emissions.means.trainable = False
    props.emissions.scale_diags.trainable = False
    props.initial.probs.trainable = False

    params, _ = hmm.fit_em(params, props, np.array(data), num_iters=100, verbose=False)

    return hmm, params, props

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

def permtest(data1, data2, class_func, latdim=6, reps=50):
    length = math.floor(np.average([len(data1), len(data2)]))
    if class_func == svmcv:
        realacc = svmcv(data1,data2)
    else:
        realacc = class_func(data1, data2, latdim)
    permaccs = []

    for i in range(reps):
        alldatas = data1 + data2
        random.shuffle(alldatas)
        data1 = alldatas[:length]
        data2 = alldatas[length:2*length]
        if class_func == svmcv:
            acc = svmcv(data1, data2)
        else:
            acc = class_func(data1, data2, latdim)

        permaccs.append(acc)

    plt.hist(permaccs,bins=50)
    plt.axvline(x=realacc)
    plt.show()

def getstats(model, params, datas):
    '''
    takes a model and a dataset, returns avg occupancy time in each state across run, avg consecutive dwell time in each
    state and avg # of transitions in a run

    Args:
        model: hidden markov model trained to the data
        datas: data from which to extract statistics

    Returns:

    '''
    num_states = np.shape(params.emissions.means)[0]
    changes = []
    occs = []
    dwells = [[]]
    for i in range(num_states - 1):
        dwells.append([])
    for data in datas:
        state = model.most_likely_states(params, data)
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
    test_ll = np.zeros(maxstates-1)
    print(len(test_ll))
    obsdim = np.shape(data[0])[1]
    hiddenstates = np.arange(2, maxstates + 1)
    print(len(hiddenstates))
    kf = KFold(n_splits=folds)
    for train, test in kf.split(data):
        for num_states in hiddenstates:
            print(num_states)
            #hmm = DiagonalGaussianHMM(num_states, obsdim)
            hmm = LinearAutoregressiveHMM(num_states, obsdim, num_lags=1)
            params, props = hmm.initialize(method="kmeans", emissions=np.array(data)[train])
            inputs = []
            for index in train:
                inputs.append(hmm.compute_inputs(np.array(data)[index]))
            params, _ = hmm.fit_em(params, props, np.array(data)[train], inputs=np.array(inputs), num_iters=100, verbose=False)
            for index in test:
                inputs = hmm.compute_inputs(np.array(data)[index])
                test_ll[num_states - 2] += hmm.marginal_log_prob(params, np.array(data)[index], inputs=inputs)
    plt.plot(hiddenstates, test_ll)
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

def main():
    tues, thurs = import_tuesthurs(7)
    normal = []
    ar = []
    hidden_states = range(3, 12)
    for state in hidden_states:
        normal.append(loocv(tues, thurs, state))
        ar.append(loocv_ar(tues, thurs, state, 1))
    plt.plot(hidden_states, normal)
    plt.plot(hidden_states, ar)
    plt.show()


if __name__ == "__main__":
    main()



