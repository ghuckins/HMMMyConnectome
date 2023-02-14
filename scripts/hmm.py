import numpy as np
import ssm
import os
import pandas as pd
import random
from scipy.stats import zscore
from sklearn import svm
import math
import matplotlib.pyplot as plt
import pickle

def import_networks(num_networks,full_data = False):
    if full_data:
        data = []
        path = f"../results/fulldata{num_networks}"
        if os.path.exists(path):
            for file_name in os.listdir(path):
                data.append(np.loadtxt(f"{path}/{file_name}"))
            return data
        else:
            os.mkdir(path)
            return import_raw(num_networks,full_data)

    else:
        tues_data = []
        thurs_data = []
        tues_path = f"../results/tuesdata{num_networks}"
        thurs_path = f"../results/thursdata{num_networks}"
        if os.path.exists(tues_path):
            for file_name in os.listdir(tues_path):
                tues_data.append(np.loadtxt(f"{tues_path}/{file_name}"))
            for file_name in os.listdir(thurs_path):
                thurs_data.append(np.loadtxt(f"{thurs_path}/{file_name}"))
            return tues_data, thurs_data
        else:
            os.mkdir(tues_path)
            os.mkdir(thurs_path)
            return import_raw(num_networks, full_data)

def import_raw(num_networks,full_data = False):
    metadata = pd.read_table("../data/trackingdata.txt")

    if full_data:
        data = []
    else:
        tues_data = []
        thurs_data = []

    for filename in os.listdir("../data"):
        if filename.startswith("sub"):
            day = metadata.loc[metadata["subcode"] == filename.replace(".txt", ""), "day_of_week"]
            try:
                day = int(day)
            except:
                continue
            if not (full_data or day in [2, 4]):
                continue
            raw_data = np.loadtxt(f"../data/{filename}")
            activities = getNetworkActivity(raw_data, num_networks)
            activities = zscore(gsr(activities), axis=0)
            if full_data:
                data.append(activities)
                np.savetxt(f"../results/fulldata{num_networks}/{filename}", activities)
            else:
                if int(day) == 2:
                    tues_data.append(activities)
                    np.savetxt(f"../results/tuesdata{num_networks}/{filename}", activities)
                if int(day) == 4:
                    thurs_data.append(activities)
                    np.savetxt(f"../results/thursdata{num_networks}/{filename}", activities)

    if full_data:
        return data
    else:
        return tues_data, thurs_data

def getNetworkActivity(data, num_networks):
    parcellation = f"{num_networks}networks"
    parcels = pd.read_table("../data/parcel_data.txt")[parcellation]
    roughnetworks = pd.unique(parcels)
    activities = []
    for network in roughnetworks:
        if network.lower().startswith(parcellation):
            netactivity = np.average(
                data[:, (parcels == network)], axis=1
            ).reshape((-1, 1))
            activities.append(netactivity)
    return np.concatenate(activities, axis=1)

# global signal regression
def gsr(data):
    gsignal = np.average(data, axis=1)
    gsignal = np.reshape(gsignal, (-1, 1))
    ginverse = np.linalg.inv(gsignal.T @ gsignal) @ gsignal.T
    return data - ginverse @ data

def loocv(data1, data2, latdim1, latdim2):
    '''
    Performs LOO cross-validation by fitting separate HMMs to each dataset and classifying the left-out data
    based on which HMM assigns it a higher log likelihood

    Args:
        data1:
        data2:
        latdim1:
        latdim2:

    Returns: Balanced accuracy of the classifier

    '''
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    print(obsdim)
    print(np.shape(data2[0])[1])
    assert np.shape(data2[0])[1] == obsdim, "Datasets must have the same number of features"
    length = min(len(data1), len(data2))
    accs = []
    correct = 0
    model2 = ssm.HMM(latdim2, obsdim, observations="diagonal_gaussian")
    model2.fit(data2[: length - 1], method="em", num_em_iters=100)
    for test in data1:
        model1 = ssm.HMM(latdim1, obsdim, observations="diagonal_gaussian")
        temp = data1.copy()
        temp.remove([test])
        random.shuffle(temp)
        model1.fit(temp[:length-1], method="em", num_em_iters=100)
        if model1.log_likelihood(test) > model2.log_likelihood(test):
            correct += 1
    accs.append(correct/len(data1))
    correct = 0
    model1 = ssm.HMM(latdim1, obsdim, observations="diagonal_gaussian")
    model1.fit(data1[: length - 1], method="em", num_em_iters=100)
    for test in data2:
        model2 = ssm.HMM(latdim2, obsdim, observations="diagonal_gaussian")
        temp = data2.copy()
        temp.remove([test])
        random.shuffle(temp)
        model2.fit(temp[:length-1], method="em", num_em_iters=100)
        if model1.log_likelihood(test) < model2.log_likelihood(test):
            correct += 1
    accs.append(corrent / len(data2))
    return accs

tues_data,thurs_data = import_networks(17,full_data=False)
print(loocv(tues_data,thurs_data,6,6))











def findhiddenstates(data, obsdim, maxstates, cv):
    i = 0
    test_ll = np.zeros(maxstates)
    # bayesinfocrit = np.zeros(maxstates)
    hiddenstates = np.arange(1, maxstates + 1)
    while i < cv:
        random.shuffle(data)
        if cv == 1:
            train = data[: math.floor(len(data) * 0.75)]
            test = data[math.floor(len(data) * 0.75) :]
        else:
            train = data[: math.floor(len(data) * (cv - 1) / cv)]
            test = data[math.floor(len(data) * (cv - 1) / cv) :]
        for num_states in hiddenstates:
            hmm = ssm.HMM(num_states, obsdim, observations="diagonal_gaussian")
            hmm.fit(train, method="em", num_em_iters=100)
            test_ll[num_states - 1] += hmm.log_likelihood(test)
            # bayesinfocrit[num_states-1] += bic(hmm,test)
        i += 1
    plt.plot(test_ll / cv)
    # plt.plot(bayesinfocrit)
    plt.show()
    # return bayesinfocrit/cv
    return test_ll / cv


def svmcv(data1, data2, featurelim=10):
    length = min(len(data1), len(data2))
    # length = math.floor(length/2)
    d1 = np.zeros(length)
    d2 = np.ones(length)
    y = np.concatenate([d1, d2])
    y = y.tolist()
    d1train = data1[:length]  #:2*length]
    d2train = data2[:length]  #:2*length]
    # x = []
    x = np.concatenate([d1train, d2train])
    # trainedcoeffs = np.loadtxt('svm_coefficients.txt')
    # for train in d1train:
    #    corr = np.corrcoef(train.T)
    #    x.append(corr[np.triu_indices(17)])#[np.where(np.abs(trainedcoeffs)>=featurelim)])
    # for train in d2train:
    #    corr = np.corrcoef(train.T)
    #    x.append(corr[np.triu_indices(17)])#[np.where(np.abs(trainedcoeffs)>=featurelim)])
    x = zscore(x, axis=0).tolist()
    # print(len(x[0]))
    # if(len(x[0])) < 1:
    #    return
    d1tot = 0
    d2tot = 0
    counter = 0
    # coeffs = np.zeros((1, np.shape(x[0])[0]))
    for item in x:
        xcopy = x.copy()
        ycopy = y.copy()
        del xcopy[counter]
        del ycopy[counter]
        if counter < length:
            del xcopy[2 * length - counter - 2]
            del ycopy[2 * length - counter - 2]
        else:
            del xcopy[2 * length - counter - 1]
            del ycopy[2 * length - counter - 1]
        classifier = svm.SVC(kernel="linear")
        classifier.fit(xcopy, ycopy)
        # coeffs += classifier.coef_
        if counter < length:
            d1tot += 1 - classifier.predict([item])
        else:
            d2tot += classifier.predict([item])
        counter += 1
    # np.savetxt('svm_coefficients.txt',coeffs)
    return d1tot / length, d2tot / length  # , len(x[0])


#
def getstats(model, datas, num_states):
    '''
    takes a model and a dataset, returns avg occupancy time in each state across run, avg consecutive dwell time in each
    state and avg # of transitions in a run

    Args:
        model:
        datas:
        num_states:

    Returns:

    '''
    changes = []
    occs = []
    dwells = [[]]
    for i in range(num_states - 1):
        dwells.append([])
    for data in datas:
        state = model.most_likely_states(data)
        change = np.nonzero(np.diff(state))
        change = change[0]
        nochange = len(change)
        changes.append(nochange)
        occ = np.histogram(state, bins=num_states, range=(0, num_states))[0]
        occs.append(occ)
        for i in range(nochange):
            if i == 0:
                dwells[state[change[0]]].append(change[0])
            else:
                dwells[state[change[i]]].append(change[i] - change[i - 1])
    avgchanges = np.mean(changes)
    avgoccs = np.mean(occs, axis=0)
    avgdwells = []
    for item in dwells:
        avgdwells.append(np.mean(item))
    return (avgoccs, avgdwells, avgchanges)


def bic(model, data):
    outdim = model.D
    latdim = model.K
    # for diagonal gaussian
    freeparams = latdim**2 + 2 * latdim * outdim
    ll = model.log_likelihood(data)
    return math.log(np.shape(data[0])[0] * len(data)) * freeparams - 2 * ll


def utnodiag(dim):
    r = np.arange(dim)
    return r[:, None] < r

def permtest(data1, data2, reps):
    length = min(len(data1), len(data2))
    data1 = data1[:length]
    data2 = data2[:length]
    realacc = np.average(svmcv(data1,data2))
    print(realacc)
    permaccs = []

    for i in range(reps):
        alldatas = data1 + data2
        random.shuffle(alldatas)
        data1 = alldatas[:length]
        data2 = alldatas[length:]
        acc = svmcv(data1,data2)
        print(acc)
        permaccs.append(np.average(acc))

    plt.hist(permaccs)
    plt.axvline(x=realacc)
    plt.show()




parcels = "17networks"
metadata = pd.read_table("../data/trackingdata.txt")

# import all myconnectome rsfmri runs
tuesdata = []
thursdata = []
fulldata = []
directory = "../data"
for filename in os.listdir(directory):
    if filename.startswith("sub"):
        day = metadata.loc[
            metadata["subcode"] == filename.replace(".txt", ""), "day_of_week"
        ]
        data = np.loadtxt(directory + "/" + filename)
        __, activities = getNetworkActivity(data, parcels)
        activities = gsr(activities)
        activities = zscore(activities, axis=0)
        try:
            int(day)
        except:
            pass
        else:
            if int(day) == 2:
                tuesdata.append(activities)
            if int(day) == 4:
                thursdata.append(activities)
        fulldata.append(activities)
        # if counter == 50:
        # break

fcacc = 0.712
states = range(3,10)

# svm classification with transition matrices

allacc = []

reps = 5

alltuesoccs = []
allthursoccs = []

for i in range(reps):
    random.shuffle(tuesdata)
    random.shuffle(thursdata)
    accs = []
    # tuesdata = tuesdata[math.floor(len(tuesdata) / 2):]
    # thursdata = thursdata[math.floor(len(thursdata) / 2):]

    for state in states:
        # get observations and initial states
        if not (os.path.exists("../results/observations" + str(state))):
            hmm = ssm.HMM(state, 17, observations="diagonal_gaussian")
            hmm.fit(
                fulldata,
                method="em",
                num_em_iters=500,
                transonly=False,
                initialize=True,
            )
            file = open("../results/observations" + str(state), "wb")
            pickle.dump(hmm.observations, file)
            file.close()
            file = open("../results/initialstate" + str(state), "wb")
            pickle.dump(hmm.init_state_distn, file)
            file.close()

        # get transition matrices for each run

        tuestrans = []
        thurstrans = []
        tueshmms = []
        thurshmms = []
        tuesoccs = []
        thursoccs = []

        file = open("../results/observations" + str(state), "rb")
        observations = pickle.load(file)
        file.close()
        file = open("../results/initialstate" + str(state), "rb")
        initialstate = pickle.load(file)
        file.close()

        for data in tuesdata:
            hmm = ssm.HMM(state, 17, observations="diagonal_gaussian")
            hmm.observations = observations
            hmm.init_state_distn = initialstate
            hmm.fit(
                data, method="em", num_em_iters=100, transonly=True, initialize=False
            )
            tuestrans.append(
                hmm.transitions.transition_matrix[np.triu_indices(state)])#[utnodiag(state)])
            occs, dwells, changes = getstats(hmm, [data], state)
            tuesoccs.append(occs)
            tueshmms.append(hmm)

        for data in thursdata:
            hmm = ssm.HMM(state, 17, observations="diagonal_gaussian")
            hmm.observations = observations
            hmm.init_state_distn = initialstate
            hmm.fit(
                data, method="em", num_em_iters=100, transonly=True, initialize=False
            )
            thurstrans.append(
                hmm.transitions.transition_matrix[np.triu_indices(state)])#[utnodiag(state)])
            occs, dwells, changes = getstats(hmm, [data], state)
            thursoccs.append(occs)
            thurshmms.append(hmm)
        #permtest(tuestrans,thurstrans,10)
        acc = svmcv(tuestrans,thurstrans)

        accs.append(np.average(acc))
    allacc.append(accs)

plt.plot(states,np.average(accs))
plt.show()
quit()
# plot transition matrices

meantues = np.mean(tuestrans, axis=0)
meanthurs = np.mean(thurstrans, axis=0)
np.fill_diagonal(meantues, 0)
np.fill_diagonal(meanthurs, 0)
fig, axs = plt.subplots(1, 3)
axs[0].imshow(meantues)
im = axs[1].imshow(meanthurs)
axs[2].axis("off")
plt.colorbar(im, ax=axs[2])
plt.show()

fig, axs = plt.subplots(2, 4)
np.fill_diagonal(tuestrans[0], 0)
np.fill_diagonal(tuestrans[1], 0)
np.fill_diagonal(tuestrans[2], 0)
np.fill_diagonal(thurstrans[0], 0)
np.fill_diagonal(thurstrans[1], 0)
np.fill_diagonal(thurstrans[2], 0)

axs[0, 0].imshow(tuestrans[0])
axs[0, 1].imshow(tuestrans[1])
axs[0, 2].imshow(tuestrans[2])
axs[1, 0].imshow(thurstrans[0])
axs[1, 1].imshow(thurstrans[1])
im = axs[1, 2].imshow(thurstrans[2])
axs[0, 3].axis("off")
axs[1, 3].axis("off")
plt.colorbar(im, ax=axs[:, 3])
plt.show()

quit()


plt.plot(states, np.mean(allacc, axis=0), label="transition matrix")
plt.plot(states, np.ones(np.shape(states)) * fcacc, label="functional connectivity")
plt.legend()
plt.xlabel("number of hidden states")
plt.ylabel("classification accuracy")
plt.show()
quit()

# occupancy states from full hmm

occaccs = []
for state in states:
    fullhmm = ssm.HMM(state, 17, observations="diagonal_gaussian")
    fullhmm.fit(fulldata, method="em", num_em_iters=100)

    tuesoccs = []
    tueschange = []
    tuesdwells = []
    thursoccs = []
    thurschange = []
    thursdwells = []
    for data in tuesdata:
        occs, dwells, changes = getstats(fullhmm, [data], state)
        tuesoccs.append(occs)
        tueschange.append(changes)
        tuesdwells.append(dwells)
    for data in thursdata:
        occs, dwells, changes = getstats(fullhmm, [data], state)
        thursoccs.append(occs)
        thurschange.append(changes)
        thursdwells.append(dwells)

    occaccs.append(np.mean(svmcv(tuesoccs, thursoccs)))


plt.subplot(121)
for item in tuesoccs:
    plt.plot(range(states[0]), item)
plt.subplot(122)
for item in thursoccs:
    plt.plot(range(states[0]), item)
plt.show()
# plt.errorbar(range(states[0]),np.mean(tuesoccs,axis=0),yerr = np.std(tuesoccs,axis=0), label ='Tuesday')
# plt.errorbar(range(states[0]),np.mean(thursoccs,axis=0),yerr = np.std(thursoccs,axis=0), label='Thursday')
quit()


plt.plot(
    states, np.average(np.array(allacc), axis=0), label="occupancy time classification"
)
plt.plot(
    states,
    np.ones(np.shape(states)) * fcacc,
    label="functional connectivity classification",
)
plt.legend()
plt.xlabel("number of hidden states")
plt.ylabel("classification accuracy")
plt.title("classification by occupancies")
plt.ylim([0.4, 1])
plt.show()

quit()


# plot transition matrices

meantues = np.mean(tuestrans, axis=0)
meanthurs = np.mean(thurstrans, axis=0)
np.fill_diagonal(meantues, 0)
np.fill_diagonal(meanthurs, 0)
fig, axs = plt.subplots(1, 3)
axs[0].imshow(meantues)
im = axs[1].imshow(meanthurs)
axs[2].axis("off")
plt.colorbar(im, ax=axs[2])
plt.show()

quit()

# svm: classification with only few most informative features

accs = []
features = []
for lim in range(30):
    a = svmcv(tuesdata, thursdata, lim)
    try:
        (accs.append(np.average(a[:2])))
    except:
        break
    features.append(a[2])
plt.plot(features, accs)
plt.show()

# testing state number for classification, same # in each model

max_states = 20
accs = []
tuesaccs = []
thursaccs = []
for i in range(max_states):
    acc = loocv(tuesdata, thursdata, i + 1, i + 1, 7)
    tuesaccs.append(acc[0])
    thursaccs.append(acc[1])
    accs.append(
        np.dot(acc, [len(tuesdata), len(thursdata)]) / (len(tuesdata) + len(thursdata))
    )

# testing lattice of different state combinations

max_states = 4
accuracies = np.zeros((max_states, max_states))
tuesaccs = np.zeros((max_states, max_states))
thursaccs = np.zeros((max_states, max_states))

for i in range(max_states):
    for j in range(max_states):
        accs = loocv(tuesdata, thursdata, i + 2, j + 2, 17)
        accuracies[i, j] = np.dot(accs, [len(tuesdata), len(thursdata)]) / (
            len(tuesdata) + len(thursdata)
        )
        tuesaccs[i, j] = accs[0]
        thursaccs[i, j] = accs[1]

fix, axs = plt.subplots(1, 4)
axs[0].imshow(accuracies)
axs[1].imshow(tuesaccs)
im = axs[2].imshow(thursaccs)
axs[3].axis("off")
plt.colorbar(im, ax=axs[3])

plt.show()


reps = 25
num_states = 6
tueshmms = []
thurshmms = []
for i in range(reps):
    random.shuffle(tuesdata)
    subsample = tuesdata[: int(len(tuesdata) / 2)]
    hmm = ssm.HMM(num_states, 17, observations="diagonal_gaussian")
    hmm.fit(subsample, method="em", num_em_iters=100)
    tueshmms.append(hmm)
    random.shuffle(thursdata)
    subsample = thursdata[: int(len(thursdata) / 2)]
    hmm = ssm.HMM(num_states, 17, observations="diagonal_gaussian")
    hmm.fit(subsample, method="em", num_em_iters=100)
    thurshmms.append(hmm)

tueslikelystates = tueshmms[0].most_likely_states(fulldata[0])
thurslikelystates = thurshmms[0].most_likely_states(fulldata[0])
thurshmms[0].permute(ssm.util.find_permutation(tueslikelystates, thurslikelystates))
thurslikelystates = thurshmms[0].most_likely_states(fulldata[0])
for i in range(1, reps):
    newlikelystates = tueshmms[i].most_likely_states(fulldata[0])
    tueshmms[i].permute(ssm.util.find_permutation(tueslikelystates, newlikelystates))
    newlikelystates = thurshmms[i].most_likely_states(fulldata[0])
    thurshmms[i].permute(ssm.util.find_permutation(thurslikelystates, newlikelystates))

# note—is this the way that i should be doing it, averaging over all models and all data? should I apply both models
# to the same data?
# plot the odds of occupancy in different states across timecourse

tuesstateodds = np.zeros((np.shape(fulldata[0])[0], num_states))
thursstateodds = np.zeros((np.shape(fulldata[0])[0], num_states))
countertues = 0
counterthurs = 0
for i in range(reps):
    for data in tuesdata:
        tuesstateodds += tueshmms[i].expected_states(tuesdata[0])[0]
        countertues += 1
    for data in thursdata:
        thursstateodds += thurshmms[i].expected_states(thursdata[0])[0]
        counterthurs += 1
tuesstateodds = tuesstateodds / countertues
thursstateodds = thursstateodds / counterthurs
num_bins = 1
bins = np.arange(num_bins) * np.shape(tuesstateodds)[0] / num_bins
binlabels = np.digitize(np.arange(np.shape(tuesstateodds)[0]), bins)
tuesbinneddata = np.zeros((num_bins, num_states))
thursbinneddata = np.zeros((num_bins, num_states))
for i in range(num_bins):
    tuesbinneddata[i, :] = np.dot((binlabels == i + 1).astype(int), tuesstateodds)
    thursbinneddata[i, :] = np.dot((binlabels == i + 1).astype(int), thursstateodds)
plt.subplot(121)
for i in range(num_states):
    plt.bar(
        range(num_bins),
        height=tuesbinneddata[:, i],
        bottom=np.sum(tuesbinneddata[:, :i], axis=1),
    )
plt.subplot(122)
for i in range(num_states):
    plt.bar(
        range(num_bins),
        height=thursbinneddata[:, i],
        bottom=np.sum(thursbinneddata[:, :i], axis=1),
    )
plt.show()

# calculating 95% confidence intervals for the transition probabilities

tuestransitions = []
thurstransitions = []
for i in range(reps):
    tuestransitions.append(tueshmms[i].transitions.transition_matrix)
    thurstransitions.append(thurshmms[i].transitions.transition_matrix)
alltues = np.stack(tuestransitions)
allthurs = np.stack(thurstransitions)

# making transition plot

meantues = np.mean(alltues, axis=0)
inttues = np.std(alltues, axis=0) * 2.064 / math.sqrt(reps)
meanthurs = np.mean(allthurs, axis=0)
intthurs = np.std(allthurs, axis=0) * 2.064 / math.sqrt(reps)
np.fill_diagonal(meantues, 0)
np.fill_diagonal(inttues, 0)
np.fill_diagonal(meanthurs, 0)
np.fill_diagonal(intthurs, 0)
fig, axs = plt.subplots(2, 4)
axs[0, 0].imshow(meantues - inttues)
axs[0, 1].imshow(meantues)
axs[0, 2].imshow(meantues + inttues)
axs[1, 0].imshow(meanthurs - intthurs)
axs[1, 1].imshow(meanthurs)
im = axs[1, 2].imshow(meanthurs + intthurs)
axs[0, 3].axis("off")
axs[1, 3].axis("off")
plt.colorbar(im, ax=axs[:, 3])
plt.show()

# making state plots

fig, axs = plt.subplots(2, num_states, subplot_kw=dict(projection="polar"))
tuesmus = []
thursmus = []
for i in range(reps):
    tuesmus.append(tueshmms[i].observations.mus)
    thursmus.append(thurshmms[i].observations.mus)
theta = np.arange(17) / 17 * 2 * math.pi
i = 0
while i < num_states:
    axs[0, i].set_ylim([-1.5, 1.5])
    for mu in tuesmus:
        axs[0, i].plot(theta, mu[i, :])
    axs[0, i].set_xticklabels([])
    axs[0, i].set_yticklabels([])
    for t, r in zip(theta, tuesmus[0][i, :]):
        axs[0, i].annotate(str(round(t * 17 / (2 * math.pi) + 1)), xy=[t, r])
    i += 1
i = 0
while i < num_states:
    axs[1, i].set_ylim([-1.5, 1.5])
    for mu in thursmus:
        axs[1, i].plot(theta, mu[i, :])
    axs[1, i].set_xticklabels([])
    axs[1, i].set_yticklabels([])
    for t, r in zip(theta, thursmus[0][i, :]):
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
