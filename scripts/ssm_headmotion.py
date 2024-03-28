import ssm
import numpy as np
import random
import os
from hmm import import_raw
import math

root = "/Users/gracehuckins/PycharmProjects/HMMMyConnectome"

def split_data(num_networks):
    """
    Splits MyConnectome data into all continuous head motion-free sequences and saves those sequences

    Args:
        num_networks: int (7 or 17) indicating whether to use 7- or 17-network data

    Returns:
        None
    """
    path = os.path.join(root, "data", f"data{num_networks}")
    dest_path = os.path.join(root, "data", f"data{num_networks}_split")
    mask_path = os.path.join(root, "data", "tmasks")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    os.mkdir(dest_path)
    for filename in os.listdir(path):
        data = np.loadtxt(os.path.join(path, filename))
        mask = np.loadtxt(os.path.join(mask_path, filename[:6] + ".txt"))
        curr = mask[0]
        counter = 0
        start = 0
        for i in range(len(mask)):
            if mask[i] - curr == 1:
                start = i
                curr = mask[i]
            elif mask[i] - curr == -1:
                np.savetxt(
                    os.path.join(dest_path, filename[:-4] + f"_{counter}.txt"),
                    data[start:i, :],
                )
                curr = mask[i]
                counter += 1
        if mask[i] == 1:
            np.savetxt(
                os.path.join(dest_path, filename[:-4] + f"_{counter}.txt"),
                data[start:, :],
            )
    return None


def loocv_split(data1, data2, latdim):
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
    alldata = data1 + data2
    random.shuffle(alldata)
    data1 = alldata[:math.floor(len(alldata)/2)]
    data2 = alldata[math.floor(len(alldata)/2):]
    # finding total number of timepoint-to-timepoint transitions in each dataset
    tot_transitions1 = np.sum([(len(i) - 1) for i in data1])
    tot_transitions2 = np.sum([(len(i) - 1) for i in data2])
    transitions = min(tot_transitions1, tot_transitions2)
    correct1 = 0
    for i in range(len(data1)):
        model1 = ssm.HMM(latdim, obsdim, observations="ar", lags=1) #alt—"diagonal_gaussian"
        model2 = ssm.HMM(latdim, obsdim, observations="ar", lags=1)
        temp = data1.copy()
        temp.pop(i)
        random.shuffle(temp)
        random.shuffle(data2)
        curr_length = np.sum([(len(i) - 1) for i in temp])
        if curr_length > transitions:
            temp = relength(transitions, temp)
            temp_data2 = data2.copy()
        else:
            temp_data2 = relength(curr_length, data2)
        model1.fit(temp, method="em", num_em_iters=100, verbose=0)
        model2.fit(temp_data2, method="em", num_em_iters=100, verbose=0)
        if model1.log_likelihood(data1[i]) > model2.log_likelihood(data1[i]):
            correct1 += 1

    correct2 = 0
    for i in range(len(data2)):
        model1 = ssm.HMM(latdim, obsdim, observations="ar",lags=1)
        model2 = ssm.HMM(latdim, obsdim, observations="ar",lags=1)
        temp = data2.copy()
        temp.pop(i)
        random.shuffle(temp)
        random.shuffle(data1)
        curr_length = np.sum([(len(i) - 1) for i in temp])
        if curr_length > transitions:
            temp = relength(transitions, temp)
            temp_data1 = data1.copy()
        else:
            temp_data1 = relength(curr_length, data1)
        model1.fit(temp_data1, method="em", num_em_iters=100, verbose=0)
        model2.fit(temp, method="em", num_em_iters=100, verbose=0)
        if model2.log_likelihood(data2[i]) > model1.log_likelihood(data2[i]):
            correct2 += 1

    return np.average([correct1 / len(data1), correct2 / len(data2)])


def relength(length, data):
    """
    Subsamples a dataset to yield a new dataset with a desired total number of timepoint-to-timepoint transitions

    Args:
        length: the total number of transitions in the desired final dataset
        data: a list, the original dataset

    Returns:
        new_data: a list, the new dataset with a total number of transitions dictated by length
    """
    new_data = []
    curr_length = 0
    for item in data:
        curr_length += len(item) - 1
        if curr_length > length:
            new_data.append(item[curr_length-length:])
            return new_data
        else:
            new_data.append(item)
    return new_data


def load_hm_data(num_networks):
    """
    Loads both the head motion-free split data and a head motion-containing version of the data
    with the same statistical characteristics

    Args:
        num_networks: int (7 or 17) indicating whether to use 7- or 17-network data

    Returns:
        hmdata: a dict of the split data with head motion-containing sequences included
        nohmdata: a dict of the head motion-free split data
    """
    hmdata = {}
    hmdata["t"] = []
    hmdata["r"] = []
    nohmdata = {}
    nohmdata["t"] = []
    nohmdata["r"] = []

    hmpath = os.path.join(root, "data", f"data{num_networks}")
    nohmpath = os.path.join(root, "data", f"data{num_networks}_split")

    for filename in os.listdir(hmpath):
        if filename[6] != "t" and filename[6] != "r":
            continue

        data = np.loadtxt(os.path.join(hmpath, filename))

        num = 0
        hm_position = 0

        while os.path.exists(os.path.join(nohmpath, filename[:7] + f"_{num}.txt")):
            nohm_toadd = np.loadtxt(os.path.join(nohmpath, filename[:7] + f"_{num}.txt"))
            nohmdata[filename[6]].append(nohm_toadd)
            hmdata[filename[6]].append(data[hm_position:hm_position + len(nohm_toadd)])
            hm_position += len(nohm_toadd)
            num += 1

    return hmdata, nohmdata

def to_edit():

    path = os.path.join(root, "results", "fits", "hmm7_split")
    iters = 5
    states = [2, 3, 4, 5]
    for iter in range(iters):
        for state in states:
            hmacc = np.array(loocv_split(hmdata["t"], hmdata["r"], state))
            hmacc = np.reshape(hmacc, (1, 1))
            with open(os.path.join(path, f"fit{state}_motion_ar"), "ab") as file:
                np.savetxt(file, hmacc)
            nohmacc = np.array(loocv_split(nohmdata["t"], nohmdata["r"], state))
            nohmacc = np.reshape(nohmacc, (1, 1))
            with open(os.path.join(path, f"fit{state}_nomotion_ar"), "ab") as file:
                np.savetxt(file, nohmacc)


def main():
    split_data(17)
    quit()
    path = os.path.join(root, "results", "fits", "Head Motion")
    hmdata, nohmdata = load_hm_data(7)
    #state =
    reps = 10
    for rep in range(reps):
        hmacc = [loocv_split(hmdata["t"], hmdata["r"], state)]
        nohmacc = [loocv_split(nohmdata["t"], nohmdata["r"], state)]
        with open(os.path.join(path, f"fit{state}_motion_ar"), "ab") as file:
            np.savetxt(file, hmacc)
        with open(os.path.join(path, f"fit{state}_nomotion_ar"), "ab") as file:
            np.savetxt(file, nohmacc)

if __name__ == "__main__":
    main()


