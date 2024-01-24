import numpy as np
import os
import pandas as pd
from hmm import gsr, loocv_batch, svmcv
from scipy.stats import zscore
import sys

root = "/Users/gracehuckins/Documents/HMMMyConnectome"

def import_all():
    path = os.path.join(root,"data","psychosis")
    for filename in os.listdir(path):
        if filename.endswith("timeseries.tsv"):
            if not os.path.exists(os.path.join(root,"results","psychosis",filename[:35]+"_999.txt")):
                if not os.path.exists(os.path.join(root,"results","psychosis",filename[:35]+"_888.txt")):
                    import_raw(filename)
def import_raw(filename):
    path = os.path.join(root,"data/psychosis")
    metadata = pd.read_table(os.path.join(path, "participants.tsv"))
    metadata = metadata.set_index("participant_id")
    savepath = os.path.join(root, "results", "psychosis")
    if not os.path.exists(savepath):
        os.mkdir(savepath)

    psychosis = metadata.loc[filename[:12]]["is_this_subject_a_patient"]
    data = pd.read_table(os.path.join(path,filename))
    data.dropna(axis=1, inplace=True)
    numpy_data = np.array(data)
    numpy_data = interpolate(numpy_data, pd.read_csv(os.path.join(path,"outliers",filename[:35]+"_outliers.tsv")).values)
    data = pd.DataFrame(gsr(numpy_data), columns=data.columns.str.split("_").str[2])

    networks = np.load(os.path.join(path,"networks.npy"), allow_pickle=True)

    activities = []
    for network in networks:
        network_data = np.nanmean(data.filter(regex=network, axis=1).values, axis=1).reshape((-1, 1))
        activities.append(network_data)
    activities = zscore(np.concatenate(activities, axis=1))
    if np.shape(activities)[0] == 415 and np.sum(np.isnan(activities)) == 0:
        np.savetxt(os.path.join(savepath, filename[:35] + f"_{int(psychosis)}.txt"), activities)

def interpolate(timeseries,outliers):
    loc = 0
    while loc < len(outliers):
        if outliers[loc] == 0:
            loc += 1
        else:
            start = loc
            while loc < len(outliers) and outliers[loc] == 1:
                loc += 1
            begin = max(start-1,0)
            end = min(start,len(timeseries)-1)
            to_interpolate = np.linspace(timeseries[begin,:],timeseries[end,:],loc-begin+1)
            timeseries = np.concatenate((timeseries[:begin,:],to_interpolate,timeseries[end+1:,:]))[:len(outliers),:]
    return timeseries

def load_unique():
    path = os.path.join(root,"results","psychosis")
    group_1 = []
    group_2 = []
    subjects = []
    for filename in os.listdir(path):
        if filename[:12] not in subjects:
            subjects.append(filename[:12])
            if filename[-5] == "9":
                group_1.append(np.loadtxt(os.path.join(path, filename)))
            elif filename[-5] == "8":
                group_2.append(np.loadtxt(os.path.join(path, filename)))
    return group_1, group_2

def load():
    path = os.path.join(root,"results","psychosis")
    group_1 = []
    group_2 = []
    for filename in os.listdir(path):
        if np.sum(np.isnan(np.loadtxt(os.path.join(path, filename)))) == 0:
            if np.shape(np.loadtxt(os.path.join(path, filename)))[0] != 415:
                print(filename)
            if filename[-5] == "9":
                group_1.append(np.loadtxt(os.path.join(path, filename)))
            elif filename[-5] == "8":
                group_2.append(np.loadtxt(os.path.join(path, filename)))
    return group_1, group_2

def main():
    baseline = []
    reps = 10
    for rep in range(reps):
        group_1, group_2 = load_unique()
        g1_fc = []
        g2_fc = []
        for item in group_1:
            g1_fc.append(np.corrcoef(item.T)[np.triu_indices(17)])
        for item in group_2:
            g2_fc.append(np.corrcoef(item.T)[np.triu_indices(17)])
        print("data loaded")
        baseline.append(svmcv(g1_fc, g2_fc))
        print(baseline[-1])
    np.savetxt(os.path.join(root, "results", "fits", "psychosis", "baseline"), baseline)
    quit()

    path = os.path.join(root, "results", "fits")
    reps = 1
    state = int(sys.argv[1])
    accuracy = []
    for rep in range(reps):
        group_1, group_2 = load_unique()
        accuracy.append(loocv_batch(group_1, group_2, state, trans=True, ar=True, key_string="psych"))
    with open(os.path.join(path, f"psychosis{state}"), "ab") as file:
        np.savetxt(file, accuracy)

if __name__ == "__main__":
    main()