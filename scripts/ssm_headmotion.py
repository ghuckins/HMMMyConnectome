import ssm
import numpy as np
import random
import os
from hmm import import_raw
import seaborn as sns
import pandas as pd
import pickle
import matplotlib.pyplot as plt

root = "/Users/gracehuckins/Documents/HMMMyConnectome"


def split_data(num_networks):
    path = os.path.join(root, "data", f"data{num_networks}")
    dest_path = os.path.join(root, "data", f"data{num_networks}_split")
    if not os.path.exists(path):
        os.mkdir(path)
        import_raw(num_networks)
    os.mkdir(dest_path)
    for file_name in os.listdir(path):
        data = np.loadtxt(os.path.join(path, file_name))
        mask = np.loadtxt(os.path.join(root, "data", "tmasks", file_name[:6] + ".txt"))
        curr = mask[0]
        counter = 0
        start = 0
        for i in range(len(mask)):
            if mask[i] - curr == 1:
                start = i
                curr = mask[i]
            if mask[i] - curr == -1:
                np.savetxt(
                    os.path.join(dest_path, file_name[:-4] + f"_{counter}.txt"),
                    data[start:i, :],
                )
                curr = mask[i]
                counter += 1


def loocv_split(data1, data2, latdim):
    random.shuffle(data1)
    random.shuffle(data2)
    obsdim = np.shape(data1[0])[1]
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

    split_acc = np.average([correct1 / (correct1 + len(data2) - correct2), correct2 / (correct2 + len(data1) - correct1)])
    print(f'accuracy: {split_acc}')
    return split_acc
def relength(length,data):
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

print("code working")
quit()

hmdata = {}
hmdata["t"] = []
hmdata["r"] = []
nohmdata = {}
nohmdata["t"] = []
nohmdata["r"] = []

path1 = os.path.join(root,'results','data7_split')
path2 = os.path.join(root,'results','data7')

for filename in os.listdir(path2):
    if filename[6] != "t" and filename[6] != "r":
        continue

    data = np.loadtxt(os.path.join(path2,filename))
    num = 0
    hm_pos = 0
    while os.path.exists(os.path.join(path1, filename[:7] + f"_{num}.txt")):
        data1 = np.loadtxt(os.path.join(path1, filename[:7] + f"_{num}.txt"))
        nohmdata[filename[6]].append(data1)
        hmdata[filename[6]].append(data[hm_pos:hm_pos+len(data1)])
        hm_pos += len(data1)
        num += 1

path = os.path.join(root,"results","fits","hmm7_split")
iters = 5
states = [2,3,4,5]
for iter in range(iters):
    for state in states:
        hmacc = np.array(loocv_split(hmdata["t"],hmdata["r"],state))
        hmacc = np.reshape(hmacc,(1,1))
        with open(os.path.join(path, f"fit{state}_motion_ar"), "ab") as file:
            np.savetxt(file, hmacc)
        nohmacc = np.array(loocv_split(nohmdata["t"], nohmdata["r"], state))
        nohmacc = np.reshape(nohmacc, (1, 1))
        with open(os.path.join(path, f"fit{state}_nomotion_ar"), "ab") as file:
            np.savetxt(file, nohmacc)


quit()

accs = []
motion = []
states = []
for filename in os.listdir(os.path.join(root, "results/fits/hmm7_split")):
    accuracies = np.loadtxt(os.path.join(root, "results/fits/hmm7_split", filename))
    accs.extend(accuracies)
    motion.extend([filename[5]=="m"]*len(accuracies))
    states.extend([int(filename[3])]*len(accuracies))

datadict = {}
datadict["Classification Accuracy"] = accs
datadict["Hidden States"] = states
datadict["Head Motion"] = motion
data = pd.DataFrame(data=datadict)

path = os.path.join(root, "results", "fits")
with open(os.path.join(path, "dataframesplit1"), "wb") as file:
    pickle.dump(data,file)

sns.set_theme()
colors = [[51 / 255, 34 / 255, 136 / 255], [136 / 255, 204 / 255, 238 / 255], [17 / 255, 119 / 255, 51 / 255],
          [153 / 255, 153 / 255, 51 / 255], [204 / 255, 102 / 255, 119 / 255], [136 / 255, 34 / 255, 85 / 255]]
sns.set_palette(sns.color_palette(colors))
fig = sns.relplot(data=data, x="Hidden States", y="Classification Accuracy", hue="Head Motion", kind="line",
                  errorbar="ci").set_titles('Head Motion Test', weight='bold', size=14)
sns.move_legend(fig,"upper right",bbox_to_anchor=(0.817, 0.93))
fig.legend.set_title(None)
fig.legend.set(frame_on=True)

fig.fig.subplots_adjust(top=.9)
plt.ylim([0.4, 1])
plt.title("Head Motion Test", weight='bold', fontsize=14)

fig.tight_layout()
plt.savefig("./results/figs/head_motion_1", facecolor=(1, 1, 1, 0))
plt.show()
quit()


