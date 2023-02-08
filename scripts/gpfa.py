import numpy as np
from elephant.gpfa import GPFA
import quantities as pq

dimensionality = 5
data = np.loadtxt('sub014.txt')
print(data[:,0])
#gpfa = GPFA(pq.ms, x_dim = dimensionality)
#gpfa.fit(data)

#trajectories = gpfa.trqansform(data)

#print(np.shape(trajectories))