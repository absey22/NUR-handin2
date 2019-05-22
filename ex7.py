import numpy as np
import h5py



with h5py.File('/home/seymour/Documents/NUR/Numerical-Recipes/handin/handin2/NUR-handin2/colliding.hdf5', 'r') as hf:
    coords=hf.get('PartType4').get('Coordinates').value
    masses=hf.get('PartType4').get('Masses').value

cutcoords=coords[:151,:2]

import matplotlib.pyplot as plt
plt.plot(cutcoords[:,0],cutcoords[:,1],'ro')
plt.show()

#https://jheer.github.io/barnes-hut/
#http://arborjs.org/docs/barnes-hut
#https://ko.coursera.org/lecture/modeling-simulation-natural-processes/barnes-hut-algorithm-using-the-quadtree-9csRt
