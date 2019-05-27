import numpy as np
import h5py
import matplotlib.pyplot as plt


from ex7functions import quadtree_makechildren,quadtree_build

with h5py.File('/home/seymour/Documents/NUR/Numerical-Recipes/handin/handin2/NUR-handin2/colliding.hdf5', 'r') as hf:
    coords=hf.get('PartType4').get('Coordinates').value
    masses=hf.get('PartType4').get('Masses').value

#get the first 150 particle positions
cutcoords=coords[:150,:2]
particlemass=masses[0] # all masses are the same



#check if more than 1 pt in array
#div by 4


BHtree=quadtree_build(cutcoords)


children=quadtree_makechildren(cutcoords)[0]
nw,ne=children[0],children[1]
se,sw=children[2],children[3] # for plotting

plt.title("B.H. Quadtree Nodes")
plt.xlabel("x-coordinate")
plt.ylabel("y-coordinate")
plt.plot(cutcoords[:,0],cutcoords[:,1],'k^',label="Input")
plt.plot(nw[:,0],nw[:,1],'b.',label='NW')
plt.plot(ne[:,0],ne[:,1],'y.',label='NE')
plt.plot(se[:,0],se[:,1],'g.',label='SE')
plt.plot(sw[:,0],sw[:,1],'r.',label='SW')
plt.plot(cutcoords[99,0],cutcoords[99,1],'*',markersize=15,label='i = 100')
plt.legend()

plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.savefig("./plots/7_150particles1stiter.png")
plt.clf()

for i in range(len(BHtree)):
    children=BHtree[i]
    nw,ne=children[0],children[1]
    se,sw=children[2],children[3] # for plotting
    plt.suptitle("B.H. Quadtree Leafnodes")
    plt.subplot(2,2,i+1)
    plt.title("Node "+str(i+1))
    if i==2 or i==3:
        plt.xlabel("x-coordinate")
    if i==0 or i==2:
        plt.ylabel("y-coordinate")
    plt.plot(cutcoords[:,0],cutcoords[:,1],'k^',label="Input")
    plt.plot(nw[:,0],nw[:,1],'bo',label='NW')
    plt.plot(ne[:,0],ne[:,1],'yo',label='NE')
    plt.plot(se[:,0],se[:,1],'go',label='SE')
    plt.plot(sw[:,0],sw[:,1],'ro',label='SW')
    plt.plot(cutcoords[99,0],cutcoords[99,1],'*',markersize=15,label='i = 100')
    if i==4:
        plt.legend()

plt.tight_layout(pad=0.4, w_pad=2.5, h_pad=1.0)
plt.savefig("./plots/7_150particles2nditer.png")
plt.clf()



#calculate the zeroeth multipole moment of each leaf and parent node, up to the root
# the particle with index 100 is in the SW leaf node of the first NE node of the quad tree

#mass in each leaf node
M0leafnodes=np.zeros(16)
i,j=0,0
for node in BHtree:
    for leafnode in node:
        M0leafnodes[i]=len(leafnode)*particlemass
        print(j,i)
        if j==1 and i-4*j==3: # the node "coordinate" for the SW leafnode of the NE node
            print("yes")
            M0_i100leafnode=M0leafnodes[i]
        i+=1
    j+=1
#mass in each node
M0nodes=np.zeros(4)
for i in range(M0nodes.shape[0]):
    M0nodes[i]=np.sum(M0leafnodes[i+3*i:4*i+4])
    if i==1:
        M0_i100node=M0nodes[i]

#mass in root (parent) node (just the total mass of all 150 particles)
M0root=np.sum(M0nodes)

print("The i = 100th particle is located in the SW leafnode of the NE node of the B.H. Quadtree.")
print("Its leafnode has the n=0 multipole moment:",M0_i100leafnode)
print("Its parent node has the n=0 multipole moment:",M0_i100node)
print("The root nodes n=0 multipole moment is:",M0root)
print("(Which is also the total mass of all 150 particles:",particlemass*150.,")")


#https://jheer.github.io/barnes-hut/
#http://arborjs.org/docs/barnes-hut
#https://ko.coursera.org/lecture/modeling-simulation-natural-processes/barnes-hut-algorithm-using-the-quadtree-9csRt
