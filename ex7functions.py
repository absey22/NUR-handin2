import numpy as np



def quadtree_makechildren(subcoords): # take a list of x,y coords and make four children
# by dividing the space of the coordinates into 4 (as long as there is more than one coordinate)
    children=[]
    N=len(subcoords)
    if N>12: # create four child nodes
        xcoords,ycoords=subcoords[:,0],subcoords[:,1]
        mid_x,mid_y=np.mean(xcoords),np.mean(ycoords)
        nw=subcoords[np.logical_and(xcoords<mid_x,ycoords>mid_y)] # divide by location
        children.append(nw)
        ne=subcoords[np.logical_and(xcoords>mid_x,ycoords>mid_y)]
        children.append(ne)
        se=subcoords[np.logical_and(xcoords>mid_x,ycoords<mid_y)]
        children.append(se)
        sw=subcoords[np.logical_and(xcoords<mid_x,ycoords<mid_y)]
        children.append(sw)
    else:
        print("leaf node reached.") # if there would be only one element left in a box
        children=subcoords
    return children,N

def quadtree_build(coordinates): # only needs two iterations to reach 12 members, do it manually
    root=[]
    nodeholder=[] # for manual appending of 2nd BH layer
    newnodes1,N=quadtree_makechildren(coordinates)
    for node in newnodes1:
        newnodes2,N=quadtree_makechildren(node) #create 4 more children to get below N=12/node
        nodeholder.append(newnodes2)
    for leafnode in nodeholder:
        root.append(leafnode) # append to root node the 4 nodes of 4 leaf nodes each
    return root

