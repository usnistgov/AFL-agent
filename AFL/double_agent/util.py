import numpy as np
import matplotlib.pyplot as plt
import tqdm
from itertools import product

def listify(obj):
    if isinstance(obj, str) or not hasattr(obj, "__iter__"):
        obj = [obj]
    return obj

def ternary_to_xy(comps,normalize=True):
    '''Ternary composition to Cartesian coordinate'''
        
    if not (comps.shape[1]==3):
        raise ValueError('Must specify exactly three components')
    
    if normalize:
        comps = comps/comps.sum(1)[:,np.newaxis]
        
    # Convert ternary data to cartesian coordinates.
    xy = np.zeros((comps.shape[0],2))
    xy[:,1] = comps[:,1]*np.sin(60.*np.pi / 180.)
    xy[:,0] = comps[:,0] + xy[:,1]*np.sin(30.*np.pi/180.)/np.sin(60*np.pi/180)
    return xy
    
def mpl_format_ternary(ax=None,label_right=None,label_top=None,label_left=None):
    if ax is None:
        ax = plt.gca()
        
    ax.axis('off')
    ax.set_aspect('equal','box')
    ax.set(xlim = [0,1],ylim = [0,1])
    ax.plot([0,1,0.5,0],[0,0,np.sqrt(3)/2,0],ls='-',color='k')
    
    if label_right is not None:
        ax.text(1,0,label_right,ha='left')
    if label_top is not None:
        ax.text(0.5,np.sqrt(3)/2,label_top,ha='center',va='bottom')
    if label_left is not None:
        ax.text(0,0,label_left,ha='right')
        


def composition_grid_ternary(pts_per_row=50,basis=1.0,dim=3,eps=1e-9):
    try:
        from tqdm.contrib.itertools import product
    except ImportError:
        from itertools import product
    pts = []
    for i in product(*[np.linspace(0,1.0,pts_per_row)]*(dim-1)):
        if sum(i)>(1.0+eps):
            continue
            
        j = 1.0-sum(i)
        
        if j<(0.0-eps):
            continue
        pt = [k*basis for k in [*i,j]]
        pts.append(pt)
    return np.array(pts)

def composition_grid(components,range_spec):
    try:
        from tqdm.contrib.itertools import product
    except ImportError:
        from itertools import product
        
    grid_list =[]
    for component in components:
        spec = range_spec[component]
        grid_list.append(np.linspace(spec['min'],spec['max'],spec['steps']))
    
    pts = np.array(list(product(*grid_list)))
    return pts