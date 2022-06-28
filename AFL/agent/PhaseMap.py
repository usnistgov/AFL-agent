import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import OrdinalEncoder

import textwrap

@xr.register_dataarray_accessor('afl')
class AFL_DataArrayTools:
    def __init__(self,da):
        self.da = da
        self.comp = CompositionTools(da)
        self.scatt = ScatteringTools(da)
        
@xr.register_dataset_accessor('afl')
class AFL_DatasetTools:
    def __init__(self,ds):
        self.ds = ds
        self.comp = CompositionTools(ds)
        self.labels = LabelTools(ds)

    def append(self,data_dict,concat_dim='system'):
        '''Append data to current dataset (warning: much copying and data loss)'''

        # need to make sure all DataArrays have concat_dim
        # initial pass to prep for new Dataset creation
        for k in data_dict.keys():
            # make sure all variables in data_dict are already in this dataset
            if k not in self.ds:
                raise ValueError(textwrap.fill(textwrap.dedent(f'''
                All variables in data_dict {list(data_dict.keys())} must be
                in this Dataset before appending
                ''')))

            # make sure all DataArrays have concat_dim
            v = data_dict[k]
            if isinstance(v,xr.DataArray):
                if concat_dim not in v.coords:
                    data_dict[k] =  xr.DataArray([v],dims=[concat_dim,*v.dims],coords=v.coords)

        ds_new = xr.Dataset()
        for k,v in data_dict.items():
            if isinstance(v,xr.DataArray):
                ds_new[k] = xr.concat([self.ds[k].copy(),v.copy()],dim=concat_dim)
            else:
                v_xr = xr.DataArray([v],dims=concat_dim)
                ds_new[k] = xr.concat([self.ds[k].copy(),v_xr],dim=concat_dim)
        return ds_new

class LabelTools:
    def __init__(self,data):
        self.data = data

    def make_default(self,name='labels',dim='system'):
        data = self.data.copy()
        data['labels'] = xr.DataArray(np.ones(self.data[dim].shape[0]),dims=dim,coords=self.data[dim].coords)
        return data

    def make_ordinal(self,ordinal_name='labels_ordinal',labels_name='labels',sortby=None):
        encoder = OrdinalEncoder()
        data = self.data.copy()
        if sortby is not None:
            data = data.sortby(sortby)
        labels_ordinal = encoder.fit_transform(data[labels_name].values.reshape(-1,1)).flatten()
        data[ordinal_name]  = data[labels_name].copy(data=labels_ordinal)
        return data
        
class ScatteringTools:
    def __init__(self,data):
        self.data = data
        
    def plot_logwaterfall(self,x='q',dim='system',ylabel='Intensity [$cm^{-1}$]',xlabel=r'q [$Å^{-1}$]',legend=True,base=10,ax=None):
        N = self.data[dim].shape[0]
        mul = self.data[dim].copy(data=np.geomspace(1,float(base)**(N+1),N))
        lines = (self
         .data
         .pipe(lambda x:x*mul)
         .plot
         .line(
            x=x,
            xscale='log',
            yscale='log',
            marker='.',
            ls='None',
            add_legend=legend,
            ax=ax)
        )
        plt.gca().set( xlabel=xlabel,  ylabel=ylabel, )
        if legend:
            sns.move_legend(plt.gca(),loc=6,bbox_to_anchor=(1.05,0.5))
        return lines
            
    def plot_waterfall(self,x='q',dim='system',ylabel='Intensity [$cm^{-1}$]',xlabel=r'q [$Å^{-1}$]',legend=True,base=1,ax=None):
        N = self.data[dim].shape[0]
        mul = self.data[dim].copy(data=np.linspace(1,base*N,N))
        lines = (self
         .data
         .pipe(lambda x:x+mul)
         .plot
         .line(
            x=x,
            xscale='linear',
            yscale='linear',
            marker='.',
            ls='None',
            add_legend=legend,
            ax=ax)
        )
        plt.gca().set( xlabel=xlabel,  ylabel=ylabel, )
        if legend:
            sns.move_legend(plt.gca(),loc=6,bbox_to_anchor=(1.05,0.5))
        return lines
            
    def plot_loglog(self,x='q',ylabel='Intensity [$cm^{-1}$]',xlabel=r'q [$Å^{-1}$]',legend=True,ax=None):
        lines = self.data.plot.line(
            x=x,
            xscale='log',
            yscale='log',
            marker='.',
            ls='None',
            add_legend=legend,
            ax=ax)
        plt.gca().set( xlabel=xlabel,ylabel=ylabel)
        if legend:
            sns.move_legend(plt.gca(),loc=6,bbox_to_anchor=(1.05,0.5))
        return lines
    def plot_linlin(self,x='q',ylabel='Intensity [$cm^{-1}$]',xlabel=r'q [$Å^{-1}$]',legend=True,ax=None):
        lines = self.data.plot.line(
            x=x,
            marker='.',
            ls='None',
            add_legend=legend,
            ax=ax)
        plt.gca().set( xlabel=xlabel,ylabel=ylabel)
        if legend:
            sns.move_legend(plt.gca(),loc=6,bbox_to_anchor=(1.05,0.5))
        return lines
            
    def derivative(self,order=1,qlo_isel=0,qhi_isel=-1,npts=500,sgf_window_length=31,sgf_polyorder=2):
        #log scale
        data2 =self.data.copy()
        data2.name='I'
        data2 = data2.where(data2>0.0,drop=True).pipe(np.log10)
        data2 = data2.isel(q=slice(qlo_isel,qhi_isel))
        data2['q'] = np.log10(data2.q)
        
        #interpolate to constant log(dq) grid
        qnew = np.geomspace(data2.q.min(),data2.q.max(),npts)
        dq = qnew[1]-qnew[0]
        data2 = data2.interp(q=qnew)
        
        #filter out any q that have NaN
        data2 = data2.dropna('q','any')
        
        #take derivative
        dI = savgol_filter(data2.values.T,window_length=sgf_window_length,polyorder=sgf_polyorder,delta=dq,axis=0,deriv=order)
        data2_dI = data2.copy(data=dI.T)
        return data2_dI
    
        
class CompositionTools:
    def __init__(self,data):
        self.data = data
        self.default = None

    def set_default(self,components):
        self.default = components
        return self.data

    def _get_default(self,components=None):
        if (components is None):
            if self.default is not None:
                components = self.default
            else:
                raise ValueError('Must pass components or use .set_default')
        return components

    def get(self,components=None):
        components = self._get_default(components)

        da_list = []
        for k in components:
            da_list.append(self.data[k])
        comp = xr.concat(da_list,dim='component')
        comp.name='compositions'
        comp = comp.assign_coords(component=components)
        return comp.transpose()#ensures columns are components
        
    def add_grid(self,components=None,pts_per_row=50,basis=100,dim_name='grid'):
        components = self._get_default(components)

        compositions = composition_grid(
                pts_per_row = pts_per_row,
                basis = basis,
                dim=len(components),
                )

        for component in components:
            name = component+'_grid'
            if name in self.data:
                del self.data[name]

        for component,comps in zip(components,compositions.T):
            name = component+'_grid'
            self.data[name] = (dim_name,comps)
        return self.data

    
    def plot_scatter(self,components=None,labels=None,**mpl_kw):
        components = self._get_default(components)

        if len(components)==3:
            xy = self.to_xy(components)
            ternary=True
        elif len(components)==2:
            xy = np.vstack(list(self.data[c].values for c in components)).T
            ternary=False
        else:
            raise ValueError(f'Can only work with 2 or 3 components. You passed: {components}')
        
        if (labels is None):
            if ('labels' in self.data.coords):
                labels = self.data.coords['labels'].values
            elif ('labels' in self.data):
                labels = self.data['labels'].values
            else:
                labels = np.zeros(xy.shape[0])
                
        artists = []
        for label in np.unique(labels):
            mask = (labels==label)
            artists.append(plt.scatter(*xy[mask].T,**mpl_kw))
            
        if ternary:
            format_ternary(plt.gca(),*components)
        else:
            plt.gca().set(xlabel=components[0],ylabel=components[1])
        return artists
            
        
            
    def to_xy(self,components=None):
        '''Ternary composition to Cartesian coordinate'''
        components = self._get_default(components)
            
        if not (len(components)==3):
            raise ValueError('Must specify exactly three components')
        
        comps = np.vstack(list(self.data[c].values for c in components)).T
        
        return to_xy(comps)
            
    
def to_xy(comps,normalize=True):
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
    
def format_ternary(ax=None,label_right=None,label_top=None,label_left=None):
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
        


def composition_grid(pts_per_row=50,basis=100,dim=3,eps=1e-9):
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
