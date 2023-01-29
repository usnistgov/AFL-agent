import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import OrdinalEncoder
import warnings

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

    def append(self,data_dict,concat_dim='sample'):
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

    def make_default(self,name='labels',dim='sample'):
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
        
    def plot_logwaterfall(self,x='q',dim='sample',ylabel='Intensity [A.U.]',xlabel=r'q [$Å^{-1}$]',legend=True,base=10,ax=None):
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
        if legend and (len(lines)>1):
            sns.move_legend(plt.gca(),loc=6,bbox_to_anchor=(1.05,0.5))
        return lines
            
    def plot_waterfall(self,x='logq',dim='sample',ylabel='Intensity [A.U.]',xlabel=r'q [$Å^{-1}$]',legend=True,base=1,ax=None):
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
        if legend and (len(lines)>1):
            sns.move_legend(plt.gca(),loc=6,bbox_to_anchor=(1.05,0.5))
        return lines
            
    def plot_loglog(self,x='q',ylabel='Intensity [A.U.]',xlabel=r'q [$Å^{-1}$]',legend=True,ax=None,**mpl_kw):
        lines = self.data.plot.line(
            x=x,
            xscale='log',
            yscale='log',
            marker='.',
            ls='None',
            add_legend=legend,
            ax=ax,
            **mpl_kw)
        plt.gca().set( xlabel=xlabel,ylabel=ylabel)
        if legend and (len(lines)>1):
            sns.move_legend(plt.gca(),loc=6,bbox_to_anchor=(1.05,0.5))
        return lines
    def plot_linlin(self,x='logq',ylabel='Intensity [A.U.]',xlabel=r'q [$Å^{-1}$]',legend=True,ax=None,**mpl_kw):
        lines = self.data.plot.line(
            x=x,
            marker='.',
            ls='None',
            add_legend=legend,
            ax=ax,
            **mpl_kw)
        plt.gca().set( xlabel=xlabel,ylabel=ylabel)
        if legend and (len(lines)>1):
            sns.move_legend(plt.gca(),loc=6,bbox_to_anchor=(1.05,0.5))
        return lines
            
    def clean(self,qlo=None,qhi=None,qlo_isel=None,qhi_isel=None,pedestal=1e-12,npts=250,derivative=0,sgf_window_length=31,sgf_polyorder=2,apply_log_scale=True):
        data2 =self.data.copy()
        data2.name='I'
        
        if ((qlo is not None)or(qhi is not None)) and ((qlo_isel is not None)or(qhi_isel is not None)):
            warnings.warn((
                'Both value and index based indexing have been specified! '
                'Value indexing will be applied first!'
            )
                ,stacklevel=2)
            
        data2 = data2.sel(q=slice(qlo,qhi))
        data2 = data2.isel(q=slice(qlo_isel,qhi_isel))
        
        if pedestal is not None:
            data2+=pedestal
            
        qname = 'logq'
        if apply_log_scale:
            data2 = data2.where(data2>0.0,drop=True)
            data2 = data2.pipe(np.log10)
            data2['q'] = np.log10(data2.q)
            data2 = data2.rename(q=qname)
            
        #need to remove duplicate values
        data2 = data2.groupby(qname,squeeze=False).mean()
        
        #set minimum value of scattering to pedestal value and fill nans with this value
        if pedestal is not None:
            data2 += pedestal
            data2  = data2.where(~np.isnan(data2)).fillna(pedestal)
        
        #interpolate to constant log(dq) grid
        qnew = np.geomspace(data2[qname].min(),data2[qname].max(),npts)
        dq = qnew[1]-qnew[0]
        data2 = data2.interp({qname:qnew})
        
        #filter out any q that have NaN
        data2 = data2.dropna(qname,'any')
        
        #take derivative
        dI = savgol_filter(data2.values.T,window_length=sgf_window_length,polyorder=sgf_polyorder,delta=dq,axis=0,deriv=derivative)
        data2_dI = data2.copy(data=dI.T)
        return data2_dI
    
        
class CompositionTools:
    def __init__(self,data):
        self.data = data
        self.cmap = 'viridis'


    def _get_default(self,components=None,add_grid=False):
        if (components is None):
            try:
                components = self.data.attrs['components']
            except KeyError:
                raise ValueError('Must pass components or set "components" in Dataset attributes.')
        if add_grid:
            components = [c if 'grid' in c else c+'_grid' for c in components]
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

    def get_grid_old(self,components=None):
        components = self._get_default(components,add_grid=True)

        da_list = []
        for k in components:
            da_list.append(self.data[k+'_grid'])
        comp = xr.concat(da_list,dim='component')
        comp.name='compositions_grid'
        comp = comp.assign_coords(component=components)
        return comp.transpose()#ensures columns are components

    def get_grid(self,components=None):
        components = self._get_default(components,add_grid=False)
        components_grid = self._get_default(components,add_grid=True)
        
        comp   = self.data.set_index(grid=components_grid).grid#.transpose(...,'component')
        values = np.array([np.array(i) for i in comp.values])#need to convert to numpy array
        comp = comp.expand_dims(axis=1,component=len(components)).copy(data=values)
        comp = comp.assign_coords(component=components).transpose(...,'component')
        comp.name = 'composition_grid'
        return comp
        
    def add_grid(self,components=None,pts_per_row=50,basis=1.0,dim_name='grid',overwrite=False):
        components = self._get_default(components)
        print(f'--> Making grid for components {components} at {pts_per_row} pts_per_row')

        compositions = composition_grid(
                pts_per_row = pts_per_row,
                basis = basis,
                dim=len(components),
                )

        for component in components:
            name = component+'_grid'
            if name in self.data:
                if overwrite:
                    del self.data[name]
                else:
                    raise ValueError('Component {component} already in Dataset and overwrite is False.')

        components_grid = []
        for component,comps in zip(components,compositions.T):
            name = component+'_grid'
            self.data[name] = (dim_name,comps)
            components_grid.append(name)
        self.data.attrs['components_grid'] = components_grid
        return self.data
    
    def plot_surface(self,components=None,labels=None,set_axes_labels=True,**mpl_kw):
        components = self._get_default(components)
        
        if len(components)==3:
            try:
                import mpltern
            except ImportError as e:
                raise ImportError('Could not import mpltern. Please install via conda or pip') from e
                
            projection = 'ternary'
        elif len(components)==2:
            projection = None
        else:
            raise ValueError(f'plot_surface only compatible with 2 or 3 components. You passed: {components}')
            
        coords = np.vstack(list(self.data[c].values for c in components)).T

        if (labels is None):
            if ('labels' in self.data.coords):
                labels = self.data.coords['labels'].values
            elif ('labels' in self.data):
                labels = self.data['labels'].values
            else:
                labels = np.zeros(coords.shape[0])
        elif isinstance(labels,str) and (labels in self.data):
                labels = self.data[labels].values
                
        if ('cmap' not in mpl_kw) and ('color' not in mpl_kw):
             mpl_kw['cmap'] = 'viridis'
            
        if ('ax' in mpl_kw):
            ax = mpl_kw['ax']
        else:  
            fig,ax = plt.subplots(1,1,subplot_kw=dict(projection=projection))
            
        artists = ax.tripcolor(*coords.T,labels,**mpl_kw)

        if set_axes_labels:
            if projection=='ternary':
                labels = {k:v for k,v in zip(['tlabel','llabel','rlabel'],components)}
            else:
                labels = {k:v for k,v in zip(['xlabel','ylabel'],components)}
            ax.set(**labels)
            ax.grid('on',color='black')
        return artists
    
    def plot_continuous(self,components=None,labels=None,set_labels=True,**mpl_kw):
        warnings.warn('plot_continuous is deprecated and will be removed in a future release. Please use plot_surface.',DeprecationWarning,stacklevel=2)
        
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
        elif isinstance(labels,str) and (labels in self.data):
                labels = self.data[labels].values
                
        if ('cmap' not in mpl_kw) and ('color' not in mpl_kw):
             mpl_kw['cmap'] = 'viridis'
        if 'color' not in mpl_kw:
            mpl_kw['c'] = labels
        artists = plt.scatter(*xy.T,**mpl_kw)

        if set_labels:
            if ternary:
                format_ternary(plt.gca(),*components)
            else:
                plt.gca().set(xlabel=components[0],ylabel=components[1])
        return artists
    
    def plot_scatter(self,components=None,labels=None,set_axes_labels=True,**mpl_kw):
        components = self._get_default(components)
        
        if len(components)==3:
            try:
                import mpltern
            except ImportError as e:
                raise ImportError('Could not import mpltern. Please install via conda or pip') from e
                
            projection = 'ternary'
        elif len(components)==2:
            projection = None
        else:
            raise ValueError(f'plot_surface only compatible with 2 or 3 components. You passed: {components}')
            
        coords = np.vstack(list(self.data[c].values for c in components)).T
        
        if (labels is None):
            if ('labels' in self.data.coords):
                labels = self.data.coords['labels'].values
            elif ('labels' in self.data):
                labels = self.data['labels'].values
            else:
                labels = np.zeros(coords.shape[0])
        elif isinstance(labels,str) and (labels in self.data):
                labels = self.data[labels].values
                
        if ('ax' in mpl_kw):
            ax = mpl_kw.pop('ax')
        else:  
            fig,ax = plt.subplots(1,1,subplot_kw=dict(projection=projection))
            
        artists = []
        for label in np.unique(labels):
            mask = (labels==label)
            artists.append(ax.scatter(*coords[mask].T,**mpl_kw))

        if set_axes_labels:
            if projection=='ternary':
                labels = {k:v for k,v in zip(['tlabel','llabel','rlabel'],components)}
            else:
                labels = {k:v for k,v in zip(['xlabel','ylabel'],components)}
            ax.set(**labels)
            ax.grid('on',color='black')
        return artists

    def plot_discrete(self,components=None,labels=None,set_labels=True,normalize=True,**mpl_kw):
        warnings.warn('plot_discrete is deprecated and will be removed, please use plot_scatter.',DeprecationWarning,stacklevel=2)
        
        components = self._get_default(components)

        if len(components)==3:
            xy = self.to_xy(components,normalize=normalize)
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
        elif isinstance(labels,str) and (labels in self.data):
                labels = self.data[labels].values
                
        artists = []
        for label in np.unique(labels):
            mask = (labels==label)
            artists.append(plt.scatter(*xy[mask].T,**mpl_kw))
            
        if set_labels:
            if ternary:
                format_ternary(plt.gca(),*components)
            else:
                plt.gca().set(xlabel=components[0],ylabel=components[1])
        return artists

    def plot_3D(self,components=None,labels=None,set_labels=True,**mpl_kw):
        from mpl_toolkits import mplot3d

        components = self._get_default(components)

        if not (len(components)==3):
            raise ValueError(f'Can only work with  3 components. You passed: {components}')

        xy = np.vstack(list(self.data[c].values for c in components)).T
        
        if (labels is None):
            if ('labels' in self.data.coords):
                labels = self.data.coords['labels'].values
            elif ('labels' in self.data):
                labels = self.data['labels'].values
            else:
                labels = np.zeros(xy.shape[0])


        fig = plt.figure()
        ax = plt.axes(projection='3d')
                
        artists = []
        for label in np.unique(labels):
            mask = (labels==label)
            artists.append(ax.scatter3D(*xy[mask].T,**mpl_kw))

        ax.set(xlabel=components[0],ylabel=components[1],zlabel=components[2])
            
        
            
    def to_xy(self,components=None,normalize=True):
        '''Ternary composition to Cartesian coordinate'''
        components = self._get_default(components)
            
        if not (len(components)==3):
            raise ValueError('Must specify exactly three components')
        
        comps = np.vstack(list(self.data[c].values for c in components)).T
        
        return to_xy(comps,normalize)

    def plot_mask(self,mask_name='mask',components_name='components_grid'):
        self.plot_discrete(self.data.attrs[components_name],labels=self.data[mask_name].astype(int),s=1)
            
    
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
        


def composition_grid(pts_per_row=50,basis=1.0,dim=3,eps=1e-9):
    warnings.warn('composition_grid assumes that only dim-1 points are indepenent',stacklevel=2)
    #XXX Need to generalize for independent and non-indepedent coords
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
