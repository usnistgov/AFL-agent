import xarray as xr
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.preprocessing import OrdinalEncoder
import warnings

import textwrap

from AFL.agent.util import ternary_to_xy,composition_grid,mpl_format_ternary

from sklearn.preprocessing import StandardScaler


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
        self.ranges = {}
    
    def _get_default(self,components=None,add_grid=False):
        if (components is None):
            try:
                components = self.data.attrs['components']
            except KeyError:
                raise ValueError('Must pass components or set "components" in Dataset attributes.')
        if add_grid:
            components = [c if 'grid' in c else c+'_grid' for c in components]
        return components

    def get_standard_scaled(self,components=None):
        comp = self.get(components)
        comp_scaled = comp.copy(data=StandardScaler().fit_transform(comp))
        return comp_scaled
        
    def get_range_scaled(self,ranges,components=None):
        components = self._get_default(components)
        comp = self.get(components)
        
        for component in components:
            if component not in ranges:
                raise ValueError('Cannot scale components without grid ranges specified. Call dataset.afl.comp.set_grid_range()')
            comp.loc[dict(component=component)] = comp.sel(component=component).pipe(lambda x: x/ranges[component])
        return comp
        
        
    def get(self,components=None):
        components = self._get_default(components)
        
        comp = self.data[components].to_array('component')
        comp.name = 'compositions'
        return comp.transpose(...,'component')

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
        
    def add_grid(self,components=None,pts_per_row=50,basis=1.0,dim_name='grid',ternary=False,overwrite=False):
        components = self._get_default(components)
        print(f'--> Making grid for components {components} at {pts_per_row} pts_per_row')

        if ternary:
            compositions = composition_grid_ternary(
                    pts_per_row = pts_per_row,
                    basis = basis,
                    dim=len(components),
                    )
        else:
            range_spec = {}
            for component in components:
                range_spec[component] = {
                    # 'min':self.data.attrs[component+'_grid_range'][0],
                    # 'max':self.data.attrs[component+'_grid_range'][1],
                    'min':self.data.attrs[component+'_range'][0],
                    'max':self.data.attrs[component+'_range'][1],
                    'steps':pts_per_row
                }
            print(range_spec)
            compositions = composition_grid(components,range_spec)

        for component in components:
            name = component+'_grid'
            if name in self.data:
                if overwrite:
                    del self.data[name]
                else:
                    raise ValueError(f'Component {name} already in Dataset and overwrite is False.')

        components_grid = []
        for component,comps in zip(components,compositions.T):
            name = component+'_grid'
            self.data[name] = (dim_name,comps)
            components_grid.append(name)
        self.data.attrs['components_grid'] = components_grid
        for component in components:
            self.data.attrs[component+'_grid_range'] = self.data.attrs[component+'_range']
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
        if 'edgecolor' not in mpl_kw:
            mpl_kw['edgecolor'] = 'face'
            
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
    
    def plot_continuous(self,components=None,labels=None,set_labels=True,ternary=True,**mpl_kw):
        warnings.warn('plot_continuous is deprecated and will be removed in a future release. Please use plot_surface.',DeprecationWarning,stacklevel=2)
        
        components = self._get_default(components)

        if ternary and len(components)==3:
            xy = self.ternary_to_xy(components)
        elif len(components)==2:
            xy = np.vstack(list(self.data[c].values for c in components)).T
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
                mpl_format_ternary(plt.gca(),*components)
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
            
        markers = ['^','v','<','>','o','d','p','x']
        artists = []
        for label in np.unique(labels):
            mask = (labels==label)
            mpl_kw['marker'] = markers.pop(0)
            artists.append(ax.scatter(*coords[mask].T,**mpl_kw))

        if set_axes_labels:
            if projection=='ternary':
                labels = {k:v for k,v in zip(['tlabel','llabel','rlabel'],components)}
            else:
                labels = {k:v for k,v in zip(['xlabel','ylabel'],components)}
            ax.set(**labels)
            ax.grid('on',color='black')
        return artists

    def plot_discrete(self,components=None,labels=None,set_labels=True,normalize=True,ternary=True,**mpl_kw):
        warnings.warn('plot_discrete is deprecated and will be removed, please use plot_scatter.',DeprecationWarning,stacklevel=2)
        
        components = self._get_default(components)

        if ternary and len(components)==3:
            xy = self.ternary_to_xy(components,normalize=normalize)
        elif len(components)==2:
            xy = np.vstack(list(self.data[c].values for c in components)).T
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
                mpl_format_ternary(plt.gca(),*components)
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
            
    def ternary_to_xy(self,components=None,normalize=True):
        '''Ternary composition to Cartesian coordinate'''
        components = self._get_default(components)
            
        if not (len(components)==3):
            raise ValueError('Must specify exactly three components')
        
        comps = np.vstack(list(self.data[c].values for c in components)).T
        
        return ternary_to_xy(comps,normalize)
    
        

    def plot_mask(self,mask_name='mask',components_name='components_grid'):
        self.plot_discrete(self.data.attrs[components_name],labels=self.data[mask_name].astype(int),s=1)
    