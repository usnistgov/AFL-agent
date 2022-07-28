import numpy as np
import gpflow
from gpflow.ci_utils import ci_niter
from gpflow.monitor import (
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)
import tensorflow as tf
from scipy.stats import entropy
from sklearn.neighbors import KernelDensity
import tqdm
import warnings

from AFL.agent.PhaseMap import to_xy

class DummyGP:
    def predict(self,compositions):
        y_mean = np.zeros_like(compositions)
        y_var = np.zeros_like(compositions)
        return y_mean,y_var

    
class GP:
    def __init__(self,ds,components,num_classes):
        self.ds = ds

        if 'labels' not in ds:
            raise ValueError('Must have labels in Dataset before making GP!')

        if 'labels_ordinal' not in ds:
            self.ds = self.ds.afl.labels.make_ordinal()
        
        self.components = components
        self.num_classes = num_classes
        
        self.reset_GP()
        
        self.iter_monitor = lambda x: None
        self.final_monitor = lambda x: None
        
    def reset_GP(self,kernel=None):
        warnings.warn('Code is not fully generalized for non-independent coordinates. Use with care...',stacklevel=2)
        
        if len(self.components)==3:
            xy = self.ds.afl.comp.to_xy(self.components)
            data = (xy, self.ds['labels_ordinal']) 
        else:
            comp = self.ds.afl.comp.get(self.components).values#[:,:-1]#only need N-1 compositions
            data = (comp, self.ds['labels_ordinal']) 
            
        if kernel is None:
            kernel = gpflow.kernels.Matern32(variance=0.1,lengthscales=0.1) 
            # kernel +=  gpflow.kernels.White(variance=0.01)   
        invlink = gpflow.likelihoods.RobustMax(self.num_classes)  
        likelihood = gpflow.likelihoods.MultiClass(self.num_classes, invlink=invlink)  
        self.model = gpflow.models.VGP(
            data=data, 
            kernel=kernel, 
            likelihood=likelihood, 
            num_latent_gps=self.num_classes
        ) 
        self.loss = self.model.training_loss_closure(compile=True)
        self.trainable_variables = self.model.trainable_variables
        self.optimizer = tf.optimizers.Adam(learning_rate=0.001)
        
    def reset_monitoring(self,log_dir='test/',iter_period=1):
        model_task = ModelToTensorBoard(log_dir, self.model,keywords_to_monitor=['*'])
        lml_task   = ScalarToTensorBoard(log_dir, lambda: self.loss(), "Training Loss")
        
        fast_tasks = MonitorTaskGroup([model_task,lml_task],period=iter_period)
        self.iter_monitor = Monitor(fast_tasks)
        
        image_task = ImageToTensorBoard(
            log_dir, 
            self.plot, 
            "Mean/Variance",
            fig_kw=dict(figsize=(18,6)),
            subplots_kw=dict(nrows=1,ncols=3)
        )
        slow_tasks = MonitorTaskGroup(image_task) 
        self.final_monitor = Monitor(slow_tasks)

    def optimize(self,N,final_monitor_step=None,progress_bar=False):
        if progress_bar:
            for i in tqdm.tqdm(tf.range(N),total=N):
                self._step(i)
        else:
            for i in tf.range(N):
                self._step(i)
            
        if final_monitor_step is None:
            final_monitor_step = i
        self.final_monitor(final_monitor_step)
            
    @tf.function
    def _step(self,i):
        self.optimizer.minimize(self.loss,self.trainable_variables) 
        self.iter_monitor(i)
    
    def predict(self,compositions):
        if len(self.components)==3:
            xy_dense = to_xy(compositions)
            self.y = self.model.predict_y(xy_dense)
        else:
            #throw out last composition as it's not linearly indepedent
            self.y = self.model.predict_y(compositions.values[:,:-1])
        
        y_mean = self.y[0].numpy() 
        y_var = self.y[1].numpy() 
        return y_mean,y_var

    
