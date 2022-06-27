
class Labeler:
    def __init__(self,da):
        self.da = da
        self.fig,self.axes = plt.subplots(1,2,figsize=(8,4))
        AFL_Agent.format_plot_ternary(
            self.axes[1],
            label_b='PS',
            label_c='PS-PMMA',
            label_a='PMMA'
        )
        
        
        self.line_I =  self.da[0].plot(xscale='linear',yscale='log',ax=self.axes[0])[0]
        self.line_x =  self.axes[1].plot(0.5,0.5,color='red',marker='x')[0]
        self.axes[0].set(ylim=[None,1e4],xlim=(None,0.1))
        
        self.qstar = 0.01
        self.vlines = []
        for _ in range(3):
            l = self.axes[0].axvline(self.qstar,ls=':',alpha=0.5,lw=0.5,color='red')
            self.vlines.append(l)
        
        self.reset()
    
    def reset(self,*args,**kwargs):
        self.index = 0
        self.labels = []
        self.update_plot()
    
    def ternary_grid(self):
        others = [[1,2],[0,2],[0,1]]
        for i in [0,1,2]:
            for x in np.linspace(0,100,6):
                comp1 = np.zeros((2,3))
                comp1[:,i] = x
                comp1[0,others[i][0]] = 100-x
                comp1[1,others[i][1]] = 100-x
                cart = comp2cart(comp1)
                self.axes[1].plot(*cart.T,ls=':',alpha=0.5,lw=0.5,color='black')

    def update_plot(self):
        self.line_I.set_xdata(self.da[self.index].q.values)
        self.line_I.set_ydata(self.da[self.index].values)
    
        comp = np.array([self.da[self.index].concat_dim.item()])
        cart = comp2cart(comp*100.0)
        self.line_x.set_xdata(cart[0][0])
        self.line_x.set_ydata(cart[0][1])
        
        title = 'PS:{}, PS-PMMA:{}, PMMA:{}'.format(*comp[0])
        self.axes[0].set(title=title)
        
    def label(self,label):
        self.labels.append(label)
        self.index+=1
        self.update_plot()
        
    def label_C(self,*args,**kwargs):
        self.label('C')
        
    def label_L(self,*args,**kwargs):
        self.label('L')
        
    def label_S(self,*args,**kwargs):
        self.label('S')
        
    def label_D(self,*args,**kwargs):
        self.label('D')
        
    def update_peak_locations(self,*args,**kwargs):
        if self.phase_models.value=='LAM':
            locs = [1,2,3]
        elif self.phase_models.value=='HEX1':
            locs = [1,np.sqrt(3),np.sqrt(4)]
        elif self.phase_models.value=='HEX2':
            locs = [1,2/np.sqrt(3),np.sqrt(3)]
        elif self.phase_models.value=='SPH':
            locs = [1,np.sqrt(2),np.sqrt(3)]
        else:
            raise ValueError('Peak model not recognized')
        
        for line in self.vlines:
            line.set_linestyle('None')
        self.fig.canvas.draw()
        
        for loc,line in zip(locs,self.vlines):
            line.set_xdata([self.qstar*loc,self.qstar*loc])
            line.set_linestyle('-')
        self.fig.canvas.draw()
        
    def onclick(self,event):
        self.qstar=event.xdata
        self.update_peak_locations()
        # tx = 'button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata)
        # self.axes[0].set_title(tx)
        # with self.output:
        #     print(tx)
    
    def show(self):
        
        self.b0 = ipywidgets.Button(description='Cylinder (yellow)')
        self.b1 = ipywidgets.Button(description='Lamellae (purple)')
        self.b2 = ipywidgets.Button(description='Sphere (red)')
        self.b3 = ipywidgets.Button(description='Disordered (blue)')
        self.br = ipywidgets.Button(description='Reset')
        
        self.phase_models = ipywidgets.Dropdown(
            options=['LAM','HEX1','HEX2','SPH']
        )
        self.phase_models.observe(self.update_peak_locations)
        
        hbox = ipywidgets.HBox([self.b0,self.b1,self.b2,self.b3,self.br])
        self.output = ipywidgets.Output()
        vbox = ipywidgets.VBox([hbox,self.phase_models,self.output])
        
        self.b0.on_click(self.label_C)
        self.b1.on_click(self.label_L)
        self.b2.on_click(self.label_S)
        self.b3.on_click(self.label_D)
        self.br.on_click(self.reset)
            
        self.fig.canvas.mpl_connect('button_press_event',self.onclick)
        #self.axes[0].callbacks.connect('button_press_event',self.onclick)
        
        return vbox
        