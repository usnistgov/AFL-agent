#Package Tree Hierarchy
#Author Graham Roberts
#email: grahamrobertsw@gmail.com
#V0.1.0 -- A very early version of the code which contains the basic structure of the tree hierarchy, as well as some functionality for reading and writing from various json formats.

import numpy as np
from sklearn.svm import SVC
from sklearn.kernel_ridge import KernelRidge as KRR
import json
from joblib import dump, load
import joblib
from io import BytesIO
import base64

#TreeHierarchy
#   allows construction of custom hierarchical classifiers
#   these allow for multiclass classification with indepenently tuned classifiers at each branch
#   this is a recursie structure
class TreeHierarchy:

    def _init_(self):
        self.left = None
        self.right = None
        self.content = None
        self.terminal = None
        self.classA = None
        self.classB = None
        #self.__dict__ = self.vars()

    
    def vars(self):
        outdict = {}
        for k in self.dir():
            v = getattr(self, k, None)
            if v is not None:
                if isinstance(v, TreeHierarchy):
                    outdict[k] = vars(v)
                elif isinsstance(v, np.ndarray):
                    outdict[k] = list(v)
                elif isinstance(v, SVC):
                    tempdict = {}
                    for k2, v2 in v.__dict__:
                        if isinstance(v2, np.ndarray):
                            tempdict[k2] = list(v2)
                        else:
                            tempdict[k2] = v2
                    outdict[k] = tempdict
        return(outdict)
#        return(None)
    
    def add_content(self, contant, terminal):
        self.content = content
        self.terminal = terminal

    def add_left(self, entity):
        self.left = entity

    def add_right(self, entity):
        self.right = entity

    #Fit follows the sklearn fit structure and recursively calls fit on each component tree
    def fit(self, X, y):
        if not getattr(self, 'terminal', False):
           templabels = np.zeros(X.shape[0])
           for l in np.unique(self.classB):
               for i in range(len(y)):
                   if y[i] == l:
                       templabels[i] = 1
           self.entity.fit(X, templabels)
           ia = np.where(templabels == 0)[0]
           ib = np.where(templabels == 1)[0]
           self.left.fit(X[ia], y[ia])
           self.right.fit(X[ib], y[ib])
        return

    
    def predict(self, X):
        if X.shape[0] == 0:
            y = np.zeros(0)
        elif not getattr(self, 'terminal', False):
            inds = np.arange(X.shape[0])
            temp_y = self.entity.predict(X)
            ia = np.where(np.logical_not(temp_y))[0]
            ib = np.where(temp_y)[0]
            return_y = np.empty(X.shape[0], dtype=object)
            y_a = self.left.predict(X[ia])
            y_b = self.right.predict(X[ib])
            for v in range(ia.shape[0]):
                vi = ia[v]
                return_y[vi] = y_a[v]
            for v in range(ib.shape[0]):
                vi = ib[v]
                return_y[vi] = y_b[v]
            y = return_y
        else:
            y = np.array([self.terminal] * X.shape[0])
        return(y)

    #Structure from json takes as input a json structure and constructs the tree based on that structure
    def structure_from_json(self, J):
        if 'class' in J.keys():
            self.terminal = J['class']
        else:
            if 'jobfile' in J.keys():
                self.entity = load(J['jobfile'])
                print('Loading %s'%(J['jobfile']))
            elif 'classifier' in J.keys():
                print(J['classifier'])
                self.entity = classifier_from_json(J['classifier'])
                print(self.entity)
            else:
                self.entity = None
            self.left = TreeHierarchy()
            self.left.structure_from_json(J['left'])
            self.right = TreeHierarchy()
            self.right.structure_from_json(J['right'])
            self.classA = J['classA']
            self.classB = J['classB']

#Pass to json.dumps as an encoder class
#decontructs the tree, component SVCs KRRs and npArrays into primitives as well.
class TreeEncoder(json.JSONEncoder):


    def default(self, obj):
        """If input object is an ndarray it will be converted into a dict 
        holding dtype, shape and the data, base64 encoded.
        """
        if isinstance(obj, np.ndarray):
            if obj.flags['C_CONTIGUOUS']:
                obj_data = obj.data
            else:
                cont_obj = np.ascontiguousarray(obj)
                assert(cont_obj.flags['C_CONTIGUOUS'])
                obj_data = cont_obj.data
            data_b64 = base64.b64encode(obj_data)
            return dict(__ndarray__=data_b64.decode('utf-8'),
                        dtype=str(obj.dtype),
                        shape=obj.shape)
        elif isinstance(obj, SVC):
            return(obj.__dict__)
        elif isinstance(obj, KRR):
            return(obj.__dict__)
        elif isinstance(obj, TreeHierarchy):
            return(obj.__dict__)
        else:
        # Let the base class default method raise the TypeError
            super().default(obj)

def json_decoder(dct):
    """Decodes a previously encoded TreeHierarchy, numpy ndarray with proper shape and dtype, SVC, or KRR.

    :param dct: (dict) json encoded ndarray
    :return: (ndarray) if input was an encoded ndarray
    """
    if isinstance(dct, dict) and '__ndarray__' in dct:
        data = base64.b64decode(dct['__ndarray__'].encode('utf-8'))
        return np.frombuffer(data, dct['dtype']).reshape(dct['shape'])
    elif isinstance(dct, dict) and 'support_vectors_' in dct:
        obj = SVC()
        for (k, v) in dct.items():
            setattr(obj, k, json_decoder(v))
        return(obj)
    elif isinstance(dct, dict) and 'alpha' in dct:
        obj = KRR()
        for (k, v) in dct.items():
            setattr(obj, k, json_decoder(v))
        return(obj)
    elif isinstance(dct, dict) and ("classA" in dct or "terminal" in dct):
        obj = TreeHierarchy()
        for (k,v) in dct.items():
            setattr(obj, k, json_decoder(v))
        return(obj)
    elif isinstance(dct, dict):
        newdct = {k: json_decoder(v) for (k,v) in dct.items()}
        return(newdct)
    return dct

###
###    def to_dict(self):
###        if getattr(self, "terminal", False):
###            return({"class": self.terminal})
###        else:
###            return({#"classifier": self.entity,
###                    "classLeft": self.classA,
###                    "classRight": self.classB,
###                    "left":self.left.to_dict(),
###                    "right": self.right.to_dict()})
###
###        
###    def toJSON(self):
###        bc = BytesIO()
###        joblib.dump(self,bc)
###        return(json.dumps({"tree":str(bc.getvalue())}))
###
###    def to_json(self, fn):
###        with open(fn, "w") as f:
###            json.dump(self.to_dict(), f)
###        return


###def classifier_from_json(J):
###    if J['type'] in ['svc', 'SVC']:
###        print("SVC")
###        K = J['kernel']
###        classifier = SVC(C = J['C'],
###        gamma = J['gamma'],
###        kernel = K['type'],
###        degree = K['degree'] if K['type'] == 'polynomial' else 1,
###        coef0 = J['coeff0'] if 'coeff0' in J.keys() else J['coef0'])
###        print(classifier)
###    else:
###        classifier = None
###    return(classifier)