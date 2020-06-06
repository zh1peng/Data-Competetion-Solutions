## Feature extraction
```python
import numpy as np
import pandas as pd
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import PCA
from nilearn.connectome import ConnectivityMeasure
import time
 
def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
    data = []
    for subject_filename in fmri_filenames:
        data.append(pd.read_csv(subject_filename,
                                 header=None).fillna(0).values)
    return np.array(data)
 
def _load_motions(motions_filenames):
    data = []
    for subject_filename in motions_filenames:
        data.append(pd.read_csv(subject_filename,delimiter="\s+", header=None,engine="python").fillna(0).values)
    return np.array(data)
 
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # make a transformer which will load the time series and compute the
        # connectome matrix
        self.fmri_names = ["fmri_motions",
                           "fmri_basc064",
                           "fmri_basc122",
                           "fmri_craddock_scorr_mean",
                           "fmri_harvard_oxford_cort_prob_2mm",
                           "fmri_msdl",
                           "fmri_power_2011"
                           ]
        self.fmri_transformers = {col : make_pipeline(FunctionTransformer(func=_load_fmri, validate=False),
                                                      ConnectivityMeasure(kind='tangent', vectorize=True))
                                                                             for col in self.fmri_names if col!="fmri_motions"}
        self.fmri_transformers["fmri_motions"] = make_pipeline(FunctionTransformer(func=_load_motions, validate=False),ConnectivityMeasure(kind='tangent', vectorize=True))
 
        self.pca = PCA(n_components=0.99)
        
    def fit(self, X_df, y):
 
        fmri_filenames = {col : X_df[col] for col in X_df.columns if col in self.fmri_names}
        
        for fmri in self.fmri_names:
            if fmri in fmri_filenames.keys():
                print("Fitting",fmri,end="")
                start = time.time()
                self.fmri_transformers[fmri].fit(fmri_filenames[fmri],y)
                print(", Done in %.3f min"%((time.time()-start)/60))
        
        X_connectome = self._transform(X_df)
        
        self.pca.fit(X_connectome)
 
        return self
 
    def _transform(self,X_df):
        fmri_filenames = {col : X_df[col] for col in X_df.columns if col in self.fmri_names}
        X_connectome = []
        for fmri in fmri_filenames:
            print("Transforming",fmri,end="")
            start = time.time()
            X_connectome.append(self.fmri_transformers[fmri].transform(fmri_filenames[fmri]))
            print(", Done in %.3f min"%((time.time()-start)/60))
        return np.concatenate(X_connectome,axis=1)
 
    def transform(self, X_df):
        
        X_connectome = self.pca.transform(self._transform(X_df))
        X_connectome = pd.DataFrame(X_connectome, index=X_df.index)
        X_connectome.columns = ['connectome_{}'.format(i)
                                for i in range(X_connectome.columns.size)]
        # get the anatomical information
        X_part = X_df[["participants_age"]]
        X_part["participants_sex"] = X_df["participants_sex"].map(lambda x : 0 if x=="M" else 1)
        X_part.columns = ['anatomy_sex','anatomy_age']
        X_anatomy = X_df[[col for col in X_df.columns
                          if col.startswith('anatomy')]]
        X_anatomy = X_anatomy.drop(columns='anatomy_select')
        # concatenate both matrices        
        return pd.concat([X_connectome, X_anatomy, X_part], axis=1)
```
        
## classifier
```python
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import time
 
 
class Classifier(BaseEstimator):
    def __init__(self):
        
        self.connect_clfs = [LogisticRegression(penalty='l2',C=1.0),
                           LogisticRegression(penalty='l2',C=100.0),
                           LogisticRegression(penalty='l1'),
                           RandomForestClassifier(max_depth=20,n_jobs=4),
                           GradientBoostingClassifier(learning_rate=0.1, n_estimators=50),
                           GradientBoostingClassifier(loss="exponential", learning_rate=0.01, n_estimators=80),
                           SVC(probability=True),
                           SVC(C=1e2,probability=True)]
        self.anatomy_clfs = [LogisticRegression(penalty='l2',C=1.0),
                           LogisticRegression(penalty='l2',C=100.0),
                           LogisticRegression(penalty='l1'),
                           RandomForestClassifier(max_depth=20,n_jobs=4),
                           GradientBoostingClassifier(learning_rate=0.1, n_estimators=50),
                           GradientBoostingClassifier(loss="exponential",learning_rate=0.01, n_estimators=80),
                           SVC(probability=True),
                           SVC(C=1e2,probability=True)]
        
        self.clfs_connectome = [make_pipeline(StandardScaler(),
                                            reg) for reg in self.connect_clfs]
        self.clfs_anatomy = [make_pipeline(StandardScaler(),
                                            reg) for reg in self.anatomy_clfs]
        
        self.meta_clf = LogisticRegression(C=1.)
        
    def _fit_connectome(self,X_connect,y):        
        for clf in self.clfs_connectome:
            print("Fitting ",clf,end="")
            start = time.time()
            clf.fit(X_connect,y)
            print(", Done in %.3f min"%((time.time()-start)/60))
    def _predict_connectome(self,X_connect):        
        res = []
        for clf in self.clfs_connectome:
            res.append(clf.predict_proba(X_connect))
        return np.concatenate(res,axis=1)
    
    def _fit_anatomy(self,X_anatomy,y):        
        for clf in self.clfs_anatomy:
            print("Fitting ",clf,end="")
            start = time.time()
            clf.fit(X_anatomy,y)
            print(", Done in %.3f min"%((time.time()-start)/60))
    def _predict_anatomy(self,X_connect):        
        res = []
        for clf in self.clfs_anatomy:
            res.append(clf.predict_proba(X_connect))
        return np.concatenate(res,axis=1)
    def fit(self, X, y):
        X_anatomy = X[[col for col in X.columns if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns if col.startswith('connectome')]]
#        train_idx, validation_idx = train_test_split(range(y.size),test_size=0.37, shuffle=True, random_state=42)
#        X_anatomy_train = X_anatomy.iloc[train_idx]
#        X_anatomy_validation = X_anatomy.iloc[validation_idx]
#        X_connectome_train = X_connectome.iloc[train_idx]
 #       X_connectome_validation = X_connectome.iloc[validation_idx]
 #       y_train = y[train_idx]
 #       y_validation = y[validation_idx]
 
#        self._fit_connectome(X_connectome_train, y_train)
#        self._fit_anatomy(X_anatomy_train, y_train)
 
 #       y_connectome_pred = self._predict_connectome(X_connectome_validation)
 #       y_anatomy_pred = self._predict_anatomy(X_anatomy_validation)
 
        self._fit_connectome(X_connectome, y)
        self._fit_anatomy(X_anatomy, y)
        y_connectome_pred = self._predict_connectome(X_connectome)
        y_anatomy_pred = self._predict_anatomy(X_anatomy)
        
#        self.meta_clf.fit(np.concatenate([y_connectome_pred, y_anatomy_pred], axis=1),y_validation)
        self.meta_clf.fit(np.concatenate([y_connectome_pred, y_anatomy_pred], axis=1),y)
        return self
 
    def predict(self, X):
        X_anatomy = X[[col for col in X.columns if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns if col.startswith('connectome')]]
 
        y_anatomy_pred = self._predict_anatomy(X_anatomy)
        y_connectome_pred = self._predict_connectome(X_connectome)
 
        return self.meta_clf.predict(np.concatenate([y_connectome_pred, y_anatomy_pred], axis=1))
 
    def predict_proba(self, X):
        X_anatomy = X[[col for col in X.columns if col.startswith('anatomy')]]
        X_connectome = X[[col for col in X.columns
                          if col.startswith('connectome')]]
 
        y_anatomy_pred = self._predict_anatomy(X_anatomy)
        y_connectome_pred = self._predict_connectome(X_connectome)
 
        return self.meta_clf.predict_proba(
            np.concatenate([y_connectome_pred, y_anatomy_pred], axis=1))
```
