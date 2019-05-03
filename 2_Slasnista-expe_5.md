## Feature extraction
```python
import numpy as np
import pandas as pd
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
 
from nilearn.connectome import ConnectivityMeasure
 
 
def _load_fmri(fmri_filenames):
    """Load time-series extracted from the fMRI using a specific atlas."""
 
    if "motions" in fmri_filenames.name:
 
        fmri = []
        for subject_filename in fmri_filenames:
            f = pd.read_csv(subject_filename, header=None).values
 
            X_ = []
            for f_ in f:
                pass
                x = np.asarray(
                    [f__ for f__ in f_[0].split(" ") if f__ != ""]).astype(np.float)
                X_.append(x.reshape(1, -1))
            X_ = np.concatenate(X_)
            fmri.append(X_)
    else:
        fmri = np.array([pd.read_csv(subject_filename,
                         header=None).values
                        for subject_filename in fmri_filenames])
 
    return fmri
 
 
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # make a transformer which will load the time series and compute the
        # connectome matrix
 
        self.fmris = [
            'msdl',
            'basc064', 'basc122', 'basc197',
            'harvard_oxford_cort_prob_2mm', 'craddock_scorr_mean',
            'power_2011',
            "motions"
        ]
 
        self.transformer_fmri = dict()
 
        for fmri in self.fmris:
            self.transformer_fmri[fmri] = make_pipeline(
                FunctionTransformer(func=_load_fmri, validate=False),
                ConnectivityMeasure(kind='tangent', vectorize=True))
 
    def fit(self, X_df, y):
        # get only the time series for the MSDL atlas
 
        for fmri in self.fmris:
            fmri_filenames = X_df['fmri_{}'.format(fmri)]
            self.transformer_fmri[fmri].fit(fmri_filenames, y)
 
        return self
 
    def transform(self, X_df):
 
        columns_anatomy = [
            c for c in X_df.columns if c.startswith("anatomy_")]
        X_anat = X_df[columns_anatomy].values
 
        X_connectome = []
        # columns = []
        for fmri in self.fmris:
            fmri_filenames = X_df['fmri_{}'.format(fmri)]
 
            if "motions" in fmri:
                X_ = np.concatenate([
                    self.transformer_fmri[fmri].transform(
                        fmri_filenames), X_anat], axis=1)
            else:
                X_ = np.concatenate([
                    self.transformer_fmri[fmri].transform(
                        fmri_filenames), X_anat], axis=1)
 
            columns = [
                "{}_{}".format(fmri, i)
                for i in range(X_.shape[1])]
            df_ = pd.DataFrame(columns=columns, data=X_)
            X_connectome.append(df_)
 
        return pd.concat(X_connectome, axis=1)
```

## Classifier

```python
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
 
 
class Classifier(BaseEstimator):
    def __init__(self):
        self.clf = dict()
 
        self.fmris = [
            'msdl',
            'basc064', 'basc122', 'basc197',
            'harvard_oxford_cort_prob_2mm', 'craddock_scorr_mean',
            'power_2011',
            "motions"
        ]
 
        for fmri in self.fmris:
            self.clf[fmri] = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=1.))
 
    def fit(self, X, y):
 
        for fmri in self.fmris:
 
            columns = [c for c in X if fmri in c]
            X_ = X[columns].values
            self.clf[fmri].fit(X_, y)
 
        return self
 
    def predict(self, X):
 
        y_pred = dict()
 
        for fmri in self.fmris:
 
            columns = [c for c in X if fmri in c]
            X_ = X[columns].values
 
            y_pred[fmri] = np.expand_dims(
                self.clf[fmri].predict_proba(X_), axis=0)
 
        y_pred = np.mean(np.concatenate(
            [y_pred[k] for k in y_pred]), axis=0)[:, 1]
 
        return y_pred
 
    def predict_proba(self, X):
 
        y_pred = dict()
 
        for fmri in self.fmris:
 
            columns = [c for c in X if fmri in c]
            X_ = X[columns].values
 
            y_pred[fmri] = np.expand_dims(
                self.clf[fmri].predict_proba(X_), axis=0)
 
        y_pred = np.mean(np.concatenate(
            [y_pred[k] for k in y_pred]), axis=0)
 
        return y_pred


```
