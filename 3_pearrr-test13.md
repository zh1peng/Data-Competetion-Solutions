## Feature extraction
```python
import numpy as np
import pandas as pd
from nilearn.signal import clean
from scipy.special import erfinv
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from nilearn.connectome import ConnectivityMeasure
 
 
def _load_fmri(df):
    """Load time-series extracted from the fMRI using a specific atlas."""
    res = []
    for index, row in df[['subject_id', 'fmri', 'repetition_time']].iterrows():
        # path_motions_subject = './data/fmri/motions/%s/run_1/motions.txt' % row['subject_id']
        # confounds = np.loadtxt(path_motions_subject)
        timeseries = pd.read_csv(row['fmri'], header=None).values
        res.append(clean(timeseries,
                         #confounds=confounds,
                         t_r=row['repetition_time'],
                         standardize=False
                         ))
 
    return np.array(res)
 
def gauss_rank_transform(df):
    for col in df.columns.values.tolist():
        values = sorted(set(df[col]))
        # Because erfinv(1) is inf, we shrink the range into (-0.9, 0.9)
        f = pd.Series(np.linspace(0, 0.99, len(values)), index=values)
        f = np.sqrt(2) * erfinv(f)
        f -= f.mean()
        df[col] = df[col].map(f)
 
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer_harvard_oxford = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind='tangent', vectorize=True))
        self.transformer_msdl = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind='tangent', vectorize=True))
        self.transformer_basc197 = make_pipeline(
            FunctionTransformer(func=_load_fmri, validate=False),
            ConnectivityMeasure(kind='tangent', vectorize=True))
 
    def fit(self, X_df, y):
        df = X_df[['fmri_msdl', 'repetition_time']].reset_index()
        df['fmri'] = df['fmri_msdl']
        self.transformer_msdl.fit(df, y)
 
        df = X_df[['fmri_harvard_oxford_cort_prob_2mm', 'repetition_time']].reset_index()
        df['fmri'] = df['fmri_harvard_oxford_cort_prob_2mm']
        self.transformer_harvard_oxford.fit(df, y)
 
        df = X_df[['fmri_basc197', 'repetition_time']].reset_index()
        df['fmri'] = df['fmri_basc197']
        self.transformer_basc197.fit(df, y)
        return self
 
    def transform(self, X_df):
        df = X_df[['fmri_msdl', 'repetition_time']].reset_index()
        df['fmri'] = df['fmri_msdl']
        X_msdl = pd.DataFrame(self.transformer_msdl.transform(df), index=X_df.index)
        X_msdl.columns = ['msdl_{}'.format(i) for i in range(X_msdl.columns.size)]
 
        df = X_df[['fmri_harvard_oxford_cort_prob_2mm', 'repetition_time']].reset_index()
        df['fmri'] = df['fmri_harvard_oxford_cort_prob_2mm']
        X_harvard_oxford = pd.DataFrame(self.transformer_harvard_oxford.transform(df), index=X_df.index)
        X_harvard_oxford.columns = ['harvard_oxford_{}'.format(i) for i in range(X_harvard_oxford.columns.size)]
 
        df = X_df[['fmri_basc197', 'repetition_time']].reset_index()
        df['fmri'] = df['fmri_basc197']
        X_basc197 = pd.DataFrame(self.transformer_basc197.transform(df), index=X_df.index)
        X_basc197.columns = ['basc197_{}'.format(i) for i in range(X_basc197.columns.size)]
 
        X_anatomy = X_df[[col for col in X_df.columns if col.startswith('anatomy')]].drop(columns='anatomy_select')
        gauss_rank_transform(X_anatomy)
 
        X_common = X_df[['participants_site', 'participants_sex', 'participants_age']]
        gauss_rank_transform(X_common)
 
        X_select = X_df[['anatomy_select', 'fmri_select']]
 
        return pd.concat([X_msdl, X_harvard_oxford, X_basc197, X_anatomy, X_common, X_select], axis=1)
 ```
 
 
 ## Classifier
 ```python
 import numpy as np
from sklearn import decomposition
 
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import pandas as pd
 
 
class Classifier(BaseEstimator):
    def __init__(self):
        test_weights = {0: 0.5892323030907278, 1: 0.4107676969092722}
        self.clf = LogisticRegression(C=0.1, tol=2, verbose=2, penalty='l2', class_weight=test_weights, solver="liblinear")
 
    def fit(self, X, y):
        selected = X[((X['anatomy_select'] == 1) | (X['fmri_select'] == 1))].index.values
        y = pd.Series(y, index=X.index)
        y = y.loc[selected]
        X = X.loc[selected].drop(['anatomy_select', 'fmri_select'], axis=1)
        self.clf.fit(X, y)
        return self
 
    def predict(self, X):
        X = X.drop(['anatomy_select', 'fmri_select'], axis=1)
        return self.clf.predict(X)
 
    def predict_proba(self, X):
        X = X.drop(['anatomy_select', 'fmri_select'], axis=1)
        return self.clf.predict_proba(X)
 ```
