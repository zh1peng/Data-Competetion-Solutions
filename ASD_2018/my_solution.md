## Feature extraction
```python
from nilearn.signal import clean
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import make_pipeline
import pandas as pd
import numpy as np
from scipy.special import boxcox1p
from scipy import stats
from scipy.stats import norm, skew
from nilearn.connectome import ConnectivityMeasure
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.preprocessing import StandardScaler
 
 
 
def motion_load(motion_txts):
    return np.array([np.genfromtxt(subject_motion,
                                 autostrip=True)
                     for subject_motion in motion_txts])
 
def fmri_load(fmri_filenames):
    return np.array([pd.read_csv(subject_filename,
                                 header=None).values
                     for subject_filename in fmri_filenames])  
 
def clean_ts(sub_motion, sub_ts):
    RMS= np.sqrt(np.mean(sub_motion**2,1)).reshape(-1,1)
    derivatives=sub_motion-np.vstack([np.zeros([1,sub_motion.shape[1]]),sub_motion[:-1]])
    #high_variance=high_variance_confounds(sub_ts, n_confounds=5, percentile=2.0, detrend=False)
    #nuisance_all=np.hstack([sub_motion,RMS, derivatives,high_variance])
    nuisance_all=np.hstack([sub_motion,RMS, derivatives])
    clean_ts_data=clean(sub_ts, sessions=None, detrend=False, standardize=False, confounds=nuisance_all,
            low_pass=0.08, high_pass=0.009)
    return clean_ts_data
 
 
 
def get_power_fc(X_df): #264
    fmri_filenames=X_df['fmri_power_2011']
    motion_txts=X_df['fmri_motions']
    motions=motion_load(motion_txts)
    ts_data=fmri_load(fmri_filenames)
    cleaned_ts= np.array([clean_ts(sub_motion, sub_ts) for sub_motion, sub_ts in zip(motions, ts_data)])
    con=ConnectivityMeasure(kind='correlation', vectorize=True, discard_diagonal=True)
    cleaned_fc=np.array(con.fit_transform(cleaned_ts))
    n=cleaned_fc.shape[-1]
    Z_fc=np.apply_along_axis(lambda r: np.log((1 + r) / (1 - r)) * (np.sqrt(n - 3) / 2), 1, cleaned_fc) 
    FC_df=pd.DataFrame(np.array(Z_fc), index=X_df.index)
    FC_df.columns = ['power_connectome_{}'.format(i)
                                 for i in range(FC_df.columns.size)]      
    return FC_df
  
def get_anatomy(X_df):
    tmp_anatomy = X_df[[col for col in X_df.columns
               if col.startswith('anatomy')]]
    tmp_anatomy1 = tmp_anatomy.drop(columns=['anatomy_select'])
    null_features=(tmp_anatomy1 == 0).astype(int).sum(axis=0)>20
    to_drop=null_features.index[null_features].tolist()
    X_anatomy=tmp_anatomy1.drop(columns=to_drop)
    return X_anatomy
 
def correct_skewness(X_df):
    skewed_feats = X_df.apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
    skewness = pd.DataFrame({'Skew' :skewed_feats})
    skewness = skewness[abs(skewness) > 0.75]
    skewed_features = skewness.index
    for feat in skewed_features:
        X_df[feat] = boxcox1p(X_df[feat], 0.15)
    return X_df
 
class FeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.transformer_fmri = make_pipeline(
            FunctionTransformer(func= get_power_fc,validate=False))
        self.transformer_anatomy = make_pipeline(
            FunctionTransformer(func=get_anatomy,validate=False))
    def fit(self, X_df, y):
        self.transformer_fmri.fit(X_df, y)
        self.transformer_anatomy.fit(X_df,y)
        return self
    def transform(self,X_df):
        X_FC=self.transformer_fmri.transform(X_df)
        X_anatomy = self.transformer_anatomy.transform(X_df)
        sex=(X_df['participants_sex']=='F').astype(int)
        X_dem=pd.concat([X_df['participants_age'],sex,X_df['participants_site']],axis=1)
        #QC_info
        X_qc=X_df[['anatomy_select','fmri_select']]
        X_qc.columns=['anatomy_qc','con_qc']
        print('feature extraction done')
        return pd.concat([X_dem, X_anatomy, X_FC, X_qc],axis=1)
```


## classifier
```python
from sklearn.preprocessing import StandardScaler,FunctionTransformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, ElasticNetCV, SGDClassifier, LogisticRegressionCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve, KFold, cross_validate,StratifiedShuffleSplit
import pandas as pd
import numpy as np
from scipy.special import boxcox1p
from scipy import stats
from scipy.stats import norm, skew
from nilearn.connectome import ConnectivityMeasure
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
import time
from sklearn.feature_selection import mutual_info_classif, f_classif,SelectPercentile
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
class multikernal_classfier(BaseEstimator,ClassifierMixin):
    def __init__(self):
        self.model1=LogisticRegression(C=0.01)
        self.model2=GaussianNB(priors=None)
        self.model3=MLPClassifier(hidden_layer_sizes=(40, 40, 50))
        self.model4=LogisticRegression(C=100)
        self.model5=SVC(probability=True)
        self.model6=MLPClassifier(hidden_layer_sizes=(80, 50, 30))
        self.model7=SVC(C=0.01,probability=True)
        self.model8=RandomForestClassifier(n_estimators=500,max_depth=6,min_samples_leaf=2,max_features='auto')
        self.model9=SGDClassifier(loss='log', penalty='elasticnet', 
                                         alpha=0.1, l1_ratio=0.1,max_iter=1000,tol=0.0001)
        self.model10=ExtraTreesClassifier(n_estimators=500,max_depth=8,min_samples_leaf=2)
    def fit(self, X,y):
        self.model1.fit(X,y)
        self.model2.fit(X,y)
        self.model3.fit(X,y)
        self.model4.fit(X,y)
        self.model5.fit(X,y)
        self.model6.fit(X,y)
        self.model7.fit(X,y)
        self.model8.fit(X,y)
        self.model9.fit(X,y)
        self.model10.fit(X,y)
        return self
    def predict(self, X):
        pred1=self.model1.predict(X)
        pred2=self.model2.predict(X)
        pred3=self.model3.predict(X)
        pred4=self.model4.predict(X)
        pred5=self.model5.predict(X)
        pred6=self.model6.predict(X)
        pred7=self.model7.predict(X)
        pred8=self.model8.predict(X)
        pred9=self.model9.predict(X)
        pred10=self.model10.predict(X)
        return np.vstack([pred1,pred2,pred3,pred4,pred5,pred6,pred7,pred8,pred9,pred10]).T
    def predict_proba(self,X):
        prob1=self.model1.predict_proba(X)[:,1]
        prob2=self.model2.predict_proba(X)[:,1]
        prob3=self.model3.predict_proba(X)[:,1]
        prob4=self.model4.predict_proba(X)[:,1]
        prob5=self.model5.predict_proba(X)[:,1]
        prob6=self.model6.predict_proba(X)[:,1]
        prob7=self.model7.predict_proba(X)[:,1]
        prob8=self.model8.predict_proba(X)[:,1]
        prob9=self.model9.predict_proba(X)[:,1]
        prob10=self.model10.predict_proba(X)[:,1]
        return np.vstack([prob1,prob2,prob3,prob4,prob5,prob6,prob7,prob8,prob9,prob10]).T
 
 
class fold_classifier(BaseEstimator):
    def __init__(self, kernal, feature_sets, n_folds=5):
        self.feature_sets=feature_sets
        self.n_folds=n_folds
        self.kernal=kernal
        self.X2meta=[]
        self.y2meta=[]
        self.pred2meta=[]
        self.feature_fold_base=[]
    def fit(self,X,y):
        folds = list(KFold(n_splits=self.n_folds,random_state=614).split(X,y)) 
        for idx,feature2use in enumerate(self.feature_sets):
            print('*******************Feature subset: '+str(idx+1)+'****************')
            X2use=X[feature2use]
            tmp_container=[]
            tmp_container1=[]
            tmp_container2=[]
            for j, (train_idx, test_idx) in enumerate(folds):
                print('===============Fold number: '+str(j+1)+' done================')
                X_train = X2use.iloc[train_idx,:]
                y_train = y[train_idx]
                X_holdout = X2use.iloc[test_idx,:]
                y_holdout = y[test_idx]
                tmp=multikernal_classfier()
                tmp.fit(X_train, y_train)
                tmp_container.append(tmp)
                y_pred=tmp.predict_proba(X_holdout)
                tmp_container1.append(y_pred)
                tmp_container2.append(y_holdout.reshape([-1,1]))
            self.feature_fold_base.append(tmp_container)
            self.X2meta.append(np.vstack(tmp_container1))
            self.y2meta.append(np.vstack(tmp_container2))
            print('******************** Feature subset -- '+str(idx+1)+' done******************')      
        return self    
    def predict_proba(self,X):
        pred2meta=[]
        for idx,feature2use in enumerate(self.feature_sets):
            X2use=X[feature2use]
            tmp_container3=[]
            for j in np.arange(self.n_folds):
                y_pred_prob=self.feature_fold_base[idx][j].predict_proba(X2use)
                tmp_container3.append(y_pred_prob.reshape([-1,1]))              
            tmp=np.hstack(tmp_container3)
            pred2meta.append(tmp.mean(1).reshape([X.shape[0],-1]))
        return np.hstack(pred2meta)    
    def X2meta(self):
        return self.X2meta
    def y2meta(self): # if use straitified-Kfold y order will change
        return self.y2meta[0]
 
 
class Classifier(BaseEstimator):
    def __init__(self):
        self.power_bm=fold_classifier(kernal=multikernal_classfier(),
                                      feature_sets=power_5_sets,n_folds=5)
        self.meta_model=GridSearchCV(LogisticRegression(penalty='l2'),
                      {'C': np.power(10.0, np.arange(-5, 5))})
        self.sd=StandardScaler()
    def fit(self, X, y):
        start_time=time.time()
        good=X['con_qc']==1
        X_good=X[good]
        y_good=y[good] 
        self.sd.fit(X_good)
        tmp_X=self.sd.transform(X_good)
        X_norm=pd.DataFrame(tmp_X, columns=X_good.columns, index=X_good.index)
        p_info = X_good[[col for col in X_good.columns
               if col.startswith('participants_age')
               or col.startswith('participants_sex')]]
        ana_info=X_norm[anatomy2use]
        print('++++++++++++++++++++++++  fit power ++++++++++++++++++++++++++')
        self.power_bm.fit(X_norm,y_good)
        X2meta=np.hstack([np.array(p_info),
                          np.array(ana_info),
                          np.hstack(self.power_bm.X2meta)])
        self.layer1=X2meta
        print('++++++++++++++++++++++++  fit meta ++++++++++++++++++++++++++')
        self.meta_model.fit(X2meta,y_good) 
        e = int(time.time() - start_time)
        print('Time elapsed:{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))
    def predict_proba(self, X):
        p_info = X[[col for col in X.columns
               if col.startswith('participants_age')
               or col.startswith('participants_sex')]]
        tmp_X=self.sd.transform(X)
        X_norm=pd.DataFrame(tmp_X, columns=X.columns, index=X.index)
        ana_info=X_norm[anatomy2use]
         
        train2meta=np.hstack([np.array(p_info),
                              np.array(ana_info),
                              self.power_bm.predict_proba(X)])
        return self.meta_model.predict_proba(train2meta)
    def predict(self, X):
        p_info = X[[col for col in X.columns
               if col.startswith('participants_age')
               or col.startswith('participants_sex')]]
        tmp_X=self.sd.transform(X)
        X_norm=pd.DataFrame(tmp_X, columns=X.columns, index=X.index)
        ana_info=X_norm[anatomy2use] 
        train2meta=np.hstack([np.array(p_info),
                              np.array(ana_info),
                              self.power_bm.predict_proba(X)])
        return self.meta_model.predict(train2meta)
    
    def tuning_meta_clf(self):
        return self.layer1
```
