'''
All the model tuning codes are here.
'''
import warnings
warnings.filterwarnings("ignore")
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


def lrTuning(xTr, yTr, xTe, yTe, clflr, cvMethod):
    '''
    This is for multi-class logistic regression.
    '''
    gp_lr={"C":[0.001,0.01,0.1,1,10,100], "multi_class":["auto", "ovr", "multinomial"], "max_iter":[100, 200, 300, 400, 500, 1000]}
    gd_sr_lr = GridSearchCV(estimator=clflr,
	                    param_grid=gp_lr,
	                    scoring='accuracy',
	                    cv=cvMethod,
	                    n_jobs=-1,
	                    refit=True,
	                    verbose=0)
    gd_sr_lr.fit(xTr, yTr)
    best_lr = gd_sr_lr.best_estimator_
    # pred = best_lr.predict(xTe)
    # cm = confusion_matrix(yTe, pred)
    print(best_lr.score(xTe,yTe))

    return best_lr