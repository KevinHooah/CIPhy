'''
Disclamier: As the paper says, the implemenation of backdoor adjustment and baseline logistic regression model is following the AAAI 2016 paper: https://github.com/tapilab/aaai-2016-robust
'''
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append('../')
from tuning import *
from CORAL import *
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score, accuracy_score
from sklearn import preprocessing
from scipy import sparse
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
import time

start = time.time()
random_seed = np.random.randint(1,1001)
# print('Seed: ', random_seed)
np.random.seed(random_seed)
feature_start_idx = 3
file_path = 'freq_data_extended.csv'
gp_lr={"C":[0.001,0.01,0.1,1,10,100], "multi_class":["auto", "multinomial"], "max_iter":[100, 200, 300, 400, 500, 1000]}
c_ft_value = 1
log_name = './onemodel_' + str(c_ft_value) + '_ns_coral.txt'

for _ in range(100):
    speed_a = 4
    for speed_b in [1,2,3,5,6,7]:
        df = pd.read_csv(file_path, header=None)
        df.dropna(inplace=True)
        df.rename(columns={0:'subject',1:'speed',2:'trace'}, inplace=True)
        df.loc[(df["subject"] ==2) & (df["trace"] == 11) & (df["speed"] == 4), "trace"] = 10
        df = df[df['trace'] <= 10] # remove some has over 10 traces
        df_a = df[df['speed'] == speed_a]
        df_b = df[df['speed'] == speed_b]  

        person_list = np.random.permutation(10) + 1
        threshold = 4
        for idx, person in enumerate(person_list):
            # print(person)
            df_person_a = df_a.loc[df_a["subject"] == person]
            df_person_b = df_b.loc[df_b["subject"] == person]
            df_person_a = df_person_a.sample(frac=1).reset_index(drop=True)
            df_person_b = df_person_b.sample(frac=1).reset_index(drop=True)

            rand_a = np.random.randint(40,46)
            rand_b = np.random.randint(5,10)
            check_line = 50
            
            if idx > threshold:
                rand_a, rand_b = rand_b, rand_a
            # print(person, rand_a, rand_b)
            a_train_cache = df_person_a.iloc[0:rand_a,:]
            b_train_cache = df_person_b.iloc[0:rand_b:,:]
            a_test_cache = df_person_a.iloc[rand_a:,:]
            b_test_cache = df_person_b.iloc[rand_b:,:]

            if idx == 0:
                a_tr, b_tr, a_te, b_te = a_train_cache, b_train_cache, a_test_cache, b_test_cache
            else:
                a_tr = pd.concat([a_tr, a_train_cache], ignore_index=True)
                b_tr = pd.concat([b_tr, b_train_cache], ignore_index=True)
                a_te = pd.concat([a_te, a_test_cache], ignore_index=True)
                b_te = pd.concat([b_te, b_test_cache], ignore_index=True)

        if len(b_tr['subject'].value_counts())<10 or len(a_tr['subject'].value_counts())<10:
            print('Missing Trace!')

        # Prepare for the backdoor adjustment

        a_tr['Confounder_4'] = 1
        a_tr['Confounder_k'] = 0
        b_tr['Confounder_4'] = 0
        b_tr['Confounder_k'] = 1
        a_te['Confounder_4'] = 1
        a_te['Confounder_k'] = 0
        b_te['Confounder_4'] = 0
        b_te['Confounder_k'] = 1

        df_g1 =  pd.concat([a_tr, b_tr], ignore_index=True)
        df_g2 =  pd.concat([a_te, b_te], ignore_index=True)

        X_g1 = np.array(df_g1.iloc[:,feature_start_idx:-2])
        X_g1_c = np.array(df_g1.iloc[:,feature_start_idx:])
        X_g2 = np.array(df_g2.iloc[:,feature_start_idx:-2])
        X_g2_c = np.array(df_g2.iloc[:,feature_start_idx:])
        scaler = MinMaxScaler()
        X_g1_normed = scaler.fit_transform(X_g1)
        scaler = MinMaxScaler()
        X_g2_normed = scaler.fit_transform(X_g2)
        y_g1 = np.array(df_g1['subject'])
        y_g2 = np.array(df_g2['subject'])
        y_spe_g2 = np.array(df_g2['speed'])

        # baseline model
        clf_lr = LogisticRegression()
        gd_sr_lr = GridSearchCV(estimator=clf_lr,
                            param_grid=gp_lr,
                            scoring='accuracy',
                            cv=3,
                            n_jobs=-1,
                            refit=True,
                            verbose=0)
        gd_sr_lr.fit(X_g1, y_g1)
        best_base = gd_sr_lr.best_estimator_
        best_base_score = best_base.score(X_g2, y_g2)

        # coral model
        clf_coral = CORAL()
        coral_acc, _ = clf_coral.fit_predict(X_g1_normed, y_g1, X_g2_normed, y_g2)

        # coral_lr model
        coral = CORAL()
        clf_lr = LogisticRegression()
        X_g1_new = coral.fit(X_g1_normed,X_g2_normed)
        gd_sr_lr = GridSearchCV(estimator=clf_lr,
                            param_grid=gp_lr,
                            scoring='accuracy',
                            cv=3,
                            n_jobs=-1,
                            refit=True,
                            verbose=0)
        gd_sr_lr.fit(X_g1_new, y_g1)
        best_coral_lr = gd_sr_lr.best_estimator_
        best_coral_lr_score = best_coral_lr.score(X_g2_normed, y_g2)

        # backdoor adjustment
        unique, counts = np.unique(y_spe_g2, return_counts=True)
        dimention = np.sum(counts)
        first_val = counts[0]/np.sum(counts)
        second_val = counts[1]/np.sum(counts)

        if speed_a > speed_b:
            speed_four_prior = second_val
            speek_k_prior = first_val
        else:
            speed_four_prior = first_val
            speek_k_prior = second_val

        c_prob = np.array([speed_four_prior,speek_k_prior])
        l = X_g2.shape[0]                                                        
        rows = range(l*2)                                                     
        cols = list(range(2))*l 
        data = [c_ft_value]*(l*2)
        c = sparse.csr_matrix((data, (rows, cols)))
        p = np.array(c_prob).reshape(-1,1)
        p = np.tile(p, (X_g2.shape[0], 1))  
        repeat_indices = np.arange(X_g2.shape[0]).repeat(2)                      
        X_g2 = X_g2[repeat_indices]      
        Xc = sparse.hstack((X_g2,c))
        Xc_numpy = Xc.toarray()

        clf_lr = LogisticRegression()
        gd_sr_lr = GridSearchCV(estimator=clf_lr,
                            param_grid=gp_lr,
                            scoring='accuracy',
                            cv=3,
                            n_jobs=-1,
                            refit=True,
                            verbose=0)
        gd_sr_lr.fit(X_g1_c, y_g1)
        best_backdoor = gd_sr_lr.best_estimator_

        proba = best_backdoor.predict_proba(Xc_numpy)
        proba *= p
        proba = proba.reshape(-1, 2, 10)
        proba = np.sum(proba, axis=1) 
        norm = np.sum(proba, axis=1).reshape(-1,1)
        proba /= norm
        y_pred = np.array(proba.argmax(axis=1))+1
        best_backdoor_score = accuracy_score(y_g2, y_pred)

        # model coefficient

        scaler = MinMaxScaler()
        normed_base_coef = scaler.fit_transform(np.absolute(best_base.coef_[:,0]).reshape(-1, 1))
        scaler = MinMaxScaler()
        normed_coral_coef = scaler.fit_transform(np.absolute(best_coral_lr.coef_[:,0]).reshape(-1, 1))
        scaler = MinMaxScaler()
        normed_backdoor_coef = scaler.fit_transform(np.absolute(best_backdoor.coef_[:,0]).reshape(-1, 1))

        base_coef_sum = np.sum(np.absolute(best_base.coef_[:,0]))
        coral_coef_sum = np.sum(np.absolute(best_coral_lr.coef_[:,0]))
        back_coef_sum = np.sum(np.absolute(best_backdoor.coef_[:,0]))
        norm_base_coef_sum = np.sum(normed_base_coef)
        norm_coral_coef_sum = np.sum(normed_coral_coef)
        norm_back_coef_sum = np.sum(normed_backdoor_coef)

        # Simpson's Paradox
        X_a_tr = np.array(a_tr.iloc[:,feature_start_idx:])
        y_a_tr = np.array(a_tr['subject'])

        X_b_tr = np.array(b_tr.iloc[:,feature_start_idx:])
        y_b_tr = np.array(b_tr['subject'])

        backdoor_c1 = clone(best_backdoor)
        backdoor_c0 = clone(best_backdoor)
        base_c1 = clone(best_base)
        base_c0 = clone(best_base)
        coral_c0 = clone(best_coral_lr)
        coral_c1 = clone(best_coral_lr)

        backdoor_c1 = backdoor_c1.fit(X_a_tr, y_a_tr)
        backdoor_c0 = backdoor_c0.fit(X_b_tr, y_b_tr)
        base_c1 = base_c1.fit(X_a_tr[:,:-2], y_a_tr)
        base_c0 = base_c0.fit(X_b_tr[:,:-2], y_b_tr)
        coral_c1 = coral_c1.fit(X_a_tr[:,:-2], y_a_tr)
        coral_c0 = coral_c0.fit(X_b_tr[:,:-2], y_b_tr)

        # Simpson's paradox
        backdoor_spa_count = 0
        for j in range(10):
            if (backdoor_c1.coef_[j][0] * backdoor_c0.coef_[j][0] > 0 and backdoor_c1.coef_[j][0] * best_backdoor.coef_[j][0] < 0):
                backdoor_spa_count += 1
        

        base_spa_count = 0
        for j in range(10):
            if base_c1.coef_[j][0] * base_c0.coef_[j][0] > 0 and base_c1.coef_[j][0] * best_base.coef_[j][0] < 0:
                base_spa_count += 1

        coral_spa_count = 0
        for j in range(10):
            if coral_c1.coef_[j][0] * coral_c0.coef_[j][0] > 0 and coral_c1.coef_[j][0] * best_coral_lr.coef_[j][0] < 0:
                coral_spa_count += 1

        with open(log_name, 'a') as f:
            f.write(str(speed_a) + ',' + str(speed_b) + ',' + str(best_base_score) + ',' + str(coral_acc) + ',' + str(best_coral_lr_score) + ',' + str(best_backdoor_score) + ',' 
                  + str(base_coef_sum) + ',' + str(coral_coef_sum) + ',' + str(back_coef_sum) + ',' + str(norm_base_coef_sum) + ',' + str(norm_coral_coef_sum) +  ',' + str(norm_back_coef_sum) + ',' 
                  + str(base_spa_count) + ',' + str(coral_spa_count) + ',' + str(backdoor_spa_count) + '\n')
            f.close()

end = time.time()
print("Time Consumption: ", (end-start)/3600, " hours.")





