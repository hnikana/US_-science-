# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 18:35:24 2020

@author: gn8525
"""
import numpy as np
import pandas as pd 

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix , accuracy_score , precision_score , recall_score , f1_score , precision_recall_fscore_support , precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score , roc_auc_score , roc_curve

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb

#import lightgbm as lgb
#import catboost as ctb

import pickle
import time  





### HyperOpt Parameter Tuning
from hyperopt import hp , tpe , STATUS_OK , STATUS_FAIL , Trials , fmin




figfont_size = 18 
titlefont_size = 18 
labelfont_size = 16 
ticksize = 14



def validation_metrics(X_train, y_train, X_test, y_test, fitted_pipeline):
    """    Print validation metrics including precision, recall, accuracy, confusion matrix.
    :param X_train: training predictors of loan level data
    :param y_train: training target of loan level data
    :param X_test: test predictors of loan level data
    :param y_test: test target of loan level data
    :param log_model: logistic regression
    :return: predicted result with col names ['prob', 'pred_label', 'actual_label']
    """
    # TODO: move this into another function
    # test data set prediction
    y_pred_prob = fitted_pipeline.predict_proba(X_test)[:, 1]
    y_pred = fitted_pipeline.predict(X_test)
    df_ll_prediction = pd.DataFrame(np.stack([y_pred_prob, y_pred, y_test], axis=1),
                                    columns=['prob', 'pred_label', 'actual_label'])

    # print the confusion matrix
    cnf_matrix = confusion_matrix(y_pred, y_test)
    print("""
     confusion matrix: y_actual vs y_pred
                 |     Actual
                 |   0   |   1
    ----------------------------
    Predict   0  |   a   |  b
              1  |   c   |  d  
    """
          )
    print(cnf_matrix)
    # plt.ylabel('Predicted label') #0-1 top to bottom
    # plt.xlabel('Actual label') # 0-1 left to right

    print("Accuracy of logistic regression classifier on train set: {:.2f}".
          format(fitted_pipeline.score(X_train, y_train)))
    print("Accuracy of logistic regression classifier on test set: {:.2f}".
          format(fitted_pipeline.score(X_test, y_test)))
    # print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    print("Precision d/(c + d):", precision_score(y_test, y_pred))
    print("Recall d/(b + d):", recall_score(y_test, y_pred))
    print("AUC:", roc_auc_score(y_test, y_pred))
    print('F1 score:', f1_score(y_test, y_pred))
    return df_ll_prediction
   

def AUC_curve(classifier ,X_test,  y_test, y_pred, y_score , title , path = '') :
    fpr, tpr, _ = roc_curve(y_test,  y_score[:,1])
    auc = roc_auc_score(y_test , y_pred)
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(1,2,1)
    plt.plot([0,1],[0,1] , color='darkblue' , linestyle='--')
    plt.plot(fpr,tpr, color = 'orange',label=" auc="+str(auc))
    recall = recall_score(y_test,y_pred )
    precision =  precision_score(y_test,y_pred)
    f_score = f1_score(y_test,y_pred)
    textstr = '\n Recall = {:.2f} \n Percision = {:.2f} \n F1_score = {:.2f}'.format (recall ,precision , f_score )
    # textstr = '\n'.join((
    # r'$\Recall=%.2f$' % (recall, ),
    # r'$\mathrm{Percision}=%.2f$' % (precision, ),
    # r'$\F!_score=%.2f$' % (f_score, )))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
    ax1.set_xlabel('False Positive')
    ax1.set_ylabel('True Positive')
    ax1.legend(loc=4)
    ax2 = fig.add_subplot(1,2,2)
    average_precision = average_precision_score(y_test, y_score[:,1])
    lr_precision, lr_recall, _ = precision_recall_curve(y_test, y_score[:,1])
    plt.plot(lr_recall , lr_precision, label = 'Average Percision = {:.2f}'.format(average_precision))
   # plot_precision_recall_curve(classifier, X_test, y_test)
    # disp.ax_.set_title('2-class Precision-Recall curve: '
    #                 'AP={0:0.2f}'.format(average_precision))
    
    ax2.set_xlabel('Precision')
    ax2.set_ylabel('Recall')
    ax2.legend()
    
    fig.suptitle( title , fontweight = 'bold' , y=0.94, fontsize = figfont_size)
    
    fig.savefig(path+title+'.png')
    plt.show()
    
    
    
def plot_importance (importance , n_features , feature_list , title ,  path = ''  ) :
    #import pdb; pdb.set_trace()
    if importance.shape[0] != 1:
        importance = np.array(importance).reshape((1, len(importance)))
    # import pdb; pdb.set_trace()
    indices = importance.argsort()[0,:][::-1]
    fig = plt.figure(figsize=(12,8))
    plt.bar(range(n_features) , importance[0,indices[:n_features].tolist()])
    plt.xticks(range(n_features), feature_list[indices[:n_features]] , rotation='vertical')
    plt.xlabel('Feature' , fontsize = labelfont_size)
    plt.ylabel('Importance' ,  fontsize = labelfont_size)
    plt.tick_params(labelsize=ticksize, pad=6)
    #plt.table(cellText=np.array(pd.DataFrame(df_.columns[indices_tree[:20]] , index = indices_tree[:20])) , loc = 'bottom' , rowLabels = indices_tree[:20] , edges = 'open' )
    fig.suptitle( title , fontweight = 'bold' , y=0.94, fontsize = figfont_size)
    fig.savefig(path+title+'.png')
    plt.show()
 
# grid = False 
# if grid :
#     path = '/wsu/home/gn/gn85/gn8525/In_market_timing/chunked_preprocess/'
# else :
#     path = ''

# # df = pd.read_csv( path+ 'data_concat_2.csv')
    
#df = pd.read_pickle('data_concat_sample.pkl').reset_index()  # df.shape = (2259585, 73) # pd.concat requires unique indices 

df = pd.read_pickle('data_state_3/data_Michigan.pkl')  # it is in data_state


#df.dropna(axis = 'index' , inplace = True )      


cat_variable =  ['quarter' , 'income_categorical' ,'#OFCars_categorical' , 'income_census_categorical' ]
num_variable = ['CarAge'  ,   'Q_1', 'Q_2', 'Q_3', 'Q_4','p_16', 'p_15', 'p_14',
       'p_13', 'p_12', 'p_11', 'p_10', 'p_9', 'p_8', 'p_7', 'p_6', 'p_5',
       'p_4', 'p_3', 'p_2', 'p_1' , 'total_car_census' , 'OldestAge',
       'AveCar'  ] 

onehot_variable =  ['BoyAgeBw0And2', 'GirlAgeBw0And2',
       'UnknownAgeBw0And2', 'BoyAgeBw3And5', 'GirlAgeBw3And5',
       'UnknownAgeBw3And5', 'BoyAgeBw6And10', 'GirlAgeBw6And10',
       'UnknownAgeBw6And10', 'BoyAgeBw11And15', 'GirlAgeBw11And15',
       'UnknownAgeBw11And15', 'BoyAgeBw16And17', 'GirlAgeBw16And17',
       'UnknownAgeBw16And17', 'MalesAgeBw18And24', 'FemalesAgeBw18And24',
       'UnknownAgeBw18And24', 'MalesAgeBw25And34', 'FemalesAgeBw25And34',
       'UnknownAgeBw25And34', 'MalesAgeBw35And44', 'FemalesAgeBw35And44',
       'UnknownAgeBw35And44', 'MalesAgeBw45And54', 'FemalesAgeBw45And54',
       'UnknownAgeBw45And54', 'MalesAgeBw55And64', 'FemalesAgeBw55And64',
       'UnknownAgeBw55And64', 'MalesAgeBw65And74', 'FemalesAgeBw65And74',
       'UnknownAgeBw65And74', 'MalesAge75Plus', 'FemalesAge75Plus',
       'UnknownAge75Plus']

feature_drop = ['ID' , 'date' , 'income' , 'censusId', 'year' , 'cen_income' , '#OfCars' , 'SumAge' , 'ZipCode' , 'State' , 'cen_income']
          



# Dropping no_car 
#df = pd.read_pickle('data_state_3/data_Michigan.pkl')  # it is in data_state
df= df[~(df['#OFCars_categorical']=='No_car')].copy(True) # Dropping the 'no-car level from the data '

df_drop = df[feature_drop]
y = df['label'].astype('int')
df.drop('label' , axis = 1 , inplace = True) # df.shape = (2259585, 73)
df.drop(feature_drop , axis = 1 , inplace = True) # (2259585, 64)


assert  'label' in df.columns , 'label is not in the feature space'    


df_cat = pd.get_dummies(df[cat_variable]) 
df_cat.reset_index(drop = True , inplace = True)
df_num = pd.DataFrame(StandardScaler().fit_transform(df[num_variable]), columns = num_variable)
df_num.reset_index(drop = True , inplace = True)
df_one = df[onehot_variable].astype('int')
df_one.reset_index(drop = True , inplace = True)
df_final= pd.concat([df_cat, df_num , df_one ],  axis = 1 )
#df_final.index = df_drop['date']


if '#OFCars_categorical_No_car' in df_final.columns:
    df_final.drop('#OFCars_categorical_No_car' , axis = 1 , inplace = True)  


X_train, X_test, y_train, y_test = train_test_split(df_final,y,test_size=0.33, random_state=44) 


'''=================================Logistic Regression ============================================================'''


LR = LogisticRegression()
LR.fit(X_train , y_train)


train_predict = LR.predict(X_train)
y_pred = LR.predict(X_test )
y_score = LR.predict_proba(X_test)


# confusion matrix 
cm_train= confusion_matrix(y_train, train_predict)
cm = confusion_matrix(y_test, y_pred)


print("Accuracy_test", accuracy_score(y_test, y_pred))
print("Accuracy_train", accuracy_score(y_train, train_predict))

   

AUC_curve(LR, X_test , y_test , y_pred , y_score , title = 'Logistic_Regression_AUC ' , path = '')
plot_importance(LR.coef_ , 20 , df_final.columns , 'LR_Feture_importance')



'''================================= Random Forest  ============================================================'''






start = time.time()

# It is not true , I should either use cross validation or ...
# here every time instead of cross validation we use test set assuming that the large dataset 
# the distribution not changes 
def objective_cross(params):
   
    X_train, X_test, y_train, y_test = train_test_split(df_final,y,test_size=0.33, random_state=44) 
    clf = RandomForestClassifier(**params, random_state=0)
    clf.fit(X_train , y_train)
    #results = cross_validate(estimator,X_train, y_train , cv=cv,  return_estimator = True ,  scoring='f1_macro')
   
    # Extract the best score
    #score_index = list(results['test_score']).index(max(results['test_score']))
    #clf = results['estimator'][score_index]  # Here we can rtrain it again on whole withot cross validate set 
    # import pdb
    # pdb.set_trace()
    #clf.fit(X_train,y_train) # no need to train again as it is the best one among the rest already
    #clf.fit(X_train, y_train)
    
    pred= clf.predict(X_test)
    train_pred = clf.predict(X_train)
    train_acc =  accuracy_score (y_train ,train_pred  )
    test_acc = accuracy_score (y_test , pred )
    train_metric = precision_recall_fscore_support(y_train ,train_pred , average = 'binary')
    train_metric+=(train_acc,)
    
    test_metric = precision_recall_fscore_support(y_test , pred , average = 'binary')
    test_metric += (test_acc,)
    loss = 1-test_metric[2]
# testing score
    # Loss must be minimized

    # Dictionary with information for evaluation
    return {'loss':loss , 'params': params, 'status': STATUS_OK , 'model':clf , 'train_score':train_metric , 'test_score': test_metric }

#TODO : ['max_depth']

space = {
    'n_jobs':-1 ,
    'class_weight' : hp.choice('class_weight' , [{0:1,1:i+1} for i in  range(1,100,3)]),
    'max_depth' : hp.choice('max_depth', np.arange(10,100, dtype=int)),
    'n_estimators': hp.choice('n_estimators', np.arange(50,100, dtype=int)),
    'max_features' : hp.choice('max_features' ,  [.25, 'sqrt' , 'log2' ])}

# space = {
#     'n_estimators': hp.choice('n_estimators', np.arange(10,50, dtype=int)),
#     'max_features' : hp.choice('max_features' ,  [.25, 'auto' , 'sqrt' , 'log2' ])}

bayes_trail = Trials()

best_cross = fmin(objective_cross,space,algo=tpe.suggest,trials = bayes_trail, max_evals=100)

def tune_result (trials) :
    df = pd.DataFrame()
    df['loss'] = trials.losses()
    df['train_metric'] = [trials.trials[i]['result']['train_score']for i in range(100)]
    df['test_metric'] = [trials.trials[i]['result']['test_score']for i in range(100)]
    df['params'] = [trials.trials[i]['result']['model'].get_params(deep = True) for i in range(100)]
    return df


RF_tune = tune_result (bayes_trail) 

RF_tune.to_pickle('./RF_vm/RF_tune.pkl')


end = time.time() - start



f = open('bayes_RF_Trials.pkl', 'wb')
pickle.dump(bayes_trail, f , protocol = 4 )
f.close()

g = open('best_cross_RL.pckl', 'wb')
pickle.dump('best_cross_R.pkl', g)
g.close()


RF = RandomForestClassifier(n_estimators = 57 , max_features = .25 , max_depth = 30 , class_weight = {0:1,1:10} )
 

train_predict = RF.predict(X_train)
y_pred = RF.predict(X_test)
y_score = RF.predict_proba(X_test)



cm_train= confusion_matrix(y_train, train_predict)
cm = confusion_matrix(y_test, y_pred)


print("Accuracy_test", accuracy_score(y_test, y_pred))
print("Accuracy_train", accuracy_score(y_train, train_predict))

AUC_curve(RF , X_test, y_test , y_pred , y_score , title = 'Random_forest_AUC_Curve' , path = '')
plot_importance (RF.feature_importances_ , 20 , df_final.columns , title = 'Randome_Forest_feature_importance ' ,  path = ''  )





'''================================= XGBoost  ============================================================'''


class HPOpt(object):
    # import pdb
    # pdb.set_trace()
    #import pdb ; pdb.set_trace()

    def __init__(self, X, y):
        self.X = X
        self.y = y 
        
        


    def process(self, fn_name, space, trials, algo, max_evals):
        fn = getattr(self, fn_name)
        try:
            result = fmin(fn=fn, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials

    def xgb_reg(self, para):
        clf = xgb.XGBClassifier(**para['reg_params'])
        return self.train_reg(clf, para)

#    def lgb_reg(self, para):
#        reg = lgb.LGBMRegressor(**para['reg_params'])
#        return self.train_reg(reg, para)
#
#    def ctb_reg(self, para):
#        reg = ctb.CatBoostRegressor(**para['reg_params'])
#        return self.train_reg(reg, para)

    def train_reg(self, clf, para):
        X_train, X_test, y_train, y_test = train_test_split(self.X ,self.y ,test_size=0.33, random_state=44) 
        
        sample_weight = np.where(y_train ==1 , para['reg_params']['class_weight'] , 1)
        clf.set_params(sample_weight = sample_weight)
#        results = cross_validate(reg, self.x_train, self.y_train, cv=5, return_estimator = True,  scoring='f1_macro')
#        index = list(results['test_score']).index(max(results['test_score']))
#        clf = results['estimator'][index]
        clf.fit(X_train, y_train ,
                **para['fit_params'])
        
        pred = clf.predict(X_test)
        train_pred = clf.predict (X_train)
        train_acc = accuracy_score(y_train , train_pred)
        test_acc = accuracy_score(y_test , pred)
        train_metrics = precision_recall_fscore_support(y_train , train_pred , average = 'binary')
        train_metrics +=(train_acc,)
        test_metrics = precision_recall_fscore_support(y_test , pred ,  average = 'binary')
        test_metrics +=(test_acc,)
        
        loss = 1 - test_metrics[2]

        

       
       
        return {'loss': loss, 'status': STATUS_OK , 'model':clf , 'test_metrics':test_metrics ,  'train_metrics':train_metrics}





   
xgb_reg_params = {
        
    'objective' : 'binary:logistic',
    'n_jobs' : -1 , 
    'class_weight' : hp.choice ('class_weight' , [i for i in range(1,100,3)]),
    'learning_rate':    hp.choice('learning_rate',    np.arange(0.05, 0.3, 0.05)),
    'max_depth':        hp.choice('max_depth',        np.arange(20, 100, 1, dtype=int)),
    'colsample_bytree': hp.choice('colsample_bytree', np.arange(0.3, 0.8, 0.1)),
    'subsample':        hp.uniform('subsample', 0.8, 1),
    'n_estimators': hp.choice('n_estimators', np.arange(50,100, dtype=int))}

xgb_fit_params = {
    'verbose': True
}
xgb_para = dict()
xgb_para['reg_params'] = xgb_reg_params
xgb_para['fit_params'] = xgb_fit_params
#xgb_para['loss_func' ] = lambda y, pred: np.sqrt(mean_squared_error(y, pred))

   
obj = HPOpt(df_final , y)
xgb_trails = Trials()
xgb_opt = obj.process(fn_name = 'xgb_reg',space=xgb_para, trials=xgb_trails, algo=tpe.suggest, max_evals=100)  



   
def tune_result (trials) :
    df = pd.DataFrame()
    df['loss'] = trials.losses()
    df['train_metric'] = [trials.trials[i]['result']['train_metrics']for i in range(100)]
    df['test_metric'] = [trials.trials[i]['result']['test_metrics']for i in range(100)]
    df['params'] = [trials.trials[i]['result']['model'].get_xgb_params() for i in range(100)]
    return df


df_tune = tune_result(xgb_trails)


f = open('xgb_opt.pckl', 'wb')
pickle.dump(xgb_opt, f)
f.close()


g = open('xgb_trails.pkl', 'wb')
obj = pickle.dump(xgb_trails ,  g)
g.close()



'''================================= SVC ; it is very slow for now just skipped   ============================================================'''






df = pd.read_pickle('C:/Users/gn8525/In_market_Timing/data_state/data_Michigan.pkl')  # it is in data_state
df= df[~(df['#OFCars_categorical']=='No_car')].copy(True) # Dropping the 
            
            
positive_sample =y.sum()     

neg_sample_needed = positive_sample / .6


neg_sample = df[df['label']==0].sample(int(neg_sample_needed) , replace = False)

df_svm = pd.concat([neg_sample ,df[df['label']==1]] , axis = 0 , ignore_index = True)

from sklearn.utils import shuffle 
df_svm = shuffle(df_svm)


y = df_svm['label'].astype('int')
df_svm.drop('label' , axis = 1 , inplace = True) # df.shape = (2259585, 73)
df_svm.drop(feature_drop , axis = 1 , inplace = True) # (2259585, 64)


assert  'label' in df_svm.columns , 'label is not in the feature space'    


df_cat = pd.get_dummies(df_svm[cat_variable]) 
df_cat.reset_index(drop = True , inplace = True)
df_num = pd.DataFrame(StandardScaler().fit_transform(df_svm[num_variable]), columns = num_variable)
df_num.reset_index(drop = True , inplace = True)
df_one = df_svm[onehot_variable].astype('int')
df_one.reset_index(drop = True , inplace = True)
df_final= pd.concat([df_cat, df_num , df_one ],  axis = 1 )

if '#OFCars_categorical_No_car' in df_final.columns:
    df_final.drop('#OFCars_categorical_No_car' , axis = 1 , inplace = True)  





class sv_c(object):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y 
        
    def train(self, para ):
        X_train, X_test, y_train, y_test = train_test_split(self.X ,self.y ,test_size=0.33, random_state=44) 
        clf = SVC(**para)
        clf.fit(X_train , y_train)
        pred = clf.predict(X_test)
        train_pred = clf.predict (X_train)
        train_acc = accuracy_score(y_train , train_pred)
        test_acc = accuracy_score(y_test , pred)
        train_metrics = precision_recall_fscore_support( y_train , train_pred , average = 'binary')
        train_metrics +=(train_acc,)
        test_metrics = precision_recall_fscore_support(y_test , pred ,  average = 'binary')
        test_metrics +=(test_acc,)
        loss = 1 - test_metrics[2]
        
        return {'loss': loss, 'status': STATUS_OK , 'model':clf , 'test_metrics':test_metrics ,  'train_metrics':train_metrics}

    def process(self, space, trials, algo, max_evals ):
        
        try:
            result = fmin(fn=self.train, space=space, algo=algo, max_evals=max_evals, trials=trials)
        except Exception as e:
            return {'status': STATUS_FAIL,
                    'exception': str(e)}
        return result, trials
         
space = {'kernel':hp.choice('kernel' , ['rbf' , 'poly']) ,
          'degree' : hp.choice('degree' ,  [2,3,4,5]) , 
          'C' : hp.choice('C' , [.01, .1 , .5 , 1 , 3 , 5 , 10 , 20 , 50 , 100]), 
          'class_weight' : hp.choice ('class_weight' , [{0:1, 1:i} for i in range(1,10,1)])}
                
SVC_obj = sv_c(df_final , y)
SVC_trails = Trials()
SVC_opt = SVC_obj.process(space=space, trials=SVC_trails, algo=tpe.suggest, max_evals=100)  

 
'''================================= SVC ; it is very slow for now just skipped   ============================================================'''

#df = pd.read_pickle('data_state_3/data_Michigan.pkl')
rf_tune = pd.read_pickle('./RF_vm/RF_tune.pkl')
xgb_tune = pd.read_pickle('./XGB_vm/df_tune.pkl')



xgb_best_index = np.argmax(xgb_tune['loss'])
rf_best_index = np.argmax(rf_tune['loss'])

params = xgb_tune['params'][xgb_best_index]

clf = xgb.XGBClassifier()

sample_weight = np.where(y_train ==1 , clf.get_params()['class_weight'] , 1)
clf.set_params(sample_weight = sample_weight)

clf.fit(X_train , y_train)

