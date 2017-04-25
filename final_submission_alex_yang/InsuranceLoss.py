# This tells matplotlib not to try opening a new window for each plot.


import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import f_regression

from sklearn import metrics

#%matplotlib inline

#SVM
from sklearn.svm import SVR

#Library for pca decomposition
from sklearn.decomposition import PCA

from sklearn.preprocessing import Imputer, StandardScaler,\
    Binarizer, LabelEncoder, PolynomialFeatures


tot_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

#preprocessing

column_names=list(tot_data.columns)
#Set continuous and coategory columns
category_cols=[]
continuous_cols=[]

for name in column_names:
    if "cat" in name:
        category_cols.append(name)
    elif 'cont' in name:
        continuous_cols.append(name)

#combine test and train data to turn data into dummies
all_data=pd.merge(tot_data,test_data,how='outer',indicator=True)
all_data.set_index('id',inplace=True)

#find binary and multinomial columns
binary_cols=[]
multinomial_cols=[]

for feature in category_cols:
    #print feature
    if len(all_data[feature].unique())==2:
        binary_cols.append(feature)
    elif len(all_data[feature].unique())>2:
        multinomial_cols.append(feature)

# binarize binary columns
all_binary=pd.get_dummies(all_data[binary_cols],drop_first=True)
all_binary.columns=binary_cols

#turn multinomial columns into
all_multi_cat=pd.get_dummies(all_data[multinomial_cols],drop_first=True)
multi_cat_cols=list(all_multi_cat.columns)

#CREATE new version of all_data
all_data_2=pd.concat([all_binary,all_multi_cat,all_data[continuous_cols]
						,all_data['loss'],all_data['_merge']],axis=1)

#split the data in test and train again via the '_merge' column that was in from the merge method

tot_data=all_data_2[all_data_2['_merge']=='left_only']
test_data=all_data_2[all_data_2['_merge']=='right_only']

#Maintain memory
del all_data, all_data_2

#drop last two columns in test_data

del test_data['loss']
del test_data['_merge']

#New category columns
category_cols=binary_cols+multi_cat_cols

#Take log of insurance loss
tot_data['log_loss']=tot_data['loss'].apply(np.log)

#split training data into train, dev and test_dev
tot_data_length=tot_data.shape[0]

num_test=int(round(tot_data_length*.2,0))
test_dev=tot_data.loc[:num_test/2,:]

dev=tot_data.loc[num_test/2:num_test,:]
train=tot_data.loc[num_test:,:]

column_names=list(tot_data.columns)

#Finish naming columns
data=column_names[:-3]
insurance_loss=column_names[-3]
log_loss=column_names[-1]

#EDA
def continuous_hist():
    plt.close('all')
    fig = plt.figure()
    fig.set_figheight(28)
    fig.set_figwidth(16)
    #plt.title('Histograms of Continuous Features', fontsize=18)
    axes=[]
    count=1
    for col in continuous_cols:
        axes.append(fig.add_subplot(14,2,count))
        #plt.set_figheight(5)
        plt.title("Histogram of "+col)
        plt.hist(train[col])
        count+=1
    plt.tight_layout()
    plt.show()

def plot_loss():
    fig=plt.figure()
    fig.set_figheight(15)
    fig.set_figwidth(15)
    plt.suptitle("Histograms",fontsize=24)
    plt.subplot(2,1,1)
    plt.hist(train[insurance_loss], bins =50)
    plt.title('Insurance Loss Histogram',fontsize=14)
    plt.subplot(2,1,2)
    plt.title('Log Insurance Loss Histogram',fontsize=14)
    plt.hist(train[log_loss],bins=50)
    plt.show()

def continuous_scatter():
    fig = plt.figure()
    fig.set_figheight(28)
    fig.set_figwidth(16)

    axes=[]
    count=1
    for col in continuous_cols:
        axes.append(fig.add_subplot(8,2,count))
        axes[-1].scatter(train[col],train[insurance_loss],s=.5)
        plt.title("Scatterplot "+col)
        plt.xlabel(col)
        plt.ylabel('Insurance loss')
        count+=1
    plt.tight_layout()

def continuous_scatter_log():
    fig = plt.figure()
    fig.set_figheight(28)
    fig.set_figwidth(16)

    axes=[]
    count=1
    for col in continuous_cols:
        axes.append(fig.add_subplot(8,2,count))
        axes[-1].scatter(train[col],train[log_loss],s=.5)
        plt.title("Scatterplot "+col)
        plt.xlabel(col)
        plt.ylabel('log Insurance loss')
        count+=1
    plt.tight_layout()
#Basline linear model on continuous data
def base_linear():
    lm=LinearRegression()
    lm.fit(train[continuous_cols],train[insurance_loss])
    lm.score(dev[continuous_cols],dev[insurance_loss])
    print 'Train R-squared: {:.3}'.format(lm.score(train[continuous_cols],train[insurance_loss]))
    print 'dev R-squared: {:.3}'.format(lm.score(dev[continuous_cols],dev[insurance_loss]))
    baseline_predictions=lm.predict(dev[continuous_cols])
    residuals=dev[insurance_loss]-baseline_predictions
    plt.xlabel('Predicted Value')
    plt.ylabel('Residuals')
    plt.title('Linear Regression on continuous variables')
    plt.scatter(baseline_predictions,residuals,  color='blue')

def base_linear_log():
    lm=LinearRegression()
    lm.fit(train[continuous_cols],train[log_loss])
    lm.score(dev[continuous_cols],dev[log_loss])
    print 'Train R-squared: {:.3}'.format(lm.score(train[continuous_cols],train[log_loss]))
    print 'dev R-squared: {:.3}'.format(lm.score(dev[continuous_cols],dev[log_loss]))
    baseline_predictions=lm.predict(dev[continuous_cols])
    residuals=dev[insurance_loss]-baseline_predictions
    plt.xlabel('Predicted Value')
    plt.ylabel('Residuals')
    plt.title('Linear Regression on continuous variables')
    plt.scatter(baseline_predictions,residuals,  color='blue')

def best_features(n_features,X_train,y_train,X_test,y_test):
    feature_importance = f_regression(train[continuous_cols], train[insurance_loss])[0]
    idx = np.argsort(-feature_importance)[:n_features]
    lr = LinearRegression()
    lr.fit(X_train.iloc[:, idx], y_train)
    return lr.score(X_test.iloc[:, idx], y_test)

def best_features_linear_base():

    out = [best_features(i,train[continuous_cols],train[insurance_loss],
        dev[continuous_cols],dev[insurance_loss]) for i in range(1, 14)]

    fig=plt.figure()
    plt.title("Feature Importance")
    plt.xlabel("Number of Features")
    plt.ylabel("R Squared")
    plt.plot(out)
    plt.show()

def best_components(n_features,X_train,y_train,X_test,y_test):
    pca = PCA(n_components=n_features)
    X_transformed = pca.fit_transform(X_train)
    lr = LinearRegression()
    lr.fit(X_transformed, y_train)
    return lr.score(pca.transform(X_test), y_test)

def best_components_linear_model():
    out = [best_components(i,
                      train[continuous_cols],
                       train[insurance_loss],
                       dev[continuous_cols],
                       dev[insurance_loss]) for i in range(1, 15)]

    fig=plt.figure()
    plt.title("Component Importance")
    plt.xlabel("Number of Components")
    plt.ylabel("R Squared")
    plt.plot(out)
    plt.show()

def print_cum_var_ratio():
    pca=PCA()
    pca.fit(train[data])
    plt.plot(np.cumsum(pca.explained_variance_ratio_[:400]))
    plt.title('Explained Variance Ratio for PCA on all features')

def fast_best_components():
    pca=PCA(n_components=501)
    x_train=pca.fit_transform(train[data])
    x_dev=pca.transform(dev[data])
    lm=LinearRegression()
    out=[]
    for i in xrange(1,501,10):
        lm.fit(x_train[:,:i],train[insurance_loss])
        score=lm.score(x_dev[:,:i],dev[insurance_loss])
        out.append((i,score))
    plt.title('R-Squared of Linear Regressionon Development Set v PCA components')
    plt.ylabel('R-Squared')
    plt.xlabel('N Components')
    plt.plot(*zip(*out))

def large_linear_plot():
    pca_pipe=Pipeline(steps=[('decomp',PCA(n_components=400)),('lm',LinearRegression())])
    pca_pipe.fit(train[data],train[log_loss])
    train_score=pca_pipe.score(train[data],train[log_loss])
    dev_score=pca_pipe.score(dev[data],dev[log_loss])
    dev_predict=pca_pipe.predict(dev[data])
    dev_residuals=dev_predict-dev[log_loss]
    print 'Train R-squared: {:.3}'.format(train_score)
    print 'Development set R-squared: {:.3}'.format(dev_score)
    plt.scatter(dev_predict,dev_residuals)
    plt.title('Plot Residuals v Predicted')
    plt.ylabel('Residuals')
    plt.xlabel('Predicted')
    plt.show()

#large_linear_plot()