# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 22:22:00 2021

@author: arafa
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('eda_data.csv')


# what we should do

# 1. choose relevant columns
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided',
             'job_state','same_state','age','python_yn','spark','aws','excel','job_simp','seniority','desc_len']]

# 2. get dummy data
df_dum = pd.get_dummies(df_model)

# 3. train test split
from sklearn.model_selection import train_test_split

X = df_dum.drop('avg_salary', axis=1)
y = df_dum['avg_salary'].values # .values makes y to be an array instead of Series. this is because arrays are recomended to be used in models

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# models to test
# 1. multiple linear regression

#statsmodel package
import statsmodels.api as sm

X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
print(model.fit().summary())

# the model_summary
# 1. R_squared: 0.708 # this is good, tells that our model explains arround 70% of the variations in the data
# 2. p>|t| is what we want to focus on, so a P-value less than 0.05 means it's significant in our model, 
# so 'Rating is not because it has a P_value of 0.063. 'num_comp' is significant because it has a P_value of 0.010 which is < 0.05
# so for each additional competitor it looks like that we would add around 2.2505$ (coef). 
# Python is very relavant, but some other skills are not (spark, aws, excel)
# seems like you would earn like 13000 more if you are working in public companeis
# you will get more if you are working in Biotech & Pharmaceuticals and Software indeustry
# you will get paid more if you are working for a company that has a 'Revenue_$5 to $10 million'
# more details in the summary table**


from sklearn.linear_model import LinearRegression, Lasso

# creates small train vaildations sets, and apply it to the model, 
# this creates a sense of how the model will perform in reality
from sklearn.model_selection import cross_val_score

mlr_model = LinearRegression()
mlr_model.fit(X_train, y_train)

# 'neg_mean_absolute_error' is a good choise, because its the most representive this will show how far on average off our general prediction
# so if we are on average off by 21, that mean we are on average off by 21000$  
# a multiple linear regression is actually abit difficult to get a good value from that because there is such a limited data 
print(np.mean(cross_val_score(mlr_model,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))) # around 20000 off

# 2. lasso regression

# a different regression model, like lasso regression which normalizes those values and should be better for our model 

# in lasso model we use alpha (normalization term). alpha can (alpha >= 0). 
# if its 0 our model will be exactly the same as OLS multiple linear regrassion
# as we increase alpha it increases the amount the data that the data is smoothed

# without passing any alpha value, the model is less accurate than the muliple regression model
l_model = Lasso(alpha = 0.13) 
l_model.fit(X_train,y_train)
print(np.mean(cross_val_score(l_model,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))) 


# we try different values for alpha and see how our model performs

alpha = []
error = []

for i in range(100):
    alpha.append(i/100)
    l_model = Lasso(alpha = (i/100))
    error.append(np.mean(cross_val_score(l_model,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3)))
    
plt.plot(alpha,error) # from the graph, the best alpha to use it around 0.2

# combine alpha and error into a tuple, so we easily can get the error values corresponding to each alpha
err = tuple(zip(alpha,error))

# convert the err to a dataframe
err_df = pd.DataFrame(err, columns = ['alpha', 'error'])

filt = (err_df['error'] == max(err_df['error']))
print(err_df.loc[filt])

# so we can see an alpha of 0.13 is giving us the best error term, so we can use alpha = 0.13 to  build our model 

#------------------------ lasso model ends here----------------

# 3. random forrest

# here I expect raandom forrest to perform very well, especially because its kind of a tree based decision 
# process and there are a lot of 0, 1 values 
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor()
print(np.mean(cross_val_score(rf_model,X_train,y_train, scoring = 'neg_mean_absolute_error', cv= 3))) 

# we can see that Random Forest model performing very well in compare to the other models
# the error with RF is -14.871508186774685 and this was without tunning the model (using just defaults)



# To do after
# 1. tune models GridsearchCV

# now lets tune the model using grid search
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(rf_model,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

print(gs.best_score_)
print(gs.best_estimator_)

# the model is improved with very small amount

# 2. test ensembles
tpred_mlr_model = mlr_model.predict(X_test)
tpred_l_model = l_model.predict(X_test)
tpred_rf_model = gs.best_estimator_.predict(X_test)

# evaluate the models by comparing predictions with actual values
from sklearn.metrics import mean_absolute_error as mae
mae(y_test,tpred_mlr_model)
mae(y_test,tpred_l_model)
mae(y_test,tpred_rf_model)


# sometimes combining 2 models can be beneficial for the job
mae(y_test, ((tpred_l_model + tpred_rf_model)/2))