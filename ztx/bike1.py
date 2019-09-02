import pandas as pd

train_df=pd.read_csv('D:/bike/train.csv')
train_df.head()
train_df.describe()
import seaborn as sns
import matplotlib.pyplot as plt
cat_names=['season', 'holiday', 'workingday', 'weather']
i=0
for name in cat_names:
    i=i+1
    plt.subplot(2,2,i)
    sns.countplot(name,data=train_df) 
    plt.show()

cont_names=['temp','atemp','humidity','windspeed']
i=0
for name in cont_names:
    i=i+1
    plt.subplot(2,2,i)
    sns.boxplot(name,data=train_df) 
    plt.show()
from datetime import datetime

train_df['datetime']=train_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
i=1
for name_1 in cont_names:
    j=cont_names.index(name_1)
    while(j<len(cont_names)-1):
        plt.subplot(6,1,i)
        plt.title(name_1+' vs '+cont_names[j+1])
        sns.jointplot(x=name_1,y=cont_names[j+1],data=train_df) 
        j=j+1
        i=i+1
        plt.show()

type(train_df['datetime'][0])
from datetime import datetime
new_df=train_df
new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
#new_df['year']=new_df['datetime'].apply(lambda x:x.year)
#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)
new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))

print (new_df.head())
x='2012-11-30 14:00:00'
n=datetime.strptime(x,'%Y-%m-%d %H:%M:%S')
n.month
n.day
n.year
sns.swarmplot(x='hour',y='temp',data=new_df,hue='season')
plt.show()
new_df.cov()
sns.heatmap(new_df.corr())
plt.show()
new_df.corr()
cat_names=['season', 'holiday', 'workingday', 'weather']
i=1
for name in cat_names:
    plt.subplot(2,2,i)
    sns.barplot(x=name,y='count',data=new_df,estimator=sum)
    i=i+1
    plt.show()
final_df=new_df.drop(['datetime','season','holiday','atemp','holiday','windspeed','casual','registered','mnth+day','day'], axis=1)
final_df.head()

#adding dummy varibles
weather_df=pd.get_dummies(new_df['weather'],prefix='w')
#year_df=pd.get_dummies(new_df['year'],prefix='y',drop_first=True)
month_df=pd.get_dummies(new_df['month'],prefix='m',drop_first=True)
hour_df=pd.get_dummies(new_df['hour'],prefix='h',drop_first=True)
final_df=final_df.join(weather_df)
#final_df=final_df.join(year_df)
final_df=final_df.join(month_df)                     
final_df=final_df.join(hour_df)
                     
final_df.head()
final_df.columns
model_df=final_df.drop(['workingday','month','hour','weather'],axis=1)
model_df.head()
X=model_df.iloc[:,model_df.columns!='count'].values


Y=model_df.iloc[:,2].values

print ('oye',X.shape)
#splitting the data into training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)

# k-fold cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=lr,X=X_train,y=Y_train,scoring='r2',cv=10)
print (accuracies)
print (accuracies.mean())
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=300,random_state=0)
rf.fit(X,Y)





accuracies=cross_val_score(estimator=rf,X=X_train,y=Y_train,scoring='r2',cv=5)
print (accuracies)
print (accuracies.mean())
X.shape
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X_temp=sc_X.fit_transform(X)
y_temp=sc_Y.fit_transform(Y.reshape(-1,1))

from sklearn.svm import SVR
svr=SVR(kernel='rbf')
svr.fit(X_temp,y_temp)

accuracies=cross_val_score(estimator=svr,X=X_temp,y=y_temp,scoring='r2',cv=5)
print (accuracies)
print (accuracies.mean())
from sklearn.tree import DecisionTreeRegressor
dtr=DecisionTreeRegressor(random_state=0)
dtr.fit(X_train,Y_train)



accuracies=cross_val_score(estimator=dtr,X=X_train,y=Y_train,scoring='r2',cv=5)
print (accuracies)
print (accuracies.mean())
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_Y=StandardScaler()
X_temp=sc_X.fit_transform(X_train)
y_temp=sc_Y.fit_transform(Y_train.reshape(-1,1))


parameters=[{'C':[1,10,100,1000],'kernel':['linear']},
            {'C':[1,10,100,1000],'kernel':['rbf'],'gamma':[0.5,0.1,0.01,0.01,0.001]}
            ]

grid_search= GridSearchCV(estimator=svr, param_grid=parameters, cv=5,n_jobs=-1)
import numpy as np
def rmsle(y, y_):
	#np.nan_to_num replaces nan with zero and inf with finite numbers
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
	#taking squares
    calc = (log1 - log2) ** 2
	#taking mean and then square
    return np.sqrt(np.mean(calc))
test_df=pd.read_csv('../input/test.csv')
test_df['datetime']=test_df['datetime'].apply(lambda x:datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
new_df=test_df

new_df['month']=new_df['datetime'].apply(lambda x:x.month)
new_df['hour']=new_df['datetime'].apply(lambda x:x.hour)
new_df['day']=new_df['datetime'].apply(lambda x:x.day)
#new_df['year']=new_df['datetime'].apply(lambda x:x.year)
#new_df['weekday_flag']=new_df['datetime'].apply(weekday_flag)
#new_df['mnth+day']=new_df['datetime'].apply(lambda x:str(x.month)+'_'+str(x.day))

print (new_df.head())
test_df=new_df.drop(['datetime','season','holiday','atemp','holiday','windspeed','day'], axis=1)
test_df.head()
weather_df=pd.get_dummies(test_df['weather'],prefix='w',drop_first=True)
#yr_df=pd.get_dummies(test_df['year'],prefix='y',drop_first=True)
month_df=pd.get_dummies(test_df['month'],prefix='m',drop_first=True)
hour_df=pd.get_dummies(test_df['hour'],prefix='h',drop_first=True)
test_df=test_df.join(weather_df)
test_df=test_df.join(yr_df)
test_df=test_df.join(month_df)                     
test_df=test_df.join(hour_df)
                     
test_df.head()
test_df=test_df.drop(['workingday','month','hour','weather'],axis=1)

X_test=test_df.iloc[:,:].values
X_test.shape
y_output=rf.predict(X_test)
y_output
op=pd.DataFrame({'count':y_output})
op.to_csv('sub.csv')
def rmsle(y, y_):
	#np.nan_to_num replaces nan with zero and inf with finite numbers
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in y]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in y_]))
	#taking squares
    calc = (log1 - log2) ** 2
	#taking mean and then square
    return np.sqrt(np.mean(calc))
