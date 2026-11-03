##Classification
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
data =pd.read_csv("bank.csv",sep=";")
data['y'] =data['y'].map({"no":0,"yes":1})
data =pd.get_dummies(data,columns=data.select_dtypes(include='object').columns,drop_first=True,dtype=int)
model =LogisticRegression()
X_train,X_test,y_train,y_test =train_test_split(data.drop('y',axis=1).values,data['y'].values,
                                                test_size=0.2,random_state=1)
scaler =MinMaxScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
data.head()



data['y'].value_counts()


from  sklearn.metrics import classification_report
y_prediction =model.predict(X_test)
print(classification_report(y_test,y_prediction))


data.corr(numeric_only =True)


data.isnull().sum()


data.info()

##Classification1
import pandas  as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model  import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
data =pd.read_csv("bank.csv",sep=";")
data['y'] =data['y'].map({"no":0,"yes":1})
data =pd.get_dummies(data,columns=data.select_dtypes(include='object').columns,drop_first=True,dtype=int)
model =LogisticRegression()
X_train,X_test,y_train,y_test =train_test_split(data.drop('y',axis=1).values,data['y'].values,
                                                test_size=0.2,random_state=1)
scaler =MinMaxScaler()
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
data.head()


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
sampler =Pipeline(steps =[('over',SMOTE(sampling_strategy=0.7)),
                         ("under",RandomUnderSampler(sampling_strategy=0.8) )])
X_train_sampled,y_train_sampled =sampler.fit_resample(X_train,y_train)


from collections import Counter
print(Counter(y_train))
print(Counter(y_train_sampled))


data['y'].value_counts()


from  sklearn.metrics import classification_report
y_prediction =model.predict(X_test)
print(classification_report(y_test,y_prediction))


model.fit(X_train_sampled,y_train_sampled)
from  sklearn.metrics import classification_report
y_prediction =model.predict(X_test)
print(classification_report(y_test,y_prediction))


from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
positive_probability =model.predict_proba(X_test)[:,1]
fpr,tpr,thresholds =roc_curve(y_test,positive_probability)
plt.plot(fpr,tpr)


data.corr(numeric_only =True)


data.isnull().sum()


data.info()


##Clustering
#მოცემულ მონაცემებზე დაყრდნობით  გამოიყენეთ  DecisionTreeClassifier მეთოდი  რომელიც გამოიძახება
#sklearn.tree  ბიბლიოთეკიდან და  დაატრეე
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
data =pd.read_csv("income.csv")
data.drop('Name',axis=1,inplace=True)
data.head()


from sklearn.cluster  import KMeans
model =KMeans(n_clusters=3)
data =scaler.fit_transform(data) #ამის შემდეგ  data  არის უკვე მატრიცა  და არა dataframe
clusters =model.fit_predict(data)


data


new_data =[[33,70000]]
scaled_version =scaler.fit_transform(new_data)
plt.scatter(x =data[:,0],y=data[:,1],c=clusters) #c   აღნიშნავს ფერებს
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c='black',s=90) #s  არის ზომა, c  არის ფერი
plt.scatter(x =scaled_version[:,0],y=scaled_version[:,1],s=80) #c   აღნიშნავს ფერებს


new_data =[[33,70000]]
model.predict(scaler.fit_transform(new_data))

##PCA
import pandas as pd
data =pd.read_csv("/content/CarPrice_Assignment.csv")
data =data.select_dtypes(include="number")
data.drop(['car_ID','symboling'],axis=1,inplace=True)
data.head()


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
y =data['price'].values
X =data.drop('price',axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)
pca=PCA(n_components=5)
X_train =pca.fit_transform(X_train)
X_test =pca.transform(X_test)
print(pca.components_[0])
print(sum(pca.explained_variance_ratio_))
#ცვალეთ n_components  რაოდენობა  და ააგეთ რეგრესიული მოდელები შემცირებულ მონაცემებზე, აიღეთ საუკეთესო ვარიანტი(გამოიყენეთ რაც გინდათ )


from yellowbrick.features import PCA
y =data['price']
X =data.drop('price',axis=1)
visualizer = PCA(scale=True, proj_features=True, projection=2)
visualizer.fit_transform(X, y)
visualizer.show()


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
y =data['price'].values
X =data.drop('price',axis=1).values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)
X_train =scaler.fit_transform(X_train)
X_test =scaler.transform(X_test)
model =LinearRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))


data.shape


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,12))
sns.heatmap(data=data.corr(),annot=True)


data.describe().T


data.info()


data.isnull().sum()


##Multivariable Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import pandas as pd
data =pd.read_csv("50_Startups.csv")
data =pd.get_dummies(data=data,columns=["State"],drop_first=True,dtype=int)
y =data['Profit'].values
X =data[["R&D Spend","Marketing Spend"]].values
X_train,X_test,y_train,y_test =train_test_split(X,y,test_size=0.2,random_state=1)
model =LinearRegression()
model.fit(X_train,y_train)
print(model.score(X_test,y_test))
print(model.predict([[165349.20,471784.10]]))

data.head()


# import statsmodels.api as sm
# y =data['Profit']
# X =data.drop('Profit',axis=1)
# X = sm.add_constant(X)
# model =sm.OLS(endog=y,exog=X).fit()
# print(model.summary())


data.corr(numeric_only=True)


# data.info()


# data.isnull().sum()