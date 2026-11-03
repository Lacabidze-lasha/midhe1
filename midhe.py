##Classification
# Block 1: Import the libraries and load the marketing dataset for a binary classification task.
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

# Block 2: Check how many samples belong to each class in the target variable.
data['y'].value_counts()

# Block 3: Evaluate the trained model using precision, recall, and F1-score.
from  sklearn.metrics import classification_report
y_prediction =model.predict(X_test)
print(classification_report(y_test,y_prediction))

# Block 4: Inspect the correlation between numerical features in the dataset.
data.corr(numeric_only =True)

# Block 5: Detect missing values in the dataset before further analysis.
data.isnull().sum()

# Block 6: Print a summary of each column's type and non-null count.
data.info()

##Classification1
# Block 7: Repeat the classification flow and then apply resampling to balance the imbalanced training set.
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

# Block 8: Use SMOTE and random undersampling to rebalance the positive and negative classes in training data.
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
sampler =Pipeline(steps =[('over',SMOTE(sampling_strategy=0.7)),
                         ("under",RandomUnderSampler(sampling_strategy=0.8) )])
X_train_sampled,y_train_sampled =sampler.fit_resample(X_train,y_train)

# Block 9: Compare the class distribution before and after resampling.
from collections import Counter
print(Counter(y_train))
print(Counter(y_train_sampled))

# Block 10: Inspect the target distribution after the preprocessing pipeline.
data['y'].value_counts()

# Block 11: Measure classification performance without retraining on the resampled data.
from  sklearn.metrics import classification_report
y_prediction =model.predict(X_test)
print(classification_report(y_test,y_prediction))

# Block 12: Retrain the model on the balanced dataset and check if performance improved.
model.fit(X_train_sampled,y_train_sampled)
from  sklearn.metrics import classification_report
y_prediction =model.predict(X_test)
print(classification_report(y_test,y_prediction))

# Block 13: Plot the ROC curve using the model's predicted probabilities to visualize classification quality.
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
positive_probability =model.predict_proba(X_test)[:,1]
fpr,tpr,thresholds =roc_curve(y_test,positive_probability)
plt.plot(fpr,tpr)

# Block 14: Check correlations again for the processed dataset.
data.corr(numeric_only =True)

# Block 15: Verify whether any missing values remain in the data.
data.isnull().sum()

# Block 16: Print the dataset schema and column metadata for a final inspection.
data.info()

##Clustering
# Block 17: Prepare the clustering example by scaling numeric features and dropping the non-numeric name column.
#მოცემულ მონაცემებზე დაყრდნობით  გამოიყენეთ  DecisionTreeClassifier მეთოდი  რომელიც გამოიძახება
#sklearn.tree  ბიბліიოთეკიდან და  დაატრეე
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
data =pd.read_csv("income.csv")
data.drop('Name',axis=1,inplace=True)
data.head()

# Block 18: Fit a KMeans model to cluster the standardized data into three groups.
from sklearn.cluster  import KMeans
model =KMeans(n_clusters=3)
data =scaler.fit_transform(data) #ამის შემდეგ  data  არის უკვე მატრიცა  და არა dataframe
clusters =model.fit_predict(data)

# Block 19: Display the transformed dataset after standardization for clustering.
data

# Block 20: Visualize clusters and their centroids on a 2D plot, then predict the cluster for a new record.
new_data =[[33,70000]]
scaled_version =scaler.fit_transform(new_data)
plt.scatter(x =data[:,0],y=data[:,1],c=clusters) #c   აღნიშნავს ფერებს
plt.scatter(model.cluster_centers_[:,0],model.cluster_centers_[:,1],c='black',s=90) #s  არის ზომა, c  არის ფერი
plt.scatter(x =scaled_version[:,0],y=scaled_version[:,1],s=80) #c   აღნიშნავს ფერებს

new_data =[[33,70000]]
model.predict(scaler.fit_transform(new_data))

##PCA
# Block 21: Load the car-price dataset and keep only numeric columns before running PCA.
import pandas as pd
data =pd.read_csv("/content/CarPrice_Assignment.csv")
data =data.select_dtypes(include="number")
data.drop(['car_ID','symboling'],axis=1,inplace=True)
data.head()

# Block 22: Split the dataset, scale it, and reduce dimensionality using PCA before fitting a regression model.
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

# Block 23: Visualize PCA output using Yellowbrick to inspect the projected feature space.
from yellowbrick.features import PCA
y =data['price']
X =data.drop('price',axis=1)
visualizer = PCA(scale=True, proj_features=True, projection=2)
visualizer.fit_transform(X, y)
visualizer.show()

# Block 24: Train a linear regression model on the scaled feature set and compute the test score.
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

# Block 25: Check the shape of the dataset after feature selection.
data.shape

# Block 26: Plot a correlation heatmap to understand relationships between numeric variables.
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(10,12))
sns.heatmap(data=data.corr(),annot=True)

# Block 27: Show summary statistics for each column in the dataset.
data.describe().T

# Block 28: Display the dataset schema and column information.
data.info()

# Block 29: Count missing values as a final data-quality check.
data.isnull().sum()

##Multivariable Regression
# Block 30: Train a multivariable linear regression model using startup spending features to predict profit.
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

# Block 31: The code below shows an alternative OLS approach using statsmodels for detailed regression diagnostics.
# import statsmodels.api as sm
# y =data['Profit']
# X =data.drop('Profit',axis=1)
# X = sm.add_constant(X)
# model =sm.OLS(endog=y,exog=X).fit()
# print(model.summary())

# Block 32: Calculate the correlation matrix to inspect the relationships among numeric columns.
data.corr(numeric_only=True)

# Block 33: Optional data inspection lines for column types and missing values.
# data.info()

# data.isnull().sum()