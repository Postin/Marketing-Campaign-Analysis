import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.cluster import KMeans,DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier
from imblearn.over_sampling import SMOTE

import keras

df = pd.read_csv('data/marketing_campaign.csv',sep =';')
#print(df.info())

#print(df.isnull().sum())
#print(df[df['Year_Birth'] == max(df['Year_Birth'])].Income)
null_income_df = df[df['Income'].isnull()]

df.loc[(df['Kidhome'] > 0) | (df['Teenhome'] > 0), 'HasChild'] = "Yes"
df.loc[(df['Kidhome'] == 0) & (df['Teenhome'] == 0), 'HasChild'] = "No"
df.drop(['Kidhome','Teenhome'], axis=1, inplace=True)


currentYear = datetime.now().year
currentTime = datetime.now()
df['Year_Birth'] = currentYear - df['Year_Birth']
df.rename(columns={'Year_Birth':'Age'}, inplace=True)

df['Marital_Status'] = df['Marital_Status'].apply(lambda x: 'In Couple'
if x == 'Together' or x == 'Married' else 'Single')

def educ_years(x):
    if x == 'Graduation':
        return 12
    elif x == 'PhD':
        return 21
    elif x == 'Master':
        return 18
    elif x == 'Basic':
        return 5
    elif x == '2n Cycle':
        return 8


df['Educational_Years'] = df['Education'].apply(educ_years)

df['Spending'] = df['MntFishProducts'] + df['MntMeatProducts'] + df['MntFruits']\
                 + df['MntSweetProducts'] + df['MntWines'] + df['MntGoldProds']


df.drop(['ID','Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)



def seniority_func(x):
    date_format = '%Y-%m-%d'
    #convert string to date, find delta
    x = datetime.strptime(x,date_format)
    delta = currentTime - x
    return delta.days / 30.417 #Formula by Google.


df['Seniority'] = df['Dt_Customer'].apply(seniority_func)

#Removing outliers
df = df.drop(df[df.Income == max(df.Income)].index)

imputer = KNNImputer(n_neighbors=5,metric='nan_euclidean')
# fit on the dataset
imputer.fit(df[['Income', 'Spending', 'Seniority']])
# transform the dataset
X = imputer.transform(df[['Income', 'Spending', 'Seniority']])
Income_impute = pd.DataFrame(X, columns=['Income', 'Spending', 'Seniority'])
df['Income'] = Income_impute['Income'].reset_index(drop=True)

df = df.dropna()


#Standardization
scaler = StandardScaler()
scaled_df = df.copy()
col_names = ['Income','Spending', 'Seniority',]
features = scaled_df[col_names]
scaler = scaler.fit(features.values)
features = scaler.transform(features.values)
scaled_df[col_names] = features

#Normalization
transformer = Normalizer()
col_names = ['Income','Spending', 'Seniority',]
features = scaled_df[col_names]
transformer = transformer.fit(features.values)
features = transformer.transform(features.values)
scaled_df[col_names] = features



#Elbow method for determining the number of clusters
distortions = []
K = range(1,10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(features)
    distortions.append(sum(np.min(cdist(features, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / features.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
#plt.show()

kmeanModel = KMeans(n_clusters=4, init="k-means++", max_iter=2000, random_state=42).fit(features)
labels_kmeans = kmeanModel.predict(features)
plt.scatter(features[:, 0], features[:, 1], c=labels_kmeans, cmap='viridis')
#plt.show()

gmm=GaussianMixture(n_components=4, covariance_type='spherical',max_iter=2000, random_state=42).fit(features)
labels_gmm = gmm.predict(features)
plt.scatter(features[:, 0], features[:, 1], c=labels_gmm, cmap='viridis')
#plt.show()

dbscan = DBSCAN(eps=0.5).fit(features)
plt.scatter(features[:, 0], features[:, 1], c=dbscan.labels_, cmap='viridis')
#plt.show()

#evaluacija algoritama klasterovanja
print('kmeans: {}'.format(silhouette_score(features, kmeanModel.labels_,
                                           metric='cosine')))

print('GMM: {}'.format(silhouette_score(features, labels_gmm,
                                           metric='cosine')))

print('DBSCAN: {}'.format(silhouette_score(features, dbscan.labels_,
                                           metric='cosine')))
#Interpretacija klastera
scaled_df['Cluster'] = labels_gmm
df['Cluster'] = labels_gmm

#Vizualizacija klastera
fig = plt.figure(figsize=(6,6))
ax = Axes3D(fig)

x = scaled_df['Income']
y = scaled_df['Spending']
z = scaled_df['Seniority']
c = scaled_df['Cluster']

ax.set_xlabel("Income")
ax.set_ylabel("Spending")
ax.set_zlabel("Seniority")

ax.scatter(xs=x,ys=y,zs=z,c=c)
#plt.show()

df_mean = (df.groupby('Cluster').mean())

df_mean['Count'] = df['Cluster'].value_counts()
df_mean['Percent'] = (df_mean['Count'] / df_mean['Count'].sum())*100

def cluster_func(x):
    if x == 0:
        return 'Stars'
    elif x == 1:
        return 'Need attention'
    elif x == 2:
        return 'High potential'
    elif x == 3:
        return 'Leaky bucket'


df['Cluster'] = df['Cluster'].apply(cluster_func)

#More feature engineering
q1,q3 = df['MntWines'].quantile([0.25,0.75])
def bin_func(x):
    if x == 0:
        return "Non consumer"

    if x <= q1:
        return "Low consumer"
    elif q1 < x <= q3:
        return "Frequent consumer"
    else:
        return "Biggest consumer"

df_bin = df[['Education','Marital_Status','HasChild','Cluster']]
df_bin['WineGroup'] = df['MntWines'].apply(bin_func)
#df.drop(['MntWines'],axis=1,inplace=True)

q1, q3 = df['MntFruits'].quantile([0.25,0.75])
df_bin['FruitGroup'] = df['MntFruits'].apply(bin_func)
#df.drop(['MntFruits'],axis=1,inplace=True)

q1, q3 = df['MntMeatProducts'].quantile([0.25,0.75])
df_bin['MeatGroup'] = df['MntMeatProducts'].apply(bin_func)
#df.drop(['MntMeatProducts'],axis=1,inplace=True)

q1, q3 = df['MntFishProducts'].quantile([0.25,0.75])
df_bin['FishGroup'] = df['MntFishProducts'].apply(bin_func)
#df.drop(['MntFishProducts'],axis=1,inplace=True)

q1, q3 = df['MntSweetProducts'].quantile([0.25,0.75])
df_bin['SweetGroup'] = df['MntSweetProducts'].apply(bin_func)
#df.drop(['MntSweetProducts'],axis=1,inplace=True)

q1, q3 = df['MntGoldProds'].quantile([0.25,0.75])
df_bin['GoldGroup'] = df['MntGoldProds'].apply(bin_func)
#df.drop(['MntGoldProds'],axis=1,inplace=True)

def bin_senior_func(x):

    if x <= q1:
        return "New customer"
    elif q1 < x <= q3:
        return "Experienced customer"
    else:
        return "Old customer"

q1, q3 = df['Seniority'].quantile([0.25,0.75])
df_bin['SeniorityGroup'] = df['Seniority'].apply(bin_senior_func)
#df.drop(['Seniority'],axis=1,inplace=True)

def bin_income_func(x):
    if x <= q1:
        return "Low income"
    elif q1 < x <= q2:
        return "Low to medium income"
    elif q2 < x <= q3:
        return "Medium to high income"
    elif x > q3:
        return "High income"

q1,q2,q3 = df['Income'].quantile([0.25,0.5,0.75])
df_bin['IncomeGroup'] = df['Income'].apply(bin_income_func)
#df.drop(['Income'],axis=1,inplace=True)

def bin_age_func(x):
    if x <= 18:
        return "Youth"
    elif 18 < x <= 35:
        return "Young adult"
    elif 35 < x <= 65:
        return "Adult"
    else:
        return "Senior"


df_bin['AgeGroup'] = df['Age'].apply(bin_age_func)

df_assoc = pd.get_dummies(df_bin)

#Apriori min support
min_support = 0.08

#Max lenght of apriori n-grams
max_len = 10

frequent_items = apriori(df_assoc, use_colnames=True, min_support=min_support, max_len=max_len + 1)
rules = association_rules(frequent_items, metric='lift', min_threshold=1)
# We select the product and the segment we want to analyze
product='WineGroup'
segment='Biggest consumer'
target = product + "_" + segment

results_personnal_care = rules[rules['consequents'].astype(str).str.contains(target, na=False)].sort_values(by='confidence', ascending=False)
pd.set_option('display.max_columns', None)

df.drop(['Education','Dt_Customer'],axis=1,inplace=True)

f,  ax2 = plt.subplots(1, 1, figsize=(20, 10))
corr = df.corr()
sns.heatmap(corr,cmap='coolwarm_r', annot_kws={'size':30}, ax=ax2)
ax2.set_title('DF Correlation Matrix', fontsize=12)
#plt.show()


df = pd.get_dummies(df, columns=['Marital_Status','HasChild','Cluster'])

X = df.drop('Response',axis=1)
y = df['Response']
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Always split into test and train sets BEFORE trying oversampling techniques!
# Oversampling before splitting the data can allow the exact same observations to be present in both the test and train sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# transform the dataset
sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train,y_train)
