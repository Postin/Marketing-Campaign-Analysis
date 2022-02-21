from datetime import datetime
import pandas as pd


def feature_engineering(df):
    df = make_has_child(df)
    df = make_age(df)
    df = make_marital_status(df)
    df = make_educ_years(df)
    df = make_spending(df)
    df = make_seniority(df)
    # Drop excess columns
    df.drop(['ID', 'Z_CostContact', 'Z_Revenue'], axis=1, inplace=True)

    return df


def make_has_child(df):
    df.loc[(df['Kidhome'] > 0) | (df['Teenhome'] > 0), 'HasChild'] = "Yes"
    df.loc[(df['Kidhome'] == 0) & (df['Teenhome'] == 0), 'HasChild'] = "No"
    df.drop(['Kidhome', 'Teenhome'], axis=1, inplace=True)
    return df


def make_age(df):
    # Year of birth to age
    currentYear = datetime.now().year
    df['Year_Birth'] = currentYear - df['Year_Birth']
    df.rename(columns={'Year_Birth': 'Age'}, inplace=True)
    return df


def make_marital_status(df):
    #Marital Status
    df['Marital_Status'] = df['Marital_Status'].apply(lambda x: 'In Couple'
    if x == 'Together' or x == 'Married' else 'Single')
    return df


# Helper function
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


def make_educ_years(df):
    df['Educational_Years'] = df['Education'].apply(educ_years)
    return df


def make_spending(df):
    df['Spending'] = df['MntFishProducts'] + df['MntMeatProducts'] + df['MntFruits']\
                 + df['MntSweetProducts'] + df['MntWines'] + df['MntGoldProds']
    return df


# Seniority (number of months in company)
def seniority_func(x):
    currentTime = pd.to_datetime('2015-01-01')
    date_format = '%Y-%m-%d'
    #convert string to date, find delta
    x = datetime.strptime(x,date_format)
    delta = currentTime - x
    return delta.days / 30.417 #Formula by Google.


def make_seniority(df):
    df['Seniority'] = df['Dt_Customer'].apply(seniority_func)
    return df


# CLUSTER POSTPROCESS FEATURE ENGINEERING#
def cluster_func(x):
    if x == 0:
        return 'Stars'
    elif x == 1:
        return 'Need Attention'
    elif x == 2:
        return 'High Potential'
    elif x == 3:
        return 'Leaky Bucket'


def make_cluster(df):
    df['Cluster'] = df['Cluster'].apply(cluster_func)
    return df

def print_bin_explainer(df):
    q1, q3 = df['MntWines'].quantile([0.25, 0.75])
    print("WINE: Q1 - {} Q3 - {}".format(q1, q3))
    q1, q3 = df['MntFruits'].quantile([0.25, 0.75])
    print("FRUITS: Q1 - {} Q3 - {}".format(q1, q3))
    q1, q3 = df['MntMeatProducts'].quantile([0.25, 0.75])
    print("MEAT: Q1 - {} Q3 - {}".format(q1, q3))
    q1, q3 = df['MntFishProducts'].quantile([0.25, 0.75])
    print("FISH: Q1 - {} Q3 - {}".format(q1, q3))
    q1, q3 = df['MntSweetProducts'].quantile([0.25, 0.75])
    print("SWEETS: Q1 - {} Q3 - {}".format(q1, q3))
    q1, q3 = df['MntGoldProds'].quantile([0.25, 0.75])
    print("GOLD: Q1 - {} Q3 - {}".format(q1, q3))
    q1, q3 = df['Seniority'].quantile([0.25, 0.75])
    print("SENIORITY: Q1 - {} Q3 - {}".format(q1, q3))
    q1, q2, q3 = df['Income'].quantile([0.25, 0.5, 0.75])
    print("INCOME: Q1 - {} Q2- {} Q3 - {}".format(q1, q2, q3))
    print("AGE: Q1 - {} Q2- {} Q3 - {}".format(18, 35, 65))

def make_bin_df(df):
    # Make new dataframe
    df_bin = df[['Education', 'Marital_Status', 'HasChild', 'Cluster']]

    # Calculate first and third quartile for MntWines
    q1, q3 = df['MntWines'].quantile([0.25, 0.75])
    df_bin['WineGroup'] = df['MntWines'].apply(lambda x: bin_func(x, q1, q3))
    # Bin MntFruits
    q1, q3 = df['MntFruits'].quantile([0.25, 0.75])
    df_bin['FruitGroup'] = df['MntFruits'].apply(lambda x: bin_func(x, q1, q3))
    # Bin MntMeatProducts
    q1, q3 = df['MntMeatProducts'].quantile([0.25, 0.75])
    df_bin['MeatGroup'] = df['MntMeatProducts'].apply(lambda x: bin_func(x, q1, q3))
    # Bin MntFishProducts
    q1, q3 = df['MntFishProducts'].quantile([0.25, 0.75])
    df_bin['FishGroup'] = df['MntFishProducts'].apply(lambda x: bin_func(x, q1, q3))
    # Bin MntSweetProducts
    q1, q3 = df['MntSweetProducts'].quantile([0.25, 0.75])
    df_bin['SweetGroup'] = df['MntSweetProducts'].apply(lambda x: bin_func(x, q1, q3))
    # Bin MntGoldProds
    q1, q3 = df['MntGoldProds'].quantile([0.25, 0.75])
    df_bin['GoldGroup'] = df['MntGoldProds'].apply(lambda x: bin_func(x, q1, q3))
    # Bin Seniority
    q1, q3 = df['Seniority'].quantile([0.25, 0.75])
    df_bin['SeniorityGroup'] = df['Seniority'].apply(lambda x: bin_senior_func(x, q1, q3))
    # Bin Income
    q1, q2, q3 = df['Income'].quantile([0.25, 0.5, 0.75])
    df_bin['IncomeGroup'] = df['Income'].apply(lambda x: bin_income_func(x, q1, q2, q3))
    # Bin Age
    df_bin['AgeGroup'] = df['Age'].apply(bin_age_func)

    return df_bin


def bin_func(x, q1, q3):
    if x == 0:
        return "Non consumer"

    if x <= q1:
        return "Low consumer"
    elif q1 < x <= q3:
        return "Frequent consumer"
    else:
        return "Biggest consumer"


def bin_senior_func(x, q1, q3):

    if x <= q1:
        return "New customer"
    elif q1 < x <= q3:
        return "Experienced customer"
    else:
        return "Old customer"

def bin_income_func(x, q1, q2, q3):
    if x <= q1:
        return "Low income"
    elif q1 < x <= q2:
        return "Low to medium income"
    elif q2 < x <= q3:
        return "Medium to high income"
    elif x > q3:
        return "High income"

def bin_age_func(x):
    if x <= 18:
        return "Youth"
    elif 18 < x <= 35:
        return "Young adult"
    elif 35 < x <= 65:
        return "Adult"
    else:
        return "Senior"


