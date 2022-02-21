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