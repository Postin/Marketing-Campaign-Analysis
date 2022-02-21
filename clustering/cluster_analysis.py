

def cluster_mean(df):
    df_mean = (round(df.groupby('Cluster').mean(), 2))
    df_mean['Count'] = df['Cluster'].value_counts()
    df_mean['Percent'] = round((df_mean['Count'] / df_mean['Count'].sum()) * 100, 2)
    print(df_mean[['Income', 'Spending', 'Seniority', 'Count', 'Percent']])