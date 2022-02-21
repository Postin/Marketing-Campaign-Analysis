
def remove_outliers(df):
    # Remove just the one outlier of the Income feature
    df = df.drop(df[df.Income == max(df.Income)].index)
    return df