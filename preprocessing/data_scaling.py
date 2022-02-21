from sklearn.preprocessing import StandardScaler, Normalizer

from constants import scaling_cols


def standardize(df, columns=scaling_cols):
    scaler = StandardScaler()
    scaler = scaler.fit(df[columns])
    df[columns] = scaler.transform(df[columns].values)
    return df

def normalize(df, columns=scaling_cols):
    transformer = Normalizer()
    transformer = transformer.fit(df[columns].values)
    df[columns] = transformer.transform(df[columns].values)
    return df