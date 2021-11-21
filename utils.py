import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer

def extract_feat(df, ordinal_features) -> list:
    '''
    :param:
        df -> dataframe (pd.dataframe)
        ordinal_features -> ordinal_features (str or list)
        target -> label column to predict (str)

    :return:
        list of :
        numeric -> list of numeric feature names (list)
        categorical_dumm -> list of categorical (dummies) feature names (list)
        ordinal -> list of ordinal feature names (list)
        binary -> list of binary feature names (list)
    '''
    # initialize ordinal
    ordinal = []
    # count number of different unique values for each feature
    df_uniques = pd.DataFrame([[i, len(df[i].unique())] for i in df.columns],
                              columns=['Variable', 'Unique Values']).set_index('Variable')
    # retrieve binary variables/features
    binary = list(df_uniques[df_uniques['Unique Values'] == 2].index)
    # retrieve categorical variables/features
    categorical = list(df_uniques[(df_uniques['Unique Values'] <= 10) & (df_uniques['Unique Values'] > 2)].index)
    # retrieve ordinal variables/features
    if ordinal_features is not None:
        ordinal.append(ordinal_features)
    # retrieve numeric variables/features
    numeric = list(set(df.columns) - set(ordinal) - set(categorical) - set(binary))
    # encode categorical features
    df = pd.get_dummies(df, columns=categorical, drop_first=True)
    # get categorical dummies
    categorical_dumm = list(set(df.columns) - set(ordinal) - set(numeric) - set(binary))

    return [
        numeric,
        categorical_dumm,
        ordinal,
        binary,
    ]

def encode(df, ordinal_features, target) -> list:
    '''
    :param:
        df -> dataframe (pd.dataframe)
        ordinal_features -> ordinal_features (str or list)
        target -> label column to predict (str)

    :return:
        list of :
        df -> dataframe with encoded features and labels columns (pd.dataframe)
        df[numeric] -> dataframe with encoded numeric features (pd.dataframe)
        df[categorical_dumm] -> dataframe with encoded categorical features (pd.dataframe)
        df[ordinal] -> dataframe with encoded ordinal features (pd.dataframe)
        df[binary] -> dataframe with encoded binary features (pd.dataframe)
    '''
    # store target into separate dataframe before encoding
    df_target = df[target]
    # remove target feature
    df.drop(columns=target, inplace=True)
    # exctract features
    numeric, categorical_dumm, ordinal, binary = extract_feat(df, ordinal_features)
    # encode ordinal features
    Oe = OrdinalEncoder()
    df[ordinal] = Oe.fit_transform(df[ordinal])
    # encode binary features
    lb = LabelBinarizer()
    for column in binary:
        df[column] = lb.fit_transform(df[column])
    # encode ordinal and numeric features
    mm = MinMaxScaler()
    for column in [ordinal + numeric]:
        df[column] = mm.fit_transform(df[column])
    # encode labels
    Le = LabelEncoder()
    df[target] = Le.fit_transform(df_target)
    # recover low features and fille NaN values
    df_numerical = clean_numerical(df[numeric])
    # update main df
    df.update(df_numerical)
    # drop low features
    corr = get_correlation(df[numeric].join(df[target]), target)
    low_features = get_low_features(corr, threshold=0.1).index.tolist()
    df.drop(columns=low_features, inplace=True)
    # update numercial features
    numeric = list(set(numeric) - set(low_features))

    return [df,
            df[numeric],
            df[categorical_dumm],
            df[ordinal],
            df[binary]
            ]

def clean_numerical(df) -> pd.DataFrame:
    '''
    :param:
        df -> dataframe of numeric features (pd.dataframe)
    :return:
        df -> dataframe with imputed values (pd.dataframe)
    '''
    # impute residual missing values
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(df)
    # rebuild dataframe from numpy array
    df = pd.DataFrame(imputer.transform(df), index=df.index, columns=df.columns)

    return df

def get_correlation(df, target) -> pd.Series:
    '''
    :param:
        df -> dataframe input (pd.dataframe)
        target -> label column (str)
    :return:
        corr -> sorted correlation serie in respect with target (pd.series)
    '''
    corr = df.corrwith(df[target]).abs()
    corr.sort_values(ascending=False, inplace=True)

    return corr

def get_low_features(corr, threshold=0.1) -> pd.Series:
    '''
    :param:
        corr -> serie of correlations (pd.Series)
        threshold -> min correlation factor (float)
    :return:
        low_feat -> correlation serie in respect with target (pd.series)
    '''
    low_feat = corr[corr < threshold]

    return low_feat

def get_top_features(corr, top_n=20) -> pd.Series:
    '''
    :param:
        corr -> serie of correlations (pd.Series)
        top_n -> number of features (int)
    :return:
        top_feat -> correlation serie in respect with target (pd.series)
    '''
    top_feat = corr.head(top_n)

    return top_feat

def plot_correlation(df, target, top_n=20)  -> None:
    # get correlation serie
    corr = get_correlation(df, target)
    # retain first 30 with highest correlation
    top_feat = get_top_features(corr=corr, top_n=top_n)
    # plot correlation factors
    sns.heatmap(top_feat.to_frame(),cmap='rainbow',annot=True,annot_kws={"size": 10},vmin=0)
    plt.title("correlation matrix")
    plt.show()