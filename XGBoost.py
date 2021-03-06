import matplotlib

matplotlib.use('Agg')
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.cross_validation import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import csv
import gc

PLOT_FEATURES = True
PRE_PROCESS_CATEGORICAL_DATA = False
PRE_PROCESS_DATE_DATA = False

INPUT_DIRECTORY = 'input/'
OUTPUT_DIRECTORY = 'output/'
CHUNKSIZE = 50000

submission_sample_filename = 'sample_submission.csv'
submission_filename = 'submission.csv'

train_numerical_filename = 'train_numeric.csv'
train_date_filename = 'train_date.csv'
train_categorical_filename = 'train_categorical.csv'
train_categorical_leaveoneout_filename = 'train_categorical_leaveoneout.csv'
train_date_1wk_filename = 'train_date_1week.csv'

test_numerical_filename = 'test_numeric.csv'
test_date_filename = 'test_date.csv'
test_categorical_filename = 'test_categorical.csv'
test_categorical_leaveoneout_filename = 'test_categorical_leaveoneout.csv'
test_date_1wk_filename = 'test_date_1week.csv'

## Files below are small sample of data
## Uncomment for debug and development purposes

#submission_sample_filename = 'sample_submission_smallsample.csv'
#submission_filename = 'submission.csv'

#train_numerical_filename = 'train_numeric_smallsample.csv'
#train_date_filename = 'train_date_smallsample.csv'
#train_categorical_filename = 'train_categorical_smallsample.csv'
#train_categorical_leaveoneout_filename = 'train_categorical_leaveoneout.csv'
#train_date_24hr_filename = 'train_date_24hrsmallsample.csv'

#test_numerical_filename = 'test_numeric_smallsample.csv'
#test_date_filename = 'test_date_smallsample.csv'
#test_categorical_filename = 'test_categorical_smallsample.csv'
#test_categorical_leaveoneout_filename = 'test_categorical_leaveoneout.csv'
#test_date_24hr_filename = 'test_date_24hrsmallsample.csv'


def leave_one_out_encode(series):
    """
    Calculate response value for leave-one-out encoding
    :param series:
    :return: series: series with calculated response
    """
    series = (series.sum() - series) / (len(series) - 1)
    return series


def categorical_to_numeric(train_data, test_data, response_col_name):
    """
    Encodes categorical train and test data using leave-one-out technique
    :param train_data: categorical train data frame to be encoded
    :param test_data: categorical test data frame to be encoded
    :param response_col_name: response column name of test data
    :return: train_date, test_data: these are the converted dataframes with encoded categories
    """

    for col_name in train_data.columns:
        if col_name not in ['Id', 'Response']:
            mean = train_data.groupby(by=[col_name])[response_col_name].mean()
            # conversion of mean, which is a pd.series to an int
            if mean.empty is True:
                mean = np.nan
            else:
                mean = mean[0]
            test_data[col_name] = mean

            one_hot_encoded = train_data.groupby(by=[col_name])[response_col_name].apply(leave_one_out_encode)
            train_data[col_name] = one_hot_encoded

    train_data.drop('Response', axis=1, inplace=True)
    return train_data, test_data


def get_data(directory, file_list, data_type_list, use_col_list):
    # type: (str, list, list, list) -> pd.DataFrame
    """
    :param directory: input directory location
    :param file_list: list of files to read
    :param data_type_list: list of data types for each file (ex. int, str, etc)
    :param use_col_list: list of columns to read from
    :return: dataframe of all files
    """

    data = None
    for i, file_name in enumerate(file_list):
        print file_name
        subset = None
        for j, chunk in enumerate(
                pd.read_csv(directory + file_name, chunksize=CHUNKSIZE, usecols=use_col_list[i], low_memory=False,
                            dtype=data_type_list[i])):
            print j
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if data is None:
            data = subset.copy()
        else:
            data = pd.merge(data, subset.copy(), on="Id")
        del subset
        gc.collect()

    return data


def pre_process_categorical_data():
    # type: () -> bool
    """
    method for pre processing the categorical data
    :rtype: bool
    """
    directory = 'input/'
    trainfiles = [train_categorical_filename,
                  train_numerical_filename]
    testfiles = [test_categorical_filename]

    with open(INPUT_DIRECTORY + train_categorical_filename) as f:
        reader = csv.reader(f)
        Cat_columns = next(reader)  # Obtains header names from file

    cols = [Cat_columns,
            ['Id',
             'Response']]
    traindata = None
    testdata = None
    for i, f in enumerate(trainfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=CHUNKSIZE,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if traindata is None:
            traindata = subset.copy()
        else:
            traindata = pd.merge(traindata, subset.copy(), on="Id")
        del subset
        gc.collect()

    del cols[1][-1]  # Remove this column, test data doesn't have a response
    for i, f in enumerate(testfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              usecols=cols[i],
                                              chunksize=CHUNKSIZE,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if testdata is None:
            testdata = subset.copy()
        else:
            testdata = pd.merge(testdata, subset.copy(), on="Id")
        del subset
        gc.collect()

    traindata, testdata = categorical_to_numeric(traindata, testdata, 'Response')

    traindata.to_csv(INPUT_DIRECTORY + train_categorical_leaveoneout_filename)
    testdata.to_csv(INPUT_DIRECTORY + test_categorical_leaveoneout_filename)
    return True


def pre_process_date_data():
    # type: () -> bool
    """
    pre processes date data to rolling weekly format
    :rtype: bool
    """
    directory = INPUT_DIRECTORY
    trainfiles = [train_date_filename]
    testfiles = [test_date_filename]

    traindata = None
    testdata = None
    for i, f in enumerate(trainfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              chunksize=CHUNKSIZE,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if traindata is None:
            traindata = subset.copy()
        else:
            traindata = pd.merge(traindata, subset.copy(), on="Id")
        del subset
        gc.collect()

    for i, f in enumerate(testfiles):
        print(f)
        subset = None
        for i, chunk in enumerate(pd.read_csv(directory + f,
                                              chunksize=CHUNKSIZE,
                                              low_memory=False)):
            print(i)
            if subset is None:
                subset = chunk.copy()
            else:
                subset = pd.concat([subset, chunk])
            del chunk
            gc.collect()
        if testdata is None:
            testdata = subset.copy()
        else:
            testdata = pd.merge(testdata, subset.copy(), on="Id")
        del subset
        gc.collect()

    traindata = date_dataframe_converter(traindata, 100.8) # change value for desired format (24 hrs = 14.4, 1 week = 100.8)
    testdata = date_dataframe_converter(testdata, 100.8)

    traindata.to_csv(INPUT_DIRECTORY + train_date_1wk_filename)
    testdata.to_csv(INPUT_DIRECTORY + test_date_1wk_filename)
    return True


def date_dataframe_converter(df, constant):
    # type: (df, float) -> df
    """
    modifies data frame to rolling 24hr or weekly format, this value is determined by constant

    :rtype: object
    """
    col_list = list(df.columns.values)
    if 'Id' in col_list:
        col_list.remove('Id')
    if 'Response' in col_list:
        col_list.remove('Response')

    # Determine modulus of all columns by constant, except for ID and Response columns
    # modulus is used to convert to 24 hrs or to days of a week (24 hrs = 14.4, 1 week = 100.8)
    for col in col_list:
        df.loc[:,col] %= constant

    return df


def get_xgb_imp(xgb, feat_names):
    # type: (xgboost classifier, str) -> object
    """
    returns important features from xgboost classifier
    :param xgb: xgboost classifier class
    :param feat_names:
    :return:
    """
    from numpy import array
    imp_vals = xgb.booster().get_fscore()
    imp_dict = {feat_names[i]: float(imp_vals.get('f' + str(i), 0.)) for i in range(len(feat_names))}
    total = array(imp_dict.values()).sum()
    return {k: v / total for k, v in imp_dict.items()}


def important_cols2file_cols(important_cols, file_name):
    # type: (list, str) -> list
    """
    returns list of columns that are in both important_cols and in the file
    :param important_cols:
    :param file_name:
    :return: file_cols: list of files that were identified as important
    """

    with open(file_name) as f:
        reader = csv.reader(f)
        col_list = next(reader)  # gets the first line

    file_cols = [col for col in col_list if col in important_cols]

    return file_cols


def identify_important_features(training_date_filename, training_numerical_filename):
    # type: (str, str) -> list
    """
    samples training data, fits initial xgboost classifier and identifies important features
    :param training_date_filename:
    :param training_numerical_filename:
    :rtype: important_columns: list of important columns
    """
    date_chunks = pd.read_csv(INPUT_DIRECTORY + training_date_filename, index_col=0, chunksize=CHUNKSIZE, dtype=np.float32)
    num_chunks = pd.read_csv(INPUT_DIRECTORY + training_numerical_filename, index_col=0, usecols=list(range(969)),
                             chunksize=CHUNKSIZE, dtype=np.float32)

    X = pd.concat([pd.concat([dchunk, nchunk], axis=1).sample(frac=0.05) for dchunk, nchunk in
                  zip(date_chunks, num_chunks)])
    col_list = X.columns
    y = pd.read_csv(INPUT_DIRECTORY + training_numerical_filename, index_col=0, usecols=[0, 969], dtype=np.float32).loc[
        X.index].values.ravel()
    X = X.values
    y = np.nan_to_num(y)
    clf = XGBClassifier(base_score=0.005)
    clf.fit(X, y)

    # threshold for a manageable number of features
    if PLOT_FEATURES:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.hist(clf.feature_importances_[clf.feature_importances_ > 0])
        fig.savefig(OUTPUT_DIRECTORY + 'xgboost_hist.png')
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(20,50))
    plt.xlabel('xlabel', fontsize=18)
    plt.ylabel('ylabel', fontsize=18)
    plt.xticks(size=12)
    plt.yticks(size=12)
    myplt = plot_importance(clf, ax=ax)
    fig.savefig(OUTPUT_DIRECTORY + 'xgboost_importantfeatures.png')
    plt.close(fig)

    important_indices = np.where(clf.feature_importances_ > 0.005)[0]

    important_columns = [col for i, col in enumerate(col_list) if
                         i in important_indices]  # converts important_indices to col names
    print(important_columns)
    return important_columns


def load_train_data(important_cols, training_date_filename, training_numerical_filename):
    # type: (list, str, str) -> pd.DataFrame, pd.DataFrame

    """

    :param important_cols: list of important columns
    :param training_date_filename:
    :param training_numerical_filename:
    :return: X,y: X is training data, y is the response column
    """
    important_date_cols = important_cols2file_cols(important_cols, INPUT_DIRECTORY + training_date_filename)
    important_date_cols.append('Id')
    important_num_cols = important_cols2file_cols(important_cols, INPUT_DIRECTORY + training_numerical_filename)
    important_num_cols.append('Id')


    X = np.concatenate([
        pd.read_csv(INPUT_DIRECTORY + training_date_filename, index_col=0, dtype=np.float32,
                    usecols=important_date_cols).values,
        pd.read_csv(INPUT_DIRECTORY + training_numerical_filename, index_col=0, dtype=np.float32,
                    usecols=important_num_cols).values
    ], axis=1)
    y = pd.read_csv(INPUT_DIRECTORY + training_numerical_filename, index_col=0, dtype=np.float32,
                    usecols=[0, 969]).values.ravel()

    return X, y


def train_data(clf, X, response):
    # type: (xgboost classifier, pd.DataFrame, pd.DataFrame) -> xgboost classifier, float
    """

    :param clf: xgboost classfiier
    :param X: training data dataframe
    :param response: response dataframe
    :return: clf, best_threshold: returns trained classifier. best_threshold is threshold for determing pass/failing part in predictions
    """
    cv = StratifiedKFold(response, n_folds=7)
    preds = np.ones(response.shape[0])
    for i, (train, test) in enumerate(cv):
        preds[test] = clf.fit(X[train], response[train]).predict_proba(X[test])[:, 1]
        print("fold {}, ROC AUC: {:.3f}".format(i, roc_auc_score(response[test], preds[test])))
    print(roc_auc_score(response, preds))

    # pick the best threshold out-of-fold
    thresholds = np.linspace(0.01, 0.99, 50)
    mcc = np.array([matthews_corrcoef(response, preds > thr) for thr in thresholds])
    if PLOT_FEATURES:
        fig, ax = plt.subplots(nrows=1, ncols=1)
        plt.plot(thresholds, mcc)
        plt.savefig(OUTPUT_DIRECTORY + 'xgboostchart.png')
        plt.close(fig)
    best_threshold = thresholds[mcc.argmax()]
    print(mcc.max())
    return clf, best_threshold


def load_test_data(important_cols, date_filename, numerical_filename):
    # type: (list, str, str) -> pd.DataFrame
    """
    loads only important columns of test data
    :param important_cols: list of important columns
    :param date_filename:
    :param numerical_filename:
    :return: returns dataframe with only important columns
    """
    important_date_cols = important_cols2file_cols(important_cols, INPUT_DIRECTORY + date_filename)
    important_date_cols.append('Id')
    important_num_cols = important_cols2file_cols(important_cols, INPUT_DIRECTORY + numerical_filename)
    important_num_cols.append('Id')

    X = np.concatenate([
        pd.read_csv(INPUT_DIRECTORY + date_filename, index_col=0, dtype=np.float32,
                    usecols=important_date_cols).values,
        pd.read_csv(INPUT_DIRECTORY + numerical_filename, index_col=0, dtype=np.float32,
                    usecols=important_num_cols).values
    ], axis=1)

    return X


def predict_and_submit(test_data, best_threshold, clf):
    # type: (pd.DataFrame, float, xgboost classifier) -> None
    # generate predictions at the chosen threshold
    """
    uses classifer to make predictions. Predictions are saved into a csv
    :param test_data: dataframe for test_data
    :param best_threshold: threshold for evaluating passing/failing part
    :param clf: xgboost classifier
    """
    preds = (clf.predict_proba(test_data)[:, 1] > best_threshold).astype(np.int8)

    # and submit
    sub = pd.read_csv(INPUT_DIRECTORY + submission_sample_filename, index_col=0)
    sub["Response"] = preds
    sub.to_csv(OUTPUT_DIRECTORY + submission_filename)
    sub.to_csv(OUTPUT_DIRECTORY + submission_filename + ".gz", compression="gzip")


if __name__ == '__main__':
    print('Started')

    if PRE_PROCESS_CATEGORICAL_DATA:
        print('Started Pre Processing Categorical Data')
        pre_process_categorical_data()
        print('Finished Pre Processing Categorical Data')

    if PRE_PROCESS_DATE_DATA:
        print('Started Pre Processing Date Data')
        pre_process_date_data()
        print('Finished Pre Processing Date Data')

    print('Started important_cols')
    important_cols = identify_important_features(train_date_1wk_filename, train_numerical_filename)
    print('Finished important_cols')

    print('Started loading train data')
    train, response = load_train_data(important_cols, train_date_1wk_filename, train_numerical_filename)
    print('Finished loading train data')

    param = {}
    param['learning_rate'] = 0.05
    param['max_depth'] = 9
    param["subsample"] = 0.9
    param['colsample_bytree'] = 0.9
    param['base_score'] = 0.005
    clf = XGBClassifier(**param)

    clf, best_threshold = train_data(clf, train, response)
    print('Finished training')
    test_data = load_test_data(important_cols, test_date_1wk_filename, test_numerical_filename)
    predict_and_submit(test_data, best_threshold, clf)

    print('Finished')
