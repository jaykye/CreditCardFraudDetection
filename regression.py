import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix, classification_report
from sklearn.svm import SVC

if __name__ == '__main__':
    df = pd.read_csv('rsc/creditcard.csv')
    CLASS_COL = 'Class'
    # from explore.py, I want to build a model based on following fields:
    # ['V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
    features = ['V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
    x = df[features]
    y = df[CLASS_COL]
    n_class = df[CLASS_COL].nunique()
    print('Number of Features: ', len(features))
    print('Number of Classes: ', n_class)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.20, random_state=40)

    ####### Gradient Boosting #######
    # GBoost = GradientBoostingRegressor()
    # # RMSE estimated through the partition of the train set
    # GBoost.fit(X_train, y_train)
    # y_pred = GBoost.predict(X_test)
    # print(classification_report(y_test, y_pred))

    ####### Random forest #######
    # print('Random forest')
    # rfc = RandomForestClassifier()
    # rfc.fit(X_train, y_train)
    # y_pred = rfc.predict(X_test)
    # print(classification_report(y_test, y_pred))

    ####### SVM #######
    # from the KDE, it looks like it can be easily divided with lines.
    svmc = SVC()
    svmc.fit(X_train, y_train)
    y_pred = svmc.predict(X_test)
    print(classification_report(y_test, y_pred))


