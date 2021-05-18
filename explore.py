import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import my_da
from classification import plot_num_hist
if __name__ == '__main__':
    df = pd.read_csv('rsc/creditcard.csv')
    # check distribution of the data -- all columns are numeric
    plot_num_hist(df, df.columns.drop(['Time', 'Class']), 'Class')

    # from the KDE plot ['V3', 'V4', 'V9', 'V10', 'V11', 'V12', 'V14', 'V16', 'V17', 'V18']
    # look interesting.
    