import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib
import yfinance as yf
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, Dense, Reshape, LSTM

steps_into_future = 1 # 1, 10 and 30 are used in the paper
test_only_on_last_subset = False # Set to true for the last part of results in paper
features_n = 6
str_time_step = ''

if steps_into_future == 1:
    str_time_step = 'next'
else:
    str_time_step = str(steps_into_future)


stock_tickers = ['TSLA', 'AMZN', 'GOOG', 'MSFT', 'AAPL', 'PFE', 'AMT', 'VTR', 'XOM', 'CVX', 'UPS', 'FDX']
classifiers = ['SVM', 'LSTM', 'RF']

# read data from yahoo finance
#data = yf.download(' '.join(stock_tickers), period='10y', interval='1d', group_by='ticker',
#                   auto_adjust=True,
#                   prepost=True)
#print(data)
# save data to csv file
#data.to_csv(r'data_from_yfinance.csv')

# read data from csv file 
data = pd.read_csv(r'data_from_yfinance.csv', header=[0,1], index_col=0)
print(data)



class LSTMClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs, num_features=features_n):
        self.epochs = epochs
        self.model = Sequential([
            Reshape((1, num_features), batch_input_shape=(1, num_features)),
            LSTM(units=50, stateful=True),
            Dropout(0.4),
            Dense(units=1, activation='sigmoid')
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def fit(self, X, y):
        for e in range(self.epochs):
            self.model.fit(X, y, epochs=1, batch_size=1, verbose=1, shuffle=False)
            self.model.reset_states()

    def predict(self, X):
        proba = self.model.predict(X, batch_size=1)
        return (proba > 0.5).astype('int32')


def create_classifier(classifier):
    if classifier == 0:
        return SVC(kernel='poly', degree=1)
    elif classifier == 1:
        #return SVC()
        return LSTMClassifier(epochs=10)
    else:
        return RandomForestClassifier(n_estimators=30)


accuracy_plot = pd.DataFrame(index=stock_tickers, columns=pd.Index(classifiers + ['Up'], name='Accuracy'))
f1_plot = pd.DataFrame(index=stock_tickers, columns=pd.Index(classifiers + ['Up'], name='F1 Score'))
accuracy_std_plot = pd.DataFrame(index=stock_tickers, columns=pd.Index(classifiers + ['Up'], name='Accuracy Std'))
f1_std_plot = pd.DataFrame(index=stock_tickers, columns=pd.Index(classifiers + ['Up'], name='F1 Std'))
header = pd.MultiIndex.from_product([['Set 0', 'Set 1', 'Set 2', 'Set 3', 'Set 4', 'Set 5'], ['increase', 'decrease']])
data_stats = pd.DataFrame(index=stock_tickers, columns=header)

meanAcc = [0,0,0,0]
meanAccStd = [0,0,0,0]
meanFscore = [0,0,0,0]
meanFscoreStd = [0,0,0,0]

latex_confusion_matrices = ''

for ticker in stock_tickers:
    hist = data[ticker].dropna()

    rsi = pd.DataFrame(talib.RSI(hist['Close'])).fillna(0)
    stoch = pd.DataFrame(talib.STOCH(hist['High'], hist['Low'], hist['Close'])[0]).fillna(0)
    macd = pd.DataFrame(talib.MACD(hist['Close'])[0]).fillna(0)
    sma = pd.DataFrame(talib.SMA(hist['Close'], 10)).fillna(0)
    will = pd.DataFrame(talib.WILLR(hist['High'], hist['Low'], hist['Close'])).fillna(0)
    mom = pd.DataFrame(talib.MOM(hist['Close'])).fillna(0)

    # Number of days used for calculating indicators
    init_days = 50

    X = np.zeros((hist.shape[0], 6))
    X[:, 0] = np.where(rsi > 70, -1, np.where(rsi < 30, 1, np.where(rsi > rsi.shift(1), 1, -1))).squeeze()
    X[:, 1] = np.where(stoch > stoch.shift(1), 1, -1).squeeze()
    X[:, 2] = np.where(macd > macd.shift(1), 1, -1).squeeze()
    X[:, 3] = np.where(sma.squeeze() > hist['Close'], 1, -1).squeeze()
    X[:, 4] = np.where(will > will.shift(1), 1, -1).squeeze()
    X[:, 5] = np.where(mom > 0, 1, -1).squeeze()


    y = np.where(hist['Close'].shift(-steps_into_future) > hist['Close'], 1, 0)[init_days:-steps_into_future]

    X = X[init_days:-steps_into_future]

    confusion_mat_up = np.zeros((2,2))

    # Save up/down stats for every set
    tscv = TimeSeriesSplit()
    set = 0
    for train_index, val_index in tscv.split(X):
        if set == 0:
            n = len(y[train_index])
            increase = np.count_nonzero(y[train_index] == 1)
            decrease = n - np.count_nonzero(y[train_index] == 1)
            data_stats.loc[ticker, ('Set ' + str(set), 'increase')] = increase / n
            data_stats.loc[ticker, ('Set ' + str(set), 'decrease')] = decrease / n
            set += 1
        
        n = len(y[val_index])
        increase = np.count_nonzero(y[val_index] == 1)
        decrease = n - np.count_nonzero(y[val_index] == 1)
        data_stats.loc[ticker, ('Set ' + str(set), 'increase')] = increase / n
        data_stats.loc[ticker, ('Set ' + str(set), 'decrease')] = decrease / n
        set += 1

    # If we only want to save the data stats, uncomment this
    # continue

    for clf in range(len(classifiers)):
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', create_classifier(clf)),
        ])

        tscv = TimeSeriesSplit()
        f1_scores = []
        f1_scores_up = []
        accuracies = []
        scores_always_up = []
        confusion_mat = np.zeros((2,2))

        i = 0
        for train_index, val_index in tscv.split(X):
            xt, xv = X[train_index], X[val_index]
            yt, yv = y[train_index], y[val_index]
            pipe.fit(xt, yt)
            
            if i >= 4 or not test_only_on_last_subset:
                pred = pipe.predict(xv)
                f1_scores.append(f1_score(yv, pred))
                f1_scores_up.append(f1_score(yv, np.ones(yv.shape)))
                accuracies.append(pipe.score(xv, yv))
                scores_always_up.append(yv.sum() / yv.shape[0])
                confusion_mat += confusion_matrix(yv, pred)
                confusion_mat_up += confusion_matrix(yv, np.ones(yv.shape))
            i += 1
        
        confusion_mat = confusion_mat/confusion_mat.sum()

        # Normalize confusion matrix
        colsum = np.sum(confusion_mat[:,0])
        colsum2 = np.sum(confusion_mat[:,1])
        confusion_mat[0][0] = confusion_mat[0][0] / colsum
        confusion_mat[1][0] = confusion_mat[1][0] / colsum
        confusion_mat[0][1] = confusion_mat[0][1] / colsum2
        confusion_mat[1][1] = confusion_mat[1][1] / colsum2

        # print(confusion_mat_up/confusion_mat_up.sum())

        accuracy_plot[classifiers[clf]][ticker] = np.mean(accuracies)
        accuracy_plot['Up'][ticker] = np.mean(scores_always_up)
        accuracy_std_plot[classifiers[clf]][ticker] = np.std(accuracies)
        accuracy_std_plot['Up'][ticker] = np.std(scores_always_up)

        f1_plot[classifiers[clf]][ticker] = np.mean(f1_scores)
        f1_plot['Up'][ticker] = np.mean(f1_scores_up)
        f1_std_plot[classifiers[clf]][ticker] = np.std(f1_scores)
        f1_std_plot['Up'][ticker] = np.std(f1_scores_up)

        latex_confusion_matrices += """
\\begin{table}[H]
    \\centering
    \\begin{tabular}{lll}
                    & Predicted: Down     & Predicted: Up      \\\\
                    \\hline
        Actual: Down     & """ + str(confusion_mat[0][0]) + """ & """ + str(confusion_mat[0][1]) + """  \\\\
        Actual: Up      & """ + str(confusion_mat[1][0]) + """  & """ + str(confusion_mat[1][1]) + """  \\\\
    \\end{tabular}
    \\caption{Confusion matrix for """ + classifiers[clf] + """ predicting """ + ticker + """ (""" + str_time_step + """ day predictions).}
    \\label{tab:confusion_matrix_""" + classifiers[clf] + """_""" + ticker + """}
\\end{table}
        """

        print(f"""{ticker}, {classifiers[clf]}: {np.mean(accuracies):.3f} F1: {np.mean(f1_scores):.3f} (Always up: {np.mean(scores_always_up):.3f})""")

        meanAcc[clf] += np.mean(accuracies)
        meanAccStd[clf] += np.std(accuracies)
        meanFscore[clf] += np.mean(f1_scores)
        meanFscoreStd[clf] += np.std(f1_scores)
    
    confusion_mat_up = confusion_mat_up / confusion_mat_up.sum()
    latex_confusion_matrices += """
\\begin{table}[H]
    \\centering
    \\begin{tabular}{lll}
                    & Predicted: Down     & Predicted: Up      \\\\
                    \\hline
        Actual: Down     & """ + str(confusion_mat_up[0][0]) + """ & """ + str(confusion_mat_up[0][1]) + """  \\\\
        Actual: Up      & """ + str(confusion_mat_up[1][0]) + """  & """ + str(confusion_mat_up[1][1]) + """  \\\\
    \\end{tabular}
    \\caption{Confusion matrix for Up predicting """ + ticker + """ (""" + str_time_step + """ day predictions).}
    \\label{tab:confusion_matrix_Up_""" + ticker + """}
\\end{table}
        """
    
    meanAcc[3] += np.mean(scores_always_up)
    meanAccStd[3] += np.std(scores_always_up)
    meanFscore[3] += np.mean(f1_scores_up)
    meanFscoreStd[3] += np.std(f1_scores_up)



meanAcc = [x/len(stock_tickers) for x in meanAcc]
meanAccStd = [x/len(stock_tickers) for x in meanAccStd]
meanFscore = [x/len(stock_tickers) for x in meanFscore]
meanFscoreStd = [x/len(stock_tickers) for x in meanFscoreStd]

# print(latex_confusion_matrices)
str_extra = ''
if test_only_on_last_subset:
    str_extra = '_last_subset_only'

f = open("latex_confusion_matrices_" + str_time_step + str_extra + ".tex", "w")
f.write(latex_confusion_matrices)
f.close()

data_stats.to_csv(r'data_stats.csv')

print(f"""meanAcc: {meanAcc}""")
print(f"""meanAccStd: {meanAccStd}""")
print(f"""meanFscore: {meanFscore}""")
print(f"""meanFscoreStd: {meanFscoreStd}""")

accuracy_plot.plot(kind='bar', title='Accuracy (' + str_time_step + ' day predictions)')
plt.show()
plt.clf()
accuracy_std_plot.plot(kind='bar', title='Accuracy Std (' + str_time_step + ' day predictions)')
plt.show()
plt.clf()
f1_plot.plot(kind='bar', title='F1 Score (' + str_time_step + ' day predictions)')
plt.show()
plt.clf()
f1_std_plot.plot(kind='bar', title='F1 Score Std (' + str_time_step + ' day predictions)')
plt.show()
plt.clf()
