# imports

import datetime
import logging

# warnings
def warn(*args, **kwargs):
    pass

import warnings

warnings.warn = warn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
from sklearn.model_selection import train_test_split, validation_curve, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import scale, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import itertools as its
from sklearn import svm
from sklearn.utils import resample

#SMOTE
from imblearn.over_sampling import SMOTE

# graphviz
import graphviz

# scipy
import scipy
from scipy.stats import ttest_rel, wilcoxon

# debug
import pdb


tstamp = datetime.date.today()
FORMAT = ''
LOG_FILENAME = 'dtree-' + str(tstamp) + '-final.log'
logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO, format=FORMAT)

# reads csv file into pandas dataframe
sns.set()

col = ['AberturaExperiencia', 'Agradabilidade', 'Conscienciosidade', 'Extroversao', 'InstabilidadeEmocional', 'BuscaPorExperiencias', 'Impulsividade', 'Droga']

# csvfile #deve-se selecionar aqui qual base deseja analisar
csV = '/home/pedro/Área de trabalho/Bases/SomentePsicologico/Psicodelicos.csv'

# pandasdataframe 
df = pd.read_csv(csV, usecols=col, delimiter=';', low_memory=False)
df = df.round(2)

# ============================================================================ #


def final(size=0.3, state=0, max=None):
    """Trabalho final."""

    # X and Y
    X = df[['AberturaExperiencia', 'Agradabilidade', 'Conscienciosidade','Extroversao', 'InstabilidadeEmocional', 'BuscaPorExperiencias','Impulsividade']]
    Y = df['Droga']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=size, random_state=state)

    #oversampling
    sm = SMOTE(random_state=12, ratio=1.0)
    X_train, Y_train = sm.fit_sample(X_train, Y_train)

    # dtree
    dtree = DecisionTreeClassifier(max_depth=max)
    dtree.fit(X_train, Y_train)
    predDT = dtree.predict(X_test)

    ind = ["Não Usuario", "Usuario"]
    arvore(dtree, ind, max)

    # Logistic Regression
    lreg = LogisticRegression(class_weight='balanced')
    lreg.fit(X_train, Y_train)
    predLR = lreg.predict(X_test)

    coefficients = pd.concat([pd.DataFrame(X.columns),pd.DataFrame(np.transpose(lreg.coef_))], axis = 1)
    print("Coeficiêntes Regressão Logística")
    print(coefficients)

    # accuracy
    # accDT, accLR, accKN, accNB = []
    accuDT = accuracy_score(Y_test, predDT)
    accuLR = accuracy_score(Y_test, predLR)

    # confusion matrix
    con_matDT = confusion_matrix(Y_test, predDT)
    con_matLR = confusion_matrix(Y_test, predLR)

    bla = ["Não Usuario", "Usuario"]
    confusao(bla, con_matDT, 'decision_tree')
    confusao(bla, con_matLR, 'logistic_regression')

    # normalized
    norm = 2880
    confusao(bla, (con_matDT / norm), 'norm_decision_tree')
    confusao(bla, (con_matLR / norm), 'norm_logistic_regression')

    # cross val
    scoresDT = cross_val_score(dtree, X_test, Y_test, scoring='accuracy')
    scoresLR = cross_val_score(lreg, X_test, Y_test, scoring='accuracy')

    # scores roc curve
    scorRocDT = dtree.predict_proba(X_test)[:, 1]
    scorRocLR = lreg.decision_function(X_test)

    # roc
    fprDT, tprDT, _ = roc_curve(Y_test, scorRocDT)
    fprLR, tprLR, _ = roc_curve(Y_test, scorRocLR)

    # auc
    aucDT = auc(fprDT, tprDT)
    aucLR = auc(fprLR, tprLR)

    # plot roc curve
    plt.plot(fprDT, tprDT, color='green', lw=2, label='DT (area = %0.2f)' % aucDT)
    plt.plot(fprLR, tprLR, color='red', lw=2, label='LR (area = %0.2f)' % aucLR)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel('Falso Positivo')
    plt.ylabel('Verdadeiro Positivo')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig('img/curva_roc.png')


    print('# ================================================================ #')
    print('Classifier        Scores                 Acc       Std        Var')
    print('Decision Tree:   ', np.around(scoresDT, decimals=2), "{:10.2f}".format(scoresDT.mean()),
          "{:10.4f}".format(np.std(scoresDT)), "{:10.4f}".format(np.var(scoresDT), "{:10.2f}".format(accuDT)))
    print('Log. Regression: ', np.around(scoresLR, decimals=2), "{:10.2f}".format(scoresLR.mean()),
          "{:10.4f}".format(np.std(scoresLR)), "{:10.4f}".format(np.var(scoresLR), "{:10.2f}".format(accuLR)))
    print('# ================================================================ #')
    print('# ================================================================ #')
    print('Acurácia:')
    print("Decision Tree:       ", "{:10.2f}".format(accuDT))
    print("Logistic Regression: ", "{:10.2f}".format(accuLR))
    print('# ================================================================ #')


# ============================================================================ #


def confusao(index, con_matrix, tit):
    """Plots confusionx' matrix."""

    df_cm = pd.DataFrame(con_matrix, index=index, columns=index)
    plt.figure(figsize=(8, 4))
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 12}, fmt='.2f', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix - ' + tit)
    plt.ylabel('output class')
    plt.xlabel('target class')
    plt.xticks(rotation='horizontal')
    plt.yticks(rotation='horizontal')
    plt.savefig('img/con_matrix_' + str(tit) + '.png')
    print(df_cm)

# ============================================================================ #

def arvore(classif, classes, max):
    """Plot decision tree graph."""

    from subprocess import check_call
    export_graphviz(classif, out_file='img/dtree.dot', class_names=classes, max_depth=max, feature_names=['AberturaExperiencia', 'Agradabilidade', 'Conscienciosidade', 'Extroversao', 'InstabilidadeEmocional', 'BuscaPorExperiencias', 'Impulsividade'],  filled=True, rounded=True, special_characters=True)
    check_call(['dot', '-Tpng', 'img/dtree.dot', '-o', 'img/dtree.png'])


# ============================================================================ #


if __name__ == '__main__':
    final(.3)
