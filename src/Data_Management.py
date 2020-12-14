# ==================================================================
#   EL HAYANI MOUSTAMIDE Oumaima
#   CHAAIRAT Tariq
# ==================================================================

# Fichier de gestion des données issues de la BDD du lien kaggle suivant :
# https://www.kaggle.com/c/leaf-classification

# Explication des différentes méthodes présentes dans ce fichier :


import warnings
import pandas as pd
import seaborn as sns

sns.set_palette('husl')
import matplotlib.pyplot as plt

# imports des libs nécessaires
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from pandas import set_option

warnings.filterwarnings("ignore")


class Data_Management(object):
    '''
        Classe DataManagement permettant de préparer les données
    '''

    def __init__(self):
        """
            Importation des données de la BDD leaf
        """
        # ======================================================================================
        # Modifier le booleen cross_val pour l'utilisation de la validation croisée :
        # ======================================================================================
        self.cross_val = True

        # récupération des données d'entrainement
        data_train = pd.read_csv("C:/Users/Tariq/PycharmProjects/pythonProject/src/data/train.csv")
        self.data_to_print = data_train

        # récupération données de test
        self.data_test = pd.read_csv("C:/Users/Tariq/PycharmProjects/pythonProject/src/data/test.csv")

        # on enlève les 2 premières colonnes
        self.data_X_train = data_train.iloc[:, 2:]

        self.data_X_test = self.data_test.iloc[:, 1:]
        X = self.data_X_train

        # on convertit la colonne species en un dtype = category
        y = data_train['species'].astype('category')
        y = y.cat.codes.to_numpy()

        self.data_y_train = y
        sss = StratifiedShuffleSplit(10, 0.2, random_state=15)

        for train_index, test_index in sss.split(X, y):
            self.X_train, self.X_test = X.iloc[train_index], X.iloc[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]

        le = LabelEncoder().fit(data_train.iloc[:, 1])
        self.classes = list(le.classes_)
        self.test_id = self.data_test.iloc[:, 0]

    def get_cross_validation(self):
        """
        Getters du booléen de la cross_val
        :param cross_val:
        """
        return self.cross_val

    def print_data(self):
        """
            Affiche toutes les données d'entraînement
            :return:
            """
        printing = self.data_to_print[['id', 'species', 'margin20', 'shape20', 'texture20']]
        print(printing)

    def get_shape_ind_0(self):
        """
            Getters sur la forme des données
            :return:
            """
        print('Il y a {}'.format(self.data_to_print.shape[0]),
              'échantillons pour entraîner notre modèle et {}'.format(self.data_test.shape[0]),
              'échantiollons de test pour évaluer notre modèle.')

    def get_data(self):
        """
            Getters sur les données
            :return: self.x_train, self.y_train, self.x_test, self.y_test
            """
        return self.X_train, self.y_train, self.X_test, self.y_test

    def data_desc(self):
        """
            fonction qui retourne une description statiques des données
            :return: une description des données
            """
        return self.X_train[['margin20', 'shape20', 'texture20']].describe()

    def class_distribution(self):
        """
            retourne le nombre d'instance dans chaque classe
            :return: les espèces avec leurs tailles respectives
            """
        return self.data_to_print.groupby('species').size()

    def univariante_plot(self):
        """
            affiche un histogramme afin de voir si les données suivent une gaussienne
        """
        data = self.X_train.loc[:, 'texture45']
        sns.distplot(data, hist=True, kde=True, bins=int(len(data) / 20), color='darkblue',
                     hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4})
        plt.show()

    def correlation_matrix(self):
        """
            Affiche la matrice de corrélation entre toutes les données
        """
        set_option('precision', 3)
        x_corr = self.X_train[['margin10', 'shape10', 'texture10', 'margin20', 'shape20', 'texture20']]
        corr = x_corr.corr(method='pearson')
        print(corr)

    def scale(self, X):
        """
            fonction qui transforme les données dans l'intervalle [0,1]
            :param X: les données à transformer
            :return: les données transformées
            """
        return preprocessing.MinMaxScaler().fit_transform(X)

    def visualization(self):
        """
            Affiche des données d'entraînement en plot
            :return:
            """
        vis = self.data_to_print[['id', 'species', 'margin20', 'shape20', 'texture20']]
        sns.pairplot(vis, hue='species', markers='+')
        plt.show()
        print(vis)
