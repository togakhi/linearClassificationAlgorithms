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
# imports des libs nécessaires
from matplotlib import pyplot
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
        # récupération des données d'entrainement
        data_train = pd.read_csv("C:/Users/Tariq/PycharmProjects/pythonProject/src/data/train.csv")
        self.data_to_print = data_train

        # récupération données de test
        data_test = pd.read_csv("C:/Users/Tariq/PycharmProjects/pythonProject/src/data/test.csv")

        # on enlève les 2 premières colonnes
        x = data_train.iloc[:, 2:]

        self.data_x_train = x

        # on convertit la colonne species en un dtype = category
        y = data_train['species'].astype('category')
        y = y.cat.codes.to_numpy()

        self.data_y_train = y

        sss = StratifiedShuffleSplit(10, 0.2, random_state=15)

        for train_index, test_index in sss.split(x, y):
            self.x_train, self.x_test = x.iloc[train_index], x.iloc[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]

        le = LabelEncoder().fit(data_train.iloc[:, 1])

        self.className = list(le.classes_)
        # submission_data = pd.read_csv('./data/sample_submission.csv')
        # categories = submission_data.columns.values[1:]
        self.data_test = data_test.iloc[:, 1:]
        self.id = data_test.iloc[:, 0]

    def printData(self):
        """
            Affiche toutes les données d'entraînement
            :return:
            """
        printing = self.data_to_print[['id', 'species', 'margin20', 'shape20', 'texture20']]
        print(printing)

    def getShape(self):
        '''
            Getters sur la forme des données
            :return:
            '''
        print('Il y a {}'.format(self.x_train.shape[0]),
              'échantillons pour entraîner notre modèle et {}'.format(self.x_test.shape[0]),
              'échantiollons de test pour évaluer notre modèle.')

    def getData(self):
        """
            Getters sur les données
            :return: self.x_train, self.y_train, self.x_test, self.y_test
            """
        return self.x_train, self.y_train, self.x_test, self.y_test

    def dataDescription(self):
        """
            fonction qui retourne une description statiques des données
            :return: une description des données
            """
        x = self.x_train[['margin20', 'shape20', 'texture20']]
        return x.describe()

    def classDistribution(self):
        """
            retourne le nombre d'instance dans chaque classe
            :return: les espèces avec leurs tailles respectives
            """
        return self.data_to_print.groupby('species').size()

    def univariantePlot(self):
        """
            affiche un histogramme afin de voir si les données suivent une gaussienne
        """
        data = self.x_train.loc[:, 'texture45']
        sns.distplot(data, hist=True, kde=True, bins=int(len(data) / 20), color='darkblue',
                     hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4})
        pyplot.show()

    def correlationMatrix(self):
        """
            Affiche la matrice de corrélation entre toutes les données
        """
        set_option('precision', 3)
        x_corr = self.x_train[['margin10', 'shape10', 'texture10', 'margin20', 'shape20', 'texture20']]
        corr = x_corr.corr(method='pearson')
        print(corr)

    def pca(self, x):
        """
            projette les données
            :param x: les données à projeter
            :param n_components: le nombre d'élèments à garder
            :return: les données projettées dans une n_components dimension
        """
        pca_model = decomposition.PCA()
        return pca_model.fit_transform(x)

    def scale(self, X):
        """
            fonction qui transforme les données dans l'intervalle [0,1]
            :param X: les données à transformer
            :return: les données transformées
            """
        min_max_scalar = preprocessing.MinMaxScaler()
        x_transform = min_max_scalar.fit_transform(X)
        return x_transform
