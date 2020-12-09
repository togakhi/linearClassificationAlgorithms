# ==================================================================
#   EL HAYANI MOUSTAMIDE Oumaima
#   CHAAIRAT Tariq
# ==================================================================

# Fichier de gestion des données issues de la BDD du lien kaggle suivant :
# https://www.kaggle.com/c/leaf-classification

# Explication des différentes méthodes présentes dans ce fichier :


# imports des libs nécessaires
import PIL.Image
import cv2
import numpy as np
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import decomposition
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
import warnings

warnings.filterwarnings("ignore")
from pandas import set_option


class Data_Management:
    '''
        Classe DataManagement permettant de préparer les données
    '''

    def __init__(self, n_component=10):
        """
            Importation des données de la BDD leaf
            :param file_name_train: données d'entrainement
            :param file_name_test: données de test
            :param n_component:
        """
        # données d'entrainement
        data_train = pd.read_csv("C:/Users/Tariq/PycharmProjects/pythonProject/src/data/train.csv")
        self.data_to_print = data_train

        # données de test
        data_test = pd.read_csv("C:/Users/Tariq/PycharmProjects/pythonProject/src/data/test.csv")
        self.data_x_train = data_train.iloc[:, 2:]
        x = data_train.iloc[:, 2:]
        y = data_train['species'].astype('category')
        # y = y.cat.codes.as_matrix()
        self.data_y_train = y
        sss = StratifiedShuffleSplit(10, 0.2, random_state=15)
        for train_index, test_index in sss.split(x, y):
            self.x_train, self.x_test = x.iloc[train_index], x.iloc[test_index]
            self.y_train, self.y_test = y[train_index], y[test_index]
        le = LabelEncoder().fit(data_train.iloc[:, 1])
        self.className = list(le.classes_)
        # submission_data = pd.read_csv('./data/sample_submission.csv')
        # categories = submission_data.columns.values[1:]
        # self.x_test = data_test.iloc[:, 1:]
        self.id = data_test.iloc[:, 0]

    def printSomeData(self):
        """
            Affiche des données
            :return:
            """
        printing = self.data_to_print[['id', 'species', 'margin20', 'shape20', 'texture20']]
        print(printing)

    def getShape(self):
        '''
            Affiche la colonne shape des données
            :return:
            '''
        print(self.data_to_print.shape)

    def getData(self, showData=True, n=5):
        """
            Accesseurs sur les données
            :param n: Le nombre de lignes à afficher
            :param showData: si showData à True, on affiche les données
            :return: self.X_train, self.y_train, self.X_test, self.y_test
            """
        if showData:
            print('data_train:/n ')
            print(self.data_x_train.head(n))
            print('/n/n')
            print('data_test:/n')
            print(self.data_y_train.head(n))
        return self.data_x_train, self.data_y_train, self.x_test, self.y_test

    def dataDescription(self):
        """
            function for the statistic description of the data
            This function can suggest us to stardardize our data when the mean are differents
            :return: the described data: mean std etc.
            """
        x = self.x_train[['margin20', 'shape20', 'texture20']]
        description = x.describe()
        return description

    def classDistribution(self):
        """
            retourne le nombre d'instance dans chaque classe
            :return: les espèces avec leurs tailles respectives
            """
        return self.data_to_print.groupby('species').size()

    def univariantePlot(self):
        """
            since some of the algo suppose data follow gaussian distribution
            we use histogramme to see if some features follow in average gaussian distribution
            :return: histrogram figure
            """
        data = self.x_train.loc[:, 'texture45']
        sns.distplot(data, hist=True, kde=True, bins=int(len(data) / 20), color='darkblue',
                     hist_kws={'edgecolor': 'black'}, kde_kws={'linewidth': 4})
        pyplot.show()

    def correlationMatrix(self):
        """
            Teste la corrélation entre les différentes features
            :return: la matrice de corrélation
            """
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.x_train.corr(), vmin=-1, vmax=1, interpolation='none')
        fig.colorbar(cax)
        pyplot.show()

    def correlation(self):
        set_option('precision', 3)
        x_corr = self.x_train[['margin10', 'shape10', 'texture10', 'margin20', 'shape20', 'texture20']]
        corr = x_corr.corr(method='pearson')
        print(corr)

    def pca(self, x):
        """
            projette les données
            :param X: les données à projeter
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


def leaf_image(image_id, target_length=160):
    """
    `image_id` should be the index of the images in the images/ folder

    Reture the images of a given id(1~1584) with the target size (target_length x target_length)

    """

    image_name = str(image_id) + '.jpg'
    leaf_img = plt.imread("C:/Users/Tariq/PycharmProjects/pythonProject/src/images/343.png")  # Reading in the images
    leaf_img_width = leaf_img.shape[1]
    leaf_img_height = leaf_img.shape[0]
    # target_length = 160
    show_img = PIL.Image.open("C:/Users/Tariq/PycharmProjects/pythonProject/src/images/343.png")
    show_img.show()
    img_target = np.zeros((target_length, target_length), np.uint8)
    if leaf_img_width >= leaf_img_height:
        scale_img_width = target_length
        scale_img_height = int((float(scale_img_width) / leaf_img_width) * leaf_img_height)
        img_scaled = cv2.resize(leaf_img, (scale_img_width, scale_img_height), interpolation=cv2.INTER_AREA)
        copy_location = (target_length - scale_img_height) / 2
        img_target[int(copy_location):int(copy_location + scale_img_height), :] = img_scaled
    else:
        # leaf_img_width < leaf_img_height:
        scale_img_height = target_length
        print(scale_img_height)
        scale_img_width = int((float(scale_img_height) / leaf_img_height) * leaf_img_width)
        img_scaled = cv2.resize(leaf_img, (scale_img_width, scale_img_height), interpolation=cv2.INTER_AREA)
        copy_location = (target_length - scale_img_width) / 2
        print(copy_location)
        print(scale_img_width)
        # img_target[:, int(copy_location):copy_location + scale_img_width] = img_scaled

    return img_scaled
