from sklearn.model_selection import GridSearchCV as gridsearchcv, KFold as kf, cross_val_score as cvs, \
    cross_val_predict as cvp
from Data_Management import *

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class Cross_Validation(Data_Management):
    """
    Classe permettant de faire la validation croisée pour tout nos classifieurs
    """

    def __init__(self):
        super().__init__()

        # initialisation de paramètres pour les validation croisée
        self.x_data_scale = self.scale(self.x_train)
        self.num_folds = 10
        self.seed = 7
        self.score = 'accuracy'

        # initialisation des paramètres pour tout les classifiers
        print("Initialisation des paramètres : ")
        print("ADA Boost Classifier - None")
        print("Random Forest Classifier")
        print("K Nearest Neighbour Classifier")
        self.n_neighbors = None
        print("KNN - Done")
        print("Logistic Regression Classifier")
        print("SVM Classifier")
        print("Neural Networks Classifier")

    def crossValidationModel(self, model=None, transform=False):
        """
        Fonction permettant d'augmenter la précision du modèle
        passé en paramètre à l'aide d'une validation croisée
        :param model: Le modèle
        :param transform: booléen pour savoir si on doit scale les données
        :return:
        """
        if transform:
            kfold = kf(n_splits=self.num_folds, random_state=self.seed)
            precision = cvs(model, self.scale(self.data_x_train), self.data_y_train, cv=kfold, scoring=self.score)
        else:
            kfold = kf(n_splits=self.num_folds, random_state=self.seed)
            precision = cvs(model, self.data_x_train, self.data_y_train, cv=kfold, scoring=self.score)
        return precision.mean()

    def crossValidationAdaBoost(self):
        """
        Effectue la validation croisée pour le classifieurs AdaBoost
        :return:
        """
        pass

    def crossValidationRandomForest(self):
        """
        Effectue la validation croisée pour le classifieurs Random Forest
        :return:
        """
        pass

    def crossValidationKNN(self):
        """
        Effectue la validation croisée pour le classifieurs K Nearest Neighbour
        :return:
        """
        neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        grid_param = dict(n_neighbors=neighbors)
        model = KNeighborsClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.score, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.y_train)

        self.n_neighbors = grid_result.best_params_['n_neighbors']

    def crossValidationLogisticRegression(self):
        """
        Effectue la validation croisée pour le classifieurs Logistic Regression
        :return:
        """

        pass

    def crossValidationSVM(self):
        """
        Effectue la validation croisée pour le classifieurs SVM
        :return:
        """
        pass

    def crossValidationNN(self):
        """
        Effectue la validation croisée pour le classifieurs Neural Networks
        :return:
        """
        slvr = ['lbfgs', 'sgd', 'adam']

        pass
