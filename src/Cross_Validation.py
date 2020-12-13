from sklearn.model_selection import GridSearchCV as gridsearchcv, KFold as kf, cross_val_score as cvs, \
    cross_val_predict as cvp
from sklearn.neural_network import MLPClassifier

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
        self.x_data_scale = self.scale(self.data_X_train)
        self.num_folds = 10
        self.seed = 7
        self.scoring = 'accuracy'

        # initialisation des paramètres pour tout les classifiers
        self.n_neighbors = None
        self.penalty = 'l2'
        self.tol = None
        self.C = None
        self.C_svm = None
        self.kernel = None
        self.solver = 'lbfgs'
        self.activation = 'identity'
        self.learning_rate = 'constant'
        self.learning_rate_init = None

    def crossValidationModel(self, model=None):
        """
        Fonction permettant d'augmenter la précision du modèle
        passé en paramètre à l'aide d'une validation croisée
        :param model: Le modèle
        :return: la moyenne de la précision
        """
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        precision = cvs(model, self.x_data_scale, self.data_y_train, cv=kfold, scoring=self.scoring)
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
        afin de trouver les meilleurs paramètres
        """
        print("Recherche des meilleurs paramètres du K Nearest Neighbour Classifier...")
        neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
        grid_param = dict(n_neighbors=neighbors)
        model = KNeighborsClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.y_train)

        self.n_neighbors = grid_result.best_params_['n_neighbors']

    def crossValidationLogisticRegression(self):
        """
        Effectue la validation croisée pour le classifieurs Logistic Regression
        afin de trouver les meilleurs paramètres
        """
        print("Recherche des meilleurs paramètres du Logistic Regression Classifier...")
        C = [1, 10, 20, 50, 1000, 2000]
        tol = [0.005, 0.003, 0.001]
        grid_param = dict(C=C, tol=tol)
        model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.y_train)
        c = grid_result.best_params_['C']
        tol = grid_result.best_params_['tol']
        self.C = c
        self.tol = tol

    def crossValidationSVM(self):
        """
        Effectue la validation croisée pour le classifieurs SVM
        :return:
        """
        print("Recherche des meilleurs paramètres du SVM Classifier...")
        C_value = [0.1, 0.5, 0.7, 1.0, 1.3, 1.7, 2]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        param_grid = dict(C=C_value, kernel=kernel)
        model = SVC()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.y_train)
        C = grid_result.best_params_['C']
        kernel = grid_result.best_params_['kernel']
        self.C_svm = C
        self.kernel = kernel

    def crossValidationNN(self):
        """
        Effectue la validation croisée pour le classifieurs Neural Networks
        :return:
        """
        solver = ['lbfgs', 'sgd', 'adam']
        activation = ['identity', 'logistic', 'tanh', 'relu']
        learning_rate = ['constant', 'invscaling', 'adaptive']
        learning_rate_init = [0.001, 0.002, 0.01]
        param_grid = dict(activation=activation, solver=solver, learning_rate=learning_rate,
                          learning_rate_init=learning_rate_init)
        model = MLPClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.y_train)
        solver = grid_result.best_params_['solver']
        activation = grid_result.best_params_['activation']
        learning_rate = grid_result.best_params_['learning_rate']
        learning_rate_init = grid_result.best_params_['learning_rate_init']
        self.learning_rate_init = learning_rate_init
        self.solver = solver
        self.activation = activation
        self.learning_rate = learning_rate
