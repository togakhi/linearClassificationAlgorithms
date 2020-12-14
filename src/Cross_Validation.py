from sklearn.ensemble import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV as gridsearchcv, KFold as kf, cross_val_score as cvs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from Data_Management import *


class Cross_Validation(Data_Management):
    """
    Classe permettant de faire la validation croisée pour tout nos classifieurs
    """

    def __init__(self, scaled_data=True):
        super(Cross_Validation, self).__init__()
        # initialisation de paramètres pour les validation croisée
        if scaled_data:
            self.x_data_scale = self.scale(self.data_X_train)
        else:
            self.x_data_scale = self.data_X_train

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
        self.learning_rate = 'constant'
        self.n_estimators = None
        self.learning_rate_ada = 1
        self.max_depths = None
        self.bootstrap = True



    def cross_validation_ada_boost(self):
        """
        Effectue la validation croisée pour le classifieurs AdaBoost
        """
        print("Recherche des meilleurs paramètres du AdaBoost Classifier...")
        n_estimators = [25, 50, 75]
        learning_rate_ada = [0.5, 1, 1.5]
        grid_param = dict(n_estimators=n_estimators, learning_rate=learning_rate_ada)
        model = AdaBoostClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.n_estimators = grid_result.best_params_['n_estimators']
        self.learning_rate_ada = grid_result.best_params_['learning_rate']
        print("...Done")

    def cross_validation_random_forest(self):
        """
        Effectue la validation croisée pour le classifieurs Random Forest
        """
        print("Recherche des meilleurs paramètres du Random Forest Classifier...")
        max_depth = [10, 50, 100]
        bootstrap = [True, False]
        grid_param = dict(max_depth=max_depth, bootstrap=bootstrap)
        model = RandomForestClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.max_depths = grid_result.best_params_['max_depth']
        self.bootstrap = grid_result.best_params_['bootstrap']
        print("...Done")

    def cross_validation_knn(self):
        """
        Effectue la validation croisée pour le classifieurs K Nearest Neighbour
        afin de trouver les meilleurs paramètres
        """
        print("Recherche des meilleurs paramètres du K Nearest Neighbour Classifier...")
        neighbors = [1, 3, 5, 7, 9]
        grid_param = dict(n_neighbors=neighbors)
        model = KNeighborsClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)

        self.n_neighbors = grid_result.best_params_['n_neighbors']
        print("...Done")

    def cross_validation_logistic_regression(self):
        """
        Effectue la validation croisée pour le classifieurs Logistic Regression
        afin de trouver les meilleurs paramètres
        """
        print("Recherche des meilleurs paramètres du Logistic Regression Classifier...")
        C = [1, 10, 50]
        tol = [0.005, 0.003, 0.001]
        grid_param = dict(C=C, tol=tol)
        model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.C = grid_result.best_params_['C']
        self.tol = grid_result.best_params_['tol']
        print("...Done")

    def cross_validation_svm(self):
        """
        Effectue la validation croisée pour le classifieurs SVM
        """
        print("Recherche des meilleurs paramètres du SVM Classifier...")
        C_value = [0.5, 1, 1.5]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        param_grid = dict(C=C_value, kernel=kernel)
        model = SVC()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.C_svm = grid_result.best_params_['C']
        self.kernel = grid_result.best_params_['kernel']
        print("...Done")

    def cross_validation_nn(self):
        """
        Effectue la validation croisée pour le classifieurs Neural Networks
        """
        print("Recherche des meilleurs paramètres du Neural Network Classifier...")
        solver = ['lbfgs', 'sgd', 'adam']
        learning_rate = ['constant', 'invscaling', 'adaptive']
        param_grid = dict(solver=solver, learning_rate=learning_rate)
        model = MLPClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.solver = grid_result.best_params_['solver']
        self.learning_rate = grid_result.best_params_['learning_rate']
        print("...Done")
