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
        super().__init__()

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
        self.activation = 'identity'
        self.learning_rate = 'constant'
        self.learning_rate_init = None
        self.n_estimators = None
        self.learning_rate_ada = 1
        self.n_estimators_rf = 100
        self.criterion = 'gini'
        self.max_depths = None
        self.max_features = 'auto'
        self.min_samples_split = 2
        self.min_samples_leaf = 1
        self.bootstrap = True

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

    def crossValidationRandomForest(self):
        """
        Effectue la validation croisée pour le classifieurs Random Forest
        :return:
        """
        print("Recherche des meilleurs paramètres du Random Forest Classifier...")
        n_estimators = [50, 100, 150]
        criterion = ['gini', 'entropy']
        max_depth = [10, 50, 100]
        max_features = ['auto', 'sqrt', 'log2']
        min_samples_split = [1, 2, 3,]
        min_samples_leaf = [0.5, 1, 1.5]
        bootstrap = [True, False]
        grid_param = dict(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                          max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                          bootstrap=bootstrap,)
        model = RandomForestClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.n_estimators_rf = grid_result.best_params_['n_estimators']
        self.criterion = grid_result.best_params_['criterion']
        self.max_depths = grid_result.best_params_['max_depth']
        self.max_features = grid_result.best_params_['max_features']
        self.min_samples_split = grid_result.best_params_['min_samples_split']
        self.min_samples_leaf = grid_result.best_params_['min_samples_leaf']
        self.bootstrap = grid_result.best_params_['bootstrap']

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
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)

        self.n_neighbors = grid_result.best_params_['n_neighbors']

    def crossValidationLogisticRegression(self):
        """
        Effectue la validation croisée pour le classifieurs Logistic Regression
        afin de trouver les meilleurs paramètres
        """
        print("Recherche des meilleurs paramètres du Logistic Regression Classifier...")
        C = [1, 10, 50, 1000, 2000]
        tol = [0.005, 0.003, 0.001]
        grid_param = dict(C=C, tol=tol)
        model = LogisticRegression(solver='newton-cg', multi_class='multinomial')
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=grid_param, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.C = grid_result.best_params_['C']
        self.tol = grid_result.best_params_['tol']

    def crossValidationSVM(self):
        """
        Effectue la validation croisée pour le classifieurs SVM
        :return:
        """
        print("Recherche des meilleurs paramètres du SVM Classifier...")
        C_value = [0.1, 0.5, 0.7, 1.0, 1.3, 1.5, 1.9]
        kernel = ['linear', 'poly', 'rbf', 'sigmoid']
        param_grid = dict(C=C_value, kernel=kernel)
        model = SVC()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.C_svm = grid_result.best_params_['C']
        self.kernel = grid_result.best_params_['kernel']

    def crossValidationNN(self):
        """
        Effectue la validation croisée pour le classifieurs Neural Networks
        :return:
        """
        print("Recherche des meilleurs paramètres du Neural Network Classifier...")
        solver = ['lbfgs', 'sgd', 'adam']
        activation = ['identity', 'logistic', 'tanh', 'relu']
        learning_rate = ['constant', 'invscaling', 'adaptive']
        learning_rate_init = [0.001, 0.002, 0.01]
        param_grid = dict(activation=activation, solver=solver, learning_rate=learning_rate,
                          learning_rate_init=learning_rate_init)
        model = MLPClassifier()
        kfold = kf(n_splits=self.num_folds, random_state=self.seed)
        grid = gridsearchcv(estimator=model, param_grid=param_grid, scoring=self.scoring, cv=kfold)
        grid_result = grid.fit(self.x_data_scale, self.data_y_train)
        self.solver = grid_result.best_params_['solver']
        self.activation = grid_result.best_params_['activation']
        self.learning_rate = grid_result.best_params_['learning_rate']
        self.learning_rate_init = grid_result.best_params_['learning_rate_init']
