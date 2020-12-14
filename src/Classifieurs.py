# ==================================================================
#   EL HAYANI MOUSTAMIDE Oumaima
#   CHAAIRAT Tariq
# ==================================================================

# Fichier de gestion des classifieurs

# Explication des différentes méthodes présentes dans ce fichier :


from Cross_Validation import *


class Classifieurs(Cross_Validation):
    """
        Classe Classifieurs qui regroupent les 6 classifieurs utilisés
    """

    def __init__(self):
        super(Classifieurs, self).__init__()
        if self.cross_val:
            # initialisation du KNN Classifier
            self.cross_validation_knn()
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

            # initialisation du LR Classifier
            self.cross_validation_logistic_regression()
            self.lr = LogisticRegression(penalty=self.penalty, C=self.C, tol=self.tol)

            # initialisation du SVM Classifier
            self.cross_validation_svm()
            self.svm = SVC(C=self.C_svm, kernel=self.kernel, degree=3, probability=True)

            # initialisation du Neural Network Classifier
            self.cross_validation_nn()
            self.nn = MLPClassifier(solver=self.solver, learning_rate=self.learning_rate)

            # initialisation du ADA Boost Classifier
            self.cross_validation_ada_boost()
            self.ada = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate_ada)

            # initialisation du Random Forest Classifier
            self.cross_validation_random_forest()
            self.rf = RandomForestClassifier(max_depth=self.max_depths, bootstrap=self.bootstrap)
        else:
            # initialisation du KNN Classifier
            self.knn = KNeighborsClassifier()

            # initialisation du LR Classifier
            self.lr = LogisticRegression()

            # initialisation du SVM Classifier
            self.svm = SVC(probability=True)

            # initialisation du Neural Network Classifier
            self.nn = MLPClassifier()

            # initialisation du AdaBoost Classifier
            self.ada = AdaBoostClassifier()

            # initialisation du Random Forest Classifier
            self.rf = RandomForestClassifier()

    def get_all_classifiers(self):
        """
        fonction qui retourne les 6 classifieurs sous forme de liste
        :return: liste de classifieurs
        """
        return [self.svm, self.knn, self.lr, self.nn, self.ada, self.rf]

    def get_nn_classifiers(self):
        """
        fonction qui retourne le classifieurs nn
        :return: liste de classifieurs
        """
        return self.nn
