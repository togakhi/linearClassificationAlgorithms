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

    def __init__(self, cross_val=False):
        super(Classifieurs, self).__init__()
        self.cross_val = cross_val
        if self.cross_val:
            # initialisation du KNN Classifier
            self.crossValidationKNN()
            self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

            # initialisation du LR Classifier
            self.crossValidationLogisticRegression()
            self.lr = LogisticRegression(penalty=self.penalty, C=self.C, tol=self.tol)

            # initialisation du SVM Classifier
            self.crossValidationSVM()
            self.svm = SVC(C=self.C_svm, kernel=self.kernel, degree=3, probability=True)

            # initialisation du Neural Network Classifier
            self.crossValidationNN()
            self.nn = MLPClassifier(activation=self.activation, solver=self.solver,
                                    learning_rate=self.learning_rate, learning_rate_init=self.learning_rate_init)

            # initialisation du ADA Boost Classifier
            self.crossValidationAdaBoost()
            self.ada = AdaBoostClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate_ada)

            # initialisation du Random Forest Classifier
            self.crossValidationRandomForest()
            self.ada = RandomForestClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate_ada)
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
