# ==================================================================
#   EL HAYANI MOUSTAMIDE Oumaima
#   CHAAIRAT Tariq
# ==================================================================

# Fichier de gestion des classifieurs

# Explication des différentes méthodes présentes dans ce fichier :
from sklearn.metrics import accuracy_score, log_loss
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import Data_Management as dm
from Cross_Validation import *

class Classifieurs(Cross_Validation):
    """
        Classe Classifieurs qui regroupent les 6 classifieurs utilisés
    """

    # TODO Modifier commmentaire en haut

    def __init__(self):
        super(Classifieurs, self).__init__()

        # initialisation du KNN Classifier
        self.crossValidationKNN()
        self.knn = KNeighborsClassifier(n_neighbors=self.n_neighbors)

    def getAllClassifiers(self):
        """
        fonction qui retourne les 6 classifieurs sous forme de liste
        :return: liste de classifieurs
        """
        return [self.knn]

    # def entrainement(self, x_train, y_train):
    #     """
    #         Fonction qui sert à entraîner les données
    #     """
    #
    #     if self.nb_clf == 0 :
    #         print("All Classifiers - training")
    #         for clf in classifiers:
    #             clf.fit(x_train, y_train)
    #     else:
    #         if self.nb_clf == 1:
    #             # Classifieur Adaboost
    #             print("ADA Boost Classifier - training")
    #             clf = classifiers[0]
    #         elif self.nb_clf == 2:
    #             print("Random Forest Classifier - training")
    #             clf = classifiers[1]
    #         elif self.nb_clf == 3:
    #             print("Perceptron Classifier - training")
    #             clf = classifiers[2]
    #         elif self.nb_clf == 4:
    #             print("Logistic Regression Classifier - training")
    #             clf = classifiers[3]
    #         elif self.nb_clf == 5:
    #             print("SVM Classifier - training")
    #             clf = classifiers[4]
    #         elif self.nb_clf == 6:
    #             pass
    #             # TODO The last classifier
    #             # print("The last Classifier - training")
    #             # clf = classifiers[4]
    #     clf.fit(x_train, y_train)
    #
    # def prediction(self, x_test, y_test):
    #     """
    #         Fonction qui sert à calculer l'accuracy
    #     """
    #     if self.nb_clf == 0 :
    #         print("All Classifiers - prediction")
    #         for clf in classifiers:
    #             # prediction
    #             predictions_from_train_data = clf.predict(x_test)
    #             # accuracy
    #             accuracy = accuracy_score(y_test, predictions_from_train_data)
    #             print("Affichage de l'accuracy {:.4%}", format(accuracy))
    #     else:
    #         if self.nb_clf == 1:
    #             # Classifieur Adaboost
    #             print("ADA Boost Classifier - prediction")
    #             clf = classifiers[0]
    #         elif self.nb_clf == 2:
    #             print("Random Forest Classifier - prediction")
    #             clf = classifiers[1]
    #         elif self.nb_clf == 3:
    #             print("Perceptron Classifier - prediction")
    #             clf = classifiers[2]
    #         elif self.nb_clf == 4:
    #             print("Logistic Regression Classifier - prediction")
    #             clf = classifiers[3]
    #         elif self.nb_clf == 5:
    #             print("SVM Classifier - prediction")
    #             clf = classifiers[4]
    #         elif self.nb_clf == 6:
    #             pass
    #             # TODO The last classifier
    #             # print("The last Classifier - prediction")
    #             # clf = classifiers[4]
    #         # prediction
    #         predictions_from_train_data = clf.predict(x_test)
    #         # accuracy
    #         accuracy = accuracy_score(y_test, predictions_from_train_data)
    #         print("Affichage de l'accuracy {:.4%}", format(accuracy))

    # def loss_calcul(self, x_test, y_test):
    #     """
    #         Fonction qui sert à calculer la perte
    #     """
    #     if self.nb_clf == 0 :
    #         print("All Classifiers - loss")
    #         for clf in classifiers:
    #             # prediction_proba
    #             predictions_from_train_data = clf.predict_proba(x_test)
    #
    #             # loss
    #             loss = log_loss(y_test, predictions_from_train_data)
    #
    #             print("Affichage de la loss {:.4%}", format(loss))
    #     else:
    #         if self.nb_clf == 1:
    #             # Classifieur Adaboost
    #             print("ADA Boost Classifier - loss")
    #             clf = classifiers[0]
    #             # prediction_proba
    #             predictions_from_train_data = clf.predict_proba(x_test)
    #         elif self.nb_clf == 2:
    #             print("Random Forest Classifier - loss")
    #             clf = classifiers[1]
    #             # prediction_proba
    #             predictions_from_train_data = clf.predict_proba(x_test)
    #         elif self.nb_clf == 3:
    #             print("Perceptron Classifier - loss")
    #             clf = classifiers[2]
    #         elif self.nb_clf == 4:
    #             print("Logistic Regression Classifier - loss")
    #             clf = classifiers[3]
    #         elif self.nb_clf == 5:
    #             pass
    #             # print("SVM Classifier - loss")
    #             # clf = classifiers[4]
    #         elif self.nb_clf == 6:
    #             pass
    #             # TODO The last classifier
    #             # print("The last Classifier - loss")
    #             # clf = classifiers[4]
    #
    #
    #         # loss
    #         loss = log_loss(y_test, predictions_from_train_data)
    #
    #         print("Affichage de la loss {:.4%}", format(loss))
