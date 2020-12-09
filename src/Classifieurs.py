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

classifiers = [
    AdaBoostClassifier(),
    RandomForestClassifier(),
    Perceptron(),
    LogisticRegression(),
    SVC(probability=True)
    ]


class Classifieurs():
    """
        Classe Classifieurs qui regroupent les 6 classifieurs utilisés
        Si le numéro du classifieur est 0 => on utilise tout les clfs
        Si le numéro du classifieur est 1 => on utilise le classifieur AdaBoost
        Si le numéro du classifieur est 2 => on utilise le classifieur Random Forest
        Si le numéro du classifieur est 3 => on utilise le classifieur Perceptron
        Si le numéro du classifieur est 4 => on utilise le classifieur Logistic Regression
        Si le numéro du classifieur est 5 => on utilise le classifieur SVM
        Si le numéro du classifieur est 6 => on utilise le classifieur The last [à modifier]

    """
    # TODO Modifier commmentaire en haut

    def __init__(self, modele):
        self.nb_clf = modele

    def entrainement(self, x_train, y_train):
        """
            Fonction qui sert à entraîner les données
        """
        # TODO Modifier commmentaires en haut

        if self.nb_clf == 0 :
            print("All Classifiers - training")
            for clf in classifiers:
                clf.fit(x_train, y_train)
        else:
            if self.nb_clf == 1:
                # Classifieur Adaboost
                print("ADA Boost Classifier - training")
                clf = classifiers[0]
            elif self.nb_clf == 2:
                print("Random Forest Classifier - training")
                clf = classifiers[1]
            elif self.nb_clf == 3:
                print("Perceptron Classifier - training")
                clf = classifiers[2]
            elif self.nb_clf == 4:
                print("Logistic Regression Classifier - training")
                clf = classifiers[3]
            elif self.nb_clf == 5:
                print("SVM Classifier - training")
                clf = classifiers[4]
            elif self.nb_clf == 6:
                pass
                # TODO The last classifier
                # print("The last Classifier - training")
                # clf = classifiers[4]
        clf.fit(x_train, y_train)

    def prediction(self, x_test, y_test):
        """
            Fonction qui sert à calculer l'accuracy
        """
        if self.nb_clf == 0 :
            print("All Classifiers - prediction")
            for clf in classifiers:
                # prediction
                predictions_from_train_data = clf.predict(x_test)
                # accuracy
                accuracy = accuracy_score(y_test, predictions_from_train_data)
                print("Affichage de l'accuracy {:.4%}", format(accuracy))
        else:
            if self.nb_clf == 1:
                # Classifieur Adaboost
                print("ADA Boost Classifier - prediction")
                clf = classifiers[0]
            elif self.nb_clf == 2:
                print("Random Forest Classifier - prediction")
                clf = classifiers[1]
            elif self.nb_clf == 3:
                print("Perceptron Classifier - prediction")
                clf = classifiers[2]
            elif self.nb_clf == 4:
                print("Logistic Regression Classifier - prediction")
                clf = classifiers[3]
            elif self.nb_clf == 5:
                print("SVM Classifier - prediction")
                clf = classifiers[4]
            elif self.nb_clf == 6:
                pass
                # TODO The last classifier
                # print("The last Classifier - prediction")
                # clf = classifiers[4]
            # prediction
            predictions_from_train_data = clf.predict(x_test)
            # accuracy
            accuracy = accuracy_score(y_test, predictions_from_train_data)
            print("Affichage de l'accuracy {:.4%}", format(accuracy))

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


