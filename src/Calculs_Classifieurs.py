
from Classifieurs import *


class Calculs_Classifieurs(Classifieurs):
    """
    Classe qui gère l'entraînement, le calcul de la prediction, l'accuracy et la f1-score
    """

    def __init__(self):
        super(Calculs_Classifieurs, self).__init__()

    def train(self):
        """
        Fonction qui gère l'entrainement
        :return: None
        """
        print('KNN entrainement...')
        self.knn.fit(self.x_train, self.y_train)

    def predict(self):
        """
        Fonction qui retourne la prediction
        :return: la prediction
        """
        prediction = {str('KNN prediction'): ''}
        for n in range(len(self.data_test)):
            index_class = self.knn.predict([self.data_test.iloc[n]])
            prediction[str('x[') + str(n) + str(']') +
                       str(self.id[n])] = self.className[index_class[0]]

        return prediction

    def affich_predict(self, prediction):
        """
        Fonction qui affiche les predictions ligne par ligne
        :param prediction: les predictions
        """
        for key, value in prediction.items():
            print(key, ' : ', value)

    def cross_validation_results(self, transform=True):
        """
        Fonction qui retourne le score des validations croisées
        :return: le score
        """
        score = {}

        print(' Cross validation pour le classifieur KNN...\n')
        print(self.crossValidationModel(model=self.knn, transform=transform))
        score['KNN accuracy: '] = self.crossValidationModel(model=self.knn, transform=transform)

        return score
