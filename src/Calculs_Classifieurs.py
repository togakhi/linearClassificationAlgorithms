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
        x_train = self.scale(self.X_train)
        y_train = self.y_train
        print('SVM entrainement...')
        self.svm.fit(x_train, y_train)
        print('KNN entrainement...')
        self.knn.fit(x_train, y_train)
        print('LR entrainement...')
        self.lr.fit(x_train, y_train)
        print('NN entrainement...')
        self.nn.fit(x_train, y_train)
        print('AdaBoost entrainement...')
        self.ada.fit(x_train, y_train)
        print('Random Forest entrainement...')
        self.rf.fit(x_train, y_train)

    def predict(self):
        """
        Fonction qui retourne la prediction
        :return: la prediction
        """
        X_test = self.scale(self.data_test)
        prediction = {str('SVM prediction'): ''}
        for n in range(len(self.data_test)):
            class_index = self.svm.predict([X_test.iloc[n]])
            prediction[str('x[') + str(n) + str(']') +
                       str(self.id[n])] = self.className[class_index[0]]
        prediction = {str('KNN prediction'): ''}
        for n in range(len(self.data_test)):
            class_index = self.knn.predict([X_test.iloc[n]])
            prediction[str('x[') + str(n) + str(']') +
                       str(self.id[n])] = self.className[class_index[0]]
        prediction = {str('LR prediction'): ''}
        for n in range(len(self.data_test)):
            class_index = self.lr.predict([X_test.iloc[n]])
            prediction[str('x[') + str(n) + str(']') +
                       str(self.id[n])] = self.className[class_index[0]]
        prediction = {str('NN prediction'): ''}
        for n in range(len(self.data_test)):
            class_index = self.nn.predict([X_test.iloc[n]])
            prediction[str('x[') + str(n) + str(']') +
                       str(self.id[n])] = self.className[class_index[0]]
        prediction = {str('ADA prediction'): ''}
        for n in range(len(self.data_test)):
            class_index = self.ada.predict([X_test.iloc[n]])
            prediction[str('x[') + str(n) + str(']') +
                       str(self.id[n])] = self.className[class_index[0]]
        prediction = {str('RF prediction'): ''}
        for n in range(len(self.data_test)):
            class_index = self.rf.predict([X_test.iloc[n]])
            prediction[str('x[') + str(n) + str(']') +
                       str(self.id[n])] = self.className[class_index[0]]

        return prediction

    def affich_predict(self, prediction):
        """
        Fonction qui affiche les predictions ligne par ligne
        :param prediction: les predictions
        """
        for key, value in prediction.items():
            print(key, ' : ', value)

    def cross_validation_results(self):
        """
        Fonction qui retourne le score des validations croisées
        :return: le score
        """
        score = {}

        print('Cross validation pour le classifieur SVM...\n')
        print(self.crossValidationModel(model=self.lr))
        score['SVM accuracy: '] = self.crossValidationModel(model=self.svm)

        print('Cross validation pour le classifieur KNN...\n')
        print(self.crossValidationModel(model=self.knn))
        score['KNN accuracy: '] = self.crossValidationModel(model=self.knn)

        print('Cross validation pour le classifieur LR...\n')
        print(self.crossValidationModel(model=self.lr))
        score['LR accuracy: '] = self.crossValidationModel(model=self.lr)

        print('Cross validation pour le classifieur NN...\n')
        print(self.crossValidationModel(model=self.nn))
        score['NN accuracy: '] = self.crossValidationModel(model=self.nn)

        print('Cross validation pour le classifieur ADA...\n')
        print(self.crossValidationModel(model=self.ada))
        score['ADA accuracy: '] = self.crossValidationModel(model=self.ada)

        print('Cross validation pour le classifieur RF...\n')
        print(self.crossValidationModel(model=self.rf))
        score['RF accuracy: '] = self.crossValidationModel(model=self.rf)

        return score
