from Classifieurs import *
from sklearn.metrics import accuracy_score, log_loss
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


class Graphique(Classifieurs):
    """
    Classe pour représenter les résultats sous forme de graphique
    """

    def __init__(self):
        super(Graphique, self).__init__()

    def calculateAccuracy(self, model):
        """
        Fonction qui entraine, prédit et calcule l'accuracy
        :param model: the modèle à entrainer
        :return:
        """
        model.fit(self.scale(self.X_train), self.y_train)
        predict = model.predict(self.scale(self.X_test))
        accuracy = accuracy_score(self.y_test, predict)
        return accuracy

    def calculate_Accuracy_From_Cross_Validation(self, model):
        """
        Fonction qui calcule l'accuracy du modèle passé en paramètre
        après la validation croisée du modèle
        :param model: le modèle en question
        :return: l'accuracy du modèle
        """
        return self.crossValidationModel(model=model)

    def affichAccuracy(self):
        """
        Fonction qui affiche l'accuracy de chaque classifieur
        """
        clf = self.getAllClassifiers()
        # accuracy_list = list()
        columns = ['Classifieurs', 'Précision (en %)']
        columns_clf = ['SVM', 'KNN', 'LR', 'NN', 'ADA', 'RF']
        log = pd.DataFrame(columns=columns)
        i = 0
        for c in clf:
            log = log.append(pd.DataFrame([[columns_clf[i], 100 * self.calculateAccuracy(c)]], columns=columns),
                             ignore_index=True)
            i += 1

        sns.barplot(x='Précision (en %)', y='Classifieurs', data=log, color='black', alpha=0.4)
        plt.xlabel('Accuracy (en %)')
        # plt.savefig('accuracy.pdf')
        plt.show()
