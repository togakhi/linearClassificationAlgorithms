from Classifieurs import *
from sklearn import metrics
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
        model.fit(self.scale(self.x_train), self.y_train)
        predict = model.predict(self.scale(self.x_test))
        accuracy = metrics.accuracy_score(self.y_test, predict)
        return accuracy

    def affichAccuracy(self):
        """
        Fonction qui affiche l'accuracy de chaque classifieur
        """
        clf = self.getAllClassifiers()
        accuracy_list = list()
        columns = ['Classifieurs', 'Précision (en %)']
        columns_clf = ['KNN']
        log = pd.DataFrame(columns=columns)
        i=0
        for c in clf:
            print('Accuracy for KNN is ' + str(self.calculateAccuracy(c)))
            #accuracy_list = accuracy_list.append(self.calculateAccuracy(c))
        # for i in range(0, len(clf)):
            log = log.append(pd.DataFrame([[columns_clf[i], 100 * self.calculateAccuracy(c)]], columns=columns),
                       ignore_index=True)
            i += 1

        sns.barplot(x='Précision (en %)', y='Classifieurs', data=log, color='blue', alpha=0.4)
        plt.xlabel('Accuracy (en %)')
        # plt.savefig('accuracy.pdf')
        plt.show()
