from Classifieurs import *
from sklearn.metrics import accuracy_score, log_loss, f1_score
from sklearn.utils.multiclass import unique_labels
import matplotlib.pyplot as plt


class Graphique(Classifieurs):
    """
    Classe pour représenter les résultats sous forme de graphique
    """

    def __init__(self):
        super(Graphique, self).__init__()

    def calculate_accuracy(self, model, scaled_data=True):
        """
        Fonction qui entraine, prédit et calcule l'accuracy
        :param scaled_data:
        :param model: the modèle à entrainer
        :return:
        """
        if scaled_data:
            model.fit(self.scale(self.X_train), self.y_train)
            predict = model.predict(self.scale(self.X_test))
        else:
            model.fit(self.X_train, self.y_train)
            predict = model.predict(self.X_test)
        return accuracy_score(self.y_test, predict)

    def calculate_loss(self, model, scaled_data=True):
        """
        Fonction qui entraine, prédit et calcule la loss
        :param scaled_data:
        :param model: the modèle à entrainer
        :return:
        """
        if scaled_data:
            model.fit(self.scale(self.X_train), self.y_train)
            predict_proba = model.predict_proba(self.scale(self.X_test))
        else:
            model.fit(self.X_train, self.y_train)
            predict_proba = model.predict_proba(self.X_test)
        return log_loss(self.y_test, predict_proba)

    def calculate_f1_score(self, model, scaled_data=True):
        """
        Fonction qui entraine, prédit et calcule la f1_score
        :param scaled_data:
        :param model: the modèle à entrainer
        :return:
        """
        if scaled_data:
            model.fit(self.scale(self.X_train), self.y_train)
            predict = model.predict(self.scale(self.X_test))
        else:
            model.fit(self.X_train, self.y_train)
            predict = model.predict(self.X_test)
        return f1_score(self.y_test, predict, average='macro')

    def calculate_accuracy_from_cross_validation(self, model):
        """
        Fonction qui calcule l'accuracy du modèle passé en paramètre
        après la validation croisée du modèle
        :param model: le modèle en question
        :return: l'accuracy du modèle
        """
        return self.crossValidationModel(model=model)

    def affich_accuracy(self):
        """
        Fonction qui affiche l'accuracy de chaque classifieur
        """
        clf = self.get_all_classifiers()
        columns = ['Classifieurs', 'Précision (en %)']
        columns_clf = ['SVM', 'KNN', 'LR', 'NN', 'ADA', 'RF']
        log = pd.DataFrame(columns=columns)
        log_data_not_scale = pd.DataFrame(columns=columns)
        i = 0
        for c in clf:
            log = log.append(pd.DataFrame([[columns_clf[i], 100 * self.calculate_accuracy(c)]], columns=columns),
                             ignore_index=True)
            log_data_not_scale = log_data_not_scale.append(
                pd.DataFrame([[columns_clf[i], 100 * self.calculate_accuracy(c, False)]], columns=columns),
                ignore_index=True)
            i += 1

        sns.barplot(x='Précision (en %)', y='Classifieurs', data=log, color='black', alpha=0.4)
        sns.barplot(x='Précision (en %)', y='Classifieurs', data=log_data_not_scale, color='blue', alpha=0.4)
        if self.cross_val:
            plt.xlabel('Accuracy (en %) avec validation croisée')
            plt.savefig('accuracy_with_cross_val.pdf')
        else:
            plt.xlabel('Accuracy (en %) sans validation croisée')
            plt.savefig('accuracy_without_cross_val.pdf')
        plt.show()

    def affich_loss(self):
        """
        Fonction qui affiche la loss de chaque classifieur
        """
        clf = self.get_all_classifiers()
        columns = ['Classifieurs', 'Perte']
        columns_clf = ['SVM', 'KNN', 'LR', 'NN', 'ADA', 'RF']
        log = pd.DataFrame(columns=columns)
        log_data_not_scale = pd.DataFrame(columns=columns)
        i = 0
        for c in clf:
            log = log.append(pd.DataFrame([[columns_clf[i], self.calculate_loss(c)]], columns=columns),
                             ignore_index=True)
            log_data_not_scale = log_data_not_scale.append(
                pd.DataFrame([[columns_clf[i], self.calculate_loss(c, False)]], columns=columns),
                ignore_index=True)
            i += 1

        sns.barplot(x='Perte', y='Classifieurs', data=log, color='r', alpha=0.4)
        sns.barplot(x='Perte', y='Classifieurs', data=log_data_not_scale, color='r', alpha=0.4)

        if self.cross_val:
            plt.xlabel('Log loss avec validation croisée')
            plt.savefig('loss_with_cross_val.pdf')
        else:
            plt.xlabel('Log loss sans validation croisée')
            plt.savefig('loss_without_cross_val.pdf')
        plt.show()

    def affich_f1_score(self):
        """
        Fonction qui affiche la f1_score de chaque classifieur
        """
        clf = self.get_all_classifiers()
        columns = ['Classifieurs', 'F1_score']
        columns_clf = ['SVM', 'KNN', 'LR', 'NN', 'ADA', 'RF']
        log = pd.DataFrame(columns=columns)
        log_data_not_scale = pd.DataFrame(columns=columns)
        i = 0
        for c in clf:
            log = log.append(pd.DataFrame([[columns_clf[i], 100 * self.calculate_f1_score(c)]], columns=columns),
                             ignore_index=True)
            log_data_not_scale = log_data_not_scale.append(
                pd.DataFrame([[columns_clf[i], 100 * self.calculate_f1_score(c, False)]], columns=columns),
                ignore_index=True)
            i += 1

        sns.barplot(x='F1_score', y='Classifieurs', data=log, color='g', alpha=0.4)
        sns.barplot(x='F1_score', y='Classifieurs', data=log_data_not_scale, color='g', alpha=0.4)
        if self.cross_val:
            plt.xlabel('F1_score avec validation croisée')
            plt.savefig('f1_score_with_cross_val.pdf')
        else:
            plt.xlabel('F1_score sans validation croisée')
            plt.savefig('f1_score_without_cross_val.pdf')
        plt.show()
