#
from Calculs_Classifieurs import Calculs_Classifieurs
from Classifieurs import Classifieurs as clf
from Graphique import Graphique
from src.Data_Management import Data_Management as dm

if __name__ == '__main__':
    '''
    Voici le main qui run notre projet
    '''

    print("============================================================\n"
          "\t\t\tPARTIE 1 : PREPARATION DES DONNEES\n"
          "============================================================")
    data = dm()
    print("\nAffichage des données : \n")
    data.printData()
    data.getShape()

    print("\nBrève description de nos données :\n")
    print(data.dataDescription())

    print("\nAffichage de la distribution dans les classes :\n")
    print(data.classDistribution())

    print("\nMatrice de corrélation :\n")
    data.correlationMatrix()

    print("\n\n")
    print("============================================================\n"
          "\t\t\tPARTIE 2 : LES CLASSIFIEURS\n"
          "============================================================")
    # print("\n Affichage de l'histogramme...\n")
    # data.univariantePlot()

    # clf = Calculs_Classifieurs()
    graph = Graphique()
    # print("\nEntrainement des données...\n")
    # clf.train()

    print("\nAffichage de l'accuracy sans validation croisée...\n")
    graph.affich_accuracy()
    print("\nAffichage de loss sans validation croisée...\n")
    graph.affich_loss()
    print("\nAffichage de f1_score sans validation croisée...\n")
    graph.affich_f1_score()
    # print("\nValidation croisée avec les donnèes transformées...\n")
    # print(clf.cross_validation_results())

    # print("\n Prediction des données...\n")
    # clf.affich_predict(clf.predict())

    # print("\nAffichage de l'accuracy avec validation croisée...\n")
    # graph.affichAccuracy()

    # classifieur.loss_calcul(x_test, y_test)
