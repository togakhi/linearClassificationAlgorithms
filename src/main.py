#
from Calculs_Classifieurs import Calculs_Classifieurs
from Classifieurs import Classifieurs as clf
from Graphique import Graphique
from src.Data_Management import Data_Management as dm


if __name__ == '__main__':
    '''
    Voici le main qui run notre projet
    '''
    data = dm()
    print("\n Affichage des données : \n")
    data.printData()
    data.getShape()

    print("\n Brève description de nos données :\n")
    print(data.dataDescription())

    print("\n Affichage de la distribution dans les classes :\n")
    print(data.classDistribution())

    print("\n Matrice de corrélation :\n")
    data.correlationMatrix()

    # print("\n Affichage de l'histogramme...\n")
    # data.univariantePlot()

    clf = Calculs_Classifieurs()

    print("\n Entrainement des données...\n")
    clf.train()

    print("\n Validation croisée avec les donnèes transformées...\n")
    clf.cross_validation_results(transform=True)

    print("\n Validation croisée sans les donnèes transformées...\n")
    clf.cross_validation_results(transform=False)

    print("\n Prediction des données...\n")
    clf.affich_predict(clf.predict())

    graph = Graphique()
    graph.affichAccuracy()

    #classifieur.loss_calcul(x_test, y_test)
