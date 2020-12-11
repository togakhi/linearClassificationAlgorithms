#
from Classifieurs import Classifieurs as clf
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

    print("\n Affichage de l'histogramme...\n")
    data.univariantePlot()

    classifieur = clf(0)

    [x_train, y_train, x_test, y_test] = data.getData()

    classifieur.entrainement(x_train, y_train)

    classifieur.prediction(x_test, y_test)

    #classifieur.loss_calcul(x_test, y_test)
