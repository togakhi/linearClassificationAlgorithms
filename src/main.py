from Cross_Validation import Cross_Validation
from Graphique import Graphique
from src.Data_Management import Data_Management as dm

if __name__ == '__main__':
    '''
    Voici le main qui run notre projet
    '''

    # ======================================================================================
    # Pour l'utilisation de la validation croisée, il faut modifier le booléen cross_val
    # dans la classe Data_Management.py (par défaut à True)
    # ======================================================================================

    print("=" * 60)
    print("\t\t\tPARTIE 1 : PREPARATION DES DONNEES")
    print("=" * 60)
    data = dm()

    cross_val = data.get_cross_validation()

    print("\nAffichage des données : \n")
    data.print_data()
    data.get_shape_ind_0()

    print("\nBrève description de nos données :\n")
    print(data.data_desc())

    print("\nAffichage de la distribution dans les classes :\n")
    print(data.class_distribution())

    print("\nMatrice de corrélation :\n")
    data.correlation_matrix()

    # print("\n Représentation des données...\n")
    # data.univariante_plot()
    # data.visualization()

    print("=" * 60)
    print("\t\t\tPARTIE 2 : LES CLASSIFIEURS")
    print("=" * 60)
    graph = Graphique()

    if cross_val:
        print("\nAffichage de l'accuracy avec validation croisée...\n")
        graph.affich_accuracy()
        print("\nAffichage de loss avec validation croisée...\n")
        graph.affich_loss()
        print("\nAffichage de f1_score avec validation croisée...\n")
        graph.affich_f1_score()
    else:
        print("\nAffichage de l'accuracy sans validation croisée...\n")
        graph.affich_accuracy()
        print("\nAffichage de loss sans validation croisée...\n")
        graph.affich_loss()
        print("\nAffichage de f1_score sans validation croisée...\n")
        graph.affich_f1_score()

    print("Generation du fichier submission.csv...")
    graph.generate_submission_file()

    print("FIN DU PROJET ! ")
