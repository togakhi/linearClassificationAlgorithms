#
from Classifieurs import Classifieurs as clf
from src.Data_Management import Data_Management as dm


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    This is the main function for running the code
    :return:
    '''
    data = dm()
    print("\nPrinting some data for quick observation ...\n")
    data.printSomeData()

    print("\nPrinting the shape of the data ...\n")
    data.getShape()

    print("\n Printing a quick description of the data ...\n")
    print(data.dataDescription())

    print("\n Printing the Class distribution ...\n")
    print(data.classDistribution())

    print("\n Printing the Correlation between variables ...\n")
    print(data.correlation())

    classifieur = clf(0)

    [x_train, y_train, x_test, y_test] = data.getData()

    classifieur.entrainement(x_train, y_train)

    classifieur.prediction(x_test, y_test)

    #classifieur.loss_calcul(x_test, y_test)
    # Test the leaf_image function


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
