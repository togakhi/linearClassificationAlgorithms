#
from src.Data_Management import Data_Management


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    '''
    This is the main function for running the code
    :return:
    '''
    data = Data_Management()
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

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
