from basic_nn import *
from complex_nn import *

if __name__=="__main__":

    window = 10
    set_name = "matlab"

    bnn_evaluate(window, 0, set_name)
    bnn_evaluate(window, 1, set_name)
    bnn_evaluate(window, 7, set_name)

    cnn_evaluate(window, 0, set_name)
    cnn_evaluate(window, 1, set_name)
    cnn_evaluate(window, 7, set_name)