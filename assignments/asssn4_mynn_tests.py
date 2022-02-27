from asssn4_mynn import *
def test1():
    obj = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=2, learning_rate=0.01)
    ob2 = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=2, learning_rate=0.01)
    return 0

def test2():
    obj1 = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=2, learning_rate=0.01)
    x = np.array([1,2,3])
    out = obj1.forward_propagation(x)
    assert out.shape == (2,)
    return 0

def test3():
    obj1 = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=3, learning_rate=0.01)
    x = np.array([[1,2,3], [3,2,1] , [0,1,0]])
    y = np.array([1,0,2])
    assert obj1.compute_loss(x,y).shape == ()
    return 0
