from simple_mynn import *
# def test1():
#     obj = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=1, learning_rate=0.01)
#     ob2 = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=1, learning_rate=0.01)
#     return 0

# def test2():
#     obj1 = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=1, learning_rate=0.01)
#     x = np.array([1,2,3])
#     out = obj1.forward_propagation(x)
#     assert out.shape == (1,)
#     return 0

# def test3():
#     obj1 = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=1, learning_rate=0.01)
#     x = np.array([[1,2,3], [3,2,1] , [0,1,0]])
#     y = np.array([1,0,0])
#     assert obj1.compute_loss(x,y).shape == ()
#     return 0

# def test4():
#     obj1 = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=1, learning_rate=0.01)
#     assert obj1.sigmoid_derivative(np.random.rand(4,3)).shape == (4,3)
#     np.random.seed(8)
#     x = np.random.rand(2,5)
#     assert (obj1.sigmoid_derivative(x) - obj1.sigmoid_derivative(-x) == np.zeros_like(x)).all()
#     assert obj1.sigmoid_derivative(0.) == 0.25

# def test_backprop_shapes():
#     obj1 = myNeuralNetwork(n_in=3, n_layer1=4, n_layer2=5, n_out=1, learning_rate=0.1)
#     x = np.array([1,2,3])
#     obj1.backpropagate(x, 1)

# def test_fit():
#     np.random.seed(253)
#     obj1 = myNeuralNetwork(n_in=1, n_layer1=4, n_layer2=4, n_out=1, learning_rate=0.0001)
#     X = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(10,1)
#     y = np.array([0,0,0,0,1,1,1,1,1,1])
#     outdict = obj1.fit(X, y, max_epochs=1000, record_rmsgrads=True)
#     probs = obj1.predict_proba(X)
#     res = obj1.predict(X)
#     assert (res==y).all()

def test_simple_fit():
    np.random.seed(253)
    obj1 = myNeuralNetwork(n_in=1, n_layer1 = 3, n_layer2=4, learning_rate=0.01, bias=True)
    X = np.array([0,1,2,3,4,5,6,7,8,9]).reshape(10,1)
    y = np.array([0,0,0,0,1,1,1,1,1,1])
    outdict = obj1.fit(X, y, max_epochs=1000, record_rmsgrads=True, bias=True)
    probs = obj1.predict_proba(X, bias=True)
    res = obj1.predict(X, bias=True)
    assert (res == y).all()
     
if __name__ == "__main__":
    test_simple_fit()