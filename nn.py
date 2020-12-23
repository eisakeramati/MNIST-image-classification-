import numpy as np
import matplotlib.pyplot as plt
import argparse
import math
from scipy import special

def softmax(x):
    """
    Compute softmax function for a batch of input values. 
    The first dimension of the input corresponds to the batch size. The second dimension
    corresponds to every class in the output. When implementing softmax, you should be careful
    to only sum over the second dimension.

    Important Note: You must be careful to avoid overflow for this function. Functions
    like softmax have a tendency to overflow when very large numbers like e^10000 are computed.
    You will know that your function is overflow resistent when it can handle input like:
    np.array([[10000, 10010, 10]]) without issues.

    Args:
        x: A 2d numpy float array of shape batch_size x number_of_classes

    Returns:
        A 2d numpy float array containing the softmax results of shape batch_size x number_of_classes
    """
    # *** START CODE HERE ***
    b = x[0].max()
    y = np.exp(x - b)
    return y / y.sum()
    # *** END CODE HERE ***

def softmax_deriv(x, y):
    res =  np.multiply(x, x) * (-1)
    ind = 0
    for i in range(len(y)):
        if y[i] == 1:
            ind = i
    res[0][ind] = x[0][ind] * (1-x[0][ind])
    return res

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Args:
        x: A numpy float array

    Returns:
        A numpy float array containing the sigmoid results
    """
    #*** START CODE HERE ***
    if x<0:
        return ( special.expit(x) / (1 + special.expit(x)))
    else:
        return 1 / (1 + special.expit(-1*(x))) 
    
    #return 1 / (1 + np.exp(-1*(x))) 
    # *** END CODE HERE ***

def sigmoid_deriv(x):
    return x * (1-x)

def get_initial_params(input_size, num_hidden, num_output):
    """
    Compute the initial parameters for the neural network.

    This function should return a dictionary mapping parameter names to numpy arrays containing
    the initial values for those parameters.

    There should be four parameters for this model:
    W1 is the weight matrix for the hidden layer of size input_size x num_hidden
    b1 is the bias vector for the hidden layer of size num_hidden
    W2 is the weight matrix for the output layers of size num_hidden x num_output
    b2 is the bias vector for the output layer of size num_output

    As specified in the PDF, weight matrices should be initialized with a random normal distribution
    centered on zero and with scale 1.
    Bias vectors should be initialized with zero.
    
    Args:
        input_size: The size of the input data
        num_hidden: The number of hidden states
        num_output: The number of output classes
    
    Returns:
        A dict mapping parameter names to numpy arrays
    """

    # *** START CODE HERE ***
    b2 = np.tile(0., (1, num_output))
    w2 = np.tile(0., (num_output, num_hidden))
    b1 = np.tile(0., (1, num_hidden))
    w1 = np.tile(0., (num_hidden, input_size))
    #w1=[]
    for i in range(num_hidden):
        for j in range(input_size):
            #temp.append(np.random.standard_normal())
            w1[i][j] = np.random.standard_normal()
    #    w1.append(temp)
   # w2=[]
    for i in range(num_output):
        #temp = []
        for j in range(num_hidden):
            w2[i][j] = np.random.standard_normal()
           # temp.append(np.random.standard_normal())
       # w2.append(temp)
    #W1 = np.array(w1)
    #W2 = np.array(w2)
    #b1 = np.zeros(num_hidden)
    #b2 = np.zeros(num_output)
    dict = {}
    dict['w1'] = w1
    dict['w2'] = w2
    dict['b1'] = b1
    dict['b2'] = b2
    return dict
    # *** END CODE HERE ***

def cross_enthropy(y, y_hat):
    count = 0
    for i in range(len(y)):
        if(y_hat[0][i] != 0):
            count = count + y[i] * np.log(y_hat[0][i])   
    return count

def cross_enthropy_deriv(y, y_hat):
    count = 0
    for i in range(len(y)):
        count = count + y[i] * 1. / (y_hat[0][i])   
    return count * (-1)


def forward_prop(data, labels, params):
    """
    Implement the forward layer given the data, labels, and params.
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network

    Returns:
        A 3 element tuple containing:
            1. A numpy array of the activations (after the sigmoid) of the hidden layer
            2. A numpy array The output (after the softmax) of the output layer
            3. The average loss for these data elements
    """
    # *** START CODE HERE ***
    #print('start')
    lay2_num = len(params.get('w1'))
    res = []
    lay2_fin = []
    lay3_fin = []
    loss = 0
    for i in range(len(data)):
        layer2 = np.zeros((1, lay2_num))
        layer3 = np.zeros((1, 10))
        for j in range(lay2_num):
            #print(len(data[i]))
            #print(len(params.get('w1')[j]))
            #print(params.get('b1')[j])
            #print(np.dot(data[i], params.get('w1')[j]) + params.get('b1')[0][j])
            layer2[0][j] = sigmoid(np.dot(data[i], params.get('w1')[j]) + params.get('b1')[0][j])
        #print(layer2)
        for j in range(10):
            layer3[0][j] = np.dot(layer2, params.get('w2')[j]) + params.get('b2')[0][j]
        #print(len(layer2))
        #print(len(layer2[0]))
        #print('..........................................................................')
        #print(layer3)
        #print(layer3)
        layer3_np = np.array(layer3)
        layer3 = softmax(layer3)
        #print(layer3)
        res.append(layer3)
        loss = loss + cross_enthropy(labels[i], layer3)
        lay2_fin.append(layer2)
        lay3_fin.append(layer3_np)
    loss = loss / len(data)
    #print(loss)
    #print(res)
    return lay2_fin, res, loss

    # *** END CODE HERE ***


def toarr(x):
    res=[]
    for i in range(len(x)):
        temp =x[i][0]
        res.append(temp.tolist())
    return res


def tonum(x):
    res = np.zeros((1, len(x)))
    for i in range(len(x)):
        res[0][i] = x[i]
    return res

def backward_prop(data, labels, params, forward_prop_func):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """
    # *** START CODE HERE ***
    x, z, d = forward_prop_func(data, labels, params)
    #print()
    #print(z)
    dict = {}
    b2 = []
    w1 = []
    w2 =[]
    temp = []
    b2_c = np.tile(0., (10, 1))
    w2_c = np.tile(0.,(10, len(params['w2'][0])))
    b1_c = np.tile(0., (1, len(params['w2'][0])))
    w1_c = np.tile(0.,(len(params['w2'][0]), len(params['w1'][0])))
    for j in range(len(z)):
        #softmd = softmax_deriv(z[j] ,labels[j])
        #corssend = cross_enthropy_deriv(labels[j], z[j])
        #temp = softmd * corssend
        b2 = z[j] - labels[j]
        b2_c = b2_c + b2.T
        w2 = np.dot(b2.T, x[j])
        #print(w2)
        w2_c = w2_c + w2
        #print(b2)
        #print('-------------')
        g = np.dot(params['w2'].T, b2.T)
        b1_c = b1_c + g
        for i in range(len(x[j])):
            x[j][i] = sigmoid_deriv(x[j][i])
        g = np.multiply(g, x[j].T)
        #print(g)
        #print(g)
        #print(tonum(data[j]))
        w1 = np.dot(g, tonum(data[j]))
        #print(w1)
        #print('jjjjjjjjjjjjjjjjjjjjj')
        w1_c = w1_c + w1
    b2_c = b2_c / (len(z))
    b1_c = b1_c / (len(z))
    w2_c = w2_c / (len(z))
    w1_c = w1_c / (len(z))
    dict['b2'] = b2_c
    dict['w2'] = w2_c
    dict['b1'] = b1_c
    dict['w1'] = w1_c
    #print(b2_c)
    #print(w1_c)
    #print(w2_c)
    #print(b1_c)
    #print('hereeeeeeeeeeeeeeee')
    return dict
    # *** END CODE HERE ***



def backward_prop_regularized(data, labels, params, forward_prop_func, reg):
    """
    Implement the backward propegation gradient computation step for a neural network
    
    Args:
        data: A numpy array containing the input
        labels: A 2d numpy array containing the labels
        params: A dictionary mapping parameter names to numpy arrays with the parameters.
            This numpy array will contain W1, b1, W2 and b2
            W1 and b1 represent the weights and bias for the hidden layer of the network
            W2 and b2 represent the weights and bias for the output layer of the network
        forward_prop_func: A function that follows the forward_prop API above
        reg: The regularization strength (lambda)

    Returns:
        A dictionary of strings to numpy arrays where each key represents the name of a weight
        and the values represent the gradient of the loss with respect to that weight.
        
        In particular, it should have 4 elements:
            W1, W2, b1, and b2
    """

    # *** START CODE HERE ***
    x, z, d = forward_prop_func(data, labels, params)
    #print()
    #print(z)
    dict = {}
    b2 = []
    w1 = []
    w2 =[]
    temp = []
    b2_c = np.tile(0., (10, 1))
    w2_c = np.tile(0.,(10, len(params['w2'][0])))
    b1_c = np.tile(0., (1, len(params['w2'][0])))
    w1_c = np.tile(0.,(len(params['w2'][0]), len(params['w1'][0])))
    for j in range(len(z)):
        #softmd = softmax_deriv(z[j] ,labels[j])
        #corssend = cross_enthropy_deriv(labels[j], z[j])
        #temp = softmd * corssend
        b2 = z[j] - labels[j]
        b2_c = b2_c + b2.T
        w2 = np.dot(b2.T, x[j]) + reg * (2 * params['w2'])
        #print(w2)
        w2_c = w2_c + w2
        #print(b2)
        #print('-------------')
        g = np.dot(params['w2'].T, b2.T)
        b1_c = b1_c + g
        for i in range(len(x[j])):
            x[j][i] = sigmoid_deriv(x[j][i])
        g = np.multiply(g, x[j].T)
        #print(g)
        #print(g)
        #print(tonum(data[j]))
        w1 = np.dot(g, tonum(data[j])) + reg * (2 * params['w1'])
        #print(w1)
        #print('jjjjjjjjjjjjjjjjjjjjj')
        w1_c = w1_c + w1
    b2_c = b2_c / (len(z))
    b1_c = b1_c / (len(z))
    w2_c = w2_c / (len(z))
    w1_c = w1_c / (len(z))
    dict['b2'] = b2_c
    dict['w2'] = w2_c
    dict['b1'] = b1_c
    dict['w1'] = w1_c
    #print(b2_c)
    #print(w1_c)
    #print(w2_c)
    #print(b1_c)
    #print('hereeeeeeeeeeeeeeee')
    return dict
    # *** END CODE HERE ***



def gradient_descent_epoch(train_data, train_labels, learning_rate, batch_size, params, forward_prop_func, backward_prop_func):
    """
    Perform one epoch of gradient descent on the given training data using the provided learning rate.

    This code should update the parameters stored in params.
    It should not return anything

    Args:
        train_data: A numpy array containing the training data
        train_labels: A numpy array containing the training labels
        learning_rate: The learning rate
        batch_size: The amount of items to process in each batch
        params: A dict of parameter names to parameter values that should be updated.
        forward_prop_func: A function that follows the forward_prop API
        backward_prop_func: A function that follows the backwards_prop API

    Returns: This function returns nothing.
    """

    # *** START CODE HERE ***
    for i in range(50):
        dict_grad = backward_prop_func(train_data[i*batch_size:i*batch_size+batch_size], train_labels[i*batch_size:i*batch_size+batch_size], params, forward_prop_func)
        b1 = params['b1']
        #print('-------------------------------------')
        #print(b1)
        b1_grad = dict_grad['b1']
        #print(b1_grad)
        params['b1'] = b1 - b1_grad*learning_rate
        #print(b1 - b1_grad*learning_rate)
        b2 = params['b2']
        b2_grad = dict_grad['b2']
        params['b2'] = b2 - b2_grad*learning_rate
        w1 = params['w1']
        w1_grad = dict_grad['w1']
        params['w1'] = w1 - w1_grad*learning_rate
        w2 = params['w2']
        w2_grad = dict_grad['w2']
        params['w2'] = w2 - w2_grad*learning_rate

    # *** END CODE HERE ***

    # This function does not return anything
    return

def nn_train(
    train_data, train_labels, dev_data, dev_labels, 
    get_initial_params_func, forward_prop_func, backward_prop_func,
    num_hidden=300, learning_rate=5, num_epochs=30, batch_size=1000):

    (nexp, dim) = train_data.shape

    params = get_initial_params_func(dim, num_hidden, 10)

    cost_train = []
    cost_dev = []
    accuracy_train = []
    accuracy_dev = []
    for epoch in range(num_epochs):
       # print(params['w1'])
       # print(params['b1'])
        gradient_descent_epoch(train_data, train_labels, 
            learning_rate, batch_size, params, forward_prop_func, backward_prop_func)
       # print(params['w1'])
       # print('ttttttttt')
       # print(params['b1'])
        h, output, cost = forward_prop_func(train_data, train_labels, params)
        cost_train.append(cost)
        accuracy_train.append(compute_accuracy(toarr(output),train_labels))
        #print("ACC:")
        #print(compute_accuracy(toarr(output),train_labels))
        h, output, cost = forward_prop_func(dev_data, dev_labels, params)
        cost_dev.append(cost)
        accuracy_dev.append(compute_accuracy(toarr(output), dev_labels))
        #print("ACC2:")
        #print(compute_accuracy(toarr(output),dev_labels))

    return params, cost_train, cost_dev, accuracy_train, accuracy_dev

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(toarr(output), labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == 
        np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size),labels.astype(int)] = 1
    return one_hot_labels

def read_data(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y

def run_train_test(name, all_data, all_labels, backward_prop_func, num_epochs, plot=True):
    params, cost_train, cost_dev, accuracy_train, accuracy_dev = nn_train(
        all_data['train'], all_labels['train'], 
        all_data['dev'], all_labels['dev'],
        get_initial_params, forward_prop, backward_prop_func,
        num_hidden=300, learning_rate=5, num_epochs=num_epochs, batch_size=1000
    )

    t = np.arange(num_epochs)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1)

        ax1.plot(t, cost_train,'r', label='train')
        ax1.plot(t, cost_dev, 'b', label='dev')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        if name == 'baseline':
            ax1.set_title('Without Regularization')
        else:
            ax1.set_title('With Regularization')
        ax1.legend()

        ax2.plot(t, accuracy_train,'r', label='train')
        ax2.plot(t, accuracy_dev, 'b', label='dev')
        ax2.set_xlabel('epochs')
        ax2.set_ylabel('accuracy')
        ax2.legend()

        fig.savefig('./' + name + '.pdf')

    accuracy = nn_test(all_data['test'], all_labels['test'], params)
    print('For model %s, got accuracy: %f' % (name, accuracy))
    
    return accuracy

def main(plot=True):
    parser = argparse.ArgumentParser(description='Train a nn model.')
    parser.add_argument('--num_epochs', type=int, default=30)

    args = parser.parse_args()

    np.random.seed(100)
    train_data, train_labels = read_data('./images_train.csv', './labels_train.csv')
    train_labels = one_hot_labels(train_labels)
    p = np.random.permutation(60000)
    train_data = train_data[p,:]
    train_labels = train_labels[p,:]

    dev_data = train_data[0:10000,:]
    dev_labels = train_labels[0:10000,:]
    train_data = train_data[10000:,:]
    train_labels = train_labels[10000:,:]

    mean = np.mean(train_data)
    std = np.std(train_data)
    train_data = (train_data - mean) / std
    dev_data = (dev_data - mean) / std

    test_data, test_labels = read_data('./images_test.csv', './labels_test.csv')
    test_labels = one_hot_labels(test_labels)
    test_data = (test_data - mean) / std

    all_data = {
        'train': train_data,
        'dev': dev_data,
        'test': test_data
    }

    all_labels = {
        'train': train_labels,
        'dev': dev_labels,
        'test': test_labels,
    }
    
    baseline_acc = run_train_test('baseline', all_data, all_labels, backward_prop, args.num_epochs, plot)
    reg_acc = run_train_test('regularized', all_data, all_labels, 
        lambda a, b, c, d: backward_prop_regularized(a, b, c, d, reg=0.0001),
        args.num_epochs, plot)
        
    return baseline_acc, reg_acc

if __name__ == '__main__':
    main()
