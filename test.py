import numpy as np

s = np.random.standard_normal()
for i in range(50):
    print(s)
    s = np.random.normal()

a= [1, 2, 3]
d = [2,2,3]
print(np.transpose(a))

c = np.array(d)
b = np.array(a)
print(len(b))
print(np.dot(c, b))
print(np.matmul(c, b))
print(5*b)
b2_c = np.tile(0,3)
print(b2_c+b+c)

a = [[1, 0]]
a = np.array(a)
b = [[4], [2]]
b= np.array(b)
print(np.dot(a.T, b.T))

a = np.zeros((1,3))
a[0][0]=4
print(a.T)
x = np.tile(0,(10, 2))
x[1][1] = 5
x = np.zeros(5)
print(x)
t = [1, 2, 3, 4, 5]
t = np.array(t)
x[2] = 0.567
b2 = np.tile(0., (1, 3))
b2[0][1] = 10010
#b2[0][3] =3.2
print(b2)
print(np.random.standard_normal())

def cross_enthropy_deriv(y, y_hat):
    return (-1) * np.dot(y, 1. / y_hat)

b2[0][0] = 10000
b2[0][2] = 10
print(a*2)

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
    z = x - max(x)
    numerator = np.exp(z)
    denominator = np.sum(numerator)
    softmax = numerator/denominator
    
def exp_normalize(x):
    b = x[0].max()
    y = np.exp(x - b)
    return y / y.sum()
x = np.array([10000, 10010, 10])
print(exp_normalize(b2))
print('8888888')
print(b2 / 10)
print(softmax(x))
print('9999999999')
print(np.sum(np.exp(b2), axis=1))
print(b2.tolist())
def toarr(x):
    res=[]
    for i in range(len(x)):
        temp =x[i]
        res.append(temp.tolist())
    return res

print(toarr(b2))

