import numpy as np

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    y_tr = np.load("age_regression_ytr.npy")
    Xte = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")

    # Splitting training data(80% for training and 20% for validation)
    N = X_tr.shape[0]
    split = int(0.8 * N)
    Xtr = X_tr[:split]
    Xv = X_tr[split:]
    ytr = y_tr[:split]
    yv = ytr[split:]

    return Xtr,ytr,Xv,yv,Xte,yte

def cost_function(x,y,w,b):
    n = x.shape[0]
    y_hat = np.dot(x,w) + b
    cost = (1/(2)) * np.mean((np.square(y_hat-y)))

    return cost

def gradient_w(x,y,w,b):
    n = x.shape[0]
    y_hat = np.dot(x,w) + b
    grad_w = (1/n) * (np.dot(x.T,(y_hat - y)))

    return grad_w

def gradient_b(x,y,w,b):
    n = x.shape[0]
    y_hat = np.dot(x,w) + b
    grad_b = (1/n) * np.sum(y_hat - y)

    return grad_b

def stochastic_grad_descent(x,y,mini_batch,num_epoch,epsilon=0.001972):

    # Assigning initial values for parameters and hyperparameters
    n,m = np.shape(x)
    w = np.zeros(m)
    b = 0.0
    dldw = 0.0
    dldb = 0.0

    # Tracking the cost function value in every 10 epochs
    cost_list = []
    epoch_count = []

    # If mini batch size exceeds total no. of samples
    if mini_batch > n:
        mini_batch = n

    for i in range(num_epoch):
        # Shuffling the dataset
        ids = np.random.permutation(n)
        x_sh = x[ids]
        y_sh = y[ids]

        for j in range(0,n,mini_batch):
            # Creating a mini batch for input x and output y
            xj = x_sh[j:j+mini_batch]
            yj = y_sh[j:j+mini_batch]

            # Gradient of f(MSE) w.r.t w and b
            dldw = gradient_w(xj,yj,w,b)
            dldb = gradient_b(xj,yj,w,b)

            # Updating the values w and b based on the current mini batch gradient
            w -= (epsilon * dldw)
            b -= (epsilon * dldb)

        cost = cost_function(x,y,w,b)

        if i%10 == 0:
            # Keeping the records of cost function value after every 10 epochs
            cost_list.append(cost)
            epoch_count.append(i)

    return w,b,cost,cost_list,epoch_count

X_train, y_train, X_val, y_val, X_test, y_test = train_age_regressor()
weight,bias,cost,cost_list,epoch_count = stochastic_grad_descent(X_train,y_train,32,800)
print(cost)
