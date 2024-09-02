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
    yv = y_tr[split:]

    return Xtr,ytr,Xv,yv,Xte,yte

def cost_function(x,y,w,b):
    n = x.shape[0]
    y_hat = np.dot(x,w) + b
    cost = (1/2) * np.mean((np.square(y_hat-y)))

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

def stochastic_grad_descent(x,y,xv,yv,mini_batch,num_epoch,epsilon):

    # Assigning initial values for parameters and hyperparameters
    n,m = np.shape(x)
    w = np.zeros(m)
    b = 0.0
    dldw = 0.0
    dldb = 0.0

    # Tracking the training and validation cost function value
    train_cost_list = []
    val_cost_list = []

    # If mini batch size exceeds total no. of samples
    if mini_batch > n:
        mini_batch = n

    for i in range(num_epoch):
        # Shuffling the dataset
        ids = np.random.permutation(n)
        x_shuffle = x[ids]
        y_shuffle = y[ids]

        for j in range(0,n,mini_batch):
            # Creating a mini batch for input x and output y
            xj = x_shuffle[j:j+mini_batch]
            yj = y_shuffle[j:j+mini_batch]

            # Gradient of f(MSE) w.r.t w and b
            dldw = gradient_w(xj,yj,w,b)
            dldb = gradient_b(xj,yj,w,b)

            # Updating the values w and b based on the current mini batch gradient
            w -= (epsilon * dldw)
            b -= (epsilon * dldb)

        # Computing Training and Validation cost values
        train_cost = cost_function(X_train,y_train,w,b)
        train_cost_list.append(train_cost)

        val_cost = cost_function(X_val,y_val,w,b)
        val_cost_list.append(val_cost)

    return w,b,train_cost_list,val_cost_list

X_train, y_train, X_val, y_val, X_test, y_test = train_age_regressor()


# Hyperparameter Tuning using Grid Search

# Assigning 2 values for each hyperparameter
learning_rates = [0.001,0.0019]
mini_batch_size = [16,32]
epochs = [500,800]

# Recording the best values of both parameters and hyperparameters
best_cost_val = float('inf')
best_hparameters = {}
best_w, best_b = None, None

# Nested for loops to get the best hyperparameter values
for l_rate in learning_rates:
    for n_mini in mini_batch_size:
        for e in epochs:
            print(f'Training set with learning rate:{l_rate}, mini batch size:{n_mini} and number of epochs:{e}')
            weight,bias,train_cost,val_cost = stochastic_grad_descent(X_train,y_train,X_val,y_val,n_mini,e,l_rate)
            val_cost = val_cost[-1]

            if val_cost < best_cost_val:
                best_cost_val = val_cost
                best_hparameters = {'learning_rate':l_rate,'mini_batch_size':n_mini,'epochs':e}
                best_w, best_b = weight,bias

# Getting Cost value for Test dataset
test_cost_value = cost_function(X_test,y_test,best_w,best_b)

print(f'Best Hyper Parameters: {best_hparameters}')
print(f'Test Cost Value: {test_cost_value}')
print(f'Best Validation cost value:{best_cost_val}')
print(f'Cost values of Training Dataset of last 10 iterations: {train_cost[-10:]}')
