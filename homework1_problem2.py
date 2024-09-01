import numpy as np

def train_age_regressor ():
    # Load data
    X_tr = np.reshape(np.load("age_regression_Xtr.npy"), (-1, 48*48))
    ytr = np.load("age_regression_ytr.npy")
    X_te = np.reshape(np.load("age_regression_Xte.npy"), (-1, 48*48))
    yte = np.load("age_regression_yte.npy")




def stochastic_grad_descent(x,y,mini_batch,num_epoch,epsilon):

    # Assigning initial values for parameters and hyperparameters
    N = np.shape(x[0])
    w = np.ones(shape=np.shape(x)[1])
    b = 0
    dldw,dldb = 0

    # Tracking the cost function value in every 10 epochs
    cost_list = []
    epoch_count = []

    # If mini batch size exceeds total no. of samples
    if mini_batch > N:
        mini_batch = N

    for i in range(num_epoch):
        # Shuffling the dataset
        ids = np.random.permutation(5000)
        x_sh = x[ids]
        y_sh = y[ids]

        for j in range(0,N,mini_batch):
            # Creating a mini batch for input x and output y
            xj = x_sh[j:j+mini_batch]
            yj = y_sh[j:j+mini_batch]

            # Prediction Model equation( y = x.T*w + b )
            y_predict = xj.T.dot(w) + b

            # Gradient of f(MSE) w.r.t w and b
            dldw = xj.T.dot(y_predict - yj)
            dldb = (y_predict - yj)

            # Updating the values w and b based on the current mini batch gradient
            w -= (epsilon * dldw)
            b -= (epsilon * dldb)

            costfunc = np.mean(np.square(y_predict-y_sh))

        if i%10 == 0:
            # Keeping the records of cost function value after every 10 epochs
            cost_list = np.append(costfunc)
            epoch_count = np.append(i)

    return w,b,costfunc,cost_list,epoch_count

train_age_regressor()

