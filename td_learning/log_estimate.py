import numpy as np
from numpy.random import randint
from math import log10

N = 10
x = np.array([10**randint(-10, -1) for i in range(N)])
x_avg = np.mean(x)
print("Take a list of numbers:")
print("x = ", x)
print("Its average is:")
print("E[x] = ", x_avg)

log_x = np.log10(x)
log_x_avg = np.mean(log_x)
print("The log of the average is different from the average of the log!")
print("log(x) = ", log_x)
print("E[log(x)] = ", log_x_avg)
print("log(E[x]) = ", log10(x_avg))

# INCREMENTAL ESTIMATION TD-LEARNING STYLE
print("********************************")
print("Now let us use incremental estimation to compute an approximate average")
epochs = 1000
x_hat = 0.0
y_hat1 = y_hat2 = -10
alpha = 0.1

log_alpha = log10(alpha)
log_1_m_alpha = log10(1-alpha)
def log_sum(l1, l2):
    # compute log(x1+x2) given: l1=log(x1) and l2=log(x2)
    return max(l1, l2) + log10(1+ 10**(-abs(l1-l2)))

y_hat2_history = []
for e in range(epochs):
    x_sample = x[randint(N)]
    x_hat += alpha * (x_sample - x_hat)

    y_sample = log10(x_sample)
    y_hat1 += alpha * (y_sample - y_hat1)

    l1 = y_hat2 + log_1_m_alpha
    l2 = y_sample + log_alpha
    y_hat2 = log_sum(l1, l2)
    y_hat2_history.append(y_hat2)

import matplotlib.pyplot as plt
plt.figure()
plt.plot(y_hat2_history)
plt.show()
    

print("Using a standard technique we get:")
print("x_hat = ", x_hat)
print("log(x_hat) = ", log10(x_hat))
print("Working in log space naively we get the wrong result")
print("y_hat = log(x_hat) = ", y_hat1)
print("10^y_hat = x_hat = ", 10**y_hat1)
print("However it is possible to work in log space and get the right result")
print("y_hat = log(x_hat) = ", y_hat2)
print("10^y_hat = x_hat = ", 10**y_hat2)



# def log_sum(l1, l2,alpha):
#     l1 += np.log10(1-alpha)
#     l2 += np.log10(alpha)
#     return max(l1, l2) + np.log10(1+ 10**(-np.abs(l1-l2)))