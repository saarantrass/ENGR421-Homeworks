import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
def safelog(x):
    return(np.log(x + 1e-100))
np.random.seed(421)

# mean parameters
class_means = np.array([[+0.0, +2.5], 
                        [-2.5, -2.0], 
                        [+2.5, -2.0]])
# covariance parameters
class_covariances = np.array([[[+3.2, +0.0], 
                               [+0.0, +1.2]],
                              [[+1.2, -0.8], 
                               [-0.8, +1.2]],
                              [[+1.2, +0.8], 
                               [+0.8, +1.2]]])
# sample sizes
class_sizes = np.array([120, 90, 90])

# generate random samples
points1 = np.random.multivariate_normal(class_means[0,:], class_covariances[0,:,:], class_sizes[0])
points2 = np.random.multivariate_normal(class_means[1,:], class_covariances[1,:,:], class_sizes[1])
points3 = np.random.multivariate_normal(class_means[2,:], class_covariances[2,:,:], class_sizes[2])
X = np.vstack((points1, points2, points3))

# generate corresponding labels
y = np.concatenate((np.repeat(1, class_sizes[0]), np.repeat(2, class_sizes[1]), np.repeat(3, class_sizes[2])))
# write data to a file
np.savetxt("hw1_data_set.csv", np.hstack((X, y[:, None])), fmt = "%f,%f,%d")

# # plot data points generated if wanted
# plt.figure(figsize = (10, 10))
# plt.plot(points1[:,0], points1[:,1], "r.", markersize = 10)
# plt.plot(points2[:,0], points2[:,1], "g.", markersize = 10)
# plt.plot(points3[:,0], points3[:,1], "b.", markersize = 10)
# plt.xlabel("x1")
# plt.ylabel("x2")
# plt.show()

# read data into memory
data_set = np.genfromtxt("hw1_data_set.csv", delimiter = ",")

# get X and y values
X = data_set[:,[0, 1]]
y_truth = data_set[:,2].astype(int)


# get number of classes and number of samples
K = np.max(y_truth)
N = data_set.shape[0]

# parameter estimations
sample_means = [np.mean(X[y == (c + 1)], axis = 0) for c in range(K)]
sample_covariances= [(np.matmul(np.transpose(X[y == (c + 1)]-sample_means[c]),(X[y == (c + 1)] - sample_means[c])))/class_sizes[c] for c in range(K)]
class_priors = [np.mean(y == (c + 1)) for c in range(K)]
# print("\nSample means:", sample_means)
# print("\nSample covariances:\n", sample_covariances[0],"\n ",sample_covariances[1],"\n ",sample_covariances[2],"\n ")
# print("\nClass probabilities:", class_priors)

# calculations for gc(x) function
W=[-0.5*np.linalg.inv(sample_covariances[c]) for c in range(K)]
wc=[np.transpose(np.matmul(np.linalg.inv(sample_covariances[c]),sample_means[c])) for c in range(K)]
wco=[-0.5* np.matmul( np.matmul((sample_means[c]),np.linalg.inv(sample_covariances[c])) ,np.transpose(sample_means[c])) 
               -0.5*np.log(np.linalg.det(sample_covariances[c])) + np.log(class_priors[c]) for c in range(K)]

# remaining calculations for gc(x)
Y_predicted =[]
for i in range (N): 
    triple=[]
    for c in range(K):
        a=(np.matmul( np.matmul(np.transpose(X[i]) , W[c])  , X[i]))
        b=(np.matmul( wc[c], X[i] ))
        c=(wco[c])
        triple.append(a+b+c)
    Y_predicted.append(triple)

# argmax to print confusion matrix
y_predicted = np.argmax(Y_predicted,axis=1)+1
confusion_matrix = pd.crosstab(y_predicted, y_truth, rownames = ['y_pred'], colnames = ['y_truth'])
print(confusion_matrix)

# create grids to plot our data
x1_interval = np.linspace(-8, +8, 1201)
x2_interval = np.linspace(-8, +8, 1201)
xx, yy = np.meshgrid(x1_interval, x2_interval)
discriminant_values = np.zeros((len(x1_interval), len(x2_interval), K))


# create decision boundaries
for c in range(K):
    discriminant_values[:,:,c] =W[c][0][0]*xx**2+ W[c][1][1]*yy**2+  W[c][1][0]*xx*yy+ wc[c][0] * xx + wc[c][1] * yy + wco[c]
    
A = discriminant_values[:,:,0]
B = discriminant_values[:,:,1]
C = discriminant_values[:,:,2]
A[(A < B) & (A < C)] = np.nan
B[(B < A) & (B < C)] = np.nan
C[(C < A) & (C < B)] = np.nan
discriminant_values[:,:,0] = A
discriminant_values[:,:,1] = B
discriminant_values[:,:,2] = C
plt.figure(figsize = (10, 10))
plt.plot(X[y_truth == 1, 0], X[y_truth == 1, 1], "r.", markersize = 10)
plt.plot(X[y_truth == 2, 0], X[y_truth == 2, 1], "g.", markersize = 10)
plt.plot(X[y_truth == 3, 0], X[y_truth == 3, 1], "b.", markersize = 10)
plt.plot(X[y_predicted != y_truth, 0], X[y_predicted != y_truth, 1], "ko", markersize = 12, fillstyle = "none")
#plot lines
plt.contour(xx, yy, discriminant_values[:,:,0] - discriminant_values[:,:,1], levels = 0, colors = "k")
plt.contour(xx, yy, discriminant_values[:,:,0] - discriminant_values[:,:,2], levels = 0, colors = "k")
plt.contour(xx, yy, discriminant_values[:,:,1] - discriminant_values[:,:,2], levels = 0, colors = "k")
#color areas
plt.contourf(xx, yy, discriminant_values[:,:,0] - discriminant_values[:,:,1],levels = 0, colors=["g","r"],alpha=0.3)
plt.contourf(xx, yy, discriminant_values[:,:,0] - discriminant_values[:,:,2],levels = 0,colors=["b","r"],alpha=0.3)
plt.contourf(xx, yy, discriminant_values[:,:,1] - discriminant_values[:,:,2],levels = 0,colors=["b","g"],alpha=0.3)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()