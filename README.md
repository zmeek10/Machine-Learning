# EE-399-HW
This is the first homework assignment for EE 399 which is a machine learning class

HW1:
The goal of this homework was to take a function explicitly given to us on the handout with an array of data points and write python code to determine the minimum error,
and determine the paramters of the function to cause this minimum error. The following problems exapned on this result.

#1: This example was given to us in class, and professor had already layed out the structure of how to find the curve fit and RMSE. Therefore I printed the parameters
c[0]-c[3] to display the paramters for the given minimum error.

#2: In this problem we were generating a error landscape by keeping two parameters fixed, and sweeping through the other paramters. The 
error landscape is visualized through a grid. I created the error grid and swept parameters by creating a range of values the swept paramters could be through np.linspace
and then created a function that looped through the C values and B values that were being swept, and then set the error grid of B and C values equal to the RMSE of the 
function. I then created a mesh grid of the B and C values and plotted this with a color feature.

#3: In this problem the goal was to use the first 20 data points in the given array and fit a line, parabola, and 19th degree polynomial to the data, as well as computing
the least square error. The rest of the data, or the test data is put through the same process. After some research I figured the easiest way to implement this was to use
np.polyfit for each fit, as it gave flexibility and ease for dealing with the 19th degree polynomial. I then found the RMSE of each fit and printed the values for this.
I then printed each error on the screen as well as the scatter plots and line fits.

#4: This problem is the same as the previous except the first 10 and last 10 data points are used for the training data and the middle remaining is used for the test data.
The process for this problem was the exact same as #3. The question was to find the difference between the last two problems, and the answer was in the error. The error was
much larger for #4 than #3 and this makes sense because the data array is a gradual increase in value and therefore skipping the middle data points would cause a huge 
jump in value creating a larger error.



HW 2:
This file has a total of 39 different faces with about 65 lighting scenes for each
face (2414 faces in all).

The individual images are columns of the matrix X, where each image has been downsampled to 32×32

pixels and converted into gray scale with values between 0 and 1. So the matrix is size 1024 × 2414. To important the file, use the following

import numpy as np
from scipy.io import loadmat
results=loadmat(’yalefaces.mat’)
X=results[’X’]

(a) Compute a 100 × 100 correlation matrix C where you will compute the dot product (correlation)
between the first 100 images in the matrix X. Thus each element is given by cjk = x
T
j xk where xj is
the jth column of the matrix. Plot the correlation matrix using pcolor.
(b) From the correlation matrix for part (a), which two images are most highly correlated? Which are
most uncorrelated? Plot these faces.
(c) Repeat part (a) but now compute the 10 × 10 correlation matrix between images and plot the
correlation matrix between them.
[1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005].
(Just for clarification, the first image is labeled as one, not zero like python might do)
(d) Create the matrix Y = XXT and find the first six eigenvectors with the largest magnitude eigenvalue.
(e) SVD the matrix X and find the first six principal component directions.
(f) Compare the first eigenvector v1 from (d) with the first SVD mode u1 from (e) and compute the
norm of difference of their absolute values.
(g) Compute the percentage of variance captured by each of the first 6 SVD modes. Plot the first 6
SVD modes

a. For the first problem I loaded the file into the notebook using the given method to do so. I then compute a 100x100 correlation matrix using the dot product between the first 100 images in the matrix. The np function np.dot computes the dot product for you. I then used pcolor, and other plt functions to visualize the matrix.

b. To find the most correlated and the most uncorrelated images was pretty simple. There is an np function called unravel_index, which allows you to then find the argmax and argmin of the dot product found in the previous problem. I then used plt functions like imshow to display the images that are most correlated and most uncorrelated.

c. This was a step by step repeat of a except it was a 10x10 correlation matrix with specific data points that would be uploaded.

d. To start I loaded the file in the same as before, and then dot prodcut the matrix. I then used the built in np fucntion np.linalg.eig, to find the eigen values and eigen vector of the matrix, and then sorted the eigen values in descending order. The eigenvectors are then manipulated to the covariance matrix by dot producting it with X.T, and normalized.

e. To find the SVD I used the np function np.linalg.svd(x) to extract the principal component directions from the v matrix

f. To compute the norm difference I took the absolute value of the first principal component, and the first value computed in the SVD problem before. After this I used the np function np.linalg.norm of the difference I computed.

g. To calculate the variance I used the first 6 values from the SVD, squared it and then divided by the np.sum of s squared. Where s is the vector of singular values. I then multiplied it by 100 to calculate the total variance. I then printed each of the first 6 SVD modes. Then reshaped the SVD modes to 32x32, and looped through 6 times to print out the images in the yalefaces file in gray.

