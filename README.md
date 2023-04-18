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

a. For the first problem I loaded the file into the notebook using the given method to do so. I then followed 

