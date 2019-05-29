#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'notebooks'))
	print(os.getcwd())
except:
	pass

#%%
#  Ebnable HTML/CSS 
from IPython.core.display import HTML
HTML("<link href='https://fonts.googleapis.com/css?family=Passion+One' rel='stylesheet' type='text/css'><style>div.attn { font-family: 'Helvetica Neue'; font-size: 30px; line-height: 40px; color: #FFFFFF; text-align: center; margin: 30px 0; border-width: 10px 0; border-style: solid; border-color: #5AAAAA; padding: 30px 0; background-color: #DDDDFF; }hr { border: 0; background-color: #ffffff; border-top: 1px solid black; }hr.major { border-top: 10px solid #5AAA5A; }hr.minor { border: none; background-color: #ffffff; border-top: 5px dotted #CC3333; }div.bubble { width: 65%; padding: 20px; background: #DDDDDD; border-radius: 15px; margin: 0 auto; font-style: italic; color: #f00; }em { color: #AAA; }div.c1{visibility:hidden;margin:0;height:0;}div.note{color:red;}</style>")

#%% [markdown]
# ___
# Enter Team Member Names here (double click to edit):
# 
# - Carson Drake
# - Andy Heroy
# - Che' Cobb
# - David Josephs
# 
#%% [markdown]
# # In Class Assignment One
# In the following assignment you will be asked to fill in python code and derivations for a number of different problems. Please read all instructions carefully and turn in the rendered notebook (or HTML of the rendered notebook)  before the end of class (or right after class). The initial portion of this notebook is given before class and the remainder is given during class. Please answer the initial questions before class. Once class has started you may rework your answers as a team for the initial part of the assignment. 
# 
# <a id="top"></a>
# ## Contents
# * <a href="#Loading">Loading the Data</a>
# * <a href="#linearnumpy">Linear Regression</a>
# * <a href="#sklearn">Using Scikit Learn for Regression</a>
# * <a href="#classification">Linear Classification</a>
# 
# ________________________________________________________________________________________________________
# 
# <a id="Loading"></a>
# <a href="#top">Back to Top</a>
# ## Loading the Data
# Please run the following code to read in the "diabetes" dataset from sklearn's data loading module. 
# 
# This will load the data into the variable `ds`. `ds` is a dictionary object with fields like `ds.data`, which is a matrix of the continuous features in the dataset. The object is not a pandas dataframe. It is a numpy matrix. Each row is a set of observed instances, each column is a different feature. It also has a field called `ds.target` that is a continuous value we are trying to predict. Each entry in `ds.target` is a label for each row of the `ds.data` matrix. 

#%%
from sklearn.datasets import load_diabetes
import numpy as np


ds = load_diabetes()

# this holds the continuous feature data
# because ds.data is a matrix, there are some special properties we can access (like 'shape')
print('features shape:', ds.data.shape, 'format is:', ('rows','columns')) # there are 442 instances and 10 features per instance
print('range of target:', np.min(ds.target),np.max(ds.target))


#%%
from pprint import pprint

# we can set the fields inside of ds and set them to new variables in python
pprint(ds.data) # prints out elements of the matrix
pprint(ds.target) # prints the vector (all 442 items)

#%% [markdown]
# ________________________________________________________________________________________________________
# <a id="linearnumpy"></a>
# <a href="#top">Back to Top</a>
# ## Using Linear Regression 
# In the videos, we derived the formula for calculating the optimal values of the regression weights (you must be connected to the internet for this equation to show up properly):
# 
# $$ w = (X^TX)^{-1}X^Ty $$
# 
# where $X$ is the matrix of values with a bias column of ones appended onto it. For the diabetes dataset one could construct this $X$ matrix by stacking a column of ones onto the `ds.data` matrix. 
# 
# $$ 
# X=\begin{bmatrix}
#          & \vdots &        &  1 \\
#         \dotsb & \text{ds.data} & \dotsb &  \vdots\\
#          & \vdots &         &  1\\
#      \end{bmatrix}
# $$
# 
# **Question 1:** For the diabetes dataset, how many elements will the vector $w$ contain?

#%%
# Enter your answer here (or write code to calculate it)

X = np.hstack((np.ones((ds.data.shape[0], 1)), ds.data))
X2 = np.hstack((np.ones((ds.data.shape[0], 1)), ds.data))
pprint(X.shape[1])

#

#%% [markdown]
# ________________________________________________________________________________________________________
# 
# **Exercise 1:** In the following empty cell, use this equation and numpy matrix operations to find the values of the vector $w$. You will need to be sure $X$ and $y$ are created like the instructor talked about in the video. Don't forget to include any modifications to $X$ to account for the bias term in $w$. You might be interested in the following functions:
# 
# - `np.hstack((mat1,mat2))` stack two matrices horizontally, to create a new matrix
# - `np.ones((rows,cols))` create a matrix full of ones
# - `my_mat.T` takes transpose of numpy matrix named `my_mat`
# - `np.dot(mat1,mat2)` is matrix multiplication for two matrices
# - `np.linalg.inv(mat)` gets the inverse of the variable `mat`

#%%
# Write you code here, print the values of the regression weights using the 'print()' function in python

y = ds.target # set target array as column vector
w = np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,y))
print('The intercept is:', w[0])
print('The weight coefficients are:', w[1:])


#%% [markdown]
# 
# ___
# <a id="sklearn"></a>
# <a href="#top">Back to Top</a>
# # Start of Live Session Coding
# 
# **Exercise 2:** Scikit-learn also has a linear regression fitting implementation. Look at the scikit learn API and learn to use the linear regression method. The API is here: 
# 
# - API Reference: http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
# 
# Use the sklearn `LinearRegression` module to check your results from the previous question. 
# 
# **Question 2**: Did you get the same parameters? 

#%%
from sklearn.linear_model import LinearRegression

# write your code here, print the values of model by accessing 
#    its properties that you looked up from the API
X = X[:,1:]
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_


print('model coefficients are:', reg.coef_)
print('model intercept is', reg.intercept_)
print('Answer to question is', "True")

#%% [markdown]
# ________________________________________________________________________________________________________
#
# Recall that to predict the output from our model, $\hat{y}$, from $w$ and $X$
# we need to use the following formula:
#
# - $\hat{y}=w^TX^T$
#
# Where $X$ is a matrix with example instances in *each row* of the matrix. 
#
# **Exercise 3:** 
# - *Part A:* Use matrix multiplication to predict output using numpy,
#   $\hat{y}_{numpy}$ and also using the sklearn regression object,
#   $\hat{y}_{sklearn}$.
#  - **Note**: you may need to make the regression weights a column vector using
#    the following code: `w = w.reshape((len(w),1))` This assumes your weights
#    vector is assigned to the variable named `w`.
# - *Part B:* Calculate the mean squared error between your prediction from
#   numpy and the target, $\sum_i(y-\hat{y}_{numpy})^2$. 
# - *Part C:* Calculate the mean squared error between your sklearn prediction
#   and the target, $\sum_i(y-\hat{y}_{sklearn})^2$.

#%%
# Use this block to answer the questions
from sklearn.metrics import mean_squared_error as mse

yhats = np.dot(w.T,X2.T)


print('MSE Sklearn is:', mse(y, reg.predict(X)))
print('MSE Numpy is:', np.mean((y - yhats)**2))

#%% [markdown]
# ________________________________________________________________________________________________________
# <a id="classification"></a> <a href="#top">Back to Top</a>
# ## Using Linear Classification
# Now lets use the code you created to make a classifier with linear boundaries.
# Run the following code in order to load the iris dataset.

#%%
from sklearn.datasets import load_iris
import numpy as np

# this will overwrite the diabetes dataset
ds = load_iris()
print('features shape:', ds.data.shape) # there are 150 instances and 4 features per instance
print('original number of classes:', len(np.unique(ds.target)))

# now let's make this a binary classification task
ds.target = ds.target>1
print ('new number of classes:', len(np.unique(ds.target)))

#%% [markdown]
# ________________________________________________________________________________________________________
#
# **Exercise 4:** Now use linear regression to come up with a set of weights,
# `w`, that predict the class value. This is exactly like you did before for the
# *diabetes* dataset. However, instead of regressing to continuous values, you
# are just regressing to the integer value of the class (0 or 1), like we talked
# about in the video. Remember to account for the bias term when constructing
# the feature matrix, `X`. Print the weights of the linear classifier.

#%%
# write your code here and print the values of the weights 

regr = LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
X = ds.data
y = ds.target
# Train the model using the training sets
regr.fit(X, y)

# Print the weights of the linear classifier.
print('model coefficients are:', regr.coef_)
print('model intercept is', regr.intercept_)


#

#%% [markdown]
# ________________________________________________________________________________________________________
# 
# **Exercise 5:** Finally, use a hard decision function on the output of the linear regression to make this a binary classifier. This is just like we talked about in the video, where the output of the linear regression passes through a function: 
# 
# - $\hat{y}=g(w^TX^T)$ where
#  - $g(w^TX^T)$ for $w^TX^T < \alpha$ maps the predicted class to `0` 
#  - $g(w^TX^T)$ for $w^TX^T \geq \alpha$ maps the predicted class to `1`. 
# 
# Here, alpha is a threshold for deciding the class. 
# 
# **Question 3**: What value for $\alpha$ makes the most sense? What is the accuracy of the classifier given the $\alpha$ you chose? 
# 
# Note: You can calculate the accuracy with the following code: `accuracy = float(sum(yhat==y)) / len(y)` assuming you choose variable names `y` and `yhat` for the target and prediction, respectively.

#%%
# use this box to predict the classification output

pred = regr.predict(X)
alpha = 0.5

y_hat = pred > alpha

accuracy = float(sum(y_hat==y)) / len(y)

print('Percentage accuracy:', accuracy)

#%% [markdown]
# ________________________________________________________________________________________________________
# 
# That's all! Please **save (make sure you saved!!!) and upload your rendered notebook** and please include **team member names** in the notebook submission.

#%%



