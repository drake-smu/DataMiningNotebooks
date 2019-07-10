#%%
#  Ebnable HTML/CSS
from IPython.core.display import HTML
HTML("<link href='https://fonts.googleapis.com/css?family=Passion+One' rel='stylesheet' type='text/css'><style>div.attn { font-family: 'Helvetica Neue'; font-size: 30px; line-height: 40px; color: #FFFFFF; text-align: center; margin: 30px 0; border-width: 10px 0; border-style: solid; border-color: #5AAAAA; padding: 30px 0; background-color: #DDDDFF; }hr { border: 0; background-color: #ffffff; border-top: 1px solid black; }hr.major { border-top: 10px solid #5AAA5A; }hr.minor { border: none; background-color: #ffffff; border-top: 5px dotted #CC3333; }div.bubble { width: 65%; padding: 20px; background: #DDDDDD; border-radius: 15px; margin: 0 auto; font-style: italic; color: #f00; }em { color: #AAA; }div.c1{visibility:hidden;margin:0;height:0;}div.note{color:red;}</style>")

#%% [markdown]
# ___
# Enter Team Member Names here:
#
# - Name 1: Carson Drake
# - Name 2: Che Cobb
# - Name 3: David Josephs
# - Name 4: Andy Heroy
#
#
# ________
#
# # In Class Assignment Three
# In the following assignment you will be asked to fill in python code and
# derivations for a number of different problems. Please read all instructions
# carefully and turn in the rendered notebook (or HTML of the rendered notebook)
# before the end of class.
#
# <a id="top"></a>
# ## Contents
# * <a href="#Loading">Loading the Data</a>
# * <a href="#distance">Measuring Distances</a>
# * <a href="#KNN">K-Nearest Neighbors</a>
# * <a href="#naive">Naive Bayes</a>
#
# ________________________________________________________________________________________________________
# <a id="Loading"></a> <a href="#top">Back to Top</a>
# ## Downloading the Document Data
# Please run the following code to read in the "20 newsgroups" dataset from
# sklearn's data loading module.

#%%
from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np

# this takes about 30 seconds to compute, read the next section while this downloads
ds = fetch_20newsgroups_vectorized(subset='train')

# this holds the continuous feature data (which is tfidf)
print('features shape:', ds.data.shape) # there are ~11000 instances and ~130k features per instance
print('target shape:', ds.target.shape)
print('range of target:', np.min(ds.target),np.max(ds.target))
print('Data type is', type(ds.data), float(ds.data.nnz)/(ds.data.shape[0]*ds.data.shape[1])*100, '% of the data is non-zero')
print('Number of keys is', ds.keys())
print(ds['DESCR'])
#%% [markdown]
# ## Understanding the Dataset
# Look at the description for the 20 newsgroups dataset at
# http://qwone.com/~jason/20Newsgroups/. You have just downloaded the
# "vectorized" version of the dataset, which means all the words inside the
# articles have gone through a transformation that binned them into 130 thousand
# features related to the words in them.
#
# **Question Set 1**:
# - How many instances are in the dataset?
# - What does each instance represent?
# - How many classes are in the da taset and what does each class represent?
# - Would you expect a classifier trained on this data would generalize to
#   documents written in the past week? Why or why not?
# - Is the data represented as a sparse or dense matrix?
#%% [markdown]
#   ___
# Enter your answer here:
#

# 1.  There are 11314 instances in this dataset
# 2.  Each instance represents 130,107 features
# 3.  There are 20 classes in the dataset and each represent a different news category
# 4.  I would think just the last week wouldn't be enough data to get a desired accuracy level
# 5.  It is represented as a sparce matrix.  Only .12% of the values are zero.
#
#%% [markdown]
# ___
# <a id="distance"></a> <a href="#top">Back to Top</a>
# ## Measures of Distance
# In the following block of code, we isolate three instances from the dataset.
# The instance "`a`" is from the group *computer graphics*, "`b`" is from from
# the group *recreation autos*, and "`c`" is from group *recreation motorcycle*.
#
#
# **Exercise for part 2**:
#
# Calculate the:
# - (1) Euclidean distance
# - (2) Cosine distance
# - (3) Jaccard similarity
#
#
# between each pair of instances using the imported functions below. Remember
# that the Jaccard similarity is only for binary valued vectors, so convert
# vectors to binary using a threshold.
#
#
# **Question for part 2**: Which distance seems more appropriate to use for this
# data? **Why**?

#%%
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import jaccard
import numpy as np

# get first instance (comp)
idx = 550
a = ds.data[idx].todense()
a_class = ds.target_names[ds.target[idx]]
print('Instance A is from class', a_class)

# get second instance (autos)
idx = 4000
b = ds.data[idx].todense()
b_class = ds.target_names[ds.target[idx]]
print('Instance B is from class', b_class)

# get third instance (motorcycle)
idx = 7000
c = ds.data[idx].todense()
c_class = ds.target_names[ds.target[idx]]
print('Instance C is from class', c_class)

# Euclidean Distance
e_ab = euclidean(a, b)
e_ac = euclidean(a, c)
e_bc = euclidean(b, c)

# Cosine Distance
c_ab = cosine(a, b)
c_ac = cosine(a, c)
c_bc = cosine(b, c)

# converting to boolean
a = a > 0
b = b > 0
c = c > 0
# Jaccard Distance

j_ab = jaccard(a, b)
j_ac = jaccard(a, c)
j_bc = jaccard(b, c)


# Enter distance comparison below for each pair of vectors:

print('\n\nEuclidean Distance\n ab:', e_ab, 'ac:', e_ac, 'bc:',e_bc)
print('Cosine Distance\n ab:', c_ab, 'ac:', c_ac, 'bc:', c_bc)
print('Jaccard Dissimilarity (vectors should be boolean values)\n ab:', j_ab, 'ac:', j_ac, 'bc:', j_bc)

print('\n\nThe most appropriate distance is...Cosine Distance')
print('\nThe Cosine distance is best in this scenario because \nif the angle between the two vectors is small, then they \nare closer together and therefore more similar.')

#%% [markdown]
# ___
# # Start of Live Session Assignment
# ___
# <a id="KNN"></a> <a href="#top">Back to Top</a>
# ## Using scikit-learn with KNN
# Now let's use stratified cross validation with a holdout set to train a KNN
# model in `scikit-learn`. Use the example below to train a KNN classifier. The
# documentation for `KNeighborsClassifier` is here:
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#
#
# **Exercise for part 3**: Use the code below to test what value of
# `n_neighbors` works best for the given data. *Note: do NOT change the metric
# to be anything other than `'euclidean'`. Other distance functions are not
# optimized for the amount of data we are working with.*
#
# **Question for part 3**: What is the accuracy of the best classifier you can
# create for this data (by changing only the `n_neighbors` parameter)?

#%%
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from IPython.html import widgets

sss = StratifiedShuffleSplit(n_splits= 1, test_size = 0.2, train_size=0.8)
cv = sss.split(X= ds.data, y = ds.target)

# fill in the training and testing data and save as separate variables
for trainidx, testidx in cv:
    # note that these are sparse matrices
    X_train = ds.data[trainidx]
    X_test = ds.data[testidx]
    y_train = ds.target[trainidx]
    y_test = ds.target[testidx]

# fill in your code  here to train and test
# calculate the accuracy and print it for various values of K
clf = KNeighborsClassifier(weights='uniform', metric='euclidean')
accuracies = []
for k in range(1,10):
    clf.n_neighbors = k
    clf.fit(X_train,y_train)
    acc = clf.score(X_test,y_test)
    accuracies.append(acc)
    print('Accuracy of classifier with %d neighbors is: %.3f'%(k,acc))




#%% [markdown]
#=====================================
#
# The best accuracy is 68.9% with k=1 neighbors.  Because we're only optimizing
# one point, the bias is fairly low and therefore performs better on the
# training data.  Unfortunately, this also means the variance is probably
# higher within our model.
#
#

#%% [markdown]
# **Question for part 3**:
#With sparse data, does the use of a KDTree representation make sense? Why or
#Why not?
#
#%% [markdown]
# Enter your answer below:
#
#KDtree won't work well for this dataset because of the multitude of features.
#As you increase dimensionality its going to run slower and slower because its
#trying to calculate the angles/vectors between 130,000 features.  Thats alot to
#compute and why we would refrain from doing so with this dataset.
#
#_____
#%% [markdown]
#_____
### KNN extensions - Centroids
#
#Now lets look at a very closely related classifier to KNN, called nearest
#centroid. In this classifier (which is more appropriate for big data scenarios
#and sparse data), the training step is used to calculate the centroids for each
#class. These centroids are saved. Unknown attributes, at prediction time, only
#need to have distances calculated for each saved centroid, drastically
#decreasing the time required for a prediction.
#
#**Exercise for part 4**: Use the template code below to create a nearest
#centroid classifier. Test which metric has the best cross validated
#performance: Euclidean, Cosine, or Manhattan. In `scikit-learn` you can see the
#documentation for NearestCentroid here:
#- http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestCentroid.html#sklearn.neighbors.NearestCentroid
#
#and for supported distance metrics here:
#- http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics

#%%
from sklearn.neighbors.nearest_centroid import NearestCentroid

# the parameters for the nearest centroid metric to test are:
#    l1, l2, and cosine (all are optimized)
# fill in the training and testing data and save as separate variables

for d in ['l1', 'l2', 'cosine', 'euclidean', 'manhattan']:
   clf = NearestCentroid(metric=d)
   clf.fit(X_train, y_train)
   yhat = clf.predict(X_test)
   acc = accuracy_score(y_test, yhat)
   print(d, acc)

p = 'cosine'
print('The best distance metric is: ', p)

#%% [markdown]
# ___
# <a id="naive"></a> <a href="#top">Back to Top</a>
# ## Naive Bayes Classification
# Now let's look at the use of the Naive Bayes classifier. The 20 newsgroups
# dataset has 20 classes and about 130,000 features per instance. Recall that
# the Naive Bayes classifer calculates a posterior distribution for each
# possible class. Each posterior distribution is a multiplication of many
# conditional distributions:
#
# $${\arg \max}_{j} \left(p(class=j)\prod_{i} p(attribute=i|class=j) \right)$$
#
# where $p(class=j)$ is the prior and $p(attribute=i|class=j)$ is the
# conditional probability.
#
# **Question for part 5**: With this many classes and features, how many
#different conditional probabilities need to be parameterized? How many priors
#need to be parameterized?
# %% [markdown]
# Enter you answer here:
#
# There are 2600000 conditionals probabilities that need to be parameterized. There are 20 priors that need to be parameterized.

#%% [markdown]
#
# Use this space for any calculations you might want to do.
#
# The above number was found because 130k features x 20 classes according to the
# argmax function

#%% [markdown]
# ___
# ## Naive Bayes in Scikit-learn
# Scikit has several implementations of the Naive Bayes classifier:
# `GaussianNB`, `MultinomialNB`, and `BernoulliNB`. Look at the documentation
# here: http://scikit-learn.org/stable/modules/naive_bayes.html Take a look at
# each implementation and then answer this question:
#
# **Questions for part 6**:
# - If the instances contain mostly continuous attributes, would it be better to
#   use Gaussian Naive Bayes, Multinomial Naive Bayes, or Bernoulli? And Why?
# - What if the data is sparse, does this change your answer? Why or Why not?
#
#%%  [markdown]
# Enter you answer here:
#
#
#
# A Gaussian Naive Bayes algorithm is a special type of NB algorithm. It's
# specifically used when the features have continuous values. It's also assumed
# that all the features are following a gaussian distribution i.e, normal
# distribution.  If the data is sparce then our answer does not change because
# they're still continuous values.
# ___
#%% [markdown]
# ## Naive Bayes Comparison
# For the final section of this notebook let's compare the performance of Naive
# Bayes for document classification. Look at the parameters for `MultinomialNB`,
# and `BernoulliNB` (especially `alpha` and `binarize`).
#
# **Exercise for part 7**:
#
# Using the example code below, change the parameters for each classifier and
# see how accurate you can make the classifiers on the test set.
#
# **Question for part 7**:
#
# Why are these implementations so fast to train? What does the `'alpha'` value
# control in these models (*i.e.*, how does it change the parameterizations)?
#
#
# 1.  They're both faster to train on because multinomial is using counts on a
#     multinomial distribution.  Bernoulli does so on a gaussian distribution
#     and uses a binary analysis.  Both operations are quite fast.
# 2.  The Alpha value's control smoothing within the model.

#%%
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

for a in [0.0, 0.001, 0.01, 0.1, 1]:
   clf_mnb = MultinomialNB(alpha=a)
   clf_mnb.fit(X_train, y_train)
   yhat = clf_mnb.predict(X_test)
   acc = accuracy_score(y_test, yhat)
   print("MultinomialNB (alpha=%f): %f" % (a, acc))
   for b in [0.0, 0.002, 0.02, 0.04, 0.06, 0.08, 0.2]:
      clf_bnb = BernoulliNB(alpha=a, binarize=b)
      clf_bnb.fit(X_train, y_train)
      yhat = clf_bnb.predict(X_test)
      acc = accuracy_score(y_test, yhat)
      print("BernoulliNB (alpha=%f, binarize=%f): %f" % (a, b, acc))



print('These classifiers are so fast because multinomial is using counts\n on a multinomial distribution.  Bernoulli does so on a gaussian\n distribution and uses a binary analysis.  Both operations are quite fast.\n\n')
print('The Alpha values control smoothing within the model. ')

#%% [markdown]
# ________________________________________________________________________________________________________
#
# That's all! Please **upload your rendered notebook to blackboard** and please include **team member names** in the notebook submission.

#%%


