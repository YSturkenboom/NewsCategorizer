# Resources used:
# - D. Greene and P. Cunningham. "Practical Solutions to the Problem
# of Diagonal Dominance in Kernel Document Clustering", Proc. ICML 
# 2006.
# - http://scikit-learn.org/stable/tutorial/statistical_inference/
# supervised_learning.html

# TODO: allow piping of text to the script to classify
# TODO: output human-readable labels
# TODO: buy a faster machine so training doesn't take forever

import nltk
from lxml import etree
import os
import io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

classified_dir = './data/bbc/bbc.classes'
data_main_dir = './data/bbc/'
data_sub_dirs = ['business', 'entertainment', 'politics', 'sport', 'tech']
#data_sub_dirs = ['business', 'entertainment']


# process classes data file into target vector
y = []
f = open(classified_dir)
for line in f:
	y.append(line.split()[1])

# this is a set because we want only unique tokens
all_tokens = set()

# The term frequencies of each article will serve as features for
# machine learning. Below we generate them per article.
freqdists = []
for data_sub_dir in data_sub_dirs:
	for filename in os.listdir(data_main_dir + data_sub_dir + '/'):
		f = io.open(data_main_dir + data_sub_dir + '/' + filename, encoding='latin-1')
		text = f.read()

		# Turn text from an article into tokens
		tokens = nltk.word_tokenize(text.lower())

		# Update the set of all tokens (ignoring duplicates)
		all_tokens.update(tokens)

		# Use nltk to get the frequency per term, and save this in
		# a list
		freqdists.append(dict(nltk.FreqDist(tokens)))

# First an empty base vector is created, all terms occur zero times.
base_vector = {key: 0 for key in all_tokens}

# Now, expand all term frequency dicts for each article to include
# all tokens, so we can compare them meaningfully.
# This is done by taking the base vector and updating the term
# frequencies that were found to the correct number.
X = []

# copy() because of weird Python referencing stuff
# The following code explains some unexpected (for me) behavior of
# the dict.update function
# a = {'a': 0, 'b': 0, 'c':0}
# b = {'b': 1}
# z = a.copy()
# z.update(b)
# >>> {'a': 0, 'b':1, 'c':0}
for x in freqdists:
	new_vec = base_vector.copy()
	new_vec.update(x)
	new_vec = [new_vec[key] for key in sorted(new_vec)]
	X.append(new_vec)

# Make numpy array for better sklearn integration
X = np.array(X)
y = np.array(y)

# Make sure results are replicable
np.random.seed(0)

# Randomize train and test sets
indices = np.random.permutation(len(X))
X_train = X[indices[:-10]]
y_train = y[indices[:-10]]
X_test  = X[indices[-10:]]
y_test  = y[indices[-10:]]

# Create and fit a nearest-neighbor classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

result = knn.predict(X_test)

# Print results! :)
print result
file = open('output.txt','w') 
file.write(result)
file.close() 

print y_test
print 'accuracy = ' + str(accuracy_score(y_test, result))


