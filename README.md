# Analysis-of-Iris-Flower-Dataset-using-Machine-Learning

I have used various Machine Learning algorithms, specifically classified under Supervised and Unsupervised learning, to analyze the Iris Flower dataset. 
Here, I have used Fisher's Iris dataset,and downloaded from kaggle.com as a .csv file. Our task is to classify Iris flowers among three species( namely Setosa, Virginica and Versicolor), from measurements of length of sepals and petals.

### Iris-Versicolor


![330px-Iris_versicolor_3](https://user-images.githubusercontent.com/76059423/102658762-3aad3480-419e-11eb-96ee-a1418cb67da8.jpg)


### Iris-Virginica



![330px-Iris_virginica](https://user-images.githubusercontent.com/76059423/102659020-b6a77c80-419e-11eb-8dc6-1908cbe2b08d.jpg)



### Iris-Setosa


![Kosaciec_szczecinkowaty_Iris_setosa](https://user-images.githubusercontent.com/76059423/102659174-f8d0be00-419e-11eb-83e1-2e9da80ebd0e.jpg)



## Libraries to be installed :

1. pandas
2. sklearn
3. matplotlib
4. numpy
5. scipy

## Contents :

A. Supervised Learning
   1. K-Nearest Neighbor
   2. Logistic Regression
   3. Decision Tree
   4. Naive Bayes
   5. Support Vector Machine
   6. Random Forest

B. Unsupervised Learning
   1. K-Means Clustering
   2. Hierarchical Clustering
   3. Principal Component Analysis
   4. Single Value Decomposition

## Supervised Learning

In a supervised machine learning model, the algorithm learns on a labeled dataset, providing an answer key that the algorithm can use to evaluate its accuracy on training data.
Here, the dataset is first transformed to numeric form using LabelEncoder of sklearn. Then, the obtained data is divided into training data and testing data respectively.

#### 1. K-Nearest Neighbor

The KNN algorithm is simple and easy to use, and can be used for both classification and regression problems. Here, it is used for classification of Iris flowers using KNeighborstClassifier of sklearn. In this section, KNN is applied on the training data and then accuracy score is calculated, which is found to be 90.0%. An unknown flower can also be predicted, that it belongs to which species, by simply taking any random values for length and width of the sepal and petal respectively.

#### 2. Logistic Regression

Logistic regression is a classification algorithm which is used to predict the probability of a target variable. The target or dependent variable is dichotomous or binary in nature, i.e. having only two values 0 or 1. Here, LogisticRegression classifier of sklearn is imported, and the accuracy calculated is found to be 87.5%. Classification of unknown flower is also predicted.

#### 3. Decision Tree

Decision trees predict values of responses by learning decision rules derived from features. They can be used in both regression and classification. Here, DecisionTreeClassifier of sklearn is imported and the accuracy calculated is found to be 90.0%.

#### 4. Naive Bayes

It is based on Bayes theorem used for solving classification problems. It is a probabilistic classifier and predicts on the basis of the probability of an object. Here, GaussianNB classifier of sklearn is imported and the accuracy obtained is 87.5%.

#### 5. Support Vector Machine

This algorithm analyze data for classification and regression analysis by constructing a hyperplane or set of hyperplanes in a high or infinite dimensional space.
Here, SVC classifier of sklearn is imported and accuracy obtained is 97.5%.

#### 6. Random Forest

Random decision forests are an ensemble learning method for classification, regression and other tasks that operate by constructing a multitude of decision trees at training time and outputting the class that is the mode of the classes or mean prediction of the individual trees.Here, RandomForestClassifier of sklearn is imported and the accuracy calculated is 90.0%.



The accuracies of all the algorithms is then visualized using matplotlib.



## Unsupervised Learning

An unsupervised learning model provides unlabeled data that the algorithm tries to make sense of byextracting features and patterns on its own.


#### 1. K-Means Clustering

This algorithm identifies 'k' number of centroids, and then allocates every data point to the nearest cluster, while keeping the centroids as small as possible.
Here, KMeans of sklearn is imported to find the number of clusters, which is then visualized in a graph using the Elbow method. The scattered clusters are also visualized using matplotlib library.

#### 2. Hierarchical Clustering

It is used to group together the unlabeled data points having similar characteristics. Here, Agglomerative type of hierarchical clustering is visualized by diving the dataset into three clusters. Also, dendrogram of the clusters is plotted using scipy.

#### 3. Principal Component Analysis

This technique is used to preprocess and reduce the dimensionality of high-dimensional datasets while preserving the original structure and relationships inherent to the original dataset so that machine learning models can still learn from them and be used to make accurate predictions. Here, StandardScaler of sklearn is imported to transform the data and then PCA is imported to select two principal components from the dataset, which is then graphically visualized using matplotlib.

##### 4. Single Value Decomposition

SVD is a matrix decomposition method for reducing a matrix to its constituent parts in order to make certain subsequent matrix calculations simpler. Here, TruncatedSVD of sklearn is imported and and target names are visualized.

