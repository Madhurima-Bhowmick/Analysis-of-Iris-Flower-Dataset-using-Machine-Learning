# Analysis-of-Iris-Flower-Dataset-using-Machine-Learning

I have used various Machine Learning algorithms, specifically classified under Supervised and Unsupervised learning, to analyze the Iris Flower dataset. 
Here, I have used Fisher's Iris dataset,and downloaded from kaggle.com as a .csv file. Our task is to classify Iris flowers among three species( namely Setosa, Virginica and Versicolor), from measurements of length of sepals and petals.

### Iris-Versicolor


![330px-Iris_versicolor_3](https://user-images.githubusercontent.com/76059423/102658762-3aad3480-419e-11eb-96ee-a1418cb67da8.jpg)


### Iris-Virginica


![330px-Iris_virginica](https://user-images.githubusercontent.com/76059423/102659020-b6a77c80-419e-11eb-8dc6-1908cbe2b08d.jpg)


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
   3. Regression Tree
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

### 1. K-Nearest Neighbor

The KNN algorithm is simple and easy to use, and can be used for both classification and regression problems. Here, it is used for classification of Iris flowers. In this section, KNN is applied on the training data and then accuracy score is calculated, which is found to be 90.0%. An unknown flower can also be predicted, that it belongs to which species, by simply taking any random values for length and width of the sepal and petal respectively.

