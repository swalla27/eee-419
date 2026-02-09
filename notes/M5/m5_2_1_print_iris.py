# script to print the iris data set

from sklearn import datasets            # load the data sets
iris = datasets.load_iris()             # load the iris set

print(iris.data)                        # print the data

# data values:
print("sepal length, sepal width, petal length, petal width")

# data for iris types:
print("setosa = 0, versicolor = 1, and virginica = 2")
print(iris.target)

