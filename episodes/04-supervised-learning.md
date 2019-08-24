[Go to main page](../README.md)

## Supervised Learning

Supervised learning is the branch of Machine Learning (ML) that involves predicting labels, such as 'Survived' or 'Not'. Such models learn from labelled data, which is data that includes whether a passenger survived (called "model training"), and then predict on unlabeled data.

These are generally called train and test sets because
- You want to build a model that learns patterns in the training set, and
- You then use the model to make predictions on the test set.

We can then calculate the percentage that you got correct: this is known as the accuracy of your model.

### How To Start with Supervised Learning

As you might already know, a good way to approach supervised learning is the following:
- Perform an Exploratory Data Analysis (EDA) on your data set;
- Build a quick and dirty model, or a baseline model, which can serve as a comparison against later models that you will build;
- Iterate this process. You will do more EDA and build another model;
- Engineer features: take the features that you already have and combine them or extract more information from them to eventually come to the last point, which is
- Get a model that performs better.

A common practice in all supervised learning is the construction and use of the **train- and test- datasets**. This process takes all of the input randomly splits into the two datasets (training and test); the ratio of the split is usually up to the researcher, and can be anything: 80/20, 70/30, 60/40...

## Supervised Learning I: classification

There are various classifiers available:

- **Decision Trees** – These are organized in the form of sets of questions and answers in the tree structure.
- **Naive Bayes Classifiers** – A probabilistic machine learning model that is used for classification.
- **K-NN Classifiers** – Based on the similarity measures like distance, it classifies new cases.
- **Support Vector Machines** – It is a non-probabilistic binary linear classifier that builds a model to classify a case into one of the two categories.

### Decision trees

It is a type of supervised learning algorithm. We use it for classification problems. It works for both types of input and output variables. In this technique, we split the population into two or more homogeneous sets. Moreover, it is based on the most significant splitter/differentiator in input variables.

The Decision Tree is a powerful non-linear classifier. A Decision Tree makes use of a tree-like structure to generate relationship among the various features and potential outcomes. It makes use of branching decisions as its core structure.

There are two types of decision trees:
- **Categorical (classification)** Variable Decision Tree: Decision Tree which has a categorical target variable.
- **Continuous (Regression)** Variable Decision Tree: Decision Tree has a continuous target variable.

Regression trees are used when the dependent variable is continuous while classification trees are used when the dependent variable is categorical. In continuous, a value obtained is a mean response of observation. In classification, a value obtained by a terminal node is a mode of observations.

Here, we will use the `rpart` and the `rpart.plot` package in order to produce and visualize a decision tree. First of all, we'll create the train and test datasets using a 70/30 ratio and a fixed seed so that we can reproduce the results.

```r
# split into training and test subsets
set.seed(5)
ind <- sample(2, nrow(breastCancerData), replace=TRUE, prob=c(0.7, 0.3))
breastCancerData.train <- breastCancerData[ind==1,]
breastCancerData.test <- breastCancerData[ind==2,]
```

Now, we will load the library and create our model. We would like to create a model that predicts the `Diagnosis` based on the mean of the radius and the area, as well as the SE of the texture. For ths reason we'll use the notation of `myFormula <- Diagnosis ~ Radius.Mean + Area.Mean + Texture.SE`. If we wanted to create a prediction model based on all variables, we will have used `myFormula <- Diagnosis ~ .` instead. Finally, `minsplit` stands for the the minimum number of instances in a node so that it is split.

```r
library(rpart)
library(rpart.plot)
myFormula <- Diagnosis ~ Radius.Mean + Area.Mean + Texture.SE

breastCancerData.model <- rpart(myFormula,
                                method = "class",
                                data = breastCancerData.train,
                                minsplit = 10,
                                minbucket = 1,
                                maxdepth = 3,
                                cp = -1)

print(breastCancerData.model$cptable)
rpart.plot(breastCancerData.model)
```

We see the following output and a figure:

```
      CP       nsplit rel error   xerror     xstd
1  0.69930070      0 1.0000000 1.0000000 0.06688883
2  0.02797203      1 0.3006993 0.3006993 0.04330166
3  0.00000000      2 0.2727273 0.3006993 0.04330166
4 -1.00000000      6 0.2727273 0.3006993 0.04330166
```

![Full decision tree](https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/static/images/decisionTreeFull.png "Full decision tree")

The parameters that we used reflect the following aspects of the model:
- `minsplit`: the minimum number of instances in a node so that it is split
- `minbucket`: the minimum allowed number of instances in each leaf of the tree
- `maxdepth`: the maximum depth of the tree
- `cp`: parameter that controls the complexity for a split and is set intuitively (the larger its value, the more probable to apply pruning to the tree)

As we can observe, this might not be the best model. So we can select the tree with the minimum prediction error:

```r
opt <- which.min(breastCancerData.model$cptable[, "xerror"])
cp <- breastCancerData.model$cptable[opt, "CP"]
# prune tree
breastCancerData.pruned.model <- prune(breastCancerData.model, cp = cp)
# plot tree
rpart.plot(breastCancerData.pruned.model)

table(predict(breastCancerData.pruned.model, type="class"), breastCancerData.train$Diagnosis)
```

The output now is the following Confusion Matrix and pruned tree:

```r
    B    M
B  245  34
M   9   109
```

![Pruned decision tree](https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/static/images/decisionTreePruned.png "Pruned decision tree")

_Question: **What does the above "Confusion Matrix" tells you?**_

Now that we have a model, we should check how the prediction works in our test dataset.


```r
## make prediction
BreastCancer_pred <- predict(breastCancerData.pruned.model, newdata = breastCancerData.test, type="class")
plot(BreastCancer_pred ~ Diagnosis, data = breastCancerData.test,
     xlab = "Observed",
     ylab = "Prediction")
table(BreastCancer_pred, breastCancerData.test$Diagnosis)
```

The new Confusion Matrix is the following:

```r
BreastCancer_pred   B   M
                B 102  16
                M   1  53
```

![Prediction Plot](https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/static/images/predictionPlot.png "Prediction Plot")

| **Exercises**  |   |
|--------|----------|
| 1 | Can we improve the above model? What are the key parameters that have the most impact?|
| 2 | We have been using only some of the variables in our model. What is the impact of using all variables / features for our prediction? Is this a good or a bad plan?|

## Supervised Learning II: regression

### What if the target variable is numerical rather than categorical?
