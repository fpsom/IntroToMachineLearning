[Go to main page](../README.md)

### Loading and exploring omics data

The data we are going to be using for this workshop are from the following two sources:

- the [Breast Cancer Wisconsin (Diagnostic) Data Set](http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29) from the [UCI Machine Learning repository](http://archive.ics.uci.edu/ml/).
- RNA-Seq data from the study of tooth growth in mouse embryos from the [Gene Expression Omnibus ID:GSE76316](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE76316)

We will first load up the UCI dataset; the dataset itself does not contain column names, but we've created a second file with only the column names that we will use.

```r
breastCancerData <- read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
               col_names = FALSE)

breastCancerDataColNames <- read_csv("https://raw.githubusercontent.com/fpsom/IntroToMachineLearning/gh-pages/data/wdbc.colnames.csv",
                                     col_names = FALSE)

colnames(breastCancerData) <- breastCancerDataColNames$X1

# Check out head of dataframe
breastCancerData %>% head()
```

If all goes well, we can see that our dataset contains 569 observations across 32 variables. This is what the first 6 lines look like:

```r
# A tibble: 6 x 32
      ID Diagnosis Radius.Mean Texture.Mean Perimeter.Mean Area.Mean Smoothness.Mean
   <dbl> <chr>           <dbl>        <dbl>          <dbl>     <dbl>           <dbl>
1 8.42e5 M                18.0         10.4          123.      1001           0.118
2 8.43e5 M                20.6         17.8          133.      1326           0.0847
3 8.43e7 M                19.7         21.2          130       1203           0.110
4 8.43e7 M                11.4         20.4           77.6      386.          0.142
5 8.44e7 M                20.3         14.3          135.      1297           0.100
6 8.44e5 M                12.4         15.7           82.6      477.          0.128
# ... with 25 more variables: Compactness.Mean <dbl>, Concavity.Mean <dbl>,
#   Concave.Points.Mean <dbl>, Symmetry.Mean <dbl>, Fractal.Dimension.Mean <dbl>,
#   Radius.SE <dbl>, Texture.SE <dbl>, Perimeter.SE <dbl>, Area.SE <dbl>,
#   Smoothness.SE <dbl>, Compactness.SE <dbl>, Concavity.SE <dbl>, Concave.Points.SE <dbl>,
#   Symmetry.SE <dbl>, Fractal.Dimension.SE <dbl>, Radius.Worst <dbl>, Texture.Worst <dbl>,
#   Perimeter.Worst <dbl>, Area.Worst <dbl>, Smoothness.Worst <dbl>,
#   Compactness.Worst <dbl>, Concavity.Worst <dbl>, Concave.Points.Worst <dbl>,
#   Symmetry.Worst <dbl>, Fractal.Dimension.Worst <dbl>
```

We will also make our `Diagnosis` column a factor (_Question: **What is a factor?**_):

```r
# Make Diagnosis a factor
breastCancerData$Diagnosis <- as.factor(breastCancerData$Diagnosis)
```

### What is Exploratory Data Analysis (EDA) and why is it useful?

Before thinking about modeling, have a look at your data. There is no point in throwing a 10000 layer convolutional neural network (whatever that means) at your data before you even know what you're dealing with.

You’ll first remove the first column, which is the unique identifier of each row (_Question: **why?**_):

```r
# Remove first column
breastCancerDataNoID <- breastCancerData[2:ncol(breastCancerData)]

# View head
breastCancerDataNoID %>% head()
```

The output should like like this:

```r
# A tibble: 6 x 31
  Diagnosis Radius.Mean Texture.Mean Perimeter.Mean Area.Mean Smoothness.Mean
  <fct>           <dbl>        <dbl>          <dbl>     <dbl>           <dbl>
1 M                18.0         10.4          123.      1001           0.118
2 M                20.6         17.8          133.      1326           0.0847
3 M                19.7         21.2          130       1203           0.110
4 M                11.4         20.4           77.6      386.          0.142
5 M                20.3         14.3          135.      1297           0.100
6 M                12.4         15.7           82.6      477.          0.128
# ... with 25 more variables: Compactness.Mean <dbl>, Concavity.Mean <dbl>,
#   Concave.Points.Mean <dbl>, Symmetry.Mean <dbl>, Fractal.Dimension.Mean <dbl>,
#   Radius.SE <dbl>, Texture.SE <dbl>, Perimeter.SE <dbl>, Area.SE <dbl>,
#   Smoothness.SE <dbl>, Compactness.SE <dbl>, Concavity.SE <dbl>, Concave.Points.SE <dbl>,
#   Symmetry.SE <dbl>, Fractal.Dimension.SE <dbl>, Radius.Worst <dbl>, Texture.Worst <dbl>,
#   Perimeter.Worst <dbl>, Area.Worst <dbl>, Smoothness.Worst <dbl>,
#   Compactness.Worst <dbl>, Concavity.Worst <dbl>, Concave.Points.Worst <dbl>,
#   Symmetry.Worst <dbl>, Fractal.Dimension.Worst <dbl>
```

We already have a lot of variables so, for the interest of time, we will focus only on the first five. Let's have a look at a plot:

```r
ggpairs(breastCancerDataNoID[1:5], aes(color=Diagnosis, alpha=0.4))
```

![ggpairs output of the first 5 variables](static/images/ggpairs5variables.png "ggpairs output of the first 5 variables")

### Unsupervised Learning.


### How could unsupervised learning be used to analyze omics data?
