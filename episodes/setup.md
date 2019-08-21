[Go to main page](../README.md)

## Download R and R Studio
1. Go to the [CRAN website](https://cran.r-project.org) and follow the instructions to download and install R.
2. Download and install [RStudio](https://www.rstudio.com/products/rstudio/download/#download).

## Installing additional packages
Open RStudio and install the following packages.

```{r}
# Run this cell to install & load the required packages
install.packages("tidyverse")
install.packages("kernlab")
install.packages("ddalpha")
install.packages("caret")
install.packages("GGally")
install.packages("gmodels")
install.packages("glmnet", repos = "http://cran.us.r-project.org")
install.packages("e1071")
```


Load them to make sure they were successfully installed.
```
# Load packages
library(tidyverse)
library(kernlab)
library(ddalpha)
library(caret)
library(GGally)
library(gmodels)
library(glmnet)
```
