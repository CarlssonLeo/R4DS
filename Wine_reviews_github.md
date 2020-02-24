Wine reviews
================

# Intro

Following [Julia Silge’s
guide](https://juliasilge.com/blog/intro-tidymodels/), I decided to try
to predict the score of wine, using the [Wine Reviews
dataset](https://www.kaggle.com/zynicide/wine-reviews) from Kaggle.

## Summary

First, just let’s look at the data.

``` r
library(skimr)
skim(wine)
```

|                                                  |        |
| :----------------------------------------------- | :----- |
| Name                                             | wine   |
| Number of rows                                   | 129971 |
| Number of columns                                | 10     |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_   |        |
| Column type frequency:                           |        |
| character                                        | 8      |
| numeric                                          | 2      |
| \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_ |        |
| Group variables                                  | None   |

Data summary

**Variable type:
character**

| skim\_variable | n\_missing | complete\_rate | min | max | empty | n\_unique | whitespace |
| :------------- | ---------: | -------------: | --: | --: | ----: | --------: | ---------: |
| country        |         63 |           1.00 |   2 |  22 |     0 |        43 |          0 |
| designation    |      37465 |           0.71 |   1 |  95 |     0 |     37979 |          0 |
| province       |         63 |           1.00 |   3 |  31 |     0 |       425 |          0 |
| region\_1      |      21247 |           0.84 |   3 |  50 |     0 |      1229 |          0 |
| region\_2      |      79460 |           0.39 |   4 |  17 |     0 |        17 |          0 |
| title          |          0 |           1.00 |  12 | 136 |     0 |    118840 |          0 |
| variety        |          1 |           1.00 |   4 |  35 |     0 |       707 |          0 |
| winery         |          0 |           1.00 |   1 |  54 |     0 |     16757 |          0 |

**Variable type:
numeric**

| skim\_variable | n\_missing | complete\_rate |  mean |    sd | p0 | p25 | p50 | p75 | p100 | hist  |
| :------------- | ---------: | -------------: | ----: | ----: | -: | --: | --: | --: | ---: | :---- |
| points         |          0 |           1.00 | 88.45 |  3.04 | 80 |  86 |  88 |  91 |  100 | ▂▇▇▂▁ |
| price          |       8996 |           0.93 | 35.36 | 41.02 |  4 |  17 |  25 |  42 | 3300 | ▇▁▁▁▁ |

There’s a lot of missing values all over. This needs to be dealt with,
in some form.

## EDA

Then, lets see if we can see something interesting by plotting the
points by country in a box plot. For ease of reading, I’ve sorted the
boxplot with the highest mean in descending order. Thanks to the R4DS
slack-group for helping me with this\!

``` r
wine %>% 
        drop_na(country) %>%
        mutate(country = fct_reorder(country, .x = points, .fun = mean)) %>% 
        ggplot(aes(x=country, y=points)) +
        geom_boxplot()+
        coord_flip()
```

![](Wine_reviews_github_files/figure-gfm/Boxplot-1.png)<!-- -->

And I don’t know about you, but I was higly surpriced to see that
England and India were the two countries with the on average best
wines\!

# Building a model

Now, let’s build a model\! First let’s drop the NA’s. Sadly, I don’t
know much imputation yet, so this is the way it has to be(for now\!).
Then split it into training and testing sets.

``` r
wine <- wine %>% drop_na()
set.seed(1234)
wine_split <- initial_split(wine)

wine_test <- testing(wine_split)
wine_train <- training(wine_split)
```

## Specs

Using a Random Forest in regression mode, we set the engine to use the
ranger package.

``` r
rf_spec <- rand_forest(mode = "regression") %>%
        set_engine("ranger")
rf_spec
```

    ## Random Forest Model Specification (regression)
    ## 
    ## Computational engine: ranger

## Fitting

Then let’s fit it, using `points` as the target and every other variable
as a predictor.

``` r
rf_fit <- rf_spec %>%
        fit(points ~.,
            data=wine_train)
rf_fit
```

    ## parsnip model object
    ## 
    ## Fit time:  29.6s 
    ## Ranger result
    ## 
    ## Call:
    ##  ranger::ranger(formula = formula, data = data, num.threads = 1,      verbose = FALSE, seed = sample.int(10^5, 1)) 
    ## 
    ## Type:                             Regression 
    ## Number of trees:                  500 
    ## Sample size:                      25563 
    ## Number of independent variables:  9 
    ## Mtry:                             3 
    ## Target node size:                 5 
    ## Variable importance mode:         none 
    ## Splitrule:                        variance 
    ## OOB prediction error (MSE):       5.54125 
    ## R squared (OOB):                  0.4196467

## Results

So, let’s look at the results\!

``` r
results_train <- rf_fit %>%
  predict(new_data = wine_train) %>%
  mutate(truth = wine_train$points) %>% 
        rmse(truth = truth, estimate = .pred) %>% 
        mutate(split = "training")

results_test <- rf_fit %>%
  predict(new_data = wine_test) %>%
  mutate(truth = wine_test$points) %>% 
        rmse(truth = truth, estimate = .pred) %>% 
        mutate(split = "test")

results <- results_train %>% bind_rows(results_test)
results
```

    ## # A tibble: 2 x 4
    ##   .metric .estimator .estimate split   
    ##   <chr>   <chr>          <dbl> <chr>   
    ## 1 rmse    standard        1.25 training
    ## 2 rmse    standard        2.62 test

Sadly, as we can see, the model seem to have been overfitted to the
training data. Let’s correct that\!

## Cross validating

``` r
set.seed(1234)
wine_folds <- vfold_cv(wine_train)

rf_res <- fit_resamples(
  points ~ .,
  rf_spec,
  wine_folds)

rf_res %>%
  collect_metrics()
```

which sadly gives me the following error: x Fold01: formula: Error in
`contrasts<-`(`*tmp*`, value = contr.funs\[1 + isOF\[nn\]\]): contrasts
can be applied only to factors with 2 or more… x Fold02: formula: Error
in `contrasts<-`(`*tmp*`, value = contr.funs\[1 + isOF\[nn\]\]):
contrasts can be applied only to factors with 2 or more… x Fold03:
formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs\[1 +
isOF\[nn\]\]): contrasts can be applied only to factors with 2 or more…
x Fold04: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs\[1
+ isOF\[nn\]\]): contrasts can be applied only to factors with 2 or
more… x Fold05: formula: Error in `contrasts<-`(`*tmp*`, value =
contr.funs\[1 + isOF\[nn\]\]): contrasts can be applied only to factors
with 2 or more… x Fold06: formula: Error in `contrasts<-`(`*tmp*`, value
= contr.funs\[1 + isOF\[nn\]\]): contrasts can be applied only to
factors with 2 or more… x Fold07: formula: Error in
`contrasts<-`(`*tmp*`, value = contr.funs\[1 + isOF\[nn\]\]): contrasts
can be applied only to factors with 2 or more… x Fold08: formula: Error
in `contrasts<-`(`*tmp*`, value = contr.funs\[1 + isOF\[nn\]\]):
contrasts can be applied only to factors with 2 or more… x Fold09:
formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs\[1 +
isOF\[nn\]\]): contrasts can be applied only to factors with 2 or more…
x Fold10: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs\[1
+ isOF\[nn\]\]): contrasts can be applied only to factors with 2 or
more… All models failed in \[fit\_resamples()\]. See the `.notes`
column.

Any ideas?
