Wine reviews
================

``` r
library(tidyverse)
```

    ## ── Attaching packages ────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse 1.2.1 ──

    ## ✓ ggplot2 3.2.1     ✓ purrr   0.3.3
    ## ✓ tibble  2.1.3     ✓ dplyr   0.8.4
    ## ✓ tidyr   1.0.0     ✓ stringr 1.4.0
    ## ✓ readr   1.3.1     ✓ forcats 0.4.0

    ## ── Conflicts ───────────────────────────────────────────────────────────────────────────────────────────────────────────── tidyverse_conflicts() ──
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(tidymodels)
```

    ## Registered S3 method overwritten by 'xts':
    ##   method     from
    ##   as.zoo.xts zoo

    ## ── Attaching packages ───────────────────────────────────────────────────────────────────────────────────────────────────────── tidymodels 0.1.0 ──

    ## ✓ broom     0.5.4     ✓ rsample   0.0.5
    ## ✓ dials     0.0.4     ✓ tune      0.0.1
    ## ✓ infer     0.5.1     ✓ workflows 0.1.0
    ## ✓ parsnip   0.0.5     ✓ yardstick 0.0.5
    ## ✓ recipes   0.1.9

    ## ── Conflicts ──────────────────────────────────────────────────────────────────────────────────────────────────────────── tidymodels_conflicts() ──
    ## x scales::discard() masks purrr::discard()
    ## x dplyr::filter()   masks stats::filter()
    ## x recipes::fixed()  masks stringr::fixed()
    ## x dplyr::lag()      masks stats::lag()
    ## x dials::margin()   masks ggplot2::margin()
    ## x yardstick::spec() masks readr::spec()
    ## x recipes::step()   masks stats::step()

``` r
wine <- read_csv("https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-28/winemag-data-130k-v2.csv") %>% select(-X1,-description,-taster_name, -taster_twitter_handle)
```

    ## Warning: Missing column names filled in: 'X1' [1]

    ## Parsed with column specification:
    ## cols(
    ##   X1 = col_double(),
    ##   country = col_character(),
    ##   description = col_character(),
    ##   designation = col_character(),
    ##   points = col_double(),
    ##   price = col_double(),
    ##   province = col_character(),
    ##   region_1 = col_character(),
    ##   region_2 = col_character(),
    ##   taster_name = col_character(),
    ##   taster_twitter_handle = col_character(),
    ##   title = col_character(),
    ##   variety = col_character(),
    ##   winery = col_character()
    ## )

# Intro

Following [Julia Silge’s
guide](https://juliasilge.com/blog/intro-tidymodels/), I decided to try
to predict the score of wine, using the [Wine Reviews
dataset](https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-05-28)
from R4DS’s Tidy Tuesday.

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
    ## Fit time:  25s 
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

which sadly gives me the following
error:

``` r
x Fold01: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold02: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold03: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold04: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold05: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold06: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold07: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold08: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold09: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
x Fold10: formula: Error in `contrasts<-`(`*tmp*`, value = contr.funs[1 + isOF[nn]]): contrasts can be applied only to factors with 2 or more...
All models failed in [fit_resamples()]. See the `.notes` column.
```

Any ideas?
