library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(ggmosaic)

train <- vroom('Data//train.csv')
test <- vroom('Data//test.csv')

ggplot(data = train) +
  geom_bar(aes(x = ACTION))

ggplot(data = train) +
  geom_mosaic(aes(x = 'ROLE_ROLLUP_1', fill=ACTION))
summary(train$ROLE_ROLLUP_1)
summary(train$ROLE_ROLLUP_2)
summary(train$ROLE_DEPTNAME)
summary(train$ROLE_TITLE)
summary(train$ROLE_FAMILY)


length(unique(train$ROLE_ROLLUP_1))
length(unique(train$ROLE_ROLLUP_2))
length(unique(train$ROLE_DEPTNAME))
length(unique(train$ROLE_TITLE))
length(unique(train$ROLE_FAMILY))

smallplot <- train %>%
  filter(ROLE_FAMILY < 20000) %>%
  ggplot(aes(x = ROLE_FAMILY, y = ACTION)) +
  geom_col()
smallplot

smallplot2 <- train %>%
  filter(ROLE_FAMILY > 200000) %>%
  ggplot(aes(x = ROLE_FAMILY, y = ACTION)) +
  geom_col()
smallplot2
skimr::skim(train)


my_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())
  
  
prepped_recipe <- prep(my_recipe)
baked_data <- bake(prepped_recipe, new_data = NULL)
