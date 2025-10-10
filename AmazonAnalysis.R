library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(ggmosaic)

train <- vroom('C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Data//train.csv') %>%
  mutate(ACTION=factor(ACTION))
test <- vroom('C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Data//test.csv')

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




logregmodel <- logistic_reg() %>%
  set_engine('glm')

log_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(logregmodel) %>%
  fit(data=train)

amazon_predictions <- predict(log_workflow,
                              new_data=test,
                              type='prob')

kaggle_submission1 <- data.frame(
  id = test$id,
  Action = amazon_predictions$.pred_1
)
vroom_write(x=kaggle_submission1, 
            file="C:\\Users\\cjmsp\\Desktop\\Stat348\\AmazonAccess\\AmazonAccess\\Preds\\logreg_preds.csv", 
            delim=",")


### Penalized Regression ###
penalty_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_lencode_mixed(all_nominal_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_numeric_predictors())


pen_mod <- logistic_reg(mixture=tune(), penalty = tune()) %>%
  set_engine('glmnet')

pen_workflow <- workflow() %>%
  add_recipe(penalty_recipe) %>%
  add_model(pen_mod)

tuning_grid <- grid_regular(penalty(),
                            mixture(),
                            levels = 4)

folds <- vfold_cv(train, v = 5, repeats = 1)
CV_results <- pen_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- pen_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

pen_preds <- final_wf %>%
  predict(new_data=test, type='prob')

kaggle_submission2 <- data.frame(
  id = test$id,
  Action = pen_preds$.pred_1
)
vroom_write(x=kaggle_submission2, 
            file="C:\\Users\\cjmsp\\Desktop\\Stat348\\AmazonAccess\\AmazonAccess\\Preds\\pen_preds.csv", 
            delim=",")
