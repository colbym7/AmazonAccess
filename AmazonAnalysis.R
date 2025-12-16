library(tidyverse)
library(tidymodels)
library(embed)
library(vroom)
library(rpart)
library(kknn)
library(naivebayes)
library(discrim)
library(keras)
library(themis)
# library(ggmosaic)

#C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess
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
            file=".\\Preds\\logreg_preds.csv", 
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
            file=".\\Preds\\pen_preds.csv", 
            delim=",")

## Random Forest ##
rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,30)),
                            min_n(),
                            levels = 4)

folds <- vfold_cv(train, v = 5, repeats = 1)
CV_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

rf_preds <- final_wf %>%
  predict(new_data=test, type='prob')

kaggle_submission3 <- data.frame(
  id = test$id,
  Action = rf_preds$.pred_1
)
vroom_write(x=kaggle_submission3, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//rf_preds.csv", 
            delim=",")


## KNN ##
knn_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_nominal_predictors())

knn_model <- nearest_neighbor(neighbors=tune()) %>%
  set_mode('classification') %>%
  set_engine('kknn')

knn_wf <- workflow() %>%
  add_recipe(knn_recipe) %>%
  add_model(knn_model)

tuning_grid <- grid_regular(neighbors(),
                            levels = 4)

folds <- vfold_cv(train, v = 5, repeats = 1)
CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

knn_preds <- final_wf %>%
  predict(new_data=test, type='prob')

kaggle_submission4 <- data.frame(
  id = test$id,
  Action = knn_preds$.pred_1
)
vroom_write(x=kaggle_submission4, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//knn_preds.csv", 
            delim=",")


## Naive Bayes ##
nb_model <- naive_Bayes(Laplace = tune(), smoothness=tune()) %>%
  set_mode('classification') %>%
  set_engine('naivebayes')

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(), smoothness(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)
CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

nb_preds <- final_wf %>%
  predict(new_data=test, type='prob')


kaggle_submission5 <- data.frame(
  id = test$id,
  Action = nb_preds$.pred_1
)
vroom_write(x=kaggle_submission5, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//nb_preds.csv", 
            delim=",")


nn_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors())

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine('keras') %>%
  set_mode('classification')

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 15)),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)
tuned_nn <- nn_wf %>%
  tune_grid(resamples = folds,
            grid=nn_tuneGrid,
            metrics = metric_set(roc_auc, accuracy))
bestTune <- tuned_nn %>%
  select_best(metric = 'roc_auc')

final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

nn_preds <- final_wf %>%
  predict(new_data=test, type='prob')


kaggle_submission6 <- data.frame(
  id = test$id,
  Action = nn_preds$.pred_1
)
vroom_write(x=kaggle_submission6, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//nn_preds.csv", 
            delim=",")

tuned_nn %>% collect_metrics() %>%
  filter(.metric=='accuracy') %>%
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()



  ### PCA ###
pca_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_pca(all_predictors(), threshold=.15)

prepped_recipe <- prep(pca_recipe)
baked_data <- bake(prepped_recipe, new_data = NULL)

pca_logmodel <- logistic_reg() %>%
  set_engine('glm')

pca_log_workflow <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(pca_logmodel) %>%
  fit(data=train)

pca_logpreds <- predict(pca_log_workflow,
                              new_data=test,
                              type='prob')

kaggle_submission7 <- data.frame(
  id = test$id,
  Action = pca_logpreds$.pred_1
)
vroom_write(x=kaggle_submission7, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//pca_logreg_preds.csv", 
            delim=",")


pen_mod <- logistic_reg(mixture=tune(), penalty = tune()) %>%
  set_engine('glmnet')

pen_workflow <- workflow() %>%
  add_recipe(pca_recipe) %>%
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

kaggle_submission8 <- data.frame(
  id = test$id,
  Action = pen_preds$.pred_1
)
vroom_write(x=kaggle_submission8, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//pca_pen_preds.csv", 
            delim=",")


rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,30)),
                            min_n(),
                            levels = 4)

folds <- vfold_cv(train, v = 5, repeats = 1)
CV_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

rf_preds <- final_wf %>%
  predict(new_data=test, type='prob')

kaggle_submission9 <- data.frame(
  id = test$id,
  Action = rf_preds$.pred_1
)
vroom_write(x=kaggle_submission9, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//pcarf_preds.csv", 
            delim=",")



### SVM ###
svmPoly <- svm_poly(degree=tune(), cost=tune()) %>%
  set_mode('classification') %>%
  set_engine('kernlab')

svmRadial <- svm_rbf(rbf_sigma=tune(), cost=tune()) %>%
  set_mode('classification') %>%
  set_engine('kernlab')

svmLinear <- svm_linear(cost=tune()) %>%
  set_mode('classification') %>%
  set_engine('kernlab')

svm1_workflow <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(svmPoly)

tuning_grid <- grid_regular(degree(),
                            cost(),
                            levels = 2)

folds <- vfold_cv(train, v = 3, repeats = 1)
CV_results <- svm1_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- svm1_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

svm1_preds <- final_wf %>%
  predict(new_data=test, type='prob')

kaggle_submission10 <- data.frame(
  id = test$id,
  Action = svm1_preds$.pred_1
)
vroom_write(x=kaggle_submission10, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//svm1_preds.csv", 
            delim=",")

# SVM Radial #
svm2_workflow <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(svmRadial)

tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 2)

folds <- vfold_cv(train, v = 3, repeats = 1)
CV_results <- svm2_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- svm2_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

svm2_preds <- final_wf %>%
  predict(new_data=test, type='prob')

kaggle_submission11 <- data.frame(
  id = test$id,
  Action = svm2_preds$.pred_1
)
vroom_write(x=kaggle_submission11, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//svm2_preds.csv", 
            delim=",")

## SVM Linear ##
svm3_workflow <- workflow() %>%
  add_recipe(pca_recipe) %>%
  add_model(svmLinear)

tuning_grid <- grid_regular(cost(),
                            levels = 3)

folds <- vfold_cv(train, v = 3, repeats = 1)
CV_results <- svm3_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- svm3_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

svm3_preds <- final_wf %>%
  predict(new_data=test, type='prob')

kaggle_submission12 <- data.frame(
  id = test$id,
  Action = svm3_preds$.pred_1
)
vroom_write(x=kaggle_submission12, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//svm3_preds.csv", 
            delim=",")


### SMOTE ###
smote_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_other(all_nominal_predictors(), threshold = .001) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_smote(all_outcomes(), neighbors=5) %>%
  step_downsample()

smote_logmodel <- logistic_reg() %>%
  set_engine('glm')

smote_log_workflow <- workflow() %>%
  add_recipe(smote_recipe) %>%
  add_model(smote_logmodel) %>%
  fit(data=train)

smote_logpreds <- predict(smote_log_workflow,
                        new_data=test,
                        type='prob')

kaggle_submission13 <- data.frame(
  id = test$id,
  Action = smote_logpreds$.pred_1
)
vroom_write(x=kaggle_submission13, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//smote_logreg_preds.csv", 
            delim=",")

# factor
# lencode mixed
# normalize
rf_recipe <- recipe(ACTION ~ ., data = train) %>%
  step_mutate_at(all_numeric_predictors(), fn = factor) %>%
  step_lencode_mixed(all_predictors(), outcome = vars(ACTION)) %>%
  step_normalize(all_predictors())
prepped_recipe <- prep(rf_recipe)
baked_data <- bake(prepped_recipe, new_data = NULL)



rf_mod <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) %>%
  set_engine("ranger") %>%
  set_mode("classification")

rf_workflow <- workflow() %>%
  add_recipe(rf_recipe) %>%
  add_model(rf_mod)

tuning_grid <- grid_regular(mtry(range=c(1,9)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(train, v = 5, repeats = 1)
CV_results <- rf_workflow %>%
  tune_grid(resamples = folds,
            grid=tuning_grid,
            metrics = metric_set(roc_auc))
bestTune <- CV_results %>%
  select_best(metric = 'roc_auc')

final_wf <- rf_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data=train)

rf_preds <- final_wf %>%
  predict(new_data=test, type='prob')

kaggle_submission14 <- data.frame(
  id = test$id,
  Action = rf_preds$.pred_1
)
vroom_write(x=kaggle_submission14, 
            file="C://Users//cjmsp//Desktop//Stat348//AmazonAccess//AmazonAccess//Preds//final_preds.csv", 
            delim=",")
