library(tidymodels)
library(embed)
library(vroom)
library(kernlab)
library(themis)
library(randomForest)

train <- vroom("./train.csv")
test <- vroom("./test.csv")


colnames(train)[2] = "Date"
colnames(test)[2] = "Date"
colnames(train)[4] = "City_Group"
colnames(test)[4] = "City_Group"

train$Date <- as.Date(train$Date, "%m/%d/%y")
test$Date <- as.Date(test$Date, "%m/%d/%y")

my_recipe <- recipe(revenue ~ ., data = train) %>%
  step_date(Date, features = c("month", "dow", "year", "decimal")) %>% 
  step_other(all_nominal_predictors(), threshold = .05) %>% 
  step_rm(c(Id, Date))
  
prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)
view(baked)

set.seed(2662)

rf_model <- rand_forest(
  mtry = tune(),
  min_n = tune(),
  trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("regression")

rf_wf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(rf_model)

grid <- grid_regular(
  parameters(mtry(range = c(1, ncol(train) - 1)), 
             min_n()),
  levels = 5
)

folds <- vfold_cv(train, v = 5, repeats = 2)

cv_results <- rf_wf %>% 
  tune_grid(
    resamples = folds,
    grid = grid,
    metrics = metric_set(rmse, mae, rsq)
  )

besttune <- cv_results %>% 
  select_best(metric = "rmse")

final <- rf_wf %>% 
  finalize_workflow(besttune) %>% 
  fit(data = train) 

predict_rf <- final %>%
  predict(new_data = test)

kaggle_submission <- predict_rf %>% 
  bind_cols(., test) %>% 
  rename(Prediction = .pred) %>% 
  select(Id, Prediction)

vroom_write(x = kaggle_submission, file = "./Final_RF.csv" , delim = ",")

################KNN###########
library(kknn)

k_neighbor_mod <- nearest_neighbor(
  mode = "regression",
  engine = "kknn",
  neighbors = 6,
  weight_func = "rank",
  dist_power = NULL
)

knn_rf <- workflow() %>% 
  add_recipe(my_recipe) %>% 
  add_model(k_neighbor_mod) %>% 
  fit(data = train)

predict_k_neighbor <- knn_rf %>%
  predict(new_data = test)

kaggle_submission <- predict_k_neighbor %>% 
  bind_cols(., test) %>% 
  rename(Prediction = .pred) %>% 
  select(Id, Prediction)

vroom_write(x = kaggle_submission, file = "./Final_KNN.csv" , delim = ",")


########LINEAR MODEL########
new_train <- train
new_test <- test

new_train$year <- format(new_train$Date, "%Y")
new_train$month <- format(new_train$Date, "%m")
new_train$day <- format(new_train$Date, "%d")
new_train$year <- as.factor(new_train$year)
new_train$month <- as.factor(new_train$month)
new_train$day <- as.factor(new_train$day)
new_train$City_Group <- as.factor(new_train$City_Group)
new_train$Type <- as.factor(new_train$Type)
new_train$City <- as.factor(new_train$City)

new_train <- new_train %>% select(-Id, -Type, -City, -day)

new_test$year <- format(new_test$Date, "%Y")
new_test$month <- format(new_test$Date, "%m")
new_test$day <- format(new_test$Date, "%d")
new_test$year <- as.factor(new_test$year)
new_test$month <- as.factor(new_test$month)
new_test$day <- as.factor(new_test$day)
new_test$City_Group <- as.factor(new_test$City_Group)
new_test$Type <- as.factor(new_test$Type)
new_test$City <- as.factor(new_test$City)

new_test <- new_test %>% select(-Id, -Type, -City, -day)

levels(new_train)



my_linear_model <- linear_reg() %>% 
  set_engine("lm") %>% 
  set_mode("regression") %>% 
  fit(formula = revenue ~ ., data = new_train)

linear_predictions <- predict(my_linear_model,
                            new_data=new_test)

kaggle_submission <- linear_predictions %>% 
  bind_cols(., test) %>% 
  rename(Prediction = .pred) %>% 
  select(Id, Prediction)

vroom_write(x = kaggle_submission, file = "./Final_LINEAR.csv" , delim = ",")

(my_linear_model)
