# Modeling and evaluation 

library(tidyverse)
library(foreign)
library(haven)
library(readr)
library(ggplot2)
library(factoextra)
library(tidymodels)
library(caret)
library(glmnet)
library(pROC)
library(MLmetrics)
library(logistf)

########################### data preparation #########################

# Demographics only 
data_demo = read_csv("./002_data/data_wide.csv") |>
  dplyr::select(-1) |>
  select(seqn, gender, age, race, education, marital_status, pir, bmi) |>
  mutate(gender = as.character(gender), 
         race = as.character(race), 
         education = as.character(education), 
         marital_status = as.character(marital_status)) 

# GPT with 1536 embedding dimension 
data_gpt1536 = read.csv("./002_data/data_wide_embedding_gpt1536.csv") |>
  janitor::clean_names() |>
  dplyr::select(-x,-combined)

data_gpt1536$embedding = gsub("\\[", "", data_gpt1536$embedding)
data_gpt1536$embedding = gsub("\\]", "", data_gpt1536$embedding)

data_gpt1536 = data_gpt1536 |>
  separate(embedding, into = paste0("var", 1:1536), sep = ",\\s*", convert = TRUE) |>
  mutate(across(starts_with("var"), as.numeric))

data_gpt1536$gender <- as.character(data_gpt1536$gender)
data_gpt1536$race <- as.character(data_gpt1536$race)
data_gpt1536$education <- as.character(data_gpt1536$education)
data_gpt1536$married <- as.character(data_gpt1536$married)


### added for fairness
# GPT truncated with 1536 embedding dimension 
trunc_gpt1536 = read.csv("./002_data/trunc_wide_embedding_gpt1536.csv") |>
  janitor::clean_names() |>
  dplyr::select(-x,-combined)

trunc_gpt1536$embedding = gsub("\\[", "", trunc_gpt1536$embedding)
trunc_gpt1536$embedding = gsub("\\]", "", trunc_gpt1536$embedding)

trunc_gpt1536 = trunc_gpt1536 |>
  separate(embedding, into = paste0("var", 1:1536), sep = ",\\s*", convert = TRUE) |>
  mutate(across(starts_with("var"), as.numeric))

trunc_gpt1536$gender <- as.character(trunc_gpt1536$gender)
trunc_gpt1536$race <- as.character(trunc_gpt1536$race)
trunc_gpt1536$education <- as.character(trunc_gpt1536$education)
trunc_gpt1536$marital_status <- as.character(trunc_gpt1536$marital_status)

# BERT with 768 embedding dimension
data_bert768 = read.csv("./002_data/data_wide_embedding_bert768.csv") |>
  janitor::clean_names() |>
  dplyr::select(-x,-combined)

data_bert768$embedding <- gsub("\\[", "", data_bert768$embedding)
data_bert768$embedding <- gsub("\\]", "", data_bert768$embedding)
data_bert768$embedding <- gsub("\n", " ", data_bert768$embedding)

data_bert768 = data_bert768 |>
  mutate(embedding = str_trim(embedding),  # Remove leading and trailing spaces
         embedding = str_replace_all(embedding, "\\s+", " ")) |> # Replace multiple spaces with a single space 
  separate(embedding, into = paste0("var", 1:768), sep = "\\s+", convert = TRUE) |>
  mutate(across(starts_with("var"), as.numeric))

data_bert768$gender <- as.character(data_bert768$gender)
data_bert768$race <- as.character(data_bert768$race)
data_bert768$education <- as.character(data_bert768$education)
data_bert768$married <- as.character(data_bert768$married)

# Cohere with 1024 embedding dimension
data_cohere1024 = read.csv("./002_data/data_wide_embedding_cohere1024.csv") |>
  janitor::clean_names() |>
  dplyr::select(-x,-n_tokens)

data_cohere1024$embedding = gsub("\\[", "", data_cohere1024$embedding) 
data_cohere1024$embedding = gsub("\\]", "", data_cohere1024$embedding)
data_cohere1024$embedding = gsub(",", " ", data_cohere1024$embedding)

data_cohere1024 = data_cohere1024 |> 
  mutate(embedding = str_trim(embedding),  # Remove leading and trailing spaces
         embedding = str_replace_all(embedding, "\\s+", " ")) |> # Replace multiple spaces with a single space
  separate(embedding, into = paste0("var", 1:1024), sep = "\\s+", convert = TRUE) |>
  mutate(across(starts_with("var"), as.numeric))

data_cohere1024$gender <- as.character(data_cohere1024$gender)
data_cohere1024$race <- as.character(data_cohere1024$race)
data_cohere1024$education <- as.character(data_cohere1024$education)
data_cohere1024$married <- as.character(data_cohere1024$married)

# Entropy
data_entropy = read.csv("./002_data/data_wide_entropy.csv") |>
  janitor::clean_names() |>
  dplyr::select(-x)

data_entropy$gender <- as.character(data_entropy$gender)
data_entropy$race <- as.character(data_entropy$race)
data_entropy$education <- as.character(data_entropy$education)
data_entropy$married <- as.character(data_entropy$married)

# GPT1536 + entropy
data_gpt1536_entropy = data_entropy |>
  dplyr::select(seqn, entropy_day1:entropy_day7) |> 
  inner_join(data_gpt1536, by = "seqn") |>
  select("seqn", "gender", "age", "race", "education", "married", "pir", "bmi", everything())

### added for fairness
# GPT1536 truncated + entropy
trunc_gpt1536_entropy = data_entropy |>
  dplyr::select(seqn, entropy_day1:entropy_day7) |> 
  inner_join(trunc_gpt1536, by = "seqn") |>
  select("seqn", "gender", "age", "race", "education", "marital_status", "pir", "bmi", everything())

# BERT768 + entropy
data_bert768_entropy = data_entropy |>
  dplyr::select(seqn, entropy_day1:entropy_day7) |> 
  inner_join(data_bert768, by = "seqn") |>
  select("seqn", "gender", "age", "race", "education", "married", "pir", "bmi", everything())

# Cohere1024 + entropy
data_cohere1024_entropy = data_entropy |>
  dplyr::select(seqn, entropy_day1:entropy_day7) |> 
  inner_join(data_cohere1024, by = "seqn") |>
  select("seqn", "gender", "age", "race", "education", "married", "pir", "bmi", everything())


# MOMENT with 1024 embedding dimension  
data_moment1024 = read.csv("./002_data/embeddings_min5_moment1024.csv") |>
  janitor::clean_names() |>
  select(-x) |>
  arrange(seqn) 

data_moment1024$gender <- as.character(data_moment1024$gender)
data_moment1024$race <- as.character(data_moment1024$race)
data_moment1024$education <- as.character(data_moment1024$education)
data_moment1024$marital_status <- as.character(data_moment1024$marital_status)

# MOMENT raw data embedding with 1024 dimension  
data_raw_moment1024 = read.csv("./002_data/embeddings_min1_moment1024.csv") |>
  janitor::clean_names() |>
  select(-x) |>
  arrange(seqn) 
data_raw_moment1024$gender <- as.character(data_raw_moment1024$gender)
data_raw_moment1024$race <- as.character(data_raw_moment1024$race)
data_raw_moment1024$education <- as.character(data_raw_moment1024$education)
data_raw_moment1024$married <- as.character(data_raw_moment1024$married)

### added for fairness
# MOMENT min20 data embedding with 1024 dimension  
data_min20_moment1024 = read_csv("./002_data/embeddings_min20_moment1024.csv") |>
  mutate(gender = as.character(gender), 
         race = as.character(race), 
         education = as.character(education), 
         marital_status = as.character(marital_status)) 


# Clinical covariates and outcomes 
load("002_data/clinical_cov.RData")
load("002_data/outcomes.RData")




########################### data for modeling ########################### 

# EntroGPT + clinical covariates + outcomes 
data_gpt1536_entropy_cov2 = data_gpt1536_entropy |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909

### added for fairness
## EntroGPT truncated + clinical covariates + outcomes 
trunc_gpt1536_entropy_cov2 = trunc_gpt1536_entropy |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909

# EntroBERT + clinical covariates + outcomes 
data_bert768_entropy_cov2 = data_bert768_entropy |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909

# EntroCohere + clinical covariates + outcomes 
data_cohere1024_entropy_cov2 = data_cohere1024_entropy |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909

# Moment + clinical covariates + outcomes 
data_moment1024_cov2 = data_moment1024 |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909

# Raw Moment + clinical covariates + outcomes 
data_raw_moment1024_cov2 = data_raw_moment1024 |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909

### added for fairness
# min20 Moment + clinical covariates + outcomes  
data_min20_moment1024_cov2 = data_min20_moment1024 |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909

# demo + clinical covariates + outcomes 
data_demo_cov2 = data_demo |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909

# entropy + clinical covariates + outcomes 
data_entropy_cov2 = data_entropy |>
  left_join(clinical_cov, by = "seqn") |> 
  left_join(outcomes, by = "seqn") |> 
  drop_na() ## 2909



##### Variables #####  
# med_sbp: median systolic blood pressure in mmHg
# med_dbp: median diastolic blood pressure in mmHg
# chol_total: choleterol in mg/dL
# chol_ldl: LDL in mg/dL
# chol_hdl: HDL in mg/dL 
# glucose in mg/dL 
# triglyceride in mg/dL 
# crp in mg/dL 
# malig: ever told to have cancer of malignancy by doctor (1=yes, 0=no) 
# breat cancer (1=yes, 0=no) 
# skin cancer (1=yes, 0=no) 


# Convert some covariates into cateogrical (keep both cont and cat)

make_clinical_categories <- function(df) {
  df |>
    mutate(
      chol_total_cat = dplyr::case_when(chol_total < 200 ~ 0, chol_total >= 200 ~ 1),
      chol_ldl_cat = dplyr::case_when(chol_ldl < 100 ~ 0, chol_ldl >= 100 ~ 1),
      chol_hdl_cat = dplyr::case_when(chol_hdl >= 60 ~ 0, chol_hdl < 60 ~ 1),
      trigly_cat = dplyr::case_when(triglyceride < 150 ~ 0, triglyceride >= 150 ~ 1),
      gluc_cat = dplyr::case_when(glucose < 100 ~ 0, glucose >= 100 ~ 1),
      sbp_cat = dplyr::case_when(med_sbp < 120 ~ 0, med_sbp >= 120 ~ 1),
      dbp_cat = dplyr::case_when(med_dbp < 80 ~ 0, med_dbp >= 80 ~ 1),
      age_cat = dplyr::case_when(age >= 20 & age <= 39 ~ 0,
                                 age >= 40 & age <= 64 ~ 1,
                                 age >= 65             ~ 2)) |>
    mutate(across(c(chol_total_cat, chol_ldl_cat, chol_hdl_cat, trigly_cat,
                    gluc_cat, sbp_cat, dbp_cat, age_cat), as.character))
}

data_gpt1536_entropy_cat2 = make_clinical_categories(data_gpt1536_entropy_cov2)
data_bert768_entropy_cat2 = make_clinical_categories(data_bert768_entropy_cov2)
data_cohere1024_entropy_cat2 <- make_clinical_categories(data_cohere1024_entropy_cov2)
data_moment1024_cat2 <- make_clinical_categories(data_moment1024_cov2)
data_raw_moment1024_cat2 <- make_clinical_categories(data_raw_moment1024_cov2)
data_demo_cat2 <- make_clinical_categories(data_demo_cov2)
data_entropy_cat2 <- make_clinical_categories(data_entropy_cov2)

data_min20_moment1024_cat2 = make_clinical_categories(data_min20_moment1024_cov2)
trunc_gpt1536_entropy_cat2 = make_clinical_categories(trunc_gpt1536_entropy_cov2)

##### Cont -> Cat Variables #####  
# chol_total_cat: 0=low risk, 1=high risk 
# chol_ldl_cat: 0=low risk, 1=high risk 
# chol_hdl_cat: 0=low risk, 1=high risk 
# trigly_cat: 0=low risk, 1=high risk 
# gluc_cat: 0=low risk, 1=high risk 
# age: 0=20~39, 1=40~64, 2=65+ 
# sbp_cat: 0=normal, 1=at risk 
# dbp_cat: 0=normal, 1=at risk 



########################### modeling #########################

##### RMSE metrics - cholesterol #####
models_cov_chol = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))
  for(i in 1:sim){
    set.seed(i)

    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |> select(-chol_total) |> as.matrix()
    train_y = train$chol_total
    test_x = test |> select(-chol_total) |> as.matrix()
    test_y = test$chol_total
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet", 
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)))
    pred_ridge = predict(ridge, newdata = test_x) 
    
    metrics = postResample(pred = pred_ridge, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}
entropy_model_cov_chol = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |> select(-chol_total) |> as.matrix()
    train_y = train$chol_total
    test_x = test |> select(-chol_total) |> as.matrix()
    test_y = test$chol_total
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### glm  
    glm = train(chol_total~., 
                data = train,
                method = "glm",
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"))
    
    pred_class = predict(glm, newdata = test)
    
    metrics = postResample(pred = pred_class, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}

res_moment_cov_chol = models_cov_chol(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_chol = models_cov_chol(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_chol = models_cov_chol(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_chol = models_cov_chol(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_chol = models_cov_chol(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_chol_2 = models_cov_chol(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_chol = entropy_model_cov_chol(data_entropy_cov2,sim = 10, "Entropy_cov")

res_cov_chol = rbind(res_moment_cov_chol, res_raw_moment_cov_chol, 
                     res_gpt_cov_chol, res_bert_cov_chol, res_cohere_cov_chol,
                     res_demo_cov_chol, res_entropy_cov_chol)
res_cov_chol_avg = res_cov_chol |>
  group_by(method) |>
  summarize(
    avg_rmse = mean(test_rmse),
    avg_rsquared = mean(test_rsquared), 
    avg_mae = mean(test_mae),
    .groups = 'drop') |>
  arrange(method)


##### RMSE metrics - LDL ##### 
models_cov_ldl = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, 
                           triglyceride, glucose, crp))
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |> select(-chol_ldl) |> as.matrix()
    train_y = train$chol_ldl
    test_x = test |> select(-chol_ldl) |> as.matrix()
    test_y = test$chol_ldl
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet", 
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)))
    pred_ridge = predict(ridge, newdata = test_x) 
    
    metrics = postResample(pred = pred_ridge, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}
entropy_model_cov_ldl = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, 
                           triglyceride, glucose, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |> select(-chol_ldl) |> as.matrix()
    train_y = train$chol_ldl
    test_x = test |> select(-chol_ldl) |> as.matrix()
    test_y = test$chol_ldl
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### glm  
    glm = train(chol_ldl~., 
                data = train,
                method = "glm",
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"))
    
    pred_class = predict(glm, newdata = test) 
    
    metrics = postResample(pred = pred_class, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}

res_moment_cov_ldl = models_cov_ldl(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_ldl = models_cov_ldl(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_ldl = models_cov_ldl(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_ldl = models_cov_ldl(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_ldl = models_cov_ldl(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_ldl = models_cov_ldl(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_ldl = entropy_model_cov_ldl(data_entropy_cov2,sim = 10, "Entropy_cov")

res_cov_ldl = rbind(res_moment_cov_ldl, res_raw_moment_cov_ldl, 
                    res_gpt_cov_ldl, res_bert_cov_ldl, res_cohere_cov_ldl,
                    res_demo_cov_ldl, res_entropy_cov_ldl)
res_cov_ldl_avg = res_cov_ldl |>
  group_by(method) |>
  summarize(
    avg_rmse = mean(test_rmse),
    avg_rsquared = mean(test_rsquared), 
    avg_mae = mean(test_mae),
    .groups = 'drop') 


##### RMSE metrics - HDL ##### 
models_cov_hdl = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_ldl, 
                           triglyceride, glucose, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-chol_hdl) |> 
      as.matrix()
    train_y = train$chol_hdl
    
    test_x = test |>
      select(-chol_hdl) |>
      as.matrix()
    test_y = test$chol_hdl
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet", 
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)))
    pred_ridge = predict(ridge, newdata = test_x)
    
    metrics = postResample(pred = pred_ridge, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}
entropy_model_cov_hdl = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_ldl, 
                           triglyceride, glucose, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-chol_hdl) |> 
      as.matrix()
    train_y = train$chol_hdl
    
    test_x = test |>
      select(-chol_hdl) |>
      as.matrix()
    test_y = test$chol_hdl
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### glm  
    glm = train(chol_hdl~., 
                data = train,
                method = "glm",
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"))
    
    pred_class = predict(glm, newdata = test) 
    
    metrics = postResample(pred = pred_class, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}

res_moment_cov_hdl = models_cov_hdl(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_hdl = models_cov_hdl(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_hdl = models_cov_hdl(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_hdl = models_cov_hdl(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_hdl = models_cov_hdl(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_hdl = models_cov_hdl(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_hdl = entropy_model_cov_hdl(data_entropy_cov2,sim = 10, "Entropy_cov")

res_cov_hdl = rbind(res_moment_cov_hdl, res_raw_moment_cov_hdl, 
                    res_gpt_cov_hdl, res_bert_cov_hdl, res_cohere_cov_hdl,
                    res_demo_cov_hdl, res_entropy_cov_hdl)
res_cov_hdl_avg = res_cov_hdl |>
  group_by(method) |>
  summarize(
    avg_rmse = mean(test_rmse),
    avg_rsquared = mean(test_rsquared), 
    avg_mae = mean(test_mae),
    .groups = 'drop') 


##### RMSE metrics - triglyceride #####
models_cov_trigly = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           glucose, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-triglyceride) |> 
      as.matrix()
    train_y = train$triglyceride
    
    test_x = test |>
      select(-triglyceride) |>
      as.matrix()
    test_y = test$triglyceride
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)))
    pred_ridge = predict(ridge, newdata = test_x) 
    
    metrics = postResample(pred = pred_ridge, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}
entropy_model_cov_trigly = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           glucose, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-triglyceride) |> 
      as.matrix()
    train_y = train$triglyceride
    
    test_x = test |>
      select(-triglyceride) |>
      as.matrix()
    test_y = test$triglyceride
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### glm  
    glm = train(triglyceride~., 
                data = train,
                method = "glm",
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"))
    
    pred_class = predict(glm, newdata = test) 
    
    metrics = postResample(pred = pred_class, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}

res_moment_cov_trigly = models_cov_trigly(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_trigly = models_cov_trigly(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_trigly = models_cov_trigly(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_trigly = models_cov_trigly(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_trigly = models_cov_trigly(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_trigly = models_cov_trigly(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_trigly = entropy_model_cov_trigly(data_entropy_cov2,sim = 10, "Entropy_cov")

res_cov_trigly = rbind(res_moment_cov_trigly, res_raw_moment_cov_trigly, 
                       res_gpt_cov_trigly, res_bert_cov_trigly, res_cohere_cov_trigly,
                       res_demo_cov_trigly, res_entropy_cov_trigly)
res_cov_trigly_avg = res_cov_trigly |>
  group_by(method) |>
  summarize(
    avg_rmse = mean(test_rmse),
    avg_rsquared = mean(test_rsquared), 
    avg_mae = mean(test_mae),
    .groups = 'drop') 


##### RMSE metrics - glucose #####
models_cov_gluc = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-glucose) |> 
      as.matrix()
    train_y = train$glucose
    
    test_x = test |>
      select(-glucose) |>
      as.matrix()
    test_y = test$glucose
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)))
    pred_ridge = predict(ridge, newdata = test_x) 
    
    metrics = postResample(pred = pred_ridge, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}
entropy_model_cov_gluc = function(data, sim, name){
  res = list()
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-glucose) |> 
      as.matrix()
    train_y = train$glucose
    
    test_x = test |>
      select(-glucose) |>
      as.matrix()
    test_y = test$glucose
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE)
    
    ### glm  
    glm = train(glucose~., 
                data = train,
                method = "glm",
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"))
    
    pred_class = predict(glm, newdata = test) 
    
    metrics = postResample(pred = pred_class, obs = test_y)
    
    rmse = metrics["RMSE"]
    rsquared = metrics["Rsquared"]
    mae = metrics["MAE"]
    
    # output result 
    res[[i]] = list(rmse = rmse, rsquared = rsquared, mae = mae)
  }
  
  test_rmse = sapply(res, function(x) x$rmse)
  test_rsquared = sapply(res, function(x) x$rsquared)
  test_mae = sapply(res, function(x) x$mae)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_rmse = test_rmse, 
                         test_rsquared = test_rsquared, 
                         test_mae = test_mae)
  return(final_res)
}

res_moment_cov_gluc = models_cov_gluc(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_gluc = models_cov_gluc(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_gluc = models_cov_gluc(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_gluc = models_cov_gluc(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_gluc = models_cov_gluc(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_gluc = models_cov_gluc(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_gluc = entropy_model_cov_gluc(data_entropy_cov2,sim = 10, "Entropy_cov")

res_cov_gluc = rbind(res_moment_cov_gluc, res_raw_moment_cov_gluc, 
                     res_gpt_cov_gluc, res_bert_cov_gluc, res_cohere_cov_gluc,
                     res_demo_cov_gluc, res_entropy_cov_gluc)
res_cov_gluc_avg = res_cov_gluc |>
  group_by(method) |>
  summarize(
    avg_rmse = mean(test_rmse),
    avg_rsquared = mean(test_rsquared), 
    avg_mae = mean(test_mae),
    .groups = 'drop') 


##### AUC metrics - cholesterol #####
models_cat_chol = function(data, sim, name){
  res = list()
  data$chol_total_cat = factor(data$chol_total_cat, 
                               levels = c("0", "1"), 
                               labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_ldl_cat, chol_hdl_cat, trigly_cat, 
                           gluc_cat, sbp_cat, dbp_cat, age_cat))
  
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-chol_total_cat) |> 
      as.matrix()
    train_y = train$chol_total_cat
    
    test_x = test |>
      select(-chol_total_cat) |>
      as.matrix()
    test_y = test$chol_total_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  family = "binomial",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)), 
                  metric = "ROC")
    
    pred_class = predict(ridge, newdata = test)
    pred_probs = predict(ridge, newdata = test_x, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}
entropy_model_cat_chol = function(data, sim, name){
  res = list()
  data$chol_total_cat = factor(data$chol_total_cat, 
                               levels = c("0", "1"), 
                               labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_ldl_cat, chol_hdl_cat, trigly_cat, 
                           gluc_cat, sbp_cat, dbp_cat, age_cat))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-chol_total_cat) |> 
      as.matrix()
    train_y = train$chol_total_cat
    
    test_x = test |>
      select(-chol_total_cat) |>
      as.matrix()
    test_y = test$chol_total_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### glm   
    glm = train(chol_total_cat ~ ., 
                data = train,
                method = "glm",   
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"), 
                metric = "ROC")
    
    pred_class = predict(glm, newdata = test)
    pred_probs = predict(glm, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}

res_moment_cat_chol = models_cat_chol(data_moment1024_cat2, sim = 10, "Moment_cat")
res_raw_moment_cat_chol = models_cat_chol(data_raw_moment1024_cat2, sim = 10, "Raw_Moment_cat")
res_gpt_cat_chol = models_cat_chol(data_gpt1536_entropy_cat2, sim = 10, "EntroGPT_cat")
res_bert_cat_chol = models_cat_chol(data_bert768_entropy_cat2, sim = 10, "EntroBert_cat")
res_cohere_cat_chol = models_cat_chol(data_cohere1024_entropy_cat2, sim = 10, "EntroCohere_cat")
res_demo_cat_chol = models_cat_chol(data_demo_cat2, sim = 10, "Demo_cat")
res_entropy_cat_chol = entropy_model_cat_chol(data_entropy_cat2,sim = 10, "Entropy_cat")

res_cat_chol = rbind(res_moment_cat_chol, res_raw_moment_cat_chol, 
                     res_gpt_cat_chol, res_bert_cat_chol, res_cohere_cat_chol,
                     res_demo_cat_chol, res_entropy_cat_chol)
res_cat_chol_avg = res_cat_chol |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))


##### AUC metrics - LDL #####
models_cat_ldl = function(data, sim, name){
  res = list()
  data$chol_ldl_cat = factor(data$chol_ldl_cat, 
                             levels = c("0", "1"), 
                             labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_total_cat, chol_hdl_cat, trigly_cat, 
                           gluc_cat, sbp_cat, dbp_cat, age_cat))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-chol_ldl_cat) |> 
      as.matrix()
    train_y = train$chol_ldl_cat
    
    test_x = test |>
      select(-chol_ldl_cat) |>
      as.matrix()
    test_y = test$chol_ldl_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  family = "binomial",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)), 
                  metric = "ROC")
    
    pred_class = predict(ridge, newdata = test)
    pred_probs = predict(ridge, newdata = test_x, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}
entropy_model_cat_ldl = function(data, sim, name){
  res = list()
  data$chol_ldl_cat = factor(data$chol_ldl_cat, 
                             levels = c("0", "1"), 
                             labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_total_cat, chol_hdl_cat, trigly_cat, 
                           gluc_cat, sbp_cat, dbp_cat, age_cat))
  
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-chol_ldl_cat) |> 
      as.matrix()
    train_y = train$chol_ldl_cat
    
    test_x = test |>
      select(-chol_ldl_cat) |>
      as.matrix()
    test_y = test$chol_ldl_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### glm   
    glm = train(chol_ldl_cat ~ ., 
                data = train,
                method = "glm", 
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"), 
                metric = "ROC")
    
    pred_class = predict(glm, newdata = test)
    pred_probs = predict(glm, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}

res_moment_cat_ldl = models_cat_ldl(data_moment1024_cat2, sim = 10, "Moment_cat")
res_raw_moment_cat_ldl = models_cat_ldl(data_raw_moment1024_cat2, sim = 10, "Raw_Moment_cat")
res_gpt_cat_ldl = models_cat_ldl(data_gpt1536_entropy_cat2, sim = 10, "EntroGPT_cat")
res_bert_cat_ldl = models_cat_ldl(data_bert768_entropy_cat2, sim = 10, "EntroBert_cat")
res_cohere_cat_ldl = models_cat_ldl(data_cohere1024_entropy_cat2, sim = 10, "EntroCohere_cat")
res_demo_cat_ldl = models_cat_ldl(data_demo_cat2, sim = 10, "Demo_cat")
res_entropy_cat_ldl = entropy_model_cat_ldl(data_entropy_cat2, sim = 10, "Entropy_cat")

res_cat_ldl = rbind(res_moment_cat_ldl, res_raw_moment_cat_ldl, 
                    res_gpt_cat_ldl, res_bert_cat_ldl, res_cohere_cat_ldl, 
                    res_demo_cat_ldl, res_entropy_cat_ldl) 
res_cat_ldl_avg = res_cat_ldl |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc),
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))


##### AUC metrics - HDL #####
models_cat_hdl = function(data, sim, name){
  res = list()
  data$chol_hdl_cat = factor(data$chol_hdl_cat, 
                             levels = c("0", "1"), 
                             labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_total_cat, chol_ldl_cat, trigly_cat, 
                           gluc_cat, sbp_cat, dbp_cat, age_cat))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-chol_hdl_cat) |> 
      as.matrix()
    train_y = train$chol_hdl_cat
    
    test_x = test |>
      select(-chol_hdl_cat) |>
      as.matrix()
    test_y = test$chol_hdl_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  family = "binomial",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)), 
                  metric = "ROC")
    
    pred_class = predict(ridge, newdata = test)
    pred_probs = predict(ridge, newdata = test_x, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}
entropy_model_cat_hdl = function(data, sim, name){
  res = list()
  data$chol_hdl_cat = factor(data$chol_hdl_cat, 
                             levels = c("0", "1"), 
                             labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_total_cat, chol_ldl_cat, trigly_cat, 
                           gluc_cat, sbp_cat, dbp_cat, age_cat))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-chol_hdl_cat) |> 
      as.matrix()
    train_y = train$chol_hdl_cat
    
    test_x = test |>
      select(-chol_hdl_cat) |>
      as.matrix()
    test_y = test$chol_hdl_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### glm   
    glm = train(chol_hdl_cat ~ ., 
                data = train,
                method = "glm", 
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"), 
                metric = "ROC")
    
    pred_class = predict(glm, newdata = test)
    pred_probs = predict(glm, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}

res_moment_cat_hdl = models_cat_hdl(data_moment1024_cat2, sim = 10, "Moment_cat")
res_raw_moment_cat_hdl = models_cat_hdl(data_raw_moment1024_cat2, sim = 10, "Raw_Moment_cat")
res_gpt_cat_hdl = models_cat_hdl(data_gpt1536_entropy_cat2, sim = 10, "EntroGPT_cat")
res_bert_cat_hdl = models_cat_hdl(data_bert768_entropy_cat2, sim = 10, "EntroBert_cat")
res_cohere_cat_hdl = models_cat_hdl(data_cohere1024_entropy_cat2, sim = 10, "EntroCohere_cat")
res_demo_cat_hdl = models_cat_hdl(data_demo_cat2, sim = 10, "Demo_cat")
res_entropy_cat_hdl = entropy_model_cat_hdl(data_entropy_cat2, sim = 10, "Entropy_cat")

res_cat_hdl = rbind(res_moment_cat_hdl, res_raw_moment_cat_hdl, 
                    res_gpt_cat_hdl, res_bert_cat_hdl, res_cohere_cat_hdl,
                    res_demo_cat_hdl, res_entropy_cat_hdl) 
res_cat_hdl_avg = res_cat_hdl |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc),
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))


##### AUC metrics - triglyceride #####
models_cat_trigly = function(data, sim, name){
  res = list()
  data$trigly_cat = factor(data$trigly_cat, 
                           levels = c("0", "1"), 
                           labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_total_cat, chol_ldl_cat, chol_hdl_cat, 
                           gluc_cat, sbp_cat, dbp_cat, age_cat))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-trigly_cat) |> 
      as.matrix()
    train_y = train$trigly_cat
    
    test_x = test |>
      select(-trigly_cat) |>
      as.matrix()
    test_y = test$trigly_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  family = "binomial",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)), 
                  metric = "ROC")
    
    pred_class = predict(ridge, newdata = test)
    pred_probs = predict(ridge, newdata = test_x, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}
entropy_model_cat_trigly = function(data, sim, name){
  res = list()
  data$trigly_cat = factor(data$trigly_cat, 
                           levels = c("0", "1"), 
                           labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_total_cat, chol_ldl_cat, chol_hdl_cat, 
                           gluc_cat, sbp_cat, dbp_cat, age_cat))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-trigly_cat) |> 
      as.matrix()
    train_y = train$trigly_cat
    
    test_x = test |>
      select(-trigly_cat) |>
      as.matrix()
    test_y = test$trigly_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### glm   
    glm = train(trigly_cat ~ ., 
                data = train,
                method = "glm", 
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"), 
                metric = "ROC")
    
    pred_class = predict(glm, newdata = test)
    pred_probs = predict(glm, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}

res_moment_cat_trigly = models_cat_trigly(data_moment1024_cat2, sim = 10, "Moment_cat")
res_raw_moment_cat_trigly = models_cat_trigly(data_raw_moment1024_cat2, sim = 10, "Raw_Moment_cat")
res_gpt_cat_trigly = models_cat_trigly(data_gpt1536_entropy_cat2, sim = 10, "EntroGPT_cat")
res_bert_cat_trigly = models_cat_trigly(data_bert768_entropy_cat2, sim = 10, "EntroBert_cat")
res_cohere_cat_trigly = models_cat_trigly(data_cohere1024_entropy_cat2, sim = 10, "EntroCohere_cat")
res_demo_cat_trigly = models_cat_trigly(data_demo_cat2, sim = 10, "Demo_cat")
res_entropy_cat_trigly = entropy_model_cat_trigly(data_entropy_cat2, sim = 10, "Entropy_cat")

res_cat_trigly = rbind(res_moment_cat_trigly, res_raw_moment_cat_trigly, 
                       res_gpt_cat_trigly, res_bert_cat_trigly, res_cohere_cat_trigly, 
                       res_demo_cat_trigly, res_entropy_cat_trigly)
res_cat_trigly_avg = res_cat_trigly |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))


##### AUC metrics - glucose #####
models_cat_gluc = function(data, sim, name){
  res = list()
  data$gluc_cat = factor(data$gluc_cat, 
                         levels = c("0", "1"), 
                         labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_total_cat, chol_ldl_cat, chol_hdl_cat, 
                           trigly_cat, sbp_cat, dbp_cat, age_cat))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-gluc_cat) |> 
      as.matrix()
    train_y = train$gluc_cat
    
    test_x = test |>
      select(-gluc_cat) |>
      as.matrix()
    test_y = test$gluc_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  family = "binomial",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)), 
                  metric = "ROC")
    
    pred_class = predict(ridge, newdata = test)
    pred_probs = predict(ridge, newdata = test_x, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}
entropy_model_cat_gluc = function(data, sim, name){
  res = list()
  data$gluc_cat = factor(data$gluc_cat, 
                         levels = c("0", "1"), 
                         labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp, 
                           chol_total_cat, chol_ldl_cat, chol_hdl_cat, 
                           trigly_cat, sbp_cat, dbp_cat, age_cat))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-gluc_cat) |> 
      as.matrix()
    train_y = train$gluc_cat
    
    test_x = test |>
      select(-gluc_cat) |>
      as.matrix()
    test_y = test$gluc_cat
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE,
                        summaryFunction = twoClassSummary)
    
    ### glm   
    glm = train(gluc_cat ~ ., 
                data = train,
                method = "glm", 
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"), 
                metric = "ROC")
    
    pred_class = predict(glm, newdata = test)
    pred_probs = predict(glm, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}

res_moment_cat_gluc = models_cat_gluc(data_moment1024_cat2, sim = 10, "Moment_cat")
res_raw_moment_cat_gluc = models_cat_gluc(data_raw_moment1024_cat2, sim = 10, "Raw_Moment_cat")
res_gpt_cat_gluc = models_cat_gluc(data_gpt1536_entropy_cat2, sim = 10, "EntroGPT_cat")
res_bert_cat_gluc = models_cat_gluc(data_bert768_entropy_cat2, sim = 10, "EntroBert_cat")
res_cohere_cat_gluc = models_cat_gluc(data_cohere1024_entropy_cat2, sim = 10, "EntroCohere_cat")
res_demo_cat_gluc = models_cat_gluc(data_demo_cat2, sim = 10, "Demo_cat")
res_entropy_cat_gluc = entropy_model_cat_gluc(data_entropy_cat2, sim = 10, "Entropy_cat")

res_cat_gluc = rbind(res_moment_cat_gluc, res_raw_moment_cat_gluc, 
                     res_gpt_cat_gluc, res_bert_cat_gluc, res_cohere_cat_gluc, 
                     res_demo_cat_gluc, res_entropy_cat_gluc)
res_cat_gluc_avg = res_cat_gluc |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))


##### AUC metrics - BMI #####
models_cov_bmi = function(data, sim, name){
  res = list()
  data$bmi = factor(data$bmi, 
                    levels = c("0", "1"), 
                    labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-bmi) |> 
      as.matrix()
    train_y = train$bmi
    
    test_x = test |>
      select(-bmi) |>
      as.matrix()
    test_y = test$bmi
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)), 
                  metric = "ROC")
    
    pred_class = predict(ridge, newdata = test)
    pred_probs = predict(ridge, newdata = test_x, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}
entropy_model_cov_bmi = function(data, sim, name){
  res = list()
  data$bmi = factor(data$bmi, 
                    levels = c("0", "1"), 
                    labels = c("normal", "high")) 
  data = data |> select(-c(seqn, arthritis, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp)) 
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-bmi) |> 
      as.matrix()
    train_y = train$bmi
    
    test_x = test |>
      select(-bmi) |>
      as.matrix()
    test_y = test$bmi
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)
    
    ### glm   
    glm = train(bmi ~ ., 
                data = train,
                method = "glm",   
                family = "binomial",
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"), 
                metric = "ROC")
    
    pred_class = predict(glm, newdata = test)
    pred_probs = predict(glm, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$high)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}

res_moment_cov_bmi = models_cov_bmi(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_bmi = models_cov_bmi(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_bmi = models_cov_bmi(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_bmi = models_cov_bmi(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_bmi = models_cov_bmi(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_bmi = models_cov_bmi(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_bmi = entropy_model_cov_bmi(data_entropy_cov2, sim = 10, "Entropy_cov")

res_cov_bmi = rbind(res_moment_cov_bmi, res_raw_moment_cov_bmi, 
                    res_gpt_cov_bmi, res_bert_cov_bmi, res_cohere_cov_bmi, 
                    res_demo_cov_bmi, res_entropy_cov_bmi)
res_cov_bmi_avg = res_cov_bmi |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc),
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)


##### AUC metrics - Arthritis #####
models_cov_arthritis = function(data, sim, name){
  res = list()
  data$arthritis = factor(data$arthritis, 
                          levels = c("1", "0"), 
                          labels = c("yes", "no")) 
  data = data |> select(-c(seqn, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-arthritis) |> 
      as.matrix()
    train_y = train$arthritis
    
    test_x = test |>
      select(-arthritis) |>
      as.matrix()
    test_y = test$arthritis
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-3, 1, length = 50)), 
                  metric = "ROC")
    pred_class = predict(ridge, newdata = test_x)
    pred_probs = predict(ridge, newdata = test_x, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"] 
    auc_value = pROC::auc(test_y, pred_probs$yes)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc)
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc, 
                         auc = auc_vals)
  return(final_res)
}
entropy_model_cov_arthritis = function(data, sim, name){
  res = list()
  data$arthritis = factor(data$arthritis, 
                          levels = c("0", "1"), 
                          labels = c("yes", "no")) 
  data = data |> select(-c(seqn, malig, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-arthritis) |> 
      as.matrix()
    train_y = train$arthritis
    
    test_x = test |>
      select(-arthritis) |>
      as.matrix()
    test_y = test$arthritis
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)
    
    ### glm   
    glm = train(arthritis ~ ., 
                data = train,
                method = "glm",   
                family = "binomial",
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"), 
                metric = "ROC")
    
    pred_class = predict(glm, newdata = test)
    pred_probs = predict(glm, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"]
    auc_value = pROC::auc(test_y, pred_probs$yes)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc)
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc, 
                         auc = auc_vals)
  return(final_res)
}

res_moment_cov_arthritis = models_cov_arthritis(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_arthritis = models_cov_arthritis(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_arthritis = models_cov_arthritis(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_arthritis = models_cov_arthritis(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_arthritis = models_cov_arthritis(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_arthritis = models_cov_arthritis(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_arthritis = entropy_model_cov_arthritis(data_entropy_cov2, sim = 10, "Entropy_cov")

res_cov_arthritis = rbind(res_moment_cov_arthritis, res_raw_moment_cov_arthritis, 
                          res_gpt_cov_arthritis, res_bert_cov_arthritis, 
                          res_cohere_cov_arthritis, 
                          res_demo_cov_arthritis, res_entropy_cov_arthritis)
res_cov_arthritis_avg = res_cov_arthritis |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))


##### AUC metrics - Malignancy #####
models_cov_malig = function(data, sim, name){
  res = list()
  data$malig = factor(data$malig, 
                      levels = c("1", "0"), 
                      labels = c("yes", "no")) 
  data = data |> select(-c(seqn, arthritis, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))  
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8, strata = malig)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-malig) |> 
      as.matrix()
    train_y = train$malig
    
    test_x = test |>
      select(-malig) |>
      as.matrix()
    test_y = test$malig
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE,
                        classProbs = TRUE)
    
    ### logistic ridge (alpha = 0)
    ridge = train(x = train_x, y = train_y,
                  method = "glmnet",
                  family = "binomial",
                  trControl = ctrl, 
                  tuneGrid = expand.grid(alpha = 0, 
                                         lambda = 10^seq(-4, 1, length = 50)))
    pred_class = predict(ridge, newdata = test)
    pred_probs = predict(ridge, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"]
    auc_value = auc(test_y, pred_probs$yes)
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc)
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc, 
                         auc = auc_vals)
  return(final_res)
}
entropy_model_cov_malig = function(data, sim, name){
  res = list()
  data$malig = factor(data$malig, 
                      levels = c("1", "0"), 
                      labels = c("yes", "no")) 
  data = data |> select(-c(seqn, arthritis, breast, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))  
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8, strata = malig)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |>
      select(-malig) |> 
      as.matrix()
    train_y = train$malig
    
    test_x = test |>
      select(-malig) |>
      as.matrix()
    test_y = test$malig
    
    ctrl = trainControl(method = "cv", 
                        number = 10, 
                        savePredictions = TRUE, 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary)
    
    ### glm   
    glm = train(malig ~ ., 
                data = train,
                method = "glm",   
                family = "binomial",
                trControl = ctrl,
                preProcess = c("center", "scale", "zv"), 
                metric = "ROC")
    
    pred_class = predict(glm, newdata = test)
    pred_probs = predict(glm, newdata = test, type = "prob")
    
    cm = confusionMatrix(pred_class, test_y)
    acc = cm$overall["Accuracy"]
    auc_value = pROC::auc(test_y, pred_probs$yes)
    
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  test_acc = sapply(res, function(x) x$acc)
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc, 
                         auc = auc_vals)
  return(final_res)
}

res_moment_cov_malig = models_cov_malig(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_malig = models_cov_malig(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_malig = models_cov_malig(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_malig = models_cov_malig(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_malig = models_cov_malig(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_malig = models_cov_malig(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_malig = entropy_model_cov_malig(data_entropy_cov2, sim = 10, "Entropy_cov")

res_cov_malig = rbind(res_moment_cov_malig, res_raw_moment_cov_malig, 
                      res_gpt_cov_malig, res_bert_cov_malig,
                      res_cohere_cov_malig, 
                      res_demo_cov_malig, res_entropy_cov_malig)
res_cov_malig_avg = res_cov_malig |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))


##### AUC metrics - Breast cancer #####
firth_lasso_cov_breast = function(data, sim, name){
  res = list()
  data$breast = factor(data$breast, 
                       levels = c("1", "0"), 
                       labels = c("yes", "no")) 
  data = data |> select(-c(seqn, arthritis, malig, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))  
  
  
  covariates_indices = c(1:7, (ncol(data)-8):(ncol(data)-1))
  covariates_names = names(data)[covariates_indices]
  embeddings_names = setdiff(names(data), c("breast", covariates_names))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8, strata = "breast")
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train[, embeddings_names, drop = FALSE] |> as.matrix()
    train_y = train$breast
    
    
    ### lasso to embeddings (alpha = 1)
    cv_lasso = cv.glmnet(x = train_x, y = as.numeric(train_y == "yes"),
                         family = "binomial",
                         alpha = 1,
                         standardize = TRUE)
    
    lasso_coefs = coef(cv_lasso, s = cv_lasso$lambda.1se)
    
    selected_embeddings = rownames(lasso_coefs)[lasso_coefs[,1] != 0]
    selected_embeddings = selected_embeddings[selected_embeddings != "(Intercept)"]
    
    final_features = unique(c("breast", selected_embeddings, covariates_names))
    
    train_final = train[, final_features]
    test_final = test[, final_features]
    
    
    ### firth after lasso 
    firth = logistf(breast~., data = train_final, 
                    control = logistf.control(maxit = 1000, maxstep = 1),
                    pl = FALSE)
    
    pred_probs_yes = 1 - predict(firth, newdata = test_final, type = "response")
    pred_probs = data.frame(no = 1 - pred_probs_yes, 
                            yes = pred_probs_yes)
    roc_obj = pROC::roc(as.numeric(test$breast == "yes"), pred_probs_yes)
    auc_value = pROC::auc(roc_obj)
    
    threshold = coords(roc_obj, "best", best.method = "youden")$threshold
    pred_class = factor(ifelse(pred_probs_yes > threshold, "yes", "no"), 
                        levels = c("yes", "no"))
    test_breast = factor(test$breast, levels = c("yes", "no"))
    
    cm = confusionMatrix(pred_class, test_breast, positive = "yes")
    acc = cm$overall["Accuracy"] 
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}
firth_model_cov_breast = function(data, sim, name){
  res = list()
  data$breast = factor(data$breast, 
                       levels = c("1", "0"), 
                       labels = c("yes", "no")) 
  data = data |> select(-c(seqn, arthritis, malig, lung, ovary, skin, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))  
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8, strata = "breast")
    train = training(data_split)  
    test = testing(data_split) 
    
    ### firth correction   
    firth = logistf(breast ~ ., data = train, 
                    control = logistf.control(maxit = 1000, maxstep = 1),
                    pl = FALSE) 
    
    pred_probs_yes = 1 - predict(firth, newdata = test, type = "response")
    pred_probs = data.frame(no = 1 - pred_probs_yes, 
                            yes = pred_probs_yes)
    
    roc_obj = pROC::roc(as.numeric(test$breast == "yes"), pred_probs_yes)
    auc_value = pROC::auc(roc_obj)
    
    threshold = coords(roc_obj, "best", best.method = "youden")$threshold
    
    pred_class = factor(ifelse(pred_probs_yes > threshold, "yes", "no"), 
                        levels = c("yes", "no"))
    
    cm = confusionMatrix(pred_class, test$breast)
    acc = cm$overall["Accuracy"] 
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}

res_moment_cov_breast = firth_lasso_cov_breast(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_breast = firth_lasso_cov_breast(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_breast = firth_lasso_cov_breast(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_breast = firth_lasso_cov_breast(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_breast = firth_lasso_cov_breast(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_breast = firth_model_cov_breast(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_breast = firth_model_cov_breast(data_entropy_cov2, sim = 10, "Entropy_cov")

res_cov_breast = rbind(res_moment_cov_breast, res_raw_moment_cov_breast, 
                       res_gpt_cov_breast, res_bert_cov_breast, 
                       res_cohere_cov_breast, 
                       res_demo_cov_breast, res_entropy_cov_breast)
res_cov_breast_avg = res_cov_breast |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))


##### AUC metrics - Skin cancer #####
firth_lasso_cov_skin = function(data, sim, name){
  res = list()
  data$skin = factor(data$skin, 
                     levels = c("1", "0"), 
                     labels = c("yes", "no")) 
  data = data |> select(-c(seqn, arthritis, malig, lung, ovary, breast, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))  
  
  covariates_indices = c(1:7, (ncol(data)-8):(ncol(data)-1))
  covariates_names = names(data)[covariates_indices]
  embeddings_names = setdiff(names(data), c("skin", covariates_names))
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8, strata = "skin")
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train[, embeddings_names, drop = FALSE] |> as.matrix()
    train_y = train$skin
    
    
    ### lasso to embeddings (alpha = 1)
    cv_lasso = cv.glmnet(x = train_x, y = as.numeric(train_y == "yes"),
                         family = "binomial",
                         alpha = 1,
                         standardize = TRUE)
    
    lasso_coefs = coef(cv_lasso, s = cv_lasso$lambda.1se)
    
    selected_embeddings = rownames(lasso_coefs)[lasso_coefs[,1] != 0]
    selected_embeddings = selected_embeddings[selected_embeddings != "(Intercept)"]
    
    final_features = unique(c("skin", selected_embeddings, covariates_names))
    
    train_final = train[, final_features]
    test_final = test[, final_features]
    
    
    ### firth after lasso 
    firth = logistf(skin~., data = train_final, 
                    control = logistf.control(maxit = 1000, maxstep = 1),
                    pl = FALSE)
    
    pred_probs_yes = 1 - predict(firth, newdata = test_final, type = "response")
    pred_probs = data.frame(no = 1 - pred_probs_yes, 
                            yes = pred_probs_yes)
    roc_obj = pROC::roc(as.numeric(test$skin == "yes"), pred_probs_yes)
    auc_value = pROC::auc(roc_obj)
    
    threshold = coords(roc_obj, "best", best.method = "youden")$threshold
    pred_class = factor(ifelse(pred_probs_yes > threshold, "yes", "no"), 
                        levels = c("yes", "no"))
    test_skin = factor(test$skin, levels = c("yes", "no"))
    
    cm = confusionMatrix(pred_class, test_skin, positive = "yes")
    acc = cm$overall["Accuracy"] 
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}
firth_model_cov_skin = function(data, sim, name){
  res = list()
  data$skin = factor(data$skin, 
                     levels = c("1", "0"), 
                     labels = c("yes", "no")) 
  data = data |> select(-c(seqn, arthritis, malig, lung, ovary, breast, 
                           med_sbp, med_dbp, chol_total, chol_hdl, chol_ldl, 
                           triglyceride, glucose, crp))  
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8, strata = "skin")
    train = training(data_split)  
    test = testing(data_split) 
    
    ### firth correction   
    firth = logistf(skin ~ ., data = train, 
                    control = logistf.control(maxit = 1000, maxstep = 1),
                    pl = FALSE) 
    
    pred_probs_yes = 1 - predict(firth, newdata = test, type = "response")
    pred_probs = data.frame(no = 1 - pred_probs_yes, 
                            yes = pred_probs_yes)
    
    roc_obj = pROC::roc(as.numeric(test$skin == "yes"), pred_probs_yes)
    auc_value = pROC::auc(roc_obj)
    
    threshold = coords(roc_obj, "best", best.method = "youden")$threshold
    
    pred_class = factor(ifelse(pred_probs_yes > threshold, "yes", "no"), 
                        levels = c("yes", "no"))
    
    cm = confusionMatrix(pred_class, test$skin)
    acc = cm$overall["Accuracy"] 
    
    # output result 
    res[[i]] = list(acc = acc, auc = auc_value)
  }
  
  test_acc = sapply(res, function(x) x$acc) 
  auc_vals = sapply(res, function(x) x$auc)
  
  final_res = data.frame(method = rep(name, sim), 
                         test_acc = test_acc,  
                         auc = auc_vals)
  return(final_res)
}

res_moment_cov_skin = firth_lasso_cov_skin(data_moment1024_cov2, sim = 10, "Moment_cov")
res_raw_moment_cov_skin = firth_lasso_cov_skin(data_raw_moment1024_cov2, sim = 10, "Raw_Moment_cov")
res_gpt_cov_skin = firth_lasso_cov_skin(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT_cov")
res_bert_cov_skin = firth_lasso_cov_skin(data_bert768_entropy_cov2, sim = 10, "EntroBert_cov")
res_cohere_cov_skin = firth_lasso_cov_skin(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere_cov")
res_demo_cov_skin = firth_model_cov_skin(data_demo_cov2, sim = 10, "Demo_cov")
res_entropy_cov_skin = firth_model_cov_skin(data_entropy_cov2, sim = 10, "Entropy_cov")

res_cov_skin = rbind(res_moment_cov_skin, res_raw_moment_cov_skin, 
                     res_gpt_cov_skin, res_bert_cov_skin, res_cohere_cov_skin, 
                     res_demo_cov_skin, res_entropy_cov_skin)
res_cov_skin_avg = res_cov_skin |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc))



########################### box plot #########################
# Boxplots for RMSE 
all_cov_res_cont = 
  bind_rows(
    res_cov_chol |> mutate(response = "cholesterol"),
    res_cov_ldl |> mutate(response = "ldl"),
    res_cov_hdl |> mutate(response = "hdl"),
    res_cov_trigly |> mutate(response = "triglyceride"),
    res_cov_gluc |> mutate(response = "glucose"),
    .id = NULL) |>
  mutate(method = case_when(
    method == "Demo_cov" ~ "Demo", 
    method == "Entropy_cov" ~ "entropy", 
    method == "EntroCohere_cov" ~ "EntroCohere", 
    method == "EntroBert_cov" ~ "EntroBert", 
    method == "EntroGPT_cov" ~ "EntroGPT",  
    method == "Moment_cov" ~ "Moment", 
    method == "Raw_Moment_cov" ~ "Raw_Moment", 
    TRUE ~ method))

p_all_cov_rmse = ggplot(all_cov_res_cont, 
                        aes(x = method, y = test_rmse, fill = method)) +
  geom_boxplot(outlier.shape = 21, alpha = 0.8) +
  facet_wrap(~ response, scales = "free_y") +
  labs(title = "Test RMSE Distributions by Method for Each Response",
       x = "Method",
       y = "Test RMSE") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")


# Boxplots for AUC 
all_res_total = bind_rows( 
  res_cov_bmi |> mutate(response = "bmi"),
  res_cat_chol |> mutate(response = "cholesterol"),
  res_cat_ldl |> mutate(response = "ldl"),
  res_cat_hdl |> mutate(response = "hdl"),
  res_cat_trigly |> mutate(response = "triglyceride"),
  res_cat_gluc |> mutate(response = "glucose"),
  res_cov_arthritis |> mutate(response = "arthritis"),
  res_cov_malig |> mutate(response = "malignancy"),
  res_cov_breast |> mutate(response = "breast"),
  res_cov_skin |> mutate(response = "skin"),
  .id = NULL) |>
  mutate(method = case_when(
    method == "Entropy_cat" ~ "Entropy",
    method == "Demo_cat" ~ "Demo",
    method == "EntroGPT_cat" ~ "EntroGPT",
    method == "EntroBert_cat" ~ "EntroBert",
    method == "EntroCohere_cat" ~ "EntroCohere",
    method == "Moment_cat" ~ "Moment",
    method == "Raw_Moment_cat" ~ "Raw_Moment",
    TRUE ~ method)) |>
  mutate(method = case_when(
    method == "Entropy_cov" ~ "Entropy",
    method == "Demo_cov" ~ "Demo",
    method == "EntroGPT_cov" ~ "EntroGPT",
    method == "EntroBert_cov" ~ "EntroBert",
    method == "EntroCohere_cov" ~ "EntroCohere",
    method == "Moment_cov" ~ "Moment",
    method == "Raw_Moment_cov" ~ "Raw_Moment",
    TRUE ~ method))

all_res_total_out = all_res_total |> 
  filter(response %in% c("malignancy", "breast", "skin", "arthritis"))

all_res_total_cov = all_res_total |> 
  filter(!response %in% c("malignancy", "breast", "skin", "arthritis"))

p_all_res_total_out = ggplot(all_res_total_out, 
                             aes(x = method, y = auc, fill = method)) +
  geom_boxplot(outlier.shape = 21, alpha = 0.8) +
  facet_wrap(~ response, scales = "free_y") +
  labs(title = "AUC Distributions by Method for Each Response",
       x = "Method",
       y = "AUC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")

p_all_res_total_cov = ggplot(all_res_total_cov,
                             aes(x = method, y = auc, fill = method)) +
  geom_boxplot(outlier.shape = 21, alpha = 0.8) +
  facet_wrap(~ response, scales = "free_y") +
  labs(title = "AUC Distributions by Method for Each Response",
       x = "Method",
       y = "AUC") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        legend.position = "none")