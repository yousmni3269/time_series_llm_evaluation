# Evaluate LLM predictions 
library(tidyverse)
library(jsonlite)
library(openxlsx)
library(readr)
library(ggplot2)
library(tidymodels)
library(caret)
library(glmnet)
library(pROC)

########################### data import #########################

# gpt4o-mini 
data_gpt4o_mini = fromJSON("003_llm_data/llm_bmi_gpt4o_mini.json")
data_gpt4o_mini = as.data.frame(data_gpt4o_mini) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt4o_mini, "003_llm_data/llm_bmi_gpt4o_mini.csv")

## gpt4o-mini with baseline  
data_gpt4o_mini_baseline = fromJSON("003_llm_data/llm_bmi_gpt4o_mini_baseline.json")
data_gpt4o_mini_baseline = as.data.frame(data_gpt4o_mini_baseline) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt4o_mini_baseline, "003_llm_data/llm_bmi_gpt4o_mini_baseline.csv")

## gpt4o-mini with clinical covariates 
data_gpt4o_mini_cov = fromJSON("003_llm_data/llm_bmi_gpt4o_mini_cov.json") 
data_gpt4o_mini_cov = as.data.frame(data_gpt4o_mini_cov) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt4o_mini_cov, "003_llm_data/llm_bmi_gpt4o_mini_cov.csv")

## gpt4o-mini with clinical covariates with baseline 
data_gpt4o_mini_cov_baseline = fromJSON("006_llm_data/llm_bmi_gpt4o_mini_cov_baseline.json") 
data_gpt4o_mini_cov_baseline = as.data.frame(data_gpt4o_mini_cov_baseline) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont) 
write.csv(data_gpt4o_mini_cov_baseline, "006_llm_data/llm_bmi_gpt4o_mini_cov_baseline.csv")


## gpt5  
data_gpt5 = fromJSON("003_llm_data/gpt5_nobaseline.json")
data_gpt5 = as.data.frame(data_gpt5) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt5, "003_llm_data/llm_bmi_gpt5.csv")

## gpt5 with baseline 
data_gpt5_baseline = fromJSON("003_llm_data/gpt5_baseline.json")
data_gpt5_baseline = as.data.frame(data_gpt5_baseline) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt5_baseline, "003_llm_data/llm_bmi_gpt5_baseline.csv")

## gpt5 with clinical covariates 
data_gpt5_cov = fromJSON("003_llm_data/llm_bmi_gpt5_cov.json") 
data_gpt5_cov = as.data.frame(data_gpt5_cov) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt5_cov, "003_llm_data/llm_bmi_gpt5_cov.csv")

## gpt5 with clinical covariates with baseline 
data_gpt5_cov_baseline = fromJSON("003_llm_data/llm_bmi_gpt5_cov_baseline.json") 
data_gpt5_cov_baseline = as.data.frame(data_gpt5_cov_baseline) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt5_cov_baseline, "003_llm_data/llm_bmi_gpt5_cov_baseline.csv")


## gpt4o 
data_gpt4o = fromJSON("003_llm_data/gpt4o_nobaseline.json") 
data_gpt4o = as.data.frame(data_gpt4o) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt4o, "003_llm_data/llm_bmi_gpt4o.csv")

## gpt4o with baseline 
data_gpt4o_baseline = fromJSON("003_llm_data/gpt4o_baseline.json") 
data_gpt4o_baseline = as.data.frame(data_gpt4o_baseline) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt4o_baseline, "003_llm_data/llm_bmi_gpt4o_baseline.csv")

## gpt4o with clinical covariates 
data_gpt4o_cov = fromJSON("003_llm_data/llm_bmi_gpt4o_cov.json") 
data_gpt4o_cov = as.data.frame(data_gpt4o_cov) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt4o_cov, "003_llm_data/llm_bmi_gpt4o_cov.csv")

## gpt4o with clinical covariates with baseline 
data_gpt4o_cov_baseline = fromJSON("003_llm_data/llm_bmi_gpt4o_cov_baseline.json") 
data_gpt4o_cov_baseline = as.data.frame(data_gpt4o_cov_baseline) |> 
  janitor::clean_names() |> 
  rename(pred_bmi_cont = bmi) |> 
  rename(pred_bmi = overweight_status) |>
  left_join(data_demo_bmi, by ="seqn") |>
  select(seqn, pred_bmi, pred_bmi_cont, bmi, bmi_cont)
write.csv(data_gpt4o_cov_baseline, "003_llm_data/llm_bmi_gpt4o_cov_baseline.csv")


########################### modeling #########################

models_bmi = function(data, sim, name){
  res = list()
  data$bmi = factor(data$bmi, 
                    levels = c("0", "1"), 
                    labels = c("normal", "high")) 
  data = data |> select(-seqn, arthritis, malig, breast, lung, ovary, skin)
  
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
entropy_model_bmi = function(data, sim, name){
  res = list()
  data$bmi = factor(data$bmi, 
                    levels = c("0", "1"), 
                    labels = c("normal", "high")) 
  data = data |> select(-seqn, arthritis, malig, breast, lung, ovary, skin)
  
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

res_moment_bmi = models_bmi(data_moment1024_cov2, sim = 10, "Moment")
res_raw_moment_bmi = models_bmi(data_raw_moment1024_cov2, sim = 10, "Raw_Moment")
res_gpt_bmi = models_bmi(data_gpt1536_entropy_cov2, sim = 10, "EntroGPT") 
res_bert_bmi = models_bmi(data_bert768_entropy_cov2, sim = 10, "EntroBert")
res_cohere_bmi = models_bmi(data_cohere1024_entropy_cov2, sim = 10, "EntroCohere")
res_demo_bmi = models_bmi(data_demo_cov2, sim = 10, "Demo")
res_entropy_bmi = entropy_model_bmi(data_entropy_cov2, sim = 10, "Entropy")

res_bmi = rbind(res_moment_bmi, res_raw_moment_bmi, 
                res_gpt_bmi, res_bert_bmi, res_cohere_bmi, 
                res_demo_bmi, res_entropy_bmi)
res_bmi_avg = res_bmi |>
  group_by(method) |>
  summarize(
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc)) |>
  mutate(method = case_when(
    method == "Entropy" ~ "Entropy_cov", 
    method == "Demo" ~ "Demo_cov", 
    method == "EntroGPT" ~ "EntroGPT_cov", 
    method == "EntroBert" ~ "EntroBert_cov", 
    method == "EntroCohere" ~ "EntroCohere_cov", 
    method == "Moment" ~ "Moment_cov", 
    method == "Raw_Moment" ~ "Raw_Moment_cov", 
    TRUE ~ method))

# with covariates (only as responses)  
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
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc)) |>
  mutate(method = case_when(
    method == "EntroGPT_cov" ~ "EntroGPT", 
    method == "Entropy_cov" ~ "Entropy", 
    method == "EntroBert_cov" ~ "EntroBert", 
    method == "EntroCohere_cov" ~ "EntroCohere", 
    method == "Demo_cov" ~ "Demo", 
    method == "Moment_cov" ~ "Moment", 
    method == "Raw_Moment_cov" ~ "Raw_Moment", 
    TRUE ~ method))

res_bmi_all = rbind(res_cov_bmi_avg, res_bmi_avg) 


## gpt4o-mini 
gpt4o_mini_roc = pROC::roc(response = data_gpt4o_mini$bmi,
                           predictor = data_gpt4o_mini$pred_bmi_cont)
gpt4o_mini_auc = pROC::auc(gpt4o_mini_roc)
gpt4o_mini_auc

## gpt4o-mini with baseline 
gpt4o_mini_baseline_roc = pROC::roc(response = data_gpt4o_mini_baseline$bmi,
                                    predictor = data_gpt4o_mini_baseline$pred_bmi_cont)
gpt4o_mini_baseline_auc = pROC::auc(gpt4o_mini_baseline_roc)
gpt4o_mini_baseline_auc

## gpt4o-mini with clinical covariates 
gpt4o_mini_cov_roc = pROC::roc(response = data_gpt4o_mini_cov$bmi,
                               predictor = data_gpt4o_mini_cov$pred_bmi_cont)
gpt4o_mini_cov_auc = pROC::auc(gpt4o_mini_cov_roc)
gpt4o_mini_cov_auc

## gpt4o-mini with clinical covariates with baseline 
gpt4o_mini_cov_baseline_roc = pROC::roc(response = data_gpt4o_mini_cov_baseline$bmi,
                                        predictor = data_gpt4o_mini_cov_baseline$pred_bmi_cont)
gpt4o_mini_cov_baseline_auc = pROC::auc(gpt4o_mini_cov_baseline_roc)
gpt4o_mini_cov_baseline_auc


## gpt5 
gpt5_roc = pROC::roc(response = data_gpt5$bmi,
                     predictor = data_gpt5$pred_bmi_cont)
gpt5_auc = pROC::auc(gpt5_roc)
gpt5_auc

## gpt5 with baseline 
gpt5_baseline_roc = pROC::roc(response = data_gpt5_baseline$bmi,
                              predictor = data_gpt5_baseline$pred_bmi_cont)
gpt5_baseline_auc = pROC::auc(gpt5_baseline_roc)
gpt5_baseline_auc

## gpt5 with clinical covariates 
gpt5_cov_roc = pROC::roc(response = data_gpt5_cov$bmi,
                         predictor = data_gpt5_cov$pred_bmi_cont)
gpt5_cov_auc = pROC::auc(gpt5_cov_roc)
gpt5_cov_auc

## gpt5 with clinical covariates with baseline 
gpt5_cov_baseline_roc = pROC::roc(response = data_gpt5_cov_baseline$bmi,
                                  predictor = data_gpt5_cov_baseline$pred_bmi_cont)
gpt5_cov_baseline_auc = pROC::auc(gpt5_cov_baseline_roc)
gpt5_cov_baseline_auc


## gpt4o 
gpt4o_roc = pROC::roc(response = data_gpt4o$bmi,
                      predictor = data_gpt4o$pred_bmi_cont)
gpt4o_auc = pROC::auc(gpt4o_roc)
gpt4o_auc

## gpt4o with baseline
gpt4o_baseline_roc = pROC::roc(response = data_gpt4o_baseline$bmi,
                               predictor = data_gpt4o_baseline$pred_bmi_cont)
gpt4o_baseline_auc = pROC::auc(gpt4o_baseline_roc)
gpt4o_baseline_auc

## gpt4o with clinical covariates 
gpt4o_cov_roc = pROC::roc(response = data_gpt4o_cov$bmi,
                          predictor = data_gpt4o_cov$pred_bmi_cont)
gpt4o_cov_auc = pROC::auc(gpt4o_cov_roc)
gpt4o_cov_auc

## gpt4o with clinical covariates with baseline 
gpt4o_cov_baseline_roc = pROC::roc(response = data_gpt4o_cov_baseline$bmi,
                                   predictor = data_gpt4o_cov_baseline$pred_bmi_cont)
gpt4o_cov_baseline_auc = pROC::auc(gpt4o_cov_baseline_roc)
gpt4o_cov_baseline_auc 


########################### summary #########################
auc_merge_llm = tibble(
  method = c("gpt4o_mini", "gpt4o_mini_baseline", 
             "gpt4o_mini_cov", "gpt4o_mini_cov_baseline", 
             "gpt5", "gpt5_baseline", "gpt5_cov", "gpt5_cov_baseline", 
             "gpt4o", "gpt4o_baseline", "gpt4o_cov", "gpt4o_cov_baseline"),
  avg_auc = c(gpt4o_mini_auc, gpt4o_mini_baseline_auc, 
              gpt4o_mini_cov_auc, gpt4o_mini_cov_baseline_auc, 
              gpt5_auc, gpt5_baseline_auc, gpt5_cov_auc, gpt5_cov_baseline_auc, 
              gpt4o_auc, gpt4o_baseline_auc, gpt4o_cov_auc, gpt4o_cov_baseline_auc)) 
auc_merge_all = rbind(res_bmi_all, auc_merge_llm) 
auc_merge_all_cov = auc_merge_all |> 
  filter(str_detect(method, 'cov')) |>
  arrange(method) |> 
  knitr::kable(digits = 3)

auc_merge_all_no_cov = auc_merge_all |> 
  filter(!str_detect(method, 'cov')) |>
  arrange(method) |> 
  knitr::kable(digits = 3)

