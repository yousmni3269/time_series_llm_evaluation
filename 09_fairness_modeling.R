library(tidyverse)
library(tidymodels)
library(caret)
library(glmnet)
library(pROC)
library(logistf)

########################### fairness test modeling ########################### 

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

min20_moment_cat_chol = models_cat_chol(data_min20_moment1024_cat2, sim = 10, "min20_Moment_cat")
trunc_gpt_cat_chol = models_cat_chol(trunc_gpt1536_entropy_cat2, sim = 10, "trunc_EntroGPT_cat")

res_cat_chol = rbind(res_moment_cat_chol, res_raw_moment_cat_chol, 
                     res_gpt_cat_chol, res_bert_cat_chol, res_cohere_cat_chol,
                     res_demo_cat_chol, res_entropy_cat_chol, 
                     min20_moment_cat_chol, trunc_gpt_cat_chol)
res_cat_chol_avg = res_cat_chol |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)

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

min20_moment_cat_ldl = models_cat_ldl(data_min20_moment1024_cat2, sim = 10, "min20_Moment_cat")
trunc_gpt_cat_ldl = models_cat_ldl(trunc_gpt1536_entropy_cat2, sim = 10, "trunc_EntroGPT_cat")

res_cat_ldl = rbind(res_moment_cat_ldl, res_raw_moment_cat_ldl, 
                    res_gpt_cat_ldl, res_bert_cat_ldl, res_cohere_cat_ldl, 
                    res_demo_cat_ldl, res_entropy_cat_ldl, 
                    min20_moment_cat_ldl, trunc_gpt_cat_ldl) 
res_cat_ldl_avg = res_cat_ldl |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc),
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)

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

min20_moment_cat_hdl = models_cat_hdl(data_min20_moment1024_cat2, sim = 10, "min20_Moment_cat")
trunc_gpt_cat_hdl = models_cat_hdl(trunc_gpt1536_entropy_cat2, sim = 10, "trunc_EntroGPT_cat")

res_cat_hdl = rbind(res_moment_cat_hdl, res_raw_moment_cat_hdl, 
                    res_gpt_cat_hdl, res_bert_cat_hdl, res_cohere_cat_hdl,
                    res_demo_cat_hdl, res_entropy_cat_hdl, 
                    min20_moment_cat_hdl, trunc_gpt_cat_hdl) 
res_cat_hdl_avg = res_cat_hdl |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc),
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)

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

min20_moment_cat_trigly = models_cat_trigly(data_min20_moment1024_cat2, sim = 10, "min20_Moment_cat")
trunc_gpt_cat_trigly = models_cat_trigly(trunc_gpt1536_entropy_cat2, sim = 10, "trunc_EntroGPT_cat")

res_cat_trigly = rbind(res_moment_cat_trigly, res_raw_moment_cat_trigly, 
                       res_gpt_cat_trigly, res_bert_cat_trigly, res_cohere_cat_trigly, 
                       res_demo_cat_trigly, res_entropy_cat_trigly, 
                       min20_moment_cat_trigly, trunc_gpt_cat_trigly)
res_cat_trigly_avg = res_cat_trigly |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)

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

min20_moment_cat_gluc = models_cat_gluc(data_min20_moment1024_cat2, sim = 10, "min20_Moment_cat")
trunc_gpt_cat_gluc = models_cat_gluc(trunc_gpt1536_entropy_cat2, sim = 10, "trunc_EntroGPT_cat")

res_cat_gluc = rbind(res_moment_cat_gluc, res_raw_moment_cat_gluc, 
                     res_gpt_cat_gluc, res_bert_cat_gluc, res_cohere_cat_gluc, 
                     res_demo_cat_gluc, res_entropy_cat_gluc, 
                     min20_moment_cat_gluc, trunc_gpt_cat_gluc)
res_cat_gluc_avg = res_cat_gluc |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)


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

min20_moment_cov_bmi = models_cov_bmi(data_min20_moment1024_cov2, sim = 10, "min20_Moment_cov")
trunc_gpt_cov_bmi = models_cov_bmi(trunc_gpt1536_entropy_cov2, sim = 10, "trunc_EntroGPT_cov")

res_cov_bmi = rbind(res_moment_cov_bmi, res_raw_moment_cov_bmi, 
                    res_gpt_cov_bmi, res_bert_cov_bmi, res_cohere_cov_bmi, 
                    res_demo_cov_bmi, res_entropy_cov_bmi, 
                    min20_moment_cov_bmi, trunc_gpt_cov_bmi)
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

min20_moment_cov_arthritis = models_cov_arthritis(data_min20_moment1024_cov2, sim = 10, "min20_Moment_cov")
trunc_gpt_cov_arthritis = models_cov_arthritis(trunc_gpt1536_entropy_cov2, sim = 10, "trunc_EntroGPT_cov")

res_cov_arthritis = rbind(res_moment_cov_arthritis, res_raw_moment_cov_arthritis, 
                          res_gpt_cov_arthritis, res_bert_cov_arthritis, 
                          res_cohere_cov_arthritis, 
                          res_demo_cov_arthritis, res_entropy_cov_arthritis, 
                          min20_moment_cov_arthritis, trunc_gpt_cov_arthritis)
res_cov_arthritis_avg = res_cov_arthritis |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)


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

min20_moment_cov_malig = models_cov_malig(data_min20_moment1024_cov2, sim = 10, "min20_Moment_cov") 
trunc_gpt_cov_malig = models_cov_malig(trunc_gpt1536_entropy_cov2, sim = 10, "trunc_EntroGPT_cov")

res_cov_malig = rbind(res_moment_cov_malig, res_raw_moment_cov_malig, 
                      res_gpt_cov_malig, res_bert_cov_malig,
                      res_cohere_cov_malig, 
                      res_demo_cov_malig, res_entropy_cov_malig, 
                      min20_moment_cov_malig, trunc_gpt_cov_malig)
res_cov_malig_avg = res_cov_malig |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)


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

min20_moment_cov_breast = firth_lasso_cov_breast(data_min20_moment1024_cov2, sim = 10, "min20_Moment_cov")
trunc_gpt_cov_breast = firth_lasso_cov_breast(trunc_gpt1536_entropy_cov2, sim = 10, "trunc_EntroGPT_cov")

res_cov_breast = rbind(res_moment_cov_breast, res_raw_moment_cov_breast, 
                       res_gpt_cov_breast, res_bert_cov_breast, 
                       res_cohere_cov_breast, 
                       res_demo_cov_breast, res_entropy_cov_breast, 
                       min20_moment_cov_breast, trunc_gpt_cov_breast)
res_cov_breast_avg = res_cov_breast |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)


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

min20_moment_cov_skin = firth_lasso_cov_skin(data_min20_moment1024_cov2, sim = 10, "min20_Moment_cov")
trunc_gpt_cov_skin = firth_lasso_cov_skin(trunc_gpt1536_entropy_cov2, sim = 10, "trunc_EntroGPT_cov")

res_cov_skin = rbind(res_moment_cov_skin, res_raw_moment_cov_skin, 
                     res_gpt_cov_skin, res_bert_cov_skin, res_cohere_cov_skin, 
                     res_demo_cov_skin, res_entropy_cov_skin, 
                     min20_moment_cov_skin, trunc_gpt_cov_skin)
res_cov_skin_avg = res_cov_skin |>
  group_by(method) |>
  summarize(
    avg_acc = mean(test_acc), 
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(method)


all_results = rbind(
  res_cat_chol_avg |> mutate(outcome = "chol"), 
  res_cat_ldl_avg |> mutate(outcome = "ldl"),
  res_cat_hdl_avg |> mutate(outcome = "hdl"),
  res_cat_trigly_avg |> mutate(outcome = "trigly"),
  res_cat_gluc_avg |> mutate(outcome = "gluc"),
  res_cov_bmi_avg |> mutate(outcome = "bmi"),
  res_cov_arthritis_avg |> mutate(outcome = "arthritis"),
  res_cov_malig_avg |> mutate(outcome = "malig"),
  res_cov_breast_avg |> mutate(outcome = "breast"),
  res_cov_skin_avg |> mutate(outcome = "skin")) |>
    select(-avg_acc) |> 
    mutate(method = str_replace(method, "_cat$", "_cov")) |>   
    pivot_wider(names_from = outcome, values_from = avg_auc)
  