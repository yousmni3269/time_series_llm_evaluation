# Sensitivity analysis 

# compare AUC for BMI response across 2909 participants dataset with the full 6943 participants dataset

models_bmi_full = function(data, sim, name){
  res = list()
  data$bmi = factor(data$bmi, 
                    levels = c("0", "1"), 
                    labels = c("normal", "high")) 
  data = data |> select(-seqn)
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |> select(-bmi) |> as.matrix()
    train_y = train$bmi
    
    test_x = test |> select(-bmi) |> as.matrix()
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
entropy_model_bmi_full = function(data, sim, name){
  res = list()
  data$bmi = factor(data$bmi, 
                    levels = c("0", "1"), 
                    labels = c("normal", "high")) 
  data = data |> select(-seqn)
  
  for(i in 1:sim){
    set.seed(i)
    
    data_split = initial_split(data, prop = 0.8)
    train = training(data_split)  
    test = testing(data_split) 
    
    train_x = train |> select(-bmi) |> as.matrix()
    train_y = train$bmi
    
    test_x = test |> select(-bmi) |> as.matrix()
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

res_moment_bmi_full = models_bmi_full(data_moment1024, sim = 10, "Moment")
res_raw_moment_bmi_full = models_bmi_full(data_raw_moment1024, sim = 10, "Raw_Moment")
res_gpt_bmi_full = models_bmi_full(data_gpt1536_entropy, sim = 10, "EntroGPT") 
res_bert_bmi_full = models_bmi_full(data_bert768_entropy, sim = 10, "EntroBert")
res_cohere_bmi_full = models_bmi_full(data_cohere1024_entropy, sim = 10, "EntroCohere")
res_demo_bmi_full = entropy_model_bmi_full(data_demo, sim = 10, "Demo")
res_entropy_bmi_full = entropy_model_bmi_full(data_entropy, sim = 10, "Entropy")

res_bmi_full = rbind(res_moment_bmi_full, res_raw_moment_bmi_full, 
                     res_gpt_bmi_full, res_bert_bmi_full, res_cohere_bmi_full, 
                     res_demo_bmi_full, res_entropy_bmi_full)

res_bmi_full_avg = res_bmi_full |>
  group_by(method) |>
  summarize(
    avg_auc = mean(auc),
    .groups = 'drop') |>
  arrange(desc(avg_auc)) |> 
  rename(avg_auc_full = avg_auc) 

left_join(res_bmi_full_avg, res_cov_bmi_avg, by = "method") |>
  arrange(method) |>
  mutate(diff = avg_auc_full - avg_auc)




```