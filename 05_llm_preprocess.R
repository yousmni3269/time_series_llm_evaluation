# load clean data and covariates 
load("002_data/data_wide.RData")
load("002_data/clinical_cov.RData")


# data: no bmi + clinical covariates 
data_wide_cov_no_bmi = data_wide |> select(-bmi)
  left_join(clinical_cov, by = "seqn") |>
  drop_na()
write.csv(data_wide_cov_no_bmi, file = "003_llm_data/data_wide_cov_no_bmi.csv") 

# data: no bmi + no clinical covariates 
data_wide_no_cov_no_bmi = data_wide |> select(-bmi)
  left_join(clinical_cov, by = "seqn") |>
  drop_na()
write.csv(data_wide_cov, file = "003_llm_data/data_wide_no_cov_no_bmi.csv") 
