# Data Preprocessing 
# Data source: https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&

library(tidyverse)
library(foreign)
library(haven)


# Import 2003-2004 NHANES data 

bmx_d1 = read_xpt("001_raw_data/BMX_C.xpt") |>
  janitor::clean_names() |>
  drop_na(bmxbmi) |>
  select(seqn, bmxbmi)

demo_d1 = read_xpt("001_raw_data/DEMO_C.xpt") |>
  janitor::clean_names() |>
  filter(ridageyr >= 20 & ridageyr < 85 & dmdmartl != 77 & dmdmartl != 99 & ridexprg %in% c(2,3,NA) & dmdeduc2 != 7 & dmdeduc2 != 9) |>
  select(seqn, riagendr, ridageyr, ridreth1, dmdeduc2, dmdmartl, indfmpir) |>
  drop_na()

paxraw_d1 = read_xpt("001_raw_data/paxraw_c.xpt") |>
  janitor::clean_names() |>
  filter(paxstat == 1 & paxcal == 1) |>  
  drop_na(paxinten) |>
  select(seqn, paxn, paxinten)


# Import 2005-2006 NHANES data 

bmx_d2 = read_xpt("001_raw_data/BMX_D.xpt") |>
  janitor::clean_names() |>
  drop_na(bmxbmi) |>
  select(seqn, bmxbmi)

demo_d2 = read_xpt("001_raw_data/DEMO_D.xpt") |>
  janitor::clean_names() |>
  filter(ridageyr >= 20 & ridageyr < 85 & dmdmartl != 77 & dmdmartl != 99 & ridexprg %in% c(2,3,NA) & dmdeduc2 != 7 & dmdeduc2 != 9) |>
  select(seqn, riagendr, ridageyr, ridreth1, dmdeduc2, dmdmartl, indfmpir) |>
  drop_na()

paxraw_d2 = read_xpt("001_raw_data/paxraw_d.xpt", col_select = c("SEQN", "PAXN", "PAXINTEN", "PAXSTAT", "PAXCAL")) |> 
  # only select required columns due to memory limit issue
  janitor::clean_names() |>
  filter(paxstat == 1 & paxcal == 1) |>  
  drop_na(paxinten) |>
  select(seqn, paxn, paxinten)



# Remove missing values 
# Aggregate the physical activity data over 5-minute intervals for each subject

missing1 = paxraw_d1 |>
  group_by(seqn) |> 
  summarize(n = n()) |>
  filter(n != 24*60*7)

data1 = paxraw_d1 |>
  mutate(min5 = floor((paxn - 1) / 5) + 1) |> # min5 interval added ranging from 1 to 2016
  group_by(seqn, min5) |> 
  summarise(intensity = mean(paxinten)) |> 
  anti_join(missing1, by = "seqn") |>
  inner_join(bmx_d1, by = "seqn") |>
  inner_join(demo_d1, by = "seqn") # 3486 subjects 


missing2 = paxraw_d2 |>
  group_by(seqn) |> 
  summarize(n = n()) |>
  filter(n != 24*60*7)

data2 = paxraw_d2 |>
  mutate(min5 = floor((paxn - 1) / 5) + 1) |> # min5 interval added ranging from 1 to 2016
  group_by(seqn, min5) |> 
  summarise(intensity = mean(paxinten)) |> 
  anti_join(missing2, by = "seqn") |>
  inner_join(bmx_d2, by="seqn") |>
  inner_join(demo_d2, by="seqn") # 3457 subjects

data2 |>
  group_by(seqn) |>
  summarize()

data_total = rbind(data1,data2) |>
  rename(time = min5, 
         gender = riagendr, 
         age = ridageyr, 
         race = ridreth1, 
         education = dmdeduc2, 
         marital_status = dmdmartl, 
         pir = indfmpir) |>
  dplyr::select(seqn, bmxbmi, time, gender, age, race, education, marital_status, pir, intensity) # 6943 subjects 


##### Variables #####  
# BMI = 0: < 25, 1: >= 25 
# gender = 1: male, 2: female 
# race = 1: Mexican American, 2: Other Hispanic, 3: Non-Hispanic White, 4: Non-Hispanic Black, 5: Other
# education = 1: less than 9th grade, 2: 9-11th grade (includes 12th grade and no diploma), 3: high school grad/GED or equivalent, 4: some college or associates (AA) degree, 5: college graduate or higher 
# marital_status = 1: married, 2: widowed, 3: divorced, 4: separated, 5: never married, 6: living with partner 
# PA (physical activity) = 0: intensity < 100 counts/min (sedentary), 1: 100 <= intensity < 760 counts/min (light), 2: 760 <= intensity < 2200 counts/min (lifestyle), 3: 2200 <= intensity < 6000 counts/min (moderate), 4: intensity >= 6000 counts/min (vigorous)
  

# restructure 
data_wide = data_total |> 
  mutate(
    bmi = ifelse(bmxbmi >= 25, 1, 0),
    pa = ifelse(intensity < 100, 0,
                ifelse(intensity < 760, 1,
                       ifelse(intensity < 2200, 2,
                              ifelse(intensity < 6000, 3, 4))))) |>
  select(-bmxbmi, -intensity) |>
  pivot_wider(
    names_from = time, 
    values_from = pa, 
    names_prefix = "time")
save(data_wide, file = "002_data/data_wide.RData")
write.csv(data_wide, file = "002_data/data_wide.csv") 


# Clean the physical activity data over 1-minute intervals for each subject
# This is for raw_MOMENT  
data1_raw = paxraw_d1 |> 
  mutate(time = paste0("x", paxn)) |>  
  select(seqn, time, paxinten) |>
  pivot_wider(names_from = time, values_from = paxinten) |>
  anti_join(missing1, by = "seqn") |>
  inner_join(bmx_d1, by = "seqn") |>
  inner_join(demo_d1, by = "seqn") # 3486 subjects

data2_raw = paxraw_d2 |> 
  mutate(time = paste0("x", paxn)) |>  
  select(seqn, time, paxinten) |>
  pivot_wider(names_from = time, values_from = paxinten) |>
  anti_join(missing2, by = "seqn") |>
  inner_join(bmx_d2, by="seqn") |>
  inner_join(demo_d2, by="seqn") # 3457 subjects

data_raw_full = bind_rows(data1_raw, data2_raw) |>
  rename(gender = riagendr, 
         age = ridageyr, 
         race = ridreth1, 
         education = dmdeduc2, 
         marital_status = dmdmartl, 
         pir = indfmpir) |>
  mutate(bmi = ifelse(bmxbmi >= 25, 1, 0)) |>
  select(-bmxbmi) |>
  select(seqn, gender, age, race, education, marital_status, pir, bmi, everything()) # 6943 subjects 

save(data_raw_full, file = "002_data/data_raw_full.RData")
write.csv(data_raw_full, file = "002_data/data_raw_full.csv")  


# Aggregate the physical activity data over 20-minute intervals for each subject
# This is for min20_MOMENT  
data1_min20 = paxraw_d1 |>
  mutate(min20 = floor((paxn - 1) / 20) + 1) |> # min20 interval added ranging from 1 to 504
  group_by(seqn, min20) |> 
  summarise(intensity = mean(paxinten)) |> 
  anti_join(missing1, by = "seqn") |>
  inner_join(bmx_d1, by = "seqn") |>
  inner_join(demo_d1, by = "seqn") # 3486 subjects 

data2_min20 = paxraw_d2 |>
  mutate(min20 = floor((paxn - 1) / 20) + 1) |> # min20 interval added ranging from 1 to 504
  group_by(seqn, min20) |> 
  summarise(intensity = mean(paxinten)) |> 
  anti_join(missing2, by = "seqn") |>
  inner_join(bmx_d2, by="seqn") |>
  inner_join(demo_d2, by="seqn") # 3457 subjects

data_total_min20 = rbind(data1_min20, data2_min20) |>
  rename(time = min20, 
         gender = riagendr, 
         age = ridageyr, 
         race = ridreth1, 
         education = dmdeduc2, 
         marital_status = dmdmartl, 
         pir = indfmpir) |>
  dplyr::select(seqn, bmxbmi, time, gender, age, race, education, marital_status, pir, intensity) # 6943 subjects 

# restructure 
data_wide_min20 = data_total_min20 |> 
  mutate(
    bmi = ifelse(bmxbmi >= 25, 1, 0),
    pa = ifelse(intensity < 100, 0,
                ifelse(intensity < 760, 1,
                       ifelse(intensity < 2200, 2,
                              ifelse(intensity < 6000, 3, 4))))) |>
  select(-bmxbmi, -intensity) |>
  pivot_wider(
    names_from = time, 
    values_from = pa, 
    names_prefix = "time")
save(data_wide_min20, file = "002_data/data_wide_min20.RData")
write.csv(data_wide_min20, file = "002_data/data_wide_min20.csv") 




##### Add more variables #####  

# NAHNES 2003-2004 clinical variables 

bp_d1 = read_xpt("001_raw_data/BPX_C.xpt") |>
  janitor::clean_names() |>
  rowwise() |>
  mutate(
    med_sbp = median(c(bpxsy1, bpxsy2, bpxsy3, bpxsy4), na.rm = TRUE),
    med_dbp = median(c(bpxdi1, bpxdi2, bpxdi3, bpxdi4), na.rm = TRUE)
  ) |>
  select(seqn, med_sbp, med_dbp) |> # blood pressure in mmHg
  drop_na()

ldl_d1 = read_xpt("001_raw_data/L13AM_C.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbdldl, lbxtr) # lipid biomarkers (LDL, HDL, Triglyceride) in mg/dL

choles_d1 = read_xpt("001_raw_data/L13_C.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbxtc, lbxhdd) |>
  left_join(ldl_d1, by = "seqn") |>
  rename(chol_total = lbxtc, 
         chol_hdl = lbxhdd, 
         chol_ldl = lbdldl, 
         triglyceride = lbxtr) # cholesterol in mg/dL

glucose_d1 = read_xpt("001_raw_data/L10AM_C.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbxglu) |>
  rename(glucose = lbxglu) # glucose 

crp_d1 = read_xpt("001_raw_data/L11_C.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbxcrp) |>
  rename(crp = lbxcrp) # C-reactive protein (CRP) in mg/dL

clinical_d1 = bp_d1 |> 
  left_join(choles_d1, by = "seqn") |>
  left_join(glucose_d1, by = "seqn") |>
  left_join(crp_d1, by ="seqn")



# NAHNES 2005-2006 clinical variables 

bp_d2 = read_xpt("001_raw_data/BPX_D.xpt") |>
  janitor::clean_names() |>
  rowwise() |>
  mutate(
    med_sbp = median(c(bpxsy1, bpxsy2, bpxsy3, bpxsy4), na.rm = TRUE),
    med_dbp = median(c(bpxdi1, bpxdi2, bpxdi3, bpxdi4), na.rm = TRUE)
  ) |>
  select(seqn, med_sbp, med_dbp) |> # blood pressure in mmHg
  drop_na()

ldl_d2 = read_xpt("001_raw_data/TRIGLY_D.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbdldl, lbxtr)  

hdl_d2 = read_xpt("001_raw_data/HDL_D.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbdhdd)  

choles_d2 = read_xpt("001_raw_data/TCHOL_D.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbxtc) |>
  left_join(ldl_d2, by = "seqn") |>
  left_join(hdl_d2, by = "seqn") |>
  rename(chol_total = lbxtc, 
         chol_hdl = lbdhdd, 
         chol_ldl = lbdldl, 
         triglyceride = lbxtr) # lipid biomarkers (cholesterol, LDL, HDL, Triglyceride) in mg/dL

glucose_d2 = read_xpt("001_raw_data/GLU_D.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbxglu) |>
  rename(glucose = lbxglu) # glucose 

crp_d2 = read_xpt("001_raw_data/CRP_D.xpt") |>
  janitor::clean_names() |>
  select(seqn, lbxcrp) |>
  rename(crp = lbxcrp) # C-reactive protein (CRP) in mg/dL

clinical_d2 = bp_d2 |> 
  left_join(choles_d2, by = "seqn") |>
  left_join(glucose_d2, by = "seqn") |>
  left_join(crp_d2, by ="seqn")

clinical_cov = rbind(clinical_d1, clinical_d2) |> 
  drop_na()
save(clinical_cov, file = "002_data/clinical_cov.RData")
write.csv(clinical_cov, file = "002_data/clinical_cov.csv") 


# NAHNES 2003-2004 health outcome variables 

mc_d1 = read_xpt("001_raw_data/MCQ_C.xpt") |>
  janitor::clean_names() |>
  select(seqn, mcq160a, mcq220, mcq230a, mcq230b, mcq230c, mcq230d) |>
  filter(mcq220 != 7 & mcq220 != 9 & mcq160a != 7 & mcq160a != 9) |> #7=refused, 9=dontknow 
  mutate(arthritis = if_else(mcq160a == 1, 1, 0), #1=yes, 0=no
         malig = if_else(mcq220 == 1, 1, 0), #1=yes, 0=no
         breast = case_when(mcq230a==14~1, mcq230b==14~1, TRUE~0), 
         lung = case_when(mcq230a==23~1, mcq230b==23~1, mcq230c==23~1, TRUE~0), 
         ovary = case_when(mcq230a==28~1, mcq230b==28~1, TRUE~0), 
         skin = case_when(mcq230a%in%c(25, 32, 33)~1, 
                          mcq230b%in%c(25, 32, 33)~1, 
                          mcq230c%in%c(25, 32, 33)~1, TRUE~0)) |>
  select(-c(mcq160a, mcq220, mcq230a, mcq230b, mcq230c, mcq230d)) #1=yes, 0=no

# NAHNES 2005-2006 health outcome variables 

mc_d2 = read_xpt("001_raw_data/MCQ_D.xpt") |>
  janitor::clean_names() |>
  select(seqn, mcq160a, mcq220, mcq230a, mcq230b, mcq230c, mcq230d) |>
  filter(mcq220 != 7 & mcq220 != 9 & mcq160a != 7 & mcq160a != 9) |> #7=refused, 9=dontknow 
  mutate(arthritis = if_else(mcq160a == 1, 1, 0), #1=yes, 0=no
         malig = if_else(mcq220 ==1, 1, 0), #1=yes, 0=no
         breast = case_when(mcq230a==14~1, mcq230b==14~1, TRUE~0), 
         #lung = case_when(mcq230a==23~1, TRUE~2), 
         #ovary = case_when(mcq230a==28~1, mcq230b==28~1, mcq230c==28~1, TRUE~0), 
         skin = case_when(mcq230a%in%c(25, 32, 33)~1, 
                          mcq230b%in%c(25, 32, 33)~1, 
                          mcq230c%in%c(25, 32, 33)~1, TRUE~0)) |>
  select(-c(mcq160a, mcq220, mcq230a, mcq230b, mcq230c, mcq230d)) #1=yes, 0=no 

outcomes = rbind(mc_d1, mc_d2) |>
  drop_na()
save(outcomes, file = "002_data/outcomes.RData")
write.csv(outcomes, file = "002_data/outcomes.csv") 