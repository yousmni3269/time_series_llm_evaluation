# time_series_llm_evaluation

This repository contains the code associated with the paper "Evaluation of Time-Series Versus General-Purpose LLM on Health Outcome Prediction."

The data is from NHANES (https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&) and MOMENT time-series foundation model (https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb) is used. 

The code implements the methods and analyses described in the paper, including data preprocessing, modeling, and evaluation.


## Contents
README.md – Overview of the repository

01_preprocessing.R – R script for cleaning and preparing the dataset

02_embedding_moment.ipynb – Jupyter Notebook (Python) for obtaining MOMENT representations 

03_Modeling.R – R script for modeling and evaluation. 

04_sensitivity.R - R script for sensitivity analysis 

05_llm_preprocess.R - R script for preparing data for general-purpose LLM predictions  

06_llm_prediction.ipynb - Jupyter Notebook (Python) for obtaining LLM predictions on overweight status usinsg OpenAI API

07_llm_evaluation.R - R script for evaluating the LLM predictions against the embedding-based approaches 

