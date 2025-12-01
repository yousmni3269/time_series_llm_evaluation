# time_series_llm_evaluation

This repository contains the code associated with the paper "Evaluation of Time-Series Versus General-Purpose LLM on Health Outcome Prediction."
The analysis uses accelerometer data from NHANES 2003–2006 (https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&) and compares:

* Time-series foundation model embeddings (MOMENT) (https://github.com/moment-timeseries-foundation-model/moment/blob/main/tutorials/representation_learning.ipynb) 

* General-purpose LLM embeddings and entropy (EntroLLM) (https://github.com/huangxq63/EntroLLM/tree/main) 

* Prompt-based LLM predictions via OpenAI ChatGPT 


The repository implements the data processing, modeling, and evaluation pipeline described in the paper.


## Contents
README.md – Overview of the repository

01_preprocessing.R – R script for cleaning and preparing the dataset

02_embedding_moment.ipynb – Jupyter Notebook (Python) for obtaining MOMENT representations 

03_Modeling.R – R script for modeling and evaluation. 

04_sensitivity.R - R script for sensitivity analysis 

05_llm_preprocess.R - R script for preparing data for general-purpose LLM predictions  

06_llm_prediction.ipynb - Jupyter Notebook (Python) for obtaining LLM predictions on overweight status usinsg OpenAI API

07_llm_evaluation.R - R script for evaluating the LLM predictions against the embedding-based approaches 

