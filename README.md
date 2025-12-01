# time_series_llm_evaluation

This repository contains the code associated with the paper **"Evaluating Representation Embeddings from LLMs and Time-Series Foundation Models for Wearable Accelerometer-Based Health Prediction."**
The analysis uses accelerometer data from NHANES 2003–2006 (https://wwwn.cdc.gov/nchs/nhanes/search/datapage.aspx?Component=Examination&) and compares:

* **[Time-series foundation model embeddings (MOMENT)](https://github.com/moment-timeseries-foundation-model/moment)**
* **[General-purpose LLM embeddings and entropy (EntroLLM)](https://github.com/huangxq63/EntroLLM)**
* **Prompt-based LLM predictions via [OpenAI ChatGPT](https://platform.openai.com)**

The repository implements the data processing, modeling, and evaluation pipeline described in the paper.

---

## Contents

- `README.md` – Overview of the repository  
- `01_preprocessing.R` – Data cleaning and preparation  
- `02_embedding_moment.ipynb` – Jupyter Notebook (Python) for obtaining MOMENT representations  
- `03_Modeling.R` – Predictive modeling and evaluation  
- `04_sensitivity.R` – Sensitivity analysis  
- `05_llm_preprocess.R` – Preparing data for general-purpose LLM predictions  
- `06_llm_prediction.ipynb` – LLM-based overweight prediction using the OpenAI API  
- `07_llm_evaluation.R` – Evaluation of LLM predictions vs. embedding-based approaches  

---

## Supplements  

Supplemental materials include:

- **Missing data counts:** Summary of missing data after restricting to the complete-case cohort.  
- **Divergence case studies:** Participants where entropy-only and embedding-based models produced different predictions.  

Additional supplemental materials are available in the `docs/` folder.
