# Supplementaries 


## Missing data counts  
NHANES laboratory data was collected only from eligible participants, resulting in substantial biomarker missingness. After restricting to complete biomarker data, the analytic sample included 2,909 participants. Table 1 summarizes missing counts for all variables.

<div align="center">
  
**Table 1.** Missing Counts per Clinical Variables 
| Clinical Variable               | Missing Count (%) |
|---------------------------------|-------------------|
| LDL cholesterol                 | 4,018 (57.9%)     |
| Triglycerides                   | 3,922 (56.5%)     |
| Glucose                         | 3,904 (56.2%)     |
| Total cholesterol               | 565 (8.1%)        |
| HDL cholesterol                 | 565 (8.1%)        |
| C-reactive protein              | 552 (8.0%)        |
| Median systolic blood pressure  | 320 (4.6%)        |
| Median diastolic blood pressure | 320 (4.6%)        |
</div>

We also performed a sensitivity analysis using the full dataset (N = 6,943) to assess whether the reduced sample size affects inference, keeping all modeling conditions, penalized logistic ridge regression, and simulation settings identical. The sensitivity analyses confirmed that the complete-case (N=2,909) cohort is valid and representative of the full sample size. 


## Divergence Case Studies 
To better understand the comparative behavior of entropy and embedding representations, we examined cases where the entropy-only model and embedding-based models disagreed on overweight classification (Figure 1). In cases with intermittent low-intensity movement and little moderate or vigorous activity, entropy accurately classified overweight status while embeddings over-smoothed the temporal pattern and misclassified the participant. Conversely, for participants with long sedentary stretches interrupted by distinctive bursts of activity, embeddings sometimes captured temporal regularity that entropy did not. Although informative, such cases were uncommon and did not change overall model rankings.

<p align="center">
<img width="450" height="330" alt="Picture2" src="https://github.com/user-attachments/assets/305004c1-ff6a-4605-b497-98d14a06baa2" />

**Figure 1.** Physical activity patterns from participants with diverging results. (a–c) show accelerometer data from a participant whose overweight status was correctly classified as normal by the entropy-only model, while embedding-based models produced differing results. (d–f) show a participant whose overweight status was incorrectly classified as normal by the entropy-only model but correctly classified as overweight by embedding-based models.
</p>



## Fairness Test 
To address potential information disadvantages caused by the 512-token input limit of foundation models, we performed two additional fairness evaluations: (1) we evaluated MOMENT embeddings using 20-minute aggregated intervals to allow the model to capture the full 7-day recording period within its token limit ; and (2) we evaluated EntroGPT using input tokens truncated to 512 to directly compare its performance against EntroBERT, EntroCohere, and MOMENT under identical sequence-length constraints. 

**Table 2.** Comparative Performance ACross Fairness Evaluation Models (AUC)  
| Method | Cholesterol | LDL | HDL | Triglyceride | Glucose | BMI | Arthritis | Malignancy | Breast Cancer | Skin Cancer |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Demo_cov** | 0.593 | 0.569 | 0.724 | 0.670 | 0.719 | 0.566 | 0.788 | 0.783 | 0.883 | 0.864 |
| **EntroBert_cov** | 0.576 | 0.570 | 0.712 | 0.652 | 0.711 | 0.589 | 0.778 | 0.679 | 0.875 | 0.852 |
| **EntroCohere_cov** | 0.562 | 0.559 | 0.702 | 0.639 | 0.702 | 0.578 | 0.768 | 0.694 | 0.875 | 0.856 |
| **EntroGPT_cov** | 0.580 | 0.581 | 0.692 | 0.653 | 0.707 | 0.630 | 0.770 | 0.744 | 0.878 | 0.865 |
| **EntroGPT_truncated_cov** | 0.554 | 0.559 | 0.708 | 0.652 | 0.691 | 0.583 | 0.761 | 0.695 | 0.877 | 0.842 |
| **Entropy_cov** | 0.588 | 0.570 | 0.726 | 0.675 | 0.718 | 0.618 | 0.790 | 0.798 | 0.875 | 0.856 |
| **Moment_cov** | 0.532 | 0.541 | 0.667 | 0.615 | 0.669 | 0.565 | 0.731 | 0.630 | 0.879 | 0.862 |
| **Raw_Moment_cov** | 0.552 | 0.522 | 0.679 | 0.615 | 0.680 | 0.524 | 0.743 | 0.541 | 0.871 | 0.863 |
| **min20_Moment_cov** | 0.578 | 0.581 | 0.658 | 0.624 | 0.690 | 0.578 | 0.736 | 0.679 | 0.879 | 0.861 |
</div>

Truncating EntroGPT input tokens to 512 negatively affected its performance compared to the full-sequence version. 
However, the model rankings remained largely identical and the EntroGPT truncated still achieved higher AUC values than the 5-minute MOMENT model for most outcomes, including BMI (0.583 vs. 0.565) and glucose (0.691 vs. 0.669). 
Furthermore, using a 20-minute interval (min20_Moment) to provide the foundation model with the full week of data did not yield significant improvements or change the overall model rankings for most clinical outcomes. 



