## Missing data counts  

NHANES laboratory data was collected only from eligible participants, resulting in substantial biomarker missingness. After restricting to complete biomarker data, the analytic sample included 2,909 participants. Table 1 summarizes missing counts for all variables.

Table 1: Missing Counts per Clinical Variables 
## Clinical Variable Missingness (N = 6,943)

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

We also performed a sensitivity analysis using the full dataset (N = 6,943) to assess whether the reduced sample size affects inference, keeping all modeling conditions, penalized logistic ridge regression, and simulation settings identical. The sensitivity analyses confirmed that the complete-case (N=2,909) cohort is valid and representative of the full (N=6,943) sample size. 

## Divergence Case Studies 
To better understand the comparative behavior of entropy and embedding representations, we examined cases where the entropy-only model and embedding-based models disagreed on overweight classification (Figure 1). In cases with intermittent low-intensity movement and little moderate or vigorous activity, entropy accurately classified overweight status while embeddings over-smoothed the temporal pattern and misclassified the participant. Conversely, for participants with long sedentary stretches interrupted by distinctive bursts of activity, embeddings sometimes captured temporal regularity that entropy did not. Although informative, such cases were uncommon and did not change overall model rankings.
<img width="294" height="214" alt="Picture1" src="https://github.com/user-attachments/assets/a950098d-d2a1-4597-8e18-a69429de3a08" />


Figure 1: Physical activity pattern from participants with diverging results: (a-c) show accelerometer data from a participant whose overweight status was correctly classified as normal by the Entropy-only model, while other models produced diverging results. (d-f) are from a participant whose overweight status was incorrectly specified as normal by the Entropy-only model but correctly classified as overweight by other models.



