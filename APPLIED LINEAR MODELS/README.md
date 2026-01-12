# Applied Linear Models – IgA in Celiac Disease

This folder contains a statistical modelling project developed for the course *Applied Linear Models*.  
The work is based on a clinical dataset related to **Celiac Disease** and focuses on modelling **IgA antibody levels** using linear regression techniques.

The analysis was carried out in **R**.

---

## Dataset

The dataset comes from Wageningen University & Research and contains clinical, demographic and immunological information on individuals evaluated for Celiac Disease.

After data cleaning and variable selection based on clinical relevance, the dataset includes:

- Age  
- Gender  
- Diabetes  
- Diarrhoea type  
- Abdominal pain  
- Short stature classification  
- IgA, IgG, IgM antibody levels  
- Marsh classification (histological grading of intestinal damage)  
- Disease diagnosis  

The response variable used in the modelling is **IgA**, which was log-transformed to address skewness.

---

## Objectives

The goal of the project is to study how IgA levels are associated with:
- demographic variables,
- clinical conditions,
- immunological markers,
- and disease severity indicators,

and to build a linear regression model that can be used for prediction in terms of Celiac Disease.

---

## What was done

The project follows the full applied linear modelling workflow:

### 1. Exploratory data analysis
- Distribution of IgA and log(IgA)
- Correlation analysis for continuous variables
- Boxplots for all categorical predictors

### 2. Model selection
Best subset selection was performed using the `regsubsets()` function, with model comparison based on:
- AIC  
- BIC  
- Adjusted R²  
- Mallows’ Cp  
- 10-fold cross-validation  

This was used to determine the number of predictors to include in the final regression model.

### 3. Model fitting
An ordinary least squares regression was fitted using:
- Age  
- Gender  
- Diabetes  
- Short stature  
- IgM  
- Marsh classification  

with log(IgA) as the response variable.

### 4. Collinearity analysis
Variance Inflation Factors (VIF and GVIF) were computed to assess multicollinearity, taking into account categorical predictors with multiple levels.

### 5. Diagnostic analysis
The standard regression assumptions were evaluated using:
- residuals vs fitted values,
- histograms and Q–Q plots of residuals,
- Shapiro–Wilk test,
- leverage values,
- standardized residuals,
- Cook’s distance.

Outliers and leverage points were identified and assessed.

### 6. Model comparison
Nested models were compared using ANOVA to evaluate whether dropping predictors led to a significant loss of information.

### 7. Prediction
The fitted model was used to generate predictions and confidence intervals for new observations.

### 8. Simulation
Simulated IgA values were generated from the fitted model using the estimated coefficients and residual variance, and compared to observed values.

---

## Contents

This folder contains:
- the Rmd scripts used for the analysis,
- the cleaned dataset,
- and the final report knitted to pdf for full visualizations of the results.

---
