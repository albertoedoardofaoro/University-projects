# Scoring Model – Customer Upselling

This folder contains a group project developed for a data-driven marketing course.  
The goal of the project is to build a **scoring model** to support a bank’s upselling campaign by identifying customers who are more likely to accept a commercial offer.

The analysis was carried out using a customer-level dataset with financial and behavioural indicators, and the model was implemented using **logistic regression**.

---

## Dataset

The dataset contains information on **19,120 customers** and **35 variables**, including one binary target variable (`FLG_TARGET`) indicating whether a customer accepted the upselling offer :contentReference[oaicite:1]{index=1}.

The remaining variables consist of **key performance indicators (KPIs)** describing:
- customers’ financial capacity (e.g. assets, inflows, movements),
- their activity and engagement with the bank (e.g. number of logins, transactions),
- and behavioural measures related to account usage.

---

## Objective

The objective is to:
- model the probability that a customer accepts the offer,
- rank customers according to this probability,
- and evaluate how this ranking can be used to guide a marketing campaign.

---

## What was done

The project follows a standard scoring model workflow.

### 1. Preliminary analysis
The structure of the dataset was analysed by:
- counting rows and columns,
- computing the average acceptance rate,
- and inspecting the univariate distributions of selected numerical KPIs.

This step was used to understand the scale and variability of the main predictors :contentReference[oaicite:2]{index=2}.

---

### 2. Data audit
Data quality checks were performed to identify:
- variables with zero variance,
- variables with a high proportion of missing values,
- and sets of highly correlated predictors.

Highly correlated variables were removed in order to avoid redundancy and instability in the model :contentReference[oaicite:3]{index=3}.

---

### 3. Model estimation
The dataset was split into:
- a **training set** (70% of observations),
- a **test set** (30% of observations).

A **logistic regression** model was fitted on the training data to predict `FLG_TARGET`.  
Model performance was evaluated using **ROC curves** and AUC on both training and test sets to assess generalization.

---

### 4. Variable selection
Starting from the full model, variable selection was performed using a **stepwise procedure based on AIC** in order to remove predictors that did not contribute to the model while balancing fit and complexity :contentReference[oaicite:4]{index=4}.

---

### 5. Scoring and ranking
The fitted model was used to compute a **score** for each customer, corresponding to the estimated probability of acceptance.  
Customers were then ranked and grouped into percentiles (ventiles) based on this score.

---

### 6. Lift and gain analysis
The quality of the ranking was evaluated using:
- **cumulative lift charts**,
- and **gain charts**,

which compare the concentration of positive responses in the top-scoring groups to what would be obtained under random targeting :contentReference[oaicite:5]{index=5}.

---

### 7. Campaign simulation
A marketing scenario was defined using assumptions on:
- total number of customers,
- value per customer,
- response rate,
- contact costs.

Using the score distribution and the lift values, cumulative revenues, costs, and profits were computed across ventiles to evaluate different targeting strategies :contentReference[oaicite:6]{index=6}.

---

## Contents

This folder contains:
- the data used for the analysis,
- the scripts for model estimation and scoring,
- and the presentation describing the full workflow.

---

