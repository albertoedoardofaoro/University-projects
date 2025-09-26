
# =====================================================
# SCORING MODEL - Upselling Campaign
# =====================================================

set.seed(123)
library(caret)
library(dplyr)
library(MASS)
library(pROC)
library(ggplot2)

# -----------------------------------------------------
# 01. PRELIMINARY ANALYSIS
# -----------------------------------------------------

# Dataset structure
rows <- nrow(data_clean)
cols <- ncol(data_clean)
cat("Rows:", rows, "Columns:", cols, "\n")

# Average acceptance rate
avg_acceptance <- mean(data_clean$FLG_TARGET)
cat("Average acceptance rate:", round(avg_acceptance, 3), "\n")

# -----------------------------------------------------
# 02. DATA AUDIT
# -----------------------------------------------------

# Check constant columns
const_cols <- sapply(data_clean, function(x) sd(as.numeric(x), na.rm = TRUE) == 0)
cat("Constant columns:", names(data_clean)[const_cols], "\n")

# Check NAs
na_pct <- sapply(data_clean, function(x) mean(is.na(x)))
cat("Columns with >70% NA:", names(data_clean)[na_pct > 0.7], "\n")

# Remove highly correlated predictors (threshold 0.7)
num_vars <- data_clean %>% select(where(is.numeric))
corr_mat <- cor(num_vars, use = "pairwise.complete.obs")
high_corr <- findCorrelation(corr_mat, cutoff = 0.7, names = TRUE)
data_clean <- data_clean %>% select(-all_of(high_corr))

# -----------------------------------------------------
# 03. MODEL ESTIMATION
# -----------------------------------------------------

# Train/test split
train_idx <- createDataPartition(data_clean$FLG_TARGET, p = 0.7, list = FALSE)
train_data <- data_clean[train_idx, ]
test_data  <- data_clean[-train_idx, ]

# Convert categorical
train_data$TIPO_MULTIB <- as.factor(train_data$TIPO_MULTIB)
test_data$TIPO_MULTIB  <- as.factor(test_data$TIPO_MULTIB)

# Logistic regression (full + stepwise AIC)
model <- glm(FLG_TARGET ~ ., data = train_data, family = "binomial")
stepwise_model <- stepAIC(model, direction = "both")

# Predict probabilities
train_data$pred_prob <- predict(stepwise_model, newdata = train_data, type = "response")
test_data$pred_prob  <- predict(stepwise_model, newdata = test_data, type = "response")

# Overfitting: ROC & AUC
roc_train <- roc(train_data$FLG_TARGET, train_data$pred_prob)
roc_test  <- roc(test_data$FLG_TARGET,  test_data$pred_prob)

plot(roc_train, col = "blue", lwd = 2,
     main = "ROC Curve: Train vs Test", legacy.axes = TRUE)
lines(roc_test, col = "red", lwd = 2)
legend("bottomright",
       legend = c(sprintf("Train AUC = %.3f", auc(roc_train)),
                  sprintf("Test AUC = %.3f", auc(roc_test))),
       col = c("blue", "red"), lwd = 2)

# -----------------------------------------------------
# 04. LIFT CHART
# -----------------------------------------------------

compute_lift_table <- function(data) {
  total_population <- nrow(data)
  total_target     <- sum(data$FLG_TARGET)
  
  data %>%
    mutate(ventile = ntile(-pred_prob, 20)) %>%
    group_by(ventile) %>%
    summarise(Population = n(),
              Target = sum(FLG_TARGET),
              .groups = "drop") %>%
    arrange(ventile) %>%
    mutate(
      Cum_population = cumsum(Population),
      Cum_target = cumsum(Target),
      Redemption = Target / Population,
      Captured = Target / total_target,
      Cum_captured = Cum_target / total_target,
      Population_pct = Population / total_population,
      Cum_population_pct = Cum_population / total_population,
      Lift = Captured / Population_pct,
      Cum_lift = Cum_captured / Cum_population_pct
    )
}

train_lift <- compute_lift_table(train_data)
test_lift  <- compute_lift_table(test_data)

# Overfitting check (first ventile)
train_first_lift <- train_lift$Lift[1]
test_first_lift  <- test_lift$Lift[1]
lift_diff_pct    <- 100 * (train_first_lift - test_first_lift) / test_first_lift

cat("Train first ventile Lift:", round(train_first_lift, 2), "\n")
cat("Test first ventile Lift:", round(test_first_lift, 2), "\n")
cat("Difference (%):", round(lift_diff_pct, 2), "\n")

# -----------------------------------------------------
# 05. CAMPAIGN MANAGEMENT
# -----------------------------------------------------

total_customers <- 1e6
mean_target_rate <- mean(train_data$FLG_TARGET)
net_profit_per_upsell <- 120
fixed_cost <- 100000
variable_cost_per_customer <- 20
response_rate <- 0.20

lift_table <- test_lift %>%
  mutate(
    cum_customers = total_customers * ventile / 20,
    target_rate_bin = mean_target_rate * Cum_lift,
    expected_responders = cum_customers * target_rate_bin * response_rate,
    revenue = expected_responders * net_profit_per_upsell,
    variable_cost = cum_customers * variable_cost_per_customer,
    net_profit = revenue - variable_cost - fixed_cost
  )

optimal_row <- lift_table[which.max(lift_table$net_profit), ]
cat(sprintf("✅ Optimal cut-off: top %d ventiles\n💰 Max Net Profit: €%.0f\n",
            optimal_row$ventile, optimal_row$net_profit))

ggplot(lift_table, aes(x = ventile, y = net_profit)) +
  geom_line(color = "blue", linewidth = 1) +
  geom_point() +
  labs(title = "Net Profit vs % Customers Contacted",
       x = "Ventiles (5% each)", y = "Net Profit (€)") +
  theme_minimal()
