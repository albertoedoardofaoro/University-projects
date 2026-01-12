# Computer Vision – Parking Lot Occupancy Estimation

This folder contains a deep learning project focused on the **automatic estimation of parking lot occupancy** from camera images.

The project uses a dataset of annotated parking lot images and implements two learning tasks:
- a **regression task** to predict the number of occupied parking spaces in an image,
- a **binary classification task** to determine whether a parking lot is mostly occupied or mostly empty.

The analysis was performed using **Python** on **Visual Studio Code**.

---

## Dataset

The dataset is based on the **PKLot** collection, which contains camera images of parking areas acquired under different weather conditions at the Universidade Federal do Paraná (Brazil).  
Each image is annotated with **bounding boxes** identifying individual parking spots, provided in **COCO format** :contentReference[oaicite:1]{index=1}.

From these annotations, project-specific targets were constructed:
- for regression, the target is the number of occupied parking spaces in an image,
- for classification, the target is a binary label based on the occupancy ratio, using a threshold of 50%.

---

## Problem formulation

The project is structured around two related tasks:

1. **Regression**  
   Predict the number of occupied parking spaces in a given image.

2. **Binary classification**  
   Determine whether the parking lot is mostly occupied or mostly empty, based on the ratio  
   \( \text{ratio}_{occ} = \frac{n_{occ}}{n_{total}} \) :contentReference[oaicite:2]{index=2}.

---

## What was done

### 1. Data loading and format
The dataset is handled using custom dataset classes that return:
- `(image, target)` pairs for regression,
- `(image, label)` pairs for classification.

Mini-batches are generated using a data loader with configurable batch size and shuffling :contentReference[oaicite:3]{index=3}.

---

### 2. Exploratory data analysis and preprocessing
The distribution of the binary labels was examined to check for class imbalance.  
Image preprocessing included:
- resizing,
- tensor conversion,
- normalization,
- and color jittering to increase robustness to lighting and weather conditions :contentReference[oaicite:4]{index=4}.

---

### 3. Neural network architecture
A **ResNet-18** backbone is used to extract visual features, producing a 512-dimensional feature vector.  
Two separate heads are attached:
- a **regression head** consisting of a single linear layer producing a scalar output,
- a **classification head** with two fully connected layers, ReLU activations, and dropout, producing a scalar logit :contentReference[oaicite:5]{index=5}.

---

### 4. Regression model
The regression task is trained using:
- mean squared error loss,
- Adam optimizer,
- Kaiming (He) initialization,
- a learning rate selected via a learning rate finder,
- warm-up followed by cosine annealing,
- early stopping based on validation loss.

The best model is selected and restored according to the minimum validation loss :contentReference[oaicite:6]{index=6}.

---

### 5. Regression evaluation
Performance is evaluated using:
- MSE, RMSE, and MAE,
- a relative error measure (ROE),
- and MAE computed across different occupancy ranges to assess robustness across low and high occupancy levels :contentReference[oaicite:7]{index=7}.

---

### 6. Classification model
The classification model reuses the **ResNet-18 backbone trained for regression**.  
The regression head is replaced with a new binary classification head, while the backbone parameters are frozen.

Training uses:
- binary cross-entropy loss,
- Adam optimizer with cosine annealing,
- early stopping based on validation accuracy :contentReference[oaicite:8]{index=8}.

---

### 7. Classification evaluation
The classifier is evaluated on a held-out test set using:
- confusion matrix,
- accuracy, precision, recall, and F1-score,
- ROC curve and AUC :contentReference[oaicite:9]{index=9}.

---

### 8. Model interpretation
Grad-CAM and Ablation-CAM are applied to visualize which image regions contribute most to the network’s predictions for both regression and classification tasks :contentReference[oaicite:10]{index=10}.

---

## Contents

This folder contains:
- the scripts for dataset construction, training, and evaluation,
- the trained models and checkpoints,
- and the presentation describing the full workflow.


- Grad-CAM and Ablation-CAM
