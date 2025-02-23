# Machine Learning Optimization for Climate Resilience

## Project Overview
This project explores the implementation of various machine learning models to enhance climate resilience through data-driven weather classification. The study leverages the Seattle Weather Dataset, applying classical ML algorithms (SVM) alongside deep learning models with optimization techniques such as regularization, early stopping, dropout, and hyperparameter tuning.

## Problem Statement
Traditional methods of predicting weather and crop yields often fail to
account for complex interactions between climate variables and
agricultural conditions

## Dataset Description
- **Dataset**: Seattle Weather Dataset here
- **Source**: Publicly available dataset here
- **Problem Statement**: Classifying weather conditions based on historical climate data.
- **Features Used**: Temperature, Precipitation, Wind Speed,  etc.
- **Target Variable**: Weather category (Clear, Fog, Rain, Snow)

---
## Performance Metrics
Each model is evaluated using the following metrics:

- Accuracy
- Loss
- F1-score
- Precision
- Recall
  
---
## Implemented Models

| Model Name     | Optimizer Used | Regularization | Dropout | Early Stopping | Learning Rate | Accuracy | Precision | Recall | F1-Score |
|---------------|---------------|---------------|--------|--------------|--------------|---------|----------|--------|----------|
| **SVM Model** | N/A           | N/A           | N/A    | No           | N/A          | **89%**  | 88%      | 87%    | 88%      |
| **Instance 1** | Adam          | None          | None   | No           | 0.001        | 85%      | 84%      | 83%    | 84%      |
| **Instance 2** | Adam          | L2 (0.01)     | None   | No           | 0.001        | **90%**  | 89%      | 88%    | 89%      |
| **Instance 3** | RMSprop       | None          | 0.3, 0.3, 0.3 | Yes (Patience: 3) | 0.0005 | **91%**  | 90%      | 89%    | 90%      |
| **Instance 4** | SGD           | L1 (0.01)     | 0.2, 0.2, 0.2 | Yes (Patience: 5) | 0.01 | **92%**  | 91%      | 90%    | **91%**  |
| **Instance 5** | Adam          | L2 (0.001)    | 0.3, 0.4, 0.3 | Yes (Patience: 5) | 0.0001 | **93%**  | **92%**  | **91%**  | **92%**  |

---
## Key Findings & Best Model
**- Instance 5 (Adam + L2 Regularization + Dropout + Early Stopping)** achieved the highest accuracy (93%) and F1-score (92%).
**- Neural Network models** generally **outperformed the classical ML models** due to deep learning’s capacity to capture complex weather patterns.
**- Regularization (L1/L2) and dropout improved generalization and prevented overfitting.**

## File Structure
```
Climate Resilience/
├── notebook.ipynb  # Jupyter Notebook with full implementation
├── saved_models/   # Directory containing saved models
│   ├── nn_instance_1.h5
│   ├── nn_instance_2.h5
│   ├── nn_instance_3.h5
│   ├── nn_instance_4.h5
│   ├── nn_instance_5.h5
│   ├── ml_model_svm.pkl
└── README.md       # Project documentation

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository_link>
   cd climate-resilient
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook (`notebook.ipynb`) to train and evaluate models.
4. Load the best-performing model (`nn_instance_5.h5`):
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('saved_models/nn_instance_5.h5')
   ```
5. Make predictions using test data.

---

## Contributors
👤 **Denys Ntwaritaganzwa**  
📧 ntwaridenis@gmail.com  
🌍 African Leadership University, Rwanda  

This project aligns with Rwanda’s strategic **climate resilience goals** and demonstrates how AI can transform **sustainable agriculture.** 🌱🚀


## Video Presentation
Watch the Demo **[video Link](https://drive.google.com/file/d/1xPVuOydu7vtM7dhopXVRdfLkX4m6mvhY/view?usp=sharing)** here for more.

## Conclusion
This project demonstrates the impact of optimization techniques on machine learning models for climate resilience. By leveraging deep learning and hyperparameter tuning, Instance 5 emerged as the best model, achieving the highest accuracy and robustness in predictions.

---

## Contributors
👤 **Denys Ntwaritaganzwa**  
📧 d.ntwaritag@alustudent.com  
🌍 Developed for educational purpose  

This project aligns with Rwanda’s strategic **climate resilience goals** and demonstrates how AI can transform **sustainable agriculture.** 🌱🚀



