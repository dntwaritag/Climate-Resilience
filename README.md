# Machine Learning Optimization for Climate Resilience

## Project Overview
This project explores the implementation of various **machine learning models** to enhance **climate resilience** through data-driven weather classification. The study leverages the **Seattle Weather Dataset**, applying classical **ML algorithms (Logistic Regression, SVM, XGBoost)** alongside **deep learning models** with optimization techniques such as **regularization, early stopping, dropout, and hyperparameter tuning**.

## Dataset Description
- **Dataset**: Seattle Weather Dataset
- **Source**: Publicly available dataset
- **Problem Statement**: Classifying weather conditions based on historical climate data.
- **Features Used**: Temperature, Precipitation, Wind Speed, etc.
- **Target Variable**: Weather category (Clear, Fog, Rain, Snow)

## Implemented Models

| Model Name  | Optimizer Used | Regularization | Dropout | Early Stopping | Learning Rate | Accuracy | Precision | Recall | F1-Score |
|-------------|---------------|---------------|---------|----------------|---------------|----------|-----------|--------|---------|
| Logistic Regression | N/A | N/A | N/A | No | N/A | 86.5% | 85.4% | 87.2% | 86.3% |
| Instance 1  | Adam          | None          | None    | No             | 0.001         | 88.1%    | 87.2%     | 89.0%  | 88.0%   |
| Instance 2  | Adam          | L2 (0.01)     | None    | Yes            | 0.001         | 89.6%    | 88.5%     | 90.1%  | 89.3%   |
| Instance 3  | SGD           | None          | 0.3, 0.2 | No             | 0.01          | 87.4%    | 86.8%     | 88.0%  | 87.4%   |
| Instance 4  | RMSprop       | L1 (0.005)    | 0.3, 0.2 | Yes            | 0.0005        | 90.2%    | 89.3%     | 91.0%  | 90.1%   |

## Key Findings & Best Model
- **Instance 4 (RMSprop + L1 Regularization + Dropout + Early Stopping)** achieved the highest accuracy (90.2%) and F1-score.
- **Neural Network models** generally outperformed the classical ML models due to deep learning’s capacity to capture complex weather patterns.
- **Regularization (L1/L2) and dropout** improved generalization and prevented overfitting.

## File Structure
```
Climate Resilience/
├── notebook.ipynb  # Jupyter Notebook with full implementation
├── saved_models/   # Directory containing saved models
│   ├── Instance_1.h5
│   ├── Instance_2.h5
│   ├── Instance_3.h5
│   ├── Instance_4.h5
│   ├── logistic_regression.pkl
└── README.md       # Project documentation
```

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository_link>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook (`notebook.ipynb`) to train and evaluate models.
4. Load the best-performing model (`Instance_4.h5`):
   ```python
   from tensorflow.keras.models import load_model
   model = load_model('saved_models/Instance_4.h5')
   ```
5. Make predictions using test data.

## Video Presentation
.....

## Conclusion
This project demonstrates the **impact of optimization techniques** on machine learning models for climate resilience. By leveraging deep learning and tuning hyperparameters, **Instance 4 emerged as the best model, achieving the highest accuracy and robustness in predictions.**

---
**Prepared by:** Denys Ntwaritaganzwa  
**Course:** Introduction to Machine Learning  
**Institution:** African Leadership University

