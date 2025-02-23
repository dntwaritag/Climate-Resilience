# Machine Learning Optimization for Climate Resilience

## Project Overview
This project explores the implementation of various machine learning models to enhance climate resilience through data-driven weather classification. The study leverages the Seattle Weather Dataset, applying classical ML algorithms (SVM) alongside deep learning models with optimization techniques such as regularization, early stopping, dropout, and hyperparameter tuning.

## Problem Statement
Climate change poses significant challenges to weather prediction and agricultural planning. Traditional forecasting models struggle to capture the complex relationships between meteorological variables, leading to inaccurate predictions. This project aims to develop machine learning models that improve weather classification accuracy, enabling better decision-making for climate resilience and sustainable agriculture.

## Dataset Description
- **Dataset**: Seattle Weather **[Dataset here](https://drive.google.com/file/d/1-Rdcuv-yQCjVdkCMQkgyccx4eGAkjOmL/view?usp=sharing)**
- **Source**: Publicly available **[Dataset here](https://www.kaggle.com/code/petalme/seattle-weather-prediction/input)**
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
| **nn_instance_1** | Adam          | None          | None   | No           | 0.001        | 78.23%     | 67.48%     | 78.23%    | 72.12%   |
| **nn_instance_2** | Adam          | L2 (0.01)     | None   | No           | 0.001        |77.55%  | 66.76%      | 77.55%    | 71.48%      |
| **nn_instance_3** | RMSprop       | None          | 0.3, 0.3, 0.3 | Yes (Patience: 3) | 0.0005 | 75.51% | 64.71%      | 75.51%   | 69.53%      |
| **nn_instance_4** | SGD           | L1 (0.01)     | 0.2, 0.2, 0.2 | Yes (Patience: 5) | 0.01 | 76.87%  | 67.26%      | 76.87%    | 70.98%  |
| **nn_instance_5** | Adam          | L2 (0.001)    | 0.3, 0.4, 0.3 | Yes (Patience: 5) | 0.0001 | 68.03%  | 58.11%  | 68.03%  | 62.57% |

---
## Key Findings & Best Model
**- nn_instance_1** achieved the highest accuracy (78.23%) and F1-score (72.12%).

- Performance varied across models due to different optimization techniques and hyperparameter settings.

**- Precision and recall trade-offs** were observed, indicating varying effectiveness in different weather conditions.



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



