# Logistics Delivery Time Prediction & Optimization

## Week 4 Internship Project

A machine learning project for predicting last-mile delivery time and supporting logistics optimization using Python and Scikit-learn.

## Project Objective

The project develops predictive models to estimate delivery duration from operational logistics factors and converts the resulting insights into practical optimization strategies. The workflow covers data preparation, model training, cross-validation, hyperparameter tuning, performance evaluation, visualization, and logistics recommendations.

## Dataset

The project uses a simulated logistics dataset containing 6,000 delivery routes. Features include route distance, number of stops, package weight, departure time, weather, vehicle type, driver experience, warehouse location, and traffic conditions.

**Target variable:** Delivery Time

## Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor

Five-fold cross-validation and hyperparameter tuning are used to improve model reliability and performance. Evaluation metrics include **MAE, RMSE, and R²**.

## Optimization Strategies

Model insights are used to support:

- Driver and vehicle assignment
- Departure-time optimization
- SLA risk identification
- Delivery-delay reduction
- Better resource utilization

## Repository Structure

```text
logistics-delivery-time-prediction/
├── data/
│   └── delivery_data.csv
├── figures/
│   ├── fig_actual_pred.png
│   ├── fig_importance.png
│   ├── fig_models.png
│   ├── fig_resid.png
│   └── fig_target.png
├── report/
│   └── Logistics_Predictive_Modeling_Report.docx
├── results/
│   ├── cv.csv
│   ├── results.json
│   └── test.csv
├── delivery_time_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib

## How to Run

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the model:

```bash
python delivery_time_model.py
```

## Deliverables

The `report` folder contains the complete Week 4 internship report. The `figures` and `results` folders contain supporting outputs generated during the analysis.

## Project Outcome

This project demonstrates how predictive analytics can support data-driven logistics decisions by forecasting delivery time and identifying opportunities to improve operational efficiency, resource allocation, and service reliability.

## GitHub Description

Machine learning project for last-mile delivery time prediction and logistics optimization using Python, Scikit-learn, and predictive analytics.
