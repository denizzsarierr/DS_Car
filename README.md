# Vehicle Price Prediction

Machine learning project to predict used vehicle prices using structured Craigslist vehicle data.

## 🔧 Tech Stack
- Python
- Pandas
- Scikit-learn
- Matplotlib

## Data Cleaning & Feature Engineering
- Removed extreme price outliers (1st–99th percentile)
- Engineered features:
  - Vehicle age
  - Mileage per year
  - Time since posting
  - Numeric condition mapping
  - Cylinder count extraction
- Grouped rare vehicle models into `"Other"`
- Dropped high-cardinality and identifier columns (VIN, URLs, descriptions)

## Preprocessing
- Numerical features: median imputation
- Low-cardinality categoricals: one-hot encoding
- High-cardinality categoricals: ordinal encoding
- Implemented using `ColumnTransformer` and `Pipeline`

## Model
- RandomForest Regressor
- Validation strategy: Train / Validation split (80/20)
- Metric: Mean Absolute Error (MAE)

## Result
- Achieved reasonable MAE on validation data
- Demonstrates a complete regression pipeline with real-world feature engineering

## Notes
This project focuses on handling messy real-world data and building a clean, reproducible ML pipeline.

## How to Run

```bash
pip install -r requirements.txt
python cars.py
```
