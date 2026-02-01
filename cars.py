import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import datetime,date
import pytz
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

data = pd.read_csv('vehicles.csv')
lower = data.price.quantile(0.01)  # 1st percentile
upper = data.price.quantile(0.99)  # 99th percentile

data = data[(data.price >= lower) & (data.price <= upper)]

print(data.price.describe())
# svalues = set(data['condition'].values)

y = data.price
X = data.drop("price",axis = 1)

X_train_full, X_val_full, y_train, y_val = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)

# my_columns = numerical_columns + object_columns

def clean_numeric_features(X_train_full):

    condition_map ={"salvage": 0,
                    "fair": 1,
                    "good": 2,
                    "excellent": 3,
                    "like new": 4,
                    "new": 5}
    
    #! FEATURE ENGINEERING
    # MAPPING CONDITION

    X_train_full["condition_num"] = X_train_full["condition"].map(condition_map)   
    X_val_full["condition_num"] = X_val_full["condition"].map(condition_map)   
    

    # cylinders
    X_train_full["cylinders"] = (
        X_train_full["cylinders"]
        .str.extract(r'(\d+)')
        .astype(float)
    )

    X_val_full["cylinders"] = (
        X_val_full["cylinders"]
        .str.extract(r'(\d+)')
        .astype(float)
    )


    
    X_train_full['posting_date'] = pd.to_datetime(
    X_train_full['posting_date'],
    format='%Y-%m-%dT%H:%M:%S%z',
    errors='coerce',
    utc=True
    )

    X_val_full['posting_date'] = pd.to_datetime(
        X_val_full['posting_date'],
        format='%Y-%m-%dT%H:%M:%S%z',
        errors='coerce',
        utc=True
    )

    current_date = date.today()

    current_year = current_date.year

    X_train_full["age"] = current_year - X_train_full["year"]
    X_val_full["age"] = current_year - X_val_full["year"]

    
    model_counts = X_train_full['model'].value_counts()

  
    common_models = model_counts[model_counts >= 50].index

    X_train_full['model_grouped'] = X_train_full['model'].where(
        X_train_full['model'].isin(common_models), other='Other'
    )
    X_val_full['model_grouped'] = X_val_full['model'].where(
        X_val_full['model'].isin(common_models), other='Other'
    )
    

    X_train_full["mileage_per_year"] = X_train_full["odometer"] / X_train_full["age"]
    X_val_full["mileage_per_year"] = X_val_full["odometer"] / X_val_full["age"]

    #!

    current_time = datetime.now(pytz.UTC)

    X_train_full["time_since_posted"] = (

        current_time - X_train_full["posting_date"]

    ).dt.days

    X_val_full["time_since_posted"] = (

        current_time - X_val_full["posting_date"]

    ).dt.days

    # we use time_since_posted instead. this is called #! Feature engineering
    X_train_full.drop("posting_date",axis=1,inplace= True)
    X_val_full.drop("posting_date",axis = 1,inplace = True)

    X_train_full.drop("year",axis = 1,inplace = True)
    X_val_full.drop("year",axis = 1,inplace = True)

    X_train_full.drop("condition",axis = 1,inplace = True)
    X_val_full.drop("condition",axis = 1,inplace = True)

    X_train_full.drop("model",axis = 1,inplace = True)
    X_val_full.drop("model",axis = 1,inplace = True)

    # ----- Dropping unique columns-----

    X_train_full.drop('url',axis = 1,inplace = True)
    X_val_full.drop('url',axis = 1 , inplace = True)

    X_train_full.drop('region_url',axis = 1,inplace = True)
    X_val_full.drop('region_url',axis = 1,inplace = True)

    X_train_full.drop("VIN",axis = 1,inplace = True)
    X_val_full.drop("VIN",axis = 1,inplace = True)

    X_train_full.drop('image_url',axis = 1,inplace = True)
    X_val_full.drop('image_url',axis = 1,inplace = True)



    # ! MIGHT BE IMPORTANT LATER
    X_train_full.drop('description',axis = 1,inplace=True)
    X_val_full.drop('description',axis = 1,inplace=True)

    # ! ERROR BECAUSE ALL NaN

    X_train_full.drop('county',axis = 1,inplace =True)
    X_val_full.drop('county',axis = 1,inplace =True)



clean_numeric_features(X_train_full)

# ENCODING AND IMPUTING

numerical_columns = [col for col in X_train_full if X_train_full[col].dtype in ["float64","int64"]]

low_car_columns = [col for col in X_train_full if X_train_full[col].nunique() < 10 and X_train_full[col].dtype == "object"]
high_car_columns = [col for col in X_train_full if X_train_full[col].nunique() >= 10 and X_train_full[col].dtype == "object"]

numerical_transformer = SimpleImputer(strategy='median')


low_categorical_transformer = Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                              ('onehot',OneHotEncoder(handle_unknown='ignore'))])  

high_categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                               ('ordinal',OrdinalEncoder(handle_unknown='use_encoded_value',unknown_value=-1))])

preprocessor = ColumnTransformer(transformers=[
    ('num',numerical_transformer,numerical_columns),
    ('low_cat',low_categorical_transformer,low_car_columns),
    ('high-cat',high_categorical_transformer,high_car_columns)
])


model = RandomForestRegressor(n_estimators=10,random_state=0)

pipeline = Pipeline(steps=[('preprocessor',preprocessor),
                           ('model',model)])

pipeline.fit(X_train_full,y_train)
pred = pipeline.predict(X_val_full)

score = mean_absolute_error(y_val,pred)

print(score)
####



