import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


@st.cache_data
def load_data():
    df = pd.read_csv("BostonHousing.csv")
    df['rm'].fillna(df['rm'].mean(), inplace=True)
    return df

df = load_data()


menu = ["Dataset Overview", "Visualisations", "Model Performance", "Prediction"]
choice = st.sidebar.selectbox("Navigation", menu)


if choice == "Dataset Overview":
    st.title("Boston Housing Prices - Dataset Overview")
    st.write("*Shape:*", df.shape)
    st.write("*Columns:*", df.columns.tolist())
    st.dataframe(df.head())
    st.write("*Missing Values:*")
    st.write(df.isnull().sum())


elif choice == "Visualisations":
    st.title("Visualisations")

  
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

 
    st.subheader("Distribution of MEDV (Target)")
    fig, ax = plt.subplots()
    sns.histplot(df['medv'], bins=20, kde=True, ax=ax)
    st.pyplot(fig)

    
    st.subheader("RM vs MEDV")
    fig, ax = plt.subplots()
    sns.scatterplot(x='rm', y='medv', data=df, ax=ax)
    st.pyplot(fig)


elif choice == "Model Performance":
    st.title("Model Training and Evaluation")

    X = df.drop(columns=['medv'])
    y = df['medv']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))
    lr_r2 = r2_score(y_test, lr_preds)
    lr_cv = cross_val_score(lr, X, y, cv=5, scoring='r2').mean()

   
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_r2 = r2_score(y_test, rf_preds)
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring='r2').mean()

    
    results_df = pd.DataFrame({
        "Model": ["Linear Regression", "Random Forest"],
        "RMSE": [lr_rmse, rf_rmse],
        "R²": [lr_r2, rf_r2],
        "CV R²": [lr_cv, rf_cv]
    })
    st.write(results_df)

    
    if rf_r2 > lr_r2:
        pickle.dump(rf, open("model.pkl", "wb"))
        st.success("Random Forest saved as best model!")
    else:
        pickle.dump(lr, open("model.pkl", "wb"))
        st.success("Linear Regression saved as best model!")


elif choice == "Prediction":
    st.title("Predict House Price")

    try:
        model = pickle.load(open("model.pkl", "rb"))
    except FileNotFoundError:
        st.error("No model found. Please run 'Model Performance' first.")
        st.stop()

    st.write("Enter feature values to predict:")

    crim = st.number_input("CRIM (per capita crime rate)", 0.0, 100.0, 0.1)
    zn = st.number_input("ZN (residential land zoned %)", 0.0, 100.0, 0.0)
    indus = st.number_input("INDUS (non-retail business acres)", 0.0, 30.0, 5.0)
    chas = st.selectbox("CHAS (Charles River dummy variable)", [0, 1])
    nox = st.number_input("NOX (nitric oxides concentration)", 0.0, 1.0, 0.5)
    rm = st.number_input("RM (average number of rooms)", 0.0, 10.0, 6.0)
    age = st.number_input("AGE (% built before 1940)", 0.0, 100.0, 60.0)
    dis = st.number_input("DIS (distance to employment centers)", 0.0, 15.0, 4.0)
    rad = st.number_input("RAD (accessibility to radial highways)", 1, 24, 1)
    tax = st.number_input("TAX (property-tax rate)", 100, 800, 300)
    ptratio = st.number_input("PTRATIO (pupil–teacher ratio)", 10.0, 30.0, 15.0)
    b = st.number_input("B (proportion of Black residents)", 0.0, 400.0, 350.0)
    lstat = st.number_input("LSTAT (% lower status)", 0.0, 40.0, 10.0)

    if st.button("Predict"):
        input_data = np.array([[crim, zn, indus, chas, nox, rm, age, dis, rad,
                                tax, ptratio, b, lstat]])
        prediction = model.predict(input_data)[0]
        st.success(f"Predicted Median House Price: ${prediction * 1000:.2f}")