import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import LabelEncoder
from pycaret.regression import RegressionExperiment
from pycaret.classification import ClassificationExperiment

def perform_eda(dataframe):
  st.header("Exploratory Data Analysis (EDA):-")

  st.subheader("Data Shape")
  st.write(f"The dataset has {dataframe.shape[0]} rows and {dataframe.shape[1]} columns.") 
            
  st.subheader("Data Types")
  st.write(dataframe.dtypes)
  
  analyze_data = st.checkbox("Perform EDA?")
  if analyze_data:
    visualiz_columns = st.multiselect("Select the columns for visualization:", options=dataframe.columns)
    if visualiz_columns:
      numeric_columns = dataframe[visualiz_columns].select_dtypes(include=['number']).columns
      st.subheader("Histograms")
      for column in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(data=dataframe, x=column, kde=True)
        plt.title(f"Histogram For numerical data - {column}")
        plt.xlabel(column)
        plt.ylabel("Frequency of values")
        plt.show()
        st.pyplot()
      st.subheader("boxplot")
      for column in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=dataframe, y=column)
        plt.title(f"Box Plot For numerical data - {column}")
        plt.ylabel(column)
        plt.show()
        st.pyplot()
      st.subheader("scatterplot")
      if len(numeric_columns) >= 2:
        for i in range(len(numeric_columns)):
          for j in range(i + 1, len(numeric_columns)):
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=dataframe, x=numeric_columns[i], y=numeric_columns[j])
            plt.title(f"Scatter Plot VS {numeric_columns[i]} and {numeric_columns[j]}")
            plt.xlabel(numeric_columns[i])
            plt.ylabel(numeric_columns[j])
            plt.show()
            st.pyplot()
      st.subheader("correlation_matrix")
      if len(numeric_columns) >= 2:
        correlation_matrix = dataframe[visualiz_columns].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Heatmap For correlation coefficients between numerical features")
        plt.show()
        st.pyplot()

def encode_categorical(dataframe):
    categorical_features = dataframe.select_dtypes(include=['object']).columns
    encoding_method = st.radio("Select encoding method for categorical data:", ("Label Encoding", "One-Hot Encoding"))
    if encoding_method == "Label Encoding":
        for col in categorical_features:
            dataframe[col] = LabelEncoder().fit_transform(dataframe[col])

def drop_duplicates(dataframe):
    st.header("select Drop duplicate rows?")
    drop_duplicates_option = st.checkbox("Drop duplicate", key="drop_duplicates_checkbox")
    
    if drop_duplicates_option:
        dataframe.drop_duplicates(inplace=True)
        
    return dataframe


def normalize(dataframe):
    numerical_features = dataframe.select_dtypes(include=['number']).columns
    normalization_method = st.radio("Select normalization method:", ("Min-Max Scaling", "Standard Scaling"))
    
    if normalization_method == "Min-Max Scaling":
        scaler = MinMaxScaler()
        dataframe[numerical_features] = scaler.fit_transform(dataframe[numerical_features])
    
    elif normalization_method == "Standard Scaling":
        scaler = StandardScaler()
        dataframe[numerical_features] = scaler.fit_transform(dataframe[numerical_features])
    
    return dataframe

def choose_variables(dataframe):
    st.header("Choose X and Y variables")
    X_variables = st.multiselect("Select independent variables (X):", options=dataframe.columns)
    Y_variable = st.selectbox("Select dependent variable (Y):", options=dataframe.columns)
    return X_variables, Y_variable

def main():
  with st.sidebar:
    st.header("The steps to prediction accuracy:-")
    st.text("1- upload csv file or excel file ")
    st.text("2- choose target feature")
    st.text("3- some perform eda")
    st.text("4- handle missing values")
    st.text("5- drop_duplicates")
    st.text("6- choose x and y ")
    st.text("7- encode categorical data ")
    st.text("8- data normalizetion ")
    st.text("9- using pycaret")

  dataframe = pd.DataFrame()
  target = ""

  
  data = st.file_uploader("upload the file:-", type=['csv', 'xslx','json'])
  if data is not None:
  
    if "csv" in data.name:
      dataframe = pd.read_csv(data)
    elif "json" in data.name:
      dataframe = pd.read_json(data)
    elif "excel" in data.name:
      dataframe = pd.read_excel(data)    
    else:
      raise ValueError("Unsupported file format")

    st.write(dataframe.head())
    
    target = st.selectbox("choose The target:-", dataframe.columns)
            
  if not dataframe.empty:

    perform_eda(dataframe)

    
    select_columns = st.multiselect("Select features to remove from the dataset:", options=dataframe.columns)
    if select_columns:
      dataframe.drop(select_columns, axis=1, inplace=True)
     

    numerical_features = dataframe.select_dtypes(['int64', 'float64']).columns
    categorical_feature = dataframe.select_dtypes(['object']).columns
    missing_value_num = st.radio("Set missing value for numerical value ", ["mean", "median"])
    missing_value_cat = st.radio("Set missing value for categorical value ", ['most frequent', "put additional class"])

    for col in numerical_features:
      dataframe[col] = SimpleImputer(strategy=missing_value_num, missing_values=np.nan).fit_transform(
              dataframe[col].values.reshape(-1, 1))
    for col in categorical_feature:
      if dataframe[col].nunique() > 7:
        dataframe[col] = SimpleImputer(strategy='most_frequent', missing_values=np.nan).fit_transform(
                    dataframe[col].values.reshape(-1, 1))
    else:
      dataframe[col] = LabelEncoder().fit_transform(dataframe[col])
    
    drop_duplicates(dataframe)
    
    X_variables, Y_variable = choose_variables(dataframe)

    encode_categorical(dataframe)
    
    #normalize(dataframe)
    
    if (len(numerical_features) != 0):
      st.header("Numerical Columns")
      st.write(numerical_features)
    if (len(categorical_feature) != 0):
      st.header("Categorical columns")
      st.write(categorical_feature)
    if (len(categorical_feature) != 0 or len(numerical_features) != 0):
      st.header("Number of null values")
      st.write(dataframe.isna().sum())

    
    if target and X_variables and Y_variable:
      
      option = "Regression" if dataframe[Y_variable].dtype in ['int64', 'float64'] else "Classification"
      st.header(f"Detected Task Type: {option}")

      if option == 'Regression':
        s = RegressionExperiment()
      elif option == 'Classification':
       s = ClassificationExperiment()

      s.setup(dataframe, target=Y_variable, session_id=123)
      best = s.compare_models()
      st.header("Best Algorithm")
      st.write(best)
      st.write(s.evaluate_model(best))
      st.header("20 rows of Prediction")
      predictions = s.predict_model(best, data=dataframe)
      st.write(predictions.head(20))

if __name__ == "__main__":
  main()