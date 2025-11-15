import streamlit as st
import pandas as pd
import joblib
from feature_eng import FeatureEngineer
st.title("📈 Demand Prediction App")

# Load the trained pipeline
pipeline = joblib.load('pipeline.pkl')

# Example: Input fields for user to enter feature values
col1, col2 = st.columns(2)

with col1:
    st.subheader("🛒 Product Details")
    store_id = st.selectbox("Store ID", options=['S001','S002','S003','S004','S005'])
    product_id = st.selectbox("Product ID", options=[
        'P0001','P0002','P0003','P0004','P0005',
        'P0006','P0007','P0008','P0009','P0010',
        'P0011','P0012','P0013','P0014','P0015',
        'P0016','P0017','P0018','P0019','P0020'
    ])
    category = st.selectbox("Category", options=['Electronics','Clothing','Groceries','Toys','Furniture'])
    price = st.number_input("Price", min_value=0.0, step=1.0)
    discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, step=1.0)
with col2:
    st.subheader("📦 Sales Data")
    promotion = st.number_input("Promotion (0/1)", min_value=0, max_value=1, step=1)
    competitor_pricing = st.number_input("Competitor Pricing", min_value=0.0, step=1.0)
    inventory_level = st.number_input("Inventory Level", min_value=0, step=1)
    units_ordered = st.number_input("Units Ordered", min_value=0, step=1)
    epidemic = st.number_input("Epidemic (0/1)", min_value=0, max_value=1, step=1)

# Another row of columns for environment details
col3, col4, col5 = st.columns(3)

with col3:
    region = st.selectbox("Region", options=['North','South','East','West'])

with col4:
    weather_condition = st.selectbox("Weather Condition", options=['Snowy','Cloudy','Sunny','Rainy'])

with col5:
    seasonality = st.selectbox("Seasonality", options=['Winter','Spring','Summer','Autumn'])

# Date input at the bottom
st.subheader("📅 Date of Sale")
date = st.date_input("Date")

# Predict Button
if st.button("🚀 Predict Demand"):
    input_data = pd.DataFrame({
        'Store ID': [store_id],
        'Product ID': [product_id],
        'Category': [category],
        'Region': [region],
        'Weather Condition': [weather_condition],
        'Seasonality': [seasonality],
        'Price': [price],
        'Units Sold': [155],  # Placeholder, adjust if needed
        'Discount': [discount],
        'Promotion': [promotion],
        'Competitor Pricing': [competitor_pricing],
        'Inventory Level': [inventory_level],
        'Units Ordered': [units_ordered],
        'Epidemic': [epidemic],
        'Date': pd.to_datetime([date])
    })
    
    # Use the pipeline to transform and predict
    
    prediction = pipeline.predict(input_data)
    # print(pipeline.named_steps['preprocessor'].get_feature_names_out())
    st.success(f"📊 Predicted Demand: {prediction[0]:.2f}")
if st.button("🔍 Show Features and Values"):
    # Prepare your input data (same as you do in prediction)
    input_data = pd.DataFrame({
        'Store ID': [store_id],
        'Product ID': [product_id],
        'Category': [category],
        'Region': [region],
        'Weather Condition': [weather_condition],
        'Seasonality': [seasonality],
        'Price': [price],
        'Units Sold':[155],  # Dummy value, adjust as needed
        'Discount': [discount],
        'Promotion': [promotion],
        'Competitor Pricing': [competitor_pricing],
        'Inventory Level': [inventory_level],
        'Units Ordered': [units_ordered],
        'Epidemic': [epidemic],
        'Date': pd.to_datetime([date])
    })
    
    # Transform the input data
    transformed_data = pipeline[:-1].transform(input_data)  # skip the model step for now

    # Get feature names
    try:
        # First try getting them from the preprocessor step
        feature_names = pipeline.named_steps['preprocessing'].get_feature_names_out()
    except AttributeError:
        # If preprocessor does not have get_feature_names_out, use columns from FeatureEngineer
        feature_names = pipeline.named_steps['feature_engineering'].columns_
    
    # Display feature names
    st.subheader("📝 Feature Names")
    st.write(list(feature_names))

    # Display transformed values
    st.subheader("📊 Transformed Values")
    if isinstance(transformed_data, pd.DataFrame):
        st.write(transformed_data)
    else:
        # If the transformer returns a NumPy array, convert it to DataFrame
        transformed_df = pd.DataFrame(transformed_data, columns=feature_names)
        st.write(transformed_df)
