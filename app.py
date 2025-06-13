import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = load_model('bigmart_best_model')

st.title("🛒 BigMart Sales Prediction App")
st.markdown("Predict sales using trained AutoML model from PyCaret.")

# Input form
with st.form(key='input_form'):
    Item_Identifier = st.text_input("Item Identifier", value="FDA15")
    Item_Weight = st.number_input("Item Weight (grams)", value=9.3)
    Item_Fat_Content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
    Item_Visibility = st.slider("Item Visibility", min_value=0.0, max_value=0.3, value=0.05)
    Item_Type = st.selectbox("Item Type", [
        'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
        'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene',
        'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']
    )
    Item_MRP = st.number_input("Item MRP", value=249.8)
    Outlet_Identifier = st.selectbox("Outlet Identifier", [
        'OUT049', 'OUT018', 'OUT010', 'OUT013', 'OUT027',
        'OUT045', 'OUT017', 'OUT046', 'OUT035']
    )
    Outlet_Establishment_Year = st.selectbox("Outlet Establishment Year", [1985, 1987, 1997, 1999, 2002, 2004, 2007, 2009])
    Outlet_Size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
    Outlet_Location_Type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    Outlet_Type = st.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

    submit = st.form_submit_button(label='Predict Sales')

if submit:
    input_dict = {
        'Item_Identifier': Item_Identifier,
        'Item_Weight': Item_Weight,
        'Item_Fat_Content': Item_Fat_Content,
        'Item_Visibility': Item_Visibility,
        'Item_Type': Item_Type,
        'Item_MRP': Item_MRP,
        'Outlet_Identifier': Outlet_Identifier,
        'Outlet_Establishment_Year': Outlet_Establishment_Year,
        'Outlet_Size': Outlet_Size,
        'Outlet_Location_Type': Outlet_Location_Type,
        'Outlet_Type': Outlet_Type,
    }

    input_df = pd.DataFrame([input_dict])

    

    # Replace fat content inconsistencies
    input_df['Item_Fat_Content'] = input_df['Item_Fat_Content'].replace({
        'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'
    })

    # Fill missing Item_Weight by group mean (simulate training)
    # Normally Item_Type group mean; here just fallback
    if pd.isnull(input_df['Item_Weight']).any():
        input_df['Item_Weight'].fillna(12.5, inplace=True)  # replace with global mean if needed

    # Fill missing Outlet_Size with mode from training group
    if pd.isnull(input_df['Outlet_Size']).any():
        input_df['Outlet_Size'].fillna('Small', inplace=True)

    # Label Encoding
    categorical_cols = ['Item_Fat_Content', 'Item_Type', 'Outlet_Identifier',
                        'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type']

    le = LabelEncoder()
    for col in categorical_cols:
        input_df[col] = le.fit_transform(input_df[col])

    # Predict using PyCaret
    result = predict_model(model, data=input_df)

    st.write("🔍 Prediction Output:", result)

# Safe handling for prediction column
    predicted_column = 'Label' if 'Label' in result.columns else result.columns[-1]

    predicted_value = result[predicted_column].iloc[0]

    st.subheader("📈 Predicted Item Outlet Sales:")
    st.success(f"₹ {round(predicted_value, 2)}")
