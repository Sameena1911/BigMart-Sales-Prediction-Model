import streamlit as st
import pandas as pd
import numpy as np
from pycaret.regression import load_model, predict_model
from sklearn.preprocessing import LabelEncoder

# Load trained model
model = load_model('bigmart_best_model')

st.title("🛒 BigMart Sales Prediction App")
st.markdown("Predict sales using trained AutoML model from PyCaret.")

with st.form(key='input_form'):
    Item_Weight = st.number_input("Item Weight (grams)", value=12.5)
    Item_Fat_Content = st.selectbox("Item Fat Content", ['Low Fat', 'Regular'])
    
    Item_Type = st.selectbox("Item Type", [
        'Dairy', 'Soft Drinks', 'Meat', 'Fruits and Vegetables', 'Household',
        'Baking Goods', 'Snack Foods', 'Frozen Foods', 'Breakfast', 'Health and Hygiene',
        'Hard Drinks', 'Canned', 'Breads', 'Starchy Foods', 'Others', 'Seafood']
    )
    Item_MRP = st.number_input("Item MRP", value=249.8)
    Outlet_Establishment_Year = st.selectbox("Outlet Establishment Year", [1985, 1987, 1997, 1999, 2002, 2004, 2007, 2009])
    Outlet_Size = st.selectbox("Outlet Size", ['Small', 'Medium', 'High'])
    Outlet_Location_Type = st.selectbox("Outlet Location Type", ['Tier 1', 'Tier 2', 'Tier 3'])
    Outlet_Type = st.selectbox("Outlet Type", ['Supermarket Type1', 'Supermarket Type2', 'Supermarket Type3', 'Grocery Store'])

    submit = st.form_submit_button(label='Predict Sales')

if submit:
    input_dict = {
        'Item_Weight': Item_Weight,
        'Item_Fat_Content': Item_Fat_Content,
        
        'Item_Type': Item_Type,
        'Item_MRP': Item_MRP,
        'Outlet_Establishment_Year': Outlet_Establishment_Year,
        'Outlet_Size': Outlet_Size,
        'Outlet_Location_Type': Outlet_Location_Type,
        'Outlet_Type': Outlet_Type
    }

    input_df = pd.DataFrame([input_dict])

    # Label Encoding (same encoding done in training)
    encoders = {
        'Item_Fat_Content': {'Low Fat': 0, 'Regular': 1},
        'Item_Type': {
            'Baking Goods': 0, 'Breads': 1, 'Breakfast': 2, 'Canned': 3, 'Dairy': 4,
            'Frozen Foods': 5, 'Fruits and Vegetables': 6, 'Hard Drinks': 7,
            'Health and Hygiene': 8, 'Household': 9, 'Meat': 10, 'Others': 11,
            'Seafood': 12, 'Snack Foods': 13, 'Soft Drinks': 14, 'Starchy Foods': 15
        },
        'Outlet_Size': {'High': 0, 'Medium': 1, 'Small': 2},
        'Outlet_Location_Type': {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2},
        'Outlet_Type': {'Grocery Store': 0, 'Supermarket Type1': 1, 'Supermarket Type2': 2, 'Supermarket Type3': 3}
    }

    for col, mapping in encoders.items():
        input_df[col] = input_df[col].map(mapping)

    result = predict_model(model, data=input_df)

    predicted_column = 'Label' if 'Label' in result.columns else result.columns[-1]
    predicted_value = result[predicted_column].iloc[0]

    st.subheader("📈 Predicted Item Outlet Sales:")
    st.success(f"₹ {round(predicted_value, 2)}")
