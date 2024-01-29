import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import category_encoders as ce
import plotly.express as px

# Load your dataset
data = pd.read_excel(r"walmart3.xlsx")

# Data preprocessing
date_format = '%d/%m/%Y'
data['Order Date'] = pd.to_datetime(data['Order Date'], format=date_format)
data['Ship Date'] = pd.to_datetime(data['Ship Date'], format=date_format)

duplicate = data.duplicated()
data.drop_duplicates(inplace=True)

data['Sales'].fillna(value=data['Sales'].mean(), inplace=True)
data.sort_values(by='Order Date', inplace=True)

data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Day'] = data['Order Date'].dt.day


data.drop(['Order ID', 'Order Date', 'Ship Date', 'Customer Name', 'Profit'], axis=1, inplace=True)
data.drop(['Country', 'Category', 'State'], axis=1, inplace=True)

# Target encode 'City' and 'Product Name' columns
tar_encoders_city = ce.TargetEncoder(cols=['City'])
tar_encoders_product = ce.TargetEncoder(cols=['Product Name'])

# Create a separate DataFrame to store the encoded values
encoded_data = data.copy()
encoded_data['City'] = tar_encoders_city.fit_transform(data['City'], data['Sales'])
encoded_data['Product Name'] = tar_encoders_product.fit_transform(data['Product Name'], data['Sales'])

# Create reverse mapping dictionaries
city_reverse_mapping = dict(zip(encoded_data['City'], data['City']))
product_reverse_mapping = dict(zip(encoded_data['Product Name'], data['Product Name']))

# Model training
X = encoded_data.drop(['Sales'], axis=1)
Y = data['Sales']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=101)

scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

rf = RandomForestRegressor(n_estimators=1000, max_depth=15)
rf.fit(X_train_std, Y_train)
Y_pred_rf = rf.predict(X_test_std)

# Streamlit Dashboard
st.title("Walmart Sales Prediction Dashboard")

# Sidebar with user input
st.sidebar.header("User Input")

# Collecting user input for features
quantity = st.sidebar.text_input("Quantity", "")
city_options_encoded = list(encoded_data['City'].unique())
city_options_original = [city_reverse_mapping[val] for val in city_options_encoded]
city = st.sidebar.selectbox("Select City", city_options_original, index=0)  # Default to the first city

product_options_encoded = list(encoded_data['Product Name'].unique())
product_options_original = [product_reverse_mapping[val] for val in product_options_encoded]
product_name = st.sidebar.selectbox("Select Product Name", product_options_original, index=0)  # Default to the first product

year = st.sidebar.number_input("Year", min_value=int(data['Year'].min()), max_value=int(data['Year'].max()), value=int(data['Year'].mean()))
month = st.sidebar.number_input("Month", min_value=int(data['Month'].min()), max_value=int(data['Month'].max()), value=int(data['Month'].mean()))
day = st.sidebar.number_input("Day", min_value=int(data['Day'].min()), max_value=int(data['Day'].max()), value=int(data['Day'].mean()))

# Handling missing values in user input
if quantity == "":
    st.warning("Please provide a value for Quantity.")
    st.stop()
if city == "":
    st.warning("Please provide a value for City.")
    st.stop()

if product_name == "":
    st.warning("Please provide a value for Product Name.")
    st.stop()

# Encode user input
user_input_city = tar_encoders_city.transform(pd.DataFrame({'City': [city]}), pd.DataFrame({'Sales': [0]})).squeeze()
user_input_product = tar_encoders_product.transform(pd.DataFrame({'Product Name': [product_name]}), pd.DataFrame({'Sales': [0]})).squeeze()

# Creating a DataFrame with user input
user_input = pd.DataFrame({'City': [user_input_city], 'Product Name': [user_input_product], 'Quantity': [quantity], 'Year': [year], 'Month': [month], 'Day': [day]})

# Standardize user input
user_input_std = scaler.transform(user_input)

# Predicting Sales
predicted_sales = rf.predict(user_input_std)

# Display the prediction
st.subheader("Sales Prediction")
st.write("Predicted Sales:", predicted_sales[0])

# Reset button for user input
if st.sidebar.button("Reset"):
    st.caching.clear_cache()

# Displaying additional information (optional)
st.subheader("Additional Information")
st.write("Yearly Sales Visualization:")
# Add your visualization code here if needed


# Interactive Plot using Plotly
fig = px.bar(data, x='Year', y='Sales', title='Yearly Sales Over Time')
st.plotly_chart(fig)
city_contribution = data.groupby('City')['Sales'].sum() / data['Sales'].sum()
# Select top contributors (e.g., top 5)
top_contributors = city_contribution.nlargest(5).index
# Create a new column 'City Grouped' to include 'Others'
data['City Grouped'] = np.where(data['City'].isin(top_contributors), data['City'], 'Others')
fig_pie_city = px.pie(data, names='City Grouped', title='City Contributors to Sales Distribution')
st.plotly_chart(fig_pie_city)


fig_line_product = px.line(data[data['Product Name'] == product_name], x='Year', y='Sales', title=f'Sales Over Time for {product_name}')
st.plotly_chart(fig_line_product)
fig_line = px.line(data, x='Year', y='Sales', title='Yearly Sales Over Time')
st.plotly_chart(fig_line)


