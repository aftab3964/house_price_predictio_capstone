import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
st.sidebar.title("🏠 House Price Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Predict Price",
        "Model Performance",
        "About"
    ]
)

# ---------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------
if page == "Home":

    st.title("🏠 House Price Prediction using Machine Learning")

    st.image(
        "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=1200",
        use_container_width=True
    )

    st.markdown("---")

    st.header("📌 Project Overview")

    st.write("""
This project predicts residential house prices using Machine Learning.

The dataset contains information such as:

- Bedrooms
- Bathrooms
- Living Area
- Lot Size
- Floors
- Year Built
- City
- State ZIP

Three regression models were developed:

- Linear Regression
- Decision Tree
- Random Forest

Among these, **Linear Regression** achieved the highest prediction accuracy after preprocessing and outlier removal.
""")

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Size", "4600")
    col2.metric("Features", "18")
    col3.metric("Best R²", "0.7766")

    st.markdown("---")

    st.header("🛠 Technologies Used")

    st.write("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- Streamlit
""")

    st.markdown("---")

    st.success("Use the sidebar to navigate through the application.")

# ---------------------------------------------------------
# OTHER PAGES (Coming Next)
# ---------------------------------------------------------

elif page == "Predict Price":

    import joblib
    import pandas as pd
    import numpy as np

    st.title("🏡 Predict House Price")

    # Load saved files
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    st.write("Enter the property details below:")

    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Bathrooms", 1.0, 10.0, 2.0)
        sqft_living = st.number_input("Living Area (sqft)", 500, 10000, 2000)
        sqft_lot = st.number_input("Lot Size (sqft)", 500, 1000000, 5000)
        floors = st.number_input("Floors", 1.0, 4.0, 2.0)
        waterfront = st.selectbox("Waterfront", [0, 1])

    with col2:
        view = st.slider("View", 0, 4, 0)
        condition = st.slider("Condition", 1, 5, 3)
        sqft_above = st.number_input("Above Ground Area", 500, 10000, 1800)
        sqft_basement = st.number_input("Basement Area", 0, 5000, 200)
        yr_built = st.number_input("Year Built", 1900, 2025, 1995)

    city = st.selectbox(
        "City",
        [
            "Seattle",
            "Bellevue",
            "Redmond",
            "Kent",
            "Kirkland",
            "Mercer Island",
            "Bothell",
            "Sammamish",
            "Issaquah",
            "Renton"
        ]
    )

    statezip = st.selectbox(
        "State ZIP",
        [
            "WA 98004",
            "WA 98005",
            "WA 98006",
            "WA 98007",
            "WA 98008",
            "WA 98033",
            "WA 98052",
            "WA 98053",
            "WA 98074",
            "WA 98103",
            "WA 98105"
        ]
    )

    if st.button("Predict House Price"):

        # Create dictionary with numeric inputs
        input_dict = {
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "sqft_living": sqft_living,
            "sqft_lot": sqft_lot,
            "floors": floors,
            "waterfront": waterfront,
            "view": view,
            "condition": condition,
            "sqft_above": sqft_above,
            "sqft_basement": sqft_basement,
            "yr_built": yr_built,
        }

        # Create dataframe
        input_df = pd.DataFrame([input_dict])

        # Add all feature columns
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Activate selected city
        city_col = f"city_{city}"
        if city_col in input_df.columns:
            input_df[city_col] = 1

        # Activate selected statezip
        zip_col = f"statezip_{statezip}"
        if zip_col in input_df.columns:
            input_df[zip_col] = 1

        # Arrange columns
        input_df = input_df[feature_columns]

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]

        st.success(f"🏠 Predicted House Price: ${prediction:,.2f}")

elif page == "Model Performance":

    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    st.title("📊 Model Performance")

    st.markdown("### Regression Model Comparison")

    # -------------------------------
    # Model Comparison Table
    # -------------------------------
    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Random Forest",
            "Decision Tree"
        ],
        "MAE": [
            119106.83,
            76718.22,
            103688.23
        ],
        "RMSE": [
            213066.74,
            108988.69,
            139060.89
        ],
        "R² Score": [
            0.7766,
            0.7169,
            0.5391
        ]
    })

    st.dataframe(results, use_container_width=True)

    st.markdown("---")

    # -------------------------------
    # R² Comparison Chart
    # -------------------------------
    st.subheader("Model Comparison (R² Score)")

    fig, ax = plt.subplots(figsize=(7,4))

    sns.barplot(
        data=results,
        x="Model",
        y="R² Score",
        ax=ax
    )

    ax.set_ylim(0,1)
    ax.set_ylabel("R² Score")

    st.pyplot(fig)

    st.markdown("---")

    # -------------------------------
    # Best Model
    # -------------------------------
    st.subheader("Best Performing Model")

    st.success("""
    **Linear Regression** achieved the highest R² score (**0.7766**) after
    data preprocessing and outlier removal. It was selected as the final
    model for this project.
    """)

    st.markdown("---")

    # -------------------------------
    # Project Highlights
    # -------------------------------
    st.subheader("Project Highlights")

    st.write("""
    ✅ 4600 housing records analyzed

    ✅ Data Cleaning & Preprocessing

    ✅ Outlier Removal using IQR

    ✅ Feature Engineering

    ✅ Three Machine Learning Models Developed

    ✅ Best R² Score: **0.7766**

    ✅ Most Important Feature:
    **sqft_living**
    """)

elif page == "About":

    st.title("👨‍💻 About the Project")

    st.markdown("---")

    st.header("🏠 House Price Prediction using Machine Learning")

    st.write("""
This application was developed as part of the **Masai School Data Science Capstone Project**.

The objective of this project is to predict residential house prices using machine learning techniques based on structural and location-related features such as living area, number of bedrooms, bathrooms, property condition, construction year, and geographical location.

The project follows a complete machine learning pipeline, including data preprocessing, exploratory data analysis (EDA), feature engineering, model development, evaluation, and interpretation.
""")

    st.markdown("---")

    st.header("📂 Dataset")

    st.write("""
- **Dataset Size:** 4600 records
- **Features:** 18
- **Target Variable:** House Price

The dataset contains residential property information such as:

- Bedrooms
- Bathrooms
- Living Area
- Lot Size
- Floors
- Waterfront
- View
- Condition
- Above Ground Area
- Basement Area
- Year Built
- City
- State ZIP
""")

    st.markdown("---")

    st.header("⚙️ Machine Learning Workflow")

    st.write("""
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Outlier Removal using IQR
5. Feature Engineering
6. Data Encoding
7. Feature Scaling
8. Model Training
9. Model Evaluation
10. Feature Importance Analysis
""")

    st.markdown("---")

    st.header("🛠 Technologies Used")

    st.write("""
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib
""")

    st.markdown("---")

    st.header("📊 Model Summary")

    st.success("""
**Best Model:** Linear Regression

**R² Score:** 0.7766

The model demonstrates good predictive performance after preprocessing and outlier removal.
""")

    st.markdown("---")

    st.header("👤 Author")

    st.write("""
**Aftab Ahmad**

Researcher | Data Science Enthusiast | Structural Engineering

This project demonstrates the application of machine learning techniques for solving a real-world regression problem and serves as a portfolio project showcasing data preprocessing, model development, evaluation, and deployment using Streamlit.
""")

    st.markdown("---")

    st.header("🔗 Connect")

    st.markdown("""
- **GitHub:** https://github.com/aftab3964
- **LinkedIn:** https://linkedin.com/in/ahmad-aftab-ba020925a
- **Email:** xaftabahmad@gmail.com
""")

    st.markdown("---")

    st.info("Thank you for exploring this project!")