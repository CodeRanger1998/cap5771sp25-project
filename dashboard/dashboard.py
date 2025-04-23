# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time # To add a small delay for user experience

# --- Page Configuration ---
st.set_page_config(
    page_title="Movie Success Predictor",
    page_icon="🎬",
    layout="wide"
)

# --- Load Model and Scaler ---
try:
    model = joblib.load('random_forest_model.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("Error: Model ('random_forest_model.joblib') or Scaler ('scaler.joblib') file not found.")
    st.stop() # Stop execution if files are missing
except Exception as e:
    st.error(f"An error occurred loading model/scaler: {e}")
    st.stop()

# --- Load Data (Optional, for context/exploration) ---
try:
    # Using the cleaned data file name mentioned in reports
    df_cleaned = pd.read_csv('cleaned_rotten_movies.csv')
except FileNotFoundError:
    st.warning("Warning: 'cleaned_rotten_movies.csv' not found. Data exploration features disabled.")
    df_cleaned = None
except Exception as e:
    st.error(f"An error occurred loading the CSV data: {e}")
    df_cleaned = None


# --- Define Features and Target Mapping ---
# IMPORTANT: List features in the exact order the model expects them!
# Based on your Milestone 2 report/output:
expected_features = [
    'tomatometer_rating',
    'runtime_rating_interaction',
    'tomatometer_fresh_critics_count',
    'tomatometer_count',
    'tomatometer_rotten_critics_count',
    'audience_rating',
    'tomatometer_top_critics_count',
    'runtime',
    'movie_age',
    'audience_count'
]

# Mapping from numerical prediction back to labels (Based on Milestone 2 Report)
# Check your notebook's LabelEncoder output for exact mapping if different
target_mapping = {0: 'Certified-Fresh', 1: 'Fresh', 2: 'Rotten'}

# --- Dashboard Title ---
st.title("🎬 Movie Tomatometer Status Predictor")
st.markdown("Predict whether a movie will be 'Rotten', 'Fresh', or 'Certified-Fresh' based on its features.")

# --- Sidebar for Inputs ---
st.sidebar.header("Input Movie Features:")

input_data = {}

# Create input fields for each feature
# Using sliders for numerical inputs - adjust min/max/value based on your data knowledge
# You might want to derive realistic ranges from df_cleaned.describe() if loaded
input_data['tomatometer_rating'] = st.sidebar.slider("Tomatometer Rating (%)", 0, 100, 70)
# Note: runtime_rating_interaction is engineered, maybe calculate it based on runtime and rating inputs?
# Or ask for it directly if you expect the user to know it?
# Let's calculate it for simplicity:
input_data['runtime'] = st.sidebar.slider("Runtime (minutes)", 30, 240, 110)
input_data['runtime_rating_interaction'] = input_data['runtime'] * input_data['tomatometer_rating']
st.sidebar.text(f"Calculated Runtime*Rating: {input_data['runtime_rating_interaction']:.0f}")

input_data['tomatometer_fresh_critics_count'] = st.sidebar.number_input("Fresh Critic Reviews Count", min_value=0, value=50, step=1)
input_data['tomatometer_count'] = st.sidebar.number_input("Total Critic Reviews Count", min_value=0, value=80, step=1)
input_data['tomatometer_rotten_critics_count'] = st.sidebar.number_input("Rotten Critic Reviews Count", min_value=0, value=10, step=1)
input_data['audience_rating'] = st.sidebar.slider("Audience Rating (%)", 0, 100, 65)
input_data['tomatometer_top_critics_count'] = st.sidebar.number_input("Top Critic Reviews Count", min_value=0, value=20, step=1)
input_data['movie_age'] = st.sidebar.slider("Movie Age (Years)", 0, 100, 5)
input_data['audience_count'] = st.sidebar.number_input("Audience Reviews Count", min_value=0, value=50000, step=1000)


# --- Prediction Button and Output ---
if st.sidebar.button("Predict Status"):
    # 1. Create DataFrame from inputs in the correct order
    input_df = pd.DataFrame([input_data])
    # Reorder columns to match the training order
    try:
        input_df = input_df[expected_features]
    except KeyError as e:
        st.error(f"Input data is missing expected feature columns: {e}. Check 'expected_features' list.")
        st.stop()

    # 2. Scale the input data using the loaded scaler
    try:
        input_scaled = input_df
    except ValueError as e:
        st.error(f"Error scaling input data: {e}. Ensure input values are valid numbers.")
        st.stop()
    except Exception as e:
         st.error(f"An unexpected error occurred during scaling: {e}")
         st.stop()


    # 3. Make prediction
    try:
        prediction_num = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.stop()


    # 4. Map prediction to label
    predicted_label = target_mapping.get(prediction_num, "Unknown Prediction")

    # 5. Display results
    st.subheader("Prediction Result")
    with st.spinner('Generating prediction...'):
        time.sleep(1) # Simulate calculation
        if predicted_label == 'Certified-Fresh':
            st.success(f"Predicted Status: **{predicted_label}** 🎉")
        elif predicted_label == 'Fresh':
            st.info(f"Predicted Status: **{predicted_label}** 👍")
        else: # Rotten
            st.warning(f"Predicted Status: **{predicted_label}** 🍅")

        st.write("Prediction Confidence:")
        prob_df = pd.DataFrame({
            'Status': [target_mapping.get(i, f"Class {i}") for i in model.classes_],
            'Probability': probabilities
        })
        st.dataframe(prob_df.sort_values(by='Probability', ascending=False))

# --- Optional: Display Model Info/Data ---
st.divider() # Visual separator

# Tabs for optional sections
tab1, tab2, tab3 = st.tabs(["Data Sample", "Feature Importance", "Model Performance"])

with tab1:
    st.header("Cleaned Data Sample")
    if df_cleaned is not None:
        st.dataframe(df_cleaned.head(10))
    else:
        st.info("Cleaned data file was not loaded.")

with tab2:
    st.header("Top 10 Features Used by Model")
    st.markdown("Based on Random Forest Feature Importance:")
    # Manually list the features based on your report
    st.code("""
1. tomatometer_rating
2. runtime_rating_interaction
3. tomatometer_fresh_critics_count
4. tomatometer_count
5. tomatometer_rotten_critics_count
6. audience_rating
7. tomatometer_top_critics_count
8. runtime
9. movie_age
10. audience_count
    """)
    # Optionally, load feature importance scores and create a bar chart if you saved them

with tab3:
    st.header("Model Performance (on Test Set)")
    st.metric("Accuracy", "0.9845") # From your report
    st.metric("ROC AUC (weighted OvR)", "0.9987") # From your report
    st.metric("F1-Score (Weighted)", "0.9846") # From your report

    st.subheader("Test Set Confusion Matrix")
    try:
        st.image('confusion_matrix_test_set.png', caption='Confusion Matrix for Random Forest on Test Set')
    except FileNotFoundError:
        st.warning("Warning: 'confusion_matrix_test_set.png' not found.")
    except Exception as e:
        st.error(f"Could not load confusion matrix image: {e}")


st.sidebar.divider()
st.sidebar.info("Dashboard developed based on Milestone 2 results.")