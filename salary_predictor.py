import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# ===== Load trained model =====
model = load_model(r'C:\Users\FINE LAPTOP\Desktop\AI Projects\Salary prediction\salary_model.keras')

# ===== Load encoders =====
with open(r'C:\Users\FINE LAPTOP\Desktop\AI Projects\Salary prediction\sex_encoder.pkl', 'rb') as f:
    sex_encoder = pickle.load(f)

with open(r'C:\Users\FINE LAPTOP\Desktop\AI Projects\Salary prediction\unit_encoder.pkl', 'rb') as f:
    unit_encoder = pickle.load(f)

with open(r'C:\Users\FINE LAPTOP\Desktop\AI Projects\Salary prediction\designation_encoder.pkl', 'rb') as f:
    designation_encoder = pickle.load(f)

# ===== Load scaler =====
with open(r'C:\Users\FINE LAPTOP\Desktop\AI Projects\Salary prediction\scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# ===== Streamlit UI =====
st.title("Real Salary Predictor ")
st.markdown("Enter employee details to predict expected salary:")

# ===== User Input =====
age = st.number_input("Age", min_value=18, max_value=70, step=1)
sex = st.selectbox("Gender", sex_encoder.classes_.tolist())
unit = st.selectbox("Unit", unit_encoder.classes_.tolist())
designation = st.selectbox("Designation", designation_encoder.classes_.tolist())
leaves_used = st.number_input("Leaves Used", min_value=0, max_value=60)
leaves_remaining = st.number_input("Leaves Remaining", min_value=0, max_value=60)
ratings = st.slider("Performance Rating", 1, 5)
past_exp = st.number_input("Past Experience (Years)", min_value=0, max_value=40)

# ===== Predict Button =====
if st.button("Predict Salary"):
    try:
        # Encode inputs
        sex_encoded = sex_encoder.transform([sex])[0]
        unit_encoded = unit_encoder.transform([unit])[0]
        designation_encoded = designation_encoder.transform([designation])[0]

        # Prepare input
        input_data = np.array([[age, sex_encoded, unit_encoded, leaves_used,
                                leaves_remaining, ratings, past_exp, designation_encoded]])

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)[0][0]

        # Show result
        st.success(f"Predicted Salary: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"Prediction failed: {e}")




