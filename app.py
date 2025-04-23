# ====== SETUP (MUST BE FIRST STREAMLIT COMMAND) ======
import streamlit as st
st.set_page_config(
    page_title="✨ Diabetes AI Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====== IMPORTS ======
import pandas as pd
import numpy as np
import joblib
from streamlit_lottie import st_lottie
import requests
import time
import plotly.express as px
import plotly.graph_objects as go
import base64

# ====== BACKGROUND IMAGE ======
def get_base64_of_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Replace with your actual image path
background_image = "background.png"  # Save your image with this name
background_base64 = get_base64_of_image(background_image)

# ====== LOAD MODEL AND SCALER ======
@st.cache_resource
def load_models():
    try:
        model = joblib.load("diabetes_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model files: {str(e)}")
        st.stop()

model, scaler = load_models()

# ====== CUSTOM CSS ======
def inject_custom_css():
    st.markdown(f"""
    <style>
    :root {{
        --primary: #00d2ff;
        --secondary: #3a7bd5;
        --dark: #0f172a;
        --light: #f8fafc;
        --danger: #ff4b4b;
        --success: #2ecc71;
    }}
    
    .stApp {{
        background-image: url("data:image/png;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background-blend-mode: overlay;
        background-color: rgba(15, 23, 42, 0.85);
    }}
    
    .hero-section {{
        text-align: center;
        padding: 2rem 0;
        margin-bottom: 2rem;
        animation: fadeIn 1s ease-out;
        background-color: rgba(15, 23, 42, 0.7);
        border-radius: 15px;
        backdrop-filter: blur(5px);
        margin-top: 1rem;
    }}
    
    .gradient-text {{
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent;
        animation: gradient 8s ease infinite;
    }}
    
    @keyframes gradient {{
        0% {{ background-position: 0% 50%; }}
        50% {{ background-position: 100% 50%; }}
        100% {{ background-position: 0% 50%; }}
    }}
    
    .glass-panel {{
        background: rgba(15, 23, 42, 0.7);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: all 0.3s ease;
        margin-bottom: 1rem;
    }}
    
    .glass-panel:hover {{
        transform: translateY(-5px);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.3);
    }}
    
    .result-card {{
        background: rgba(15, 23, 42, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem 0;
        animation: fadeIn 0.8s ease;
    }}
    
    .high-risk {{
        border-left: 5px solid var(--danger);
    }}
    
    .low-risk {{
        border-left: 5px solid var(--success);
    }}
    
    .stButton>button {{
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }}
    
    .stButton>button:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.5);
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    
    .footer {{
        text-align: center;
        margin-top: 3rem;
        opacity: 0.7;
        font-size: 0.9rem;
        background-color: rgba(15, 23, 42, 0.7);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(5px);
    }}
    
    .metric-card {{
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }}
    
    /* Theme toggle button styling */
    .theme-toggle-container {{
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 1000;
    }}
    
    .theme-toggle-btn {{
        background: rgba(15, 23, 42, 0.7) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 50% !important;
        width: 40px !important;
        height: 40px !important;
        padding: 0 !important;
        display: flex;
        align-items: center;
        justify-content: center;
        backdrop-filter: blur(5px);
    }}
    
    /* Make sure content is readable over background */
    .stMarkdown, .stSlider, .stSelectbox, .stTextInput, .stNumberInput {{
        color: var(--light) !important;
    }}
    
    /* Plotly chart background */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly .main-svg {{
        background-color: transparent !important;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_custom_css()

# ====== THEME TOGGLE ======
def toggle_theme():
    current_theme = st.session_state.get("theme", "dark")
    new_theme = "light" if current_theme == "dark" else "dark"
    st.session_state.theme = new_theme
    # Update CSS variables based on theme
    if new_theme == "light":
        st.markdown("""
        <style>
        .stApp {
            background-color: rgba(248, 250, 252, 0.85);
        }
        .glass-panel, .result-card, .hero-section, .footer {
            background-color: rgba(248, 250, 252, 0.7) !important;
            color: #0f172a !important;
        }
        .metric-card {
            background: rgba(0, 0, 0, 0.05) !important;
        }
        .stMarkdown, .stSlider, .stSelectbox, .stTextInput, .stNumberInput {
            color: #0f172a !important;
        }
        </style>
        """, unsafe_allow_html=True)
    else:
        inject_custom_css()  # Re-inject dark mode styles

# ====== LOTTIE ANIMATIONS ======
def load_lottie(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
        return None
    except:
        return None

lottie_healthy = load_lottie("https://assets1.lottiefiles.com/packages/lf20_5njp3vgg.json")
lottie_diabetic = load_lottie("https://assets4.lottiefiles.com/packages/lf20_7skp4qvy.json")
lottie_doctor = load_lottie("https://assets4.lottiefiles.com/packages/lf20_yyja8qra.json")

# ====== SIDEBAR ======
with st.sidebar:
    # Theme toggle button at the top right
    with st.container():
        cols = st.columns([1, 1])
        with cols[1]:
            if st.button("🌙", key="theme_toggle", help="Toggle dark/light mode", on_click=toggle_theme):
                pass
    
    st.markdown("""
    <div class="glass-panel">
        <h2>👨‍⚕️ Patient Health Metrics</h2>
        <p class="subtext">Adjust sliders & predict diabetes risk</p>
    """, unsafe_allow_html=True)
    
    if lottie_doctor:
        st_lottie(lottie_doctor, height=180, key="sidebar_anim")

    with st.form("input_form"):
        pregnancies = st.slider("🤰 Pregnancies", 0, 20, 1)
        glucose = st.slider("🩸 Glucose (mg/dL)", 50, 300, 120)
        bp = st.slider("💓 BP (mm Hg)", 40, 130, 70)
        skin = st.slider("🖐️ Skin Thickness (mm)", 0, 100, 20)
        insulin = st.slider("💉 Insulin (μU/mL)", 0, 300, 80)
        bmi = st.slider("⚖️ BMI", 10.0, 60.0, 25.0)
        dpf = st.slider("🧬 DPF", 0.1, 2.5, 0.5)
        age = st.slider("👴 Age", 10, 100, 33)
        
        submitted = st.form_submit_button("🔮 Predict Risk", use_container_width=True)

# ====== MAIN UI ======
st.markdown("""
<div class="hero-section">
    <h1>AI-Powered <span class="gradient-text">Diabetes</span> Predictor</h1>
    <p class="hero-subtitle">Advanced machine learning for early health insights</p>
</div>
""", unsafe_allow_html=True)

# ====== PREDICTION LOGIC ======
if submitted:
    with st.spinner("🚀 Analyzing health data..."):
        time.sleep(1.5)  # Simulate processing
        
        # Create input dataframe
        input_df = pd.DataFrame({
            "Pregnancies": [pregnancies],
            "Glucose": [glucose],
            "BloodPressure": [bp],
            "SkinThickness": [skin],
            "Insulin": [insulin],
            "BMI": [bmi],
            "DiabetesPedigreeFunction": [dpf],
            "Age": [age]
        })
        
        # Scale input features
        try:
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)
            prediction_proba = model.predict_proba(scaled_input)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
            st.stop()

    # ====== RESULTS DISPLAY ======
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### 📊 Risk Probability")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_proba[0][1] * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diabetes Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#00d2ff"},
                'steps': [
                    {'range': [0, 30], 'color': "#2ecc71"},
                    {'range': [30, 70], 'color': "#f39c12"},
                    {'range': [70, 100], 'color': "#e74c3c"}
                ],
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Key metrics display
        st.markdown("### 📈 Key Metrics")
        st.markdown(f"""
        <div class="metric-card">
            <p>Glucose: <strong>{glucose} mg/dL</strong></p>
        </div>
        <div class="metric-card">
            <p>BMI: <strong>{bmi:.1f}</strong></p>
        </div>
        <div class="metric-card">
            <p>Age: <strong>{age} years</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        risk_class = "high-risk" if prediction[0] == 1 else "low-risk"
        risk_message = "⚠️ High Risk Detected" if prediction[0] == 1 else "✅ Low Risk Detected"
        recommendation = "🩺 Consult a doctor immediately" if prediction[0] == 1 else "💪 Maintain healthy habits!"
        
        st.markdown(f"""
        <div class="result-card {risk_class}">
            <h2>{risk_message}</h2>
            <p>AI prediction confidence: <strong>{prediction_proba[0][1]*100:.1f}%</strong></p>
        """, unsafe_allow_html=True)
        
        if prediction[0] == 1 and lottie_diabetic:
            st_lottie(lottie_diabetic, height=200)
        elif prediction[0] == 0 and lottie_healthy:
            st_lottie(lottie_healthy, height=200)
            
        st.markdown(f"""
            <div class="metric-card">
                <p><strong>Recommendation:</strong> {recommendation}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

