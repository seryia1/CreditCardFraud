import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import os
from streamlit_option_menu import option_menu

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .prediction-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .fraud-alert {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        animation: pulse 2s infinite;
    }
    
    .safe-alert {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .feedback-container {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'feedback_data' not in st.session_state:
        st.session_state.feedback_data = []
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []

# Load model and column ranges
@st.cache_resource
def load_model_and_ranges():
    try:
        model = joblib.load('stacked_model.joblib')
        column_ranges = joblib.load('column_ranges.joblib')
        return model, column_ranges
    except FileNotFoundError:
        st.error("Model files not found. Please ensure 'fraud_model.joblib' and 'column_ranges.joblib' are in the project directory.")
        return None, None

# Authentication
def authenticate(username, password):
    return username == "admin" and password == "admin"

# Save feedback
def save_feedback(prediction, confidence, user_feedback, actual_result):
    feedback_entry = {
        'timestamp': datetime.now().isoformat(),
        'prediction': prediction,
        'confidence': confidence,
        'user_feedback': user_feedback,
        'actual_result': actual_result
    }
    st.session_state.feedback_data.append(feedback_entry)
    
    # Save to file
    try:
        with open('feedback_data.json', 'w') as f:
            json.dump(st.session_state.feedback_data, f)
    except:
        pass

# Load feedback data
def load_feedback_data():
    try:
        with open('feedback_data.json', 'r') as f:
            st.session_state.feedback_data = json.load(f)
    except FileNotFoundError:
        st.session_state.feedback_data = []

def main():
    load_css()
    init_session_state()
    load_feedback_data()
    
    # Navigation
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=300&h=200&fit=crop", 
                 caption="Credit Card Security", use_column_width=True)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Home", "🔍 Fraud Detection", "👨‍💼 Admin Dashboard"],
            icons=["house", "search", "person-gear"],
            menu_icon="cast",
            default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "orange", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "text-align": "left", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#02ab21"},
            }
        )
    
    if selected == "🏠 Home":
        show_home_page()
    elif selected == "🔍 Fraud Detection":
        show_prediction_page()
    elif selected == "👨‍💼 Admin Dashboard":
        show_admin_dashboard()

def show_home_page():
    st.markdown('<h1 class="main-header">💳 Credit Card Fraud Detection System</h1>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.image("https://images.unsplash.com/photo-1563013544-824ae1b704d3?w=600&h=400&fit=crop", 
                 caption="Advanced AI-Powered Fraud Detection")
    
    st.markdown("""
    <div class="prediction-container">
        <h2 style="color: white; text-align: center;">🛡️ Protect Your Financial Future</h2>
        <p style="color: white; text-align: center; font-size: 1.2rem;">
            Our advanced machine learning system analyzes transaction patterns using Principal Component Analysis (PCA) 
            to detect fraudulent activities with high accuracy.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p>Advanced ML algorithms trained on extensive datasets for precise fraud detection.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Analysis</h3>
            <p>Instant transaction analysis with immediate fraud risk assessment.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🔒 Secure Processing</h3>
            <p>All data is processed securely with privacy protection measures.</p>
        </div>
        """, unsafe_allow_html=True)

def show_prediction_page():
    st.markdown('<h1 class="main-header">🔍 Fraud Detection Analysis</h1>', unsafe_allow_html=True)
    
    model, column_ranges = load_model_and_ranges()
    
    if model is None or column_ranges is None:
        st.error("Unable to load model. Please check if model files exist.")
        return
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">Enter Transaction Features (PCA Components)</h3>
        <p style="color: white; margin: 0;">Adjust the sliders below to input the PCA-transformed features of the transaction.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create input sliders
    col1, col2 = st.columns(2)
    features = {}
    
    num_features = len(column_ranges)
    half_features = num_features // 2
    
    with col1:
        for i in range(half_features):
            feature_name = f'PC{i+1}'
            min_val = float(column_ranges[feature_name]['min'])
            max_val = float(column_ranges[feature_name]['max'])
            features[feature_name] = st.slider(
                f'{feature_name}',
                min_value=min_val,
                max_value=max_val,
                value=(min_val + max_val) / 2,
                step=(max_val - min_val) / 100
            )
    
    with col2:
        for i in range(half_features, num_features):
            feature_name = f'PC{i+1}'
            min_val = float(column_ranges[feature_name]['min'])
            max_val = float(column_ranges[feature_name]['max'])
            features[feature_name] = st.slider(
                f'{feature_name}',
                min_value=min_val,
                max_value=max_val,
                value=(min_val + max_val) / 2,
                step=(max_val - min_val) / 100
            )
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            # Make prediction
            feature_array = np.array([list(features.values())])
            prediction = model.predict(feature_array)[0]
            prediction_proba = model.predict_proba(feature_array)[0]
            
            confidence = max(prediction_proba) * 100
            
            # Store prediction
            prediction_data = {
                'timestamp': datetime.now().isoformat(),
                'features': features,
                'prediction': int(prediction),
                'confidence': confidence
            }
            st.session_state.prediction_history.append(prediction_data)
            
            # Display result
            if prediction == 1:  # Fraud
                st.markdown(f"""
                <div class="fraud-alert">
                    🚨 FRAUD DETECTED 🚨<br>
                    Confidence: {confidence:.2f}%<br>
                    This transaction shows high risk patterns!
                </div>
                """, unsafe_allow_html=True)
            else:  # Normal
                st.markdown(f"""
                <div class="safe-alert">
                    ✅ TRANSACTION SAFE ✅<br>
                    Confidence: {confidence:.2f}%<br>
                    This transaction appears legitimate.
                </div>
                """, unsafe_allow_html=True)
            
            # Confidence visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Prediction Confidence"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red" if prediction == 1 else "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Feedback section
            st.markdown("""
            <div class="feedback-container">
                <h3>📝 Provide Feedback</h3>
                <p>Help us improve our model by providing feedback on this prediction.</p>
            </div>
            """, unsafe_allow_html=True)
            
            feedback_col1, feedback_col2 = st.columns(2)
            
            with feedback_col1:
                user_feedback = st.radio(
                    "Was this prediction accurate?",
                    ["Correct", "Incorrect"],
                    key=f"feedback_{len(st.session_state.prediction_history)}"
                )
            
            with feedback_col2:
                actual_result = st.selectbox(
                    "What was the actual result?",
                    ["Fraud", "Normal"],
                    key=f"actual_{len(st.session_state.prediction_history)}"
                )
            
            if st.button("Submit Feedback", key=f"submit_{len(st.session_state.prediction_history)}"):
                save_feedback(prediction, confidence, user_feedback, actual_result)
                st.success("Thank you for your feedback! 🙏")

def show_admin_dashboard():
    st.markdown('<h1 class="main-header">👨‍💼 Admin Dashboard</h1>', unsafe_allow_html=True)
    
    if not st.session_state.authenticated:
        st.markdown("""
        <div class="prediction-container">
            <h2 style="color: white; text-align: center;">🔐 Admin Login</h2>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            
            if st.button("Login", type="primary", use_container_width=True):
                if authenticate(username, password):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials! Use admin/admin")
    else:
        # Logout button
        if st.button("Logout", type="secondary"):
            st.session_state.authenticated = False
            st.rerun()
        
        # Dashboard content
        st.markdown("### 📊 System Overview")
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", len(st.session_state.prediction_history))
        
        with col2:
            fraud_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 1)
            st.metric("Fraud Detected", fraud_count)
        
        with col3:
            st.metric("Feedback Received", len(st.session_state.feedback_data))
        
        with col4:
            if st.session_state.feedback_data:
                correct_feedback = sum(1 for f in st.session_state.feedback_data if f['user_feedback'] == 'Correct')
                accuracy = (correct_feedback / len(st.session_state.feedback_data)) * 100
                st.metric("User Accuracy", f"{accuracy:.1f}%")
            else:
                st.metric("User Accuracy", "N/A")
        
        # Prediction history chart
        if st.session_state.prediction_history:
            st.markdown("### 📈 Prediction Timeline")
            
            df_predictions = pd.DataFrame(st.session_state.prediction_history)
            df_predictions['timestamp'] = pd.to_datetime(df_predictions['timestamp'])
            df_predictions['result'] = df_predictions['prediction'].map({0: 'Normal', 1: 'Fraud'})
            
            fig = px.scatter(df_predictions, x='timestamp', y='confidence', 
                           color='result', title='Prediction Confidence Over Time',
                           color_discrete_map={'Normal': 'green', 'Fraud': 'red'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Feedback analysis
        if st.session_state.feedback_data:
            st.markdown("### 💬 Feedback Analysis")
            
            df_feedback = pd.DataFrame(st.session_state.feedback_data)
            
            col1, col2 = st.columns(2)
            
            with col1:
                feedback_counts = df_feedback['user_feedback'].value_counts()
                fig_feedback = px.pie(values=feedback_counts.values, names=feedback_counts.index,
                                    title="User Feedback Distribution")
                st.plotly_chart(fig_feedback, use_container_width=True)
            
            with col2:
                actual_counts = df_feedback['actual_result'].value_counts()
                fig_actual = px.pie(values=actual_counts.values, names=actual_counts.index,
                                  title="Actual Results Distribution")
                st.plotly_chart(fig_actual, use_container_width=True)
            
            # Detailed feedback table
            st.markdown("### 📋 Detailed Feedback")
            st.dataframe(df_feedback, use_container_width=True)

if __name__ == "__main__":
    main()
