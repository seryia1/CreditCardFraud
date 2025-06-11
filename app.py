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
import sys

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
    
    .error-container {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    
    .input-group {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .input-group .number-input {
        width: 100px;
        margin-right: 10px;
    }
    
    .slider-container {
        flex-grow: 1;
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
    if 'input_values' not in st.session_state:
        st.session_state.input_values = {}
        for i in range(1, 29):
            st.session_state.input_values[f'V{i}'] = 0.0
        st.session_state.input_values['Time'] = 0.0
        st.session_state.input_values['Amount'] = 0.0

# Load model and column ranges
@st.cache_resource
def load_model():
    try:
        # Check if file exists first
        if not os.path.exists('stacked_pipeline1.joblib'):
            st.error("❌ stacked_pipeline1.joblib not found in the project directory")
            return None
        
        # Load your specific model
        st.info("🔄 Loading your stacked model...")
        model = joblib.load('stacked_pipeline1.joblib')
        
        st.success("✅ Your stacked model loaded successfully!")
        return model
        
    except Exception as e:
        st.markdown(f"""
        <div class="error-container">
            <h4>❌ Error Loading Your Model</h4>
            <p><strong>Error Details:</strong> {str(e)}</p>
            <p><strong>Possible Solutions:</strong></p>
            <ul>
                <li>Your model was saved with a different version of scikit-learn</li>
                <li>Try updating your requirements.txt to match the training environment</li>
                <li>Re-save your model with the current environment versions</li>
            </ul>
            <p><strong>Current Python Version:</strong> {sys.version}</p>
        </div>
        """, unsafe_allow_html=True)
        return None

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
    
    model = load_model()
    
    if model is None:
        st.markdown("""
        <div class="error-container">
            <h3>⚠️ Model Loading Failed</h3>
            <p>Unable to load your stacked model. Please ensure:</p>
            <ul>
                <li><code>stacked_pipeline1.joblib</code> is in the project root directory</li>
                <li>The model file is compatible with the current environment</li>
            </ul>
            <p>The fraud detection feature is currently unavailable.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
        <h3 style="color: white; margin: 0;">Enter Transaction Features (PCA Components)</h3>
        <p style="color: white; margin: 0;">Adjust the sliders or enter values directly for each PCA component.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📊 Interactive Sliders", "📝 Batch Input"])
    
    with tab1:
        # Create input sliders with number inputs
        st.markdown("### PCA Components")
        
        # Create 4 columns for better layout
        cols = st.columns(4)
        
        # Create sliders for V1-V28
        for i in range(1, 29):
            col_idx = (i-1) % 4
            with cols[col_idx]:
                # Create a container for each input group
                with st.container():
                    # Number input and slider in the same row
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        # Direct number input
                        value = st.number_input(f"V{i}", 
                                               value=st.session_state.input_values[f'V{i}'],
                                               min_value=-50.0, 
                                               max_value=50.0,
                                               step=0.1,
                                               format="%.1f",
                                               key=f"num_V{i}")
                        st.session_state.input_values[f'V{i}'] = value
                    
                    with col2:
                        # Slider that updates when number input changes
                        st.slider(f"", 
                                 min_value=-50.0, 
                                 max_value=50.0, 
                                 value=st.session_state.input_values[f'V{i}'],
                                 step=0.1,
                                 key=f"slider_V{i}",
                                 label_visibility="collapsed")
                        
        # Time and Amount inputs
        st.markdown("### Transaction Details")
        col1, col2 = st.columns(2)
        
        with col1:
            time_value = st.number_input("Time (seconds)", 
                                        value=st.session_state.input_values['Time'],
                                        min_value=0.0, 
                                        max_value=200000.0,
                                        step=1.0)
            st.session_state.input_values['Time'] = time_value
            st.slider("Time Slider", 
                     min_value=0.0, 
                     max_value=200000.0, 
                     value=st.session_state.input_values['Time'],
                     step=1.0)
        
        with col2:
            amount_value = st.number_input("Amount ($)", 
                                          value=st.session_state.input_values['Amount'],
                                          min_value=0.0, 
                                          max_value=25000.0,
                                          step=1.0)
            st.session_state.input_values['Amount'] = amount_value
            st.slider("Amount Slider", 
                     min_value=0.0, 
                     max_value=25000.0, 
                     value=st.session_state.input_values['Amount'],
                     step=1.0)
    
    with tab2:
        st.markdown("### Paste Multiple Values")
        st.markdown("Enter comma-separated values for all features (V1-V28, Time, Amount) in one go:")
        
        # Text area for batch input
        batch_input = st.text_area("Format: V1,V2,...,V28,Time,Amount", 
                                  placeholder="Example: 0.1,-0.2,0.3,...,120,50.5")
        
        if st.button("Apply Batch Values"):
            try:
                values = [float(x.strip()) for x in batch_input.split(',')]
                if len(values) == 30:  # 28 V features + Time + Amount
                    for i in range(1, 29):
                        st.session_state.input_values[f'V{i}'] = values[i-1]
                    st.session_state.input_values['Time'] = values[28]
                    st.session_state.input_values['Amount'] = values[29]
                    st.success("✅ Values applied successfully!")
                else:
                    st.error(f"❌ Expected 30 values, but got {len(values)}. Please check your input.")
            except Exception as e:
                st.error(f"❌ Error parsing values: {str(e)}")
    
    # Prediction button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔍 Analyze Transaction", type="primary", use_container_width=True):
            try:
                # Prepare input array
                inputs = {}
                for i in range(1, 29):
                    inputs[f'V{i}'] = st.session_state.input_values[f'V{i}']
                
                inputs['Time'] = st.session_state.input_values['Time']
                inputs['Amount'] = st.session_state.input_values['Amount']
                
                # Convert inputs to array
                input_array = np.array([list(inputs.values())])
                
                # Make prediction
                prediction = model.predict(input_array)[0]
                proba = model.predict_proba(input_array)[0][1]
                confidence = proba * 100
                
                # Store prediction
                prediction_data = {
                    'timestamp': datetime.now().isoformat(),
                    'features': inputs,
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
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check that your input values are valid.")

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
