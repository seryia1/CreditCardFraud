import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import os
from streamlit_option_menu import option_menu
import sys
import time
import psutil
import logging
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
logger = logging.getLogger(__name__)

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
    
    .alert-card {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
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
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    if 'system_metrics' not in st.session_state:
        st.session_state.system_metrics = []
    if 'alerts' not in st.session_state:
        st.session_state.alerts = []
    if 'input_values' not in st.session_state:
        st.session_state.input_values = {}
        for i in range(1, 29):
            st.session_state.input_values[f'V{i}'] = 0.0
        st.session_state.input_values['Time'] = 0.0
        st.session_state.input_values['Amount'] = 0.0

# Load model
@st.cache_resource
def load_model():
    try:
        if not os.path.exists('stacked_pipeline1.joblib'):
            st.error("❌ stacked_pipeline1.joblib not found in the project directory")
            logger.error("Model file not found")
            return None
        
        st.info("🔄 Loading your stacked model...")
        start_time = time.time()
        model = joblib.load('stacked_pipeline1.joblib')
        load_time = time.time() - start_time
        
        st.success("✅ Your stacked model loaded successfully!")
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        return model
        
    except Exception as e:
        st.markdown(f"""
        <div class="error-container">
            <h4>❌ Error Loading Your Model</h4>
            <p><strong>Error Details:</strong> {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        logger.error(f"Model loading failed: {str(e)}")
        return None

# System monitoring functions
def get_system_metrics():
    """Get current system performance metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return None

def log_prediction(inputs, prediction, confidence, response_time):
    """Log prediction details for monitoring"""
    prediction_data = {
        'timestamp': datetime.now().isoformat(),
        'features': inputs,
        'prediction': int(prediction),
        'confidence': confidence,
        'response_time_ms': response_time * 1000,
        'fraud_probability': confidence if prediction == 1 else 100 - confidence
    }
    
    st.session_state.prediction_history.append(prediction_data)
    
    # Log to file
    logger.info(f"Prediction made: {prediction} (confidence: {confidence:.2f}%, response_time: {response_time*1000:.2f}ms)")
    
    # Save to file
    try:
        with open('prediction_history.json', 'w') as f:
            json.dump(st.session_state.prediction_history, f)
    except Exception as e:
        logger.error(f"Error saving prediction history: {str(e)}")

def check_for_alerts():
    """Check for system alerts and anomalies"""
    alerts = []
    
    if len(st.session_state.prediction_history) > 0:
        recent_predictions = st.session_state.prediction_history[-100:]  # Last 100 predictions
        
        # Check fraud rate
        fraud_rate = sum(1 for p in recent_predictions if p['prediction'] == 1) / len(recent_predictions)
        if fraud_rate > 0.1:  # More than 10% fraud rate
            alerts.append({
                'type': 'HIGH_FRAUD_RATE',
                'message': f'High fraud rate detected: {fraud_rate*100:.1f}%',
                'severity': 'HIGH',
                'timestamp': datetime.now().isoformat()
            })
        
        # Check response time
        avg_response_time = np.mean([p['response_time_ms'] for p in recent_predictions])
        if avg_response_time > 1000:  # More than 1 second
            alerts.append({
                'type': 'SLOW_RESPONSE',
                'message': f'Slow response time: {avg_response_time:.0f}ms',
                'severity': 'MEDIUM',
                'timestamp': datetime.now().isoformat()
            })
        
        # Check for low confidence predictions
        low_confidence_count = sum(1 for p in recent_predictions if p['confidence'] < 60)
        if low_confidence_count > len(recent_predictions) * 0.3:  # More than 30% low confidence
            alerts.append({
                'type': 'LOW_CONFIDENCE',
                'message': f'High number of low confidence predictions: {low_confidence_count}',
                'severity': 'MEDIUM',
                'timestamp': datetime.now().isoformat()
            })
    
    # Add new alerts to session state
    for alert in alerts:
        if alert not in st.session_state.alerts:
            st.session_state.alerts.append(alert)
            logger.warning(f"Alert generated: {alert['type']} - {alert['message']}")
    
    return alerts

def calculate_performance_metrics():
    """Calculate model performance metrics"""
    if len(st.session_state.prediction_history) < 10:
        return None
    
    recent_predictions = st.session_state.prediction_history[-1000:]  # Last 1000 predictions
    
    # Calculate basic statistics
    total_predictions = len(recent_predictions)
    fraud_predictions = sum(1 for p in recent_predictions if p['prediction'] == 1)
    normal_predictions = total_predictions - fraud_predictions
    
    avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
    avg_response_time = np.mean([p['response_time_ms'] for p in recent_predictions])
    
    # Calculate confidence distribution
    high_confidence = sum(1 for p in recent_predictions if p['confidence'] > 80)
    medium_confidence = sum(1 for p in recent_predictions if 60 <= p['confidence'] <= 80)
    low_confidence = sum(1 for p in recent_predictions if p['confidence'] < 60)
    
    return {
        'timestamp': datetime.now().isoformat(),
        'total_predictions': total_predictions,
        'fraud_predictions': fraud_predictions,
        'normal_predictions': normal_predictions,
        'fraud_rate': fraud_predictions / total_predictions * 100,
        'avg_confidence': avg_confidence,
        'avg_response_time_ms': avg_response_time,
        'high_confidence_pct': high_confidence / total_predictions * 100,
        'medium_confidence_pct': medium_confidence / total_predictions * 100,
        'low_confidence_pct': low_confidence / total_predictions * 100
    }

# Authentication
def authenticate(username, password):
    return username == "admin" and password == "admin"

def main():
    load_css()
    init_session_state()
    
    # Navigation
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1556742049-0cfed4f6a45d?w=300&h=200&fit=crop", 
                 caption="Credit Card Security", use_column_width=True)
        
        selected = option_menu(
            menu_title="Navigation",
            options=["🏠 Home", "🔍 Fraud Detection", "📊 Admin Dashboard"],
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
    elif selected == "📊 Admin Dashboard":
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
            to detect fraudulent activities with high accuracy and comprehensive monitoring.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Features section
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 High Accuracy</h3>
            <p>Advanced ML algorithms with real-time performance monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>⚡ Real-time Analysis</h3>
            <p>Instant predictions with comprehensive response time tracking.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📊 System Monitoring</h3>
            <p>Complete system health monitoring with automated alerts.</p>
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
                with st.container():
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        value = st.number_input(f"V{i}", 
                                               value=st.session_state.input_values[f'V{i}'],
                                               min_value=-50.0, 
                                               max_value=50.0,
                                               step=0.1,
                                               format="%.1f",
                                               key=f"num_V{i}")
                        st.session_state.input_values[f'V{i}'] = value
                    
                    with col2:
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
        
        batch_input = st.text_area("Format: V1,V2,...,V28,Time,Amount", 
                                  placeholder="Example: 0.1,-0.2,0.3,...,120,50.5")
        
        if st.button("Apply Batch Values"):
            try:
                values = [float(x.strip()) for x in batch_input.split(',')]
                if len(values) == 30:
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
                # Record system metrics before prediction
                system_metrics = get_system_metrics()
                if system_metrics:
                    st.session_state.system_metrics.append(system_metrics)
                
                # Prepare input array
                inputs = {}
                for i in range(1, 29):
                    inputs[f'V{i}'] = st.session_state.input_values[f'V{i}']
                
                inputs['Time'] = st.session_state.input_values['Time']
                inputs['Amount'] = st.session_state.input_values['Amount']
                
                # Convert inputs to array and measure response time
                input_array = np.array([list(inputs.values())])
                
                start_time = time.time()
                prediction = model.predict(input_array)[0]
                proba = model.predict_proba(input_array)[0][1]
                response_time = time.time() - start_time
                
                confidence = proba * 100
                
                # Log prediction for monitoring
                log_prediction(inputs, prediction, confidence, response_time)
                
                # Display result
                if prediction == 1:  # Fraud
                    st.markdown(f"""
                    <div class="fraud-alert">
                        🚨 FRAUD DETECTED 🚨<br>
                        Confidence: {confidence:.2f}%<br>
                        Response Time: {response_time*1000:.0f}ms<br>
                        This transaction shows high risk patterns!
                    </div>
                    """, unsafe_allow_html=True)
                else:  # Normal
                    st.markdown(f"""
                    <div class="safe-alert">
                        ✅ TRANSACTION SAFE ✅<br>
                        Confidence: {confidence:.2f}%<br>
                        Response Time: {response_time*1000:.0f}ms<br>
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
                
                # Check for alerts after prediction
                check_for_alerts()
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")

def show_admin_dashboard():
    st.markdown('<h1 class="main-header">📊 Admin Dashboard</h1>', unsafe_allow_html=True)
    
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
        
        # 🔍 Performance Monitoring - Always show basic stats
        st.markdown("### 🔍 Performance Monitoring")
        
        total_predictions = len(st.session_state.prediction_history)
        fraud_count = sum(1 for p in st.session_state.prediction_history if p['prediction'] == 1) if total_predictions > 0 else 0
        fraud_rate = (fraud_count / total_predictions * 100) if total_predictions > 0 else 0
        avg_confidence = np.mean([p['confidence'] for p in st.session_state.prediction_history]) if total_predictions > 0 else 0
        avg_response_time = np.mean([p['response_time_ms'] for p in st.session_state.prediction_history]) if total_predictions > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Predictions", total_predictions)
        
        with col2:
            st.metric("Fraud Rate", f"{fraud_rate:.1f}%")
        
        with col3:
            st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
        
        with col4:
            st.metric("Avg Response Time", f"{avg_response_time:.0f}ms")
        
        # Show confidence distribution even with no data
        if total_predictions > 0:
            high_confidence = sum(1 for p in st.session_state.prediction_history if p['confidence'] > 80)
            medium_confidence = sum(1 for p in st.session_state.prediction_history if 60 <= p['confidence'] <= 80)
            low_confidence = sum(1 for p in st.session_state.prediction_history if p['confidence'] < 60)
            
            confidence_data = {
                'Confidence Level': ['High (>80%)', 'Medium (60-80%)', 'Low (<60%)'],
                'Count': [high_confidence, medium_confidence, low_confidence]
            }
        else:
            confidence_data = {
                'Confidence Level': ['High (>80%)', 'Medium (60-80%)', 'Low (<60%)'],
                'Count': [0, 0, 0]
            }
        
        fig_confidence = px.pie(confidence_data, values='Count', names='Confidence Level',
                              title="Confidence Distribution")
        st.plotly_chart(fig_confidence, use_container_width=True)
        
        # 🚨 Alert System - Always show current status
        st.markdown("### 🚨 Alert System")
        
        # Check for alerts
        current_alerts = check_for_alerts()
        
        if st.session_state.alerts:
            st.success(f"🔔 {len(st.session_state.alerts)} alerts generated")
            for alert in st.session_state.alerts[-3:]:  # Show last 3 alerts
                severity_color = {"HIGH": "#ff4444", "MEDIUM": "#ffaa00", "LOW": "#44ff44"}
                st.markdown(f"""
                <div class="alert-card" style="border-left-color: {severity_color.get(alert['severity'], '#cccccc')}">
                    <strong>{alert['type']}</strong> ({alert['severity']})<br>
                    {alert['message']}<br>
                    <small>{alert['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("✅ No alerts - System running normally")
        
        # Alert thresholds info
        st.markdown("""
        **Alert Thresholds:**
        - 🔴 High Fraud Rate: >10% fraud predictions
        - 🟡 Slow Response: >1000ms response time  
        - 🟡 Low Confidence: >30% predictions with <60% confidence
        """)
        
        # 📊 System Health Monitoring - Always show current system stats
        st.markdown("### 📊 System Health Monitoring")
        
        # Get current system metrics
        current_metrics = get_system_metrics()
        if current_metrics:
            st.session_state.system_metrics.append(current_metrics)
        
        if current_metrics:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("CPU Usage", f"{current_metrics['cpu_percent']:.1f}%")
            
            with col2:
                st.metric("Memory Usage", f"{current_metrics['memory_percent']:.1f}%")
            
            with col3:
                st.metric("Available Memory", f"{current_metrics['memory_available_gb']:.1f}GB")
            
            with col4:
                st.metric("Disk Usage", f"{current_metrics['disk_percent']:.1f}%")
        else:
            st.warning("⚠️ Unable to get system metrics")
        
        # Model Status - Always show
        model = load_model()
        if model is not None:
            st.success("✅ **Model Status:** Loaded and Ready")
        else:
            st.error("❌ **Model Status:** Failed to Load")
        
        # Simple system info
        st.markdown("### ℹ️ System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Python Version:** {sys.version.split()[0]}")
            st.info(f"**Streamlit Version:** {st.__version__}")
        
        with col2:
            st.info(f"**Current Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.info(f"**Session Duration:** {len(st.session_state.prediction_history)} predictions made")
        
        # Manual refresh button
        if st.button("🔄 Refresh Dashboard", type="secondary"):
            st.rerun()

if __name__ == "__main__":
    main()
