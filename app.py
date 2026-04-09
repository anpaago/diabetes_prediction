import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import io
from datetime import datetime

# Load the trained model and scaler with error handling
try:
    model_file = 'DiabetesML_model.joblib'
    scaler_file = 'scaler.joblib'
    
    if not os.path.exists(model_file):
        model_file = 'random_forest_model.joblib'
    if not os.path.exists(scaler_file):
        scaler_file = 'standard_scaler.joblib'
    
    model = joblib.load(model_file)
    scaler = joblib.load(scaler_file)
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS styling
st.markdown("""
    <style>
    /* Overall styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Header styling */
    .header-main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.3);
    }
    
    .header-main h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-main p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-left: 5px solid #667eea;
        color: #1a202c;
    }
    
    .metric-card h3,
    .metric-card h4 {
        color: #1a202c;
        margin-top: 0;
    }
    
    .metric-card ul {
        color: #1a202c;
    }
    
    .metric-card ol {
        color: #1a202c;
    }
    
    /* Prediction result cards */
    .prediction-negative {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #0d7377;
        box-shadow: 0 4px 12px rgba(13, 115, 119, 0.2);
        color: #0d3b3d;
    }
    
    .prediction-negative h3 {
        color: #0d3b3d;
        margin-top: 0;
    }
    
    .prediction-negative p {
        color: #0d3b3d;
    }
    
    .prediction-positive {
        background: linear-gradient(135deg, #fa8072 0%, #ff6b6b 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 6px solid #c92a2a;
        box-shadow: 0 4px 12px rgba(201, 42, 42, 0.2);
        color: #ffffff;
    }
    
    .prediction-positive h3 {
        color: #ffffff;
        margin-top: 0;
    }
    
    .prediction-positive p {
        color: #ffffff;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] button {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        padding: 0.8rem 1.5rem;
    }
    
    .stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 3px solid #667eea;
    }
    
    /* Input section */
    .input-section {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid #e0e0e0;
        color: #212529;
    }
    
    .input-section strong {
        color: #212529;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #2d3748;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Info boxes */
    .info-box {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
        color: #0d47a1;
    }
    
    .info-box strong {
        color: #0d47a1;
    }
    
    .info-box h3,
    .info-box h4 {
        color: #0d47a1;
        margin-top: 0;
    }
    
    .info-box p,
    .info-box ul,
    .info-box ol {
        color: #0d47a1;
    }
    
    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
        color: #856404;
    }
    
    .warning-box strong {
        color: #856404;
    }
    
    .warning-box h3,
    .warning-box h4 {
        color: #856404;
        margin-top: 0;
    }
    
    .warning-box p,
    .warning-box ul,
    .warning-box ol {
        color: #856404;
    }
    
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
        color: #155724;
    }
    
    .success-box strong {
        color: #155724;
    }
    
    .success-box h3,
    .success-box h4 {
        color: #155724;
        margin-top: 0;
    }
    
    .success-box p,
    .success-box ul,
    .success-box ol {
        color: #155724;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    }
    
    /* Slider styling */
    .stSlider {
        padding: 0.5rem 0;
    }
    
    /* Cards for batch results */
    .result-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        color: #2d3748;
    }
    
    /* Footer styling */
    .footer-text {
        color: #4a5568;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("""
    <div class="header-main">
        <h1>🏥 Diabetes Prediction AI</h1>
        <p>Advanced Machine Learning Model for Intelligent Diabetes Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)

# Imputation values from training data
GLUCOSE_MEAN = 120.894531
BLOODPRESSURE_MEAN = 69.105469
BMI_MEAN = 31.992578
SKINTHICKNESS_MEDIAN = 23.0
INSULIN_MEDIAN = 30.5

# Feature information
FEATURE_HELP = {
    'Pregnancies': 'Number of pregnancies (0-17)',
    'Glucose': 'Blood glucose level in mg/dL (0-200)',
    'BloodPressure': 'Diastolic blood pressure in mmHg (0-122)',
    'SkinThickness': 'Triceps skin fold thickness in mm (0-99)',
    'Insulin': '2-hour serum insulin in µU/ml (0-846)',
    'BMI': 'Body Mass Index in kg/m² (0-67.1)',
    'DiabetesPedigreeFunction': 'Family history of diabetes (0.078-2.42)',
    'Age': 'Patient age in years (21-81)'
}

# Function to preprocess data
def preprocess_data(df):
    """Preprocess input data with imputation and scaling"""
    temp_df = df.copy()
    
    # Apply imputation for zero values
    if 'Glucose' in temp_df.columns and (temp_df['Glucose'] == 0).any():
        temp_df.loc[temp_df['Glucose'] == 0, 'Glucose'] = GLUCOSE_MEAN
    if 'BloodPressure' in temp_df.columns and (temp_df['BloodPressure'] == 0).any():
        temp_df.loc[temp_df['BloodPressure'] == 0, 'BloodPressure'] = BLOODPRESSURE_MEAN
    if 'BMI' in temp_df.columns and (temp_df['BMI'] == 0).any():
        temp_df.loc[temp_df['BMI'] == 0, 'BMI'] = BMI_MEAN
    if 'SkinThickness' in temp_df.columns and (temp_df['SkinThickness'] == 0).any():
        temp_df.loc[temp_df['SkinThickness'] == 0, 'SkinThickness'] = SKINTHICKNESS_MEDIAN
    if 'Insulin' in temp_df.columns and (temp_df['Insulin'] == 0).any():
        temp_df.loc[temp_df['Insulin'] == 0, 'Insulin'] = INSULIN_MEDIAN
    
    return temp_df

# Function to make predictions
def predict_diabetes(df):
    """Make diabetes predictions"""
    processed_df = preprocess_data(df)
    scaled_data = scaler.transform(processed_df)
    predictions = model.predict(scaled_data)
    probabilities = model.predict_proba(scaled_data)
    return predictions, probabilities

# Function to get prediction label
def get_prediction_label(pred):
    """Convert prediction to label"""
    return "Diabetic ⚠️" if pred == 1 else "Non-Diabetic ✅"

# Input fields function
def user_input_features(key_prefix=""):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pregnancies = st.slider('👶 Pregnancies', 0, 17, 3, help=FEATURE_HELP['Pregnancies'], key=f"{key_prefix}pregnancies")
        glucose = st.slider('🍬 Glucose', 0, 200, 120, help=FEATURE_HELP['Glucose'], key=f"{key_prefix}glucose")
    
    with col2:
        blood_pressure = st.slider('💓 Blood Pressure', 0, 122, 70, help=FEATURE_HELP['BloodPressure'], key=f"{key_prefix}blood_pressure")
        skin_thickness = st.slider('📏 Skin Thickness', 0, 99, 20, help=FEATURE_HELP['SkinThickness'], key=f"{key_prefix}skin_thickness")
    
    with col3:
        insulin = st.slider('💉 Insulin', 0, 846, 79, help=FEATURE_HELP['Insulin'], key=f"{key_prefix}insulin")
        bmi = st.slider('⚖️ BMI', 0.0, 67.1, 32.0, help=FEATURE_HELP['BMI'], key=f"{key_prefix}bmi")
    
    with col4:
        dpf = st.slider('🧬 Diabetes Pedigree Function', 0.078, 2.42, 0.471, help=FEATURE_HELP['DiabetesPedigreeFunction'], key=f"{key_prefix}dpf")
        age = st.slider('📅 Age', 21, 81, 33, help=FEATURE_HELP['Age'], key=f"{key_prefix}age")
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

# Create tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Single Prediction", "📁 Batch Prediction", "📊 Analytics", "📈 Model Info", "ℹ️ About"])

# TAB 1: Single Prediction
with tab1:
    st.markdown('<div class="section-header">👤 Patient Data & Prediction</div>', unsafe_allow_html=True)
    
    df_input = user_input_features(key_prefix="tab1_")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("**Your Input Values:**")
        input_display = df_input.copy()
        st.dataframe(input_display, use_container_width=True, key="input_table_single", hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_right:
        # Make predictions
        predictions, probabilities = predict_diabetes(df_input)
        prediction = predictions[0]
        non_diabetic_prob = probabilities[0][0] * 100
        diabetic_prob = probabilities[0][1] * 100
        
        # Display prediction with gauge
        st.markdown('<div class="section-header">🎯 Prediction Result</div>', unsafe_allow_html=True)
        
        if prediction == 0:
            st.markdown("""
                <div class="prediction-negative">
                    <h3>✅ Non-Diabetic</h3>
                    <p style="font-size: 1.1rem; margin: 0.5rem 0;">The model predicts this patient is <strong>unlikely to have diabetes</strong>.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="prediction-positive">
                    <h3>⚠️ Diabetic</h3>
                    <p style="font-size: 1.1rem; margin: 0.5rem 0;">The model predicts this patient <strong>may have diabetes</strong>. Medical consultation recommended.</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=diabetic_prob,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Diabetes Risk %"},
            delta={'reference': 50},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightcoral"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig.update_layout(height=350, margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True, key="gauge_chart")
    
    st.markdown("---")
    
    # Probability breakdown
    col_prob1, col_prob2, col_prob3 = st.columns(3)
    
    with col_prob1:
        st.metric("Non-Diabetic Probability", f"{non_diabetic_prob:.1f}%", delta=f"{non_diabetic_prob - 50:+.1f}%")
    with col_prob2:
        st.metric("Diabetic Probability", f"{diabetic_prob:.1f}%", delta=f"{diabetic_prob - 50:+.1f}%")
    with col_prob3:
        confidence = max(non_diabetic_prob, diabetic_prob)
        st.metric("Model Confidence", f"{confidence:.1f}%", delta=f"{confidence - 70:.1f}%")

# TAB 2: Batch Prediction
with tab2:
    st.markdown('<div class="section-header">📁 Batch Prediction</div>', unsafe_allow_html=True)
    
    st.markdown("""
        <div class="info-box">
            <strong>📤 Upload Instructions:</strong><br>
            Upload a CSV file with the following columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv", key="batch_upload")
    
    if uploaded_file is not None:
        try:
            df_batch = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            missing_columns = [col for col in required_columns if col not in df_batch.columns]
            
            if missing_columns:
                st.error(f"❌ Missing columns: {', '.join(missing_columns)}")
            else:
                st.success(f"✅ Loaded {len(df_batch)} records")
                
                # Show preview
                with st.expander("📋 Data Preview"):
                    st.dataframe(df_batch.head(10), use_container_width=True)
                
                # Make batch predictions
                if st.button("🚀 Run Batch Prediction", key="batch_predict"):
                    with st.spinner("Processing predictions..."):
                        predictions_batch, probabilities_batch = predict_diabetes(df_batch)
                        
                        # Create results dataframe
                        results_df = df_batch.copy()
                        results_df['Prediction'] = predictions_batch
                        results_df['Prediction_Label'] = [get_prediction_label(p) for p in predictions_batch]
                        results_df['Diabetic_Risk_%'] = (probabilities_batch[:, 1] * 100).round(2)
                        results_df['Non_Diabetic_%'] = (probabilities_batch[:, 0] * 100).round(2)
                        
                        # Display statistics
                        st.markdown("---")
                        st.markdown('<div class="section-header">📊 Batch Results Summary</div>', unsafe_allow_html=True)
                        
                        col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                        
                        total_records = len(results_df)
                        diabetic_count = (results_df['Prediction'] == 1).sum()
                        non_diabetic_count = (results_df['Prediction'] == 0).sum()
                        avg_risk = results_df['Diabetic_Risk_%'].mean()
                        
                        with col_stat1:
                            st.metric("Total Records", total_records)
                        with col_stat2:
                            st.metric("Diabetic Cases", diabetic_count, f"{(diabetic_count/total_records)*100:.1f}%")
                        with col_stat3:
                            st.metric("Non-Diabetic Cases", non_diabetic_count, f"{(non_diabetic_count/total_records)*100:.1f}%")
                        with col_stat4:
                            st.metric("Average Risk %", f"{avg_risk:.1f}%")
                        
                        st.markdown("---")
                        
                        # Results visualization
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            fig_dist = go.Figure(data=[
                                go.Histogram(x=results_df['Diabetic_Risk_%'], nbinsx=20, 
                                           marker_color='#667eea', name='Risk Distribution')
                            ])
                            fig_dist.update_layout(title="Diabetes Risk Distribution", 
                                                  xaxis_title="Risk %", 
                                                  yaxis_title="Count",
                                                  height=350)
                            st.plotly_chart(fig_dist, use_container_width=True, key="batch_hist")
                        
                        with col_chart2:
                            pred_counts = results_df['Prediction_Label'].value_counts()
                            colors = ['#84fab0', '#fa8072']
                            fig_pie = go.Figure(data=[go.Pie(
                                labels=pred_counts.index,
                                values=pred_counts.values,
                                marker=dict(colors=colors),
                                textinfo='label+percent+value'
                            )])
                            fig_pie.update_layout(title="Prediction Distribution", height=350)
                            st.plotly_chart(fig_pie, use_container_width=True, key="batch_pie")
                        
                        st.markdown("---")
                        
                        # Detailed results table
                        st.markdown('<div class="section-header">📋 Detailed Results</div>', unsafe_allow_html=True)
                        
                        display_cols = ['Pregnancies', 'Glucose', 'BMI', 'Age', 'Prediction_Label', 'Diabetic_Risk_%']
                        st.dataframe(results_df[display_cols], use_container_width=True, hide_index=True)
                        
                        # Download button
                        csv_results = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv_results,
                            file_name=f"diabetes_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key="download_batch"
                        )
        
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")

# TAB 3: Analytics
with tab3:
    st.markdown('<div class="section-header">📊 Feature Analysis</div>', unsafe_allow_html=True)
    
    df_analytics = user_input_features(key_prefix="tab3_")
    
    col_analysis1, col_analysis2 = st.columns(2)
    
    with col_analysis1:
        st.markdown("**Metabolic Markers**")
        metabolic_data = {
            'Marker': ['Glucose', 'BMI', 'Insulin'],
            'Value': [df_analytics['Glucose'].iloc[0], df_analytics['BMI'].iloc[0], df_analytics['Insulin'].iloc[0]]
        }
        metabolic_df = pd.DataFrame(metabolic_data)
        fig_metabolic = px.bar(metabolic_df, x='Marker', y='Value', color='Marker',
                               title="Key Metabolic Indicators",
                               color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
        fig_metabolic.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_metabolic, use_container_width=True, key="metabolic_chart")
    
    with col_analysis2:
        st.markdown("**Demographic Information**")
        demo_data = {
            'Category': ['Pregnancies', 'Age'],
            'Count': [df_analytics['Pregnancies'].iloc[0], df_analytics['Age'].iloc[0]]
        }
        demo_df = pd.DataFrame(demo_data)
        fig_demo = px.bar(demo_df, x='Category', y='Count', color='Category',
                          title="Demographic Data",
                          color_discrete_sequence=['#4facfe', '#00f2fe'])
        fig_demo.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig_demo, use_container_width=True, key="demo_chart")
    
    st.markdown("---")
    
    st.markdown('<div class="section-header">Feature Importance & Model Insights</div>', unsafe_allow_html=True)
    
    feature_importance_data = {
        'Feature': ['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 
                   'BloodPressure', 'Pregnancies', 'Insulin', 'SkinThickness'],
        'Importance': [0.264023, 0.164255, 0.130515, 0.122300, 
                      0.089039, 0.081850, 0.076212, 0.071807]
    }
    feature_importance_df = pd.DataFrame(feature_importance_data)
    
    fig_importance = px.bar(feature_importance_df, x='Feature', y='Importance',
                            title="Feature Importance in Prediction Model",
                            color='Importance',
                            color_continuous_scale='Viridis',
                            orientation='v')
    fig_importance.update_layout(height=450, xaxis_tickangle=-45)
    st.plotly_chart(fig_importance, use_container_width=True, key="importance_chart")
    
    st.markdown("""
        <div class="success-box">
            <strong>🔬 Key Insights:</strong><br>
            • <strong>Glucose</strong> is the strongest predictor (26.4% importance)<br>
            • <strong>BMI</strong> is critical (16.4% importance)<br>
            • <strong>Age</strong> significantly impacts predictions (13.1%)<br>
            • Family history matters: Diabetes Pedigree Function (12.2%)
        </div>
    """, unsafe_allow_html=True)

# TAB 4: Model Information
with tab4:
    st.markdown('<div class="section-header">📈 Model Details</div>', unsafe_allow_html=True)
    
    col_model1, col_model2 = st.columns(2)
    
    with col_model1:
        st.markdown("""
            <div class="metric-card">
                <h4>🤖 Algorithm Details</h4>
                <ul style="line-height: 2;">
                    <li><strong>Model Type:</strong> Random Forest Classifier</li>
                    <li><strong>Number of Trees:</strong> 100</li>
                    <li><strong>Training Dataset:</strong> Pima Indians Diabetes</li>
                    <li><strong>Total Samples:</strong> 768 patients</li>
                    <li><strong>Features Used:</strong> 8 medical/demographic</li>
                    <li><strong>Target Classes:</strong> Binary (Diabetic/Non-Diabetic)</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col_model2:
        st.markdown("""
            <div class="metric-card">
                <h4>📊 Preprocessing Pipeline</h4>
                <ul style="line-height: 2;">
                    <li><strong>Imputation:</strong> Mean/Median for zero values</li>
                    <li><strong>Scaling:</strong> StandardScaler normalization</li>
                    <li><strong>Feature Range:</strong> Clinically validated</li>
                    <li><strong>Missing Values:</strong> Handled systematically</li>
                    <li><strong>Data Quality:</strong> Verified & validated</li>
                    <li><strong>Model State:</strong> Serialized with joblib</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div class="warning-box">
            <h4>⚠️ Important Model Limitations & Disclaimers</h4>
            <ul>
                <li>This model is trained on the <strong>Pima Indians Diabetes Database</strong> and may not generalize perfectly to other populations</li>
                <li>Predictions are <strong>probabilistic estimates</strong>, not definitive diagnoses</li>
                <li>Clinical validation should always be performed by healthcare professionals</li>
                <li>The model may have bias based on the training data demographics</li>
                <li>For critical medical decisions, always consult with qualified healthcare providers</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

# TAB 5: About & Instructions
with tab5:
    col_about1, col_about2 = st.columns(2)
    
    with col_about1:
        st.markdown("""
            <div class="metric-card">
                <h3>🎯 How to Use This Application</h3>
                <ol style="line-height: 2;">
                    <li><strong>Single Prediction:</strong> Use sliders to input patient data and get instant predictions</li>
                    <li><strong>Batch Prediction:</strong> Upload a CSV file with multiple patients</li>
                    <li><strong>Analytics:</strong> Explore feature importance and model insights</li>
                    <li><strong>Model Info:</strong> Understand the model's architecture and limitations</li>
                    <li>Review confidence scores and probabilities</li>
                    <li>Export batch results for further analysis</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
    
    with col_about2:
        st.markdown("""
            <div class="metric-card">
                <h3>📚 Input Feature Guide</h3>
                <ul style="line-height: 2.2; font-size: 0.95rem;">
                    <li><strong>👶 Pregnancies:</strong> 0-17</li>
                    <li><strong>🍬 Glucose:</strong> 0-200 mg/dL</li>
                    <li><strong>💓 Blood Pressure:</strong> 0-122 mmHg</li>
                    <li><strong>📏 Skin Thickness:</strong> 0-99 mm</li>
                    <li><strong>💉 Insulin:</strong> 0-846 µU/ml</li>
                    <li><strong>⚖️ BMI:</strong> 0-67.1 kg/m²</li>
                    <li><strong>🧬 Diabetes Pedigree:</strong> 0.078-2.42</li>
                    <li><strong>📅 Age:</strong> 21-81 years</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
        <div class="info-box">
            <h3>🏥 Medical Disclaimer</h3>
            <p>This application is designed for <strong>educational and research purposes only</strong>. 
            It should <strong>NOT</strong> be used as a substitute for professional medical advice, 
            diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #4a5568; font-size: 0.9rem;">
            <p><strong>Diabetes Prediction AI v2.0</strong> | Built with Streamlit & Machine Learning</p>
            <p>Dataset: Pima Indians Diabetes Database | Model: Random Forest Classifier</p>
        </div>
    """, unsafe_allow_html=True)

