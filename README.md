# 🏥 Diabetes Prediction AI

A sophisticated machine learning application for diabetes risk prediction with an interactive, beautifully designed Streamlit web interface. This application uses a trained Random Forest classifier to predict the likelihood of diabetes based on medical and demographic parameters.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Application Guide](#application-guide)
- [Model Information](#model-information)
- [Input Parameters](#input-parameters)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)

## 🎯 Overview

**Diabetes Prediction AI** is an advanced machine learning application designed to assess diabetes risk using the Pima Indians Diabetes Database. The application provides both single-patient predictions and batch processing capabilities through an intuitive web interface.

### Key Highlights

- ✨ **Beautiful UI**: Modern gradient design with excellent color contrast and accessibility
- 🔮 **Single Predictions**: Interactive sliders for easy patient data input
- 📁 **Batch Processing**: Upload CSV files for predictions on multiple patients
- 📊 **Advanced Analytics**: Feature importance visualization and data analysis
- 📈 **Probability Gauges**: Visual representation of diabetes risk percentages
- 💾 **Data Export**: Download batch results with timestamps
- 📱 **Responsive Design**: Works on desktop and tablet browsers

## ✨ Features

### 🔮 Single Prediction Tab
- Interactive sliders for all 8 patient features with helpful tooltips
- Real-time prediction results with color-coded outcomes
- Diabetes risk gauge chart showing probability distribution
- Confidence score breakdown
- Live data visualization

### 📁 Batch Prediction Tab
- CSV file upload for bulk predictions
- Automatic data validation and error handling
- Summary statistics showing:
  - Total number of records processed
  - Count of diabetic vs non-diabetic predictions
  - Percentage distribution of outcomes
  - Average diabetes risk percentage
- Visual analytics:
  - Risk distribution histogram
  - Prediction category pie chart
  - Detailed results table
- CSV export functionality with timestamp

### 📊 Analytics Tab
- Metabolic markers visualization (Glucose, BMI, Insulin)
- Demographic information charts
- Feature importance analysis showing which features impact predictions most
- Key insights about model decision factors

### 📈 Model Info Tab
- Detailed algorithm specifications
- Preprocessing pipeline documentation
- Training data information
- Important model limitations and disclaimers
- Feature requirements and ranges

### ℹ️ About Tab
- Complete usage instructions
- Input feature guide with ranges
- Medical disclaimers
- Application credits and information

## 📂 Project Structure

```
Diabetes_ML/
├── app.py                      # Main Streamlit application
├── DiabetesML_model.joblib     # Trained Random Forest model
├── scaler.joblib               # StandardScaler for feature normalization
├── requirements.txt            # Python package dependencies
├── README.md                   # This file
└── .venv/                      # Virtual environment (optional)
```

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Windows, macOS, or Linux

### Step-by-Step Installation

#### 1. Clone or Download the Project
```bash
# Download the Diabetes_ML folder to your desired location
cd Diabetes_ML
```

#### 2. Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

If you don't have a requirements.txt, install these packages:
```bash
pip install streamlit pandas joblib scikit-learn numpy plotly
```

#### 4. Verify Installation
```bash
python -c "import streamlit; print(streamlit.__version__)"
```

## 💻 Usage

### Running the Application

#### From Command Line
```bash
# Navigate to project directory
cd Diabetes_ML

# Activate virtual environment (if using one)
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Run the Streamlit app
streamlit run app.py
```

#### Expected Output
```
You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.x.x:8501
```

#### Access the App
Open your web browser and navigate to:
- **Local Access**: `http://localhost:8501`
- **Network Access**: Use the provided Network URL to access from other devices on the same network

### Stopping the Application
Press `Ctrl+C` in the terminal to stop the Streamlit server.

## 📖 Application Guide

### 🔮 Single Prediction Workflow

1. **Navigate to "Single Prediction" tab** (default view)
2. **Adjust Patient Parameters** using the sliders:
   - 👶 Pregnancies (0-17)
   - 🍬 Glucose (0-200 mg/dL)
   - 💓 Blood Pressure (0-122 mmHg)
   - 📏 Skin Thickness (0-99 mm)
   - 💉 Insulin (0-846 µU/ml)
   - ⚖️ BMI (0-67.1 kg/m²)
   - 🧬 Diabetes Pedigree Function (0.078-2.42)
   - 📅 Age (21-81 years)
3. **View Results**:
   - See input values in the table
   - Check prediction result (Non-Diabetic ✅ or Diabetic ⚠️)
   - Monitor the risk gauge chart
   - Review probability percentages

### 📁 Batch Prediction Workflow

1. **Prepare CSV File**:
   - Headers must include: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
   - Example format:
     ```csv
     Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
     6,148,72,35,0,33.6,0.627,50
     1,85,66,29,0,26.6,0.351,31
     8,183,64,0,0,23.3,0.672,32
     ```

2. **Upload File**:
   - Click "Choose a CSV file"
   - Select your prepared CSV file
   - System validates columns and shows preview

3. **Run Predictions**:
   - Click "🚀 Run Batch Prediction" button
   - Wait for processing (shows spinner)
   - View summary statistics and visualizations
   - Review detailed results table

4. **Export Results**:
   - Click "📥 Download Results as CSV"
   - File automatically downloads with timestamp (e.g., `diabetes_predictions_20260409_153000.csv`)
   - Results include original data plus predictions and risk percentages

### 📊 Analytics Tab

1. **View Feature Analysis**:
   - Adjust sliders to see metabolic and demographic data visualizations
   - Charts update in real-time

2. **Explore Feature Importance**:
   - See which features have the most impact on predictions
   - Glucose: 26.4% (strongest predictor)
   - BMI: 16.4% (second most important)
   - Age: 13.1% (significant impact)
   - Diabetes Pedigree Function: 12.2% (family history matters)

## 📈 Model Information

### Algorithm Details
- **Model Type**: Random Forest Classifier
- **Number of Trees**: 100 decision trees
- **Training Dataset**: Pima Indians Diabetes Database
- **Total Training Samples**: 768 patients
- **Number of Features**: 8 medical/demographic attributes
- **Target Classes**: Binary (0: Non-Diabetic, 1: Diabetic)

### Preprocessing Pipeline

1. **Imputation Strategy**:
   - Glucose: Mean imputation (120.89 mg/dL)
   - Blood Pressure: Mean imputation (69.11 mmHg)
   - BMI: Mean imputation (31.99 kg/m²)
   - Skin Thickness: Median imputation (23.0 mm)
   - Insulin: Median imputation (30.5 µU/ml)

2. **Feature Scaling**:
   - StandardScaler normalization applied to all features
   - Ensures features are on comparable scales
   - Improves model performance and convergence

3. **Data Quality**:
   - Zero values handled systematically
   - Clinically validated feature ranges
   - Verified and validated preprocessing logic

### Model Performance
- Trained on imbalanced dataset (268 diabetic, 500 non-diabetic)
- Uses robust preprocessing to handle missing values
- Provides probability estimates with confidence scores

## 📝 Input Parameters

All input parameters are validated within clinically realistic ranges based on the Pima Indians dataset:

| Feature | Range | Unit | Description |
|---------|-------|------|-------------|
| **Pregnancies** | 0-17 | Count | Number of times pregnant |
| **Glucose** | 0-200 | mg/dL | Plasma glucose concentration |
| **BloodPressure** | 0-122 | mmHg | Diastolic blood pressure |
| **SkinThickness** | 0-99 | mm | Triceps skin fold thickness |
| **Insulin** | 0-846 | µU/ml | 2-hour serum insulin |
| **BMI** | 0-67.1 | kg/m² | Body Mass Index |
| **DiabetesPedigreeFunction** | 0.078-2.42 | Ratio | Family history score |
| **Age** | 21-81 | years | Patient age |

## 🔧 Technical Details

### Dependencies

```
streamlit>=1.28.0          # Web application framework
pandas>=2.0.0              # Data manipulation
joblib>=1.3.0              # Model serialization
scikit-learn>=1.4.0        # Machine learning library
numpy>=1.24.0              # Numerical computing
plotly>=5.0.0              # Interactive visualizations
pandas>=2.0.0              # Data analysis
```

### Python Version
- Python 3.8+
- Tested on Python 3.9, 3.10, 3.11

### File Sizes
- `DiabetesML_model.joblib`: ~5-10 MB
- `scaler.joblib`: ~1 KB
- `app.py`: ~30 KB

### Memory Requirements
- Minimum: 512 MB RAM
- Recommended: 2+ GB RAM for batch processing (1000+ records)

### Browser Compatibility
- ✅ Chrome/Chromium 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

## 🔍 Troubleshooting

### Issue: "Module not found" Error

**Problem**: `ModuleNotFoundError: No module named 'streamlit'`

**Solution**:
```bash
# Ensure virtual environment is activated
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Issue: Model Files Not Found

**Problem**: `Error loading model files: [Errno 2] No such file or directory`

**Solution**:
- Ensure both `DiabetesML_model.joblib` and `scaler.joblib` are in the same directory as `app.py`
- Check file names are exactly correct (case-sensitive on Linux/macOS)

### Issue: Port Already in Use

**Problem**: `Address already in use`

**Solution**:
```bash
# Kill the process on port 8501
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# macOS/Linux
lsof -ti:8501 | xargs kill -9

# Or use a different port
streamlit run app.py --server.port 8502
```

### Issue: CSV Upload Fails

**Problem**: `Error processing file`

**Solution**:
- Verify CSV has all required columns: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- Check for proper formatting (comma-separated)
- Ensure no special characters in column names
- Try with a sample CSV file first

### Issue: Slow Performance

**Problem**: App is slow or unresponsive

**Solution**:
- Close unnecessary applications
- Use 64-bit Python instead of 32-bit
- Reduce batch size (process fewer records at once)
- Check available disk space

### Issue: Charts Not Displaying

**Problem**: Blank spaces where charts should appear

**Solution**:
- Clear browser cache (Ctrl+Shift+Delete)
- Try a different browser
- Restart the Streamlit app
- Check browser console for errors (F12)

## ⚠️ Important Disclaimers

### Medical Disclaimer
```
⚠️ CRITICAL: This application is for EDUCATIONAL and RESEARCH PURPOSES ONLY.

❌ DO NOT use this application as a substitute for professional medical advice, 
   diagnosis, or treatment.

✅ ALWAYS consult with a qualified healthcare provider for medical decisions.
```

### Model Limitations

1. **Dataset Bias**: Model trained on Pima Indians dataset - may not generalize perfectly to other populations
2. **Probabilistic Estimates**: Predictions are probability estimates, not definitive diagnoses
3. **Clinical Validation Required**: All results must be validated by healthcare professionals
4. **Population Specificity**: Results may vary based on ethnic and demographic differences
5. **Missing Clinical Context**: Model doesn't consider medication, treatment history, or other medical factors
6. **No Real-Time Updates**: Model uses fixed feature importance - doesn't learn from new data

### Proper Usage
- Use only as a screening tool
- Never replace professional medical judgment
- Consider results alongside clinical expertise
- Maintain patient confidentiality
- Keep audit logs of predictions for accountability

## 📞 Support & Contributing

### Reporting Issues
If you encounter bugs or issues:
1. Check the Troubleshooting section above
2. Verify all files are present and correct
3. Ensure Python version compatibility
4. Check requirements are installed: `pip list`

### Suggested Improvements
- Train on larger datasets
- Include additional health metrics
- Implement cross-validation results
- Add confidence intervals
- Include explainability features (SHAP values)

## 📄 File Descriptions

### `app.py`
The main Streamlit application file containing:
- UI layout and styling
- User input handling
- Model prediction logic
- Data visualization
- Batch processing pipeline
- CSV export functionality

### `DiabetesML_model.joblib`
Serialized Random Forest classifier model trained on the Pima Indians dataset. Contains all decision trees and feature information necessary for predictions.

**Note**: This file was created using scikit-learn 1.6.1. Version mismatches may produce warnings but shouldn't affect functionality.

### `scaler.joblib`
Serialized StandardScaler object containing the mean and standard deviation values for each feature. Used to normalize input data before prediction.

### `requirements.txt`
List of all Python package dependencies with versions. Use with:
```bash
pip install -r requirements.txt
```

## 🎓 Learning Resources

### Understanding the Model
- [Random Forest in scikit-learn](https://scikit-learn.org/stable/modules/ensemble.html#random-forests)
- [Feature Scaling & Normalization](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Cross-validation for Model Evaluation](https://scikit-learn.org/stable/modules/cross_validation.html)

### Streamlit Documentation
- [Streamlit Official Docs](https://docs.streamlit.io/)
- [Streamlit Components](https://docs.streamlit.io/library/api-reference)

### Diabetes Information
- [Mayo Clinic - Diabetes](https://www.mayoclinic.org/diseases-conditions/diabetes/)
- [CDC Diabetes Info](https://www.cdc.gov/diabetes/)

## 📊 Example CSV for Batch Prediction

Create a file named `patients.csv`:

```csv
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age
6,148,72,35,0,33.6,0.627,50
1,85,66,29,0,26.6,0.351,31
8,183,64,0,0,23.3,0.672,32
1,89,66,23,94,28.1,0.167,21
0,137,40,35,168,43.1,2.288,33
5,116,74,0,0,25.6,0.201,30
3,78,50,32,88,31.0,0.248,26
```

## 🔐 Data Privacy & Security

- ✅ All processing happens locally on your machine
- ✅ No data is sent to external servers
- ✅ CSV files are processed in-memory and not stored
- ✅ No user tracking or analytics collected
- ⚠️ Downloaded results files should be handled securely
- ⚠️ Follow HIPAA/GDPR guidelines if handling real patient data

## 📝 License & Attribution

This project uses:
- **Pima Indians Diabetes Database**: Available from UCI Machine Learning Repository
- **Streamlit**: Open-source framework (Apache 2.0)
- **scikit-learn**: Open-source machine learning library (BSD 3-Clause)
- **Plotly**: Interactive visualization library

## 📅 Version History

### Version 2.0 (Current)
- ✨ Enhanced UI with gradient designs and modern styling
- 📁 Batch prediction functionality with CSV export
- 📊 Advanced analytics and feature importance visualization
- 🎨 Improved color contrast for accessibility
- 💾 Timestamp-based file downloads
- ⚠️ Comprehensive disclaimers and documentation

### Version 1.0 (Previous)
- Basic single prediction interface
- Simple feature input sliders
- Basic prediction output

## 🎉 Getting Started Checklist

- [ ] Python 3.8+ installed
- [ ] Project folder downloaded/cloned
- [ ] Virtual environment created (optional but recommended)
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Both .joblib files present in project directory
- [ ] Streamlit installed successfully (`streamlit --version`)
- [ ] App launched (`streamlit run app.py`)
- [ ] Browser opened to `http://localhost:8501`
- [ ] All tabs functional and responsive
- [ ] Can make single predictions
- [ ] Can upload and process batch CSV
- [ ] Can export results

---

**Last Updated**: April 9, 2026  
**Application Version**: 2.0  
**Status**: ✅ Production Ready

For questions or issues, refer to the Troubleshooting section or the embedded help tooltips in the application.

**Happy Predicting! 🏥✨**
