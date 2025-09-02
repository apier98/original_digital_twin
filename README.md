# Digital Twin Fork

## Project Overview
This project is a Digital Twin application for monitoring, analyzing, and optimizing industrial processes using computer vision and predictive modeling. The application is built with Python and Streamlit, providing an interactive web interface for users to:
- Monitor quality in real-time using computer vision models
- Retrieve and analyze historical and real-time process data
- Train and deploy predictive models for process optimization
- Suggest optimal process parameters based on data-driven inference

## Directory Structure

```
main.py
Quality_Monitoring.py
modifiche.txt
requirements.txt
shortcut_to_run_app.bat
pages/
    DataRetrieval.py
    StartupMode.py
    TrainingMode.py
utils/
    build_interface.py
    data_retrieval.py
    quality_monitoring.py
    settings.py
    startup_mode.py
    training_mode.py
    utils.py
    __pycache__/
data/
    datasets/
        process_parameters/
        quality_monitoring/
        training_mode/
    prediction_models/
    weights/
```

## Main Components

### 1. Entry Points
- `main.py`: Launches the Quality Monitoring interface.
- `Quality_Monitoring.py`: Main page for computer vision-based quality monitoring.
- `pages/`: Contains Streamlit pages for different modes:
  - `DataRetrieval.py`: Retrieve historical and real-time process data.
  - `StartupMode.py`: Model training, Molding Window analysis, and suggestions.
  - `TrainingMode.py`: Train computer vision models and manage acquisition sessions.

### 2. Utilities (`utils/`)
- `build_interface.py`: Sets up the Streamlit UI layout and navigation.
- `data_retrieval.py`: Functions for loading and processing process data.
- `quality_monitoring.py`: Computer vision defect detection, result plotting, and acquisition management.
- `settings.py`: Centralized configuration for data paths and constants.
- `startup_mode.py`: Predictive modeling, feature engineering, inference, and suggestions.
- `training_mode.py`: Camera acquisition, session management, and model training routines.
- `utils.py`: Helper functions for state management, device setup, model loading, and saving results.

### 3. Data (`data/`)
- `datasets/`: Raw and processed datasets for process parameters, quality monitoring, and training.
- `prediction_models/`: Saved predictive models (pickle files).
- `weights/`: Computer vision model weights (e.g., .pt files).

## Functionality Overview

### Quality Monitoring
- Real-time defect detection using computer vision (RTDETR models).
- Supports live webcam/video stream and user trial modes.
- Results are visualized and stored for further analysis.

### Data Retrieval
- Load historical or real-time process data from CSV files.
- Data coupling between process parameters and detected defects.

### Training Mode
- Camera acquisition and streaming for new data collection.
- Train computer vision models with new data.
- Manage acquisition sessions and save results.

### Startup Mode
- Train predictive models (Random Forest, XGBoost, etc.) on coupled datasets.
- Feature engineering and selection.
- Molding Window: Analyze process parameter ranges for optimal quality.
- Suggestion: Generate optimal process settings using model inference and Latin Hypercube Sampling.
- Save and load predictive models.

## How to Run
1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Start the app:
   ```powershell
   streamlit run main.py
   ```
   Or use `shortcut_to_run_app.bat` for quick launch.

## Notes
- All configuration paths are managed in `utils/settings.py`.
- Models and data are stored in the `data/` directory.
- The app is modular: each mode (Quality Monitoring, Training, Startup) is accessible via the sidebar.

## Authors
- Developed by Andrea and contributors.