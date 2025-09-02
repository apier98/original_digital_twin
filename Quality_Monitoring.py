import os
import streamlit as st
from utils import settings
from utils import helper
from utils import helper_2

# Setting page layout
st.set_page_config(
    page_title="Digital Twin App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Digital Twin App - Computer Vision Model")

# Sidebar
st.sidebar.header("Available Options: ")

model_confidence = float(st.sidebar.slider(
    "Select Computer Vision Model Confidence", 25, 100, 40)) / 100

# Working mode Radio
working_mode_radio = st.sidebar.radio(
    "Select Working Mode", settings.WORKING_MODE_LIST)

if working_mode_radio == settings.QUALITY_MONITORING:
    # Model loading
    #model_path = settings.DETECTION_MODEL_PLACCA
    # load the model
    try:
        model, model_path = helper.load_rtdetr_model(settings.MODEL_DIR)
        if model: st.info(f"The model has been uploaded! --> {os.path.basename(model_path)}")
    except: st.error(f"Unable to load model")

    qm_source_radio = st.sidebar.radio('Select Quality Monitoring Source:', settings.QUALITY_MONITORING_SOURCES)

    if qm_source_radio == settings.WEBCAM:
        # webcam player
        helper_2.display_live_stream(model, model_confidence)
    if qm_source_radio == settings.VIDEOS:
        # video player
        helper_2.play_stored_video(model_confidence, model)
    ############################################################
    if qm_source_radio == settings.USER_TRIAL:
        helper_2.user_cycle_trial()
    ############################################################

if working_mode_radio == settings.TRAINING:
    st.info('Retrieve params and couple with defects!')
    #helper.training_mode(settings.TRAINING_MODE_LIST)
    # import the model
    model, model_path = helper.load_rtdetr_model(settings.MODEL_DIR)
    helper_2.training_mode_webcam(model_confidence, model)






