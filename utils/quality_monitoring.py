import streamlit as st
import numpy as np
import pandas as pd
import json
import os

from ultralytics import settings

import utils.settings
from utils.data_retrieval import get_start_time
from utils.settings import QUALITY_MONITORING

from utils.utils import *

defect_dict = {0:'componente', 1:'linea_giunzione', 2:'risucchio', 3:'macchia', 4:'sfiammatura', 5:'bava'}

def apply_computer_vision_model(image, model, conf):
    res = model.track(image, conf=conf, persist=True)
    return res
def plot_computer_vision_res(c_placeholder, res):
    res_plot = res[0].plot()
    c_placeholder.image(res_plot, caption='Detected Video', channels="BGR")

def update_acquisition_dict(res, acquisition_dictionary):
    acquisition_keys = list(acquisition_dictionary.keys())
    json_original_str = res[0].tojson()
    json_original = json.loads(json_original_str)
    for detection in json_original:
        conf = detection.get('conf', None)  # get confidence if present
        if detection['class'] == 0: # only components are chosen here
            try:
                if str(detection['track_id']) not in list(acquisition_dictionary.keys()): # new comp
                    acquisition_dictionary[str(detection['track_id'])] = {
                        'componente':1, 'linea_giunzione':0, 'risucchio':0, 'macchia':0, 'sfiammatura':0, 'bava':0,
                        'confidence': {'linea_giunzione':[], 'risucchio':[], 'macchia':[], 'sfiammatura':[], 'bava':[]}
                    }
                elif str(detection['track_id']) in list(acquisition_dictionary.keys()):
                    acquisition_dictionary[str(detection['track_id'])]['componente'] += 1  # one more comp identification
            except: pass
        if detection['class'] > 0: #defects
            def_box = detection['box'] #dictionary with the defective box
            defect_name = defect_dict[detection['class']]
            for det in json_original:
                if det['class']==0:
                    try:
                        if det['box']['x1'] < def_box['x1'] and det['box']['y1'] < def_box['y1']:
                            if det['box']['x2'] > def_box['x2'] and det['box']['y2'] > def_box['y2']:
                                acquisition_dictionary[str(det['track_id'])][defect_name] += 1
                                # Store confidence for this defect type
                                acquisition_dictionary[str(det['track_id'])]['confidence'][defect_name].append(conf)
                    except:pass
    if list(acquisition_dictionary.keys()) != acquisition_keys:
        acquisition_keys = list(acquisition_dictionary.keys())
    return acquisition_dictionary

def get_dataframe(acquisition_dictionary):
    # Flatten confidence lists to strings for each defect type
    flat_dict = {}
    for comp_id, comp_data in acquisition_dictionary.items():
        flat_comp = {}
        for key, value in comp_data.items():
            if key == 'confidence':
                for defect_type, conf_list in value.items():
                    flat_comp[f'confidence_{defect_type}'] = ','.join([str(c) for c in conf_list])
            else:
                flat_comp[key] = value
        flat_dict[comp_id] = flat_comp
    acquisition_dataframe = pd.DataFrame(flat_dict)
    for id_ in acquisition_dataframe:
        if acquisition_dataframe.loc['componente', id_] < 10:
            acquisition_dataframe = acquisition_dataframe.drop(id_, axis=1)
    return acquisition_dataframe

def plot_acquisition(c_placeholder, acquisition_dictionary):
    acquisition_dataframe = pd.DataFrame(acquisition_dictionary)
    for id_ in acquisition_dataframe:
        if acquisition_dataframe.loc['componente', id_] < 10:
            acquisition_dataframe = acquisition_dataframe.drop(id_, axis=1)
    container = c_placeholder.container(border=True)
    container.dataframe(acquisition_dataframe, use_container_width=True)

def quality_monitoring():
    # define model confidence
    confidence = float(st.sidebar.slider(
        "Select Computer Vision Model Confidence", 25, 100, 40)) / 100
    # load model
    model_name_list = os.listdir(utils.settings.WEIGHTS)
    model_name = st.sidebar.selectbox(
        "Choose a model...", model_name_list)
    model = load_model(model_name)

    # create device
    devices = create_devices_with_tries()
    device = system.select_device(devices)
    num_channels = setup(device)

    initialize_state(["run_stream", "acquisition_dictionary"], [False, {}])
    stream_container = st.container(border=True)
    stream_empty = stream_container.empty()
    acquisition_expander = stream_container.expander("Acquisitions")
    run_stream_button = stream_container.button("Run Stream")
    delete_state_button = stream_container.button("Delete Session")
    if delete_state_button:
        delete_state(["acquisition_dictionary"], [{}])
    if run_stream_button:
        st.session_state["run_stream"] = not st.session_state.run_stream

    while st.session_state.run_stream:
        with device.start_stream():
            while True:
                buffer = device.get_buffer()
                item = BufferFactory.copy(buffer)
                device.requeue_buffer(buffer)
                buffer_bytes_per_pixel = int(len(item.data) / (item.width * item.height))
                array = (ctypes.c_ubyte * num_channels * item.width * item.height).from_address(
                    ctypes.addressof(item.pbytes))
                npndarray = np.ndarray(buffer=array, dtype=np.uint8,
                                       shape=(item.height, item.width, buffer_bytes_per_pixel))
                res = apply_computer_vision_model(npndarray, model, confidence)
                st.session_state.acquisition_dictionary = update_acquisition_dict(res, st.session_state.acquisition_dictionary)
                plot_computer_vision_res(stream_empty, res)
                BufferFactory.destroy(item)
        device.stop_stream()
    system.destroy_device()
    plot_acquisition(acquisition_expander, st.session_state.acquisition_dictionary)
    # save session
    save_session_button = stream_container.button("Save Session")
    if save_session_button:
        df_to_save = get_dataframe(st.session_state.acquisition_dictionary)
        curr_time, curr_date = get_start_time()
        curr_date = str(curr_date).replace("-","")
        curr_time = str(curr_time).replace(":","-")
        save_results(os.path.join(QUALITY_MONITORING, curr_date + "_" + curr_time +'.csv'), df_to_save)






