import streamlit as st

from utils.data_retrieval import *
from utils.quality_monitoring import *
from utils.settings import TRANINING_MODE
from utils.utils import *

def camera_acquisition(c_placeholder, confidence, model, device, num_channels):
    stream_empty = c_placeholder.empty()
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
    return st.session_state.acquisition_dictionary


def training_mode():
    col1, col2 = st.columns(2)
    # create device
    devices = create_devices_with_tries()
    device = system.select_device(devices)
    num_channels = setup(device)
    confidence = float(st.sidebar.slider(
        "Select Computer Vision Model Confidence", 25, 100, 40)) / 100
    model_name_list = os.listdir(utils.settings.WEIGHTS)
    model_name = st.sidebar.selectbox(
        "Choose a model...", model_name_list)
    model = load_model(model_name)
    initialize_state(['start_time', 'start_date', 'acquisition_dictionary'], [get_start_time()[0], get_start_time()[1], {}])
    container_1 = col1.container(border=True) #streaming
    container_2 = col2.container(border=True) # data coupling
    stream_button = container_1.button('Start Streaming')
    delete_state_button = container_1.button("Delete Session")
    if delete_state_button:
        delete_state(["acquisition_dictionary"], [{}])
    initialize_state(['run_stream'], [False])
    if stream_button:
        st.session_state['run_stream'] = not st.session_state.run_stream
    if st.session_state.run_stream:
        st.session_state.start_time, st.session_state.start_date = get_start_time()
        try: st.session_state.acquisition_dictionary = camera_acquisition(container_1, confidence, model, device, num_channels)
        except Exception as e:
            st.sidebar.error("Error loading camera: " + str(e))
            st.write(e)
    else: st.session_state.end_time, st.session_state.end_date = get_end_time()
    cv_expander = container_1.expander("CV retrieved defects")
    plot_acquisition(cv_expander, st.session_state.acquisition_dictionary)
    try:
        retrieval_info_container = st.container(border=True)
        retrieval_info_expander = retrieval_info_container.expander("Acquired PP")
        machine_input = retrieval_info_container.text_input("Insert machine name", value='01926')
        retrieval_info_container.info(st.session_state.start_date)
        retrieval_info_container.info(st.session_state.start_time)
        retrieval_info_container.info(st.session_state.end_date)
        retrieval_info_container.info(st.session_state.end_time)
        st.session_state.pp_df = query_server(
            machine_input,
            st.session_state.start_date,
            st.session_state.start_time,
            st.session_state.end_date,
            st.session_state.end_time
        )
        try:
            retrieval_info_expander.dataframe(st.session_state.pp_df, use_container_width=True)
        except: pass
    except NameError:
        retrieval_info_container.info('Run one cycle to get times!')

    if isinstance(st.session_state.pp_df, pd.DataFrame):
        # reverse the dataframe
        params_dataframe = st.session_state.pp_df.iloc[::-1].reset_index(drop=True)
        #params_dataframe = st.session_state.pp_df.copy().reset_index()
        acquisition_dataframe_t = pd.DataFrame(st.session_state.acquisition_dictionary).T.reset_index(drop=True)
        coupled_dataframe = pd.concat([acquisition_dataframe_t, params_dataframe], axis=1)
        coupled_dataframe = coupled_dataframe.dropna()
        container_2.info('Coupled results: ')
        container_2.dataframe(coupled_dataframe)
        # save coupled
        save_session_button = container_2.button("Save Session")
        if save_session_button:
            curr_time, curr_date = get_start_time()
            curr_date = str(curr_date).replace("-", "")
            curr_time = str(curr_time).replace(":", "-")
            if machine_input:
                save_results(os.path.join(TRANINING_MODE, "Machine" + machine_input + "_" + curr_date + "_" + curr_time + '.csv'), coupled_dataframe)
            else:
                save_results(os.path.join(TRANINING_MODE, curr_date + "_" + curr_time + '.csv'), coupled_dataframe)
    else:
        container_2.info('No data from machine')


def camera_acquisition2(c_placeholder, confidence, model, device, num_channels):
    stream_empty = c_placeholder.empty()
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
    return st.session_state.acquisition_dictionary


def training_mode2():
    col1, col2 = st.columns(2)
    # create device
    devices = create_devices_with_tries()
    device = system.select_device(devices)
    num_channels = setup(device)
    confidence = float(st.sidebar.slider(
        "Select Computer Vision Model Confidence", 25, 100, 40)) / 100
    model_name_list = os.listdir(utils.settings.WEIGHTS) #computer vision weights
    model_name = st.sidebar.selectbox(
        "Choose a model...", model_name_list)
    model = load_model(model_name)
    initialize_state(['start_time', 'start_date', 'acquisition_dictionary'], [get_start_time()[0], get_start_time()[1], {}])
    container_1 = col1.container(border=True) #streaming
    container_2 = col2.container(border=True) # data coupling
    stream_button = container_1.button('Start Streaming')
    delete_state_button = container_1.button("Delete Session") #delete current session
    if delete_state_button:
        delete_state(["acquisition_dictionary"], [{}])
    initialize_state(['run_stream'], [False])
    if stream_button:
        st.session_state['run_stream'] = not st.session_state.run_stream
    if st.session_state.run_stream:
        st.session_state.start_time, st.session_state.start_date = get_start_time()
        try: st.session_state.acquisition_dictionary = camera_acquisition2(container_1, confidence, model, device, num_channels)
        except Exception as e:
            st.sidebar.error("Error loading camera: " + str(e))
            st.write(e)
    else: st.session_state.end_time, st.session_state.end_date = get_end_time()
    cv_expander = container_1.expander("CV retrieved defects")
    plot_acquisition(cv_expander, st.session_state.acquisition_dictionary)
    try:
        retrieval_info_container = st.container(border=True)
        retrieval_info_expander = retrieval_info_container.expander("Acquired PP")
        machine_input = retrieval_info_container.text_input("Insert machine name", value='01926')
        retrieval_info_container.info(st.session_state.start_date)
        retrieval_info_container.info(st.session_state.start_time)
        retrieval_info_container.info(st.session_state.end_date)
        retrieval_info_container.info(st.session_state.end_time)
        st.session_state.pp_df = query_server(
            machine_input,
            st.session_state.start_date,
            st.session_state.start_time,
            st.session_state.end_date,
            st.session_state.end_time
        )
        try:
            retrieval_info_expander.dataframe(st.session_state.pp_df, use_container_width=True)
        except: pass
    except NameError:
        retrieval_info_container.info('Run one cycle to get times!')

    if isinstance(st.session_state.pp_df, pd.DataFrame):
        # reverse the dataframe
        params_dataframe = st.session_state.pp_df.iloc[::-1].reset_index(drop=True)
        #params_dataframe = st.session_state.pp_df.copy().reset_index()
        # change confidence data type
        for obj_id in st.session_state.acquisition_dictionary:
            for acq_field in st.st.session_state.acquisition_dictionary[obj_id]:
                if "confidence" in acq_field:
                    st.st.session_state.acquisition_dictionary[obj_id][acq_field] = str(st.st.session_state.acquisition_dictionary[obj_id][acq_field])
        acquisition_dataframe_t = pd.DataFrame(st.session_state.acquisition_dictionary).T.reset_index(drop=True)
        coupled_dataframe = pd.concat([acquisition_dataframe_t, params_dataframe], axis=1)
        coupled_dataframe = coupled_dataframe.dropna()
        container_2.info('Coupled results: ')
        container_2.dataframe(coupled_dataframe)
        # save coupled
        save_session_button = container_2.button("Save Session")
        if save_session_button:
            curr_time, curr_date = get_start_time()
            curr_date = str(curr_date).replace("-", "")
            curr_time = str(curr_time).replace(":", "-")
            if machine_input:
                save_results(os.path.join(TRANINING_MODE, "Machine" + machine_input + "_" + curr_date + "_" + curr_time + '.csv'), coupled_dataframe)
            else:
                save_results(os.path.join(TRANINING_MODE, curr_date + "_" + curr_time + '.csv'), coupled_dataframe)
    else:
        container_2.info('No data from machine')