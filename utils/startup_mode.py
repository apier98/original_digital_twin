import os
from ast import Index
from datetime import datetime, timedelta

from skmultilearn.problem_transform import BinaryRelevance, ClassifierChain
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss, precision_score, recall_score, f1_score, jaccard_score
import numpy as np
from utils.data_retrieval import get_start_time, query_server
import pickle
import pandas as pd
import pyDOE
import streamlit as st
from utils.quality_monitoring import *
from utils.settings import *
from utils.utils import *

def retrieve_date(c_placeholder):
    datetime = c_placeholder.date_input("Date Filter", value="today")
    return datetime
def postprocess_datetime(datetime):
    return str(datetime).replace("-", "")
def df_to_load(c_placeholder, coupled_df_folder, datetime, date_filter):
    if date_filter:
        user_choice = c_placeholder.multiselect("Select Datasets", [data_name for data_name in os.listdir(coupled_df_folder) if datetime in data_name])
    else: user_choice = c_placeholder.multiselect("Select Datasets", [data_name for data_name in os.listdir(coupled_df_folder)])
    return user_choice
def get_global_defect_dict():
    return {0: 'componente',1: 'linea_giunzione', 2: 'risucchio',3 : 'macchia', 4: 'sfiammatura', 5: 'bava'}
def concatenate_df_to_load(df_list, targets):
    df_list = [pd.read_csv(os.path.join(TRANINING_MODE, df_name), sep=';') for df_name in df_list]
    concatenated = pd.concat(df_list, axis=0)
    if 'index' in concatenated.columns:
        concatenated = concatenated.drop(columns='index')
    #concatenated = concatenated.dropna()
    for defect_name in targets:
        concatenated[defect_name] = concatenated[defect_name].apply(lambda x: 1 if x > 5 else 0)
    return concatenated
def get_ft_from_df(coupled_df, targets):
    features = []
    for col in coupled_df.columns:
        if col not in targets and col not in ['componente', 'id_stamp', 'Cavities', 'TimeStamp', 'Quantity',
                                              'ActualOrderCode']:
            features.append(col)
    return features
def post_process_ft(features):
    allowed = ["V201", "V202", "V203", "V204", "V205", "S201", "S202", "S203", "S204", "S205", "P12", "P118", "P119", "TH320.1", "TH401", "TH402", "TH403"]
    return [all_ for all_ in features if all_ in allowed]
def save_model(model):
    curr_time, curr_date = get_start_time()
    curr_date = str(curr_date).replace("-", "")
    curr_time = str(curr_time).replace(":", "-")
    with open(os.path.join(PREDICTION_MODELS, 'Model'+curr_date+'_'+curr_time+'.pkl'), 'wb') as file:
        pickle.dump(model, file)
def compute_metrics(y_true, y_pred):
    exact_match_score = accuracy_score(y_true, y_pred)
    hamming_loss_value = hamming_loss(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='micro')
    recall = recall_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    jaccard = jaccard_score(y_true, y_pred, average='samples')
    return exact_match_score, hamming_loss_value, precision, recall, f1, jaccard
def ps_sl_main_modeling():
    #classifier = BinaryRelevance(classifier=RandomForestClassifier(), require_dense=[False, True])
    #classifier = BinaryRelevance(classifier=XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    classifier = ClassifierChain(classifier=XGBClassifier(use_label_encoder=False, eval_metric='logloss'))

    # get global dict
    st.session_state.global_defect = get_global_defect_dict()
    container_ = st.container(border=True)
    container_1 = container_.container(border=True)

    date_filter = st.sidebar.selectbox("Date Filter", [True, False])
    datetime = retrieve_date(container_1)
    datetime = postprocess_datetime(datetime)
    to_load = df_to_load(container_1, TRANINING_MODE, datetime, date_filter)

    if to_load:
        targets = [defect_name for defect_name in st.session_state.global_defect.values() if
                   defect_name != 'componente']
        concatenated = concatenate_df_to_load(to_load, targets)
        container_1.dataframe(concatenated)
        # features
        features = get_ft_from_df(concatenated, targets)
        features = post_process_ft(features)

        X = concatenated[features]  # features
        Y = concatenated[targets]  # targets
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        col1_cont, col2_cont = container_1.columns(2)
        train_model_button = col1_cont.button("Train Model")
        save_model_button = col2_cont.button("Save Model")
        if train_model_button:
            classifier = classifier.fit(X_train, Y_train)
            st.session_state.classifier = classifier # send to session state
            col1_cont.success(f"Model Trained\n\nModel count is there? {hasattr(classifier, 'model_count_')}")
            expander_1 = col1_cont.expander("See Training Metrics")
            Y_pred = classifier.predict(X_test)
            Y_pred_dense = Y_pred.toarray()
            exact_match_score, hamming_loss_value, precision, recall, f1, jaccard = compute_metrics(Y_test, Y_pred_dense)
            expander_1.info(f"**Exact Match Score:** {exact_match_score}"
                            f"\n\n**Hamming Loss:** {hamming_loss_value}\n\n"
                            f"\n\n**Precision:** {precision}\n\n"
                            f"\n\n**Recall:** {recall}\n\n"
                            f"\n\n**F1 value:** {f1}\n\n"
                            f"\n\n**Jaccard:** {jaccard}\n\n")
        if save_model_button:
            save_model(st.session_state.classifier)
            col2_cont.container(border=True).success(f"Model Saved\n\nModel count is there? {hasattr(st.session_state.classifier, 'model_count_')}")

########################################
def get_reference_df(c_placeholder, targets, datetime, date_filter):
    #df_choice = c_placeholder.multiselect("Select Reference Data", list(os.listdir(TRANINING_MODE)))
    if date_filter:
        try:
            df_choice = c_placeholder.multiselect("Select Reference Datasets",
                                                [data_name for data_name in os.listdir(TRANINING_MODE) if datetime in data_name],
                                              default=[data_name for data_name in os.listdir(TRANINING_MODE) if datetime in data_name])
        except IndexError:
            c_placeholder.error(f"No Dataset Build on {datetime}")
            df_choice = None
    else:
        df_choice = c_placeholder.multiselect("Select Reference Datasets",
                                              [data_name for data_name in os.listdir(TRANINING_MODE)],
                                              default=[data_name for data_name in os.listdir(TRANINING_MODE)][0])
    if df_choice:
        reference_df = concatenate_df_to_load(df_choice, targets)
        #reference_df = pd.read_csv(os.path.join(TRANINING_MODE, df_choice), sep=";")
        for defect_name in targets:
            #reference_df[defect_name] = reference_df[defect_name].apply(lambda x: 1 if x > 5 else 0)
            reference_df[defect_name] = reference_df[defect_name].apply(lambda x: 1 if x == 1 else 0)
        return reference_df
def load_pred_model(c_placeholder, datetime, date_filter):
    if date_filter:
        p_model_list = [model_name for model_name in os.listdir(PREDICTION_MODELS) if datetime in model_name]
    else:
        p_model_list = [model_name for model_name in os.listdir(PREDICTION_MODELS)]
    model_name = c_placeholder.selectbox("Select Prediction Model...", p_model_list)
    with open(os.path.join(PREDICTION_MODELS, model_name), "rb") as file:
        model = pickle.load(file)
    return model
def ps_sl_main_mw():
    container_1 = st.container(border=True)
    datetime = retrieve_date(container_1)
    datetime = postprocess_datetime(datetime)
    date_filter = st.sidebar.selectbox("Date Filter", [True, False])
    try:
        st.session_state.pred_model = load_pred_model(container_1, datetime, date_filter)
    except TypeError: container_1.error("Remove Data Filter")
    st.session_state.global_defect = get_global_defect_dict()
    targets = [defect_name for defect_name in st.session_state.global_defect.values() if defect_name != 'componente']

    col1, col2 = container_1.columns(2)
    # select a dataset just for reference
    reference_df = get_reference_df(col1, targets, datetime, date_filter)
    ref_expander = col1.expander("Reference DF")
    if reference_df is not None:
        ref_expander.dataframe(reference_df)
        # get features
        features = get_ft_from_df(reference_df, targets)
        features = post_process_ft(features)
        mean_features = reference_df[features].mean()
        # build dictionary to dinamically allow the user store number values
        number_input_dict = {}

        for ft in features:
            try:
                number_input_dict[ft] = col1.number_input(f"Enter {ft} value", value=mean_features[ft], step=0.5, max_value=2000.0, min_value=0.0)
            except st.errors.StreamlitValueAboveMaxError: col1.error("error")

        # build the array
        #prediction_input = np.array([value for value in number_input_dict.values()]).reshape(-1,1)
        prediction_input = pd.DataFrame(number_input_dict, index=[0])
        predictions = st.session_state.pred_model.predict(prediction_input)
        predictions_dense = predictions.toarray()
        def_indexes = np.where(predictions_dense[0] == 1)[0]
        # show predictions
        container_2 = col2.container(border=True)
        container_2.markdown("""
        ### Expected Result: 
        """)
        predictions_list = []
        if def_indexes.shape[0] == 0:
            container_2.success("Componente Conforme")
        for ind in def_indexes:
            predictions_list.append(st.session_state.global_defect[ind+1])
            container_2.error(st.session_state.global_defect[ind+1])

########################################
def make_predictions(scaled_lhs, features, model):
    lhs_data = pd.DataFrame(scaled_lhs, columns=features)
    predictions = model.predict(lhs_data)
    proba = model.predict_proba(scaled_lhs)
    predictions_dense = predictions.toarray()
    proba_dense = proba.toarray()
    return predictions_dense, proba_dense
def build_dataframe(features, mw_dict, par_to_remove='max'):
    mw_df = pd.DataFrame(mw_dict, index=[0])
    for ft in features:
        mw_df = mw_df.drop(columns=[ft + '_' + par_to_remove])
    return mw_df
def sort_predictions(input_data, predictions, prediction_probas):
    proba_sum = prediction_probas.sum(axis=1)
    sort_idx = np.argsort(proba_sum)
    proba_s = prediction_probas[sort_idx]
    predictions_s = predictions[sort_idx]
    input_data_s = input_data[sort_idx]
    return input_data_s, predictions_s, proba_s
def ps_sl_main_s():
    container_1 = st.container(border=True)
    datetime = retrieve_date(container_1)
    datetime = postprocess_datetime(datetime)
    date_filter = st.sidebar.selectbox("Date Filter", [True, False])
    try:
        st.session_state.pred_model = load_pred_model(container_1, datetime, date_filter)
    except TypeError: st.sidebar.error("Remove Date Filter")
    st.session_state.global_defect = get_global_defect_dict()
    targets = [defect_name for defect_name in st.session_state.global_defect.values() if defect_name != 'componente']
    col1, col2 = container_1.columns(2)
    # select a dataset just for reference
    reference_df = get_reference_df(col1, targets, datetime, date_filter)
    ref_expander = col1.expander("Reference DF")
    col1, col2 = st.columns(2)
    if reference_df is not None:
        ref_expander.dataframe(reference_df)
        features = get_ft_from_df(reference_df, targets)
        features = post_process_ft(features)
        max_features = reference_df[features].max()
        min_features = reference_df[features].min()
        container1_1 = col1.container(border=True)
        mw_edges = {}
        for ft in features:
            col_min, col_max = container1_1.columns(2)
            mw_edges[ft + '_min'] = col_min.number_input(label=f"{ft}_min", value=min_features[ft])
            mw_edges[ft + '_max'] = col_max.number_input(label=f"{ft}_max", value=max_features[ft])

        # build min and max dataframe
        min_dataframe = build_dataframe(features, mw_edges, par_to_remove='max')
        max_dataframe = build_dataframe(features, mw_edges, par_to_remove='min')
        min_array = min_dataframe.values
        max_array = max_dataframe.values
        run_inference = container1_1.button("Run Inference")
        suggestion_pt = 100000
        lhs = pyDOE.lhs(len(features), suggestion_pt)
        lhs_ = min_array + (max_array - min_array) * lhs
        if run_inference:
            predictions, probas = make_predictions(lhs_, features, st.session_state.pred_model)
            lhs_s, predictions_s, proba_s = sort_predictions(lhs_, predictions, probas)
            input_data_s = pd.DataFrame(lhs_s, columns=features)
            predictions_s_df = pd.DataFrame(predictions_s, columns=targets)
            pred_proba_s_df = pd.DataFrame(proba_s, columns=targets)
            container2_1 = col2.container(border=True)
            container2_1.markdown("""
            ### PREDICTIONS
            """)
            container2_1.markdown("""**Process Parameters**""")
            container2_1.dataframe(input_data_s.iloc[:3].round(1).T, use_container_width=True)
            container2_1.markdown("""**Model Confidence**""")
            container2_1.dataframe(pred_proba_s_df.iloc[:3].T, use_container_width=False)
########################################
def camera_acquisition_startup(c_placeholder, confidence, model, device, num_channels):
    stream_empty = c_placeholder.empty()
    detected_defect = None
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
                for id_track in st.session_state.acquisition_dictionary:
                    for def_name in st.session_state.acquisition_dictionary[id_track]:
                        if def_name != "componente":
                            if st.session_state.acquisition_dictionary[id_track][def_name] > 10:
                                detected_defect = def_name
                                break
                    if detected_defect: break
                if detected_defect:
                    # main loop
                    break
        device.stop_stream()
        if detected_defect:
            delete_state(["run_stream", "acquisition_dictionary"], [False, {}])
            break
    system.destroy_device()
    return detected_defect

def make_suggestions(c_placeholder, prediction_model, reference_df, targets, features, pp_df):
    with st.spinner("Inference..."):
        time.sleep(2)
        # build molding window
        max_features = reference_df[features].max()
        min_features = reference_df[features].min()
        mw_edges = {}
        for ft in features:
            mw_edges[ft + '_min'] = min_features[ft]
            mw_edges[ft + '_max'] = max_features[ft]
        min_dataframe = build_dataframe(features, mw_edges, par_to_remove='max')
        max_dataframe = build_dataframe(features, mw_edges, par_to_remove='min')
        min_array = min_dataframe.values
        max_array = max_dataframe.values
        suggestion_pt = 10000
        lhs = pyDOE.lhs(len(features), suggestion_pt)
        lhs_ = min_array + (max_array - min_array) * lhs
        # run inference
        predictions, probas = make_predictions(lhs_, features, prediction_model)
        lhs_s, predictions_s, proba_s = sort_predictions(lhs_, predictions, probas)
        input_data_s = pd.DataFrame(lhs_s, columns=features)
        predictions_s_df = pd.DataFrame(predictions_s, columns=targets)
        pred_proba_s_df = pd.DataFrame(proba_s, columns=targets)
        c_placeholder.markdown("""
        ### SUGGESTIONS
        """)
        #c_placeholder.markdown("""**Process Parameters**""")
        #c_placeholder.dataframe(input_data_s.iloc[0].round(1), use_container_width=True)

        modification = pd.concat([pp_df[features].iloc[0], input_data_s.iloc[0]], axis=1)
        modification.columns = ['Current', 'Proposed']
        modification["Delta"] = np.array(modification['Current']).astype(np.float32) - np.array(modification["Proposed"]).astype(np.float32)
        modification["Delta"] = modification["Delta"].apply(lambda x: "Alzare di " + str(-x) if x < 0 else "Abbassare di " + str(x))
        c_placeholder.markdown("""**Modification**""")
        c_placeholder.dataframe(modification, use_container_width=True)

        c_placeholder.markdown("""**Model Confidence**""")
        c_placeholder.dataframe(pred_proba_s_df.iloc[0], use_container_width=False)

def get_start_time_st():
    # Get current time and date from the system clock
    dt = datetime.now()  - timedelta(hours=1)
    start_date = dt.date()
    start_time = dt.strftime("%H:%M:%S")  # Format as HH:MM:SS
    return start_time, start_date

def ps_sl_main_ff():
    col1, col2 = st.columns(2)
    container_1 = col1.container(border=True)
    container_2 = col2.container(border=True)
    st.session_state.global_defect = get_global_defect_dict()
    targets = [defect_name for defect_name in st.session_state.global_defect.values() if defect_name != 'componente']
    date_filter = st.sidebar.selectbox("Date Filter", [True, False])
    datetime = retrieve_date(container_1)
    datetime = postprocess_datetime(datetime)
    st.session_state.reference_df = get_reference_df(container_2, targets, datetime, date_filter)
    try:
        features = get_ft_from_df(st.session_state.reference_df, targets)
        features = post_process_ft(features)
    except AttributeError: container_2.error("No dataset retrieved, so unable to load features")
    # create device
    devices = create_devices_with_tries()
    device = system.select_device(devices)
    num_channels = setup(device)
    confidence = float(st.sidebar.slider(
        "Select Computer Vision Model Confidence", 25, 100, 40)) / 100
    model_name_list = os.listdir(utils.settings.WEIGHTS)
    model_name = st.sidebar.selectbox(
        "Select Computer Vision Model...", model_name_list)
    st.session_state.model = load_model(model_name)
    machine_input = st.sidebar.text_input("Input Machine", value="01936")
    try:
        st.session_state.prediction_model = load_pred_model(st.sidebar, datetime, date_filter)
    except TypeError: st.sidebar.error("Remove Date Filter")

    # initialize states
    initialize_state(["acquisition_dictionary", "run_stream"], [{}, False])
    stream_button = container_1.button('Start Streaming')
    if stream_button:
        st.session_state['run_stream'] = not st.session_state.run_stream
        delete_state(["acquisition_dictionary"], [{}])
    if st.session_state.run_stream:
        try:
            detected_defect = camera_acquisition_startup(container_1, confidence, st.session_state.model, device, num_channels)
            if detected_defect:
                container_1.error(detected_defect)
                # retrieve last process data
                start_time, start_date = get_start_time_st()
                pp_df = query_server(machine_input, start_date, start_time, "", "", end_=False)
                #container_2.dataframe(pp_df[features].iloc[0])

                make_suggestions(container_2, st.session_state.prediction_model, st.session_state.reference_df, targets,
                                 features, pp_df)
            else: container_1.success("No detected defect")

        except Exception as e:
            st.sidebar.error("Error loading camera: " + str(e))
            st.write(e)




