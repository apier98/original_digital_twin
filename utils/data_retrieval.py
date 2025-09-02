import streamlit as st
import ntplib
from datetime import datetime
import requests
import pandas as pd
import time
import os

from utils.settings import PROCESS_PARAMETERS
from utils.utils import initialize_state, save_results

def get_start_time():
    # Get current time and date from the system clock
    dt = datetime.now()  # Use the system clock
    start_date = dt.date()
    start_time = dt.strftime("%H:%M:%S")  # Format as HH:MM:SS
    return start_time, start_date
def get_end_time():
    # Get current time and date from the system clock
    dt = datetime.now()  # Use the system clock
    end_date = dt.date()
    end_time = dt.strftime("%H:%M:%S")  # Format as HH:MM:SS
    return end_time, end_date

def get_start_time_():
    # get current time and date
    client = ntplib.NTPClient()
    response_time = client.request('pool.ntp.org')
    timestamp = response_time.tx_time
    dt = datetime.fromtimestamp(timestamp)
    start_date = dt.date()
    start_time = dt.strftime("%H:%M:%S")
    return start_time, start_date
def get_end_time_():
    client = ntplib.NTPClient()
    response_time = client.request('pool.ntp.org')
    timestamp = response_time.tx_time
    dt = datetime.fromtimestamp(timestamp)
    end_date = dt.date()
    end_time = dt.strftime("%H:%M:%S")
    return end_time, end_date

def query_server(machine_input, start_date, start_time, end_date, end_time, end_=True):
    #format_dict = {'Cavities': '{:.0f}', 'TimeStamp': '{}', 'Quantity': '{:.0f}', 'ActualOrderCode':'{}'}
    format_dict = {'Cavities': '{:.0f}', 'TimeStamp': '{}', 'Quantity': '{:.0f}'}
    url = 'http://stwivicc02.vimar.net/api/devices/PEaBtShQlLvExcCCZbPDVMxpMESq18GYVYfY7SJ7/telemetries'
    bearer_token = '1|xpnMND7Jxwj2jPcEaNLflL64OkjhzHEG82DEtzSq'
    if end_:
        query = {
            "query": {"data.MachineInfo.SapCode": machine_input},
            "since": str(start_date) + " " + str(start_time),
            "until": str(end_date) + " " + str(end_time),
            "limit": 100,
            "page": 1
        }
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    else:
        query = {
            "query": {"data.MachineInfo.SapCode": machine_input},
            "since": str(start_date) + " " + str(start_time),
            "limit": 100,
            "page": 1
        }
        headers = {
            'Authorization': f'Bearer {bearer_token}',
            'Content-Type': 'application/json'
        }
    params_d_list = []
    response = requests.post(url, json=query, headers=headers)
    json_data = response.json()
    stampate = json_data['data']
    for i, stamp in enumerate(stampate):
        custom_d = {}
        custom_d['id_stamp'] = i + 1
        custom_d['Cavities'] = int(stamp['data']['Cavities'])
        custom_d['TimeStamp'] = stamp['data']['Timestamp']
        custom_d['Quantity'] = float(stamp['data']['Quantity'])
        #custom_d['ActualOrderCode'] = int(stamp['data']['ActualOrderCode'])
        for par_n in stamp['data']['Params'].keys():
            custom_d[stamp['data']['Params'][par_n]['ParamName']] = float(stamp['data']['Params'][par_n]['Value'])
            format_dict[stamp['data']['Params'][par_n]['ParamName']] = '{:.2f}'
        retrieved_dataframe = pd.DataFrame(custom_d, index=[0])
        params_d_list.append(retrieved_dataframe)
    try:
        pp_df = pd.concat(params_d_list).set_index('id_stamp', inplace=False)
        for key, value in format_dict.items():
            pp_df[key] = pp_df[key].map(value.format)
        return pp_df
    except ValueError:
        pp_df = None
        return pp_df

def retrieve_data_1():
    """Retrieve Hystorical Data"""
    col1, col2 = st.columns(2)
    container_1 = col1.container(border=True)
    machine_input = container_1.text_input("Insert machine name", value='01926')
    start_date = container_1.text_input("Start date: ", value='2024-10-09')
    end_date = container_1.text_input("End date: ", value='2024-10-09')
    start_session = container_1.text_input("Session start: ", value='10:00:00')
    end_session = container_1.text_input("Session end: ", value='10:10:00')
    query_button = container_1.button('Query')
    if query_button:
        st.session_state.pp_df = query_server(machine_input, start_date, start_session, end_date, end_session)
    try:
        container_2 = col2.container(border=True)
        container_2.info('Numero di stampate: {}'.format(st.session_state.pp_df.shape[0]))
        container_2.dataframe(st.session_state.pp_df)
        save_session_button = container_2.button("Save Session")
        if save_session_button:
            curr_time, curr_date = get_start_time()
            curr_date = str(curr_date).replace("-", "")
            curr_time = str(curr_time).replace(":", "-")

            save_results(os.path.join(PROCESS_PARAMETERS, "Machine" + machine_input + "_" + curr_date + "_" + curr_time + '.csv'), st.session_state.pp_df)
    except AttributeError:
        container_1.error('No data retrieved')

def retrieve_data_2():
    """Retrieve Real Time Data"""
    default_retrieval_time = 6
    retrieval_time = st.sidebar.slider(label='Set Retrieval Time',
                                       min_value=1,
                                       max_value=40,
                                       value=default_retrieval_time,
                                       step=1)
    container_1 = st.container(border=True)
    machine_input = container_1.text_input("Insert machine name", value='01926')
    # get current time and date
    #client = ntplib.NTPClient()
    #response_time = client.request('pool.ntp.org')
    #timestamp = response_time.tx_time
    #dt = datetime.fromtimestamp(timestamp)
    #date = dt.date()
    #current_time = dt.strftime("%H:%M:%S")
    current_time, date = get_start_time()
    container_1.info('Current date:  {}'.format(date))
    container_1.info('Current time:  {}'.format(current_time))
    start_stop_button = container_1.button('Start/Stop acquisition')
    retrieval_placeholder = container_1.empty()
    initialize_state(["run_retrieval"], [False])
    if start_stop_button:
        st.session_state['run_retrieval'] = not st.session_state.run_retrieval
    while st.session_state.run_retrieval:
        pp_df = query_server(machine_input, date, current_time, end_date='', end_time='', end_=False)
        retrieval_placeholder.dataframe(pp_df)
        time.sleep(retrieval_time)
