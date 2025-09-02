# camera and other stuff
import pandas as pd
from ultralytics import RTDETR
import utils.settings
import streamlit as st
import time
import os

from arena_api.system import system
from arena_api.buffer import *
import ctypes

def initialize_state(var_list, init_state_list):
	for var, init_state in zip(var_list, init_state_list):
		if var not in st.session_state:
			st.session_state[var] = init_state

def delete_state(var_list, new_state_list):
	for var, new_state in zip(var_list, new_state_list):
		if var in st.session_state:
			st.session_state[var] = new_state

def save_results(file_path: str, file):
	if file_path.endswith('.csv'): #csv file
		if isinstance(file, pd.DataFrame):
			file.to_csv(file_path, sep=';', index=False)

def load_model(model_name: str):
    """load RTDETR model"""
    model_path = os.path.join(utils.settings.WEIGHTS, model_name)
    model = RTDETR(model_path)
    return model

def create_devices_with_tries():
	'''
	This function waits for the user to connect a device before raising
		an exception
	'''
	tries = 0
	tries_max = 6
	sleep_time_secs = 10
	while tries < tries_max:  # Wait for device for 60 seconds
		devices = system.create_device()
		if not devices:
			info_container = st.container(border=True)
			info_placeholder = info_container.empty()
			info_container.info(
				f'Try {tries+1} of {tries_max}: waiting for {sleep_time_secs} '
				f'secs for a device to be connected!')
			for sec_count in range(sleep_time_secs):
				time.sleep(1)
				info_placeholder.info(f'{sec_count + 1 } seconds passed ')
			tries += 1
		else:
			#st.info(f'Created {len(devices)} device(s)')
			return devices
	else:
		raise Exception(f'No device found! Please connect a device and run '
						f'the example again.')



def setup(device):
    """
    Setup stream dimensions and stream nodemap
        num_channels changes based on the PixelFormat
        Mono 8 would has 1 channel, RGB8 has 3 channels

    """
    nodemap = device.nodemap
    nodes = nodemap.get_node(['Width', 'Height', 'PixelFormat', 'OffsetX', 'OffsetY', 'ExposureAuto', 'BalanceWhiteAuto', 'GainAuto', 'Gain'])

    nodes['Width'].value = 2448//2
    nodes['Height'].value = 2048//2
    nodes['OffsetX'].value = 2048 // 4
    nodes['OffsetY'].value = 2448//4
    nodes['ExposureAuto'].value = 'Off'
    nodes['BalanceWhiteAuto'].value = 'Continuous' ############## Off
    nodes['GainAuto'].value = 'Off'
    nodes['Gain'].value = 5.0
    nodes['PixelFormat'].value = 'RGB8'

    num_channels = 3

    # Stream nodemap
    tl_stream_nodemap = device.tl_stream_nodemap

    tl_stream_nodemap["StreamBufferHandlingMode"].value = "NewestOnly"
    tl_stream_nodemap['StreamAutoNegotiatePacketSize'].value = True
    tl_stream_nodemap['StreamPacketResendEnable'].value = False

    return num_channels


