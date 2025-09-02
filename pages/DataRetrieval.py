from utils.build_interface import build_interface
from utils.data_retrieval import *

# Setting page layout
st.set_page_config(
    page_title="Digital Twin App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# markdown
build_interface()
# Main page heading
st.title("Digital Twin App - Data Retrieval")
# Sidebar
st.sidebar.header("Available Working Modes: ")

# Working mode Radio
working_mode = st.sidebar.radio(
    "Select Working Mode", ['Hystorical Data', 'Real Time Data'])
if working_mode == 'Hystorical Data':
    retrieve_data_1()
if working_mode == 'Real Time Data':
    retrieve_data_2()