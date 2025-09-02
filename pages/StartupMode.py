from utils.build_interface import build_interface
from utils.startup_mode import  *

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
st.title("Digital Twin App - Startup Mode")
# Sidebar
st.sidebar.header("Available Working Modes: ")

# Working mode Radio
working_mode = st.sidebar.radio(
    "Select Working Mode", ['Modeling', 'Molding Window', 'Suggestion', 'Suggestion From Frames'])
if working_mode == 'Modeling':
    ps_sl_main_modeling()
elif working_mode == 'Molding Window':
    ps_sl_main_mw()
elif working_mode == 'Suggestion':
    ps_sl_main_s()
elif working_mode == 'Suggestion From Frames':
    ps_sl_main_ff()



#"""
#-----Modeling: addestrare il modello predittivo a partire dai Dataframe
#---dataframe: dati ottenuti da Training Mode (dictionary-difetti + dictionary-parametri processo)
#
#-----Molding Window:
# """