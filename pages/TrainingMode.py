from utils.build_interface import build_interface
from utils.training_mode import  *

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
st.title("Digital Twin App - Training Mode")
# Sidebar
st.sidebar.header("Available Working Modes: ")

# Working mode Radio
working_mode = st.sidebar.radio(
    "Select Working Mode", ['Training Mode'])
if working_mode == 'Training Mode':
    training_mode2()