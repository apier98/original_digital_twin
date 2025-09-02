import streamlit as st

def build_interface():
    st.markdown("""
        <style>
            /* Main app background */
            .main {
                background-color: #ffffff;
                font-family: 'Arial', sans-serif;
                color: #333333;
            }

            /* Sidebar background and text styling */
            [data-testid="stSidebar"] {
                background-color: #f8f9fa; /* Light gray for sidebar */
                border-right: 1px solid #dcdcdc; /* Subtle border for sidebar */
            }

            [data-testid="stSidebar"] .css-1d391kg, 
            [data-testid="stSidebar"] .css-1lcbmhc, 
            [data-testid="stSidebar"] .css-1cpxqw2, 
            [data-testid="stSidebar"] .css-1eth2k1 {
                color: #4a4a4a; /* Dark gray text for sidebar */
            }

            /* Sidebar title styling */
            [data-testid="stSidebar"] h1, 
            [data-testid="stSidebar"] h2, 
            [data-testid="stSidebar"] h3 {
                color: #333333; /* Darker text for headers */
            }

            /* Buttons */
            button {
                background-color: #0066cc !important; /* Nice blue color */
                color: white !important;
                border-radius: 8px !important;
                border: none !important;
                padding: 8px 16px !important;
                font-size: 16px !important;
            }

            button:hover {
                background-color: #005bb5 !important; /* Slightly darker blue on hover */
            }

            /* Input fields */
            input, textarea, select {
                background-color: #ffffff !important; /* White input background */
                border: 1px solid #dcdcdc !important;
                border-radius: 6px !important;
                padding: 8px !important;
                font-size: 14px !important;
            }

            input:focus, textarea:focus, select:focus {
                border-color: #0066cc !important;
                box-shadow: 0 0 5px rgba(0, 102, 204, 0.3) !important;
            }

            /* Header styles */
            h1 {
                color: #0066cc; /* Main header in blue */
                font-size: 2.5em;
            }

            h2 {
                color: #333333;
                font-size: 2em;
            }

            h3 {
                color: #4a4a4a;
                font-size: 1.5em;
            }

            /* Footer styles */
            footer {
                text-align: center;
                font-size: 0.9em;
                color: #6c757d;
                margin-top: 20px;
            }
        </style>
    """, unsafe_allow_html=True)