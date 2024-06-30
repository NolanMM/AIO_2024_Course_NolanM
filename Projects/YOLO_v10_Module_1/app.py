from streamlit_extras.switch_page_button import switch_page
import streamlit as st
import Multipage
from pages import (
    checking_pytorch_env,
    set_up_yolo_v10_module,
)

# Create an instance of the app
app = Multipage.MultiPage()
st.set_page_config(page_title="YOLO_v10_Comprehensive",
                   page_icon=":computer:", layout="wide", initial_sidebar_state="collapsed")
# Title of the main page
st.title("Multi-Page App")

# Add your pages here
app.add_page("Page 1", checking_pytorch_env.checking_pytorch_env().app)

# Run the app
app.run()
