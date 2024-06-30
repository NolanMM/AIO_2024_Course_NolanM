from pages.Helpers.Document_remake import Document
from pages.safe_helmet_project import SafeHelmetProject
import streamlit as st


class set_up_yolo_v10_module():
    def app(self, pages):
        if "yolo_v10_set_up_render_done" not in st.session_state:
            st.session_state.yolo_v10_set_up_render_done = False
        doc = Document()
        st.title("Set Up YOLOv10")

        st.markdown("---")

        doc.intro_render()

        st.markdown("---")

        doc.yolo_v10_set_up_render()
        if st.session_state.yolo_v10_set_up_render_done:
            st.markdown("---")
            doc.test_yolov10_link_render(SafeHelmetProject())
