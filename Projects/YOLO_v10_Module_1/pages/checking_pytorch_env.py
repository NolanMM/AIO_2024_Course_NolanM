from pages.set_up_yolo_v10_module import set_up_yolo_v10_module
import streamlit as st
import torch


class checking_pytorch_env():
    def app(self, pages):
        # st.set_page_config(
        #     page_title="Checking PyTorch Environment", layout="wide")

        st.title("Checking PyTorch Environment")
        if "pytorch_checked" not in st.session_state:
            st.session_state.pytorch_checked = False
        if "pytorch_checking" not in st.session_state:
            st.session_state.pytorch_checking = False
        st.divider()

        st.markdown("""
            1. Check the PyTorch version.
            Using the code below, you can check the version of PyTorch installed in your environment.
                    """, unsafe_allow_html=True)

        checking_pytorch_env_code = """
            print("PyTorch version:", torch.__version__)
        """
        st.code(checking_pytorch_env_code, language="python")

        st.markdown("""
            2. Check if CUDA is available.
            Using the code below, you can check if CUDA is available in your environment.
                    """, unsafe_allow_html=True)

        checking_CUDA_env_code = """
            cuda_available = torch.cuda.is_available()
            print("CUDA available:", cuda_available)

            if cuda_available:
                print("CUDA version:", torch.version.cuda)

                """
        st.code(checking_CUDA_env_code, language="python")

        st.markdown("""
            3. Check the number of available GPUs.
            Using the code below, you can check the number of available GPUs in your environment.
                    """, unsafe_allow_html=True)

        checking_num_GPUs_code = """
                print("Number of GPUs available:", torch.cuda.device_count())
                print("Current GPU device name:", torch.cuda.get_device_name(
                    torch.cuda.current_device()))
        """
        st.code(checking_num_GPUs_code, language="python")
        checking_button = st.button("Check PyTorch Environment")
        if checking_button:
            st.session_state.pytorch_checking = True
        if st.session_state.pytorch_checking:
            st.write("PyTorch version:", torch.__version__)
            cuda_available = torch.cuda.is_available()
            st.write("CUDA available:", cuda_available)
            if cuda_available:
                st.write("CUDA version:", torch.version.cuda)
                st.write("Number of GPUs available:",
                         torch.cuda.device_count())
                st.write("Current GPU device name:",
                         torch.cuda.get_device_name(torch.cuda.current_device()))
                st.success("PyTorch environment is set up successfull")
                st.markdown("""
                    - Click the button below to Set Up YOLOv10.
                            """, unsafe_allow_html=True)
                st.divider()
                if st.button("Continue to YOLOv10 Setup"):
                    st.session_state.pytorch_checked = True
            else:
                st.session_state.pytorch_checked = False
                st.error(
                    "PyTorch is not available. Please set up PyTorch with CUDA support.")
        st.divider()

        st.markdown("""
            3. The full code to check the PyTorch environment is given below:
                    """, unsafe_allow_html=True)

        full_code = """
            print("PyTorch version:", torch.__version__)

            cuda_available = torch.cuda.is_available()
            print("CUDA available:", cuda_available)

            if cuda_available:
                print("CUDA version:", torch.version.cuda)
                print("Number of GPUs available:", torch.cuda.device_count())
                print("Current GPU device name:", torch.cuda.get_device_name(
                    torch.cuda.current_device()))
        """
        st.code(full_code, language="python")

        if st.session_state.pytorch_checked:
            st.divider()
            set_up_yolo_v10_module().app(pages)
            # st.page_link("./pages/set_up_yolo_v10_module.py",
            #              label="Continue Set Up YOLOv10...", icon="🔥")
