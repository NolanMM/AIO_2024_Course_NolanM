from multiprocessing.pool import ThreadPool

import requests
from pages.Helpers.Helpers import check_folder_exists
from pathlib import Path
import streamlit as st
import subprocess
import time
import os

PROCESS_TIME = 240

pool = ThreadPool(processes=4)

current_dir_ = Path(__file__).parent if "__file__" in locals() else Path.cwd()

current_dir = str(current_dir_.parent.parent)


class Document:
    def __init__(self):
        self.MODEL_PATH = None
        self.model = None
        self.yolov10_path = None
        self.dataset_path = None
        self.zip_file = None
        self.yolov10_model_file = None

        if "yolov10_path" not in st.session_state:
            st.session_state.yolov10_path = current_dir + "/yolov10"
            st.session_state.dataset_path = current_dir + "/safety_helmet_dataset"
            st.session_state.zip_file = current_dir + "/Safety_Helmet_Dataset.zip"
            st.session_state.yolov10_model_file = current_dir + "/yolov10n.pt"

        self.objective_text = """
        ## Objective
            Object Detection is a classical problem in the field of Computer Vision. The goal of this problem is to automatically identify the locations of objects within an image. It is one of the essential and complex problems in Computer Vision,
            with widespread applications ranging from facial recognition and license plate recognition to object tracking in videos and autonomous driving.
        """

        self.style_text = """
            <style>
                .st-emotion-cache-1v0mbdj.e115fcil1 {
                    margin-left: auto;
                    margin-right: auto;
                }
            </style>
        """

        self.image1_path = "./Assets/Fig1.png"
        self.image1_caption = "Figure 1: Object detection program identifying individuals wearing helmets. AIO-2024"

        self.project_text = """
            In this project, we will develop a program to detect whether employees are wearing safety helmets at construction sites. The model we will use is the YOLOv10 model. Accordingly, the Input and Output of the program are:

            **Input**: An image.
            <br>
            **Output**: The coordinates (bounding box) of the employees and the helmets.
        """

        self.process_text = """
        ## II. Process Steps
        """

        self.image2_path = "./Assets/Fig2.png"
        self.image2_caption = "Figure 2: Process Steps- AIO-2024"

        self.pipeline_text = """
        ## III. Pipeline
        """

        self.image3_path = "./Assets/Fig3.png"
        self.image3_caption = "Figure 3: Pipeline- AIO-2024"

        self.set_up_yolo_v10_text = """
        ## IV. Set up YOLOv10
        """
        self.set_up_yolo_v10_git_code_markdown = """
            1. Use the following code to clone the YOLOv10 repository:
        """
        self.set_up_yolo_v10_git_code = """
            !git clone https://github.com/THU-MIG/yolov10.git
        """
        self.set_up_yolo_v10_setup_code_markdown = """
            2. Use the following code to set up YOLOv10:
        """
        self.set_up_yolo_v10_setup_code = """
            cd yolov10
            !pip install -r requirements.txt
            !pip install -e .
        """
        self.download_yolo_v10_ver_code_markdown = """
            3. Use the following code to download the YOLOv10 model:
        """
        self.download_yolo_v10_ver_code = """
            !wget https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt
        """

        self.test_yolov10_link_page_markdown = """
            ## V. Test YOLOv10
        """

        self.project_implementation_text = """
            ## VI. Project Implementation
        """

    def intro_render(self):
        st.markdown(self.objective_text, unsafe_allow_html=True)
        st.markdown(self.style_text, unsafe_allow_html=True)
        st.image(self.image1_path, use_column_width=False, width=800,
                 output_format="PNG", caption=self.image1_caption)
        st.markdown(self.project_text, unsafe_allow_html=True)
        st.divider()
        st.markdown(self.process_text, unsafe_allow_html=True)
        st.image(self.image2_path, use_column_width=False, width=600,
                 output_format="PNG", caption=self.image2_caption)
        st.markdown(self.pipeline_text, unsafe_allow_html=True)
        st.image(self.image3_path, use_column_width=False, width=800,
                 output_format="PNG", caption=self.image3_caption)

    def git_clone_implementation(self):
        command_git_clone = ["git", "clone",
                             "https://github.com/THU-MIG/yolov10.git"]
        process_git_clone = subprocess.Popen(command_git_clone,
                                             stdout=subprocess.PIPE,
                                             stderr=subprocess.PIPE)
        return process_git_clone.communicate()

    def git_clone_step(self):
        st.markdown(self.set_up_yolo_v10_git_code_markdown,
                    unsafe_allow_html=True)
        st.code(self.set_up_yolo_v10_git_code, language="python")
        git_process_button = st.button("Run Git Clone Command")
        if git_process_button:
            if not check_folder_exists(st.session_state.yolov10_path):
                async_result = pool.apply_async(self.git_clone_implementation)
                bar = st.progress(0)
                per = 10 / 100
                for i in range(100):
                    time.sleep(per)
                    bar.progress(i + 1)
                output_git_clone, error_git_clone = async_result.get()
                if output_git_clone:
                    st.write("Output:")
                    st.code(output_git_clone.decode("utf-8"))
                    st.session_state.git_clone_done = True
                if error_git_clone:
                    st.session_state.git_clone_done = True
                    st.write("Output:")
                    st.code(error_git_clone.decode("utf-8"))
            else:
                st.session_state.git_clone_done = True
                st.write("The yolov10 folder already exists.")

    def set_up_yolov10_requirements_implementation_install_requirement(self):
        return run_command("pip install -r requirements.txt")

    def set_up_yolov10_requirements_implementation_set_up(self):
        return run_command("pip install -e .")

    def set_up_yolov10_requirements(self):
        st.markdown(self.set_up_yolo_v10_setup_code_markdown,
                    unsafe_allow_html=True)
        st.code(self.set_up_yolo_v10_setup_code, language="python")
        if st.button("Install YOLOv10 Requirements"):
            if check_folder_exists(st.session_state.yolov10_path):
                st.write("Installing YOLOv10 requirements...")
                # Change to the yolov10 directory and run pip install commands
                os.chdir(st.session_state.yolov10_path)
                st.write("Processing...")
                output_requirement_clone, error_requirement_clone = self.set_up_yolov10_requirements_implementation_install_requirement()
                if output_requirement_clone:
                    st.session_state.yolov10_requirements_done = True
                    st.write("Output (requirements.txt):")
                    st.code(output_requirement_clone)
                if error_requirement_clone:
                    st.session_state.yolov10_requirements_done = False
                    st.write("Error (requirements.txt):")
                    st.code(error_requirement_clone)
                st.write("Processing...")
                # output_requirement_clone_, _error_requirement_clone = self.set_up_yolov10_requirements_implementation_set_up()
                # if output_requirement_clone_:
                #     st.session_state.yolov10_requirements_done = True
                #     st.write("Output (requirements.txt):")
                #     st.code(output_requirement_clone_)
                # if _error_requirement_clone:
                #     st.session_state.yolov10_requirements_done = False
                #     st.write("Error (requirements.txt):")
                #     st.code(_error_requirement_clone)
                os.chdir(current_dir)
            else:
                yolo_v10_path = st.session_state.yolov10_path
                st.write("The folder '{}' does not exist. Please clone the repository first.".format(
                    yolo_v10_path))

    def download_yolov10_model(self):
        st.markdown(self.download_yolo_v10_ver_code_markdown,
                    unsafe_allow_html=True)
        st.code(self.download_yolo_v10_ver_code, language="python")
        if st.button("Download YOLOv10 Model"):
            if check_folder_exists(st.session_state.yolov10_model_file):
                st.write("The file '{}' already exists.".format(
                    st.session_state.yolov10_model_file))
                st.session_state.yolo_v10_set_up_render_done = True
            else:
                st.write("Downloading YOLOv10 model...")
                url = 'https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.pt'
                output_path = "./yolov10n.pt"
                try:
                    response = requests.get(url, stream=True)
                    response.raise_for_status()
                    with open(output_path, 'wb') as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    print(f'File downloaded and saved to {output_path}')
                    st.session_state.yolo_v10_set_up_render_done = True

                except requests.exceptions.HTTPError as errh:
                    print("Http Error:", errh)
                    st.session_state.yolo_v10_set_up_render_done = False
                    st.write("Http Error:", errh)

    def yolo_v10_set_up_render(self):
        if "git_clone_done" not in st.session_state:
            st.session_state.git_clone_done = False
        if "yolov10_requirements_done" not in st.session_state:
            st.session_state.yolov10_requirements_done = False

        st.markdown(self.set_up_yolo_v10_text, unsafe_allow_html=True)
        self.git_clone_step()
        if st.session_state.git_clone_done:
            self.set_up_yolov10_requirements()
            if st.session_state.yolov10_requirements_done:
                self.download_yolov10_model()

    def test_yolov10_link_render(self, safehelmetproject):
        if "yolo_v10_basic_render_mode" not in st.session_state:
            st.session_state.yolo_v10_basic_render_mode = False
        if "project_render_mode" not in st.session_state:
            st.session_state.project_render_mode = False
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(self.test_yolov10_link_page_markdown,
                        unsafe_allow_html=True)
            if st.button("Yolov10 Basic Implementation 🔥"):
                st.session_state.yolo_v10_basic_render_mode = True
                st.session_state.project_render_mode = False

        with col2:
            st.markdown(self.project_implementation_text,
                        unsafe_allow_html=True)
            if st.button("Yolov10 Safe Helmet Project 🔥"):
                st.session_state.project_render_mode = True
                st.session_state.yolo_v10_basic_render_mode = False

        if st.session_state.yolo_v10_basic_render_mode:
            st.write("Basic Implementation")
        if st.session_state.project_render_mode:
            safehelmetproject.project_implementation_render()


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    output, error = process.communicate()
    return output, error
