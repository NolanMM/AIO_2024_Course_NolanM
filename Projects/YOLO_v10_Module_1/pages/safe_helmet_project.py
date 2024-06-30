from pages.Helpers.Helpers import check_folder_exists
from multiprocessing.pool import ThreadPool
from pathlib import Path
from io import StringIO
import streamlit as st
import subprocess
import zipfile
import gdown
import time
import sys
import os

PROCESS_TIME = 240

pool = ThreadPool(processes=4)

current_dir = str(
    Path(__file__).parent.parent if "__file__" in locals() else Path.cwd())


class SafeHelmetProject():
    def __init__(self):
        self.project_implementation_text = """
        ## VI. Project Implementation
        """
        st.write(current_dir)
        self.project_implementation_step_1_code_markdown = """
            1. Use the following code to clone the project repository:
        """
        self.project_implementation_step_1_code = """
            !gdown '1twdtZEfcw4ghSZIiPDypJurZnNXzMO7R'
            !mkdir safety_helmet_dataset
            !unzip -q '/Safety_Helmet_Dataset.zip' -d '/safety_helmet_dataset'
        """

        self.project_implementation_step_2_code_markdown = """
            2. Use the following code to import the necessary libraries:
        """
        self.project_implementation_step_2_code = """
            import os
            from ultralytics import YOLO

            MODEL_PATH = '/yolov10n.pt'

            model = YOLO(MODEL_PATH)
        """

        self.project_implementation_step_3_code_markdown = """
            3. Use the following code to train the model with **50 epochs** with the image size of **640**:
        """
        self.project_implementation_step_3_code = """
            YAML_PATH = '/safety_helmet_dataset/data.yaml'
            IMG_SIZE = 640
            EPOCHS = 50
            BATCH_SIZE = 256

            model.train(data=YAML_PATH, imgsz=IMG_SIZE,
                        epochs=EPOCHS, batch=BATCH_SIZE)
        """

        self.project_implementation_step_4_code_markdown = """
            4. Use the following code to validate/test the model:
        """
        self.project_implementation_step_4_code = """
            TRAINED_MODEL_PATH = '/yolov10/runs/detect/train/weights/best.pt'
            model = YOLO(TRAINED_MODEL_PATH)

            model.val(data=YAML_PATH, imgsz=IMG_SIZE, split='test')
        """

    def step_1_download_dataset_implementation(self):
        try:
            gdown.download(
                'https://drive.google.com/uc?id=1twdtZEfcw4ghSZIiPDypJurZnNXzMO7R', current_dir + "/Safety_Helmet_Dataset.zip", quiet=False)
            return True, "Downloaded the Safety Helmet Dataset Done"
        except Exception as e:
            return False, str(e)

    def step_1_download_dataset(self):
        if st.button("Download Dataset"):
            st.write("Downloading Safety Helmet Dataset...")
            if not check_folder_exists(st.session_state.zip_file):
                # Run the gdown command
                async_result_download = pool.apply_async(
                    self.step_1_download_dataset_implementation)
                bar = st.progress(0)
                per = 15 / 100
                for i in range(100):
                    time.sleep(per)
                    bar.progress(i + 1)
                output_download, msg_download = async_result_download.get()
                if output_download:
                    st.write("Output (Download):")
                    st.code(msg_download)
                else:
                    st.write("Error (Download):")
                    st.code(msg_download)
            else:
                st.write("The file '{}' already exists.".format(
                    st.session_state.zip_file))

    def step_1_unzip_dataset_implementation(self):
        dataset_folder = current_dir + "/safety_helmet_dataset"
        zip_file = current_dir + "/Safety_Helmet_Dataset.zip"
        try:
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)
            return True, "Unzipped the Safety Helmet Dataset Done"
        except Exception as e:
            return False, str(e)

    def step_1_unzip_dataset(self):
        if st.button("Prepare Dataset"):
            dataset_folder = st.session_state.dataset_path
            zip_file = st.session_state.zip_file
            # Check if the dataset folder exists
            if not check_folder_exists(dataset_folder):
                st.write("Creating directory for the dataset...")
                # Create the directory
                os.makedirs(dataset_folder)
                if check_folder_exists(dataset_folder):
                    st.write("Directory {} created successfully.".format(
                        dataset_folder))
                else:
                    st.write(f"Failed to create directory '{dataset_folder}'.")
            # Check if the zip file exists before attempting to unzip
            if check_folder_exists(zip_file):
                st.write("Unzipping the dataset...")
                # Unzip the dataset
                async_result_unzip = pool.apply_async(
                    self.step_1_unzip_dataset_implementation)
                bar = st.progress(0)
                per = 30 / 100
                for i in range(100):
                    time.sleep(per)
                    bar.progress(i + 1)
                output_unzip, error_unzip = async_result_unzip.get()
                if output_unzip:
                    st.write("Output (Unzip):")
                    st.code(error_unzip)
                else:
                    st.write("Error (Unzip):")
                    st.code(error_unzip)
            else:
                st.write(
                    "The file '{}' does not exist. Please download the dataset first.".format(zip_file))

    def step_1_implementation(self):
        st.markdown(self.project_implementation_text, unsafe_allow_html=True)
        st.markdown(self.project_implementation_step_1_code_markdown,
                    unsafe_allow_html=True)
        st.code(self.project_implementation_step_1_code, language="python")
        self.step_1_download_dataset()
        self.step_1_unzip_dataset()

    def step_2_implementation(self):
        from ultralytics import YOLO
        st.markdown(self.project_implementation_step_2_code_markdown,
                    unsafe_allow_html=True)
        st.code(self.project_implementation_step_2_code, language="python")
        if st.button("Run Step 2"):
            self.MODEL_PATH = current_dir + "/yolov10n.pt"
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            try:
                self.model = YOLO(self.MODEL_PATH)
            finally:
                sys.stdout = old_stdout

            st.write("Done")
            st.write(buffer.getvalue())

    def step_3_implementation(self):
        from ultralytics import YOLO
        st.markdown(self.project_implementation_step_3_code_markdown,
                    unsafe_allow_html=True)
        st.code(self.project_implementation_step_3_code, language="python")
        if st.button("Run Step 3"):
            self.MODEL_PATH = current_dir + "/yolov10n.pt"
            YAML_PATH = current_dir + "/safety_helmet_dataset/data.yaml"
            IMG_SIZE = 640
            EPOCHS = 50
            BATCH_SIZE = 16
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            try:
                self.model = YOLO(self.MODEL_PATH)
                self.model.train(data=YAML_PATH, imgsz=IMG_SIZE,
                                 epochs=EPOCHS, batch=BATCH_SIZE)
            finally:
                sys.stdout = old_stdout

            st.write("Output")
            st.write(buffer.getvalue())

    def step_4_implementation(self):
        from ultralytics import YOLO
        st.markdown(self.project_implementation_step_4_code_markdown,
                    unsafe_allow_html=True)
        st.code(self.project_implementation_step_4_code, language="python")
        if st.button("Run Step 4"):
            YAML_PATH = current_dir + "/safety_helmet_dataset/data.yaml"
            IMG_SIZE = 640
            TRAINED_MODEL_PATH = current_dir + '/yolov10/runs/detect/train/weights/best.pt'
            old_stdout = sys.stdout
            sys.stdout = buffer = StringIO()
            try:
                self.model = YOLO(TRAINED_MODEL_PATH)
                self.model.val(data=YAML_PATH, imgsz=IMG_SIZE, split='test')
            finally:
                sys.stdout = old_stdout

            st.write("Output")
            st.write(buffer.getvalue())

    def project_implementation_render(self):
        self.step_1_implementation()
        self.step_2_implementation()
        self.step_3_implementation()
        self.step_4_implementation()


def run_command(command):
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
    output, error = process.communicate()
    return output, error
