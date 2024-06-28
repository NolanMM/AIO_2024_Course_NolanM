import numpy as np
from PIL import Image
import streamlit as st
from ImageProcessor import ImageProcessor


def main():
    st.title('Object Detection for Images')
    file = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])
    if file is not None:
        st.image(file, caption="Uploaded Image")
        image_processor = ImageProcessor()

        image = Image.open(file)
        image = np.array(image)
        detections = image_processor.process_image(image)
        processed_image = image_processor.annotate_image(image, detections)
        st.image(processed_image, caption="Processed Image")


if __name__ == "__main__":
    main()
