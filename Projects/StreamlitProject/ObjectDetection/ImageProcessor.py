import cv2
import numpy as np


class ImageProcessor:
    def __init__(self):
        self.model = "../model/MobileNetSSD_deploy.caffemodel"
        self.prototxt = "../model/MobileNetSSD_deploy.prototxt.txt"
        self.net = cv2.dnn.readNetFromCaffe(self.prototxt, self.model)

    def process_image(self, image):
        """
        Processes the input image to prepare it for object detection.

        Parameters:
        image (numpy.ndarray): The input image to be processed.

        Returns:
        numpy.ndarray: The detection results after processing the image through the neural network.
        """
        blob = cv2.dnn.blobFromImage(
            # - Scale the pixel values to the range [0, 1] by multiplying by 0.007843
            # - Subtract the mean value (127.5) from each pixel to center the data around zero
            cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5
        )
        # Set the blob as the input to the network
        self.net.setInput(blob)
        # Perform a forward pass through the network to get the detection results
        return self.net.forward()

    def annotate_image(self, image, detections, confidence_threshold=0.5):
        """
        Annotates the input image with bounding boxes based on detection results.

        Parameters:
        image (numpy.ndarray): The input image on which to draw the bounding boxes.
        detections (numpy.ndarray): The detection results, typically obtained from a deep learning model.
                                    This array contains the confidence scores, class IDs, and bounding box coordinates.
        confidence_threshold (float): The minimum confidence score required to draw a bounding box. Default is 0.5.

        Returns:
        numpy.ndarray: The annotated image with bounding boxes drawn around detected objects.
        """
        (h, w) = image.shape[:2]
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > confidence_threshold:
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (start_x, start_y, end_x, end_y) = box.astype("int")
                cv2.rectangle(image, (start_x, start_y),
                              (end_x, end_y), (70, 2))
        return image
