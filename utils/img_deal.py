import cv2, carla
import numpy as np

class ImageProcessor:
    def __init__(self):
        self.img = None
        self.select_seg = [1, 7, 24]
        self.traffic_seg = [14, 15, 16, 17, 18, 19]

    def get_process_semantic_image(self, vehicle):
        image = self.img
        # Convert the CARLA image to a numpy array
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))

        # Extract the semantic segmentation image (ignoring the alpha channel)
        semantic_image = array[:, :, 2]

        # Convert to RGB using the predefined color palette
        semantic_image_rgb = np.zeros((image.height, image.width, 3), dtype=np.uint8)
        for label in range(0, 27):
            if label in self.traffic_seg:
                semantic_image_rgb[semantic_image == label] = [0, 0, 255]
            elif label == self.select_seg[0]:
                semantic_image_rgb[semantic_image == label] = [255, 0, 0]
            elif label == self.select_seg[1]:       #50 red, 100 yellow, 150 green, 200 off/unknown
                try:
                    traffic_light = vehicle.get_traffic_light()
                    traffic_light_state = traffic_light.get_state()
                    if traffic_light_state == carla.TrafficLightState.Red:
                        semantic_image_rgb[semantic_image == label] = [50, 50, 50]
                    elif traffic_light_state == carla.TrafficLightState.Yellow:
                        semantic_image_rgb[semantic_image == label] = [100, 100, 100]
                    elif traffic_light_state == carla.TrafficLightState.Green:
                        semantic_image_rgb[semantic_image == label] = [150, 150, 150]
                    else:
                        semantic_image_rgb[semantic_image == label] = [200, 200, 200]
                except: semantic_image_rgb[semantic_image == label] = [200, 200, 200]   #找不到红绿灯，默认unknown
            elif label == self.select_seg[2]:
                semantic_image_rgb[semantic_image == label] = [0, 255, 0]

        return semantic_image_rgb

    def store_semantic_image(self, image):
        # Process the image to extract the semantic segmentation image
        self.img = image

    def show_image(self):
        if self.img is not None:
                cv2.imshow("CARLA Semantic Segmentation", self.img)
                cv2.waitKey(1)
