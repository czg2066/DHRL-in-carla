import carla
import random
import cv2,time
import numpy as np

# Define the CARLA semantic segmentation color palette
COLOR_PALETTE = np.array([
    [0, 0, 0],        # None
    [0, 255, 0],   # Road
    [0, 0, 0],     # Building
    [0, 0, 0],  # Fence

    [0, 0, 0],      # Other
    [0, 0, 0],    # Pedestrian
    [0, 0, 0],  # Pole
    [255, 255, 255],   # TrafficLight
    
    
    [0, 0, 0],   # Sidewalk
    [0, 0, 0],   # Vegetation
    [0, 0, 0],  # Wall
    [0, 0, 0],    # TrafficSign

    [0, 0, 0],   # Sky
    [0, 0, 0],      # Ground
    [0, 0, 255],      # Vehicles
    [0, 0, 0],  # Bridge

    [0, 0, 0],  # RailTrack
    [0, 0, 0],  # GuardRail
    [0, 0, 0],  # Static
    [0, 0, 0],   # Dynamic

    [0, 0, 0],    # Water
    [0, 0, 0],  # Terrain
    [0, 0, 0],  # Terrain
    [0, 0, 0],  # Terrain

    [255, 255, 0],   # RoadLine
    [0, 0, 0],  # Terrain
    [0, 0, 0],  # Terrain
], dtype=np.uint8)

img = None
ori_img = None

def process_semantic_image(image):
    # Convert the CARLA image to a numpy array
    array = np.frombuffer(image.raw_data, dtype=np.uint8)
    array = np.reshape(array, (image.height, image.width, 4))

    # Extract the semantic segmentation image (ignoring the alpha channel)
    semantic_image = array[:, :, 2]

    # Convert to RGB using the predefined color palette
    semantic_image_rgb = np.zeros((image.height, image.width, 3), dtype=np.uint8)
    for label in range(0, 27):
        semantic_image_rgb[semantic_image == label] = COLOR_PALETTE[label]

    return semantic_image_rgb

def process_image(image):
    global img
    
    # Process the image to extract the semantic segmentation image
    img = process_semantic_image(image)

def main():
    # Connect to CARLA server
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    # Get the blueprint library
    blueprint_library = world.get_blueprint_library()

    # Choose a random vehicle
    vehicle_bp = random.choice(blueprint_library.filter('vehicle'))
    spawn_point = random.choice(world.get_map().get_spawn_points())
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # Set up the camera blueprint
    camera_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
    camera_bp.set_attribute('image_size_x', '400')
    camera_bp.set_attribute('image_size_y', '300')
    camera_bp.set_attribute('fov', '110')

    # Attach the camera to the vehicle
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)


    # Listen to the camera sensor data
    camera.listen(lambda image: process_image(image))

    try:
        vehicle.set_autopilot(True)
        # Run the simulation1
        while True:
            time.sleep(0.1)
            if vehicle.is_at_traffic_light():
                traffic_light = vehicle.get_traffic_light()
                traffic_light_state = traffic_light.get_state()
                print(traffic_light_state)
            print("thread is online")
            if img is not None:
                cv2.imshow("CARLA Semantic Segmentation", img)
                cv2.waitKey(1)

    except KeyboardInterrupt:
        print("Simulation stopped.")

    finally:
        # Destroy the camera and vehicle actors
        camera.destroy()
        vehicle.destroy()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
