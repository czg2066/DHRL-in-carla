import carla
import pygame
import numpy as np

def main():
    # 初始化pygame
    pygame.init()
    display = pygame.display.set_mode((800, 600), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("CARLA Client")

    # 连接到CARLA服务端
    client = carla.Client('157.122.209.70', 21000)
    client.set_timeout(60.0)
    world = client.get_world()
    #world.unload_map_layer(carla.MapLayer.All)

    # 获取蓝图库和生成蓝图
    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('model3')[0]

    # 生成车辆和设置初始位置
    spawn_point = world.get_map().get_spawn_points()[0]
    vehicle = world.spawn_actor(vehicle_bp, spawn_point)

    # 创建摄像机
    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '320')
    camera_bp.set_attribute('image_size_y', '120') # 设置摄像机分辨率
    camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
    camera_center_transform = carla.Transform(carla.Location(x=0, z=50), carla.Rotation(pitch=-90))
    camera_center = world.spawn_actor(camera_bp, camera_center_transform, attach_to=vehicle)

    vehicle.set_autopilot(True) # 启用自动驾驶

    # 定义处理图像的回调函数
    def process_image(image):
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]  # 只保留RGB通道
        array = array[:, :, ::-1]  # 将BGR通道转换为RGB通道
        surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        display.blit(surface, (0, 0))
        pygame.display.flip()

    # 绑定回调函数到摄像机
    #camera.listen(lambda image: process_image(image))
    camera_center.listen(lambda image: process_image(image))

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
    finally:
        # 清理
        camera.stop()
        vehicle.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()
