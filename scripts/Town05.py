import carla, random
from utils.img_deal import ImageProcessor
from utils.waypoint_search import WaypointSearch
from utils.global_path import GlobalPath
from utils.caculate_feedback import Upfeedback, Downfeedback

class CarlaEnv_T5:
    def __init__(self, env_name='Town05',Outtime = 20.0, delta_seconds=0.05, **kwargs) -> None:  # env_name = Town05):
        self.carla_world = env_name
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(Outtime)  # 设置超时时间
        self.world = self.client.get_world()
        settings = self.world.get_settings()
        settings.synchronous_mode = True# 设置为同步模式
        settings.fixed_delta_seconds = delta_seconds  # 设置每帧的时间间隔为0.05秒
        self.world.apply_settings(settings)
        self.traffic_manager = self.client.get_trafficmanager(8000)  # 默认端口为8000
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5) # 设置最小距离为2.5米
        self.traffic_manager.set_synchronous_mode(True)  # 如果世界已经设置为同步模式

    def reset(self, location=-1):
        # Reset the environment
        self.get_3kind_actors(self.world, destroy=True) # 销毁所有车辆、行人和传感器actor

        self.img_processor = ImageProcessor()
        self.waypoint_search = WaypointSearch()
        self.global_path = GlobalPath()
        self.upfeedback = Upfeedback()
        self.downfeedback = Downfeedback()

        self.blueprint_library = self.world.get_blueprint_library()
        
        model_3_bp = self.blueprint_library.find('vehicle.tesla.model3')
        spawn_points = self.world.get_map().get_spawn_points()
        if location < 0: spawn_point = random.choice(spawn_points)
        else: spawn_point = spawn_points[location]
        self.ego_vehicle = self.world.spawn_actor(model_3_bp, spawn_point)

        self.spectator = self.world.get_spectator()
        transform = self.ego_vehicle.get_transform()  # 获取车辆当前位置
        spectator_transform = carla.Transform(carla.Location(x=transform.location.x, y=transform.location.y, z=transform.location.z + 10), carla.Rotation(pitch=-90, yaw=transform.rotation.yaw))
        self.spectator.set_transform(spectator_transform)

        seg_camera_bp = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        seg_camera_bp.set_attribute('image_size_x', '800')
        seg_camera_bp.set_attribute('image_size_y', '600')
        seg_camera_bp.set_attribute('fov', '120')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        segmentation_camera = self.world.spawn_actor(seg_camera_bp, camera_transform, attach_to=self.ego_vehicle)
        segmentation_camera.listen(self.img_processor.store_semantic_image)

        self.CollisionSensor = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(self.CollisionSensor, self.vehicle_transform, attach_to=self.vehicle)
        self.collision_sensor.listen(self.downfeedback.detect_collision)

        self.Lane_invasionSensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lane_invasion_sensor = self.world.spawn_actor(self.Lane_invasionSensor, self.vehicle_transform, attach_to=self.vehicle)
        self.lane_invasion_sensor.listen(self.downfeedback.detect_lane)

        self.last_doaction = [0, 0]
        self.last_upoutput = [-1, -1]

        self.global_dest = self.global_path.get_destination(self.vehicle)
        self.golab_route = self.global_path.get_route(self.vehicle, self.global_dest)

        self.step_num = 0

        return self.get_states([5, 0], mode="all")

    
    def step(self, action, up_output=[5, 0], mode="all"):
        self.step_num += 1
        self.control_vehicle(action)
        self.world.tick()
        vwaypoint = self.map.get_waypoint(self.ego_vehicle.get_location())
        up_states, down_states = self.get_states(up_output, mode)
        fglobal_goal = self.global_path.update_route(self.ego_vehicle)
        goals = self.waypoint_search.get_goals(vwaypoint)
        finish_subgoal = self.waypoint_search.arrive_subgoal(self.ego_vehicle, up_output, goals)
        dreward = self.downfeedback.caculate_reward(self.ego_vehicle, goals, finish_subgoal, fglobal_goal)
        dcost = self.downfeedback.caculate_cost(self.ego_vehicle, up_output)
        if mode=="all":
            ureward = self.upfeedback.caculate_reward(up_output, finish_subgoal, fglobal_goal, subgoal_step=self.step_num)
            ucost = self.upfeedback.caculate_cost(up_output)
        else:
            ureward = None
            ucost = None
        done = self.get_done(fglobal_goal)
        return down_states, dreward-dcost, done, up_states, ureward-ucost, done

    def control_vehicle(self, action):
        vehicle_control = carla.VehicleControl()
        vehicle_control.steer = action[0]
        vehicle_control.throttle = action[1]
        self.ego_vehicle.apply_control(vehicle_control)

    def get_states(self, up_output, layer_state="all"): # layer_state:down, all
        up_states = []
        down_states = []
        if layer_state == "all":
            up_states.append(self.img_processor.get_semantic_image())
        down_states = self.get_sur_vehicle()
        if self.last_upoutput != up_output:
            goals_location = self.waypoint_search.get_goals[up_output]
            self.goals_list = [up_output[0], goals_location.x, goals_location.y, 0]
        down_states.append(self.goals_list)
        return up_states, down_states
    
    def get_done(self, fglobal_goal):
        if fglobal_goal or self.downfeedback.collision or self.downfeedback.lane_inv:
            return True
        return False


    def get_sur_vehicle(self, max_num=6, max_distance=100):
        vehicles = self.world.get_actors().filter('vehicle.*')
        distance_to_vehicles = []
        ego_location = self.ego_vehicle.get_transform().location
        ego_velocity = self.ego_vehicle.get_velocity()
        vehicle_vectors = [[ego_location.x, ego_location.y, ego_velocity.x, ego_velocity.y]]

        for vehicle in vehicles:
            if vehicle.id != self.ego_vehicle.id:  # 确保不计算自车
                location = vehicle.get_transform().location
                distance = location.distance(self.ego_vehicle.get_transform().location)
                if distance <= max_distance:  # 距离小于或等于100米
                    distance_to_vehicles.append((vehicle, distance))
        # 按距离排序并取前mun个
        act_mun = min(max_num, len(distance_to_vehicles))
        distance_to_vehicles.sort(key=lambda x: x[1])
        nearest_vehicles = distance_to_vehicles[:act_mun]

        for vehicle, distance in nearest_vehicles:
            location = vehicle.get_transform().location
            velocity = vehicle.get_velocity()
            vehicle_vectors.append([location.x, location.y, velocity.x, velocity.y])
        while len(vehicle_vectors) < max_num:
            vehicle_vectors.append([0, 0, 0, 0]) # 补齐不足的车辆信息

    def set_env_vehicle(self, num_vehicles=10):
        vehicle_blueprints = self.blueprint_library.filter('vehicle.*')
        spawn_points = self.world.get_map().get_spawn_points()
        finish_num = min(num_vehicles, len(spawn_points))
        while finish_num > 0:
            finish_num -= 1
            try :
                vehicle_bp = random.choice(vehicle_blueprints)
                spawn_point = random.choice(spawn_points)
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                vehicle.set_autopilot(True, self.traffic_manager.get_port())
            except Exception as e:
                finish_num += 1
                print("Can't spawn vehicle, becase:", e, "Will try again!")

    def get_3kind_actors(self,world=None, destroy=False):
        if world is None:
            world = self.world
        actors = world.get_actors()
        vehicles = actors.filter('vehicle.*')
        pedestrians = actors.filter('walker.pedestrian.*')
        sensors = actors.filter('sensor.*')
        if destroy:
            self.destroy_actors(vehicles)
            self.destroy_actors(pedestrians)
            self.destroy_actors(sensors)
        return vehicles, pedestrians, sensors
    
    def destroy_actors(self, actors):
        try:
            for actor in actors:
                actor.destroy()
            return 0
        except Exception as e:
            print(e)
            return -1