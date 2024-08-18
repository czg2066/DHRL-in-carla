import carla

class Upfeedback:
    def __init__(self) -> None:
        self.rewards = 0
        self.costs = 0
        self.maxstep_subgoal = 10

    def update(self, reward, cost):
        pass

    def caculate_reward(self, vehcile, finish_subgoal=False, global_goal=False, speed_limit=60, subgoal_step=-1):
        self.rewards = 0
        vspeed = (vehcile.get_velocity().x**2 + vehcile.get_velocity().y**2)**0.5
        if self.is_passing_intersection(vehcile):   #红绿灯准确停车行驶
            self.rewards += 1
        if finish_subgoal:
            self.rewards += 10
        if global_goal:
            self.rewards += 50
        if vspeed <= speed_limit:
            self.rewards += 5*(speed_limit - vspeed)/speed_limit
        if subgoal_step != -1 and subgoal_step <= self.maxstep_subgoal: #在10步内完成目标
            self.rewards += self.maxstep_subgoal - subgoal_step
        return self.rewards
    
    def caculate_cost(self, vehcile, goals, up_output, speed_limit=60):
        self.costs = 0
        if goals[up_output[1]] is None:
            self.costs += 10
        if up_output[0] > speed_limit:
            self.costs += 1
        if self.is_passing_intersection(vehcile) is False:
            self.costs += 5
        return self.costs

    
    def is_passing_intersection(self, vehcile):
        if vehcile.is_at_traffic_light():
            traffic_light = vehcile.get_traffic_light()
            traffic_light_state = traffic_light.get_state()
            v_vehcile =  vehcile.get_velocity()
            if (traffic_light_state == carla.TrafficLightState.Red or traffic_light_state == carla.TrafficLightState.Yellow) and (v_vehcile.x**2 + v_vehcile.y**2)**0.5 <= 0.5:
                return True
            elif (traffic_light_state == carla.TrafficLightState.Green or traffic_light_state == carla.TrafficLightState.Off or traffic_light_state == carla.TrafficLightState.Unknown)\
                  and (v_vehcile.x**2 + v_vehcile.y**2)**0.5 > 0.5:
                return True
            else:
                return False

class Downfeedback:
    def __init__(self) -> None:
        self.rewards = 0
        self.costs = 0
        self.maxstep_subgoal = 10
        self.collision = False
        self.lane_inv = False

    def update(self, reward, cost):
        pass

    def caculate_reward(self, vehcile, up_output, goals, finish_subgoal=False, global_goal=False):
        self.rewards = 0
        vspeed = (vehcile.get_velocity().x**2 + vehcile.get_velocity().y**2)**0.5
        vlocation = vehcile.get_location()
        if finish_subgoal:
            self.rewards += 10
        if global_goal:
            self.rewards += 50
        if abs(up_output[0] - vspeed) <= 15:
            self.rewards += 1*(15-abs(up_output[0]-vspeed))/10
        if vlocation.distance(goals[up_output[1]]) < 15:
            dis = vlocation.distance(goals[up_output[1]])
            self.rewards += (15 - dis)/3
        return self.rewards

    def caculate_cost(self, vehcile, up_output):
        self.costs = 0
        vmap = vehcile.get_world().get_map()
        vspeed = (vehcile.get_velocity().x**2 + vehcile.get_velocity().y**2)**0.5
        if self.collision == True:
            self.costs += 50
        if abs(up_output[0] - vspeed) > 15:
            self.costs += 1*(abs(up_output[0]-vspeed)-15)/10
        if self.lane_inv == True:
            vehicle_location = vehcile.get_location()
            waypoint = vmap.get_waypoint(vehicle_location)
            vector_lane_to_vehicle = vehicle_location - waypoint.transform.location
            waypoint_direction = waypoint.transform.rotation.get_forward_vector()
            left_right_inv = waypoint_direction.cross(vector_lane_to_vehicle).z
            lanetype = [waypoint.right_lane_marking.type, waypoint.left_lane_marking.type]
            if (left_right_inv < 0 and lanetype[1] is not carla.LaneMarkingType.Broken) or \
                (left_right_inv > 0 and lanetype[0] is not carla.LaneMarkingType.Broken):
                    self.costs += 50
        return self.costs

    #检测汽车碰撞函数
    def detect_collision(self, event):
        impulse = event.normal_impulse
        self.intensity = sum([impulse.x**2, impulse.y**2, impulse.z**2])**0.5  # 计算碰撞强度
        self.collision = True

    #检测汽车车道变化函数
    def detect_lane(self, event):
        '''
        车道入侵传感器回调函数,触发后减少rewards 增加cost 设置done 
        参数: carla event事件类型数据
        返回: 无
        '''
        self.lane_inv = True
        