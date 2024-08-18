import carla
import numpy as np

class WaypointSearch:
    def __init__(self):
        pass
        
    def get_lane_center(self, waypoint, distance=15):
        
        left_lane_waypoint = waypoint.get_left_lane() if waypoint.lane_change & carla.LaneChange.Left else None
        right_lane_waypoint = waypoint.get_right_lane() if waypoint.lane_change & carla.LaneChange.Right else None
        # 在当前道路前方 distance 米的位置获取中心点
        while True:
            try:
                forward_waypoint = waypoint.next(distance)[0] 
                if left_lane_waypoint is not None: left_lane_waypoint = left_lane_waypoint.next(distance)[0] 
                if right_lane_waypoint is not None: right_lane_waypoint = right_lane_waypoint.next(distance)[0]
                break
            except:
                if distance < 2: break
                print("Can't find next waypoint after {} meters, try -2m.".format(distance))
                distance -= 2
        
        return [forward_waypoint.transform.location, left_lane_waypoint.transform.location, right_lane_waypoint.transform.location]
    
    def get_intersection_center(self, waypoint):
        exits = [None, None, None]
        # 探索可能的路径
        next_waypoints = waypoint.next(3.0)
        for next_wp in next_waypoints:
            if next_wp.is_junction:  # 还在路口内
                # 根据转向角度分类
                angle = next_wp.transform.rotation.yaw - waypoint.transform.rotation.yaw
                if angle < -45:
                    exits[1] = next_wp.transform.location
                elif angle > 45:
                    exits[2] = next_wp.transform.location
                else:
                    exits[0] = next_wp.transform.location
        return exits
    
    def get_goals(self, waypoint):
        # 检查当前waypoint是否在路口
        if waypoint.is_junction:
            return self.get_intersection_center(waypoint)
        else:
            return self.get_lane_center(waypoint)
        
    def arrive_subgoal(self, vehcile, up_output, goals):
        vlocation = vehcile.get_location()
        waypoint = map.get_waypoint(vlocation)
        goal = goals[up_output[1]]
        if vlocation.distance(goal) < 0.25:
            return True
        else:
            return False

