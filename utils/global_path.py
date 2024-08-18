from global_route_planner import GlobalRoutePlanner
from global_route_planner_dao import GlobalRoutePlannerDAO
import random


class GlobalPath:
    def __init__(self) -> None:
        self.route = None
        self.arrive_route = 0

    def get_route(self, vehicle, dest_location):
        if self.route is None:
            vmap = vehicle.get_world().get_map()
            dest_waypoint = vmap.get_waypoint(dest_location)
            dao = GlobalRoutePlannerDAO(map, 15)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self.route = grp.trace_route(vehicle.get_location(), dest_waypoint.transform.location)

        return self.route
    
    def get_destination(self, vehicle, location=-1, min_distance=100):
        vmap = vehicle.get_world().get_map()
        if location == -1:
            possible_spawn_points = vmap.get_spawn_points()

            filtered_spawn_points = [
                spawn_point for spawn_point in possible_spawn_points
                if vehicle.get_location().distance(spawn_point.location) > min_distance
            ]
            if filtered_spawn_points:
                return random.choice(filtered_spawn_points)
            else:
                raise ValueError("No suitable spawn point found outside the specified radius.")
        else:
            return vmap.get_waypoint(location).transform.location
        
    def get_next_globalwp(self):
        if self.arrive_route == 0:
            return self.route[0][0].transform.location
        elif len(self.route) == 1:
            return self.route[0][0].transform.location
        else:
            return self.route[1][0].transform.location
    
    def update_route(self, vehicle):
        if vehicle.get_location().distance(self.route[0].location) < 2:
            if self.arrive_route != 0: self.route.pop(0)
            self.arrive_route += 1
        if len(self.route) == 0:
            return True
        else: return False

    def leave_route(self, vehicle):
        vmap = vehicle.get_world().get_map()
        if vmap.get_waypoint(vehicle.get_location()).road_id == self.route[0][0].road_id and \
            self.route[1][0].road_id == self.route[0][0].road_id:
            return False
        elif min(self.route[0][0].transform.location.x, self.route[1][0].transform.location.x)-5 <= vehicle.get_location().x \
            <= max(self.route[0][0].transform.location.x, self.route[1][0].transform.location.x)+5 and \
            min(self.route[0][0].transform.location.y, self.route[1][0].transform.location.y)-5 <= vehicle.get_location().y \
            <= max(self.route[0][0].transform.location.y, self.route[1][0].transform.location.y)+5:
            return False
        else:
            return True
