import carla

client = carla.Client('localhost', 21000)
client.set_timeout(10.0)
world = client.get_world()

print(world.get_map().name)