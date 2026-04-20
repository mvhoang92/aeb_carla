import argparse
import math
import queue
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pygame

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from carla_bootstrap import bootstrap_carla

bootstrap_carla()

import carla

from configs import aeb_config

CAM_W = aeb_config.CAM_WIDTH
CAM_H = aeb_config.CAM_HEIGHT
CAM_FOV = aeb_config.CAM_FOV
DEFAULT_OUTPUT_DIR = CURRENT_DIR / 'data'
CLASS_NAMES = {
    0: 'person',
    1: 'bike_motorbike',
    2: 'car',
    3: 'truck',
}

BLUEPRINT_TO_CLASS = {
    'walker.pedestrian': (0, CLASS_NAMES[0]),
    'vehicle.bh.crossbike': (1, CLASS_NAMES[1]),
    'vehicle.diamondback.century': (1, CLASS_NAMES[1]),
    'vehicle.gazelle.omafiets': (1, CLASS_NAMES[1]),
    'vehicle.harley-davidson.low_rider': (1, CLASS_NAMES[1]),
    'vehicle.kawasaki.ninja': (1, CLASS_NAMES[1]),
    'vehicle.vespa.zx125': (1, CLASS_NAMES[1]),
    'vehicle.yamaha.yzf': (1, CLASS_NAMES[1]),
}

TRUCK_KEYWORDS = (
    'truck',
    'ambulance',
    'sprinter',
    'carlacola',
    'fusorosa',
    'firetruck',
    'volkswagen.t2',
)

BIKE_KEYWORDS = (
    'bike',
    'century',
    'omafiets',
    'low_rider',
    'ninja',
    'yzf',
    'zx125',
    'vespa',
)

BANNED_VEHICLES = ('vehicle.mini.cooper_s_2021',)
WEATHER_FIELDS = (
    'cloudiness',
    'precipitation',
    'precipitation_deposits',
    'wind_intensity',
    'sun_azimuth_angle',
    'sun_altitude_angle',
    'fog_density',
    'fog_distance',
    'wetness',
)


def clamp(value, minimum=0.0, maximum=100.0):
    return max(minimum, min(value, maximum))


class Sun:
    def __init__(self, azimuth, altitude):
        self.azimuth = azimuth
        self.altitude = altitude
        self._time = 0.0

    def tick(self, delta_seconds):
        self._time += 0.008 * delta_seconds
        self._time %= 2.0 * math.pi
        self.azimuth = (self.azimuth + 0.25 * delta_seconds) % 360.0
        self.altitude = (70.0 * math.sin(self._time)) - 20.0


class Storm:
    def __init__(self, precipitation):
        self._time = precipitation if precipitation > 0.0 else -50.0
        self._increasing = True
        self.clouds = 0.0
        self.rain = 0.0
        self.wetness = 0.0
        self.puddles = 0.0
        self.wind = 0.0
        self.fog = 0.0

    def tick(self, delta_seconds):
        delta = (1.3 if self._increasing else -1.3) * delta_seconds
        self._time = clamp(self._time + delta, -250.0, 100.0)
        self.clouds = clamp(self._time + 40.0, 0.0, 90.0)
        self.rain = clamp(self._time, 0.0, 80.0)
        delay = -10.0 if self._increasing else 90.0
        self.puddles = clamp(self._time + delay, 0.0, 85.0)
        self.wetness = clamp(self._time * 5.0, 0.0, 100.0)
        self.wind = 5.0 if self.clouds <= 20.0 else 90.0 if self.clouds >= 70.0 else 40.0
        self.fog = clamp(self._time - 10.0, 0.0, 30.0)
        if self._time == -250.0:
            self._increasing = True
        if self._time == 100.0:
            self._increasing = False


class DynamicWeatherCycle:
    def __init__(self, weather, speed_factor):
        self.weather = weather
        self.speed_factor = max(0.01, speed_factor)
        self.sun = Sun(weather.sun_azimuth_angle, weather.sun_altitude_angle)
        self.storm = Storm(weather.precipitation)
        self.elapsed = 0.0
        self.update_freq = 0.1 / self.speed_factor

    def tick(self, world, delta_seconds):
        self.elapsed += delta_seconds
        if self.elapsed < self.update_freq:
            return
        scaled_delta = self.speed_factor * self.elapsed
        self.sun.tick(scaled_delta)
        self.storm.tick(scaled_delta)
        self.weather.cloudiness = self.storm.clouds
        self.weather.precipitation = self.storm.rain
        self.weather.precipitation_deposits = self.storm.puddles
        self.weather.wind_intensity = self.storm.wind
        self.weather.fog_density = self.storm.fog
        self.weather.wetness = self.storm.wetness
        self.weather.sun_azimuth_angle = self.sun.azimuth
        self.weather.sun_altitude_angle = self.sun.altitude
        world.set_weather(self.weather)
        self.elapsed = 0.0


def clone_weather(weather):
    cloned = carla.WeatherParameters()
    for field in WEATHER_FIELDS:
        if hasattr(weather, field):
            setattr(cloned, field, float(getattr(weather, field)))
    return cloned


def build_weather_profile(name, cloudiness, rain, puddles, wetness, wind, fog, sun_altitude, fog_distance=30.0):
    weather = carla.WeatherParameters()
    weather.cloudiness = float(cloudiness)
    weather.precipitation = float(rain)
    weather.precipitation_deposits = float(puddles)
    weather.wetness = float(wetness)
    weather.wind_intensity = float(wind)
    weather.fog_density = float(fog)
    weather.fog_distance = float(fog_distance)
    weather.sun_altitude_angle = float(sun_altitude)
    weather.sun_azimuth_angle = random.uniform(0.0, 360.0)
    return {'name': name, 'weather': weather}


class WeatherDirector:
    def __init__(self, world, speed_factor, change_interval):
        self.world = world
        self.speed_factor = max(0.1, speed_factor)
        self.change_interval = max(10.0, change_interval)
        self.elapsed = 0.0
        self.profile_name = 'Unknown'
        self.dynamic_cycle = None
        self.profiles = [
            build_weather_profile('Clear Day', 5.0, 0.0, 0.0, 0.0, 10.0, 0.0, 65.0, 80.0),
            build_weather_profile('Cloudy Day', 75.0, 0.0, 0.0, 10.0, 35.0, 5.0, 42.0, 60.0),
            build_weather_profile('Wet Cloudy Day', 85.0, 20.0, 45.0, 70.0, 50.0, 10.0, 30.0, 45.0),
            build_weather_profile('Hard Rain Day', 100.0, 85.0, 95.0, 100.0, 95.0, 25.0, 18.0, 30.0),
            build_weather_profile('Clear Night', 10.0, 0.0, 0.0, 0.0, 15.0, 0.0, -70.0, 80.0),
            build_weather_profile('Foggy Night', 55.0, 0.0, 10.0, 15.0, 20.0, 65.0, -75.0, 18.0),
            build_weather_profile('Rainy Night', 100.0, 90.0, 100.0, 100.0, 100.0, 40.0, -55.0, 20.0),
            build_weather_profile('Storm Sunset', 95.0, 70.0, 85.0, 90.0, 100.0, 20.0, -8.0, 28.0),
        ]
        self.apply_random_profile(initial=True)

    def apply_random_profile(self, initial=False):
        profile = random.choice(self.profiles)
        self.profile_name = profile['name']
        weather = clone_weather(profile['weather'])
        self.world.set_weather(weather)
        self.dynamic_cycle = DynamicWeatherCycle(clone_weather(weather), self.speed_factor)
        self.elapsed = 0.0
        prefix = '[*]' if initial else '[~]'
        print(f'{prefix} Weather -> {self.profile_name}')

    def tick(self, delta_seconds):
        self.elapsed += delta_seconds
        self.dynamic_cycle.tick(self.world, delta_seconds)
        if self.elapsed >= self.change_interval:
            self.apply_random_profile()


class VehicleBlueprintPicker:
    def __init__(self, blueprint_library):
        self.cars = []
        self.bikes = []
        self.trucks = []

        for blueprint in blueprint_library.filter('vehicle.*'):
            if blueprint.id in BANNED_VEHICLES:
                continue
            classification = classify_actor_type(blueprint.id)
            if not classification:
                continue
            if classification[0] == 1:
                self.bikes.append(blueprint)
            elif classification[0] == 3:
                self.trucks.append(blueprint)
            else:
                self.cars.append(blueprint)

        self.cars.sort(key=lambda bp: bp.id)
        self.bikes.sort(key=lambda bp: bp.id)
        self.trucks.sort(key=lambda bp: bp.id)

    def choose(self):
        roll = random.random()
        if self.bikes and roll < 0.18:
            return random.choice(self.bikes)
        if self.trucks and roll < 0.30:
            return random.choice(self.trucks)
        if self.cars:
            return random.choice(self.cars)
        pools = self.bikes + self.trucks
        if not pools:
            raise RuntimeError('Khong tim thay blueprint xe phu hop trong map hien tai.')
        return random.choice(pools)


def classify_actor_type(type_id):
    if type_id.startswith('walker.pedestrian'):
        return 0, CLASS_NAMES[0]

    for prefix, class_info in BLUEPRINT_TO_CLASS.items():
        if type_id.startswith(prefix):
            return class_info

    if type_id.startswith('vehicle.'):
        if any(keyword in type_id for keyword in BIKE_KEYWORDS):
            return 1, CLASS_NAMES[1]
        if any(keyword in type_id for keyword in TRUCK_KEYWORDS):
            return 3, CLASS_NAMES[3]
        return 2, CLASS_NAMES[2]

    return None


def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    matrix = np.identity(3)
    matrix[0, 0] = matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def get_actor_info(actor_id, world, cache):
    cached = cache.get(actor_id)
    if cached is not None:
        actor = cached['actor_obj']
        if actor is not None and actor.is_alive:
            return cached
        cache.pop(actor_id, None)

    actor = world.get_actor(actor_id)
    if actor and hasattr(actor, 'bounding_box') and actor.bounding_box is not None:
        cache[actor_id] = {
            'actor_obj': actor,
            'type_id': actor.type_id,
            'bbox': actor.bounding_box,
            'extent_x': actor.bounding_box.extent.x,
        }
        return cache[actor_id]

    cache[actor_id] = None
    return None


def get_2d_bbox(actor_snapshot, actor_type_id, actor_bbox, camera_transform, projection_matrix, image_w, image_h, max_dist):
    actor_loc = actor_snapshot.get_transform().location
    if actor_loc.distance(camera_transform.location) > max_dist:
        return None

    center_x = actor_bbox.location.x
    center_y = actor_bbox.location.y
    center_z = actor_bbox.location.z
    extent = actor_bbox.extent

    is_bike = any(token in actor_type_id for token in BIKE_KEYWORDS)
    if is_bike:
        extent_x = extent.x * 1.1
        extent_y = extent.y * 1.1
        extent_z = extent.z * 1.4
    else:
        extent_x = extent.x
        extent_y = extent.y
        extent_z = extent.z

    vertices = [
        [center_x + extent_x, center_y + extent_y, center_z + extent_z],
        [center_x + extent_x, center_y - extent_y, center_z + extent_z],
        [center_x + extent_x, center_y + extent_y, center_z - extent_z],
        [center_x + extent_x, center_y - extent_y, center_z - extent_z],
        [center_x - extent_x, center_y + extent_y, center_z + extent_z],
        [center_x - extent_x, center_y - extent_y, center_z + extent_z],
        [center_x - extent_x, center_y + extent_y, center_z - extent_z],
        [center_x - extent_x, center_y - extent_y, center_z - extent_z],
    ]

    world_coords = []
    actor_transform = actor_snapshot.get_transform()
    for vertex in vertices:
        world_vertex = actor_transform.transform(
            carla.Location(x=vertex[0], y=vertex[1], z=vertex[2])
        )
        world_coords.append([world_vertex.x, world_vertex.y, world_vertex.z, 1.0])

    world_coords = np.transpose(np.array(world_coords))
    world_to_camera = np.array(camera_transform.get_inverse_matrix())
    camera_coords = np.dot(world_to_camera, world_coords)

    if np.any(camera_coords[0, :] < 0.1):
        return None

    min_depth = float(np.min(camera_coords[0, :]))
    camera_x = camera_coords[0, :]
    camera_y = camera_coords[1, :]
    camera_z = camera_coords[2, :]
    points_2d = np.transpose(
        np.dot(projection_matrix, np.array([camera_y, -camera_z, camera_x]))
    )
    points_2d = points_2d / points_2d[:, 2][:, None]

    x_coords = points_2d[:, 0]
    y_coords = points_2d[:, 1]
    raw_x1 = float(np.min(x_coords))
    raw_y1 = float(np.min(y_coords))
    raw_x2 = float(np.max(x_coords))
    raw_y2 = float(np.max(y_coords))

    if raw_x2 < 0 or raw_x1 > image_w or raw_y2 < 0 or raw_y1 > image_h:
        return None

    raw_width = max(raw_x2 - raw_x1, 1.0)
    raw_height = max(raw_y2 - raw_y1, 1.0)
    raw_area = raw_width * raw_height

    x1 = int(max(0.0, raw_x1))
    y1 = int(max(0.0, raw_y1))
    x2 = int(min(float(image_w), raw_x2))
    y2 = int(min(float(image_h), raw_y2))
    clipped_area = max(x2 - x1, 0) * max(y2 - y1, 0)
    frame_visible_ratio = clipped_area / raw_area

    return min_depth, (
        x1,
        y1,
        x2,
        y2,
    ), frame_visible_ratio


def compute_visibility_ratio(depth_frame, x1, y1, x2, y2, min_depth, tolerance):
    height, width = depth_frame.shape
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(width - 1, int(x2))
    y2 = min(height - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return True

    xs = np.linspace(x1, x2, 8).astype(int)
    ys = np.linspace(y1, y2, 8).astype(int)
    visible_points = 0

    for px in xs:
        for py in ys:
            pixel_depth = depth_frame[py, px]
            if min_depth - 2.0 <= pixel_depth <= min_depth + tolerance:
                visible_points += 1

    return visible_points / 64.0


def sensor_callback(data, sensor_queue):
    try:
        if sensor_queue.full():
            sensor_queue.get_nowait()
        sensor_queue.put_nowait(data)
    except Exception:
        pass


def receive_latest_frame(sensor_queue, target_frame, timeout=2.0):
    data = sensor_queue.get(timeout=timeout)
    while data.frame < target_frame:
        data = sensor_queue.get(timeout=timeout)
    return data


def speed_kmh(vehicle):
    velocity = vehicle.get_velocity()
    return 3.6 * math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def actor_speed_mps(actor):
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)


def yaw_delta_deg(yaw_a, yaw_b):
    return abs((yaw_a - yaw_b + 180.0) % 360.0 - 180.0)


def is_dark_weather(weather):
    return (
        weather.sun_altitude_angle < 10.0
        or (weather.sun_altitude_angle < 20.0 and weather.cloudiness > 70.0)
        or weather.precipitation > 45.0
        or weather.fog_density > 25.0
    )


def set_vehicle_headlights(vehicles, enable):
    required_bits = int(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam)
    for vehicle in vehicles:
        if vehicle is None or not vehicle.is_alive:
            continue
        try:
            current_bits = int(vehicle.get_light_state())
            next_bits = (current_bits | required_bits) if enable else (current_bits & ~required_bits)
            if next_bits != current_bits:
                vehicle.set_light_state(carla.VehicleLightState(next_bits))
        except RuntimeError:
            continue


def set_spectator_over_vehicle(world, vehicle):
    transform = vehicle.get_transform()
    spectator = world.get_spectator()
    spectator.set_transform(
        carla.Transform(
            transform.location + carla.Location(z=35.0),
            carla.Rotation(pitch=-90.0),
        )
    )


def clear_sensor_queue(sensor_queue):
    while not sensor_queue.empty():
        try:
            sensor_queue.get_nowait()
        except queue.Empty:
            break


def snap_transform_to_driving_lane(world, transform, z_offset=0.05):
    waypoint = world.get_map().get_waypoint(
        transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if waypoint is None:
        snapped = carla.Transform(transform.location, transform.rotation)
    else:
        snapped = carla.Transform(waypoint.transform.location, waypoint.transform.rotation)
    snapped.location.z += z_offset
    snapped.rotation.roll = 0.0
    snapped.rotation.pitch = 0.0
    return snapped


def find_recovery_spawn_point(world, current_location, min_distance):
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for spawn_point in spawn_points:
        candidate = snap_transform_to_driving_lane(world, spawn_point)
        if candidate.location.distance(current_location) >= min_distance:
            return candidate
    fallback = random.choice(spawn_points)
    return snap_transform_to_driving_lane(world, fallback)


def pick_walker_target(world, origin_location, min_distance, max_distance, max_attempts):
    fallback = None
    for _ in range(max_attempts):
        target_location = world.get_random_location_from_navigation()
        if target_location is None:
            continue
        if fallback is None:
            fallback = target_location
        distance = origin_location.distance(target_location)
        if min_distance <= distance <= max_distance:
            return target_location
    return fallback


def assign_walker_target(world, controller, walker_actor, min_distance, max_distance, max_attempts):
    if controller is None or walker_actor is None:
        return False
    if not controller.is_alive or not walker_actor.is_alive:
        return False
    origin_location = walker_actor.get_location()
    target_location = pick_walker_target(
        world,
        origin_location,
        min_distance,
        max_distance,
        max_attempts,
    )
    if target_location is None:
        return False
    controller.go_to_location(target_location)
    return True


def recover_ego_vehicle(world, ego, traffic_manager, sensor_queues, min_distance):
    spawn_point = find_recovery_spawn_point(world, ego.get_transform().location, min_distance)
    ego.set_autopilot(False, traffic_manager.get_port())
    ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
    try:
        ego.set_simulate_physics(False)
    except RuntimeError:
        pass
    ego.set_transform(spawn_point)
    for sensor_queue in sensor_queues:
        clear_sensor_queue(sensor_queue)
    set_spectator_over_vehicle(world, ego)
    world.tick()
    try:
        ego.set_simulate_physics(True)
    except RuntimeError:
        pass
    ego.set_target_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.set_target_angular_velocity(carla.Vector3D(0.0, 0.0, 0.0))
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0))
    world.tick()
    ego.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, steer=0.0))
    ego.set_autopilot(True, traffic_manager.get_port())
    return spawn_point


def connect_to_carla(host, port, rpc_timeout, startup_wait):
    client = carla.Client(host, port)
    deadline = time.time() + max(startup_wait, rpc_timeout)
    attempt = 0
    last_error = None

    while time.time() < deadline:
        attempt += 1
        try:
            client.set_timeout(rpc_timeout)
            world = client.get_world()
            return client, world
        except RuntimeError as exc:
            last_error = exc
            remaining = max(0, int(deadline - time.time()))
            print(
                f'[!] CARLA chua san sang o {host}:{port} '
                f'(lan {attempt}, con doi toi da {remaining}s)...'
            )
            time.sleep(5.0)

    raise RuntimeError(
        f'Khong ket noi duoc CARLA tai {host}:{port} sau {startup_wait:.0f}s. '
        f'Process/co cong co the da mo, nhung RPC get_world() van khong phan hoi. '
        f'Hay doi map load xong hoan toan hoac restart lai server. '
        f'Chi tiet cuoi: {last_error}'
    )


def rpc_call_with_retry(label, func, startup_wait, retry_interval=5.0):
    deadline = time.time() + max(startup_wait, retry_interval)
    attempt = 0
    last_error = None

    while time.time() < deadline:
        attempt += 1
        try:
            return func()
        except RuntimeError as exc:
            last_error = exc
            remaining = max(0, int(deadline - time.time()))
            print(
                f'[!] CARLA dang cham o buoc {label} '
                f'(lan {attempt}, con doi toi da {remaining}s)...'
            )
            time.sleep(retry_interval)

    raise RuntimeError(
        f'Buoc RPC "{label}" khong hoan tat duoc sau {startup_wait:.0f}s. '
        f'Chi tiet cuoi: {last_error}'
    )


def prepare_output_dir(output_dir):
    images_dir = output_dir / 'images'
    labels_dir = output_dir / 'labels'
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, labels_dir


def spawn_ego_vehicle(world):
    vehicle_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    if vehicle_bp.has_attribute('color'):
        vehicle_bp.set_attribute(
            'color',
            random.choice(vehicle_bp.get_attribute('color').recommended_values),
        )
    vehicle_bp.set_attribute('role_name', 'hero')

    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)
    for spawn_point in spawn_points:
        ego = world.try_spawn_actor(vehicle_bp, spawn_point)
        if ego is not None:
            set_spectator_over_vehicle(world, ego)
            return ego
    raise RuntimeError('Khong spawn duoc Tesla Model 3. Hay reset map hoac xoa bot actor dang ton tai.')


def spawn_npc_vehicles(client, world, traffic_manager, vehicle_count):
    if vehicle_count <= 0:
        return []

    picker = VehicleBlueprintPicker(world.get_blueprint_library())
    spawn_points = world.get_map().get_spawn_points()
    random.shuffle(spawn_points)

    vehicle_count = min(vehicle_count, len(spawn_points))
    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    SetVehicleLightState = carla.command.SetVehicleLightState
    FutureActor = carla.command.FutureActor

    batch = []
    for spawn_point in spawn_points[:vehicle_count]:
        blueprint = picker.choose()
        if blueprint.has_attribute('color'):
            blueprint.set_attribute(
                'color',
                random.choice(blueprint.get_attribute('color').recommended_values),
            )
        if blueprint.has_attribute('driver_id'):
            blueprint.set_attribute(
                'driver_id',
                random.choice(blueprint.get_attribute('driver_id').recommended_values),
            )
        blueprint.set_attribute('role_name', 'autopilot')

        batch.append(
            SpawnActor(blueprint, spawn_point)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
            .then(SetVehicleLightState(FutureActor, carla.VehicleLightState.NONE))
        )

    actor_ids = []
    for response in client.apply_batch_sync(batch, True):
        if not response.error:
            actor_ids.append(response.actor_id)
    return actor_ids


def spawn_walkers(client, world, walker_count):
    if walker_count <= 0:
        return [], []

    walker_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    spawn_points = []
    for _ in range(walker_count * 3):
        location = world.get_random_location_from_navigation()
        if location is not None:
            spawn_points.append(carla.Transform(location))
        if len(spawn_points) >= walker_count:
            break

    SpawnActor = carla.command.SpawnActor
    walker_batch = []
    walker_speeds = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(walker_blueprints)
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        speed = 1.4
        if walker_bp.has_attribute('speed'):
            values = walker_bp.get_attribute('speed').recommended_values
            speed = float(values[1 if len(values) > 1 else 0])
        walker_speeds.append(speed)
        walker_batch.append(SpawnActor(walker_bp, spawn_point))

    walker_ids = []
    valid_speeds = []
    for index, response in enumerate(client.apply_batch_sync(walker_batch, True)):
        if not response.error:
            walker_ids.append(response.actor_id)
            valid_speeds.append(walker_speeds[index])

    if not walker_ids:
        return [], []

    controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    controller_batch = [
        SpawnActor(controller_bp, carla.Transform(), walker_id)
        for walker_id in walker_ids
    ]

    controller_ids = []
    kept_walker_ids = []
    kept_speeds = []
    destroy_batch = []
    for index, response in enumerate(client.apply_batch_sync(controller_batch, True)):
        if response.error:
            destroy_batch.append(carla.command.DestroyActor(walker_ids[index]))
            continue
        controller_ids.append(response.actor_id)
        kept_walker_ids.append(walker_ids[index])
        kept_speeds.append(valid_speeds[index])

    if destroy_batch:
        client.apply_batch(destroy_batch)

    world.tick()
    world.set_pedestrians_cross_factor(0.35)

    walker_pairs = []
    for controller_id, walker_id, max_speed in zip(controller_ids, kept_walker_ids, kept_speeds):
        controller = world.get_actor(controller_id)
        walker_actor = world.get_actor(walker_id)
        if controller is None or walker_actor is None:
            continue
        controller.start()
        controller.set_max_speed(max_speed)
        assign_walker_target(
            world,
            controller,
            walker_actor,
            min_distance=8.0,
            max_distance=60.0,
            max_attempts=12,
        )
        walker_pairs.append((controller_id, walker_id))

    return controller_ids, kept_walker_ids, walker_pairs


def retarget_walkers(world, walker_pairs, idle_speed_threshold, min_distance, max_distance, max_attempts):
    for controller_id, walker_id in walker_pairs:
        controller = world.get_actor(controller_id)
        walker_actor = world.get_actor(walker_id)
        if controller is None or walker_actor is None:
            continue
        if not controller.is_alive or not walker_actor.is_alive:
            continue
        if actor_speed_mps(walker_actor) > idle_speed_threshold:
            continue
        assign_walker_target(
            world,
            controller,
            walker_actor,
            min_distance=min_distance,
            max_distance=max_distance,
            max_attempts=max_attempts,
        )


def destroy_actors(client, actor_ids):
    valid_ids = [actor_id for actor_id in actor_ids if actor_id]
    if valid_ids:
        client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in valid_ids])


def render_preview(display, rgb_frame, preview_frame):
    width = CAM_W // 2
    height = CAM_H // 2
    third_person = cv2.resize(rgb_frame, (width, height))
    ai_view = cv2.resize(preview_frame, (width, height))

    third_surface = pygame.surfarray.make_surface(third_person.swapaxes(0, 1))
    ai_surface = pygame.surfarray.make_surface(ai_view.swapaxes(0, 1))
    display.blit(third_surface, (0, 0))
    display.blit(ai_surface, (width, 0))
    pygame.display.flip()


def main(args):
    output_dir = Path(args.output).resolve()
    images_dir, labels_dir = prepare_output_dir(output_dir)

    client, world = connect_to_carla(
        args.host,
        args.port,
        args.rpc_timeout,
        args.startup_wait,
    )
    traffic_manager = client.get_trafficmanager(args.tm_port)
    blueprint_library = rpc_call_with_retry(
        'get_blueprint_library',
        world.get_blueprint_library,
        args.startup_wait,
    )

    original_settings = rpc_call_with_retry(
        'get_settings',
        world.get_settings,
        args.startup_wait,
    )
    original_weather = clone_weather(
        rpc_call_with_retry(
            'get_weather',
            world.get_weather,
            args.startup_wait,
        )
    )

    settings = rpc_call_with_retry(
        'get_settings',
        world.get_settings,
        args.startup_wait,
    )
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1.0 / max(float(aeb_config.CAM_FPS), 1.0)
    rpc_call_with_retry(
        'apply_settings',
        lambda: world.apply_settings(settings),
        args.startup_wait,
    )
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_global_distance_to_leading_vehicle(3.0)
    traffic_manager.global_percentage_speed_difference(20.0)

    display = None
    if not args.no_display:
        pygame.init()
        display = pygame.display.set_mode((CAM_W, CAM_H // 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('Tesla Model 3 Dataset Collector')

    ego = None
    camera = None
    depth_camera = None
    third_camera = None
    spawned_vehicle_ids = []
    walker_controller_ids = []
    walker_ids = []
    walker_pairs = []
    actor_cache = {}

    camera_queue = queue.LifoQueue(maxsize=3)
    depth_queue = queue.LifoQueue(maxsize=3)
    third_queue = queue.LifoQueue(maxsize=3)

    projection_matrix = build_projection_matrix(CAM_W, CAM_H, CAM_FOV)
    saved_count = len(list(images_dir.glob('*.jpg')))
    frame_count = 0
    last_log_time = time.time()
    last_light_update = 0.0
    last_walker_retarget = 0.0
    current_headlights = None
    last_saved_transform = None
    stuck_seconds = 0.0
    last_recovery_time = -1e9

    try:
        print(f'[*] Map hien tai: {world.get_map().name}')
        ego = spawn_ego_vehicle(world)
        spawned_vehicle_ids = spawn_npc_vehicles(client, world, traffic_manager, args.vehicles)
        walker_controller_ids, walker_ids, walker_pairs = spawn_walkers(client, world, args.walkers)

        traffic_manager.auto_lane_change(ego, False)
        traffic_manager.distance_to_leading_vehicle(ego, 5.0)
        traffic_manager.vehicle_percentage_speed_difference(ego, args.ego_speed_diff)
        ego.set_autopilot(True, traffic_manager.get_port())

        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(CAM_W))
        rgb_bp.set_attribute('image_size_y', str(CAM_H))
        rgb_bp.set_attribute('fov', str(CAM_FOV))
        rgb_bp.set_attribute('sensor_tick', str(settings.fixed_delta_seconds))

        depth_bp = blueprint_library.find('sensor.camera.depth')
        depth_bp.set_attribute('image_size_x', str(CAM_W))
        depth_bp.set_attribute('image_size_y', str(CAM_H))
        depth_bp.set_attribute('fov', str(CAM_FOV))
        depth_bp.set_attribute('sensor_tick', str(settings.fixed_delta_seconds))

        third_bp = blueprint_library.find('sensor.camera.rgb')
        third_bp.set_attribute('image_size_x', str(CAM_W))
        third_bp.set_attribute('image_size_y', str(CAM_H))
        third_bp.set_attribute('fov', '90')
        third_bp.set_attribute('sensor_tick', str(settings.fixed_delta_seconds))

        front_transform = carla.Transform(
            carla.Location(x=aeb_config.CAM_X, z=aeb_config.CAM_Z)
        )
        third_transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8),
            carla.Rotation(pitch=-15.0),
        )

        camera = world.spawn_actor(rgb_bp, front_transform, attach_to=ego)
        depth_camera = world.spawn_actor(depth_bp, front_transform, attach_to=ego)
        third_camera = world.spawn_actor(third_bp, third_transform, attach_to=ego)

        camera.listen(lambda data: sensor_callback(data, camera_queue))
        depth_camera.listen(lambda data: sensor_callback(data, depth_queue))
        third_camera.listen(lambda data: sensor_callback(data, third_queue))

        weather_director = WeatherDirector(world, args.weather_speed, args.weather_change_interval)

        print(f'[*] Tesla Model 3 da bat autopilot va bat dau thu du lieu vao {output_dir}')
        print(f'[*] Muc tieu: {args.frames} frame | traffic={args.vehicles} | walkers={args.walkers}')

        while saved_count < args.frames:
            world.tick()
            snapshot = world.get_snapshot()
            snapshot_frame = snapshot.frame

            weather_director.tick(snapshot.timestamp.delta_seconds)

            now = time.time()
            if now - last_walker_retarget >= args.walker_retarget_interval:
                retarget_walkers(
                    world,
                    walker_pairs,
                    idle_speed_threshold=args.walker_idle_speed,
                    min_distance=args.walker_target_min_distance,
                    max_distance=args.walker_target_max_distance,
                    max_attempts=args.walker_target_attempts,
                )
                last_walker_retarget = now

            weather = world.get_weather()
            enable_headlights = is_dark_weather(weather)
            if current_headlights is None or enable_headlights != current_headlights or now - last_light_update >= 1.0:
                vehicles = [ego]
                vehicles.extend(world.get_actor(actor_id) for actor_id in spawned_vehicle_ids)
                set_vehicle_headlights(vehicles, enable_headlights)
                current_headlights = enable_headlights
                last_light_update = now

            try:
                image_data = receive_latest_frame(camera_queue, snapshot_frame)
                depth_data = receive_latest_frame(depth_queue, image_data.frame)
                third_data = receive_latest_frame(third_queue, image_data.frame)
            except queue.Empty:
                continue

            while snapshot.frame < image_data.frame:
                world.tick()
                snapshot = world.get_snapshot()

            camera_snapshot = snapshot.find(camera.id)
            if camera_snapshot is None:
                continue

            rgb_raw = np.frombuffer(image_data.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
            rgb_frame = cv2.cvtColor(rgb_raw, cv2.COLOR_BGRA2RGB)

            third_raw = np.frombuffer(third_data.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
            third_frame = cv2.cvtColor(third_raw, cv2.COLOR_BGRA2RGB)

            depth_raw = np.frombuffer(depth_data.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4)).astype(np.float32)
            blue_channel = depth_raw[:, :, 0]
            green_channel = depth_raw[:, :, 1]
            red_channel = depth_raw[:, :, 2]
            depth_map = (red_channel + green_channel * 256.0 + blue_channel * 65536.0) / (256.0 ** 3 - 1.0) * 1000.0

            preview_frame = rgb_frame.copy()
            annotations = []
            camera_transform = camera_snapshot.get_transform()

            for actor_snapshot in snapshot:
                if actor_snapshot.id == ego.id:
                    continue

                info = get_actor_info(actor_snapshot.id, world, actor_cache)
                if not info:
                    continue

                if not (
                    info['type_id'].startswith('vehicle.')
                    or info['type_id'].startswith('walker.')
                ):
                    continue

                class_info = classify_actor_type(info['type_id'])
                if not class_info:
                    continue

                bbox_result = get_2d_bbox(
                    actor_snapshot,
                    info['type_id'],
                    info['bbox'],
                    camera_transform,
                    projection_matrix,
                    CAM_W,
                    CAM_H,
                    args.max_distance,
                )
                if not bbox_result:
                    continue

                min_depth, bbox, frame_visible_ratio = bbox_result
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                if bbox_width < 10 or bbox_height < 10 or (bbox_width * bbox_height) < 150:
                    continue
                if frame_visible_ratio < args.min_frame_visibility:
                    continue

                region = rgb_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                if region.size > 0:
                    gray_region = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
                    if np.max(gray_region) < 25:
                        continue

                tolerance = 5.0 + info['extent_x'] * 3.0
                visible_ratio = compute_visibility_ratio(
                    depth_map,
                    bbox[0],
                    bbox[1],
                    bbox[2],
                    bbox[3],
                    min_depth,
                    tolerance=tolerance,
                )
                if visible_ratio < args.min_occlusion_visibility:
                    continue

                class_id = class_info[0]
                class_name = class_info[1]
                cx = ((bbox[0] + bbox[2]) / 2.0) / CAM_W
                cy = ((bbox[1] + bbox[3]) / 2.0) / CAM_H
                bw = bbox_width / CAM_W
                bh = bbox_height / CAM_H
                annotations.append(f'{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')

                cv2.rectangle(preview_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                label = class_name
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    preview_frame,
                    (bbox[0], bbox[1] - label_height - 6),
                    (bbox[0] + label_width + 6, bbox[1]),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    preview_frame,
                    label,
                    (bbox[0] + 3, bbox[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

            if display is not None:
                render_preview(display, third_frame, preview_frame)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        saved_count = args.frames
                        break

            frame_count += 1
            moving_speed = speed_kmh(ego)
            current_transform = ego.get_transform()
            waiting_at_light = False
            try:
                waiting_at_light = ego.is_at_traffic_light()
            except RuntimeError:
                waiting_at_light = False

            if moving_speed < args.stuck_speed_kmh and not waiting_at_light:
                stuck_seconds += snapshot.timestamp.delta_seconds
            else:
                stuck_seconds = 0.0

            if stuck_seconds >= args.stuck_timeout and (now - last_recovery_time) >= args.recovery_cooldown:
                print('[!] Ego bi ket qua lau, dang doi vi tri de tiep tuc thu du lieu...')
                recover_ego_vehicle(
                    world,
                    ego,
                    traffic_manager,
                    [camera_queue, depth_queue, third_queue],
                    args.recovery_min_distance,
                )
                stuck_seconds = 0.0
                last_recovery_time = now
                last_saved_transform = ego.get_transform()
                continue

            moved_enough = last_saved_transform is None
            if last_saved_transform is not None:
                traveled = current_transform.location.distance(last_saved_transform.location)
                rotated = yaw_delta_deg(current_transform.rotation.yaw, last_saved_transform.rotation.yaw)
                moved_enough = traveled >= args.min_save_distance or rotated >= args.min_save_yaw_delta
            should_save = (
                frame_count % args.skip == 0
                and moving_speed >= args.min_speed_kmh
                and moved_enough
                and (annotations or args.save_empty)
            )

            if should_save:
                stem = f'frame_{saved_count:06d}_{image_data.frame}'
                image_path = images_dir / f'{stem}.jpg'
                label_path = labels_dir / f'{stem}.txt'

                cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 93],
                )
                label_path.write_text('\n'.join(annotations), encoding='utf-8')

                saved_count += 1
                last_saved_transform = current_transform
                if saved_count % 50 == 0:
                    elapsed = max(time.time() - last_log_time, 0.001)
                    print(
                        f'  -> Da luu {saved_count}/{args.frames} frame | '
                        f'~{50.0 / elapsed:.1f} frame/s | toc do ego {moving_speed:.1f} km/h'
                    )
                    last_log_time = time.time()

    except KeyboardInterrupt:
        print('\n[*] Dung collector theo yeu cau nguoi dung.')
    finally:
        print('[*] Dang don dep actor va tra lai setting cho CARLA...')
        if camera is not None:
            camera.stop()
            camera.destroy()
        if depth_camera is not None:
            depth_camera.stop()
            depth_camera.destroy()
        if third_camera is not None:
            third_camera.stop()
            third_camera.destroy()

        for controller_id in walker_controller_ids:
            controller = world.get_actor(controller_id)
            if controller is not None:
                try:
                    controller.stop()
                except RuntimeError:
                    pass

        destroy_actors(client, [ego.id] if ego is not None else [])
        destroy_actors(client, spawned_vehicle_ids)
        destroy_actors(client, walker_controller_ids + walker_ids)

        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)
        world.set_weather(original_weather)

        if display is not None:
            pygame.quit()

        print(f'[*] Hoan tat. Du lieu da luu trong: {output_dir}')
        print(f'[*] Tong so anh hien co trong dataset: {saved_count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tesla Model 3 CARLA dataset collector')
    parser.add_argument('--host', default=aeb_config.HOST)
    parser.add_argument('--port', default=aeb_config.PORT, type=int)
    parser.add_argument('--tm-port', default=8000, type=int)
    parser.add_argument('--rpc-timeout', default=20.0, type=float, help='Timeout cho moi lan goi RPC toi CARLA')
    parser.add_argument('--startup-wait', default=180.0, type=float, help='Tong thoi gian toi da de doi CARLA san sang')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--frames', default=5000, type=int, help='So frame can luu vao data/images')
    parser.add_argument('--skip', default=5, type=int, help='Luu 1 frame sau moi N frame simulation')
    parser.add_argument('--vehicles', default=aeb_config.NUM_TRAFFIC_VEHICLES, type=int)
    parser.add_argument('--walkers', default=aeb_config.NUM_PEDESTRIANS, type=int)
    parser.add_argument('--ego-speed-diff', default=-18.0, type=float, help='Am hon nghia la ego chay nhanh hon gioi han toc do')
    parser.add_argument('--weather-speed', default=6.0, type=float, help='Toc do bien doi thoi tiet')
    parser.add_argument('--weather-change-interval', default=45.0, type=float, help='Sau bao nhieu giay simulation thi doi profile weather')
    parser.add_argument('--walker-retarget-interval', default=8.0, type=float)
    parser.add_argument('--walker-idle-speed', default=0.35, type=float, help='Chi doi dich walker neu dang di rat cham')
    parser.add_argument('--walker-target-min-distance', default=8.0, type=float, help='Khoang cach toi thieu cho dich moi cua walker')
    parser.add_argument('--walker-target-max-distance', default=60.0, type=float, help='Khoang cach toi da cho dich moi cua walker')
    parser.add_argument('--walker-target-attempts', default=12, type=int, help='So lan thu lay dich moi cho moi walker')
    parser.add_argument('--min-speed-kmh', default=5.0, type=float)
    parser.add_argument('--min-save-distance', default=3.0, type=float, help='Chi luu khi ego da di chuyen toi thieu N met')
    parser.add_argument('--min-save-yaw-delta', default=8.0, type=float, help='Hoac khi huong xe thay doi toi thieu N do')
    parser.add_argument('--max-distance', default=80.0, type=float)
    parser.add_argument('--min-frame-visibility', default=0.5, type=float, help='Ty le toi thieu cua bbox nam trong khung camera')
    parser.add_argument('--min-occlusion-visibility', default=0.5, type=float, help='Ty le diem nhin thay duoc de giu nhan')
    parser.add_argument('--stuck-speed-kmh', default=1.5, type=float, help='Duoi nguong nay thi xem nhu xe dang dung yen')
    parser.add_argument('--stuck-timeout', default=12.0, type=float, help='Neu ego dung yen qua lau thi tu recover')
    parser.add_argument('--recovery-cooldown', default=20.0, type=float, help='Khoang cho toi thieu giua 2 lan recover')
    parser.add_argument('--recovery-min-distance', default=30.0, type=float, help='Khoang cach toi thieu khi doi spawn moi sau recover')
    parser.add_argument('--save-empty', action='store_true', help='Luu ca frame khong co annotation')
    parser.add_argument('--no-display', action='store_true', help='Tat preview pygame')
    main(parser.parse_args())
