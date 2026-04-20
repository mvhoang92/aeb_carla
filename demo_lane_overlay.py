import argparse
import queue
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
from collect_dataset import (
    WeatherDirector,
    clone_weather,
    connect_to_carla,
    is_dark_weather,
    receive_latest_frame,
    recover_ego_vehicle,
    retarget_walkers,
    rpc_call_with_retry,
    sensor_callback,
    set_vehicle_headlights,
    spawn_ego_vehicle,
    spawn_npc_vehicles,
    spawn_walkers,
    speed_kmh,
)
from collect_lane_dataset import (
    CAM_FOV,
    CAM_H,
    CAM_W,
    apply_lane_overlay,
    build_ego_lane_mask,
    build_projection_matrix,
    configure_ego_autopilot,
    extract_semantic_labels,
    render_preview,
)


def main(args):
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

    pygame.init()
    display = pygame.display.set_mode((CAM_W, CAM_H // 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption('CARLA Lane Overlay Demo')
    font = pygame.font.SysFont('consolas', 20, bold=True)

    ego_vehicle = None
    front_camera = None
    semantic_camera = None
    third_camera = None
    spawned_vehicle_ids = []
    walker_controller_ids = []
    walker_ids = []
    walker_pairs = []

    front_queue = queue.LifoQueue(maxsize=3)
    semantic_queue = queue.LifoQueue(maxsize=3)
    third_queue = queue.LifoQueue(maxsize=3)
    projection_matrix = build_projection_matrix(CAM_W, CAM_H, CAM_FOV)

    last_light_update = 0.0
    last_walker_retarget = 0.0
    last_recovery_time = -1e9
    current_headlights = None
    stuck_seconds = 0.0

    try:
        print(f'[*] Map hien tai: {world.get_map().name}')
        ego_vehicle = spawn_ego_vehicle(world)
        if args.vehicles > 0:
            spawned_vehicle_ids = spawn_npc_vehicles(client, world, traffic_manager, args.vehicles)
        if args.walkers > 0:
            walker_controller_ids, walker_ids, walker_pairs = spawn_walkers(client, world, args.walkers)

        configure_ego_autopilot(
            traffic_manager,
            ego_vehicle,
            args.ego_speed_diff,
            args.ignore_lights_percentage,
        )

        rgb_bp = blueprint_library.find('sensor.camera.rgb')
        rgb_bp.set_attribute('image_size_x', str(CAM_W))
        rgb_bp.set_attribute('image_size_y', str(CAM_H))
        rgb_bp.set_attribute('fov', str(CAM_FOV))
        rgb_bp.set_attribute('sensor_tick', str(settings.fixed_delta_seconds))

        semantic_bp = blueprint_library.find('sensor.camera.semantic_segmentation')
        semantic_bp.set_attribute('image_size_x', str(CAM_W))
        semantic_bp.set_attribute('image_size_y', str(CAM_H))
        semantic_bp.set_attribute('fov', str(CAM_FOV))
        semantic_bp.set_attribute('sensor_tick', str(settings.fixed_delta_seconds))

        third_bp = blueprint_library.find('sensor.camera.rgb')
        third_bp.set_attribute('image_size_x', str(CAM_W))
        third_bp.set_attribute('image_size_y', str(CAM_H))
        third_bp.set_attribute('fov', '90')
        third_bp.set_attribute('sensor_tick', str(settings.fixed_delta_seconds))

        front_transform = carla.Transform(carla.Location(x=aeb_config.CAM_X, z=aeb_config.CAM_Z))
        third_transform = carla.Transform(
            carla.Location(x=-5.5, z=2.8),
            carla.Rotation(pitch=-15.0),
        )

        front_camera = world.spawn_actor(rgb_bp, front_transform, attach_to=ego_vehicle)
        semantic_camera = world.spawn_actor(semantic_bp, front_transform, attach_to=ego_vehicle)
        third_camera = world.spawn_actor(third_bp, third_transform, attach_to=ego_vehicle)
        front_camera.listen(lambda data: sensor_callback(data, front_queue))
        semantic_camera.listen(lambda data: sensor_callback(data, semantic_queue))
        third_camera.listen(lambda data: sensor_callback(data, third_queue))

        weather_director = WeatherDirector(world, args.weather_speed, args.weather_change_interval)
        print('[*] Dang chay demo lane overlay. Nhan ESC de thoat.')

        while True:
            world.tick()
            snapshot = world.get_snapshot()
            snapshot_frame = snapshot.frame
            weather_director.tick(snapshot.timestamp.delta_seconds)

            now = time.time()
            if walker_pairs and now - last_walker_retarget >= args.walker_retarget_interval:
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
                vehicles = [ego_vehicle]
                vehicles.extend(world.get_actor(actor_id) for actor_id in spawned_vehicle_ids)
                set_vehicle_headlights(vehicles, enable_headlights)
                current_headlights = enable_headlights
                last_light_update = now

            try:
                front_data = receive_latest_frame(front_queue, snapshot_frame)
                semantic_data = receive_latest_frame(semantic_queue, front_data.frame)
                third_data = receive_latest_frame(third_queue, front_data.frame)
            except queue.Empty:
                continue

            while snapshot.frame < front_data.frame:
                world.tick()
                snapshot = world.get_snapshot()

            camera_snapshot = snapshot.find(front_camera.id)
            if camera_snapshot is None:
                continue

            rgb_raw = np.frombuffer(front_data.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
            rgb_frame = cv2.cvtColor(rgb_raw, cv2.COLOR_BGRA2RGB)
            semantic_labels = extract_semantic_labels(semantic_data, CAM_H, CAM_W)
            third_raw = np.frombuffer(third_data.raw_data, dtype=np.uint8).reshape((CAM_H, CAM_W, 4))
            third_frame = cv2.cvtColor(third_raw, cv2.COLOR_BGRA2RGB)

            lane_mask, line_mask, current_waypoint, _, _ = build_ego_lane_mask(
                world,
                ego_vehicle,
                camera_snapshot.get_transform(),
                projection_matrix,
                CAM_W,
                CAM_H,
                lookahead_distance=args.lookahead_distance,
                behind_distance=args.behind_distance,
                sample_step=args.sample_step,
                lane_margin=args.lane_margin,
                semantic_labels=semantic_labels,
                boundary_search_px=args.semantic_boundary_search,
                min_lane_width_px=args.min_lane_width_px,
                min_direct_support_rows=args.min_direct_support_rows,
                min_direct_support_ratio=args.min_direct_support_ratio,
                support_vertical_radius=args.support_vertical_radius,
            )
            overlay_frame = apply_lane_overlay(rgb_frame, lane_mask, line_mask)
            lane_pixels = int(np.count_nonzero(lane_mask))
            ego_speed = speed_kmh(ego_vehicle)
            no_lane_zone = current_waypoint is None or current_waypoint.is_junction or lane_pixels == 0

            if current_waypoint is not None:
                cv2.putText(
                    overlay_frame,
                    f'road {current_waypoint.road_id} lane {current_waypoint.lane_id}',
                    (18, 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            lane_status = 'no lane markings' if no_lane_zone else f'lane pixels: {lane_pixels}'
            cv2.putText(
                overlay_frame,
                lane_status,
                (18, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay_frame,
                f'speed: {ego_speed:.1f} km/h',
                (18, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            render_preview(display, third_frame, overlay_frame)
            title = font.render(f'Weather: {weather_director.profile_name}', True, (255, 255, 255))
            display.blit(title, (12, 10))
            pygame.display.flip()

            waiting_at_light = False
            try:
                waiting_at_light = ego_vehicle.is_at_traffic_light()
            except RuntimeError:
                waiting_at_light = False

            if ego_speed < args.stuck_speed_kmh and not waiting_at_light:
                stuck_seconds += snapshot.timestamp.delta_seconds
            else:
                stuck_seconds = 0.0

            if stuck_seconds >= args.stuck_timeout and (now - last_recovery_time) >= args.recovery_cooldown:
                print('[!] Ego bi ket qua lau, dang recover de tiep tuc demo...')
                recover_ego_vehicle(
                    world,
                    ego_vehicle,
                    traffic_manager,
                    [front_queue, semantic_queue, third_queue],
                    args.recovery_min_distance,
                )
                configure_ego_autopilot(
                    traffic_manager,
                    ego_vehicle,
                    args.ego_speed_diff,
                    args.ignore_lights_percentage,
                )
                stuck_seconds = 0.0
                last_recovery_time = now
                continue

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                ):
                    return

    finally:
        print('[*] Dang don dep demo lane overlay...')
        if front_camera is not None:
            front_camera.stop()
            front_camera.destroy()
        if semantic_camera is not None:
            semantic_camera.stop()
            semantic_camera.destroy()
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

        if ego_vehicle is not None:
            client.apply_batch([carla.command.DestroyActor(ego_vehicle.id)])
        if spawned_vehicle_ids:
            client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in spawned_vehicle_ids])
        if walker_controller_ids or walker_ids:
            all_walkers = walker_controller_ids + walker_ids
            client.apply_batch([carla.command.DestroyActor(actor_id) for actor_id in all_walkers])

        traffic_manager.set_synchronous_mode(False)
        world.apply_settings(original_settings)
        world.set_weather(original_weather)
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live CARLA ego-lane overlay demo')
    parser.add_argument('--host', default=aeb_config.HOST)
    parser.add_argument('--port', default=aeb_config.PORT, type=int)
    parser.add_argument('--tm-port', default=8000, type=int)
    parser.add_argument('--rpc-timeout', default=20.0, type=float)
    parser.add_argument('--startup-wait', default=180.0, type=float)
    parser.add_argument('--vehicles', default=0, type=int)
    parser.add_argument('--walkers', default=0, type=int)
    parser.add_argument('--ego-speed-diff', default=-45.0, type=float)
    parser.add_argument('--ignore-lights-percentage', default=100.0, type=float)
    parser.add_argument('--weather-speed', default=10.0, type=float)
    parser.add_argument('--weather-change-interval', default=25.0, type=float)
    parser.add_argument('--walker-retarget-interval', default=8.0, type=float)
    parser.add_argument('--walker-idle-speed', default=0.35, type=float)
    parser.add_argument('--walker-target-min-distance', default=8.0, type=float)
    parser.add_argument('--walker-target-max-distance', default=60.0, type=float)
    parser.add_argument('--walker-target-attempts', default=12, type=int)
    parser.add_argument('--lookahead-distance', default=55.0, type=float)
    parser.add_argument('--behind-distance', default=8.0, type=float)
    parser.add_argument('--sample-step', default=0.5, type=float)
    parser.add_argument('--lane-margin', default=0.10, type=float)
    parser.add_argument('--semantic-boundary-search', default=96, type=int)
    parser.add_argument('--min-lane-width-px', default=28, type=int)
    parser.add_argument('--min-direct-support-rows', default=12, type=int)
    parser.add_argument('--min-direct-support-ratio', default=0.18, type=float)
    parser.add_argument('--support-vertical-radius', default=10, type=int)
    parser.add_argument('--stuck-speed-kmh', default=1.5, type=float)
    parser.add_argument('--stuck-timeout', default=12.0, type=float)
    parser.add_argument('--recovery-cooldown', default=20.0, type=float)
    parser.add_argument('--recovery-min-distance', default=30.0, type=float)
    main(parser.parse_args())
