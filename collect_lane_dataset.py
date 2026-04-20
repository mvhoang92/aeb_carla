import argparse
import math
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
    yaw_delta_deg,
)

CAM_W = aeb_config.CAM_WIDTH
CAM_H = aeb_config.CAM_HEIGHT
CAM_FOV = aeb_config.CAM_FOV
DEFAULT_OUTPUT_DIR = CURRENT_DIR / 'lane_data'
SEMANTIC_ROADLINE = 6
SEMANTIC_ROAD = 7


def build_projection_matrix(width, height, fov):
    focal = width / (2.0 * np.tan(fov * np.pi / 360.0))
    matrix = np.identity(3)
    matrix[0, 0] = matrix[1, 1] = focal
    matrix[0, 2] = width / 2.0
    matrix[1, 2] = height / 2.0
    return matrix


def configure_ego_autopilot(traffic_manager, ego_vehicle, speed_diff, ignore_lights_percentage=100.0):
    traffic_manager.auto_lane_change(ego_vehicle, False)
    traffic_manager.distance_to_leading_vehicle(ego_vehicle, 5.0)
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, speed_diff)
    traffic_manager.ignore_lights_percentage(ego_vehicle, ignore_lights_percentage)
    ego_vehicle.set_autopilot(True, traffic_manager.get_port())


def prepare_output_dir(output_dir):
    images_dir = output_dir / 'images'
    masks_dir = output_dir / 'masks'
    overlays_dir = output_dir / 'overlays'
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, masks_dir, overlays_dir


def lateral_shift(transform, shift):
    shifted = carla.Transform(transform.location, transform.rotation)
    shifted.rotation.yaw += 90.0
    return shifted.location + shift * shifted.get_forward_vector()


def extract_semantic_labels(image_data, image_h, image_w):
    semantic_raw = np.frombuffer(image_data.raw_data, dtype=np.uint8).reshape((image_h, image_w, 4))
    # CARLA stores semantic tags in the red channel for raw semantic cameras.
    return semantic_raw[:, :, 2].copy()


def project_world_point(location, camera_transform, projection_matrix, image_w, image_h):
    world_point = np.array([location.x, location.y, location.z, 1.0])
    world_to_camera = np.array(camera_transform.get_inverse_matrix())
    camera_point = np.dot(world_to_camera, world_point)

    if camera_point[0] <= 0.1:
        return None

    point_2d = np.dot(
        projection_matrix,
        np.array([camera_point[1], -camera_point[2], camera_point[0]]),
    )
    point_2d = point_2d / point_2d[2]

    x_coord = float(np.clip(point_2d[0], 0.0, image_w - 1.0))
    y_coord = float(np.clip(point_2d[1], 0.0, image_h - 1.0))
    return x_coord, y_coord


def choose_best_waypoint(candidates, reference_waypoint):
    if not candidates:
        return None

    reference_yaw = reference_waypoint.transform.rotation.yaw
    return min(
        candidates,
        key=lambda waypoint: yaw_delta_deg(waypoint.transform.rotation.yaw, reference_yaw),
    )


def gather_lane_waypoints(current_waypoint, step_distance, distance_ahead, distance_behind):
    lane_waypoints = [current_waypoint]

    travelled = 0.0
    reference_waypoint = current_waypoint
    while travelled < distance_ahead:
        next_waypoints = reference_waypoint.next(step_distance)
        next_waypoint = choose_best_waypoint(next_waypoints, reference_waypoint)
        if next_waypoint is None or next_waypoint.is_junction:
            break
        lane_waypoints.append(next_waypoint)
        reference_waypoint = next_waypoint
        travelled += step_distance

    previous_waypoints = []
    travelled = 0.0
    reference_waypoint = current_waypoint
    while travelled < distance_behind:
        prev_waypoints = reference_waypoint.previous(step_distance)
        prev_waypoint = choose_best_waypoint(prev_waypoints, reference_waypoint)
        if prev_waypoint is None or prev_waypoint.is_junction:
            break
        previous_waypoints.append(prev_waypoint)
        reference_waypoint = prev_waypoint
        travelled += step_distance

    merged = list(reversed(previous_waypoints)) + lane_waypoints
    unique_waypoints = []
    seen_ids = set()
    for waypoint in merged:
        if waypoint.id in seen_ids:
            continue
        seen_ids.add(waypoint.id)
        unique_waypoints.append(waypoint)
    return unique_waypoints


def build_geometry_lane_mask(
    world,
    ego_vehicle,
    camera_transform,
    projection_matrix,
    image_w,
    image_h,
    lookahead_distance,
    behind_distance,
    sample_step,
    lane_margin,
):
    lane_mask = np.zeros((image_h, image_w), dtype=np.uint8)
    line_mask = np.zeros((image_h, image_w), dtype=np.uint8)

    current_waypoint = world.get_map().get_waypoint(
        ego_vehicle.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if current_waypoint is None:
        return lane_mask, line_mask, None, [], []
    if current_waypoint.is_junction:
        return lane_mask, line_mask, current_waypoint, [], []

    lane_waypoints = gather_lane_waypoints(
        current_waypoint,
        step_distance=sample_step,
        distance_ahead=lookahead_distance,
        distance_behind=behind_distance,
    )

    left_pixels = []
    right_pixels = []
    for waypoint in lane_waypoints:
        half_width = max((waypoint.lane_width * 0.5) - lane_margin, 0.75)
        left_world = lateral_shift(waypoint.transform, -half_width) + carla.Location(z=0.05)
        right_world = lateral_shift(waypoint.transform, half_width) + carla.Location(z=0.05)

        left_pixel = project_world_point(
            left_world,
            camera_transform,
            projection_matrix,
            image_w,
            image_h,
        )
        right_pixel = project_world_point(
            right_world,
            camera_transform,
            projection_matrix,
            image_w,
            image_h,
        )
        if left_pixel is None or right_pixel is None:
            continue

        left_pixels.append(left_pixel)
        right_pixels.append(right_pixel)

    if len(left_pixels) < 3 or len(right_pixels) < 3:
        return lane_mask, line_mask, current_waypoint, left_pixels, right_pixels

    polygon = np.array(
        left_pixels + list(reversed(right_pixels)),
        dtype=np.int32,
    )
    if cv2.contourArea(polygon) < 200.0:
        return lane_mask, line_mask, current_waypoint, left_pixels, right_pixels

    cv2.fillPoly(lane_mask, [polygon], 255)
    cv2.polylines(line_mask, [np.array(left_pixels, dtype=np.int32)], False, 255, 5)
    cv2.polylines(line_mask, [np.array(right_pixels, dtype=np.int32)], False, 255, 5)
    return lane_mask, line_mask, current_waypoint, left_pixels, right_pixels


def find_nearest_boundary_x(candidates, target_x, max_offset):
    if candidates.size == 0:
        return None
    nearest_idx = int(np.argmin(np.abs(candidates - target_x)))
    nearest_x = int(candidates[nearest_idx])
    if abs(nearest_x - target_x) > max_offset:
        return None
    return nearest_x


def build_support_mask(binary_mask, kernel_width, kernel_height):
    if kernel_width <= 1 and kernel_height <= 1:
        return binary_mask.astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
    return cv2.dilate(binary_mask.astype(np.uint8), kernel)


def find_nearest_boundary_in_band(support_mask, y_coord, target_x, max_offset, vertical_radius):
    image_h, image_w = support_mask.shape
    x0 = max(0, int(target_x - max_offset))
    x1 = min(image_w, int(target_x + max_offset + 1))
    y0 = max(0, int(y_coord - vertical_radius))
    y1 = min(image_h, int(y_coord + vertical_radius + 1))
    if x0 >= x1 or y0 >= y1:
        return None

    band = support_mask[y0:y1, x0:x1]
    band_y, band_x = np.where(band > 0)
    if band_x.size == 0:
        return None

    band_x = band_x + x0
    band_y = band_y + y0
    scores = np.abs(band_x - target_x) + (0.35 * np.abs(band_y - y_coord))
    nearest_idx = int(np.argmin(scores))
    return int(band_x[nearest_idx])


def smooth_boundary_positions(values, radius=4):
    if not values:
        return values
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - radius)
        end = min(len(values), index + radius + 1)
        smoothed.append(int(round(float(np.median(values[start:end])))))
    return smoothed


def refine_lane_mask_with_semantics(
    coarse_lane_mask,
    semantic_labels,
    boundary_search_px,
    min_lane_width_px,
    min_direct_support_rows,
    min_direct_support_ratio,
    support_vertical_radius,
):
    if semantic_labels is None:
        return coarse_lane_mask.copy(), np.zeros_like(coarse_lane_mask)

    refined_lane_mask = np.zeros_like(coarse_lane_mask)
    refined_line_mask = np.zeros_like(coarse_lane_mask)

    if not np.any(coarse_lane_mask):
        return refined_lane_mask, refined_line_mask

    roadline_mask = semantic_labels == SEMANTIC_ROADLINE
    road_mask = semantic_labels == SEMANTIC_ROAD
    if not np.any(roadline_mask) and not np.any(road_mask):
        return refined_lane_mask, refined_line_mask

    roadline_support_mask = build_support_mask(roadline_mask, kernel_width=7, kernel_height=5)

    rows = []
    left_bounds = []
    right_bounds = []
    previous_left = None
    previous_right = None
    paired_support_rows = 0
    either_support_rows = 0
    left_support_rows = 0
    right_support_rows = 0

    row_indices = np.where(np.any(coarse_lane_mask > 0, axis=1))[0]
    for y_coord in row_indices:
        lane_pixels = np.flatnonzero(coarse_lane_mask[y_coord] > 0)
        if lane_pixels.size < 2:
            continue

        coarse_left = int(lane_pixels[0])
        coarse_right = int(lane_pixels[-1])

        snapped_left = find_nearest_boundary_in_band(
            roadline_support_mask,
            y_coord,
            coarse_left,
            boundary_search_px,
            support_vertical_radius,
        )
        snapped_right = find_nearest_boundary_in_band(
            roadline_support_mask,
            y_coord,
            coarse_right,
            boundary_search_px,
            support_vertical_radius,
        )
        support_left = snapped_left is not None
        support_right = snapped_right is not None

        if snapped_left is None:
            snapped_left = coarse_left if previous_left is None else int(round((0.7 * coarse_left) + (0.3 * previous_left)))
        if snapped_right is None:
            snapped_right = coarse_right if previous_right is None else int(round((0.7 * coarse_right) + (0.3 * previous_right)))

        if snapped_right <= snapped_left:
            continue

        if (snapped_right - snapped_left) < min_lane_width_px:
            center_x = 0.5 * (snapped_left + snapped_right)
            half_width = max(int(round((coarse_right - coarse_left) * 0.5)), min_lane_width_px // 2)
            snapped_left = max(int(round(center_x - half_width)), 0)
            snapped_right = min(int(round(center_x + half_width)), coarse_lane_mask.shape[1] - 1)

        if road_mask.any():
            road_pixels = np.flatnonzero(road_mask[y_coord] | roadline_mask[y_coord])
            if road_pixels.size > 0:
                snapped_left = max(snapped_left, int(road_pixels[0]))
                snapped_right = min(snapped_right, int(road_pixels[-1]))
                if snapped_right <= snapped_left:
                    continue

        rows.append(int(y_coord))
        left_bounds.append(int(snapped_left))
        right_bounds.append(int(snapped_right))
        if support_left:
            left_support_rows += 1
        if support_right:
            right_support_rows += 1
        if support_left or support_right:
            either_support_rows += 1
        if support_left and support_right:
            paired_support_rows += 1
        previous_left = int(snapped_left)
        previous_right = int(snapped_right)

    if len(rows) < 3:
        return refined_lane_mask, refined_line_mask
    strongest_side_rows = max(left_support_rows, right_support_rows)
    if either_support_rows < min_direct_support_rows:
        return np.zeros_like(coarse_lane_mask), np.zeros_like(coarse_lane_mask)
    if (either_support_rows / float(len(rows))) < min_direct_support_ratio:
        return np.zeros_like(coarse_lane_mask), np.zeros_like(coarse_lane_mask)
    if strongest_side_rows < max(6, int(round(min_direct_support_rows * 0.55))):
        return np.zeros_like(coarse_lane_mask), np.zeros_like(coarse_lane_mask)
    if paired_support_rows == 0:
        return np.zeros_like(coarse_lane_mask), np.zeros_like(coarse_lane_mask)

    left_bounds = smooth_boundary_positions(left_bounds)
    right_bounds = smooth_boundary_positions(right_bounds)

    left_poly = np.column_stack([left_bounds, rows]).astype(np.int32)
    right_poly = np.column_stack([right_bounds, rows]).astype(np.int32)
    polygon = np.vstack([left_poly, right_poly[::-1]])

    refined_lane_mask = np.zeros_like(coarse_lane_mask)
    cv2.fillPoly(refined_lane_mask, [polygon], 255)
    cv2.polylines(refined_line_mask, [left_poly], False, 255, 4)
    cv2.polylines(refined_line_mask, [right_poly], False, 255, 4)

    close_kernel = np.ones((7, 7), dtype=np.uint8)
    refined_lane_mask = cv2.morphologyEx(refined_lane_mask, cv2.MORPH_CLOSE, close_kernel)
    return refined_lane_mask, refined_line_mask


def build_ego_lane_mask(
    world,
    ego_vehicle,
    camera_transform,
    projection_matrix,
    image_w,
    image_h,
    lookahead_distance,
    behind_distance,
    sample_step,
    lane_margin,
    semantic_labels=None,
    boundary_search_px=72,
    min_lane_width_px=24,
    min_direct_support_rows=18,
    min_direct_support_ratio=0.32,
    support_vertical_radius=8,
):
    coarse_lane_mask, coarse_line_mask, current_waypoint, left_pixels, right_pixels = build_geometry_lane_mask(
        world,
        ego_vehicle,
        camera_transform,
        projection_matrix,
        image_w,
        image_h,
        lookahead_distance,
        behind_distance,
        sample_step,
        lane_margin,
    )

    refined_lane_mask, refined_line_mask = refine_lane_mask_with_semantics(
        coarse_lane_mask,
        semantic_labels,
        boundary_search_px,
        min_lane_width_px,
        min_direct_support_rows,
        min_direct_support_ratio,
        support_vertical_radius,
    )
    if semantic_labels is not None:
        return refined_lane_mask, refined_line_mask, current_waypoint, left_pixels, right_pixels
    if np.any(refined_line_mask):
        return refined_lane_mask, refined_line_mask, current_waypoint, left_pixels, right_pixels
    return coarse_lane_mask, coarse_line_mask, current_waypoint, left_pixels, right_pixels


def apply_lane_overlay(rgb_frame, lane_mask, line_mask):
    overlay = rgb_frame.copy()

    if np.any(lane_mask):
        lane_color = np.zeros_like(rgb_frame)
        lane_color[:, :] = (0, 220, 170)
        blended = cv2.addWeighted(rgb_frame, 0.6, lane_color, 0.4, 0.0)
        overlay[lane_mask > 0] = blended[lane_mask > 0]

    if np.any(line_mask):
        overlay[line_mask > 0] = (255, 220, 0)

    return overlay


def render_mask_preview(lane_mask, line_mask):
    mask_preview = np.zeros((lane_mask.shape[0], lane_mask.shape[1], 3), dtype=np.uint8)

    if np.any(lane_mask):
        mask_preview[lane_mask > 0] = (0, 220, 170)
    if np.any(line_mask):
        mask_preview[line_mask > 0] = (255, 220, 0)

    return mask_preview


def compose_lane_label_panel(rgb_frame, overlay_frame, lane_mask, line_mask):
    mask_preview = render_mask_preview(lane_mask, line_mask)
    return np.hstack([rgb_frame, overlay_frame, mask_preview])


def render_preview(display, third_frame, overlay_frame):
    width = CAM_W // 2
    height = CAM_H // 2
    third_person = cv2.resize(third_frame, (width, height))
    overlay_view = cv2.resize(overlay_frame, (width, height))

    third_surface = pygame.surfarray.make_surface(third_person.swapaxes(0, 1))
    overlay_surface = pygame.surfarray.make_surface(overlay_view.swapaxes(0, 1))
    display.blit(third_surface, (0, 0))
    display.blit(overlay_surface, (width, 0))
    pygame.display.flip()


def write_dataset_meta(output_dir):
    meta_path = output_dir / 'lane_data.yaml'
    meta_path.write_text(
        '\n'.join([
            'task: segmentation',
            f'path: {output_dir.as_posix()}',
            'images: images',
            'masks: masks',
            'overlays: overlays',
            'nc: 2',
            'names:',
            '  0: background',
            '  1: ego_lane',
            'mask_encoding:',
            '  background: 0',
            '  ego_lane: 255',
        ]) + '\n',
        encoding='utf-8',
    )


def main(args):
    output_dir = Path(args.output).resolve()
    images_dir, masks_dir, overlays_dir = prepare_output_dir(output_dir)
    write_dataset_meta(output_dir)

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
    original_weather = rpc_call_with_retry(
        'get_weather',
        world.get_weather,
        args.startup_wait,
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
    display = None
    if not args.no_display:
        display = pygame.display.set_mode((CAM_W, CAM_H // 2), pygame.HWSURFACE | pygame.DOUBLEBUF)
        pygame.display.set_caption('CARLA Ego Lane Dataset Collector')

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
    saved_count = len(list(images_dir.glob('*.jpg')))
    frame_count = 0
    last_log_time = time.time()
    last_light_update = 0.0
    last_walker_retarget = 0.0
    last_saved_transform = None
    current_headlights = None
    stuck_seconds = 0.0
    last_recovery_time = -1e9

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

        front_transform = carla.Transform(
            carla.Location(x=aeb_config.CAM_X, z=aeb_config.CAM_Z)
        )
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

        print(f'[*] Dang thu du lieu lane vao {output_dir}')
        print(f'[*] Muc tieu: {args.frames} frame | traffic={args.vehicles} | walkers={args.walkers}')

        while saved_count < args.frames:
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
            lane_pixels = int(np.count_nonzero(lane_mask))
            overlay_frame = apply_lane_overlay(rgb_frame, lane_mask, line_mask)
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

            if display is not None:
                render_preview(display, third_frame, overlay_frame)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (
                        event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
                    ):
                        saved_count = args.frames
                        break

            frame_count += 1
            ego_speed = speed_kmh(ego_vehicle)
            current_transform = ego_vehicle.get_transform()
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
                print('[!] Ego bi ket qua lau, dang doi vi tri de tiep tuc thu du lieu lane...')
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
                last_saved_transform = None
                continue

            moved_enough = last_saved_transform is None
            if last_saved_transform is not None:
                travelled = current_transform.location.distance(last_saved_transform.location)
                rotated = yaw_delta_deg(current_transform.rotation.yaw, last_saved_transform.rotation.yaw)
                moved_enough = travelled >= args.min_save_distance or rotated >= args.min_save_yaw_delta

            save_empty_mask = (
                args.save_empty_interval > 0
                and frame_count % args.save_empty_interval == 0
                and no_lane_zone
            )

            should_save = (
                frame_count % args.skip == 0
                and ego_speed >= args.min_speed_kmh
                and moved_enough
                and (lane_pixels >= args.min_lane_pixels or save_empty_mask)
            )

            if should_save:
                stem = f'frame_{saved_count:06d}_{front_data.frame}'
                image_path = images_dir / f'{stem}.jpg'
                mask_path = masks_dir / f'{stem}.png'
                overlay_path = overlays_dir / f'{stem}.jpg'

                cv2.imwrite(
                    str(image_path),
                    cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 93],
                )
                cv2.imwrite(str(mask_path), lane_mask)
                label_panel = compose_lane_label_panel(rgb_frame, overlay_frame, lane_mask, line_mask)
                cv2.imwrite(
                    str(overlay_path),
                    cv2.cvtColor(label_panel, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )

                saved_count += 1
                last_saved_transform = current_transform
                if saved_count % 50 == 0:
                    elapsed = max(time.time() - last_log_time, 0.001)
                    lane_state = 'co lane' if lane_pixels >= args.min_lane_pixels else 'mask rong'
                    print(
                        f'  -> Da luu {saved_count}/{args.frames} frame lane | '
                        f'~{50.0 / elapsed:.1f} frame/s | toc do ego {ego_speed:.1f} km/h | {lane_state}'
                    )
                    last_log_time = time.time()

    except KeyboardInterrupt:
        print('\n[*] Dung collector lane theo yeu cau nguoi dung.')
    finally:
        print('[*] Dang don dep actor va tra lai setting cho CARLA...')
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

        print(f'[*] Hoan tat. Du lieu lane da luu trong: {output_dir}')
        print(f'[*] Tong so anh hien co trong dataset lane: {saved_count}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CARLA ego-lane dataset collector')
    parser.add_argument('--host', default=aeb_config.HOST)
    parser.add_argument('--port', default=aeb_config.PORT, type=int)
    parser.add_argument('--tm-port', default=8000, type=int)
    parser.add_argument('--rpc-timeout', default=20.0, type=float)
    parser.add_argument('--startup-wait', default=180.0, type=float)
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument('--frames', default=5000, type=int)
    parser.add_argument('--skip', default=4, type=int)
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
    parser.add_argument('--min-speed-kmh', default=12.0, type=float)
    parser.add_argument('--min-save-distance', default=4.5, type=float)
    parser.add_argument('--min-save-yaw-delta', default=10.0, type=float)
    parser.add_argument('--min-lane-pixels', default=3500, type=int)
    parser.add_argument('--save-empty-interval', default=12, type=int)
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
    parser.add_argument('--no-display', action='store_true')
    main(parser.parse_args())
