import sys
import glob
try:
    sys.path.append(glob.glob('../PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')[0])
except IndexError:
    pass

import carla
import pygame

from configs import aeb_config
from modules.environment import CarlaEnv
from modules.sensors import CameraSensor
from modules.perception import YoloDetector # <-- MỚI
from modules.lane_detection import LaneDetector
from modules.lane_controller import LaneController
from modules.lane_warning import LaneDepartureWarning
from modules.adaptive_speed import AdaptiveSpeedController

def main():
    pygame.init()
    pygame.font.init() # Khởi tạo Font vẽ chữ
    sys_font = pygame.font.SysFont('consolas', 16, bold=True)
    
    display = pygame.display.set_mode((aeb_config.WINDOW_WIDTH, aeb_config.WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Tesla AEB - AI Perception Active")
    clock = pygame.time.Clock()

    env = CarlaEnv(aeb_config.HOST, aeb_config.PORT)
    # Khởi tạo Bộ não AI
    detector = YoloDetector(aeb_config.YOLO_WEIGHTS, aeb_config.CONFIDENCE_THRESHOLD)

    # Khởi tạo Lane Detection & Controller
    lane_detector = LaneDetector(aeb_config.CAM_HEIGHT, aeb_config.CAM_WIDTH)
    lane_controller = LaneController(aeb_config.CAM_WIDTH, aeb_config.LKA_MAX_STEERING)
    lane_warning = LaneDepartureWarning(aeb_config.CAM_WIDTH, warning_threshold=80)
    speed_controller = AdaptiveSpeedController(max_speed=0.8, min_speed=0.3)

    cam_front = None
    cam_third = None
    
    try:
        env.spawn_ego_vehicle()
        env.spawn_traffic(aeb_config.NUM_TRAFFIC_VEHICLES)
        env.spawn_pedestrians(aeb_config.NUM_PEDESTRIANS) # Thả thêm người
        
        # Gắn Camera
        front_transform = carla.Transform(carla.Location(x=1.5, z=1.4))
        cam_front = CameraSensor(env.ego_vehicle, aeb_config.CAM_WIDTH, aeb_config.CAM_HEIGHT, front_transform)

        third_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        cam_third = CameraSensor(env.ego_vehicle, aeb_config.CAM_WIDTH, aeb_config.CAM_HEIGHT, third_transform)

        env.ego_vehicle.set_autopilot(True)
        print("\n=> Hệ thống đang chạy. Bấm dấu X ở cửa sổ Pygame hoặc Ctrl+C để thoát.")

        while True:
            clock.tick_busy_loop(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # 1. Render ảnh nền Camera
            cam_third.render(display, x_offset=0, y_offset=0)
            cam_front.render(display, x_offset=aeb_config.CAM_WIDTH, y_offset=0)

            # Kẻ vạch chia đôi
            pygame.draw.line(display, (255, 255, 255), (aeb_config.CAM_WIDTH, 0), (aeb_config.CAM_WIDTH, aeb_config.CAM_HEIGHT), 2)

            # 2. LANE KEEPING ASSIST (LKA) + ADAPTIVE SPEED
            steering_angle = 0.0
            throttle = 0.5
            warning_msg = "Lane OK"

            if aeb_config.LKA_ENABLED and cam_front.image_array is not None:
                lane_center, left_lines, right_lines = lane_detector.get_lane_center(cam_front.image_array)
                steering_angle = lane_controller.calculate_steering_angle(lane_center)

                # Adaptive Speed Control
                throttle = speed_controller.calculate_adaptive_speed(steering_angle)

                # Lane Departure Warning
                warning_active, warning_side, distance = lane_warning.check_departure(lane_center, left_lines, right_lines)
                warning_msg = lane_warning.get_warning_message()

                # Áp dụng steering angle vào xe
                if env.ego_vehicle:
                    control = carla.VehicleControl()
                    control.throttle = throttle
                    control.steer = steering_angle
                    env.ego_vehicle.apply_control(control)

                # Vẽ vạch làn lên ảnh
                lane_image = lane_detector.draw_lanes(cam_front.image_array, left_lines, right_lines)
                lane_surface = pygame.surfarray.make_surface(lane_image.swapaxes(0, 1))
                display.blit(lane_surface, (aeb_config.CAM_WIDTH, 0))

                # Vẽ tâm làn
                if lane_center is not None:
                    color = (255, 0, 0) if warning_active else (0, 255, 0)
                    pygame.draw.circle(display, color, (int(lane_center) + aeb_config.CAM_WIDTH, int(aeb_config.CAM_HEIGHT * 0.75)), 5)

            # 3. CHẠY AI & VẼ BOUNDING BOX TRỰC TIẾP
            if cam_front.image_array is not None:
                boxes = detector.detect(cam_front.image_array)

                for box in boxes:
                    # Vì cam_front hiển thị ở nửa bên phải màn hình, ta phải cộng thêm CAM_WIDTH vào trục X
                    x = box['xmin'] + aeb_config.CAM_WIDTH
                    y = box['ymin']
                    w = box['xmax'] - box['xmin']
                    h = box['ymax'] - box['ymin']

                    color = detector.colors.get(box['class_id'], (255, 255, 255))

                    # Vẽ khung chữ nhật
                    pygame.draw.rect(display, color, (x, y, w, h), 2)

                    # Vẽ text Tên Class + Độ tự tin
                    label = f"{box['name']} {box['conf']:.2f}"
                    text_surface = sys_font.render(label, True, (0, 0, 0), color) # Chữ đen, nền màu
                    display.blit(text_surface, (x, max(0, y - 20)))

            # Vẽ thông tin steering angle, throttle, warning
            steering_text = f"Steering: {steering_angle:.3f}"
            throttle_text = f"Throttle: {throttle:.2f}"
            warning_text = warning_msg

            steering_surface = sys_font.render(steering_text, True, (255, 255, 0))
            throttle_surface = sys_font.render(throttle_text, True, (255, 255, 0))

            # Màu cảnh báo
            warning_color = (0, 0, 255) if warning_active else (0, 255, 0)
            warning_surface = sys_font.render(warning_text, True, warning_color)

            display.blit(steering_surface, (10, 10))
            display.blit(throttle_surface, (10, 35))
            display.blit(warning_surface, (10, 60))

            pygame.display.flip()

    except KeyboardInterrupt:
        pass
    finally:
        if cam_front: cam_front.destroy()
        if cam_third: cam_third.destroy()
        env.cleanup()
        pygame.quit()

if __name__ == '__main__':
    main()