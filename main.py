import os
import sys
import glob
import pygame
import time

# Sync paths
current_dir = os.path.dirname(os.path.abspath(__file__))
carla_egg_path = glob.glob(os.path.join(current_dir, '../PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg'))
if carla_egg_path:
    sys.path.append(carla_egg_path[0])

import carla
from configs import aeb_config
from modules.environment import CarlaEnv
from modules.sensors import CameraSensor
from modules.perception import YoloDetector
from modules.aeb_system import AEBSystem
from modules.controller import VehicleController

def main():
    # 1. Khởi tạo Giao diện & Hệ thống
    pygame.init()
    display = pygame.display.set_mode((aeb_config.WINDOW_WIDTH, aeb_config.WINDOW_HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
    pygame.display.set_caption("Tesla AEB System - Modular Refactored")
    clock = pygame.time.Clock()
    sys_font = pygame.font.SysFont('consolas', 18, bold=True)

    # 2. Khởi tạo Modules (Sense - Think - Act)
    env = CarlaEnv(aeb_config.HOST, aeb_config.PORT)
    detector = YoloDetector(aeb_config.YOLO_WEIGHTS, aeb_config.CONFIDENCE_THRESHOLD) # Sense (Perception)
    aeb_brain = AEBSystem() # Think
    
    try:
        # 3. Chuẩn bị môi trường & Actor
        ego_vehicle = env.spawn_ego_vehicle()
        env.spawn_traffic(aeb_config.NUM_TRAFFIC_VEHICLES)
        env.spawn_pedestrians(aeb_config.NUM_PEDESTRIANS)
        
        controller = VehicleController(ego_vehicle) # Act
        controller.set_autopilot(True)

        # 4. Gắn Cảm biến (Sense - Data collection)
        # Camera quan sát thứ 3
        third_transform = carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15.0))
        cam_third = CameraSensor(ego_vehicle, aeb_config.CAM_WIDTH, aeb_config.CAM_HEIGHT, third_transform)
        
        # Camera AEB (Kính lái - Tesla Specs)
        front_transform = carla.Transform(carla.Location(x=aeb_config.CAM_X, z=aeb_config.CAM_Z))
        cam_front = CameraSensor(ego_vehicle, aeb_config.CAM_WIDTH, aeb_config.CAM_HEIGHT, front_transform)

        print("\n[+] Hệ thống AEB đã sẵn sàng. Đang giám sát...")

        while True:
            clock.tick_busy_loop(60)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            # --- LUỒNG XỬ LÝ CHÍNH ---
            
            # STEP 1: RENDER (Hiển thị preview)
            cam_third.render(display, x_offset=0, y_offset=0)
            cam_front.render(display, x_offset=aeb_config.CAM_WIDTH, y_offset=0)
            
            # STEP 2: PERCEPTION (Nhận diện vật thể)
            if cam_front.image_array is not None:
                boxes = detector.detect(cam_front.image_array)
                
                # STEP 3: THINK (Phân tích rủi ro AEB)
                should_brake, risk, target = aeb_brain.analyze_risk(boxes)
                
                # STEP 4: ACT (Điều khiển phanh)
                if should_brake:
                    controller.apply_aeb_brake(brake_value=1.0)
                else:
                    controller.release_aeb()

                # VẼ BOX LÊN MÀN HÌNH ĐỂ GIÁM SÁT
                for box in boxes:
                    color = detector.colors.get(box['class_id'], (255, 255, 255))
                    # Nếu là vật đang gây nguy hiểm, vẽ màu đỏ đậm
                    if target and box == target:
                        color = (255, 0, 0)
                        pygame.draw.rect(display, color, (box['xmin'] + aeb_config.CAM_WIDTH, box['ymin'], box['xmax']-box['xmin'], box['ymax']-box['ymin']), 4)
                    else:
                        pygame.draw.rect(display, color, (box['xmin'] + aeb_config.CAM_WIDTH, box['ymin'], box['xmax']-box['xmin'], box['ymax']-box['ymin']), 2)
                    
                    label = f"{box['name']} {box['conf']:.2f}"
                    text_surf = sys_font.render(label, True, (0,0,0), color)
                    display.blit(text_surf, (box['xmin'] + aeb_config.CAM_WIDTH, max(0, box['ymin'] - 20)))

            # HIỂN THỊ HUD TRẠNG THÁI
            state = controller.get_state()
            status_color = (255, 0, 0) if state['aeb_active'] else (0, 255, 0)
            status_text = "AEB ACTIVE - BRAKING!" if state['aeb_active'] else "AEB MONITORING"
            
            hud_surf = sys_font.render(f"STATUS: {status_text}", True, status_color)
            display.blit(hud_surf, (20, 20))
            speed_surf = sys_font.render(f"SPEED: {state['speed']:.1f} km/h", True, (255, 255, 255))
            display.blit(speed_surf, (20, 50))

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