import carla
import numpy as np
import pygame
import weakref

class CameraSensor:
    def __init__(self, parent_actor, width, height, transform):
        self.surface = None
        self.image_array = None  # <-- MỚI: Biến chứa ảnh thô cho AI
        self.sensor = None
        self.width = width
        self.height = height
        
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find('sensor.camera.rgb')
        bp.set_attribute('image_size_x', str(width))
        bp.set_attribute('image_size_y', str(height))
        
        self.sensor = world.spawn_actor(bp, transform, attach_to=parent_actor)
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda image: CameraSensor._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self: return
        
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        bgr_array = array[:, :, :3].copy() # Copy BGR cho YOLO
        
        self.image_array = bgr_array # YOLO mặc định lấy đầu vào BGR giống OpenCV
        
        # BGR sang RGB cho Pygame
        rgb_array = bgr_array[:, :, ::-1] 
        self.surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))

    def render(self, display, x_offset=0, y_offset=0):
        if self.surface is not None:
            display.blit(self.surface, (x_offset, y_offset))

    def destroy(self):
        if self.sensor is not None:
            if self.sensor.is_alive:
                self.sensor.stop()
                self.sensor.destroy()
            self.sensor = None