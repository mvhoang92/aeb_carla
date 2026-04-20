import carla

class VehicleController:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.current_control = carla.VehicleControl()
        self.aeb_active = False

    def apply_aeb_brake(self, brake_value=1.0):
        """Áp dụng phanh khẩn cấp, ghi đè ga"""
        self.current_control.brake = brake_value
        self.current_control.throttle = 0.0
        self.aeb_active = True
        self.vehicle.apply_control(self.current_control)

    def release_aeb(self):
        """Nhả phanh khẩn cấp để xe tiếp tục chạy"""
        if self.aeb_active:
            self.current_control.brake = 0.0
            self.aeb_active = False
            self.vehicle.apply_control(self.current_control)

    def set_autopilot(self, enabled):
        """Bật/tắt autopilot của CARLA"""
        self.vehicle.set_autopilot(enabled)

    def get_state(self):
        """Trả về trạng thái hiện tại của xe"""
        v = self.vehicle.get_velocity()
        speed = 3.6 * (v.x**2 + v.y**2 + v.z**2)**0.5 # km/h
        return {
            'speed': speed,
            'aeb_active': self.aeb_active,
            'throttle': self.current_control.throttle,
            'brake': self.current_control.brake
        }
