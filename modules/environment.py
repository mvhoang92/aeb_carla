import carla
import random

class CarlaEnv:
    def __init__(self, host, port):
        print(f"[*] Đang kết nối tới CARLA Server tại {host}:{port}...")
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle = None
        
        # Danh sách này sẽ chứa cả ID của xe giao thông và người đi bộ
        self.npc_list = [] 
        print("[*] Kết nối thành công!")

    def spawn_ego_vehicle(self):
        """Khởi tạo xe Tesla Model 3 cho hệ thống AEB"""
        print("[*] Đang thả xe Ego (Tesla Model 3)...")
        tesla_bp = self.blueprint_library.find('vehicle.tesla.model3')
        
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            print("[!] Lỗi: Map không có điểm thả xe hợp lệ.")
            return None
            
        spawn_point = random.choice(spawn_points)
        
        self.ego_vehicle = self.world.try_spawn_actor(tesla_bp, spawn_point)
        if self.ego_vehicle:
            # Ép Camera của Unreal Engine (Spectator) bay đến chỗ con Tesla 
            # để nhìn từ trên cao xuống cho dễ quan sát
            spectator = self.world.get_spectator()
            spectator_transform = carla.Transform(
                spawn_point.location + carla.Location(z=50), 
                carla.Rotation(pitch=-90) 
            )
            spectator.set_transform(spectator_transform)
            print(f"[+] Đã thả thành công xe Ego (ID: {self.ego_vehicle.id})")
        else:
            print("[!] Lỗi: Không thể thả xe Ego tại vị trí này.")
        
        return self.ego_vehicle

    def spawn_traffic(self, num_vehicles):
        """Thả dàn xe giao thông tự chạy (NPC) bằng Batch Sync an toàn"""
        print(f"[*] Đang thả {num_vehicles} xe giao thông...")
        traffic_manager = self.client.get_trafficmanager(8000)
        # Bắt xe NPC đi chậm hơn giới hạn 30% để bác dễ test phanh
        traffic_manager.global_percentage_speed_difference(30.0)

        # Lọc chỉ lấy các xe 4 bánh
        blueprints = self.blueprint_library.filter('vehicle.*')
        blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
        
        spawn_points = self.world.get_map().get_spawn_points()
        random.shuffle(spawn_points)

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        batch = []
        for n, transform in enumerate(spawn_points):
            if n >= num_vehicles: 
                break
            bp = random.choice(blueprints)
            if bp.has_attribute('color'):
                color = random.choice(bp.get_attribute('color').recommended_values)
                bp.set_attribute('color', color)
            
            # Gộp lệnh Spawn và bật Autopilot vào 1 batch
            batch.append(SpawnActor(bp, transform).then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        count = 0
        for response in self.client.apply_batch_sync(batch, False):
            if not response.error:
                self.npc_list.append(response.actor_id)
                count += 1
                
        print(f"[+] Đã thả thành công {count} xe NPC!")

    def spawn_pedestrians(self, num_walkers):
        """Thả người đi bộ đứng trên vỉa hè hoặc lòng đường"""
        print(f"[*] Đang thả {num_walkers} người đi bộ...")
        walker_bps = self.blueprint_library.filter('walker.pedestrian.*')
        spawn_points = []
        
        # Lấy random các điểm trên map để thả người
        for i in range(num_walkers):
            loc = self.world.get_random_location_from_navigation()
            if loc is not None:
                spawn_points.append(carla.Transform(loc))
                
        batch = []
        SpawnActor = carla.command.SpawnActor
        for spawn_point in spawn_points:
            bp = random.choice(walker_bps)
            # Cài cho người đi bộ "bất tử" để xe Tesla lỡ đâm vào không bị văng game
            if bp.has_attribute('is_invincible'):
                bp.set_attribute('is_invincible', 'true')
            batch.append(SpawnActor(bp, spawn_point))
            
        count = 0
        for response in self.client.apply_batch_sync(batch, False):
            if not response.error:
                self.npc_list.append(response.actor_id)
                count += 1
                
        print(f"[+] Đã thả thành công {count} người đi bộ!")

    def cleanup(self):
        """Hủy toàn bộ xe và người khi tắt chương trình để tránh rác map"""
        print("\n[*] Đang dọn dẹp hiện trường...")
        if self.ego_vehicle:
            self.ego_vehicle.destroy()
        
        # Dùng batch destroy để xóa hàng loạt NPC siêu nhanh
        if self.npc_list:
            self.client.apply_batch([carla.command.DestroyActor(x) for x in self.npc_list])
            
        print("[+] Dọn dẹp hoàn tất. Hẹn gặp lại!")