#!/usr/bin/env python3.7
"""Đổi map CARLA. Dùng: python3.7 change_map.py Town01"""
import sys, glob

egg = glob.glob('../PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg')
if egg:
    sys.path.insert(0, egg[0])

import carla

map_name = sys.argv[1] if len(sys.argv) > 1 else 'Town03'
print(f'[*] Đang load {map_name}...')
c = carla.Client('localhost', 2000)
c.set_timeout(60.0)
c.load_world(map_name)
print(f'[*] Xong! Map hiện tại: {map_name}')
