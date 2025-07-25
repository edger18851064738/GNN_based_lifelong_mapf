#!/usr/bin/env python3
"""
🛡️ 路口GNN预训练系统 - 支持intersection_edges格式
专门为lifelong_planning.py的路口场景设计

主要特性:
✅ 支持intersection_edges地图格式
✅ 生成路口冲突激烈场景
✅ 每条边一个任务的训练数据
✅ 安全感知的路口标签生成
✅ 兼容lifelong_planning.py的特征维度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data, DataLoader, Batch
from torch.utils.data import Dataset
import numpy as np
import math
import json
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time
from tqdm import tqdm
import random

# 导入基础组件
try:
    from trying import UnstructuredEnvironment, VehicleState, VehicleParameters
    HAS_TRYING = True
    print("✅ 成功导入trying.py环境组件")
except ImportError:
    HAS_TRYING = False
    print("⚠️ 无法导入trying.py，将使用简化实现")
    
    @dataclass
    class VehicleState:
        x: float
        y: float
        theta: float
        v: float
        t: float
    
    class VehicleParameters:
        def __init__(self):
            self.max_speed = 8.0
            self.max_accel = 2.0
            self.length = 4.0
            self.width = 2.0
    
    class UnstructuredEnvironment:
        def __init__(self, size=100):
            self.size = size
            self.obstacle_map = np.zeros((size, size), dtype=bool)
        
        def is_valid_position(self, x, y):
            ix, iy = int(round(x)), int(round(y))
            if 0 <= ix < self.size and 0 <= iy < self.size:
                return not self.obstacle_map[iy, ix]
            return False

# 导入lifelong组件
try:
    from lifelong_planning import IntersectionEdge, Task, Vehicle
    HAS_LIFELONG = True
    print("✅ 成功导入lifelong_planning.py组件")
except ImportError:
    HAS_LIFELONG = False
    print("⚠️ 将使用内置路口组件")
    
    @dataclass
    class IntersectionEdge:
        edge_id: str
        center_x: int
        center_y: int
        length: int = 5
        direction: str = ""
        
        def get_random_integer_position(self) -> Tuple[int, int]:
            return (self.center_x, self.center_y)
    
    @dataclass
    class Task:
        task_id: int
        start_edge: IntersectionEdge
        end_edge: IntersectionEdge
        start_pos: Tuple[int, int]
        end_pos: Tuple[int, int]
        priority: int = 1
    
    @dataclass
    class Vehicle:
        vehicle_id: int
        task: Task
        trajectory: List[VehicleState] = None
        color: str = "blue"

@dataclass
class IntersectionTrainingConfig:
    """🛡️ 路口训练配置"""
    batch_size: int = 4
    learning_rate: float = 0.0008
    num_epochs: int = 40
    hidden_dim: int = 64
    num_layers: int = 3
    dropout: float = 0.15
    weight_decay: float = 1e-4
    
    # 🆕 路口场景配置
    num_scenarios: int = 2000
    num_map_variants: int = 8
    max_vehicles_per_scenario: int = 8
    min_vehicles_per_scenario: int = 3
    
    # 🛡️ 安全相关配置
    min_safe_distance: float = 8.0
    safety_priority_weight: float = 1.8  # 路口安全权重更高
    high_conflict_ratio: float = 0.4     # 40%高冲突场景
    
    # 验证配置
    val_split: float = 0.2
    early_stopping_patience: int = 10

class IntersectionMapGenerator:
    """路口地图生成器"""
    
    def __init__(self, config: IntersectionTrainingConfig):
        self.config = config
        self.real_maps = []
        self.synthetic_maps = []
        
        print("🗺️ 初始化路口地图生成器...")
        self._scan_intersection_maps()
        self._generate_synthetic_intersection_maps()
    
    def _scan_intersection_maps(self):
        """扫描真实路口地图"""
        print("🔍 扫描路口地图文件...")
        
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        intersection_files = [f for f in json_files if any(keyword in f.lower() 
                            for keyword in ['lifelong', 'intersection', 'cross', 'junction'])]
        
        for json_file in intersection_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    map_data = json.load(f)
                
                # 验证是否为路口地图
                if 'intersection_edges' in map_data and map_data['intersection_edges']:
                    env = UnstructuredEnvironment(size=100)
                    if hasattr(env, 'load_from_json'):
                        success = env.load_from_json(json_file)
                        if success:
                            self.real_maps.append({
                                'name': json_file,
                                'environment': env,
                                'data': map_data
                            })
                            print(f"  ✅ 加载路口地图: {json_file}")
                    
            except Exception as e:
                continue
        
        print(f"📊 发现 {len(self.real_maps)} 个路口地图文件")
    
    def _generate_synthetic_intersection_maps(self):
        """生成合成路口地图"""
        print(f"🏗️ 生成 {self.config.num_map_variants} 种合成路口地图...")
        
        for i in range(self.config.num_map_variants):
            map_data = self._create_synthetic_intersection(i)
            env = UnstructuredEnvironment(size=100)
            
            # 设置障碍物
            if 'grid' in map_data:
                grid = np.array(map_data['grid'])
                for row in range(min(grid.shape[0], env.size)):
                    for col in range(min(grid.shape[1], env.size)):
                        if grid[row, col] == 1:
                            env.obstacle_map[row, col] = True
            
            self.synthetic_maps.append({
                'name': f"synthetic_intersection_{i}",
                'environment': env,
                'data': map_data
            })
        
        print(f"✅ 生成了 {len(self.synthetic_maps)} 个合成路口地图")
    
    def _create_synthetic_intersection(self, variant_id: int) -> Dict:
        """创建合成路口地图"""
        # 基于variant_id创建不同类型的路口
        intersection_types = ['four_way', 'three_way', 'complex', 'roundabout']
        intersection_type = intersection_types[variant_id % len(intersection_types)]
        
        # 基础地图信息
        map_data = {
            "map_info": {
                "name": f"synthetic_intersection_{variant_id}",
                "width": 100,
                "height": 100,
                "type": intersection_type
            },
            "intersection_edges": [],
            "grid": np.zeros((100, 100), dtype=int).tolist()
        }
        
        if intersection_type == 'four_way':
            # 四向路口
            edges = [
                {"edge_id": "N", "center_x": 50, "center_y": 10, "direction": "north", "length": 8},
                {"edge_id": "S", "center_x": 50, "center_y": 90, "direction": "south", "length": 8},
                {"edge_id": "E", "center_x": 90, "center_y": 50, "direction": "east", "length": 8},
                {"edge_id": "W", "center_x": 10, "center_y": 50, "direction": "west", "length": 8}
            ]
            # 添加中央障碍物
            for x in range(45, 56):
                for y in range(45, 56):
                    map_data["grid"][y][x] = 1
                    
        elif intersection_type == 'three_way':
            # T型路口
            edges = [
                {"edge_id": "N", "center_x": 50, "center_y": 15, "direction": "north", "length": 10},
                {"edge_id": "E", "center_x": 85, "center_y": 50, "direction": "east", "length": 10},
                {"edge_id": "W", "center_x": 15, "center_y": 50, "direction": "west", "length": 10}
            ]
            # 添加障碍物
            for x in range(40, 61):
                for y in range(60, 80):
                    map_data["grid"][y][x] = 1
                    
        elif intersection_type == 'complex':
            # 复杂路口
            edges = [
                {"edge_id": "N1", "center_x": 40, "center_y": 10, "direction": "north", "length": 6},
                {"edge_id": "N2", "center_x": 60, "center_y": 10, "direction": "north", "length": 6},
                {"edge_id": "S", "center_x": 50, "center_y": 90, "direction": "south", "length": 8},
                {"edge_id": "E", "center_x": 90, "center_y": 50, "direction": "east", "length": 8},
                {"edge_id": "W", "center_x": 10, "center_y": 50, "direction": "west", "length": 8}
            ]
            # 复杂障碍物布局
            for x in range(30, 35):
                for y in range(30, 70):
                    map_data["grid"][y][x] = 1
            for x in range(65, 70):
                for y in range(30, 70):
                    map_data["grid"][y][x] = 1
                    
        else:  # roundabout
            # 环岛
            edges = [
                {"edge_id": "N", "center_x": 50, "center_y": 20, "direction": "north", "length": 6},
                {"edge_id": "S", "center_x": 50, "center_y": 80, "direction": "south", "length": 6},
                {"edge_id": "E", "center_x": 80, "center_y": 50, "direction": "east", "length": 6},
                {"edge_id": "W", "center_x": 20, "center_y": 50, "direction": "west", "length": 6}
            ]
            # 中央圆形障碍物
            center_x, center_y = 50, 50
            radius = 12
            for x in range(100):
                for y in range(100):
                    if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                        map_data["grid"][y][x] = 1
        
        map_data["intersection_edges"] = edges
        return map_data
    
    def get_random_intersection_map(self) -> Tuple[UnstructuredEnvironment, Dict, str]:
        """获取随机路口地图"""
        all_maps = self.real_maps + self.synthetic_maps
        
        if not all_maps:
            # 创建最简单的默认路口
            default_map = self._create_synthetic_intersection(0)
            env = UnstructuredEnvironment(size=100)
            return env, default_map, "default_intersection"
        
        selected = random.choice(all_maps)
        return selected['environment'], selected['data'], selected['name']

class IntersectionVehicleScenarioGenerator:
    """路口车辆场景生成器"""
    
    def __init__(self, config: IntersectionTrainingConfig):
        self.config = config
        self.map_generator = IntersectionMapGenerator(config)
        
    def generate_training_data(self) -> List[Tuple]:
        """生成路口训练数据"""
        print(f"🛡️ 生成 {self.config.num_scenarios} 个路口训练场景...")
        
        data_list = []
        failed_scenarios = 0
        
        # 分配高冲突和普通场景
        num_high_conflict = int(self.config.num_scenarios * self.config.high_conflict_ratio)
        num_normal = self.config.num_scenarios - num_high_conflict
        
        print(f"📊 场景分配: {num_high_conflict} 高冲突 + {num_normal} 普通场景")
        
        for i in tqdm(range(self.config.num_scenarios)):
            try:
                # 获取随机路口地图
                environment, map_data, map_name = self.map_generator.get_random_intersection_map()
                
                # 生成车辆场景
                is_high_conflict = i < num_high_conflict
                vehicles = self._generate_intersection_vehicles(map_data, is_high_conflict)
                
                if not vehicles:
                    failed_scenarios += 1
                    continue
                
                # 构建图数据
                graph_data = self._build_intersection_graph(vehicles, environment)
                
                if graph_data.x.size(0) == 0:
                    failed_scenarios += 1
                    continue
                
                # 生成路口安全标签
                labels = self._generate_intersection_safety_labels(vehicles, map_data)
                
                # 验证数据一致性
                if self._validate_data_consistency(graph_data, labels, len(vehicles)):
                    data_list.append((graph_data, labels))
                else:
                    failed_scenarios += 1
                
            except Exception as e:
                failed_scenarios += 1
                if i < 10:
                    print(f"⚠️ 生成场景 {i} 时出错: {str(e)}")
                continue
        
        print(f"✅ 成功生成 {len(data_list)} 个路口训练场景")
        print(f"📊 统计: 成功 {len(data_list)}, 失败 {failed_scenarios}")
        
        return data_list
    
    def _generate_intersection_vehicles(self, map_data: Dict, is_high_conflict: bool) -> List[Vehicle]:
        """生成路口车辆场景"""
        edges_data = map_data.get("intersection_edges", [])
        if not edges_data:
            return []
        
        # 创建路口边对象
        edges = []
        for edge_data in edges_data:
            edge = IntersectionEdge(
                edge_id=edge_data["edge_id"],
                center_x=edge_data["center_x"],
                center_y=edge_data["center_y"],
                length=edge_data.get("length", 5),
                direction=edge_data.get("direction", "")
            )
            edges.append(edge)
        
        if len(edges) < 2:
            return []
        
        vehicles = []
        
        if is_high_conflict:
            # 高冲突场景：每条边都有车辆，增加对角线冲突
            for i, start_edge in enumerate(edges):
                # 选择冲突目标边
                if len(edges) >= 4:
                    # 优先选择对角线边
                    target_edges = [e for e in edges if e.edge_id != start_edge.edge_id]
                    if len(target_edges) >= 2:
                        end_edge = target_edges[i % len(target_edges)]
                    else:
                        end_edge = random.choice(target_edges)
                else:
                    # 边数较少时随机选择
                    others = [e for e in edges if e.edge_id != start_edge.edge_id]
                    end_edge = random.choice(others)
                
                task = Task(
                    task_id=i + 1,
                    start_edge=start_edge,
                    end_edge=end_edge,
                    start_pos=start_edge.get_random_integer_position(),
                    end_pos=end_edge.get_random_integer_position(),
                    priority=random.randint(2, 5)  # 高优先级范围
                )
                
                vehicle = Vehicle(
                    vehicle_id=i + 1,
                    task=task,
                    color='red'
                )
                vehicles.append(vehicle)
                
            # 额外添加一些汇聚车辆
            if len(edges) >= 3:
                target_edge = random.choice(edges)
                source_edges = [e for e in edges if e.edge_id != target_edge.edge_id][:2]
                
                for j, source_edge in enumerate(source_edges):
                    task = Task(
                        task_id=len(vehicles) + j + 1,
                        start_edge=source_edge,
                        end_edge=target_edge,
                        start_pos=source_edge.get_random_integer_position(),
                        end_pos=target_edge.get_random_integer_position(),
                        priority=random.randint(1, 4)
                    )
                    
                    vehicle = Vehicle(
                        vehicle_id=len(vehicles) + j + 1,
                        task=task,
                        color='orange'
                    )
                    vehicles.append(vehicle)
        
        else:
            # 普通场景：适度数量的车辆，避免过度冲突
            num_vehicles = random.randint(self.config.min_vehicles_per_scenario, 
                                        min(len(edges), self.config.max_vehicles_per_scenario))
            
            selected_edges = random.sample(edges, min(num_vehicles, len(edges)))
            
            for i, start_edge in enumerate(selected_edges):
                # 选择非相邻边
                others = [e for e in edges if e.edge_id != start_edge.edge_id]
                if len(others) >= 3:
                    # 排除最近的边
                    others.sort(key=lambda e: 
                        math.sqrt((e.center_x - start_edge.center_x)**2 + 
                                 (e.center_y - start_edge.center_y)**2))
                    end_edge = random.choice(others[1:])  # 排除最近的
                else:
                    end_edge = random.choice(others)
                
                task = Task(
                    task_id=i + 1,
                    start_edge=start_edge,
                    end_edge=end_edge,
                    start_pos=start_edge.get_random_integer_position(),
                    end_pos=end_edge.get_random_integer_position(),
                    priority=random.randint(1, 4)
                )
                
                vehicle = Vehicle(
                    vehicle_id=i + 1,
                    task=task,
                    color='blue'
                )
                vehicles.append(vehicle)
        
        return vehicles
    
    def _build_intersection_graph(self, vehicles: List[Vehicle], environment) -> Data:
        """构建路口交互图 - 兼容lifelong_planning.py的特征"""
        
        # 🎯 生成8维节点特征 (与lifelong_planning.py兼容)
        node_features = self._extract_8d_node_features(vehicles)
        
        # 构建边特征
        edge_indices, edge_features = self._build_intersection_edges(vehicles)
        
        # 全局特征
        global_features = self._extract_global_features(vehicles)
        
        return Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=torch.tensor(edge_indices, dtype=torch.long).T if edge_indices else torch.zeros((2, 0), dtype=torch.long),
            edge_attr=torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros((0, 6), dtype=torch.float32),
            global_features=torch.tensor(global_features, dtype=torch.float32)
        )
    
    def _extract_8d_node_features(self, vehicles: List[Vehicle]) -> List[List[float]]:
        """提取8维节点特征 - 与lifelong_planning.py完全兼容"""
        features = []
        
        for vehicle in vehicles:
            task = vehicle.task
            start_x, start_y = task.start_pos
            end_x, end_y = task.end_pos
            
            # 计算基础特征
            dx = end_x - start_x
            dy = end_y - start_y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)
            goal_bearing = math.atan2(dy, dx)
            
            # 🎯 8维特征向量 (与lifelong_planning.py完全一致)
            node_feature = [
                (start_x - 50.0) / 50.0,          # [0] 相对环境中心x
                math.cos(goal_bearing),           # [1] 航向余弦
                math.sin(goal_bearing),           # [2] 航向正弦
                3.0 / 8.0,                        # [3] 归一化速度 (固定3.0)
                0.0,                              # [4] 归一化加速度
                distance_to_goal / 100.0,         # [5] 归一化目标距离
                math.cos(goal_bearing),           # [6] 目标方向余弦
                task.priority / 10.0              # [7] 归一化优先级
            ]
            
            features.append(node_feature)
        
        return features
    
    def _build_intersection_edges(self, vehicles: List[Vehicle]) -> Tuple[List, List]:
        """构建路口边特征"""
        edge_indices = []
        edge_features = []
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                
                # 计算交互特征
                dist = math.sqrt((v1.task.start_pos[0] - v2.task.start_pos[0])**2 + 
                               (v1.task.start_pos[1] - v2.task.start_pos[1])**2)
                
                if dist < 50.0:  # 交互范围
                    # 路径交叉检测
                    crossing = self._check_path_crossing(v1.task, v2.task)
                    
                    # 6维边特征
                    edge_feat = [
                        dist / 50.0,                    # 归一化距离
                        6.0 / 16.0,                     # 平均速度
                        0.0,                            # 角度差
                        1.0 if crossing else 0.0,      # 路径交叉
                        (v1.task.priority + v2.task.priority) / 10.0,  # 优先级
                        0.5                             # 冲突风险
                    ]
                    
                    edge_indices.extend([[i, j], [j, i]])
                    edge_features.extend([edge_feat, edge_feat])
        
        return edge_indices, edge_features
    
    def _check_path_crossing(self, task1: Task, task2: Task) -> bool:
        """检查两个任务的路径是否交叉"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
        return intersect(task1.start_pos, task1.end_pos, task2.start_pos, task2.end_pos)
    
    def _extract_global_features(self, vehicles: List[Vehicle]) -> List[float]:
        """提取全局特征"""
        if not vehicles:
            return [0.0] * 8
        
        n_vehicles = len(vehicles)
        priorities = [v.task.priority for v in vehicles]
        
        # 计算冲突对数
        conflicts = 0
        total_pairs = 0
        for i in range(n_vehicles):
            for j in range(i + 1, n_vehicles):
                total_pairs += 1
                if self._check_path_crossing(vehicles[i].task, vehicles[j].task):
                    conflicts += 1
        
        conflict_ratio = conflicts / max(total_pairs, 1)
        
        return [
            n_vehicles / 10.0,           # 车辆数
            3.0 / 8.0,                   # 平均速度
            0.1,                         # 速度方差
            50.0 / 100.0,                # 平均距离
            10.0 / 100.0,                # 距离方差
            sum(priorities) / (n_vehicles * 10),  # 平均优先级
            conflict_ratio,              # 冲突比例
            0.5                          # 预留特征
        ]
    
    def _generate_intersection_safety_labels(self, vehicles: List[Vehicle], map_data: Dict) -> Dict:
        """生成路口安全标签"""
        labels = {
            'priority': [],
            'cooperation': [],
            'urgency': [],
            'safety': [],
            'speed_adjustment': [],
            'route_preference': []
        }
        
        # 分析全局冲突情况
        n_vehicles = len(vehicles)
        total_conflicts = 0
        for i in range(n_vehicles):
            for j in range(i + 1, n_vehicles):
                if self._check_path_crossing(vehicles[i].task, vehicles[j].task):
                    total_conflicts += 1
        
        global_conflict_level = total_conflicts / max(n_vehicles * (n_vehicles - 1) / 2, 1)
        
        for vehicle in vehicles:
            # 计算该车辆的冲突数
            vehicle_conflicts = 0
            for other in vehicles:
                if other.vehicle_id != vehicle.vehicle_id:
                    if self._check_path_crossing(vehicle.task, other.task):
                        vehicle_conflicts += 1
            
            conflict_factor = vehicle_conflicts / max(n_vehicles - 1, 1)
            
            # 🛡️ 路口安全标签生成
            
            # 优先级调整
            base_priority = (vehicle.task.priority - 3) / 3.0
            if conflict_factor > 0.5:
                priority_adj = base_priority * 0.7  # 高冲突时降低优先级
            else:
                priority_adj = base_priority
            labels['priority'].append([np.tanh(priority_adj)])
            
            # 合作倾向
            if global_conflict_level > 0.3:
                cooperation = 0.8  # 高冲突环境下提高合作
            else:
                cooperation = 0.5 + conflict_factor * 0.3
            labels['cooperation'].append([cooperation])
            
            # 紧急程度
            if conflict_factor > 0.6:
                urgency = 0.3  # 高冲突时降低紧急程度，优先安全
            else:
                urgency = 0.4 + conflict_factor * 0.2
            labels['urgency'].append([urgency])
            
            # 🛡️ 安全系数 (路口最重要)
            if conflict_factor > 0.5:
                safety = 0.9  # 高冲突时最高安全要求
            elif conflict_factor > 0.3:
                safety = 0.8
            else:
                safety = 0.6 + conflict_factor * 0.2
            labels['safety'].append([safety])
            
            # 速度调整
            if conflict_factor > 0.4:
                speed_adj = -0.3  # 高冲突时减速
            elif global_conflict_level > 0.4:
                speed_adj = -0.2  # 全局冲突时适度减速
            else:
                speed_adj = 0.0
            labels['speed_adjustment'].append([speed_adj])
            
            # 路径偏好
            if conflict_factor > 0.3:
                # 高冲突时偏向避让
                route_pref = [0.4, 0.2, 0.4]  # 左/直/右
            else:
                route_pref = [0.3, 0.4, 0.3]  # 均衡偏好
            labels['route_preference'].append(route_pref)
        
        # 转换为张量
        for key in labels:
            labels[key] = torch.tensor(labels[key], dtype=torch.float32)
        
        return labels
    
    def _validate_data_consistency(self, graph_data: Data, labels: Dict, expected_nodes: int) -> bool:
        """验证数据一致性"""
        try:
            actual_nodes = graph_data.x.size(0)
            if actual_nodes != expected_nodes:
                return False
            
            for key, label_tensor in labels.items():
                if label_tensor.size(0) != expected_nodes:
                    return False
            
            return True
            
        except Exception:
            return False

# 复用Pretraining_gnn.py的模型和训练器类
class IntersectionVehicleCoordinationGNN(nn.Module):
    """路口车辆协调GNN - 兼容8维输入"""
    
    def __init__(self, config: IntersectionTrainingConfig):
        super().__init__()
        
        self.config = config
        self.node_dim = 8      # 🎯 兼容lifelong_planning.py的8维特征
        self.edge_dim = 6
        self.global_dim = 8
        self.hidden_dim = config.hidden_dim
        
        # 编码器
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # GNN层
        self.gnn_layers = nn.ModuleList([
            pyg_nn.GCNConv(self.hidden_dim, self.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # 决策输出头
        self.decision_heads = nn.ModuleDict({
            'priority': nn.Sequential(
                nn.Linear(self.hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Tanh()
            ),
            'cooperation': nn.Sequential(
                nn.Linear(self.hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'urgency': nn.Sequential(
                nn.Linear(self.hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'safety': nn.Sequential(
                nn.Linear(self.hidden_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.05),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()
            ),
            'speed_adjustment': nn.Sequential(
                nn.Linear(self.hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 1),
                nn.Tanh()
            ),
            'route_preference': nn.Sequential(
                nn.Linear(self.hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(32, 3),
                nn.Softmax(dim=-1)
            )
        })
    
    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """前向传播"""
        x, edge_index = batch.x, batch.edge_index
        
        if x.size(0) == 0:
            return self._empty_output()
        
        # 节点编码
        x = self.node_encoder(x)
        
        # GNN层
        for gnn_layer in self.gnn_layers:
            x = F.relu(gnn_layer(x, edge_index))
            x = F.dropout(x, p=self.config.dropout, training=self.training)
        
        # 生成决策
        decisions = {}
        for decision_type, head in self.decision_heads.items():
            decisions[decision_type] = head(x)
        
        return decisions
    
    def _empty_output(self) -> Dict[str, torch.Tensor]:
        """空输出"""
        return {
            'priority': torch.zeros((0, 1)),
            'cooperation': torch.zeros((0, 1)),
            'urgency': torch.zeros((0, 1)),
            'safety': torch.zeros((0, 1)),
            'speed_adjustment': torch.zeros((0, 1)),
            'route_preference': torch.zeros((0, 3))
        }

class IntersectionGraphDataset(Dataset):
    """路口图数据集"""
    
    def __init__(self, scenarios_data: List[Tuple]):
        self.data = []
        
        print(f"🔄 处理 {len(scenarios_data)} 个路口场景数据...")
        
        for i, (graph_data, labels) in enumerate(scenarios_data):
            try:
                data_obj = Data(
                    x=graph_data.x,
                    edge_index=graph_data.edge_index,
                    edge_attr=graph_data.edge_attr,
                    global_features=graph_data.global_features,
                    y_priority=labels['priority'],
                    y_cooperation=labels['cooperation'],
                    y_urgency=labels['urgency'],
                    y_safety=labels['safety'],
                    y_speed_adjustment=labels['speed_adjustment'],
                    y_route_preference=labels['route_preference']
                )
                
                num_nodes = data_obj.x.size(0)
                if (data_obj.y_priority.size(0) == num_nodes and
                    data_obj.y_cooperation.size(0) == num_nodes):
                    self.data.append(data_obj)
                else:
                    if i < 5:
                        print(f"⚠️ 跳过场景 {i}: 标签与节点数不匹配")
                    
            except Exception as e:
                if i < 5:
                    print(f"⚠️ 处理场景 {i} 时出错: {str(e)}")
                continue
        
        print(f"✅ 成功处理 {len(self.data)} 个有效路口场景")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

class IntersectionGNNTrainer:
    """路口GNN训练器"""
    
    def __init__(self, config: IntersectionTrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🔧 使用设备: {self.device}")
        
        self.model = IntersectionVehicleCoordinationGNN(config).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=12, gamma=0.7
        )
        
        # 损失函数
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def compute_batch_loss(self, predictions: Dict, batch: Batch) -> torch.Tensor:
        """计算批次损失"""
        total_loss = 0.0
        
        # 🛡️ 路口安全优先的损失权重
        loss_weights = {
            'priority': 1.0,
            'cooperation': 1.2,
            'urgency': 1.0,
            'safety': self.config.safety_priority_weight,  # 路口安全权重最高
            'speed_adjustment': 0.8,
            'route_preference': 1.0
        }
        
        for task in ['priority', 'cooperation', 'urgency', 'safety', 'speed_adjustment']:
            if task in predictions and hasattr(batch, f'y_{task}'):
                y_true = getattr(batch, f'y_{task}')
                if y_true.size(0) > 0:
                    if task in ['cooperation', 'urgency', 'safety']:
                        loss = self.bce_loss(predictions[task], y_true)
                        # 🛡️ 安全额外惩罚
                        if task == 'safety':
                            safety_penalty = torch.mean(torch.relu(y_true - predictions[task])) * 0.3
                            loss += safety_penalty
                    else:
                        loss = self.mse_loss(predictions[task], y_true)
                    total_loss += loss_weights[task] * loss
        
        if 'route_preference' in predictions and hasattr(batch, 'y_route_preference'):
            y_route = batch.y_route_preference
            if y_route.size(0) > 0:
                loss = self.ce_loss(predictions['route_preference'], y_route)
                total_loss += loss_weights['route_preference'] * loss
        
        return total_loss
    
    def train_epoch(self, dataloader) -> float:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="🛡️ 路口训练"):
            try:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)
                predictions = self.model(batch)
                loss = self.compute_batch_loss(predictions, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            except Exception as e:
                print(f"⚠️ 训练批次出错: {str(e)}")
                continue
        
        return total_loss / max(num_batches, 1)
    
    def validate(self, dataloader) -> float:
        """验证"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="🛡️ 路口验证"):
                try:
                    batch = batch.to(self.device)
                    predictions = self.model(batch)
                    loss = self.compute_batch_loss(predictions, batch)
                    total_loss += loss.item()
                    num_batches += 1
                except Exception as e:
                    continue
        
        return total_loss / max(num_batches, 1)
    
    def train(self, train_dataset: IntersectionGraphDataset, val_dataset: IntersectionGraphDataset):
        """完整训练流程"""
        print(f"🛡️ 开始训练路口GNN模型...")
        print(f"  训练数据: {len(train_dataset)} 个路口场景")
        print(f"  验证数据: {len(val_dataset)} 个路口场景")
        print(f"  🛡️ 安全损失权重: {self.config.safety_priority_weight}")
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            follow_batch=['x']
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            follow_batch=['x']
        )
        
        for epoch in range(self.config.num_epochs):
            print(f"\n📊 Epoch {epoch+1}/{self.config.num_epochs} (🛡️ 路口安全训练)")
            
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            self.scheduler.step()
            
            print(f"  训练损失: {train_loss:.4f}")
            print(f"  验证损失: {val_loss:.4f}")
            print(f"  学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_model('best_intersection_gnn_model.pth')
                print(f"  ✅ 验证损失改善，保存最佳路口模型")
            else:
                self.patience_counter += 1
                print(f"  ⏳ 验证损失未改善 ({self.patience_counter}/{self.config.early_stopping_patience})")
            
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"  🛑 早停")
                break
        
        print(f"\n🎉 路口GNN训练完成！最佳验证损失: {self.best_val_loss:.4f}")
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }, filepath)

def main():
    """🛡️ 路口GNN预训练主流程"""
    print("🛡️ 路口GNN预训练系统")
    print("=" * 80)
    print("🎯 专为lifelong_planning.py路口场景设计:")
    print("   ✅ 🛡️ 支持intersection_edges地图格式")
    print("   ✅ 🛡️ 每条边一个任务的训练数据生成")
    print("   ✅ 🛡️ 8维特征兼容性")
    print("   ✅ 🛡️ 路口冲突安全标签生成")
    print("   ✅ 🛡️ 高冲突场景专门训练")
    print("=" * 80)
    
    # 路口训练配置
    config = IntersectionTrainingConfig(
        batch_size=4,
        learning_rate=0.0008,
        num_epochs=40,
        hidden_dim=64,
        num_layers=3,
        dropout=0.15,
        
        # 路口特化配置
        num_scenarios=2000,
        num_map_variants=8,
        max_vehicles_per_scenario=8,
        min_vehicles_per_scenario=3,
        
        # 安全配置
        min_safe_distance=8.0,
        safety_priority_weight=1.8,
        high_conflict_ratio=0.4
    )
    
    print(f"\n📋 🛡️ 路口训练配置:")
    print(f"  数据集大小: {config.num_scenarios}")
    print(f"  高冲突场景比例: {config.high_conflict_ratio*100:.0f}%")
    print(f"  安全损失权重: {config.safety_priority_weight}x")
    print(f"  地图变体数: {config.num_map_variants}")
    print(f"  特征维度: 8维节点 + 6维边 + 8维全局")
    
    try:
        # 1. 生成路口训练数据
        print(f"\n📊 步骤1: 生成路口训练数据")
        generator = IntersectionVehicleScenarioGenerator(config)
        all_data = generator.generate_training_data()
        
        if len(all_data) < 20:
            print("❌ 生成的有效路口数据太少，无法训练")
            return
        
        # 2. 划分数据集
        val_size = max(5, int(len(all_data) * config.val_split))
        train_size = len(all_data) - val_size
        
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]
        
        print(f"  训练集: {len(train_data)} 个路口场景")
        print(f"  验证集: {len(val_data)} 个路口场景")
        
        # 3. 创建数据集
        print(f"\n🔄 步骤2: 创建路口数据集")
        train_dataset = IntersectionGraphDataset(train_data)
        val_dataset = IntersectionGraphDataset(val_data)
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            print("❌ 路口数据集创建失败")
            return
        
        # 4. 训练路口GNN模型
        print(f"\n🛡️ 步骤3: 训练路口GNN模型")
        trainer = IntersectionGNNTrainer(config)
        trainer.train(train_dataset, val_dataset)
        
        # 5. 保存模型
        trainer.save_model('final_intersection_gnn_model.pth')
        
        print(f"\n✅ 🛡️ 路口GNN预训练完成！")
        print(f"  最佳模型: best_intersection_gnn_model.pth")
        print(f"  最终模型: final_intersection_gnn_model.pth")
        print(f"\n🎯 🛡️ 路口模型特性:")
        print(f"  ✅ 兼容lifelong_planning.py的8维特征")
        print(f"  ✅ 支持intersection_edges地图格式")
        print(f"  ✅ 40%高冲突场景训练安全避让")
        print(f"  ✅ 路口安全优先决策 (权重1.8x)")
        print(f"  ✅ 每条边一个任务的专门训练")
        
        print(f"\n🛡️ 使用方法:")
        print(f"  现在可以运行 lifelong_planning.py")
        print(f"  系统将自动加载预训练的路口GNN模型")
        print(f"  享受路口安全智能规划！")
        
    except Exception as e:
        print(f"❌ 路口预训练过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()