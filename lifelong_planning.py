#!/usr/bin/env python3
"""
终身GNN路口规划系统 - 完整版
基于原有 lifelong_planning.py，集成 GNN 增强功能

核心特性:
✅ 每条出入口边一个任务，边上随机整数起点 → 非相邻边随机整数终点
✅ 可选的GNN增强规划 (继承 GNN_try.py)
✅ 智能优先级分配 (继承 priority.py)
✅ 高级可视化 (继承 trying.py)
✅ 冲突强度分析和控制

流程:
1. 载入路口地图
2. 每条出入口边生成一个任务：边上随机整数起点 → 排除最近两条边后随机选择终点边上随机整数终点
3. 可选使用GNN增强规划或基础规划
4. 智能优先级分配
5. 批次规划所有车辆
6. 高级可视化展示
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import json
import time
import math
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import deque
from Pretraining_gnn import SafetyEnhancedTrainingConfig as TrainingConfig
# 导入核心规划模块
from trying import (
    VehicleState, VehicleParameters, OptimizationLevel,
    UnstructuredEnvironment, VHybridAStarPlanner
)

# 导入GNN增强组件 (可选)
try:
    from GNN_try import (
        PretrainedGNNIntegratedCoordinator, 
        GNNEnhancementLevel
    )
    HAS_GNN_INTEGRATION = True
    print("✅ GNN增强模块可用")
except ImportError:
    HAS_GNN_INTEGRATION = False
    print("⚠️ GNN增强模块不可用，将使用基础规划")

# 导入高级可视化组件 (可选)
try:
    from trying import MultiVehicleCoordinator
    HAS_ADVANCED_VISUALIZATION = True
    print("✅ 高级可视化模块可用")
except ImportError:
    HAS_ADVANCED_VISUALIZATION = False
    print("⚠️ 将使用简单可视化")

# 导入智能优先级模块 (可选)
try:
    from priority import IntelligentPriorityAssigner
    HAS_INTELLIGENT_PRIORITY = True
    print("✅ 智能优先级模块可用")
except ImportError:
    HAS_INTELLIGENT_PRIORITY = False
    print("⚠️ 将使用默认优先级")

@dataclass
class IntersectionEdge:
    """进出口边"""
    edge_id: str
    center_x: int
    center_y: int  
    length: int = 5
    direction: str = ""  # 仅用于可视化
    
    def get_points(self) -> List[Tuple[int, int]]:
        """获取边界覆盖的所有整数点位"""
        points = []
        half_length = self.length // 2
        
        if self.direction in ["north", "south"]:
            # 水平边界
            for x in range(self.center_x - half_length, self.center_x + half_length + 1):
                points.append((x, self.center_y))
        elif self.direction in ["east", "west"]:
            # 垂直边界  
            for y in range(self.center_y - half_length, self.center_y + half_length + 1):
                points.append((self.center_x, y))
        else:
            # 默认水平边界
            for x in range(self.center_x - half_length, self.center_x + half_length + 1):
                points.append((x, self.center_y))
        
        return points
    
    def get_random_integer_position(self) -> Tuple[int, int]:
        """在边界上获取随机整数位置"""
        points = self.get_points()
        if points:
            return random.choice(points)
        return (self.center_x, self.center_y)

@dataclass  
class Task:
    """简单任务"""
    task_id: int
    start_edge: IntersectionEdge
    end_edge: IntersectionEdge
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    priority: int = 1

@dataclass
class Vehicle:
    """简单车辆"""
    vehicle_id: int
    task: Task
    trajectory: List[VehicleState] = None
    color: str = "blue"
    planning_time: float = 0.0

class ConflictIntensityAnalyzer:
    """冲突强度分析器"""
    
    @staticmethod
    def analyze_scenario_conflicts(vehicles: List[Vehicle]) -> Dict:
        """分析场景冲突强度"""
        if len(vehicles) < 2:
            return {'intensity': 0.0, 'conflicts': [], 'total_pairs': 0, 'conflict_count': 0}
        
        conflicts = []
        total_pairs = 0
        
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                v1, v2 = vehicles[i], vehicles[j]
                total_pairs += 1
                
                # 检查路径是否交叉
                if ConflictIntensityAnalyzer._paths_intersect(
                    v1.task.start_pos, v1.task.end_pos,
                    v2.task.start_pos, v2.task.end_pos
                ):
                    conflicts.append((v1.vehicle_id, v2.vehicle_id))
        
        intensity = len(conflicts) / max(total_pairs, 1)
        
        return {
            'intensity': intensity,
            'conflicts': conflicts,
            'total_pairs': total_pairs,
            'conflict_count': len(conflicts)
        }
    
    @staticmethod
    def _paths_intersect(start1: tuple, end1: tuple, start2: tuple, end2: tuple) -> bool:
        """检查两条路径是否交叉"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        
        def intersect(A, B, C, D):
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
        
        return intersect(start1, end1, start2, end2)

class FirstRoundPlanner:
    """第一轮多车规划器 - 基础版"""
    
    def __init__(self, map_file: str, optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED):
        # 加载地图
        self.environment = UnstructuredEnvironment()
        self.map_data = self.environment.load_from_json(map_file)
        
        if not self.map_data:
            raise ValueError(f"无法加载地图文件: {map_file}")
        
        self.params = VehicleParameters()
        self.optimization_level = optimization_level
        
        # 加载出入口边
        self.edges = self._load_edges()
        
        # 生成任务
        self.tasks = self._generate_tasks()
        
        # 创建车辆
        self.vehicles = self._create_vehicles()
        
        # 统计
        self.total_vehicles = len(self.vehicles)
        self.successful_plannings = 0
        self.planning_start_time = time.time()
        
        print(f"🚀 第一轮规划器初始化")
        print(f"   地图: {self.map_data.get('map_info', {}).get('name', 'Unknown')}")
        print(f"   出入口边: {len(self.edges)} 个")
        print(f"   生成任务: {len(self.tasks)} 个")
        print(f"   创建车辆: {len(self.vehicles)} 个")
    
    def _load_edges(self) -> List[IntersectionEdge]:
        """加载出入口边"""
        edges = []
        for edge_data in self.map_data.get("intersection_edges", []):
            edge = IntersectionEdge(
                edge_id=edge_data["edge_id"],
                center_x=edge_data["center_x"],
                center_y=edge_data["center_y"],
                length=edge_data.get("length", 5),
                direction=edge_data.get("direction", "")
            )
            edges.append(edge)
        return edges
    
    def _generate_tasks(self) -> List[Task]:
        """为每个出入口边生成一个任务 - 边上随机整数起点 → 非相邻边随机整数终点"""
        tasks = []
        
        for i, start_edge in enumerate(self.edges):
            # 选择非相邻的终点边（排除距离最近的两条）
            end_edge = self._select_non_adjacent_edge(start_edge)
            if not end_edge:
                continue
            
            # 🎯 在边上生成随机整数坐标的起点和终点
            start_pos = start_edge.get_random_integer_position()
            end_pos = end_edge.get_random_integer_position()
            
            task = Task(
                task_id=i + 1,
                start_edge=start_edge,
                end_edge=end_edge,
                start_pos=start_pos,
                end_pos=end_pos,
                priority=1  # 默认优先级
            )
            tasks.append(task)
            
            print(f"  任务 T{task.task_id}: {start_edge.edge_id}({start_pos}) -> {end_edge.edge_id}({end_pos})")
        
        return tasks
    
    def _select_non_adjacent_edge(self, start_edge: IntersectionEdge) -> Optional[IntersectionEdge]:
        """选择非相邻边（排除距离最近的两条）"""
        if len(self.edges) <= 3:
            # 边数太少，随便选一个不同的边
            others = [e for e in self.edges if e.edge_id != start_edge.edge_id]
            return random.choice(others) if others else None
        
        # 计算距离并排序
        edge_distances = []
        for edge in self.edges:
            if edge.edge_id == start_edge.edge_id:
                continue
            
            distance = math.sqrt(
                (edge.center_x - start_edge.center_x)**2 + 
                (edge.center_y - start_edge.center_y)**2
            )
            edge_distances.append((edge, distance))
        
        edge_distances.sort(key=lambda x: x[1])
        
        # 🎯 排除最近的两条边
        if len(edge_distances) <= 2:
            return edge_distances[0][0] if edge_distances else None
        else:
            valid_edges = [ed[0] for ed in edge_distances[2:]]  # 排除最近的两条
            return random.choice(valid_edges)
    
    def _create_vehicles(self) -> List[Vehicle]:
        """为每个任务创建车辆"""
        vehicles = []
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for task in self.tasks:
            vehicle = Vehicle(
                vehicle_id=task.task_id,
                task=task,
                color=colors[(task.task_id - 1) % len(colors)]
            )
            vehicles.append(vehicle)
        
        return vehicles
    
    def apply_intelligent_priorities(self):
        """应用智能优先级（如果可用）"""
        if not HAS_INTELLIGENT_PRIORITY:
            print("📋 使用默认优先级（智能优先级模块不可用）")
            return
        
        try:
            # 转换为priority模块需要的格式
            scenarios = []
            for vehicle in self.vehicles:
                task = vehicle.task
                start_x, start_y = task.start_pos
                end_x, end_y = task.end_pos
                
                # 计算朝向
                dx = end_x - start_x
                dy = end_y - start_y
                theta = math.atan2(dy, dx)
                
                scenario = {
                    'id': vehicle.vehicle_id,
                    'priority': task.priority,
                    'color': vehicle.color,
                    'start': VehicleState(x=start_x, y=start_y, theta=theta, v=3.0, t=0.0),
                    'goal': VehicleState(x=end_x, y=end_y, theta=theta, v=2.0, t=0.0),
                    'description': f'Vehicle {vehicle.vehicle_id} ({task.start_edge.edge_id}->{task.end_edge.edge_id})'
                }
                scenarios.append(scenario)
            
            # 应用智能优先级
            priority_assigner = IntelligentPriorityAssigner(self.environment)
            intelligent_scenarios = priority_assigner.assign_intelligent_priorities(scenarios)
            
            # 更新任务优先级
            for scenario in intelligent_scenarios:
                vehicle_id = scenario['id']
                new_priority = scenario['priority']
                
                for vehicle in self.vehicles:
                    if vehicle.vehicle_id == vehicle_id:
                        vehicle.task.priority = int(new_priority)
                        break
            
            print("✅ 智能优先级应用成功")
            
        except Exception as e:
            print(f"⚠️ 智能优先级应用失败: {e}")
    
    def plan_all_vehicles(self):
        """同时规划所有车辆 - 基础版本"""
        print(f"\n🎯 基础规划模式: {len(self.vehicles)} 个车辆...")
        
        # 按优先级排序
        self.vehicles.sort(key=lambda v: v.task.priority, reverse=True)
        
        # 同时规划所有车辆
        successful_trajectories = []
        
        for vehicle in self.vehicles:
            print(f"   规划车辆 V{vehicle.vehicle_id} (优先级 {vehicle.task.priority})")
            
            trajectory = self._plan_single_vehicle(vehicle, successful_trajectories)
            if trajectory:
                vehicle.trajectory = trajectory
                successful_trajectories.append(trajectory)
                self.successful_plannings += 1
                print(f"      ✅ 成功: {len(trajectory)} 个轨迹点")
            else:
                print(f"      ❌ 失败")
        
        total_time = time.time() - self.planning_start_time
        success_rate = (self.successful_plannings / self.total_vehicles) * 100
        
        print(f"\n📊 基础规划结果:")
        print(f"   总车辆: {self.total_vehicles}")
        print(f"   成功: {self.successful_plannings}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   总时间: {total_time:.2f}s")
        
        return success_rate >= 50  # 成功率超过50%认为成功
    
    def _plan_single_vehicle(self, vehicle: Vehicle, existing_trajectories: List) -> Optional[List[VehicleState]]:
        """规划单个车辆"""
        task = vehicle.task
        start_x, start_y = task.start_pos
        end_x, end_y = task.end_pos
        
        # 计算朝向
        dx = end_x - start_x
        dy = end_y - start_y
        theta = math.atan2(dy, dx)
        
        start_state = VehicleState(x=start_x, y=start_y, theta=theta, v=3.0, t=0.0)
        goal_state = VehicleState(x=end_x, y=end_y, theta=theta, v=2.0, t=0.0)
        
        # 创建规划器并规划
        planner = VHybridAStarPlanner(self.environment, self.optimization_level)
        
        planning_start = time.time()
        try:
            trajectory = planner.search_with_waiting(
                start_state, goal_state, vehicle.vehicle_id, existing_trajectories
            )
            vehicle.planning_time = time.time() - planning_start
            return trajectory
        except Exception as e:
            vehicle.planning_time = time.time() - planning_start
            print(f"      异常: {str(e)}")
            return None
    
    def get_successful_vehicles(self) -> List[Vehicle]:
        """获取规划成功的车辆"""
        return [v for v in self.vehicles if v.trajectory is not None]

class LifelongGNNPlanner(FirstRoundPlanner):
    """终身GNN规划器 - 继承第一轮规划器，添加GNN增强"""
    
    def __init__(self, map_file: str, 
                 optimization_level: OptimizationLevel = OptimizationLevel.FULL,
                 use_gnn: bool = True):
        
        # 调用父类初始化
        super().__init__(map_file, optimization_level)
        
        self.use_gnn = use_gnn and HAS_GNN_INTEGRATION
        
        if self.use_gnn:
            print(f"🧠 启用GNN增强模式")
            # 创建GNN集成协调器
            self.gnn_coordinator = PretrainedGNNIntegratedCoordinator(
                map_file_path=map_file,
                optimization_level=optimization_level,
                gnn_enhancement_level=GNNEnhancementLevel.PRETRAINED_FULL
            )
            
            # 🔧 修补GNN图构建器以处理特征维度不匹配
            self._patch_gnn_graph_builder()
        else:
            print(f"📋 使用基础规划模式")
            self.gnn_coordinator = None
        
        print(f"🚀 终身GNN规划器初始化完成")
        print(f"   GNN状态: {'启用' if self.use_gnn else '未启用'}")
        print(f"   优化级别: {optimization_level.value}")
    
    def _patch_gnn_graph_builder(self):
        """修补GNN图构建器以处理特征维度不匹配"""
        if hasattr(self.gnn_coordinator, 'pretrained_gnn_planner') and \
           hasattr(self.gnn_coordinator.pretrained_gnn_planner, 'graph_builder'):
            
            graph_builder = self.gnn_coordinator.pretrained_gnn_planner.graph_builder
            original_extract = graph_builder._extract_enhanced_node_features
            
            def patched_extract_features(vehicles_info):
                """修补的特征提取 - 确保特征维度匹配"""
                features = original_extract(vehicles_info)
                
                # 检查特征维度并进行适配
                if features and len(features[0]) == 8:
                    # 从8维扩展到10维
                    print(f"🔧 特征维度适配: 8维 → 10维")
                    adapted_features = []
                    for feature_vec in features:
                        # 添加两个补充特征
                        extended_vec = feature_vec + [
                            0.5,  # [8] 占位特征1
                            0.5   # [9] 占位特征2
                        ]
                        adapted_features.append(extended_vec)
                    return adapted_features
                elif features and len(features[0]) == 12:
                    # 从12维截断到10维
                    print(f"🔧 特征维度适配: 12维 → 10维")
                    adapted_features = []
                    for feature_vec in features:
                        truncated_vec = feature_vec[:10]  # 截断到前10维
                        adapted_features.append(truncated_vec)
                    return adapted_features
                else:
                    # 维度已经匹配或其他情况
                    return features
            
            # 替换原有方法
            graph_builder._extract_enhanced_node_features = patched_extract_features
            print(f"✅ GNN图构建器已修补，支持特征维度适配")
    
    def plan_all_vehicles_with_gnn(self):
        """使用GNN增强规划所有车辆"""
        
        if not self.use_gnn or not self.gnn_coordinator:
            print("⚠️ GNN不可用，回退到基础规划")
            return super().plan_all_vehicles()
        
        print(f"\n🧠 GNN增强多车规划: {len(self.vehicles)}辆车")
        
        # 为GNN协调器准备兼容的地图数据
        self._prepare_gnn_compatible_map_data()
        
        # 更新GNN协调器的地图数据
        self.gnn_coordinator.map_data = self.map_data
        
        try:
            # 使用GNN集成协调器规划
            planning_start = time.time()
            gnn_results, gnn_scenarios = self.gnn_coordinator.plan_with_pretrained_gnn_integration()
            planning_time = time.time() - planning_start
            
            # 转换结果回原格式
            success_count = self._convert_gnn_results_back(gnn_results)
            
            print(f"📊 GNN增强规划结果:")
            print(f"   成功: {success_count}/{len(self.vehicles)}")
            print(f"   成功率: {100*success_count/len(self.vehicles):.1f}%")
            print(f"   规划时间: {planning_time:.2f}s")
            
            self.successful_plannings = success_count
            return success_count >= len(self.vehicles) * 0.5
            
        except Exception as e:
            print(f"❌ GNN规划失败: {str(e)}")
            print("🔄 回退到基础规划")
            return super().plan_all_vehicles()
    
    def _prepare_gnn_compatible_map_data(self):
        """为GNN协调器准备兼容的地图数据"""
        # 如果地图数据中没有 point_pairs，从我们的 vehicles 创建
        if not self.map_data.get("point_pairs"):
            start_points = []
            end_points = []
            point_pairs = []
            
            for vehicle in self.vehicles:
                task = vehicle.task
                start_x, start_y = task.start_pos
                end_x, end_y = task.end_pos
                
                # 创建起点
                start_point = {
                    "id": vehicle.vehicle_id,
                    "x": start_x,
                    "y": start_y
                }
                start_points.append(start_point)
                
                # 创建终点
                end_point = {
                    "id": vehicle.vehicle_id,
                    "x": end_x,
                    "y": end_y
                }
                end_points.append(end_point)
                
                # 创建配对
                pair = {
                    "start_id": vehicle.vehicle_id,
                    "end_id": vehicle.vehicle_id
                }
                point_pairs.append(pair)
            
            # 更新地图数据
            self.map_data["start_points"] = start_points
            self.map_data["end_points"] = end_points
            self.map_data["point_pairs"] = point_pairs
            
            print(f"🔄 为GNN协调器创建了兼容数据: {len(point_pairs)}个配对")
    
    def _convert_to_gnn_scenarios(self) -> List[Dict]:
        """转换为GNN协调器需要的场景格式"""
        # 为GNN协调器准备兼容的地图数据
        self._prepare_gnn_compatible_map_data()
        
        scenarios = []
        
        for vehicle in self.vehicles:
            task = vehicle.task
            start_x, start_y = task.start_pos
            end_x, end_y = task.end_pos
            
            # 计算朝向
            dx = end_x - start_x
            dy = end_y - start_y
            theta = math.atan2(dy, dx)
            
            scenario = {
                'id': vehicle.vehicle_id,
                'priority': task.priority,
                'color': vehicle.color,
                'start': VehicleState(x=start_x, y=start_y, theta=theta, v=3.0, t=0.0),
                'goal': VehicleState(x=end_x, y=end_y, theta=theta, v=2.0, t=0.0),
                'description': f'Vehicle {vehicle.vehicle_id} ({task.start_edge.edge_id}->{task.end_edge.edge_id})'
            }
            scenarios.append(scenario)
        
        return scenarios
    
    def _convert_gnn_results_back(self, gnn_results: Dict) -> int:
        """将GNN结果转换回原格式"""
        success_count = 0
        
        for vehicle in self.vehicles:
            vehicle_id = vehicle.vehicle_id
            
            if vehicle_id in gnn_results and gnn_results[vehicle_id]['trajectory']:
                vehicle.trajectory = gnn_results[vehicle_id]['trajectory']
                vehicle.planning_time = gnn_results[vehicle_id].get('planning_time', 0.0)
                success_count += 1
            else:
                vehicle.trajectory = None
                vehicle.planning_time = 0.0
        
        return success_count
    
    def create_advanced_visualization(self):
        """创建高级可视化"""
        
        if not HAS_ADVANCED_VISUALIZATION or not self.gnn_coordinator:
            print("🎬 使用简单可视化")
            visualizer = SimpleVisualizer(self)
            return visualizer.create_animation()
        
        print("🎬 创建GNN增强可视化")
        
        try:
            # 转换为高级可视化需要的格式
            results = {}
            scenarios = []
            
            for vehicle in self.vehicles:
                if vehicle.trajectory:
                    results[vehicle.vehicle_id] = {
                        'trajectory': vehicle.trajectory,
                        'color': vehicle.color,
                        'description': f'Vehicle {vehicle.vehicle_id} ({vehicle.task.start_edge.edge_id}->{vehicle.task.end_edge.edge_id})',
                        'planning_time': vehicle.planning_time
                    }
                    
                    scenarios.append({
                        'id': vehicle.vehicle_id,
                        'priority': vehicle.task.priority,
                        'color': vehicle.color,
                        'description': f'Vehicle {vehicle.vehicle_id} ({vehicle.task.start_edge.edge_id}->{vehicle.task.end_edge.edge_id})'
                    })
            
            if results:
                # 使用trying.py的高级可视化
                coordinator = MultiVehicleCoordinator(
                    optimization_level=self.optimization_level
                )
                coordinator.environment = self.environment
                return coordinator.create_animation(results, scenarios)
            else:
                print("❌ 没有成功轨迹可显示")
                return None
                
        except Exception as e:
            print(f"⚠️ 高级可视化失败: {str(e)}")
            print("🔄 回退到简单可视化")
            visualizer = SimpleVisualizer(self)
            return visualizer.create_animation()

class SimpleVisualizer:
    """简单可视化器"""
    
    def __init__(self, planner: FirstRoundPlanner):
        self.planner = planner
        self.fig, (self.ax_map, self.ax_stats) = plt.subplots(1, 2, figsize=(16, 8))
    
    def create_animation(self):
        """创建动画"""
        successful_vehicles = self.planner.get_successful_vehicles()
        
        if not successful_vehicles:
            print("❌ 没有成功的轨迹可以显示")
            return None
        
        # 计算最大时间
        max_time = max(max(state.t for state in v.trajectory) for v in successful_vehicles)
        
        def animate(frame):
            self.ax_map.clear()
            self.ax_stats.clear()
            
            current_time = frame * 0.2
            
            # 绘制环境
            self._draw_environment()
            
            # 绘制车辆
            active_count = 0
            for vehicle in successful_vehicles:
                current_state = self._get_state_at_time(vehicle.trajectory, current_time)
                if current_state:
                    self._draw_vehicle(current_state, vehicle.color)
                    active_count += 1
                
                # 绘制轨迹
                xs = [s.x for s in vehicle.trajectory]
                ys = [s.y for s in vehicle.trajectory]
                self.ax_map.plot(xs, ys, color=vehicle.color, alpha=0.6, linewidth=2)
            
            # 绘制任务起终点
            self._draw_tasks()
            
            self.ax_map.set_title(f'终身路口规划 - {self.planner.environment.map_name}\n'
                                 f'时间: {current_time:.1f}s | 活跃车辆: {active_count}')
            self.ax_map.set_xlim(0, self.planner.environment.size)
            self.ax_map.set_ylim(0, self.planner.environment.size)
            self.ax_map.grid(True, alpha=0.3)
            
            # 统计图
            self._draw_statistics()
            
            return []
        
        frames = int(max_time / 0.2) + 20
        anim = animation.FuncAnimation(self.fig, animate, frames=frames, 
                                     interval=200, blit=False, repeat=False)
        
        # 保存GIF
        try:
            writer = PillowWriter(fps=5)
            gif_filename = f"lifelong_gnn_{self.planner.environment.map_name}.gif"
            anim.save(gif_filename, writer=writer)
            print(f"✅ 动画已保存: {gif_filename}")
        except Exception as e:
            print(f"⚠️ 动画保存失败: {str(e)}")
        
        plt.tight_layout()
        plt.show()
        return anim
    
    def _draw_environment(self):
        """绘制环境"""
        env = self.planner.environment
        
        # 绘制障碍物
        obs_y, obs_x = np.where(env.obstacle_map)
        if len(obs_x) > 0:
            self.ax_map.scatter(obs_x, obs_y, c='darkred', s=3, alpha=0.8)
        
        # 绘制出入口边
        for edge in self.planner.edges:
            self._draw_edge(edge)
    
    def _draw_edge(self, edge: IntersectionEdge):
        """绘制出入口边"""
        color_map = {"north": "red", "south": "blue", "east": "green", "west": "orange"}
        color = color_map.get(edge.direction, "purple")
        
        edge_points = edge.get_points()
        for x, y in edge_points:
            self.ax_map.add_patch(patches.Rectangle(
                (x-0.5, y-0.5), 1, 1, 
                facecolor=color, alpha=0.6, edgecolor='white', linewidth=1
            ))
        
        self.ax_map.text(edge.center_x, edge.center_y, edge.edge_id,
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='white')
    
    def _draw_vehicle(self, state: VehicleState, color: str):
        """绘制车辆"""
        length, width = self.planner.params.length, self.planner.params.width
        
        cos_theta, sin_theta = math.cos(state.theta), math.sin(state.theta)
        
        corners = np.array([
            [-length/2, -width/2], [length/2, -width/2],
            [length/2, width/2], [-length/2, width/2],
            [-length/2, -width/2]
        ])
        
        rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_corners = corners @ rotation.T
        translated_corners = rotated_corners + np.array([state.x, state.y])
        
        vehicle_patch = patches.Polygon(translated_corners[:-1], 
                                      facecolor=color, alpha=0.8, 
                                      edgecolor='black', linewidth=1)
        self.ax_map.add_patch(vehicle_patch)
    
    def _draw_tasks(self):
        """绘制任务起终点"""
        for vehicle in self.planner.vehicles:
            task = vehicle.task
            start_x, start_y = task.start_pos
            end_x, end_y = task.end_pos
            
            # 起点 (绿色圆圈)
            self.ax_map.plot(start_x, start_y, 'go', markersize=6, markeredgecolor='darkgreen')
            # 终点 (红色方形)
            self.ax_map.plot(end_x, end_y, 'rs', markersize=6, markeredgecolor='darkred')
            # 连线
            self.ax_map.plot([start_x, end_x], [start_y, end_y], 
                           'k--', alpha=0.3, linewidth=1)
    
    def _draw_statistics(self):
        """绘制统计信息"""
        total = self.planner.total_vehicles
        successful = self.planner.successful_plannings
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        labels = ['成功', '失败']
        sizes = [successful, total - successful]
        colors = ['lightgreen', 'lightcoral']
        
        if sum(sizes) > 0:
            self.ax_stats.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        self.ax_stats.set_title(f'规划结果统计\n成功率: {success_rate:.1f}%')
    
    def _get_state_at_time(self, trajectory: List[VehicleState], target_time: float) -> Optional[VehicleState]:
        """获取指定时间的状态"""
        if not trajectory:
            return None
        
        if target_time <= trajectory[0].t:
            return trajectory[0]
        elif target_time >= trajectory[-1].t:
            return trajectory[-1]
        
        for i in range(len(trajectory) - 1):
            if trajectory[i].t <= target_time <= trajectory[i+1].t:
                # 线性插值
                t1, t2 = trajectory[i].t, trajectory[i+1].t
                if abs(t2 - t1) < 1e-6:
                    return trajectory[i]
                
                alpha = (target_time - t1) / (t2 - t1)
                
                return VehicleState(
                    x=trajectory[i].x + alpha * (trajectory[i+1].x - trajectory[i].x),
                    y=trajectory[i].y + alpha * (trajectory[i+1].y - trajectory[i].y),
                    theta=trajectory[i].theta + alpha * (trajectory[i+1].theta - trajectory[i].theta),
                    v=trajectory[i].v + alpha * (trajectory[i+1].v - trajectory[i].v),
                    t=target_time
                )
        
        return None

def main():
    """主函数"""
    print("🧠 终身GNN路口规划系统")
    print("🎯 每条出入口边一个任务：边上随机整数起点 → 非相邻边随机整数终点")
    print("🚀 可选GNN增强 + 智能优先级 + 高级可视化")
    print("=" * 70)
    
    # 查找地图文件
    import os
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    map_files = [f for f in json_files if any(keyword in f.lower() 
                for keyword in ['lifelong', 'intersection', 'cross', 'junction'])]
    
    if not map_files:
        print("❌ 未找到路口地图文件")
        print("💡 请使用 lifelong_map.py 创建路口地图")
        return
    
    print(f"📁 发现 {len(map_files)} 个地图文件:")
    for i, f in enumerate(map_files):
        print(f"   {i+1}. {f}")
    
    # 选择地图
    if len(map_files) == 1:
        selected_map = map_files[0]
    else:
        try:
            choice = input(f"选择地图 (1-{len(map_files)}) 或回车使用第1个: ").strip()
            if choice:
                selected_map = map_files[int(choice) - 1]
            else:
                selected_map = map_files[0]
        except:
            selected_map = map_files[0]
    
    print(f"🗺️ 使用地图: {selected_map}")
    
    # 选择规划模式
    print(f"\n🎯 选择规划模式:")
    print(f"   1. GNN增强模式 (推荐)")
    print(f"   2. 基础规划模式")
    
    mode_choice = input("选择模式 (1/2) 或回车使用GNN模式: ").strip()
    use_gnn = mode_choice != '2'
    
    # 选择优化级别
    print(f"\n⚙️ 选择优化级别:")
    print(f"   1. BASIC (快速)")
    print(f"   2. ENHANCED (平衡)")
    print(f"   3. FULL (完整，推荐)")
    
    opt_choice = input("选择优化级别 (1/2/3) 或回车使用FULL: ").strip()
    opt_levels = {
        '1': OptimizationLevel.BASIC,
        '2': OptimizationLevel.ENHANCED, 
        '3': OptimizationLevel.FULL
    }
    opt_level = opt_levels.get(opt_choice, OptimizationLevel.FULL)
    
    try:
        # 创建规划器
        if use_gnn:
            planner = LifelongGNNPlanner(selected_map, opt_level, use_gnn=True)
        else:
            planner = FirstRoundPlanner(selected_map, opt_level)
        
        # 分析冲突强度
        conflict_analysis = ConflictIntensityAnalyzer.analyze_scenario_conflicts(planner.vehicles)
        print(f"\n📊 场景分析:")
        print(f"   车辆数量: {len(planner.vehicles)}")
        print(f"   冲突强度: {conflict_analysis['intensity']:.3f}")
        print(f"   冲突对数: {conflict_analysis['conflict_count']}/{conflict_analysis['total_pairs']}")
        
        # 应用智能优先级（如果可用）
        planner.apply_intelligent_priorities()
        
        # 执行规划
        print(f"\n🚀 开始{'GNN增强' if use_gnn else '基础'}规划...")
        
        if use_gnn and isinstance(planner, LifelongGNNPlanner):
            success = planner.plan_all_vehicles_with_gnn()
        else:
            success = planner.plan_all_vehicles()
        
        # 创建可视化
        if success:
            print(f"\n🎬 创建可视化...")
            if use_gnn and isinstance(planner, LifelongGNNPlanner):
                planner.create_advanced_visualization()
            else:
                visualizer = SimpleVisualizer(planner)
                visualizer.create_animation()
            
            print(f"🎉 终身GNN路口规划完成！")
        else:
            print(f"⚠️ 规划成功率较低，仍会显示结果")
            visualizer = SimpleVisualizer(planner)
            visualizer.create_animation()
            
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()