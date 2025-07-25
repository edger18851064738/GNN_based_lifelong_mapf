#!/usr/bin/env python3
"""
第一轮多车轨迹规划系统
基于IEEE TITS论文的数学模型，实现多车协同路径规划

特性:
- 为每个出入口边生成一个任务
- 智能选择非相邻边作为终点
- 多车协同轨迹规划
- 完整的数学模型集成
- 实时可视化和性能监控

任务生成规则:
- 起点：在该出入口边上的随机位置
- 终点：在非相邻出入口边上的随机位置
- 排除规则：直线距离最近的两条边不作为终点
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
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import threading
import queue

# 导入核心规划模块
from trying import (
    VehicleState, VehicleParameters, OptimizationLevel,
    UnstructuredEnvironment, MultiVehicleCoordinator,
    VHybridAStarPlanner
)

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"       # 等待分配
    ASSIGNED = "assigned"     # 已分配车辆
    PLANNING = "planning"     # 规划中
    EXECUTING = "executing"   # 执行中
    COMPLETED = "completed"   # 已完成
    FAILED = "failed"         # 失败

class VehicleLifecycleStatus(Enum):
    """车辆生命周期状态"""
    SPAWNING = "spawning"     # 生成中
    ACTIVE = "active"         # 活跃
    COMPLETING = "completing" # 即将完成
    DESPAWNED = "despawned"   # 已消失

@dataclass
class IntersectionEdge:
    """进出口边 - 简化版，只关注位置信息"""
    edge_id: str
    center_x: int
    center_y: int  
    length: int = 5
    direction: str = ""  # 保留用于可视化，但规划时不使用
    
    def get_points(self) -> List[Tuple[int, int]]:
        """获取边界覆盖的所有点位 - 简化为以中心点为基础的线段"""
        points = []
        half_length = self.length // 2
        
        # 简化：默认创建水平线段，如果需要垂直可以通过direction判断
        if self.direction in ["north", "south"]:
            # 水平边界
            for x in range(self.center_x - half_length, self.center_x + half_length + 1):
                points.append((x, self.center_y))
        elif self.direction in ["east", "west"]:
            # 垂直边界  
            for y in range(self.center_y - half_length, self.center_y + half_length + 1):
                points.append((self.center_x, y))
        else:
            # 如果没有方向信息，默认创建水平边界
            for x in range(self.center_x - half_length, self.center_x + half_length + 1):
                points.append((x, self.center_y))
        
        return points
    
    def get_random_position(self) -> Tuple[float, float]:
        """在边界上获取随机位置"""
        points = self.get_points()
        if points:
            x, y = random.choice(points)
            # 添加少量随机偏移使位置更自然
            x += random.uniform(-0.3, 0.3)
            y += random.uniform(-0.3, 0.3)
            return (float(x), float(y))
        return (float(self.center_x), float(self.center_y))

@dataclass  
class LifelongTask:
    """持续任务"""
    task_id: int
    start_edge: IntersectionEdge
    end_edge: IntersectionEdge
    priority: int = 1
    creation_time: float = field(default_factory=time.time)
    status: TaskStatus = TaskStatus.PENDING
    assigned_vehicle_id: Optional[int] = None
    
    # 具体的起点和终点位置
    start_position: Optional[Tuple[float, float]] = None
    end_position: Optional[Tuple[float, float]] = None
    optimal_start_heading: Optional[float] = None
    
    def __post_init__(self):
        """生成具体位置和朝向"""
        if self.start_position is None:
            self.start_position = self.start_edge.get_random_position()
        
        if self.end_position is None:
            self.end_position = self.end_edge.get_random_position()
        
        if self.optimal_start_heading is None:
            # 根据起点到终点的方向计算最优朝向
            dx = self.end_position[0] - self.start_position[0]
            dy = self.end_position[1] - self.start_position[1]
            self.optimal_start_heading = math.atan2(dy, dx)

@dataclass
class LifelongVehicle:
    """持续系统中的车辆"""
    vehicle_id: int
    current_task: Optional[LifelongTask] = None
    trajectory: List[VehicleState] = field(default_factory=list)
    status: VehicleLifecycleStatus = VehicleLifecycleStatus.SPAWNING
    spawn_time: float = field(default_factory=time.time)
    last_update_time: float = field(default_factory=time.time)
    color: str = "blue"
    
    # 性能统计
    total_distance: float = 0.0
    planning_time: float = 0.0
    execution_time: float = 0.0

class EdgeManager:
    """进出口边管理器 - 简化版"""
    
    def __init__(self, edges_data: List[Dict]):
        self.edges: List[IntersectionEdge] = []
        
        # 加载边界数据
        for edge_data in edges_data:
            edge = IntersectionEdge(
                edge_id=edge_data["edge_id"],
                center_x=edge_data["center_x"],
                center_y=edge_data["center_y"],
                length=edge_data.get("length", 5),
                direction=edge_data.get("direction", "")  # 可选，仅用于可视化
            )
            self.edges.append(edge)
        
        print(f"📍 EdgeManager 初始化: {len(self.edges)} 个进出口边")
    
    def get_non_adjacent_edges(self, start_edge: IntersectionEdge) -> List[IntersectionEdge]:
        """获取非相邻的边界（排除距离最近的两条边）"""
        if len(self.edges) <= 3:
            # 如果边数太少，返回所有其他边
            return [edge for edge in self.edges if edge.edge_id != start_edge.edge_id]
        
        # 计算所有其他边到起始边的距离
        edge_distances = []
        for edge in self.edges:
            if edge.edge_id == start_edge.edge_id:
                continue
            
            distance = math.sqrt(
                (edge.center_x - start_edge.center_x)**2 + 
                (edge.center_y - start_edge.center_y)**2
            )
            edge_distances.append((edge, distance))
        
        # 按距离排序
        edge_distances.sort(key=lambda x: x[1])
        
        # 排除距离最近的两条边，返回其余的边
        if len(edge_distances) <= 2:
            return [ed[0] for ed in edge_distances]  # 如果只有2条或更少，全部返回
        else:
            return [ed[0] for ed in edge_distances[2:]]  # 排除最近的两条
    
    def get_random_non_adjacent_edge(self, start_edge: IntersectionEdge) -> Optional[IntersectionEdge]:
        """选择一个随机的非相邻边界"""
        valid_edges = self.get_non_adjacent_edges(start_edge)
        return random.choice(valid_edges) if valid_edges else None

class TaskGenerator:
    """任务生成器 - 第一轮版本：为每个出入口边生成一个任务"""
    
    def __init__(self, edge_manager: EdgeManager):
        self.edge_manager = edge_manager
        self.task_id_counter = 1
        
        # 统计信息
        self.total_generated = 0
        
    def generate_initial_tasks(self) -> List[LifelongTask]:
        """为每个出入口边生成一个初始任务"""
        initial_tasks = []
        
        print(f"🎯 为 {len(self.edge_manager.edges)} 个出入口边生成初始任务...")
        
        for start_edge in self.edge_manager.edges:
            # 选择非相邻的终点边
            end_edge = self.edge_manager.get_random_non_adjacent_edge(start_edge)
            
            if end_edge is None:
                print(f"⚠️ 边界 {start_edge.edge_id} 无法找到合适的终点边")
                continue
            
            # 计算基础优先级（可以基于距离）
            distance = math.sqrt(
                (end_edge.center_x - start_edge.center_x)**2 + 
                (end_edge.center_y - start_edge.center_y)**2
            )
            base_priority = min(5, max(1, int(distance / 15)))
            
            task = LifelongTask(
                task_id=self.task_id_counter,
                start_edge=start_edge,
                end_edge=end_edge,
                priority=base_priority
            )
            
            initial_tasks.append(task)
            self.task_id_counter += 1
            self.total_generated += 1
            
            print(f"  ✅ 任务 T{task.task_id}: {start_edge.edge_id} -> {end_edge.edge_id} "
                  f"(距离: {distance:.1f}m, 优先级: {base_priority})")
        
        print(f"📋 初始任务生成完成: {len(initial_tasks)} 个任务")
        return initial_tasks
    
    def get_generation_stats(self) -> Dict:
        """获取生成统计"""
        return {
            'total_generated': self.total_generated,
            'generation_mode': 'initial_round'
        }

class VehicleManager:
    """车辆管理器"""
    
    def __init__(self, params: VehicleParameters):
        self.params = params
        self.vehicles: Dict[int, LifelongVehicle] = {}
        self.vehicle_id_counter = 1
        self.completed_vehicles = 0
        self.failed_vehicles = 0
        
        # 颜色池
        self.color_pool = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 
                          'gray', 'olive', 'cyan', 'magenta', 'yellow', 'navy', 'lime']
        self.used_colors = set()
        
    def create_vehicle(self, task: LifelongTask) -> LifelongVehicle:
        """创建新车辆"""
        # 选择颜色
        available_colors = [c for c in self.color_pool if c not in self.used_colors]
        if not available_colors:
            available_colors = self.color_pool
            self.used_colors.clear()
        
        color = random.choice(available_colors)
        self.used_colors.add(color)
        
        vehicle = LifelongVehicle(
            vehicle_id=self.vehicle_id_counter,
            current_task=task,
            color=color
        )
        
        self.vehicles[self.vehicle_id_counter] = vehicle
        self.vehicle_id_counter += 1
        
        # 更新任务状态
        task.status = TaskStatus.ASSIGNED
        task.assigned_vehicle_id = vehicle.vehicle_id
        
        return vehicle
    
    def update_vehicle_trajectory(self, vehicle_id: int, trajectory: List[VehicleState]):
        """更新车辆轨迹"""
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            vehicle.trajectory = trajectory
            vehicle.status = VehicleLifecycleStatus.ACTIVE
            vehicle.last_update_time = time.time()
            
            if vehicle.current_task:
                vehicle.current_task.status = TaskStatus.EXECUTING
    
    def remove_vehicle(self, vehicle_id: int, completed: bool = True):
        """移除车辆"""
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            
            # 释放颜色
            if vehicle.color in self.used_colors:
                self.used_colors.remove(vehicle.color)
            
            # 更新统计
            if completed:
                self.completed_vehicles += 1
                if vehicle.current_task:
                    vehicle.current_task.status = TaskStatus.COMPLETED
            else:
                self.failed_vehicles += 1
                if vehicle.current_task:
                    vehicle.current_task.status = TaskStatus.FAILED
            
            del self.vehicles[vehicle_id]
    
    def get_active_vehicles(self) -> List[LifelongVehicle]:
        """获取活跃车辆"""
        return [v for v in self.vehicles.values() 
                if v.status in [VehicleLifecycleStatus.ACTIVE, VehicleLifecycleStatus.SPAWNING]]
    
    def get_active_trajectories(self) -> List[List[VehicleState]]:
        """获取所有活跃车辆的轨迹"""
        trajectories = []
        for vehicle in self.get_active_vehicles():
            if vehicle.trajectory:
                trajectories.append(vehicle.trajectory)
        return trajectories
    
    def cleanup_completed_vehicles(self, current_time: float):
        """清理已完成的车辆"""
        to_remove = []
        
        for vehicle_id, vehicle in self.vehicles.items():
            if not vehicle.trajectory:
                continue
            
            # 检查是否到达终点
            if vehicle.current_task and vehicle.trajectory:
                last_state = vehicle.trajectory[-1]
                end_pos = vehicle.current_task.end_position
                
                if end_pos:
                    distance_to_end = math.sqrt(
                        (last_state.x - end_pos[0])**2 + 
                        (last_state.y - end_pos[1])**2
                    )
                    
                    # 如果接近终点，标记为完成
                    if distance_to_end < 2.0:
                        to_remove.append((vehicle_id, True))
                        continue
            
            # 检查超时
            if current_time - vehicle.last_update_time > 120:  # 2分钟超时
                to_remove.append((vehicle_id, False))
        
        # 移除车辆
        for vehicle_id, completed in to_remove:
            self.remove_vehicle(vehicle_id, completed)
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        active_count = len(self.get_active_vehicles())
        
        return {
            'active_vehicles': active_count,
            'completed_vehicles': self.completed_vehicles,
            'failed_vehicles': self.failed_vehicles,
            'total_spawned': self.vehicle_id_counter - 1,
            'success_rate': self.completed_vehicles / max(1, self.completed_vehicles + self.failed_vehicles) * 100
        }

class LifelongPlanner:
    """第一轮多车轨迹规划器"""
    
    def __init__(self, map_file: str, optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED):
        self.optimization_level = optimization_level
        
        # 加载地图
        self.environment = UnstructuredEnvironment()
        self.map_data = self.environment.load_from_json(map_file)
        
        if not self.map_data:
            raise ValueError(f"无法加载地图文件: {map_file}")
        
        # 初始化组件
        self.params = VehicleParameters()
        self.edge_manager = EdgeManager(self.map_data.get("intersection_edges", []))
        self.task_generator = TaskGenerator(self.edge_manager)
        self.vehicle_manager = VehicleManager(self.params)
        
        # 生成第一轮任务
        self.pending_tasks: List[LifelongTask] = self.task_generator.generate_initial_tasks()
        self.completed_tasks: List[LifelongTask] = []
        self.all_tasks_assigned = False
        
        # 系统状态
        self.system_start_time = time.time()
        self.total_planning_attempts = 0
        self.successful_plannings = 0
        
        # 性能监控
        self.performance_history = deque(maxlen=200)
        
        print(f"🚀 第一轮规划器初始化完成")
        print(f"   地图: {self.map_data.get('map_info', {}).get('name', 'Unknown')}")
        print(f"   进出口边: {len(self.edge_manager.edges)} 个")
        print(f"   初始任务: {len(self.pending_tasks)} 个")
        print(f"   优化级别: {optimization_level.value}")
    
    def assign_tasks_to_vehicles(self):
        """为所有待分配任务创建车辆并同时规划"""
        if not self.pending_tasks:
            if not self.all_tasks_assigned:
                print(f"📋 所有任务已分配完成")
                self.all_tasks_assigned = True
            return
        
        # 按优先级排序任务
        self.pending_tasks.sort(key=lambda t: t.priority, reverse=True)
        
        # 批量创建车辆（不立即规划）
        new_vehicles = []
        max_simultaneous = min(len(self.pending_tasks), 8)  # 限制同时规划的车辆数
        
        print(f"🚗 批量创建 {max_simultaneous} 个车辆...")
        
        for _ in range(max_simultaneous):
            if not self.pending_tasks:
                break
                
            task = self.pending_tasks.pop(0)
            vehicle = self.vehicle_manager.create_vehicle(task)
            new_vehicles.append(vehicle)
            
            print(f"   创建车辆 V{vehicle.vehicle_id} 执行任务 T{task.task_id}")
        
        if new_vehicles:
            print(f"🎯 开始同时规划 {len(new_vehicles)} 个车辆的轨迹...")
            # 同时规划所有新车辆
            self.plan_vehicles_simultaneously(new_vehicles)
    
    def plan_vehicles_simultaneously(self, new_vehicles: List[LifelongVehicle]):
        """同时规划多个车辆的轨迹"""
        if not new_vehicles:
            return
        
        # 获取已有车辆的轨迹作为静态障碍物
        existing_trajectories = self.vehicle_manager.get_active_trajectories()
        
        print(f"   考虑 {len(existing_trajectories)} 个已有车辆轨迹作为动态障碍物")
        
        # 第一阶段：为所有新车辆进行初始规划（不考虑彼此）
        initial_plans = {}
        successful_initial_plans = 0
        
        print(f"   阶段1: 初始规划（不考虑新车辆间冲突）")
        
        for vehicle in new_vehicles:
            trajectory = self._plan_single_vehicle(vehicle, existing_trajectories)
            if trajectory:
                initial_plans[vehicle.vehicle_id] = trajectory
                successful_initial_plans += 1
                print(f"      ✅ V{vehicle.vehicle_id} 初始规划成功: {len(trajectory)} 个点")
            else:
                print(f"      ❌ V{vehicle.vehicle_id} 初始规划失败")
                self.vehicle_manager.remove_vehicle(vehicle.vehicle_id, completed=False)
        
        if successful_initial_plans == 0:
            print(f"   ⚠️ 所有车辆初始规划都失败")
            return
        
        # 第二阶段：冲突检测与解决
        print(f"   阶段2: 冲突检测与解决...")
        final_plans = self._resolve_conflicts(initial_plans, new_vehicles)
        
        # 第三阶段：应用最终轨迹
        print(f"   阶段3: 应用最终轨迹")
        successful_final_plans = 0
        
        for vehicle_id, trajectory in final_plans.items():
            if trajectory:
                self.vehicle_manager.update_vehicle_trajectory(vehicle_id, trajectory)
                successful_final_plans += 1
                print(f"      ✅ V{vehicle_id} 最终轨迹: {len(trajectory)} 个点")
            else:
                print(f"      ❌ V{vehicle_id} 最终规划失败")
                self.vehicle_manager.remove_vehicle(vehicle_id, completed=False)
        
        print(f"🎊 同时规划完成: {successful_final_plans}/{len(new_vehicles)} 成功")
    
    def _plan_single_vehicle(self, vehicle: LifelongVehicle, 
                           existing_trajectories: List[List[VehicleState]]) -> Optional[List[VehicleState]]:
        """为单个车辆规划轨迹"""
        if not vehicle.current_task:
            return None
    
    def plan_vehicle_trajectory(self, vehicle: LifelongVehicle):
        """单车辆规划轨迹（向后兼容）"""
        print(f"🎯 单独规划车辆 V{vehicle.vehicle_id}")
        trajectory = self._plan_single_vehicle(vehicle, self.vehicle_manager.get_active_trajectories())
        if trajectory:
            self.vehicle_manager.update_vehicle_trajectory(vehicle.vehicle_id, trajectory)
            print(f"✅ 车辆 V{vehicle.vehicle_id} 规划成功: {len(trajectory)} 个轨迹点")
        else:
            print(f"❌ 车辆 V{vehicle.vehicle_id} 规划失败")
            self.vehicle_manager.remove_vehicle(vehicle.vehicle_id, completed=False)
        
        task = vehicle.current_task
        task.status = TaskStatus.PLANNING
        
        # 创建起点和终点状态
        start_pos = task.start_position
        end_pos = task.end_position
        
        start_state = VehicleState(
            x=start_pos[0], y=start_pos[1],
            theta=task.optimal_start_heading,
            v=2.0, t=0.0
        )
        
        end_state = VehicleState(
            x=end_pos[0], y=end_pos[1], 
            theta=task.optimal_start_heading,
            v=1.0, t=0.0
        )
        
        # 创建规划器
        planner = VHybridAStarPlanner(self.environment, self.optimization_level)
        
        # 规划轨迹
        planning_start_time = time.time()
        self.total_planning_attempts += 1
        
        try:
            trajectory = planner.search_with_waiting(
                start_state, end_state, vehicle.vehicle_id, existing_trajectories
            )
            
            planning_time = time.time() - planning_start_time
            vehicle.planning_time = planning_time
            
            if trajectory:
                self.successful_plannings += 1
                
                # 记录性能
                self.performance_history.append({
                    'timestamp': time.time(),
                    'planning_time': planning_time,
                    'trajectory_length': len(trajectory),
                    'vehicle_id': vehicle.vehicle_id,
                    'success': True
                })
                
                return trajectory
            else:
                # 记录失败
                self.performance_history.append({
                    'timestamp': time.time(),
                    'planning_time': planning_time,
                    'trajectory_length': 0,
                    'vehicle_id': vehicle.vehicle_id,
                    'success': False
                })
                return None
                
        except Exception as e:
            planning_time = time.time() - planning_start_time
            print(f"      💥 V{vehicle.vehicle_id} 规划异常: {str(e)}")
            return None
    
    def _resolve_conflicts(self, initial_plans: Dict[int, List[VehicleState]], 
                         vehicles: List[LifelongVehicle]) -> Dict[int, List[VehicleState]]:
        """解决车辆间冲突"""
        # 检测冲突
        conflicts = self._detect_conflicts(initial_plans)
        
        if not conflicts:
            print(f"      ✅ 无冲突检测到")
            return initial_plans
        
        print(f"      ⚠️ 检测到 {len(conflicts)} 个冲突对")
        
        # 简化的冲突解决策略：按优先级重新规划冲突车辆
        final_plans = initial_plans.copy()
        
        # 收集所有冲突车辆
        conflicted_vehicles = set()
        for v1_id, v2_id in conflicts:
            conflicted_vehicles.add(v1_id)
            conflicted_vehicles.add(v2_id)
        
        print(f"      🔄 重新规划 {len(conflicted_vehicles)} 个冲突车辆")
        
        # 按优先级排序冲突车辆
        conflicted_vehicle_objects = [v for v in vehicles if v.vehicle_id in conflicted_vehicles]
        conflicted_vehicle_objects.sort(
            key=lambda v: v.current_task.priority if v.current_task else 0, 
            reverse=True
        )
        
        # 逐个重新规划冲突车辆（低优先级的需要避开高优先级的）
        for vehicle in conflicted_vehicle_objects:
            # 收集需要避开的轨迹（优先级更高的车辆 + 非冲突车辆）
            avoid_trajectories = []
            
            for other_vehicle in vehicles:
                if (other_vehicle.vehicle_id != vehicle.vehicle_id and 
                    other_vehicle.vehicle_id in final_plans):
                    
                    # 如果是更高优先级的车辆，或者不在冲突中，则需要避开
                    other_priority = other_vehicle.current_task.priority if other_vehicle.current_task else 0
                    current_priority = vehicle.current_task.priority if vehicle.current_task else 0
                    
                    if (other_priority > current_priority or 
                        other_vehicle.vehicle_id not in conflicted_vehicles):
                        avoid_trajectories.append(final_plans[other_vehicle.vehicle_id])
            
            print(f"        🎯 重新规划 V{vehicle.vehicle_id} (优先级 {vehicle.current_task.priority if vehicle.current_task else 0})")
            
            # 重新规划
            new_trajectory = self._plan_single_vehicle(vehicle, avoid_trajectories)
            if new_trajectory:
                final_plans[vehicle.vehicle_id] = new_trajectory
                print(f"        ✅ V{vehicle.vehicle_id} 冲突解决成功")
            else:
                print(f"        ❌ V{vehicle.vehicle_id} 冲突解决失败")
                final_plans[vehicle.vehicle_id] = None
        
        return final_plans
    
    def _detect_conflicts(self, plans: Dict[int, List[VehicleState]]) -> List[Tuple[int, int]]:
        """检测轨迹间的冲突"""
        conflicts = []
        vehicle_ids = list(plans.keys())
        safety_distance = self.params.get_current_safety_distance()
        
        for i, v1_id in enumerate(vehicle_ids):
            for v2_id in vehicle_ids[i+1:]:
                if self._trajectories_conflict(plans[v1_id], plans[v2_id], safety_distance):
                    conflicts.append((v1_id, v2_id))
        
        return conflicts
    
    def _trajectories_conflict(self, traj1: List[VehicleState], traj2: List[VehicleState], 
                             safety_distance: float) -> bool:
        """检查两条轨迹是否冲突"""
        if not (traj1 and traj2):
            return False
        
        # 检查时间重叠的轨迹段
        max_time = min(traj1[-1].t, traj2[-1].t)
        
        # 以一定时间间隔检查冲突
        check_interval = 0.5
        current_time = 0.0
        
        while current_time <= max_time:
            state1 = self._interpolate_trajectory_at_time(traj1, current_time)
            state2 = self._interpolate_trajectory_at_time(traj2, current_time)
            
            if state1 and state2:
                distance = math.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
                if distance < safety_distance:
                    return True
            
            current_time += check_interval
        
        return False
    
    def _interpolate_trajectory_at_time(self, trajectory: List[VehicleState], 
                                      target_time: float) -> Optional[VehicleState]:
        """在轨迹中插值指定时间的状态"""
        if not trajectory:
            return None
        
        if target_time <= trajectory[0].t:
            return trajectory[0]
        elif target_time >= trajectory[-1].t:
            return trajectory[-1]
        
        for i in range(len(trajectory) - 1):
            if trajectory[i].t <= target_time <= trajectory[i+1].t:
                t1, t2 = trajectory[i].t, trajectory[i+1].t
                
                if abs(t2 - t1) < 1e-6:
                    return trajectory[i]
                
                alpha = (target_time - t1) / (t2 - t1)
                
                # 角度插值处理
                theta1, theta2 = trajectory[i].theta, trajectory[i+1].theta
                theta_diff = theta2 - theta1
                if theta_diff > math.pi:
                    theta_diff -= 2 * math.pi
                elif theta_diff < -math.pi:
                    theta_diff += 2 * math.pi
                interpolated_theta = theta1 + alpha * theta_diff
                
                return VehicleState(
                    x=trajectory[i].x + alpha * (trajectory[i+1].x - trajectory[i].x),
                    y=trajectory[i].y + alpha * (trajectory[i+1].y - trajectory[i].y),
                    theta=interpolated_theta,
                    v=trajectory[i].v + alpha * (trajectory[i+1].v - trajectory[i].v),
                    t=target_time
                )
        
        return None
    
    def update_system_state(self, current_time: float):
        """更新系统状态 - 第一轮版本"""
        # 清理完成的车辆
        self.vehicle_manager.cleanup_completed_vehicles(current_time)
        
        # 分配剩余任务（如果有的话）
        if self.pending_tasks and len(self.vehicle_manager.get_active_vehicles()) < 10:
            self.assign_tasks_to_vehicles()
    
    def is_round_completed(self) -> bool:
        """检查第一轮是否完成"""
        return (len(self.pending_tasks) == 0 and 
                len(self.vehicle_manager.get_active_vehicles()) == 0)
    
    def get_system_statistics(self) -> Dict:
        """获取系统统计"""
        current_time = time.time()
        runtime = current_time - self.system_start_time
        
        vehicle_stats = self.vehicle_manager.get_statistics()
        task_stats = self.task_generator.get_generation_stats()
        
        # 计算平均规划时间
        recent_performance = [p for p in self.performance_history 
                            if current_time - p['timestamp'] <= 300]  # 最近5分钟
        
        avg_planning_time = 0.0
        if recent_performance:
            avg_planning_time = sum(p['planning_time'] for p in recent_performance) / len(recent_performance)
        
        planning_success_rate = 0.0
        if self.total_planning_attempts > 0:
            planning_success_rate = (self.successful_plannings / self.total_planning_attempts) * 100
        
        # 第一轮完成度
        total_initial_tasks = task_stats['total_generated']
        completed_tasks = vehicle_stats['completed_vehicles']
        completion_rate = (completed_tasks / max(1, total_initial_tasks)) * 100
        
        return {
            'runtime_seconds': runtime,
            'vehicle_stats': vehicle_stats,
            'task_stats': task_stats,
            'planning_stats': {
                'total_attempts': self.total_planning_attempts,
                'successful_plannings': self.successful_plannings,
                'success_rate': planning_success_rate,
                'avg_planning_time': avg_planning_time
            },
            'round_progress': {
                'total_tasks': total_initial_tasks,
                'pending_tasks': len(self.pending_tasks),
                'active_vehicles': vehicle_stats['active_vehicles'],
                'completed_tasks': completed_tasks,
                'completion_rate': completion_rate,
                'round_completed': self.is_round_completed()
            }
        }

class LifelongVisualizer:
    """Lifelong系统可视化器"""
    
    def __init__(self, planner: LifelongPlanner):
        self.planner = planner
        self.fig, self.axes = plt.subplots(2, 2, figsize=(16, 12))
        self.ax_map = self.axes[0, 0]
        self.ax_stats = self.axes[0, 1] 
        self.ax_timeline = self.axes[1, 0]
        self.ax_performance = self.axes[1, 1]
        
        # 可视化历史
        self.stats_history = deque(maxlen=300)
        self.timeline_vehicles = deque(maxlen=50)
        
        plt.subplots_adjust(hspace=0.3, wspace=0.3)
    
    def setup_map_visualization(self):
        """设置地图可视化"""
        self.ax_map.clear()
        
        # 绘制环境
        env = self.planner.environment
        
        # 绘制网格背景
        self.ax_map.add_patch(patches.Rectangle((0, 0), env.size, env.size,
                                              facecolor='lightgray', alpha=0.1))
        
        # 绘制障碍物
        obs_y, obs_x = np.where(env.obstacle_map)
        if len(obs_x) > 0:
            self.ax_map.scatter(obs_x, obs_y, c='darkred', s=3, alpha=0.8)
        
        # 绘制进出口边
        for edge in self.planner.edge_manager.edges:
            self.draw_intersection_edge(edge)
        
        self.ax_map.set_xlim(0, env.size)
        self.ax_map.set_ylim(0, env.size)
        self.ax_map.set_aspect('equal')
        self.ax_map.grid(True, alpha=0.3)
        self.ax_map.set_title(f'Lifelong MAPF - {env.map_name}')
    
    def draw_intersection_edge(self, edge: IntersectionEdge):
        """绘制进出口边"""
        # 颜色映射
        color_map = {
            "north": "red", "south": "blue", 
            "east": "green", "west": "orange"
        }
        color = color_map.get(edge.direction, "gray")
        
        # 绘制边界区域
        edge_points = edge.get_points()
        for x, y in edge_points:
            self.ax_map.add_patch(patches.Rectangle(
                (x-0.5, y-0.5), 1, 1, 
                facecolor=color, alpha=0.6, edgecolor='white', linewidth=1
            ))
        
        # 添加标签
        self.ax_map.text(edge.center_x, edge.center_y, edge.edge_id,
                        ha='center', va='center', fontsize=8, fontweight='bold',
                        color='white')
    
    def update_visualization(self, current_time: float):
        """更新可视化"""
        # 清除地图上的动态元素
        self.setup_map_visualization()
        
        # 绘制活跃车辆
        active_vehicles = self.planner.vehicle_manager.get_active_vehicles()
        
        for vehicle in active_vehicles:
            if vehicle.trajectory:
                self.draw_vehicle_trajectory(vehicle, current_time)
        
        # 绘制等待任务
        self.draw_pending_tasks()
        
        # 更新统计图表
        self.update_statistics_plot()
        self.update_timeline_plot(current_time)
        self.update_performance_plot()
        
        # 显示系统信息
        self.display_system_info(current_time)
    
    def draw_vehicle_trajectory(self, vehicle: LifelongVehicle, current_time: float):
        """绘制车辆轨迹"""
        trajectory = vehicle.trajectory
        if not trajectory:
            return
        
        # 绘制完整轨迹
        xs = [state.x for state in trajectory]
        ys = [state.y for state in trajectory]
        self.ax_map.plot(xs, ys, color=vehicle.color, alpha=0.6, linewidth=2)
        
        # 插值当前位置
        current_state = self.get_interpolated_state(trajectory, current_time)
        if current_state:
            self.draw_vehicle_at_state(current_state, vehicle.color)
            
            # 显示车辆ID
            self.ax_map.text(current_state.x + 1, current_state.y + 1, 
                           f'V{vehicle.vehicle_id}',
                           fontsize=8, fontweight='bold', color=vehicle.color)
    
    def draw_vehicle_at_state(self, state: VehicleState, color: str):
        """在指定状态绘制车辆"""
        length, width = self.planner.params.length, self.planner.params.width
        
        cos_theta, sin_theta = math.cos(state.theta), math.sin(state.theta)
        
        # 车辆矩形的四个角点
        corners = np.array([
            [-length/2, -width/2], [length/2, -width/2],
            [length/2, width/2], [-length/2, width/2],
            [-length/2, -width/2]
        ])
        
        # 旋转和平移
        rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_corners = corners @ rotation.T
        translated_corners = rotated_corners + np.array([state.x, state.y])
        
        # 绘制车辆
        vehicle_patch = patches.Polygon(translated_corners[:-1], 
                                      facecolor=color, alpha=0.8, 
                                      edgecolor='black', linewidth=1)
        self.ax_map.add_patch(vehicle_patch)
        
        # 绘制方向箭头
        arrow_length = 1.5
        dx = arrow_length * cos_theta
        dy = arrow_length * sin_theta
        self.ax_map.arrow(state.x, state.y, dx, dy, 
                         head_width=0.5, head_length=0.5,
                         fc=color, ec='black', alpha=0.9, linewidth=1)
    
    def draw_pending_tasks(self):
        """绘制等待中的任务"""
        for i, task in enumerate(self.planner.pending_tasks[:5]):  # 只显示前5个
            start_pos = task.start_position
            end_pos = task.end_position
            
            if start_pos and end_pos:
                # 绘制任务连线
                self.ax_map.plot([start_pos[0], end_pos[0]], 
                               [start_pos[1], end_pos[1]], 
                               'k--', alpha=0.5, linewidth=1)
                
                # 绘制起点
                self.ax_map.plot(start_pos[0], start_pos[1], 'go', markersize=8)
                self.ax_map.text(start_pos[0], start_pos[1] - 1, f'T{task.task_id}',
                               ha='center', fontsize=7, color='green', fontweight='bold')
                
                # 绘制终点
                self.ax_map.plot(end_pos[0], end_pos[1], 'rs', markersize=8)
    
    def get_interpolated_state(self, trajectory: List[VehicleState], target_time: float) -> Optional[VehicleState]:
        """获取插值状态"""
        if not trajectory:
            return None
        
        # 调整时间基准
        system_runtime = target_time - self.planner.system_start_time
        
        if system_runtime <= trajectory[0].t:
            return trajectory[0]
        elif system_runtime >= trajectory[-1].t:
            return trajectory[-1]
        
        # 时间插值
        for i in range(len(trajectory) - 1):
            if trajectory[i].t <= system_runtime <= trajectory[i+1].t:
                t1, t2 = trajectory[i].t, trajectory[i+1].t
                
                if abs(t2 - t1) < 1e-6:
                    return trajectory[i]
                
                alpha = (system_runtime - t1) / (t2 - t1)
                
                # 角度插值
                theta1, theta2 = trajectory[i].theta, trajectory[i+1].theta
                theta_diff = theta2 - theta1
                if theta_diff > math.pi:
                    theta_diff -= 2 * math.pi
                elif theta_diff < -math.pi:
                    theta_diff += 2 * math.pi
                interpolated_theta = theta1 + alpha * theta_diff
                
                return VehicleState(
                    x=trajectory[i].x + alpha * (trajectory[i+1].x - trajectory[i].x),
                    y=trajectory[i].y + alpha * (trajectory[i+1].y - trajectory[i].y),
                    theta=interpolated_theta,
                    v=trajectory[i].v + alpha * (trajectory[i+1].v - trajectory[i].v),
                    t=system_runtime
                )
        
        return None
    
    def update_statistics_plot(self):
        """更新统计图表"""
        self.ax_stats.clear()
        
        stats = self.planner.get_system_statistics()
        self.stats_history.append(stats)
        
        if len(self.stats_history) > 1:
            timestamps = [s['runtime_seconds'] for s in self.stats_history]
            active_counts = [s['vehicle_stats']['active_vehicles'] for s in self.stats_history]
            completed_counts = [s['vehicle_stats']['completed_vehicles'] for s in self.stats_history]
            
            self.ax_stats.plot(timestamps, active_counts, 'b-', label='活跃车辆', linewidth=2)
            self.ax_stats.plot(timestamps, completed_counts, 'g-', label='已完成', linewidth=2)
            
            self.ax_stats.set_xlabel('运行时间 (秒)')
            self.ax_stats.set_ylabel('车辆数量')
            self.ax_stats.set_title('车辆统计')
            self.ax_stats.legend()
            self.ax_stats.grid(True, alpha=0.3)
    
    def update_timeline_plot(self, current_time: float):
        """更新时间线图表"""
        self.ax_timeline.clear()
        
        active_vehicles = self.planner.vehicle_manager.get_active_vehicles()
        
        for i, vehicle in enumerate(active_vehicles[-20:]):  # 显示最近20个车辆
            y_pos = i
            spawn_time = vehicle.spawn_time - self.planner.system_start_time
            current_time_relative = current_time - self.planner.system_start_time
            
            # 绘制车辆生命周期
            self.ax_timeline.plot([spawn_time, current_time_relative], [y_pos, y_pos], 
                                color=vehicle.color, linewidth=4, alpha=0.7)
            
            # 标记当前位置
            self.ax_timeline.plot(current_time_relative, y_pos, 'o', 
                                color=vehicle.color, markersize=6)
            
            # 添加车辆ID
            self.ax_timeline.text(current_time_relative + 1, y_pos, f'V{vehicle.vehicle_id}',
                                fontsize=8, va='center')
        
        self.ax_timeline.set_xlabel('时间 (秒)')
        self.ax_timeline.set_ylabel('车辆')
        self.ax_timeline.set_title('车辆时间线')
        self.ax_timeline.grid(True, alpha=0.3)
    
    def update_performance_plot(self):
        """更新性能图表"""
        self.ax_performance.clear()
        
        if len(self.planner.performance_history) > 1:
            recent_performance = list(self.planner.performance_history)[-50:]  # 最近50次规划
            
            planning_times = [p['planning_time'] for p in recent_performance]
            success_flags = [1 if p['success'] else 0 for p in recent_performance]
            
            # 绘制规划时间
            self.ax_performance.plot(range(len(planning_times)), planning_times, 
                                   'b-', alpha=0.7, label='规划时间')
            
            # 绘制成功率（移动窗口）
            window_size = 10
            if len(success_flags) >= window_size:
                success_rates = []
                for i in range(window_size, len(success_flags) + 1):
                    window_success = sum(success_flags[i-window_size:i]) / window_size * 100
                    success_rates.append(window_success)
                
                self.ax_performance.plot(range(window_size-1, len(success_flags)), success_rates, 
                                       'g-', alpha=0.7, label='成功率 (%)')
            
            self.ax_performance.set_xlabel('规划序号')
            self.ax_performance.set_ylabel('时间 (秒) / 成功率 (%)')
            self.ax_performance.set_title('规划性能')
            self.ax_performance.legend()
            self.ax_performance.grid(True, alpha=0.3)
    
    def display_system_info(self, current_time: float):
        """显示系统信息"""
        stats = self.planner.get_system_statistics()
        
        info_text = f"运行时间: {stats['runtime_seconds']:.0f}s\n"
        info_text += f"活跃车辆: {stats['vehicle_stats']['active_vehicles']}\n"
        info_text += f"已完成: {stats['vehicle_stats']['completed_vehicles']}\n"
        info_text += f"成功率: {stats['vehicle_stats']['success_rate']:.1f}%\n"
        info_text += f"等待任务: {stats['system_load']['pending_tasks']}\n"
        info_text += f"规划成功率: {stats['planning_stats']['success_rate']:.1f}%"
        
        self.ax_map.text(0.02, 0.98, info_text, transform=self.ax_map.transAxes,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                        verticalalignment='top', fontsize=9)

def run_first_round_simulation(map_file: str, max_duration: int = 300, 
                              optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED):
    """运行第一轮多车规划仿真"""
    print(f"🚀 启动第一轮多车规划仿真")
    print(f"   地图文件: {map_file}")
    print(f"   最大时长: {max_duration} 秒") 
    print(f"   优化级别: {optimization_level.value}")
    print("=" * 60)
    
    # 创建规划器
    planner = LifelongPlanner(map_file, optimization_level)
    
    if len(planner.pending_tasks) == 0:
        print("❌ 没有生成任何任务，请检查地图文件")
        return None, {}
    
    # 立即分配第一批任务
    print(f"📋 开始分配第一轮任务...")
    planner.assign_tasks_to_vehicles()
    
    # 创建可视化器
    visualizer = LifelongVisualizer(planner)
    
    # 仿真参数
    update_interval = 0.5  # 更新间隔
    visualization_interval = 2.0  # 可视化更新间隔
    
    last_visualization_time = 0
    simulation_completed = False
    
    def update_simulation(frame):
        nonlocal last_visualization_time, simulation_completed
        
        current_time = time.time()
        
        # 检查是否完成
        if planner.is_round_completed():
            if not simulation_completed:
                print(f"🎉 第一轮任务全部完成！")
                simulation_completed = True
            return []
        
        # 检查超时
        runtime = current_time - planner.system_start_time
        if runtime > max_duration:
            if not simulation_completed:
                print(f"⏰ 达到最大仿真时长，强制结束")
                simulation_completed = True
            return []
        
        # 更新系统状态
        planner.update_system_state(current_time)
        
        # 定期更新可视化
        if current_time - last_visualization_time >= visualization_interval:
            visualizer.update_visualization(current_time)
            last_visualization_time = current_time
            
            # 输出进度信息
            stats = planner.get_system_statistics()
            round_progress = stats['round_progress']
            print(f"⏱️ 运行时间: {stats['runtime_seconds']:.0f}s | "
                  f"活跃: {round_progress['active_vehicles']} | "
                  f"已完成: {round_progress['completed_tasks']}/{round_progress['total_tasks']} | "
                  f"待分配: {round_progress['pending_tasks']} | "
                  f"完成率: {round_progress['completion_rate']:.1f}%")
        
        return []
    
    # 初始可视化设置
    visualizer.setup_map_visualization()
    
    # 创建动画
    estimated_frames = max_duration // update_interval
    anim = animation.FuncAnimation(visualizer.fig, update_simulation, 
                                 frames=int(estimated_frames * 1.2), interval=int(update_interval * 1000), 
                                 blit=False, repeat=False)
    
    # 保存动画
    print(f"🎬 开始仿真并保存动画...")
    try:
        writer = PillowWriter(fps=2)
        gif_filename = f"first_round_{planner.environment.map_name}_{optimization_level.value}.gif"
        anim.save(gif_filename, writer=writer)
        print(f"✅ 动画已保存: {gif_filename}")
    except Exception as e:
        print(f"⚠️ 动画保存失败: {str(e)}")
    
    plt.tight_layout()
    plt.show()
    
    # 最终统计
    final_stats = planner.get_system_statistics()
    round_progress = final_stats['round_progress']
    
    print(f"\n📊 第一轮最终统计:")
    print(f"   总运行时间: {final_stats['runtime_seconds']:.1f} 秒")
    print(f"   总任务数: {round_progress['total_tasks']}")
    print(f"   已完成任务: {round_progress['completed_tasks']}")
    print(f"   完成率: {round_progress['completion_rate']:.1f}%")
    print(f"   规划成功率: {final_stats['planning_stats']['success_rate']:.1f}%")
    print(f"   平均规划时间: {final_stats['planning_stats']['avg_planning_time']:.2f}s")
    
    if round_progress['completion_rate'] >= 80:
        print(f"🎊 第一轮任务基本完成！")
    elif round_progress['completion_rate'] >= 50:
        print(f"👍 第一轮任务部分完成")
    else:
        print(f"⚠️ 第一轮任务完成率较低，可能需要调整参数")
    
    return planner, final_stats

def main():
    """主函数"""
    print("🎯 第一轮多车轨迹规划系统")
    print("🔬 基于IEEE TITS论文的数学模型")
    print("📋 为每个出入口边生成一个任务，进行多车协同规划")
    print("=" * 60)
    
    # 查找可用的地图文件
    import os
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    lifelong_maps = [f for f in json_files if any(keyword in f.lower() 
                    for keyword in ['lifelong', 'intersection', 'cross', 'junction'])]
    
    if not lifelong_maps:
        print("❌ 未找到适合的地图文件")
        print("💡 请使用 lifelong_map.py 创建地图文件")
        return
    
    print(f"📁 发现 {len(lifelong_maps)} 个路口地图:")
    for i, map_file in enumerate(lifelong_maps):
        try:
            with open(map_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            map_info = data.get('map_info', {})
            name = map_info.get('name', map_file)
            edges_count = len(data.get('intersection_edges', []))
            print(f"   {i+1}. {map_file}")
            print(f"      名称: {name}")
            print(f"      出入口边: {edges_count} 个")
        except:
            print(f"   {i+1}. {map_file} (无法读取详细信息)")
    
    # 选择地图
    if len(lifelong_maps) == 1:
        selected_map = lifelong_maps[0]
        print(f"🎯 自动选择: {selected_map}")
    else:
        while True:
            try:
                choice = input(f"\n请选择地图 (1-{len(lifelong_maps)}) 或按Enter使用第1个: ").strip()
                if choice == "":
                    selected_map = lifelong_maps[0]
                    break
                
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(lifelong_maps):
                    selected_map = lifelong_maps[choice_idx]
                    break
                else:
                    print(f"❌ 请输入 1-{len(lifelong_maps)} 之间的数字")
            except ValueError:
                print("❌ 请输入有效的数字")
    
    # 选择优化级别
    print(f"\n⚙️ 优化级别:")
    print(f"   1. BASIC - 基础规划")
    print(f"   2. ENHANCED - 增强规划 (推荐)")
    print(f"   3. FULL - 完整QP优化 (需要CVXPY)")
    
    optimization_choice = input("请选择优化级别 (1-3) 或按Enter使用推荐级别: ").strip()
    
    if optimization_choice == "1":
        opt_level = OptimizationLevel.BASIC
    elif optimization_choice == "3":
        opt_level = OptimizationLevel.FULL
    else:
        opt_level = OptimizationLevel.ENHANCED
    
    # 最大仿真时长
    duration_input = input("最大仿真时长 (秒，默认300): ").strip()
    try:
        max_duration = int(duration_input) if duration_input else 300
    except ValueError:
        max_duration = 300
    
    print(f"\n🚀 启动第一轮仿真...")
    print(f"   地图: {selected_map}")
    print(f"   优化级别: {opt_level.value}")
    print(f"   最大时长: {max_duration} 秒")
    print(f"   任务模式: 每个出入口边一个任务")
    
    # 运行仿真
    try:
        planner, stats = run_first_round_simulation(selected_map, max_duration, opt_level)
        
        if planner:
            print(f"\n🎉 第一轮仿真完成!")
            
            # 询问是否继续下一轮
            if stats and stats['round_progress']['completion_rate'] >= 80:
                print(f"\n💡 第一轮成功率很高，可以考虑实现lifelong版本或RHCR增强!")
            
        else:
            print(f"\n❌ 仿真无法启动")
        
    except KeyboardInterrupt:
        print(f"\n⏹️ 仿真被用户中断")
    except Exception as e:
        print(f"\n💥 仿真异常: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()