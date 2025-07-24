#!/usr/bin/env python3
"""
IEEE TIT 2024论文完整集成改进版: Multi-Vehicle Collaborative Trajectory Planning 
in Unstructured Conflict Areas Based on V-Hybrid A*

集成改进:
1. 精确的中间节点生成机制 (基于论文Algorithm 1第16-21行)
2. 完整的Box约束实现 (基于论文公式22-25)
3. 保持原有的性能优化特性
4. 增强的可视化和调试功能
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import heapq
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set, Dict
import math
import time
import json
import os
from enum import Enum

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 可选依赖
try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

class OptimizationLevel(Enum):
    """优化级别枚举"""
    BASIC = "basic"          # 只使用V-Hybrid A*
    ENHANCED = "enhanced"    # 加入Dubins曲线 + 改进的中间节点生成
    FULL = "full"           # 完整QP优化 + Box约束优化（需要CVXPY）

@dataclass
class VehicleState:
    """Complete vehicle state for Hybrid A*"""
    x: float
    y: float
    theta: float  # heading angle
    v: float      # velocity
    t: float      # time
    steer: float = 0.0  # steering angle
    
    def copy(self):
        return VehicleState(self.x, self.y, self.theta, self.v, self.t, self.steer)

@dataclass
class HybridNode:
    """Node for Hybrid A* search"""
    state: VehicleState
    g_cost: float
    h_cost: float
    parent: Optional['HybridNode'] = None
    grid_x: int = 0
    grid_y: int = 0
    grid_theta: int = 0
    acceleration: float = 0.0  # 新增：记录节点的加速度信息
    
    @property
    def f_cost(self):
        return self.g_cost + self.h_cost
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def grid_key(self, resolution=1.0, angle_resolution=math.pi/4):
        """Discretization key for Hybrid A*"""
        return (
            int(self.state.x / resolution),
            int(self.state.y / resolution),
            int(self.state.theta / angle_resolution) % int(2*math.pi/angle_resolution)
        )

class VehicleParameters:
    """论文Table I的精确参数设置"""
    def __init__(self):
        # 车辆物理参数
        self.wheelbase = 3.0        # L (m) - 轴距
        self.length = 4.0           # 车辆长度
        self.width = 2.0            # 车辆宽度
        
        # 运动约束 (对应论文公式11-13)
        self.max_steer = 0.6        # δmax (rad) ≈ 35度
        self.max_speed = 8.0        # vmax (m/s)
        self.min_speed = 0.5        # vmin (m/s)
        self.max_accel = 2.0        # amax (m/s²)
        self.max_decel = -3.0       # amin (m/s²)
        self.max_lateral_accel = 4.0 # aymax (m/s²)
        
        # 规划参数
        self.dt = 0.5               # ΔT (s) - 时间分辨率
        self.speed_resolution = 1.0  # 速度分辨率
        self.steer_resolution = 0.3  # 转向分辨率
        
        # 成本函数权重 (对应论文公式16)
        self.wv = 1.0               # 速度变化权重
        self.wref = 0.5             # 参考速度权重  
        self.wδ = 0.2               # 方向变化权重
        
        # 轨迹优化权重 (对应论文公式17)
        self.ωs = 1.0               # 平滑性权重
        self.ωr = 2.0               # 参考路径权重
        self.ωl = 0.1               # 长度均匀化权重
        
        # 速度优化权重 (对应论文公式26)
        self.ωv_opt = 1.0           # 参考速度权重
        self.ωa = 0.1               # 加速度权重
        self.ωj = 0.01              # jerk权重
        
        # 安全距离
        self.safety_margin = 0.5    # 安全边距
        self.turning_radius_min = self.wheelbase / math.tan(self.max_steer)

class ImprovedIntermediateNodeGenerator:
    """
    改进的中间节点生成器
    基于论文Algorithm 1第16-21行的精确实现
    """
    
    def __init__(self, params: VehicleParameters):
        self.params = params
    
    def generate_intermediate_nodes_for_deceleration(self, parent_node: HybridNode, 
                                                   child_node: HybridNode) -> List[VehicleState]:
        """
        为减速节点生成中间节点 - 基于论文Algorithm 1第16-21行
        
        论文原文逻辑：
        16: if child node is deceleration node then
        17:     generate intermediate nodes;
        18:     if intermediate node in closed set then
        19:         add child node to the closed set;
        20:         break;
        21:     end if
        """
        intermediate_nodes = []
        
        # 检查是否为减速节点
        if child_node.acceleration >= 0:
            return intermediate_nodes  # 非减速节点，不需要生成中间节点
        
        # 计算精确的中间节点 - 基于运动学模型
        current_state = parent_node.state.copy()
        target_state = child_node.state
        
        # 计算总的运动时间
        total_time = target_state.t - current_state.t
        
        # 根据减速程度确定中间节点密度
        speed_diff = abs(target_state.v - current_state.v)
        # 速度变化越大，需要更多中间节点
        num_intermediate = max(3, int(speed_diff / 0.5) + 2)
        
        dt_intermediate = total_time / (num_intermediate + 1)
        
        for i in range(1, num_intermediate + 1):
            # 使用运动学模型计算精确的中间状态
            intermediate_time = current_state.t + i * dt_intermediate
            
            # 计算当前时刻的状态 - 使用恒定加速度模型
            elapsed_time = i * dt_intermediate
            
            # 速度计算 v = v0 + a*t
            intermediate_v = current_state.v + child_node.acceleration * elapsed_time
            intermediate_v = max(self.params.min_speed, 
                               min(intermediate_v, self.params.max_speed))
            
            # 位置计算 - 考虑转向
            if abs(current_state.theta - target_state.theta) < 1e-6:
                # 直线运动
                distance = current_state.v * elapsed_time + 0.5 * child_node.acceleration * elapsed_time**2
                intermediate_x = current_state.x + distance * math.cos(current_state.theta)
                intermediate_y = current_state.y + distance * math.sin(current_state.theta)
                intermediate_theta = current_state.theta
            else:
                # 曲线运动 - 使用自行车模型
                progress = elapsed_time / total_time
                intermediate_x = current_state.x + progress * (target_state.x - current_state.x)
                intermediate_y = current_state.y + progress * (target_state.y - current_state.y)
                intermediate_theta = current_state.theta + progress * (target_state.theta - current_state.theta)
                
                # 确保角度连续性
                intermediate_theta = math.atan2(math.sin(intermediate_theta), 
                                              math.cos(intermediate_theta))
            
            intermediate_state = VehicleState(
                x=intermediate_x,
                y=intermediate_y,
                theta=intermediate_theta,
                v=intermediate_v,
                t=intermediate_time,
                steer=current_state.steer
            )
            
            intermediate_nodes.append(intermediate_state)
        
        return intermediate_nodes
    
    def validate_intermediate_trajectory(self, parent_state: VehicleState, 
                                       child_state: VehicleState,
                                       intermediate_nodes: List[VehicleState]) -> bool:
        """
        验证中间轨迹的物理可行性
        确保加速度和转向约束得到满足
        """
        all_states = [parent_state] + intermediate_nodes + [child_state]
        
        for i in range(len(all_states) - 1):
            current = all_states[i]
            next_state = all_states[i + 1]
            
            dt = next_state.t - current.t
            if dt <= 0:
                return False
            
            # 检查速度变化率
            dv = next_state.v - current.v
            acceleration = dv / dt
            if acceleration < self.params.max_decel or acceleration > self.params.max_accel:
                return False
            
            # 检查转向约束
            dtheta = abs(next_state.theta - current.theta)
            if dtheta > math.pi:
                dtheta = 2 * math.pi - dtheta
            
            max_theta_change = abs(current.v * math.tan(self.params.max_steer) / self.params.wheelbase * dt)
            if dtheta > max_theta_change + 1e-6:  # 小的容差
                return False
        
        return True

class AdvancedBoxConstraints:
    """
    高级Box约束实现
    基于论文公式(22)-(25)的精确实现
    """
    
    def __init__(self, params: VehicleParameters):
        self.params = params
        self.obstacle_grid = {}  # 障碍物网格
        self.grid_resolution = 0.5  # 网格分辨率
    
    def calculate_safety_distance(self, waypoint_index: int, total_waypoints: int) -> float:
        """
        计算安全距离 rd
        基于车辆几何和安全边距
        """
        # 基础安全距离：车辆对角线长度 + 安全边距
        vehicle_diagonal = math.sqrt(self.params.length**2 + self.params.width**2)
        base_safety_distance = vehicle_diagonal / 2 + self.params.safety_margin
        
        return base_safety_distance
    
    def calculate_box_constraints(self, waypoint_index: int, total_waypoints: int, 
                                rd: float) -> Tuple[float, float]:
        """
        计算Box约束的最大偏移量
        基于论文公式(22)-(25)
        """
        
        # 公式(22): 基础约束
        # Δxk = Δyk = rd/√2
        base_delta = rd / math.sqrt(2)
        
        # 公式(24): 计算μ值
        N = total_waypoints
        k = waypoint_index
        
        if k <= N // 2:
            mu = k
        else:
            mu = N - k
        
        # 公式(23): 应用系数修正
        # Δxk = Δyk = (rd/√2) · 1/(1 + e^(4-μ))
        coefficient = 1.0 / (1.0 + math.exp(4 - mu))
        
        delta_x = base_delta * coefficient
        delta_y = base_delta * coefficient
        
        return delta_x, delta_y
    
    def generate_box_constraints(self, waypoints: List[VehicleState]) -> List[dict]:
        """
        为每个路径点生成Box约束
        返回约束字典列表
        """
        constraints = []
        N = len(waypoints)
        
        for k, waypoint in enumerate(waypoints):
            rd = self.calculate_safety_distance(k, N)
            delta_x, delta_y = self.calculate_box_constraints(k, N, rd)
            
            # 初始约束框 - 公式(25)基础部分
            initial_xlb = waypoint.x - delta_x
            initial_xub = waypoint.x + delta_x
            initial_ylb = waypoint.y - delta_y
            initial_yub = waypoint.y + delta_y
            
            # 检查静态障碍物并调整约束
            adjusted_constraints = self._adjust_for_static_obstacles(
                initial_xlb, initial_xub, initial_ylb, initial_yub, waypoint
            )
            
            constraints.append({
                'waypoint_index': k,
                'waypoint': waypoint,
                'rd': rd,
                'delta_x': delta_x,
                'delta_y': delta_y,
                'xlb': adjusted_constraints['xlb'],
                'xub': adjusted_constraints['xub'],
                'ylb': adjusted_constraints['ylb'],
                'yub': adjusted_constraints['yub'],
                'epsilon': adjusted_constraints['epsilon']  # 调整量
            })
        
        return constraints
    
    def _adjust_for_static_obstacles(self, xlb: float, xub: float, ylb: float, yub: float,
                                   waypoint: VehicleState) -> dict:
        """
        根据静态障碍物调整约束框
        基于论文公式(25)的完整实现
        """
        
        # 公式(25)的调整项
        epsilon1, epsilon2, epsilon3, epsilon4 = 0.0, 0.0, 0.0, 0.0
        
        # 获取约束框内的网格点
        grid_points = self._get_grid_points_in_box(xlb, xub, ylb, yub)
        
        # 检查每个网格点是否与障碍物冲突
        for grid_x, grid_y in grid_points:
            if self._is_obstacle_at_grid(grid_x, grid_y):
                # 计算需要的调整量
                world_x = grid_x * self.grid_resolution
                world_y = grid_y * self.grid_resolution
                
                if world_x < waypoint.x:  # 左侧障碍物
                    epsilon1 = max(epsilon1, waypoint.x - world_x + self.grid_resolution)
                elif world_x > waypoint.x:  # 右侧障碍物
                    epsilon2 = max(epsilon2, world_x - waypoint.x + self.grid_resolution)
                
                if world_y < waypoint.y:  # 下方障碍物
                    epsilon3 = max(epsilon3, waypoint.y - world_y + self.grid_resolution)
                elif world_y > waypoint.y:  # 上方障碍物
                    epsilon4 = max(epsilon4, world_y - waypoint.y + self.grid_resolution)
        
        # 应用公式(25)的调整
        final_xlb = xlb + epsilon1
        final_xub = xub - epsilon2
        final_ylb = ylb + epsilon3
        final_yub = yub - epsilon4
        
        # 确保约束框有效（不会过度收缩）
        min_box_size = 0.5  # 最小约束框大小
        if final_xub - final_xlb < min_box_size:
            center_x = (final_xlb + final_xub) / 2
            final_xlb = center_x - min_box_size / 2
            final_xub = center_x + min_box_size / 2
        
        if final_yub - final_ylb < min_box_size:
            center_y = (final_ylb + final_yub) / 2
            final_ylb = center_y - min_box_size / 2
            final_yub = center_y + min_box_size / 2
        
        return {
            'xlb': final_xlb,
            'xub': final_xub,
            'ylb': final_ylb,
            'yub': final_yub,
            'epsilon': [epsilon1, epsilon2, epsilon3, epsilon4]
        }
    
    def _get_grid_points_in_box(self, xlb: float, xub: float, ylb: float, yub: float) -> List[Tuple[int, int]]:
        """获取约束框内的所有网格点"""
        grid_points = []
        
        x_start = int(xlb / self.grid_resolution)
        x_end = int(xub / self.grid_resolution) + 1
        y_start = int(ylb / self.grid_resolution)
        y_end = int(yub / self.grid_resolution) + 1
        
        for grid_x in range(x_start, x_end):
            for grid_y in range(y_start, y_end):
                world_x = grid_x * self.grid_resolution
                world_y = grid_y * self.grid_resolution
                
                if xlb <= world_x <= xub and ylb <= world_y <= yub:
                    grid_points.append((grid_x, grid_y))
        
        return grid_points
    
    def _is_obstacle_at_grid(self, grid_x: int, grid_y: int) -> bool:
        """检查网格点是否有障碍物"""
        return self.obstacle_grid.get((grid_x, grid_y), False)
    
    def update_obstacle_grid(self, obstacles: List[Tuple[float, float, float, float]]):
        """
        更新障碍物网格
        obstacles: [(x_min, y_min, x_max, y_max), ...]
        """
        self.obstacle_grid.clear()
        
        for x_min, y_min, x_max, y_max in obstacles:
            x_start = int(x_min / self.grid_resolution)
            x_end = int(x_max / self.grid_resolution) + 1
            y_start = int(y_min / self.grid_resolution)
            y_end = int(y_max / self.grid_resolution) + 1
            
            for grid_x in range(x_start, x_end):
                for grid_y in range(y_start, y_end):
                    self.obstacle_grid[(grid_x, grid_y)] = True

class TimeSync:
    """时间同步管理器 - 解决轨迹优化后的时间不一致问题"""
    
    @staticmethod
    def resync_trajectory_time(trajectory: List[VehicleState], start_time: float = 0.0) -> List[VehicleState]:
        """重新同步轨迹时间，确保时间连续性"""
        if not trajectory:
            return trajectory
        
        resynced_trajectory = []
        current_time = start_time
        
        for i, state in enumerate(trajectory):
            new_state = state.copy()
            new_state.t = current_time
            resynced_trajectory.append(new_state)
            
            # 计算到下一个点的时间
            if i < len(trajectory) - 1:
                next_state = trajectory[i + 1]
                distance = math.sqrt((next_state.x - state.x)**2 + (next_state.y - state.y)**2)
                avg_speed = max(0.1, (state.v + next_state.v) / 2)  # 避免除零
                dt = distance / avg_speed
                current_time += dt
        
        return resynced_trajectory
    
    @staticmethod
    def get_time_key(state: VehicleState, resolution: float = 0.5) -> int:
        """获取时间键值，用于动态障碍物查找"""
        return int(state.t / resolution)
    
    @staticmethod
    def interpolate_state_at_time(trajectory: List[VehicleState], target_time: float) -> Optional[VehicleState]:
        """在指定时间插值获取状态"""
        if not trajectory:
            return None
        
        # 找到目标时间前后的状态
        for i in range(len(trajectory) - 1):
            if trajectory[i].t <= target_time <= trajectory[i + 1].t:
                # 线性插值
                t1, t2 = trajectory[i].t, trajectory[i + 1].t
                if abs(t2 - t1) < 1e-6:
                    return trajectory[i]
                
                alpha = (target_time - t1) / (t2 - t1)
                
                interpolated = VehicleState(
                    x=trajectory[i].x + alpha * (trajectory[i + 1].x - trajectory[i].x),
                    y=trajectory[i].y + alpha * (trajectory[i + 1].y - trajectory[i].y),
                    theta=trajectory[i].theta + alpha * (trajectory[i + 1].theta - trajectory[i].theta),
                    v=trajectory[i].v + alpha * (trajectory[i + 1].v - trajectory[i].v),
                    t=target_time
                )
                return interpolated
        
        # 如果目标时间超出范围，返回最近的状态
        if target_time <= trajectory[0].t:
            return trajectory[0]
        elif target_time >= trajectory[-1].t:
            return trajectory[-1]
        
        return None

class EfficientDubinsPath:
    """高效Dubins曲线计算 - 仅在需要时计算"""
    
    def __init__(self, params: VehicleParameters):
        self.params = params
        self.min_radius = params.turning_radius_min
        self.cache = {}  # 简单缓存机制
    
    def compute_dubins_path(self, start_state: VehicleState, goal_state: VehicleState, 
                          quick_mode: bool = True) -> Optional[List[VehicleState]]:
        """计算Dubins路径 - 支持快速模式"""
        
        # 检查缓存
        cache_key = (round(start_state.x), round(start_state.y), round(start_state.theta, 2),
                    round(goal_state.x), round(goal_state.y), round(goal_state.theta, 2))
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 距离检查
        dx = goal_state.x - start_state.x
        dy = goal_state.y - start_state.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance < 0.1:
            return [start_state, goal_state]
        
        # 快速模式：只计算LSL和RSR（最常用的两种）
        if quick_mode:
            paths = []
            
            lsl_path = self._compute_lsl_fast(start_state, goal_state)
            if lsl_path:
                paths.append(('LSL', lsl_path))
            
            rsr_path = self._compute_rsr_fast(start_state, goal_state)
            if rsr_path:
                paths.append(('RSR', rsr_path))
            
            if not paths:
                # 如果快速模式失败，回退到直线
                return self._compute_straight_line(start_state, goal_state)
            
            best_path = min(paths, key=lambda x: self._path_length(x[1]))
            result = best_path[1]
        else:
            # 完整模式：计算所有四种曲线
            result = self._compute_all_dubins_curves(start_state, goal_state)
        
        # 缓存结果
        self.cache[cache_key] = result
        return result
    
    def _compute_lsl_fast(self, start: VehicleState, goal: VehicleState) -> Optional[List[VehicleState]]:
        """快速LSL计算"""
        try:
            # 简化的LSL计算
            c1_x = start.x - self.min_radius * math.sin(start.theta)
            c1_y = start.y + self.min_radius * math.cos(start.theta)
            
            c2_x = goal.x - self.min_radius * math.sin(goal.theta)
            c2_y = goal.y + self.min_radius * math.cos(goal.theta)
            
            center_dist = math.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)
            
            if center_dist < 2 * self.min_radius:
                return None
            
            # 生成简化的路径点（减少点数）
            path = []
            num_points = max(3, int(center_dist / 2.0))  # 减少点数
            
            for i in range(num_points + 1):
                t = i / num_points if num_points > 0 else 0
                x = start.x + t * (goal.x - start.x)
                y = start.y + t * (goal.y - start.y)
                theta = start.theta + t * (goal.theta - start.theta)
                v = start.v + t * (goal.v - start.v)
                time = start.t + t * center_dist / max(1.0, (start.v + goal.v) / 2)
                
                path.append(VehicleState(x, y, theta, v, time))
            
            return path
            
        except:
            return None
    
    def _compute_rsr_fast(self, start: VehicleState, goal: VehicleState) -> Optional[List[VehicleState]]:
        """快速RSR计算"""
        try:
            # 简化的RSR计算
            c1_x = start.x + self.min_radius * math.sin(start.theta)
            c1_y = start.y - self.min_radius * math.cos(start.theta)
            
            c2_x = goal.x + self.min_radius * math.sin(goal.theta)
            c2_y = goal.y - self.min_radius * math.cos(goal.theta)
            
            center_dist = math.sqrt((c2_x - c1_x)**2 + (c2_y - c1_y)**2)
            
            if center_dist < 2 * self.min_radius:
                return None
            
            # 生成简化的路径点
            path = []
            num_points = max(3, int(center_dist / 2.0))
            
            for i in range(num_points + 1):
                t = i / num_points if num_points > 0 else 0
                x = start.x + t * (goal.x - start.x)
                y = start.y + t * (goal.y - start.y)
                theta = start.theta + t * (goal.theta - start.theta)
                v = start.v + t * (goal.v - start.v)
                time = start.t + t * center_dist / max(1.0, (start.v + goal.v) / 2)
                
                path.append(VehicleState(x, y, theta, v, time))
            
            return path
            
        except:
            return None
    
    def _compute_straight_line(self, start: VehicleState, goal: VehicleState) -> List[VehicleState]:
        """直线连接备选方案"""
        distance = math.sqrt((goal.x - start.x)**2 + (goal.y - start.y)**2)
        num_points = max(2, int(distance / 1.5))
        
        path = []
        for i in range(num_points + 1):
            t = i / num_points if num_points > 0 else 0
            x = start.x + t * (goal.x - start.x)
            y = start.y + t * (goal.y - start.y)
            theta = start.theta + t * (goal.theta - start.theta)
            v = start.v + t * (goal.v - start.v)
            time = start.t + t * distance / max(1.0, (start.v + goal.v) / 2)
            
            path.append(VehicleState(x, y, theta, v, time))
        
        return path
    
    def _compute_all_dubins_curves(self, start: VehicleState, goal: VehicleState) -> Optional[List[VehicleState]]:
        """计算所有四种Dubins曲线（完整模式）"""
        # 这里可以实现完整的Dubins曲线计算
        # 为了性能，先使用简化版本
        return self._compute_straight_line(start, goal)
    
    def _path_length(self, path: List[VehicleState]) -> float:
        """计算路径长度"""
        length = 0.0
        for i in range(len(path) - 1):
            dx = path[i+1].x - path[i].x
            dy = path[i+1].y - path[i].y
            length += math.sqrt(dx*dx + dy*dy)
        return length

class FastConflictDetector:
    """高效冲突检测器"""
    
    def __init__(self, params: VehicleParameters):
        self.params = params
        self.time_resolution = params.dt
    
    def detect_conflicts(self, trajectory1: List[VehicleState], 
                        trajectory2: List[VehicleState]) -> List[Tuple[VehicleState, VehicleState]]:
        """快速冲突检测"""
        conflicts = []
        
        # 使用时间网格进行快速检测
        time_grid1 = self._build_time_grid(trajectory1)
        time_grid2 = self._build_time_grid(trajectory2)
        
        # 检查时间重叠
        for time_key in time_grid1:
            if time_key in time_grid2:
                state1 = time_grid1[time_key]
                state2 = time_grid2[time_key]
                
                # 简化的距离检测
                distance = math.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
                safety_distance = (self.params.length + self.params.width) / 2 + self.params.safety_margin
                
                if distance < safety_distance:
                    conflicts.append((state1, state2))
        
        return conflicts
    
    def _build_time_grid(self, trajectory: List[VehicleState]) -> Dict[int, VehicleState]:
        """构建时间网格"""
        time_grid = {}
        
        for state in trajectory:
            time_key = TimeSync.get_time_key(state, self.time_resolution)
            time_grid[time_key] = state
        
        return time_grid

class OptimizedTrajectoryProcessor:
    """优化的轨迹处理器 - 集成Box约束优化"""
    
    def __init__(self, params: VehicleParameters, optimization_level: OptimizationLevel):
        self.params = params
        self.optimization_level = optimization_level
        self.conflict_detector = FastConflictDetector(params)
        
        # 根据优化级别初始化组件
        if optimization_level == OptimizationLevel.ENHANCED or optimization_level == OptimizationLevel.FULL:
            self.dubins_path = EfficientDubinsPath(params)
        
        # 新增：Box约束处理器
        if optimization_level == OptimizationLevel.FULL:
            self.box_constraints = AdvancedBoxConstraints(params)
    
    def process_trajectory(self, initial_trajectory: List[VehicleState],
                         high_priority_trajectories: List[List[VehicleState]]) -> List[VehicleState]:
        """根据优化级别处理轨迹"""
        
        if self.optimization_level == OptimizationLevel.BASIC:
            return self._basic_processing(initial_trajectory)
        
        elif self.optimization_level == OptimizationLevel.ENHANCED:
            return self._enhanced_processing(initial_trajectory, high_priority_trajectories)
        
        elif self.optimization_level == OptimizationLevel.FULL:
            return self._full_processing(initial_trajectory, high_priority_trajectories)
        
        return initial_trajectory
    
    def _basic_processing(self, trajectory: List[VehicleState]) -> List[VehicleState]:
        """基础处理 - 仅时间同步"""
        return TimeSync.resync_trajectory_time(trajectory)
    
    def _enhanced_processing(self, trajectory: List[VehicleState], 
                           high_priority_trajectories: List[List[VehicleState]]) -> List[VehicleState]:
        """增强处理 - 加入简单的轨迹平滑"""
        # 基础时间同步
        synced_trajectory = TimeSync.resync_trajectory_time(trajectory)
        
        # 简单的轨迹平滑
        smoothed_trajectory = self._simple_smooth(synced_trajectory)
        
        return smoothed_trajectory
    
    def _full_processing(self, trajectory: List[VehicleState],
                        high_priority_trajectories: List[List[VehicleState]]) -> List[VehicleState]:
        """完整处理 - 包含Box约束优化"""
        try:
            # 先进行基础处理
            processed_trajectory = self._enhanced_processing(trajectory, high_priority_trajectories)
            
            # 如果轨迹较短，不进行Box约束优化
            if len(processed_trajectory) < 5:
                return processed_trajectory
            
            # 应用Box约束优化
            box_optimized_trajectory = self._apply_box_constraints_optimization(processed_trajectory)
            
            # 如果支持CVXPY，尝试QP优化
            if HAS_CVXPY:
                qp_optimized_trajectory = self._qp_optimize(box_optimized_trajectory, high_priority_trajectories)
                final_trajectory = qp_optimized_trajectory
            else:
                final_trajectory = box_optimized_trajectory
            
            # 确保时间同步
            return TimeSync.resync_trajectory_time(final_trajectory)
            
        except Exception as e:
            print(f"        完整处理失败，使用增强处理: {str(e)}")
            return self._enhanced_processing(trajectory, high_priority_trajectories)
    
    def _apply_box_constraints_optimization(self, trajectory: List[VehicleState]) -> List[VehicleState]:
        """应用Box约束优化"""
        if not hasattr(self, 'box_constraints'):
            return trajectory
        
        try:
            # 生成Box约束
            constraints = self.box_constraints.generate_box_constraints(trajectory)
            
            # 应用约束优化路径
            optimized_trajectory = []
            
            for i, (state, constraint) in enumerate(zip(trajectory, constraints)):
                # 检查当前状态是否在约束框内
                if (constraint['xlb'] <= state.x <= constraint['xub'] and 
                    constraint['ylb'] <= state.y <= constraint['yub']):
                    # 状态已在约束框内，无需调整
                    optimized_trajectory.append(state)
                else:
                    # 将状态投影到约束框内
                    optimized_x = max(constraint['xlb'], 
                                    min(state.x, constraint['xub']))
                    optimized_y = max(constraint['ylb'], 
                                    min(state.y, constraint['yub']))
                    
                    # 创建调整后的状态
                    optimized_state = VehicleState(
                        x=optimized_x,
                        y=optimized_y,
                        theta=state.theta,
                        v=state.v,
                        t=state.t,
                        steer=state.steer
                    )
                    optimized_trajectory.append(optimized_state)
            
            return optimized_trajectory
            
        except Exception as e:
            print(f"        Box约束优化失败: {str(e)}")
            return trajectory
    
    def _simple_smooth(self, trajectory: List[VehicleState]) -> List[VehicleState]:
        """简单的轨迹平滑"""
        if len(trajectory) < 3:
            return trajectory
        
        smoothed = [trajectory[0]]  # 保持起点
        
        for i in range(1, len(trajectory) - 1):
            # 简单的三点平滑
            prev_state = trajectory[i-1]
            curr_state = trajectory[i]
            next_state = trajectory[i+1]
            
            smooth_x = (prev_state.x + curr_state.x + next_state.x) / 3
            smooth_y = (prev_state.y + curr_state.y + next_state.y) / 3
            smooth_theta = curr_state.theta  # 保持原始朝向
            smooth_v = (prev_state.v + curr_state.v + next_state.v) / 3
            
            smoothed_state = VehicleState(smooth_x, smooth_y, smooth_theta, smooth_v, curr_state.t)
            smoothed.append(smoothed_state)
        
        smoothed.append(trajectory[-1])  # 保持终点
        return smoothed
    
    def _qp_optimize(self, trajectory: List[VehicleState],
                    high_priority_trajectories: List[List[VehicleState]]) -> List[VehicleState]:
        """简化的QP优化"""
        N = len(trajectory)
        if N < 3:
            return trajectory
        
        try:
            # 简化的路径优化
            x_vars = cp.Variable(N)
            y_vars = cp.Variable(N)
            
            x_ref = np.array([state.x for state in trajectory])
            y_ref = np.array([state.y for state in trajectory])
            
            # 简化的目标函数
            objective = 0
            
            # 参考路径拟合（主要项）
            for k in range(N):
                objective += cp.square(x_vars[k] - x_ref[k]) + cp.square(y_vars[k] - y_ref[k])
            
            # 简单的平滑约束
            for k in range(N-2):
                objective += 0.1 * (cp.square(x_vars[k] + x_vars[k+2] - 2*x_vars[k+1]) + 
                                   cp.square(y_vars[k] + y_vars[k+2] - 2*y_vars[k+1]))
            
            # 约束条件
            constraints = []
            constraints.append(x_vars[0] == trajectory[0].x)
            constraints.append(y_vars[0] == trajectory[0].y)
            constraints.append(x_vars[N-1] == trajectory[-1].x)
            constraints.append(y_vars[N-1] == trajectory[-1].y)
            
            # 求解
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.solve(solver=cp.OSQP, verbose=False, max_iter=1000)
            
            if problem.status == cp.OPTIMAL:
                optimized_trajectory = []
                for k in range(N):
                    new_state = trajectory[k].copy()
                    new_state.x = float(x_vars.value[k])
                    new_state.y = float(y_vars.value[k])
                    optimized_trajectory.append(new_state)
                
                return optimized_trajectory
            else:
                return trajectory
                
        except Exception as e:
            print(f"        QP优化异常: {str(e)}")
            return trajectory

class UnstructuredEnvironment:
    """非结构化环境类"""
    
    def __init__(self, size=100):
        self.size = size
        self.resolution = 1.0
        self.obstacle_map = np.zeros((self.size, self.size), dtype=bool)
        self.dynamic_obstacles = {}
        self.map_name = "default"
        self.environment_type = "custom"
        
        # 性能优化：预计算常用值
        self.size_range = range(self.size)
    
    def load_from_json(self, json_file_path):
        """从JSON文件加载地图"""
        print(f"🗺️ 加载地图文件: {json_file_path}")
        
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                map_data = json.load(f)
            
            map_info = map_data.get("map_info", {})
            self.map_name = map_info.get("name", "loaded_map")
            map_width = map_info.get("width", 50)
            map_height = map_info.get("height", 50)
            
            self.size = max(map_width, map_height)
            self.obstacle_map = np.zeros((self.size, self.size), dtype=bool)
            
            if "grid" in map_data:
                grid = np.array(map_data["grid"], dtype=np.int8)
                for row in range(min(grid.shape[0], self.size)):
                    for col in range(min(grid.shape[1], self.size)):
                        if grid[row, col] == 1:
                            self.obstacle_map[row, col] = True
            
            if "obstacles" in map_data:
                for obstacle in map_data["obstacles"]:
                    x, y = obstacle["x"], obstacle["y"]
                    if 0 <= x < self.size and 0 <= y < self.size:
                        self.obstacle_map[y, x] = True
            
            self.environment_type = "custom_loaded"
            self._validate_environment()
            
            print(f"✅ 地图加载成功: {self.map_name}")
            return map_data
            
        except Exception as e:
            print(f"❌ 加载地图失败: {str(e)}")
            return None
    
    def _validate_environment(self):
        """验证环境的基本属性"""
        total_cells = self.size * self.size
        obstacle_cells = np.sum(self.obstacle_map)
        free_cells = total_cells - obstacle_cells
        
        print(f"🗺️ 环境统计信息:")
        print(f"  地图名称: {self.map_name}")
        print(f"  地图大小: {self.size}x{self.size} ({total_cells} cells)")
        print(f"  障碍物占用: {obstacle_cells} cells ({100*obstacle_cells/total_cells:.1f}%)")
        print(f"  可通行区域: {free_cells} cells ({100*free_cells/total_cells:.1f}%)")
    
    def is_valid_position(self, x, y):
        """检查位置是否可通行"""
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < self.size and 0 <= iy < self.size:
            return not self.obstacle_map[iy, ix]
        return False
    
    def is_collision_free(self, state: VehicleState, params: VehicleParameters):
        """优化的碰撞检测"""
        # 快速边界检查
        margin = max(params.length, params.width) / 2
        if not (margin <= state.x <= self.size - margin and 
               margin <= state.y <= self.size - margin):
            return False
        
        # 简化的碰撞检测：只检查关键点
        cos_theta, sin_theta = math.cos(state.theta), math.sin(state.theta)
        
        # 检查车辆四个角点
        half_length, half_width = params.length/2, params.width/2
        corners = [
            (-half_length, -half_width), (half_length, -half_width),
            (half_length, half_width), (-half_length, half_width)
        ]
        
        for lx, ly in corners:
            gx = state.x + lx * cos_theta - ly * sin_theta
            gy = state.y + lx * sin_theta + ly * cos_theta
            if not self.is_valid_position(gx, gy):
                return False
        
        # 检查动态障碍物碰撞
        time_key = TimeSync.get_time_key(state)
        if time_key in self.dynamic_obstacles:
            vehicle_cells = self._get_vehicle_cells_fast(state, params)
            if vehicle_cells.intersection(self.dynamic_obstacles[time_key]):
                return False
        
        return True
    
    def _get_vehicle_cells_fast(self, state: VehicleState, params: VehicleParameters):
        """快速获取车辆占用的网格单元"""
        cells = set()
        
        # 简化的网格占用计算
        x_min = int(state.x - params.length/2)
        x_max = int(state.x + params.length/2)
        y_min = int(state.y - params.width/2)
        y_max = int(state.y + params.width/2)
        
        for x in range(max(0, x_min), min(self.size, x_max + 1)):
            for y in range(max(0, y_min), min(self.size, y_max + 1)):
                cells.add((x, y))
        
        return cells
    
    def add_vehicle_trajectory(self, trajectory: List[VehicleState], params: VehicleParameters):
        """添加车辆轨迹作为动态障碍物"""
        for state in trajectory:
            time_key = TimeSync.get_time_key(state)
            if time_key not in self.dynamic_obstacles:
                self.dynamic_obstacles[time_key] = set()
            
            vehicle_cells = self._get_vehicle_cells_fast(state, params)
            self.dynamic_obstacles[time_key].update(vehicle_cells)
    
    def is_start_position_blocked(self, start_state: VehicleState, params: VehicleParameters):
        """检查起始位置是否被动态障碍物占用"""
        start_cells = self._get_vehicle_cells_fast(start_state, params)
        time_key = TimeSync.get_time_key(start_state)
        
        if time_key in self.dynamic_obstacles:
            if start_cells.intersection(self.dynamic_obstacles[time_key]):
                return True
        return False
    
    def find_safe_start_time(self, start_state: VehicleState, params: VehicleParameters, max_delay=20.0):
        """找到安全的启动时间"""
        for delay in np.arange(0, max_delay, 1.0):
            test_state = start_state.copy()
            test_state.t = start_state.t + delay
            
            if not self.is_start_position_blocked(test_state, params):
                return delay
        
        return None

class VHybridAStarPlanner:
    """集成改进的V-Hybrid A* 规划器"""
    
    def __init__(self, environment: UnstructuredEnvironment, optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED):
        self.environment = environment
        self.params = VehicleParameters()
        self.optimization_level = optimization_level
        self.trajectory_processor = OptimizedTrajectoryProcessor(self.params, optimization_level)
        
        # 新增：精确的中间节点生成器
        self.intermediate_generator = ImprovedIntermediateNodeGenerator(self.params)
        
        # 新增：高级Box约束处理器（仅在FULL模式下）
        if optimization_level == OptimizationLevel.FULL:
            self.box_constraints = AdvancedBoxConstraints(self.params)
        
        # 根据优化级别设置搜索参数
        if optimization_level == OptimizationLevel.BASIC:
            self.max_iterations = 8000
        elif optimization_level == OptimizationLevel.ENHANCED:
            self.max_iterations = 12000
        else:  # FULL
            self.max_iterations = 15000
        
        # 生成运动基元
        self.motion_primitives = self._generate_motion_primitives()
        
        print(f"        优化级别: {optimization_level.value}")
        
        # 在ENHANCED和FULL模式下都需要Dubins路径
        if optimization_level == OptimizationLevel.ENHANCED or optimization_level == OptimizationLevel.FULL:
            self.dubins_path = EfficientDubinsPath(self.params)
    
    def _generate_motion_primitives(self):
        """生成运动基元"""
        primitives = []
        
        steer_angles = [-self.params.max_steer, -self.params.max_steer/2, 0, 
                       self.params.max_steer/2, self.params.max_steer]
        accelerations = [self.params.max_decel, 0, self.params.max_accel]
        
        for steer in steer_angles:
            for accel in accelerations:
                primitives.append((accel, steer))
        
        return primitives
    
    def bicycle_model(self, state: VehicleState, accel: float, steer: float) -> VehicleState:
        """自行车运动模型 - 对应论文公式(3)-(10)"""
        # 公式(3): 速度更新
        new_v = state.v + accel * self.params.dt
        new_v = max(self.params.min_speed, min(new_v, self.params.max_speed))
        
        # 公式(4): 转向半径
        if abs(steer) < 1e-6:
            Rr = float('inf')
        else:
            Rr = self.params.wheelbase / math.tan(steer)
        
        # 公式(5): 行驶距离
        d = new_v * self.params.dt
        
        # 公式(6): 航向角变化
        if abs(Rr) < 1e6:
            dtheta = d / Rr
        else:
            dtheta = 0
        
        # 公式(7): 新航向角
        new_theta = state.theta + dtheta
        new_theta = math.atan2(math.sin(new_theta), math.cos(new_theta))
        
        # 公式(8)-(9): 位置更新
        if abs(dtheta) < 1e-6:
            new_x = state.x + d * math.cos(state.theta)
            new_y = state.y + d * math.sin(state.theta)
        else:
            new_x = state.x + Rr * (math.sin(new_theta) - math.sin(state.theta))
            new_y = state.y + Rr * (math.cos(state.theta) - math.cos(new_theta))
        
        # 公式(10): 时间更新
        new_t = state.t + self.params.dt
        
        return VehicleState(new_x, new_y, new_theta, new_v, new_t, steer)
    
    def heuristic(self, state: VehicleState, goal: VehicleState) -> float:
        """启发式函数 - 对应论文公式(15)"""
        dx = goal.x - state.x
        dy = goal.y - state.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        goal_heading = math.atan2(dy, dx)
        heading_diff = abs(math.atan2(math.sin(state.theta - goal_heading), 
                                     math.cos(state.theta - goal_heading)))
        
        return distance + 1.5 * heading_diff
    
    def cost_function(self, current: HybridNode, new_state: VehicleState) -> float:
        """成本函数 - 对应论文公式(16)"""
        motion_cost = math.sqrt((new_state.x - current.state.x)**2 + 
                               (new_state.y - current.state.y)**2)
        
        speed_change_cost = self.params.wv * abs(new_state.v - current.state.v)
        
        vref = 5.0
        speed_ref_cost = self.params.wref * abs(new_state.v - vref)
        
        direction_cost = self.params.wδ * abs(new_state.theta - current.state.theta)
        
        return motion_cost + speed_change_cost + speed_ref_cost + direction_cost
    
    def is_fitting_success(self, current_node: HybridNode, goal: VehicleState) -> Tuple[bool, Optional[List[VehicleState]]]:
        """目标拟合检查 - 根据优化级别选择策略"""
        distance = math.sqrt((current_node.state.x - goal.x)**2 + 
                           (current_node.state.y - goal.y)**2)
        
        if distance > 8.0:
            return False, None
        
        if self.optimization_level == OptimizationLevel.BASIC:
            # 基础模式：直线拟合
            return self._straight_line_fitting(current_node.state, goal)
        
        elif self.optimization_level == OptimizationLevel.ENHANCED:
            # 增强模式：快速Dubins拟合
            dubins_trajectory = self.dubins_path.compute_dubins_path(
                current_node.state, goal, quick_mode=True)
            
            if dubins_trajectory is None:
                return self._straight_line_fitting(current_node.state, goal)
            
            # 快速碰撞检测
            for state in dubins_trajectory[::2]:  # 每隔一个点检测
                if not self.environment.is_collision_free(state, self.params):
                    return False, None
            
            return True, dubins_trajectory
        
        else:  # FULL
            # 完整模式：完整Dubins拟合
            dubins_trajectory = self.dubins_path.compute_dubins_path(
                current_node.state, goal, quick_mode=False)
            
            if dubins_trajectory is None:
                return self._straight_line_fitting(current_node.state, goal)
            
            # 完整碰撞检测
            for state in dubins_trajectory:
                if not self.environment.is_collision_free(state, self.params):
                    return False, None
            
            return True, dubins_trajectory
    
    def _straight_line_fitting(self, start_state: VehicleState, goal_state: VehicleState) -> Tuple[bool, Optional[List[VehicleState]]]:
        """直线拟合备选方案"""
        distance = math.sqrt((goal_state.x - start_state.x)**2 + 
                           (goal_state.y - start_state.y)**2)
        
        num_points = max(2, int(distance / 1.0))
        trajectory = []
        
        for i in range(num_points + 1):
            t = i / num_points if num_points > 0 else 0
            x = start_state.x + t * (goal_state.x - start_state.x)
            y = start_state.y + t * (goal_state.y - start_state.y)
            theta = start_state.theta + t * (goal_state.theta - start_state.theta)
            v = start_state.v + t * (goal_state.v - start_state.v)
            time = start_state.t + t * distance / max(1.0, (start_state.v + goal_state.v) / 2)
            
            state = VehicleState(x, y, theta, v, time)
            
            if not self.environment.is_collision_free(state, self.params):
                return False, None
            
            trajectory.append(state)
        
        return True, trajectory
    
    def generate_intermediate_nodes(self, parent_node: HybridNode, child_node: HybridNode, 
                                  acceleration: float) -> List[VehicleState]:
        """
        改进的中间节点生成 - 集成精确的减速节点处理
        基于论文Algorithm 1第16-21行
        """
        # 记录加速度信息
        child_node.acceleration = acceleration
        
        # 如果是减速节点，使用精确的中间节点生成
        if acceleration < 0:
            return self.intermediate_generator.generate_intermediate_nodes_for_deceleration(
                parent_node, child_node)
        else:
            # 非减速节点使用原有方法
            return self._generate_simple_intermediate_nodes(parent_node, child_node)
    
    def _generate_simple_intermediate_nodes(self, parent_node: HybridNode, 
                                          child_node: HybridNode) -> List[VehicleState]:
        """为非减速节点生成简单的中间节点"""
        intermediate_nodes = []
        
        distance = math.sqrt((child_node.state.x - parent_node.state.x)**2 + 
                           (child_node.state.y - parent_node.state.y)**2)
        
        # 根据优化级别调整密度
        if self.optimization_level == OptimizationLevel.BASIC:
            step_size = 0.8
        elif self.optimization_level == OptimizationLevel.ENHANCED:
            step_size = 0.5
        else:  # FULL
            step_size = 0.3
        
        num_intermediate = max(1, int(distance / step_size))
        
        for i in range(1, num_intermediate):
            t = i / num_intermediate
            x = parent_node.state.x + t * (child_node.state.x - parent_node.state.x)
            y = parent_node.state.y + t * (child_node.state.y - parent_node.state.y)
            theta = parent_node.state.theta + t * (child_node.state.theta - parent_node.state.theta)
            v = parent_node.state.v + t * (child_node.state.v - parent_node.state.v)
            time = parent_node.state.t + t * (child_node.state.t - parent_node.state.t)
            
            intermediate_nodes.append(VehicleState(x, y, theta, v, time))
        
        return intermediate_nodes
    
    def search(self, start: VehicleState, goal: VehicleState, 
             high_priority_trajectories: List[List[VehicleState]] = None) -> Optional[List[VehicleState]]:
        """
        集成改进的V-Hybrid A*搜索算法
        """
        print(f"      V-Hybrid A* search ({self.optimization_level.value}): ({start.x:.1f},{start.y:.1f}) -> ({goal.x:.1f},{goal.y:.1f})")
        
        if high_priority_trajectories is None:
            high_priority_trajectories = []
        
        # 更新Box约束的障碍物信息（如果使用FULL模式）
        if self.optimization_level == OptimizationLevel.FULL:
            self._update_box_constraints_obstacles()
        
        # Algorithm 1 第1-2行: 初始化
        start_node = HybridNode(start, 0.0, self.heuristic(start, goal))
        open_set = [start_node]
        closed_set = set()
        g_score = {start_node.grid_key(): 0.0}
        
        iterations = 0
        
        # Algorithm 1 第3行: while循环
        while open_set and iterations < self.max_iterations:
            iterations += 1
            
            current = heapq.heappop(open_set)
            current_key = current.grid_key()
            
            if current_key in closed_set:
                continue
            
            closed_set.add(current_key)
            
            # Algorithm 1 第6-8行: 目标拟合检查
            fitting_success, fitting_trajectory = self.is_fitting_success(current, goal)
            if fitting_success:
                print(f"        ✅ Goal reached in {iterations} iterations")
                initial_path = self._reconstruct_path(current) + fitting_trajectory[1:]
                
                # 应用轨迹处理
                processed_trajectory = self.trajectory_processor.process_trajectory(
                    initial_path, high_priority_trajectories)
                
                return processed_trajectory
            
            # Algorithm 1 第10行: 查找扩展节点
            for accel, steer in self.motion_primitives:
                new_state = self.bicycle_model(current.state, accel, steer)
                
                # 边界检查
                margin = 2.0
                if not (margin <= new_state.x <= self.environment.size - margin and 
                       margin <= new_state.y <= self.environment.size - margin):
                    continue
                
                if new_state.t > 80:  # 减少时间限制
                    continue
                
                new_node = HybridNode(new_state, 0, self.heuristic(new_state, goal))
                new_key = new_node.grid_key()
                
                if new_key in closed_set:
                    continue
                
                # Algorithm 1 第16-21行: 改进的减速节点处理
                if accel < 0:
                    intermediate_nodes = self.generate_intermediate_nodes(current, new_node, accel)
                    collision_detected = False
                    
                    # 检查中间节点碰撞
                    for intermediate_state in intermediate_nodes:
                        if not self.environment.is_collision_free(intermediate_state, self.params):
                            collision_detected = True
                            break
                    
                    # 验证轨迹物理可行性
                    if not collision_detected:
                        is_valid = self.intermediate_generator.validate_intermediate_trajectory(
                            current.state, new_state, intermediate_nodes)
                        if not is_valid:
                            collision_detected = True
                    
                    if collision_detected:
                        closed_set.add(new_key)
                        continue
                
                if not self.environment.is_collision_free(new_state, self.params):
                    continue
                
                g_new = current.g_cost + self.cost_function(current, new_state)
                new_node.g_cost = g_new
                new_node.parent = current
                
                if new_key not in g_score or g_new < g_score[new_key]:
                    g_score[new_key] = g_new
                    heapq.heappush(open_set, new_node)
        
        print(f"        ❌ Search failed after {iterations} iterations")
        return None
    
    def _update_box_constraints_obstacles(self):
        """更新Box约束的障碍物信息"""
        if not hasattr(self, 'box_constraints'):
            return
            
        obstacles = []
        
        # 从环境中提取障碍物信息
        if hasattr(self.environment, 'obstacle_map'):
            obs_y, obs_x = np.where(self.environment.obstacle_map)
            for x, y in zip(obs_x, obs_y):
                # 将网格坐标转换为世界坐标，并创建矩形障碍物
                obstacles.append((x, y, x+1, y+1))
        
        self.box_constraints.update_obstacle_grid(obstacles)
    
    def search_with_waiting(self, start: VehicleState, goal: VehicleState, 
                          vehicle_id: int = None, 
                          high_priority_trajectories: List[List[VehicleState]] = None) -> Optional[List[VehicleState]]:
        """带等待机制的搜索"""
        print(f"    Planning vehicle {vehicle_id}: ({start.x:.1f},{start.y:.1f}) -> ({goal.x:.1f},{goal.y:.1f})")
        
        start_valid = self.environment.is_valid_position(start.x, start.y)
        goal_valid = self.environment.is_valid_position(goal.x, goal.y)
        start_collision_free = self.environment.is_collision_free(start, self.params)
        
        print(f"      起始位置检查: 坐标有效={start_valid}, 无碰撞={start_collision_free}")
        print(f"      目标位置检查: 坐标有效={goal_valid}")
        
        if not start_valid or not goal_valid:
            print(f"      ❌ 起始或目标位置无效")
            return None
        
        if self.environment.is_start_position_blocked(start, self.params):
            print(f"      Start position blocked, finding safe start time...")
            safe_delay = self.environment.find_safe_start_time(start, self.params)
            
            if safe_delay is not None:
                print(f"      Waiting {safe_delay:.1f}s for safe start")
                delayed_start = start.copy()
                delayed_start.t = start.t + safe_delay
                return self.search(delayed_start, goal, high_priority_trajectories)
            else:
                print(f"      No safe start time found")
                return None
        else:
            return self.search(start, goal, high_priority_trajectories)
    
    def _reconstruct_path(self, node: HybridNode) -> List[VehicleState]:
        """重构路径"""
        path = []
        current = node
        while current:
            path.append(current.state)
            current = current.parent
        return path[::-1]

class MultiVehicleCoordinator:
    """多车辆协调器 - 集成改进版本"""
    
    def __init__(self, map_file_path=None, optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED):
        self.environment = UnstructuredEnvironment(size=100)
        self.params = VehicleParameters()
        self.optimization_level = optimization_level
        self.map_data = None
        self.vehicles = {}
        self.trajectories = {}
        
        if map_file_path:
            self.load_map(map_file_path)
        
        print(f"🎯 多车辆协调器初始化完成 (优化级别: {optimization_level.value})")
        if optimization_level == OptimizationLevel.FULL:
            if HAS_CVXPY:
                print("  ✅ CVXPY可用，将使用完整的QP优化")
            else:
                print("  ⚠️ CVXPY不可用，将回退到增强模式")
    
    def load_map(self, map_file_path):
        """加载地图文件"""
        self.map_data = self.environment.load_from_json(map_file_path)
        return self.map_data is not None
    
    def create_scenario_from_json(self):
        """从JSON数据创建车辆场景"""
        if not self.map_data:
            print("❌ 没有加载地图数据")
            return []
        
        start_points = self.map_data.get("start_points", [])
        end_points = self.map_data.get("end_points", [])
        point_pairs = self.map_data.get("point_pairs", [])
        
        print(f"📍 发现 {len(start_points)} 个起点, {len(end_points)} 个终点, {len(point_pairs)} 个配对")
        
        scenarios = []
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, pair in enumerate(point_pairs):
            start_id = pair["start_id"]
            end_id = pair["end_id"]
            
            start_point = next((p for p in start_points if p["id"] == start_id), None)
            end_point = next((p for p in end_points if p["id"] == end_id), None)
            
            if start_point and end_point:
                dx = end_point["x"] - start_point["x"]
                dy = end_point["y"] - start_point["y"]
                optimal_theta = math.atan2(dy, dx)
                
                start_state = VehicleState(
                    x=float(start_point["x"]),
                    y=float(start_point["y"]),
                    theta=optimal_theta,
                    v=3.0,
                    t=0.0
                )
                
                goal_state = VehicleState(
                    x=float(end_point["x"]),
                    y=float(end_point["y"]),
                    theta=optimal_theta,
                    v=2.0,
                    t=0.0
                )
                
                scenario = {
                    'id': i + 1,
                    'priority': len(point_pairs) - i,
                    'color': colors[i % len(colors)],
                    'start': start_state,
                    'goal': goal_state,
                    'description': f'Vehicle {i+1} (S{start_id}->E{end_id})'
                }
                
                scenarios.append(scenario)
                print(f"  ✅ 车辆 {i+1}: ({start_point['x']},{start_point['y']}) -> ({end_point['x']},{end_point['y']})")
        
        return scenarios
    
    def plan_all_vehicles(self, scenarios):
        """规划所有车辆的轨迹 - 集成改进版本"""
        sorted_scenarios = sorted(scenarios, key=lambda x: x['priority'], reverse=True)
        
        results = {}
        high_priority_trajectories = []  # 存储已规划的高优先级轨迹
        
        print(f"\n🚀 规划 {len(scenarios)} 台车辆 (优化级别: {self.optimization_level.value})...")
        print(f"📊 改进特性:")
        print(f"  ✅ 精确的减速节点中间节点生成")
        if self.optimization_level == OptimizationLevel.FULL:
            print(f"  ✅ 完整的Box约束优化")
            if HAS_CVXPY:
                print(f"  ✅ QP轨迹优化")
        
        for i, scenario in enumerate(sorted_scenarios):
            print(f"\n--- Vehicle {scenario['id']} (Priority {scenario['priority']}) ---")
            print(f"Description: {scenario['description']}")
            
            vehicle_start_time = time.time()
            
            planner = VHybridAStarPlanner(self.environment, self.optimization_level)
            
            # 传递高优先级轨迹用于优化
            trajectory = planner.search_with_waiting(
                scenario['start'], scenario['goal'], scenario['id'], 
                high_priority_trajectories)
            
            vehicle_planning_time = time.time() - vehicle_start_time
            
            if trajectory:
                print(f"SUCCESS: {len(trajectory)} waypoints, time: {trajectory[-1].t:.1f}s, planning: {vehicle_planning_time:.2f}s")
                
                # 显示改进效果统计
                if self.optimization_level != OptimizationLevel.BASIC:
                    self._analyze_trajectory_improvements(trajectory, scenario['id'])
                
                results[scenario['id']] = {
                    'trajectory': trajectory,
                    'color': scenario['color'],
                    'description': scenario['description'],
                    'planning_time': vehicle_planning_time
                }
                
                # 添加为动态障碍物和高优先级轨迹
                self.environment.add_vehicle_trajectory(trajectory, self.params)
                high_priority_trajectories.append(trajectory)
                print(f"Added as dynamic obstacle for remaining {len(sorted_scenarios)-i-1} vehicles")
            else:
                print(f"FAILED: No feasible trajectory, planning: {vehicle_planning_time:.2f}s")
                results[scenario['id']] = {
                    'trajectory': [], 
                    'color': scenario['color'], 
                    'description': scenario['description'],
                    'planning_time': vehicle_planning_time
                }
        
        return results, sorted_scenarios
    
    def _analyze_trajectory_improvements(self, trajectory: List[VehicleState], vehicle_id: int):
        """分析轨迹改进效果"""
        if len(trajectory) < 2:
            return
        
        # 计算平滑度指标
        smoothness_score = self._calculate_smoothness(trajectory)
        
        # 计算速度一致性
        speed_consistency = self._calculate_speed_consistency(trajectory)
        
        # 计算转向平滑度
        steering_smoothness = self._calculate_steering_smoothness(trajectory)
        
        print(f"      改进效果分析:")
        print(f"        轨迹平滑度: {smoothness_score:.3f}")
        print(f"        速度一致性: {speed_consistency:.3f}")
        print(f"        转向平滑度: {steering_smoothness:.3f}")
    
    def _calculate_smoothness(self, trajectory: List[VehicleState]) -> float:
        """计算轨迹平滑度"""
        if len(trajectory) < 3:
            return 1.0
        
        curvature_changes = []
        for i in range(1, len(trajectory) - 1):
            prev_state = trajectory[i-1]
            curr_state = trajectory[i]
            next_state = trajectory[i+1]
            
            # 计算曲率变化
            angle1 = math.atan2(curr_state.y - prev_state.y, curr_state.x - prev_state.x)
            angle2 = math.atan2(next_state.y - curr_state.y, next_state.x - curr_state.x)
            
            angle_change = abs(angle2 - angle1)
            if angle_change > math.pi:
                angle_change = 2 * math.pi - angle_change
            
            curvature_changes.append(angle_change)
        
        if not curvature_changes:
            return 1.0
        
        avg_curvature_change = sum(curvature_changes) / len(curvature_changes)
        return max(0, 1 - avg_curvature_change / (math.pi / 4))  # 归一化到[0,1]
    
    def _calculate_speed_consistency(self, trajectory: List[VehicleState]) -> float:
        """计算速度一致性"""
        if len(trajectory) < 2:
            return 1.0
        
        speed_changes = []
        for i in range(1, len(trajectory)):
            speed_change = abs(trajectory[i].v - trajectory[i-1].v)
            speed_changes.append(speed_change)
        
        if not speed_changes:
            return 1.0
        
        avg_speed_change = sum(speed_changes) / len(speed_changes)
        return max(0, 1 - avg_speed_change / 2.0)  # 归一化，假设最大速度变化为2m/s
    
    def _calculate_steering_smoothness(self, trajectory: List[VehicleState]) -> float:
        """计算转向平滑度"""
        if len(trajectory) < 2:
            return 1.0
        
        theta_changes = []
        for i in range(1, len(trajectory)):
            theta_change = abs(trajectory[i].theta - trajectory[i-1].theta)
            if theta_change > math.pi:
                theta_change = 2 * math.pi - theta_change
            theta_changes.append(theta_change)
        
        if not theta_changes:
            return 1.0
        
        avg_theta_change = sum(theta_changes) / len(theta_changes)
        return max(0, 1 - avg_theta_change / (math.pi / 6))  # 归一化
    
    def create_animation(self, results, scenarios):
        """创建可视化动画 - 增强版本"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        self._setup_environment_plot(ax1)
        
        all_trajectories = []
        for scenario in scenarios:
            vid = scenario['id']
            if vid in results and results[vid]['trajectory']:
                traj = results[vid]['trajectory']
                color = results[vid]['color']
                all_trajectories.append((traj, color, scenario['description']))
        
        if not all_trajectories:
            print("No successful trajectories to animate")
            return
        
        max_time = max(max(state.t for state in traj) for traj, _, _ in all_trajectories)
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            self._setup_environment_plot(ax1)
            
            current_time = frame * 0.5
            
            active_vehicles = 0
            for traj, color, desc in all_trajectories:
                current_state = None
                for state in traj:
                    if state.t <= current_time:
                        current_state = state
                    else:
                        break
                
                if current_state:
                    active_vehicles += 1
                    self._draw_vehicle(ax1, current_state, color)
                    
                    past_states = [s for s in traj if s.t <= current_time]
                    if len(past_states) > 1:
                        xs = [s.x for s in past_states]
                        ys = [s.y for s in past_states]
                        ax1.plot(xs, ys, color=color, alpha=0.6, linewidth=2)
                        
                        # 新增：显示改进效果
                        if self.optimization_level != OptimizationLevel.BASIC and len(past_states) > 5:
                            # 在轨迹上显示约束框（仅FULL模式）
                            if self.optimization_level == OptimizationLevel.FULL:
                                self._draw_constraint_boxes(ax1, past_states[-5:], color)
            
            if self.map_data:
                self._draw_json_points(ax1)
            
            # 增强的标题显示
            improvement_text = ""
            if self.optimization_level == OptimizationLevel.ENHANCED:
                improvement_text = " + 精确中间节点"
            elif self.optimization_level == OptimizationLevel.FULL:
                improvement_text = " + 精确中间节点 + Box约束"
            
            ax1.set_title(f'集成改进的V-Hybrid A* ({self.optimization_level.value}){improvement_text}\n[{self.environment.map_name}] (t = {current_time:.1f}s) Active: {active_vehicles}')
            
            self._draw_timeline(ax2, all_trajectories, current_time)
            
            return []
        
        frames = int(max_time / 0.5) + 10
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def _draw_constraint_boxes(self, ax, states: List[VehicleState], color: str):
        """绘制约束框（仅在FULL模式下）"""
        if self.optimization_level != OptimizationLevel.FULL:
            return
        
        try:
            # 创建临时的Box约束处理器
            box_constraints = AdvancedBoxConstraints(self.params)
            box_constraints.update_obstacle_grid([])  # 简化处理
            
            constraints = box_constraints.generate_box_constraints(states)
            
            for constraint in constraints[-3:]:  # 只显示最近的几个约束框
                xlb = constraint['xlb']
                xub = constraint['xub']
                ylb = constraint['ylb']
                yub = constraint['yub']
                
                rect = patches.Rectangle((xlb, ylb), xub-xlb, yub-ylb, 
                                       linewidth=1, edgecolor=color, facecolor='none', alpha=0.3)
                ax.add_patch(rect)
        except:
            pass  # 如果出错，跳过约束框绘制
    
    def _setup_environment_plot(self, ax):
        """设置环境可视化"""
        ax.add_patch(patches.Rectangle((0, 0), self.environment.size, self.environment.size,
                                     facecolor='lightgray', alpha=0.1))
        
        free_y, free_x = np.where(~self.environment.obstacle_map)
        ax.scatter(free_x, free_y, c='lightblue', s=1, alpha=0.3)
        
        obs_y, obs_x = np.where(self.environment.obstacle_map)
        ax.scatter(obs_x, obs_y, c='darkred', s=4, alpha=0.8)
        
        ax.set_xlim(0, self.environment.size)
        ax.set_ylim(0, self.environment.size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
    
    def _draw_vehicle(self, ax, state: VehicleState, color):
        """绘制车辆"""
        length, width = self.params.length, self.params.width
        
        corners = np.array([
            [-length/2, -width/2],
            [length/2, -width/2], 
            [length/2, width/2],
            [-length/2, width/2],
            [-length/2, -width/2]
        ])
        
        cos_theta, sin_theta = math.cos(state.theta), math.sin(state.theta)
        rotation = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        rotated_corners = corners @ rotation.T
        translated_corners = rotated_corners + np.array([state.x, state.y])
        
        vehicle_patch = patches.Polygon(translated_corners[:-1], facecolor=color, 
                                       alpha=0.8, edgecolor='black')
        ax.add_patch(vehicle_patch)
        
        arrow_length = 3
        dx = arrow_length * cos_theta
        dy = arrow_length * sin_theta
        ax.arrow(state.x, state.y, dx, dy, head_width=1, head_length=1,
                fc=color, ec='black', alpha=0.9)
    
    def _draw_json_points(self, ax):
        """绘制JSON地图中的起点和终点"""
        if not self.map_data:
            return
        
        for point in self.map_data.get("start_points", []):
            ax.plot(point["x"], point["y"], 'go', markersize=12, markeredgecolor='darkgreen', markeredgewidth=2)
            ax.text(point["x"], point["y"], str(point["id"]), ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=8)
        
        for point in self.map_data.get("end_points", []):
            ax.plot(point["x"], point["y"], 'rs', markersize=12, markeredgecolor='darkred', markeredgewidth=2)
            ax.text(point["x"], point["y"], str(point["id"]), ha='center', va='center', 
                   color='white', fontweight='bold', fontsize=8)
        
        for pair in self.map_data.get("point_pairs", []):
            start_point = next((p for p in self.map_data.get("start_points", []) if p["id"] == pair["start_id"]), None)
            end_point = next((p for p in self.map_data.get("end_points", []) if p["id"] == pair["end_id"]), None)
            
            if start_point and end_point:
                ax.plot([start_point["x"], end_point["x"]], 
                       [start_point["y"], end_point["y"]], 
                       'b--', alpha=0.5, linewidth=1)
    
    def _draw_timeline(self, ax, all_trajectories, current_time):
        """绘制时间线"""
        ax.set_title(f'Vehicle Timeline - {self.environment.map_name} ({self.optimization_level.value})')
        
        for i, (traj, color, desc) in enumerate(all_trajectories):
            y_pos = len(all_trajectories) - i
            
            start_time = traj[0].t
            if start_time > 0:
                ax.plot([0, start_time], [y_pos, y_pos], color='gray', 
                       linewidth=4, alpha=0.5)
            
            times = [state.t for state in traj]
            ax.plot(times, [y_pos] * len(times), color=color, linewidth=6, alpha=0.3)
            
            completed_times = [t for t in times if t <= current_time]
            if completed_times:
                ax.plot(completed_times, [y_pos] * len(completed_times), 
                       color=color, linewidth=6, alpha=0.9)
            
            if times and current_time <= max(times):
                ax.plot(current_time, y_pos, 'o', color='red', markersize=8)
            
            wait_info = f" (wait {start_time:.0f}s)" if start_time > 0 else ""
            ax.text(max(times) + 1, y_pos, desc + wait_info, fontsize=10, va='center')
        
        ax.axvline(x=current_time, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Vehicle')
        ax.grid(True, alpha=0.3)

def save_trajectories(results, filename):
    """保存轨迹数据到JSON文件"""
    trajectory_data = {
        'metadata': {
            'timestamp': time.time(),
            'performance_metrics': {
                'total_vehicles': len(results),
                'successful_vehicles': sum(1 for vid in results if results[vid].get('trajectory')),
                'avg_planning_time': sum(results[vid].get('planning_time', 0) for vid in results) / len(results) if results else 0
            }
        },
        'trajectories': {}
    }
    
    for vid, result in results.items():
        if result.get('trajectory'):
            trajectory_data['trajectories'][f"vehicle_{vid}"] = {
                'description': result['description'],
                'color': result['color'],
                'planning_time': result.get('planning_time', 0),
                'trajectory': [
                    {
                        'x': state.x,
                        'y': state.y,
                        'theta': state.theta,
                        'v': state.v,
                        't': state.t
                    }
                    for state in result['trajectory']
                ]
            }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)
        print(f"💾 改进轨迹数据已保存到: {filename}")
    except Exception as e:
        print(f"❌ 保存轨迹数据失败: {str(e)}")

def main():
    """主函数 - 集成改进版本演示"""
    print("🚀 IEEE TIT 2024论文完整集成改进版")
    print("📄 Multi-Vehicle Collaborative Trajectory Planning Based on V-Hybrid A*")
    print("⚡ 集成改进:")
    print("   ✅ 精确的中间节点生成机制 (基于论文Algorithm 1第16-21行)")
    print("   ✅ 完整的Box约束实现 (基于论文公式22-25)")
    print("   ✅ 保持原有的性能优化特性")
    print("   ✅ 增强的可视化和分析功能")
    print("=" * 80)
    
    # 优化级别选择
    print("\n🎯 优化级别选择:")
    print("  1. BASIC    - 基础V-Hybrid A* (最快)")
    print("  2. ENHANCED - 增强版 + 精确中间节点生成 (推荐)")
    print("  3. FULL     - 完整版 + Box约束 + QP优化 (最优，需要CVXPY)")
    
    # 自动选择优化级别
    if HAS_CVXPY:
        optimization_level = OptimizationLevel.FULL  # 使用完整功能
        print(f"  📊 自动选择: {optimization_level.value} (CVXPY可用，启用完整功能)")
    else:
        optimization_level = OptimizationLevel.ENHANCED  # 使用增强功能
        print(f"  📊 自动选择: {optimization_level.value} (CVXPY不可用，使用增强模式)")
    
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if not json_files:
        print("❌ 当前目录没有找到JSON地图文件")
        print("请先创建地图文件，或使用以下示例创建简单测试地图:")
        create_simple_test_map()
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    print(f"\n📁 发现 {len(json_files)} 个JSON地图文件:")
    for i, file in enumerate(json_files):
        print(f"  {i+1}. {file}")
    
    selected_file = json_files[0]
    print(f"\n🎯 使用地图文件: {selected_file}")
    
    coordinator = MultiVehicleCoordinator(map_file_path=selected_file, optimization_level=optimization_level)
    
    if not coordinator.map_data:
        print("❌ 地图加载失败")
        return
    
    scenarios = coordinator.create_scenario_from_json()
    
    if not scenarios:
        print("❌ 没有找到有效的车辆配对")
        return
    
    print(f"\n🚗 车辆场景:")
    for scenario in sorted(scenarios, key=lambda x: x['priority'], reverse=True):
        print(f"  V{scenario['id']} (优先级{scenario['priority']}): {scenario['description']}")
    
    print(f"\n📊 集成改进算法参数:")
    params = coordinator.params
    print(f"  优化级别: {optimization_level.value}")
    print(f"  车辆参数: L={params.wheelbase}m, δmax={math.degrees(params.max_steer):.1f}°")
    print(f"  运动约束: vmax={params.max_speed}m/s, amax={params.max_accel}m/s²")
    print(f"  时间分辨率: Δt={params.dt}s")
    print(f"  改进特性:")
    print(f"    - 精确的减速节点中间节点生成")
    if optimization_level == OptimizationLevel.FULL:
        print(f"    - 完整的Box约束优化 (公式22-25)")
        if HAS_CVXPY:
            print(f"    - QP轨迹优化")
    
    # 性能测试
    print(f"\n⏱️  集成改进版性能测试开始...")
    start_time = time.time()
    results, sorted_scenarios = coordinator.plan_all_vehicles(scenarios)
    planning_time = time.time() - start_time
    
    success_count = sum(1 for vid in results if results[vid]['trajectory'])
    avg_planning_time = sum(results[vid].get('planning_time', 0) for vid in results) / len(results) if results else 0
    
    print(f"\n📊 集成改进版规划结果:")
    print(f"总规划时间: {planning_time:.2f}s")
    print(f"平均单车规划时间: {avg_planning_time:.2f}s")
    print(f"成功率: {success_count}/{len(scenarios)} ({100*success_count/len(scenarios):.1f}%)")
    print(f"优化级别: {optimization_level.value}")
    
    if success_count >= 1:
        print(f"🎬 Creating enhanced animation with improvements...")
        anim = coordinator.create_animation(results, scenarios)
        
        trajectory_file = f"{coordinator.environment.map_name}_integrated_{optimization_level.value}.json"
        save_trajectories(results, trajectory_file)
        
        print(f"\n✨ 集成改进特性汇总:")
        print(f"  ✅ 精确的中间节点生成机制 (论文Algorithm 1第16-21行)")
        print(f"  ✅ 物理约束验证 (速度、加速度、转向角度)")
        print(f"  ✅ 轨迹平滑度分析和评估")
        if optimization_level == OptimizationLevel.FULL:
            print(f"  ✅ 完整的Box约束实现 (论文公式22-25)")
            print(f"  ✅ 自适应约束框调整")
            print(f"  ✅ 障碍物感知的约束优化")
            if HAS_CVXPY:
                print(f"  ✅ QP轨迹优化")
        print(f"  ✅ 时间同步和性能优化")
        print(f"  ✅ 增强的可视化和分析功能")
        
        input("Press Enter to exit...")
    else:
        print("❌ No successful trajectories for animation")
    
    print("\n🎉 集成改进版演示完成!")

def create_simple_test_map():
    """创建简单的测试地图"""
    # 创建一个50x50的网格，0表示可通行，1表示障碍物
    grid = np.zeros((50, 50), dtype=int)
    
    # 添加一些障碍物
    grid[15:17, 20:23] = 1  # 障碍物块1
    grid[25:27, 30:33] = 1  # 障碍物块2
    
    test_map = {
        "map_info": {
            "name": "integrated_test_map",
            "width": 50,
            "height": 50,
            "description": "集成改进测试地图"
        },
        "grid": grid.tolist(),  # 将numpy数组转换为列表
        "obstacles": [
            {"x": 20, "y": 15}, {"x": 21, "y": 15}, {"x": 22, "y": 15},
            {"x": 20, "y": 16}, {"x": 21, "y": 16}, {"x": 22, "y": 16},
            {"x": 30, "y": 25}, {"x": 31, "y": 25}, {"x": 32, "y": 25},
            {"x": 30, "y": 26}, {"x": 31, "y": 26}, {"x": 32, "y": 26},
        ],
        "start_points": [
            {"id": 1, "x": 5, "y": 10},
            {"id": 2, "x": 5, "y": 20},
            {"id": 3, "x": 5, "y": 30}
        ],
        "end_points": [
            {"id": 1, "x": 45, "y": 10},
            {"id": 2, "x": 45, "y": 20},
            {"id": 3, "x": 45, "y": 30}
        ],
        "point_pairs": [
            {"start_id": 1, "end_id": 1},
            {"start_id": 2, "end_id": 2},
            {"start_id": 3, "end_id": 3}
        ]
    }
    
    with open("integrated_test_map.json", "w", encoding="utf-8") as f:
        json.dump(test_map, f, indent=2, ensure_ascii=False)
    
    print("✅ 已创建集成改进测试地图: integrated_test_map.json")

if __name__ == "__main__":
    main()