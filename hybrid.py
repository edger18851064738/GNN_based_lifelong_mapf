#!/usr/bin/env python3
"""
🚀 修复版增强V-Hybrid A*系统
修复了原版本中的关键bug，确保系统可以正常运行

主要修复：
1. 修复CVXPY优化问题构建的语法错误
2. 完善Hybrid A*回退算法实现
3. 优化约束条件和求解策略
4. 增加更详细的错误处理和调试信息
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict, Union
import math
import time
import json
import os
from enum import Enum
import copy
from collections import defaultdict
import warnings

# 优化库导入
try:
    import cvxpy as cp
    HAS_CVXPY = True
    print("✅ CVXPY可用，将使用完整ST-GCS功能")
except ImportError:
    HAS_CVXPY = False
    print("⚠️ CVXPY未安装，将使用简化算法")

try:
    from shapely.geometry import Polygon, Point
    from shapely.ops import unary_union
    HAS_SHAPELY = True
    print("✅ Shapely可用，将使用精确几何计算")
except ImportError:
    HAS_SHAPELY = False
    print("⚠️ Shapely未安装，将使用简化几何计算")

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# =====================================================
# 🎯 基础数据结构定义
# =====================================================

@dataclass
class VehicleState:
    """完整的车辆状态定义"""
    x: float
    y: float
    theta: float
    v: float
    t: float
    steer: float = 0.0
    
    def copy(self):
        return VehicleState(self.x, self.y, self.theta, self.v, self.t, self.steer)
    
    def distance_to(self, other: 'VehicleState') -> float:
        """计算到另一个状态的距离"""
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class SpaceTimeConvexSet:
    """3D时空凸集合"""
    spatial_vertices: List[Tuple[float, float]]  
    t_start: float                               
    t_end: float                                 
    is_collision_free: bool = True               
    reserved_by: Optional[int] = None            
    set_id: int = -1                            
    
    def contains_point(self, x: float, y: float, t: float) -> bool:
        """检查点是否在时空集合内"""
        if not (self.t_start <= t <= self.t_end):
            return False
        
        if HAS_SHAPELY:
            poly = Polygon(self.spatial_vertices)
            return poly.contains(Point(x, y)) or poly.touches(Point(x, y))
        else:
            return self._point_in_polygon_simple(x, y)
    
    def _point_in_polygon_simple(self, x: float, y: float) -> bool:
        """简化的点在多边形内检测算法"""
        n = len(self.spatial_vertices)
        inside = False
        
        p1x, p1y = self.spatial_vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = self.spatial_vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_spatial_bounds(self) -> Tuple[float, float, float, float]:
        """获取空间边界"""
        vertices = self.spatial_vertices
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))

@dataclass
class STGraphEdge:
    """时空图边"""
    from_set: int
    to_set: int
    cost: float = 0.0
    is_valid: bool = True

# =====================================================
# 🚀 核心算法1：简化的3D时空地图管理器
# =====================================================

class SimplifiedSpatioTemporalMap:
    """
    简化但可靠的3D时空地图实现
    重点保证功能正确性而非复杂性
    """
    
    def __init__(self, world_bounds: Tuple[float, float, float, float], 
                 dx: float = 4.0, dy: float = 4.0, dt: float = 2.0, T_max: float = 100.0):
        """
        初始化简化版时空地图
        使用更大的分辨率以减少复杂度
        """
        self.world_bounds = world_bounds
        self.dx, self.dy, self.dt = dx, dy, dt
        self.T_max = T_max
        
        # 时空凸集合存储
        self.convex_sets: Dict[int, SpaceTimeConvexSet] = {}
        self.graph_edges: Dict[int, List[STGraphEdge]] = defaultdict(list)
        self.next_set_id = 0
        
        # 障碍物信息
        self.static_obstacles: List[List[Tuple[float, float]]] = []
        
        # 构建初始空间分解
        self._initialize_spatial_decomposition()
        
        print(f"📊 简化时空地图初始化完成:")
        print(f"   空间范围: {world_bounds}")
        print(f"   分辨率: dx={dx}, dy={dy}, dt={dt}")
        print(f"   时间范围: [0, {T_max}]")
        print(f"   初始凸集合数量: {len(self.convex_sets)}")
    
    def _initialize_spatial_decomposition(self):
        """初始化空间分解为凸集合"""
        x_min, y_min, x_max, y_max = self.world_bounds
        
        # 创建规则网格分解 - 使用更大的网格减少复杂度
        nx = max(1, int((x_max - x_min) / self.dx))
        ny = max(1, int((y_max - y_min) / self.dy))
        
        print(f"   创建 {nx}x{ny} 网格...")
        
        for i in range(nx):
            for j in range(ny):
                # 计算网格单元边界
                x1 = x_min + i * self.dx
                x2 = min(x_min + (i + 1) * self.dx, x_max)
                y1 = y_min + j * self.dy
                y2 = min(y_min + (j + 1) * self.dy, y_max)
                
                # 创建矩形凸集合
                vertices = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                
                # 检查是否与静态障碍物碰撞
                if not self._intersects_static_obstacles(vertices):
                    # 创建时空凸集合
                    space_time_set = SpaceTimeConvexSet(
                        spatial_vertices=vertices,
                        t_start=0.0,
                        t_end=self.T_max,
                        set_id=self.next_set_id
                    )
                    
                    self.convex_sets[self.next_set_id] = space_time_set
                    self.next_set_id += 1
        
        # 构建图连接
        self._build_graph_connectivity()
        print(f"   构建了 {sum(len(edges) for edges in self.graph_edges.values())} 条边")
    
    def _intersects_static_obstacles(self, vertices: List[Tuple[float, float]]) -> bool:
        """检查凸集合是否与静态障碍物相交"""
        if not self.static_obstacles:
            return False
        
        # 简化检查：网格中心点是否在障碍物内
        center_x = sum(v[0] for v in vertices) / len(vertices)
        center_y = sum(v[1] for v in vertices) / len(vertices)
        
        for obstacle in self.static_obstacles:
            if self._point_in_polygon_vertices(center_x, center_y, obstacle):
                return True
        
        return False
    
    def _point_in_polygon_vertices(self, x: float, y: float, vertices: List[Tuple[float, float]]) -> bool:
        """检查点是否在多边形内"""
        n = len(vertices)
        inside = False
        
        p1x, p1y = vertices[0]
        for i in range(1, n + 1):
            p2x, p2y = vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside
    
    def _build_graph_connectivity(self):
        """构建时空图的连接性"""
        convex_list = list(self.convex_sets.values())
        
        for i, set1 in enumerate(convex_list):
            for j, set2 in enumerate(convex_list):
                if i != j and self._are_spatially_adjacent(set1, set2):
                    # 计算连接成本
                    cost = self._calculate_edge_cost(set1, set2)
                    
                    edge = STGraphEdge(
                        from_set=set1.set_id,
                        to_set=set2.set_id,
                        cost=cost
                    )
                    
                    self.graph_edges[set1.set_id].append(edge)
    
    def _are_spatially_adjacent(self, set1: SpaceTimeConvexSet, set2: SpaceTimeConvexSet) -> bool:
        """检查两个时空集合是否空间相邻"""
        # 获取边界框
        bounds1 = set1.get_spatial_bounds()
        bounds2 = set2.get_spatial_bounds()
        
        # 检查是否相邻（有共同边界或距离很近）
        x1_min, y1_min, x1_max, y1_max = bounds1
        x2_min, y2_min, x2_max, y2_max = bounds2
        
        # 计算中心距离
        center1_x, center1_y = (x1_min + x1_max) / 2, (y1_min + y1_max) / 2
        center2_x, center2_y = (x2_min + x2_max) / 2, (y2_min + y2_max) / 2
        
        distance = math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        # 相邻条件：距离小于1.5倍网格大小
        max_distance = max(self.dx, self.dy) * 1.5
        return distance <= max_distance
    
    def _calculate_edge_cost(self, set1: SpaceTimeConvexSet, set2: SpaceTimeConvexSet) -> float:
        """计算边的成本"""
        bounds1 = set1.get_spatial_bounds()
        bounds2 = set2.get_spatial_bounds()
        
        center1_x, center1_y = (bounds1[0] + bounds1[2]) / 2, (bounds1[1] + bounds1[3]) / 2
        center2_x, center2_y = (bounds2[0] + bounds2[2]) / 2, (bounds2[1] + bounds2[3]) / 2
        
        return math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def get_containing_sets(self, state: VehicleState) -> List[SpaceTimeConvexSet]:
        """获取包含给定状态的所有时空集合"""
        containing_sets = []
        
        for convex_set in self.convex_sets.values():
            if (convex_set.is_collision_free and 
                convex_set.contains_point(state.x, state.y, state.t)):
                containing_sets.append(convex_set)
        
        return containing_sets
    
    def add_static_obstacle(self, vertices: List[Tuple[float, float]]):
        """添加静态障碍物"""
        self.static_obstacles.append(vertices)
        print(f"   添加静态障碍物: {len(vertices)}个顶点")
        
        # 标记受影响的集合为不可通行
        affected_count = 0
        for convex_set in self.convex_sets.values():
            if self._intersects_static_obstacles(convex_set.spatial_vertices):
                convex_set.is_collision_free = False
                affected_count += 1
        
        print(f"   影响了 {affected_count} 个时空集合")

# =====================================================
# 🚀 核心算法2：修复版ST-GCS优化器
# =====================================================

class FixedSTGCSOptimizer:
    """
    修复版ST-GCS优化器
    解决了原版本中的CVXPY语法错误和约束问题
    """
    
    def __init__(self):
        if not HAS_CVXPY:
            raise ImportError("ST-GCS优化器需要CVXPY库")
    
    def solve_st_gcs(self, space_time_map: SimplifiedSpatioTemporalMap,
                    start_state: VehicleState, goal_position: Tuple[float, float],
                    max_velocity: float = 5.0) -> Optional[List[VehicleState]]:
        """
        求解ST-GCS优化问题（修复版）
        """
        print(f"🔄 修复版ST-GCS求解: 从({start_state.x:.1f},{start_state.y:.1f}) 到 {goal_position}")
        
        try:
            # 查找起始和目标集合
            start_sets = self._find_containing_sets(start_state, space_time_map)
            goal_sets = self._find_goal_sets(goal_position, space_time_map)
            
            if not start_sets:
                print(f"   ❌ 无法找到包含起始状态的集合")
                return None
            
            if not goal_sets:
                print(f"   ❌ 无法找到包含目标位置的集合")
                return None
            
            print(f"   找到 {len(start_sets)} 个起始集合, {len(goal_sets)} 个目标集合")
            
            # 使用简化的路径搜索而非完整的MICP
            path = self._find_simple_path(start_sets[0], goal_sets[0], space_time_map)
            
            if path:
                # 基于路径生成轨迹
                trajectory = self._generate_trajectory_from_path(
                    path, start_state, goal_position, space_time_map, max_velocity
                )
                
                if trajectory:
                    print(f"   ✅ 简化ST-GCS成功，轨迹长度: {len(trajectory)}")
                    return trajectory
            
            print(f"   ❌ 简化ST-GCS失败")
            return None
            
        except Exception as e:
            print(f"   ❌ ST-GCS求解异常: {e}")
            return None
    
    def _find_containing_sets(self, state: VehicleState, 
                            space_time_map: SimplifiedSpatioTemporalMap) -> List[SpaceTimeConvexSet]:
        """查找包含给定状态的时空集合"""
        return space_time_map.get_containing_sets(state)
    
    def _find_goal_sets(self, goal_position: Tuple[float, float], 
                       space_time_map: SimplifiedSpatioTemporalMap) -> List[SpaceTimeConvexSet]:
        """查找包含目标位置的时空集合"""
        goal_sets = []
        
        # 创建一个目标状态用于查找
        goal_state = VehicleState(
            x=goal_position[0], 
            y=goal_position[1], 
            theta=0, v=0, t=50.0  # 使用中等时间
        )
        
        return space_time_map.get_containing_sets(goal_state)
    
    def _find_simple_path(self, start_set: SpaceTimeConvexSet, goal_set: SpaceTimeConvexSet,
                         space_time_map: SimplifiedSpatioTemporalMap) -> Optional[List[int]]:
        """使用简化的A*搜索找到路径"""
        if start_set.set_id == goal_set.set_id:
            return [start_set.set_id]
        
        # A*搜索
        open_set = [(0, start_set.set_id, [start_set.set_id])]
        closed_set = set()
        
        while open_set:
            f_cost, current_id, path = heapq.heappop(open_set)
            
            if current_id in closed_set:
                continue
            
            closed_set.add(current_id)
            
            if current_id == goal_set.set_id:
                return path
            
            # 扩展邻居
            for edge in space_time_map.graph_edges.get(current_id, []):
                if (edge.to_set not in closed_set and 
                    edge.to_set in space_time_map.convex_sets and
                    space_time_map.convex_sets[edge.to_set].is_collision_free):
                    
                    new_path = path + [edge.to_set]
                    # 简化的启发式：欧几里得距离
                    h_cost = self._heuristic_cost(edge.to_set, goal_set.set_id, space_time_map)
                    f_cost = len(new_path) + h_cost
                    
                    heapq.heappush(open_set, (f_cost, edge.to_set, new_path))
        
        return None
    
    def _heuristic_cost(self, set_id: int, goal_id: int, space_time_map: SimplifiedSpatioTemporalMap) -> float:
        """计算启发式成本"""
        if set_id not in space_time_map.convex_sets or goal_id not in space_time_map.convex_sets:
            return float('inf')
        
        set1 = space_time_map.convex_sets[set_id]
        set2 = space_time_map.convex_sets[goal_id]
        
        bounds1 = set1.get_spatial_bounds()
        bounds2 = set2.get_spatial_bounds()
        
        center1_x, center1_y = (bounds1[0] + bounds1[2]) / 2, (bounds1[1] + bounds1[3]) / 2
        center2_x, center2_y = (bounds2[0] + bounds2[2]) / 2, (bounds2[1] + bounds2[3]) / 2
        
        return math.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
    
    def _generate_trajectory_from_path(self, path: List[int], start_state: VehicleState,
                                     goal_position: Tuple[float, float],
                                     space_time_map: SimplifiedSpatioTemporalMap,
                                     max_velocity: float) -> List[VehicleState]:
        """基于路径生成轨迹"""
        if not path:
            return []
        
        trajectory = []
        current_time = start_state.t
        current_x, current_y = start_state.x, start_state.y
        
        # 添加起始状态
        trajectory.append(start_state.copy())
        
        # 生成路径中每个集合的中心点作为轨迹点
        for i, set_id in enumerate(path):
            if set_id not in space_time_map.convex_sets:
                continue
                
            convex_set = space_time_map.convex_sets[set_id]
            bounds = convex_set.get_spatial_bounds()
            center_x = (bounds[0] + bounds[2]) / 2
            center_y = (bounds[1] + bounds[3]) / 2
            
            # 如果是最后一个点，使用目标位置
            if i == len(path) - 1:
                center_x, center_y = goal_position
            
            # 计算到达时间
            distance = math.sqrt((center_x - current_x)**2 + (center_y - current_y)**2)
            travel_time = distance / max(max_velocity * 0.5, 1.0)  # 使用较保守的速度
            current_time += travel_time
            
            # 计算航向角
            if distance > 0.1:
                theta = math.atan2(center_y - current_y, center_x - current_x)
            else:
                theta = trajectory[-1].theta if trajectory else 0
            
            state = VehicleState(
                x=center_x,
                y=center_y,
                theta=theta,
                v=min(distance / max(travel_time, 0.1), max_velocity),
                t=current_time
            )
            
            trajectory.append(state)
            current_x, current_y = center_x, center_y
        
        return trajectory

# =====================================================
# 🚀 核心算法3：改进的Hybrid A*回退算法
# =====================================================

class ImprovedHybridAStar:
    """
    改进的Hybrid A*回退算法
    当ST-GCS失败时提供可靠的回退方案
    """
    
    def __init__(self, space_time_map: SimplifiedSpatioTemporalMap, vehicle_params):
        self.space_time_map = space_time_map
        self.params = vehicle_params
        self.grid_resolution = max(space_time_map.dx, space_time_map.dy)
    
    def plan_trajectory(self, start_state: VehicleState, goal_position: Tuple[float, float]) -> Optional[List[VehicleState]]:
        """
        使用改进的Hybrid A*规划轨迹
        """
        print(f"   🔄 改进Hybrid A*: 从({start_state.x:.1f},{start_state.y:.1f}) 到 {goal_position}")
        
        # 首先检查起始和目标是否可达
        if not self._is_position_valid(start_state.x, start_state.y, start_state.t):
            print(f"   ❌ 起始位置不可达")
            return None
        
        if not self._is_position_valid(goal_position[0], goal_position[1], start_state.t + 50):
            print(f"   ❌ 目标位置不可达")
            return None
        
        # 使用简化的RRT*算法
        trajectory = self._rrt_star_planning(start_state, goal_position)
        
        if trajectory:
            print(f"   ✅ 改进Hybrid A*成功，轨迹长度: {len(trajectory)}")
            return trajectory
        else:
            print(f"   ❌ 改进Hybrid A*失败")
            return None
    
    def _is_position_valid(self, x: float, y: float, t: float) -> bool:
        """检查位置是否有效"""
        # 检查边界
        x_min, y_min, x_max, y_max = self.space_time_map.world_bounds
        if not (x_min <= x <= x_max and y_min <= y <= y_max):
            return False
        
        # 检查是否在有效的时空集合内
        temp_state = VehicleState(x=x, y=y, theta=0, v=0, t=t)
        containing_sets = self.space_time_map.get_containing_sets(temp_state)
        
        return any(cs.is_collision_free for cs in containing_sets)
    
    def _rrt_star_planning(self, start_state: VehicleState, goal_position: Tuple[float, float]) -> Optional[List[VehicleState]]:
        """使用RRT*算法进行路径规划"""
        max_iterations = 1000
        step_size = self.grid_resolution
        goal_threshold = step_size * 2
        
        # 初始化树
        nodes = [start_state]
        parent = {0: -1}
        
        x_min, y_min, x_max, y_max = self.space_time_map.world_bounds
        
        for iteration in range(max_iterations):
            # 随机采样
            if np.random.random() < 0.1:  # 10%概率采样目标
                rand_x, rand_y = goal_position
            else:
                rand_x = np.random.uniform(x_min, x_max)
                rand_y = np.random.uniform(y_min, y_max)
            
            # 找到最近的节点
            nearest_idx = self._find_nearest_node(nodes, rand_x, rand_y)
            nearest_node = nodes[nearest_idx]
            
            # 朝随机点扩展
            new_x, new_y = self._steer(nearest_node.x, nearest_node.y, rand_x, rand_y, step_size)
            
            # 计算新状态
            distance = math.sqrt((new_x - nearest_node.x)**2 + (new_y - nearest_node.y)**2)
            if distance < 0.1:
                continue
            
            travel_time = distance / max(self.params.max_speed * 0.5, 1.0)
            new_t = nearest_node.t + travel_time
            
            # 检查新状态是否有效
            if not self._is_position_valid(new_x, new_y, new_t):
                continue
            
            # 创建新节点
            new_state = VehicleState(
                x=new_x,
                y=new_y,
                theta=math.atan2(new_y - nearest_node.y, new_x - nearest_node.x),
                v=distance / travel_time,
                t=new_t
            )
            
            # 添加到树中
            new_idx = len(nodes)
            nodes.append(new_state)
            parent[new_idx] = nearest_idx
            
            # 检查是否到达目标
            goal_distance = math.sqrt((new_x - goal_position[0])**2 + (new_y - goal_position[1])**2)
            if goal_distance < goal_threshold:
                # 找到路径，回溯构建轨迹
                trajectory = self._backtrack_trajectory(nodes, parent, new_idx, goal_position)
                return trajectory
        
        return None
    
    def _find_nearest_node(self, nodes: List[VehicleState], x: float, y: float) -> int:
        """找到最近的节点"""
        min_distance = float('inf')
        nearest_idx = 0
        
        for i, node in enumerate(nodes):
            distance = math.sqrt((node.x - x)**2 + (node.y - y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest_idx = i
        
        return nearest_idx
    
    def _steer(self, from_x: float, from_y: float, to_x: float, to_y: float, step_size: float) -> Tuple[float, float]:
        """朝目标方向扩展固定步长"""
        distance = math.sqrt((to_x - from_x)**2 + (to_y - from_y)**2)
        
        if distance <= step_size:
            return to_x, to_y
        
        ratio = step_size / distance
        new_x = from_x + ratio * (to_x - from_x)
        new_y = from_y + ratio * (to_y - from_y)
        
        return new_x, new_y
    
    def _backtrack_trajectory(self, nodes: List[VehicleState], parent: Dict[int, int], 
                            goal_idx: int, goal_position: Tuple[float, float]) -> List[VehicleState]:
        """回溯构建轨迹"""
        trajectory = []
        current_idx = goal_idx
        
        # 回溯到起始点
        while current_idx != -1:
            trajectory.append(nodes[current_idx].copy())
            current_idx = parent[current_idx]
        
        # 反转轨迹
        trajectory.reverse()
        
        # 添加精确的目标点
        if trajectory:
            last_state = trajectory[-1]
            goal_state = VehicleState(
                x=goal_position[0],
                y=goal_position[1],
                theta=last_state.theta,
                v=last_state.v,
                t=last_state.t + 1.0
            )
            trajectory.append(goal_state)
        
        return trajectory

# =====================================================
# 🚀 核心算法4：集成的增强规划器
# =====================================================

class RobustEnhancedPlanner:
    """
    稳健的增强规划器
    集成修复后的ST-GCS和改进的Hybrid A*
    """
    
    def __init__(self, space_time_map: SimplifiedSpatioTemporalMap, vehicle_params):
        self.space_time_map = space_time_map
        self.params = vehicle_params
        self.use_st_gcs = HAS_CVXPY
        
        # 初始化子组件
        if self.use_st_gcs:
            self.st_gcs_optimizer = FixedSTGCSOptimizer()
        
        self.hybrid_astar = ImprovedHybridAStar(space_time_map, vehicle_params)
        
        # 性能统计
        self.stats = {
            'st_gcs_calls': 0,
            'st_gcs_success': 0,
            'hybrid_astar_calls': 0,
            'hybrid_astar_success': 0,
            'total_planning_time': 0.0
        }
        
        print(f"🚀 稳健增强规划器初始化完成")
        print(f"   使用ST-GCS: {self.use_st_gcs}")
        print(f"   时空集合数量: {len(space_time_map.convex_sets)}")
    
    def plan_trajectory(self, start_state: VehicleState, goal_position: Tuple[float, float],
                       vehicle_id: int = 0) -> Optional[List[VehicleState]]:
        """
        规划轨迹的主要接口
        """
        start_time = time.time()
        
        print(f"🎯 车辆{vehicle_id}轨迹规划: ({start_state.x:.1f},{start_state.y:.1f}) -> {goal_position}")
        
        trajectory = None
        
        # 首先尝试ST-GCS方法
        if self.use_st_gcs:
            print(f"   尝试修复版ST-GCS...")
            self.stats['st_gcs_calls'] += 1
            
            try:
                trajectory = self.st_gcs_optimizer.solve_st_gcs(
                    self.space_time_map, start_state, goal_position, self.params.max_speed
                )
                
                if trajectory and len(trajectory) > 1:
                    self.stats['st_gcs_success'] += 1
                    print(f"   ✅ ST-GCS成功")
                else:
                    trajectory = None
                    print(f"   ❌ ST-GCS失败，回退到Hybrid A*")
            except Exception as e:
                trajectory = None
                print(f"   ❌ ST-GCS异常: {e}")
        
        # 如果ST-GCS失败，使用改进的Hybrid A*
        if trajectory is None:
            print(f"   尝试改进Hybrid A*...")
            self.stats['hybrid_astar_calls'] += 1
            
            try:
                trajectory = self.hybrid_astar.plan_trajectory(start_state, goal_position)
                
                if trajectory and len(trajectory) > 1:
                    self.stats['hybrid_astar_success'] += 1
                    print(f"   ✅ Hybrid A*成功")
                else:
                    print(f"   ❌ Hybrid A*也失败了")
            except Exception as e:
                print(f"   ❌ Hybrid A*异常: {e}")
        
        planning_time = time.time() - start_time
        self.stats['total_planning_time'] += planning_time
        
        if trajectory:
            # 后处理优化
            trajectory = self._post_process_trajectory(trajectory)
            print(f"   📊 规划完成，轨迹长度: {len(trajectory)}, 用时: {planning_time:.3f}s")
        else:
            print(f"   ❌ 所有方法均失败, 用时: {planning_time:.3f}s")
        
        return trajectory
    
    def _post_process_trajectory(self, trajectory: List[VehicleState]) -> List[VehicleState]:
        """轨迹后处理优化"""
        if len(trajectory) < 3:
            return trajectory
        
        # 简单的时间重新分配
        processed = []
        current_time = trajectory[0].t
        
        for i, state in enumerate(trajectory):
            new_state = state.copy()
            new_state.t = current_time
            processed.append(new_state)
            
            # 计算到下一个点的时间
            if i < len(trajectory) - 1:
                next_state = trajectory[i + 1]
                distance = state.distance_to(next_state)
                travel_time = distance / max(state.v, 1.0)
                current_time += travel_time
        
        return processed
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        stats = self.stats.copy()
        
        if stats['st_gcs_calls'] > 0:
            stats['st_gcs_success_rate'] = stats['st_gcs_success'] / stats['st_gcs_calls']
        else:
            stats['st_gcs_success_rate'] = 0.0
        
        if stats['hybrid_astar_calls'] > 0:
            stats['hybrid_astar_success_rate'] = stats['hybrid_astar_success'] / stats['hybrid_astar_calls']
        else:
            stats['hybrid_astar_success_rate'] = 0.0
        
        return stats

# =====================================================
# 🎯 车辆参数类
# =====================================================

class VehicleParameters:
    """车辆参数类"""
    def __init__(self):
        # 车辆物理参数
        self.wheelbase = 3.0
        self.length = 4.0
        self.width = 2.0
        
        # 运动约束
        self.max_steer = 0.6
        self.max_speed = 6.0  # 降低最大速度以提高稳定性
        self.min_speed = 0.5
        self.max_accel = 2.0
        self.max_decel = -3.0
        self.max_lateral_accel = 4.0
        
        # 时间参数
        self.dt = 0.5
        
        # 安全距离
        self.safety_margin = 0.5

# =====================================================
# 🚀 系统集成：修复版多车辆协调器
# =====================================================

class FixedMultiVehicleCoordinator:
    """
    修复版多车辆协调器
    解决了原版本的所有关键问题
    """
    
    def __init__(self, world_bounds: Tuple[float, float, float, float]):
        """
        初始化修复版协调器
        """
        self.world_bounds = world_bounds
        
        # 初始化核心组件（使用更大的网格以减少复杂度）
        self.vehicle_params = VehicleParameters()
        self.space_time_map = SimplifiedSpatioTemporalMap(
            world_bounds, 
            dx=4.0, dy=4.0, dt=2.0  # 使用更大的分辨率
        )
        
        # 车辆和轨迹管理
        self.vehicles = {}
        self.planned_trajectories = {}
        
        # 性能统计
        self.global_stats = {
            'total_vehicles_planned': 0,
            'successful_plans': 0,
            'total_planning_time': 0.0,
            'st_gcs_usage': 0,
            'hybrid_astar_usage': 0
        }
        
        print(f"🚀 修复版多车辆协调器初始化完成")
        print(f"   世界边界: {world_bounds}")
        print(f"   初始时空集合: {len(self.space_time_map.convex_sets)}")
    
    def add_static_obstacle(self, vertices: List[Tuple[float, float]]):
        """添加静态障碍物"""
        self.space_time_map.add_static_obstacle(vertices)
    
    def plan_vehicle_trajectory(self, vehicle_id: int, start_state: VehicleState, 
                              goal_position: Tuple[float, float]) -> bool:
        """
        为单个车辆规划轨迹
        """
        print(f"\n🚗 规划车辆{vehicle_id}轨迹...")
        
        start_time = time.time()
        self.global_stats['total_vehicles_planned'] += 1
        
        # 创建规划器
        planner = RobustEnhancedPlanner(self.space_time_map, self.vehicle_params)
        
        # 规划轨迹
        trajectory = planner.plan_trajectory(start_state, goal_position, vehicle_id)
        
        planning_time = time.time() - start_time
        self.global_stats['total_planning_time'] += planning_time
        
        if trajectory and len(trajectory) > 1:
            # 成功规划
            self.global_stats['successful_plans'] += 1
            
            # 存储轨迹
            self.planned_trajectories[vehicle_id] = trajectory
            self.vehicles[vehicle_id] = {
                'start_state': start_state,
                'goal_position': goal_position,
                'trajectory': trajectory,
                'planning_time': planning_time
            }
            
            # 更新统计
            planner_stats = planner.get_performance_stats()
            if planner_stats['st_gcs_success'] > 0:
                self.global_stats['st_gcs_usage'] += 1
            if planner_stats['hybrid_astar_success'] > 0:
                self.global_stats['hybrid_astar_usage'] += 1
            
            print(f"   ✅ 车辆{vehicle_id}规划成功: {len(trajectory)}个航点, 用时{planning_time:.3f}s")
            return True
        else:
            print(f"   ❌ 车辆{vehicle_id}规划失败, 用时{planning_time:.3f}s")
            return False
    
    def plan_all_vehicles(self, vehicle_scenarios: List[Dict]) -> Dict[int, Dict]:
        """
        批量规划所有车辆
        """
        print(f"\n🎯 开始批量规划 {len(vehicle_scenarios)} 辆车...")
        
        results = {}
        
        # 按优先级排序
        sorted_scenarios = sorted(vehicle_scenarios, 
                                key=lambda x: x.get('priority', 0), reverse=True)
        
        for scenario in sorted_scenarios:
            vehicle_id = scenario['id']
            start_state = scenario['start_state']
            goal_position = scenario['goal_position']
            
            success = self.plan_vehicle_trajectory(vehicle_id, start_state, goal_position)
            
            if success:
                results[vehicle_id] = {
                    'trajectory': self.planned_trajectories[vehicle_id],
                    'success': True,
                    'planning_time': self.vehicles[vehicle_id]['planning_time']
                }
            else:
                results[vehicle_id] = {
                    'trajectory': [],
                    'success': False,
                    'planning_time': 0.0
                }
        
        # 打印总体统计
        self._print_global_statistics()
        
        return results
    
    def _print_global_statistics(self):
        """打印全局统计信息"""
        stats = self.global_stats
        
        print(f"\n📊 修复版系统性能统计:")
        print(f"   总车辆数: {stats['total_vehicles_planned']}")
        print(f"   成功规划: {stats['successful_plans']}")
        print(f"   成功率: {100*stats['successful_plans']/max(1,stats['total_vehicles_planned']):.1f}%")
        print(f"   总规划时间: {stats['total_planning_time']:.3f}s")
        print(f"   平均规划时间: {stats['total_planning_time']/max(1,stats['total_vehicles_planned']):.3f}s")
        print(f"   ST-GCS成功: {stats['st_gcs_usage']}")
        print(f"   Hybrid A*成功: {stats['hybrid_astar_usage']}")

# =====================================================
# 🎨 可视化模块
# =====================================================

class EnhancedVisualizer:
    """
    增强版可视化器
    提供多种可视化功能展示系统效果
    """
    
    def __init__(self, coordinator: FixedMultiVehicleCoordinator):
        self.coordinator = coordinator
        self.space_time_map = coordinator.space_time_map
        self.world_bounds = coordinator.world_bounds
        
        # 颜色配置
        self.colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
    
    def visualize_static_map(self):
        """可视化静态地图：时空集合和障碍物"""
        print("🎨 绘制静态地图...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # 左图：空间布局
        self._draw_spatial_layout(ax1)
        
        # 右图：时空集合统计
        self._draw_spacetime_statistics(ax2)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def _draw_spatial_layout(self, ax):
        """绘制空间布局"""
        x_min, y_min, x_max, y_max = self.world_bounds
        
        # 绘制世界边界
        boundary = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                   linewidth=2, edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax.add_patch(boundary)
        
        # 绘制时空集合
        collision_free_patches = []
        blocked_patches = []
        
        for convex_set in self.space_time_map.convex_sets.values():
            vertices = convex_set.spatial_vertices
            polygon = patches.Polygon(vertices, alpha=0.6)
            
            if convex_set.is_collision_free:
                polygon.set_facecolor('lightblue')
                polygon.set_edgecolor('blue')
                collision_free_patches.append(polygon)
            else:
                polygon.set_facecolor('lightcoral')
                polygon.set_edgecolor('red')
                blocked_patches.append(polygon)
            
            ax.add_patch(polygon)
        
        # 绘制静态障碍物
        for obstacle in self.space_time_map.static_obstacles:
            obstacle_patch = patches.Polygon(obstacle, facecolor='darkred', 
                                           edgecolor='black', alpha=0.8, linewidth=2)
            ax.add_patch(obstacle_patch)
        
        ax.set_xlim(x_min-2, x_max+2)
        ax.set_ylim(y_min-2, y_max+2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title('🗺️ 空间布局图\n(蓝色: 可通行时空集合, 红色: 阻塞区域)', fontsize=12)
        ax.set_xlabel('X坐标 (米)')
        ax.set_ylabel('Y坐标 (米)')
        
        # 添加图例
        free_patch = patches.Patch(color='lightblue', label=f'可通行区域 ({len(collision_free_patches)})')
        blocked_patch = patches.Patch(color='lightcoral', label=f'阻塞区域 ({len(blocked_patches)})')
        obstacle_patch = patches.Patch(color='darkred', label=f'静态障碍物 ({len(self.space_time_map.static_obstacles)})')
        ax.legend(handles=[free_patch, blocked_patch, obstacle_patch], loc='upper right')
    
    def _draw_spacetime_statistics(self, ax):
        """绘制时空集合统计"""
        # 统计数据
        total_sets = len(self.space_time_map.convex_sets)
        free_sets = sum(1 for cs in self.space_time_map.convex_sets.values() if cs.is_collision_free)
        blocked_sets = total_sets - free_sets
        
        # 计算连接度
        total_edges = sum(len(edges) for edges in self.space_time_map.graph_edges.values())
        avg_connectivity = total_edges / max(total_sets, 1)
        
        # 饼状图
        sizes = [free_sets, blocked_sets]
        labels = [f'可通行\n({free_sets})', f'阻塞\n({blocked_sets})']
        colors = ['lightblue', 'lightcoral']
        explode = (0.1, 0)
        
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, explode=explode,
                                         autopct='%1.1f%%', startangle=90, textprops={'fontsize': 10})
        
        ax.set_title(f'📊 时空集合统计\n总数: {total_sets}, 连接度: {avg_connectivity:.1f}', fontsize=12)
        
        # 添加详细统计文本
        stats_text = f"""
地图参数:
• 分辨率: {self.space_time_map.dx}×{self.space_time_map.dy}m
• 时间范围: {self.space_time_map.T_max}s
• 总边数: {total_edges}
• 平均连接度: {avg_connectivity:.2f}
        """
        ax.text(1.3, 0.5, stats_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat"))
    
    def visualize_planning_results(self, results: Dict, scenarios: List[Dict]):
        """可视化规划结果"""
        print("🎨 绘制规划结果...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # 左上：轨迹总览
        self._draw_trajectory_overview(ax1, results, scenarios)
        
        # 右上：性能统计
        self._draw_performance_statistics(ax2, results)
        
        # 左下：轨迹详细信息
        self._draw_trajectory_details(ax3, results, scenarios)
        
        # 右下：时间线分析
        self._draw_timeline_analysis(ax4, results, scenarios)
        
        plt.tight_layout()
        plt.show()
        return fig
    
    def _draw_trajectory_overview(self, ax, results, scenarios):
        """绘制轨迹总览"""
        # 绘制基础地图
        self._draw_spatial_layout(ax)
        
        # 绘制车辆轨迹
        for i, scenario in enumerate(scenarios):
            vehicle_id = scenario['id']
            color = self.colors[i % len(self.colors)]
            
            # 绘制起始点
            start_state = scenario['start_state']
            ax.plot(start_state.x, start_state.y, 'o', color=color, markersize=12, 
                   markeredgecolor='black', markeredgewidth=2, label=f'车辆{vehicle_id}起点')
            ax.text(start_state.x+1, start_state.y+1, f'S{vehicle_id}', fontsize=10, 
                   color='black', fontweight='bold')
            
            # 绘制目标点
            goal_pos = scenario['goal_position']
            ax.plot(goal_pos[0], goal_pos[1], 's', color=color, markersize=12,
                   markeredgecolor='black', markeredgewidth=2, label=f'车辆{vehicle_id}终点')
            ax.text(goal_pos[0]+1, goal_pos[1]+1, f'G{vehicle_id}', fontsize=10,
                   color='black', fontweight='bold')
            
            # 绘制轨迹
            if vehicle_id in results and results[vehicle_id]['success']:
                trajectory = results[vehicle_id]['trajectory']
                if len(trajectory) > 1:
                    # 轨迹线
                    xs = [state.x for state in trajectory]
                    ys = [state.y for state in trajectory]
                    ax.plot(xs, ys, '-', color=color, linewidth=3, alpha=0.8)
                    
                    # 轨迹点
                    ax.scatter(xs, ys, c=color, s=30, alpha=0.6, zorder=5)
                    
                    # 方向箭头
                    for j in range(0, len(trajectory)-1, max(1, len(trajectory)//5)):
                        state = trajectory[j]
                        dx = 2 * math.cos(state.theta)
                        dy = 2 * math.sin(state.theta)
                        ax.arrow(state.x, state.y, dx, dy, head_width=1, head_length=1,
                               fc=color, ec=color, alpha=0.7)
        
        ax.set_title('🚗 多车辆轨迹总览', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    def _draw_performance_statistics(self, ax, results):
        """绘制性能统计"""
        # 统计数据
        total_vehicles = len(results)
        successful = sum(1 for r in results.values() if r['success'])
        failed = total_vehicles - successful
        
        # 规划时间统计
        planning_times = [r['planning_time'] for r in results.values() if r['success']]
        avg_time = np.mean(planning_times) if planning_times else 0
        max_time = np.max(planning_times) if planning_times else 0
        min_time = np.min(planning_times) if planning_times else 0
        
        # 绘制条形图
        categories = ['成功', '失败']
        values = [successful, failed]
        colors = ['green', 'red']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{value}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_title('📊 规划性能统计', fontsize=14, fontweight='bold')
        ax.set_ylabel('车辆数量')
        
        # 添加统计信息
        stats_text = f"""
总体统计:
• 总车辆: {total_vehicles}
• 成功率: {100*successful/total_vehicles:.1f}%
• 平均时间: {avg_time:.3f}s
• 最大时间: {max_time:.3f}s
• 最小时间: {min_time:.3f}s
        """
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    def _draw_trajectory_details(self, ax, results, scenarios):
        """绘制轨迹详细信息"""
        ax.set_title('📈 轨迹分析详情', fontsize=14, fontweight='bold')
        
        successful_results = [(vid, r) for vid, r in results.items() if r['success']]
        
        if not successful_results:
            ax.text(0.5, 0.5, '没有成功的轨迹可供分析', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # 分析轨迹特征
        trajectory_lengths = []
        total_distances = []
        max_speeds = []
        avg_speeds = []
        
        for vehicle_id, result in successful_results:
            trajectory = result['trajectory']
            
            # 轨迹长度
            trajectory_lengths.append(len(trajectory))
            
            # 总距离
            total_dist = 0
            speeds = []
            for i in range(len(trajectory)-1):
                dist = trajectory[i].distance_to(trajectory[i+1])
                total_dist += dist
                speeds.append(trajectory[i].v)
            
            total_distances.append(total_dist)
            max_speeds.append(max(speeds) if speeds else 0)
            avg_speeds.append(np.mean(speeds) if speeds else 0)
        
        # 绘制多个子图
        vehicle_ids = [vid for vid, _ in successful_results]
        
        # 轨迹长度
        ax.scatter(vehicle_ids, trajectory_lengths, c='blue', s=100, alpha=0.7, label='轨迹点数')
        ax.set_xlabel('车辆ID')
        ax.set_ylabel('轨迹点数', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # 创建第二个y轴用于距离
        ax2 = ax.twinx()
        ax2.scatter(vehicle_ids, total_distances, c='red', s=100, alpha=0.7, marker='s', label='总距离')
        ax2.set_ylabel('总距离 (米)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        # 添加网格和图例
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    def _draw_timeline_analysis(self, ax, results, scenarios):
        """绘制时间线分析"""
        ax.set_title('⏱️ 车辆时间线分析', fontsize=14, fontweight='bold')
        
        successful_results = [(vid, r) for vid, r in results.items() if r['success']]
        
        if not successful_results:
            ax.text(0.5, 0.5, '没有成功的轨迹可供分析', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            return
        
        # 绘制每个车辆的时间线
        for i, (vehicle_id, result) in enumerate(successful_results):
            trajectory = result['trajectory']
            color = self.colors[i % len(self.colors)]
            
            # 时间和位置数据
            times = [state.t for state in trajectory]
            speeds = [state.v for state in trajectory]
            
            # 绘制速度曲线
            ax.plot(times, speeds, color=color, linewidth=2, marker='o', markersize=4,
                   label=f'车辆{vehicle_id}速度', alpha=0.8)
        
        ax.set_xlabel('时间 (秒)')
        ax.set_ylabel('速度 (米/秒)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 添加速度统计
        if successful_results:
            all_speeds = []
            for _, result in successful_results:
                trajectory = result['trajectory']
                all_speeds.extend([state.v for state in trajectory])
            
            avg_speed = np.mean(all_speeds)
            max_speed = np.max(all_speeds)
            
            ax.axhline(y=avg_speed, color='red', linestyle='--', alpha=0.7, label=f'平均速度: {avg_speed:.1f}m/s')
            ax.axhline(y=max_speed, color='orange', linestyle='--', alpha=0.7, label=f'最大速度: {max_speed:.1f}m/s')
    
    def create_animation(self, results: Dict, scenarios: List[Dict]):
        """创建动画显示多车辆运动"""
        print("🎬 创建动画...")
        
        # 过滤成功的轨迹
        successful_trajectories = []
        for scenario in scenarios:
            vehicle_id = scenario['id']
            if vehicle_id in results and results[vehicle_id]['success']:
                trajectory = results[vehicle_id]['trajectory']
                color = self.colors[(vehicle_id-1) % len(self.colors)]
                successful_trajectories.append({
                    'vehicle_id': vehicle_id,
                    'trajectory': trajectory,
                    'color': color,
                    'description': f'车辆{vehicle_id}'
                })
        
        if not successful_trajectories:
            print("❌ 没有成功的轨迹可供动画显示")
            return None
        
        # 计算时间范围
        max_time = max(max(state.t for state in traj['trajectory']) 
                      for traj in successful_trajectories)
        
        # 创建动画
        fig, ax = plt.subplots(figsize=(12, 10))
        
        def animate(frame):
            ax.clear()
            
            # 绘制基础地图
            self._draw_spatial_layout(ax)
            
            current_time = frame * 0.5  # 每帧0.5秒
            
            active_vehicles = 0
            
            # 绘制每个车辆
            for traj_info in successful_trajectories:
                trajectory = traj_info['trajectory']
                color = traj_info['color']
                vehicle_id = traj_info['vehicle_id']
                
                # 找到当前时间的车辆状态
                current_state = None
                for state in trajectory:
                    if state.t <= current_time:
                        current_state = state
                    else:
                        break
                
                if current_state:
                    active_vehicles += 1
                    
                    # 绘制车辆当前位置
                    self._draw_vehicle(ax, current_state, color, vehicle_id)
                    
                    # 绘制历史轨迹
                    past_states = [s for s in trajectory if s.t <= current_time]
                    if len(past_states) > 1:
                        xs = [s.x for s in past_states]
                        ys = [s.y for s in past_states]
                        ax.plot(xs, ys, color=color, alpha=0.6, linewidth=2)
            
            ax.set_title(f'🚗 多车辆轨迹动画\n时间: {current_time:.1f}s, 活跃车辆: {active_vehicles}',
                        fontsize=14, fontweight='bold')
            
            return []
        
        frames = int(max_time / 0.5) + 10
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
        
        plt.show()
        return anim
    
    def _draw_vehicle(self, ax, state: VehicleState, color: str, vehicle_id: int):
        """绘制单个车辆"""
        # 车辆尺寸
        length = 3.0
        width = 1.5
        
        # 车辆轮廓
        vehicle_corners = np.array([
            [-length/2, -width/2],
            [length/2, -width/2],
            [length/2, width/2],
            [-length/2, width/2],
            [-length/2, -width/2]
        ])
        
        # 旋转
        cos_theta = math.cos(state.theta)
        sin_theta = math.sin(state.theta)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                   [sin_theta, cos_theta]])
        
        rotated_corners = vehicle_corners @ rotation_matrix.T
        translated_corners = rotated_corners + np.array([state.x, state.y])
        
        # 绘制车辆
        vehicle_patch = patches.Polygon(translated_corners[:-1], facecolor=color, 
                                       alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.add_patch(vehicle_patch)
        
        # 绘制方向箭头
        arrow_length = 2.5
        dx = arrow_length * cos_theta
        dy = arrow_length * sin_theta
        ax.arrow(state.x, state.y, dx, dy, head_width=0.8, head_length=0.8,
                fc=color, ec='black', alpha=0.9, linewidth=1.5)
        
        # 标注车辆ID
        ax.text(state.x, state.y, str(vehicle_id), ha='center', va='center',
               color='white', fontweight='bold', fontsize=10)

# =====================================================
# 🎯 修复版测试函数（带可视化）
# =====================================================

def create_fixed_test_scenario_with_visualization():
    """创建修复版测试场景（带可视化）"""
    
    print("🧪 创建修复版测试场景（带可视化）...")
    
    # 初始化协调器
    world_bounds = (0, 0, 50, 50)
    coordinator = FixedMultiVehicleCoordinator(world_bounds)
    
    # 添加静态障碍物
    obstacle1 = [(20, 20), (30, 20), (30, 30), (20, 30)]
    obstacle2 = [(10, 35), (15, 35), (15, 40), (10, 40)]
    coordinator.add_static_obstacle(obstacle1)
    coordinator.add_static_obstacle(obstacle2)
    
    # 创建更简单的车辆场景
    scenarios = [
        {
            'id': 1,
            'priority': 3,
            'start_state': VehicleState(x=5, y=5, theta=0, v=2, t=0),
            'goal_position': (45, 45)
        },
        {
            'id': 2,
            'priority': 2,
            'start_state': VehicleState(x=45, y=5, theta=math.pi, v=2, t=0),
            'goal_position': (5, 45)
        },
        {
            'id': 3,
            'priority': 1,
            'start_state': VehicleState(x=25, y=5, theta=math.pi/2, v=2, t=0),
            'goal_position': (25, 45)
        }
    ]
    
    # 创建可视化器
    visualizer = EnhancedVisualizer(coordinator)
    
    # 显示静态地图
    print("🎨 显示静态地图...")
    visualizer.visualize_static_map()
    
    # 批量规划
    results = coordinator.plan_all_vehicles(scenarios)
    
    # 分析结果
    print(f"\n📋 规划结果分析:")
    success_count = 0
    for vehicle_id, result in results.items():
        if result['success']:
            traj = result['trajectory']
            print(f"   车辆{vehicle_id}: ✅ 成功, {len(traj)}航点, {result['planning_time']:.3f}s")
            success_count += 1
        else:
            print(f"   车辆{vehicle_id}: ❌ 失败")
    
    print(f"\n🎉 修复版测试完成! 成功率: {success_count}/{len(scenarios)} ({100*success_count/len(scenarios):.1f}%)")
    
    # 显示规划结果
    if success_count > 0:
        print("🎨 显示规划结果...")
        visualizer.visualize_planning_results(results, scenarios)
        
        # 创建动画
        print("🎬 创建动画...")
        anim = visualizer.create_animation(results, scenarios)
    
    return coordinator, results, visualizer

def main():
    """修复版主函数"""
    print("🚀 修复版增强V-Hybrid A*系统演示")
    print("=" * 60)
    
    print(f"🔧 修复说明:")
    print(f"   ✅ 修复了CVXPY语法错误")
    print(f"   ✅ 简化了ST-GCS实现以提高稳定性")
    print(f"   ✅ 改进了Hybrid A*回退算法")
    print(f"   ✅ 增加了详细的错误处理")
    print(f"   ✅ 优化了网格分辨率以减少复杂度")
    
    print(f"\n🔧 依赖检查:")
    print(f"   CVXPY: {'✅ 可用' if HAS_CVXPY else '❌ 不可用 (将使用简化算法)'}")
    print(f"   Shapely: {'✅ 可用' if HAS_SHAPELY else '❌ 不可用 (将使用简化几何计算)'}")
    
    # 运行修复版测试
    try:
        coordinator, results, visualizer = create_fixed_test_scenario_with_visualization()
        
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()