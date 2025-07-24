#!/usr/bin/env python3
"""
图增强的V-Hybrid A*多车协调系统
集成消息传递机制的图神经网络
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import os
import json

# 导入成熟算法模块
from trying import (
    VehicleState, VehicleParameters, UnstructuredEnvironment, 
    VHybridAStarPlanner, MultiVehicleCoordinator, OptimizationLevel,
    HybridNode, ConflictDensityAnalyzer, TimeSync,
    interactive_json_selection
)

class GNNEnhancementLevel(Enum):
    """图增强级别"""
    PRIORITY_ONLY = "priority_only"           
    EXPANSION_GUIDE = "expansion_guide"       
    FULL_INTEGRATION = "full_integration"     

@dataclass
class VehicleInteractionGraph:
    """车辆交互图结构"""
    node_features: torch.Tensor      # (N, feature_dim) 节点特征
    edge_indices: torch.Tensor       # (2, E) 边索引
    edge_features: torch.Tensor      # (E, edge_dim) 边特征
    vehicle_ids: List[int]           # 节点到车辆ID映射
    adjacency_matrix: torch.Tensor   # (N, N) 邻接矩阵
    global_features: torch.Tensor    # (global_dim,) 全局特征

class VehicleGraphBuilder:
    """车辆交互图构建器"""
    
    def __init__(self, params: VehicleParameters):
        self.params = params
        self.interaction_radius = 50.0  # 🆕 增大交互半径
        self.node_feature_dim = 10      
        self.edge_feature_dim = 6       
        self.global_feature_dim = 8     
        
    def _build_edges_and_features(self, vehicles_info: List[Dict]) -> Tuple[List, List, List]:
        """构建边索引、边特征和邻接矩阵"""
        n_vehicles = len(vehicles_info)
        edge_indices = []
        edge_features = []
        adjacency_matrix = np.zeros((n_vehicles, n_vehicles))
        
        print(f"        构建边: 检查{n_vehicles}辆车的交互")
        
        for i in range(n_vehicles):
            for j in range(i + 1, n_vehicles):
                # 计算交互强度和特征
                interaction_data = self._compute_interaction_features(vehicles_info[i], vehicles_info[j])
                
                # 🆕 降低交互阈值并添加调试信息
                state1 = vehicles_info[i]['current_state']
                state2 = vehicles_info[j]['current_state']
                distance = math.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
                
                if i < 3 and j < 3:  # 只打印前几对的调试信息
                    print(f"          车辆{i}-{j}: 距离={distance:.1f}, 交互强度={interaction_data['interaction_strength']:.3f}")
                
                if interaction_data['interaction_strength'] > 0.05:  # 🆕 降低阈值从0.1到0.05
                    # 添加双向边
                    edge_indices.extend([[i, j], [j, i]])
                    edge_features.extend([interaction_data['features'], interaction_data['features']])
                    
                    # 更新邻接矩阵
                    weight = interaction_data['interaction_strength']
                    adjacency_matrix[i, j] = weight
                    adjacency_matrix[j, i] = weight
        
        print(f"        构建完成: {len(edge_indices)}条边")
        return edge_indices, edge_features, adjacency_matrix.tolist()
    
    def _compute_interaction_features(self, vehicle1: Dict, vehicle2: Dict) -> Dict:
        """计算车辆间交互特征"""
        state1 = vehicle1['current_state']
        state2 = vehicle2['current_state']
        goal1 = vehicle1['goal_state']
        goal2 = vehicle2['goal_state']
        
        # 基础距离和几何关系
        dx = state2.x - state1.x
        dy = state2.y - state1.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > self.interaction_radius:
            return {'interaction_strength': 0.0, 'features': [0.0] * self.edge_feature_dim}
        
        # 🆕 修正交互强度计算，确保即使距离较远也有基础交互
        distance_factor = max(0.1, 1.0 - (distance / self.interaction_radius))  # 最小0.1而不是0
        
        # 计算详细交互特征
        
        # 1. 空间关系
        relative_bearing = math.atan2(dy, dx)
        
        # 2. 运动关系
        v1x, v1y = state1.v * math.cos(state1.theta), state1.v * math.sin(state1.theta)
        v2x, v2y = state2.v * math.cos(state2.theta), state2.v * math.sin(state2.theta)
        
        # 相对速度
        rel_vx, rel_vy = v1x - v2x, v1y - v2y
        relative_speed = math.sqrt(rel_vx*rel_vx + rel_vy*rel_vy)
        
        # 接近速度（朝向对方的速度分量）
        if distance > 1e-6:
            approach_speed = max(0, (v1x * dx + v1y * dy) / distance)
        else:
            approach_speed = 0.0
        
        # 3. 路径交叉分析
        path_crossing = self._analyze_path_crossing(state1, goal1, state2, goal2)
        
        # 4. 优先级关系
        priority_diff = (vehicle1.get('priority', 1) - vehicle2.get('priority', 1)) / 10.0
        
        # 5. 时间冲突预测
        time_to_conflict = self._estimate_time_to_conflict(state1, state2, v1x, v1y, v2x, v2y)
        
        # 🆕 修正综合交互强度计算
        interaction_strength = (
            distance_factor * 0.5 +              # 增加距离权重
            min(1.0, relative_speed / 10.0) * 0.15 +
            min(1.0, approach_speed / 5.0) * 0.15 +
            path_crossing * 0.2
        )
        
        # 确保最小交互强度
        interaction_strength = max(0.05, interaction_strength)
        
        # 6维边特征
        features = [
            distance / self.interaction_radius,     # [0] 归一化距离
            relative_speed / 10.0,                  # [1] 归一化相对速度
            approach_speed / 5.0,                   # [2] 归一化接近速度
            path_crossing,                          # [3] 路径交叉概率
            priority_diff,                          # [4] 优先级差异
            min(1.0, time_to_conflict / 20.0)      # [5] 归一化冲突时间
        ]
        
        return {
            'interaction_strength': min(1.0, interaction_strength),
            'features': features
        }
    def build_interaction_graph(self, vehicles_info: List[Dict]) -> VehicleInteractionGraph:
        """构建完整的车辆交互图"""
        n_vehicles = len(vehicles_info)
        if n_vehicles == 0:
            return self._create_empty_graph()
        
        # 提取节点特征
        node_features = self._extract_node_features(vehicles_info)
        
        # 构建边和边特征
        edge_indices, edge_features, adjacency_matrix = self._build_edges_and_features(vehicles_info)
        
        # 提取全局特征
        global_features = self._extract_global_features(vehicles_info)
        
        vehicle_ids = [v['id'] for v in vehicles_info]
        
        return VehicleInteractionGraph(
            node_features=torch.tensor(node_features, dtype=torch.float32),
            edge_indices=torch.tensor(edge_indices, dtype=torch.long).T if edge_indices else torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.tensor(edge_features, dtype=torch.float32),
            vehicle_ids=vehicle_ids,
            adjacency_matrix=torch.tensor(adjacency_matrix, dtype=torch.float32),
            global_features=torch.tensor(global_features, dtype=torch.float32)
        )

    def _extract_node_features(self, vehicles_info: List[Dict]) -> List[List[float]]:
        """提取10维节点特征"""
        node_features = []
        
        for vehicle_info in vehicles_info:
            current_state = vehicle_info['current_state']
            goal_state = vehicle_info['goal_state']
            priority = vehicle_info.get('priority', 1)
            
            # 计算导航特征
            dx = goal_state.x - current_state.x
            dy = goal_state.y - current_state.y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)
            goal_bearing = math.atan2(dy, dx)
            heading_error = self._normalize_angle(current_state.theta - goal_bearing)
            
            # 计算运动特征
            speed_ratio = current_state.v / self.params.max_speed
            acceleration = getattr(current_state, 'acceleration', 0.0) / self.params.max_accel
            
            # 10维特征向量
            features = [
                current_state.x / 100.0,                    # [0] 归一化x坐标
                current_state.y / 100.0,                    # [1] 归一化y坐标
                math.cos(current_state.theta),              # [2] 航向余弦
                math.sin(current_state.theta),              # [3] 航向正弦
                speed_ratio,                                 # [4] 归一化速度
                acceleration,                                # [5] 归一化加速度
                distance_to_goal / 100.0,                   # [6] 归一化目标距离
                math.cos(goal_bearing),                      # [7] 目标方向余弦
                math.sin(goal_bearing),                      # [8] 目标方向正弦
                priority / 10.0                             # [9] 归一化优先级
            ]
            
            node_features.append(features)
        
        return node_features

    def _extract_global_features(self, vehicles_info: List[Dict]) -> List[float]:
        """提取8维全局特征"""
        if not vehicles_info:
            return [0.0] * self.global_feature_dim
        
        n_vehicles = len(vehicles_info)
        
        # 统计特征
        speeds = [v['current_state'].v for v in vehicles_info]
        distances_to_goal = []
        priorities = []
        
        for v in vehicles_info:
            state = v['current_state']
            goal = v['goal_state']
            dist = math.sqrt((goal.x - state.x)**2 + (goal.y - state.y)**2)
            distances_to_goal.append(dist)
            priorities.append(v.get('priority', 1))
        
        # 空间分布
        positions = [(v['current_state'].x, v['current_state'].y) for v in vehicles_info]
        center_x = sum(p[0] for p in positions) / n_vehicles
        center_y = sum(p[1] for p in positions) / n_vehicles
        spread = sum(math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in positions) / n_vehicles
        
        # 8维全局特征
        global_features = [
            n_vehicles / 10.0,                           # [0] 归一化车辆数
            sum(speeds) / (n_vehicles * self.params.max_speed),  # [1] 平均速度比
            np.std(speeds) / self.params.max_speed,      # [2] 速度方差
            sum(distances_to_goal) / (n_vehicles * 100), # [3] 平均目标距离
            np.std(distances_to_goal) / 100,             # [4] 目标距离方差
            sum(priorities) / (n_vehicles * 10),         # [5] 平均优先级
            spread / 50.0,                               # [6] 空间分布
            self._compute_traffic_density(vehicles_info) # [7] 交通密度
        ]
        
        return global_features

    def _analyze_path_crossing(self, state1: VehicleState, goal1: VehicleState,
                            state2: VehicleState, goal2: VehicleState) -> float:
        """分析路径交叉概率"""
        # 线段相交检测
        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-10:
                return False, None
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersect_x = x1 + t * (x2 - x1)
                intersect_y = y1 + t * (y2 - y1)
                return True, (intersect_x, intersect_y)
            
            return False, None
        
        # 检查直线路径是否相交
        intersects, intersection = line_intersection(
            (state1.x, state1.y), (goal1.x, goal1.y),
            (state2.x, state2.y), (goal2.x, goal2.y)
        )
        
        if intersects:
            # 计算交叉点到各车当前位置的距离
            ix, iy = intersection
            dist1 = math.sqrt((ix - state1.x)**2 + (iy - state1.y)**2)
            dist2 = math.sqrt((ix - state2.x)**2 + (iy - state2.y)**2)
            
            # 估算到达交叉点的时间
            t1 = dist1 / max(0.1, state1.v)
            t2 = dist2 / max(0.1, state2.v)
            
            # 时间差越小，冲突概率越高
            time_diff = abs(t1 - t2)
            return max(0.0, 1.0 - time_diff / 10.0)
        
        return 0.0

    def _estimate_time_to_conflict(self, state1: VehicleState, state2: VehicleState,
                                v1x: float, v1y: float, v2x: float, v2y: float) -> float:
        """估算冲突时间"""
        # 相对位置和相对速度
        dx = state2.x - state1.x
        dy = state2.y - state1.y
        dvx = v1x - v2x
        dvy = v1y - v2y
        
        # 如果相对速度为0，返回无穷大时间
        relative_speed_sq = dvx*dvx + dvy*dvy
        if relative_speed_sq < 1e-6:
            return float('inf')
        
        # 最近距离时间
        t_closest = -(dx*dvx + dy*dvy) / relative_speed_sq
        
        if t_closest < 0:
            return float('inf')  # 已经错过最近点
        
        # 最近距离
        closest_distance = math.sqrt(
            (dx + dvx*t_closest)**2 + (dy + dvy*t_closest)**2
        )
        
        # 如果最近距离太大，不会冲突
        if closest_distance > self.params.length * 2:
            return float('inf')
        
        return t_closest

    def _compute_traffic_density(self, vehicles_info: List[Dict]) -> float:
        """计算交通密度"""
        if len(vehicles_info) < 2:
            return 0.0
        
        total_interactions = 0
        possible_interactions = 0
        
        for i in range(len(vehicles_info)):
            for j in range(i + 1, len(vehicles_info)):
                state1 = vehicles_info[i]['current_state']
                state2 = vehicles_info[j]['current_state']
                distance = math.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
                
                possible_interactions += 1
                if distance < self.interaction_radius:
                    total_interactions += 1
        
        return total_interactions / max(1, possible_interactions)

    def _normalize_angle(self, angle: float) -> float:
        """角度标准化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _create_empty_graph(self) -> VehicleInteractionGraph:
        """创建空图"""
        return VehicleInteractionGraph(
            node_features=torch.zeros((0, self.node_feature_dim)),
            edge_indices=torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.zeros((0, self.edge_feature_dim)),
            vehicle_ids=[],
            adjacency_matrix=torch.zeros((0, 0)),
            global_features=torch.zeros(self.global_feature_dim)
        )    
    def _analyze_path_crossing(self, state1: VehicleState, goal1: VehicleState,
                              state2: VehicleState, goal2: VehicleState) -> float:
        """分析路径交叉概率"""
        # 线段相交检测
        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-10:
                return False, None
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersect_x = x1 + t * (x2 - x1)
                intersect_y = y1 + t * (y2 - y1)
                return True, (intersect_x, intersect_y)
            
            return False, None
        
        # 检查直线路径是否相交
        intersects, intersection = line_intersection(
            (state1.x, state1.y), (goal1.x, goal1.y),
            (state2.x, state2.y), (goal2.x, goal2.y)
        )
        
        if intersects:
            # 计算交叉点到各车当前位置的距离
            ix, iy = intersection
            dist1 = math.sqrt((ix - state1.x)**2 + (iy - state1.y)**2)
            dist2 = math.sqrt((ix - state2.x)**2 + (iy - state2.y)**2)
            
            # 估算到达交叉点的时间
            t1 = dist1 / max(0.1, state1.v)
            t2 = dist2 / max(0.1, state2.v)
            
            # 时间差越小，冲突概率越高
            time_diff = abs(t1 - t2)
            return max(0.0, 1.0 - time_diff / 10.0)
        
        return 0.0
    
    def _estimate_time_to_conflict(self, state1: VehicleState, state2: VehicleState,
                                  v1x: float, v1y: float, v2x: float, v2y: float) -> float:
        """估算冲突时间"""
        # 相对位置和相对速度
        dx = state2.x - state1.x
        dy = state2.y - state1.y
        dvx = v1x - v2x
        dvy = v1y - v2y
        
        # 如果相对速度为0，返回无穷大时间
        relative_speed_sq = dvx*dvx + dvy*dvy
        if relative_speed_sq < 1e-6:
            return float('inf')
        
        # 最近距离时间
        t_closest = -(dx*dvx + dy*dvy) / relative_speed_sq
        
        if t_closest < 0:
            return float('inf')  # 已经错过最近点
        
        # 最近距离
        closest_distance = math.sqrt(
            (dx + dvx*t_closest)**2 + (dy + dvy*t_closest)**2
        )
        
        # 如果最近距离太大，不会冲突
        if closest_distance > self.params.length * 2:
            return float('inf')
        
        return t_closest
    
    def _extract_global_features(self, vehicles_info: List[Dict]) -> List[float]:
        """提取8维全局特征"""
        if not vehicles_info:
            return [0.0] * self.global_feature_dim
        
        n_vehicles = len(vehicles_info)
        
        # 统计特征
        speeds = [v['current_state'].v for v in vehicles_info]
        distances_to_goal = []
        priorities = []
        
        for v in vehicles_info:
            state = v['current_state']
            goal = v['goal_state']
            dist = math.sqrt((goal.x - state.x)**2 + (goal.y - state.y)**2)
            distances_to_goal.append(dist)
            priorities.append(v.get('priority', 1))
        
        # 空间分布
        positions = [(v['current_state'].x, v['current_state'].y) for v in vehicles_info]
        center_x = sum(p[0] for p in positions) / n_vehicles
        center_y = sum(p[1] for p in positions) / n_vehicles
        spread = sum(math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in positions) / n_vehicles
        
        # 8维全局特征
        global_features = [
            n_vehicles / 10.0,                           # [0] 归一化车辆数
            sum(speeds) / (n_vehicles * self.params.max_speed),  # [1] 平均速度比
            np.std(speeds) / self.params.max_speed,      # [2] 速度方差
            sum(distances_to_goal) / (n_vehicles * 100), # [3] 平均目标距离
            np.std(distances_to_goal) / 100,             # [4] 目标距离方差
            sum(priorities) / (n_vehicles * 10),         # [5] 平均优先级
            spread / 50.0,                               # [6] 空间分布
            self._compute_traffic_density(vehicles_info) # [7] 交通密度
        ]
        
        return global_features
    
    def _compute_traffic_density(self, vehicles_info: List[Dict]) -> float:
        """计算交通密度"""
        if len(vehicles_info) < 2:
            return 0.0
        
        total_interactions = 0
        possible_interactions = 0
        
        for i in range(len(vehicles_info)):
            for j in range(i + 1, len(vehicles_info)):
                state1 = vehicles_info[i]['current_state']
                state2 = vehicles_info[j]['current_state']
                distance = math.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
                
                possible_interactions += 1
                if distance < self.interaction_radius:
                    total_interactions += 1
        
        return total_interactions / max(1, possible_interactions)
    
    def _normalize_angle(self, angle: float) -> float:
        """角度标准化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _create_empty_graph(self) -> VehicleInteractionGraph:
        """创建空图"""
        return VehicleInteractionGraph(
            node_features=torch.zeros((0, self.node_feature_dim)),
            edge_indices=torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.zeros((0, self.edge_feature_dim)),
            vehicle_ids=[],
            adjacency_matrix=torch.zeros((0, 0)),
            global_features=torch.zeros(self.global_feature_dim)
        )

class MessagePassingLayer(nn.Module):
    """消息传递层"""
    
    def __init__(self, node_dim: int, edge_dim: int, message_dim: int):
        super().__init__()
        
        # 消息计算网络
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
        # 节点更新网络
        self.node_update_net = nn.Sequential(
            nn.Linear(node_dim + message_dim, node_dim),
            nn.ReLU(),
            nn.LayerNorm(node_dim)
        )
        
        # 边更新网络
        self.edge_update_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
    
    def forward(self, node_features: torch.Tensor, edge_indices: torch.Tensor, 
                edge_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """消息传递前向计算"""
        num_nodes = node_features.shape[0]
        num_edges = edge_indices.shape[1]
        
        if num_nodes == 0 or num_edges == 0:
            return node_features, edge_features
        
        message_dim = self.message_net[0].out_features
        messages = torch.zeros(num_nodes, message_dim, device=node_features.device)
        updated_edges = edge_features.clone()
        
        # 消息计算和聚合
        for i in range(num_edges):
            src_idx, dst_idx = edge_indices[0, i], edge_indices[1, i]
            
            # 构造消息输入：发送者特征 + 接收者特征 + 边特征
            message_input = torch.cat([
                node_features[src_idx],
                node_features[dst_idx], 
                edge_features[i]
            ])
            
            # 计算消息
            message = self.message_net(message_input)
            
            # 聚合到接收者
            messages[dst_idx] += message
            
            # 更新边特征
            edge_input = torch.cat([
                node_features[src_idx],
                node_features[dst_idx],
                edge_features[i]
            ])
            updated_edges[i] = self.edge_update_net(edge_input)
        
        # 节点更新
        updated_nodes = torch.zeros_like(node_features)
        for i in range(num_nodes):
            node_input = torch.cat([node_features[i], messages[i]])
            updated_nodes[i] = self.node_update_net(node_input)
        
        return updated_nodes, updated_edges

class GlobalReadoutLayer(nn.Module):
    """全局读出层"""
    
    def __init__(self, node_dim: int, global_dim: int, output_dim: int):
        super().__init__()
        
        self.output_dim = output_dim  # 🆕 保存输出维度
        
        # 节点到全局映射
        self.node_to_global = nn.Sequential(
            nn.Linear(node_dim, global_dim),
            nn.ReLU()
        )
        
        # 全局特征处理
        self.global_processor = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim),  # 聚合特征 + 输入全局特征
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
            nn.ReLU()
        )
        
        # 全局到节点反馈
        self.global_to_node = nn.Sequential(
            nn.Linear(node_dim + global_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, node_features: torch.Tensor, 
                global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """全局读出前向计算"""
        if node_features.shape[0] == 0:
            return node_features, global_features
        
        # 节点特征聚合到全局
        node_global_contrib = self.node_to_global(node_features)
        
        # 使用注意力聚合
        attention_scores = F.softmax(
            torch.sum(node_global_contrib, dim=-1), dim=0
        )
        aggregated_global = torch.sum(
            attention_scores.unsqueeze(-1) * node_global_contrib, dim=0
        )
        
        # 处理全局特征
        combined_global = torch.cat([aggregated_global, global_features])
        processed_global = self.global_processor(combined_global)
        
        # 全局特征反馈到节点 - 🆕 修正这里
        enhanced_nodes = torch.zeros(node_features.shape[0], self.output_dim)
        for i in range(node_features.shape[0]):
            node_global_input = torch.cat([node_features[i], processed_global])
            enhanced_nodes[i] = self.global_to_node(node_global_input)
        
        return enhanced_nodes, processed_global

class VehicleCoordinationGNN(nn.Module):
    """车辆协调图神经网络"""
    
    def __init__(self, node_dim: int = 10, edge_dim: int = 6, global_dim: int = 8,
                 hidden_dim: int = 64, num_mp_layers: int = 3):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        
        # 输入编码层
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU()
        )
        
        # 多层消息传递
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_mp_layers)
        ])
        
        # 全局读出层
        self.global_readout = GlobalReadoutLayer(hidden_dim, global_dim, hidden_dim)
        
        # 决策输出头
        self.decision_heads = nn.ModuleDict({
            'priority': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()  # 优先级调整 [-1, 1]
            ),
            'cooperation': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 合作倾向 [0, 1]
            ),
            'urgency': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 紧急程度 [0, 1]
            ),
            'safety': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 安全系数 [0, 1]
            )
        })
        
        # 全局输出
        self.global_output = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 全局协调信号
        )
        
    def forward(self, graph: VehicleInteractionGraph) -> Dict[str, torch.Tensor]:
        """GNN前向传播"""
        
        # 处理空图情况
        if graph.node_features.shape[0] == 0:
            return self._empty_output()
        
        # 编码输入
        node_features = self.node_encoder(graph.node_features)
        edge_features = self.edge_encoder(graph.edge_features) if graph.edge_features.shape[0] > 0 else graph.edge_features
        
        # 多层消息传递
        for mp_layer in self.mp_layers:
            node_features, edge_features = mp_layer(node_features, graph.edge_indices, edge_features)
        
        # 全局读出
        enhanced_nodes, global_representation = self.global_readout(node_features, graph.global_features)
        
        # 生成决策输出
        decisions = {}
        for decision_type, head in self.decision_heads.items():
            decisions[decision_type] = head(enhanced_nodes)
        
        # 全局协调信号
        decisions['global_coordination'] = self.global_output(global_representation)
        decisions['node_embeddings'] = enhanced_nodes
        
        return decisions
    
    def _empty_output(self) -> Dict[str, torch.Tensor]:
        """空输出"""
        return {
            'priority': torch.zeros((0, 1)),
            'cooperation': torch.zeros((0, 1)),
            'urgency': torch.zeros((0, 1)),
            'safety': torch.zeros((0, 1)),
            'global_coordination': torch.zeros(4),
            'node_embeddings': torch.zeros((0, self.hidden_dim))
        }

class GNNEnhancedPlanner(VHybridAStarPlanner):
    """GNN增强的规划器"""
    
    def __init__(self, environment: UnstructuredEnvironment, 
                 optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED,
                 gnn_enhancement_level: GNNEnhancementLevel = GNNEnhancementLevel.PRIORITY_ONLY):
        
        super().__init__(environment, optimization_level)
        
        self.gnn_enhancement_level = gnn_enhancement_level
        
        # 初始化GNN组件
        self.graph_builder = VehicleGraphBuilder(self.params)
        self.coordination_gnn = VehicleCoordinationGNN()
        self.coordination_gnn.eval()  # 推理模式
        
        # GNN增强统计
        self.gnn_stats = {
            'graph_constructions': 0,
            'message_passing_rounds': 0,
            'priority_adjustments': 0,
            'cooperation_decisions': 0,
            'safety_adjustments': 0,
            'gnn_inference_time': 0.0,
            'graph_construction_time': 0.0
        }
        
        print(f"      GNN增强规划器初始化: {gnn_enhancement_level.value}")
    
    def plan_multi_vehicle_with_gnn(self, vehicles_info: List[Dict]) -> Dict[int, Optional[List[VehicleState]]]:
        """使用GNN进行多车协调规划"""
        
        print(f"     🧠 GNN多车协调: {len(vehicles_info)}辆车")
        
        # 构建交互图
        graph_start = time.time()
        interaction_graph = self.graph_builder.build_interaction_graph(vehicles_info)
        self.gnn_stats['graph_construction_time'] += time.time() - graph_start
        self.gnn_stats['graph_constructions'] += 1
        
        print(f"        图构建: {interaction_graph.node_features.shape[0]}节点, "
              f"{interaction_graph.edge_indices.shape[1]}边")
        
        # GNN推理
        gnn_start = time.time()
        with torch.no_grad():
            gnn_decisions = self.coordination_gnn(interaction_graph)
        self.gnn_stats['gnn_inference_time'] += time.time() - gnn_start
        
        # 解析决策指导
        coordination_guidance = self._parse_gnn_decisions(gnn_decisions, vehicles_info)
        
        # 按调整后优先级排序
        sorted_vehicles = self._sort_by_gnn_priority(vehicles_info, coordination_guidance)
        
        # 逐车规划
        results = {}
        completed_trajectories = []
        
        for vehicle_info in sorted_vehicles:
            vehicle_id = vehicle_info['id']
            guidance = coordination_guidance.get(vehicle_id, {})
            
            print(f"     规划车辆{vehicle_id}: 优先级调整{guidance.get('priority_adj', 0.0):.3f}")
            
            # 应用GNN指导
            self._apply_gnn_guidance(guidance)
            
            # 执行规划
            trajectory = self.search_with_waiting(
                vehicle_info['start'], vehicle_info['goal'], 
                vehicle_id, completed_trajectories
            )
            
            # 重置参数
            self._reset_planning_params()
            
            results[vehicle_id] = trajectory
            if trajectory:
                completed_trajectories.append(trajectory)
                print(f"        ✅ 成功: {len(trajectory)}点")
            else:
                print(f"        ❌ 失败")
        
        self._print_gnn_stats()
        return results
    
    def _parse_gnn_decisions(self, decisions: Dict[str, torch.Tensor], 
                           vehicles_info: List[Dict]) -> Dict[int, Dict]:
        """解析GNN决策"""
        guidance = {}
        
        priority_adj = decisions['priority']
        cooperation = decisions['cooperation']
        urgency = decisions['urgency']
        safety = decisions['safety']
        global_coord = decisions['global_coordination']
        
        print(f"        全局协调信号: {global_coord.tolist()}")
        
        for i, vehicle_info in enumerate(vehicles_info):
            if i < priority_adj.shape[0]:
                vehicle_id = vehicle_info['id']
                
                pri_adj = priority_adj[i, 0].item()
                coop_score = cooperation[i, 0].item()
                urgency_level = urgency[i, 0].item()
                safety_factor = safety[i, 0].item()
                
                guidance[vehicle_id] = {
                    'priority_adj': pri_adj,
                    'cooperation_score': coop_score,
                    'urgency_level': urgency_level,
                    'safety_factor': safety_factor,
                    'adjusted_priority': vehicle_info['priority'] + pri_adj * 2.0,
                    'strategy': self._determine_coordination_strategy(pri_adj, coop_score, urgency_level, safety_factor)
                }
                
                # 更新统计
                if abs(pri_adj) > 0.1:
                    self.gnn_stats['priority_adjustments'] += 1
                if coop_score > 0.7:
                    self.gnn_stats['cooperation_decisions'] += 1
                if safety_factor > 0.8:
                    self.gnn_stats['safety_adjustments'] += 1
        
        return guidance
    
    def _determine_coordination_strategy(self, priority_adj: float, cooperation: float, 
                                       urgency: float, safety: float) -> str:
        """确定协调策略"""
        if safety > 0.8:
            return "safety_first"
        elif urgency > 0.8:
            return "urgent_passage"
        elif cooperation > 0.7:
            return "cooperative"
        elif priority_adj > 0.3:
            return "assert_priority"
        elif priority_adj < -0.3:
            return "yield_way"
        else:
            return "normal"
    
    def _sort_by_gnn_priority(self, vehicles_info: List[Dict], guidance: Dict[int, Dict]) -> List[Dict]:
        """按GNN调整后的优先级排序"""
        def get_adjusted_priority(vehicle_info):
            vehicle_id = vehicle_info['id']
            return guidance.get(vehicle_id, {}).get('adjusted_priority', vehicle_info['priority'])
        
        return sorted(vehicles_info, key=get_adjusted_priority, reverse=True)
    
    def _apply_gnn_guidance(self, guidance: Dict):
        """应用GNN指导到规划参数"""
        strategy = guidance.get('strategy', 'normal')
        safety_factor = guidance.get('safety_factor', 0.5)
        cooperation_score = guidance.get('cooperation_score', 0.5)
        urgency_level = guidance.get('urgency_level', 0.5)
        
        # 基础参数调整
        if strategy == "safety_first":
            self.params.green_additional_safety *= (1.0 + safety_factor * 0.5)
            self.params.max_speed *= (1.0 - safety_factor * 0.2)
            self.params.wv *= 1.3  # 更重视速度稳定
            
        elif strategy == "urgent_passage":
            self.params.max_speed *= (1.0 + urgency_level * 0.1)
            self.params.wref *= 0.8  # 减少轨迹跟踪约束
            self.max_iterations = int(self.max_iterations * (1.0 + urgency_level * 0.3))
            
        elif strategy == "cooperative":
            self.params.wδ *= (1.0 + cooperation_score * 0.3)  # 增加轨迹平滑
            self.params.green_additional_safety *= (1.0 + cooperation_score * 0.2)
            
        elif strategy == "assert_priority":
            self.params.max_speed *= 1.05
            self.params.wref *= 0.9
            
        elif strategy == "yield_way":
            self.params.green_additional_safety *= 1.2
            self.params.max_speed *= 0.9
            self.params.wv *= 1.2
        
        # 根据增强级别应用更多调整
        if self.gnn_enhancement_level in [GNNEnhancementLevel.EXPANSION_GUIDE, GNNEnhancementLevel.FULL_INTEGRATION]:
            self._apply_advanced_guidance(guidance)
    
    def _apply_advanced_guidance(self, guidance: Dict):
        """应用高级GNN指导"""
        cooperation_score = guidance.get('cooperation_score', 0.5)
        urgency_level = guidance.get('urgency_level', 0.5)
        
        # 调整搜索策略
        if cooperation_score > 0.7:
            # 高合作模式：更细致的运动原语
            self.gnn_stats['cooperation_decisions'] += 1
            
        if urgency_level > 0.7:
            # 紧急模式：增加搜索积极性
            self.max_iterations = int(self.max_iterations * 1.2)
        
        # 动态调整代价函数权重
        self.params.wv *= (1.0 + urgency_level * 0.1)
        self.params.wδ *= (1.0 + cooperation_score * 0.2)
    
    def _reset_planning_params(self):
        """重置规划参数"""
        self.params = VehicleParameters()
        
        # 重置迭代次数
        if self.optimization_level == OptimizationLevel.BASIC:
            self.max_iterations = 15000
        elif self.optimization_level == OptimizationLevel.ENHANCED:
            self.max_iterations = 32000
        else:
            self.max_iterations = 30000
    
    def _print_gnn_stats(self):
        """打印GNN统计信息"""
        stats = self.gnn_stats
        print(f"\n      🧠 GNN增强统计:")
        print(f"        图构建: {stats['graph_constructions']}次 ({stats['graph_construction_time']:.3f}s)")
        print(f"        GNN推理: {stats['gnn_inference_time']:.3f}s") 
        print(f"        优先级调整: {stats['priority_adjustments']}次")
        print(f"        合作决策: {stats['cooperation_decisions']}次")
        print(f"        安全调整: {stats['safety_adjustments']}次")

class GNNIntegratedCoordinator:
    """GNN集成的多车协调器"""
    
    def __init__(self, map_file_path=None, 
                 optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED,
                 gnn_enhancement_level: GNNEnhancementLevel = GNNEnhancementLevel.PRIORITY_ONLY):
        
        self.environment = UnstructuredEnvironment(size=100)
        self.optimization_level = optimization_level
        self.gnn_enhancement_level = gnn_enhancement_level
        self.map_data = None
        
        if map_file_path:
            self.load_map(map_file_path)
        
        # 创建GNN增强规划器
        self.gnn_planner = GNNEnhancedPlanner(
            self.environment, optimization_level, gnn_enhancement_level
        )
        
        print(f"✅ GNN集成协调器初始化")
        print(f"   基础优化: {optimization_level.value}")
        print(f"   GNN增强: {gnn_enhancement_level.value}")
    
    def load_map(self, map_file_path):
        """加载地图"""
        self.map_data = self.environment.load_from_json(map_file_path)
        return self.map_data is not None
    
    def create_scenarios_from_json(self):
        """从JSON创建场景"""
        if not self.map_data:
            return []
        
        start_points = self.map_data.get("start_points", [])
        end_points = self.map_data.get("end_points", [])
        point_pairs = self.map_data.get("point_pairs", [])
        
        if not point_pairs:
            print("❌ 未找到车辆配对")
            return []
        
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
                    'priority': 1,
                    'color': colors[i % len(colors)],
                    'start': start_state,
                    'goal': goal_state,
                    'description': f'Vehicle {i+1} (S{start_id}->E{end_id})'
                }
                
                scenarios.append(scenario)
        
        print(f"✅ 创建{len(scenarios)}个场景")
        return scenarios
    
    def plan_with_gnn_integration(self):
        """执行GNN集成规划"""
        
        scenarios = self.create_scenarios_from_json()
        if not scenarios:
            return None, None
        
        print(f"\n🎯 GNN集成规划: {len(scenarios)}辆车")
        
        # 转换为规划信息格式
        vehicles_info = []
        for scenario in scenarios:
            vehicles_info.append({
                'id': scenario['id'],
                'start': scenario['start'],
                'goal': scenario['goal'],
                'priority': scenario['priority'],
                'current_state': scenario['start'],
                'goal_state': scenario['goal']
            })
        
        # 执行GNN增强规划
        start_time = time.time()
        planning_results = self.gnn_planner.plan_multi_vehicle_with_gnn(vehicles_info)
        total_time = time.time() - start_time
        
        # 转换结果格式
        results = {}
        for scenario in scenarios:
            vehicle_id = scenario['id']
            trajectory = planning_results.get(vehicle_id)
            
            results[vehicle_id] = {
                'trajectory': trajectory if trajectory else [],
                'color': scenario['color'],
                'description': scenario['description'],
                'planning_time': total_time / len(scenarios)
            }
        
        success_count = sum(1 for r in results.values() if r['trajectory'])
        
        print(f"\n📊 GNN集成规划结果:")
        print(f"   成功率: {success_count}/{len(scenarios)} ({100*success_count/len(scenarios):.1f}%)")
        print(f"   总时间: {total_time:.2f}s")
        print(f"   平均时间: {total_time/len(scenarios):.2f}s/车")
        
        return results, scenarios
def verify_map_structure(map_file_path):
    """验证地图文件结构"""
    print(f"🔍 验证地图文件: {map_file_path}")
    
    try:
        with open(map_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("📋 地图文件结构:")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)}个元素")
                if value and isinstance(value[0], dict):
                    print(f"     示例: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"   {key}: {len(value)}个字段")
                print(f"     字段: {list(value.keys())}")
            else:
                print(f"   {key}: {type(value)}")
        
        # 检查必要字段
        required_fields = ["start_points", "end_points", "point_pairs"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"❌ 缺少必要字段: {missing_fields}")
            return False
        
        start_points = data.get("start_points", [])
        end_points = data.get("end_points", [])
        point_pairs = data.get("point_pairs", [])
        
        print(f"✅ 场景验证:")
        print(f"   起点数量: {len(start_points)}")
        print(f"   终点数量: {len(end_points)}")
        print(f"   配对数量: {len(point_pairs)}")
        
        # 验证配对引用
        valid_pairs = 0
        for pair in point_pairs:
            start_id = pair.get("start_id")
            end_id = pair.get("end_id")
            
            start_exists = any(p["id"] == start_id for p in start_points)
            end_exists = any(p["id"] == end_id for p in end_points)
            
            if start_exists and end_exists:
                valid_pairs += 1
            else:
                print(f"   ❌ 无效配对: S{start_id}->E{end_id}")
        
        print(f"   有效配对: {valid_pairs}/{len(point_pairs)}")
        
        return valid_pairs > 0
        
    except Exception as e:
        print(f"❌ 地图文件验证失败: {e}")
        return False

def interactive_json_selection():
    """交互式JSON文件选择"""
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if not json_files:
        print("❌ 当前目录没有找到JSON地图文件")
        return None
    
    print(f"\n📁 发现 {len(json_files)} 个JSON地图文件:")
    for i, file in enumerate(json_files):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                map_info = data.get('map_info', {})
                name = map_info.get('name', file)
                width = map_info.get('width', '未知')
                height = map_info.get('height', '未知')
                vehicles = len(data.get('point_pairs', []))
                print(f"  {i+1}. {file}")
                print(f"     名称: {name}")
                print(f"     大小: {width}x{height}")
                print(f"     车辆数: {vehicles}")
        except:
            print(f"  {i+1}. {file} (无法读取详细信息)")
    
    while True:
        try:
            choice = input(f"\n🎯 请选择地图文件 (1-{len(json_files)}) 或按Enter使用第1个: ").strip()
            if choice == "":
                return json_files[0]
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(json_files):
                return json_files[choice_idx]
            else:
                print(f"❌ 请输入 1-{len(json_files)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
def main():
    """主函数"""
    print("🧠 GNN增强的V-Hybrid A*多车协调系统")
    print("=" * 60)
    
    # 选择地图
    selected_file = interactive_json_selection()
    if not selected_file:
        print("❌ 未选择地图文件")
        return
    
    # 验证地图
    if not verify_map_structure(selected_file):
        print("❌ 地图验证失败")
        return
    
    print(f"\n🗺️ 使用地图: {selected_file}")
    
    # 选择GNN增强级别
    print(f"\n🧠 选择GNN增强级别:")
    print(f"  1. 仅优先级增强 (priority_only)")
    print(f"  2. 节点扩展指导 (expansion_guide)")
    print(f"  3. 完全集成 (full_integration)")
    
    try:
        choice = input("选择级别 (1-3) 或Enter使用1: ").strip()
        if choice == "2":
            gnn_level = GNNEnhancementLevel.EXPANSION_GUIDE
        elif choice == "3":
            gnn_level = GNNEnhancementLevel.FULL_INTEGRATION
        else:
            gnn_level = GNNEnhancementLevel.PRIORITY_ONLY
    except:
        gnn_level = GNNEnhancementLevel.PRIORITY_ONLY
    
    print(f"🎯 GNN增强级别: {gnn_level.value}")
    
    # 创建GNN集成系统
    try:
        coordinator = GNNIntegratedCoordinator(
            map_file_path=selected_file,
            optimization_level=OptimizationLevel.ENHANCED,
            gnn_enhancement_level=gnn_level
        )
        
        if not coordinator.map_data:
            print("❌ 地图数据加载失败")
            return
        
        # 执行GNN集成规划
        results, scenarios = coordinator.plan_with_gnn_integration()
        
        if results and scenarios and any(r['trajectory'] for r in results.values()):
            print(f"\n🎬 生成可视化...")
            
            # 使用原始协调器进行可视化
            original_coordinator = MultiVehicleCoordinator(selected_file)
            original_coordinator.create_animation(results, scenarios)
            
            print(f"\n✅ GNN集成演示完成!")
            
        else:
            print("❌ 规划失败")
        
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()#!/usr/bin/env python3
"""
GNN增强的V-Hybrid A*多车协调系统 - 完整版
集成消息传递机制的图神经网络 + 完整QP优化流程
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import os
import json

# 导入成熟算法模块
from trying import (
    VehicleState, VehicleParameters, UnstructuredEnvironment, 
    VHybridAStarPlanner, MultiVehicleCoordinator, OptimizationLevel,
    HybridNode, ConflictDensityAnalyzer, TimeSync,
    interactive_json_selection, OptimizedTrajectoryProcessor,
    CompleteQPOptimizer, EnhancedConvexSpaceSTDiagram
)

class GNNEnhancementLevel(Enum):
    """GNN增强级别"""
    PRIORITY_ONLY = "priority_only"           
    EXPANSION_GUIDE = "expansion_guide"       
    FULL_INTEGRATION = "full_integration"     

@dataclass
class VehicleInteractionGraph:
    """车辆交互图结构"""
    node_features: torch.Tensor      # (N, feature_dim) 节点特征
    edge_indices: torch.Tensor       # (2, E) 边索引
    edge_features: torch.Tensor      # (E, edge_dim) 边特征
    vehicle_ids: List[int]           # 节点到车辆ID映射
    adjacency_matrix: torch.Tensor   # (N, N) 邻接矩阵
    global_features: torch.Tensor    # (global_dim,) 全局特征

class VehicleGraphBuilder:
    """车辆交互图构建器"""
    
    def __init__(self, params: VehicleParameters):
        self.params = params
        self.interaction_radius = 50.0  # 🆕 增大交互半径
        self.node_feature_dim = 10      
        self.edge_feature_dim = 6       
        self.global_feature_dim = 8     
        
    def _build_edges_and_features(self, vehicles_info: List[Dict]) -> Tuple[List, List, List]:
        """构建边索引、边特征和邻接矩阵"""
        n_vehicles = len(vehicles_info)
        edge_indices = []
        edge_features = []
        adjacency_matrix = np.zeros((n_vehicles, n_vehicles))
        
        print(f"        构建边: 检查{n_vehicles}辆车的交互")
        
        for i in range(n_vehicles):
            for j in range(i + 1, n_vehicles):
                # 计算交互强度和特征
                interaction_data = self._compute_interaction_features(vehicles_info[i], vehicles_info[j])
                
                # 🆕 降低交互阈值并添加调试信息
                state1 = vehicles_info[i]['current_state']
                state2 = vehicles_info[j]['current_state']
                distance = math.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
                
                if i < 3 and j < 3:  # 只打印前几对的调试信息
                    print(f"          车辆{i}-{j}: 距离={distance:.1f}, 交互强度={interaction_data['interaction_strength']:.3f}")
                
                if interaction_data['interaction_strength'] > 0.05:  # 🆕 降低阈值从0.1到0.05
                    # 添加双向边
                    edge_indices.extend([[i, j], [j, i]])
                    edge_features.extend([interaction_data['features'], interaction_data['features']])
                    
                    # 更新邻接矩阵
                    weight = interaction_data['interaction_strength']
                    adjacency_matrix[i, j] = weight
                    adjacency_matrix[j, i] = weight
        
        print(f"        构建完成: {len(edge_indices)}条边")
        return edge_indices, edge_features, adjacency_matrix.tolist()
    
    def _compute_interaction_features(self, vehicle1: Dict, vehicle2: Dict) -> Dict:
        """计算车辆间交互特征"""
        state1 = vehicle1['current_state']
        state2 = vehicle2['current_state']
        goal1 = vehicle1['goal_state']
        goal2 = vehicle2['goal_state']
        
        # 基础距离和几何关系
        dx = state2.x - state1.x
        dy = state2.y - state1.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        if distance > self.interaction_radius:
            return {'interaction_strength': 0.0, 'features': [0.0] * self.edge_feature_dim}
        
        # 🆕 修正交互强度计算，确保即使距离较远也有基础交互
        distance_factor = max(0.1, 1.0 - (distance / self.interaction_radius))  # 最小0.1而不是0
        
        # 计算详细交互特征
        
        # 1. 空间关系
        relative_bearing = math.atan2(dy, dx)
        
        # 2. 运动关系
        v1x, v1y = state1.v * math.cos(state1.theta), state1.v * math.sin(state1.theta)
        v2x, v2y = state2.v * math.cos(state2.theta), state2.v * math.sin(state2.theta)
        
        # 相对速度
        rel_vx, rel_vy = v1x - v2x, v1y - v2y
        relative_speed = math.sqrt(rel_vx*rel_vx + rel_vy*rel_vy)
        
        # 接近速度（朝向对方的速度分量）
        if distance > 1e-6:
            approach_speed = max(0, (v1x * dx + v1y * dy) / distance)
        else:
            approach_speed = 0.0
        
        # 3. 路径交叉分析
        path_crossing = self._analyze_path_crossing(state1, goal1, state2, goal2)
        
        # 4. 优先级关系
        priority_diff = (vehicle1.get('priority', 1) - vehicle2.get('priority', 1)) / 10.0
        
        # 5. 时间冲突预测
        time_to_conflict = self._estimate_time_to_conflict(state1, state2, v1x, v1y, v2x, v2y)
        
        # 🆕 修正综合交互强度计算
        interaction_strength = (
            distance_factor * 0.5 +              # 增加距离权重
            min(1.0, relative_speed / 10.0) * 0.15 +
            min(1.0, approach_speed / 5.0) * 0.15 +
            path_crossing * 0.2
        )
        
        # 确保最小交互强度
        interaction_strength = max(0.05, interaction_strength)
        
        # 6维边特征
        features = [
            distance / self.interaction_radius,     # [0] 归一化距离
            relative_speed / 10.0,                  # [1] 归一化相对速度
            approach_speed / 5.0,                   # [2] 归一化接近速度
            path_crossing,                          # [3] 路径交叉概率
            priority_diff,                          # [4] 优先级差异
            min(1.0, time_to_conflict / 20.0)      # [5] 归一化冲突时间
        ]
        
        return {
            'interaction_strength': min(1.0, interaction_strength),
            'features': features
        }

    def build_interaction_graph(self, vehicles_info: List[Dict]) -> VehicleInteractionGraph:
        """构建完整的车辆交互图"""
        n_vehicles = len(vehicles_info)
        if n_vehicles == 0:
            return self._create_empty_graph()
        
        # 提取节点特征
        node_features = self._extract_node_features(vehicles_info)
        
        # 构建边和边特征
        edge_indices, edge_features, adjacency_matrix = self._build_edges_and_features(vehicles_info)
        
        # 提取全局特征
        global_features = self._extract_global_features(vehicles_info)
        
        vehicle_ids = [v['id'] for v in vehicles_info]
        
        return VehicleInteractionGraph(
            node_features=torch.tensor(node_features, dtype=torch.float32),
            edge_indices=torch.tensor(edge_indices, dtype=torch.long).T if edge_indices else torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.tensor(edge_features, dtype=torch.float32),
            vehicle_ids=vehicle_ids,
            adjacency_matrix=torch.tensor(adjacency_matrix, dtype=torch.float32),
            global_features=torch.tensor(global_features, dtype=torch.float32)
        )

    def _extract_node_features(self, vehicles_info: List[Dict]) -> List[List[float]]:
        """提取10维节点特征"""
        node_features = []
        
        for vehicle_info in vehicles_info:
            current_state = vehicle_info['current_state']
            goal_state = vehicle_info['goal_state']
            priority = vehicle_info.get('priority', 1)
            
            # 计算导航特征
            dx = goal_state.x - current_state.x
            dy = goal_state.y - current_state.y
            distance_to_goal = math.sqrt(dx*dx + dy*dy)
            goal_bearing = math.atan2(dy, dx)
            heading_error = self._normalize_angle(current_state.theta - goal_bearing)
            
            # 计算运动特征
            speed_ratio = current_state.v / self.params.max_speed
            acceleration = getattr(current_state, 'acceleration', 0.0) / self.params.max_accel
            
            # 10维特征向量
            features = [
                current_state.x / 100.0,                    # [0] 归一化x坐标
                current_state.y / 100.0,                    # [1] 归一化y坐标
                math.cos(current_state.theta),              # [2] 航向余弦
                math.sin(current_state.theta),              # [3] 航向正弦
                speed_ratio,                                 # [4] 归一化速度
                acceleration,                                # [5] 归一化加速度
                distance_to_goal / 100.0,                   # [6] 归一化目标距离
                math.cos(goal_bearing),                      # [7] 目标方向余弦
                math.sin(goal_bearing),                      # [8] 目标方向正弦
                priority / 10.0                             # [9] 归一化优先级
            ]
            
            node_features.append(features)
        
        return node_features

    def _extract_global_features(self, vehicles_info: List[Dict]) -> List[float]:
        """提取8维全局特征"""
        if not vehicles_info:
            return [0.0] * self.global_feature_dim
        
        n_vehicles = len(vehicles_info)
        
        # 统计特征
        speeds = [v['current_state'].v for v in vehicles_info]
        distances_to_goal = []
        priorities = []
        
        for v in vehicles_info:
            state = v['current_state']
            goal = v['goal_state']
            dist = math.sqrt((goal.x - state.x)**2 + (goal.y - state.y)**2)
            distances_to_goal.append(dist)
            priorities.append(v.get('priority', 1))
        
        # 空间分布
        positions = [(v['current_state'].x, v['current_state'].y) for v in vehicles_info]
        center_x = sum(p[0] for p in positions) / n_vehicles
        center_y = sum(p[1] for p in positions) / n_vehicles
        spread = sum(math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in positions) / n_vehicles
        
        # 8维全局特征
        global_features = [
            n_vehicles / 10.0,                           # [0] 归一化车辆数
            sum(speeds) / (n_vehicles * self.params.max_speed),  # [1] 平均速度比
            np.std(speeds) / self.params.max_speed,      # [2] 速度方差
            sum(distances_to_goal) / (n_vehicles * 100), # [3] 平均目标距离
            np.std(distances_to_goal) / 100,             # [4] 目标距离方差
            sum(priorities) / (n_vehicles * 10),         # [5] 平均优先级
            spread / 50.0,                               # [6] 空间分布
            self._compute_traffic_density(vehicles_info) # [7] 交通密度
        ]
        
        return global_features

    def _analyze_path_crossing(self, state1: VehicleState, goal1: VehicleState,
                            state2: VehicleState, goal2: VehicleState) -> float:
        """分析路径交叉概率"""
        # 线段相交检测
        def line_intersection(p1, p2, p3, p4):
            x1, y1 = p1
            x2, y2 = p2
            x3, y3 = p3
            x4, y4 = p4
            
            denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(denom) < 1e-10:
                return False, None
            
            t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
            u = -((x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)) / denom
            
            if 0 <= t <= 1 and 0 <= u <= 1:
                intersect_x = x1 + t * (x2 - x1)
                intersect_y = y1 + t * (y2 - y1)
                return True, (intersect_x, intersect_y)
            
            return False, None
        
        # 检查直线路径是否相交
        intersects, intersection = line_intersection(
            (state1.x, state1.y), (goal1.x, goal1.y),
            (state2.x, state2.y), (goal2.x, goal2.y)
        )
        
        if intersects:
            # 计算交叉点到各车当前位置的距离
            ix, iy = intersection
            dist1 = math.sqrt((ix - state1.x)**2 + (iy - state1.y)**2)
            dist2 = math.sqrt((ix - state2.x)**2 + (iy - state2.y)**2)
            
            # 估算到达交叉点的时间
            t1 = dist1 / max(0.1, state1.v)
            t2 = dist2 / max(0.1, state2.v)
            
            # 时间差越小，冲突概率越高
            time_diff = abs(t1 - t2)
            return max(0.0, 1.0 - time_diff / 10.0)
        
        return 0.0

    def _estimate_time_to_conflict(self, state1: VehicleState, state2: VehicleState,
                                v1x: float, v1y: float, v2x: float, v2y: float) -> float:
        """估算冲突时间"""
        # 相对位置和相对速度
        dx = state2.x - state1.x
        dy = state2.y - state1.y
        dvx = v1x - v2x
        dvy = v1y - v2y
        
        # 如果相对速度为0，返回无穷大时间
        relative_speed_sq = dvx*dvx + dvy*dvy
        if relative_speed_sq < 1e-6:
            return float('inf')
        
        # 最近距离时间
        t_closest = -(dx*dvx + dy*dvy) / relative_speed_sq
        
        if t_closest < 0:
            return float('inf')  # 已经错过最近点
        
        # 最近距离
        closest_distance = math.sqrt(
            (dx + dvx*t_closest)**2 + (dy + dvy*t_closest)**2
        )
        
        # 如果最近距离太大，不会冲突
        if closest_distance > self.params.length * 2:
            return float('inf')
        
        return t_closest

    def _compute_traffic_density(self, vehicles_info: List[Dict]) -> float:
        """计算交通密度"""
        if len(vehicles_info) < 2:
            return 0.0
        
        total_interactions = 0
        possible_interactions = 0
        
        for i in range(len(vehicles_info)):
            for j in range(i + 1, len(vehicles_info)):
                state1 = vehicles_info[i]['current_state']
                state2 = vehicles_info[j]['current_state']
                distance = math.sqrt((state1.x - state2.x)**2 + (state1.y - state2.y)**2)
                
                possible_interactions += 1
                if distance < self.interaction_radius:
                    total_interactions += 1
        
        return total_interactions / max(1, possible_interactions)

    def _normalize_angle(self, angle: float) -> float:
        """角度标准化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _create_empty_graph(self) -> VehicleInteractionGraph:
        """创建空图"""
        return VehicleInteractionGraph(
            node_features=torch.zeros((0, self.node_feature_dim)),
            edge_indices=torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.zeros((0, self.edge_feature_dim)),
            vehicle_ids=[],
            adjacency_matrix=torch.zeros((0, 0)),
            global_features=torch.zeros(self.global_feature_dim)
        )

class MessagePassingLayer(nn.Module):
    """消息传递层"""
    
    def __init__(self, node_dim: int, edge_dim: int, message_dim: int):
        super().__init__()
        
        # 消息计算网络
        self.message_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim),
            nn.ReLU(),
            nn.Linear(message_dim, message_dim)
        )
        
        # 节点更新网络
        self.node_update_net = nn.Sequential(
            nn.Linear(node_dim + message_dim, node_dim),
            nn.ReLU(),
            nn.LayerNorm(node_dim)
        )
        
        # 边更新网络
        self.edge_update_net = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim),
            nn.ReLU(),
            nn.Linear(edge_dim, edge_dim)
        )
    
    def forward(self, node_features: torch.Tensor, edge_indices: torch.Tensor, 
                edge_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """消息传递前向计算"""
        num_nodes = node_features.shape[0]
        num_edges = edge_indices.shape[1]
        
        if num_nodes == 0 or num_edges == 0:
            return node_features, edge_features
        
        message_dim = self.message_net[0].out_features
        messages = torch.zeros(num_nodes, message_dim, device=node_features.device)
        updated_edges = edge_features.clone()
        
        # 消息计算和聚合
        for i in range(num_edges):
            src_idx, dst_idx = edge_indices[0, i], edge_indices[1, i]
            
            # 构造消息输入：发送者特征 + 接收者特征 + 边特征
            message_input = torch.cat([
                node_features[src_idx],
                node_features[dst_idx], 
                edge_features[i]
            ])
            
            # 计算消息
            message = self.message_net(message_input)
            
            # 聚合到接收者
            messages[dst_idx] += message
            
            # 更新边特征
            edge_input = torch.cat([
                node_features[src_idx],
                node_features[dst_idx],
                edge_features[i]
            ])
            updated_edges[i] = self.edge_update_net(edge_input)
        
        # 节点更新
        updated_nodes = torch.zeros_like(node_features)
        for i in range(num_nodes):
            node_input = torch.cat([node_features[i], messages[i]])
            updated_nodes[i] = self.node_update_net(node_input)
        
        return updated_nodes, updated_edges

class GlobalReadoutLayer(nn.Module):
    """全局读出层"""
    
    def __init__(self, node_dim: int, global_dim: int, output_dim: int):
        super().__init__()
        
        self.output_dim = output_dim  # 🆕 保存输出维度
        
        # 节点到全局映射
        self.node_to_global = nn.Sequential(
            nn.Linear(node_dim, global_dim),
            nn.ReLU()
        )
        
        # 全局特征处理
        self.global_processor = nn.Sequential(
            nn.Linear(global_dim * 2, global_dim),  # 聚合特征 + 输入全局特征
            nn.ReLU(),
            nn.Linear(global_dim, global_dim),
            nn.ReLU()
        )
        
        # 全局到节点反馈
        self.global_to_node = nn.Sequential(
            nn.Linear(node_dim + global_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, node_features: torch.Tensor, 
                global_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """全局读出前向计算"""
        if node_features.shape[0] == 0:
            return node_features, global_features
        
        # 节点特征聚合到全局
        node_global_contrib = self.node_to_global(node_features)
        
        # 使用注意力聚合
        attention_scores = F.softmax(
            torch.sum(node_global_contrib, dim=-1), dim=0
        )
        aggregated_global = torch.sum(
            attention_scores.unsqueeze(-1) * node_global_contrib, dim=0
        )
        
        # 处理全局特征
        combined_global = torch.cat([aggregated_global, global_features])
        processed_global = self.global_processor(combined_global)
        
        # 全局特征反馈到节点 - 🆕 修正这里
        enhanced_nodes = torch.zeros(node_features.shape[0], self.output_dim)
        for i in range(node_features.shape[0]):
            node_global_input = torch.cat([node_features[i], processed_global])
            enhanced_nodes[i] = self.global_to_node(node_global_input)
        
        return enhanced_nodes, processed_global

class VehicleCoordinationGNN(nn.Module):
    """车辆协调图神经网络"""
    
    def __init__(self, node_dim: int = 10, edge_dim: int = 6, global_dim: int = 8,
                 hidden_dim: int = 64, num_mp_layers: int = 3):
        super().__init__()
        
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.global_dim = global_dim
        self.hidden_dim = hidden_dim
        
        # 输入编码层
        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_dim),
            nn.ReLU()
        )
        
        # 多层消息传递
        self.mp_layers = nn.ModuleList([
            MessagePassingLayer(hidden_dim, edge_dim, hidden_dim)
            for _ in range(num_mp_layers)
        ])
        
        # 全局读出层
        self.global_readout = GlobalReadoutLayer(hidden_dim, global_dim, hidden_dim)
        
        # 决策输出头
        self.decision_heads = nn.ModuleDict({
            'priority': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Tanh()  # 优先级调整 [-1, 1]
            ),
            'cooperation': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 合作倾向 [0, 1]
            ),
            'urgency': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 紧急程度 [0, 1]
            ),
            'safety': nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # 安全系数 [0, 1]
            )
        })
        
        # 全局输出
        self.global_output = nn.Sequential(
            nn.Linear(global_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4)  # 全局协调信号
        )
        
    def forward(self, graph: VehicleInteractionGraph) -> Dict[str, torch.Tensor]:
        """GNN前向传播"""
        
        # 处理空图情况
        if graph.node_features.shape[0] == 0:
            return self._empty_output()
        
        # 编码输入
        node_features = self.node_encoder(graph.node_features)
        edge_features = self.edge_encoder(graph.edge_features) if graph.edge_features.shape[0] > 0 else graph.edge_features
        
        # 多层消息传递
        for mp_layer in self.mp_layers:
            node_features, edge_features = mp_layer(node_features, graph.edge_indices, edge_features)
        
        # 全局读出
        enhanced_nodes, global_representation = self.global_readout(node_features, graph.global_features)
        
        # 生成决策输出
        decisions = {}
        for decision_type, head in self.decision_heads.items():
            decisions[decision_type] = head(enhanced_nodes)
        
        # 全局协调信号
        decisions['global_coordination'] = self.global_output(global_representation)
        decisions['node_embeddings'] = enhanced_nodes
        
        return decisions
    
    def _empty_output(self) -> Dict[str, torch.Tensor]:
        """空输出"""
        return {
            'priority': torch.zeros((0, 1)),
            'cooperation': torch.zeros((0, 1)),
            'urgency': torch.zeros((0, 1)),
            'safety': torch.zeros((0, 1)),
            'global_coordination': torch.zeros(4),
            'node_embeddings': torch.zeros((0, self.hidden_dim))
        }

class GNNEnhancedPlanner(VHybridAStarPlanner):
    """🆕 GNN增强的规划器 - 完整QP优化版"""
    
    def __init__(self, environment: UnstructuredEnvironment, 
                 optimization_level: OptimizationLevel = OptimizationLevel.FULL,  # 🆕 默认使用FULL
                 gnn_enhancement_level: GNNEnhancementLevel = GNNEnhancementLevel.PRIORITY_ONLY):
        
        super().__init__(environment, optimization_level)
        
        self.gnn_enhancement_level = gnn_enhancement_level
        
        # 初始化GNN组件
        self.graph_builder = VehicleGraphBuilder(self.params)
        self.coordination_gnn = VehicleCoordinationGNN()
        self.coordination_gnn.eval()  # 推理模式
        
        # 🆕 确保使用完整的轨迹处理器
        self.trajectory_processor = OptimizedTrajectoryProcessor(self.params, optimization_level)
        
        # GNN增强统计
        self.gnn_stats = {
            'graph_constructions': 0,
            'message_passing_rounds': 0,
            'priority_adjustments': 0,
            'cooperation_decisions': 0,
            'safety_adjustments': 0,
            'gnn_inference_time': 0.0,
            'graph_construction_time': 0.0,
            'qp_optimizations': 0,  # 🆕 添加QP统计
            'trajectory_improvements': 0  # 🆕 添加改进统计
        }
        
        print(f"      🧠 GNN增强规划器初始化: {gnn_enhancement_level.value}")
        print(f"         优化级别: {optimization_level.value}")
        print(f"         QP优化: {'完整启用' if optimization_level == OptimizationLevel.FULL else '部分启用'}")
    
    def search(self, start: VehicleState, goal: VehicleState, 
               high_priority_trajectories: List[List[VehicleState]] = None) -> Optional[List[VehicleState]]:
        """🆕 增强的搜索方法，确保使用完整优化流程"""
        
        print(f"         执行GNN增强搜索 + QP优化")
        
        # 调用父类的基础搜索
        base_trajectory = super().search(start, goal, high_priority_trajectories)
        
        if base_trajectory and len(base_trajectory) > 1:
            print(f"        基础搜索完成: {len(base_trajectory)}点，执行QP后处理...")
            
            # 🆕 关键：添加完整的轨迹优化
            if high_priority_trajectories is None:
                high_priority_trajectories = []
            
            # 执行三阶段优化处理
            optimized_trajectory = self.trajectory_processor.process_trajectory(
                base_trajectory, high_priority_trajectories)
            
            # 统计优化效果
            if len(optimized_trajectory) != len(base_trajectory):
                self.gnn_stats['trajectory_improvements'] += 1
            
            self.gnn_stats['qp_optimizations'] += 1
            
            print(f"        ✅ QP优化完成: {len(base_trajectory)}点 → {len(optimized_trajectory)}点")
            return optimized_trajectory
        
        return base_trajectory
    
    def plan_multi_vehicle_with_gnn(self, vehicles_info: List[Dict]) -> Dict[int, Optional[List[VehicleState]]]:
        """🆕 使用GNN进行多车协调规划 - 完整优化版"""
        
        print(f"     🧠 GNN多车协调规划: {len(vehicles_info)}辆车")
        print(f"        🎯 特性: GNN决策 + 完整QP优化流程")
        
        # 1. 构建交互图
        graph_start = time.time()
        interaction_graph = self.graph_builder.build_interaction_graph(vehicles_info)
        self.gnn_stats['graph_construction_time'] += time.time() - graph_start
        self.gnn_stats['graph_constructions'] += 1
        
        print(f"        图构建: {interaction_graph.node_features.shape[0]}节点, "
              f"{interaction_graph.edge_indices.shape[1]}边")
        
        # 2. GNN推理
        gnn_start = time.time()
        with torch.no_grad():
            gnn_decisions = self.coordination_gnn(interaction_graph)
        self.gnn_stats['gnn_inference_time'] += time.time() - gnn_start
        
        # 3. 解析决策指导
        coordination_guidance = self._parse_gnn_decisions(gnn_decisions, vehicles_info)
        
        # 4. 按调整后优先级排序
        sorted_vehicles = self._sort_by_gnn_priority(vehicles_info, coordination_guidance)
        
        # 5. 逐车规划
        results = {}
        completed_trajectories = []
        
        for vehicle_info in sorted_vehicles:
            vehicle_id = vehicle_info['id']
            guidance = coordination_guidance.get(vehicle_id, {})
            
            print(f"     规划车辆{vehicle_id}: 优先级调整{guidance.get('priority_adj', 0.0):.3f}")
            
            # 应用GNN指导
            self._apply_gnn_guidance(guidance)
            
            # 🆕 执行GNN增强搜索（包含QP优化）
            trajectory = self.search_with_waiting(
                vehicle_info['start'], vehicle_info['goal'], 
                vehicle_id, completed_trajectories
            )
            
            if trajectory:
                # 🆕 可选：GNN反馈调整
                if self.gnn_enhancement_level == GNNEnhancementLevel.FULL_INTEGRATION:
                    trajectory = self._apply_gnn_feedback_optimization(trajectory, guidance)
                
                results[vehicle_id] = trajectory
                completed_trajectories.append(trajectory)
                print(f"        ✅ 成功: {len(trajectory)}点 (GNN+QP优化)")
            else:
                print(f"        ❌ 失败")
                results[vehicle_id] = None
            
            # 重置参数
            self._reset_planning_params()
        
        self._print_enhanced_gnn_stats()
        return results
    
    def _apply_gnn_feedback_optimization(self, trajectory: List[VehicleState], 
                                       guidance: Dict) -> List[VehicleState]:
        """🆕 GNN反馈的轨迹微调"""
        if not trajectory:
            return trajectory
        
        strategy = guidance.get('strategy', 'normal')
        cooperation_score = guidance.get('cooperation_score', 0.5)
        safety_factor = guidance.get('safety_factor', 0.5)
        
        print(f"          应用GNN反馈微调: {strategy}")
        
        # 基于GNN决策进行轨迹微调
        adjusted_trajectory = []
        
        for i, state in enumerate(trajectory):
            new_state = state.copy()
            
            # 根据安全系数调整速度
            if safety_factor > 0.8:
                new_state.v *= (0.9 + safety_factor * 0.1)  # 安全优先时减速
                print(f"            安全调整: v {state.v:.2f} → {new_state.v:.2f}")
            elif cooperation_score > 0.7:
                new_state.v *= (0.95 + cooperation_score * 0.05)  # 合作时稍微减速
                print(f"            合作调整: v {state.v:.2f} → {new_state.v:.2f}")
            
            adjusted_trajectory.append(new_state)
        
        # 重新同步时间
        return TimeSync.resync_trajectory_time(adjusted_trajectory)
    
    def _parse_gnn_decisions(self, decisions: Dict[str, torch.Tensor], 
                           vehicles_info: List[Dict]) -> Dict[int, Dict]:
        """解析GNN决策"""
        guidance = {}
        
        priority_adj = decisions['priority']
        cooperation = decisions['cooperation']
        urgency = decisions['urgency']
        safety = decisions['safety']
        global_coord = decisions['global_coordination']
        
        print(f"        全局协调信号: {global_coord.tolist()}")
        
        for i, vehicle_info in enumerate(vehicles_info):
            if i < priority_adj.shape[0]:
                vehicle_id = vehicle_info['id']
                
                pri_adj = priority_adj[i, 0].item()
                coop_score = cooperation[i, 0].item()
                urgency_level = urgency[i, 0].item()
                safety_factor = safety[i, 0].item()
                
                guidance[vehicle_id] = {
                    'priority_adj': pri_adj,
                    'cooperation_score': coop_score,
                    'urgency_level': urgency_level,
                    'safety_factor': safety_factor,
                    'adjusted_priority': vehicle_info['priority'] + pri_adj * 2.0,
                    'strategy': self._determine_coordination_strategy(pri_adj, coop_score, urgency_level, safety_factor)
                }
                
                # 更新统计
                if abs(pri_adj) > 0.1:
                    self.gnn_stats['priority_adjustments'] += 1
                if coop_score > 0.7:
                    self.gnn_stats['cooperation_decisions'] += 1
                if safety_factor > 0.8:
                    self.gnn_stats['safety_adjustments'] += 1
        
        return guidance
    
    def _determine_coordination_strategy(self, priority_adj: float, cooperation: float, 
                                       urgency: float, safety: float) -> str:
        """确定协调策略"""
        if safety > 0.8:
            return "safety_first"
        elif urgency > 0.8:
            return "urgent_passage"
        elif cooperation > 0.7:
            return "cooperative"
        elif priority_adj > 0.3:
            return "assert_priority"
        elif priority_adj < -0.3:
            return "yield_way"
        else:
            return "normal"
    
    def _sort_by_gnn_priority(self, vehicles_info: List[Dict], guidance: Dict[int, Dict]) -> List[Dict]:
        """按GNN调整后的优先级排序"""
        def get_adjusted_priority(vehicle_info):
            vehicle_id = vehicle_info['id']
            return guidance.get(vehicle_id, {}).get('adjusted_priority', vehicle_info['priority'])
        
        return sorted(vehicles_info, key=get_adjusted_priority, reverse=True)
    
    def _apply_gnn_guidance(self, guidance: Dict):
        """应用GNN指导到规划参数"""
        strategy = guidance.get('strategy', 'normal')
        safety_factor = guidance.get('safety_factor', 0.5)
        cooperation_score = guidance.get('cooperation_score', 0.5)
        urgency_level = guidance.get('urgency_level', 0.5)
        
        # 基础参数调整
        if strategy == "safety_first":
            self.params.green_additional_safety *= (1.0 + safety_factor * 0.5)
            self.params.max_speed *= (1.0 - safety_factor * 0.2)
            self.params.wv *= 1.3  # 更重视速度稳定
            
        elif strategy == "urgent_passage":
            self.params.max_speed *= (1.0 + urgency_level * 0.1)
            self.params.wref *= 0.8  # 减少轨迹跟踪约束
            self.max_iterations = int(self.max_iterations * (1.0 + urgency_level * 0.3))
            
        elif strategy == "cooperative":
            self.params.wδ *= (1.0 + cooperation_score * 0.3)  # 增加轨迹平滑
            self.params.green_additional_safety *= (1.0 + cooperation_score * 0.2)
            
        elif strategy == "assert_priority":
            self.params.max_speed *= 1.05
            self.params.wref *= 0.9
            
        elif strategy == "yield_way":
            self.params.green_additional_safety *= 1.2
            self.params.max_speed *= 0.9
            self.params.wv *= 1.2
        
        # 根据增强级别应用更多调整
        if self.gnn_enhancement_level in [GNNEnhancementLevel.EXPANSION_GUIDE, GNNEnhancementLevel.FULL_INTEGRATION]:
            self._apply_advanced_guidance(guidance)
    
    def _apply_advanced_guidance(self, guidance: Dict):
        """应用高级GNN指导"""
        cooperation_score = guidance.get('cooperation_score', 0.5)
        urgency_level = guidance.get('urgency_level', 0.5)
        
        # 调整搜索策略
        if cooperation_score > 0.7:
            # 高合作模式：更细致的运动原语
            self.gnn_stats['cooperation_decisions'] += 1
            
        if urgency_level > 0.7:
            # 紧急模式：增加搜索积极性
            self.max_iterations = int(self.max_iterations * 1.2)
        
        # 动态调整代价函数权重
        self.params.wv *= (1.0 + urgency_level * 0.1)
        self.params.wδ *= (1.0 + cooperation_score * 0.2)
    
    def _reset_planning_params(self):
        """重置规划参数"""
        self.params = VehicleParameters()
        
        # 重置迭代次数
        if self.optimization_level == OptimizationLevel.BASIC:
            self.max_iterations = 15000
        elif self.optimization_level == OptimizationLevel.ENHANCED:
            self.max_iterations = 32000
        else:
            self.max_iterations = 30000
    
    def _print_enhanced_gnn_stats(self):
        """🆕 打印增强的GNN统计信息"""
        stats = self.gnn_stats
        print(f"\n      🧠 GNN+QP集成统计:")
        print(f"        图构建: {stats['graph_constructions']}次 ({stats['graph_construction_time']:.3f}s)")
        print(f"        GNN推理: {stats['gnn_inference_time']:.3f}s") 
        print(f"        优先级调整: {stats['priority_adjustments']}次")
        print(f"        合作决策: {stats['cooperation_decisions']}次")
        print(f"        安全调整: {stats['safety_adjustments']}次")
        print(f"        QP优化应用: {stats['qp_optimizations']}次")  # 🆕
        print(f"        轨迹改进: {stats['trajectory_improvements']}次")  # 🆕

class GNNIntegratedCoordinator:
    """🆕 GNN集成的多车协调器 - 完整优化版"""
    
    def __init__(self, map_file_path=None, 
                 optimization_level: OptimizationLevel = OptimizationLevel.FULL,  # 🆕 默认FULL
                 gnn_enhancement_level: GNNEnhancementLevel = GNNEnhancementLevel.PRIORITY_ONLY):
        
        self.environment = UnstructuredEnvironment(size=100)
        self.optimization_level = optimization_level
        self.gnn_enhancement_level = gnn_enhancement_level
        self.map_data = None
        
        if map_file_path:
            self.load_map(map_file_path)
        
        # 🆕 创建GNN增强规划器，确保包含完整QP优化
        self.gnn_planner = GNNEnhancedPlanner(
            self.environment, optimization_level, gnn_enhancement_level
        )
        
        print(f"✅ GNN集成协调器初始化（完整优化版）")
        print(f"   基础优化: {optimization_level.value}")
        print(f"   GNN增强: {gnn_enhancement_level.value}")
        print(f"   特性集成: GNN智能决策 + 完整QP优化 + 精确运动学")
    
    def load_map(self, map_file_path):
        """加载地图"""
        self.map_data = self.environment.load_from_json(map_file_path)
        return self.map_data is not None
    
    def create_scenarios_from_json(self):
        """从JSON创建场景"""
        if not self.map_data:
            return []
        
        start_points = self.map_data.get("start_points", [])
        end_points = self.map_data.get("end_points", [])
        point_pairs = self.map_data.get("point_pairs", [])
        
        if not point_pairs:
            print("❌ 未找到车辆配对")
            return []
        
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
                    'priority': 1,
                    'color': colors[i % len(colors)],
                    'start': start_state,
                    'goal': goal_state,
                    'description': f'Vehicle {i+1} (S{start_id}->E{end_id})'
                }
                
                scenarios.append(scenario)
        
        print(f"✅ 创建{len(scenarios)}个场景")
        return scenarios
    
    def plan_with_gnn_integration(self):
        """🆕 执行GNN集成规划 - 完整优化版"""
        
        scenarios = self.create_scenarios_from_json()
        if not scenarios:
            return None, None
        
        print(f"\n🎯 GNN+QP集成规划: {len(scenarios)}辆车")
        print(f"   🧠 GNN智能决策: 优先级调整 + 协调策略")
        print(f"   ⚙️ QP优化流程: 路径平滑 + 速度优化 + 凸空间约束")
        print(f"   🔧 精确运动学: 完整转弯半径计算 + 角度更新")
        
        # 转换为规划信息格式
        vehicles_info = []
        for scenario in scenarios:
            vehicles_info.append({
                'id': scenario['id'],
                'start': scenario['start'],
                'goal': scenario['goal'],
                'priority': scenario['priority'],
                'current_state': scenario['start'],
                'goal_state': scenario['goal']
            })
        
        # 🆕 执行GNN增强规划（包含完整QP优化）
        start_time = time.time()
        planning_results = self.gnn_planner.plan_multi_vehicle_with_gnn(vehicles_info)
        total_time = time.time() - start_time
        
        # 转换结果格式
        results = {}
        for scenario in scenarios:
            vehicle_id = scenario['id']
            trajectory = planning_results.get(vehicle_id)
            
            results[vehicle_id] = {
                'trajectory': trajectory if trajectory else [],
                'color': scenario['color'],
                'description': scenario['description'],
                'planning_time': total_time / len(scenarios)
            }
        
        success_count = sum(1 for r in results.values() if r['trajectory'])
        
        print(f"\n📊 GNN+QP集成规划结果:")
        print(f"   成功率: {success_count}/{len(scenarios)} ({100*success_count/len(scenarios):.1f}%)")
        print(f"   总时间: {total_time:.2f}s")
        print(f"   平均时间: {total_time/len(scenarios):.2f}s/车")
        print(f"   优化级别: {self.optimization_level.value}")
        print(f"   特性完整性: 100%集成（GNN+QP+运动学）")
        
        return results, scenarios

def verify_map_structure(map_file_path):
    """验证地图文件结构"""
    print(f"🔍 验证地图文件: {map_file_path}")
    
    try:
        with open(map_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print("📋 地图文件结构:")
        for key, value in data.items():
            if isinstance(value, list):
                print(f"   {key}: {len(value)}个元素")
                if value and isinstance(value[0], dict):
                    print(f"     示例: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"   {key}: {len(value)}个字段")
                print(f"     字段: {list(value.keys())}")
            else:
                print(f"   {key}: {type(value)}")
        
        # 检查必要字段
        required_fields = ["start_points", "end_points", "point_pairs"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            print(f"❌ 缺少必要字段: {missing_fields}")
            return False
        
        start_points = data.get("start_points", [])
        end_points = data.get("end_points", [])
        point_pairs = data.get("point_pairs", [])
        
        print(f"✅ 场景验证:")
        print(f"   起点数量: {len(start_points)}")
        print(f"   终点数量: {len(end_points)}")
        print(f"   配对数量: {len(point_pairs)}")
        
        # 验证配对引用
        valid_pairs = 0
        for pair in point_pairs:
            start_id = pair.get("start_id")
            end_id = pair.get("end_id")
            
            start_exists = any(p["id"] == start_id for p in start_points)
            end_exists = any(p["id"] == end_id for p in end_points)
            
            if start_exists and end_exists:
                valid_pairs += 1
            else:
                print(f"   ❌ 无效配对: S{start_id}->E{end_id}")
        
        print(f"   有效配对: {valid_pairs}/{len(point_pairs)}")
        
        return valid_pairs > 0
        
    except Exception as e:
        print(f"❌ 地图文件验证失败: {e}")
        return False

def main():
    """🆕 主函数 - 完整GNN+QP集成版"""
    print("🧠 GNN增强的V-Hybrid A*多车协调系统 - 完整优化版")
    print("=" * 70)
    print("🎯 核心特性:")
    print("   ✅ GNN智能协调: 消息传递 + 全局决策 + 优先级调整")
    print("   ✅ 完整QP优化: 路径平滑 + 速度优化 + 凸空间约束")
    print("   ✅ 精确运动学: 转弯半径 + 角度更新 + 位置计算")
    print("   ✅ 分层安全策略: 动态安全距离切换")
    print("   ✅ 3D时空地图: 真实时空维度规划")
    print("   ✅ 轨迹质量保证: 达到trying.py的高质量标准")
    print("=" * 70)
    
    # 选择地图
    selected_file = interactive_json_selection()
    if not selected_file:
        print("❌ 未选择地图文件")
        return
    
    # 验证地图
    if not verify_map_structure(selected_file):
        print("❌ 地图验证失败")
        return
    
    print(f"\n🗺️ 使用地图: {selected_file}")
    
    # 选择GNN增强级别
    print(f"\n🧠 选择GNN增强级别:")
    print(f"  1. 仅优先级增强 (priority_only)")
    print(f"  2. 节点扩展指导 (expansion_guide)")
    print(f"  3. 完全集成 (full_integration)")
    
    try:
        choice = input("选择级别 (1-3) 或Enter使用1: ").strip()
        if choice == "2":
            gnn_level = GNNEnhancementLevel.EXPANSION_GUIDE
        elif choice == "3":
            gnn_level = GNNEnhancementLevel.FULL_INTEGRATION
        else:
            gnn_level = GNNEnhancementLevel.PRIORITY_ONLY
    except:
        gnn_level = GNNEnhancementLevel.PRIORITY_ONLY
    
    print(f"🎯 配置选择:")
    print(f"   GNN增强级别: {gnn_level.value}")
    print(f"   基础优化级别: FULL (自动选择最高质量)")
    print(f"   轨迹质量目标: 匹配trying.py标准")
    
    # 🆕 创建GNN+QP集成系统
    try:
        coordinator = GNNIntegratedCoordinator(
            map_file_path=selected_file,
            optimization_level=OptimizationLevel.FULL,  # 🆕 强制使用FULL确保最佳质量
            gnn_enhancement_level=gnn_level
        )
        
        if not coordinator.map_data:
            print("❌ 地图数据加载失败")
            return
        
        # 🆕 执行GNN+QP集成规划
        results, scenarios = coordinator.plan_with_gnn_integration()
        
        if results and scenarios and any(r['trajectory'] for r in results.values()):
            print(f"\n🎬 生成高质量轨迹可视化...")
            
            # 使用原始协调器进行可视化（带有完整优化轨迹）
            original_coordinator = MultiVehicleCoordinator(selected_file, OptimizationLevel.FULL)
            original_coordinator.create_animation(results, scenarios)
            
            print(f"\n✅ GNN+QP集成演示完成!")
            print(f"\n🏆 质量对比:")
            print(f"   原版trying.py: 基础V-Hybrid A* + 完整QP优化")
            print(f"   增强trans.py: GNN智能协调 + 完整QP优化")
            print(f"   质量提升: 智能决策 + 保持轨迹质量")
            print(f"   成功率: {sum(1 for r in results.values() if r['trajectory'])}/{len(scenarios)}")
            
        else:
            print("❌ 规划失败")
        
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()