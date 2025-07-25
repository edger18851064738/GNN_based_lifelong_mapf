#!/usr/bin/env python3
"""
重构版GAT多车协调系统 - 清晰模块化架构
消除复杂继承关系，采用组合模式，职责分离清晰
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum

# 导入trying.py的基础组件（不继承，只使用）
from trying import (
    VehicleState, VehicleParameters, UnstructuredEnvironment,
    VHybridAStarPlanner, MultiVehicleCoordinator, OptimizationLevel
)

# 检查PyTorch Geometric可用性
try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    from torch_geometric.data import Data
    HAS_TORCH_GEOMETRIC = True
except ImportError:
    HAS_TORCH_GEOMETRIC = False

# ================================
# 1. 数据结构定义
# ================================

@dataclass
class VehicleGraphData:
    """车辆图数据结构"""
    node_features: torch.Tensor      # (N, node_dim)
    edge_indices: torch.Tensor       # (2, E)  
    edge_features: torch.Tensor      # (E, edge_dim)
    global_features: torch.Tensor    # (global_dim,)
    vehicle_ids: List[int]           # 车辆ID映射
    num_nodes: int                   # 节点数量

@dataclass 
class GATDecisions:
    """GAT决策输出结构"""
    priority_adjustments: torch.Tensor    # (N, 1)
    cooperation_scores: torch.Tensor      # (N, 1)
    urgency_levels: torch.Tensor          # (N, 1)
    safety_factors: torch.Tensor          # (N, 1)
    strategies: torch.Tensor              # (N, 5)
    global_signal: torch.Tensor           # (4,)

@dataclass
class CoordinationGuidance:
    """协调指导信息"""
    vehicle_id: int
    strategy: str                    # 'normal', 'cooperative', 'aggressive', 'defensive', 'adaptive'
    priority_adjustment: float       # [-1, 1]
    cooperation_score: float         # [0, 1]
    urgency_level: float            # [0, 1]
    safety_factor: float            # [0, 1]
    adjusted_priority: float        # 调整后优先级

# ================================
# 2. 图数据处理模块
# ================================

class VehicleGraphBuilder:
    """车辆图数据构建器 - 专注于图数据构建"""
    
    def __init__(self, interaction_radius: float = 50.0):
        self.interaction_radius = interaction_radius
        self.node_dim = 15
        self.edge_dim = 10
        self.global_dim = 8
    
    def build_graph(self, vehicles_info: List[Dict]) -> VehicleGraphData:
        """构建车辆交互图"""
        if not vehicles_info:
            return self._empty_graph()
        
        print(f"        📊 构建车辆图: {len(vehicles_info)}个节点")
        
        # 1. 提取节点特征
        node_features = self._extract_node_features(vehicles_info)
        
        # 2. 构建边连接
        edge_indices, edge_features = self._build_edges(vehicles_info)
        
        # 3. 计算全局特征
        global_features = self._compute_global_features(vehicles_info)
        
        print(f"          边连接数: {len(edge_indices)}")
        
        return VehicleGraphData(
            node_features=torch.tensor(node_features, dtype=torch.float32),
            edge_indices=torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.tensor(edge_features, dtype=torch.float32) if edge_features else torch.zeros((0, self.edge_dim), dtype=torch.float32),
            global_features=torch.tensor(global_features, dtype=torch.float32),
            vehicle_ids=[v['id'] for v in vehicles_info],
            num_nodes=len(vehicles_info)
        )
    
    def _extract_node_features(self, vehicles_info: List[Dict]) -> List[List[float]]:
        """提取15维节点特征"""
        features = []
        
        for vehicle in vehicles_info:
            current = vehicle['current_state']
            goal = vehicle['goal_state']
            priority = vehicle.get('priority', 1)
            
            # 计算导航特征
            dx = goal.x - current.x
            dy = goal.y - current.y
            dist_to_goal = math.sqrt(dx*dx + dy*dy)
            goal_bearing = math.atan2(dy, dx)
            heading_error = self._normalize_angle(current.theta - goal_bearing)
            
            # 15维特征向量
            node_feature = [
                current.x / 100.0,                           # 0: 归一化x
                current.y / 100.0,                           # 1: 归一化y  
                math.cos(current.theta),                      # 2: 航向cos
                math.sin(current.theta),                      # 3: 航向sin
                current.v / 15.0,                            # 4: 归一化速度
                getattr(current, 'acceleration', 0.0) / 3.0, # 5: 归一化加速度
                current.v * math.cos(current.theta) / 15.0,  # 6: x速度分量
                dist_to_goal / 100.0,                        # 7: 目标距离
                math.cos(goal_bearing),                       # 8: 目标方向cos
                math.sin(goal_bearing),                       # 9: 目标方向sin
                heading_error / math.pi,                      # 10: 航向误差
                priority / 10.0,                             # 11: 优先级
                current.t / 100.0,                           # 12: 时间
                1.0 if dist_to_goal < 5.0 else 0.0,         # 13: 接近目标
                math.tanh(current.v / 5.0)                   # 14: 速度饱和
            ]
            
            features.append(node_feature)
        
        return features
    
    def _build_edges(self, vehicles_info: List[Dict]) -> Tuple[List, List]:
        """构建边连接和特征"""
        edge_indices = []
        edge_features = []
        
        for i in range(len(vehicles_info)):
            for j in range(len(vehicles_info)):
                if i != j:
                    interaction_data = self._compute_interaction(vehicles_info[i], vehicles_info[j])
                    
                    if interaction_data['should_connect']:
                        edge_indices.append([i, j])
                        edge_features.append(interaction_data['features'])
        
        return edge_indices, edge_features
    
    def _compute_interaction(self, vehicle1: Dict, vehicle2: Dict) -> Dict:
        """计算车辆间交互"""
        state1 = vehicle1['current_state']
        state2 = vehicle2['current_state']
        
        # 基础距离
        dx = state2.x - state1.x
        dy = state2.y - state1.y
        distance = math.sqrt(dx*dx + dy*dy)
        
        # 连接判断
        should_connect = (
            distance <= self.interaction_radius or
            self._predict_conflict(vehicle1, vehicle2)
        )
        
        if not should_connect:
            return {'should_connect': False, 'features': [0.0] * self.edge_dim}
        
        # 计算10维边特征
        features = [
            distance / self.interaction_radius,                    # 0: 距离
            self._compute_relative_speed(state1, state2) / 10.0,   # 1: 相对速度
            self._compute_approach_speed(state1, state2) / 5.0,    # 2: 接近速度
            self._compute_path_crossing(vehicle1, vehicle2),       # 3: 路径交叉
            (vehicle1.get('priority', 1) - vehicle2.get('priority', 1)) / 10.0,  # 4: 优先级差
            min(1.0, self._estimate_conflict_time(state1, state2) / 20.0),  # 5: 冲突时间
            math.cos(math.atan2(dy, dx)),                          # 6: 方位cos
            math.sin(math.atan2(dy, dx)),                          # 7: 方位sin
            self._assess_coordination_need(vehicle1, vehicle2),    # 8: 协调需求
            min(1.0, max(0.1, 1.0 - distance / self.interaction_radius))  # 9: 交互强度
        ]
        
        return {'should_connect': True, 'features': features}
    
    def _compute_global_features(self, vehicles_info: List[Dict]) -> List[float]:
        """计算8维全局特征"""
        n = len(vehicles_info)
        if n == 0:
            return [0.0] * self.global_dim
        
        speeds = [v['current_state'].v for v in vehicles_info]
        priorities = [v.get('priority', 1) for v in vehicles_info]
        
        return [
            n / 10.0,                                    # 0: 车辆数
            sum(speeds) / (n * 15.0),                    # 1: 平均速度
            np.std(speeds) / 15.0,                       # 2: 速度方差
            sum(priorities) / (n * 10.0),                # 3: 平均优先级
            np.std(priorities) / 10.0,                   # 4: 优先级方差
            self._compute_spatial_spread(vehicles_info), # 5: 空间分布
            self._compute_traffic_density(vehicles_info), # 6: 交通密度
            self._compute_complexity(vehicles_info)      # 7: 协调复杂度
        ]
    
    # 辅助方法
    def _normalize_angle(self, angle: float) -> float:
        while angle > math.pi: angle -= 2 * math.pi
        while angle < -math.pi: angle += 2 * math.pi
        return angle
    
    def _predict_conflict(self, v1: Dict, v2: Dict) -> bool:
        """简单冲突预测"""
        s1, s2 = v1['current_state'], v2['current_state']
        g1, g2 = v1['goal_state'], v2['goal_state']
        
        # 检查路径是否可能相交
        return self._lines_intersect((s1.x, s1.y), (g1.x, g1.y), (s2.x, s2.y), (g2.x, g2.y))
    
    def _lines_intersect(self, p1, p2, p3, p4) -> bool:
        """检查两条线段是否相交"""
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
        return ccw(p1,p3,p4) != ccw(p2,p3,p4) and ccw(p1,p2,p3) != ccw(p1,p2,p4)
    
    def _compute_relative_speed(self, s1, s2) -> float:
        v1x, v1y = s1.v * math.cos(s1.theta), s1.v * math.sin(s1.theta)
        v2x, v2y = s2.v * math.cos(s2.theta), s2.v * math.sin(s2.theta)
        return math.sqrt((v1x - v2x)**2 + (v1y - v2y)**2)
    
    def _compute_approach_speed(self, s1, s2) -> float:
        dx, dy = s2.x - s1.x, s2.y - s1.y
        distance = math.sqrt(dx*dx + dy*dy)
        if distance < 1e-6: return 0.0
        v1x, v1y = s1.v * math.cos(s1.theta), s1.v * math.sin(s1.theta)
        return max(0.0, (v1x * dx + v1y * dy) / distance)
    
    def _compute_path_crossing(self, v1: Dict, v2: Dict) -> float:
        """路径交叉概率 [0,1]"""
        return 0.5 if self._predict_conflict(v1, v2) else 0.0
    
    def _estimate_conflict_time(self, s1, s2) -> float:
        """估算冲突时间"""
        return 10.0  # 简化实现
    
    def _assess_coordination_need(self, v1: Dict, v2: Dict) -> float:
        """评估协调需求 [0,1]"""
        return 0.5  # 简化实现
    
    def _compute_spatial_spread(self, vehicles_info: List[Dict]) -> float:
        """计算空间分布"""
        positions = [(v['current_state'].x, v['current_state'].y) for v in vehicles_info]
        if len(positions) < 2: return 0.0
        center_x = sum(p[0] for p in positions) / len(positions)
        center_y = sum(p[1] for p in positions) / len(positions)
        spread = sum(math.sqrt((p[0]-center_x)**2 + (p[1]-center_y)**2) for p in positions)
        return spread / (len(positions) * 50.0)
    
    def _compute_traffic_density(self, vehicles_info: List[Dict]) -> float:
        """计算交通密度"""
        if len(vehicles_info) < 2: return 0.0
        close_pairs = 0
        total_pairs = 0
        for i in range(len(vehicles_info)):
            for j in range(i+1, len(vehicles_info)):
                s1, s2 = vehicles_info[i]['current_state'], vehicles_info[j]['current_state']
                distance = math.sqrt((s1.x - s2.x)**2 + (s1.y - s2.y)**2)
                if distance < self.interaction_radius: close_pairs += 1
                total_pairs += 1
        return close_pairs / max(1, total_pairs)
    
    def _compute_complexity(self, vehicles_info: List[Dict]) -> float:
        """计算协调复杂度"""
        return min(1.0, len(vehicles_info) / 10.0)
    
    def _empty_graph(self) -> VehicleGraphData:
        """空图"""
        return VehicleGraphData(
            node_features=torch.zeros((0, self.node_dim)),
            edge_indices=torch.zeros((2, 0), dtype=torch.long),
            edge_features=torch.zeros((0, self.edge_dim)),
            global_features=torch.zeros(self.global_dim),
            vehicle_ids=[],
            num_nodes=0
        )

# ================================
# 3. GAT网络模块
# ================================

class GATLayer(nn.Module):
    """简洁的GAT层实现"""
    
    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.out_dim = out_dim
        
        if HAS_TORCH_GEOMETRIC:
            self.gat_conv = GATConv(in_dim, out_dim // num_heads, heads=num_heads, 
                                  dropout=dropout, edge_dim=10, concat=True)
        else:
            # 简化的自定义实现
            self.W = nn.Linear(in_dim, out_dim)
            self.attention = nn.MultiheadAttention(out_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dim)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor = None):
        if HAS_TORCH_GEOMETRIC:
            x_new = self.gat_conv(x, edge_index, edge_attr)
        else:
            # 简化处理：当作全连接自注意力
            x_proj = self.W(x)
            x_new, _ = self.attention(x_proj.unsqueeze(0), x_proj.unsqueeze(0), x_proj.unsqueeze(0))
            x_new = x_new.squeeze(0)
        
        # 残差连接和标准化
        if x.shape == x_new.shape:
            x_new = x + x_new
        
        return self.layer_norm(self.dropout(x_new))

class VehicleGATNetwork(nn.Module):
    """车辆协调GAT网络"""
    
    def __init__(self, node_dim: int = 15, edge_dim: int = 10, global_dim: int = 8,
                 hidden_dim: int = 128, num_layers: int = 3, num_heads: int = 4):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 输入投影
        self.node_encoder = nn.Linear(node_dim, hidden_dim)
        
        # GAT层
        self.gat_layers = nn.ModuleList([
            GATLayer(hidden_dim, hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # 全局池化
        self.global_pooling = nn.Sequential(
            nn.Linear(hidden_dim * 3 + global_dim, global_dim),
            nn.ReLU(),
            nn.LayerNorm(global_dim)
        )
        
        # 多任务输出头
        self.task_heads = nn.ModuleDict({
            'priority': nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Tanh()),
            'cooperation': nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()),
            'urgency': nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()),
            'safety': nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()),
            'strategy': nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 5), nn.Softmax(dim=-1))
        })
        
        # 全局决策头
        self.global_head = nn.Sequential(nn.Linear(global_dim, 32), nn.ReLU(), nn.Linear(32, 4), nn.Tanh())
    
    def forward(self, graph_data: VehicleGraphData) -> GATDecisions:
        """前向传播"""
        x = graph_data.node_features
        edge_index = graph_data.edge_indices
        edge_attr = graph_data.edge_features
        u = graph_data.global_features
        
        if x.size(0) == 0:
            return self._empty_decisions()
        
        # 节点编码
        x = F.relu(self.node_encoder(x))
        
        # GAT层
        for gat_layer in self.gat_layers:
            x = gat_layer(x, edge_index, edge_attr)
        
        # 全局池化
        pooled_mean = torch.mean(x, dim=0, keepdim=True)
        pooled_max = torch.max(x, dim=0, keepdim=True)[0]
        pooled_sum = torch.sum(x, dim=0, keepdim=True)
        
        global_input = torch.cat([pooled_mean, pooled_max, pooled_sum, u.unsqueeze(0)], dim=-1)
        global_features = self.global_pooling(global_input)
        
        # 多任务输出
        return GATDecisions(
            priority_adjustments=self.task_heads['priority'](x),
            cooperation_scores=self.task_heads['cooperation'](x),
            urgency_levels=self.task_heads['urgency'](x),
            safety_factors=self.task_heads['safety'](x),
            strategies=self.task_heads['strategy'](x),
            global_signal=self.global_head(global_features).squeeze(0)
        )
    
    def _empty_decisions(self) -> GATDecisions:
        """空决策"""
        return GATDecisions(
            priority_adjustments=torch.zeros((0, 1)),
            cooperation_scores=torch.zeros((0, 1)),
            urgency_levels=torch.zeros((0, 1)),
            safety_factors=torch.zeros((0, 1)),
            strategies=torch.zeros((0, 5)),
            global_signal=torch.zeros(4)
        )

# ================================
# 4. 决策解析模块
# ================================

class DecisionParser:
    """GAT决策解析器"""
    
    def __init__(self):
        self.strategy_names = ['normal', 'cooperative', 'aggressive', 'defensive', 'adaptive']
    
    def parse_decisions(self, decisions: GATDecisions, vehicles_info: List[Dict]) -> List[CoordinationGuidance]:
        """解析GAT决策为协调指导"""
        guidance_list = []
        
        for i, vehicle_info in enumerate(vehicles_info):
            if i < decisions.priority_adjustments.size(0):
                vehicle_id = vehicle_info['id']
                
                # 提取决策值
                priority_adj = decisions.priority_adjustments[i, 0].item()
                cooperation = decisions.cooperation_scores[i, 0].item()
                urgency = decisions.urgency_levels[i, 0].item()
                safety = decisions.safety_factors[i, 0].item()
                
                # 确定策略
                strategy_idx = torch.argmax(decisions.strategies[i]).item()
                strategy = self.strategy_names[strategy_idx]
                
                guidance = CoordinationGuidance(
                    vehicle_id=vehicle_id,
                    strategy=strategy,
                    priority_adjustment=priority_adj,
                    cooperation_score=cooperation,
                    urgency_level=urgency,
                    safety_factor=safety,
                    adjusted_priority=vehicle_info['priority'] + priority_adj * 2.0
                )
                
                guidance_list.append(guidance)
        
        return guidance_list

# ================================
# 5. 集成规划模块
# ================================

class IntegratedPlanner:
    """集成规划器 - 将GAT与trying.py组件集成"""
    
    def __init__(self, environment: UnstructuredEnvironment, optimization_level: OptimizationLevel):
        self.environment = environment
        self.optimization_level = optimization_level
        
        # 创建trying.py的规划器实例（组合，不继承）
        self.base_planner = VHybridAStarPlanner(environment, optimization_level)
        self.params = VehicleParameters()
        
        print(f"         集成规划器初始化: {optimization_level.value}")
    
    def plan_single_vehicle(self, start: VehicleState, goal: VehicleState, 
                          vehicle_id: int, guidance: CoordinationGuidance,
                          existing_trajectories: List[List[VehicleState]]) -> Optional[List[VehicleState]]:
        """规划单个车辆轨迹"""
        
        # 应用GAT指导调整参数
        self._apply_guidance(guidance)
        
        # 使用trying.py的规划器进行搜索
        trajectory = self.base_planner.search_with_waiting(
            start, goal, vehicle_id, existing_trajectories
        )
        
        # 重置参数
        self._reset_params()
        
        return trajectory
    
    def _apply_guidance(self, guidance: CoordinationGuidance):
        """应用GAT指导调整规划参数"""
        strategy = guidance.strategy
        safety_factor = guidance.safety_factor
        cooperation_score = guidance.cooperation_score
        urgency_level = guidance.urgency_level
        
        if strategy == "cooperative":
            self.params.green_additional_safety *= (1.0 + cooperation_score * 0.5)
            self.params.wδ *= (1.0 + cooperation_score * 0.2)
            
        elif strategy == "aggressive":
            self.params.max_speed *= (1.0 + urgency_level * 0.1)
            self.params.wref *= 0.9
            
        elif strategy == "defensive":
            self.params.green_additional_safety *= (1.0 + safety_factor * 0.8)
            self.params.max_speed *= (1.0 - safety_factor * 0.1)
            
        elif strategy == "adaptive":
            adaptation = (cooperation_score + safety_factor) / 2
            self.params.green_additional_safety *= (1.0 + adaptation * 0.4)
            self.params.wv *= (1.0 + adaptation * 0.1)
        
        # 将调整后的参数应用到base_planner
        self.base_planner.params = self.params
    
    def _reset_params(self):
        """重置参数"""
        self.params = VehicleParameters()

# ================================
# 6. 主协调器
# ================================

class GATCoordinator:
    """GAT多车协调器 - 主要接口"""
    
    def __init__(self, map_file_path: str = None, 
                 optimization_level: OptimizationLevel = OptimizationLevel.FULL):
        
        # 初始化环境
        self.environment = UnstructuredEnvironment(size=100)
        self.optimization_level = optimization_level
        
        if map_file_path:
            self.map_data = self.environment.load_from_json(map_file_path)
        else:
            self.map_data = None
        
        # 初始化组件（组合模式）
        self.graph_builder = VehicleGraphBuilder()
        self.gat_network = VehicleGATNetwork()
        self.decision_parser = DecisionParser()
        self.planner = IntegratedPlanner(self.environment, optimization_level)
        
        self.gat_network.eval()
        
        print(f"✅ GAT协调器初始化完成")
        print(f"   优化级别: {optimization_level.value}")
        print(f"   GAT框架: {'PyTorch Geometric' if HAS_TORCH_GEOMETRIC else 'Custom'}")
    
    def plan_vehicles(self, vehicles_info: List[Dict]) -> Dict[int, Optional[List[VehicleState]]]:
        """规划多车辆轨迹"""
        
        print(f"\n🎯 GAT多车协调规划: {len(vehicles_info)}辆车")
        
        # 1. 构建图数据
        print(f"     📊 Step 1: 构建车辆交互图")
        graph_data = self.graph_builder.build_graph(vehicles_info)
        
        # 2. GAT推理
        print(f"     🧠 Step 2: GAT智能决策推理")
        start_time = time.time()
        try:
            with torch.no_grad():
                gat_decisions = self.gat_network(graph_data)
            print(f"          GAT推理成功: {time.time() - start_time:.3f}s")
        except Exception as e:
            print(f"          ⚠️ GAT推理失败: {e}")
            return {}
        
        # 3. 解析决策
        print(f"     📋 Step 3: 解析协调策略")
        guidance_list = self.decision_parser.parse_decisions(gat_decisions, vehicles_info)
        
        # 按调整后优先级排序
        guidance_list.sort(key=lambda g: g.adjusted_priority, reverse=True)
        
        # 4. 逐车规划
        print(f"     🛣️ Step 4: 执行轨迹规划")
        results = {}
        completed_trajectories = []
        
        for guidance in guidance_list:
            vehicle_info = next(v for v in vehicles_info if v['id'] == guidance.vehicle_id)
            
            print(f"       车辆{guidance.vehicle_id}: {guidance.strategy} (优先级:{guidance.adjusted_priority:.1f})")
            
            trajectory = self.planner.plan_single_vehicle(
                vehicle_info['start'], vehicle_info['goal'],
                guidance.vehicle_id, guidance, completed_trajectories
            )
            
            results[guidance.vehicle_id] = trajectory
            if trajectory:
                completed_trajectories.append(trajectory)
                print(f"         ✅ 成功: {len(trajectory)}个路径点")
            else:
                print(f"         ❌ 失败")
        
        success_count = sum(1 for t in results.values() if t is not None)
        print(f"\n📊 规划完成: {success_count}/{len(vehicles_info)} 成功")
        
        return results
    
    def create_scenarios_from_json(self) -> List[Dict]:
        """从JSON创建车辆场景"""
        if not self.map_data:
            return []
        
        start_points = self.map_data.get("start_points", [])
        end_points = self.map_data.get("end_points", [])
        point_pairs = self.map_data.get("point_pairs", [])
        
        scenarios = []
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        
        for i, pair in enumerate(point_pairs):
            start_point = next((p for p in start_points if p["id"] == pair["start_id"]), None)
            end_point = next((p for p in end_points if p["id"] == pair["end_id"]), None)
            
            if start_point and end_point:
                dx = end_point["x"] - start_point["x"]
                dy = end_point["y"] - start_point["y"]
                theta = math.atan2(dy, dx)
                
                scenario = {
                    'id': i + 1,
                    'priority': 1,
                    'color': colors[i % len(colors)],
                    'start': VehicleState(start_point["x"], start_point["y"], theta, 3.0, 0.0),
                    'goal': VehicleState(end_point["x"], end_point["y"], theta, 2.0, 0.0),
                    'description': f'Vehicle {i+1}'
                }
                scenarios.append(scenario)
        
        print(f"✅ 创建 {len(scenarios)} 个车辆场景")
        return scenarios
    
    def run_complete_demo(self) -> Tuple[Dict, List]:
        """运行完整演示"""
        
        scenarios = self.create_scenarios_from_json()
        if not scenarios:
            print("❌ 没有有效场景")
            return {}, []
        
        # 转换格式
        vehicles_info = []
        for scenario in scenarios:
            vehicles_info.append({
                'id': scenario['id'],
                'priority': scenario['priority'],
                'start': scenario['start'],
                'goal': scenario['goal'],
                'current_state': scenario['start'],
                'goal_state': scenario['goal']
            })
        
        # 执行规划
        planning_results = self.plan_vehicles(vehicles_info)
        
        # 转换结果格式
        results = {}
        for scenario in scenarios:
            vehicle_id = scenario['id']
            trajectory = planning_results.get(vehicle_id)
            
            results[vehicle_id] = {
                'trajectory': trajectory if trajectory else [],
                'color': scenario['color'],
                'description': scenario['description']
            }
        
        return results, scenarios

# ================================
# 7. 工具函数
# ================================

def test_system():
    """测试系统功能"""
    print("🧪 GAT系统测试")
    print("=" * 30)
    
    # 创建测试数据
    from dataclasses import dataclass
    
    @dataclass
    class MockVehicleState:
        x: float
        y: float
        theta: float
        v: float
        t: float
        acceleration: float = 0.0
    
    vehicles_info = [
        {
            'id': 1,
            'current_state': MockVehicleState(10, 10, 0, 5, 0),
            'goal_state': MockVehicleState(90, 90, 0, 3, 0),
            'priority': 1
        },
        {
            'id': 2,
            'current_state': MockVehicleState(20, 80, math.pi/2, 4, 0),
            'goal_state': MockVehicleState(80, 20, math.pi/2, 2, 0),
            'priority': 2
        }
    ]
    
    try:
        # 测试图构建
        print("1. 测试图构建...")
        builder = VehicleGraphBuilder()
        graph_data = builder.build_graph(vehicles_info)
        print(f"   ✅ 图构建成功: {graph_data.num_nodes}节点")
        
        # 测试GAT网络
        print("2. 测试GAT网络...")
        network = VehicleGATNetwork()
        with torch.no_grad():
            decisions = network(graph_data)
        print(f"   ✅ GAT推理成功")
        
        # 测试决策解析
        print("3. 测试决策解析...")
        parser = DecisionParser()
        guidance = parser.parse_decisions(decisions, vehicles_info)
        print(f"   ✅ 解析成功: {len(guidance)}个指导策略")
        
        print("🎉 所有测试通过!")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

def interactive_json_selection():
    """交互式选择JSON文件"""
    import os
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if not json_files:
        print("❌ 未找到JSON文件")
        return None
    
    print(f"\n📁 发现 {len(json_files)} 个JSON文件:")
    for i, file in enumerate(json_files):
        print(f"  {i+1}. {file}")
    
    try:
        choice = input(f"选择文件 (1-{len(json_files)}) 或Enter使用第1个: ").strip()
        if choice == "":
            return json_files[0]
        idx = int(choice) - 1
        if 0 <= idx < len(json_files):
            return json_files[idx]
    except:
        pass
    
    return json_files[0] if json_files else None

# ================================
# 8. 主程序
# ================================

def main():
    """主函数"""
    print("🎯 重构版GAT多车协调系统")
    print("=" * 50)
    print("✨ 特性:")
    print("   🧠 标准GAT架构 + 智能协调决策")
    print("   🏗️ 清晰模块化架构，职责分离")
    print("   🔗 与trying.py完整集成（组合模式）")
    print("   📊 15维节点特征 + 10维边特征")
    print("   🎯 5任务多目标学习")
    print("=" * 50)
    
    # 系统测试
    print("\n🧪 运行系统测试...")
    if not test_system():
        print("❌ 系统测试失败")
        return
    
    # 选择地图文件
    selected_file = interactive_json_selection()
    if not selected_file:
        print("❌ 未选择地图文件")
        return
    
    print(f"\n🗺️ 使用地图: {selected_file}")
    
    # 选择优化级别
    print(f"\n⚙️ 选择优化级别:")
    print(f"  1. BASIC - 基础功能")
    print(f"  2. ENHANCED - 增强功能") 
    print(f"  3. FULL - 完整功能（包含QP优化）")
    
    try:
        choice = input("选择级别 (1-3) 或Enter使用FULL: ").strip()
        if choice == "1":
            opt_level = OptimizationLevel.BASIC
        elif choice == "2":
            opt_level = OptimizationLevel.ENHANCED
        else:
            opt_level = OptimizationLevel.FULL
    except:
        opt_level = OptimizationLevel.FULL
    
    print(f"🎯 优化级别: {opt_level.value}")
    
    # 创建协调器并运行
    try:
        coordinator = GATCoordinator(
            map_file_path=selected_file,
            optimization_level=opt_level
        )
        
        if not coordinator.map_data:
            print("❌ 地图加载失败")
            return
        
        # 运行完整演示
        print(f"\n🚀 开始GAT协调规划...")
        results, scenarios = coordinator.run_complete_demo()
        
        if results and any(r['trajectory'] for r in results.values()):
            print(f"\n🎬 生成可视化...")
            
            # 使用trying.py的可视化
            visualizer = MultiVehicleCoordinator(selected_file, opt_level)
            visualizer.create_animation(results, scenarios)
            
            print(f"\n✅ GAT演示完成!")
        else:
            print("❌ 规划失败")
            
    except Exception as e:
        print(f"❌ 系统错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()