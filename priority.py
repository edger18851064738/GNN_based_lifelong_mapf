#!/usr/bin/env python3
"""
智能优先级系统 - Multi-Factor Priority Assignment
基于多因素分析的车辆优先级动态分配系统
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json

# 假设这些类已经从原始代码导入
# from trying import VehicleState, VehicleParameters, UnstructuredEnvironment

class PriorityFactor(Enum):
    """优先级影响因素"""
    DISTANCE = "distance"           # 路径距离
    COMPLEXITY = "complexity"       # 路径复杂度  
    URGENCY = "urgency"            # 紧急程度
    CONFLICT_DENSITY = "conflict"   # 冲突密度
    VEHICLE_TYPE = "vehicle_type"   # 车辆类型
    SAFETY_CRITICALITY = "safety"  # 安全关键性
    TRAFFIC_FLOW = "traffic_flow"   # 交通流影响

@dataclass
class PriorityProfile:
    """车辆优先级档案"""
    vehicle_id: int
    base_priority: float           # 基础优先级
    distance_factor: float         # 距离因子 [0-1]
    complexity_factor: float       # 复杂度因子 [0-1]  
    urgency_factor: float         # 紧急度因子 [0-1]
    conflict_factor: float        # 冲突因子 [0-1]
    safety_factor: float          # 安全因子 [0-1]
    traffic_factor: float         # 交通流因子 [0-1]
    final_priority: float         # 最终优先级
    priority_reasoning: str       # 优先级推理说明

class IntelligentPriorityAssigner:
    """智能优先级分配器"""
    
    def __init__(self, environment, factor_weights: Optional[Dict[PriorityFactor, float]] = None):
        self.environment = environment
        
        # 🎯 可配置的因子权重
        self.factor_weights = factor_weights or {
            PriorityFactor.DISTANCE: 0.20,        # 距离权重
            PriorityFactor.COMPLEXITY: 0.25,      # 复杂度权重  
            PriorityFactor.URGENCY: 0.15,         # 紧急度权重
            PriorityFactor.CONFLICT_DENSITY: 0.20, # 冲突密度权重
            PriorityFactor.SAFETY_CRITICALITY: 0.15, # 安全权重
            PriorityFactor.TRAFFIC_FLOW: 0.05      # 交通流权重
        }
        
        # 验证权重和为1
        total_weight = sum(self.factor_weights.values())
        if abs(total_weight - 1.0) > 1e-6:
            print(f"⚠️ 权重和不为1.0: {total_weight:.3f}，自动归一化")
            for factor in self.factor_weights:
                self.factor_weights[factor] /= total_weight
        
        print(f"🎯 智能优先级系统初始化")
        print(f"   因子权重: {[(f.value, w) for f, w in self.factor_weights.items()]}")
    
    def assign_intelligent_priorities(self, scenarios: List[Dict]) -> List[Dict]:
        """为所有车辆分配智能优先级"""
        
        print(f"\n🧮 智能优先级分析: {len(scenarios)}辆车")
        
        # 1. 提取所有车辆的基础信息
        vehicle_infos = self._extract_vehicle_infos(scenarios)
        
        # 2. 计算各种因子
        priority_profiles = []
        for info in vehicle_infos:
            profile = self._compute_priority_profile(info, vehicle_infos)
            priority_profiles.append(profile)
        
        # 3. 应用权重计算最终优先级
        self._compute_final_priorities(priority_profiles)
        
        # 4. 更新scenarios
        updated_scenarios = self._update_scenarios_with_priorities(scenarios, priority_profiles)
        
        # 5. 打印优先级分析报告
        self._print_priority_analysis(priority_profiles)
        
        return updated_scenarios
    
    def _extract_vehicle_infos(self, scenarios: List[Dict]) -> List[Dict]:
        """提取车辆基础信息"""
        vehicle_infos = []
        
        for scenario in scenarios:
            start = scenario['start']
            goal = scenario['goal']
            
            # 基础几何信息
            dx = goal.x - start.x
            dy = goal.y - start.y
            euclidean_distance = math.sqrt(dx*dx + dy*dy)
            straight_line_bearing = math.atan2(dy, dx)
            heading_alignment = abs(self._normalize_angle(start.theta - straight_line_bearing))
            
            info = {
                'vehicle_id': scenario['id'],
                'scenario': scenario,
                'start_pos': (start.x, start.y),
                'goal_pos': (goal.x, goal.y),
                'start_heading': start.theta,
                'euclidean_distance': euclidean_distance,
                'straight_line_bearing': straight_line_bearing,
                'heading_alignment': heading_alignment,
                'original_priority': scenario.get('priority', 1)
            }
            
            vehicle_infos.append(info)
        
        return vehicle_infos
    
    def _compute_priority_profile(self, vehicle_info: Dict, all_vehicles: List[Dict]) -> PriorityProfile:
        """计算单个车辆的优先级档案"""
        
        vehicle_id = vehicle_info['vehicle_id']
        
        # 🚗 1. 距离因子：距离越短优先级越高
        distance_factor = self._compute_distance_factor(vehicle_info, all_vehicles)
        
        # 🌀 2. 复杂度因子：路径越复杂优先级越高（需要更多规划时间）
        complexity_factor = self._compute_complexity_factor(vehicle_info)
        
        # ⚡ 3. 紧急度因子：基于车辆类型、距离、初始速度等
        urgency_factor = self._compute_urgency_factor(vehicle_info)
        
        # 💥 4. 冲突因子：与其他车辆路径冲突越多，优先级需要调整
        conflict_factor = self._compute_conflict_factor(vehicle_info, all_vehicles)
        
        # 🛡️ 5. 安全因子：安全关键性分析
        safety_factor = self._compute_safety_factor(vehicle_info)
        
        # 🌊 6. 交通流因子：对整体交通流的影响
        traffic_factor = self._compute_traffic_flow_factor(vehicle_info, all_vehicles)
        
        # 生成推理说明
        reasoning = self._generate_priority_reasoning(
            vehicle_id, distance_factor, complexity_factor, urgency_factor,
            conflict_factor, safety_factor, traffic_factor
        )
        
        return PriorityProfile(
            vehicle_id=vehicle_id,
            base_priority=vehicle_info['original_priority'],
            distance_factor=distance_factor,
            complexity_factor=complexity_factor,
            urgency_factor=urgency_factor,
            conflict_factor=conflict_factor,
            safety_factor=safety_factor,
            traffic_factor=traffic_factor,
            final_priority=0.0,  # 待计算
            priority_reasoning=reasoning
        )
    
    def _compute_distance_factor(self, vehicle_info: Dict, all_vehicles: List[Dict]) -> float:
        """计算距离因子：相对距离排名"""
        distances = [v['euclidean_distance'] for v in all_vehicles]
        current_distance = vehicle_info['euclidean_distance']
        
        if len(distances) <= 1:
            return 0.5
        
        # 距离排名：距离越短排名越高，因子越大
        sorted_distances = sorted(distances)
        rank = sorted_distances.index(current_distance)
        
        # 归一化到[0,1]，距离越短因子越大
        distance_factor = 1.0 - (rank / (len(distances) - 1))
        
        return distance_factor
    
    def _compute_complexity_factor(self, vehicle_info: Dict) -> float:
        """计算路径复杂度因子"""
        
        # 基于多个复杂度指标
        complexity_score = 0.0
        
        # 1. 航向偏差：起始航向与目标方向的偏差
        heading_deviation = vehicle_info['heading_alignment']
        heading_complexity = min(1.0, heading_deviation / (math.pi/2))  # 归一化到[0,1]
        complexity_score += heading_complexity * 0.4
        
        # 2. 路径通过障碍区域的复杂度
        obstacle_complexity = self._analyze_obstacle_complexity(vehicle_info)
        complexity_score += obstacle_complexity * 0.6
        
        return min(1.0, complexity_score)
    
    def _analyze_obstacle_complexity(self, vehicle_info: Dict) -> float:
        """分析路径穿越障碍物的复杂度"""
        start_x, start_y = vehicle_info['start_pos']
        goal_x, goal_y = vehicle_info['goal_pos']
        
        # 简化分析：检查直线路径上的障碍物密度
        if not hasattr(self.environment, 'obstacle_map'):
            return 0.3  # 默认中等复杂度
        
        # 在直线路径上采样点检查障碍物
        num_samples = 20
        obstacle_encounters = 0
        
        for i in range(num_samples + 1):
            t = i / num_samples
            sample_x = start_x + t * (goal_x - start_x)
            sample_y = start_y + t * (goal_y - start_y)
            
            # 检查采样点周围的障碍物
            ix, iy = int(sample_x), int(sample_y)
            if (0 <= ix < self.environment.size and 0 <= iy < self.environment.size):
                # 检查3x3区域的障碍物密度
                local_obstacles = 0
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        check_x, check_y = ix + dx, iy + dy
                        if (0 <= check_x < self.environment.size and 
                            0 <= check_y < self.environment.size and
                            self.environment.obstacle_map[check_y, check_x]):
                            local_obstacles += 1
                
                if local_obstacles > 0:
                    obstacle_encounters += local_obstacles / 9.0  # 归一化
        
        # 归一化复杂度
        complexity = min(1.0, obstacle_encounters / num_samples)
        return complexity
    
    def _compute_urgency_factor(self, vehicle_info: Dict) -> float:
        """计算紧急度因子"""
        
        urgency_score = 0.0
        
        # 1. 距离紧急度：距离越短越紧急（需要快速决策）
        distance = vehicle_info['euclidean_distance']
        if distance < 20:
            urgency_score += 0.6  # 短距离高紧急度
        elif distance < 50:
            urgency_score += 0.3  # 中距离中等紧急度
        else:
            urgency_score += 0.1  # 长距离低紧急度
        
        # 2. 航向紧急度：大幅转向需要更多规划时间
        heading_deviation = vehicle_info['heading_alignment']
        if heading_deviation > math.pi * 0.75:  # 大于135度
            urgency_score += 0.4  # 需要掉头，紧急度高
        elif heading_deviation > math.pi * 0.25:  # 大于45度
            urgency_score += 0.2  # 需要转向
        
        return min(1.0, urgency_score)
    
    def _compute_conflict_factor(self, vehicle_info: Dict, all_vehicles: List[Dict]) -> float:
        """计算冲突因子：与其他车辆的潜在冲突"""
        
        if len(all_vehicles) <= 1:
            return 0.0
        
        current_start = vehicle_info['start_pos']
        current_goal = vehicle_info['goal_pos']
        
        total_conflicts = 0.0
        
        for other_vehicle in all_vehicles:
            if other_vehicle['vehicle_id'] == vehicle_info['vehicle_id']:
                continue
            
            other_start = other_vehicle['start_pos']
            other_goal = other_vehicle['goal_pos']
            
            # 分析路径交叉和空间冲突
            conflict_score = self._analyze_path_conflict(
                current_start, current_goal, other_start, other_goal
            )
            
            total_conflicts += conflict_score
        
        # 归一化：冲突越多，需要更高优先级来先行规划
        normalized_conflicts = min(1.0, total_conflicts / (len(all_vehicles) - 1))
        
        return normalized_conflicts
    
    def _analyze_path_conflict(self, start1: Tuple, goal1: Tuple, 
                              start2: Tuple, goal2: Tuple) -> float:
        """分析两条路径的冲突程度"""
        
        # 1. 路径交叉检测
        intersection_score = self._compute_path_intersection(start1, goal1, start2, goal2)
        
        # 2. 空间距离分析
        spatial_conflict = self._compute_spatial_conflict(start1, goal1, start2, goal2)
        
        # 3. 综合冲突分数
        total_conflict = intersection_score * 0.7 + spatial_conflict * 0.3
        
        return total_conflict
    
    def _compute_path_intersection(self, start1: Tuple, goal1: Tuple, 
                                  start2: Tuple, goal2: Tuple) -> float:
        """计算路径交叉分数"""
        
        # 线段相交算法
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
        
        has_intersection, intersection_point = line_intersection(start1, goal1, start2, goal2)
        
        if has_intersection:
            # 计算交点位置的重要性
            ix, iy = intersection_point
            
            # 交点越靠近起点，冲突越严重
            dist1_to_intersect = math.sqrt((ix - start1[0])**2 + (iy - start1[1])**2)
            dist2_to_intersect = math.sqrt((ix - start2[0])**2 + (iy - start2[1])**2)
            
            path1_length = math.sqrt((goal1[0] - start1[0])**2 + (goal1[1] - start1[1])**2)
            path2_length = math.sqrt((goal2[0] - start2[0])**2 + (goal2[1] - start2[1])**2)
            
            # 归一化交点位置
            pos1_ratio = dist1_to_intersect / max(path1_length, 1.0)
            pos2_ratio = dist2_to_intersect / max(path2_length, 1.0)
            
            # 早期交叉冲突更严重
            intersection_severity = 2.0 - pos1_ratio - pos2_ratio
            return min(1.0, intersection_severity / 2.0)
        
        return 0.0
    
    def _compute_spatial_conflict(self, start1: Tuple, goal1: Tuple, 
                                 start2: Tuple, goal2: Tuple) -> float:
        """计算空间冲突分数"""
        
        # 计算路径端点间的最小距离
        distances = [
            math.sqrt((start1[0] - start2[0])**2 + (start1[1] - start2[1])**2),
            math.sqrt((start1[0] - goal2[0])**2 + (start1[1] - goal2[1])**2),
            math.sqrt((goal1[0] - start2[0])**2 + (goal1[1] - start2[1])**2),
            math.sqrt((goal1[0] - goal2[0])**2 + (goal1[1] - goal2[1])**2)
        ]
        
        min_distance = min(distances)
        
        # 距离越近冲突越高
        conflict_threshold = 30.0  # 冲突阈值
        if min_distance < conflict_threshold:
            spatial_conflict = 1.0 - (min_distance / conflict_threshold)
            return spatial_conflict
        
        return 0.0
    
    def _compute_safety_factor(self, vehicle_info: Dict) -> float:
        """计算安全关键性因子"""
        
        safety_score = 0.5  # 基础安全分数
        
        # 1. 基于位置的安全性：靠近边界或障碍物的车辆安全性更关键
        start_x, start_y = vehicle_info['start_pos']
        goal_x, goal_y = vehicle_info['goal_pos']
        
        # 检查是否靠近边界
        boundary_distance = min(start_x, start_y, 
                               self.environment.size - start_x, 
                               self.environment.size - start_y)
        
        if boundary_distance < 10:
            safety_score += 0.3  # 靠近边界，安全关键性增加
        
        # 2. 基于路径长度的安全性：长路径需要更多安全考虑
        distance = vehicle_info['euclidean_distance']
        if distance > 60:
            safety_score += 0.2
        
        return min(1.0, safety_score)
    
    def _compute_traffic_flow_factor(self, vehicle_info: Dict, all_vehicles: List[Dict]) -> float:
        """计算对交通流的影响因子"""
        
        if len(all_vehicles) <= 1:
            return 0.5
        
        # 分析该车辆对整体交通流的影响
        flow_impact = 0.0
        
        current_bearing = vehicle_info['straight_line_bearing']
        current_distance = vehicle_info['euclidean_distance']
        
        # 统计相似方向的车辆
        similar_direction_vehicles = 0
        
        for other_vehicle in all_vehicles:
            if other_vehicle['vehicle_id'] == vehicle_info['vehicle_id']:
                continue
            
            other_bearing = other_vehicle['straight_line_bearing']
            bearing_diff = abs(self._normalize_angle(current_bearing - other_bearing))
            
            # 方向相似（差异小于45度）
            if bearing_diff < math.pi / 4:
                similar_direction_vehicles += 1
        
        # 如果有很多车辆同方向，该车辆对流量影响更大
        if similar_direction_vehicles > 0:
            flow_impact = min(1.0, similar_direction_vehicles / (len(all_vehicles) - 1))
        
        return flow_impact
    
    def _compute_final_priorities(self, priority_profiles: List[PriorityProfile]):
        """计算最终优先级分数"""
        
        for profile in priority_profiles:
            # 加权综合各个因子
            weighted_score = (
                profile.distance_factor * self.factor_weights[PriorityFactor.DISTANCE] +
                profile.complexity_factor * self.factor_weights[PriorityFactor.COMPLEXITY] +
                profile.urgency_factor * self.factor_weights[PriorityFactor.URGENCY] +
                profile.conflict_factor * self.factor_weights[PriorityFactor.CONFLICT_DENSITY] +
                profile.safety_factor * self.factor_weights[PriorityFactor.SAFETY_CRITICALITY] +
                profile.traffic_factor * self.factor_weights[PriorityFactor.TRAFFIC_FLOW]
            )
            
            # 转换为1-10的优先级范围
            profile.final_priority = 1.0 + weighted_score * 9.0
    
    def _generate_priority_reasoning(self, vehicle_id: int, distance_f: float, 
                                   complexity_f: float, urgency_f: float,
                                   conflict_f: float, safety_f: float, 
                                   traffic_f: float) -> str:
        """生成优先级推理说明"""
        
        factors = [
            (distance_f, "距离", "短" if distance_f > 0.7 else "中" if distance_f > 0.3 else "长"),
            (complexity_f, "复杂度", "高" if complexity_f > 0.7 else "中" if complexity_f > 0.3 else "低"),
            (urgency_f, "紧急度", "高" if urgency_f > 0.7 else "中" if urgency_f > 0.3 else "低"),
            (conflict_f, "冲突密度", "高" if conflict_f > 0.7 else "中" if conflict_f > 0.3 else "低"),
            (safety_f, "安全关键性", "高" if safety_f > 0.7 else "中" if safety_f > 0.3 else "低"),
            (traffic_f, "交通流影响", "高" if traffic_f > 0.7 else "中" if traffic_f > 0.3 else "低")
        ]
        
        # 找出主要影响因子
        sorted_factors = sorted(factors, key=lambda x: x[0], reverse=True)
        top_factors = [f"{name}({level})" for _, name, level in sorted_factors[:3]]
        
        reasoning = f"V{vehicle_id}主要因子: {', '.join(top_factors)}"
        
        return reasoning
    
    def _update_scenarios_with_priorities(self, scenarios: List[Dict], 
                                        priority_profiles: List[PriorityProfile]) -> List[Dict]:
        """更新scenarios的优先级"""
        
        # 创建优先级映射
        priority_map = {profile.vehicle_id: profile.final_priority 
                       for profile in priority_profiles}
        
        # 更新scenarios
        updated_scenarios = []
        for scenario in scenarios:
            updated_scenario = scenario.copy()
            vehicle_id = scenario['id']
            if vehicle_id in priority_map:
                updated_scenario['priority'] = priority_map[vehicle_id]
                updated_scenario['original_priority'] = scenario.get('priority', 1)
            updated_scenarios.append(updated_scenario)
        
        # 按新优先级排序
        updated_scenarios.sort(key=lambda x: x['priority'], reverse=True)
        
        return updated_scenarios
    
    def _print_priority_analysis(self, priority_profiles: List[PriorityProfile]):
        """打印优先级分析报告"""
        
        print(f"\n📊 智能优先级分析报告:")
        print(f"{'车辆':<4} {'原始':<4} {'最终':<5} {'距离':<5} {'复杂':<5} {'紧急':<5} {'冲突':<5} {'安全':<5} {'流量':<5} {'推理说明'}")
        print("-" * 90)
        
        # 按最终优先级排序
        sorted_profiles = sorted(priority_profiles, key=lambda x: x.final_priority, reverse=True)
        
        for profile in sorted_profiles:
            print(f"V{profile.vehicle_id:<3} "
                  f"{profile.base_priority:<4.1f} "
                  f"{profile.final_priority:<5.1f} "
                  f"{profile.distance_factor:<5.2f} "
                  f"{profile.complexity_factor:<5.2f} "
                  f"{profile.urgency_factor:<5.2f} "
                  f"{profile.conflict_factor:<5.2f} "
                  f"{profile.safety_factor:<5.2f} "
                  f"{profile.traffic_factor:<5.2f} "
                  f"{profile.priority_reasoning}")
        
        print(f"\n🎯 优先级分配结果:")
        for i, profile in enumerate(sorted_profiles):
            priority_change = profile.final_priority - profile.base_priority
            change_symbol = "↗️" if priority_change > 0.5 else "↘️" if priority_change < -0.5 else "➡️"
            print(f"   #{i+1} V{profile.vehicle_id}: {profile.base_priority:.1f} → {profile.final_priority:.1f} {change_symbol}")
    
    def _normalize_angle(self, angle: float) -> float:
        """角度标准化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# 🔧 智能优先级系统的集成示例
class EnhancedMultiVehicleCoordinator:
    """集成智能优先级的多车协调器"""
    
    def __init__(self, map_file_path=None, priority_config: Optional[Dict] = None):
        self.environment = UnstructuredEnvironment(size=100)
        
        # 加载地图
        if map_file_path:
            self.environment.load_from_json(map_file_path)
        
        # 初始化智能优先级分配器
        self.priority_assigner = IntelligentPriorityAssigner(
            self.environment, priority_config
        )
        
        print(f"✅ 增强型多车协调器初始化完成")
    
    def create_intelligent_scenarios(self, json_data):
        """创建智能优先级场景"""
        
        # 1. 创建基础场景（使用简单优先级）
        basic_scenarios = self._create_basic_scenarios(json_data)
        
        # 2. 应用智能优先级分析
        intelligent_scenarios = self.priority_assigner.assign_intelligent_priorities(basic_scenarios)
        
        print(f"\n✨ 智能优先级分配完成:")
        print(f"   基础优先级: 简单的载入顺序")
        print(f"   智能优先级: 6因子综合分析")
        print(f"   优先级变化: {self._analyze_priority_changes(basic_scenarios, intelligent_scenarios)}")
        
        return intelligent_scenarios
    
    def _create_basic_scenarios(self, json_data):
        """创建基础场景"""
        # 这里是原始的简单优先级分配逻辑
        # ... (与原始代码相同)
        pass
    
    def _analyze_priority_changes(self, basic: List[Dict], intelligent: List[Dict]) -> str:
        """分析优先级变化"""
        changes = 0
        for b, i in zip(basic, intelligent):
            if abs(b['priority'] - i['priority']) > 0.5:
                changes += 1
        
        return f"{changes}/{len(basic)}辆车优先级发生显著变化"


# 🎯 使用示例
def demo_intelligent_priority():
    """演示智能优先级系统"""
    
    # 创建测试场景
    test_scenarios = [
        {
            'id': 1,
            'priority': 4,  # 原始简单优先级
            'start': VehicleState(x=10, y=10, theta=0, v=3, t=0),
            'goal': VehicleState(x=90, y=90, theta=0, v=2, t=0),
        },
        {
            'id': 2, 
            'priority': 3,
            'start': VehicleState(x=90, y=10, theta=math.pi, v=3, t=0),
            'goal': VehicleState(x=10, y=90, theta=math.pi, v=2, t=0),
        },
        {
            'id': 3,
            'priority': 2,
            'start': VehicleState(x=50, y=10, theta=math.pi/2, v=3, t=0),
            'goal': VehicleState(x=50, y=40, theta=math.pi/2, v=2, t=0),
        },
        {
            'id': 4,
            'priority': 1,
            'start': VehicleState(x=20, y=80, theta=0, v=3, t=0),
            'goal': VehicleState(x=80, y=80, theta=0, v=2, t=0),
        }
    ]
    
    # 创建环境和智能优先级分配器
    environment = UnstructuredEnvironment(size=100)
    
    # 自定义权重配置
    custom_weights = {
        PriorityFactor.DISTANCE: 0.15,
        PriorityFactor.COMPLEXITY: 0.30,  # 增加复杂度权重
        PriorityFactor.URGENCY: 0.10,
        PriorityFactor.CONFLICT_DENSITY: 0.35,  # 增加冲突权重
        PriorityFactor.SAFETY_CRITICALITY: 0.05,
        PriorityFactor.TRAFFIC_FLOW: 0.05
    }
    
    assigner = IntelligentPriorityAssigner(environment, custom_weights)
    
    # 应用智能优先级
    intelligent_scenarios = assigner.assign_intelligent_priorities(test_scenarios)
    
    return intelligent_scenarios

if __name__ == "__main__":
    demo_intelligent_priority()