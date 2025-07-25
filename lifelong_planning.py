#!/usr/bin/env python3
"""
🚀 增强版第一轮多车轨迹规划系统 - 集成GAT智能协调
在原有lifelong_planning基础上集成GAT模块，提供智能协调决策

主要增强：
1. 集成GAT智能协调系统
2. 车辆交互图分析
3. 动态优先级调整
4. 智能协调策略应用
5. 保持原有接口兼容性

流程:
1. 载入地图，生成任务
2. 构建车辆交互图
3. GAT智能决策推理
4. 应用协调指导规划所有车辆
5. 可视化结果
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

# 导入核心规划模块
from trying import (
    VehicleState, VehicleParameters, OptimizationLevel,
    UnstructuredEnvironment, VHybridAStarPlanner
)

# 🆕 导入GAT协调模块
try:
    from GAT import (
        VehicleGraphBuilder, VehicleGATNetwork, DecisionParser, 
        IntegratedPlanner, CoordinationGuidance, VehicleGraphData,
        GATDecisions
    )
    HAS_GAT = True
    print("✅ GAT模块导入成功")
except ImportError as e:
    HAS_GAT = False
    print(f"⚠️ GAT模块导入失败: {e}")

# 可选导入智能优先级模块
try:
    from priority import IntelligentPriorityAssigner
    HAS_INTELLIGENT_PRIORITY = True
except ImportError:
    HAS_INTELLIGENT_PRIORITY = False

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
    """任务定义"""
    task_id: int
    start_edge: IntersectionEdge
    end_edge: IntersectionEdge
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    priority: int = 1
    # 🆕 GAT相关字段
    gat_strategy: str = "normal"
    cooperation_score: float = 0.5
    urgency_level: float = 0.5
    safety_factor: float = 0.5

@dataclass
class Vehicle:
    """车辆定义"""
    vehicle_id: int
    task: Task
    trajectory: List[VehicleState] = None
    color: str = "blue"
    planning_time: float = 0.0
    # 🆕 GAT相关字段
    gat_guidance: Optional[CoordinationGuidance] = None

class EnhancedFirstRoundPlanner:
    """🚀 增强版第一轮多车规划器 - 集成GAT智能协调"""
    
    def __init__(self, map_file: str, optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED,
                 enable_gat: bool = True):
        # 基础环境设置
        self.environment = UnstructuredEnvironment()
        self.map_data = self.environment.load_from_json(map_file)
        
        if not self.map_data:
            raise ValueError(f"无法加载地图文件: {map_file}")
        
        self.params = VehicleParameters()
        self.optimization_level = optimization_level
        
        # 🆕 GAT系统初始化
        self.enable_gat = enable_gat and HAS_GAT
        if self.enable_gat:
            self._initialize_gat_system()
        else:
            print("ℹ️ GAT系统未启用，使用传统规划")
        
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
        
        # 🆕 GAT性能统计
        self.gat_stats = {
            'graph_construction_time': 0.0,
            'inference_time': 0.0,
            'decision_parsing_time': 0.0,
            'coordination_applications': 0
        }
        
        print(f"🚀 增强版第一轮规划器初始化完成")
        print(f"   地图: {self.map_data.get('map_info', {}).get('name', 'Unknown')}")
        print(f"   出入口边: {len(self.edges)} 个")
        print(f"   生成任务: {len(self.tasks)} 个")
        print(f"   创建车辆: {len(self.vehicles)} 个")
        print(f"   GAT智能协调: {'✅ 启用' if self.enable_gat else '❌ 禁用'}")
        print(f"   优化级别: {optimization_level.value}")
    
    def _initialize_gat_system(self):
        """🆕 初始化GAT系统组件"""
        try:
            self.graph_builder = VehicleGraphBuilder(interaction_radius=50.0)
            self.gat_network = VehicleGATNetwork()
            self.decision_parser = DecisionParser()
            self.integrated_planner = IntegratedPlanner(self.environment, self.optimization_level)
            
            # 设置GAT网络为推理模式
            self.gat_network.eval()
            
            print("   🧠 GAT系统组件初始化成功")
            print(f"     - 图构建器: 交互半径50.0m")
            print(f"     - GAT网络: 15维节点 + 10维边 + 8维全局特征")
            print(f"     - 集成规划器: {self.optimization_level.value}级别")
            
        except Exception as e:
            print(f"❌ GAT系统初始化失败: {e}")
            self.enable_gat = False
    
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
        """为每个出入口边生成一个任务"""
        tasks = []
        
        for i, start_edge in enumerate(self.edges):
            # 选择非相邻的终点边
            end_edge = self._select_non_adjacent_edge(start_edge)
            if not end_edge:
                continue
            
            # 生成整数坐标的起点和终点
            start_pos = start_edge.get_random_integer_position()
            end_pos = end_edge.get_random_integer_position()
            
            task = Task(
                task_id=i + 1,
                start_edge=start_edge,
                end_edge=end_edge,
                start_pos=start_pos,
                end_pos=end_pos,
                priority=1  # 默认优先级，后续可能被GAT调整
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
        
        # 排除最近的两条边
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
    
    def _convert_vehicles_to_gat_format(self) -> List[Dict]:
        """🆕 将车辆信息转换为GAT模块需要的格式"""
        vehicles_info = []
        
        for vehicle in self.vehicles:
            task = vehicle.task
            start_x, start_y = task.start_pos
            end_x, end_y = task.end_pos
            
            # 计算朝向
            dx = end_x - start_x
            dy = end_y - start_y
            theta = math.atan2(dy, dx)
            
            # 创建起始和目标状态
            start_state = VehicleState(x=start_x, y=start_y, theta=theta, v=3.0, t=0.0)
            goal_state = VehicleState(x=end_x, y=end_y, theta=theta, v=2.0, t=0.0)
            
            vehicle_info = {
                'id': vehicle.vehicle_id,
                'priority': task.priority,
                'start': start_state,
                'goal': goal_state,
                'current_state': start_state,  # GAT需要
                'goal_state': goal_state,      # GAT需要
                'color': vehicle.color,
                'description': f'V{vehicle.vehicle_id}({task.start_edge.edge_id}->{task.end_edge.edge_id})'
            }
            vehicles_info.append(vehicle_info)
        
        return vehicles_info
    
    def apply_gat_coordination(self):
        """🆕 应用GAT智能协调决策"""
        if not self.enable_gat:
            print("ℹ️ GAT系统未启用，跳过智能协调")
            return
        
        print(f"\n🧠 开始GAT智能协调分析...")
        
        try:
            # Step 1: 转换车辆信息格式
            vehicles_info = self._convert_vehicles_to_gat_format()
            print(f"   📊 车辆信息转换: {len(vehicles_info)}个车辆")
            
            # Step 2: 构建车辆交互图
            start_time = time.time()
            graph_data = self.graph_builder.build_graph(vehicles_info)
            self.gat_stats['graph_construction_time'] = time.time() - start_time
            
            print(f"   📈 交互图构建完成: {graph_data.num_nodes}节点, 耗时{self.gat_stats['graph_construction_time']:.3f}s")
            
            # Step 3: GAT智能推理
            start_time = time.time()
            with np.errstate(all='ignore'):  # 忽略numpy警告
                import torch
                with torch.no_grad():
                    gat_decisions = self.gat_network(graph_data)
            self.gat_stats['inference_time'] = time.time() - start_time
            
            print(f"   🎯 GAT推理完成: 耗时{self.gat_stats['inference_time']:.3f}s")
            
            # Step 4: 解析决策指导
            start_time = time.time()
            guidance_list = self.decision_parser.parse_decisions(gat_decisions, vehicles_info)
            self.gat_stats['decision_parsing_time'] = time.time() - start_time
            
            print(f"   📋 决策解析完成: {len(guidance_list)}个指导策略, 耗时{self.gat_stats['decision_parsing_time']:.3f}s")
            
            # Step 5: 应用协调指导到车辆
            self._apply_coordination_guidance(guidance_list)
            
            print(f"✅ GAT智能协调应用成功")
            
        except Exception as e:
            print(f"❌ GAT智能协调失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _apply_coordination_guidance(self, guidance_list: List[CoordinationGuidance]):
        """🆕 应用协调指导到车辆和任务"""
        print(f"   🎯 应用协调指导:")
        
        for guidance in guidance_list:
            # 找到对应的车辆
            vehicle = next((v for v in self.vehicles if v.vehicle_id == guidance.vehicle_id), None)
            if not vehicle:
                continue
            
            # 保存GAT指导
            vehicle.gat_guidance = guidance
            
            # 更新任务优先级和策略
            vehicle.task.priority = guidance.adjusted_priority
            vehicle.task.gat_strategy = guidance.strategy
            vehicle.task.cooperation_score = guidance.cooperation_score
            vehicle.task.urgency_level = guidance.urgency_level
            vehicle.task.safety_factor = guidance.safety_factor
            
            self.gat_stats['coordination_applications'] += 1
            
            print(f"     V{guidance.vehicle_id}: {guidance.strategy}, "
                  f"优先级{guidance.adjusted_priority:.1f}, "
                  f"合作{guidance.cooperation_score:.2f}, "
                  f"紧急{guidance.urgency_level:.2f}, "
                  f"安全{guidance.safety_factor:.2f}")
    
    def plan_all_vehicles(self):
        """🚀 使用GAT增强的多车规划"""
        print(f"\n🎯 开始GAT增强的多车规划...")
        print(f"   车辆数量: {len(self.vehicles)}")
        print(f"   GAT系统: {'✅ 启用' if self.enable_gat else '❌ 禁用'}")
        
        # 按优先级排序（可能已被GAT调整）
        self.vehicles.sort(key=lambda v: v.task.priority, reverse=True)
        
        # 显示最终优先级排序
        print(f"   📊 最终优先级排序:")
        for i, vehicle in enumerate(self.vehicles):
            strategy_info = f" [{vehicle.task.gat_strategy}]" if self.enable_gat else ""
            print(f"     {i+1}. V{vehicle.vehicle_id}: 优先级{vehicle.task.priority}{strategy_info}")
        
        # 执行规划
        successful_trajectories = []
        
        for vehicle in self.vehicles:
            print(f"\n   🚗 规划车辆 V{vehicle.vehicle_id}")
            print(f"      优先级: {vehicle.task.priority}")
            if self.enable_gat and vehicle.gat_guidance:
                guidance = vehicle.gat_guidance
                print(f"      GAT策略: {guidance.strategy}")
                print(f"      协调参数: 合作{guidance.cooperation_score:.2f}, 紧急{guidance.urgency_level:.2f}, 安全{guidance.safety_factor:.2f}")
            
            trajectory = self._plan_single_vehicle_enhanced(vehicle, successful_trajectories)
            
            if trajectory:
                vehicle.trajectory = trajectory
                successful_trajectories.append(trajectory)
                self.successful_plannings += 1
                print(f"      ✅ 成功: {len(trajectory)} 个轨迹点, 耗时{vehicle.planning_time:.2f}s")
            else:
                print(f"      ❌ 失败, 耗时{vehicle.planning_time:.2f}s")
        
        # 统计结果
        total_time = time.time() - self.planning_start_time
        success_rate = (self.successful_plannings / self.total_vehicles) * 100
        
        print(f"\n📊 规划结果总结:")
        print(f"   总车辆: {self.total_vehicles}")
        print(f"   成功: {self.successful_plannings}")
        print(f"   成功率: {success_rate:.1f}%")
        print(f"   总时间: {total_time:.2f}s")
        print(f"   平均时间: {total_time/self.total_vehicles:.2f}s/车")
        
        # 🆕 GAT性能统计
        if self.enable_gat:
            print(f"\n🧠 GAT系统性能:")
            print(f"   图构建: {self.gat_stats['graph_construction_time']:.3f}s")
            print(f"   推理时间: {self.gat_stats['inference_time']:.3f}s") 
            print(f"   决策解析: {self.gat_stats['decision_parsing_time']:.3f}s")
            print(f"   应用次数: {self.gat_stats['coordination_applications']}")
            
            total_gat_time = (self.gat_stats['graph_construction_time'] + 
                             self.gat_stats['inference_time'] + 
                             self.gat_stats['decision_parsing_time'])
            print(f"   GAT总耗时: {total_gat_time:.3f}s ({100*total_gat_time/total_time:.1f}%)")
        
        return success_rate >= 50  # 成功率超过50%认为成功
    
    def _plan_single_vehicle_enhanced(self, vehicle: Vehicle, existing_trajectories: List) -> Optional[List[VehicleState]]:
        """🆕 使用GAT指导的增强单车规划"""
        task = vehicle.task
        start_x, start_y = task.start_pos
        end_x, end_y = task.end_pos
        
        # 计算朝向
        dx = end_x - start_x
        dy = end_y - start_y
        theta = math.atan2(dy, dx)
        
        start_state = VehicleState(x=start_x, y=start_y, theta=theta, v=3.0, t=0.0)
        goal_state = VehicleState(x=end_x, y=end_y, theta=theta, v=2.0, t=0.0)
        
        planning_start = time.time()
        
        try:
            if self.enable_gat and vehicle.gat_guidance:
                # 🆕 使用GAT集成规划器
                trajectory = self.integrated_planner.plan_single_vehicle(
                    start_state, goal_state, vehicle.vehicle_id, 
                    vehicle.gat_guidance, existing_trajectories
                )
            else:
                # 使用传统规划器
                planner = VHybridAStarPlanner(self.environment, self.optimization_level)
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
    
    def print_detailed_results(self):
        """🆕 打印详细的规划结果分析"""
        print(f"\n📈 详细结果分析:")
        
        successful_vehicles = self.get_successful_vehicles()
        failed_vehicles = [v for v in self.vehicles if v.trajectory is None]
        
        print(f"\n✅ 成功车辆 ({len(successful_vehicles)}):")
        for vehicle in successful_vehicles:
            traj_length = len(vehicle.trajectory) if vehicle.trajectory else 0
            total_time = vehicle.trajectory[-1].t if vehicle.trajectory else 0
            avg_speed = sum(s.v for s in vehicle.trajectory) / len(vehicle.trajectory) if vehicle.trajectory else 0
            
            gat_info = ""
            if self.enable_gat and vehicle.gat_guidance:
                gat_info = f" | GAT: {vehicle.gat_guidance.strategy}"
            
            print(f"   V{vehicle.vehicle_id}: {traj_length}点, {total_time:.1f}s, {avg_speed:.1f}m/s{gat_info}")
        
        if failed_vehicles:
            print(f"\n❌ 失败车辆 ({len(failed_vehicles)}):")
            for vehicle in failed_vehicles:
                task_distance = math.sqrt((vehicle.task.end_pos[0] - vehicle.task.start_pos[0])**2 + 
                                        (vehicle.task.end_pos[1] - vehicle.task.start_pos[1])**2)
                print(f"   V{vehicle.vehicle_id}: 距离{task_distance:.1f}m, 规划时间{vehicle.planning_time:.2f}s")
        
        # 🆕 GAT效果分析
        if self.enable_gat and successful_vehicles:
            print(f"\n🧠 GAT效果分析:")
            strategy_count = {}
            for vehicle in successful_vehicles:
                if vehicle.gat_guidance:
                    strategy = vehicle.gat_guidance.strategy
                    strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
            
            print(f"   策略分布: {strategy_count}")
            
            cooperation_scores = [v.gat_guidance.cooperation_score for v in successful_vehicles if v.gat_guidance]
            if cooperation_scores:
                avg_cooperation = sum(cooperation_scores) / len(cooperation_scores)
                print(f"   平均合作度: {avg_cooperation:.3f}")

class EnhancedVisualizer:
    """🚀 增强版可视化器 - 支持GAT信息显示"""
    
    def __init__(self, planner: EnhancedFirstRoundPlanner):
        self.planner = planner
        self.fig, (self.ax_map, self.ax_stats) = plt.subplots(1, 2, figsize=(18, 9))
    
    def create_animation(self):
        """创建增强版动画，显示GAT协调信息"""
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
            
            # 绘制车辆和GAT信息
            active_count = 0
            gat_info_text = []
            
            for vehicle in successful_vehicles:
                current_state = self._get_state_at_time(vehicle.trajectory, current_time)
                if current_state:
                    self._draw_vehicle_with_gat_info(current_state, vehicle)
                    active_count += 1
                    
                    # 收集GAT信息
                    if self.planner.enable_gat and vehicle.gat_guidance:
                        gat_info_text.append(f"V{vehicle.vehicle_id}:{vehicle.gat_guidance.strategy}")
                
                # 绘制轨迹
                xs = [s.x for s in vehicle.trajectory]
                ys = [s.y for s in vehicle.trajectory]
                self.ax_map.plot(xs, ys, color=vehicle.color, alpha=0.6, linewidth=2)
            
            # 绘制任务起终点
            self._draw_tasks()
            
            # 🆕 标题包含GAT信息
            gat_status = "🧠GAT协调" if self.planner.enable_gat else "传统规划"
            title = f'增强版第一轮多车规划 - {self.planner.environment.map_name}\n'
            title += f'{gat_status} | 时间: {current_time:.1f}s | 活跃车辆: {active_count}'
            
            self.ax_map.set_title(title)
            self.ax_map.set_xlim(0, self.planner.environment.size)
            self.ax_map.set_ylim(0, self.planner.environment.size)
            self.ax_map.grid(True, alpha=0.3)
            
            # 🆕 显示GAT策略信息
            if gat_info_text:
                gat_text = " | ".join(gat_info_text)
                self.ax_map.text(0.02, 0.02, f"GAT策略: {gat_text}", 
                               transform=self.ax_map.transAxes, fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            
            # 增强统计图
            self._draw_enhanced_statistics()
            
            return []
        
        frames = int(max_time / 0.2) + 20
        anim = animation.FuncAnimation(self.fig, animate, frames=frames, 
                                     interval=200, blit=False, repeat=False)
        
        # 保存GIF
        try:
            writer = PillowWriter(fps=5)
            gat_suffix = "_gat" if self.planner.enable_gat else "_traditional"
            gif_filename = f"enhanced_first_round_{self.planner.environment.map_name}{gat_suffix}.gif"
            anim.save(gif_filename, writer=writer)
            print(f"✅ 动画已保存: {gif_filename}")
        except Exception as e:
            print(f"⚠️ 动画保存失败: {str(e)}")
        
        plt.tight_layout()
        plt.show()
        return anim
    
    def _draw_vehicle_with_gat_info(self, state: VehicleState, vehicle: Vehicle):
        """绘制带GAT信息的车辆"""
        # 绘制基础车辆
        self._draw_vehicle(state, vehicle.color)
        
        # 🆕 GAT信息显示
        if self.planner.enable_gat and vehicle.gat_guidance:
            guidance = vehicle.gat_guidance
            
            # 根据策略调整车辆边框样式
            linewidth = 2
            linestyle = '-'
            
            if guidance.strategy == "cooperative":
                linewidth = 3
                # 绘制合作指示
                self.ax_map.plot(state.x, state.y, 'o', color='green', 
                               markersize=6, alpha=0.7)
            elif guidance.strategy == "aggressive":
                linestyle = '--'
                linewidth = 3
            elif guidance.strategy == "defensive":
                # 绘制防御圈
                circle = plt.Circle((state.x, state.y), radius=2.0, 
                                  fill=False, color='orange', alpha=0.5)
                self.ax_map.add_patch(circle)
            elif guidance.strategy == "adaptive":
                linewidth = 2
                # 绘制适应性标记
                self.ax_map.plot(state.x, state.y, '^', color='purple', 
                               markersize=6, alpha=0.7)
            
            # 显示优先级调整
            if abs(guidance.priority_adjustment) > 0.1:
                priority_color = 'red' if guidance.priority_adjustment > 0 else 'blue'
                self.ax_map.text(state.x + 1, state.y + 1, 
                               f"{guidance.priority_adjustment:+.1f}", 
                               fontsize=8, color=priority_color, weight='bold')
    
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
            
            # 起点
            self.ax_map.plot(start_x, start_y, 'go', markersize=6)
            # 终点
            self.ax_map.plot(end_x, end_y, 'rs', markersize=6)
            # 连线
            self.ax_map.plot([start_x, end_x], [start_y, end_y], 
                           'k--', alpha=0.3, linewidth=1)
    
    def _draw_enhanced_statistics(self):
        """🆕 绘制增强统计信息（包含GAT信息）"""
        total = self.planner.total_vehicles
        successful = self.planner.successful_plannings
        success_rate = (successful / total) * 100 if total > 0 else 0
        
        # 基础饼图
        labels = ['成功', '失败']
        sizes = [successful, total - successful]
        colors = ['lightgreen', 'lightcoral']
        
        wedges, texts, autotexts = self.ax_stats.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
        
        # 🆕 GAT统计信息
        if self.planner.enable_gat:
            gat_text = f"GAT增强规划\n成功率: {success_rate:.1f}%\n"
            
            # 策略分布
            successful_vehicles = self.planner.get_successful_vehicles()
            strategy_count = {}
            for vehicle in successful_vehicles:
                if vehicle.gat_guidance:
                    strategy = vehicle.gat_guidance.strategy
                    strategy_count[strategy] = strategy_count.get(strategy, 0) + 1
            
            if strategy_count:
                gat_text += "策略分布:\n"
                for strategy, count in strategy_count.items():
                    gat_text += f"{strategy}: {count}\n"
            
            # GAT性能
            total_gat_time = (self.planner.gat_stats['graph_construction_time'] + 
                             self.planner.gat_stats['inference_time'] + 
                             self.planner.gat_stats['decision_parsing_time'])
            gat_text += f"GAT耗时: {total_gat_time:.3f}s"
            
        else:
            gat_text = f"传统规划\n成功率: {success_rate:.1f}%"
        
        self.ax_stats.set_title(gat_text)
    
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
    print("🚀 增强版第一轮多车轨迹规划系统")
    print("🧠 集成GAT智能协调 + 传统规划器")
    print("=" * 60)
    
    # 查找地图文件
    import os
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    map_files = [f for f in json_files if any(keyword in f.lower() 
                for keyword in ['lifelong', 'intersection', 'cross', 'junction'])]
    
    if not map_files:
        print("❌ 未找到地图文件，请使用 lifelong_map.py 创建")
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
    
    print(f"🎯 使用地图: {selected_map}")
    
    # 选择优化级别
    opt_levels = {
        '1': OptimizationLevel.BASIC,
        '2': OptimizationLevel.ENHANCED, 
        '3': OptimizationLevel.FULL
    }
    
    choice = input("优化级别 (1=BASIC, 2=ENHANCED, 3=FULL) 或回车使用ENHANCED: ").strip()
    opt_level = opt_levels.get(choice, OptimizationLevel.ENHANCED)
    
    # 🆕 GAT选项
    gat_choice = input("启用GAT智能协调? (y/N) 或回车使用默认: ").strip().lower()
    enable_gat = gat_choice in ['y', 'yes'] if gat_choice else HAS_GAT
    
    print(f"🎯 优化级别: {opt_level.value}")
    print(f"🧠 GAT智能协调: {'✅ 启用' if enable_gat else '❌ 禁用'}")
    
    try:
        # 创建增强规划器
        planner = EnhancedFirstRoundPlanner(selected_map, opt_level, enable_gat)
        
        # 应用智能优先级（如果可用）
        planner.apply_intelligent_priorities()
        
        # 🆕 应用GAT智能协调
        planner.apply_gat_coordination()
        
        # 执行规划
        success = planner.plan_all_vehicles()
        
        # 🆕 打印详细结果
        planner.print_detailed_results()
        
        if success:
            # 创建增强可视化
            visualizer = EnhancedVisualizer(planner)
            visualizer.create_animation()
            print("🎉 增强版第一轮规划完成！")
        else:
            print("⚠️ 规划成功率较低")
            
    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()