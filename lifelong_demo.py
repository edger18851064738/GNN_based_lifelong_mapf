#!/usr/bin/env python3
"""
Lifelong MAPF 演示程序
基于Enhanced V-Hybrid A*算法和出入口边地图

功能：
1. 加载lifelong_map.py创建的出入口边地图
2. 在出入口边上生成随机任务（整数坐标）
3. 使用Enhanced V-Hybrid A*进行路径规划
4. 可视化结果
"""

# 从demo.py导入所有必要的组件
from demo import (
    # 核心算法组件
    VHybridAStarPlanner, MultiVehicleCoordinator, UnstructuredEnvironment,
    # 数据结构
    VehicleState, VehicleParameters, OptimizationLevel,
    # 工具类
    TimeSync, ConflictDensityAnalyzer, AdaptiveTimeResolution,
    ImprovedIntermediateNodeGenerator, AdvancedBoxConstraints,
    # 其他必要组件
    EfficientDubinsPath, FastConflictDetector, OptimizedTrajectoryProcessor
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
import json
import os
import random
import math
import time
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class LifelongEnvironment(UnstructuredEnvironment):
    """🚀 Lifelong专用环境：针对出入口边场景优化的碰撞检测"""
    
    def __init__(self, size=100):
        super().__init__(size)
        
    def is_collision_free(self, state: VehicleState, params: VehicleParameters):
        """改进的碰撞检测，适用于出入口边在边界的情况"""
        # 🚀 修复1：更宽松的边界检查 - 允许车辆中心靠近边界
        boundary_margin = 1.0  # 减小边界边距
        
        # 车辆中心必须在地图范围内，但允许更接近边界
        if not (boundary_margin <= state.x <= self.size - boundary_margin and 
                boundary_margin <= state.y <= self.size - boundary_margin):
            # 对于边界位置，进一步放宽检查
            if not (0 <= state.x < self.size and 0 <= state.y < self.size):
                return False
        
        # 🚀 修复2：更智能的车辆角点碰撞检测
        cos_theta, sin_theta = math.cos(state.theta), math.sin(state.theta)
        
        half_length, half_width = params.length/2, params.width/2
        corners = [
            (-half_length, -half_width), (half_length, -half_width),
            (half_length, half_width), (-half_length, half_width)
        ]
        
        # 检查车辆四角
        valid_corners = 0
        total_corners = len(corners)
        
        for lx, ly in corners:
            gx = state.x + lx * cos_theta - ly * sin_theta
            gy = state.y + lx * sin_theta + ly * cos_theta
            
            # 如果角点在地图内，检查是否有障碍物
            if (0 <= gx < self.size and 0 <= gy < self.size):
                if self.is_valid_position(gx, gy):
                    valid_corners += 1
                # 如果角点撞到静态障碍物，这是真正的碰撞
                else:
                    return False
            else:
                # 角点在地图外，这对于边界出入口边是允许的
                valid_corners += 1
        
        # 🚀 修复3：至少要有一半的角点是有效的
        if valid_corners < total_corners // 2:
            return False
        
        # 🚀 修复4：动态障碍物检测保持原有逻辑
        time_key = TimeSync.get_time_key(state)
        if time_key in self.dynamic_obstacles:
            vehicle_cells = self._get_vehicle_cells_fast(state, params)
            if vehicle_cells.intersection(self.dynamic_obstacles[time_key]):
                return False
        
        return True

@dataclass
class LifelongTask:
    """Lifelong MAPF任务"""
    task_id: int
    source_gateway_id: int
    target_gateway_id: int
    start_point: Tuple[int, int]  # 整数坐标
    goal_point: Tuple[int, int]   # 整数坐标
    priority: int = 1
    creation_time: float = 0.0

class LifelongMapLoader:
    """Lifelong地图加载器"""
    
    def __init__(self):
        self.map_data = None
        self.gateways = []
        self.environment = None
        
    def load_map(self, map_file_path: str) -> bool:
        """加载lifelong地图文件"""
        try:
            with open(map_file_path, 'r', encoding='utf-8') as f:
                self.map_data = json.load(f)
            
            # 检查是否为lifelong地图
            map_info = self.map_data.get("map_info", {})
            map_type = map_info.get("map_type", "")
            
            if map_type != "lifelong_gateway":
                print(f"⚠️ 警告：地图类型为 {map_type}，可能不是lifelong地图")
            
            # 提取出入口边信息
            self.gateways = self.map_data.get("gateways", [])
            
            # 创建环境 - 🚀 使用lifelong专用环境
            width = map_info.get("width", 60)
            height = map_info.get("height", 60)
            
            self.environment = LifelongEnvironment(size=max(width, height))
            
            # 加载网格数据
            if "grid" in self.map_data:
                grid = np.array(self.map_data["grid"], dtype=np.int8)
                for row in range(min(grid.shape[0], self.environment.size)):
                    for col in range(min(grid.shape[1], self.environment.size)):
                        if grid[row, col] == 1:
                            self.environment.obstacle_map[row, col] = True
            
            self.environment.map_name = map_info.get("name", "lifelong_map")
            
            print(f"✅ 成功加载lifelong地图: {self.environment.map_name}")
            print(f"   地图大小: {width}x{height}")
            print(f"   出入口边数量: {len(self.gateways)}")
            
            return True
            
        except Exception as e:
            print(f"❌ 加载地图失败: {str(e)}")
            return False
    
    def get_gateway_info(self) -> Dict:
        """获取出入口边统计信息"""
        if not self.gateways:
            return {}
        
        info = {
            'total_gateways': len(self.gateways),
            'total_capacity': sum(g['capacity'] for g in self.gateways),
            'types': {}
        }
        
        for gateway in self.gateways:
            gateway_type = gateway['type']
            if gateway_type not in info['types']:
                info['types'][gateway_type] = {'count': 0, 'capacity': 0}
            info['types'][gateway_type]['count'] += 1
            info['types'][gateway_type]['capacity'] += gateway['capacity']
        
        return info

class LifelongTaskGenerator:
    """Lifelong任务生成器"""
    
    def __init__(self, gateways: List[Dict]):
        self.gateways = gateways
        self.task_counter = 1
        
    def generate_random_point_on_gateway(self, gateway: Dict) -> Tuple[int, int]:
        """在出入口边上生成随机点并取整"""
        # 在出入口边上随机采样
        t = random.uniform(0, 1)
        x = gateway['start_x'] + t * (gateway['end_x'] - gateway['start_x'])
        y = gateway['start_y'] + t * (gateway['end_y'] - gateway['start_y'])
        
        # 取整确保坐标为整数
        return (int(round(x)), int(round(y)))
    
    def generate_single_task(self, source_gateway_id: int = None, 
                           target_gateway_id: int = None) -> Optional[LifelongTask]:
        """生成单个任务"""
        if len(self.gateways) < 2:
            return None
        
        # 选择源出入口边
        if source_gateway_id is None:
            source_gateway = random.choice(self.gateways)
        else:
            source_gateway = next((g for g in self.gateways if g['id'] == source_gateway_id), None)
            if not source_gateway:
                return None
        
        # 选择目标出入口边（不能与源相同）
        available_targets = [g for g in self.gateways if g['id'] != source_gateway['id']]
        if not available_targets:
            return None
            
        if target_gateway_id is None:
            target_gateway = random.choice(available_targets)
        else:
            target_gateway = next((g for g in available_targets if g['id'] == target_gateway_id), None)
            if not target_gateway:
                target_gateway = random.choice(available_targets)
        
        # 生成起点和终点
        start_point = self.generate_random_point_on_gateway(source_gateway)
        goal_point = self.generate_random_point_on_gateway(target_gateway)
        
        task = LifelongTask(
            task_id=self.task_counter,
            source_gateway_id=source_gateway['id'],
            target_gateway_id=target_gateway['id'],
            start_point=start_point,
            goal_point=goal_point,
            priority=1,
            creation_time=time.time()
        )
        
        self.task_counter += 1
        return task
    
    def generate_batch_tasks(self, num_tasks: int, 
                           enforce_diversity: bool = True) -> List[LifelongTask]:
        """生成批量任务"""
        tasks = []
        
        if enforce_diversity and len(self.gateways) >= 2:
            # 确保每个出入口边都有机会作为起点
            for _ in range(num_tasks):
                # 轮流选择不同的源出入口边
                source_gateway = self.gateways[len(tasks) % len(self.gateways)]
                task = self.generate_single_task(source_gateway['id'])
                if task:
                    tasks.append(task)
        else:
            # 完全随机生成
            for _ in range(num_tasks):
                task = self.generate_single_task()
                if task:
                    tasks.append(task)
        
        return tasks

class LifelongCoordinator:
    """Lifelong MAPF协调器"""
    
    def __init__(self, map_loader: LifelongMapLoader, 
                 optimization_level: OptimizationLevel = OptimizationLevel.ENHANCED):
        self.map_loader = map_loader
        self.environment = map_loader.environment
        self.gateways = map_loader.gateways
        self.optimization_level = optimization_level
        
        # 任务生成器
        self.task_generator = LifelongTaskGenerator(self.gateways)
        
        # 车辆参数
        self.params = VehicleParameters()
        
        print(f"🚀 Lifelong MAPF协调器初始化完成")
        print(f"   优化级别: {optimization_level.value}")
        print(f"   地图: {self.environment.map_name}")
        print(f"   出入口边数量: {len(self.gateways)}")
        print(f"   ✅ 已启用等待机制和改进碰撞检测（适用于多车协调）")
    
    def plan_lifelong_tasks(self, tasks: List[LifelongTask]) -> Dict:
        """规划lifelong任务"""
        print(f"\n🔄 开始Lifelong MAPF规划...")
        print(f"   任务数量: {len(tasks)}")
        
        # 转换为demo.py的格式
        scenarios = []
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
        
        for i, task in enumerate(tasks):
            # 创建起始状态
            start_x, start_y = task.start_point
            goal_x, goal_y = task.goal_point
            
            # 计算最优朝向
            dx = goal_x - start_x
            dy = goal_y - start_y
            optimal_theta = math.atan2(dy, dx) if abs(dx) > 0.1 or abs(dy) > 0.1 else 0.0
            
            start_state = VehicleState(
                x=float(start_x),
                y=float(start_y),
                theta=optimal_theta,
                v=3.0,
                t=0.0
            )
            
            goal_state = VehicleState(
                x=float(goal_x),
                y=float(goal_y),
                theta=optimal_theta,
                v=2.0,
                t=0.0
            )
            
            scenario = {
                'id': task.task_id,
                'priority': len(tasks) - i,  # 按顺序分配优先级
                'color': colors[i % len(colors)],
                'start': start_state,
                'goal': goal_state,
                'description': f'Lifelong Task {task.task_id} (G{task.source_gateway_id}→G{task.target_gateway_id})',
                'lifelong_task': task  # 保存原始任务信息
            }
            
            scenarios.append(scenario)
            print(f"   T{task.task_id}: G{task.source_gateway_id}({start_x},{start_y}) → G{task.target_gateway_id}({goal_x},{goal_y})")
        
        # 使用原有的规划逻辑
        sorted_scenarios = sorted(scenarios, key=lambda x: x['priority'], reverse=True)
        
        results = {}
        high_priority_trajectories = []
        
        print(f"\n🚀 Lifelong规划执行...")
        
        for i, scenario in enumerate(sorted_scenarios):
            print(f"\n--- Lifelong Task {scenario['id']} (Priority {scenario['priority']}) ---")
            print(f"Description: {scenario['description']}")
            
            vehicle_start_time = time.time()
            
            # 创建规划器 - 🚀 为lifelong场景优化搜索参数
            planner = VHybridAStarPlanner(self.environment, self.optimization_level)
            
            # 🚀 关键改进：增加最大迭代次数以提高成功率
            if self.optimization_level == OptimizationLevel.BASIC:
                planner.max_iterations = 15000  # 从8000增加
            elif self.optimization_level == OptimizationLevel.ENHANCED:
                planner.max_iterations = 20000  # 从12000增加
            else:
                planner.max_iterations = 25000  # 从15000增加
            
            print(f"      🎯 Max iterations set to {planner.max_iterations} for lifelong planning")
            
            # 🚀 使用专门的lifelong规划方法（包含等待机制）
            trajectory = self.lifelong_search_with_waiting(
                planner, scenario['start'], scenario['goal'], scenario['id'], 
                high_priority_trajectories)
            
            # 🚀 新增：如果规划失败，尝试重新生成任务点
            if not trajectory and i < len(sorted_scenarios) - 1:  # 不对最后一个任务重试，避免无限循环
                print(f"      🔄 Planning failed, trying alternative start/goal points...")
                
                # 重新生成该任务的起点和终点
                original_task = scenario['lifelong_task']
                
                for retry in range(3):  # 最多重试3次
                    print(f"      Retry {retry + 1}/3...")
                    
                    # 重新生成起点和终点
                    new_start_point = self.task_generator.generate_random_point_on_gateway(
                        next(g for g in self.gateways if g['id'] == original_task.source_gateway_id))
                    new_goal_point = self.task_generator.generate_random_point_on_gateway(
                        next(g for g in self.gateways if g['id'] == original_task.target_gateway_id))
                    
                    # 创建新的状态
                    start_x, start_y = new_start_point
                    goal_x, goal_y = new_goal_point
                    
                    dx = goal_x - start_x
                    dy = goal_y - start_y
                    optimal_theta = math.atan2(dy, dx) if abs(dx) > 0.1 or abs(dy) > 0.1 else 0.0
                    
                    new_start_state = VehicleState(
                        x=float(start_x), y=float(start_y), theta=optimal_theta, v=3.0, t=0.0)
                    new_goal_state = VehicleState(
                        x=float(goal_x), y=float(goal_y), theta=optimal_theta, v=2.0, t=0.0)
                    
                    print(f"        New points: ({start_x},{start_y}) -> ({goal_x},{goal_y})")
                    
                    # 重新尝试规划
                    trajectory = self.lifelong_search_with_waiting(
                        planner, new_start_state, new_goal_state, scenario['id'], 
                        high_priority_trajectories)
                    
                    if trajectory:
                        print(f"      ✅ Retry {retry + 1} succeeded!")
                        # 更新scenario中的状态
                        scenario['start'] = new_start_state
                        scenario['goal'] = new_goal_state
                        break
                else:
                    print(f"      ❌ All retries failed")
            
            vehicle_planning_time = time.time() - vehicle_start_time
            
            if trajectory:
                print(f"SUCCESS: {len(trajectory)} waypoints, time: {trajectory[-1].t:.1f}s, planning: {vehicle_planning_time:.2f}s")
                
                results[scenario['id']] = {
                    'trajectory': trajectory,
                    'color': scenario['color'],
                    'description': scenario['description'],
                    'planning_time': vehicle_planning_time,
                    'lifelong_task': scenario['lifelong_task']
                }
                
                # 添加为动态障碍物
                self.environment.add_vehicle_trajectory(trajectory, self.params)
                high_priority_trajectories.append(trajectory)
                print(f"Added as dynamic obstacle for remaining {len(sorted_scenarios)-i-1} tasks")
            else:
                print(f"FAILED: No feasible trajectory, planning: {vehicle_planning_time:.2f}s")
                results[scenario['id']] = {
                    'trajectory': [], 
                    'color': scenario['color'], 
                    'description': scenario['description'],
                    'planning_time': vehicle_planning_time,
                    'lifelong_task': scenario['lifelong_task']
                }
        
        return results, sorted_scenarios
    
    def is_lifelong_start_position_blocked(self, start: VehicleState) -> bool:
        """🚀 Lifelong专用：检查起始位置是否被动态障碍物阻塞（忽略边界问题）"""
        # 只检查动态障碍物，不检查静态边界碰撞
        start_cells = self.environment._get_vehicle_cells_fast(start, self.params)
        time_key = TimeSync.get_time_key(start)
        
        if time_key in self.environment.dynamic_obstacles:
            if start_cells.intersection(self.environment.dynamic_obstacles[time_key]):
                return True
        return False
    
    def find_lifelong_safe_start_time(self, start: VehicleState, max_delay: float = 25.0) -> Optional[float]:
        """🚀 Lifelong专用：为被阻塞的起始位置找到安全启动时间"""
        # 使用更精细的时间步长和更长的最大延迟
        for delay in np.arange(0.5, max_delay, 0.5):  # 从0.5秒开始，每0.5秒检查一次
            test_state = start.copy()
            test_state.t = start.t + delay
            
            if not self.is_lifelong_start_position_blocked(test_state):
                return delay
        
        return None
    
    def lifelong_search_with_waiting(self, planner: VHybridAStarPlanner, 
                                   start: VehicleState, goal: VehicleState, 
                                   vehicle_id: int = None, 
                                   high_priority_trajectories: List = None) -> Optional[List]:
        """🚀 Lifelong专用搜索方法：支持等待机制但放宽边界限制"""
        print(f"    🚀 Enhanced lifelong planning vehicle {vehicle_id}: ({start.x:.1f},{start.y:.1f}) -> ({goal.x:.1f},{goal.y:.1f})")
        
        # 基本位置有效性检查
        start_valid = (0 <= start.x < self.environment.size and 0 <= start.y < self.environment.size)
        goal_valid = (0 <= goal.x < self.environment.size and 0 <= goal.y < self.environment.size)
        
        print(f"      起始位置检查: 坐标有效={start_valid}")
        print(f"      目标位置检查: 坐标有效={goal_valid}")
        
        if not start_valid or not goal_valid:
            print(f"      ❌ 起始或目标位置超出地图范围")
            return None
        
        # 🚀 关键修复：恢复等待机制，但使用lifelong专用的阻塞检查
        if self.is_lifelong_start_position_blocked(start):
            print(f"      Start position blocked by dynamic obstacles, finding safe start time...")
            safe_delay = self.find_lifelong_safe_start_time(start)
            
            if safe_delay is not None:
                print(f"      Waiting {safe_delay:.1f}s for safe start")
                delayed_start = start.copy()
                delayed_start.t = start.t + safe_delay
                return planner.search(delayed_start, goal, high_priority_trajectories)
            else:
                print(f"      No safe start time found within reasonable delay")
                return None
        else:
            # 直接搜索
            return planner.search(start, goal, high_priority_trajectories)
    
    def create_lifelong_animation(self, results: Dict, scenarios: List, tasks: List[LifelongTask]):
        """创建Lifelong MAPF动画"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        self._setup_environment_plot(ax1)
        
        # 收集成功的轨迹
        all_trajectories = []
        for scenario in scenarios:
            vid = scenario['id']
            if vid in results and results[vid]['trajectory']:
                traj = results[vid]['trajectory']
                color = results[vid]['color']
                all_trajectories.append((traj, color, scenario['description']))
        
        if not all_trajectories:
            print("❌ 没有成功的轨迹用于动画")
            return None
        
        max_time = max(max(state.t for state in traj) for traj, _, _ in all_trajectories)
        
        def save_gif(anim, filename, fps=8):
            """保存GIF动画"""
            try:
                print(f"🎬 正在保存Lifelong动画: {filename}")
                writer = PillowWriter(fps=fps)
                anim.save(filename, writer=writer)
                print(f"✅ 动画已保存: {filename}")
            except Exception as e:
                print(f"❌ 保存失败: {e}")
        
        def animate(frame):
            ax1.clear()
            ax2.clear()
            
            self._setup_environment_plot(ax1)
            
            current_time = frame * 0.5
            
            # 绘制活跃车辆
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
                    
                    # 绘制轨迹历史
                    past_states = [s for s in traj if s.t <= current_time]
                    if len(past_states) > 1:
                        xs = [s.x for s in past_states]
                        ys = [s.y for s in past_states]
                        ax1.plot(xs, ys, color=color, alpha=0.6, linewidth=2)
            
            # 绘制出入口边和任务信息
            self._draw_gateways(ax1)
            
            ax1.set_title(f'🔄 Lifelong MAPF with Enhanced V-Hybrid A* ({self.optimization_level.value})\n'
                         f'[{self.environment.map_name}] t={current_time:.1f}s, Active:{active_vehicles}, Tasks:{len(tasks)}')
            
            # 绘制时间线
            self._draw_lifelong_timeline(ax2, all_trajectories, current_time, tasks)
            
            return []
        
        frames = int(max_time / 0.5) + 10
        anim = animation.FuncAnimation(fig, animate, frames=frames, interval=200, blit=False)
        
        # 保存动画
        gif_filename = f"lifelong_{self.environment.map_name}_{self.optimization_level.value}.gif"
        save_gif(anim, gif_filename)
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def _setup_environment_plot(self, ax):
        """设置环境绘图"""
        # 绘制背景
        ax.add_patch(patches.Rectangle((0, 0), self.environment.size, self.environment.size,
                                     facecolor='lightgray', alpha=0.1))
        
        # 绘制可通行区域
        free_y, free_x = np.where(~self.environment.obstacle_map)
        ax.scatter(free_x, free_y, c='lightblue', s=1, alpha=0.3)
        
        # 绘制障碍物
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
        
        # 绘制方向箭头
        arrow_length = 3
        dx = arrow_length * cos_theta
        dy = arrow_length * sin_theta
        ax.arrow(state.x, state.y, dx, dy, head_width=1, head_length=1,
                fc=color, ec='black', alpha=0.9)
    
    def _draw_gateways(self, ax):
        """绘制出入口边"""
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#F4A460", "#87CEEB"]
        
        for i, gateway in enumerate(self.gateways):
            color = colors[i % len(colors)]
            
            start_x = gateway["start_x"]
            start_y = gateway["start_y"]
            end_x = gateway["end_x"]
            end_y = gateway["end_y"]
            
            # 绘制出入口边主线
            ax.plot([start_x, end_x], [start_y, end_y], 
                   color=color, linewidth=8, alpha=0.7, solid_capstyle='round')
            
            # 绘制标识
            mid_x = (start_x + end_x) / 2
            mid_y = (start_y + end_y) / 2
            
            # 背景圆圈
            circle = patches.Circle((mid_x, mid_y), 2, facecolor='white', 
                                  edgecolor=color, linewidth=2)
            ax.add_patch(circle)
            
            # ID标签
            ax.text(mid_x, mid_y, str(gateway['id']),
                   ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    
    def _draw_lifelong_timeline(self, ax, all_trajectories, current_time, tasks):
        """绘制Lifelong时间线"""
        ax.set_title(f'Lifelong MAPF Timeline - {self.environment.map_name} ({self.optimization_level.value})')
        
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
            
            # 任务信息
            wait_info = f" (wait {start_time:.0f}s)" if start_time > 0 else ""
            ax.text(max(times) + 1, y_pos, desc + wait_info, fontsize=10, va='center')
        
        ax.axvline(x=current_time, color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Task')
        ax.grid(True, alpha=0.3)

def interactive_lifelong_map_selection():
    """交互式选择lifelong地图文件"""
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    
    if not json_files:
        print("❌ 当前目录没有找到JSON地图文件")
        return None
    
    # 过滤lifelong地图文件
    lifelong_files = []
    for file in json_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                map_info = data.get('map_info', {})
                if map_info.get('map_type') == 'lifelong_gateway' or 'gateways' in data:
                    lifelong_files.append(file)
        except:
            continue
    
    if not lifelong_files:
        print("❌ 没有找到lifelong地图文件")
        print("请使用lifelong_map.py创建包含出入口边的地图")
        return None
    
    print(f"\n📁 发现 {len(lifelong_files)} 个Lifelong地图文件:")
    for i, file in enumerate(lifelong_files):
        try:
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                map_info = data.get('map_info', {})
                name = map_info.get('name', file)
                width = map_info.get('width', '未知')
                height = map_info.get('height', '未知')
                gateways = len(data.get('gateways', []))
                print(f"  {i+1}. {file}")
                print(f"     名称: {name}")
                print(f"     大小: {width}x{height}")
                print(f"     出入口边: {gateways} 个")
        except:
            print(f"  {i+1}. {file} (无法读取详细信息)")
    
    while True:
        try:
            choice = input(f"\n🎯 请选择地图文件 (1-{len(lifelong_files)}) 或按Enter使用第1个: ").strip()
            if choice == "":
                return lifelong_files[0]
            
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(lifelong_files):
                return lifelong_files[choice_idx]
            else:
                print(f"❌ 请输入 1-{len(lifelong_files)} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")

def main():
    """Lifelong MAPF演示主函数"""
    print("🔄 Lifelong MAPF 演示程序")
    print("📄 基于Enhanced V-Hybrid A*和出入口边地图")
    print("=" * 60)
    
    # 选择地图文件
    selected_file = interactive_lifelong_map_selection()
    if not selected_file:
        print("❌ 未选择有效的地图文件")
        return
    
    print(f"\n🎯 使用地图文件: {selected_file}")
    
    # 加载地图
    map_loader = LifelongMapLoader()
    if not map_loader.load_map(selected_file):
        print("❌ 地图加载失败")
        return
    
    # 显示地图信息
    gateway_info = map_loader.get_gateway_info()
    print(f"\n📊 地图统计:")
    print(f"   总出入口边: {gateway_info.get('total_gateways', 0)} 个")
    print(f"   总容量: {gateway_info.get('total_capacity', 0)} 辆车")
    if 'types' in gateway_info:
        for gate_type, info in gateway_info['types'].items():
            print(f"   {gate_type}: {info['count']} 个 (容量 {info['capacity']})")
    
    # 创建协调器
    optimization_level = OptimizationLevel.ENHANCED  # 可以修改为FULL或BASIC
    coordinator = LifelongCoordinator(map_loader, optimization_level)
    
    # 获取任务数量
    while True:
        try:
            num_tasks = input(f"\n🚗 请输入要生成的任务数量 (1-{len(map_loader.gateways)*2}, 默认4): ").strip()
            if num_tasks == "":
                num_tasks = 4
            else:
                num_tasks = int(num_tasks)
            
            if 1 <= num_tasks <= len(map_loader.gateways) * 2:
                break
            else:
                print(f"❌ 请输入 1-{len(map_loader.gateways)*2} 之间的数字")
        except ValueError:
            print("❌ 请输入有效的数字")
    
    # 生成任务
    print(f"\n🎲 生成 {num_tasks} 个lifelong任务...")
    tasks = coordinator.task_generator.generate_batch_tasks(num_tasks, enforce_diversity=True)
    
    if not tasks:
        print("❌ 任务生成失败")
        return
    
    print(f"✅ 成功生成 {len(tasks)} 个任务:")
    for task in tasks:
        print(f"   T{task.task_id}: G{task.source_gateway_id}({task.start_point[0]},{task.start_point[1]}) → "
              f"G{task.target_gateway_id}({task.goal_point[0]},{task.goal_point[1]})")
    
    # 执行规划
    print(f"\n⏱️ 开始Lifelong MAPF规划...")
    start_time = time.time()
    results, scenarios = coordinator.plan_lifelong_tasks(tasks)
    planning_time = time.time() - start_time
    
    # 统计结果
    success_count = sum(1 for tid in results if results[tid]['trajectory'])
    avg_planning_time = sum(results[tid].get('planning_time', 0) for tid in results) / len(results) if results else 0
    
    print(f"\n📊 Lifelong规划结果:")
    print(f"总规划时间: {planning_time:.2f}s")
    print(f"平均单任务规划时间: {avg_planning_time:.2f}s")
    print(f"成功率: {success_count}/{len(tasks)} ({100*success_count/len(tasks):.1f}%)")
    print(f"优化级别: {optimization_level.value}")
    
    if success_count >= 1:
        print(f"\n🎬 创建Lifelong MAPF动画...")
        anim = coordinator.create_lifelong_animation(results, scenarios, tasks)
        
        # 保存轨迹数据
        trajectory_file = f"lifelong_{coordinator.environment.map_name}_{optimization_level.value}.json"
        save_lifelong_trajectories(results, tasks, trajectory_file)
        
        print(f"\n✨ Lifelong MAPF特性:")
        print(f"  ✅ 出入口边任务生成: 在{len(map_loader.gateways)}个出入口边间生成任务")
        print(f"  ✅ 整数坐标处理: 起点终点坐标已取整")
        print(f"  ✅ 智能等待机制: 动态障碍物阻塞时自动寻找安全启动时间")
        print(f"  ✅ 改进碰撞检测: 平衡边界出入口边支持和准确碰撞检测")
        print(f"  ✅ 增强搜索参数: 更高迭代次数，提高复杂场景成功率")
        print(f"  ✅ 任务重试机制: 失败任务自动重新生成起点终点并重试")
        print(f"  ✅ Enhanced V-Hybrid A*: 使用增强算法进行路径规划")
        print(f"  ✅ 动态避障: 后续任务避开已规划轨迹")
        print(f"  ✅ 可视化展示: 出入口边和任务执行过程")
        
        input("按Enter键退出...")
    else:
        print("❌ 没有成功的任务用于可视化")
    
    print("\n🎉 Lifelong MAPF演示完成!")

def save_lifelong_trajectories(results: Dict, tasks: List[LifelongTask], filename: str):
    """保存lifelong轨迹数据"""
    trajectory_data = {
        'metadata': {
            'timestamp': time.time(),
            'algorithm': '🔄 Lifelong MAPF with Enhanced V-Hybrid A*',
            'performance_metrics': {
                'total_tasks': len(results),
                'successful_tasks': sum(1 for tid in results if results[tid].get('trajectory')),
                'avg_planning_time': sum(results[tid].get('planning_time', 0) for tid in results) / len(results) if results else 0,
                'lifelong_features': [
                    'Gateway-based Task Generation',
                    'Integer Coordinate Processing',
                    'Enhanced V-Hybrid A* Planning',
                    'Dynamic Obstacle Avoidance',
                    'Interactive Visualization'
                ]
            }
        },
        'tasks': [],
        'trajectories': {}
    }
    
    # 保存任务信息
    for task in tasks:
        trajectory_data['tasks'].append({
            'task_id': task.task_id,
            'source_gateway_id': task.source_gateway_id,
            'target_gateway_id': task.target_gateway_id,
            'start_point': task.start_point,
            'goal_point': task.goal_point,
            'priority': task.priority,
            'creation_time': task.creation_time
        })
    
    # 保存轨迹数据
    for tid, result in results.items():
        if result.get('trajectory'):
            trajectory_data['trajectories'][f"task_{tid}"] = {
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
        print(f"💾 Lifelong轨迹数据已保存到: {filename}")
    except Exception as e:
        print(f"❌ 保存轨迹数据失败: {str(e)}")

if __name__ == "__main__":
    main()