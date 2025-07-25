#!/usr/bin/env python3
"""
🧠 GAT训练解决方案 - 为lifelong_planning提供训练好的GAT模型

解决方案包括：
1. 启发式规则基础的GAT替代方案
2. 简单的监督学习训练
3. 自监督学习训练  
4. 强化学习训练框架
5. 预训练权重保存/加载
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import math
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# 尝试导入GAT相关模块
try:
    from GAT import (
        VehicleGraphBuilder, VehicleGATNetwork, DecisionParser,
        CoordinationGuidance, VehicleGraphData
    )
    from trying import VehicleState
    HAS_GAT = True
except ImportError:
    HAS_GAT = False

@dataclass
class TrainingData:
    """训练数据结构"""
    graph_data: 'VehicleGraphData'
    optimal_decisions: Dict[str, float]  # 最优决策标签
    scenario_info: Dict
    
@dataclass 
class TrainingConfig:
    """训练配置"""
    learning_rate: float = 0.001
    batch_size: int = 16
    num_epochs: int = 100
    save_interval: int = 10
    model_save_path: str = "gat_coordination_model.pth"

class HeuristicCoordinator:
    """🎯 启发式协调器 - GAT的规则基础替代方案"""
    
    def __init__(self):
        self.strategy_rules = {
            'distance_based': self._distance_strategy,
            'density_based': self._density_strategy, 
            'priority_based': self._priority_strategy,
            'hybrid': self._hybrid_strategy
        }
    
    def generate_coordination_guidance(self, vehicles_info: List[Dict]) -> List[CoordinationGuidance]:
        """生成启发式协调指导"""
        print("🎯 使用启发式协调器 (GAT替代方案)")
        
        guidance_list = []
        
        for vehicle_info in vehicles_info:
            # 分析车辆情况
            analysis = self._analyze_vehicle_situation(vehicle_info, vehicles_info)
            
            # 生成协调指导
            guidance = CoordinationGuidance(
                vehicle_id=vehicle_info['id'],
                strategy=analysis['strategy'],
                priority_adjustment=analysis['priority_adjustment'],
                cooperation_score=analysis['cooperation_score'],
                urgency_level=analysis['urgency_level'],
                safety_factor=analysis['safety_factor'],
                adjusted_priority=vehicle_info['priority'] + analysis['priority_adjustment']
            )
            
            guidance_list.append(guidance)
            print(f"   V{vehicle_info['id']}: {guidance.strategy}, "
                  f"优先级调整{guidance.priority_adjustment:+.1f}")
        
        return guidance_list
    
    def _analyze_vehicle_situation(self, vehicle: Dict, all_vehicles: List[Dict]) -> Dict:
        """分析单个车辆的情况"""
        current = vehicle['current_state']
        goal = vehicle['goal_state']
        
        # 1. 距离分析
        distance_to_goal = math.sqrt((goal.x - current.x)**2 + (goal.y - current.y)**2)
        
        # 2. 密度分析  
        nearby_count = self._count_nearby_vehicles(vehicle, all_vehicles, radius=20.0)
        
        # 3. 冲突分析
        conflict_count = self._count_potential_conflicts(vehicle, all_vehicles)
        
        # 4. 路径复杂度分析
        path_complexity = self._assess_path_complexity(current, goal)
        
        # 5. 规则决策
        if conflict_count >= 2:
            strategy = "defensive"
            priority_adjustment = -0.5
            cooperation_score = 0.8
            urgency_level = 0.3
            safety_factor = 0.9
        elif distance_to_goal > 50:
            strategy = "normal"
            priority_adjustment = 0.0
            cooperation_score = 0.5
            urgency_level = 0.5
            safety_factor = 0.5
        elif nearby_count >= 3:
            strategy = "cooperative"
            priority_adjustment = 0.2
            cooperation_score = 0.9
            urgency_level = 0.4
            safety_factor = 0.7
        elif path_complexity > 0.7:
            strategy = "adaptive"
            priority_adjustment = 0.1
            cooperation_score = 0.6
            urgency_level = 0.6
            safety_factor = 0.6
        else:
            strategy = "normal"
            priority_adjustment = 0.0
            cooperation_score = 0.5
            urgency_level = 0.5
            safety_factor = 0.5
        
        return {
            'strategy': strategy,
            'priority_adjustment': priority_adjustment,
            'cooperation_score': cooperation_score,
            'urgency_level': urgency_level,
            'safety_factor': safety_factor,
            'analysis': {
                'distance_to_goal': distance_to_goal,
                'nearby_count': nearby_count,
                'conflict_count': conflict_count,
                'path_complexity': path_complexity
            }
        }
    
    def _count_nearby_vehicles(self, vehicle: Dict, all_vehicles: List[Dict], radius: float) -> int:
        """计算附近车辆数量"""
        current = vehicle['current_state']
        count = 0
        
        for other in all_vehicles:
            if other['id'] == vehicle['id']:
                continue
            
            other_pos = other['current_state']
            distance = math.sqrt((current.x - other_pos.x)**2 + (current.y - other_pos.y)**2)
            
            if distance <= radius:
                count += 1
        
        return count
    
    def _count_potential_conflicts(self, vehicle: Dict, all_vehicles: List[Dict]) -> int:
        """计算潜在冲突数量"""
        conflicts = 0
        
        for other in all_vehicles:
            if other['id'] == vehicle['id']:
                continue
            
            if self._paths_intersect(vehicle, other):
                conflicts += 1
        
        return conflicts
    
    def _paths_intersect(self, vehicle1: Dict, vehicle2: Dict) -> bool:
        """简单的路径相交检测"""
        start1 = vehicle1['current_state']
        goal1 = vehicle1['goal_state']
        start2 = vehicle2['current_state']
        goal2 = vehicle2['goal_state']
        
        # 使用简单的线段相交检测
        return self._line_segments_intersect(
            (start1.x, start1.y), (goal1.x, goal1.y),
            (start2.x, start2.y), (goal2.x, goal2.y)
        )
    
    def _line_segments_intersect(self, p1: Tuple, p2: Tuple, p3: Tuple, p4: Tuple) -> bool:
        """线段相交检测"""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
    
    def _assess_path_complexity(self, start: VehicleState, goal: VehicleState) -> float:
        """评估路径复杂度"""
        # 基于角度变化评估复杂度
        dx = goal.x - start.x
        dy = goal.y - start.y
        goal_bearing = math.atan2(dy, dx)
        
        heading_change = abs(start.theta - goal_bearing)
        if heading_change > math.pi:
            heading_change = 2 * math.pi - heading_change
        
        # 标准化到[0,1]
        complexity = heading_change / math.pi
        return complexity
    
    def _distance_strategy(self, vehicle: Dict, context: Dict) -> Dict:
        """基于距离的策略"""
        pass
    
    def _density_strategy(self, vehicle: Dict, context: Dict) -> Dict:
        """基于密度的策略"""
        pass
    
    def _priority_strategy(self, vehicle: Dict, context: Dict) -> Dict:
        """基于优先级的策略"""
        pass
    
    def _hybrid_strategy(self, vehicle: Dict, context: Dict) -> Dict:
        """混合策略"""
        pass

class GATTrainer:
    """🎓 GAT训练器 - 提供多种训练方法"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🎓 GAT训练器初始化: 设备={self.device}")
        
        if HAS_GAT:
            self.gat_network = VehicleGATNetwork()
            self.gat_network.to(self.device)
            self.optimizer = optim.Adam(self.gat_network.parameters(), 
                                      lr=config.learning_rate)
            print("✅ GAT网络加载成功")
        else:
            print("❌ GAT模块不可用")
    
    def collect_training_data_from_scenarios(self, scenarios_data: List[Dict]) -> List[TrainingData]:
        """从规划场景中收集训练数据"""
        print("📊 从场景中收集训练数据...")
        
        training_data = []
        graph_builder = VehicleGraphBuilder()
        heuristic_coordinator = HeuristicCoordinator()
        
        for scenario in scenarios_data:
            try:
                # 构建图数据
                vehicles_info = scenario['vehicles_info']
                graph_data = graph_builder.build_graph(vehicles_info)
                
                # 生成启发式最优决策作为标签
                optimal_guidance = heuristic_coordinator.generate_coordination_guidance(vehicles_info)
                
                # 转换为训练标签格式
                optimal_decisions = {}
                for guidance in optimal_guidance:
                    vehicle_idx = guidance.vehicle_id - 1  # 假设vehicle_id从1开始
                    if vehicle_idx < len(vehicles_info):
                        optimal_decisions[f'priority_{vehicle_idx}'] = guidance.priority_adjustment
                        optimal_decisions[f'cooperation_{vehicle_idx}'] = guidance.cooperation_score
                        optimal_decisions[f'urgency_{vehicle_idx}'] = guidance.urgency_level
                        optimal_decisions[f'safety_{vehicle_idx}'] = guidance.safety_factor
                        
                        # 策略编码
                        strategy_encoding = {
                            'normal': 0, 'cooperative': 1, 'aggressive': 2,
                            'defensive': 3, 'adaptive': 4
                        }
                        optimal_decisions[f'strategy_{vehicle_idx}'] = strategy_encoding.get(guidance.strategy, 0)
                
                training_data.append(TrainingData(
                    graph_data=graph_data,
                    optimal_decisions=optimal_decisions,
                    scenario_info=scenario
                ))
                
            except Exception as e:
                print(f"⚠️ 处理场景数据失败: {e}")
                continue
        
        print(f"✅ 收集到 {len(training_data)} 条训练数据")
        return training_data
    
    def supervised_training(self, training_data: List[TrainingData]):
        """监督学习训练"""
        if not HAS_GAT or not training_data:
            print("❌ 监督训练失败: GAT不可用或无训练数据")
            return
        
        print("🎓 开始监督学习训练...")
        
        self.gat_network.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            
            # 简单的批处理
            for i in range(0, len(training_data), self.config.batch_size):
                batch = training_data[i:i + self.config.batch_size]
                
                batch_loss = 0.0
                
                for data in batch:
                    self.optimizer.zero_grad()
                    
                    # 前向传播
                    graph_data = data.graph_data
                    # 将数据移到设备
                    graph_data.node_features = graph_data.node_features.to(self.device)
                    graph_data.edge_indices = graph_data.edge_indices.to(self.device)
                    graph_data.edge_features = graph_data.edge_features.to(self.device)
                    graph_data.global_features = graph_data.global_features.to(self.device)
                    
                    predictions = self.gat_network(graph_data)
                    
                    # 计算损失
                    loss = self._compute_supervised_loss(predictions, data.optimal_decisions)
                    
                    # 反向传播
                    loss.backward()
                    batch_loss += loss.item()
                
                self.optimizer.step()
                epoch_loss += batch_loss
            
            avg_loss = epoch_loss / len(training_data)
            print(f"   Epoch {epoch+1}/{self.config.num_epochs}: Loss = {avg_loss:.4f}")
            
            # 定期保存模型
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_model(f"gat_epoch_{epoch+1}.pth")
        
        # 保存最终模型
        self.save_model(self.config.model_save_path)
        print("✅ 监督学习训练完成")
    
    def _compute_supervised_loss(self, predictions, targets) -> torch.Tensor:
        """计算监督学习损失"""
        loss = 0.0
        num_vehicles = predictions.priority_adjustments.size(0)
        
        for i in range(num_vehicles):
            # 优先级调整损失
            if f'priority_{i}' in targets:
                target_priority = torch.tensor(targets[f'priority_{i}'], device=self.device)
                loss += nn.MSELoss()(predictions.priority_adjustments[i, 0], target_priority)
            
            # 合作度损失
            if f'cooperation_{i}' in targets:
                target_coop = torch.tensor(targets[f'cooperation_{i}'], device=self.device)
                loss += nn.MSELoss()(predictions.cooperation_scores[i, 0], target_coop)
            
            # 紧急度损失
            if f'urgency_{i}' in targets:
                target_urgency = torch.tensor(targets[f'urgency_{i}'], device=self.device)
                loss += nn.MSELoss()(predictions.urgency_levels[i, 0], target_urgency)
            
            # 安全因子损失
            if f'safety_{i}' in targets:
                target_safety = torch.tensor(targets[f'safety_{i}'], device=self.device)
                loss += nn.MSELoss()(predictions.safety_factors[i, 0], target_safety)
            
            # 策略损失
            if f'strategy_{i}' in targets:
                target_strategy = torch.tensor(targets[f'strategy_{i}'], dtype=torch.long, device=self.device)
                loss += nn.CrossEntropyLoss()(predictions.strategies[i:i+1], target_strategy.unsqueeze(0))
        
        return loss
    
    def self_supervised_training(self, scenarios_data: List[Dict]):
        """自监督学习训练"""
        print("🎓 开始自监督学习训练...")
        
        # 基于对比学习的自监督训练
        # 正样本：相似情况下的协调决策应该相似
        # 负样本：不同情况下的协调决策应该不同
        
        # 这里可以实现更复杂的自监督学习逻辑
        pass
    
    def reinforcement_learning_training(self, environment):
        """强化学习训练框架"""
        print("🎓 开始强化学习训练...")
        
        # 这里可以实现基于强化学习的GAT训练
        # 奖励信号：规划成功率、冲突数量、效率等
        pass
    
    def save_model(self, path: str):
        """保存模型 - 兼容PyTorch 2.6"""
        if HAS_GAT:
            try:
                # 🔧 修复: 只保存基本参数，避免自定义类序列化问题
                save_dict = {
                    'model_state_dict': self.gat_network.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # 🆕 只保存配置的基本参数，不保存整个config对象
                    'config_params': {
                        'learning_rate': self.config.learning_rate,
                        'batch_size': self.config.batch_size,
                        'num_epochs': self.config.num_epochs,
                        'save_interval': self.config.save_interval,
                        'model_save_path': self.config.model_save_path
                    },
                    'pytorch_version': torch.__version__,
                    'save_timestamp': time.time()
                }
                
                torch.save(save_dict, path)
                print(f"💾 模型已保存: {path}")
                print(f"   PyTorch版本: {torch.__version__}")
                
            except Exception as e:
                print(f"❌ 模型保存失败: {str(e)}")
    
    def load_model(self, path: str) -> bool:
        """加载模型 - 兼容PyTorch 2.6"""
        try:
            if HAS_GAT:
                print(f"📂 尝试加载模型: {path}")
                print(f"   当前PyTorch版本: {torch.__version__}")
                
                # 🔧 修复: 使用weights_only=False来兼容旧版本保存的模型
                try:
                    # 首先尝试安全模式加载
                    checkpoint = torch.load(path, map_location=self.device, weights_only=True)
                    print("   ✅ 使用安全模式加载成功")
                except Exception as safe_load_error:
                    print(f"   ⚠️ 安全模式加载失败: {safe_load_error}")
                    print("   🔄 尝试兼容模式加载...")
                    
                    # 兼容模式加载（针对旧版本保存的模型）
                    checkpoint = torch.load(path, map_location=self.device, weights_only=False)
                    print("   ✅ 兼容模式加载成功")
                
                # 加载模型权重
                if 'model_state_dict' in checkpoint:
                    self.gat_network.load_state_dict(checkpoint['model_state_dict'])
                    print("   ✅ 模型权重加载成功")
                else:
                    print("   ❌ 模型权重不存在")
                    return False
                
                # 加载优化器状态
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                        print("   ✅ 优化器状态加载成功")
                    except Exception as opt_error:
                        print(f"   ⚠️ 优化器状态加载失败: {opt_error}")
                        print("   ℹ️ 将使用默认优化器状态")
                
                # 检查版本兼容性
                if 'pytorch_version' in checkpoint:
                    saved_version = checkpoint['pytorch_version']
                    print(f"   📋 模型保存时PyTorch版本: {saved_version}")
                    if saved_version != torch.__version__:
                        print("   ⚠️ PyTorch版本不同，可能存在兼容性问题")
                
                # 加载配置参数（如果存在）
                if 'config_params' in checkpoint:
                    config_params = checkpoint['config_params']
                    print("   ✅ 配置参数加载成功")
                elif 'config' in checkpoint:
                    print("   ℹ️ 检测到旧版本配置格式")
                
                print(f"📂 模型加载完成: {path}")
                return True
                
        except Exception as e:
            print(f"❌ 模型加载失败: {str(e)}")
            print("💡 解决建议:")
            print("   1. 检查模型文件是否完整")
            print("   2. 重新训练生成新模型") 
            print("   3. 或删除旧模型文件使用默认设置")
            
        return False

class SmartGATCoordinator:
    """🧠 智能GAT协调器 - 集成训练和启发式方案"""
    
    def __init__(self, enable_training: bool = True, model_path: str = None):
        self.heuristic_coordinator = HeuristicCoordinator()
        self.enable_training = enable_training and HAS_GAT
        self.use_gat = False
        
        if self.enable_training:
            self.config = TrainingConfig()
            self.trainer = GATTrainer(self.config)
            
            # 尝试加载预训练模型
            if model_path and self.trainer.load_model(model_path):
                self.use_gat = True
                print("✅ 使用预训练GAT模型")
            else:
                print("ℹ️ 未找到预训练模型，将使用启发式方法")
        
        print(f"🧠 智能GAT协调器初始化完成")
        print(f"   GAT状态: {'✅ 启用' if self.use_gat else '❌ 使用启发式'}")
    
    def train_on_scenarios(self, scenarios_data: List[Dict]):
        """在场景数据上训练GAT"""
        if not self.enable_training:
            print("⚠️ 训练未启用")
            return
        
        print("🎓 开始在场景数据上训练GAT...")
        
        # 收集训练数据
        training_data = self.trainer.collect_training_data_from_scenarios(scenarios_data)
        
        if training_data:
            # 执行监督学习训练
            self.trainer.supervised_training(training_data)
            self.use_gat = True
            print("✅ GAT训练完成，现在可以使用训练好的模型")
        else:
            print("❌ 无有效训练数据")
    
    def generate_coordination_guidance(self, vehicles_info: List[Dict]) -> List[CoordinationGuidance]:
        """生成协调指导"""
        if self.use_gat and HAS_GAT:
            return self._gat_coordination(vehicles_info)
        else:
            return self.heuristic_coordinator.generate_coordination_guidance(vehicles_info)
    
    def _gat_coordination(self, vehicles_info: List[Dict]) -> List[CoordinationGuidance]:
        """使用训练好的GAT进行协调"""
        try:
            print("🧠 使用训练好的GAT进行协调")
            
            # 构建图数据
            graph_builder = VehicleGraphBuilder()
            graph_data = graph_builder.build_graph(vehicles_info)
            
            # GAT推理
            self.trainer.gat_network.eval()
            with torch.no_grad():
                # 移到设备
                graph_data.node_features = graph_data.node_features.to(self.trainer.device)
                graph_data.edge_indices = graph_data.edge_indices.to(self.trainer.device)
                graph_data.edge_features = graph_data.edge_features.to(self.trainer.device)
                graph_data.global_features = graph_data.global_features.to(self.trainer.device)
                
                predictions = self.trainer.gat_network(graph_data)
            
            # 解析决策
            decision_parser = DecisionParser()
            guidance_list = decision_parser.parse_decisions(predictions, vehicles_info)
            
            print(f"✅ GAT协调完成: {len(guidance_list)}个指导策略")
            return guidance_list
            
        except Exception as e:
            print(f"❌ GAT协调失败，回退到启发式方法: {e}")
            return self.heuristic_coordinator.generate_coordination_guidance(vehicles_info)

def create_demo_training_scenarios() -> List[Dict]:
    """创建演示训练场景"""
    print("📊 创建演示训练场景...")
    
    scenarios = []
    
    # 场景1: 简单无冲突
    scenarios.append({
        'name': 'simple_no_conflict',
        'vehicles_info': [
            {
                'id': 1, 'priority': 1,
                'current_state': type('', (), {'x': 10, 'y': 10, 'theta': 0, 'v': 3, 't': 0})(),
                'goal_state': type('', (), {'x': 90, 'y': 10, 'theta': 0, 'v': 2, 't': 0})(),
            },
            {
                'id': 2, 'priority': 1,
                'current_state': type('', (), {'x': 10, 'y': 30, 'theta': 0, 'v': 3, 't': 0})(),
                'goal_state': type('', (), {'x': 90, 'y': 30, 'theta': 0, 'v': 2, 't': 0})(),
            }
        ]
    })
    
    # 场景2: 高冲突密度
    scenarios.append({
        'name': 'high_conflict_density',
        'vehicles_info': [
            {
                'id': 1, 'priority': 1,
                'current_state': type('', (), {'x': 10, 'y': 50, 'theta': 0, 'v': 3, 't': 0})(),
                'goal_state': type('', (), {'x': 90, 'y': 50, 'theta': 0, 'v': 2, 't': 0})(),
            },
            {
                'id': 2, 'priority': 1,
                'current_state': type('', (), {'x': 50, 'y': 10, 'theta': math.pi/2, 'v': 3, 't': 0})(),
                'goal_state': type('', (), {'x': 50, 'y': 90, 'theta': math.pi/2, 'v': 2, 't': 0})(),
            },
            {
                'id': 3, 'priority': 1,
                'current_state': type('', (), {'x': 90, 'y': 50, 'theta': math.pi, 'v': 3, 't': 0})(),
                'goal_state': type('', (), {'x': 10, 'y': 50, 'theta': math.pi, 'v': 2, 't': 0})(),
            }
        ]
    })
    
    print(f"✅ 创建了 {len(scenarios)} 个演示场景")
    return scenarios

def main():
    """演示GAT训练解决方案"""
    print("🧠 GAT训练解决方案演示")
    print("=" * 50)
    
    # 1. 创建演示场景数据
    scenarios_data = create_demo_training_scenarios()
    
    # 2. 创建智能协调器
    coordinator = SmartGATCoordinator(enable_training=True)
    
    # 3. 在场景上训练GAT（如果可用）
    if coordinator.enable_training:
        coordinator.train_on_scenarios(scenarios_data)
    
    # 4. 测试协调器
    print("\n🧪 测试协调器性能...")
    test_vehicles = scenarios_data[1]['vehicles_info']  # 使用高冲突场景测试
    
    guidance = coordinator.generate_coordination_guidance(test_vehicles)
    
    print("\n📊 协调结果:")
    for g in guidance:
        print(f"   V{g.vehicle_id}: {g.strategy}, 优先级{g.adjusted_priority:.1f}, "
              f"合作{g.cooperation_score:.2f}, 安全{g.safety_factor:.2f}")
    
    print("\n💡 解决方案总结:")
    print("   ✅ 启发式协调器: 立即可用，基于规则的智能决策")
    print("   ✅ GAT训练框架: 可在实际数据上训练改进")
    print("   ✅ 智能回退机制: GAT失败时自动使用启发式方法")
    print("   ✅ 模型保存/加载: 支持预训练模型复用")

if __name__ == "__main__":
    main()