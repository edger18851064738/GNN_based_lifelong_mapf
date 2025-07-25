#!/usr/bin/env python3
"""
GAT自监督学习训练系统
基于最新GNN自监督学习研究，专为多车协调设计

核心思想：
1. 对比学习 - 学习好坏协调方案的区别
2. 重建学习 - 预测协调策略和结果
3. 多视图学习 - 从不同角度理解车辆交互
4. 经验回放 - 从历史成功案例中学习

参考文献：
- TrajRCL: Self-supervised contrastive representation learning for trajectories
- HeCo: Self-supervised Heterogeneous Graph Neural Network with Co-contrastive Learning  
- ST-A-PGCL: Spatiotemporal adaptive periodical graph contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import time
from collections import deque, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class GATPlanningExperience:
    """GAT规划经验数据"""
    # 输入状态
    graph_data: dict                    # 车辆交互图数据
    vehicles_info: List[Dict]           # 车辆信息
    
    # GAT决策
    gat_decisions: dict                 # GAT输出的决策
    guidance_list: List                 # 解析后的协调指导
    
    # 规划结果
    planning_results: Dict[int, bool]   # 每辆车的规划是否成功
    success_rate: float                 # 整体成功率
    planning_time: float                # 规划耗时
    conflict_count: int                 # 冲突数量
    
    # 质量评估
    coordination_quality: float         # 协调质量评分 [0,1]
    efficiency_score: float             # 效率评分 [0,1]
    safety_score: float                 # 安全评分 [0,1]
    
    # 时间戳
    timestamp: float
    map_name: str

class TrajectoryQualityEvaluator:
    """轨迹质量评估器 - 为自监督学习提供信号"""
    
    def __init__(self, environment):
        self.environment = environment
        
    def evaluate_coordination_quality(self, planning_results: Dict, 
                                    vehicles_info: List[Dict], 
                                    guidance_list: List) -> Dict[str, float]:
        """综合评估协调质量"""
        
        successful_vehicles = [vid for vid, result in planning_results.items() 
                              if result.get('trajectory')]
        total_vehicles = len(planning_results)
        
        # 1. 成功率评分
        success_rate = len(successful_vehicles) / max(1, total_vehicles)
        
        # 2. 效率评分
        efficiency_score = self._evaluate_efficiency(planning_results, vehicles_info)
        
        # 3. 安全评分  
        safety_score = self._evaluate_safety(planning_results)
        
        # 4. 协调一致性评分
        coordination_score = self._evaluate_coordination_consistency(guidance_list)
        
        # 5. 综合质量评分
        overall_quality = (0.4 * success_rate + 
                          0.25 * efficiency_score + 
                          0.25 * safety_score + 
                          0.1 * coordination_score)
        
        return {
            'overall_quality': overall_quality,
            'success_rate': success_rate,
            'efficiency_score': efficiency_score,
            'safety_score': safety_score,
            'coordination_score': coordination_score
        }
    
    def _evaluate_efficiency(self, planning_results: Dict, vehicles_info: List[Dict]) -> float:
        """评估规划效率"""
        if not planning_results:
            return 0.0
        
        total_efficiency = 0.0
        valid_count = 0
        
        for vehicle_info in vehicles_info:
            vid = vehicle_info['id']
            if vid in planning_results and planning_results[vid].get('trajectory'):
                trajectory = planning_results[vid]['trajectory']
                planning_time = planning_results[vid].get('planning_time', float('inf'))
                
                # 计算轨迹长度
                traj_length = len(trajectory)
                final_time = trajectory[-1].t if trajectory else float('inf')
                
                # 计算直线距离
                start = vehicle_info['start']
                goal = vehicle_info['goal']
                euclidean_dist = math.sqrt((goal.x - start.x)**2 + (goal.y - start.y)**2)
                
                # 效率 = 直线距离 / (轨迹时间 * 规划时间权重)
                if final_time > 0 and planning_time < 60:  # 合理范围内
                    efficiency = euclidean_dist / (final_time + planning_time * 0.1)
                    total_efficiency += min(1.0, efficiency / 5.0)  # 归一化
                    valid_count += 1
        
        return total_efficiency / max(1, valid_count)
    
    def _evaluate_safety(self, planning_results: Dict) -> float:
        """评估安全性 - 检查轨迹间的最小距离"""
        trajectories = []
        for result in planning_results.values():
            if result.get('trajectory'):
                trajectories.append(result['trajectory'])
        
        if len(trajectories) < 2:
            return 1.0  # 单车或无车情况认为是安全的
        
        min_distance = float('inf')
        
        # 检查所有轨迹对在所有时间点的最小距离
        for i in range(len(trajectories)):
            for j in range(i + 1, len(trajectories)):
                traj1, traj2 = trajectories[i], trajectories[j]
                
                for state1 in traj1:
                    for state2 in traj2:
                        if abs(state1.t - state2.t) < 0.5:  # 时间接近
                            distance = math.sqrt((state1.x - state2.x)**2 + 
                                               (state1.y - state2.y)**2)
                            min_distance = min(min_distance, distance)
        
        # 安全评分：距离越大越安全
        safety_threshold = 4.0  # 安全距离阈值
        if min_distance == float('inf'):
            return 1.0
        
        return min(1.0, min_distance / safety_threshold)
    
    def _evaluate_coordination_consistency(self, guidance_list: List) -> float:
        """评估协调策略的一致性"""
        if not guidance_list:
            return 0.0
        
        # 统计策略分布
        strategies = [g.strategy for g in guidance_list]
        strategy_counts = {}
        for strategy in strategies:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        # 计算策略多样性 (Shannon熵)
        total = len(strategies)
        entropy = 0.0
        for count in strategy_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # 归一化熵值
        max_entropy = math.log2(min(5, total))  # 最多5种策略
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        # 优先级分布合理性
        priorities = [g.adjusted_priority for g in guidance_list]
        priority_std = np.std(priorities) if len(priorities) > 1 else 0
        priority_score = min(1.0, priority_std / 2.0)  # 优先级应该有一定差异
        
        return (normalized_entropy + priority_score) / 2.0

class VehicleGraphAugmentor:
    """车辆图增强器 - 生成对比学习的不同视图"""
    
    def __init__(self, interaction_radius: float = 50.0):
        self.interaction_radius = interaction_radius
        
    def create_augmented_views(self, vehicles_info: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """创建两个不同的增强视图用于对比学习"""
        
        # 视图1: 位置噪声增强 (模拟感知误差)
        view1 = self._position_noise_augmentation(vehicles_info)
        
        # 视图2: 速度时间增强 (模拟动态变化)  
        view2 = self._velocity_time_augmentation(vehicles_info)
        
        return view1, view2
    
    def _position_noise_augmentation(self, vehicles_info: List[Dict]) -> List[Dict]:
        """位置噪声增强 - 模拟GPS定位误差"""
        augmented = []
        
        for vehicle in vehicles_info:
            aug_vehicle = vehicle.copy()
            
            # 为当前位置添加小幅噪声 (0.5-1.0米)
            noise_x = random.uniform(-1.0, 1.0)
            noise_y = random.uniform(-1.0, 1.0)
            
            # 创建新的状态副本
            current_state = aug_vehicle['current_state']
            aug_current = type(current_state)(
                x=current_state.x + noise_x,
                y=current_state.y + noise_y,
                theta=current_state.theta,
                v=current_state.v,
                t=current_state.t
            )
            aug_vehicle['current_state'] = aug_current
            
            augmented.append(aug_vehicle)
        
        return augmented
    
    def _velocity_time_augmentation(self, vehicles_info: List[Dict]) -> List[Dict]:
        """速度和时间增强 - 模拟动态变化"""
        augmented = []
        
        for vehicle in vehicles_info:
            aug_vehicle = vehicle.copy()
            
            # 速度扰动 (±10%)
            velocity_factor = random.uniform(0.9, 1.1)
            
            # 时间偏移 (±0.2秒)
            time_offset = random.uniform(-0.2, 0.2)
            
            current_state = aug_vehicle['current_state']
            aug_current = type(current_state)(
                x=current_state.x,
                y=current_state.y,
                theta=current_state.theta,
                v=current_state.v * velocity_factor,
                t=current_state.t + time_offset
            )
            aug_vehicle['current_state'] = aug_current
            
            augmented.append(aug_vehicle)
        
        return augmented

class GATContrastiveLearner:
    """GAT对比学习器 - 核心自监督学习组件"""
    
    def __init__(self, gat_network, graph_builder, temperature: float = 0.1):
        self.gat_network = gat_network
        self.graph_builder = graph_builder
        self.temperature = temperature
        self.augmentor = VehicleGraphAugmentor()
        
        # 优化器
        self.optimizer = torch.optim.Adam(
            self.gat_network.parameters(), 
            lr=1e-4, weight_decay=1e-5
        )
        
        print("🧠 GAT对比学习器初始化完成")
        print(f"   温度参数: {temperature}")
        print(f"   学习率: 1e-4")
    
    def contrastive_learning_step(self, vehicles_info: List[Dict], 
                                 quality_scores: Dict[str, float]) -> Dict[str, float]:
        """执行一步对比学习"""
        
        # 1. 生成增强视图
        view1, view2 = self.augmentor.create_augmented_views(vehicles_info)
        
        # 2. 构建图数据
        graph1 = self.graph_builder.build_graph(view1)
        graph2 = self.graph_builder.build_graph(view2)
        
        # 3. GAT编码
        self.gat_network.train()
        
        # 前向传播
        decisions1 = self.gat_network(graph1)
        decisions2 = self.gat_network(graph2)
        
        # 4. 提取表征向量
        repr1 = self._extract_representation(decisions1, graph1)
        repr2 = self._extract_representation(decisions2, graph2)
        
        # 5. 计算对比损失
        contrastive_loss = self._compute_contrastive_loss(repr1, repr2)
        
        # 6. 质量引导损失 (基于历史经验)
        quality_loss = self._compute_quality_guided_loss(
            decisions1, quality_scores)
        
        # 7. 总损失
        total_loss = contrastive_loss + 0.5 * quality_loss
        
        # 8. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.gat_network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'contrastive_loss': contrastive_loss.item(),
            'quality_loss': quality_loss.item()
        }
    
    def _extract_representation(self, decisions, graph_data) -> torch.Tensor:
        """从GAT决策中提取表征向量"""
        
        # 将多任务输出合并为统一表征
        priority_repr = decisions.priority_adjustments.flatten()
        cooperation_repr = decisions.cooperation_scores.flatten()  
        urgency_repr = decisions.urgency_levels.flatten()
        safety_repr = decisions.safety_factors.flatten()
        strategy_repr = decisions.strategies.flatten()
        global_repr = decisions.global_signal
        
        # 拼接所有表征
        combined_repr = torch.cat([
            priority_repr, cooperation_repr, urgency_repr, 
            safety_repr, strategy_repr, global_repr
        ], dim=0)
        
        return combined_repr
    
    def _compute_contrastive_loss(self, repr1: torch.Tensor, 
                                 repr2: torch.Tensor) -> torch.Tensor:
        """计算对比损失 - 基于InfoNCE"""
        
        # L2标准化
        repr1_norm = F.normalize(repr1, dim=0)
        repr2_norm = F.normalize(repr2, dim=0)
        
        # 计算相似度
        similarity = torch.dot(repr1_norm, repr2_norm) / self.temperature
        
        # 由于我们只有正样本对，使用简化的对比损失
        # 目标：最大化两个视图间的相似度
        contrastive_loss = -similarity + torch.log(torch.exp(similarity) + 1)
        
        return contrastive_loss
    
    def _compute_quality_guided_loss(self, decisions, 
                                   quality_scores: Dict[str, float]) -> torch.Tensor:
        """基于质量评分的引导损失"""
        
        if not quality_scores:
            return torch.tensor(0.0)
        
        overall_quality = quality_scores.get('overall_quality', 0.5)
        
        # 如果质量高，鼓励当前决策；如果质量低，惩罚当前决策
        quality_target = torch.tensor(overall_quality)
        
        # 计算决策的"激进程度" - 作为质量的代理指标
        priority_variance = torch.var(decisions.priority_adjustments)
        cooperation_mean = torch.mean(decisions.cooperation_scores)
        
        decision_intensity = priority_variance + (1.0 - cooperation_mean)
        
        # 质量引导损失：高质量时鼓励适中的决策强度
        if overall_quality > 0.7:
            # 高质量：鼓励适中决策
            target_intensity = 0.3
        elif overall_quality > 0.4:
            # 中等质量：保持当前水平
            target_intensity = decision_intensity.detach()
        else:
            # 低质量：鼓励更保守的决策
            target_intensity = 0.1
        
        quality_loss = F.mse_loss(decision_intensity, 
                                torch.tensor(target_intensity))
        
        return quality_loss

class ExperienceBuffer:
    """经验缓冲区 - 存储和管理GAT规划经验"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.experiences = deque(maxlen=max_size)
        self.quality_stats = {
            'mean_quality': 0.0,
            'std_quality': 0.0,
            'best_quality': 0.0,
            'worst_quality': 1.0
        }
        
        print(f"📚 经验缓冲区初始化: 最大容量 {max_size}")
    
    def add_experience(self, experience: GATPlanningExperience):
        """添加新的规划经验"""
        self.experiences.append(experience)
        self._update_quality_stats()
        
        print(f"   📝 新增经验: 质量 {experience.coordination_quality:.3f}, "
              f"成功率 {experience.success_rate:.1%}, "
              f"缓冲区大小 {len(self.experiences)}")
    
    def get_high_quality_experiences(self, top_k: int = 50) -> List[GATPlanningExperience]:
        """获取高质量经验"""
        if not self.experiences:
            return []
        
        sorted_experiences = sorted(self.experiences, 
                                  key=lambda x: x.coordination_quality, 
                                  reverse=True)
        return sorted_experiences[:top_k]
    
    def get_diverse_experiences(self, num_samples: int = 20) -> List[GATPlanningExperience]:
        """获取多样化经验样本"""
        if not self.experiences:
            return []
        
        # 按质量分层采样
        high_quality = [exp for exp in self.experiences if exp.coordination_quality > 0.7]
        medium_quality = [exp for exp in self.experiences if 0.4 <= exp.coordination_quality <= 0.7]
        low_quality = [exp for exp in self.experiences if exp.coordination_quality < 0.4]
        
        samples = []
        
        # 分层采样
        if high_quality:
            samples.extend(random.sample(high_quality, min(num_samples//3, len(high_quality))))
        if medium_quality:
            samples.extend(random.sample(medium_quality, min(num_samples//3, len(medium_quality))))
        if low_quality:
            samples.extend(random.sample(low_quality, min(num_samples//3, len(low_quality))))
        
        # 补充随机样本
        remaining = num_samples - len(samples)
        if remaining > 0:
            all_samples = list(self.experiences)
            additional = random.sample(all_samples, min(remaining, len(all_samples)))
            samples.extend(additional)
        
        return samples[:num_samples]
    
    def _update_quality_stats(self):
        """更新质量统计信息"""
        if not self.experiences:
            return
        
        qualities = [exp.coordination_quality for exp in self.experiences]
        self.quality_stats = {
            'mean_quality': np.mean(qualities),
            'std_quality': np.std(qualities),
            'best_quality': np.max(qualities),
            'worst_quality': np.min(qualities),
            'count': len(qualities)
        }

class GATSelfSupervisedTrainer:
    """🎯 GAT自监督学习训练器 - 主要接口"""
    
    def __init__(self, gat_network, graph_builder, environment):
        self.gat_network = gat_network
        self.graph_builder = graph_builder
        self.environment = environment
        
        # 核心组件
        self.quality_evaluator = TrajectoryQualityEvaluator(environment)
        self.contrastive_learner = GATContrastiveLearner(gat_network, graph_builder)
        self.experience_buffer = ExperienceBuffer(max_size=1000)
        
        # 训练统计
        self.training_stats = {
            'total_episodes': 0,
            'training_steps': 0,
            'avg_loss': 0.0,
            'learning_rate': 1e-4
        }
        
        print("🚀 GAT自监督训练器初始化完成")
        print("   🎯 核心功能:")
        print("     - 对比学习 (多视图增强)")
        print("     - 质量引导学习 (成功案例学习)")
        print("     - 经验回放 (历史经验利用)")
        print("     - 自适应优化 (动态调整)")
    
    def record_planning_experience(self, vehicles_info: List[Dict], 
                                 gat_decisions, guidance_list: List,
                                 planning_results: Dict, 
                                 planning_time: float,
                                 map_name: str):
        """记录一次完整的规划经验"""
        
        # 评估协调质量
        quality_metrics = self.quality_evaluator.evaluate_coordination_quality(
            planning_results, vehicles_info, guidance_list)
        
        # 创建经验记录
        experience = GATPlanningExperience(
            graph_data=self.graph_builder.build_graph(vehicles_info).__dict__,
            vehicles_info=vehicles_info,
            gat_decisions=gat_decisions.__dict__ if hasattr(gat_decisions, '__dict__') else gat_decisions,
            guidance_list=guidance_list,
            planning_results={vid: bool(result.get('trajectory')) for vid, result in planning_results.items()},
            success_rate=quality_metrics['success_rate'],
            planning_time=planning_time,
            conflict_count=0,  # TODO: 实现冲突计数
            coordination_quality=quality_metrics['overall_quality'],
            efficiency_score=quality_metrics['efficiency_score'],
            safety_score=quality_metrics['safety_score'],
            timestamp=time.time(),
            map_name=map_name
        )
        
        # 存储经验
        self.experience_buffer.add_experience(experience)
        
        # 检查是否需要训练
        if len(self.experience_buffer.experiences) >= 10:  # 累积足够经验后开始训练
            self._perform_training_step()
    
    def _perform_training_step(self):
        """执行一步训练"""
        
        # 获取多样化训练样本
        training_samples = self.experience_buffer.get_diverse_experiences(num_samples=5)
        
        if not training_samples:
            return
        
        total_loss = 0.0
        
        for sample in training_samples:
            # 使用经验中的车辆信息进行对比学习
            quality_scores = {
                'overall_quality': sample.coordination_quality,
                'efficiency_score': sample.efficiency_score,
                'safety_score': sample.safety_score
            }
            
            loss_info = self.contrastive_learner.contrastive_learning_step(
                sample.vehicles_info, quality_scores)
            
            total_loss += loss_info['total_loss']
        
        # 更新训练统计
        self.training_stats['training_steps'] += 1
        self.training_stats['avg_loss'] = (self.training_stats['avg_loss'] * 0.9 + 
                                          (total_loss / len(training_samples)) * 0.1)
        
        if self.training_stats['training_steps'] % 10 == 0:
            self._print_training_progress()
    
    def _print_training_progress(self):
        """打印训练进度"""
        stats = self.training_stats
        buffer_stats = self.experience_buffer.quality_stats
        
        print(f"\n🧠 GAT自监督学习进度:")
        print(f"   训练步数: {stats['training_steps']}")
        print(f"   平均损失: {stats['avg_loss']:.4f}")
        print(f"   经验数量: {buffer_stats.get('count', 0)}")
        print(f"   平均质量: {buffer_stats.get('mean_quality', 0):.3f}")
        print(f"   最佳质量: {buffer_stats.get('best_quality', 0):.3f}")
    
    def save_model(self, filepath: str):
        """保存训练好的模型"""
        checkpoint = {
            'gat_network_state': self.gat_network.state_dict(),
            'optimizer_state': self.contrastive_learner.optimizer.state_dict(),
            'training_stats': self.training_stats,
            'buffer_stats': self.experience_buffer.quality_stats
        }
        
        torch.save(checkpoint, filepath)
        print(f"💾 GAT模型已保存: {filepath}")
    
    def load_model(self, filepath: str):
        """加载预训练模型"""
        try:
            checkpoint = torch.load(filepath)
            self.gat_network.load_state_dict(checkpoint['gat_network_state'])
            self.contrastive_learner.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.training_stats = checkpoint.get('training_stats', self.training_stats)
            
            print(f"✅ GAT模型已加载: {filepath}")
            print(f"   训练步数: {self.training_stats.get('training_steps', 0)}")
            print(f"   平均损失: {self.training_stats.get('avg_loss', 0):.4f}")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
    
    def get_learning_summary(self) -> Dict:
        """获取学习总结"""
        return {
            'training_stats': self.training_stats,
            'buffer_stats': self.experience_buffer.quality_stats,
            'model_info': {
                'total_parameters': sum(p.numel() for p in self.gat_network.parameters()),
                'trainable_parameters': sum(p.numel() for p in self.gat_network.parameters() if p.requires_grad)
            }
        }

def create_gat_self_supervised_system(gat_network, graph_builder, environment):
    """创建完整的GAT自监督学习系统"""
    
    trainer = GATSelfSupervisedTrainer(gat_network, graph_builder, environment)
    
    print("🎉 GAT自监督学习系统创建完成！")
    print("\n🎯 使用方法:")
    print("1. 在每次规划后调用 trainer.record_planning_experience()")
    print("2. 系统会自动进行对比学习和质量引导学习")
    print("3. 定期调用 trainer.save_model() 保存学习进度")
    print("4. 使用 trainer.load_model() 加载预训练模型")
    
    print("\n📚 学习策略:")
    print("   🔄 对比学习: 从不同视图学习鲁棒表征")
    print("   🎯 质量引导: 从成功案例学习优秀策略")
    print("   📈 经验回放: 利用历史经验持续改进")
    print("   🚀 自适应: 根据规划质量动态调整")
    
    return trainer

# ========================================
# 集成到现有系统中的示例
# ========================================

def integrate_self_supervised_learning_example():
    """集成自监督学习到现有GAT系统的示例"""
    
    print("\n🔧 集成示例:")
    print("""
    # 在 EnhancedFirstRoundPlanner 中集成自监督学习
    
    class EnhancedFirstRoundPlanner:
        def __init__(self, ...):
            # ... 现有初始化代码 ...
            
            # 🆕 添加自监督学习组件
            if self.enable_gat:
                self.gat_trainer = create_gat_self_supervised_system(
                    self.gat_network, 
                    self.graph_builder, 
                    self.environment
                )
                
                # 尝试加载预训练模型
                model_path = f"gat_model_{self.map_data.get('map_info', {}).get('name', 'default')}.pt"
                if os.path.exists(model_path):
                    self.gat_trainer.load_model(model_path)
        
        def plan_all_vehicles(self):
            # ... 现有规划代码 ...
            
            # 🆕 记录规划经验用于学习
            if self.enable_gat and hasattr(self, 'gat_trainer'):
                vehicles_info = self._convert_vehicles_to_gat_format()
                
                # 获取GAT决策和指导
                graph_data = self.graph_builder.build_graph(vehicles_info)
                gat_decisions = self.gat_network(graph_data)
                guidance_list = self.decision_parser.parse_decisions(gat_decisions, vehicles_info)
                
                # 收集规划结果
                planning_results = {}
                for vehicle in self.vehicles:
                    planning_results[vehicle.vehicle_id] = {
                        'trajectory': vehicle.trajectory,
                        'planning_time': vehicle.planning_time
                    }
                
                # 🎯 记录经验，触发自监督学习
                self.gat_trainer.record_planning_experience(
                    vehicles_info=vehicles_info,
                    gat_decisions=gat_decisions,
                    guidance_list=guidance_list,
                    planning_results=planning_results,
                    planning_time=total_planning_time,
                    map_name=self.environment.map_name
                )
                
                # 定期保存模型
                if len(self.gat_trainer.experience_buffer.experiences) % 50 == 0:
                    self.gat_trainer.save_model(model_path)
                    print(f"🎓 GAT模型学习进度已保存")
    """)

if __name__ == "__main__":
    print("🧠 GAT自监督学习系统")
    print("=" * 60)
    print("基于最新GNN自监督学习研究，专为多车协调设计")
    print("\n核心特性:")
    print("✅ 对比学习 - 从多视图中学习鲁棒协调策略")
    print("✅ 质量引导 - 从成功案例中学习优秀决策")
    print("✅ 经验回放 - 利用历史数据持续改进")
    print("✅ 自适应学习 - 根据规划质量动态调整")
    print("✅ 零人工标注 - 完全自监督学习")
    
    integrate_self_supervised_learning_example()