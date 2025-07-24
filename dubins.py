import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class DubinsPath:
    """Dubins路径数据结构"""
    length: float
    path_type: str  # "LSL", "RSR", "LSR", "RSL"
    params: Tuple[float, float, float]  # (t, p, q) parameters
    
class MinimalDubinsPlanner:
    """
    最小Dubins曲线规划器 - 仅实现核心功能
    用于V-Hybrid A*中的目标拟合
    """
    
    def __init__(self, turning_radius: float = 3.0):
        self.turning_radius = turning_radius
        print(f"        🔄 Dubins曲线集成: 转弯半径={turning_radius}m")
    
    def plan_dubins_path(self, start_x: float, start_y: float, start_theta: float,
                        goal_x: float, goal_y: float, goal_theta: float) -> Optional[DubinsPath]:
        """规划Dubins路径"""
        
        # 标准化参数
        dx = goal_x - start_x
        dy = goal_y - start_y
        D = math.sqrt(dx*dx + dy*dy)
        d = D / self.turning_radius
        
        if d < 1e-6:
            return None
        
        # 角度标准化
        alpha = math.atan2(dy, dx)
        theta1 = self._normalize_angle(start_theta - alpha)
        theta2 = self._normalize_angle(goal_theta - alpha)
        
        # 尝试四种基本Dubins路径
        paths = []
        
        # LSL路径
        lsl_path = self._lsl_path(theta1, theta2, d)
        if lsl_path:
            paths.append(lsl_path)
        
        # RSR路径
        rsr_path = self._rsr_path(theta1, theta2, d)
        if rsr_path:
            paths.append(rsr_path)
        
        # LSR路径
        lsr_path = self._lsr_path(theta1, theta2, d)
        if lsr_path:
            paths.append(lsr_path)
        
        # RSL路径
        rsl_path = self._rsl_path(theta1, theta2, d)
        if rsl_path:
            paths.append(rsl_path)
        
        # 选择最短路径
        if not paths:
            return None
        
        best_path = min(paths, key=lambda p: p.length)
        best_path.length *= self.turning_radius  # 恢复实际长度
        
        return best_path
    
    def generate_trajectory(self, start_x: float, start_y: float, start_theta: float,
                          goal_x: float, goal_y: float, goal_theta: float,
                          start_v: float, goal_v: float, num_points: int = 20) -> List:
        """生成Dubins轨迹点"""
        
        dubins_path = self.plan_dubins_path(start_x, start_y, start_theta,
                                           goal_x, goal_y, goal_theta)
        
        if not dubins_path:
            return []
        
        # 生成轨迹点
        trajectory_points = []
        
        for i in range(num_points + 1):
            s = (i / num_points) * dubins_path.length
            x, y, theta = self._sample_dubins_path(
                start_x, start_y, start_theta, dubins_path, s)
            
            # 线性插值速度
            alpha = i / num_points
            v = start_v + alpha * (goal_v - start_v)
            
            # 估算时间（简化）
            avg_speed = (start_v + goal_v) / 2
            if avg_speed > 0.1:
                t = s / avg_speed
            else:
                t = s / 1.0
            
            trajectory_points.append({
                'x': x, 'y': y, 'theta': theta, 'v': v, 't': t
            })
        
        return trajectory_points
    
    def _normalize_angle(self, angle: float) -> float:
        """角度标准化到[-π, π]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    def _lsl_path(self, theta1: float, theta2: float, d: float) -> Optional[DubinsPath]:
        """LSL路径计算"""
        tmp0 = d + math.sin(theta1) - math.sin(theta2)
        p_squared = 2 + (d*d) - (2*math.cos(theta1 - theta2)) + (2*d*(math.sin(theta1) - math.sin(theta2)))
        
        if p_squared < 0:
            return None
        
        tmp1 = math.atan2((math.cos(theta2) - math.cos(theta1)), tmp0)
        t = self._normalize_angle(-theta1 + tmp1)
        p = math.sqrt(p_squared)
        q = self._normalize_angle(theta2 - tmp1)
        
        if t >= 0 and p >= 0 and q >= 0:
            return DubinsPath(t + p + q, "LSL", (t, p, q))
        return None
    
    def _rsr_path(self, theta1: float, theta2: float, d: float) -> Optional[DubinsPath]:
        """RSR路径计算"""
        tmp0 = d - math.sin(theta1) + math.sin(theta2)
        p_squared = 2 + (d*d) - (2*math.cos(theta1 - theta2)) + (2*d*(math.sin(theta2) - math.sin(theta1)))
        
        if p_squared < 0:
            return None
        
        tmp1 = math.atan2((math.cos(theta1) - math.cos(theta2)), tmp0)
        t = self._normalize_angle(theta1 - tmp1)
        p = math.sqrt(p_squared)
        q = self._normalize_angle(-theta2 + tmp1)
        
        if t >= 0 and p >= 0 and q >= 0:
            return DubinsPath(t + p + q, "RSR", (t, p, q))
        return None
    
    def _lsr_path(self, theta1: float, theta2: float, d: float) -> Optional[DubinsPath]:
        """LSR路径计算"""
        p_squared = -2 + (d*d) + (2*math.cos(theta1 - theta2)) + (2*d*(math.sin(theta1) + math.sin(theta2)))
        
        if p_squared < 0:
            return None
        
        p = math.sqrt(p_squared)
        tmp2 = math.atan2((-math.cos(theta1) - math.cos(theta2)), (d + math.sin(theta1) + math.sin(theta2))) - math.atan2(-2.0, p)
        t = self._normalize_angle(-theta1 + tmp2)
        q = self._normalize_angle(-self._normalize_angle(theta2) + tmp2)
        
        if t >= 0 and p >= 0 and q >= 0:
            return DubinsPath(t + p + q, "LSR", (t, p, q))
        return None
    
    def _rsl_path(self, theta1: float, theta2: float, d: float) -> Optional[DubinsPath]:
        """RSL路径计算"""
        p_squared = (d*d) - 2 + (2*math.cos(theta1 - theta2)) - (2*d*(math.sin(theta1) + math.sin(theta2)))
        
        if p_squared < 0:
            return None
        
        p = math.sqrt(p_squared)
        tmp2 = math.atan2((math.cos(theta1) + math.cos(theta2)), (d - math.sin(theta1) - math.sin(theta2))) - math.atan2(2.0, p)
        t = self._normalize_angle(theta1 - tmp2)
        q = self._normalize_angle(theta2 - tmp2)
        
        if t >= 0 and p >= 0 and q >= 0:
            return DubinsPath(t + p + q, "RSL", (t, p, q))
        return None
    
    def _sample_dubins_path(self, start_x: float, start_y: float, start_theta: float,
                           dubins_path: DubinsPath, s: float) -> Tuple[float, float, float]:
        """在Dubins路径上采样点"""
        
        t, p, q = dubins_path.params
        path_type = dubins_path.path_type
        
        # 当前位置和角度
        x, y, theta = start_x, start_y, start_theta
        
        # 处理第一段
        if s <= t * self.turning_radius:
            # 在第一段圆弧上
            angle_traveled = s / self.turning_radius
            if path_type[0] == 'L':
                # 左转
                center_x = x - self.turning_radius * math.sin(theta)
                center_y = y + self.turning_radius * math.cos(theta)
                new_theta = theta + angle_traveled
                new_x = center_x + self.turning_radius * math.sin(new_theta)
                new_y = center_y - self.turning_radius * math.cos(new_theta)
            else:
                # 右转
                center_x = x + self.turning_radius * math.sin(theta)
                center_y = y - self.turning_radius * math.cos(theta)
                new_theta = theta - angle_traveled
                new_x = center_x - self.turning_radius * math.sin(new_theta)
                new_y = center_y + self.turning_radius * math.cos(new_theta)
            
            return new_x, new_y, self._normalize_angle(new_theta)
        
        # 移动到第一段结束
        s -= t * self.turning_radius
        if path_type[0] == 'L':
            theta += t
        else:
            theta -= t
        
        theta = self._normalize_angle(theta)
        
        if path_type[0] == 'L':
            x -= self.turning_radius * math.sin(start_theta)
            y += self.turning_radius * math.cos(start_theta)
            x += self.turning_radius * math.sin(theta)
            y -= self.turning_radius * math.cos(theta)
        else:
            x += self.turning_radius * math.sin(start_theta)
            y -= self.turning_radius * math.cos(start_theta)
            x -= self.turning_radius * math.sin(theta)
            y += self.turning_radius * math.cos(theta)
        
        # 处理第二段（直线）
        if s <= p * self.turning_radius:
            # 在直线段上
            x += s * math.cos(theta)
            y += s * math.sin(theta)
            return x, y, theta
        
        # 移动到第二段结束
        s -= p * self.turning_radius
        x += p * self.turning_radius * math.cos(theta)
        y += p * self.turning_radius * math.sin(theta)
        
        # 处理第三段（最后的圆弧）
        angle_traveled = s / self.turning_radius
        if path_type[2] == 'L':
            # 左转
            center_x = x - self.turning_radius * math.sin(theta)
            center_y = y + self.turning_radius * math.cos(theta)
            new_theta = theta + angle_traveled
            new_x = center_x + self.turning_radius * math.sin(new_theta)
            new_y = center_y - self.turning_radius * math.cos(new_theta)
        else:
            # 右转
            center_x = x + self.turning_radius * math.sin(theta)
            center_y = y - self.turning_radius * math.cos(theta)
            new_theta = theta - angle_traveled
            new_x = center_x - self.turning_radius * math.sin(new_theta)
            new_y = center_y + self.turning_radius * math.cos(new_theta)
        
        return new_x, new_y, self._normalize_angle(new_theta)