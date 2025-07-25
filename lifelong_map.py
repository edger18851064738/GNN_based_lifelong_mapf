#!/usr/bin/env python3
"""
Lifelong路口地图编辑器 - 简洁版
专注于：障碍物绘制 + 出入口边放置 + 地图保存加载
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import json
import os
from typing import List, Tuple
from dataclasses import dataclass
from enum import Enum

class EdgeDirection(Enum):
    """出入口边方向"""
    NORTH = "north"    # 北边界
    SOUTH = "south"    # 南边界
    EAST = "east"      # 东边界
    WEST = "west"      # 西边界

@dataclass
class IntersectionEdge:
    """出入口边"""
    edge_id: str
    direction: EdgeDirection
    center_x: int      # 边界中心x坐标
    center_y: int      # 边界中心y坐标
    length: int = 5    # 边界长度，固定为5

    def get_points(self) -> List[Tuple[int, int]]:
        """获取边界覆盖的所有点位"""
        points = []
        half_length = self.length // 2
        
        if self.direction in [EdgeDirection.NORTH, EdgeDirection.SOUTH]:
            # 水平边界
            for x in range(self.center_x - half_length, self.center_x + half_length + 1):
                points.append((x, self.center_y))
        else:
            # 垂直边界
            for y in range(self.center_y - half_length, self.center_y + half_length + 1):
                points.append((self.center_x, y))
        
        return points

class LifelongMapEditor(tk.Tk):
    """Lifelong地图编辑器"""
    
    def __init__(self):
        super().__init__()
        self.title("Lifelong路口地图编辑器")
        self.geometry("1200x800")
        
        # 地图数据
        self.rows = 50
        self.cols = 50
        self.cell_size = 10
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)  # 0=可通行, 1=障碍物
        
        # 出入口边
        self.edges: List[IntersectionEdge] = []
        self.edge_id_counter = 1
        
        # 工具状态
        self.current_tool = "obstacle"  # obstacle, passable, edge
        self.current_edge_direction = EdgeDirection.NORTH
        self.brush_size = 2
        self.is_drawing = False
        
        self.create_ui()
        self.init_map()
    
    def create_ui(self):
        """创建界面"""
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 左侧控制面板
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        # 右侧画布
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.create_controls(control_frame)
        self.create_canvas(canvas_frame)
    
    def create_controls(self, parent):
        """创建控制面板"""
        
        # 1. 地图尺寸
        size_frame = tk.LabelFrame(parent, text="📐 地图尺寸", padx=5, pady=5)
        size_frame.pack(fill=tk.X, padx=2, pady=3)
        
        size_grid = tk.Frame(size_frame)
        size_grid.pack(fill=tk.X)
        
        tk.Label(size_grid, text="宽度:").grid(row=0, column=0, sticky=tk.W)
        self.col_entry = tk.Entry(size_grid, width=8)
        self.col_entry.grid(row=0, column=1, padx=5, pady=2)
        self.col_entry.insert(0, str(self.cols))
        
        tk.Label(size_grid, text="高度:").grid(row=1, column=0, sticky=tk.W)
        self.row_entry = tk.Entry(size_grid, width=8)
        self.row_entry.grid(row=1, column=1, padx=5, pady=2)
        self.row_entry.insert(0, str(self.rows))
        
        tk.Button(size_frame, text="更新尺寸", command=self.update_map_size,
                 bg="#4CAF50", fg="white").pack(pady=3)
        
        # 2. 绘图工具
        tool_frame = tk.LabelFrame(parent, text="🔧 绘图工具", padx=5, pady=5)
        tool_frame.pack(fill=tk.X, padx=2, pady=3)
        
        self.tool_var = tk.StringVar(value="obstacle")
        tools = [
            ("🚫 障碍物", "obstacle"),
            ("✅ 可通行", "passable"),
            ("🚪 出入口边", "edge")
        ]
        
        for text, value in tools:
            tk.Radiobutton(tool_frame, text=text, value=value, variable=self.tool_var,
                          command=self.update_current_tool).pack(anchor=tk.W)
        
        # 画笔大小
        brush_frame = tk.Frame(tool_frame)
        brush_frame.pack(fill=tk.X, pady=3)
        
        tk.Label(brush_frame, text="画笔大小:").pack(side=tk.LEFT)
        self.brush_scale = tk.Scale(brush_frame, from_=1, to=5, orient=tk.HORIZONTAL,
                                   command=self.update_brush_size)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)
        
        # 3. 出入口边设置
        edge_frame = tk.LabelFrame(parent, text="🚪 出入口边", padx=5, pady=5)
        edge_frame.pack(fill=tk.X, padx=2, pady=3)
        
        tk.Label(edge_frame, text="边界方向:", font=("Arial", 9, "bold")).pack(anchor=tk.W)
        
        self.edge_direction_var = tk.StringVar(value="north")
        directions = [
            ("北边界 ↑", "north"),
            ("南边界 ↓", "south"), 
            ("东边界 →", "east"),
            ("西边界 ←", "west")
        ]
        
        for text, value in directions:
            tk.Radiobutton(edge_frame, text=text, value=value,
                          variable=self.edge_direction_var,
                          command=self.update_edge_direction).pack(anchor=tk.W)
        
        # 边界操作按钮
        edge_btn_frame = tk.Frame(edge_frame)
        edge_btn_frame.pack(fill=tk.X, pady=3)
        
        tk.Button(edge_btn_frame, text="🗑️ 清除所有边界", 
                 command=self.clear_all_edges,
                 bg="#f44336", fg="white").pack(fill=tk.X)
        
        # 已放置的边界列表
        tk.Label(edge_frame, text="已放置的边界:", font=("Arial", 8, "bold")).pack(anchor=tk.W, pady=(10,0))
        
        list_frame = tk.Frame(edge_frame)
        list_frame.pack(fill=tk.X, pady=2)
        
        self.edge_listbox = tk.Listbox(list_frame, height=4, font=("Arial", 8))
        edge_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.edge_listbox.yview)
        self.edge_listbox.configure(yscrollcommand=edge_scrollbar.set)
        
        self.edge_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        edge_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 双击删除边界
        self.edge_listbox.bind("<Double-Button-1>", self.delete_selected_edge)
        
        # 4. 快速模板
        template_frame = tk.LabelFrame(parent, text="🚦 快速模板", padx=5, pady=5)
        template_frame.pack(fill=tk.X, padx=2, pady=3)
        
        tk.Button(template_frame, text="十字路口", 
                 command=self.create_cross_template,
                 bg="#2196F3", fg="white").pack(fill=tk.X, pady=1)
        
        tk.Button(template_frame, text="T型路口", 
                 command=self.create_t_template,
                 bg="#FF9800", fg="white").pack(fill=tk.X, pady=1)
        
        # 5. 地图信息
        info_frame = tk.LabelFrame(parent, text="📊 地图信息", padx=5, pady=5)
        info_frame.pack(fill=tk.X, padx=2, pady=3)
        
        self.info_text = tk.Text(info_frame, height=3, state=tk.DISABLED, 
                               font=("Arial", 8), bg="#f0f0f0")
        self.info_text.pack(fill=tk.X, pady=2)
        
        # 6. 文件操作
        file_frame = tk.LabelFrame(parent, text="💾 文件操作", padx=5, pady=5)
        file_frame.pack(fill=tk.X, padx=2, pady=3)
        
        tk.Label(file_frame, text="地图名称:").pack(anchor=tk.W)
        self.name_entry = tk.Entry(file_frame)
        self.name_entry.pack(fill=tk.X, pady=2)
        self.name_entry.insert(0, "lifelong_map")
        
        file_btn_frame = tk.Frame(file_frame)
        file_btn_frame.pack(fill=tk.X, pady=2)
        
        tk.Button(file_btn_frame, text="💾 保存", command=self.save_map, 
                 bg="#4CAF50", fg="white", width=12).pack(side=tk.LEFT, padx=2)
        
        tk.Button(file_btn_frame, text="📂 加载", command=self.load_map, 
                 width=12).pack(side=tk.RIGHT, padx=2)
        
        tk.Button(file_frame, text="🗑️ 清空地图", command=self.clear_map,
                 bg="#f44336", fg="white").pack(fill=tk.X, pady=2)
        
        # 状态栏
        self.status_label = tk.Label(parent, text="就绪", 
                                   bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                   font=("Arial", 8))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))
    
    def create_canvas(self, parent):
        """创建画布"""
        self.canvas = tk.Canvas(parent, bg="white")
        h_scrollbar = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.canvas.xview)
        v_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.canvas.yview)
        
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas.configure(xscrollcommand=h_scrollbar.set, yscrollcommand=v_scrollbar.set)
        
        # 绑定事件
        self.canvas.bind("<ButtonPress-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<Button-3>", self.on_right_click)
    
    def update_current_tool(self):
        """更新当前工具"""
        self.current_tool = self.tool_var.get()
        if self.current_tool == "edge":
            self.set_status("点击位置放置出入口边（长度5）")
        else:
            self.set_status(f"当前工具: {self.current_tool}")
    
    def update_edge_direction(self):
        """更新边界方向"""
        self.current_edge_direction = EdgeDirection(self.edge_direction_var.get())
    
    def update_brush_size(self, value):
        """更新画笔大小"""
        self.brush_size = int(value)
    
    def update_map_size(self):
        """更新地图尺寸"""
        try:
            new_cols = int(self.col_entry.get())
            new_rows = int(self.row_entry.get())
            
            if new_rows <= 0 or new_cols <= 0:
                messagebox.showerror("错误", "宽度和高度必须大于0")
                return
            
            # 创建新网格
            new_grid = np.zeros((new_rows, new_cols), dtype=np.int8)
            
            # 复制现有数据
            min_rows = min(self.rows, new_rows)
            min_cols = min(self.cols, new_cols)
            new_grid[:min_rows, :min_cols] = self.grid[:min_rows, :min_cols]
            
            # 过滤超出范围的边界
            self.edges = [edge for edge in self.edges 
                         if 0 <= edge.center_x < new_cols and 0 <= edge.center_y < new_rows]
            
            self.grid = new_grid
            self.rows = new_rows
            self.cols = new_cols
            
            self.update_edge_list()
            self.draw_grid()
            self.update_map_info()
            self.set_status(f"地图尺寸已更新为 {new_cols}x{new_rows}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")
    
    def on_canvas_click(self, event):
        """处理画布点击"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= col < self.cols and 0 <= row < self.rows:
            if self.current_tool == "edge":
                self.place_edge(col, row)
            else:
                # 清理之前的拖拽状态
                if hasattr(self, 'last_drag_pos'):
                    delattr(self, 'last_drag_pos')
                self.is_drawing = True
                self.apply_tool(row, col)
    
    def on_canvas_release(self, event):
        """处理鼠标释放"""
        self.is_drawing = False
        # 清理拖拽位置记录
        if hasattr(self, 'last_drag_pos'):
            delattr(self, 'last_drag_pos')
        # 确保最终状态正确显示并更新信息
        self.draw_grid()
        self.update_map_info()
    
    def on_canvas_drag(self, event):
        """处理拖拽绘制"""
        if self.current_tool in ["obstacle", "passable"] and self.is_drawing:
            canvas_x = self.canvas.canvasx(event.x)
            canvas_y = self.canvas.canvasy(event.y)
            col = int(canvas_x // self.cell_size)
            row = int(canvas_y // self.cell_size)
            
            if 0 <= col < self.cols and 0 <= row < self.rows:
                # 避免重复绘制同一个位置
                if not hasattr(self, 'last_drag_pos') or self.last_drag_pos != (row, col):
                    self.last_drag_pos = (row, col)
                    self.apply_tool(row, col)
    
    def on_right_click(self, event):
        """右键删除边界"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        # 查找点击位置的边界
        for edge in self.edges[:]:  # 使用切片避免修改列表时的问题
            edge_points = edge.get_points()
            if (col, row) in edge_points:
                self.edges.remove(edge)
                self.update_edge_list()
                self.draw_grid()
                self.set_status(f"已删除边界: {edge.edge_id}")
                break
    
    def apply_tool(self, row, col):
        """应用工具"""
        if self.current_tool == "obstacle":
            self.paint_obstacle(row, col)
        elif self.current_tool == "passable":
            self.clear_obstacle(row, col)
    
    def paint_obstacle(self, center_row, center_col):
        """绘制障碍物"""
        radius = self.brush_size
        changed = False
        
        for r in range(max(0, center_row - radius), min(self.rows, center_row + radius + 1)):
            for c in range(max(0, center_col - radius), min(self.cols, center_col + radius + 1)):
                if ((r - center_row) ** 2 + (c - center_col) ** 2) <= radius ** 2:
                    if self.grid[r, c] != 1:  # 只有当状态改变时才标记
                        self.grid[r, c] = 1
                        changed = True
        
        # 实时更新显示
        if changed:
            self.draw_grid()
    
    def clear_obstacle(self, center_row, center_col):
        """清除障碍物"""
        radius = self.brush_size
        changed = False
        
        for r in range(max(0, center_row - radius), min(self.rows, center_row + radius + 1)):
            for c in range(max(0, center_col - radius), min(self.cols, center_col + radius + 1)):
                if ((r - center_row) ** 2 + (c - center_col) ** 2) <= radius ** 2:
                    if self.grid[r, c] != 0:  # 只有当状态改变时才标记
                        self.grid[r, c] = 0
                        changed = True
        
        # 实时更新显示
        if changed:
            self.draw_grid()
    
    def place_edge(self, col, row):
        """放置出入口边"""
        # 检查是否已有边界在此位置
        for existing_edge in self.edges:
            if existing_edge.center_x == col and existing_edge.center_y == row:
                messagebox.showwarning("警告", "此位置已有边界")
                return
        
        # 创建新边界
        edge_id = f"{self.current_edge_direction.value}_{self.edge_id_counter}"
        new_edge = IntersectionEdge(
            edge_id=edge_id,
            direction=self.current_edge_direction,
            center_x=col,
            center_y=row,
            length=5
        )
        
        # 确保边界覆盖的区域都是可通行的
        edge_points = new_edge.get_points()
        for x, y in edge_points:
            if 0 <= x < self.cols and 0 <= y < self.rows:
                self.grid[y, x] = 0  # 设为可通行
        
        self.edges.append(new_edge)
        self.edge_id_counter += 1
        
        self.update_edge_list()
        self.draw_grid()
        self.set_status(f"已放置边界: {edge_id} 在位置 ({col}, {row})")
    
    def clear_all_edges(self):
        """清除所有边界"""
        if self.edges:
            reply = messagebox.askyesno("确认", "确定要清除所有出入口边界吗？")
            if reply:
                self.edges.clear()
                self.update_edge_list()
                self.draw_grid()
                self.set_status("已清除所有边界")
    
    def delete_selected_edge(self, event):
        """删除选中的边界"""
        selection = self.edge_listbox.curselection()
        if selection:
            edge_index = selection[0]
            if edge_index < len(self.edges):
                edge = self.edges[edge_index]
                self.edges.remove(edge)
                self.update_edge_list()
                self.draw_grid()
                self.set_status(f"已删除边界: {edge.edge_id}")
    
    def update_edge_list(self):
        """更新边界列表"""
        self.edge_listbox.delete(0, tk.END)
        for edge in self.edges:
            text = f"{edge.edge_id} ({edge.center_x}, {edge.center_y})"
            self.edge_listbox.insert(tk.END, text)
    
    def create_cross_template(self):
        """创建十字路口模板"""
        self.clear_map()
        
        center_x, center_y = self.cols // 2, self.rows // 2
        road_width = 8
        
        # 创建十字道路
        # 水平道路
        for x in range(self.cols):
            for y in range(center_y - road_width//2, center_y + road_width//2 + 1):
                if 0 <= y < self.rows:
                    self.grid[y, x] = 0
        
        # 垂直道路
        for y in range(self.rows):
            for x in range(center_x - road_width//2, center_x + road_width//2 + 1):
                if 0 <= x < self.cols:
                    self.grid[y, x] = 0
        
        # 添加建筑物
        buildings = [
            (5, 5, 10, 10), (35, 5, 10, 10),
            (5, 35, 10, 10), (35, 35, 10, 10)
        ]
        
        for bx, by, bw, bh in buildings:
            for y in range(by, min(by + bh, self.rows)):
                for x in range(bx, min(bx + bw, self.cols)):
                    self.grid[y, x] = 1
        
        # 添加四个方向的出入口边
        margin = 5
        self.edges = [
            IntersectionEdge("north_1", EdgeDirection.NORTH, center_x, margin, 5),
            IntersectionEdge("south_1", EdgeDirection.SOUTH, center_x, self.rows - margin, 5),
            IntersectionEdge("east_1", EdgeDirection.EAST, self.cols - margin, center_y, 5),
            IntersectionEdge("west_1", EdgeDirection.WEST, margin, center_y, 5)
        ]
        self.edge_id_counter = 5
        
        self.update_edge_list()
        self.draw_grid()
        self.update_map_info()
        self.set_status("已创建十字路口模板")
    
    def create_t_template(self):
        """创建T型路口模板"""
        self.clear_map()
        
        center_x, center_y = self.cols // 2, self.rows // 2
        road_width = 8
        
        # 水平主干道
        for x in range(self.cols):
            for y in range(center_y - road_width//2, center_y + road_width//2 + 1):
                if 0 <= y < self.rows:
                    self.grid[y, x] = 0
        
        # 垂直支路（只向北）
        for y in range(0, center_y + road_width//2 + 1):
            for x in range(center_x - road_width//2, center_x + road_width//2 + 1):
                self.grid[y, x] = 0
        
        # 添加建筑物
        buildings = [
            (5, 5, 15, 15), (30, 5, 15, 15),
            (5, 30, 40, 15)
        ]
        
        for bx, by, bw, bh in buildings:
            for y in range(by, min(by + bh, self.rows)):
                for x in range(bx, min(bx + bw, self.cols)):
                    self.grid[y, x] = 1
        
        # 添加T型路口的三个出入口边
        margin = 5
        self.edges = [
            IntersectionEdge("north_1", EdgeDirection.NORTH, center_x, margin, 5),
            IntersectionEdge("east_1", EdgeDirection.EAST, self.cols - margin, center_y, 5),
            IntersectionEdge("west_1", EdgeDirection.WEST, margin, center_y, 5)
        ]
        self.edge_id_counter = 4
        
        self.update_edge_list()
        self.draw_grid()
        self.update_map_info()
        self.set_status("已创建T型路口模板")
    
    def init_map(self):
        """初始化地图"""
        self.draw_grid()
        self.update_map_info()
        self.set_status("地图已初始化")
    
    def draw_grid(self):
        """绘制地图"""
        self.canvas.delete("all")
        
        canvas_width = self.cols * self.cell_size
        canvas_height = self.rows * self.cell_size
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))
        
        # 绘制网格线
        for i in range(0, canvas_width + 1, self.cell_size):
            self.canvas.create_line(i, 0, i, canvas_height, fill="lightgray", width=1)
        for i in range(0, canvas_height + 1, self.cell_size):
            self.canvas.create_line(0, i, canvas_width, i, fill="lightgray", width=1)
        
        # 绘制障碍物
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == 1:
                    x1 = col * self.cell_size
                    y1 = row * self.cell_size
                    x2 = x1 + self.cell_size
                    y2 = y1 + self.cell_size
                    self.canvas.create_rectangle(x1, y1, x2, y2, 
                                               fill="black", outline="gray")
        
        # 绘制出入口边
        for edge in self.edges:
            self.draw_edge(edge)
    
    def draw_edge(self, edge):
        """绘制出入口边"""
        edge_points = edge.get_points()
        
        # 颜色映射
        color_map = {
            EdgeDirection.NORTH: "red",
            EdgeDirection.SOUTH: "blue",
            EdgeDirection.EAST: "green", 
            EdgeDirection.WEST: "orange"
        }
        color = color_map[edge.direction]
        
        # 绘制边界覆盖的格子
        for x, y in edge_points:
            if 0 <= x < self.cols and 0 <= y < self.rows:
                x1 = x * self.cell_size
                y1 = y * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                self.canvas.create_rectangle(x1, y1, x2, y2, 
                                           fill=color, outline="white", width=2)
        
        # 添加标签
        center_x = edge.center_x * self.cell_size + self.cell_size // 2
        center_y = edge.center_y * self.cell_size + self.cell_size // 2
        
        # 方向箭头
        arrow_map = {
            EdgeDirection.NORTH: "↑",
            EdgeDirection.SOUTH: "↓",
            EdgeDirection.EAST: "→",
            EdgeDirection.WEST: "←"
        }
        arrow = arrow_map[edge.direction]
        
        self.canvas.create_text(center_x, center_y - 8, text=arrow, 
                               fill="white", font=("Arial", 8, "bold"))
        self.canvas.create_text(center_x, center_y + 8, text=edge.edge_id, 
                               fill="white", font=("Arial", 6, "bold"))
    
    def update_map_info(self):
        """更新地图信息"""
        obstacle_count = np.sum(self.grid == 1)
        total_cells = self.rows * self.cols
        obstacle_percentage = (obstacle_count / total_cells) * 100
        
        info_text = f"地图尺寸: {self.cols}×{self.rows}\n"
        info_text += f"障碍物: {obstacle_count} 个 ({obstacle_percentage:.1f}%)\n"
        info_text += f"出入口边: {len(self.edges)} 个"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.info_text.config(state=tk.DISABLED)
    
    def clear_map(self):
        """清空地图"""
        reply = messagebox.askyesno("确认", "确定要清空整个地图吗？")
        if reply:
            self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
            self.edges.clear()
            self.edge_id_counter = 1
            self.update_edge_list()
            self.draw_grid()
            self.update_map_info()
            self.set_status("地图已清空")
    
    def save_map(self):
        """保存地图"""
        map_name = self.name_entry.get().strip()
        if not map_name:
            messagebox.showerror("错误", "请输入地图名称")
            return
        
        # 转换边界数据
        edges_data = []
        for edge in self.edges:
            edges_data.append({
                "edge_id": edge.edge_id,
                "direction": edge.direction.value,
                "center_x": edge.center_x,
                "center_y": edge.center_y,
                "length": edge.length
            })
        
        # 创建地图数据
        map_data = {
            "map_info": {
                "name": map_name,
                "type": "lifelong_intersection",
                "width": self.cols,
                "height": self.rows,
                "cell_size": self.cell_size,
                "created_with": "LifelongMapEditor",
                "version": "1.0"
            },
            "grid": self.grid.tolist(),
            "intersection_edges": edges_data,
            "statistics": {
                "total_cells": self.rows * self.cols,
                "obstacle_cells": int(np.sum(self.grid == 1)),
                "edges_count": len(self.edges)
            }
        }
        
        filename = f"{map_name}.json"
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("保存成功", f"地图已保存: {filename}")
            self.set_status(f"已保存: {filename}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")
    
    def load_map(self):
        """加载地图"""
        file_path = filedialog.askopenfilename(
            title="选择地图文件",
            filetypes=[("JSON地图文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 加载基础信息
                map_info = data.get("map_info", {})
                self.cols = map_info.get("width", 50)
                self.rows = map_info.get("height", 50)
                
                # 更新UI
                self.col_entry.delete(0, tk.END)
                self.col_entry.insert(0, str(self.cols))
                self.row_entry.delete(0, tk.END)
                self.row_entry.insert(0, str(self.rows))
                
                # 加载网格
                if "grid" in data:
                    self.grid = np.array(data["grid"], dtype=np.int8)
                else:
                    self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
                
                # 加载边界
                self.edges.clear()
                for edge_data in data.get("intersection_edges", []):
                    edge = IntersectionEdge(
                        edge_id=edge_data["edge_id"],
                        direction=EdgeDirection(edge_data["direction"]),
                        center_x=edge_data["center_x"],
                        center_y=edge_data["center_y"],
                        length=edge_data.get("length", 5)
                    )
                    self.edges.append(edge)
                
                # 更新计数器
                if self.edges:
                    max_id = max([int(edge.edge_id.split('_')[-1]) for edge in self.edges 
                                 if edge.edge_id.split('_')[-1].isdigit()], default=0)
                    self.edge_id_counter = max_id + 1
                else:
                    self.edge_id_counter = 1
                
                # 更新地图名称
                map_name = map_info.get("name", os.path.basename(file_path).split('.')[0])
                self.name_entry.delete(0, tk.END)
                self.name_entry.insert(0, map_name)
                
                # 更新显示
                self.update_edge_list()
                self.draw_grid()
                self.update_map_info()
                
                messagebox.showinfo("加载成功", f"已加载地图: {map_name}")
                self.set_status(f"已加载: {map_name}")
                self.title(f"Lifelong路口地图编辑器 - {map_name}")
                
            except Exception as e:
                messagebox.showerror("错误", f"加载失败: {str(e)}")
    
    def set_status(self, message):
        """设置状态"""
        self.status_label.config(text=message)

if __name__ == "__main__":
    app = LifelongMapEditor()
    app.mainloop()