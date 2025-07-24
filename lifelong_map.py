import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import json
import os
import math
import random

class GatewayMapCreator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Lifelong MAPF 出入口边地图创建工具")
        self.geometry("1400x900")

        # 地图数据
        self.rows = 60
        self.cols = 60
        self.cell_size = 8  # 稍微小一点以便显示更大地图

        # 出入口边数据 - 新增核心功能
        self.gateways = []  # [{"id": 1, "type": "north", "start_x": 10, "start_y": 0, "end_x": 20, "end_y": 0, "capacity": 5}, ...]
        self.gateway_colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#F4A460", "#87CEEB"]
        
        # 网格 - 0为可通行，1为障碍物
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        
        self.current_tool = "obstacle"  # 默认工具
        self.brush_size = 2
        self.last_painted_cell = None
        self.next_gateway_id = 1
        self.is_drawing = False
        self.selected_gateway = None  # 当前选择的出入口
        
        # 任务生成设置
        self.initial_vehicles_per_gateway = 1
        self.max_vehicles_per_gateway = 5
        
        # 文件历史记录
        self.recent_files = self.load_recent_files()
        self.current_file_path = None
        
        # 创建UI
        self.create_ui()
        
        # 初始化方向提示
        self.on_gateway_type_change()

    def create_ui(self):
        # 主布局
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧控制面板
        control_frame = tk.Frame(main_frame, width=300)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)
        
        # 右侧画布
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建控制面板
        self.create_controls(control_frame)
        
        # 创建画布
        self.create_canvas(canvas_frame)
        
        # 初始化地图
        self.init_map()

    def create_controls(self, parent):
        # 地图信息显示
        info_frame = tk.LabelFrame(parent, text="🗺️ Lifelong MAPF 地图信息", padx=5, pady=5, 
                                 font=("Arial", 10, "bold"))
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, height=5, state=tk.DISABLED, 
                               font=("Arial", 8), bg="#f0f0f0")
        self.info_text.pack(fill=tk.X, pady=2)

        # 地图尺寸设置
        size_frame = tk.LabelFrame(parent, text="📐 地图尺寸", padx=5, pady=5)
        size_frame.pack(fill=tk.X, padx=5, pady=5)

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

        tk.Button(size_frame, text="更新尺寸", command=self.update_map_size).pack(pady=5)

        # 工具选择
        tool_frame = tk.LabelFrame(parent, text="🔧 绘图工具", padx=5, pady=5)
        tool_frame.pack(fill=tk.X, padx=5, pady=5)

        self.tool_var = tk.StringVar(value="obstacle")
        tools = [
            ("🚫 障碍物", "obstacle"),
            ("✅ 可通行", "passable"),
            ("🚪 出入口边", "gateway")
        ]

        for text, value in tools:
            rb = tk.Radiobutton(tool_frame, text=text, value=value, variable=self.tool_var, 
                              command=self.update_current_tool)
            rb.pack(anchor=tk.W)

        # 画笔大小
        brush_frame = tk.Frame(tool_frame)
        brush_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(brush_frame, text="画笔大小:").pack(side=tk.LEFT)
        self.brush_scale = tk.Scale(brush_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                   command=self.update_brush_size)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 🚪 出入口边管理 - 核心新功能
        gateway_frame = tk.LabelFrame(parent, text="🚪 出入口边管理", padx=5, pady=5, 
                                    font=("Arial", 10, "bold"))
        gateway_frame.pack(fill=tk.X, padx=5, pady=5)

        # 出入口边列表
        self.gateway_listbox = tk.Listbox(gateway_frame, height=6, font=("Arial", 8))
        gateway_scrollbar = ttk.Scrollbar(gateway_frame, orient=tk.VERTICAL, command=self.gateway_listbox.yview)
        self.gateway_listbox.configure(yscrollcommand=gateway_scrollbar.set)
        
        gateway_list_frame = tk.Frame(gateway_frame)
        gateway_list_frame.pack(fill=tk.X, pady=2)
        self.gateway_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        gateway_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 绑定选择事件
        self.gateway_listbox.bind("<<ListboxSelect>>", self.on_gateway_select)

        # 出入口边类型选择
        gateway_type_frame = tk.Frame(gateway_frame)
        gateway_type_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(gateway_type_frame, text="类型:").pack(side=tk.LEFT)
        self.gateway_type_var = tk.StringVar(value="north")
        self.gateway_type_combo = ttk.Combobox(gateway_type_frame, textvariable=self.gateway_type_var,
                                             values=["north", "south", "east", "west"], 
                                             state="readonly", width=8)
        self.gateway_type_combo.pack(side=tk.LEFT, padx=5)
        self.gateway_type_combo.bind("<<ComboboxSelected>>", self.on_gateway_type_change)
        
        # 方向提示
        self.direction_label = tk.Label(gateway_type_frame, text="(水平方向)", 
                                      font=("Arial", 7), fg="gray")
        self.direction_label.pack(side=tk.LEFT, padx=5)

        # 容量设置
        capacity_frame = tk.Frame(gateway_frame)
        capacity_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(capacity_frame, text="容量:").pack(side=tk.LEFT)
        self.capacity_entry = tk.Entry(capacity_frame, width=5)
        self.capacity_entry.pack(side=tk.LEFT, padx=5)
        self.capacity_entry.insert(0, "3")

        # 出入口边操作按钮
        gateway_btn_frame = tk.Frame(gateway_frame)
        gateway_btn_frame.pack(fill=tk.X, pady=5)

        tk.Button(gateway_btn_frame, text="🔄 自动生成", command=self.auto_generate_gateways, 
                width=10, bg="#4CAF50", fg="white").pack(side=tk.LEFT, padx=2)
        tk.Button(gateway_btn_frame, text="🗑️ 删除选中", command=self.delete_selected_gateway, 
                width=10).pack(side=tk.RIGHT, padx=2)

        # 快速布局按钮
        quick_layout_frame = tk.Frame(gateway_frame)
        quick_layout_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(quick_layout_frame, text="快速布局:").pack(anchor=tk.W)
        
        layout_buttons = [
            ("四边对称", self.create_symmetric_layout),
            ("仓库式", self.create_warehouse_layout),
            ("停车场", self.create_parking_layout)
        ]
        
        for text, command in layout_buttons:
            tk.Button(quick_layout_frame, text=text, command=command, width=8).pack(side=tk.LEFT, padx=1)

        # Lifelong 参数设置
        lifelong_frame = tk.LabelFrame(parent, text="🔄 Lifelong 参数", padx=5, pady=5)
        lifelong_frame.pack(fill=tk.X, padx=5, pady=5)

        # 车辆生成参数
        vehicle_frame = tk.Frame(lifelong_frame)
        vehicle_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(vehicle_frame, text="初始车辆/出入口:").grid(row=0, column=0, sticky=tk.W)
        self.initial_vehicles_entry = tk.Entry(vehicle_frame, width=5)
        self.initial_vehicles_entry.grid(row=0, column=1, padx=5)
        self.initial_vehicles_entry.insert(0, str(self.initial_vehicles_per_gateway))

        tk.Label(vehicle_frame, text="最大车辆/出入口:").grid(row=1, column=0, sticky=tk.W)
        self.max_vehicles_entry = tk.Entry(vehicle_frame, width=5)
        self.max_vehicles_entry.grid(row=1, column=1, padx=5)
        self.max_vehicles_entry.insert(0, str(self.max_vehicles_per_gateway))

        # 任务生成测试
        tk.Button(lifelong_frame, text="🧪 测试任务生成", command=self.test_task_generation,
                bg="#FF9800", fg="white").pack(fill=tk.X, pady=5)

        # 文件操作
        file_frame = tk.LabelFrame(parent, text="💾 文件操作", padx=5, pady=5)
        file_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(file_frame, text="地图名称:").pack(anchor=tk.W)
        self.name_entry = tk.Entry(file_frame)
        self.name_entry.pack(fill=tk.X, pady=2)
        self.name_entry.insert(0, "lifelong_gateway_map")

        # 文件操作按钮
        file_btn_frame = tk.Frame(file_frame)
        file_btn_frame.pack(fill=tk.X, pady=2)
        
        tk.Button(file_btn_frame, text="📁 加载", command=self.load_map, width=8).pack(side=tk.LEFT, padx=2)
        tk.Button(file_btn_frame, text="💾 保存", command=self.save_map, 
                bg="#2196F3", fg="white", width=8).pack(side=tk.RIGHT, padx=2)

        tk.Button(file_frame, text="🗑️ 清空地图", command=self.clear_map).pack(fill=tk.X, pady=2)

        # 状态信息
        self.status_label = tk.Label(parent, text="就绪 - 出入口边地图创建工具", bd=1, relief=tk.SUNKEN, 
                                   anchor=tk.W, font=("Arial", 8))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def create_canvas(self, parent):
        # 创建带滚动条的画布
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

    def init_map(self):
        """初始化地图"""
        self.draw_grid()
        self.update_gateway_list()
        self.update_map_info()
        self.set_status("Lifelong MAPF 地图已初始化")

    def draw_grid(self):
        """绘制网格和所有元素"""
        self.canvas.delete("all")
        
        # 计算画布大小并设置滚动区域
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
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray", tags="cell")
        
        # 🚪 绘制出入口边 - 核心功能
        for i, gateway in enumerate(self.gateways):
            color = self.gateway_colors[i % len(self.gateway_colors)]
            self.draw_gateway(gateway, color)

    def draw_gateway(self, gateway, color):
        """绘制单个出入口边"""
        start_x = gateway["start_x"] * self.cell_size
        start_y = gateway["start_y"] * self.cell_size
        end_x = gateway["end_x"] * self.cell_size
        end_y = gateway["end_y"] * self.cell_size
        
        # 绘制出入口边主线 - 加粗
        line_width = 6
        self.canvas.create_line(start_x, start_y, end_x, end_y, 
                              fill=color, width=line_width, capstyle=tk.ROUND,
                              tags=f"gateway_{gateway['id']}")
        
        # 绘制出入口边标识
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        # 绘制标识圆圈
        radius = 12
        self.canvas.create_oval(mid_x - radius, mid_y - radius, 
                              mid_x + radius, mid_y + radius,
                              fill="white", outline=color, width=2)
        
        # 绘制ID标签
        self.canvas.create_text(mid_x, mid_y, 
                              text=str(gateway['id']),
                              font=("Arial", 8, "bold"), fill=color)
        
        # 绘制类型标识
        type_text = gateway['type'][0].upper()  # N, S, E, W
        offset = 20
        if gateway['type'] == 'north':
            text_y = mid_y - offset
        elif gateway['type'] == 'south':
            text_y = mid_y + offset
        elif gateway['type'] == 'east':
            text_x = mid_x + offset
            text_y = mid_y
        else:  # west
            text_x = mid_x - offset
            text_y = mid_y
        
        if gateway['type'] in ['north', 'south']:
            text_x = mid_x
        
        self.canvas.create_text(text_x, text_y, 
                              text=f"{type_text}:{gateway['capacity']}", 
                              font=("Arial", 7), fill=color)

    # 🚪 出入口边管理方法
    def auto_generate_gateways(self):
        """自动生成出入口边"""
        self.gateways = []
        
        # 北边
        self.gateways.append({
            "id": 1, "type": "north", 
            "start_x": 15, "start_y": 0, "end_x": 25, "end_y": 0, "capacity": 3
        })
        self.gateways.append({
            "id": 2, "type": "north",
            "start_x": 35, "start_y": 0, "end_x": 45, "end_y": 0, "capacity": 3
        })
        
        # 南边
        self.gateways.append({
            "id": 3, "type": "south",
            "start_x": 10, "start_y": self.rows-1, "end_x": 20, "end_y": self.rows-1, "capacity": 3
        })
        self.gateways.append({
            "id": 4, "type": "south",
            "start_x": 40, "start_y": self.rows-1, "end_x": 50, "end_y": self.rows-1, "capacity": 3
        })
        
        # 东边
        self.gateways.append({
            "id": 5, "type": "east",
            "start_x": self.cols-1, "start_y": 15, "end_x": self.cols-1, "end_y": 25, "capacity": 3
        })
        
        # 西边
        self.gateways.append({
            "id": 6, "type": "west",
            "start_x": 0, "start_y": 20, "end_x": 0, "end_y": 30, "capacity": 3
        })
        
        self.next_gateway_id = 7
        self.draw_grid()
        self.update_gateway_list()
        self.update_map_info()
        self.set_status("已自动生成6个出入口边")

    def create_symmetric_layout(self):
        """创建四边对称布局"""
        self.gateways = []
        gateway_length = 8
        
        # 四边各2个出入口
        for i in range(2):
            offset = 15 + i * 20
            
            # 北边
            self.gateways.append({
                "id": len(self.gateways) + 1, "type": "north",
                "start_x": offset, "start_y": 0, 
                "end_x": offset + gateway_length, "end_y": 0, "capacity": 3
            })
            
            # 南边
            self.gateways.append({
                "id": len(self.gateways) + 1, "type": "south",
                "start_x": offset, "start_y": self.rows-1,
                "end_x": offset + gateway_length, "end_y": self.rows-1, "capacity": 3
            })
            
            # 东边
            self.gateways.append({
                "id": len(self.gateways) + 1, "type": "east",
                "start_x": self.cols-1, "start_y": offset,
                "end_x": self.cols-1, "end_y": offset + gateway_length, "capacity": 3
            })
            
            # 西边
            self.gateways.append({
                "id": len(self.gateways) + 1, "type": "west",
                "start_x": 0, "start_y": offset,
                "end_x": 0, "end_y": offset + gateway_length, "capacity": 3
            })
        
        self.next_gateway_id = len(self.gateways) + 1
        self._update_display()

    def create_warehouse_layout(self):
        """创建仓库式布局"""
        self.gateways = []
        
        # 装卸区域（北边多个小出入口）
        for i in range(4):
            x_start = 8 + i * 12
            self.gateways.append({
                "id": len(self.gateways) + 1, "type": "north",
                "start_x": x_start, "start_y": 0,
                "end_x": x_start + 6, "end_y": 0, "capacity": 2
            })
        
        # 运输通道（东西两边）
        for i in range(2):
            y_start = 15 + i * 20
            # 西边
            self.gateways.append({
                "id": len(self.gateways) + 1, "type": "west",
                "start_x": 0, "start_y": y_start,
                "end_x": 0, "end_y": y_start + 10, "capacity": 4
            })
            # 东边
            self.gateways.append({
                "id": len(self.gateways) + 1, "type": "east",
                "start_x": self.cols-1, "start_y": y_start,
                "end_x": self.cols-1, "end_y": y_start + 10, "capacity": 4
            })
        
        self.next_gateway_id = len(self.gateways) + 1
        self._add_warehouse_obstacles()
        self._update_display()

    def create_parking_layout(self):
        """创建停车场布局"""
        self.gateways = []
        
        # 主入口（南边一个大入口）
        self.gateways.append({
            "id": 1, "type": "south",
            "start_x": 25, "start_y": self.rows-1,
            "end_x": 35, "end_y": self.rows-1, "capacity": 6
        })
        
        # 出口（北边分散的小出口）
        for i in range(3):
            x_start = 10 + i * 15
            self.gateways.append({
                "id": len(self.gateways) + 1, "type": "north",
                "start_x": x_start, "start_y": 0,
                "end_x": x_start + 8, "end_y": 0, "capacity": 3
            })
        
        self.next_gateway_id = len(self.gateways) + 1
        self._add_parking_obstacles()
        self._update_display()

    def _add_warehouse_obstacles(self):
        """添加仓库风格的障碍物"""
        # 中央存储区域
        for row in range(20, 40):
            for col in range(20, 40):
                if (row - 20) % 8 < 6 and (col - 20) % 8 < 6:
                    self.grid[row, col] = 1

    def _add_parking_obstacles(self):
        """添加停车场风格的障碍物"""
        # 停车位行
        for row_start in [10, 20, 30, 40]:
            for col in range(5, 55, 12):
                for r in range(row_start, row_start + 6):
                    for c in range(col, col + 8):
                        if r < self.rows and c < self.cols:
                            self.grid[r, c] = 1

    def _update_display(self):
        """更新显示"""
        self.draw_grid()
        self.update_gateway_list()
        self.update_map_info()

    def on_canvas_click(self, event):
        """处理画布点击事件"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.last_painted_cell = (row, col)
            self.is_drawing = True
            
            if self.current_tool == "gateway":
                self.start_gateway_creation(row, col)
            else:
                self.apply_tool(row, col)

    def start_gateway_creation(self, row, col):
        """开始创建出入口边 - 在鼠标点击位置创建"""
        gateway_type = self.gateway_type_var.get()
        try:
            capacity = int(self.capacity_entry.get())
        except ValueError:
            capacity = 3
        
        # 根据类型确定出入口边的方向和长度
        gateway_length = 10
        
        if gateway_type == "north" or gateway_type == "south":
            # 水平方向的出入口边，以点击位置为中心
            start_x = max(0, col - gateway_length//2)
            end_x = min(self.cols-1, col + gateway_length//2)
            start_y = end_y = row  # 在点击的行上
        else:  # east 或 west
            # 垂直方向的出入口边，以点击位置为中心
            start_y = max(0, row - gateway_length//2)
            end_y = min(self.rows-1, row + gateway_length//2)
            start_x = end_x = col  # 在点击的列上
        
        # 创建新出入口边
        new_gateway = {
            "id": self.next_gateway_id,
            "type": gateway_type,
            "start_x": start_x, "start_y": start_y,
            "end_x": end_x, "end_y": end_y,
            "capacity": capacity
        }
        
        self.gateways.append(new_gateway)
        self.next_gateway_id += 1
        
        self.is_drawing = False
        self.draw_grid()
        self.update_gateway_list()
        self.update_map_info()
        self.set_status(f"已在 ({col},{row}) 创建出入口边 {new_gateway['id']}: {gateway_type}")

    def on_gateway_type_change(self, event=None):
        """出入口边类型改变时更新提示"""
        gateway_type = self.gateway_type_var.get()
        if gateway_type in ["north", "south"]:
            self.direction_label.config(text="(水平方向)")
        else:  # east, west
            self.direction_label.config(text="(垂直方向)")

    def on_gateway_select(self, event):
        """出入口边选择事件"""
        selection = self.gateway_listbox.curselection()
        if selection:
            self.selected_gateway = selection[0]

    def delete_selected_gateway(self):
        """删除选中的出入口边"""
        if self.selected_gateway is not None and self.selected_gateway < len(self.gateways):
            deleted_gateway = self.gateways.pop(self.selected_gateway)
            self.selected_gateway = None
            self.draw_grid()
            self.update_gateway_list()
            self.update_map_info()
            self.set_status(f"已删除出入口边 {deleted_gateway['id']}")

    def update_gateway_list(self):
        """更新出入口边列表显示"""
        self.gateway_listbox.delete(0, tk.END)
        
        for gateway in self.gateways:
            display_text = f"G{gateway['id']}: {gateway['type']} (容量:{gateway['capacity']})"
            self.gateway_listbox.insert(tk.END, display_text)

    def test_task_generation(self):
        """测试任务生成"""
        if len(self.gateways) < 2:
            messagebox.showwarning("警告", "需要至少2个出入口边才能生成任务")
            return
        
        # 模拟生成几个任务
        test_results = []
        for i in range(5):
            source_gateway = random.choice(self.gateways)
            target_gateways = [g for g in self.gateways if g['id'] != source_gateway['id']]
            target_gateway = random.choice(target_gateways)
            
            # 在出入口边上生成随机点
            source_point = self.get_random_point_on_gateway(source_gateway)
            target_point = self.get_random_point_on_gateway(target_gateway)
            
            test_results.append({
                'task_id': i + 1,
                'source_gateway': source_gateway['id'],
                'target_gateway': target_gateway['id'],
                'source_point': source_point,
                'target_point': target_point
            })
        
        # 显示结果
        result_text = "🧪 任务生成测试结果:\n\n"
        for task in test_results:
            result_text += f"任务 {task['task_id']}: G{task['source_gateway']} → G{task['target_gateway']}\n"
            result_text += f"  起点: ({task['source_point'][0]:.1f}, {task['source_point'][1]:.1f})\n"
            result_text += f"  终点: ({task['target_point'][0]:.1f}, {task['target_point'][1]:.1f})\n\n"
        
        messagebox.showinfo("任务生成测试", result_text)

    def get_random_point_on_gateway(self, gateway):
        """在出入口边上生成随机点"""
        t = random.uniform(0, 1)
        x = gateway['start_x'] + t * (gateway['end_x'] - gateway['start_x'])
        y = gateway['start_y'] + t * (gateway['end_y'] - gateway['start_y'])
        return (x, y)

    def update_map_info(self):
        """更新地图信息显示"""
        obstacle_count = np.sum(self.grid == 1)
        total_cells = self.rows * self.cols
        obstacle_percentage = (obstacle_count / total_cells) * 100
        
        total_capacity = sum(g['capacity'] for g in self.gateways)
        
        info_text = f"🗺️ Lifelong MAPF 地图信息\n"
        info_text += f"尺寸: {self.cols}×{self.rows} ({total_cells} 格子)\n"
        info_text += f"障碍物: {obstacle_count} 个 ({obstacle_percentage:.1f}%)\n"
        info_text += f"🚪 出入口边: {len(self.gateways)} 个\n"
        info_text += f"总容量: {total_capacity} 辆车\n"
        
        if self.gateways:
            types = {}
            for g in self.gateways:
                types[g['type']] = types.get(g['type'], 0) + 1
            type_info = ", ".join([f"{k}:{v}" for k, v in types.items()])
            info_text += f"类型分布: {type_info}"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.info_text.config(state=tk.DISABLED)

    # 保持原有的基础方法
    def update_current_tool(self):
        self.current_tool = self.tool_var.get()
        self.set_status(f"当前工具: {self.current_tool}")

    def update_brush_size(self, value):
        self.brush_size = int(value)

    def on_canvas_release(self, event):
        """处理鼠标释放事件"""
        self.last_painted_cell = None
        self.is_drawing = False
        # 释放时重绘整个画布以确保显示正确
        self.draw_grid()
        self.update_map_info()  # 更新地图信息

    def on_canvas_drag(self, event):
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= col < self.cols and 0 <= row < self.rows and (row, col) != self.last_painted_cell:
            self.last_painted_cell = (row, col)
            if self.current_tool in ["obstacle", "passable"]:
                self.apply_tool(row, col)

    def on_right_click(self, event):
        # 右键可以用来删除出入口边
        pass

    def apply_tool(self, row, col):
        if self.current_tool == "obstacle":
            self.paint_obstacle(row, col)
        elif self.current_tool == "passable":
            self.clear_obstacle(row, col)

    def paint_obstacle(self, center_row, center_col):
        """绘制障碍物"""
        radius = self.brush_size
        
        # 将圆形区域内的点设为障碍物
        for r in range(max(0, center_row - radius), min(self.rows, center_row + radius + 1)):
            for c in range(max(0, center_col - radius), min(self.cols, center_col + radius + 1)):
                if ((r - center_row) ** 2 + (c - center_col) ** 2) <= radius ** 2:
                    if self.grid[r, c] != 1:  # 只有当状态改变时才更新
                        self.grid[r, c] = 1
                        # 绘制过程中只更新单个格子
                        if self.is_drawing:
                            self.draw_single_cell(r, c)
        
        # 如果不是在绘制过程中，重绘整个画布
        if not self.is_drawing:
            self.draw_grid()

    def clear_obstacle(self, center_row, center_col):
        """清除障碍物"""
        radius = self.brush_size
        
        # 将圆形区域内的点设为可通行
        for r in range(max(0, center_row - radius), min(self.rows, center_row + radius + 1)):
            for c in range(max(0, center_col - radius), min(self.cols, center_col + radius + 1)):
                if ((r - center_row) ** 2 + (c - center_col) ** 2) <= radius ** 2:
                    if self.grid[r, c] != 0:  # 只有当状态改变时才更新
                        self.grid[r, c] = 0
                        # 绘制过程中只更新单个格子
                        if self.is_drawing:
                            self.draw_single_cell(r, c)
        
        # 如果不是在绘制过程中，重绘整个画布
        if not self.is_drawing:
            self.draw_grid()

    def draw_single_cell(self, row, col):
        """绘制单个格子"""
        x1 = col * self.cell_size
        y1 = row * self.cell_size
        x2 = x1 + self.cell_size
        y2 = y1 + self.cell_size
        
        # 删除该位置的现有矩形
        overlapping = self.canvas.find_overlapping(x1, y1, x2, y2)
        for item in overlapping:
            tags = self.canvas.gettags(item)
            if "cell" in tags:
                self.canvas.delete(item)
        
        # 绘制新的格子状态
        if self.grid[row, col] == 1:
            self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", outline="gray", tags="cell")
        # 可通行区域不需要绘制，背景就是白色

    def update_map_size(self):
        try:
            new_cols = int(self.col_entry.get())
            new_rows = int(self.row_entry.get())
            
            if new_rows <= 0 or new_cols <= 0:
                messagebox.showerror("错误", "宽度和高度必须大于0")
                return
            
            new_grid = np.zeros((new_rows, new_cols), dtype=np.int8)
            min_rows = min(self.rows, new_rows)
            min_cols = min(self.cols, new_cols)
            new_grid[:min_rows, :min_cols] = self.grid[:min_rows, :min_cols]
            
            # 过滤超出范围的出入口边
            self.gateways = [g for g in self.gateways 
                           if (g['start_x'] < new_cols and g['start_y'] < new_rows and
                               g['end_x'] < new_cols and g['end_y'] < new_rows)]
            
            self.grid = new_grid
            self.rows = new_rows
            self.cols = new_cols
            
            self.draw_grid()
            self.update_gateway_list()
            self.update_map_info()
            self.set_status(f"地图大小已更新为 {new_cols}x{new_rows}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")

    def save_map(self):
        map_name = self.name_entry.get()
        if not map_name:
            messagebox.showerror("错误", "请输入地图名称")
            return
        
        # 🚪 新的保存格式，包含出入口边信息
        map_data = {
            "map_info": {
                "name": map_name,
                "width": self.cols,
                "height": self.rows,
                "cell_size": self.cell_size,
                "map_type": "lifelong_gateway",  # 标识为lifelong地图
                "created_with": "GatewayMapCreator",
                "version": "1.0"
            },
            "grid": self.grid.tolist(),
            "gateways": self.gateways,  # 核心：出入口边信息
            "lifelong_config": {
                "initial_vehicles_per_gateway": int(self.initial_vehicles_entry.get()),
                "max_vehicles_per_gateway": int(self.max_vehicles_entry.get()),
                "task_generation_method": "random_gateway_to_gateway"
            },
            "statistics": {
                "total_cells": self.rows * self.cols,
                "obstacle_cells": int(np.sum(self.grid == 1)),
                "gateway_count": len(self.gateways),
                "total_capacity": sum(g['capacity'] for g in self.gateways)
            }
        }
        
        filename = f"{map_name}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=2, ensure_ascii=False)
            
            messagebox.showinfo("保存成功", f"Lifelong地图已保存为: {filename}")
            self.set_status(f"Lifelong地图已保存: {filename}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load_map(self):
        file_path = filedialog.askopenfilename(
            title="选择Lifelong地图文件",
            filetypes=[("JSON地图文件", "*.json"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.load_map_from_path(file_path)

    def load_map_from_path(self, file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载基本信息
            map_info = data.get("map_info", {})
            self.cols = map_info.get("width", 60)
            self.rows = map_info.get("height", 60)
            
            # 加载网格
            if "grid" in data:
                self.grid = np.array(data["grid"], dtype=np.int8)
            
            # 🚪 加载出入口边
            self.gateways = data.get("gateways", [])
            if self.gateways:
                self.next_gateway_id = max(g["id"] for g in self.gateways) + 1
            
            # 加载lifelong配置
            lifelong_config = data.get("lifelong_config", {})
            if lifelong_config:
                self.initial_vehicles_entry.delete(0, tk.END)
                self.initial_vehicles_entry.insert(0, str(lifelong_config.get("initial_vehicles_per_gateway", 1)))
                self.max_vehicles_entry.delete(0, tk.END)
                self.max_vehicles_entry.insert(0, str(lifelong_config.get("max_vehicles_per_gateway", 5)))
            
            # 更新显示
            self.col_entry.delete(0, tk.END)
            self.col_entry.insert(0, str(self.cols))
            self.row_entry.delete(0, tk.END)
            self.row_entry.insert(0, str(self.rows))
            
            map_name = map_info.get("name", os.path.basename(file_path).split('.')[0])
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, map_name)
            
            self.draw_grid()
            self.update_gateway_list()
            self.update_map_info()
            self.set_status(f"已加载Lifelong地图: {os.path.basename(file_path)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"加载地图失败: {str(e)}")

    def clear_map(self):
        reply = messagebox.askyesno("确认", "确定要清空地图吗？")
        if reply:
            self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
            self.gateways = []
            self.next_gateway_id = 1
            self.draw_grid()
            self.update_gateway_list()
            self.update_map_info()
            self.set_status("Lifelong地图已清空")

    def load_recent_files(self):
        return []

    def set_status(self, message):
        self.status_label.config(text=message)

if __name__ == "__main__":
    app = GatewayMapCreator()
    app.mainloop()