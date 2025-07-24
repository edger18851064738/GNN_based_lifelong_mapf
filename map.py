import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import json
import os
import tkinter.dnd as dnd

class EnhancedMapCreator(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("增强版地图创建工具 - 支持导入修改")
        self.geometry("1400x900")  # 增加窗口大小

        # 地图数据
        self.rows = 50
        self.cols = 50
        self.cell_size = 10  # 单元格大小

        # 点位数据
        self.start_points = []  # 起点列表 [{"id": 1, "x": 10, "y": 20}, ...]
        self.end_points = []    # 终点列表 [{"id": 1, "x": 30, "y": 40}, ...]
        self.point_pairs = []   # 配对列表 [{"start_id": 1, "end_id": 1}, ...]
        
        # 网格 - 0为可通行，1为障碍物
        self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
        
        self.current_tool = "obstacle"  # 默认工具
        self.brush_size = 2  # 默认画笔大小
        self.last_painted_cell = None  # 避免重复绘制
        self.next_point_id = 1  # 下一个点位ID
        self.is_drawing = False  # 是否正在绘制
        
        # 文件历史记录
        self.recent_files = self.load_recent_files()
        self.current_file_path = None
        
        # 创建UI
        self.create_ui()
        
        # 设置拖拽支持
        self.setup_drag_drop()

    def create_ui(self):
        # 主布局
        main_frame = tk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 左侧控制面板 - 增加宽度
        control_frame = tk.Frame(main_frame, width=320)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        control_frame.pack_propagate(False)  # 固定宽度
        
        # 右侧画布
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建控制面板
        self.create_controls(control_frame)
        
        # 创建画布
        self.create_canvas(canvas_frame)
        
        # 初始化地图
        self.init_map()

    def setup_drag_drop(self):
        """设置拖拽功能"""
        # 绑定拖拽事件到画布
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        
        # 绑定文件拖拽（需要tkinterdnd2库，这里提供基础支持）
        self.bind("<Control-o>", lambda e: self.load_map())  # Ctrl+O快捷键

    def create_controls(self, parent):
        # 文件导入区域 - 放在最上面，更醒目
        import_frame = tk.LabelFrame(parent, text="📁 导入地图", padx=5, pady=5, 
                                   font=("Arial", 10, "bold"))
        import_frame.pack(fill=tk.X, padx=2, pady=3)

        # 导入按钮
        import_btn = tk.Button(import_frame, text="🔍 浏览并导入地图", 
                             command=self.load_map, bg="#4CAF50", fg="white",
                             font=("Arial", 9, "bold"))
        import_btn.pack(fill=tk.X, pady=2)
        
        # 快速导入区域
        quick_import_frame = tk.Frame(import_frame)
        quick_import_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(quick_import_frame, text="快速导入:", font=("Arial", 8)).pack(anchor=tk.W)
        
        # 最近文件下拉框
        self.recent_var = tk.StringVar()
        self.recent_combo = ttk.Combobox(quick_import_frame, textvariable=self.recent_var,
                                       state="readonly", width=35)
        self.recent_combo.pack(fill=tk.X, pady=1)
        self.recent_combo.bind("<<ComboboxSelected>>", self.load_recent_file)
        self.update_recent_files_combo()
        
        # 导入状态显示
        self.import_status = tk.Label(import_frame, text="拖拽JSON文件到此处或点击浏览", 
                                    fg="gray", font=("Arial", 8))
        self.import_status.pack(pady=2)

        # 地图信息显示
        info_frame = tk.LabelFrame(parent, text="📊 地图信息", padx=5, pady=5)
        info_frame.pack(fill=tk.X, padx=2, pady=3)
        
        self.info_text = tk.Text(info_frame, height=4, state=tk.DISABLED, 
                               font=("Arial", 8), bg="#f0f0f0")
        self.info_text.pack(fill=tk.X, pady=2)

        # 地图尺寸设置
        size_frame = tk.LabelFrame(parent, text="📐 地图尺寸", padx=5, pady=5)
        size_frame.pack(fill=tk.X, padx=2, pady=3)

        size_grid = tk.Frame(size_frame)
        size_grid.pack(fill=tk.X)

        tk.Label(size_grid, text="宽度:").grid(row=0, column=0, sticky=tk.W)
        self.col_entry = tk.Entry(size_grid, width=8)
        self.col_entry.grid(row=0, column=1, padx=5, pady=2)
        self.col_entry.insert(0, str(self.cols))
        # 绑定回车键自动更新
        self.col_entry.bind("<Return>", lambda e: self.update_map_size())

        tk.Label(size_grid, text="高度:").grid(row=1, column=0, sticky=tk.W)
        self.row_entry = tk.Entry(size_grid, width=8)
        self.row_entry.grid(row=1, column=1, padx=5, pady=2)
        self.row_entry.insert(0, str(self.rows))
        # 绑定回车键自动更新
        self.row_entry.bind("<Return>", lambda e: self.update_map_size())

        tk.Button(size_frame, text="更新尺寸", command=self.update_map_size).pack(pady=3)

        # 工具选择
        tool_frame = tk.LabelFrame(parent, text="🔧 绘图工具", padx=5, pady=5)
        tool_frame.pack(fill=tk.X, padx=2, pady=3)

        self.tool_var = tk.StringVar(value="obstacle")
        tools = [
            ("🚫 障碍物", "obstacle", "black"),
            ("✅ 可通行", "passable", "white"),
            ("🟢 车辆起点", "start_point", "green"),
            ("🔴 车辆终点", "end_point", "red")
        ]

        for i, (text, value, color) in enumerate(tools):
            rb = tk.Radiobutton(tool_frame, text=text, value=value, variable=self.tool_var, 
                              command=self.update_current_tool)
            rb.pack(anchor=tk.W)

        # 画笔大小
        brush_frame = tk.Frame(tool_frame)
        brush_frame.pack(fill=tk.X, pady=3)
        
        tk.Label(brush_frame, text="画笔大小:").pack(side=tk.LEFT)
        self.brush_scale = tk.Scale(brush_frame, from_=1, to=10, orient=tk.HORIZONTAL,
                                   command=self.update_brush_size)
        self.brush_scale.set(self.brush_size)
        self.brush_scale.pack(side=tk.RIGHT, fill=tk.X, expand=True)

        # 点位配对管理
        pair_frame = tk.LabelFrame(parent, text="🔗 起点终点配对", padx=5, pady=5)
        pair_frame.pack(fill=tk.X, padx=2, pady=3)

        # 配对列表
        pair_list_frame = tk.Frame(pair_frame)
        pair_list_frame.pack(fill=tk.X, pady=2)
        
        self.pair_listbox = tk.Listbox(pair_list_frame, height=4, font=("Arial", 8))
        pair_scrollbar = ttk.Scrollbar(pair_list_frame, orient=tk.VERTICAL, command=self.pair_listbox.yview)
        self.pair_listbox.configure(yscrollcommand=pair_scrollbar.set)
        
        self.pair_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pair_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 配对操作按钮
        pair_btn_frame = tk.Frame(pair_frame)
        pair_btn_frame.pack(fill=tk.X, pady=2)

        tk.Button(pair_btn_frame, text="自动配对", command=self.auto_pair_points, width=10).pack(side=tk.LEFT, padx=1)
        tk.Button(pair_btn_frame, text="清除配对", command=self.clear_pairs, width=10).pack(side=tk.RIGHT, padx=1)

        # 手动配对
        manual_pair_frame = tk.Frame(pair_frame)
        manual_pair_frame.pack(fill=tk.X, pady=2)

        tk.Label(manual_pair_frame, text="起点ID:").pack(side=tk.LEFT)
        self.start_id_entry = tk.Entry(manual_pair_frame, width=4)
        self.start_id_entry.pack(side=tk.LEFT, padx=2)

        tk.Label(manual_pair_frame, text="终点ID:").pack(side=tk.LEFT)
        self.end_id_entry = tk.Entry(manual_pair_frame, width=4)
        self.end_id_entry.pack(side=tk.LEFT, padx=2)

        tk.Button(manual_pair_frame, text="配对", command=self.manual_pair_points).pack(side=tk.RIGHT, padx=2)

        # 文件操作
        file_frame = tk.LabelFrame(parent, text="💾 文件操作", padx=5, pady=5)
        file_frame.pack(fill=tk.X, padx=2, pady=3)

        tk.Label(file_frame, text="地图名称:").pack(anchor=tk.W)
        self.name_entry = tk.Entry(file_frame)
        self.name_entry.pack(fill=tk.X, pady=2)
        self.name_entry.insert(0, "simple_map")

        # 文件操作按钮
        file_btn_frame = tk.Frame(file_frame)
        file_btn_frame.pack(fill=tk.X, pady=2)
        
        save_btn = tk.Button(file_btn_frame, text="💾 保存", command=self.save_map, 
                           bg="#2196F3", fg="white", width=12)
        save_btn.pack(side=tk.LEFT, padx=2)
        
        save_as_btn = tk.Button(file_btn_frame, text="📄 另存为", command=self.save_as_map, 
                              width=12)
        save_as_btn.pack(side=tk.RIGHT, padx=2)

        clear_btn = tk.Button(file_frame, text="🗑️ 清空地图", command=self.clear_map)
        clear_btn.pack(fill=tk.X, pady=2)

        # 状态信息
        self.status_label = tk.Label(parent, text="就绪", bd=1, relief=tk.SUNKEN, anchor=tk.W,
                                   font=("Arial", 8))
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X, pady=(5, 0))

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
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mousewheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mousewheel)  # Linux scroll down
        self.canvas.bind("<Button-3>", self.on_right_click)  # 右键删除点位

    def load_recent_files(self):
        """加载最近打开的文件列表"""
        try:
            if os.path.exists("recent_maps.json"):
                with open("recent_maps.json", 'r', encoding='utf-8') as f:
                    return json.load(f)
        except:
            pass
        return []

    def save_recent_files(self):
        """保存最近打开的文件列表"""
        try:
            with open("recent_maps.json", 'w', encoding='utf-8') as f:
                json.dump(self.recent_files, f, indent=2, ensure_ascii=False)
        except:
            pass

    def add_to_recent_files(self, file_path):
        """添加文件到最近打开列表"""
        if file_path in self.recent_files:
            self.recent_files.remove(file_path)
        self.recent_files.insert(0, file_path)
        self.recent_files = self.recent_files[:10]  # 只保留最近10个
        self.save_recent_files()
        self.update_recent_files_combo()

    def update_recent_files_combo(self):
        """更新最近文件下拉框"""
        # 过滤存在的文件
        existing_files = [f for f in self.recent_files if os.path.exists(f)]
        self.recent_files = existing_files
        
        # 显示文件名而不是完整路径
        display_names = [os.path.basename(f) for f in existing_files]
        self.recent_combo['values'] = display_names
        
        if display_names:
            self.recent_combo.set("选择最近的地图...")

    def load_recent_file(self, event=None):
        """加载最近选择的文件"""
        selection = self.recent_combo.current()
        if selection >= 0 and selection < len(self.recent_files):
            file_path = self.recent_files[selection]
            self.load_map_from_path(file_path)

    def update_map_info(self):
        """更新地图信息显示"""
        obstacle_count = np.sum(self.grid == 1)
        total_cells = self.rows * self.cols
        obstacle_percentage = (obstacle_count / total_cells) * 100
        
        info_text = f"尺寸: {self.cols}×{self.rows} ({total_cells} 格子)\n"
        info_text += f"障碍物: {obstacle_count} 个 ({obstacle_percentage:.1f}%)\n"
        info_text += f"起点: {len(self.start_points)} 个\n"
        info_text += f"终点: {len(self.end_points)} 个 | 配对: {len(self.point_pairs)} 对"
        
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(1.0, info_text)
        self.info_text.config(state=tk.DISABLED)

    def validate_map_data(self, data):
        """验证地图数据的完整性"""
        errors = []
        
        # 检查必要字段
        if "map_info" not in data:
            errors.append("缺少地图信息 (map_info)")
        
        if "start_points" not in data:
            errors.append("缺少起点信息 (start_points)")
            
        if "end_points" not in data:
            errors.append("缺少终点信息 (end_points)")
        
        # 检查网格数据
        if "grid" not in data and "obstacles" not in data:
            errors.append("缺少地图数据 (grid 或 obstacles)")
        
        # 检查数据有效性
        if "map_info" in data:
            map_info = data["map_info"]
            if "width" not in map_info or "height" not in map_info:
                errors.append("地图尺寸信息不完整")
        
        return errors

    def load_map_from_path(self, file_path):
        """从指定路径加载地图"""
        if not os.path.exists(file_path):
            messagebox.showerror("错误", f"文件不存在: {file_path}")
            return False
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 验证数据
            errors = self.validate_map_data(data)
            if errors:
                error_msg = "地图数据验证失败:\n" + "\n".join(errors)
                messagebox.showerror("数据错误", error_msg)
                return False
            
            # 备份当前状态（用于撤销）
            self.backup_current_state()
            
            # 加载地图信息
            map_info = data.get("map_info", {})
            self.cols = map_info.get("width", 50)
            self.rows = map_info.get("height", 50)
            self.cell_size = map_info.get("cell_size", 10)
            
            # 更新UI输入框
            self.col_entry.delete(0, tk.END)
            self.col_entry.insert(0, str(self.cols))
            self.row_entry.delete(0, tk.END)
            self.row_entry.insert(0, str(self.rows))
            
            # 加载网格
            if "grid" in data:
                grid_data = np.array(data["grid"], dtype=np.int8)
                # 确保网格尺寸匹配
                if grid_data.shape != (self.rows, self.cols):
                    # 调整网格大小
                    self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
                    min_rows = min(grid_data.shape[0], self.rows)
                    min_cols = min(grid_data.shape[1], self.cols)
                    self.grid[:min_rows, :min_cols] = grid_data[:min_rows, :min_cols]
                else:
                    self.grid = grid_data
            else:
                # 从障碍物重建网格
                self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
                for obstacle in data.get("obstacles", []):
                    x, y = obstacle["x"], obstacle["y"]
                    if 0 <= x < self.cols and 0 <= y < self.rows:
                        self.grid[y, x] = 1
            
            # 加载点位
            self.start_points = data.get("start_points", [])
            self.end_points = data.get("end_points", [])
            self.point_pairs = data.get("point_pairs", [])
            
            # 更新下一个点位ID
            max_start_id = max([p["id"] for p in self.start_points], default=0)
            max_end_id = max([p["id"] for p in self.end_points], default=0)
            self.next_point_id = max(max_start_id, max_end_id) + 1
            
            # 更新地图名称
            map_name = map_info.get("name", os.path.basename(file_path).split('.')[0])
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, map_name)
            
            # 记录当前文件路径
            self.current_file_path = file_path
            
            # 添加到最近文件
            self.add_to_recent_files(file_path)
            
            # 重绘地图和更新信息
            self.draw_grid()
            self.update_pair_list()
            self.update_map_info()
            
            # 更新状态
            self.import_status.config(text=f"✅ 已导入: {os.path.basename(file_path)}", fg="green")
            self.set_status(f"已导入地图: {os.path.basename(file_path)}")
            self.title(f"增强版地图创建工具 - {map_name}")
            
            return True
            
        except json.JSONDecodeError:
            messagebox.showerror("错误", "无效的JSON文件格式")
            return False
        except Exception as e:
            messagebox.showerror("错误", f"导入地图失败: {str(e)}")
            return False

    def backup_current_state(self):
        """备份当前状态"""
        self.backup_state = {
            'grid': self.grid.copy(),
            'start_points': self.start_points.copy(),
            'end_points': self.end_points.copy(),
            'point_pairs': self.point_pairs.copy(),
            'rows': self.rows,
            'cols': self.cols,
            'next_point_id': self.next_point_id
        }

    def load_map(self):
        """加载地图文件"""
        file_path = filedialog.askopenfilename(
            title="选择要导入的地图文件",
            filetypes=[
                ("JSON地图文件", "*.json"), 
                ("所有文件", "*.*")
            ],
            initialdir=os.getcwd()
        )
        
        if file_path:
            self.load_map_from_path(file_path)

    def save_as_map(self):
        """另存为地图"""
        map_name = self.name_entry.get()
        if not map_name:
            map_name = "new_map"
            
        file_path = filedialog.asksaveasfilename(
            title="保存地图文件",
            defaultextension=".json",
            filetypes=[("JSON地图文件", "*.json"), ("所有文件", "*.*")],
            initialfilename=f"{map_name}.json"
        )
        
        if file_path:
            self.current_file_path = file_path
            map_name = os.path.basename(file_path).split('.')[0]
            self.name_entry.delete(0, tk.END)
            self.name_entry.insert(0, map_name)
            self.save_map()

    # 以下是原有的方法，保持不变但添加了一些改进
    def on_canvas_click(self, event):
        """处理画布点击事件"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.last_painted_cell = (row, col)
            self.is_drawing = True
            self.apply_tool(row, col)

    def on_canvas_release(self, event):
        """处理鼠标释放事件"""
        self.last_painted_cell = None
        self.is_drawing = False
        # 释放时更新画布滚动区域和地图信息
        self.update_canvas_scroll_region()
        self.update_map_info()

    def update_canvas_scroll_region(self):
        """更新画布滚动区域 - 自动适应地图大小"""
        canvas_width = self.cols * self.cell_size
        canvas_height = self.rows * self.cell_size
        self.canvas.configure(scrollregion=(0, 0, canvas_width, canvas_height))

    def init_map(self):
        """初始化地图"""
        self.draw_grid()
        self.update_pair_list()
        self.update_map_info()
        self.set_status("地图已初始化")

    def draw_grid(self):
        """绘制网格和所有元素"""
        self.canvas.delete("all")
        
        # 计算画布大小并设置滚动区域 - 自动更新
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
        
        # 绘制起点
        for point in self.start_points:
            x, y = point["x"], point["y"]
            x_center = x * self.cell_size + self.cell_size/2
            y_center = y * self.cell_size + self.cell_size/2
            
            # 绘制绿色圆形
            radius = self.cell_size * 0.4
            self.canvas.create_oval(
                x_center - radius, y_center - radius,
                x_center + radius, y_center + radius,
                fill="green", outline="darkgreen", width=2
            )
            
            # 添加ID标签
            self.canvas.create_text(
                x_center, y_center,
                text=str(point["id"]),
                fill="white", font=("Arial", 8, "bold")
            )
        
        # 绘制终点
        for point in self.end_points:
            x, y = point["x"], point["y"]
            x_center = x * self.cell_size + self.cell_size/2
            y_center = y * self.cell_size + self.cell_size/2
            
            # 绘制红色矩形
            radius = self.cell_size * 0.4
            self.canvas.create_rectangle(
                x_center - radius, y_center - radius,
                x_center + radius, y_center + radius,
                fill="red", outline="darkred", width=2
            )
            
            # 添加ID标签
            self.canvas.create_text(
                x_center, y_center,
                text=str(point["id"]),
                fill="white", font=("Arial", 8, "bold")
            )
        
        # 绘制配对连线
        for pair in self.point_pairs:
            start_point = next((p for p in self.start_points if p["id"] == pair["start_id"]), None)
            end_point = next((p for p in self.end_points if p["id"] == pair["end_id"]), None)
            
            if start_point and end_point:
                x1 = start_point["x"] * self.cell_size + self.cell_size/2
                y1 = start_point["y"] * self.cell_size + self.cell_size/2
                x2 = end_point["x"] * self.cell_size + self.cell_size/2
                y2 = end_point["y"] * self.cell_size + self.cell_size/2
                
                self.canvas.create_line(
                    x1, y1, x2, y2,
                    fill="blue", width=2, dash=(5, 5), arrow=tk.LAST
                )

    def update_current_tool(self):
        """更新当前工具"""
        self.current_tool = self.tool_var.get()
        self.set_status(f"当前工具: {self.current_tool}")

    def update_brush_size(self, value):
        """更新画笔大小"""
        self.brush_size = int(value)
        self.last_painted_cell = None

    def on_canvas_drag(self, event):
        """处理画布拖动事件"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= col < self.cols and 0 <= row < self.rows and (row, col) != self.last_painted_cell:
            self.last_painted_cell = (row, col)
            # 只有障碍物和可通行工具支持拖动
            if self.current_tool in ["obstacle", "passable"]:
                self.apply_tool(row, col)

    def on_right_click(self, event):
        """右键删除点位"""
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        
        col = int(canvas_x // self.cell_size)
        row = int(canvas_y // self.cell_size)
        
        if 0 <= col < self.cols and 0 <= row < self.rows:
            self.remove_point_at_position(col, row)

    def apply_tool(self, row, col):
        """应用当前工具到指定位置"""
        if self.current_tool == "obstacle":
            self.paint_obstacle(row, col)
        elif self.current_tool == "passable":
            self.clear_obstacle(row, col)
        elif self.current_tool == "start_point":
            self.place_start_point(row, col)
        elif self.current_tool == "end_point":
            self.place_end_point(row, col)

    def paint_obstacle(self, center_row, center_col):
        """绘制障碍物"""
        radius = self.brush_size
        
        # 将圆形区域内的点设为障碍物
        for r in range(max(0, center_row - radius), min(self.rows, center_row + radius + 1)):
            for c in range(max(0, center_col - radius), min(self.cols, center_col + radius + 1)):
                if ((r - center_row) ** 2 + (c - center_col) ** 2) <= radius ** 2:
                    if self.grid[r, c] != 1:  # 只有当状态改变时才更新
                        self.grid[r, c] = 1
                        self.remove_point_at_position(c, r)
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

    def place_start_point(self, row, col):
        """放置起点"""
        # 移除该位置的任何点位
        self.remove_point_at_position(col, row)
        
        # 确保该位置可通行
        self.grid[row, col] = 0
        
        # 添加起点
        self.start_points.append({
            "id": self.next_point_id,
            "x": col,
            "y": row
        })
        
        self.next_point_id += 1
        self.is_drawing = False  # 点位绘制结束
        self.draw_grid()
        self.update_pair_list()

    def place_end_point(self, row, col):
        """放置终点"""
        # 移除该位置的任何点位
        self.remove_point_at_position(col, row)
        
        # 确保该位置可通行
        self.grid[row, col] = 0
        
        # 添加终点
        self.end_points.append({
            "id": self.next_point_id,
            "x": col,
            "y": row
        })
        
        self.next_point_id += 1
        self.is_drawing = False  # 点位绘制结束
        self.draw_grid()
        self.update_pair_list()

    def remove_point_at_position(self, x, y):
        """移除指定位置的点位"""
        # 移除起点
        original_start_count = len(self.start_points)
        self.start_points = [p for p in self.start_points if not (p["x"] == x and p["y"] == y)]
        
        # 移除终点
        original_end_count = len(self.end_points)
        self.end_points = [p for p in self.end_points if not (p["x"] == x and p["y"] == y)]
        
        # 如果有点位被移除，需要移除相关配对
        if len(self.start_points) < original_start_count or len(self.end_points) < original_end_count:
            # 获取所有有效的ID
            valid_start_ids = {p["id"] for p in self.start_points}
            valid_end_ids = {p["id"] for p in self.end_points}
            
            # 移除无效配对
            self.point_pairs = [
                pair for pair in self.point_pairs 
                if pair["start_id"] in valid_start_ids and pair["end_id"] in valid_end_ids
            ]
            
            # 只有在实际移除了点位时才重绘和更新列表
            if not self.is_drawing:  # 避免在绘制障碍物时重复重绘
                self.draw_grid()
                self.update_pair_list()

    def auto_pair_points(self):
        """自动配对起点和终点"""
        self.point_pairs = []
        
        min_count = min(len(self.start_points), len(self.end_points))
        
        for i in range(min_count):
            self.point_pairs.append({
                "start_id": self.start_points[i]["id"],
                "end_id": self.end_points[i]["id"]
            })
        
        self.draw_grid()
        self.update_pair_list()
        self.update_map_info()
        self.set_status(f"已自动配对 {min_count} 对起点终点")

    def manual_pair_points(self):
        """手动配对起点和终点"""
        try:
            start_id = int(self.start_id_entry.get())
            end_id = int(self.end_id_entry.get())
            
            # 检查ID是否存在
            start_exists = any(p["id"] == start_id for p in self.start_points)
            end_exists = any(p["id"] == end_id for p in self.end_points)
            
            if not start_exists:
                messagebox.showerror("错误", f"起点ID {start_id} 不存在")
                return
            
            if not end_exists:
                messagebox.showerror("错误", f"终点ID {end_id} 不存在")
                return
            
            # 检查是否已经配对
            pair_exists = any(
                pair["start_id"] == start_id and pair["end_id"] == end_id 
                for pair in self.point_pairs
            )
            
            if pair_exists:
                messagebox.showwarning("警告", "该配对已存在")
                return
            
            # 添加配对
            self.point_pairs.append({
                "start_id": start_id,
                "end_id": end_id
            })
            
            # 清空输入框
            self.start_id_entry.delete(0, tk.END)
            self.end_id_entry.delete(0, tk.END)
            
            self.draw_grid()
            self.update_pair_list()
            self.update_map_info()
            self.set_status(f"已添加配对: 起点{start_id} -> 终点{end_id}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值ID")

    def clear_pairs(self):
        """清除所有配对"""
        self.point_pairs = []
        self.draw_grid()
        self.update_pair_list()
        self.update_map_info()
        self.set_status("已清除所有配对")

    def update_pair_list(self):
        """更新配对列表显示"""
        self.pair_listbox.delete(0, tk.END)
        
        for pair in self.point_pairs:
            self.pair_listbox.insert(tk.END, f"起点{pair['start_id']} -> 终点{pair['end_id']}")

    def update_map_size(self):
        """更新地图大小"""
        try:
            new_cols = int(self.col_entry.get())
            new_rows = int(self.row_entry.get())
            
            if new_rows <= 0 or new_cols <= 0:
                messagebox.showerror("错误", "宽度和高度必须大于0")
                return
            
            # 创建新网格
            new_grid = np.zeros((new_rows, new_cols), dtype=np.int8)
            
            # 复制现有网格数据
            min_rows = min(self.rows, new_rows)
            min_cols = min(self.cols, new_cols)
            new_grid[:min_rows, :min_cols] = self.grid[:min_rows, :min_cols]
            
            # 过滤超出新范围的点位
            self.start_points = [p for p in self.start_points if p["x"] < new_cols and p["y"] < new_rows]
            self.end_points = [p for p in self.end_points if p["x"] < new_cols and p["y"] < new_rows]
            
            # 移除无效的配对
            valid_start_ids = {p["id"] for p in self.start_points}
            valid_end_ids = {p["id"] for p in self.end_points}
            self.point_pairs = [
                pair for pair in self.point_pairs 
                if pair["start_id"] in valid_start_ids and pair["end_id"] in valid_end_ids
            ]
            
            # 更新网格
            self.grid = new_grid
            self.rows = new_rows
            self.cols = new_cols
            
            # 重绘并自动更新滚动区域
            self.draw_grid()
            self.update_pair_list()
            self.update_map_info()
            self.set_status(f"地图大小已更新为 {new_cols}x{new_rows}")
            
        except ValueError:
            messagebox.showerror("错误", "请输入有效的数值")

    def save_map(self):
        """保存地图"""
        map_name = self.name_entry.get()
        if not map_name:
            messagebox.showerror("错误", "请输入地图名称")
            return
        
        # 创建地图数据
        map_data = {
            "map_info": {
                "name": map_name,
                "width": self.cols,
                "height": self.rows,
                "cell_size": self.cell_size,
                "created_with": "EnhancedMapCreator",
                "version": "1.0"
            },
            "grid": self.grid.tolist(),
            "start_points": self.start_points,
            "end_points": self.end_points,
            "point_pairs": self.point_pairs,
            "obstacles": self.convert_grid_to_obstacles(),
            "statistics": {
                "total_cells": self.rows * self.cols,
                "obstacle_cells": int(np.sum(self.grid == 1)),
                "start_points_count": len(self.start_points),
                "end_points_count": len(self.end_points),
                "pairs_count": len(self.point_pairs)
            }
        }
        
        # 确定保存路径
        if self.current_file_path:
            filename = self.current_file_path
        else:
            filename = f"{map_name}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(map_data, f, indent=2, ensure_ascii=False)
            
            self.current_file_path = filename
            self.add_to_recent_files(filename)
            
            messagebox.showinfo("保存成功", f"地图已保存为: {os.path.basename(filename)}")
            self.set_status(f"地图已保存: {os.path.basename(filename)}")
            
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {str(e)}")

    def convert_grid_to_obstacles(self):
        """将网格转换为障碍物列表"""
        obstacles = []
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row, col] == 1:
                    obstacles.append({
                        "x": col,
                        "y": row
                    })
        return obstacles

    def clear_map(self):
        """清空地图"""
        reply = messagebox.askyesno("确认", "确定要清空地图吗？这将删除所有障碍物和点位。")
        if reply:
            self.grid = np.zeros((self.rows, self.cols), dtype=np.int8)
            self.start_points = []
            self.end_points = []
            self.point_pairs = []
            self.next_point_id = 1
            self.current_file_path = None
            self.draw_grid()
            self.update_pair_list()
            self.update_map_info()
            self.import_status.config(text="拖拽JSON文件到此处或点击浏览", fg="gray")
            self.title("增强版地图创建工具")
            self.set_status("地图已清空")

    def on_mousewheel(self, event):
        """鼠标滚轮缩放"""
        if event.num == 4 or event.delta > 0:  # 向上滚动
            self.cell_size = min(20, self.cell_size + 1)
        elif event.num == 5 or event.delta < 0:  # 向下滚动
            self.cell_size = max(3, self.cell_size - 1)
        
        self.draw_grid()

    def set_status(self, message):
        """设置状态栏消息"""
        self.status_label.config(text=message)

if __name__ == "__main__":
    app = EnhancedMapCreator()
    app.mainloop()