import sys
import os
import subprocess
import time
import re
import math
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QLabel, QTextEdit, QGraphicsView, QGraphicsScene, 
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsPathItem, QGraphicsItem,
    QMenu, QLineEdit
)
from PyQt6.QtCore import Qt, QPointF, QLineF, QTimer
from PyQt6.QtGui import QColor, QPalette, QBrush, QPen, QPolygonF, QFont, QPainter, QPainterPath

# Use the directory where scheduler.py is located for relative paths
base_path = os.path.dirname(os.path.abspath(__file__))

# Standard RELION job types to search through
JOB_TYPES = [
    "Import", "Select", "MaskCreate", "ModelAngelo", "MotionCorr", "Extract", "Class3D", 
    "PostProcess", "External", "DynaMight", "Refine3D", "CtfRefine", "Substract", "LocalRes", 
    "CtfFind", "ManualPick", "AutoPick", "InitialModel", "Polish", "JoinStar"
]

def update_pipeline_status(job_name):
    """
    Safely removes the 'Scheduled' tag using RELION's native .relion_lock system.
    Strictly NO files are automatically deleted except the temporary lock this function creates.
    """
    messages = []
    old_status = " Scheduled "
    new_status = " Running " 

    def safe_edit_star_file(file_path, lock_path):
        if not os.path.exists(file_path):
            return False
            
        # 1. Wait for the traffic light (if GUI is currently writing)
        timeout = 0
        while os.path.exists(lock_path):
            time.sleep(0.5)
            timeout += 0.5
            if timeout > 5.0:
                print(f"Timeout: {lock_path} is stuck. Please manually delete it.")
                return False

        # 2. Turn the traffic light red (Claim the lock)
        try:
            os.mkdir(lock_path)
        except FileExistsError:
            return False

        # 3. Safely edit the file
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            with open(file_path, 'w') as f:
                for line in lines:
                    if f"/{job_name}/" in line and old_status in line:
                        line = line.replace(old_status, new_status)
                    f.write(line)
        finally:
            # 4. Turn the traffic light green (Release our own lock)
            try:
                os.rmdir(lock_path)
            except OSError:
                pass
                
        return True

    # Update Master Pipeline
    if safe_edit_star_file("default_pipeline.star", ".relion_lock"):
        messages.append("Updated master pipeline")

    # Update Local Pipeline
    found_job_dir = None
    for jtype in JOB_TYPES:
        potential_dir = os.path.join(jtype, job_name)
        if os.path.isdir(potential_dir):
            found_job_dir = potential_dir
            break
            
    if found_job_dir:
        local_pipe = os.path.join(found_job_dir, "default_pipeline.star")
        local_lock = os.path.join(found_job_dir, ".relion_lock")
        if safe_edit_star_file(local_pipe, local_lock):
            messages.append(f"Updated local {job_name} pipeline")

    return " | ".join(messages) if messages else "No pipeline files updated."


class EdgeItem(QGraphicsPathItem):
    def __init__(self, source, target):
        super().__init__()
        self.source = source
        self.target = target
        
        self.source.edges.append(self)
        self.target.edges.append(self)
        
        self.setPen(QPen(QColor(255, 255, 255), 2))
        self.setZValue(-1) # Keep lines behind nodes
        self.arrow_head = QPolygonF()
        self.p2 = QPointF()
        self.ctrl2 = QPointF()
        self.adjust()

    def adjust(self):
        if not self.source or not self.target:
            return
            
        # Get centers of both nodes
        source_center = self.source.scenePos() + QPointF(60, 30)
        target_center = self.target.scenePos() + QPointF(60, 30)
        
        # Calculate intersection points with the 120x60 rectangles
        def get_intersect(node_pos, other_center):
            rect = self.source.rect().translated(node_pos)
            center = rect.center()
            line_to_other = QLineF(center, other_center)
            
            intersect_point = center
            edges = [
                QLineF(rect.topLeft(), rect.topRight()),
                QLineF(rect.bottomLeft(), rect.bottomRight()),
                QLineF(rect.topLeft(), rect.bottomLeft()),
                QLineF(rect.topRight(), rect.bottomRight())
            ]
            for edge in edges:
                intersection_type, p = line_to_other.intersects(edge)
                if intersection_type == QLineF.IntersectionType.BoundedIntersection:
                    return p
            return center

        p1 = get_intersect(self.source.scenePos(), target_center)
        p2 = get_intersect(self.target.scenePos(), source_center)
        
        # Calculate Bezier control points for a smooth horizontal S-curve
        dist_x = p2.x() - p1.x()
        
        ctrl1 = QPointF(p1.x() + dist_x * 0.5, p1.y())
        ctrl2 = QPointF(p2.x() - dist_x * 0.5, p2.y())
        
        # Store these points to calculate the arrowhead tangent later
        self.p2 = p2
        self.ctrl2 = ctrl2
        
        path = QPainterPath(p1)
        path.cubicTo(ctrl1, ctrl2, p2)
        self.setPath(path)

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        
        if self.path().isEmpty():
            return
            
        # Find the midpoint of the curve (t = 0.5)
        mid_point = self.path().pointAtPercent(0.5)
        
        # Sample points slightly before and after the midpoint to get the tangent direction
        p_before = self.path().pointAtPercent(0.49)
        p_after = self.path().pointAtPercent(0.51)
        
        dy = p_after.y() - p_before.y()
        dx = p_after.x() - p_before.x()
        
        # Fallback if the nodes are perfectly overlapping to avoid math domain errors
        if abs(dx) < 0.1 and abs(dy) < 0.1:
            return
            
        angle = math.atan2(-dy, dx)
        arrow_size = 12
        
        # Calculate arrow points relative to the midpoint
        dest_p1 = mid_point - QPointF(math.cos(angle - math.pi / 6) * arrow_size,
                                      -math.sin(angle - math.pi / 6) * arrow_size)
        dest_p2 = mid_point - QPointF(math.cos(angle + math.pi / 6) * arrow_size,
                                      -math.sin(angle + math.pi / 6) * arrow_size)
                                      
        painter.setBrush(QColor(255, 255, 255))
        painter.drawPolygon(QPolygonF([mid_point, dest_p1, dest_p2]))


class NodeItem(QGraphicsRectItem):
    def __init__(self, job_name, job_type, alias, path, main_window, is_running=False, existing_slurm_id=None):
        super().__init__(0, 0, 120, 60)
        self.job_name = job_name
        self.job_type = job_type
        self.alias = alias
        self.path = path
        self.main_window = main_window
        self.edges = []
        self.slurm_id = existing_slurm_id
        self.is_running = is_running
        
        # Styling
        self.dash_offset = 0
        if self.is_running:
            self.base_color = QColor(180, 140, 20) # Yellow-orange for active jobs
            self.pen_color = QColor(255, 215, 0)
            self.current_pen = QPen(self.pen_color, 2, Qt.PenStyle.DashLine)
            
            # Setup the animation timer for running jobs
            self.anim_timer = QTimer()
            self.anim_timer.timeout.connect(self.animate_border)
            self.anim_timer.start(50) # Update outline every 50ms
        else:
            self.base_color = QColor(62, 62, 62)
            self.pen_color = QColor(152, 195, 121)
            self.current_pen = QPen(self.pen_color, 2)
            
        self.setBrush(QBrush(self.base_color))
        self.setPen(self.current_pen)
        
        # Flags
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        
        # Text Label
        display_text = f"{self.job_name}\n{self.job_type}"
        if self.alias:
            display_text += f"\n{self.alias}"
            
        self.text_item = QGraphicsTextItem(display_text, self)
        self.text_item.setDefaultTextColor(QColor(255, 255, 255))
        font = QFont("Arial", 9, QFont.Weight.Bold)
        self.text_item.setFont(font)
        
        # Center-align multiline text
        doc = self.text_item.document()
        from PyQt6.QtGui import QTextOption
        opt = QTextOption()
        opt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        doc.setDefaultTextOption(opt)
        
        # Force the text width to match the box width (120) so AlignCenter works perfectly
        self.text_item.setTextWidth(120)
        
        # Center text vertically; horizontally it's handled by setTextWidth
        text_rect = self.text_item.boundingRect()
        self.text_item.setPos(0, 30 - text_rect.height() / 2)

    def animate_border(self):
        self.dash_offset -= 1  # Subtracting makes the dashes flow around the box like a marquee
        self.current_pen.setDashOffset(self.dash_offset)
        self.setPen(self.current_pen)
        self.update()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            for edge in self.edges:
                edge.adjust()
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            if self.isSelected():
                highlight_color = QColor(255, 215, 0) if self.is_running else QColor(152, 195, 121)
                self.setBrush(QBrush(highlight_color)) 
                self.text_item.setDefaultTextColor(QColor(43, 43, 43))
            else:
                self.setBrush(QBrush(self.base_color))
                self.text_item.setDefaultTextColor(QColor(255, 255, 255))
                
        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        # Handle linking mechanism override
        if self.main_window.linking_mode:
            self.main_window.complete_linking(self)
            event.accept()
        else:
            super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        # Prevent context menu from popping up if we are currently linking
        if self.main_window.linking_mode:
            return
            
        menu = QMenu()
        action_link_new = menu.addAction("Link to new job")
        action_parent = menu.addAction("Link to (as parent)")
        action_child = menu.addAction("Link to (as child)")
        
        if self.is_running:
            action_child.setEnabled(False)
            action_child.setText("Link to (as child) [DISABLED - Running]")
            
        menu.addSeparator()
        action_break = menu.addAction("Break links")
        action_remove = menu.addAction("Remove")
        
        # Apply theme to context menu
        menu.setStyleSheet("""
            QMenu { background-color: #2b2b2b; color: white; border: 1px solid #555; }
            QMenu::item:selected { background-color: #98c379; color: black; }
        """)

        selected_action = menu.exec(event.screenPos())
        
        if selected_action == action_parent:
            self.main_window.start_linking(self, "parent")
        elif selected_action == action_child:
            self.main_window.start_linking(self, "child")
        elif selected_action == action_link_new:
            self.main_window.quick_add_and_link(self)
        elif selected_action == action_break:
            self.main_window.break_links(self)
        elif selected_action == action_remove:
            self.main_window.remove_node(self)
        elif selected_action == action_remove:
            self.main_window.remove_node(self)


from PyQt6.QtWidgets import QInputDialog

class CustomGraphicsView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        self._right_pressed = False
        self._last_mouse_pos = None
        self._pan_distance = 0
        
        # --- Add Reset View Button ---
        self.reset_btn = QPushButton("Reset View", self)
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.setStyleSheet("""
            QPushButton { 
                background-color: #3e3e3e; 
                color: #ffffff; 
                border: 1px solid #555; 
                border-radius: 4px; 
                padding: 6px 12px; 
                font-weight: bold;
            }
            QPushButton:hover { 
                background-color: #555; 
                border: 1px solid #98c379; 
            }
            QPushButton:pressed {
                background-color: #98c379;
                color: #2b2b2b;
            }
        """)
        self.reset_btn.clicked.connect(self.reset_view)
        
        # --- Add Instructions Overlay ---
        current_project = os.path.basename(os.path.abspath(os.getcwd()))
        
        self.instructions = QLabel(
            f"<b>Project:</b> {current_project}<br><br>"
            "<b>- Set 'Submit to Queue' for All Jobs</b><br>"
            "<br>"
            "<b>Right-Click (Drag)</b>: Pan Canvas<br>"
            "<b>Scroll</b>: Zoom<br>"
            "<b>Right-Click (Node)</b>: Link/Unlink<br>"
            "<b>Right-Click (Empty)</b>: Add Job", self
        )
        self.instructions.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.instructions.setStyleSheet("""
            QLabel {
                background-color: rgba(30, 30, 30, 180);
                color: #aaaaaa;
                border: 1px solid #555;
                border-radius: 6px;
                padding: 10px;
                font-size: 10pt;
            }
        """)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Keep the button anchored to the bottom-right corner
        btn_size = self.reset_btn.sizeHint()
        margin = 15
        x = self.viewport().width() - btn_size.width() - margin
        y = self.viewport().height() - btn_size.height() - margin
        self.reset_btn.move(int(x), int(y))
        
        # Keep instructions anchored to the top-right corner
        inst_size = self.instructions.sizeHint()
        inst_x = self.viewport().width() - inst_size.width() - margin
        inst_y = margin
        self.instructions.move(int(inst_x), int(inst_y))

    def reset_view(self):
        main_win = self.window()
        # If there are no nodes, just snap back to the origin
        if not hasattr(main_win, 'nodes') or not main_win.nodes:
            self.centerOn(0, 0)
            return
        
        # Calculate the center of mass (average of all node positions)
        total_x = 0
        total_y = 0
        count = len(main_win.nodes)
        
        for node in main_win.nodes.values():
            pos = node.scenePos()
            # Offset by half the node size (120x60) to get the true center of the node
            total_x += pos.x() + 60
            total_y += pos.y() + 30
            
        center_x = total_x / count
        center_y = total_y / count
        
        self.centerOn(QPointF(center_x, center_y))

    def wheelEvent(self, event):
        import sys
        
        # Determine the OS to set the correct zoom modifier key
        is_mac = sys.platform == 'darwin'
        zoom_modifier = Qt.KeyboardModifier.AltModifier if is_mac else Qt.KeyboardModifier.ControlModifier
        
        p_delta = event.pixelDelta()
        a_delta = event.angleDelta()
        
        # Heuristic: High-precision touchpads send pixelDelta. Standard mice usually don't.
        is_touchpad = not p_delta.isNull()
        
        # We want to zoom if it's a standard mouse wheel OR if the user holds the modifier on a touchpad
        should_zoom = (not is_touchpad) or (event.modifiers() & zoom_modifier)
        
        if should_zoom:
            # --- ZOOM LOGIC ---
            delta = a_delta.y()
            if delta == 0:
                return

            zoom_factor = math.pow(1.15, delta / 120.0)
            current_scale = self.transform().m11()
            
            min_scale = 0.1   # Maximum zoom out (10%)
            max_scale = 10.0  # Maximum zoom in (1000%)

            new_scale = current_scale * zoom_factor
            if new_scale < min_scale:
                zoom_factor = min_scale / current_scale
            elif new_scale > max_scale:
                zoom_factor = max_scale / current_scale

            self.scale(zoom_factor, zoom_factor)
            
        else:
            # --- PAN LOGIC (Touchpad without modifier) ---
            dx = p_delta.x()
            dy = p_delta.y()
                
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            
            # Subtracting delta pushes the scrollbar in the correct natural scrolling direction
            h_bar.setValue(h_bar.value() - dx)
            v_bar.setValue(v_bar.value() - dy)
            
        event.accept()

    def mousePressEvent(self, event):
        main_win = self.window()
        
        # 1. Cancel linking mode if right-clicking
        if hasattr(main_win, 'linking_mode') and main_win.linking_mode:
            if event.button() == Qt.MouseButton.RightButton:
                main_win.cancel_linking()
                main_win.log("Linking operation canceled.")
                event.accept()
                return

        # 2. Start manual panning on right click in empty space
        if event.button() == Qt.MouseButton.RightButton and not self.itemAt(event.pos()):
            self._right_pressed = True
            self._last_mouse_pos = event.pos()
            self._pan_distance = 0
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
            
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._right_pressed:
            delta = event.pos() - self._last_mouse_pos
            self._pan_distance += delta.manhattanLength()
            self._last_mouse_pos = event.pos()
            
            # Manually adjust scrollbars (identical to workflow.py)
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and self._right_pressed:
            self._right_pressed = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
            # If the mouse barely moved, treat it as a click to add a job
            if self._pan_distance < 5:
                scene_pos = self.mapToScene(event.pos())
                main_win = self.window()
                if hasattr(main_win, 'context_add_job'):
                    # Use QTimer to defer the dialog. This lets Qt finish processing 
                    # the right-click release and its associated context menu event 
                    # on the empty canvas BEFORE the new node is spawned.
                    from PyQt6.QtCore import QTimer
                    QTimer.singleShot(0, lambda: main_win.context_add_job(None, scene_pos))
                
            event.accept()
            return
            
        super().mouseReleaseEvent(event)


class RelionExecutorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RELION Job Scheduler")
        self.resize(1000, 800)
        
        self.nodes = {} 
        self.edges = [] 
        
        # Linking state machine
        self.linking_mode = None  # None, "parent", or "child"
        self.linking_node = None
        
        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # --- Top Button Frame ---
        top_layout = QHBoxLayout()
        
        self.job_input = QLineEdit()
        self.job_input.setFocusPolicy(Qt.FocusPolicy.ClickFocus) # Only take focus when clicked
        self.job_input.setPlaceholderText("1,2,3...")
        self.job_input.setFixedWidth(150)
        self.job_input.returnPressed.connect(self.add_manual_job)
        top_layout.addWidget(QLabel("Add Scheduled Jobs:"))
        top_layout.addWidget(self.job_input)
        
        btn_add = QPushButton("Add to Canvas")
        btn_add.clicked.connect(self.add_manual_job)
        top_layout.addWidget(btn_add)
        
        btn_clear = QPushButton("Clear Canvas")
        btn_clear.clicked.connect(self.clear_canvas)
        top_layout.addWidget(btn_clear)
        
        instruction_label = QLabel("Right-Click on jobs to Link/Unlink")
        instruction_label.setStyleSheet("color: #777;")
        top_layout.addWidget(instruction_label)
        
        top_layout.addStretch()
        
        self.btn_exec = QPushButton("Execute Workflow")
        self.btn_exec.setStyleSheet("""
            QPushButton { background-color: #98c379; color: #2b2b2b; border: 1px solid #98c379; }
            QPushButton:hover { background-color: #a9d48a; }
        """)
        self.btn_exec.clicked.connect(self.execute_workflow)
        top_layout.addWidget(self.btn_exec)
        
        main_layout.addLayout(top_layout)

        # --- Canvas Area ---
        self.scene = QGraphicsScene()
        # Force a massive bounding box to simulate an infinite canvas
        self.scene.setSceneRect(-50000, -50000, 100000, 100000) 
        
        self.view = CustomGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing) # Antialiasing
        self.view.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444;")
        self.view.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        main_layout.addWidget(self.view, stretch=3)

        # --- Terminal Log Area ---
        log_label = QLabel("Terminal Log:")
        log_label.setStyleSheet("font-weight: bold; color: #98c379;")
        main_layout.addWidget(log_label)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("background-color: black; color: #98c379; font-family: Consolas; font-size: 10pt; border: 1px solid #444;")
        main_layout.addWidget(self.log_area, stretch=1)
        self.view.setFocus() # Ensure canvas has focus on startup
        
        # Auto-load scheduled jobs AFTER the main window has rendered
        # A short delay ensures the UI is fully visible before any blocking popup appears.
        QTimer.singleShot(100, self.auto_load_scheduled_jobs)

    def auto_load_scheduled_jobs(self):
        """Automatically parses default_pipeline.star and loads 'Scheduled' jobs."""
        if not os.path.exists("default_pipeline.star"):
            self.log("No default_pipeline.star found. Skipping auto-load.")
            return
            
        scheduled_jobs = []
        try:
            with open("default_pipeline.star", 'r') as f:
                for line in f:
                    if " Scheduled " in line:
                        match = re.search(r'(job\d+)', line)
                        if match:
                            job_name = match.group(1)
                            if job_name not in scheduled_jobs:
                                scheduled_jobs.append(job_name)
        except Exception as e:
            self.log(f"Error reading default_pipeline.star: {e}")
            return
            
        if not scheduled_jobs:
            self.log("No 'Scheduled' jobs found to auto-load.")
            return
            
        self.log(f"Auto-loading {len(scheduled_jobs)} scheduled jobs...")
        
        # Calculate starting position for the layout
        viewport_rect = self.view.viewport().rect()
        scene_center = self.view.mapToScene(viewport_rect.center())
        base_pos = scene_center - QPointF(60, 30)
        
        added_count = 0
        missing_script_jobs = []
        
        for job_name in scheduled_jobs:
            # Layout in a grid (wrap to a new row every 4 jobs)
            x_offset = (added_count % 4) * 180
            y_offset = (added_count // 4) * 120
            node_pos = base_pos + QPointF(x_offset, y_offset)
            
            # Pre-check if run_submit.script exists for the popup aggregation
            has_script = False
            for jtype in JOB_TYPES:
                job_dir = os.path.join(jtype, job_name)
                if os.path.isdir(job_dir):
                    if os.path.exists(os.path.join(job_dir, "run_submit.script")):
                        has_script = True
                    break
                    
            if not has_script:
                missing_script_jobs.append(job_name)
            
            node = self.create_job_node(job_name, node_pos, show_warning=False)
            if node:
                added_count += 1
                
        if missing_script_jobs:
            from PyQt6.QtWidgets import QMessageBox
            msg = "Scheduled jobs not submitted to queue:\n"
            for missing_job in missing_script_jobs:
                msg += f"{missing_job}\n"
            msg += "\nChoose \"Yes\" for \"Submit to Queue\" in RELION if you want to work on these jobs."
            QMessageBox.warning(self, "Missing Queue Scripts", msg)
                
        # Snap the camera to the newly added nodes
        self.view.reset_view()

    def log(self, message):
        self.log_area.append(message)
        # Scroll to bottom
        scrollbar = self.log_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def get_active_slurm_jobs(self):
        """Scans SLURM for jobs belonging to the current directory."""
        import getpass
        username = getpass.getuser()
        active_jobs = {}
        try:
            # 1. Get all SLURM IDs for current user
            res = subprocess.run(["squeue", "-u", username, "-h", "-o", "%A"], capture_output=True, text=True)
            job_ids = [j.strip() for j in res.stdout.strip().split() if j.strip()]
            if not job_ids:
                return active_jobs
            
            # 2. Iterate through each job ID one by one (scontrol expects a single ID)
            stdout_data = ""
            for jid in job_ids:
                indiv_res = subprocess.run(["scontrol", "show", "job", jid], capture_output=True, text=True)
                if indiv_res.returncode == 0:
                    stdout_data += indiv_res.stdout + "\n"
                
            # 3. Parse blocks safely
            blocks = stdout_data.split("JobId=")
            current_project_dir = os.path.basename(os.path.abspath(os.getcwd()))
            
            for block in blocks:
                if not block.strip(): continue
                jid_match = re.search(r"^(\d+)", block)
                cmd_match = re.search(r"Command=(\S+)", block)
                
                if jid_match and cmd_match:
                    jid = jid_match.group(1)
                    cmd_path = cmd_match.group(1)
                    
                    # Split path to find job name and job type
                    parts = cmd_path.split(os.sep)
                    if len(parts) >= 3 and parts[-2].startswith("job"):
                        job_name = parts[-2]
                        jtype = parts[-3]
                        
                        # Fix for HPC environments: Avoid string 'startswith' on absolute paths because of 
                        # symlinks (e.g. /home vs /mnt/home). Resolve realpaths or match directory structure.
                        local_script = os.path.abspath(os.path.join(jtype, job_name, "run_submit.script"))
                        
                        is_match = False
                        try:
                            # Method A: Do they resolve to the exact same physical disk location?
                            if os.path.realpath(cmd_path) == os.path.realpath(local_script):
                                is_match = True
                            # Method B: Does the SLURM path contain our project folder structure?
                            elif f"/{current_project_dir}/{jtype}/{job_name}/" in cmd_path:
                                is_match = True
                        except Exception:
                            pass
                            
                        if is_match:
                            active_jobs[job_name] = jid
        except Exception as e:
            self.log(f"Warning: Failed to parse SLURM data. {e}")
            
        return active_jobs

    def clear_canvas(self):
        self.scene.clear()
        self.nodes.clear()
        self.edges.clear()
        self.cancel_linking()
        self.log("\nCanvas cleared.")

    def add_manual_job(self):
        """Finds job(s) and adds them to the center of the current viewport."""
        raw_input = self.job_input.text().strip()
        if not raw_input: return
        
        # Split by comma and strip whitespaces (e.g., "41, 42, 43" works too)
        job_entries = [j.strip() for j in raw_input.split(',')]
        
        # Calculate viewport center in scene coordinates
        viewport_rect = self.view.viewport().rect()
        scene_center = self.view.mapToScene(viewport_rect.center())
        base_pos = scene_center - QPointF(60, 30)
        
        failed_jobs = []
        missing_script_jobs = []
        added_count = 0
        
        for entry in job_entries:
            if not entry: continue
            
            # Auto-fill zeros if numeric (e.g., 1 -> 001)
            if entry.isdigit():
                formatted_entry = f"{int(entry):03d}"
            else:
                formatted_entry = entry
                
            # Get canonical job name to check directory
            clean_input = formatted_entry.lower().replace("job", "").strip()
            job_name = f"job{int(clean_input):03d}" if clean_input.isdigit() else formatted_entry
            
            # Check if job folder exists but is missing run_submit.script
            has_folder = False
            has_script = False
            for jtype in JOB_TYPES:
                job_dir = os.path.join(jtype, job_name)
                if os.path.isdir(job_dir):
                    has_folder = True
                    if os.path.exists(os.path.join(job_dir, "run_submit.script")):
                        has_script = True
                    break
            
            if has_folder and not has_script:
                missing_script_jobs.append(job_name)
                continue
                
            # Offset each subsequent node by 180px horizontally so they don't overlap completely
            node_pos = base_pos + QPointF(added_count * 180, 0)
            
            node = self.create_job_node(formatted_entry, node_pos, show_warning=False)
            if node is None:
                failed_jobs.append(formatted_entry)
            else:
                added_count += 1
                
        if missing_script_jobs:
            from PyQt6.QtWidgets import QMessageBox
            msg = f"Queue script not detected\n\nPlease schedule {', '.join(missing_script_jobs)} as \"Yes\" for \"Submit to Queue\""
            QMessageBox.warning(self, "Missing Queue Script", msg)
            
        if failed_jobs:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Jobs Failed", f"The following jobs failed to load or are not scheduled:\n{', '.join(failed_jobs)}")
            
        self.job_input.clear()

    def context_add_job(self, event, scene_pos):
        """Adds node at a specific scene position (from right-click)."""
        job_num, ok = QInputDialog.getText(self, "Add Job", "Enter Scheduled Job Number:", QLineEdit.EchoMode.Normal)
        if ok and job_num:
            # Auto-fill zeros if numeric (e.g., 1 -> 001)
            if job_num.strip().isdigit():
                job_num = f"{int(job_num):03d}"
            self.create_job_node(job_num, scene_pos - QPointF(60, 30))
            
        if event:
            event.accept() # Prevents the event from bubbling up to the context menu

    def quick_add_and_link(self, parent_node):
        """Adds a new node and automatically connects the parent node to it."""
        job_num, ok = QInputDialog.getText(self, "Link New Job", f"New job following {parent_node.job_name}:", QLineEdit.EchoMode.Normal)
        if ok and job_num:
            # Place new node 180 pixels to the right for landscape layout
            new_pos = parent_node.scenePos() + QPointF(180, 0)
            new_node = self.create_job_node(job_num, new_pos)
            if new_node:
                edge = EdgeItem(parent_node, new_node)
                self.scene.addItem(edge)
                self.edges.append(edge)

    def create_job_node(self, raw_input, pos, show_warning=True):
        """Helper to create node and add to scene."""
        # Standardize: "1" -> "job001", "001" -> "job001", "job1" -> "job001"
        clean_input = raw_input.lower().replace("job", "").strip()
        if clean_input.isdigit():
            job_name = f"job{int(clean_input):03d}"
        else:
            job_name = raw_input
        if job_name in self.nodes:
            self.log(f"Job {job_name} is already on the canvas.")
            return self.nodes[job_name]

        found_path = None
        has_folder = False
        found_jtype = ""
        
        for jtype in JOB_TYPES:
            job_dir = os.path.join(jtype, job_name)
            if os.path.isdir(job_dir):
                has_folder = True
                found_jtype = jtype
                potential_path = os.path.join(job_dir, "run_submit.script")
                if os.path.exists(potential_path):
                    found_path = potential_path
                break
                
        if has_folder and not found_path:
            if show_warning:
                from PyQt6.QtWidgets import QMessageBox
                msg = f"Queue script not detected\n\nPlease schedule {job_name} as \"Yes\" for \"Submit to Queue\""
                QMessageBox.warning(self, "Missing Queue Script", msg)
            self.log(f"Skipped {job_name}: Missing run_submit.script")
            return None

        if found_path:
            # Check status in default_pipeline.star ONLY for 'Scheduled' jobs
            is_scheduled = False
            alias = ""
            
            if os.path.exists("default_pipeline.star"):
                in_processes = False
                with open("default_pipeline.star", 'r') as f:
                    for line in f:
                        if line.startswith("data_pipeline_processes"):
                            in_processes = True
                        elif line.startswith("data_"):
                            in_processes = False
                            
                        if f"/{job_name}/" in line:
                            if " Scheduled " in line:
                                is_scheduled = True
                        
                        if in_processes and f"/{job_name}/" in line:
                            parts = line.strip().split()
                            if len(parts) >= 2:
                                raw_alias = parts[1]
                                if raw_alias and raw_alias != "None":
                                    # Filter out empty strings to handle trailing slashes (e.g. "my_alias/")
                                    alias_parts = [p for p in raw_alias.split('/') if p]
                                    if alias_parts:
                                        alias = alias_parts[-1]
                            
            # Verify Running/Pending status via SLURM instead of RELION's star file
            active_slurm_jobs = self.get_active_slurm_jobs()
            is_running = job_name in active_slurm_jobs
            existing_slurm_id = active_slurm_jobs.get(job_name)
            
            if not is_scheduled and not is_running:
                if show_warning:
                    from PyQt6.QtWidgets import QMessageBox
                    QMessageBox.warning(self, "Invalid Job", f"Job {job_name} is not 'Scheduled' in RELION pipeline and not active in SLURM.")
                self.log(f"Skipped {job_name}: Not 'Scheduled' in pipeline and not active in SLURM.")
                return None

            # We can now immediately pass the automatically detected SLURM ID
            node = NodeItem(job_name, found_jtype, alias, found_path, self, is_running=is_running, existing_slurm_id=existing_slurm_id)
            node.setPos(pos)
            self.scene.addItem(node)
            self.nodes[job_name] = node
            self.log(f"Added {job_name} to canvas.")
            return node
        else:
            self.log(f"Error: Could not find job directory for '{job_name}'")
            return None

    def remove_node(self, node):
        """Safely removes a node and all associated edges from the canvas."""
        self.break_links(node)
        if node.job_name in self.nodes:
            del self.nodes[node.job_name]
        self.scene.removeItem(node)
        self.log(f"Removed {node.job_name} from canvas.")

    def start_linking(self, node, direction):
        self.linking_mode = direction
        self.linking_node = node
        self.view.viewport().setCursor(Qt.CursorShape.CrossCursor)
        
        if direction == "parent":
            self.log(f"Linking: Select the NEXT job (child) that follows {node.job_name}. Right-click to cancel.")
        else:
            self.log(f"Linking: Select the PREVIOUS job (parent) that precedes {node.job_name}. Right-click to cancel.")

    def complete_linking(self, target_node):
        if target_node == self.linking_node:
            self.log("Error: Cannot link a node to itself.")
            self.cancel_linking()
            return

        # Determine the intended source and target
        if self.linking_mode == 'parent':
            new_source, new_target = self.linking_node, target_node
        else:
            new_source, new_target = target_node, self.linking_node

        if getattr(new_target, 'is_running', False):
            self.log(f"Error: {new_target.job_name} is already running. It cannot be linked as a child.")
            self.cancel_linking()
            return

        # Check for pre-existing links to prevent duplicates or replace reverse connections
        edges_to_remove = []
        for e in self.edges:
            if e.source == new_source and e.target == new_target:
                self.cancel_linking()
                return # Link already exists, abort
            if e.source == new_target and e.target == new_source:
                edges_to_remove.append(e) # Mark reverse link for removal

        # Remove any reverse connections found
        for e in edges_to_remove:
            if e in e.source.edges: e.source.edges.remove(e)
            if e in e.target.edges: e.target.edges.remove(e)
            self.scene.removeItem(e)
            self.edges.remove(e)
            self.log(f"Overwrote reverse connection: {e.source.job_name} -> {e.target.job_name}")

        # If the parent is already running, we need its SLURM ID for the child's dependency
        if getattr(new_source, 'is_running', False) and not new_source.slurm_id:
            self.log(f"Auto-detecting SLURM ID for {new_source.job_name}...")
            
            # Fetch it automatically using your built-in scontrol parser
            active_slurm_jobs = self.get_active_slurm_jobs()
            
            if new_source.job_name in active_slurm_jobs:
                new_source.slurm_id = active_slurm_jobs[new_source.job_name]
                self.log(f"Found it! {new_source.job_name} is running under SLURM ID: {new_source.slurm_id}")
            else:
                # Fallback just in case the job just finished or SLURM is lagging
                from PyQt6.QtWidgets import QInputDialog, QLineEdit
                sid, ok = QInputDialog.getText(self, "Running Job ID", f"{new_source.job_name} is active in RELION, but wasn't found in squeue.\nEnter its SLURM ID manually:")
                if ok and sid.strip().isdigit():
                    new_source.slurm_id = sid.strip()
                else:
                    self.log(f"Warning: No valid SLURM ID provided for {new_source.job_name}. The child will NOT wait for it to finish.")

        # Create the newly assigned connection
        edge = EdgeItem(new_source, new_target)
        self.log(f"Connected: {new_source.job_name} -> {new_target.job_name}")

        self.scene.addItem(edge)
        self.edges.append(edge)
        self.cancel_linking()

    def break_links(self, node):
        edges_to_remove = [e for e in node.edges]
        for e in edges_to_remove:
            if e in e.source.edges: e.source.edges.remove(e)
            if e in e.target.edges: e.target.edges.remove(e)
            self.scene.removeItem(e)
            if e in self.edges: self.edges.remove(e)
            
        self.log(f"Unlinked all connections for {node.job_name}")

    def cancel_linking(self):
        self.linking_mode = None
        self.linking_node = None
        self.view.viewport().setCursor(Qt.CursorShape.ArrowCursor)

    def execute_workflow(self):
        if not self.nodes: 
            return
            
        self.log("\n--- INITIATING WORKFLOW ---")
        
        unsubmitted = set()
        for job, node in self.nodes.items(): 
            if node.is_running:
                self.log(f"Skipping submission for {job}: Already active (SLURM ID: {node.slurm_id})")
            else:
                node.slurm_id = None 
                unsubmitted.add(job)

        while unsubmitted:
            ready_jobs = []
            for job in unsubmitted:
                node = self.nodes[job]
                # A parent is defined as the source of an edge where this node is the target
                parents = [e.source.job_name for e in node.edges if e.target == node]
                
                if all(p not in unsubmitted for p in parents):
                    ready_jobs.append(job)
            
            if not ready_jobs:
                self.log("ERROR: Cyclic dependency detected or workflow stuck. Aborting.")
                break
                
            for job in ready_jobs:
                node = self.nodes[job]
                parents = [e.source.job_name for e in node.edges if e.target == node]
                parent_slurm_ids = [self.nodes[p].slurm_id for p in parents if self.nodes[p].slurm_id]
                
                cmd = ["sbatch"]
                if parent_slurm_ids:
                    dep_str = ":".join(parent_slurm_ids)
                    cmd.append(f"--dependency=afterok:{dep_str}")
                cmd.append(node.path)
                
                self.log(f"Submitting {job}...")
                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                    if res.returncode == 0:
                        match = re.search(r"Submitted batch job (\d+)", res.stdout)
                        if match:
                            slurm_id = match.group(1)
                            node.slurm_id = slurm_id
                            self.log(f"{job} submitted! (SLURM ID: {slurm_id})")
                            update_pipeline_status(job)
                        unsubmitted.discard(job) # Use discard to avoid KeyErrors
                    else:
                        self.log(f"Failed to submit {job}:\n{res.stderr}")
                        self.log("Workflow aborted.")
                        return
                except Exception as e:
                    self.log(f"Python execution error on {job}: {e}")
                    return
            
            # Allow the UI to process log updates and prevent "Not Responding"
            QApplication.processEvents()
                    
        self.log("All jobs in the workflow have been successfully queued! You can safely close the program!")
        self.log("To refresh your RELION GUI, go to File -> Re-read pipeline")
        self.show_success_message()

    def show_success_message(self):
        from PyQt6.QtCore import QTimer
        
        # Create a text item (default is left-aligned, just like UE5)
        success_text = QGraphicsTextItem("Workflow Successfully Executed\nSafe to close")
        
        # UE5 print strings are usually light blue/cyan, but we will stick to your green theme here
        success_text.setDefaultTextColor(QColor(152, 195, 121)) 
        
        # Switch to Consolas font to give it that authentic engine/console "Print String" feel
        font = QFont("Consolas", 16, QFont.Weight.Bold)
        success_text.setFont(font)
        
        # Map the top-left of the viewport (X:20, Y:20 for a nice margin) to scene coordinates
        top_left_scene = self.view.mapToScene(20, 20)
        success_text.setPos(top_left_scene)
        
        # Keep it layered on top of all nodes and lines
        success_text.setZValue(100)
        
        self.scene.addItem(success_text)
        
        # Remove the text safely after 4000 ms (4 seconds)
        QTimer.singleShot(4000, lambda: self.scene.removeItem(success_text) if success_text in self.scene.items() else None)


def apply_global_theme(app):
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
    dark_palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
    dark_palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(25, 25, 25))
    dark_palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    dark_palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    dark_palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    dark_palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(120, 120, 120))
    app.setPalette(dark_palette)

    app.setStyleSheet("""
        QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid #555; }
        QMainWindow { background-color: #2b2b2b; }
        QPushButton { background-color: #3e3e3e; border: 1px solid #3e3e3e; border-radius: 5px; padding: 8px 12px; color: white; font-weight: bold; outline: none; }
        QPushButton:hover { background-color: #3e3e3e; border: 1px solid #98c379; }
        QPushButton:pressed { background-color: #98c379; border: 1px solid #98c379; color: #2b2b2b; }
        QPushButton:disabled { color: #777; border: 1px solid #333; }
        QLabel { color: #eee; }
        QLineEdit { border: 1px solid #555; background: #252525; selection-background-color: #98c379; }
        QLineEdit:focus { border: 1px solid #98c379; }
    """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    apply_global_theme(app)
    
    window = RelionExecutorApp()
    window.show()
    sys.exit(app.exec())