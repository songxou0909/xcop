import sys
import os
import json
import re
import uuid
import argparse
from collections import defaultdict

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QLineEdit, QListWidget, QListWidgetItem,
    QFrame, QMenu, QDialog, QMessageBox,
    QGraphicsView, QGraphicsScene, QGraphicsItem,
    QGraphicsLineItem, QTextEdit,
    QGraphicsObject, QCheckBox, QStyleFactory, QStyledItemDelegate, QAbstractItemView
)
from PyQt6.QtCore import (
    Qt, QRectF, QTimer, pyqtSignal, QLineF, QSize
)
from PyQt6.QtGui import (
    QBrush, QPen, QColor, QFont, QPainter, QCursor, QTransform, QPainterPath, QPalette, QFontMetrics
)

# --------------------------------------------------------------------- #
# GLOBAL SETTINGS & HELPERS
# --------------------------------------------------------------------- #

BASE_FONT_SIZE = 12
MINIMAP_SIZE = 290
SIDEBAR_WIDTH = 320
WINDOW_SIZE_W = 1100
WINDOW_SIZE_H = 800
NODE_W = 120
NODE_H = 60

FOLDERS = [
    "AutoPick", "Class2D", "Class3D", "CtfFind", "Extract",
    "Import", "JoinStar", "ManualPick", "MaskCreate",
    "MotionCorr", "Refine3D", "Select",
    "PostProcess", "CtfRefine", "Polish", "LocalRes",
    "ModelAngelo", "DynaMight", "Multibody", "InitialModel"
]

JOB_PATTERN = re.compile(r'job(\d{3,})')
ROOT_DIR = os.getcwd()
print(f"[workflow] ROOT_DIR (cwd) = {ROOT_DIR}")

def _parse_relion_root_star(root_dir: str):
    """
    Parses default_pipeline.star.
    Returns:
        edges: list of (child, parent) tuples <--- CHANGED TO MATCH 1.PY LEGACY FORMAT
        aliases: dict of {job_num: alias_name}
    """
    star_path = os.path.join(root_dir, "default_pipeline.star")
    
    # Fallback: check highest job folder if not in root
    if not os.path.exists(star_path):
        highest_job = -1
        highest_path = ""
        for folder in FOLDERS:
            fpath = os.path.join(root_dir, folder)
            if os.path.isdir(fpath):
                for entry in os.listdir(fpath):
                    if entry.startswith("job") and os.path.isdir(os.path.join(fpath, entry)):
                        try:
                            num = int(entry[3:])
                            if num > highest_job:
                                highest_job = num
                                highest_path = os.path.join(fpath, entry, "default_pipeline.star")
                        except: pass
        if highest_path and os.path.exists(highest_path):
            star_path = highest_path

    if not os.path.exists(star_path):
        return [], {}

    edges = []
    aliases = {}
    
    in_processes = False
    in_input_edges = False

    try:
        with open(star_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue

                if line.startswith("data_"):
                    in_processes = (line == "data_pipeline_processes")
                    in_input_edges = (line == "data_pipeline_input_edges")
                    continue
                
                if line.startswith("loop_") or line.startswith("_"):
                    continue

                # --- 1. PARSE ALIASES ---
                if in_processes:
                    parts = line.split()
                    if len(parts) >= 2:
                        job_path = parts[0]
                        alias_raw = parts[1]

                        if "/job" in job_path:
                            seg = job_path.split('/')
                            job_str = next((s for s in seg if s.startswith("job") and s[3:].isdigit()), None)
                            
                            if job_str:
                                job_num = job_str.replace("job", "")
                                if alias_raw and alias_raw != "None":
                                    alias_parts = [p for p in alias_raw.split('/') if p]
                                    if len(alias_parts) > 1:
                                        aliases[job_num] = alias_parts[-1]
                                    elif len(alias_parts) == 1:
                                        aliases[job_num] = alias_parts[0]

                # --- 2. PARSE EDGES (Fix Direction) ---
                elif in_input_edges:
                    # Line: Parent/job035/run.star Child/job036/
                    parts = line.split()
                    if len(parts) >= 2:
                        src_str = parts[0]
                        dst_str = parts[1]

                        src_match = re.search(r'job(\d{3,})', src_str)
                        dst_match = re.search(r'job(\d{3,})', dst_str)

                        if src_match and dst_match:
                            parent_job = src_match.group(1)
                            child_job = dst_match.group(1)
                            
                            # CRITICAL FIX: Append as (Child, Parent)
                            # This matches 1.py's note.txt logic so graph_to_state swaps it correctly later.
                            edges.append((child_job, parent_job))

    except Exception as e:
        print(f"[workflow] Error parsing default_pipeline.star: {e}")

    return edges, aliases

def _parse_user_json() -> str:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--user-json", type=str)
    args, _ = parser.parse_known_args()
    path = args.user_json
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        user = data.get("user", "")
        return user
    except Exception as e:
        print(f"[workflow] Failed to read user JSON: {e}")
        return ""

def get_user_from_xcop():
    return _parse_user_json()

def get_current_user():
    return _parse_user_json()

def get_workflow_path(user):
    safe = re.sub(r'[^\w\-]', '_', user or "general")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), f"workflows_{safe}.json")

def load_all_workflows(app):
    path = get_workflow_path(get_current_user())
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[workflow] could not read workflows: {e}")
        return {}

def save_all_workflows(app):
    user = get_current_user()
    safe = re.sub(r'[^\w\-]', '_', user or "general")
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"workflows_{safe}.json")
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(app.workflows, f, indent=2)
        print(f"[workflow] SAVED ALL WORKFLOWS → {path}")
    except Exception as e:
        print(f"[workflow] could not write workflows: {e}")

# --------------------------------------------------------------------- #
# GRAPHICS ITEMS
# --------------------------------------------------------------------- #

class NodeItem(QGraphicsObject):
    positionChanged = pyqtSignal(object)

    def __init__(self, node_data, app):
        super().__init__()
        self.node_data = node_data
        self.app = app
        self.job_num = node_data.job_num
        self.width = NODE_W
        self.height = NODE_H
        self.setFlags(QGraphicsItem.GraphicsItemFlag.ItemIsMovable | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setPos(node_data.x, node_data.y)
        self.setZValue(10) # Nodes above links
        
        self.brush_color = QColor(self.app.type_colors.get(node_data.event_type, "#333333"))

    def mousePressEvent(self, event):
        self._drag_start_pos = event.screenPos()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        if hasattr(self, '_drag_start_pos'):
            moved_dist = (event.screenPos() - self._drag_start_pos).manhattanLength()
            if moved_dist < 5:
                if self.app.hide_linkers:
                    nid = self.node_data.id
                    if nid in self.app.selected_node_for_links:
                        self.app.selected_node_for_links.remove(nid)
                    else:
                        self.app.selected_node_for_links.add(nid)
                    self.app.redraw_links()
                    self.app.update_minimap()

    def boundingRect(self):
        margin = 150 
        return QRectF(
            -self.width/2 - margin, 
            -self.height/2 - margin, 
            self.width + 2*margin, 
            self.height + 2*margin
        )

    def shape(self):
        path = QPainterPath()
        path.addRect(-self.width/2, -self.height/2, self.width, self.height)
        return path

    def paint(self, painter, option, widget):
        scale = painter.transform().m11()
        MIN_PIXEL_SIZE = 4.0 
        
        current_pixel_width = self.width * scale
        
        # --- 1. Tiny Dot Rendering (LOD) ---
        if current_pixel_width < MIN_PIXEL_SIZE and scale > 0:
            painter.setBrush(QBrush(self.brush_color))
            painter.setPen(Qt.PenStyle.NoPen)
            adjusted_size = MIN_PIXEL_SIZE / scale
            adj_rect = QRectF(-adjusted_size/2, -adjusted_size/2, adjusted_size, adjusted_size)
            painter.drawRect(adj_rect)
            return

        rect = QRectF(-self.width/2, -self.height/2, self.width, self.height)

        # --- 2. Minimap Rendering ---
        if widget and hasattr(self.app, 'minimap_view') and widget.parent() is self.app.minimap_view:
            painter.setBrush(QBrush(self.brush_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(rect)
            return

        # --- 3. Main View Rendering ---
        # Prepare Text
        font = QFont("TkDefaultFont", BASE_FONT_SIZE)
        font.setBold(True)
        painter.setFont(font)
        text = f"{self.node_data.job_num}\n{self.node_data.event_type}"
        if self.node_data.alias:
            text += f"\n{self.node_data.alias}"
        
        # TextDontClip is critical here so text can flow outside
        flags = Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextDontClip 

        # Determine Colors for Box
        if hasattr(self, 'flash_active') and self.flash_active:
             brush = QBrush(QColor("#FFEB3B"))
             pen = QPen(QColor("#000000"), 4)
        else:
            brush = QBrush(self.brush_color)
            if self.isSelected():
                pen = QPen(QColor("#00E5FF"), 3, Qt.PenStyle.DashLine)
            else:
                pen = QPen(Qt.GlobalColor.white, 2)

        # --- DRAWING PASSES (The Fix) ---

        # Pass 1: Draw WHITE text everywhere
        # This ensures text spilling OUTSIDE the box is visible on the dark background.
        painter.setPen(Qt.GlobalColor.white)
        painter.drawText(rect, flags, text)

        # Pass 2: Draw the Box (Fill + Border)
        # The fill will cover the White text that is INSIDE the box.
        painter.setPen(pen)
        painter.setBrush(brush)
        painter.drawRect(rect)

        # Pass 3: Draw BLACK text, but CLIP it to the box
        # This redraws the text inside the box in black, sitting on top of the color.
        painter.save() # Good practice to save state before clipping
        painter.setClipRect(rect)
        painter.setPen(Qt.GlobalColor.black)
        painter.drawText(rect, flags, text)
        painter.restore()

    def itemChange(self, change, value):
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            self.node_data.x = value.x()
            self.node_data.y = value.y()
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            self.positionChanged.emit(self)
        return super().itemChange(change, value)

    def hoverEnterEvent(self, event):
        self.app.on_node_enter(self, event)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.app.on_node_leave(self, event)
        super().hoverLeaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.app.open_node_dialog(parent_node=self.node_data)
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        # Get all selected items in the scene
        selected_items = self.scene().selectedItems()
        # Filter to ensure we are only looking at NodeItems
        selected_nodes = [item for item in selected_items if isinstance(item, NodeItem)]

        # If more than 1 node is selected and THIS node is part of that selection
        if len(selected_nodes) > 1 and self.isSelected():
            self.app.show_multi_context_menu(selected_nodes, event.screenPos())
        else:
            # Default behavior for single node
            self.app.show_context_menu(self.node_data, event.screenPos())

class LinkItem(QGraphicsLineItem):
    def __init__(self, src_item, dst_item, app):
        super().__init__()
        self.src_item = src_item
        self.dst_item = dst_item
        self.app = app
        self.setZValue(5) # Below nodes
        self.update_position()
        
        # Default Link Color: Light Gray for Dark Mode
        self.setPen(QPen(QColor("#B0B0B0"), 2))

    def update_position(self):
        self.setLine(QLineF(self.src_item.pos(), self.dst_item.pos()))

    def update_appearance(self):
        should_show = True
        # Default to Light Gray
        color = QColor("#B0B0B0")
        
        if self.app.hide_linkers:
            if self.dst_item.node_data.id in self.app.selected_node_for_links:
                should_show = True
                # Use Parent color
                color = QColor(self.app.type_colors.get(self.src_item.node_data.event_type, "#B0B0B0"))
            else:
                should_show = False
        
        self.setVisible(should_show)
        
        if should_show:
            pen = QPen(color, 2) 
            pen.setCosmetic(True) 
            self.setPen(pen)

class ColorPreservingDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, height=20):
        super().__init__(parent)
        self.row_height = height

    def sizeHint(self, option, index):
        # Get default size (calculated from text width)
        size = super().sizeHint(option, index)
        # Force the height to match our calculation, keep width dynamic
        size.setHeight(self.row_height)
        return size

    def initStyleOption(self, option, index):
        super().initStyleOption(option, index)
        foreground_brush = index.data(Qt.ItemDataRole.ForegroundRole)
        if foreground_brush:
            option.palette.setColor(QPalette.ColorRole.HighlightedText, foreground_brush.color())

class StickyNoteItem(QGraphicsObject):
    def __init__(self, text, width, app):
        super().__init__()
        # FIX: Ignore zoom level so it stays the same pixel size on screen
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
        
        self.text = text
        self.w = width
        self.app = app
        self.setZValue(100) # Top most
        
        self.font = QFont("TkDefaultFont", 10)
        
        # Calculate height
        # Because we ignore transform, 'width' is now in screen pixels, not scene units.
        lines = text.count('\n') + 2 + (len(text) // 40)
        self.h = lines * 15 + 10

    def boundingRect(self):
        return QRectF(0, -self.h, self.w, self.h)

    def paint(self, painter, option, widget):
        # FIX: Do not draw if we are rendering on the Minimap
        # We check if the widget's parent is the MinimapView
        if widget and widget.parent() is self.app.minimap_view:
            return

        rect = self.boundingRect()
        painter.setBrush(QColor("#FFFF99"))
        painter.setPen(QPen(Qt.GlobalColor.black, 2))
        painter.drawRect(rect)
        
        painter.setPen(Qt.GlobalColor.black)
        painter.setFont(self.font)
        painter.drawText(rect.adjusted(5,5,-5,-5), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, self.text)

# --------------------------------------------------------------------- #
# DATA CLASSES
# --------------------------------------------------------------------- #

class NodeData:
    def __init__(self, job_num, event_type, alias, x, y, notes="", id=None):
        self.id = id if id else str(uuid.uuid4())
        self.job_num = job_num
        self.event_type = event_type
        self.alias = alias
        self.x = float(x)
        self.y = float(y)
        self.notes = notes

# --------------------------------------------------------------------- #
# DIALOGS
# --------------------------------------------------------------------- #

class SimpleInput(QDialog):
    def __init__(self, parent, title, prompt, default=""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.result = None
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(prompt))
        self.entry = QLineEdit(default)
        layout.addWidget(self.entry)
        
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.ok)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def ok(self):
        self.result = self.entry.text()
        self.accept()

class NodeDialog(QDialog):
    def __init__(self, parent, app, node=None, title="Create Node"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.app = app
        self.result = None
        layout = QVBoxLayout(self)
        
        layout.addWidget(QLabel("Job Number:"))
        default_job = app.get_next_job_number() if not node else node.job_num
        self.job_edit = QLineEdit(default_job)
        layout.addWidget(self.job_edit)
        
        layout.addWidget(QLabel("Event Type:"))
        self.type_combo = QComboBox()
        self.type_combo.addItems(app.event_types)
        current_type = "Import" if not node else node.event_type
        self.type_combo.setCurrentText(current_type)
        layout.addWidget(self.type_combo)
        
        layout.addWidget(QLabel("Alias (optional):"))
        default_alias = "" if not node else node.alias
        self.alias_edit = QLineEdit(default_alias)
        layout.addWidget(self.alias_edit)
        
        btns = QHBoxLayout()
        ok = QPushButton("OK")
        ok.clicked.connect(self.accept_data)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(ok)
        btns.addWidget(cancel)
        layout.addLayout(btns)

    def accept_data(self):
        job = self.job_edit.text().strip()
        if not job:
            QMessageBox.critical(self, "Error", "Job number required")
            return
        self.result = (job, self.type_combo.currentText(), self.alias_edit.text().strip())
        self.accept()

class LinkEditorDialog(QDialog):
    def __init__(self, parent, app, node_data):
        super().__init__(parent)
        self.app = app
        self.node_data = node_data
        self.setWindowTitle(f"Links for {node_data.job_num}")
        self.resize(420, 320)
        
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        layout.addWidget(self.list_widget)
        
        self.linked_nodes = []
        self.refresh()
        
        btns = QHBoxLayout()
        disc_btn = QPushButton("Disconnect")
        disc_btn.clicked.connect(self.disconnect)
        
        # --- CHANGE 1: Button Label ---
        conn_btn = QPushButton("Connect as child")
        # ------------------------------
        
        conn_btn.clicked.connect(self.connect_new)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        
        btns.addWidget(disc_btn)
        btns.addWidget(conn_btn)
        btns.addWidget(close_btn)
        layout.addLayout(btns)

    def refresh(self):
        self.list_widget.clear()
        self.linked_nodes = []
        
        # Find links involving this node
        for a_id, b_id in self.app.links:
            other_id = None
            if a_id == self.node_data.id: other_id = b_id
            elif b_id == self.node_data.id: other_id = a_id
            
            if other_id and other_id in self.app.nodes:
                self.linked_nodes.append(self.app.nodes[other_id])
                
        self.linked_nodes.sort(key=lambda n: int(n.job_num))
        
        for n in self.linked_nodes:
            self.list_widget.addItem(f"{n.job_num} {n.event_type} ({n.alias or 'no alias'})")

    def disconnect(self):
        row = self.list_widget.currentRow()
        if row < 0: return
        target = self.linked_nodes[row]
        self.app.remove_link(self.node_data.id, target.id)
        self.refresh()
        self.app.redraw_links()
        
    def connect_new(self):
        # Dialog to pick node
        d = QDialog(self)
        d.setWindowTitle("Select Parent Node")
        l = QVBoxLayout(d)
        lw = QListWidget()
        
        others = [n for n in self.app.nodes.values() if n.id != self.node_data.id]
        linked_ids = {n.id for n in self.linked_nodes}
        unconnected = [n for n in others if n.id not in linked_ids]
        unconnected.sort(key=lambda n: int(n.job_num))
        
        for n in unconnected:
            lw.addItem(f"{n.job_num} {n.event_type}")
            
        l.addWidget(lw)
        b = QPushButton("Link")
        
        def do_link():
            row = lw.currentRow()
            if row >= 0:
                parent_node = unconnected[row]
                child_node = self.node_data
                
                # --- CHANGE 2: Link Direction ---
                # We link: Selected_Parent -> Current_Node
                # This makes the current node a child, moving it down the hierarchy
                self.app.add_link(parent_node.id, child_node.id)
                # --------------------------------
                
                self.refresh()
                self.app.redraw_links()
                d.accept()
                
        b.clicked.connect(do_link)
        l.addWidget(b)
        d.exec()

class NotesDialog(QDialog):
    def __init__(self, parent, node_data):
        super().__init__(parent)
        self.node_data = node_data
        self.setWindowTitle(f"Notes – {node_data.job_num}")
        self.resize(500, 400)
        
        layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setPlainText(node_data.notes)
        layout.addWidget(self.text_edit)
        
        btns = QHBoxLayout()
        save = QPushButton("Save")
        save.clicked.connect(self.save)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(save)
        btns.addWidget(cancel)
        layout.addLayout(btns)
        
    def save(self):
        self.node_data.notes = self.text_edit.toPlainText().strip()
        self.accept()

# --------------------------------------------------------------------- #
# MAIN APPLICATION CLASS
# --------------------------------------------------------------------- #

class WorkflowScene(QGraphicsScene):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.app = app

    def mousePressEvent(self, event):
        # Standard processing to handle selection/deselection
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        # Pass double click to View for node creation
        super().mouseDoubleClickEvent(event)

class WorkflowView(QGraphicsView):
    def __init__(self, scene, app, parent=None):
        super().__init__(scene, parent)
        self.app = app
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self._middle_pressed = False
        self._last_mouse_pos = None

        

    def mouseDoubleClickEvent(self, event):
        # 1. Calculate where in the scene we clicked
        scene_pos = self.mapToScene(event.pos())
        
        # 2. Check if we clicked on an existing item (Node or Link)
        item = self.scene().itemAt(scene_pos, QTransform())

        if item:
            # If we clicked a Node, let the Node handle it (see step 1)
            super().mouseDoubleClickEvent(event)
        else:
            # 3. If we clicked Empty Space, create a new unconnected node
            self.app.create_node_at_position(scene_pos)

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.1 if zoom_in else 0.9
        self.scale(factor, factor)
        self.app.update_minimap()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton or (event.button() == Qt.MouseButton.RightButton and event.modifiers() & Qt.KeyboardModifier.NoModifier): 
            # Pan logic mimics Tkinter right-drag or middle-drag
            self._middle_pressed = True
            self._last_mouse_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            if event.button() == Qt.MouseButton.RightButton:
                # If we clicked an item, context menu is handled by item.
                # If background, maybe general menu?
                item = self.scene().itemAt(self.mapToScene(event.pos()), QTransform())
                if not item:
                    # Background right click -> Pan start
                    self._middle_pressed = True
                    self._last_mouse_pos = event.position().toPoint()
                    self.setCursor(Qt.CursorShape.ClosedHandCursor)
                    event.accept()
                    return

            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._middle_pressed:
            delta = event.pos() - self._last_mouse_pos
            self._last_mouse_pos = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            self.app.update_minimap()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._middle_pressed:
            self._middle_pressed = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            
            # If it was a short click (Right button), show context menu
            # But implementing drag threshold logic is complex here.
            # Simplified: If right click and didn't move much?
            # For now, let's assume right click on item is context menu, right click on bg is pan.
            
            event.accept()
        else:
            super().mouseReleaseEvent(event)

class MinimapView(QGraphicsView):
    def __init__(self, scene, main_view, parent=None):
        super().__init__(scene, parent)
        self.main_view = main_view
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setInteractive(False)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(240, 240, 240)))
    
    def wheelEvent(self, event):
        # Disable zooming on the minimap itself
        pass

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            # Tkinter logic: wx = left + (event.x / scale)
            # Qt equivalent: mapToScene handles the scale/offset calculation automatically
            scene_pos = self.mapToScene(event.pos())
            self.main_view.centerOn(scene_pos)
            self.main_view.app.update_minimap()

    def drawForeground(self, painter, rect):
        # Draw the main view's current viewport as a red rectangle
        super().drawForeground(painter, rect)
        
        # Calculate the visible area of the main view in scene coordinates
        viewport_rect = self.main_view.viewport().rect()
        scene_poly = self.main_view.mapToScene(viewport_rect)
        
        painter.save()
        painter.setPen(QPen(Qt.GlobalColor.white, 20))  # Thick stroke line (scales down visually)
        painter.setBrush(QBrush(QColor(153, 255, 153, 40)))
        painter.drawPolygon(scene_poly)
        painter.restore()

class MindMapApp(QMainWindow):
    def __init__(self, load_initial_data=True, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Relion Workflow")
        self.resize(WINDOW_SIZE_W, WINDOW_SIZE_H)

        # State
        self.nodes = {} # id -> NodeData
        self.links = [] # [(id, id), ...]
        self.workflows = {}
        self.current_workflow = None
        
        self.node_items = {} # id -> NodeItem
        self.link_items = {} # (id, id) -> LinkItem
        
        self.selected_node_for_links = set()
        self.hide_linkers = False
        self.sticky_notes_enabled = False
        self.hover_timer = QTimer()
        self.hover_timer.setSingleShot(True)
        self.hover_timer.timeout.connect(self._show_sticky_note)
        self.current_hovered_node_item = None
        self.current_sticky_item = None
        
        self.type_colors = {
    # --- The Light Greens (Giverny Garden) ---
    "Import": "#A5D6A7",             # Pale Emerald (Fresh leaves)
    "Subset selection": "#DCE775",   # Light Lime (Sunlit grass)
    "Mask creation": "#81C784",      # Soft Green (Distinct from Import)
    "ModelAngelo": "#4DB6AC",        # Muted Teal/Verdigris

    # --- The Light Blues (Morning Sky & Water) ---
    "Motion correction": "#81D4FA",  # Light Sky Blue
    "Particle extraction": "#80CBC4",# Soft Aqua (Pond water)
    "3D classification": "#64B5F6",  # Cornflower Blue (Distinct from Sky Blue)
    "3D multi-body": "#90A4AE",      # Blue-Grey Mist (Fog)
    "DynaMight flexibility": "#4DD0E1", # Cyan/Turquoise
    "Post-processing": "#E0F7FA",    # Ice White/Blue (Highlights)
    "External": "#B0BEC5",           # Light Grey (Clouds)

    # --- The Light Purples & Pinks (Water Lilies) ---
    "2D classification": "#CE93D8",  # Light Orchid/Lilac
    "CTF refinement": "#9575CD",     # Soft Violet (Darker than Orchid)
    "3D auto-refine": "#F8BBD0",   # Rose Pink (Flower petals)
    "Particle substraction": "#F48FB1", # Soft Flamingo
    "Local resolution": "#F06292",   # Pale Pink

    # --- The Light Warm Tones (Haystacks in Sunlight) ---
    "CTF estimation": "#FFB74D",     # Soft Apricot/Orange
    "Manual Picking": "#E57373",     # Light Red/Coral (Softened red)
    "Auto picking": "#D7CCC8",       # Beige/Sand (Earth)
    "3D initial reference": "#FFD54F", # Sunlight Yellow
    "Bayesian polishing": "#FF8A65", # Salmon (Sunset reflection)
    "Join star files": "#FFF176",  # Goldenrod/Wheat
}
        
        self.folder_to_type = {
            "Import": "Import", "MotionCorr": "Motion correction", "CtfFind": "CTF estimation",
            "ManualPick": "Manual Picking", "AutoPick": "Auto picking", "Extract": "Particle extraction",
            "Class2D": "2D classification", "InitialModel": "3D initial reference", "Class3D": "3D classification",
            "Refine3D": "3D auto-refine", "Multibody": "3D multi-body", "Select": "Subset selection",
            "CtfRefine": "CTF refinement", "Polish": "Bayesian polishing", "DynaMight": "DynaMight flexibility",
            "MaskCreate": "Mask creation", "JoinStar": "Join star files", "Substract": "Particle substraction",
            "PostProcess": "Post-processing", "LocalRes": "Local resolution", "ModelAngelo": "ModelAngelo",
            "External": "External"
        }
        self.type_to_folder = {v: k for k, v in self.folder_to_type.items()}
        self.event_types = list(self.type_colors.keys())

        self.init_ui()

        if load_initial_data:
            self.workflows = load_all_workflows(self)
            self.update_workflow_list()
            
            # FIX: Automatically open the first available workflow
            if self.workflows:
                # Pick the first one alphabetically
                first_wf = sorted(self.workflows.keys())[0]
                
                # Update the dropdown UI
                self.wf_combo.setCurrentText(first_wf)
                
                # Force the application to load and render it
                self.on_workflow_selected(first_wf)
            else:
                pass
        
    def create_node_at_position(self, pos):
        default_job = self.get_next_job_number()
        
        # Open dialog in "Create" mode (node=None)
        dlg = NodeDialog(self, self, node=None, title="Create Node")
        dlg.job_edit.setText(default_job)
        
        if dlg.exec():
            job, etype, alias = dlg.result
            
            # Use the clicked position
            new_node = NodeData(job, etype, alias, pos.x(), pos.y())
            self.nodes[new_node.id] = new_node
            
            item = NodeItem(new_node, self)
            item.positionChanged.connect(self.on_node_move)
            self.scene.addItem(item)
            self.node_items[new_node.id] = item
            
            self.update_sidebar()

    def show_multi_context_menu(self, items, pos):
        menu = QMenu()

        # Add the options but disable them as requested
        action_rename = menu.addAction("Rename")
        action_rename.setEnabled(False)

        action_links = menu.addAction("Edit Linker")
        action_links.setEnabled(False)

        action_notes = menu.addAction("Notes")
        action_notes.setEnabled(False)

        menu.addSeparator()

        # Add Delete option
        action_delete = menu.addAction(f"Delete ({len(items)} items)")
        # Connect to the bulk delete function
        action_delete.triggered.connect(lambda: self.delete_multiple_nodes(items))

        final_pos = pos.toPoint() if hasattr(pos, 'toPoint') else pos
        menu.exec(final_pos)

    def delete_multiple_nodes(self, items):
        if QMessageBox.question(self, "Delete", f"Delete {len(items)} selected nodes?") != QMessageBox.StandardButton.Yes:
            return

        # 1. Collect data objects first
        nodes_to_delete = [item.node_data for item in items if isinstance(item, NodeItem)]

        # 2. Bulk remove (Optimization: Don't redraw links/sidebar inside the loop)
        for node_data in nodes_to_delete:
            # Remove links involving this node
            self.links = [l for l in self.links if node_data.id not in l]
            
            # Remove item from scene
            if node_data.id in self.node_items:
                item = self.node_items[node_data.id]
                
                # CRITICAL FIX: Prevent C++ deletion crashes if the node is currently hovered
                if self.current_hovered_node_item == item:
                    self.hover_timer.stop()
                    self._hide_sticky_note()
                    self.current_hovered_node_item = None
                    
                self.scene.removeItem(item)
                del self.node_items[node_data.id]
            
            # Remove from data dict
            if node_data.id in self.nodes:
                del self.nodes[node_data.id]

        # 3. Update UI once at the end
        self.redraw_links()
        self.update_sidebar()

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        # 1. Left Layout (Toolbar + Graphics)
        left_layout = QVBoxLayout()
        
        # Toolbar 1
        tb1 = QFrame()
        tb1.setFrameShape(QFrame.Shape.StyledPanel)
        tb1_layout = QHBoxLayout(tb1)
        tb1_layout.setContentsMargins(2,2,2,2)
        
        tb1_layout.addWidget(QLabel("Workflow:"))
        self.wf_combo = QComboBox()
        self.wf_combo.setMinimumWidth(150)
        self.wf_combo.textActivated.connect(self.on_workflow_selected)
        tb1_layout.addWidget(self.wf_combo)
        
        self.btn_create = QPushButton("Create")
        self.btn_create.clicked.connect(self.create_workflow)
        tb1_layout.addWidget(self.btn_create)

        self.btn_rename = QPushButton("Rename")
        self.btn_rename.clicked.connect(self.rename_workflow)
        tb1_layout.addWidget(self.btn_rename)

        self.btn_copy = QPushButton("Copy")
        self.btn_copy.clicked.connect(self.copy_workflow)
        tb1_layout.addWidget(self.btn_copy)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.clicked.connect(self.delete_workflow)
        tb1_layout.addWidget(self.btn_delete)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_current)
        tb1_layout.addWidget(self.btn_clear)

        self.btn_import = QPushButton("Import Graph")
        self.btn_import.clicked.connect(self.import_dependency_graph)
        tb1_layout.addWidget(self.btn_import)

        self.btn_cont_import = QPushButton("Continue Import")
        self.btn_cont_import.clicked.connect(self.continue_import)
        tb1_layout.addWidget(self.btn_cont_import)
            
        tb1_layout.addWidget(QPushButton("Sort", clicked=self.sort_nodes))
        self.sort_start = QLineEdit(); self.sort_start.setFixedWidth(40)
        self.sort_end = QLineEdit(); self.sort_end.setFixedWidth(40)
        tb1_layout.addWidget(self.sort_start)
        tb1_layout.addWidget(QLabel("to"))
        tb1_layout.addWidget(self.sort_end)
        tb1_layout.addStretch()
        
        # Toolbar 2
        tb2 = QFrame()
        tb2_layout = QHBoxLayout(tb2)
        tb2_layout.setContentsMargins(2,2,2,2)
        
        tb2_layout.addWidget(QPushButton("Import Job Info", clicked=self.import_job_info))
        
        self.hide_linkers_cb = QCheckBox("Hide linkers")
        self.hide_linkers_cb.toggled.connect(self.update_links_visibility)
        tb2_layout.addWidget(self.hide_linkers_cb)
        
        tb2_layout.addWidget(QPushButton("Clear chains", clicked=self.clear_chains))
        
        self.sticky_cb = QCheckBox("Sticky notes")
        self.sticky_cb.toggled.connect(self.toggle_sticky_notes)
        tb2_layout.addWidget(self.sticky_cb)
        
        tb2_layout.addWidget(QPushButton("Isolate", clicked=self.isolate_jobs))
        tb2_layout.addStretch()
        
        # Search Bar
        sb = QFrame()
        sb_layout = QHBoxLayout(sb)
        sb_layout.setContentsMargins(2,2,2,2)
        sb_layout.addWidget(QLabel("Search Jobs/Alias:"))
        self.search_edit = QLineEdit()
        self.search_edit.returnPressed.connect(self.perform_search)
        sb_layout.addWidget(self.search_edit)
        sb_layout.addWidget(QPushButton("Search", clicked=self.perform_search))
        sb_layout.addStretch()

        left_layout.addWidget(tb1)
        left_layout.addWidget(tb2)
        left_layout.addWidget(sb)

        # Scene and View
        self.scene = WorkflowScene(self)
        self.scene.setSceneRect(-50000, -50000, 100000, 100000)
        
        self.view = WorkflowView(self.scene, self)
        # DARK MODE BACKGROUND
        self.view.setBackgroundBrush(QBrush(QColor(35, 35, 35)))
        left_layout.addWidget(self.view)

        # 2. Sidebar (Right)
        sidebar = QFrame()
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        sidebar.setFrameShape(QFrame.Shape.StyledPanel)
        sidebar_layout = QVBoxLayout(sidebar)
        
        sidebar_layout.addWidget(QLabel("Jobs"))
        
        # Job Lists (Num | Detail)
        job_list_frame = QWidget()
        job_list_layout = QHBoxLayout(job_list_frame)
        job_list_layout.setContentsMargins(0,0,0,0)
        job_list_layout.setSpacing(0)

        list_font = QFont("TkDefaultFont", 9) 
        fm = QFontMetrics(list_font)
        row_height = fm.height() + 4 
        
        self.job_num_list = QListWidget()
        self.job_num_list.setFont(list_font)
        self.job_num_list.setFrameShape(QFrame.Shape.NoFrame)
        self.job_num_list.setFixedWidth(40)
        self.job_num_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.job_num_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.job_num_list.setGridSize(QSize(0, row_height)) 
        self.job_num_list.setItemDelegate(ColorPreservingDelegate(self.job_num_list, row_height))

        self.job_detail_list = QListWidget()
        self.job_detail_list.setFont(list_font)
        self.job_detail_list.setFrameShape(QFrame.Shape.NoFrame)
        self.job_detail_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.job_detail_list.setWordWrap(False)
        self.job_detail_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.job_detail_list.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.job_detail_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.job_detail_list.setItemDelegate(ColorPreservingDelegate(self.job_detail_list, row_height))

        job_list_layout.addWidget(self.job_num_list)
        job_list_layout.addWidget(self.job_detail_list)
        
        sidebar_layout.addWidget(job_list_frame, 1)
        
        self.job_num_list.verticalScrollBar().valueChanged.connect(self.job_detail_list.verticalScrollBar().setValue)
        self.job_detail_list.verticalScrollBar().valueChanged.connect(self.job_num_list.verticalScrollBar().setValue)
        self.job_num_list.currentRowChanged.connect(self.on_job_select_num)
        self.job_detail_list.currentRowChanged.connect(self.on_job_select_detail)
        
        self.job_num_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.job_num_list.customContextMenuRequested.connect(self.on_job_list_context)
        self.job_detail_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.job_detail_list.customContextMenuRequested.connect(self.on_job_list_context)
        
        fc = QHBoxLayout()
        fc.addWidget(QPushButton("-", clicked=self.decrease_font))
        self.font_size_lbl = QLabel(str(BASE_FONT_SIZE))
        fc.addWidget(self.font_size_lbl)
        fc.addWidget(QPushButton("+", clicked=self.increase_font))
        fc.addWidget(QPushButton("Reset", clicked=self.reset_font))
        sidebar_layout.addLayout(fc)
        
        # Minimap
        self.minimap_view = MinimapView(self.scene, self.view)
        self.minimap_view.setFixedSize(MINIMAP_SIZE, MINIMAP_SIZE)
        # DARK MODE MINIMAP BACKGROUND
        self.minimap_view.setBackgroundBrush(QBrush(QColor(35, 35, 35)))
        
        sidebar_layout.addWidget(self.minimap_view, alignment=Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignHCenter)
        
        # --- CRITICAL FIX: Ensure layouts are attached to main structure ---
        main_layout.addLayout(left_layout)
        main_layout.addWidget(sidebar)

    # -----------------------------------------------------------------
    # LOGIC IMPLEMENTATION
    # -----------------------------------------------------------------

    def closeEvent(self, event):
        if self.current_workflow:
            self.workflows[self.current_workflow] = self.save_state()
        save_all_workflows(self)
        event.accept()

    def save_state(self):
        state = {
            "nodes": [], "links": [],
            "hide_linkers": self.hide_linkers,
            "selected_node_for_links": list(self.selected_node_for_links)
        }
        for node in self.nodes.values():
            state["nodes"].append({
                "id": node.id, "job_num": node.job_num, "event_type": node.event_type,
                "alias": node.alias, "x": node.x, "y": node.y, "notes": node.notes
            })
        for a, b in self.links:
            state["links"].append((a, b))

        return state

    def load_state(self, state):
        self.clear_current(ask=False)
        if not state: return
        
        id_map = {}
        # Load Nodes
        for nd in state.get("nodes", []):
            node = NodeData(nd["job_num"], nd["event_type"], nd["alias"], nd["x"], nd["y"], nd.get("notes", ""), nd["id"])
            self.nodes[node.id] = node
            id_map[nd["id"]] = node.id
            
            item = NodeItem(node, self)
            item.positionChanged.connect(self.on_node_move)
            self.scene.addItem(item)
            self.node_items[node.id] = item
            
        # Load Links
        for a, b in state.get("links", []):
            if a in id_map and b in id_map:
                real_a = id_map[a]
                real_b = id_map[b]
                self.links.append((real_a, real_b))
        
        self.hide_linkers = state.get("hide_linkers", False)
        self.hide_linkers_cb.setChecked(self.hide_linkers)
        self.selected_node_for_links = set(state.get("selected_node_for_links", []))
        
        self.redraw_links()
        self.update_sidebar()
        self.update_minimap()

    def clear_current(self, ask=True):
        if ask and not self.current_workflow: return
        if ask:
            if QMessageBox.question(self, "Clear", "Clear all content?") != QMessageBox.StandardButton.Yes:
                return
        
        # CRITICAL FIX: Stop timers and clear references to prevent C++ object deletion crashes
        self.hover_timer.stop()
        self._hide_sticky_note()
        self.current_hovered_node_item = None
        self.current_sticky_item = None

        self.scene.clear()
        self.nodes.clear()
        self.links.clear()

        self.node_items.clear()
        self.link_items.clear()

        self.selected_node_for_links.clear()
        
        # CRITICAL FIX: Update UI elements to reflect the cleared state
        self.update_sidebar()
        self.update_minimap()

    def on_node_move(self, node_item):
        # NEW FIX: Only update positions of existing links connected to this node
        nid = node_item.node_data.id
        for (a_id, b_id), link_item in self.link_items.items():
            if a_id == nid or b_id == nid:
                link_item.update_position()
                
        self.update_minimap()
    
    def add_link(self, id1, id2):
        # Prevent duplicates (check both directions)
        if (id1, id2) not in self.links and (id2, id1) not in self.links:
            self.links.append((id1, id2))

    def remove_link(self, id1, id2):
        # Remove link regardless of direction (a->b or b->a)
        self.links = [(a, b) for a, b in self.links 
                      if not ((a == id1 and b == id2) or (a == id2 and b == id1))]
    
    def remove_directed_link(self, src, dst):
        # Only remove the specific directional link src -> dst
        self.links = [(a, b) for a, b in self.links if not (a == src and b == dst)]
        
    def redraw_links(self):
        # In QGraphicsScene, we don't necessarily delete and recreate all lines.
        # But to keep logic simple and consistent with filtering:
        
        # Remove old links
        for item in self.link_items.values():
            self.scene.removeItem(item)
        self.link_items.clear()
        
        # Add links
        for a_id, b_id in self.links:
            if a_id in self.node_items and b_id in self.node_items:
                item = LinkItem(self.node_items[a_id], self.node_items[b_id], self)
                item.update_appearance()
                if item.isVisible():
                    self.scene.addItem(item)
                    self.link_items[(a_id, b_id)] = item
                    
        self.update_minimap()

    def update_sidebar(self):
        self.job_num_list.clear()
        self.job_detail_list.clear()
        
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: int(n.job_num))
        for node in sorted_nodes:
            self.job_num_list.addItem(node.job_num)
            
            alias = f" ({node.alias})" if node.alias else ""
            folder = self.type_to_folder.get(node.event_type, node.event_type)
            item_text = f"{folder}{alias}"
            self.job_detail_list.addItem(item_text)
            
            # Color
            c = QColor(self.type_colors.get(node.event_type, "black"))
            self.job_num_list.item(self.job_num_list.count()-1).setForeground(c)
            self.job_detail_list.item(self.job_detail_list.count()-1).setForeground(c)

    def update_minimap(self):
        if not self.view.scene(): return
        
        # 1. Get the bounding box of all items (Nodes + Areas)
        # Equivalent to Tkinter: finding min/max of all xs and ys
        items_rect = self.scene.itemsBoundingRect()
        
        # 2. Get the bounding box of the current Viewport
        # Equivalent to Tkinter: self.get_viewport_world_bounds()
        viewport_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()
        
        # 3. Combine them (The Union)
        # This ensures the minimap ALWAYS shows the nodes AND where you are looking,
        # even if you scroll far away into empty space.
        final_rect = items_rect.united(viewport_rect)
        
        # 4. Add Padding
        # Tkinter used 'padding = 200'
        padding = 200
        final_rect.adjust(-padding, -padding, padding, padding)
        
        # 5. Fit this calculated world into the Minimap View
        # Qt's KeepAspectRatio is the equivalent of Tkinter's scale calculation
        self.minimap_view.fitInView(final_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.minimap_view.viewport().update()

    # -----------------------------------------------------------------
    # WORKFLOW MANAGEMENT
    # -----------------------------------------------------------------

    def update_workflow_list(self):
        self.wf_combo.blockSignals(True)
        self.wf_combo.clear()
        names = sorted(self.workflows.keys())
        self.wf_combo.addItems(names)
        if self.current_workflow:
            self.wf_combo.setCurrentText(self.current_workflow)
        self.wf_combo.blockSignals(False)

    def on_workflow_selected(self, name):
        if self.current_workflow:
            self.workflows[self.current_workflow] = self.save_state()
        self.current_workflow = name
        self.load_state(self.workflows[name])
        self.setWindowTitle(f"Relion Workflow – {name}")

    def create_workflow(self):
        dlg = SimpleInput(self, "New Workflow", "Name:")
        if dlg.exec() and dlg.result:
            name = dlg.result.strip()
            if name in self.workflows:
                QMessageBox.warning(self, "Error", "Exists")
                return
            self.workflows[name] = {"nodes": [], "links": []}
            self.update_workflow_list()
            self.on_workflow_selected(name)
            self.wf_combo.setCurrentText(name)

    def rename_workflow(self):
        if not self.current_workflow: return
        
        dlg = SimpleInput(self, "Rename", "New Name:", self.current_workflow)
        if dlg.exec() and dlg.result:
            new_name = dlg.result.strip()
            if not new_name or new_name == self.current_workflow: return
            
            if new_name in self.workflows:
                QMessageBox.warning(self, "Error", "Name already exists")
                return

            # 1. Rename the entry in the dictionary
            # pop() removes the old key and returns the data, which we assign to the new key
            self.workflows[new_name] = self.workflows.pop(self.current_workflow)
            
            # 2. CRITICAL FIX: Update the tracking variable IMMEDIATELY
            # This prevents the system from saving data back to the old name later
            self.current_workflow = new_name
            
            # 3. Update the UI Dropdown
            self.update_workflow_list()
            
            # 4. Set the dropdown selection to the new name without triggering 'on_workflow_selected'
            # (We block signals so it doesn't try to reload the workflow we are already editing)
            self.wf_combo.blockSignals(True)
            self.wf_combo.setCurrentText(new_name)
            self.wf_combo.blockSignals(False)
            
            # 5. Update Window Title
            self.setWindowTitle(f"Relion Workflow – {new_name}")

    def copy_workflow(self):
        if not self.current_workflow: return
        
        base_name = self.current_workflow
        new_name = f"{base_name}_copy"
        i = 1
        while new_name in self.workflows:
            new_name = f"{base_name}_copy{i}"
            i += 1
            
        # Save current state so we copy the latest version
        self.workflows[self.current_workflow] = self.save_state()
        
        # Save the new copy
        self.workflows[new_name] = self.save_state()
        
        self.update_workflow_list()
        
        # FIX: Immediately switch to the new copy
        self.wf_combo.setCurrentText(new_name)
        self.on_workflow_selected(new_name)

    def delete_workflow(self):
        if not self.current_workflow: return
        
        msg = f"Delete workflow '{self.current_workflow}'?"
        if QMessageBox.question(self, "Delete", msg) == QMessageBox.StandardButton.Yes:
            # Remove from data
            del self.workflows[self.current_workflow]
            self.current_workflow = None
            self.update_workflow_list()
            
            # FIX: Determine what to show next
            available_keys = sorted(self.workflows.keys())
            
            if available_keys:
                # Switch to the first available workflow
                next_wf = available_keys[0]
                self.wf_combo.setCurrentText(next_wf)
                self.on_workflow_selected(next_wf)
            else:
                # No workflows left, clear the screen completely
                self.clear_current(ask=False)
                self.wf_combo.clear()
                self.setWindowTitle("Relion Workflow")

    # -----------------------------------------------------------------
    # IMPORT LOGIC
    # -----------------------------------------------------------------

    def import_dependency_graph(self):
        # 1. Change "Imported" to "Workflow"
        dlg = SimpleInput(self, "Import", "Workflow Name:", "Workflow")
        
        # 2. Select all text so it is highlighted immediately
        dlg.entry.selectAll()

        if dlg.exec() and dlg.result:
            name = dlg.result.strip()
            
            # --- MODIFIED UNPACKING ---
            edges, has_note, folder_of, alias_of, added_count = self.collect_edges(ROOT_DIR)
            
            unique_edges = list(set(edges))
            edges = sorted(unique_edges, key=lambda x: (int(x[0]), int(x[1])))
            
            isolated = self.find_isolated(edges, has_note)
            
            if not edges and not isolated:
                QMessageBox.information(self, "Info", "No jobs found via note.txt or default_pipeline.star.")
                return
            
            # --- POPUP REMINDER ---
            if added_count > 0:
                QMessageBox.information(
                    self, 
                    "Merged Sources", 
                    f"Loaded edges from note.txt.\n\n"
                    f"Found {added_count} additional edges in default_pipeline.star that were missing from note.txt.\n"
                    "These have been added to the graph."
                )
                
            state = self.graph_to_state(edges, isolated, folder_of, alias_of)
            self.workflows[name] = state
            self.update_workflow_list()
            self.on_workflow_selected(name)
            self.wf_combo.setCurrentText(name)

    def continue_import(self):
        if not self.current_workflow:
            QMessageBox.critical(self, "Error", "Please select or create a workflow first!")
            return
        
        # --- MODIFIED UNPACKING ---
        edges, has_note, folder_of, alias_of, added_count = self.collect_edges(ROOT_DIR)
        
        edges = self.dedup_edges(edges)
        isolated = self.find_isolated(edges, has_note)
        
        if not edges and not isolated:
            QMessageBox.information(self, "Info", "No jobs found to import.")
            return

        # --- POPUP REMINDER ---
        if added_count > 0:
            QMessageBox.information(
                self, 
                "Merged Sources", 
                f"Found {added_count} edges in default_pipeline.star missing from note.txt.\n"
                "These will be included in the import."
            )
        
        # 2. Memory Operations (Merge Logic)
        current_state = self.save_state()
        existing_job_nums = {node.job_num for node in self.nodes.values()}
        
        new_state = self.graph_to_state(edges, isolated, folder_of, alias_of, preserve_existing=True)
        
        merged_nodes = []
        job_to_id = {}
        
        # Keep all old nodes
        for old in current_state["nodes"]:
            merged_nodes.append(old)
            job_to_id[old["job_num"]] = old["id"]
            
        max_x = max((n["x"] for n in current_state["nodes"]), default=0)
        spacing = 150
        col = 0
        
        for new in new_state["nodes"]:
            if new["job_num"] in existing_job_nums:
                old = next(n for n in merged_nodes if n["job_num"] == new["job_num"])
                if new.get("alias"): 
                    old["alias"] = new["alias"]
                continue
            
            new_id = str(uuid.uuid4())
            job_to_id[new["job_num"]] = new_id
            
            x = max_x + 300 + (col % 5) * spacing
            y = 100 + (col // 5) * spacing
            col += 1
            
            merged_nodes.append({
                "id": new_id, "job_num": new["job_num"], "event_type": new["event_type"],
                "alias": new.get("alias", ""), "x": x, "y": y, "notes": ""
            })
            
        merged_links = list(current_state["links"])
        job_to_current_id = {n["job_num"]: n["id"] for n in merged_nodes}
        
        temp_id_to_job = {n['id']: n['job_num'] for n in new_state['nodes']}
        
        for src_old_id, dst_old_id in new_state['links']:
            src_job = temp_id_to_job.get(src_old_id)
            dst_job = temp_id_to_job.get(dst_old_id)
            
            if not src_job or not dst_job: continue
            
            src_id = job_to_current_id.get(src_job)
            dst_id = job_to_current_id.get(dst_job)
            
            if not src_id or not dst_id: continue
            
            if (src_id, dst_id) not in merged_links and (dst_id, src_id) not in merged_links:
                merged_links.append((src_id, dst_id))
                    
        final_state = {"nodes": merged_nodes, "links": merged_links}
        
        self.workflows[self.current_workflow] = final_state
        self.clear_current(ask=False)
        self.load_state(final_state)
        
        save_all_workflows(self)

    def collect_edges(self, root):
        edges = set()
        folder_of = {}
        has_note = set()
        
        # 1. PRIMARY SOURCE: note.txt (Legacy Priority)
        # Format: (Child, Parent)
        for folder in FOLDERS:
            path = os.path.join(root, folder)
            if not os.path.isdir(path): continue
            for entry in os.listdir(path):
                if entry.startswith("job") and os.path.isdir(os.path.join(path, entry)):
                    jn = entry[3:]
                    folder_of[jn] = folder
                    
                    note = os.path.join(path, entry, "note.txt")
                    if os.path.isfile(note):
                        has_note.add(jn)
                        try:
                            with open(note, "r") as f:
                                for line in f:
                                    for dst in JOB_PATTERN.findall(line):
                                        if dst != jn: 
                                            edges.add((jn, dst)) # Adds (Child, Parent)
                        except Exception: 
                            pass

        # 2. SECONDARY SOURCE: default_pipeline.star
        # Now returns (Child, Parent) to match above
        official_edges, official_aliases = _parse_relion_root_star(root)
        
        added_from_official = 0

        for child, parent in official_edges:
            # Only add if this link doesn't exist yet
            if (child, parent) not in edges:
                edges.add((child, parent))
                added_from_official += 1

        # 3. USE OFFICIAL ALIASES
        alias_of = official_aliases

        return edges, has_note, folder_of, alias_of, added_from_official

    def dedup_edges(self, edges):
        return list(set(edges))

    def find_isolated(self, edges, has_note):
        srcs = {s for s, _ in edges}
        return sorted(list(has_note - srcs))

    def graph_to_state(self, edges, isolated, folder_of, alias_of=None, preserve_existing=False):
        alias_of = alias_of or {}
        state = {"nodes": [], "links": []}
        
        # Collect all involved jobs
        all_jobs = {src for src, _ in edges} | {dst for _, dst in edges} | set(isolated)
        
        job_to_id = {}
        y_off = 100
        x_off = 100
        spacing = 150
        
        if preserve_existing:
            # SAFETY CHECK: Reads from internal self.nodes only.
            max_x = max((node.x for node in self.nodes.values()), default=0) + spacing
            if max_x > x_off:
                x_off = max_x
        
        for i, j in enumerate(sorted(all_jobs, key=int)):
            nid = str(uuid.uuid4())
            job_to_id[j] = nid
            folder = folder_of.get(j, "External")
            etype = self.folder_to_type.get(folder, "External")
            alias = alias_of.get(j, "")
            
            state["nodes"].append({
                "id": nid, "job_num": j, "event_type": etype, "alias": alias,
                "x": x_off + (i % 5)*spacing, "y": y_off + (i // 5)*spacing, "notes": ""
            })
            
        for s, d in edges:
            if s in job_to_id and d in job_to_id:
                # --- CHANGE IS HERE ---
                # Reference script stores links as (child, parent) / (dst, src)
                # We must swap s and d here to match the Reference sort behavior.
                state["links"].append((job_to_id[d], job_to_id[s])) 
                
        return state

    # -----------------------------------------------------------------
    # INTERACTIONS
    # -----------------------------------------------------------------

    def get_next_job_number(self):
        if not self.nodes: return "001"
        mx = max(int(n.job_num) for n in self.nodes.values())
        return f"{mx+1:03d}"

    def open_node_dialog(self, parent_node=None):
        dlg = NodeDialog(self, self, node=None, title="Create Node" if not parent_node else "Add Child")
        if dlg.exec():
            job, etype, alias = dlg.result
            x = 400
            y = 300
            if parent_node:
                x = parent_node.x
                y = parent_node.y + 120
            
            new_node = NodeData(job, etype, alias, x, y)
            self.nodes[new_node.id] = new_node
            
            item = NodeItem(new_node, self)
            item.positionChanged.connect(self.on_node_move)
            self.scene.addItem(item)
            self.node_items[new_node.id] = item
            
            if parent_node:
                self.links.append((parent_node.id, new_node.id))
                self.redraw_links()
                
            self.update_sidebar()

    def show_context_menu(self, node_data, pos):
        menu = QMenu()
        menu.addAction("Rename", lambda: self.edit_node(node_data))
        menu.addAction("Edit Linker", lambda: self.edit_links(node_data))
        menu.addAction("Notes", lambda: self.edit_notes(node_data))
        menu.addAction("Delete", lambda: self.delete_node(node_data))
        final_pos = pos.toPoint() if hasattr(pos, 'toPoint') else pos
        menu.exec(final_pos)

    def edit_node(self, node_data):
        dlg = NodeDialog(self, self, node=node_data, title="Edit Node")
        if dlg.exec():
            job, etype, alias = dlg.result
            node_data.job_num = job
            node_data.event_type = etype
            node_data.alias = alias
            
            # FIX: Update the visual item's properties
            item = self.node_items[node_data.id]
            
            # Update the color in case the type changed
            item.brush_color = QColor(self.type_colors.get(etype, "#333333"))
            
            # Redraw the node and the sidebar
            item.update() 
            self.update_sidebar()

    def edit_links(self, node_data):
        dlg = LinkEditorDialog(self, self, node_data)
        dlg.exec()

    def edit_notes(self, node_data):
        dlg = NotesDialog(self, node_data)
        dlg.exec()

    def delete_node(self, node_data):
        # Remove links
        self.links = [l for l in self.links if node_data.id not in l]
        
        # Remove item
        item = self.node_items[node_data.id]
        
        # CRITICAL FIX: Prevent C++ deletion crashes if the node is currently hovered
        if self.current_hovered_node_item == item:
            self.hover_timer.stop()
            self._hide_sticky_note()
            self.current_hovered_node_item = None
            
        self.scene.removeItem(item)
        del self.node_items[node_data.id]
        del self.nodes[node_data.id]
        
        self.redraw_links()
        self.update_sidebar()

    def sort_nodes(self):
        if not self.nodes: return

        # ---------------------------------------------------------
        # 1. CYCLE DETECTION & RESOLUTION
        # ---------------------------------------------------------
        while True:
            # Build a temporary adjacency list for detection
            temp_graph = defaultdict(list)
            for src, dst in self.links:
                if src in self.nodes and dst in self.nodes:
                    temp_graph[src].append(dst)

            visited = set()
            recursion_stack = set()
            cycle_path = [] # Will store IDs [A, B, C] if A->B->C->A

            def has_cycle(u, current_path):
                visited.add(u)
                recursion_stack.add(u)
                current_path.append(u)

                if u in temp_graph:
                    for v in temp_graph[u]:
                        if v not in visited:
                            if has_cycle(v, current_path):
                                return True
                        elif v in recursion_stack:
                            # Cycle detected! Retrieve the path from v to u
                            cycle_start_index = current_path.index(v)
                            cycle_path.extend(current_path[cycle_start_index:])
                            return True
                
                current_path.pop()
                recursion_stack.remove(u)
                return False

            found_cycle = False
            for node_id in list(self.nodes.keys()):
                if node_id not in visited:
                    if has_cycle(node_id, []):
                        found_cycle = True
                        break
            
            if not found_cycle:
                break # No cycles found, proceed to normal sorting

            # --- Handle the found cycle ---
            
            # 1. Collect Job Numbers for the popup
            involved_jobs = [self.nodes[nid].job_num for nid in cycle_path]
            involved_str = " -> ".join(involved_jobs) + " -> " + involved_jobs[0]

            # 2. Find the "Bad" link to break
            # We look for a link where Parent Job > Child Job.
            # If strictly increasing (e.g., 1->2->3->1), we break 3->1.
            link_to_remove = None
            
            # Construct pairs from the path
            cycle_pairs = []
            for i in range(len(cycle_path)):
                u = cycle_path[i]
                v = cycle_path[(i + 1) % len(cycle_path)] # Wrap around to start
                cycle_pairs.append((u, v))
            
            # Search for the specific back-edge
            for u, v in cycle_pairs:
                job_u = int(self.nodes[u].job_num)
                job_v = int(self.nodes[v].job_num)
                if job_u > job_v:
                    link_to_remove = (u, v)
                    break
            
            # Fallback: if jobs are somehow equal or logic fails, break the last link in detection
            if not link_to_remove:
                link_to_remove = cycle_pairs[-1]

            # 3. Execute Removal
            r_src, r_dst = link_to_remove
            self.remove_directed_link(r_src, r_dst)
            self.redraw_links() # Visually update immediately
            
            # 4. Notify User
            msg = (f"Infinite loop detected among jobs:\n{involved_str}\n\n"
                   f"Auto-resolving by removing link: {self.nodes[r_src].job_num} -> {self.nodes[r_dst].job_num}\n"
                   "(Enforcing smaller job number as parent)")
            QMessageBox.warning(self, "Cycle Detected", msg)
            
            # Loop continues to check if there are nested cycles left...

        # ---------------------------------------------------------
        # 2. STANDARD SORTING LOGIC
        # ---------------------------------------------------------
        
        # 1. Parse Inputs
        start_str = self.sort_start.text().strip()
        end_str   = self.sort_end.text().strip()
        
        def parse(s):
            if not s: return None
            try: return f"{int(s):03d}"
            except: return None
            
        start_job = parse(start_str)
        end_job   = parse(end_str)
        
        if (start_job and not end_job) or (end_job and not start_job):
            QMessageBox.critical(self, "Error", "Enter both start and end job numbers.")
            return

        if start_job and end_job and start_job > end_job:
            start_job, end_job = end_job, start_job
            
        target_jobs = {
            nid for nid, node in self.nodes.items()
            if (not start_job or start_job <= node.job_num <= end_job)
        }
        if not target_jobs: return

        # 2. Build Graph (Matches workflow_reference.py logic)
        graph = defaultdict(list)
        parents_of = defaultdict(list)
        in_degree = defaultdict(int)
        
        for src, dst in self.links:
            if src in self.nodes and dst in self.nodes:
                graph[src].append(dst)
                parents_of[dst].append(src)
                in_degree[dst] += 1
                
        # 3. DFS & Level Calculation
        level = {}
        refine_count = {}
        
        def dfs(nid, lvl, refines):
            if nid in level: return
            level[nid] = lvl
            # Logic: 3D auto-refine bumps the 'refine_count' for alignment
            is_refine = (self.nodes[nid].event_type == "3D auto-refine")
            refine_count[nid] = refines + (1 if is_refine else 0)
            
            for child in graph[nid]:
                dfs(child, lvl + 1, refine_count[nid])
        
        # Find roots
        roots = [nid for nid in self.nodes if in_degree[nid] == 0]
        if not roots and self.nodes:
            # Fallback (though cycle detection should prevent this case mostly)
            roots = [min(self.nodes.keys(), key=lambda k: int(self.nodes[k].job_num))]
            
        for r in roots: dfs(r, 0, 0)
        
        # 4. Refine Alignment Logic
        max_level_per_refine = defaultdict(int)
        for nid in self.nodes:
            if nid in refine_count:
                current_ref_count = refine_count[nid]
                max_level_per_refine[current_ref_count] = max(
                    max_level_per_refine[current_ref_count], 
                    level.get(nid, 0)
                )
        
        final_y_level = {}
        for nid in self.nodes:
            base_lvl = level.get(nid, 0)
            ref = refine_count.get(nid, 0)
            if self.nodes[nid].event_type == "3D auto-refine":
                baseline = max_level_per_refine[ref]
                actual_lvl = level.get(nid, 0)
                final_y_level[nid] = max(baseline, actual_lvl)
            else:
                final_y_level[nid] = base_lvl
                
        # 5. Push children down (Exact Reference Logic)
        # Since we removed cycles above, this while loop is guaranteed to terminate.
        changed = True
        while changed:
            changed = False
            for nid in self.nodes:
                if nid not in graph: continue
                parent_lvl = final_y_level.get(nid, 0)
                for child in graph[nid]:
                    child_lvl = final_y_level.get(child, 0)
                    if child_lvl <= parent_lvl:
                        final_y_level[child] = parent_lvl + 1
                        changed = True

        # 6. Grid Layout Calculation
        y_groups = defaultdict(list)
        for nid in self.nodes:
            y_groups[final_y_level.get(nid, 0)].append(nid)
            
        # 6. Grid Layout Calculation
        y_groups = defaultdict(list)
        for nid in self.nodes:
            y_groups[final_y_level.get(nid, 0)].append(nid)

        # --- MODIFIED: Adaptive Spacing ---
        # Set padding (gap) between nodes
        pad_x = 80   # Horizontal gap between node edges
        pad_y = 100  # Vertical gap between rows
        
        # Calculate step size based on Global Node Sizes
        X_STEP = NODE_W + pad_x
        Y_STEP = NODE_H + pad_y
        MIN_X = 100
        # ----------------------------------

        pos = {}
        used_x_per_row = defaultdict(set)
        
        active_levels = sorted(y_groups.keys())
        
        for i, y_lvl in enumerate(active_levels):
            nodes_in_row = y_groups[y_lvl]
            if not nodes_in_row: continue
            
            y = 100 + i * Y_STEP
            
            groups = defaultdict(list)
            for nid in nodes_in_row:
                pars = [p for p in parents_of[nid] if p in self.nodes]
                if not pars: 
                    groups[None].append(nid)
                else:
                    # Break ties using negative Job Number
                    closest = max(pars, key=lambda p: (
                        final_y_level.get(p, -1),      
                        -int(self.nodes[p].job_num)    
                    ))
                    groups[closest].append(nid)
            
            for chs in groups.values():
                chs.sort(key=lambda n: int(self.nodes[n].job_num))
                
            order = []
            for par, chs in groups.items():
                if par is None: 
                    key = "00000" # Orphans go far left
                else:
                    key = f"{pos[par][0]:020.3f}" if par in pos else "00000"
                order.append((par, chs, key))
            
            # Sort groups by parent X position (Left parents processed first)
            order.sort(key=lambda tup: tup[2])
            
            # TRACKER: Keeps track of the right-most edge of the row so far.
            current_row_limit = MIN_X
            
            for par, chs, _ in order:
                if not chs: continue
                width = (len(chs) - 1) * X_STEP
                mid = width / 2.0
                
                if par and par in pos:
                    ideal_center = pos[par][0]
                else:
                    # If no parent, place after the current limit
                    ideal_center = current_row_limit + X_STEP + mid
                
                start = ideal_center - mid
                
                # COLLISION FIX:
                # If the ideal start is to the left of our current limit, 
                # force it to push right. Never allow back-filling into gaps.
                if start < current_row_limit:
                    start = current_row_limit
                
                # Assign positions
                for i, nid in enumerate(chs):
                    x = start + i * X_STEP
                    pos[nid] = (x, y)
                    used_x_per_row[y_lvl].add(x)
                
                # Update the limit for the next group
                current_row_limit = start + width + X_STEP

        # 7. Apply Positions
        for nid in target_jobs:
            if nid not in pos: continue
            x, y = pos[nid]
            node = self.nodes[nid]
            
            node.x = x
            node.y = y
            
            if nid in self.node_items:
                item = self.node_items[nid]
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, False)
                item.setPos(x, y)
                item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        
        self.redraw_links()
        self.update_minimap()

    def decrease_font(self):
        global BASE_FONT_SIZE
        BASE_FONT_SIZE = max(1, BASE_FONT_SIZE - 1)
        self.font_size_lbl.setText(str(BASE_FONT_SIZE))
        for item in self.node_items.values(): item.update()

    def increase_font(self):
        global BASE_FONT_SIZE
        BASE_FONT_SIZE += 1
        self.font_size_lbl.setText(str(BASE_FONT_SIZE))
        for item in self.node_items.values(): item.update()

    def reset_font(self):
        global BASE_FONT_SIZE
        BASE_FONT_SIZE = 12
        self.font_size_lbl.setText(str(BASE_FONT_SIZE))
        for item in self.node_items.values(): item.update()

    # -----------------------------------------------------------------
    # SELECTION & SEARCH
    # -----------------------------------------------------------------

    def on_job_select_num(self, row):
        if row < 0: return
        nid = self.get_node_by_index(row)
        self.focus_node(nid)

    def on_job_select_detail(self, row):
        if row < 0: return
        nid = self.get_node_by_index(row)
        self.focus_node(nid)

    def get_node_by_index(self, index):
        sorted_nodes = sorted(self.nodes.values(), key=lambda n: int(n.job_num))
        if 0 <= index < len(sorted_nodes):
            return sorted_nodes[index].id
        return None

    def focus_node(self, nid):
        if nid not in self.node_items: return
        item = self.node_items[nid]
        self.view.centerOn(item)
        self.scene.clearSelection()
        item.setSelected(True)
        # Flash
        item.flash_active = True
        item.update()
        QTimer.singleShot(600, lambda: self._stop_flash(item))

    def _stop_flash(self, item):
        item.flash_active = False
        item.update()

    def perform_search(self):
        q = self.search_edit.text().strip().lower()
        if not q: return
        
        matches = []
        for n in self.nodes.values():
            if (q in n.job_num or 
               (n.alias and q in n.alias.lower()) or 
               q in n.event_type.lower()):
                matches.append(n)
        
        if not matches:
             QMessageBox.information(self, "Search", "No matches")
             return

        # Close previous window if it exists to avoid duplicates
        if hasattr(self, 'search_window') and self.search_window is not None:
            self.search_window.close()
             
        self.search_window = QDialog(self)
        self.search_window.setWindowTitle(f"Search Results ({len(matches)})")
        self.search_window.resize(600, 500)
        
        # Make it non-modal (allows interaction with main window while open)
        self.search_window.setModal(False)
        
        layout = QVBoxLayout(self.search_window)
        list_widget = QListWidget()
        layout.addWidget(list_widget)
        
        matches.sort(key=lambda x: int(x.job_num))
        
        for node in matches:
            # --- DISPLAY LOGIC ---
            # Shows: "001 Import (MyAlias)"
            display_text = f"{node.job_num} {node.event_type}"
            if node.alias:
                display_text += f" ({node.alias})"
            # ---------------------
            
            item = QListWidgetItem(display_text)
            color_code = self.type_colors.get(node.event_type, "black")
            item.setForeground(QColor(color_code))
            item.setData(Qt.ItemDataRole.UserRole, node.id)
            list_widget.addItem(item)
            
        def on_item_clicked(item):
            nid = item.data(Qt.ItemDataRole.UserRole)
            self.focus_node(nid)
            
        list_widget.itemClicked.connect(on_item_clicked)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.search_window.close)
        layout.addWidget(close_btn)
        
        self.search_window.show()
            
        def on_item_clicked(item):
            nid = item.data(Qt.ItemDataRole.UserRole)
            self.focus_node(nid)
            # Window stays open; main window updates immediately
            
        list_widget.itemClicked.connect(on_item_clicked)
        
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.search_window.close)
        layout.addWidget(close_btn)
        
        # FIX 4: Use .show() instead of .exec()
        self.search_window.show()

    def on_job_list_context(self, pos):
        # Identify list
        sender = self.sender()
        item = sender.itemAt(pos)
        if not item: return
        row = sender.row(item)
        nid = self.get_node_by_index(row)
        if nid:
            self.show_context_menu(self.nodes[nid], QCursor.pos())

    # -----------------------------------------------------------------
    # STICKY NOTES & HOVER
    # -----------------------------------------------------------------

    def toggle_sticky_notes(self, checked):
        self.sticky_notes_enabled = checked
        if not checked:
            self._hide_sticky_note()

    def on_node_enter(self, item, event):
        if self.sticky_notes_enabled and item.node_data.notes.strip():
            self.current_hovered_node_item = item
            self.hover_timer.start(500)

    def on_node_leave(self, item, event):
        self.hover_timer.stop()
        if self.current_hovered_node_item == item:
            self.current_hovered_node_item = None
            self._hide_sticky_note()

    def _show_sticky_note(self):
        if not self.current_hovered_node_item: return
        note_text = self.current_hovered_node_item.node_data.notes
        
        self._hide_sticky_note()
        
        note = StickyNoteItem(note_text, 480, self)
        pos = self.current_hovered_node_item.pos()
        # Offset above node
        note.setPos(pos.x(), pos.y() - 40) 
        
        self.scene.addItem(note)
        self.current_sticky_item = note

    def _hide_sticky_note(self):
        if self.current_sticky_item:
            self.scene.removeItem(self.current_sticky_item)
            self.current_sticky_item = None

    # -----------------------------------------------------------------
    # EXTRA FEATURES (Isolate, 3D Refine, etc)
    # -----------------------------------------------------------------

    def update_links_visibility(self, checked):
        self.hide_linkers = checked
        self.redraw_links()

    def clear_chains(self):
        self.selected_node_for_links.clear()
        self.redraw_links()

    def update_node_notes(self, node, updates):
        """
        updates is a dict: 
        {
            'twist_rise': str, 
            'refine_res': str, 
            'post_res': str,
            'b_factor': str,
            'optics_lines': list of str,
            'total_particles': int
        }
        """
        if not any(updates.values()): return

        # Regex patterns to identify lines we want to replace/update
        # We use \.? to match lines with OR without periods, so we can replace them cleanly.
        pat_twist = re.compile(r'^helical twist = .*$', re.IGNORECASE)
        pat_refine_res = re.compile(r'^Final resolution \(.*\) is: .*$', re.IGNORECASE)
        pat_post_res = re.compile(r'^Final resolution: .*$', re.IGNORECASE)
        pat_bfactor = re.compile(r'^B-factor: .*$', re.IGNORECASE)
        pat_optics_line = re.compile(r'^For optics_group .* particles on the scratch disk\.?$', re.IGNORECASE)
        pat_total_part = re.compile(r'^Total \d+ particles are used\.?$', re.IGNORECASE)

        # 1. Separation: Keep User notes, discard old Auto notes that we are about to update
        current_lines = node.notes.split('\n') if node.notes else []
        kept_lines = []

        for line in current_lines:
            s = line.strip()
            
            # Check if line matches our auto-generated patterns
            is_auto_optics = (pat_optics_line.match(s) or pat_total_part.match(s))
            is_auto_twist = pat_twist.match(s)
            is_auto_refine = pat_refine_res.match(s)
            is_auto_post = pat_post_res.match(s)
            is_auto_bfactor = pat_bfactor.match(s)

            # If it matches a pattern AND we have new data to replace it, skip (remove) the old line.
            if is_auto_optics and (updates.get('optics_lines') is not None): continue
            if is_auto_twist and updates.get('twist_rise'): continue
            if is_auto_refine and updates.get('refine_res'): continue
            if is_auto_post and updates.get('post_res'): continue
            if is_auto_bfactor and updates.get('b_factor'): continue
            
            kept_lines.append(line)

        # 2. Construction: Rebuild notes with User Notes first, then Auto Notes
        final_lines = list(kept_lines)

        # Remove trailing empty lines to keep spacing clean before appending new info
        while final_lines and not final_lines[-1].strip():
            final_lines.pop()

        # Add divider space if there were existing notes
        if final_lines:
            final_lines.append("")

        # -- Append Optics Info & Total Count --
        if updates.get('optics_lines'):
            for opt_line in updates['optics_lines']:
                # FORCE REMOVE PERIOD
                final_lines.append(opt_line.rstrip('.'))
            
            if updates.get('total_particles') is not None:
                # FORCE REMOVE PERIOD (in case it crept in)
                s = f"Total {updates['total_particles']} particles are used"
                final_lines.append(s.rstrip('.'))
            
            final_lines.append("") # Spacing

        # -- Append Twist/Rise --
        if updates.get('twist_rise'):
            s = f"helical twist = {updates['twist_rise']}"
            final_lines.append(s.rstrip('.'))
        
        # -- Append Refine Resolution --
        if updates.get('refine_res'):
            s = f"Final resolution (already with masking) is: {updates['refine_res']}"
            final_lines.append(s.rstrip('.'))
            
        # -- Append PostProcess Resolution & B-factor --
        if updates.get('post_res'):
            s = f"Final resolution: {updates['post_res']}"
            final_lines.append(s.rstrip('.'))
        
        if updates.get('b_factor'):
            if updates.get('post_res'):
                final_lines.append("") # Add empty line between resolution and B-factor
            s = f"B-factor: {updates['b_factor']}"
            final_lines.append(s.rstrip('.'))

        # Join and strip cleanup
        node.notes = '\n'.join(final_lines).strip()

    def import_job_info(self):
        if not self.current_workflow:
            QMessageBox.information(self, "Info", "Please select a workflow first.")
            return
            
        # Filter for relevant nodes
        target_nodes = [
            n for n in self.nodes.values() 
            if n.event_type in ["3D auto-refine", "Post-processing"]
        ]
        
        if not target_nodes:
            QMessageBox.information(self, "Info", "No Refine3D or PostProcess jobs found.")
            return

        updated_count = 0
        
        for node in target_nodes:
            updates = {}
            job_num = node.job_num
            
            # --- LOGIC FOR REFINE 3D ---
            if node.event_type == "3D auto-refine":
                out_path = os.path.join(ROOT_DIR, "Refine3D", f"job{job_num}", "run.out")
                if os.path.isfile(out_path):
                    try:
                        # READ ALL lines to catch optics (head) and resolution (tail)
                        with open(out_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                        
                        optics_lines = []
                        total_particles = 0

                        for line in lines:
                            s = line.strip()
                            
                            # 1. Parse Optics Groups
                            if "For optics_group" in s and "particles on the scratch disk" in s:
                                m = re.search(r'there are (\d+) particles', s)
                                if m:
                                    count = int(m.group(1))
                                    total_particles += count
                                    # Strip the period immediately when reading from file
                                    optics_lines.append(s.rstrip('.'))

                            # 2. Twist/Rise
                            if "helical twist =" in s and "rise =" in s:
                                try:
                                    # We get the values, strip periods from values just in case
                                    twist = s.split("helical twist =")[1].split("degrees")[0].strip().rstrip('.')
                                    rise = s.split("rise =")[1].split("Angstroms")[0].strip().rstrip('.')
                                    updates['twist_rise'] = f"{twist} degrees, rise = {rise} Angstroms"
                                except: pass
                            
                            # 3. Resolution
                            if "Final resolution" in s and "is:" in s:
                                try: 
                                    updates['refine_res'] = s.split("is:")[1].strip().rstrip('.')
                                except: pass
                        
                        if optics_lines:
                            updates['optics_lines'] = optics_lines
                            updates['total_particles'] = total_particles

                    except Exception as e:
                        print(f"[import_job_info] Error reading Refine3D job {job_num}: {e}")

            # --- LOGIC FOR POST PROCESS ---
            elif node.event_type == "Post-processing":
                out_path = os.path.join(ROOT_DIR, "PostProcess", f"job{job_num}", "run.out")
                if os.path.isfile(out_path):
                    try:
                        with open(out_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()[-100:]
                            
                        for line in lines:
                            s = line.strip()
                            
                            # 1. Capture Resolution
                            if "+ FINAL RESOLUTION:" in s:
                                try:
                                    val = s.split("FINAL RESOLUTION:")[1].strip().rstrip('.')
                                    updates['post_res'] = val
                                except: pass
                            
                            # 2. Capture B-factor
                            # Line looks like: " + apply b-factor of:      -46.1867"
                            if "+ apply b-factor of:" in s:
                                try:
                                    val = s.split("apply b-factor of:")[1].strip().rstrip('.')
                                    updates['b_factor'] = val
                                except: pass
                                
                    except Exception as e:
                        print(f"[import_job_info] Error reading PostProcess job {job_num}: {e}")

            # Apply updates if any found
            if updates:
                self.update_node_notes(node, updates)
                updated_count += 1
                
        if updated_count:
            # Refresh UI
            if self.current_sticky_item:
                self._hide_sticky_note()
                
            for nid in self.node_items:
                self.node_items[nid].update()
            QMessageBox.information(self, "Success", f"Imported info for {updated_count} job(s).")
        else:
            QMessageBox.information(self, "No change", "No new information found in run.out files.")

    def isolate_jobs(self):
        dlg = SimpleInput(self, "Isolate", "Job Numbers (comma sep):")
        if dlg.exec() and dlg.result:
            # Parse inputs
            nums = [x.strip().zfill(3) for x in dlg.result.replace(',', ' ').split() if x.strip()]
            if not nums: return
            
            # Check if jobs exist
            target_ids = {n.id for n in self.nodes.values() if n.job_num in nums}
            if not target_ids:
                QMessageBox.warning(self, "Error", "No matching jobs found.")
                return

            # Open Isolation Window (Pass self and the targets)
            self.iso_win = IsolationWindow(self, target_ids)
            self.iso_win.show()

# --------------------------------------------------------------------- #
# ISOLATION WINDOW
# --------------------------------------------------------------------- #
class IsolationWindow(MindMapApp):
    def __init__(self, parent_app, target_ids):
        # Initialize as a separate window, do NOT load initial data
        super().__init__(load_initial_data=False)
        self.parent_app = parent_app
        self.target_ids = target_ids
        self.setWindowTitle("Isolated View")
        self.resize(1200, 800)
        
        # 1. Disable Management Controls
        self.wf_combo.setEnabled(False)
        self.btn_create.setEnabled(False)
        self.btn_rename.setEnabled(False)
        self.btn_copy.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_import.setEnabled(False)
        self.btn_cont_import.setEnabled(False)

        # 2. Create the Checkbox
        self.show_children_cb = QCheckBox("Show Children")
        self.show_children_cb.setChecked(False) # Default OFF
        self.show_children_cb.toggled.connect(self.rebuild_scene)
        
        # 3. Insert Checkbox into Toolbar 2 (Same row as Isolate)
        # We traverse the layout to find Toolbar 2
        central = self.centralWidget()
        if central:
            # Hierarchy: Central -> Main(HBox) -> Left(VBox) -> Toolbar2(Frame)
            main_layout = central.layout()
            if main_layout and main_layout.count() > 0:
                left_layout = main_layout.itemAt(0).layout()
                if left_layout and left_layout.count() > 1:
                    # Toolbar 1 is index 0, Toolbar 2 is index 1
                    tb2_item = left_layout.itemAt(1)
                    if tb2_item and tb2_item.widget():
                        tb2_frame = tb2_item.widget()
                        tb2_layout = tb2_frame.layout()
                        
                        # We want to insert 'Next to Isolate'.
                        # In init_ui, 'Isolate' is added, then 'addStretch()'.
                        # The stretch is the last item. We insert before the stretch.
                        count = tb2_layout.count()
                        tb2_layout.insertWidget(count - 1, self.show_children_cb)
                        
                        # Add a little spacing
                        tb2_layout.insertSpacing(count - 1, 10)

        # 4. Build the View
        self.rebuild_scene()

    def rebuild_scene(self):
        """Re-calculates the visible graph based on targets and checkbox state."""
        self.scene.clear()
        self.nodes.clear()
        self.links.clear()
        self.node_items.clear()
        self.link_items.clear()
        self.selected_node_for_links.clear()

        show_children = self.show_children_cb.isChecked()
        visible_ids = self._calculate_chain(self.target_ids, show_children)
        
        # Populate Nodes (Copy from Parent)
        min_x, max_x, min_y, max_y = 99999, -99999, 99999, -99999
        
        for nid in visible_ids:
            if nid not in self.parent_app.nodes: continue
            orig = self.parent_app.nodes[nid]
            
            # Create a COPY of the node data using SAME ID
            new_node = NodeData(orig.job_num, orig.event_type, orig.alias, orig.x, orig.y, orig.notes, orig.id)
            self.nodes[new_node.id] = new_node
            
            item = NodeItem(new_node, self)
            item.positionChanged.connect(self.on_node_move)
            self.scene.addItem(item)
            self.node_items[new_node.id] = item
            
            min_x = min(min_x, new_node.x)
            max_x = max(max_x, new_node.x)
            min_y = min(min_y, new_node.y)
            max_y = max(max_y, new_node.y)

        # Populate Links
        for src, dst in self.parent_app.links:
            if src in visible_ids and dst in visible_ids:
                self.links.append((src, dst))
        
        self.redraw_links()
        
        # Center View if nodes exist
        if self.nodes:
            cx = (min_x + max_x) / 2
            cy = (min_y + max_y) / 2
            self.view.centerOn(cx, cy)
            
        self.update_sidebar()
        self.update_minimap()

    def _calculate_chain(self, targets, show_children):
        # Build Adjacency Maps
        parents_of = defaultdict(list)
        children_of = defaultdict(list)
        
        for src, dst in self.parent_app.links:
            children_of[src].append(dst)
            parents_of[dst].append(src)
            
        visited = set()
        
        # 1. Upstream (Parents) - Always
        queue = list(targets)
        seen = set(targets)
        
        while queue:
            curr = queue.pop(0)
            visited.add(curr)
            for parent in parents_of[curr]:
                if parent not in seen:
                    seen.add(parent)
                    queue.append(parent)
                    
        # 2. Downstream (Children) - Only if checked
        if show_children:
            queue = list(targets)
            # Note: seen set continues from above to avoid re-visiting targets
            while queue:
                curr = queue.pop(0)
                visited.add(curr)
                for child in children_of[curr]:
                    if child not in seen:
                        seen.add(child)
                        queue.append(child)
                        
        return visited

    def closeEvent(self, event):
        """Merge changes back to Parent App."""
        for nid, node in self.nodes.items():
            if nid in self.parent_app.nodes:
                parent_node = self.parent_app.nodes[nid]
                
                # Merge properties
                parent_node.x = node.x
                parent_node.y = node.y
                parent_node.notes = node.notes
                parent_node.alias = node.alias
                
                # Update main window visual
                if nid in self.parent_app.node_items:
                    item = self.parent_app.node_items[nid]
                    item.setPos(node.x, node.y)
                    item.update() 

        self.parent_app.redraw_links()
        self.parent_app.update_minimap()
        event.accept()

def apply_dark_theme(app):
    """Sets the application to a dark theme with white text."""
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    
    # Text OUTSIDE the nodes (General UI) remains WHITE
    palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
    
    palette.setColor(QPalette.ColorRole.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
    
    # General Text Element Color
    palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
    
    palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
    palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.white)
    palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(palette)
    
    app.setStyleSheet("""
        QPushButton {
            background-color: #3e3e42; color: #d4d4d4; border: 1px solid #3e3e42;
            padding: 5px 15px; border-radius: 4px;
        }
        QPushButton:hover { background-color: #4e4e52; border: 1px solid #98c379; }
        QPushButton:pressed, QPushButton:checked { background-color: #98c379; color: #1e1e1e; border: 1px solid #98c379; }
    """)

# --------------------------------------------------------------------- #
# ENTRY POINT
# --------------------------------------------------------------------- #

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion")) # Better cross-platform look
    # APPLY THEME HERE
    apply_dark_theme(app)
    window = MindMapApp()
    window.show()
    
    sys.exit(app.exec())