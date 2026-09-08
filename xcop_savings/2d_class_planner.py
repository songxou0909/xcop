import os
# Limit numpy/scipy/sklearn C-backend threads to prevent HPC CPU policy violations
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import AgglomerativeClustering
except Exception as e:
    print(f"Warning: sklearn failed to load at startup: {e}")

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
    QGraphicsPixmapItem, QWidget, QVBoxLayout, QPushButton, 
    QFileDialog, QInputDialog, QListWidget, QListWidgetItem, 
    QSplitter, QLabel, QHBoxLayout, QMessageBox, QDialog, QFrame,
    QGraphicsItem, QStyle, QComboBox, QCheckBox, 
    QSpinBox, QFormLayout, QDialogButtonBox, QLineEdit, QListView, QAbstractItemView,
    QProgressBar
)
from PyQt6.QtCore import (
    Qt, QMimeData, pyqtSignal as Signal, QObject, QPointF, 
    QSize, QPoint, QRectF, QThread
)
from PyQt6.QtGui import (
    QPixmap, QPainter, QDragEnterEvent, QDropEvent, 
    QIcon, QColor, QDrag, QMouseEvent, QWheelEvent, QKeyEvent,
    QPainterPath, QPen, QBrush, QKeySequence, QPalette, QImage, QAction, QFont
)
from PyQt6.QtSvg import QSvgGenerator
import numpy as np
import mrcfile
import os
import glob
import json

# --- DARK MODE THEME SETUP ---
def set_dark_theme(app):
    app.setStyle("Fusion")
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
    dark_palette.setColor(QPalette.ColorRole.Link, QColor("#98c379"))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor("#98c379"))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    
    app.setStyleSheet("""
        QWidget { background-color: #2b2b2b; color: #FFFFFF; }
        QLineEdit, QTextEdit, QListWidget, QSpinBox { background-color: #1e1e1e; border: 1px solid #444; color: white; padding: 4px; border-radius: 2px; selection-background-color: #98c379; selection-color: black; }
        QLineEdit:focus, QTextEdit:focus, QListWidget:focus, QSpinBox:focus { border: 1px solid #98c379; }
        
        QPushButton { background-color: #3e3e3e; border: 1px solid #3e3e3e; border-radius: 5px; padding: 8px; color: white; outline: none; }
        QPushButton:hover { background-color: #3e3e3e; border: 1px solid #98c379; }
        QPushButton:pressed { background-color: #2b2b2b; border: 1px solid #98c379; }
        QPushButton:disabled { background-color: #555; border: 1px solid #555; color: #888; }
        
        QSplitter::handle { background-color: #444; }
        QListWidget::item { padding: 5px; font-size: 14px; }
        QListWidget::item:hover { background-color: #444; }
        QListWidget::item:selected { background-color: #3e3e3e; border: 1px solid #98c379; color: white; }
        
        QDialog { border: 2px solid #555; background-color: #2b2b2b; color: white; }
        QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid #555; }
    """)

# --- 1. Grid Selector Widget ---
class GridSelectorWidget(QWidget):
    selectionConfirmed = Signal(int, int) # rows, cols
    selectionChanged = Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.cell_size = 30 
        self.max_rows = 10
        self.max_cols = 10
        self.selected_row = 0
        self.selected_col = 0
        
        self.setFixedSize(self.max_cols * self.cell_size + 1, self.max_rows * self.cell_size + 1)

    def mouseMoveEvent(self, event: QMouseEvent):
        col = int(event.position().x() // self.cell_size) + 1
        row = int(event.position().y() // self.cell_size) + 1
        
        col = max(1, min(col, self.max_cols))
        row = max(1, min(row, self.max_rows))
        
        if col != self.selected_col or row != self.selected_row:
            self.selected_col = col
            self.selected_row = row
            self.selectionChanged.emit(row, col)
            self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selectionConfirmed.emit(self.selected_row, self.selected_col)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        
        # Background Grid (Lighter gray for visibility)
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        
        for r in range(self.max_rows):
            for c in range(self.max_cols):
                painter.drawRect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)

        # Highlight Selection (GREEN)
        if self.selected_row > 0 and self.selected_col > 0:
            # Thick Green Lines
            painter.setPen(QPen(QColor("#98c379"), 2))
            # Semi-transparent Green Fill (RGB 152, 195, 121 with 50 opacity)
            painter.setBrush(QBrush(QColor(152, 195, 121, 50))) 
            
            for r in range(self.selected_row):
                for c in range(self.selected_col):
                    painter.drawRect(c * self.cell_size, r * self.cell_size, self.cell_size, self.cell_size)

# --- 2. Grid Selection Dialog ---
class GridSelectionDialog(QDialog):
    def __init__(self, pixmap, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Grid Size")
        self.setWindowFlags(Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setModal(True)
        
        self.result_rows = 0
        self.result_cols = 0

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(20) 

        # LEFT: Preview
        preview_container = QWidget()
        preview_layout = QVBoxLayout(preview_container)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        
        lbl_title = QLabel("Preview")
        lbl_title.setStyleSheet("font-weight: bold; color: #98c379;")
        preview_layout.addWidget(lbl_title)

        preview_lbl = QLabel()
        scaled_pix = pixmap.scaled(350, 350, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        preview_lbl.setPixmap(scaled_pix)
        preview_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview_lbl.setStyleSheet("border: 1px solid #555; background-color: black;")
        preview_layout.addWidget(preview_lbl)
        
        main_layout.addWidget(preview_container)

        # RIGHT: Grid
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_status = QLabel("Hover to select")
        self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color: #98c379;")
        font = self.lbl_status.font()
        font.setBold(True)
        font.setPointSize(12)
        self.lbl_status.setFont(font)
        right_layout.addWidget(self.lbl_status)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        right_layout.addWidget(line)

        right_layout.addStretch()
        grid_hbox = QHBoxLayout()
        grid_hbox.addStretch()
        self.grid_widget = GridSelectorWidget()
        self.grid_widget.selectionChanged.connect(self.update_label)
        self.grid_widget.selectionConfirmed.connect(self.confirm_selection)
        grid_hbox.addWidget(self.grid_widget)
        grid_hbox.addStretch()
        
        right_layout.addLayout(grid_hbox)
        right_layout.addStretch()

        main_layout.addWidget(right_container)

    def update_label(self, rows, cols):
        self.lbl_status.setText(f"{rows} x {cols}")

    def confirm_selection(self, rows, cols):
        self.result_rows = rows
        self.result_cols = cols
        self.accept()

# --- Communicator ---
class Communicate(QObject):
    highlight_stock = Signal(int)
    unhighlight_stock = Signal(int)
    stock_used = Signal(int)
    stock_unused = Signal(int)
    item_moved = Signal()
    locate_stock = Signal(int)

# --- The Canvas Item (Tile) ---
class CanvasTileItem(QGraphicsPixmapItem):
    def __init__(self, pixmap, tile_id, comms):
        super().__init__(pixmap)
        self.tile_id = tile_id
        self.comms = comms
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemSendsScenePositionChanges
        )
        self.setAcceptHoverEvents(True)
        self.setShapeMode(QGraphicsPixmapItem.ShapeMode.BoundingRectShape)
        self.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
        
        # Force smooth blending (bilinear filtering) exactly like quick_map
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        
        self.is_rotating = False
        self.last_mouse_pos = QPointF()

    def paint(self, painter, option, widget=None):
        if not self.scene():
            super().paint(painter, option, widget)
            return

        # Smart Blending Logic
        colliding_items = self.scene().collidingItems(self, Qt.ItemSelectionMode.IntersectsItemShape)
        items_below = [item for item in colliding_items 
                       if isinstance(item, CanvasTileItem) and item.zValue() < self.zValue()]

        region_map = [(self.shape(), 0)]

        for other in items_below:
            other_shape_local = self.mapFromItem(other, other.shape())
            new_region_map = []
            for path, count in region_map:
                intersect = path.intersected(other_shape_local)
                diff = path.subtracted(other_shape_local)
                if not diff.isEmpty(): new_region_map.append((diff, count))
                if not intersect.isEmpty(): new_region_map.append((intersect, count + 1))
            region_map = new_region_map

        # Enforce smooth transformation during manual custom painting
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        for path, count in region_map:
            painter.save()
            painter.setClipPath(path)
            # Opacity = 1 / (Overlap Count + 1)
            opacity = 1.0 / (count + 1)
            painter.setOpacity(opacity)
            painter.drawPixmap(0, 0, self.pixmap())
            painter.restore()

        if self.isSelected():
            painter.setPen(QPen(QColor(255, 255, 0, 200), 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.boundingRect())

    def mousePressEvent(self, event):
        items = self.scene().items()
        if items: self.setZValue(max(i.zValue() for i in items) + 0.1)
        
        if event.button() == Qt.MouseButton.RightButton:
            self.is_rotating = True
            self.last_mouse_pos = event.scenePos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)
        self.scene().update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.comms.locate_stock.emit(self.tile_id)
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_rotating:
            diff = event.scenePos().x() - self.last_mouse_pos.x()
            self.setRotation(self.rotation() + diff)
            self.last_mouse_pos = event.scenePos()
            event.accept()
            self.scene().update()
        else:
            super().mouseMoveEvent(event)
            self.scene().update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton:
            self.is_rotating = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)
        self.scene().update()
        self.comms.item_moved.emit()

    def hoverEnterEvent(self, event):
        self.comms.highlight_stock.emit(self.tile_id)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.comms.unhighlight_stock.emit(self.tile_id)
        super().hoverLeaveEvent(event)

# --- The Canvas Scene ---
class EditorScene(QGraphicsScene):
    def __init__(self, comms):
        super().__init__()
        self.comms = comms
        # Massively expanded canvas to prevent clipping on large 2D class jobs
        self.setSceneRect(-100000, -100000, 200000, 200000)

    def dragEnterEvent(self, event): event.accept()
    def dragMoveEvent(self, event): event.accept()
    def dropEvent(self, event): event.accept()

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
            for item in self.selectedItems():
                # 1. Remove from scene FIRST so the canvas count is accurate
                self.removeItem(item)
                
                # 2. THEN emit the signal to update the HUD and stock colors
                if isinstance(item, CanvasTileItem):
                    self.comms.stock_unused.emit(item.tile_id)
                    self.comms.unhighlight_stock.emit(item.tile_id)
            self.update()
        else:
            super().keyPressEvent(event)

# --- The Graphics View ---
class EditorView(QGraphicsView):
    def __init__(self, scene, main_window):
        super().__init__(scene)
        self.main_window = main_window
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setAcceptDrops(True)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self._is_panning = False
        self._pan_start_pos = QPoint()
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    # ZOOM: Scroll Wheel
    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    # PAN: Right Click Drag on Empty Space
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.RightButton and not self.itemAt(event.position().toPoint()):
            self._is_panning = True
            self._pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_panning:
            delta = event.position().toPoint() - self._pan_start_pos
            self._pan_start_pos = event.position().toPoint()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event): event.accept()
    def dragMoveEvent(self, event): event.accept()
    
    def dropEvent(self, event):
        if event.mimeData().hasFormat("application/x-tile-id"):
            try:
                tile_id = int(event.mimeData().data("application/x-tile-id").data().decode())
                pixmap = self.main_window.tile_data.get(tile_id)
                if pixmap:
                    scene_pos = self.mapToScene(event.position().toPoint())
                    item = CanvasTileItem(pixmap, tile_id, self.main_window.comms)
                    item.setPos(scene_pos.x() - pixmap.width()/2, scene_pos.y() - pixmap.height()/2)
                    
                    if self.scene().items():
                        item.setZValue(max(i.zValue() for i in self.scene().items()) + 1)
                    
                    self.scene().clearSelection() # Deselect everything currently on canvas
                    self.scene().addItem(item)
                    # We removed item.setSelected(True) so it drops cleanly
                    self.scene().update()
                    self.main_window.comms.stock_used.emit(tile_id)

            except Exception as e: print(f"Error: {e}")
            event.accept()
        else: super().dropEvent(event)

# --- The Stock List Widget ---
class StockListWidget(QListWidget):
    def __init__(self):
        super().__init__()
        self.setIconSize(QSize(100, 100))
        self.setDragEnabled(True)
        self.setViewMode(QListView.ViewMode.IconMode)
        self.setSpacing(10)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setResizeMode(QListView.ResizeMode.Adjust)

    # --- NEW: Click on empty space clears selection ---
    def mousePressEvent(self, event):
        # Check if the click is on an item or empty space
        if not self.itemAt(event.position().toPoint()):
            self.clearSelection()
            self.setCurrentItem(None)
        
        # Propagate to parent to handle item selection/dragging
        super().mousePressEvent(event)

    def startDrag(self, supportedActions):
        item = self.currentItem()
        if not item: return
        tile_id = item.data(Qt.ItemDataRole.UserRole)
        mime = QMimeData()
        mime.setData("application/x-tile-id", str(tile_id).encode())
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.setPixmap(item.icon().pixmap(50, 50))
        drag.setHotSpot(QPoint(25, 25))

        # 1. Clear selection so custom backgrounds can show during the drag
        self.clearSelection()
        
        # 2. Clear current item (Removes the stuck blue text box) during the drag
        self.setCurrentItem(None)

        drag.exec(Qt.DropAction.CopyAction)
        
        # 3. Force the widget to realize the mouse has left (Clears stuck CSS grey hover state)
        from PyQt6.QtCore import QEvent
        QApplication.postEvent(self.viewport(), QEvent(QEvent.Type.Leave))
        
        # Clear the selection so the stylesheet doesn't block the custom green background
        self.clearSelection()

# --- 3. Import Job Dialog ---
class ImportJobDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import 2D Class Job")
        
        layout = QFormLayout(self)
        
        self.job_input = QLineEdit()
        layout.addRow("Job Number:", self.job_input)
        
        self.metric_combo = QComboBox()
        self.metric_combo.addItems(["rlnClassDistribution", "rlnEstimatedResolution"])
        layout.addRow("Sort by:", self.metric_combo)
        
        self.reverse_check = QCheckBox()
        self.reverse_check.setChecked(True)
        layout.addRow("Reverse sort (highest first):", self.reverse_check)
        
        self.items_spin = QSpinBox()
        self.items_spin.setRange(1, 20)
        self.items_spin.setValue(5)
        layout.addRow("Items per row:", self.items_spin)
        
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)

    def get_values(self):
        job_num = self.job_input.text().strip()
        
        # Auto-pad with zeros if the user just types "12" instead of "012"
        if job_num.isdigit() and len(job_num) < 3:
            job_num = job_num.zfill(3)
            
        metric = f"_{self.metric_combo.currentText()}"
        reverse_sort = self.reverse_check.isChecked()
        cols = self.items_spin.value()
        return job_num, metric, reverse_sort, cols

# --- Background Import Thread Removed to prevent OpenMP crashes ---

# --- Export Thread ---
class ExportJobThread(QThread):
    progress_signal = Signal(int, str)
    finished_signal = Signal(int, int) # classes_saved, particles_saved
    error_signal = Signal(str)

    def __init__(self, select_dir, selected_classes, total_classes, current_model_star, current_data_star, relion_version, job_num):
        super().__init__()
        self.select_dir = select_dir
        self.selected_classes = selected_classes
        self.total_classes = total_classes
        self.current_model_star = current_model_star
        self.current_data_star = current_data_star
        self.relion_version = relion_version
        self.job_num = job_num

    def run(self):
        try:
            self.progress_signal.emit(5, "Writing backup_selection.star...")
            self._write_backup_selection()
            
            self.progress_signal.emit(10, "Writing class_averages.star...")
            self._write_class_averages()
            
            self.progress_signal.emit(20, "Writing particles.star (Memory Optimized)...")
            particle_count = self._write_particles_optimized()
            
            self.progress_signal.emit(95, "Finalizing job files...")
            self._write_run_out(particle_count)
            
            self.progress_signal.emit(100, "Done!")
            self.finished_signal.emit(len(self.selected_classes), particle_count)
        except Exception as e:
            self.error_signal.emit(str(e))

    def _write_backup_selection(self):
        out_path = os.path.join(self.select_dir, "backup_selection.star")
        with open(out_path, 'w') as f:
            f.write(f"\n# version {self.relion_version}\n\ndata_\n\nloop_ \n_rlnSelected #1 \n")
            for i in range(1, self.total_classes + 1):
                val = 1 if i in self.selected_classes else 0
                f.write(f"           {val} \n")
            f.write(" \n")

    def _write_class_averages(self):
        out_path = os.path.join(self.select_dir, "class_averages.star")
        with open(self.current_model_star, 'r') as f_in, open(out_path, 'w') as f_out:
            f_out.write(f"\n# version {self.relion_version}\n\ndata_\n\nloop_ \n")
            in_classes = False
            headers = []
            has_class_num = False
            class_num_idx = -1
            row_idx = 1
            
            for line in f_in:
                stripped = line.strip()
                if stripped.startswith('data_model_classes'):
                    in_classes = True
                    continue
                if in_classes and stripped.startswith('data_'):
                    break
                    
                if in_classes:
                    if stripped.startswith('_rln'):
                        headers.append(stripped.split()[0])
                        f_out.write(line)
                        if stripped.startswith('_rlnClassNumber'):
                            has_class_num = True
                            class_num_idx = len(headers) - 1
                    elif stripped and not stripped.startswith('#') and not stripped.startswith('loop_'):
                        if not has_class_num and row_idx == 1:
                            f_out.write(f"_rlnClassNumber #{len(headers) + 1} \n")
                            
                        class_num = -1
                        if has_class_num:
                            parts = stripped.split()
                            if len(parts) > class_num_idx:
                                class_num = int(parts[class_num_idx])
                        else:
                            class_num = row_idx
                            
                        if class_num in self.selected_classes:
                            if not has_class_num:
                                f_out.write(f"{line.rstrip()}{class_num:13d} \n")
                            else:
                                f_out.write(line)
                        row_idx += 1
            f_out.write(" \n")

    def _write_particles_optimized(self):
        out_path = os.path.join(self.select_dir, "particles.star")
        saved_count = 0
        
        total_size = os.path.getsize(self.current_data_star)
        processed_size = 0
        last_percent = -1
        
        in_particles = False
        passed_particles = False
        has_data = False
        headers = []
        class_num_idx = -1
        
        # Memory optimization: Only store selected particles
        class_groups = {c: [] for c in self.selected_classes}
        pre_and_headers = []
        post_lines = []
        
        with open(self.current_data_star, 'r') as f_in:
            for line in f_in:
                processed_size += len(line.encode('utf-8'))
                current_percent = 20 + int(70 * (processed_size / total_size))
                
                # Emit progress efficiently without spamming the UI thread
                if current_percent > last_percent:
                    self.progress_signal.emit(current_percent, f"Reading particles.star... {current_percent}%")
                    last_percent = current_percent
                
                stripped = line.strip()
                
                if stripped.startswith('data_particles'):
                    in_particles = True
                    pre_and_headers.append(line)
                    continue
                    
                if in_particles and stripped.startswith('data_'):
                    in_particles = False
                    passed_particles = True
                    post_lines.append(line)
                    continue
                    
                if in_particles:
                    if stripped.startswith('_rln'):
                        headers.append(stripped.split()[0])
                        if stripped.startswith('_rlnClassNumber'):
                            class_num_idx = len(headers) - 1
                        pre_and_headers.append(line)
                    elif stripped.startswith('#') or stripped.startswith('loop_'):
                        pre_and_headers.append(line)
                    elif not stripped:
                        if has_data:
                            post_lines.append(line)
                        else:
                            pre_and_headers.append(line)
                    else:
                        has_data = True
                        parts = stripped.split()
                        if len(parts) >= len(headers) and class_num_idx != -1:
                            class_num = int(parts[class_num_idx])
                            if class_num in self.selected_classes:
                                class_groups[class_num].append(line)
                                saved_count += 1
                        else:
                            pre_and_headers.append(line)
                elif passed_particles:
                    post_lines.append(line)
                else:
                    pre_and_headers.append(line)
                    
        self.progress_signal.emit(90, "Sorting and writing particles...")
        with open(out_path, 'w') as f_out:
            for line in pre_and_headers:
                f_out.write(line)
                
            sorted_classes = sorted(class_groups.keys(), key=lambda c: len(class_groups[c]), reverse=True)
            for class_num in sorted_classes:
                for line in class_groups[class_num]:
                    f_out.write(line)
                    
            for line in post_lines:
                f_out.write(line)
                
        return saved_count

    def _write_run_out(self, particle_count):
        with open(os.path.join(self.select_dir, "run.out"), 'w') as f:
            f.write(f"xcop - Saved Select/job{self.job_num}/class_averages.star with {len(self.selected_classes)} selected images.\n")
            f.write(f"xcop - Saved Select/job{self.job_num}/particles.star with {particle_count} selected particles.\n")
        
        with open(os.path.join(self.select_dir, "RELION_JOB_EXIT_SUCCESS"), 'w') as f:
            pass

# --- Save All Classes Dialog ---
class SaveAllClassesDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save All Classes")
        layout = QFormLayout(self)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "SVG"])
        layout.addRow("Format:", self.format_combo)

        self.scale_check = QCheckBox()
        self.scale_check.setChecked(True)
        layout.addRow("Include Scale Bar:", self.scale_check)

        self.scale_length_combo = QComboBox()
        self.scale_length_combo.addItems(["10", "20", "50", "100"])
        self.scale_length_combo.setCurrentText("50")
        layout.addRow("Scale Bar Length (Å):", self.scale_length_combo)

        self.scale_check.stateChanged.connect(
            lambda: self.scale_length_combo.setEnabled(self.scale_check.isChecked())
        )

        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addRow(self.buttons)

    def get_values(self):
        return {
            "format": self.format_combo.currentText(),
            "include_scale": self.scale_check.isChecked(),
            "scale_length": int(self.scale_length_combo.currentText())
        }

# --- Main Window ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Class Planner")
        self.resize(1400, 800)
        
        # --- Data Structures ---
        self.tile_data = {} 
        self.tile_usage_counts = {} 
        self.hovered_tile_id = None 
        self.cyan_highlighted_tiles = set()
        
        # --- RELION Export State ---
        self.current_mrcs_path = None
        self.current_model_star = None
        self.current_data_star = None
        self.total_classes = 0
        self.relion_class_mapping = {} 
        self.current_voxel_size = 1.0
        self.original_width = 1 

        # --- Communication ---
        self.comms = Communicate()
        self.comms.highlight_stock.connect(self.on_stock_hover)
        self.comms.unhighlight_stock.connect(self.on_stock_unhover)
        self.comms.stock_used.connect(self.on_stock_used)
        self.comms.stock_unused.connect(self.on_stock_unused)
        self.comms.item_moved.connect(self.update_stats_ui)
        self.comms.locate_stock.connect(self.on_locate_stock)

        # --- UI Layout ---
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)
        
        # 1. Toolbar (Split into Two Rows)
        toolbar_container = QVBoxLayout()

        # --- Row 1 ---
        row1 = QHBoxLayout()
        
        btn_import_job = QPushButton("Import 2D class job")
        btn_import_job.clicked.connect(self.import_relion_job)
        row1.addWidget(btn_import_job)
        
        btn_load = QPushButton("Browse Image or .mrcs")
        btn_load.clicked.connect(self.load_image)
        row1.addWidget(btn_load)
        
        paste_keys = "Cmd+V" if sys.platform == 'darwin' else "Ctrl+V"
        lbl_hint = QLabel(f"  (Tip: If run locally on your laptop, screenshot, then {paste_keys} to paste!)")
        lbl_hint.setStyleSheet("color: #aaa;")
        row1.addWidget(lbl_hint)
        
        row1.addStretch()
        toolbar_container.addLayout(row1)

        # --- Row 2 ---
        row2 = QHBoxLayout()
        
        btn_rearrange = QPushButton("Rearrange")
        btn_rearrange.setToolTip("Sort classes in each row from left to right based on stock order")
        btn_rearrange.clicked.connect(self.rearrange_canvas)
        row2.addWidget(btn_rearrange)
        
        btn_highlight = QPushButton("Highlight Selected")
        btn_highlight.setToolTip("Highlight currently selected canvas items as cyan in the stock list")
        btn_highlight.clicked.connect(self.highlight_selected_in_stock)
        row2.addWidget(btn_highlight)
        
        self.btn_auto_group = QPushButton("Auto Group")
        self.btn_auto_group.setToolTip("Automatically cluster and arrange classes on the canvas based on structural similarities")
        self.btn_auto_group.setEnabled(True)
        self.btn_auto_group.clicked.connect(self.run_auto_grouping)
        row2.addWidget(self.btn_auto_group)
        
        self.btn_save_canvas = QPushButton("Save Canvas")
        self.btn_save_canvas.setToolTip("Save the current canvas layout and stock to a JSON file")
        self.btn_save_canvas.clicked.connect(self.save_canvas)
        row2.addWidget(self.btn_save_canvas)
        
        btn_save_all = QPushButton("Save All Classes")
        btn_save_all.setToolTip("Save all classes as individual PNG or SVG files")
        btn_save_all.clicked.connect(self.save_all_classes)
        row2.addWidget(btn_save_all)
        
        btn_clear = QPushButton("Clear All")
        btn_clear.clicked.connect(self.clear_all)
        row2.addWidget(btn_clear)
        
        btn_export = QPushButton("Export Selected Classes on Canvas to RELION")
        btn_export.setToolTip("Export only the currently selected (yellow frame) canvas items directly to a RELION Select job")
        btn_export.clicked.connect(self.export_to_relion)
        btn_export.setStyleSheet("background-color: #98c379; color: black; font-weight: bold;")
        row2.addWidget(btn_export)
        
        # --- Inline Progress Bar (Hidden by Default) ---
        self.export_progress_bar = QProgressBar()
        self.export_progress_bar.setTextVisible(True)
        
        # Reserve layout space even when invisible to prevent button shifting
        sp_bar = self.export_progress_bar.sizePolicy()
        sp_bar.setRetainSizeWhenHidden(True)
        self.export_progress_bar.setSizePolicy(sp_bar)
        self.export_progress_bar.setVisible(False)
        
        self.export_progress_bar.setStyleSheet("""
            QProgressBar { border: 1px solid #555; border-radius: 5px; background-color: #1e1e1e; text-align: center; color: white; }
            QProgressBar::chunk { background-color: #98c379; width: 20px; }
        """)
        
        self.export_progress_label = QLabel("")
        self.export_progress_label.setStyleSheet("color: #aaa; font-weight: bold;")
        
        # Reserve layout space even when invisible to prevent button shifting
        sp_label = self.export_progress_label.sizePolicy()
        sp_label.setRetainSizeWhenHidden(True)
        self.export_progress_label.setSizePolicy(sp_label)
        self.export_progress_label.setVisible(False)
        
        row2.addWidget(self.export_progress_bar, 1) # Stretch factor 1
        row2.addWidget(self.export_progress_label)
        
        toolbar_container.addLayout(row2)

        layout.addLayout(toolbar_container)

        # 2. Splitter Area
        splitter = QSplitter(Qt.Orientation.Horizontal)
        # Stretch factor 1 prevents the toolbar from expanding when zooming
        layout.addWidget(splitter, 1)

        # --- LEFT: Stock Container ---
        stock_container = QWidget()
        stock_layout = QVBoxLayout(stock_container)
        
        # Header Layout for Stock (Label + Count)
        stock_header = QHBoxLayout()
        stock_header.addWidget(QLabel("<b style='color: #98c379;'>Stock: </b>"))
        
        self.lbl_stock_count = QLabel("Selected 0 classes")
        self.lbl_stock_count.setStyleSheet("color: #ccc;")
        stock_header.addWidget(self.lbl_stock_count)
        stock_header.addStretch()
        
        stock_layout.addLayout(stock_header)
        
        self.stock_list = StockListWidget()
        stock_layout.addWidget(self.stock_list)
        splitter.addWidget(stock_container)

        # --- RIGHT: Canvas Container ---
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.addWidget(QLabel("<b style='color: #98c379;'>Canvas</b>"))
        
        self.scene = EditorScene(self.comms)
        self.view = EditorView(self.scene, self)
        
        # 3. Canvas Overlay (The HUD)
        view_layout = QVBoxLayout(self.view)
        view_layout.setContentsMargins(10, 10, 40, 10) 
        
        self.lbl_canvas_overlay = QLabel("")
        self.lbl_canvas_overlay.setStyleSheet("""
            color: #98c379; 
            background-color: transparent; 
            font-weight: bold; 
            font-size: 13px;
        """)
        self.lbl_canvas_overlay.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        self.lbl_canvas_overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        
        view_layout.addWidget(self.lbl_canvas_overlay, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        
        canvas_layout.addWidget(self.view)
        splitter.addWidget(canvas_container)
        
        splitter.setStretchFactor(1, 4)

        # Sklearn is now loaded lazily inside run_auto_grouping to prevent thread-pool corruption.

    # --- UI Update Logic ---
    def update_stats_ui(self):
        active_counts = [tid for tid, count in self.tile_usage_counts.items() if count > 0]
        unique_count = len(active_counts)
        self.lbl_stock_count.setText(f"Selected {unique_count} classes")

        items = [i for i in self.scene.items() if isinstance(i, CanvasTileItem)]
        
        if not items:
            self.lbl_canvas_overlay.setText("")
            return

        def get_visual_center(item):
            return item.mapToScene(item.boundingRect().center())

        items.sort(key=lambda item: (get_visual_center(item).y(), get_visual_center(item).x()))

        groups = []
        if items:
            current_group = [items[0]]
            for i in range(1, len(items)):
                current_item = items[i]
                prev_item = current_group[-1]
                
                curr_y = get_visual_center(current_item).y()
                prev_y = get_visual_center(prev_item).y()
                
                y_diff = abs(curr_y - prev_y)
                threshold = current_item.boundingRect().height() * 0.2
                
                if y_diff <= threshold:
                    current_group.append(current_item)
                else:
                    groups.append(current_group)
                    current_group = [current_item]
            groups.append(current_group)

        text_lines = [f"Used Classes:\n"]
        
        for i, group in enumerate(groups):
            seen_ids = set()
            row_ids = []
            for item in group:
                if item.tile_id not in seen_ids:
                    row_ids.append(item.tile_id)
                    seen_ids.add(item.tile_id)
            
            for tid in row_ids:
                text_lines.append(f"Class {tid}")
            
            if i < len(groups) - 1:
                text_lines.append("=======")

        self.lbl_canvas_overlay.setText("\n".join(text_lines))

    def update_tile_visual(self, tile_id):
        target_item = None
        for i in range(self.stock_list.count()):
            item = self.stock_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == tile_id:
                target_item = item
                break
        
        if not target_item: return

        is_hovered = (self.hovered_tile_id == tile_id)
        usage_count = self.tile_usage_counts.get(tile_id, 0)
        is_cyan = tile_id in self.cyan_highlighted_tiles

        if is_cyan:
            # Cyan with 180 opacity so it stands out clearly
            target_item.setBackground(QColor(0, 255, 255, 180))
        elif is_hovered:
            target_item.setBackground(QColor(235, 171, 103, 220))
        elif usage_count > 0:
            target_item.setBackground(QColor(152, 195, 121, 150))
        else:
            target_item.setBackground(Qt.GlobalColor.transparent)

    # --- Signal Handlers ---
    def on_stock_hover(self, tile_id):
        self.hovered_tile_id = tile_id
        self.update_tile_visual(tile_id)

    def on_stock_unhover(self, tile_id):
        if self.hovered_tile_id == tile_id:
            self.hovered_tile_id = None
        self.update_tile_visual(tile_id)

    def on_stock_used(self, tile_id):
        self.tile_usage_counts[tile_id] = self.tile_usage_counts.get(tile_id, 0) + 1
        self.update_tile_visual(tile_id)
        self.update_stats_ui()

    def on_stock_unused(self, tile_id):
        if tile_id in self.tile_usage_counts:
            self.tile_usage_counts[tile_id] = max(0, self.tile_usage_counts[tile_id] - 1)
        self.update_tile_visual(tile_id)
        self.update_stats_ui()

    def on_locate_stock(self, tile_id):
        for i in range(self.stock_list.count()):
            item = self.stock_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == tile_id:
                # Scroll the stock panel so the item is dead center
                self.stock_list.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                self.stock_list.clearSelection()
                # We removed item.setSelected(True) so the default blue highlight 
                # doesn't mask your custom yellow hover color!
                break

    # --- Core Logic ---
    def keyPressEvent(self, event: QKeyEvent):
        is_paste = False
        if event.key() == Qt.Key.Key_V:
            if sys.platform == 'darwin':
                if event.modifiers() & Qt.KeyboardModifier.MetaModifier: is_paste = True
            else:
                if event.modifiers() & Qt.KeyboardModifier.ControlModifier: is_paste = True

        if is_paste:
            clipboard = QApplication.clipboard()
            if clipboard.mimeData().hasImage():
                self.ask_and_slice(clipboard.pixmap())
            else:
                QMessageBox.information(self, "No Image", "Clipboard is empty.")
        else:
            super().keyPressEvent(event)

    def import_relion_job(self):
        dialog = ImportJobDialog(self)
        if not dialog.exec():
            return
            
        job_num, sort_metric, reverse_sort, cols = dialog.get_values()
        
        if not job_num:
            return
            
        job_num = job_num.strip()
        # Auto-pad with zeros if user just types "15" instead of "015"
        if len(job_num) < 3 and job_num.isdigit():
            job_num = job_num.zfill(3)

        cwd = os.getcwd()
        job_dir = os.path.join(cwd, "Class2D", f"job{job_num}")
        
        if not os.path.exists(job_dir):
            QMessageBox.warning(self, "Error", f"Cannot find directory:\n{job_dir}")
            return
            
        # Find the highest iteration classes.mrcs
        search_pattern = os.path.join(job_dir, "run_it*_classes.mrcs")
        mrcs_files = glob.glob(search_pattern)
        
        if not mrcs_files:
            QMessageBox.warning(self, "Error", f"No run_it*_classes.mrcs found in:\n{job_dir}")
            return
            
        mrcs_files.sort()
        latest_mrcs = mrcs_files[-1]
        
        # Save paths for export functionality
        self.current_model_star = latest_mrcs.replace('_classes.mrcs', '_model.star')
        self.current_data_star = latest_mrcs.replace('_classes.mrcs', '_data.star')
        
        canvas_json_path = os.path.join(job_dir, "xcop_canvas.json")
        if os.path.exists(canvas_json_path):
            self.load_canvas_state(canvas_json_path)
        else:
            self.load_mrcs(latest_mrcs, sort_metric, reverse_sort, cols)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open", "", "Supported Files (*.png *.jpg *.bmp *.mrcs *.mrc *.json);;JSON Canvas (*.json);;Images/MRCS (*.png *.jpg *.bmp *.mrcs *.mrc)")
        if not path:
            return
            
        if path.endswith('.json'):
            self.load_canvas_state(path)
        elif path.endswith('.mrcs') or path.endswith('.mrc'):
            self.load_mrcs(path)
        else:
            self.ask_and_slice(QPixmap(path))

    def load_mrcs(self, path, sort_metric='_rlnClassDistribution', reverse_sort=True, cols=5):
        self.current_mrcs_path = path
        self.current_voxel_size = 1.0
        try:
            with mrcfile.open(path, permissive=True) as mrc:
                data = mrc.data
                if mrc.voxel_size.x > 0:
                    self.current_voxel_size = float(mrc.voxel_size.x)
                
            # Ensure data is 3D (a stack of 2D frames)
            if data.ndim == 2:
                data = data[np.newaxis, ...]
            elif data.ndim != 3:
                QMessageBox.warning(self, "Error", "Unsupported MRC dimensions.")
                return
                
            self.total_classes = data.shape[0]

            next_id = max(self.tile_data.keys()) + 1 if self.tile_data else 1
            TARGET_SIZE = 200
            
            # --- Parse corresponding _model.star file for sorting ---
            star_path = path.replace('_classes.mrcs', '_model.star')
            sort_values = {}
            if os.path.exists(star_path):
                with open(star_path, 'r') as f:
                    in_classes = False
                    headers = []
                    row_idx = 0
                    for line in f:
                        line = line.strip()
                        if line.startswith('data_model_classes'):
                            in_classes = True
                            continue
                        if in_classes and line.startswith('data_'):
                            break
                        if in_classes:
                            if line.startswith('_rln'):
                                headers.append(line.split()[0])
                            elif line and not line.startswith('#') and not line.startswith('loop_'):
                                if sort_metric in headers:
                                    idx = headers.index(sort_metric)
                                    parts = line.split()
                                    if len(parts) >= len(headers):
                                        sort_values[row_idx] = float(parts[idx])
                                        row_idx += 1

            frame_order = list(range(data.shape[0]))
            if sort_values:
                # Sort indices based on user selected metric
                frame_order.sort(key=lambda x: sort_values.get(x, 0.0), reverse=reverse_sort)

            for i in frame_order:
                relion_class_id = i + 1
                frame = data[i]
                
                # Normalize float density to 0-255 uint8 grayscale
                f_min, f_max = frame.min(), frame.max()
                if f_max > f_min:
                    norm_frame = 255.0 * (frame - f_min) / (f_max - f_min)
                else:
                    norm_frame = np.zeros_like(frame)
                
                # Convert to C-contiguous array for memory safety
                img_uint8 = np.require(norm_frame.astype(np.uint8), np.uint8, 'C')
                h, w = img_uint8.shape
                self.original_width = w
                
                # Create QImage and force a .copy() so it doesn't crash when numpy garbage collects
                qimg = QImage(img_uint8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
                pixmap = QPixmap.fromImage(qimg)
                
                pixmap = pixmap.scaled(
                    TARGET_SIZE, TARGET_SIZE, 
                    Qt.AspectRatioMode.IgnoreAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                
                self.tile_data[next_id] = pixmap
                self.relion_class_mapping[next_id] = relion_class_id
                
                # Display original RELION class ID and its metric value
                val = sort_values.get(i, 0.0)
                label_text = f"Class {relion_class_id}"
                if sort_values:
                    if sort_metric == '_rlnClassDistribution':
                        label_text += f"\n({val * 100:.1f}%)"
                    elif sort_metric == '_rlnEstimatedResolution':
                        label_text += f"\n({val:.2f} Å)"
                    else:
                        label_text += f"\n({val:.2f})"
                        
                item = QListWidgetItem(QIcon(pixmap), label_text)
                item.setData(Qt.ItemDataRole.UserRole, next_id)
                self.stock_list.addItem(item)
                
                next_id += 1

            # Resize the stock list to display the user selected items in a row
            # (cols is now passed as an argument from the dialog)
            stock_container = self.stock_list.parent()
            splitter = stock_container.parent()

            if isinstance(splitter, QSplitter):
                icon_w = self.stock_list.iconSize().width()
                spacing = self.stock_list.spacing()
                scrollbar_width = QApplication.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
                content_width = (cols * icon_w) + (cols * spacing)
                total_buffer = scrollbar_width + 80
                
                desired_stock_width = content_width + total_buffer
                total_internal_width = sum(splitter.sizes())
                safe_stock_width = min(desired_stock_width, int(total_internal_width * 0.9))
                
                new_canvas_width = total_internal_width - safe_stock_width
                splitter.setSizes([safe_stock_width, new_canvas_width])

                # --- Auto-adjust to ensure the exact number of columns ---
                self.stock_list.doItemsLayout()
                QApplication.processEvents()
                
                for _ in range(20): # Safety limit to prevent infinite loops
                    if self.stock_list.count() == 0:
                        break
                        
                    first_y = self.stock_list.visualItemRect(self.stock_list.item(0)).y()
                    items_in_first_row = 0
                    for i in range(min(cols + 2, self.stock_list.count())):
                        if self.stock_list.visualItemRect(self.stock_list.item(i)).y() == first_y:
                            items_in_first_row += 1
                        else:
                            break
                            
                    if items_in_first_row < cols:
                        safe_stock_width += 20  # Add 20 pixels
                        if safe_stock_width >= total_internal_width:
                            break # Stop if it maxes out the window
                        new_canvas_width = total_internal_width - safe_stock_width
                        splitter.setSizes([safe_stock_width, new_canvas_width])
                        self.stock_list.doItemsLayout()
                        QApplication.processEvents()
                    else:
                        break
                
        except Exception as e:
            QMessageBox.warning(self, "Error Loading MRCS", f"Failed to load MRCS:\n{str(e)}")

    def ask_and_slice(self, pixmap):
        if pixmap.isNull(): return
        
        dialog = GridSelectionDialog(pixmap, self)
        dialog.adjustSize()
        center_point = self.view.rect().center()
        view_center_global = self.view.mapToGlobal(center_point)
        dialog_rect = dialog.rect()
        pos_x = view_center_global.x() - dialog_rect.width() // 2
        pos_y = view_center_global.y() - dialog_rect.height() // 2
        dialog.move(pos_x, pos_y)

        if dialog.exec():
            rows = dialog.result_rows
            cols = dialog.result_cols
            if rows > 0 and cols > 0:
                self.slice_image(pixmap, rows, cols)

    def slice_image(self, pixmap, rows, cols):
        next_id = max(self.tile_data.keys()) + 1 if self.tile_data else 0
        cut_w = pixmap.width() // cols
        cut_h = pixmap.height() // rows
        TARGET_SIZE = 200

        for r in range(rows):
            for c in range(cols):
                tile = pixmap.copy(c * cut_w, r * cut_h, cut_w, cut_h)
                tile = tile.scaled(
                    TARGET_SIZE, TARGET_SIZE, 
                    Qt.AspectRatioMode.IgnoreAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.tile_data[next_id] = tile
                item = QListWidgetItem(QIcon(tile), f"Class {next_id}")
                item.setData(Qt.ItemDataRole.UserRole, next_id)
                self.stock_list.addItem(item)
                next_id += 1

        stock_container = self.stock_list.parent()
        splitter = stock_container.parent()

        if isinstance(splitter, QSplitter):
            icon_w = self.stock_list.iconSize().width()
            spacing = self.stock_list.spacing()
            scrollbar_width = QApplication.style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
            content_width = (cols * icon_w) + (cols * spacing)
            total_buffer = scrollbar_width + 80
            
            desired_stock_width = content_width + total_buffer
            total_internal_width = sum(splitter.sizes())
            safe_stock_width = min(desired_stock_width, int(total_internal_width * 0.9))
            
            new_canvas_width = total_internal_width - safe_stock_width
            splitter.setSizes([safe_stock_width, new_canvas_width])

            # --- Auto-adjust to ensure the exact number of columns ---
            self.stock_list.doItemsLayout()
            QApplication.processEvents()
            
            for _ in range(20): # Safety limit to prevent infinite loops
                if self.stock_list.count() == 0:
                    break
                    
                first_y = self.stock_list.visualItemRect(self.stock_list.item(0)).y()
                items_in_first_row = 0
                for i in range(min(cols + 2, self.stock_list.count())):
                    if self.stock_list.visualItemRect(self.stock_list.item(i)).y() == first_y:
                        items_in_first_row += 1
                    else:
                        break
                        
                if items_in_first_row < cols:
                    safe_stock_width += 20  # Add 20 pixels
                    if safe_stock_width >= total_internal_width:
                        break # Stop if it maxes out the window
                    new_canvas_width = total_internal_width - safe_stock_width
                    splitter.setSizes([safe_stock_width, new_canvas_width])
                    self.stock_list.doItemsLayout()
                    QApplication.processEvents()
                else:
                    break

    def save_all_classes(self):
        if not self.current_mrcs_path or not os.path.exists(self.current_mrcs_path):
            QMessageBox.warning(self, "No Data", "No original MRCS file found to export from.")
            return

        dialog = SaveAllClassesDialog(self)
        if not dialog.exec():
            return

        settings = dialog.get_values()
        
        save_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Save Classes")
        if not save_dir:
            return

        fmt = settings["format"].lower()
        angstroms = settings["scale_length"]
        saved_count = 0

        try:
            # Re-open the original file to grab uncompressed, original resolution data
            with mrcfile.open(self.current_mrcs_path, permissive=True) as mrc:
                data = mrc.data
                if data.ndim == 2:
                    data = data[np.newaxis, ...]
                    
                pixel_size = self.current_voxel_size
                
                # Iterate over the classes currently loaded in the UI
                for tile_id in self.tile_data.keys():
                    relion_id = self.relion_class_mapping.get(tile_id, tile_id)
                    # MRCS arrays are 0-indexed, RELION IDs are 1-indexed
                    mrc_index = relion_id - 1 
                    
                    if mrc_index < 0 or mrc_index >= data.shape[0]:
                        continue
                        
                    frame = data[mrc_index]
                    
                    # Normalize the high-resolution frame exactly as we do for the UI
                    f_min, f_max = frame.min(), frame.max()
                    if f_max > f_min:
                        norm_frame = 255.0 * (frame - f_min) / (f_max - f_min)
                    else:
                        norm_frame = np.zeros_like(frame)
                        
                    # Skip completely empty/black frames
                    if np.max(norm_frame) == 0:
                        continue
                        
                    # Convert to QImage at original box size
                    img_uint8 = np.require(norm_frame.astype(np.uint8), np.uint8, 'C')
                    h, w = img_uint8.shape
                    
                    qimg = QImage(img_uint8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
                    original_pixmap = QPixmap.fromImage(qimg)
                    
                    filename = os.path.join(save_dir, f"Class_{relion_id}.{fmt}")
                    
                    # Calculate true scale bar length based directly on original dimensions
                    bar_length_px = angstroms / pixel_size if pixel_size > 0 else 0
                    
                    if fmt == "png":
                        result_img = QPixmap(original_pixmap.size())
                        result_img.fill(Qt.GlobalColor.transparent)
                        painter = QPainter(result_img)
                        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                        painter.drawPixmap(0, 0, original_pixmap)
                    elif fmt == "svg":
                        result_img = QSvgGenerator()
                        result_img.setFileName(filename)
                        result_img.setSize(original_pixmap.size())
                        result_img.setViewBox(QRectF(0, 0, original_pixmap.width(), original_pixmap.height()))
                        painter = QPainter(result_img)
                        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                        painter.drawPixmap(0, 0, original_pixmap)
                        
                    if settings["include_scale"] and bar_length_px > 0:
                        painter.setPen(Qt.PenStyle.NoPen)
                        painter.setBrush(QColor(255, 255, 255))
                        
                        # Scale bar sizing using quick_map's native UI math ratios
                        bar_h = max(2, int(h * 0.015))
                        margin_x = max(2, int(w * 0.05))
                        margin_y = max(2, int(h * 0.05))
                        
                        rect_x = margin_x
                        rect_y = h - margin_y - bar_h
                        painter.drawRect(int(rect_x), int(rect_y), int(bar_length_px), int(bar_h))
                        
                        painter.setPen(QPen(QColor(255, 255, 255)))
                        font = QFont("Helvetica", max(6, int(h * 0.04)), QFont.Weight.Bold)
                        painter.setFont(font)
                        painter.drawText(int(rect_x), int(rect_y) - 2, f"{angstroms} Å")
                        
                    painter.end()
                    
                    if fmt == "png":
                        result_img.save(filename, "PNG")
                        
                    saved_count += 1
                    
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export classes from MRCS:\n{str(e)}")
            return

        QMessageBox.information(self, "Success", f"Saved {saved_count} classes in {fmt.upper()} format to:\n{save_dir}")

    def save_canvas(self):
        if not self.current_mrcs_path:
            QMessageBox.warning(self, "Save Error", "No MRCS file is currently loaded to save.")
            return
        
        items = [i for i in self.scene.items() if isinstance(i, CanvasTileItem)]
        canvas_data = []
        for item in items:
            canvas_data.append({
                "tile_id": item.tile_id,
                "x": item.pos().x(),
                "y": item.pos().y(),
                "z": item.zValue()
            })
            
        save_data = {
            "mrcs_path": self.current_mrcs_path,
            "canvas": canvas_data
        }
        
        save_dir = os.path.dirname(self.current_mrcs_path)
        save_path = os.path.join(save_dir, "xcop_canvas.json")
        
        try:
            with open(save_path, 'w') as f:
                json.dump(save_data, f, indent=4)
            QMessageBox.information(self, "Saved", f"Canvas state saved successfully to:\n{save_path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save canvas:\n{str(e)}")

    def load_canvas_state(self, json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            mrcs_path = data.get("mrcs_path")
            
            # Robust path resolution: try absolute first, then relative to the json file
            # This ensures portability if you move the folder from HPC to your laptop
            if not mrcs_path or not os.path.exists(mrcs_path):
                local_mrcs = os.path.join(os.path.dirname(json_path), os.path.basename(mrcs_path) if mrcs_path else "")
                if os.path.exists(local_mrcs):
                    mrcs_path = local_mrcs
                else:
                    QMessageBox.warning(self, "Load Error", f"Cannot find the original MRCS file:\n{mrcs_path}\nMake sure the .mrcs is in the same folder as the json file.")
                    return
                    
            self.clear_all()
            
            # Restore RELION states
            self.current_model_star = mrcs_path.replace('_classes.mrcs', '_model.star')
            self.current_data_star = mrcs_path.replace('_classes.mrcs', '_data.star')
            
            # Load the images into the stock
            self.load_mrcs(mrcs_path)
            
            # Restore the canvas layout
            canvas_items = data.get("canvas", [])
            for item_data in canvas_items:
                tile_id = item_data["tile_id"]
                if tile_id in self.tile_data:
                    pixmap = self.tile_data[tile_id]
                    item = CanvasTileItem(pixmap, tile_id, self.comms)
                    item.setPos(item_data["x"], item_data["y"])
                    item.setZValue(item_data.get("z", 0))
                    self.scene.addItem(item)
                    self.comms.stock_used.emit(tile_id)
                    
            self.scene.update()
            self.update_stats_ui()
            
        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load canvas state:\n{str(e)}")

    def rearrange_canvas(self):
        # 1. Build a dictionary of tile_id to its exact index in the stock list
        stock_order = {}
        for i in range(self.stock_list.count()):
            tile_id = self.stock_list.item(i).data(Qt.ItemDataRole.UserRole)
            stock_order[tile_id] = i

        # 2. Grab all tiles and group them by lines (using your existing Y-threshold logic)
        items = [i for i in self.scene.items() if isinstance(i, CanvasTileItem)]
        if not items: return

        def get_visual_center(item):
            return item.mapToScene(item.boundingRect().center())

        items.sort(key=lambda item: (get_visual_center(item).y(), get_visual_center(item).x()))

        groups = []
        if items:
            current_group = [items[0]]
            for i in range(1, len(items)):
                current_item = items[i]
                prev_item = current_group[-1]
                
                curr_y = get_visual_center(current_item).y()
                prev_y = get_visual_center(prev_item).y()
                
                y_diff = abs(curr_y - prev_y)
                threshold = current_item.boundingRect().height() * 0.2
                
                if y_diff <= threshold:
                    current_group.append(current_item)
                else:
                    groups.append(current_group)
                    current_group = [current_item]
            groups.append(current_group)

        # 3. For each row, align top and snap side-by-side based on stock order
        for group in groups:
            # Find the starting X position and the top-most Y position for this group
            start_x = min(item.pos().x() for item in group)
            align_y = min(item.pos().y() for item in group)
            
            # Sort the tiles themselves based on their rank/index in the stock list
            sorted_items = sorted(group, key=lambda item: stock_order.get(item.tile_id, float('inf')))
            
            # Place them perfectly side-by-side, aligned to the top Y
            current_x = start_x
            for item in sorted_items:
                item.setPos(current_x, align_y)
                current_x += item.boundingRect().width()

        # 4. Refresh the scene and the HUD 
        self.scene.update()
        self.update_stats_ui()

    def run_auto_grouping(self):
        if not self.tile_data:
            QMessageBox.warning(self, "No Data", "Please load classes into the stock first.")
            return

        try:
            # 1. Extract pixel data directly from QPixmaps and filter out empty classes
            all_tile_ids = list(self.tile_data.keys())
            flattened_data = []
            valid_tile_ids = []
            
            # Pre-filter to avoid issues with n_clusters > n_samples
            for tid in all_tile_ids:
                img = self.tile_data[tid].toImage().convertToFormat(QImage.Format.Format_Grayscale8)
                ptr = img.bits()
                ptr.setsize(img.sizeInBytes())
                arr = np.array(ptr).reshape(img.height(), img.width())
                
                # Check for completely black classes (which represent 0 particles or empty frames)
                if np.max(arr) == 0:
                    continue
                    
                flattened_data.append(arr.flatten())
                valid_tile_ids.append(tid)

            if not valid_tile_ids:
                QMessageBox.warning(self, "No Valid Classes", "All loaded classes appear to be completely empty or black.")
                return

            n_samples = len(valid_tile_ids)
            suggested_clusters = max(2, min(n_samples // 8, 15))
            
            # Ask the user for the number of groups BEFORE starting the heavy math
            n_clusters, ok = QInputDialog.getInt(
                self, 
                "Auto Group Settings", 
                "How many different types of classes?\n\nIncrease this number to force stricter separation", 
                suggested_clusters, 2, n_samples, 1
            )
            
            if not ok:
                return

            # Change mouse to hourglass during the math calculations
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            tile_ids = valid_tile_ids
            X = np.stack(flattened_data)
            
            # 2. Dimensionality reduction (PCA)
            n_components = min(n_samples, 50)
            pca = PCA(n_components=n_components)
            reduced_features = pca.fit_transform(X)
            
            clustering = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clustering.fit_predict(reduced_features)
            
            # Group the classes by their new label
            grouped_classes = {}
            for i, label in enumerate(labels):
                if label not in grouped_classes:
                    grouped_classes[label] = []
                grouped_classes[label].append(tile_ids[i])
                
            # --- SORTING LOGIC: Whitest to Blackest ---
            # Calculate mean pixel intensity (0=Black, 255=White) for each tile
            tile_intensities = {tid: np.mean(X[i]) for i, tid in enumerate(tile_ids)}
            
            # Calculate the overall mean intensity for each group
            group_intensities = {}
            for label, tids in grouped_classes.items():
                group_intensities[label] = np.mean([tile_intensities[tid] for tid in tids])
                
            # Sort group labels by intensity descending (Whitest first)
            sorted_labels = sorted(grouped_classes.keys(), key=lambda lbl: group_intensities[lbl], reverse=True)
                
            # 3. Layout on Canvas
            self.scene.clear()
            
            # Reset usage counts so the stock UI doesn't show ghost highlights
            for tid in self.tile_data.keys():
                self.tile_usage_counts[tid] = 0
                self.update_tile_visual(tid)
                
            x_spacing = 20
            y_spacing = 30
            current_y = 0
            
            for label in sorted_labels:
                current_x = 0
                max_h_in_row = 0
                
                for tid in grouped_classes[label]:
                    pixmap = self.tile_data[tid]
                    item = CanvasTileItem(pixmap, tid, self.comms)
                    item.setPos(current_x, current_y)
                    self.scene.addItem(item)
                    self.comms.stock_used.emit(tid)
                    
                    current_x += pixmap.width() + x_spacing
                    if pixmap.height() > max_h_in_row:
                        max_h_in_row = pixmap.height()
                        
                current_y += max_h_in_row + y_spacing
                
            self.scene.update()
            self.update_stats_ui()
            
        except ImportError:
            QMessageBox.critical(self, "Missing Library", "Please run: pip install scikit-learn")
        except Exception as e:
            QMessageBox.critical(self, "Grouping Error", f"Failed to group classes:\n{str(e)}")
        finally:
            QApplication.restoreOverrideCursor()

    def highlight_selected_in_stock(self):
        # Clear the previous highlights
        old_highlights = set(self.cyan_highlighted_tiles)
        self.cyan_highlighted_tiles.clear()
        
        # Find new highlights from canvas
        for item in self.scene.selectedItems():
            if isinstance(item, CanvasTileItem):
                self.cyan_highlighted_tiles.add(item.tile_id)
                
        # Update visuals for all affected tiles (old and new)
        tiles_to_update = old_highlights.union(self.cyan_highlighted_tiles)
        for tile_id in tiles_to_update:
            self.update_tile_visual(tile_id)

    def clear_all(self):
        self.stock_list.clear()
        self.tile_data.clear()
        self.tile_usage_counts.clear()
        self.hovered_tile_id = None
        self.cyan_highlighted_tiles.clear()
        self.scene.clear()
        self.update_stats_ui()
        self.current_mrcs_path = None
        self.current_model_star = None
        self.current_data_star = None
        self.total_classes = 0
        self.relion_class_mapping.clear()

    def export_to_relion(self):
        if not self.current_model_star or not self.current_data_star or not os.path.exists(self.current_model_star):
            QMessageBox.warning(self, "Export Error", "No valid RELION 2D class job is loaded.\nPlease import a job via 'Import 2D class job' first.")
            return

        # Only grab items that are currently selected (yellow frame) on the canvas
        items = [i for i in self.scene.selectedItems() if isinstance(i, CanvasTileItem)]
        if not items:
            QMessageBox.warning(self, "Export Error", "No classes are selected on the canvas. Please highlight the classes you want to export.")
            return
            
        # Get unique selected RELION classes from the canvas (Using a set automatically prevents duplicates!)
        selected_classes = set()
        for item in items:
            if item.tile_id in self.relion_class_mapping:
                selected_classes.add(self.relion_class_mapping[item.tile_id])
                
        if not selected_classes:
            QMessageBox.warning(self, "Export Error", "No valid RELION classes found on the canvas.")
            return

        select_job_str, ok = QInputDialog.getText(self, "Export to RELION", "1. Create a Select Job in RELION\n\n2. Enter the Select Job Number\n   To Assign the Classes to this Select Job:")
        if not ok or not select_job_str.strip():
            return
            
        job_num = select_job_str.strip()
        if job_num.isdigit() and len(job_num) < 3:
            job_num = job_num.zfill(3)
            
        select_dir = os.path.join(os.getcwd(), "Select", f"job{job_num}")
        
        if not os.path.exists(select_dir):
            reply = QMessageBox.question(self, "Create Directory?", f"Directory {select_dir} does not exist.\n\nShould I create it for you?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                os.makedirs(select_dir)
            else:
                return

        # --- START INLINE PROGRESS BAR & THREAD ---
        self.export_progress_bar.setValue(0)
        self.export_progress_label.setText("Initializing export...")
        self.export_progress_bar.setVisible(True)
        self.export_progress_label.setVisible(True)
        
        # Setup background thread
        relion_version = self._get_relion_version()
        self.export_thread = ExportJobThread(
            select_dir, selected_classes, self.total_classes, 
            self.current_model_star, self.current_data_star, 
            relion_version, job_num
        )
        
        # Connect signals
        self.export_thread.progress_signal.connect(self._update_export_progress)
        self.export_thread.finished_signal.connect(lambda c, p: self._on_export_finished(c, p, job_num))
        self.export_thread.error_signal.connect(self._on_export_error)
        
        self.export_thread.start()

    def _update_export_progress(self, percent, text):
        self.export_progress_bar.setValue(percent)
        self.export_progress_label.setText(text)

    def _on_export_finished(self, classes_count, particle_count, job_num):
        self.export_progress_bar.setValue(100)
        self.export_progress_bar.setVisible(False)
        self.export_progress_label.setVisible(False)
        
        msg = f"Successfully exported selections to Select/job{job_num}!\n\n"
        msg += f"Selected {classes_count} classes.\n"
        msg += f"Saved {particle_count} particles.\n\n"
        msg += "RELION will now recognize this job as Finished."
        QMessageBox.information(self, "Export Complete", msg)

    def _on_export_error(self, error_msg):
        self.export_progress_bar.setVisible(False)
        self.export_progress_label.setVisible(False)
        QMessageBox.critical(self, "Export Error", f"An error occurred during export:\n{error_msg}")

    def _get_relion_version(self):
        """Reads default_pipeline.star in Read-Only mode to dynamically extract the RELION version."""
        pipeline_path = os.path.join(os.getcwd(), "default_pipeline.star")
        if os.path.exists(pipeline_path):
            try:
                with open(pipeline_path, 'r') as f:
                    for line in f:
                        if line.startswith("# version "):
                            return line.strip().replace("# version ", "")
            except Exception:
                pass
        return "50001"  # Fallback

    def auto_group_classes(self, mrcs_data, n_clusters=5):
        """
        Extracts features from MRC data using PCA and groups them.
        mrcs_data: numpy array of shape (N, H, W)
        """
        N, H, W = mrcs_data.shape
        # Flatten the images to 1D arrays for feature extraction
        flattened_data = mrcs_data.reshape(N, H * W)
        
        # Use PCA to reduce dimensionality and isolate structural features
        n_components = min(N, 50) 
        pca = PCA(n_components=n_components)
        reduced_features = pca.fit_transform(flattened_data)
        
        # Group the features; each class index gets exactly one label
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(reduced_features)
        
        grouped_classes = {}
        for class_idx, label in enumerate(labels):
            if label not in grouped_classes:
                grouped_classes[label] = []
            grouped_classes[label].append(class_idx)
            
        return grouped_classes

    def layout_groups_on_canvas(self, grouped_classes, mrcs_data):
        """
        Lays out the grouped classes row by row on the QGraphicsScene.
        """
        self.scene.clear()
        
        x_spacing = 20
        y_spacing = 30
        current_y = 0
        
        # Iterate through the clusters, assigning one row per group
        for label, class_indices in grouped_classes.items():
            current_x = 0
            max_h_in_row = 0
            
            for class_idx in class_indices:
                img_array = mrcs_data[class_idx]
                
                # Replace this with your existing method that converts the 2D array to a QPixmap
                pixmap = self.convert_array_to_pixmap(img_array) 
                
                item = QGraphicsPixmapItem(pixmap)
                item.setPos(current_x, current_y)
                
                # Embed the original class index into the item for later retrieval
                item.setData(0, class_idx) 
                
                self.scene.addItem(item)
                
                current_x += pixmap.width() + x_spacing
                if pixmap.height() > max_h_in_row:
                    max_h_in_row = pixmap.height()
                    
            # Shift the Y coordinate down for the next group (new row)
            current_y += max_h_in_row + y_spacing

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())