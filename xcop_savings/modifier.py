import os
import sys
import math
import csv
import shutil
import atexit
import argparse

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, 
                             QPushButton, QTextEdit, QMessageBox, QFileDialog, 
                             QVBoxLayout, QHBoxLayout, QSizePolicy, QSlider,
                             QSplitter, QSpinBox, QGroupBox, QFormLayout, 
                             QCheckBox, QTableWidget, QHeaderView, QTableWidgetItem, 
                             QTextBrowser, QDialog, QTabWidget, QComboBox,
                             QTableView, QAbstractItemView)
from PyQt6.QtCore import (Qt, QTimer, QSize, QAbstractTableModel,
                          QModelIndex, QProcess, QPointF)
from PyQt6.QtGui import QColor

# Pyqtgraph caches OpenGL shader program IDs at class level. Top-level
# QOpenGLWidgets must share resources or a shader compiled by the main preview
# is not valid in a separate quick-look window. Qt requires this attribute to
# be set before QApplication is constructed.
if QApplication.instance() is None:
    QApplication.setAttribute(
        Qt.ApplicationAttribute.AA_ShareOpenGLContexts,
        True,
    )


# 1. Catch the arguments passed by xcop.py
parser = argparse.ArgumentParser()
parser.add_argument("--chimera-ver", type=str, default="chimera")
renderer_default = os.environ.get("XCOP_RENDERER", "auto").strip().lower()
if renderer_default not in {"auto", "opengl", "software"}:
    renderer_default = "auto"
parser.add_argument(
    "--renderer",
    choices=("auto", "opengl", "software"),
    default=renderer_default,
    help="3D preview renderer (or set XCOP_RENDERER)",
)
args, unknown = parser.parse_known_args()
CHIMERA_VER = args.chimera_ver

# 2. Recreate required global variables so the script doesn't crash
toplevel_windows = []

def show_error_message(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Error")
    msg.setText(message)
    msg.exec()

# ====================== Multi-layer Chain Modifier Class ======================

def open_modifier_layer_viewer(launch=True):
    try:
        import uuid # Required for unique temp filenames
        import numpy as np  # Required for the new detection logic
        import json
        import subprocess
        import pyqtgraph.opengl as gl
        from OpenGL.GL import glDepthFunc, GL_LEQUAL, GL_LESS
        from PyQt6.QtGui import (
            QDoubleValidator, QIntValidator, QMatrix4x4,
            QVector3D, QVector4D, QOpenGLContext, QPainter, QPen, QBrush,
        )
        from PyQt6.QtWidgets import QToolTip
        from scipy.spatial import cKDTree
    except ImportError as e:
        print(f"Error loading preview dependencies: {e}")
        return

    def display_server_looks_like_xquartz():
        """Detect an SSH-forwarded XQuartz display without relying on sys.platform."""
        if not os.environ.get("DISPLAY"):
            return False
        try:
            result = subprocess.run(
                ["xdpyinfo"],
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            return False
        display_info = f"{result.stdout}\n{result.stderr}".lower()
        return any(marker in display_info for marker in (
            "xquartz", "apple-wm", "apple-dri", "apple computer",
        ))

    renderer = args.renderer
    if renderer == "auto":
        renderer = "software" if display_server_looks_like_xquartz() else "opengl"
    use_software_renderer = renderer == "software"
    print(f"Modifier preview renderer: {renderer}")
    
    def create_outline_folder_icon():
        from PyQt6.QtGui import QPixmap, QIcon, QPainter, QPen, QColor, QPolygon
        from PyQt6.QtCore import Qt, QPoint
        
        # Create a 24x24 transparent canvas
        pixmap = QPixmap(24, 24)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Apply your theme's light green
        pen = QPen(QColor('#98c379'))
        pen.setWidth(2)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        
        # Draw the folder shape (tab on the left)
        poly = QPolygon([
            QPoint(2, 6),   # Top-left tab
            QPoint(9, 6),   # Top-right tab
            QPoint(11, 9),  # Fold down
            QPoint(22, 9),  # Top-right edge
            QPoint(22, 20), # Bottom-right
            QPoint(2, 20)   # Bottom-left
        ])
        painter.drawPolygon(poly)
        
        # Draw the inner line to give it that 'folder' look
        painter.drawLine(2, 9, 11, 9)
        
        painter.end()
        return QIcon(pixmap)


    def context_safe_shader_program(item_class, program_cache):
        """Return a pyqtgraph shader compiled for the current GL share-group."""
        context = QOpenGLContext.currentContext()
        if (
            context is None
            or not hasattr(item_class, "_shaderProgram")
            or not hasattr(item_class, "getShaderProgram")
        ):
            return item_class.getShaderProgram()

        share_group = context.shareGroup()
        cache_key = share_group if share_group is not None else context
        cached_program = program_cache.get(cache_key)
        if cached_program is not None:
            return cached_program

        previous_program = item_class._shaderProgram
        item_class._shaderProgram = None
        try:
            program = item_class.getShaderProgram()
        finally:
            item_class._shaderProgram = previous_program

        program_cache[cache_key] = program
        try:
            cache_key.destroyed.connect(
                lambda _object=None, key=cache_key, cache=program_cache:
                cache.pop(key, None)
            )
        except (AttributeError, TypeError):
            pass
        return program


    class ContextSafeLinePlotItem(gl.GLLinePlotItem):
        _programs_by_share_group = {}

        @staticmethod
        def getShaderProgram():
            return context_safe_shader_program(
                gl.GLLinePlotItem,
                ContextSafeLinePlotItem._programs_by_share_group,
            )


    class ContextSafeScatterPlotItem(gl.GLScatterPlotItem):
        _programs_by_share_group = {}

        @staticmethod
        def getShaderProgram():
            return context_safe_shader_program(
                gl.GLScatterPlotItem,
                ContextSafeScatterPlotItem._programs_by_share_group,
            )


    class OpenGLChainCanvas(gl.GLViewWidget):
        """OpenGL chain trace preview shared by Modifier and Layer Viewer."""

        LABEL_STYLE = (
            "QLabel { color: black; background-color: rgba(255, 255, 255, 230); "
            "border: 1px solid #333333; border-radius: 3px; padding: 2px; "
            "font-size: 11pt; font-weight: bold; }"
        )

        def __init__(
            self,
            parent=None,
            width=8,
            height=5,
            dpi=50,
            highlight_selected=False,
            center_on_labels=False,
            centered_labels=True,
        ):
            super().__init__(parent, rotationMethod='quaternion')
            # Match the preferred and minimum sizes advertised by the former
            # Matplotlib canvas. QSplitter uses these hints for its startup
            # allocation; QOpenGLWidget otherwise reports an invalid (-1, -1)
            # hint and lets the controls pane consume almost all available
            # width.
            self._preferred_size = QSize(
                max(1, round(width * dpi)),
                max(1, round(height * dpi)),
            )
            self.setBackgroundColor('#2C2C2C')
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.updateGeometry()

            self.highlight_selected = highlight_selected
            self.center_on_labels = center_on_labels
            self.centered_labels = centered_labels
            self.global_center = (0.0, 0.0, 0.0)
            self.max_range = 10.0
            self.zoom_level = 50
            self.slider_callback = None
            self._chain_labels = []
            self._label_update_pending = False
            self.setCameraPosition(distance=30.0, elevation=30, azimuth=-60)

        def sizeHint(self):
            return QSize(self._preferred_size)

        def minimumSizeHint(self):
            return QSize(10, 10)

        def _clear_scene(self):
            self.clear()
            for label, _position in self._chain_labels:
                label.hide()
                label.deleteLater()
            self._chain_labels = []

        def wheelEvent(self, event):
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta == 0:
                event.ignore()
                return
            step = 5 if delta > 0 else -5
            self.zoom_level = max(1, min(100, self.zoom_level + step))
            if self.slider_callback:
                self.slider_callback(self.zoom_level)
            self.apply_zoom()
            event.accept()

        def mouseMoveEvent(self, event):
            super().mouseMoveEvent(event)
            self._schedule_label_update()

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self._schedule_label_update()

        def set_zoom_from_slider(self, value):
            self.zoom_level = value
            self.apply_zoom()

        def apply_zoom(self):
            if self.max_range == 0:
                return
            factor = 2.0 - ((self.zoom_level / 100.0) * 1.9)
            if factor < 0.05:
                factor = 0.05
            radius = self.max_range * factor
            self.setCameraPosition(
                pos=QVector3D(*self.global_center),
                distance=max(radius * 2.25, 0.05),
            )
            self._schedule_label_update()

        def _schedule_label_update(self):
            if self._label_update_pending:
                return
            self._label_update_pending = True
            QTimer.singleShot(0, self._update_label_positions)

        def _screen_position(self, position):
            width, height = self.width(), self.height()
            if width <= 0 or height <= 0:
                return None
            vector = QVector4D(*position, 1.0)
            region = (0, 0, width, height)
            viewport = (0, 0, width, height)
            clip = (
                self.projectionMatrix(region, viewport)
                * self.viewMatrix()
                * vector
            )
            if clip.w() <= 0:
                return None
            ndc_x = clip.x() / clip.w()
            ndc_y = clip.y() / clip.w()
            if not (-1.05 <= ndc_x <= 1.05 and -1.05 <= ndc_y <= 1.05):
                return None
            return (
                (ndc_x + 1.0) * width / 2.0,
                (1.0 - ndc_y) * height / 2.0,
            )

        def _update_label_positions(self):
            self._label_update_pending = False
            for label, position in self._chain_labels:
                screen_position = self._screen_position(position)
                if screen_position is None:
                    label.hide()
                    continue
                if self.centered_labels:
                    x = int(screen_position[0] - label.width() / 2)
                    y = int(screen_position[1] - label.height() / 2)
                else:
                    # Matplotlib's default text anchor is left/baseline; keep
                    # that placement in Layer Viewer.
                    x = int(screen_position[0])
                    y = int(screen_position[1] - label.height() + 3)
                label.move(x, y)
                label.show()
                label.raise_()

        def _add_label(self, text, position):
            label = QLabel(str(text), self)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            label.setStyleSheet(self.LABEL_STYLE)
            label.adjustSize()
            self._chain_labels.append((label, tuple(position)))

        def plot_chains(
            self,
            chains_data,
            label_ids=None,
            preserve_view=False,
            waters=None,
            ions=None,
            chain_styles=None,
        ):
            self._clear_scene()
            if not chains_data:
                self.update()
                return

            label_ids = set(label_ids or ())
            chain_styles = chain_styles or {}
            chain_ids = sorted(chains_data.keys())
            should_refit = not preserve_view or self.max_range == 10.0
            if should_refit:
                calc_targets = (
                    label_ids
                    if self.center_on_labels and label_ids
                    else chain_ids
                )
                all_points = [
                    point
                    for chain_id in calc_targets
                    for point in chains_data.get(chain_id, ())
                ]
                if not all_points:
                    return
                points = np.asarray(all_points, dtype=float)
                center = points.mean(axis=0)
                self.global_center = tuple(center)
                distances = np.linalg.norm(points - center, axis=1)
                self.max_range = float(distances.max()) or 10.0

            for chain_id in chain_ids:
                coords = chains_data[chain_id]
                if not coords:
                    continue
                selected = chain_id in label_ids
                if chain_id in chain_styles:
                    color, line_width = chain_styles[chain_id]
                elif self.highlight_selected and selected:
                    color, line_width = (0.596, 0.765, 0.475, 0.9), 3.0
                else:
                    color = (1.0, 1.0, 1.0, 0.6 if not self.highlight_selected else 0.3)
                    line_width = 1.5 if not self.highlight_selected else 1.0
                line = ContextSafeLinePlotItem(
                    pos=np.asarray(coords, dtype=float),
                    color=color,
                    width=line_width,
                    antialias=True,
                    mode='line_strip',
                    glOptions='translucent',
                )
                self.addItem(line)
                if selected:
                    self._add_label(chain_id, coords[0])

            if waters:
                self.addItem(ContextSafeScatterPlotItem(
                    pos=np.asarray(waters, dtype=float),
                    color=(0.678, 0.847, 0.902, 0.8),
                    size=10.0,
                    pxMode=True,
                    glOptions='translucent',
                ))
            if ions:
                self.addItem(ContextSafeScatterPlotItem(
                    pos=np.asarray(ions, dtype=float),
                    color=(1.0, 0.714, 0.757, 0.9),
                    size=15.0,
                    pxMode=True,
                    glOptions='translucent',
                ))

            self.apply_zoom()


    class SoftwareChainCanvas(QWidget):
        """CPU-rendered 3D chain preview for XQuartz and other remote displays."""

        def __init__(
            self,
            parent=None,
            width=8,
            height=5,
            dpi=50,
            highlight_selected=False,
            center_on_labels=False,
            centered_labels=True,
        ):
            super().__init__(parent)
            self._preferred_size = QSize(
                max(1, round(width * dpi)),
                max(1, round(height * dpi)),
            )
            self.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Expanding,
            )
            self.setMouseTracking(True)
            self.highlight_selected = highlight_selected
            self.center_on_labels = center_on_labels
            self.centered_labels = centered_labels
            self.global_center = np.zeros(3, dtype=float)
            self.max_range = 10.0
            self.zoom_level = 50
            self.slider_callback = None
            self._chains = {}
            self._label_ids = set()
            self._waters = np.empty((0, 3), dtype=float)
            self._ions = np.empty((0, 3), dtype=float)
            self._chain_styles = {}
            self._rotation_x = math.radians(30.0)
            self._rotation_y = math.radians(-60.0)
            self._pan = np.zeros(2, dtype=float)
            self._last_mouse_pos = None

        def sizeHint(self):
            return QSize(self._preferred_size)

        def minimumSizeHint(self):
            return QSize(10, 10)

        @staticmethod
        def _qcolor(color):
            if isinstance(color, QColor):
                return QColor(color)
            values = tuple(color)
            if values and max(values) <= 1.0:
                values = tuple(round(value * 255) for value in values)
            if len(values) == 3:
                values += (255,)
            return QColor(*[max(0, min(255, int(value))) for value in values[:4]])

        def _rotation_matrix(self):
            cx, sx = math.cos(self._rotation_x), math.sin(self._rotation_x)
            cy, sy = math.cos(self._rotation_y), math.sin(self._rotation_y)
            rotate_x = np.array(((1, 0, 0), (0, cx, -sx), (0, sx, cx)))
            rotate_y = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)))
            return rotate_y @ rotate_x

        def _transform(self, points):
            points = np.asarray(points, dtype=float)
            if not len(points):
                return points.reshape((0, 3))
            return (points - self.global_center) @ self._rotation_matrix().T

        def _scale(self):
            zoom_divisor = max(0.05, 2.0 - (self.zoom_level / 100.0) * 1.9)
            extent = max(float(self.max_range), 0.001) * zoom_divisor
            return max(1.0, min(self.width(), self.height()) * 0.42 / extent)

        def _project(self, points):
            rotated = self._transform(points)
            scale = self._scale()
            projected = np.empty((len(rotated), 2), dtype=float)
            projected[:, 0] = self.width() / 2.0 + self._pan[0] + rotated[:, 0] * scale
            projected[:, 1] = self.height() / 2.0 + self._pan[1] - rotated[:, 1] * scale
            return projected, rotated[:, 2]

        def _screen_position(self, position):
            if self.width() <= 0 or self.height() <= 0:
                return None
            projected, _depth = self._project([position])
            x, y = projected[0]
            if not (-20 <= x <= self.width() + 20 and -20 <= y <= self.height() + 20):
                return None
            return float(x), float(y)

        def orbit(self, azimuth, elevation):
            self._rotation_y += math.radians(float(azimuth) * 0.5)
            self._rotation_x += math.radians(float(elevation) * 0.5)
            self.update()

        def pan(self, x, y, _z=0, relative="view"):
            self._pan += (float(x), float(y))
            self.update()

        def fit_structure(self):
            self.zoom_level = 50
            self._pan[:] = 0
            self.update()

        def set_zoom_from_slider(self, value):
            self.zoom_level = int(value)
            self.apply_zoom()

        def apply_zoom(self):
            self.update()

        def wheelEvent(self, event):
            delta = event.angleDelta().y() or event.angleDelta().x()
            if delta == 0:
                event.ignore()
                return
            self.zoom_level = max(1, min(100, self.zoom_level + (5 if delta > 0 else -5)))
            if self.slider_callback:
                self.slider_callback(self.zoom_level)
            self.update()
            event.accept()

        def mousePressEvent(self, event):
            self._last_mouse_pos = event.position().toPoint()
            event.accept()

        def mouseMoveEvent(self, event):
            current = event.position().toPoint()
            if self._last_mouse_pos is None:
                self._last_mouse_pos = current
            difference = current - self._last_mouse_pos
            if event.buttons() & Qt.MouseButton.LeftButton:
                self.orbit(-difference.x(), difference.y())
            elif event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
                self.pan(difference.x(), difference.y())
            self._last_mouse_pos = current
            event.accept()

        def mouseReleaseEvent(self, event):
            self._last_mouse_pos = None
            event.accept()

        def plot_chains(
            self,
            chains_data,
            label_ids=None,
            preserve_view=False,
            waters=None,
            ions=None,
            chain_styles=None,
        ):
            self._chains = {
                chain_id: np.asarray(coords, dtype=float)
                for chain_id, coords in chains_data.items()
                if len(coords)
            }
            self._label_ids = set(label_ids or ())
            self._chain_styles = dict(chain_styles or {})
            self._waters = np.asarray(
                [] if waters is None else waters, dtype=float
            ).reshape((-1, 3))
            self._ions = np.asarray(
                [] if ions is None else ions, dtype=float
            ).reshape((-1, 3))

            should_refit = not preserve_view or self.max_range == 10.0
            if should_refit and self._chains:
                target_ids = (
                    self._label_ids
                    if self.center_on_labels and self._label_ids
                    else self._chains.keys()
                )
                point_sets = [
                    self._chains[chain_id]
                    for chain_id in target_ids
                    if chain_id in self._chains and len(self._chains[chain_id])
                ]
                if point_sets:
                    points = np.vstack(point_sets)
                    self.global_center = points.mean(axis=0)
                    distances = np.linalg.norm(points - self.global_center, axis=1)
                    self.max_range = float(distances.max()) or 10.0
                    self._pan[:] = 0
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#2C2C2C"))

            if not self._chains:
                painter.setPen(QColor("#8a8a8a"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No structure loaded")
                return

            chain_layers = []
            for chain_id, coords in self._chains.items():
                projected, depths = self._project(coords)
                chain_layers.append((float(depths.mean()), chain_id, projected))

            for _depth, chain_id, projected in sorted(chain_layers):
                selected = chain_id in self._label_ids
                if chain_id in self._chain_styles:
                    color, line_width = self._chain_styles[chain_id]
                elif self.highlight_selected and selected:
                    color, line_width = (0.596, 0.765, 0.475, 0.9), 3.0
                else:
                    color = (1.0, 1.0, 1.0, 0.6 if not self.highlight_selected else 0.3)
                    line_width = 1.5 if not self.highlight_selected else 1.0
                painter.setPen(QPen(self._qcolor(color), float(line_width)))
                for first, second in zip(projected, projected[1:]):
                    painter.drawLine(QPointF(*first), QPointF(*second))

                if selected and len(projected):
                    label_position = projected[0]
                    label_rect_x = label_position[0] - (12 if self.centered_labels else 0)
                    label_rect_y = label_position[1] - 20
                    painter.setPen(QColor("#111111"))
                    painter.setBrush(QBrush(QColor(245, 245, 245, 230)))
                    painter.drawRoundedRect(
                        int(label_rect_x), int(label_rect_y), 28, 18, 3, 3
                    )
                    painter.drawText(
                        int(label_rect_x), int(label_rect_y), 28, 18,
                        Qt.AlignmentFlag.AlignCenter, str(chain_id),
                    )

            painter.setPen(Qt.PenStyle.NoPen)
            for points, color, radius in (
                (self._waters, QColor(173, 216, 235, 210), 4.0),
                (self._ions, QColor(255, 182, 193, 230), 6.0),
            ):
                if not len(points):
                    continue
                painter.setBrush(QBrush(color))
                projected, _depths = self._project(points)
                for point in projected:
                    painter.drawEllipse(QPointF(*point), radius, radius)


    ChainCanvasBase = SoftwareChainCanvas if use_software_renderer else OpenGLChainCanvas


    class Mol3DCanvas(ChainCanvasBase):
        def __init__(self, parent=None, width=8, height=5, dpi=50):
            super().__init__(
                parent,
                width,
                height,
                dpi,
                highlight_selected=False,
                center_on_labels=True,
                centered_labels=True,
            )


    def generate_chain_id(index):
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        if index < len(chars): return chars[index]
        n = index - len(chars); base = len(chars)
        return chars[n // base] + chars[n % base]


    class TrimDialog(QDialog):
        def __init__(self, layers, full_cif_data, original_filename, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Trim Layers")
            self.resize(800, 600)
            self.layers = layers  
            self.full_cif_data = full_cif_data 
            self.original_filename = original_filename
            self.initUI()
            
            # Apply global theme styling directly from the active PyQt application
            qt_app = QApplication.instance()
            if qt_app:
                self.setStyleSheet(qt_app.styleSheet())

        def initUI(self):
            layout = QVBoxLayout()
            layout.addWidget(QLabel("Select the layers you want to KEEP\nChains will be automatically renamed to maximum 2 letters upon saving to avoid truncations when opening/saving in Chimera(X)"))
            self.table = QTableWidget()
            self.table.setColumnCount(4)
            self.table.setHorizontalHeaderLabels(["Keep", "Rank", "Original Chains", "Residues"])
            self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
            self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            self.populate_table()
            layout.addWidget(self.table)
            btn_layout = QHBoxLayout(); btn_layout.addStretch()
            self.btn_save = QPushButton("Save New CIF")
            self.btn_save.clicked.connect(self.save_cif)
            btn_layout.addWidget(self.btn_save)
            layout.addLayout(btn_layout)
            self.setLayout(layout)

        def update_data(self, layers, full_cif_data, filename):
            self.layers = layers
            self.full_cif_data = full_cif_data
            self.original_filename = filename
            self.populate_table()

        def populate_table(self):
            self.table.setRowCount(len(self.layers))
            self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus) # Remove focus rectangle from table
            self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection) # Disable global selection highlighting

            for i, layer in enumerate(self.layers):
                # Column 0: Centered Checkbox Widget
                widget = QWidget()
                chk = QCheckBox()
                layout = QHBoxLayout(widget)
                layout.addWidget(chk)
                layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                self.table.setCellWidget(i, 0, widget)

                # Column 1: Rank (Unclickable)
                rank_item = QTableWidgetItem(str(i + 1))
                rank_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                rank_item.setFlags(Qt.ItemFlag.ItemIsEnabled) 
                self.table.setItem(i, 1, rank_item)

                # Column 2: Chains (Unclickable)
                chains_item = QTableWidgetItem(", ".join(layer['chains']))
                chains_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, 2, chains_item)

                # Column 3: Residues (Unclickable)
                res_item = QTableWidgetItem(str(layer['residues']))
                res_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                self.table.setItem(i, 3, res_item)

        def save_cif(self):
            selected = []
            for i in range(self.table.rowCount()):
                # Fix: Retrieve the widget containing the checkbox, not the item
                widget = self.table.cellWidget(i, 0)
                checkbox = widget.findChild(QCheckBox)
                
                if checkbox and checkbox.isChecked():
                    for cid in self.layers[i]['chains']: 
                        selected.append(cid)
            
            if not selected:
                QMessageBox.warning(self, "Warning", "No layers selected.")
                return
            
            mapping = {old: generate_chain_id(idx) for idx, old in enumerate(selected)}
            name_part, ext = os.path.splitext(self.original_filename)
            
            path, _ = QFileDialog.getSaveFileName(self, "Save Trimmed CIF", f"{name_part}_trimmed.cif", "CIF Files (*.cif)")
            if path:
                try:
                    self.write_cif(path, mapping)
                    QMessageBox.information(self, "Success", f"Saved: {path}\n(Renamed {len(mapping)} chains)")
                    self.accept()
                except Exception as e: 
                    QMessageBox.Icon.Critical(self, "Error", str(e))

        def write_cif(self, path, mapping):
            data = self.full_cif_data
            cm = data['col_map']
            idx_auth = cm.get('auth_asym_id')
            idx_label = cm.get('label_asym_id')
            target_idx = idx_auth if idx_auth is not None else idx_label
            if target_idx is None: raise ValueError("No chain ID columns found.")
            with open(path, 'w') as f:
                f.writelines(data['pre_loop'])
                f.writelines(data['loop_headers'])
                for old_id, lines in data['chain_lines'].items():
                    if old_id in mapping:
                        new_id = mapping[old_id]
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) > target_idx:
                                if idx_auth is not None: parts[idx_auth] = new_id
                                if idx_label is not None: parts[idx_label] = new_id
                                f.write(" ".join(parts) + "\n")
                            else: f.write(line)


    class CIFMol3DCanvas(ChainCanvasBase):
        def __init__(self, parent=None, width=8, height=5, dpi=50):
            super().__init__(
                parent,
                width,
                height,
                dpi,
                highlight_selected=True,
                center_on_labels=False,
                centered_labels=False,
            )


    class CIFLayerIdentifier(QWidget):
        def __init__(self):
            super().__init__()
            self.created_temp_files = [] # Track specific files to delete
            
            # Ensure cleanup happens even if app crashes
            def cleanup_files():
                for f in self.created_temp_files:
                    if os.path.exists(f):
                        try: os.remove(f)
                        except: pass
            atexit.register(cleanup_files)
            
            self.working_file_path = None; self.layers_result = []; self.full_cif_data = {}
            self.initUI()
            qt_app = QApplication.instance()
            if qt_app: 
                self.setStyleSheet(qt_app.styleSheet())

        def closeEvent(self, e):
            # Cleanup temp files on close
            for f in self.created_temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            e.accept()

        def initUI(self):
            main_l = QHBoxLayout()
            splitter = QSplitter(Qt.Orientation.Horizontal)
            left = QWidget(); l_lay = QVBoxLayout(left); l_lay.setSpacing(10); l_lay.setContentsMargins(10,10,10,10)
            
            h_load = QHBoxLayout()
            self.btn_browse = QPushButton('Load PDB/CIF File')
            self.btn_browse.setIcon(create_outline_folder_icon())
            self.btn_browse.clicked.connect(self.browse_file)
            h_load.addWidget(self.btn_browse)
            self.lbl_status = QLabel("No file loaded"); self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
            h_load.addWidget(self.lbl_status); l_lay.addLayout(h_load)
            
            self.grp_ops = QGroupBox("Operations"); self.grp_ops.setEnabled(False)
            ops_l = QVBoxLayout()
            self.btn_trim = QPushButton("Trim Best Layers")
            self.btn_trim.setToolTip("Select the layers wanted and save as a new CIF file")
            self.btn_trim.clicked.connect(self.open_trim_dialog)
            ops_l.addWidget(self.btn_trim)
            self.grp_ops.setLayout(ops_l); l_lay.addWidget(self.grp_ops)
            
            grp_p = QGroupBox("Detection Parameters"); form = QFormLayout()
            self.ed_chn = QLineEdit("4"); self.ed_chn.setValidator(QIntValidator())
            form.addRow("Chains per Layer:", self.ed_chn)
            self.ed_chn.setToolTip("Number of protofilaments you have in each layer")
            self.ed_zmn = QLineEdit("0.0"); self.ed_zmn.setValidator(QDoubleValidator())
            form.addRow("Z Shift Min (Å):", self.ed_zmn)
            self.ed_zmn.setToolTip("Within a layer, the minimum shift in Z direction that a chain can be\nUsually 0Å")
            self.ed_zmx = QLineEdit("4.0"); self.ed_zmx.setValidator(QDoubleValidator())
            form.addRow("Z Shift Max (Å):", self.ed_zmx)
            self.ed_zmx.setToolTip("Within a layer, the maximum shift in Z direction that a chain can be\nCan NOT be larger than the thickness of a typical layer")
            self.ed_chn.editingFinished.connect(self.recalc); self.ed_zmn.editingFinished.connect(self.recalc); self.ed_zmx.editingFinished.connect(self.recalc)
            grp_p.setLayout(form); l_lay.addWidget(grp_p)
            
            self.txt = QTextBrowser(); self.txt.setOpenLinks(False); self.txt.anchorClicked.connect(self.link_clk)
            self.txt.setStyleSheet("font-family: Consolas, monospace;")
            l_lay.addWidget(self.txt)
            
            right = QWidget(); r_lay = QHBoxLayout(right); r_lay.setContentsMargins(0,0,0,0)
            self.cvs = CIFMol3DCanvas(self, dpi=100); r_lay.addWidget(self.cvs, 1)
            sl = QSlider(Qt.Orientation.Vertical); sl.setRange(1, 100); sl.setValue(50); sl.valueChanged.connect(lambda v: self.cvs.set_zoom_from_slider(v))
            self.cvs.slider_callback = lambda v: sl.setValue(v)
            r_lay.addWidget(sl)
            
            splitter.addWidget(left); splitter.addWidget(right)
            splitter.setSizes([600, 750]) # Force Left Panel to 500px width
            splitter.setStretchFactor(0, 4); splitter.setStretchFactor(1, 6)
            main_l.addWidget(splitter); self.setLayout(main_l)

        def recalc(self):
            if self.working_file_path: QTimer.singleShot(10, lambda: self.process_cif(self.working_file_path))
        
        def browse_file(self):
            f, _ = QFileDialog.getOpenFileName(self, "Open PDB/CIF File", "", "Model Files (*.pdb *.cif *.mmcif);;CIF Files (*.cif *.mmcif);;PDB Files (*.pdb);;All Files (*)")
            if f:
                self.load_cif(f)

        def load_cif(self, filename):
            self.loaded_filename = os.path.basename(filename)
            self.lbl_status.setText(f"Loaded: {self.loaded_filename}")
            
            # Create temp file in the SAME folder with unique ID
            folder = os.path.dirname(filename)
            unique_suffix = uuid.uuid4().hex[:8]
            dst = os.path.join(folder, f"temp_work_{unique_suffix}.cif")
            
            shutil.copy2(filename, dst)
            self.created_temp_files.append(dst) # Track for deletion
            
            self.working_file_path = dst
            self.grp_ops.setEnabled(True)
            self.txt.clear(); self.txt.append(f"Loaded {filename}")
            self.process_cif(dst)

        def link_clk(self, url):
            try:
                idx = int(url.toString())
                if 0 <= idx < len(self.layers_result):
                    self.cvs.plot_chains(self.chains_data_plot, label_ids=set(self.layers_result[idx]['chains']), preserve_view=True, waters=getattr(self, 'current_waters', []), ions=getattr(self, 'current_ions', []))
            except: pass

        def open_trim_dialog(self):
            if not self.layers_result: return
            
            # Check if dialog is already open; if so, bring to front
            if hasattr(self, 'trim_dialog') and self.trim_dialog and self.trim_dialog.isVisible():
                self.trim_dialog.raise_()
                self.trim_dialog.activateWindow()
                return

            # Open as a non-modal window (allows interaction with main window)
            self.trim_dialog = TrimDialog(self.layers_result, self.full_cif_data, self.loaded_filename, self)
            self.trim_dialog.show()

        def parse_cif(self, path):
            chains_c = {}; waters = []; ions = []; full_d = {'pre_loop':[], 'loop_headers':[], 'chain_lines':{}, 'col_map':{}}
            with open(path,'r') as f: lines=f.readlines()
            # Simplified parsing logic for embedding
            in_h = False; buf = []; d_idx = -1; headers = []
            for i, l in enumerate(lines):
                s = l.strip()
                if d_idx == -1 and not in_h:
                    if s=="loop_": in_h=True; buf=[l]
                    else: full_d['pre_loop'].append(l)
                    continue
                if in_h:
                    if s.startswith("_"): buf.append(l)
                    else:
                        if any("_atom_site." in x for x in buf):
                            headers = [x.strip() for x in buf if x.strip().startswith("_")]
                            full_d['loop_headers']=buf; d_idx=i; break
                        else:
                            full_d['pre_loop'].extend(buf); in_h=False
                            if s=="loop_": in_h=True; buf=[l]
                            else: full_d['pre_loop'].append(l)
            if not headers: return {}, [], [], {}
            cm = {h.split('.')[1] if '.' in h else h: i for i, h in enumerate(headers)}
            for k, v in [("auth_asym_id", "auth_asym_id"), ("label_asym_id", "label_asym_id"), ("label_atom_id", "label_atom_id")]:
                for h in headers: 
                    if k in h: cm[k] = headers.index(h)
            
            full_d['col_map'] = cm
            
            ic = cm.get("auth_asym_id", cm.get("label_asym_id"))
            ia = cm.get("label_atom_id")
            ix, iy, iz = cm.get("Cartn_x"), cm.get("Cartn_y"), cm.get("Cartn_z")
            ig = cm.get("group_PDB")
            ir = cm.get("label_comp_id", cm.get("auth_comp_id"))
            
            for i in range(d_idx, len(lines)):
                ln = lines[i]; s = ln.strip()
                if not s or s.startswith(("#", "_", "loop_")): continue
                p = s.split()
                if len(p) < len(headers): continue
                cid = p[ic]
                if cid not in full_d['chain_lines']: full_d['chain_lines'][cid] = []
                full_d['chain_lines'][cid].append(ln)
                
                try:
                    coord = np.array([float(p[ix]), float(p[iy]), float(p[iz])])
                except: continue
                
                STANDARD_MODIFIED_RESIDUES = {"MSE", "SEP", "TPO", "PTR", "PCA", "CME", "CSO", "KCX", "ALY", "MLY", "SME", "CSX", "FME", "TYS", "LLP", "SAC"}
                res_name = p[ir] if ir is not None else ""
                
                if ia is not None and p[ia] == "CA":
                    if cid not in chains_c: chains_c[cid]=[]
                    chains_c[cid].append(coord)
                elif ig is not None and p[ig] == "HETATM" and res_name not in STANDARD_MODIFIED_RESIDUES:
                    atom_name = p[ia] if ia is not None else ""
                    if res_name in ["HOH", "WAT"]:
                        if atom_name.startswith("O"): waters.append(tuple(coord))
                    else:
                        ions.append(tuple(coord))
            return chains_c, waters, ions, full_d

        def process_cif(self, path):
            chn_max = int(self.ed_chn.text()); zmn = float(self.ed_zmn.text()); zmx = float(self.ed_zmx.text())
            try:
                self.txt.append("Parsing..."); coords, waters, ions, self.full_cif_data = self.parse_cif(path)
                if not coords: self.txt.append("No CA atoms found."); return
                self.chains_data_plot = {c: [tuple(p) for p in coords[c]] for c in coords}
                self.current_waters = waters
                self.current_ions = ions
                cents = {c: np.mean(coords[c], axis=0) for c in coords}
                pool = sorted(list(coords.keys()), key=lambda c: cents[c][2])
                self.layers_result = []
                while pool:
                    ref = pool.pop(0); cur = [ref]; rz = cents[ref][2]
                    cands = sorted([(abs(cents[c][2]-rz), c) for c in pool if zmn <= abs(cents[c][2]-rz) <= zmx], key=lambda x:x[0])
                    sel = [c[1] for c in cands[:chn_max-1]]
                    cur.extend(sel); 
                    for c in sel: pool.remove(c)
                    res_cnt = sum(len(coords[c]) for c in cur)
                    self.layers_result.append({'chains': cur, 'count': len(cur), 'residues': res_cnt})
                self.layers_result.sort(key=lambda x: (x['count'], x['residues']), reverse=True)
                
                h_lines = []
                high_ids = set()
                for i, lay in enumerate(self.layers_result):
                    lnk = f"<a href='{i}' style='color: #97c379; font-weight: bold; text-decoration: none;'>Layer {i+1}</a>"
                    h_lines.append(f"{lnk}: {', '.join(lay['chains'])} [Chains, Residues]:{lay['count']}, {lay['residues']}<br>")
                    if i==0: high_ids.update(lay['chains'])
                self.txt.setHtml("<br>".join(h_lines))
                self.cvs.plot_chains(self.chains_data_plot, label_ids=high_ids, waters=waters, ions=ions)
                
                # Update Trim Dialog if it is open
                if hasattr(self, 'trim_dialog') and self.trim_dialog and self.trim_dialog.isVisible():
                    self.trim_dialog.update_data(self.layers_result, self.full_cif_data, self.loaded_filename)

            except Exception as e: self.txt.append(str(e))


    class PDBLayerIdentifier(QWidget):
        
        CHAIN_RECORD_SPECS = {
            "DBREF":  [(11, 13)], "DBREF1": [(11, 13)], "DBREF2": [(11, 13)],
            "SEQADV": [(15, 17)], "MODRES": [(15, 17)], "SEQRES": [(10, 12)],
            "HET":    [(11, 13)], "SSBOND": [(14, 16), (28, 30)], "CISPEP": [(14, 16), (28, 30)],
            "LINK":   [(20, 22), (50, 52)], "SITE":   [(21, 23), (32, 34), (43, 45), (53, 56)],
            "ATOM":   [(20, 22)], "ANISOU": [(20, 22)], "TER":    [(20, 22)],
            "HETATM": [(20, 22)], "HELIX":  [(18, 20), (30, 32)], "SHEET":  [(20, 22), (31, 33)]
        }

        def __init__(self):
            super().__init__()
            self.final_sandwiches = [] 
            self.detected_layers = 0 
            self.created_temp_files = [] # Track specific files
            
            # Ensure cleanup happens even if app crashes
            def cleanup_files():
                for f in self.created_temp_files:
                    if os.path.exists(f):
                        try: os.remove(f)
                        except: pass
            atexit.register(cleanup_files)

            self.working_file_path = None 
            self.original_filename_display = ""
            self.initUI()

        def closeEvent(self, event):
            # Cleanup temp files on close
            for f in self.created_temp_files:
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
            event.accept()

        def initUI(self):
            main_layout = QHBoxLayout()
            splitter = QSplitter(Qt.Orientation.Horizontal)

            # ================= LEFT PANE =================
            left_widget = QWidget()
            layout = QVBoxLayout(left_widget)
            layout.setSpacing(10)
            layout.setContentsMargins(10, 10, 10, 10)

            hb_load = QHBoxLayout()
            self.btn_browse = QPushButton('Load PDB/CIF File')
            self.btn_browse.setIcon(create_outline_folder_icon())
            self.btn_browse.clicked.connect(self.browse_file)
            hb_load.addWidget(self.btn_browse)
            
            self.lbl_status = QLabel("No file loaded")
            self.lbl_status.setStyleSheet("color: gray; font-style: italic;")
            hb_load.addWidget(self.lbl_status)

            # Add a stretch to act like a spring, pushing the checkbox to the far right
            hb_load.addStretch()

            self.chk_anti = QCheckBox("Anti")
            self.chk_anti.setToolTip("Check this if loading an anti-parallel amyloid")
            
            def toggle_anti_mode(state):
                # Temporarily block signals so we don't trigger 3 recalculations at once
                self.edit_z_min.blockSignals(True)
                self.edit_z_max.blockSignals(True)
                self.edit_xy_limit.blockSignals(True)
                
                if state:
                    # Anti mode
                    self.edit_z_min.setText("9.0") 
                    self.edit_z_max.setText("11.5")
                    self.edit_xy_limit.setText("6.0")
                else:
                    # Standard In-register defaults
                    self.edit_z_min.setText("4.6")
                    self.edit_z_max.setText("5.0")
                    self.edit_xy_limit.setText("3.0")
                    
                self.edit_z_min.blockSignals(False)
                self.edit_z_max.blockSignals(False)
                self.edit_xy_limit.blockSignals(False)
                
                # Trigger a single recalculation now that UI is updated
                self.on_param_change()
                
            self.chk_anti.stateChanged.connect(toggle_anti_mode)
            hb_load.addWidget(self.chk_anti)
            
            layout.addLayout(hb_load)

            self.grp_ops = QGroupBox("Operations (Applied to Temporary Copy)")
            self.grp_ops.setEnabled(False) 
            v_ops = QVBoxLayout()

            hb_trim = QHBoxLayout()
            self.btn_trim = QPushButton("Trim/Expand Layers")
            self.btn_trim.setToolTip("Trim or expand the structure to the target number of layers")
            self.btn_trim.clicked.connect(self.action_trim)
            hb_trim.addWidget(self.btn_trim)
            hb_trim.addWidget(QLabel("Target Layers:"))
            self.edit_trim = QLineEdit("5")
            self.edit_trim.setValidator(QIntValidator(1, 500))
            self.edit_trim.setFixedWidth(50)
            hb_trim.addWidget(self.edit_trim)
            hb_trim.addStretch()
            v_ops.addLayout(hb_trim)

            hb_pf_trim = QHBoxLayout()
            self.btn_remove_pf = QPushButton("Trim Protofilament")
            self.btn_remove_pf.setToolTip("Trim away the selected protofilament entirely")
            self.btn_remove_pf.clicked.connect(self.action_remove_pf)
            hb_pf_trim.addWidget(self.btn_remove_pf)
            self.cmb_protofilaments = QComboBox()
            hb_pf_trim.addWidget(self.cmb_protofilaments)
            v_ops.addLayout(hb_pf_trim)

            hb_ren = QHBoxLayout()
            self.btn_rename = QPushButton("Rename Chains")
            self.btn_rename.setToolTip("Rename the chain IDs")
            self.btn_rename.clicked.connect(self.action_rename)
            hb_ren.addWidget(self.btn_rename)
            hb_ren.addWidget(QLabel("(Rename Chain ID from the Middle Out)"))
            v_ops.addLayout(hb_ren)

            # --- NEW: Evaluate Beta Sheet ---
            self.btn_beta = QPushButton("Re-evaluate β-sheet structure")
            self.btn_beta.setToolTip("Re-evaluate β-sheet structure and annote to the PDB file\nUsing the gold-standard DSSP algorithm in Chimera\nMUST be greater than 1 layer")
            self.btn_beta.clicked.connect(self.action_evaluate_beta)
            self.btn_beta.setEnabled(False) 
            v_ops.addWidget(self.btn_beta)
            
            # Row 2: Checkbox (Added directly to vertical layout to sit on the next line)
            self.chk_numbers_first = QCheckBox("Use Numbers (0-9) after Z")
            self.chk_numbers_first.setToolTip("Use 0-9 before moving to double letters (AA)\nUseful for older software like Chimera no X\nYou can rename again without numbers after you done molmapping your final models in Chimera when making FSC curves\nYou don't need this if you have less than 5] protofilaments in a single layer\nAlso there is no difference if you have more than [7 protofilaments in a single layer")
            v_ops.addWidget(self.chk_numbers_first)

            self.chk_only_numbers = QCheckBox("Use only numbers")
            self.chk_only_numbers.setToolTip("Rename chains using strictly integers (0, 1, ... 99)")
            v_ops.addWidget(self.chk_only_numbers)

            # Logic to auto de-select the other option
            self.chk_numbers_first.toggled.connect(lambda state: state and self.chk_only_numbers.setChecked(False))
            self.chk_only_numbers.toggled.connect(lambda state: state and self.chk_numbers_first.setChecked(False))

            hb_res = QHBoxLayout()
            self.btn_restrain = QPushButton("Generate Restrain File")
            self.btn_restrain.setToolTip("Generate the restrain torsion file for the current working copy")
            self.btn_restrain.clicked.connect(self.action_restrain)
            hb_res.addWidget(self.btn_restrain)
            hb_res.addWidget(QLabel("(Export .cxc File)"))
            v_ops.addLayout(hb_res)

            hb_conn = QHBoxLayout()
            self.btn_connect = QPushButton("Connect Chains")
            self.btn_connect.setToolTip("This feature helps to merge all chains into one")
            self.btn_connect.clicked.connect(self.action_connect_chain)
            hb_conn.addWidget(self.btn_connect)
            hb_conn.addWidget(QLabel("(Merge all into one chain)"))
            v_ops.addLayout(hb_conn)

            self.grp_ops.setLayout(v_ops)
            layout.addWidget(self.grp_ops)

            self.btn_save = QPushButton("Save Final PDB")
            self.btn_save.setEnabled(False)
            self.btn_save.clicked.connect(self.action_save_final)
            layout.addWidget(self.btn_save)

            # --- START: Detection Parameters with Tooltips ---
            self.grp_params = QGroupBox("Detection Parameters")
            params_layout = QFormLayout()
            params_layout.setContentsMargins(5, 5, 5, 5)

            # Helper to create text inputs with tooltips
            def create_param_input(default_val, tooltip_msg, is_int=False):
                line_edit = QLineEdit()
                line_edit.setText(str(default_val))
                if is_int:
                    line_edit.setValidator(QIntValidator())
                else:
                    line_edit.setValidator(QDoubleValidator())
                line_edit.editingFinished.connect(self.on_param_change) # Auto-reload
                line_edit.setToolTip(tooltip_msg) # Show explanation on hover
                return line_edit

            # 1. Z Min
            tt_zmin = "<b>Z Shift Min (Å)</b><br>Minimum vertical distance to be considered a layer stack<br><i>Default: 4.6</i>"
            self.edit_z_min = create_param_input(4.6, tt_zmin)
            params_layout.addRow("Z Shift Min (Å):", self.edit_z_min)

            # 2. Z Max
            tt_zmax = "<b>Z Shift Max (Å)</b><br>Maximum vertical distance to be considered a layer stack<br><i>Default: 5.0</i>"
            self.edit_z_max = create_param_input(5.0, tt_zmax)
            params_layout.addRow("Z Shift Max (Å):", self.edit_z_max)

            # 3. XY Limit
            tt_xy = "<b>XY Shift Limit (Å)</b><br>Maximum horizontal drift allowed between stacked subunits<br>You may need a higher value if your protofilaments stick out a lot from the centre<br><i>Default: 3.0</i>"
            self.edit_xy_limit = create_param_input(3.0, tt_xy)
            params_layout.addRow("XY Shift Limit (Å):", self.edit_xy_limit)

            # 4. Neighbor Search Count
            tt_n = "<b>Neighbor Search Count</b><br>How many closest chains to check for stacking (integer)<br>Smaller number can make processing faster<br>But generally does not affect a lot with small number of protofilaments<br><i>Default: 6</i>"
            self.edit_neighbor_count = create_param_input(6, tt_n, is_int=True)
            params_layout.addRow("Neighbor Search Count:", self.edit_neighbor_count)

            # 5. Water/Ion Z Shift (NEW)
            tt_wz = "<b>Water/Ion Z Shift (Å)</b><br>Maximum vertical distance from the protein layer centroids for a water/ion to be included<br>Decrease this value if extra molecule is included, vice versa<br><i>Default: 4.0</i>"
            self.edit_water_z = create_param_input(4.0, tt_wz)
            params_layout.addRow("Water/Ion Z Shift:", self.edit_water_z)

            self.grp_params.setLayout(params_layout)
            layout.addWidget(self.grp_params)
            # --- END: Detection Parameters ---

            self.text_area = QTextEdit()
            self.text_area.setReadOnly(True)
            font = self.text_area.font()
            font.setFamily("Courier New")
            font.setPointSize(9)
            self.text_area.setFont(font)
            layout.addWidget(self.text_area)

            # ================= RIGHT PANE =================
            right_widget = QWidget()
            r_layout = QHBoxLayout(right_widget)
            r_layout.setContentsMargins(0,0,0,0)
            r_layout.setSpacing(2)

            # 1. Canvas
            self.mol_canvas = Mol3DCanvas(self, width=8, height=5, dpi=100)
            r_layout.addWidget(self.mol_canvas, 1)

            # 2. Zoom Slider
            self.zoom_slider = QSlider(Qt.Orientation.Vertical)
            self.zoom_slider.setRange(1, 100)
            self.zoom_slider.setValue(50)
            self.zoom_slider.setTickPosition(QSlider.TickPosition.TicksRight)
            self.zoom_slider.setTickInterval(10)
            self.zoom_slider.valueChanged.connect(self.on_slider_change)
            r_layout.addWidget(self.zoom_slider)

            self.mol_canvas.slider_callback = self.update_slider_visual

            splitter.addWidget(left_widget)
            splitter.addWidget(right_widget)
            splitter.setStretchFactor(0, 4) 
            splitter.setStretchFactor(1, 6) 

            main_layout.addWidget(splitter)
            self.setLayout(main_layout)

        def on_slider_change(self, value):
            self.mol_canvas.set_zoom_from_slider(value)

        def update_slider_visual(self, value):
            self.zoom_slider.blockSignals(True)
            self.zoom_slider.setValue(value)
            self.zoom_slider.blockSignals(False)
        
        def on_param_change(self):
            # Only run if a file has been loaded
            if self.working_file_path:
                self.text_area.append("\n--- Parameters changed: Re-calculating ---")
                # Add a small delay or just run it directly
                QTimer.singleShot(10, lambda: self.process_pdb(self.working_file_path))

        def get_param(self, widget, default):
            try:
                val = float(widget.text())
                return val
            except ValueError:
                return default
        
        def get_kept_heteroatoms(self, keep_layer_indices, xy_limit, water_z_limit):
            if not self.working_file_path or not self.final_sandwiches: return set()
            target_z_planes = []
            chain_centroids = {cid: self.results[cid]['centroid'] for cid in self.results}
            max_depth = self.detected_layers
            
            for layer_idx in keep_layer_indices:
                if layer_idx < 0 or layer_idx >= max_depth: continue
                layer_z_values = []
                for pf in self.final_sandwiches:
                    if layer_idx < len(pf):
                        cid = pf[layer_idx]
                        if cid in chain_centroids:
                            layer_z_values.append(chain_centroids[cid][2])
                if layer_z_values:
                    target_z_planes.append(sum(layer_z_values) / len(layer_z_values))

            if not target_z_planes: return set()

            het_atoms = []
            molecule_map = {}
            is_cif = self.working_file_path.lower().endswith(('.cif', '.mmcif'))
            
            with open(self.working_file_path, 'r') as f:
                headers = []
                in_atom_site = False
                for i, line in enumerate(f):
                    s = line.strip()
                    if is_cif:
                        if s == "loop_": continue
                        if s.startswith("_atom_site."):
                            in_atom_site = True
                            headers.append(s.split('.')[1])
                            continue
                        if in_atom_site and s and not s.startswith("_") and not s.startswith("#"):
                            parts = s.split()
                            if len(parts) >= len(headers):
                                group = parts[headers.index("group_PDB")]
                                if group in ["HETATM", "ATOM"]:
                                    res_name = parts[headers.index("label_comp_id")]
                                    col_c = headers.index("auth_asym_id") if "auth_asym_id" in headers else headers.index("label_asym_id")
                                    chain_id = parts[col_c]
                                    col_seq = headers.index("auth_seq_id") if "auth_seq_id" in headers else headers.index("label_seq_id")
                                    res_seq = parts[col_seq]
                                    if chain_id in self.results: continue
                                    try:
                                        x, y, z = float(parts[headers.index("Cartn_x")]), float(parts[headers.index("Cartn_y")]), float(parts[headers.index("Cartn_z")])
                                        atom_data = {'idx': i, 'x': x, 'y': y, 'z': z, 'res': res_name, 'chain': chain_id, 'res_seq': res_seq}
                                        het_atoms.append(atom_data)
                                        mol_key = (chain_id, res_seq)
                                        if mol_key not in molecule_map: molecule_map[mol_key] = []
                                        molecule_map[mol_key].append(i)
                                    except: pass
                        elif s.startswith("#"): in_atom_site = False
                    else:
                        if line.startswith("HETATM") or line.startswith("ATOM"):
                            res_name = line[17:20].strip()
                            chain_id = line[20:22].strip()
                            res_seq = line[22:26].strip()
                            if chain_id in self.results: continue 
                            try:
                                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                                atom_data = {'idx': i, 'x': x, 'y': y, 'z': z, 'res': res_name, 'chain': chain_id, 'res_seq': res_seq}
                                het_atoms.append(atom_data)
                                
                                mol_key = (chain_id, res_seq)
                                if mol_key not in molecule_map: molecule_map[mol_key] = []
                                molecule_map[mol_key].append(i)
                            except: continue

            water_columns = [] 
            processed_indices = set()
            het_atoms.sort(key=lambda a: a['z'])
            
            for atom in het_atoms:
                if atom['idx'] in processed_indices: continue
                column = [atom]
                processed_indices.add(atom['idx'])
                matched_col = False
                for col in water_columns:
                    ref = col[0] 
                    if abs(atom['x'] - ref['x']) < 2.0 and abs(atom['y'] - ref['y']) < 2.0:
                        col.append(atom)
                        processed_indices.add(atom['idx'])
                        matched_col = True
                        break
                if not matched_col: water_columns.append(column)

            kept_line_indices = set()
            for col in water_columns:
                for target_z in target_z_planes:
                    best_atom = None
                    min_dist = float('inf')
                    for atom in col:
                        dist = abs(atom['z'] - target_z)
                        if dist < min_dist:
                            min_dist = dist
                            best_atom = atom
                    if best_atom and min_dist <= water_z_limit: 
                        mol_key = (best_atom['chain'], best_atom['res_seq'])
                        for mol_idx in molecule_map[mol_key]: kept_line_indices.add(mol_idx)

            return kept_line_indices

        # ================= LOGIC: FILE HANDLING =================

        def browse_file(self, filename=None):
            if not filename or isinstance(filename, bool):
                
                filename, _ = QFileDialog.getOpenFileName(self, "Open PDB/CIF File", "", 
                                                        "Model Files (*.pdb *.cif *.mmcif);;PDB Files (*.pdb);;CIF Files (*.cif *.mmcif);;All Files (*)")
            if filename:
                self.original_filename_display = os.path.basename(filename)
                self.lbl_status.setText(f"Loaded: {self.original_filename_display}")
                
                # Create temp file in the SAME folder with unique ID
                folder = os.path.dirname(filename)
                unique_suffix = uuid.uuid4().hex[:8]
                _, ext = os.path.splitext(filename)
                temp_path = os.path.join(folder, f"temp_working_{unique_suffix}{ext}")
                
                shutil.copy2(filename, temp_path)
                self.created_temp_files.append(temp_path) # Track it
                
                self.working_file_path = temp_path
                
                self.grp_ops.setEnabled(True)
                self.btn_save.setEnabled(True)
                
                self.text_area.clear()
                self.text_area.append(f"Loaded {filename}")

                self.check_spatial_and_id_duplicates(self.working_file_path)

                self.process_pdb(self.working_file_path)

        def action_rename(self):
            if not self.working_file_path: return
            try:
                old_sandwiches = self.final_sandwiches
                mapping = self.create_renaming_mapping(old_sandwiches)
                
                folder = os.path.dirname(self.working_file_path)
                unique_suffix = uuid.uuid4().hex[:8]
                ext = ".cif" if self.working_file_path.lower().endswith(('.cif', '.mmcif')) else ".pdb"
                new_temp = os.path.join(folder, f"temp_renamed_{unique_suffix}{ext}")
                self.created_temp_files.append(new_temp)
                
                if self.working_file_path.lower().endswith(('.cif', '.mmcif')):
                    self.write_renamed_cif(self.working_file_path, new_temp, self.final_sandwiches)
                else:
                    self.write_renamed_pdb(self.working_file_path, new_temp, self.final_sandwiches)
                
                self.working_file_path = new_temp
                self.text_area.append("\n>>>> Applied Renaming")
                self.process_pdb(self.working_file_path, old_sandwiches=old_sandwiches, rename_map=mapping)
            except Exception as e: QMessageBox.Icon.Critical(self, "Error", f"Rename failed: {e}")

        class ExpandDialog(QDialog):
            def __init__(self, current_layers, parent=None):
                super().__init__(parent)
                self.setWindowTitle("Expand Layers")
                self.current_layers = current_layers
                self.initUI()
                
            def initUI(self):
                layout = QVBoxLayout()
                self.chk_auto = QCheckBox("Auto Computing")
                self.chk_auto.setChecked(True)
                layout.addWidget(self.chk_auto)
                
                self.chk_alt = QCheckBox("Alternating Layers (A-B-A-B)")
                self.chk_alt.setToolTip("Expand using a 2-layer step to preserve alternating conformation differences.")
                layout.addWidget(self.chk_alt)
                
                self.chk_compute_axis = QCheckBox("Compute Helical Axis")
                self.chk_compute_axis.setToolTip(f"Compute the helical axis if selected, otherwise will use the z-axis directly as the helical axis\nIf the tilt is within the maixmum accuracy of the PDB/CIF file, auto-snap to the z-axis depending on how many available decimal digits in the coordinates that your file has")
                if self.current_layers <= 1:
                    self.chk_compute_axis.setChecked(False)
                else:
                    self.chk_compute_axis.setChecked(True)
                self.chk_compute_axis.stateChanged.connect(self.on_compute_axis_changed)
                layout.addWidget(self.chk_compute_axis)
                
                form_layout = QFormLayout()
                self.edit_twist = QLineEdit("0.0")
                self.edit_rise = QLineEdit("4.8")
                self.edit_twist.setEnabled(False)
                self.edit_rise.setEnabled(False)
                self.edit_twist.setStyleSheet("color: gray;")
                self.edit_rise.setStyleSheet("color: gray;")
                
                form_layout.addRow("Twist (°):", self.edit_twist)
                form_layout.addRow("Rise (Å):", self.edit_rise)
                layout.addLayout(form_layout)
                
                self.btn_expand = QPushButton("Expand")
                self.btn_expand.clicked.connect(self.accept)
                layout.addWidget(self.btn_expand)
                self.setLayout(layout)
                
                self.chk_auto.stateChanged.connect(self.toggle_manual)
                if self.current_layers <= 1:
                    self.chk_auto.setChecked(False)
                    self.chk_auto.setEnabled(False)
                    self.chk_alt.setChecked(False)
                    self.chk_alt.setEnabled(False)
                    self.toggle_manual()
                    
            def on_compute_axis_changed(self, state):
                if self.chk_compute_axis.isChecked() and self.current_layers <= 1:
                    QMessageBox.warning(self, "Reminder", f"Only 1 layer is detected\nChecking this option will force axis computation and may cause false expansion")
                    
            def toggle_manual(self):
                is_auto = self.chk_auto.isChecked()
                self.edit_twist.setEnabled(not is_auto)
                self.edit_rise.setEnabled(not is_auto)
                if is_auto:
                    self.edit_twist.setStyleSheet("color: gray;")
                    self.edit_rise.setStyleSheet("color: gray;")
                else:
                    self.edit_twist.setStyleSheet("")
                    self.edit_rise.setStyleSheet("")

        def action_remove_pf(self):
            if not self.working_file_path or not self.final_sandwiches: return
            if self.cmb_protofilaments.count() == 0: return
            
            try:
                pf_index = self.cmb_protofilaments.currentData()
                if pf_index is None or pf_index < 0 or pf_index >= len(self.final_sandwiches): return
                
                total_layers = self.detected_layers
                keep_chains = set()
                
                for i, sandwich in enumerate(self.final_sandwiches):
                    if i != pf_index:
                        for chain_id in sandwich:
                            keep_chains.add(chain_id)
                            
                if not keep_chains:
                    QMessageBox.warning(self, "Warning", "Cannot remove the last remaining protofilament.")
                    return
                
                mid_chain = self.final_sandwiches[pf_index][len(self.final_sandwiches[pf_index])//2]
                self.text_area.append(f"Removing Protofilament with middle chain {mid_chain}...")
                
                folder = os.path.dirname(self.working_file_path)
                unique_suffix = uuid.uuid4().hex[:8]
                ext = ".cif" if self.working_file_path.lower().endswith(('.cif', '.mmcif')) else ".pdb"
                new_temp = os.path.join(folder, f"temp_trimmed_pf_{unique_suffix}{ext}")
                self.created_temp_files.append(new_temp)
                
                valid_layer_indices = range(total_layers)
                xy_lim = self.get_param(self.edit_xy_limit, 3.0)
                water_z_limit = self.get_param(self.edit_water_z, 4.0)
                
                keep_het_lines = self.get_kept_heteroatoms(valid_layer_indices, xy_lim, water_z_limit)
                
                if self.working_file_path.lower().endswith(('.cif', '.mmcif')):
                    self.write_trimmed_cif(self.working_file_path, new_temp, keep_chains, keep_het_lines)
                else:
                    self.write_trimmed_pdb(self.working_file_path, new_temp, keep_chains, keep_het_lines)
                    
                self.working_file_path = new_temp
                self.text_area.append("\n>>>> Applied Protofilament Trimming")
                self.process_pdb(self.working_file_path)
                
            except Exception as e:
                QMessageBox.Icon.Critical(self, "Error", f"Action failed: {e}")

        def action_trim(self):
            if not self.working_file_path: return
            try:
                target_layers = int(self.edit_trim.text())
            except ValueError:
                QMessageBox.warning(self, "Invalid Input", "Please enter a valid integer for Target Layers.")
                return
                
            total_layers = self.detected_layers
            
            if target_layers == total_layers:
                self.text_area.append(f"\nYour model is already having {total_layers} layer(s)")
                return
            
            try:
                folder = os.path.dirname(self.working_file_path)
                unique_suffix = uuid.uuid4().hex[:8]
                ext = ".cif" if self.working_file_path.lower().endswith(('.cif', '.mmcif')) else ".pdb"
                new_temp = os.path.join(folder, f"temp_trimmed_{unique_suffix}{ext}")
                self.created_temp_files.append(new_temp)
                
                if target_layers < total_layers:
                    start_idx = (total_layers - target_layers) // 2
                    end_idx = start_idx + target_layers
                    valid_layer_indices = range(start_idx, end_idx)
                    
                    keep_chains = set()
                    for sandwich in self.final_sandwiches:
                        for i, chain_id in enumerate(sandwich):
                            if i in valid_layer_indices: keep_chains.add(chain_id)
                    
                    self.text_area.append("Calculating water/ion retention based on layer centroids...")
                    xy_lim = self.get_param(self.edit_xy_limit, 3.0)
                    water_z_limit = self.get_param(self.edit_water_z, 4.0)
                    keep_het_lines = self.get_kept_heteroatoms(valid_layer_indices, xy_lim, water_z_limit)
                    
                    if self.working_file_path.lower().endswith(('.cif', '.mmcif')):
                        self.write_trimmed_cif(self.working_file_path, new_temp, keep_chains, keep_het_lines)
                    else:
                        self.write_trimmed_pdb(self.working_file_path, new_temp, keep_chains, keep_het_lines)
                    self.working_file_path = new_temp
                    self.text_area.append(f"\n>>>> Applied Trimming (Kept middle {target_layers} layers + associated waters)")
                
                else:
                    is_anti = getattr(self, 'chk_anti', None) and self.chk_anti.isChecked()
                    if is_anti:
                        QMessageBox.warning(self, "Action Restricted", "Expansion is currently not supported in Anti-parallel mode. You can only trim layers.")
                        return

                    if not self.working_file_path.lower().endswith(('.cif', '.mmcif')):
                        current_atoms = sum(1 for line in open(self.working_file_path) if line.startswith("ATOM") or line.startswith("HETATM"))
                        if current_atoms > 0 and total_layers > 0:
                            estimated_atoms = (current_atoms / total_layers) * target_layers
                            estimated_chains = (len(self.results) / total_layers) * target_layers
                            if estimated_atoms > 99999 or estimated_chains > 700:
                                QMessageBox.warning(
                                    self, 
                                    "PDB Limit Exceeded", 
                                    "The expanded model will exceed the maximum number of atoms (99,999) or chain IDs (2 letters) a PDB file can hold. Please convert the current model to CIF format and proceed."
                                )
                                return

                    dialog = self.ExpandDialog(total_layers, self)
                    if dialog.exec() != QDialog.DialogCode.Accepted:
                        return
                    
                    use_auto = dialog.chk_auto.isChecked()
                    use_alt = dialog.chk_alt.isChecked()
                    use_computed_axis = dialog.chk_compute_axis.isChecked()
                    manual_twist = 0.0
                    manual_rise = 0.0
                    if not use_auto:
                        try:
                            manual_twist = float(dialog.edit_twist.text())
                            manual_rise = float(dialog.edit_rise.text())
                        except ValueError:
                            QMessageBox.warning(self, "Invalid Input", "Please enter valid numbers for Twist and Rise.")
                            return

                    self.text_area.append("Calculating expansion transformations...")
                    layers_to_add = target_layers - total_layers
                    water_z_limit = self.get_param(self.edit_water_z, 4.0)
                    if self.working_file_path.lower().endswith(('.cif', '.mmcif')):
                        applied_twist, applied_rise, applied_axis, tilt_deg = self.write_expanded_cif(self.working_file_path, new_temp, layers_to_add, use_auto, manual_twist, manual_rise, water_z_limit, use_alt, use_computed_axis)
                    else:
                        applied_twist, applied_rise, applied_axis, tilt_deg = self.write_expanded_pdb(self.working_file_path, new_temp, layers_to_add, use_auto, manual_twist, manual_rise, water_z_limit, use_alt, use_computed_axis)
                    self.working_file_path = new_temp
                    self.text_area.append(f"\n>>>> Applied Expansion (Added {layers_to_add} layers, Alternating: {use_alt})")

                self.process_pdb(self.working_file_path)
                
                if target_layers > total_layers:
                    self.text_area.append(f"\nApplied Twist: {applied_twist:.5f}°")
                    self.text_area.append(f"Applied Rise: {applied_rise:.5f} Å")
                    self.text_area.append(f"Applied Axis: [{applied_axis[0]:.4f}, {applied_axis[1]:.4f}, {applied_axis[2]:.4f}], with {tilt_deg:.2f}° tilted from the Z-Axis")
                
            except Exception as e: 
                import traceback
                traceback.print_exc()
                QMessageBox.Icon.Critical(self, "Error", f"Action failed: {e}")

        def get_transform(self, coords_from, coords_to):
            import numpy as np
            min_len = min(len(coords_from), len(coords_to))
            if min_len < 3:
                return np.eye(3), np.zeros(3)
            P = np.array(coords_from[:min_len])
            Q = np.array(coords_to[:min_len])
            centroid_P = np.mean(P, axis=0)
            centroid_Q = np.mean(Q, axis=0)
            P_centered = P - centroid_P
            Q_centered = Q - centroid_Q
            H = P_centered.T @ Q_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0:
                Vt[2, :] *= -1
                R = Vt.T @ U.T
            t = centroid_Q - R @ centroid_P
            return R, t

        def write_expanded_pdb(self, input_path, output_path, layers_to_add, use_auto, manual_twist, manual_rise, water_z_limit=4.0, use_alt=False, use_computed_axis=True):
            import numpy as np
            import math
            top_layers_to_add = layers_to_add // 2
            bottom_layers_to_add = layers_to_add - top_layers_to_add

            used_chain_ids = set()
            max_res_seq = {} 
            
            orig_atoms_ters = []
            orig_hetatms = []
            headers = []
            orig_conects = []

            with open(input_path, 'r') as fin:
                for line in fin:
                    if line.startswith("MASTER") or line.startswith("END") or line.startswith("SEQRES") or line.startswith("SHEET") or line.startswith("HELIX"):
                        continue
                    elif line.startswith("ATOM") or line.startswith("TER") or line.startswith("ANISOU"):
                        orig_atoms_ters.append(line)
                        if len(line) >= 22:
                            chain_id = line[20:22].strip()
                            used_chain_ids.add(chain_id)
                    elif line.startswith("HETATM"):
                        orig_hetatms.append(line)
                    elif line.startswith("CONECT"):
                        orig_conects.append(line)
                    else:
                        headers.append(line)

            het_types = set()
            for line in orig_hetatms:
                if len(line) >= 20:
                    het_types.add(line[17:20].strip())
            
            reserved_het_chains = {}
            alphabet_backwards = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
            for i, htype in enumerate(sorted(list(het_types))):
                if i < len(alphabet_backwards):
                    c = alphabet_backwards[i]
                    reserved_het_chains[htype] = c
                    used_chain_ids.add(c)
            
            updated_orig_hetatms = []
            het_counters = {c: 1 for c in reserved_het_chains.values()}
            het_molecule_map = {}
            
            for line in orig_hetatms:
                if len(line) < 26:
                    updated_orig_hetatms.append(line)
                    continue
                res_name = line[17:20].strip()
                if res_name in reserved_het_chains:
                    new_chain = reserved_het_chains[res_name]
                    orig_chain = line[20:22].strip()
                    try: orig_res_seq = int(line[22:26])
                    except: orig_res_seq = line[22:26]
                    
                    mol_key = (orig_chain, orig_res_seq, res_name)
                    if mol_key not in het_molecule_map:
                        het_molecule_map[mol_key] = het_counters[new_chain]
                        het_counters[new_chain] += 1
                        
                    new_seq = het_molecule_map[mol_key]
                    max_res_seq[new_chain] = new_seq
                    
                    l = list(line)
                    l[21] = new_chain; l[20] = ' '
                    l[22:26] = list(f"{new_seq:>4}")
                    updated_orig_hetatms.append("".join(l))
                else:
                    updated_orig_hetatms.append(line)
                    
            orig_hetatms = updated_orig_hetatms
            
            def get_new_chain_id():
                idx = 0
                while True:
                    label = self.generate_label(idx)
                    if label not in used_chain_ids:
                        used_chain_ids.add(label)
                        return label
                    idx += 1

            core_chains = set()
            for s in self.final_sandwiches:
                core_chains.update(s)

            chain_atoms = {}
            chain_ca_coords = {}
            standalone_atoms = orig_hetatms 
            
            for line in orig_atoms_ters:
                if line.startswith("ATOM") and len(line) >= 54:
                    chain_id = line[20:22].strip()
                    if chain_id in core_chains:
                        if chain_id not in chain_atoms:
                            chain_atoms[chain_id] = []
                            chain_ca_coords[chain_id] = []
                        chain_atoms[chain_id].append(line)
                        if line[12:16].strip() == "CA":
                            try:
                                x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                                chain_ca_coords[chain_id].append([x, y, z])
                            except ValueError: pass

            chain_centroids = {}
            for cid, coords in chain_ca_coords.items():
                if coords:
                    chain_centroids[cid] = np.mean(coords, axis=0)
                    
            core_ca_coords = []
            for cid in core_chains:
                if cid in chain_ca_coords:
                    core_ca_coords.extend(chain_ca_coords[cid])
            global_centroid = np.mean(core_ca_coords, axis=0) if core_ca_coords else np.zeros(3)

            associated_atoms = {cid: [] for cid in core_chains}
            for line in standalone_atoms:
                if len(line) < 54: continue
                try:
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    best_chain = None
                    min_dist = float('inf')
                    for cid, centroid in chain_centroids.items():
                        dz = abs(z - centroid[2])
                        if dz <= water_z_limit:
                            dist = math.sqrt((x - centroid[0])**2 + (y - centroid[1])**2 + (z - centroid[2])**2)
                            if dist < min_dist:
                                min_dist = dist
                                best_chain = cid
                    if best_chain:
                        associated_atoms[best_chain].append(line)
                except ValueError:
                    pass

            expansions = [] 
            reported_twist = manual_twist
            reported_rise = manual_rise
            first_auto_calc = False

            all_layer_centroids = []
            for i in range(len(self.final_sandwiches[0])):
                layer_chains = [s[i] for s in self.final_sandwiches if len(s) > i]
                if layer_chains:
                    layer_com = np.mean([chain_centroids[c] for c in layer_chains if c in chain_centroids], axis=0)
                    all_layer_centroids.append(layer_com)
            
            all_layer_centroids = np.array(all_layer_centroids)
            
            if use_computed_axis:
                mean_centroid = np.mean(all_layer_centroids, axis=0)
                centered_points = all_layer_centroids - mean_centroid
                
                _, _, Vt = np.linalg.svd(centered_points)
                
                axis_vec = Vt[0]
                
                rough_vec = all_layer_centroids[-1] - all_layer_centroids[0]
                if np.dot(axis_vec, rough_vec) < 0:
                    axis_vec = -axis_vec
                    
                axis_len = np.linalg.norm(axis_vec)
                axis_u = axis_vec / axis_len if axis_len > 0 else np.array([0.0, 0.0, 1.0])
            else:
                axis_u = np.array([0.0, 0.0, 1.0])
                
            z_axis = np.array([0.0, 0.0, 1.0])
            dot_prod = np.clip(np.dot(axis_u, z_axis), -1.0, 1.0)
            tilt_deg = math.degrees(math.acos(dot_prod))
            if tilt_deg > 90.0:
                tilt_deg = 180.0 - tilt_deg
                
            # --- SNAP TO Z THRESHOLD (PDB limits to 3 decimal places) ---
            if tilt_deg < 0.1:
                axis_u = np.array([0.0, 0.0, 1.0])
                tilt_deg = 0.0
                
            z_axis = np.array([0.0, 0.0, 1.0])
            v = np.cross(axis_u, z_axis)
            c = np.dot(axis_u, z_axis)
            if c < -0.999999:
                R_align = -np.eye(3); R_align[2,2] = 1.0
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R_align = np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + c))
            R_align_inv = R_align.T

            for sandwich in self.final_sandwiches:
                if len(sandwich) == 0: continue
                
                bottom_chain = sandwich[0]
                top_chain = sandwich[-1]
                
                if use_auto:
                    twists = []
                    rises = []
                    step = 2 if use_alt else 1
                    for i in range(len(sandwich) - step):
                        coords_1 = (np.array(chain_ca_coords[sandwich[i]]) - global_centroid) @ R_align.T
                        coords_2 = (np.array(chain_ca_coords[sandwich[i+step]]) - global_centroid) @ R_align.T
                        R_step, t_step = self.get_transform(coords_1, coords_2)
                        twists.append(math.degrees(math.atan2(R_step[1, 0], R_step[0, 0])))
                        rises.append(t_step[2])
                    
                    if not twists and len(sandwich) == 2 and use_alt:
                        coords_1 = (np.array(chain_ca_coords[sandwich[0]]) - global_centroid) @ R_align.T
                        coords_2 = (np.array(chain_ca_coords[sandwich[1]]) - global_centroid) @ R_align.T
                        R_step, t_step = self.get_transform(coords_1, coords_2)
                        twists.append(math.degrees(math.atan2(R_step[1, 0], R_step[0, 0])) * 2.0)
                        rises.append(t_step[2] * 2.0)

                    if twists:
                        avg_twist = sum(twists) / len(twists)
                        avg_rise = sum(rises) / len(rises)
                    else:
                        avg_twist, avg_rise = 0.0, 0.0

                    calc_twist = avg_twist
                    calc_rise = avg_rise
                    
                    if not first_auto_calc:
                        reported_twist = avg_twist / 2.0 if use_alt else avg_twist
                        reported_rise = avg_rise / 2.0 if use_alt else avg_rise
                        first_auto_calc = True
                else:
                    calc_twist = manual_twist * 2.0 if use_alt else manual_twist
                    calc_rise = manual_rise * 2.0 if use_alt else manual_rise

                rad = math.radians(calc_twist)
                cos_t, sin_t = math.cos(rad), math.sin(rad)
                
                R_top_local = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
                t_top_local = np.array([0.0, 0.0, calc_rise])
                
                R_bottom_local = np.array([[cos_t, sin_t, 0], [-sin_t, cos_t, 0], [0, 0, 1]])
                t_bottom_local = np.array([0.0, 0.0, -calc_rise])

                R_top = R_align_inv @ R_top_local @ R_align
                t_top = global_centroid - R_top @ global_centroid + (R_align_inv @ t_top_local)

                R_bottom = R_align_inv @ R_bottom_local @ R_align
                t_bottom = global_centroid - R_bottom @ global_centroid + (R_align_inv @ t_bottom_local)

                if use_alt:
                    if len(sandwich) >= 2:
                        prev_b1 = sandwich[1]
                        prev_b2 = sandwich[0]
                    else:
                        prev_b1 = prev_b2 = sandwich[0]
                        
                    for _ in range(bottom_layers_to_add):
                        new_chain_id = get_new_chain_id()
                        expansions.append((prev_b1, new_chain_id, R_bottom, t_bottom))
                        prev_b1 = prev_b2
                        prev_b2 = new_chain_id

                    if len(sandwich) >= 2:
                        prev_t1 = sandwich[-2]
                        prev_t2 = sandwich[-1]
                    else:
                        prev_t1 = prev_t2 = sandwich[-1]
                        
                    for _ in range(top_layers_to_add):
                        new_chain_id = get_new_chain_id()
                        expansions.append((prev_t1, new_chain_id, R_top, t_top))
                        prev_t1 = prev_t2
                        prev_t2 = new_chain_id

                else:
                    current_ref_chain = bottom_chain
                    for _ in range(bottom_layers_to_add):
                        new_chain_id = get_new_chain_id()
                        expansions.append((current_ref_chain, new_chain_id, R_bottom, t_bottom))
                        current_ref_chain = new_chain_id
                        
                    current_ref_chain = top_chain
                    for _ in range(top_layers_to_add):
                        new_chain_id = get_new_chain_id()
                        expansions.append((current_ref_chain, new_chain_id, R_top, t_top))
                        current_ref_chain = new_chain_id

            new_chain_lines = {}
            all_new_hetatms = []

            for ref_chain, new_chain, R, t in expansions:
                lines = chain_atoms.get(ref_chain, [])
                new_lines = []
                for line in lines:
                    try:
                        orig_x = float(line[30:38])
                        orig_y = float(line[38:46])
                        orig_z = float(line[46:54])
                        orig_vec = np.array([orig_x, orig_y, orig_z])
                        new_vec = R @ orig_vec + t
                        
                        l = list(line)
                        if len(new_chain) == 1:
                            l[21] = new_chain; l[20] = ' '
                        elif len(new_chain) >= 2:
                            l[20] = new_chain[0]; l[21] = new_chain[1]
                        
                        l[30:38] = list(f"{new_vec[0]:8.3f}")
                        l[38:46] = list(f"{new_vec[1]:8.3f}")
                        l[46:54] = list(f"{new_vec[2]:8.3f}")
                        
                        new_lines.append("".join(l))
                    except Exception:
                        new_lines.append(line)
                chain_atoms[new_chain] = new_lines
                new_chain_lines[new_chain] = new_lines

                het_lines = associated_atoms.get(ref_chain, [])
                het_res_map = {} 
                new_het_lines_for_next = []
                
                for line in het_lines:
                    try:
                        orig_x = float(line[30:38])
                        orig_y = float(line[38:46])
                        orig_z = float(line[46:54])
                        orig_vec = np.array([orig_x, orig_y, orig_z])
                        new_vec = R @ orig_vec + t
                        
                        l = list(line)
                        
                        orig_chain = line[20:22].strip()
                        orig_res_seq = int(line[22:26])
                        
                        res_key = (orig_chain, orig_res_seq)
                        if res_key not in het_res_map:
                            max_res_seq[orig_chain] = max_res_seq.get(orig_chain, 0) + 1
                            het_res_map[res_key] = max_res_seq[orig_chain]
                            
                        new_res_seq = het_res_map[res_key]
                        
                        l[22:26] = list(f"{new_res_seq:>4}")
                        l[30:38] = list(f"{new_vec[0]:8.3f}")
                        l[38:46] = list(f"{new_vec[1]:8.3f}")
                        l[46:54] = list(f"{new_vec[2]:8.3f}")
                        
                        new_line_str = "".join(l)
                        all_new_hetatms.append(new_line_str)
                        new_het_lines_for_next.append(new_line_str)
                    except Exception:
                        all_new_hetatms.append(line)
                        new_het_lines_for_next.append(line)
                
                associated_atoms[new_chain] = new_het_lines_for_next

            all_new_hetatms.sort(key=lambda line: (line[20:22], int(line[22:26]) if line[22:26].strip().isdigit() else 0))

            serial_counter = 1
            serial_map = {}

            chain_exp_map = {}
            for ref_c, new_c, _, _ in expansions:
                if ref_c not in chain_exp_map: chain_exp_map[ref_c] = []
                chain_exp_map[ref_c].append(new_c)

            new_headers = []
            for line in headers:
                new_headers.append(line)
                record = line[0:6].strip()
                if record in self.CHAIN_RECORD_SPECS:
                    slice_list = self.CHAIN_RECORD_SPECS[record]
                    chains_in_line = []
                    for start, end in slice_list:
                        if end <= len(line):
                            c = line[start:end].strip()
                            if c: chains_in_line.append((start, end, c))
                    
                    if chains_in_line and all(c_info[2] in chain_exp_map for c_info in chains_in_line):
                        num_expansions = len(chain_exp_map[chains_in_line[0][2]])
                        if all(len(chain_exp_map[c_info[2]]) == num_expansions for c_info in chains_in_line):
                            for i in range(num_expansions):
                                line_chars = list(line)
                                for start, end, old_c in chains_in_line:
                                    new_c = chain_exp_map[old_c][i]
                                    width = end - start
                                    formatted = f"{new_c:>{width}}"[:width]
                                    line_chars[start:end] = list(formatted)
                                new_headers.append("".join(line_chars))

            with open(output_path, 'w') as fout:
                for line in new_headers:
                    fout.write(line)
                    
                for line in orig_atoms_ters:
                    try:
                        old_serial = int(line[6:11])
                        l = list(line)
                        l[6:11] = list(f"{serial_counter:>5}")
                        fout.write("".join(l))
                        serial_map[old_serial] = serial_counter
                        serial_counter += 1
                    except: fout.write(line)
                
                for ref_chain, new_chain, R, t in expansions:
                    last_line = ""
                    for line in new_chain_lines[new_chain]:
                        try:
                            l = list(line)
                            l[6:11] = list(f"{serial_counter:>5}")
                            fout.write("".join(l))
                            serial_counter += 1
                            last_line = line
                        except: fout.write(line)
                    
                    if last_line:
                        ter_line = list("TER   " + " " * 74)
                        ter_line[6:11] = list(f"{serial_counter:>5}")
                        ter_line[17:20] = list(last_line[17:20]) 
                        ter_line[20:22] = list(last_line[20:22]) 
                        ter_line[22:26] = list(last_line[22:26]) 
                        ter_line[26] = last_line[26]             
                        fout.write("".join(ter_line).rstrip() + "\n")
                        serial_counter += 1
                
                for line in orig_hetatms:
                    try:
                        old_serial = int(line[6:11])
                        l = list(line)
                        l[6:11] = list(f"{serial_counter:>5}")
                        fout.write("".join(l))
                        serial_map[old_serial] = serial_counter
                        serial_counter += 1
                    except: fout.write(line)

                for line in all_new_hetatms:
                    try:
                        l = list(line)
                        l[6:11] = list(f"{serial_counter:>5}")
                        fout.write("".join(l))
                        serial_counter += 1
                    except: fout.write(line)

                for line in orig_conects:
                    try:
                        parts = line.split()
                        if len(parts) < 2: continue
                        old_source = int(parts[1])
                        if old_source not in serial_map: continue
                        new_source = serial_map[old_source]
                        
                        valid_targets = []
                        for p in parts[2:]:
                            try:
                                tgt_old = int(p)
                                if tgt_old in serial_map:
                                    valid_targets.append(serial_map[tgt_old])
                            except: pass
                        
                        if valid_targets:
                            out_line = "CONECT" + f"{new_source:>5}"
                            for tgt in valid_targets:
                                out_line += f"{tgt:>5}"
                            fout.write(out_line + "\n")
                    except: pass
                    
                fout.write("END   \n")

            return reported_twist, reported_rise, axis_u, tilt_deg

        def action_restrain(self):
            if not self.final_sandwiches or self.detected_layers == 0: return
            
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Restrain File", "restrain_torsion.cxc", "ChimeraX Command (*.cxc);;All Files (*)")
            if file_path:
                try:
                    mapping = self.create_renaming_mapping(self.final_sandwiches)
                    mid_layer_idx = self.detected_layers // 2 
                    with open(file_path, 'w') as f:
                        for sandwich in self.final_sandwiches:
                            mid_chain_original = sandwich[mid_layer_idx]
                            mid_id_new = mapping[mid_chain_original]
                            for layer_k in range(self.detected_layers):
                                if layer_k == mid_layer_idx: continue 
                                target_chain_original = sandwich[layer_k]
                                target_id_new = mapping[target_chain_original]
                                f.write(f"isolde restrain torsions /{target_id_new} template /{mid_id_new} angleRange 180\n")
                    self.text_area.append(f"\n[Export] Restrain file generated: {os.path.basename(file_path)}")
                except Exception as e: QMessageBox.Icon.Critical(self, "Error", f"Failed to save restrain file:\n{str(e)}")

        def action_connect_chain(self):
            if not self.working_file_path:
                return

            import math
            import shlex

            source_path = self.working_file_path
            temporary_path = source_path + ".connect.tmp"
            target_chain = "A"
            is_cif = source_path.lower().endswith((".cif", ".mmcif"))

            try:
                # Keep every record belonging to a chain together and use the
                # first atom position as the same spatial anchor used by the RF
                # Diffusion merge path.
                chain_records = {}
                first_coordinates = {}

                if is_cif:
                    prefix_lines = []
                    suffix_lines = []
                    atom_headers = []
                    atom_rows = []
                    inside_atom_loop = False
                    atom_loop_seen = False

                    with open(
                        source_path, "r", encoding="utf-8", errors="replace"
                    ) as source:
                        for line in source:
                            stripped = line.strip()

                            if stripped == "loop_":
                                if inside_atom_loop:
                                    inside_atom_loop = False
                                    suffix_lines.append(line)
                                elif atom_loop_seen:
                                    suffix_lines.append(line)
                                else:
                                    prefix_lines.append(line)
                                continue

                            if stripped.startswith("_atom_site."):
                                inside_atom_loop = True
                                atom_loop_seen = True
                                atom_headers.append(stripped.split(".", 1)[1])
                                prefix_lines.append(line)
                                continue

                            if (
                                inside_atom_loop
                                and stripped
                                and not stripped.startswith(("_", "#"))
                            ):
                                try:
                                    fields = shlex.split(
                                        stripped, comments=False, posix=True
                                    )
                                except ValueError:
                                    fields = stripped.split()
                                if len(fields) >= len(atom_headers):
                                    atom_rows.append(fields)
                                continue

                            if inside_atom_loop:
                                inside_atom_loop = False
                            if atom_loop_seen:
                                suffix_lines.append(line)
                            else:
                                prefix_lines.append(line)

                    if not atom_headers or not atom_rows:
                        raise ValueError("No _atom_site records were found.")

                    chain_column = (
                        atom_headers.index("auth_asym_id")
                        if "auth_asym_id" in atom_headers
                        else atom_headers.index("label_asym_id")
                    )
                    x_column = atom_headers.index("Cartn_x")
                    y_column = atom_headers.index("Cartn_y")
                    z_column = atom_headers.index("Cartn_z")
                    sequence_column = (
                        atom_headers.index("auth_seq_id")
                        if "auth_seq_id" in atom_headers
                        else atom_headers.index("label_seq_id")
                    )
                    insertion_column = (
                        atom_headers.index("pdbx_PDB_ins_code")
                        if "pdbx_PDB_ins_code" in atom_headers
                        else None
                    )

                    for fields in atom_rows:
                        chain_id = fields[chain_column]
                        if chain_id not in chain_records:
                            chain_records[chain_id] = []
                            try:
                                first_coordinates[chain_id] = (
                                    float(fields[x_column]),
                                    float(fields[y_column]),
                                    float(fields[z_column]),
                                )
                            except (TypeError, ValueError):
                                first_coordinates[chain_id] = (0.0, 0.0, -9999.0)
                        chain_records[chain_id].append(fields)
                else:
                    header_lines = []
                    footer_lines = []

                    with open(
                        source_path, "r", encoding="utf-8", errors="replace"
                    ) as source:
                        for line in source:
                            record_type = line[:6].strip().upper()
                            if record_type in {"ATOM", "HETATM", "ANISOU"}:
                                if len(line) < 27:
                                    continue
                                chain_id = line[20:22]
                                if chain_id not in chain_records:
                                    chain_records[chain_id] = []
                                    try:
                                        first_coordinates[chain_id] = (
                                            float(line[30:38]),
                                            float(line[38:46]),
                                            float(line[46:54]),
                                        )
                                    except (TypeError, ValueError):
                                        first_coordinates[chain_id] = (
                                            0.0, 0.0, -9999.0
                                        )
                                chain_records[chain_id].append(line)
                            elif record_type == "TER":
                                continue
                            elif (
                                line.startswith("MASTER")
                                or line.startswith("END")
                                or line.startswith("CONECT")
                            ):
                                footer_lines.append(line)
                            else:
                                header_lines.append(line)

                if not chain_records:
                    raise ValueError("No chain records were found.")

                # Match RF Diffusion's geometry-aware ordering. Start at the
                # highest remaining chain, then walk down the closest chain in
                # Z that remains within the same broad XY neighborhood.
                remaining_chains = set(chain_records)
                ordered_chains = []
                while remaining_chains:
                    current_chain = max(
                        remaining_chains,
                        key=lambda chain_id: (
                            first_coordinates[chain_id][2], chain_id
                        ),
                    )
                    ordered_chains.append(current_chain)
                    remaining_chains.remove(current_chain)

                    while True:
                        current_x, current_y, current_z = first_coordinates[
                            current_chain
                        ]
                        candidates = []
                        for candidate in remaining_chains:
                            next_x, next_y, next_z = first_coordinates[candidate]
                            if (
                                current_z - next_z > -2.0
                                and math.hypot(
                                    current_x - next_x, current_y - next_y
                                ) < 25.0
                            ):
                                candidates.append(candidate)

                        if not candidates:
                            break

                        current_chain = min(
                            candidates,
                            key=lambda chain_id: (
                                abs(
                                    current_z
                                    - first_coordinates[chain_id][2]
                                ),
                                -first_coordinates[chain_id][2],
                                chain_id,
                            ),
                        )
                        ordered_chains.append(current_chain)
                        remaining_chains.remove(current_chain)

                # Renumber every source chain into chain A. Two is added at each
                # boundary, so one residue number is intentionally absent; RF
                # Diffusion relies on that gap to preserve the chain break.
                residue_number = 0
                with open(temporary_path, "w", encoding="utf-8") as output:
                    if is_cif:
                        output.writelines(prefix_lines)
                        for chain_id in ordered_chains:
                            residue_number += 2
                            previous_residue = None

                            for fields in chain_records[chain_id]:
                                insertion_code = (
                                    fields[insertion_column]
                                    if insertion_column is not None
                                    else ""
                                )
                                source_residue = (
                                    chain_id,
                                    fields[sequence_column],
                                    insertion_code,
                                )
                                if source_residue != previous_residue:
                                    if previous_residue is not None:
                                        residue_number += 1
                                    previous_residue = source_residue

                                if "auth_asym_id" in atom_headers:
                                    fields[
                                        atom_headers.index("auth_asym_id")
                                    ] = target_chain
                                if "label_asym_id" in atom_headers:
                                    fields[
                                        atom_headers.index("label_asym_id")
                                    ] = target_chain
                                if "auth_seq_id" in atom_headers:
                                    fields[
                                        atom_headers.index("auth_seq_id")
                                    ] = str(residue_number)
                                if "label_seq_id" in atom_headers:
                                    fields[
                                        atom_headers.index("label_seq_id")
                                    ] = str(residue_number)
                                if "label_entity_id" in atom_headers:
                                    fields[
                                        atom_headers.index("label_entity_id")
                                    ] = "1"

                                serialized = [
                                    f"'{field}'"
                                    if " " in field
                                    and not field.startswith(("'", '"'))
                                    else field
                                    for field in fields
                                ]
                                output.write(" ".join(serialized) + "\n")
                        output.writelines(suffix_lines)
                    else:
                        output.writelines(header_lines)
                        for chain_id in ordered_chains:
                            residue_number += 2
                            previous_residue = None

                            for line in chain_records[chain_id]:
                                source_residue = (
                                    chain_id,
                                    line[22:26],
                                    line[26:27],
                                )
                                if source_residue != previous_residue:
                                    if previous_residue is not None:
                                        residue_number += 1
                                    previous_residue = source_residue

                                characters = list(line)
                                if len(characters) < 27:
                                    characters.extend(
                                        " " for _ in range(27 - len(characters))
                                    )
                                characters[20] = " "
                                characters[21] = target_chain
                                characters[22:26] = list(
                                    f"{residue_number:>4}"[-4:]
                                )
                                output.write("".join(characters))
                        output.writelines(footer_lines)

                os.replace(temporary_path, source_path)
                self.text_area.append(
                    "\n>>>> Chains Connected into Chain A "
                    "(RF Diffusion smart top-to-bottom ordering)"
                )
                self.process_pdb(source_path)

            except Exception as error:
                if os.path.exists(temporary_path):
                    try:
                        os.remove(temporary_path)
                    except OSError:
                        pass
                QMessageBox.critical(self, "Error", f"Action failed: {error}")

        def action_save_final(self):
            if not self.working_file_path: return
            name_part, ext = os.path.splitext(self.original_filename_display)
            proposed_name = f"{name_part}_modified{ext}"
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Final PDB", proposed_name, "PDB Files (*.pdb);;All Files (*)")
            if file_path:
                try:
                    shutil.copy2(self.working_file_path, file_path)
                    QMessageBox.information(self, "Success", f"File saved successfully:\n{file_path}")
                    self.text_area.append(f"\n>>> Saved final file to {file_path}")
                except Exception as e: QMessageBox.Icon.Critical(self, "Error", f"Save failed: {e}")

        # --- NEW HELPER: Detect Spatial & ID Duplicates ---
        def action_evaluate_beta(self):
            if not self.working_file_path: return
            
            import subprocess
            
            # Setup paths (relative execution)
            folder = os.path.dirname(self.working_file_path)
            current_filename = os.path.basename(self.working_file_path)
            # Define naming conventions based on user request
            script_name = f"chimera_script_{current_filename}.py"
            output_filename = f"{current_filename}" 
            
            script_full_path = os.path.join(folder, script_name)
            output_full_path = os.path.join(folder, output_filename)
            
            # 2. Generate Script
            script_content = f"""
import chimera
from chimera import runCommand as rc
rc("open #0 {current_filename}")
rc("ksdssp #0")
rc("write relative #0 #0 {output_filename}")
rc("close session")
"""
            try:
                with open(script_full_path, 'w') as f:
                    f.write(script_content)
                
                # 3. Subprocess
                import platform
                if platform.system() == "Windows":
                    cmd = f'chimera --nogui --script "{script_name}"'
                else:
                    # Use shell=True to handle module loading on HPC clusters
                    cmd = f'module load {CHIMERA_VER} && chimera --nogui --script "{script_name}"'
                
                self.text_area.append(f"\n>>> Re-evaluating secondary structure...")
                # Run inside the folder so relative paths work
                subprocess.run(cmd, shell=True, cwd=folder, check=True)
                
                # 4. Reload
                if os.path.exists(output_full_path):
                    self.created_temp_files.append(output_full_path) # Track new file for cleanup
                    self.working_file_path = output_full_path
                    self.process_pdb(self.working_file_path)
                    self.text_area.append(f">>> β-sheet evaluation completed \n>>> Reloaded PDB file")
                else:
                    self.text_area.append("Error: Output file not found after Chimera run.")
                    
            except Exception as e:
                self.text_area.append(f"\n>>> Chimera not found or failed. Skipping β-sheet evaluation. (Error: {e})")
            finally:
                # Always clean up the temporary script
                if os.path.exists(script_full_path):
                    try: os.remove(script_full_path)
                    except: pass

        def check_spatial_and_id_duplicates(self, file_path):
            if file_path.lower().endswith(('.cif', '.mmcif')):
                return
            split_line_indices = set()
            segments_per_chain_id = {} # e.g., {'A': 1, 'B': 1}
            
            last_chain_id = None
            last_ca_coord = None
            
            # 1. SCAN PASS: Detect Splits
            with open(file_path, 'r') as f:
                for i, line in enumerate(f):
                    # FIX: Ignore HETATM lines for duplicate detection. 
                    # Standard PDBs append HETATMs at the end, causing false positive chain splits.
                    if line.startswith("ATOM"):
                        current_chain_id = line[20:22].strip() # Fix: Support 2-char Chain IDs
                        
                        # Check for "CA" atom to do spatial check
                        is_ca = line[12:16].strip() == "CA"
                        current_coord = None
                        if is_ca:
                            try:
                                current_coord = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                            except: pass

                        # --- LOGIC: Decide if this line starts a new segment ---
                        is_new_segment = False
                        
                        # Case A: File Order Change (e.g., went from Chain A to Chain B)
                        if current_chain_id != last_chain_id:
                            is_new_segment = True
                            last_ca_coord = None # Reset spatial tracking on ID change
                        
                        # Case B: Spatial Jump (Disabled to prevent breaking chains with missing loops)
                        # elif is_ca and last_ca_coord is not None:
                        #     dist = math.sqrt(sum((a-b)**2 for a,b in zip(current_coord, last_ca_coord)))
                        #     # 15.0 Angstroms is a safe "broken chain" threshold (CA-CA is usually 3.8A)
                        #     if dist > 15.0:
                        #         is_new_segment = True
                        pass
                        
                        # --- Record Split ---
                        if is_new_segment:
                            split_line_indices.add(i)
                            
                            # Track usage counts to see if we have duplicates
                            if current_chain_id not in segments_per_chain_id:
                                segments_per_chain_id[current_chain_id] = 0
                            segments_per_chain_id[current_chain_id] += 1
                            
                            last_chain_id = current_chain_id
                        
                        # Update tracker
                        if is_ca: last_ca_coord = current_coord

            # 2. DECISION: Do we have any ID that appeared as multiple segments?
            has_duplicates = any(count > 1 for count in segments_per_chain_id.values())

            if has_duplicates:
                reply = QMessageBox.question(
                    self, 
                    "Broken Chains / Duplicates Detected", 
                    "Detected multiple segments using the same Chain ID.\n"
                    "(Based on file order OR spatial gaps > 15Å)\n\n"
                    "Rename all segments to unique IDs to proceed?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.rewrite_split_chains(file_path, split_line_indices)
                    self.text_area.append(">>> Auto-corrected duplicate/broken chains.")

        def rewrite_split_chains(self, file_path, split_indices):
            temp_out = file_path + ".tmp"
            
            # Counter starts at -1 so the first split (index 0 or later) bumps it to 0 (Chain A)
            current_segment_idx = -1 
            current_label = "A"
            
            with open(file_path, 'r') as fin, open(temp_out, 'w') as fout:
                for i, line in enumerate(fin):
                    if i in split_indices:
                        current_segment_idx += 1
                        current_label = self.generate_label(current_segment_idx)
                    
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        # Replace Chain ID at column 21
                        line_list = list(line)
                        # Handle length safety
                        if len(current_label) == 1:
                            line_list[21] = current_label
                        elif len(current_label) > 1:
                            # If label is "10" or "AA", try to fit it
                            line_list[21] = current_label[0]
                            # Try to use column 20 if empty, otherwise we just write it and hope parser handles it
                            # (Simple logic: just write char at 21)
                        fout.write("".join(line_list))
                    else:
                        fout.write(line)
            
            shutil.move(temp_out, file_path)

        # ================= LOGIC: ANALYSIS (Based on Layer_identifyer.py) =================
        def process_pdb(self, filename, old_sandwiches=None, rename_map=None):
            Z_MIN = self.get_param(self.edit_z_min, 4.6)
            Z_MAX = self.get_param(self.edit_z_max, 5.0)
            XY_SHIFT_LIMIT = self.get_param(self.edit_xy_limit, 3.0)
            NEIGHBOR_SEARCH_COUNT = int(self.get_param(self.edit_neighbor_count, 6))
            
            try:
                if filename.lower().endswith(('.cif', '.mmcif')):
                    raw_chains, waters, ions = self.parse_cif_manual_logic(filename)
                else:
                    raw_chains, waters, ions = self.parse_pdb_manual_logic(filename)
                if not raw_chains:
                    self.text_area.append("Error: No Alpha Carbons found.")
                    return
                
                chain_ids = list(raw_chains.keys())
                total_chains = len(chain_ids)

                self.chains_data_plot = {}
                for cid, residues in raw_chains.items():
                    self.chains_data_plot[cid] = [tuple(r['coord']) for r in residues]
                self.current_waters = waters
                self.current_ions = ions

                is_anti = getattr(self, 'chk_anti', None) and self.chk_anti.isChecked()
                
                if is_anti:
                    anti_pairing_max_xy = XY_SHIFT_LIMIT * 2.0
                    anti_pairing_max_z = Z_MAX * 0.7
                else:
                    anti_pairing_max_xy = 0
                    anti_pairing_max_z = 0

                base_centroids = {cid: self.get_centroid(raw_chains[cid]) for cid in chain_ids}
                unit_ids = []
                unit_chains_map = {} 
                
                if is_anti:
                    self.text_area.append("Anti mode: Grouping chains into structural pairs...")
                    paired = set()
                    for cid in chain_ids:
                        if cid in paired: continue
                        best_match = None
                        min_dist = float('inf')
                        c1 = base_centroids[cid]
                        for other_cid in chain_ids:
                            if other_cid == cid or other_cid in paired: continue
                            c2 = base_centroids[other_cid]
                            d_xy = math.hypot(c1[0] - c2[0], c1[1] - c2[1])
                            d_z = abs(c1[2] - c2[2])
                            if d_xy > anti_pairing_max_xy or d_z > anti_pairing_max_z: continue
                            d = np.linalg.norm(c1 - c2)
                            if d < min_dist:
                                min_dist = d; best_match = other_cid
                        if best_match:
                            paired.add(cid); paired.add(best_match)
                            u_id = f"{cid}|{best_match}"
                            unit_ids.append(u_id); unit_chains_map[u_id] = [cid, best_match]
                        else:
                            unit_ids.append(cid); unit_chains_map[cid] = [cid]
                else:
                    unit_ids = chain_ids[:]
                    for cid in chain_ids: unit_chains_map[cid] = [cid]

                self.results = {}
                for u_id in unit_ids:
                    chains_in_unit = unit_chains_map[u_id]
                    avg_c = np.mean([base_centroids[c] for c in chains_in_unit], axis=0)
                    self.results[u_id] = {
                        'centroid': avg_c, 'protofilament': None, 'layer': None,
                        'neighbors': {'up': None, 'down': None}, 'original_chains': chains_in_unit
                    }

                self.text_area.append("Identifying Protofilaments...")
                adjacency = {u_id: [] for u_id in unit_ids}
                
                for id_a in unit_ids:
                    center_a = self.results[id_a]['centroid']
                    distances = []
                    for id_b in unit_ids:
                        if id_a == id_b: continue
                        dist = np.linalg.norm(center_a - self.results[id_b]['centroid'])
                        distances.append((dist, id_b))
                    
                    distances.sort(key=lambda x: x[0])
                    closest = [x[1] for x in distances[:NEIGHBOR_SEARCH_COUNT]]
                    
                    for id_b in closest:
                        best_relation = 0
                        for c_a in self.results[id_a]['original_chains']:
                            for c_b in self.results[id_b]['original_chains']:
                                rel = self.check_stacking(raw_chains[c_a], raw_chains[c_b], Z_MIN, Z_MAX, XY_SHIFT_LIMIT)
                                if rel != 0: best_relation = rel; break
                            if best_relation != 0: break
                        
                        relation = best_relation
                        if relation == 1: 
                            self.results[id_a]['neighbors']['up'] = id_b
                            self.results[id_b]['neighbors']['down'] = id_a
                            adjacency[id_a].append(id_b); adjacency[id_b].append(id_a)
                        elif relation == -1: 
                            self.results[id_a]['neighbors']['down'] = id_b
                            self.results[id_b]['neighbors']['up'] = id_a
                            adjacency[id_a].append(id_b); adjacency[id_b].append(id_a)

                protofilaments = []
                visited = set()
                for u_id in unit_ids:
                    if u_id not in visited:
                        component = []
                        stack = [u_id]
                        visited.add(u_id)
                        while stack:
                            curr = stack.pop()
                            component.append(curr)
                            for neighbor in adjacency[curr]:
                                if neighbor not in visited:
                                    visited.add(neighbor)
                                    stack.append(neighbor)
                        protofilaments.append(component)

                protofilaments.sort(key=len, reverse=True)
                for idx, group in enumerate(protofilaments):
                    pf_id = idx + 1
                    for u_id in group: self.results[u_id]['protofilament'] = pf_id

                self.text_area.append("Aligning Layers...")
                layer_map = {} 
                if protofilaments:
                    ref_pf = protofilaments[0]
                    anchor_unit = ref_pf[len(ref_pf)//2]
                    layer_map[anchor_unit] = 0
                    anchor_centroid = self.results[anchor_unit]['centroid']
                    initial_layer_units = [anchor_unit]

                    for pf in protofilaments[1:]:
                        best_match = None
                        min_dist = float('inf')
                        for u_id in pf:
                            dist = np.linalg.norm(self.results[u_id]['centroid'] - anchor_centroid)
                            if dist < min_dist: min_dist = dist; best_match = u_id
                        if best_match:
                            layer_map[best_match] = 0
                            initial_layer_units.append(best_match)

                    queue = initial_layer_units[:] 
                    processed = set(initial_layer_units)
                    
                    while queue:
                        curr = queue.pop(0)
                        curr_layer = layer_map[curr]
                        up = self.results[curr]['neighbors']['up']
                        if up and up not in processed:
                            layer_map[up] = curr_layer + 1; processed.add(up); queue.append(up)
                        down = self.results[curr]['neighbors']['down']
                        if down and down not in processed:
                            layer_map[down] = curr_layer - 1; processed.add(down); queue.append(down)
                    
                    if layer_map:
                        min_L = min(layer_map.values())
                        offset = 1 - min_L
                        for k in layer_map: layer_map[k] += offset

                unpacked_protofilaments = []
                unpacked_layer_map = {}
                
                for pf in protofilaments:
                    if is_anti:
                        pf_chain1, pf_chain2 = [], []
                        for u_id in pf:
                            chains = self.results[u_id]['original_chains']
                            c1 = chains[0]; c2 = chains[1] if len(chains) > 1 else None
                            pf_chain1.append(c1)
                            if c2: pf_chain2.append(c2)
                            if u_id in layer_map:
                                unpacked_layer_map[c1] = layer_map[u_id]
                                if c2: unpacked_layer_map[c2] = layer_map[u_id]
                        if pf_chain1: unpacked_protofilaments.append(pf_chain1)
                        if pf_chain2: unpacked_protofilaments.append(pf_chain2)
                    else:
                        pf_chains = []
                        for u_id in pf:
                            c1 = self.results[u_id]['original_chains'][0]
                            pf_chains.append(c1)
                            if u_id in layer_map: unpacked_layer_map[c1] = layer_map[u_id]
                        if pf_chains: unpacked_protofilaments.append(pf_chains)

                protofilaments = unpacked_protofilaments
                layer_map = unpacked_layer_map

                final_results = {}
                for u_id in unit_ids:
                    for c_id in self.results[u_id]['original_chains']:
                        final_results[c_id] = {'centroid': base_centroids[c_id]}
                self.results = final_results

                self.final_sandwiches = []
                if protofilaments:
                    pf_centroids = []
                    for pf in protofilaments:
                        group_centroids = [self.results[c]['centroid'] for c in pf]
                        pf_centroids.append(np.mean(group_centroids, axis=0))
                    
                    global_center = np.mean(pf_centroids, axis=0)
                    indexed_centroids = list(enumerate(pf_centroids))
                    start_idx, start_centroid = min(indexed_centroids, key=lambda p: p[1][0])
                    sorted_indices = [start_idx]; visited_indices = {start_idx}
                    
                    def get_xy_dist(c1, c2): return math.sqrt((c1[0]-c2[0])**2 + (c1[1]-c2[1])**2)
                    def find_symmetry(ref_idx, tolerance=15.0):
                        ref_c = pf_centroids[ref_idx]
                        vec_x = ref_c[0] - global_center[0]; vec_y = ref_c[1] - global_center[1]
                        target_x = global_center[0] - vec_x; target_y = global_center[1] - vec_y
                        target = (target_x, target_y)
                        best_sym_idx = -1; min_sym_dist = float('inf')
                        for idx, c in indexed_centroids:
                            if idx in visited_indices: continue
                            d = math.sqrt((c[0]-target[0])**2 + (c[1]-target[1])**2)
                            if d < min_sym_dist: min_sym_dist = d; best_sym_idx = idx
                        if best_sym_idx != -1 and min_sym_dist <= tolerance: return best_sym_idx
                        return -1

                    sym_a = find_symmetry(start_idx)
                    if sym_a != -1: sorted_indices.append(sym_a); visited_indices.add(sym_a)

                    remaining = []
                    for idx, c in indexed_centroids:
                        if idx not in visited_indices:
                            d = get_xy_dist(start_centroid, c)
                            remaining.append((d, idx))
                    
                    remaining.sort(key=lambda x: x[0]) 
                    
                    for dist, idx in remaining:
                        if idx in visited_indices: continue
                        sorted_indices.append(idx); visited_indices.add(idx)
                        sym_partner = find_symmetry(idx)
                        if sym_partner != -1: sorted_indices.append(sym_partner); visited_indices.add(sym_partner)

                    protofilaments = [protofilaments[i] for i in sorted_indices]

                for group in protofilaments:
                    group_with_layers = [c for c in group if c in layer_map]
                    if not group_with_layers: continue
                    sorted_group = sorted(group_with_layers, key=lambda c: layer_map[c])
                    self.final_sandwiches.append(sorted_group)

                if self.final_sandwiches: self.detected_layers = max(len(s) for s in self.final_sandwiches)
                else: self.detected_layers = 0
                
                self.btn_beta.setEnabled(self.detected_layers >= 1)

                self.cmb_protofilaments.blockSignals(True)
                self.cmb_protofilaments.clear()
                if self.final_sandwiches:
                    for i, pf in enumerate(self.final_sandwiches):
                        mid_chain = pf[len(pf)//2] if pf else "?"
                        self.cmb_protofilaments.addItem(f"PF with Chain {mid_chain}", userData=i)
                self.cmb_protofilaments.blockSignals(False)

                output_lines = []
                output_lines.append("--------------- Analysis Result ---------------")
                output_lines.append(f"Total Chains: {total_chains}")
                output_lines.append(f"Max Layers: {self.detected_layers}")
                
                is_anti = getattr(self, 'chk_anti', None) and self.chk_anti.isChecked()
                
                if rename_map is not None and old_sandwiches is not None: display_sandwiches = old_sandwiches; is_renaming = True
                else: display_sandwiches = self.final_sandwiches; is_renaming = False

                if is_anti:
                    merged_pfs = []
                    i = 0
                    while i < len(display_sandwiches):
                        s1 = display_sandwiches[i]
                        if i + 1 < len(display_sandwiches) and len(s1) == len(display_sandwiches[i+1]):
                            s2 = display_sandwiches[i+1]
                            merged_old = [f"({c1},{c2})" for c1, c2 in zip(s1, s2)]
                            if is_renaming: merged_new = [f"({rename_map.get(c1, '?')},{rename_map.get(c2, '?')})" for c1, c2 in zip(s1, s2)]
                            else: merged_new = []
                            merged_pfs.append((s1[len(s1)//2], merged_old, merged_new))
                            i += 2 
                        else:
                            merged_pfs.append((s1[len(s1)//2], list(s1), []))
                            i += 1
                    
                    output_lines.append(f"Protofilaments: {len(merged_pfs)}")
                    for i, (mid_chain, m_old, m_new) in enumerate(merged_pfs):
                        if is_renaming:
                            new_mid = rename_map.get(mid_chain, "?")
                            output_lines.append(f"\nAnti-Protofilament #{i+1} (Contains Chain {mid_chain} -> {new_mid})")
                            output_lines.append(f"Old Units: {', '.join(m_old)}")
                            output_lines.append(f"New Units: {', '.join(m_new)}")
                        else:
                            output_lines.append(f"\nAnti-Protofilament #{i+1} (Contains Chain {mid_chain})")
                            output_lines.append(", ".join(m_old))
                else:
                    output_lines.append(f"Protofilaments: {len(display_sandwiches)}")
                    for i, sandwich in enumerate(display_sandwiches):
                        mid_chain = sandwich[len(sandwich)//2]
                        if is_renaming:
                            new_mid = rename_map.get(mid_chain, "?")
                            output_lines.append(f"\nProtofilament #{i+1} (Contains Chain {mid_chain} -> {new_mid})")
                            output_lines.append(f"Old: {', '.join(sandwich)}")
                            new_seq = [rename_map.get(c, "?") for c in sandwich]
                            output_lines.append(f"New: {', '.join(new_seq)}")
                        else:
                            output_lines.append(f"\nProtofilament #{i+1} (Contains Chain {mid_chain})")
                            output_lines.append(", ".join(sandwich))

                self.text_area.append("\n".join(output_lines))
                
                highlight_ids = set()
                if self.final_sandwiches:
                    mid_idx = self.detected_layers // 2
                    for s in self.final_sandwiches:
                        if mid_idx < len(s): 
                            highlight_ids.add(s[mid_idx])
                
                self.mol_canvas.plot_chains(
                    self.chains_data_plot, 
                    label_ids=highlight_ids, 
                    waters=self.current_waters, 
                    ions=self.current_ions
                )

            except Exception as e:
                self.text_area.append(f"Analysis Error: {str(e)}")

        def parse_cif_manual_logic(self, filepath):
            import numpy as np
            chains = {}; waters = []; ions = []
            headers = []; in_loop = False; start_idx = -1
            
            with open(filepath, 'r') as f:
                lines = f.readlines()
                
            for i, line in enumerate(lines):
                s = line.strip()
                if s == "loop_":
                    in_loop = True; headers = []
                elif s.startswith("_atom_site."):
                    headers.append(s.split('.')[1])
                elif in_loop and s.startswith("_"): pass
                elif in_loop and headers and "group_PDB" in headers:
                    start_idx = i; break
                else: in_loop = False
                    
            if start_idx == -1: return chains, waters, ions
            
            try:
                i_group = headers.index("group_PDB")
                i_atom = headers.index("label_atom_id")
                i_res = headers.index("label_comp_id")
                i_chain = headers.index("auth_asym_id") if "auth_asym_id" in headers else headers.index("label_asym_id")
                i_seq = headers.index("auth_seq_id") if "auth_seq_id" in headers else headers.index("label_seq_id")
                i_x = headers.index("Cartn_x")
                i_y = headers.index("Cartn_y")
                i_z = headers.index("Cartn_z")
            except ValueError: return chains, waters, ions
                
            for line in lines[start_idx:]:
                if line.strip() == "#" or line.startswith("loop_"): break
                parts = line.split()
                if len(parts) < len(headers): continue
                
                group, atom_name, res_name = parts[i_group], parts[i_atom], parts[i_res]
                chain_id, res_id = parts[i_chain], parts[i_seq]
                
                try: x, y, z = float(parts[i_x]), float(parts[i_y]), float(parts[i_z])
                except ValueError: continue
                    
                STANDARD_MODIFIED_RESIDUES = {"MSE", "SEP", "TPO", "PTR", "PCA", "CME", "CSO", "KCX", "ALY", "MLY", "SME", "CSX", "FME", "TYS", "LLP", "SAC"}
                if (group == "ATOM" or res_name in STANDARD_MODIFIED_RESIDUES) and ("CA" == atom_name or "CA" in atom_name):
                    if chain_id not in chains: chains[chain_id] = []
                    chains[chain_id].append({'id': res_id, 'coord': np.array([x, y, z])})
                elif group == "HETATM" and res_name not in STANDARD_MODIFIED_RESIDUES:
                    if res_name in ["HOH", "WAT"]:
                        if atom_name.startswith("O"): waters.append((x, y, z))
                    else: ions.append((x, y, z))
                    
            return chains, waters, ions

        def parse_pdb_manual_logic(self, filepath):
            chains = {}; waters = []; ions = []
            with open(filepath, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        atom_name = line[12:16].strip()
                        res_name = line[17:20].strip()
                        try: x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        except ValueError: continue

                        STANDARD_MODIFIED_RESIDUES = {"MSE", "SEP", "TPO", "PTR", "PCA", "CME", "CSO", "KCX", "ALY", "MLY", "SME", "CSX", "FME", "TYS", "LLP", "SAC"}
                        if "CA" == atom_name or "CA " in line[12:16]:
                            chain_id = line[20:22].strip()
                            res_id = line[22:27].strip()
                            if chain_id not in chains: chains[chain_id] = []
                            chains[chain_id].append({'id': res_id, 'coord': np.array([x, y, z])})
                        elif line.startswith("HETATM") and res_name not in STANDARD_MODIFIED_RESIDUES:
                            if res_name in ["HOH", "WAT"]:
                                if atom_name.startswith("O"): waters.append((x, y, z))
                            else: ions.append((x, y, z))
            return chains, waters, ions

        def get_centroid(self, chain_data):
            if not chain_data: return np.array([0.0, 0.0, 0.0])
            coords = [res['coord'] for res in chain_data]
            return np.mean(coords, axis=0)

        def check_stacking(self, chain_a, chain_b, z_min, z_max, xy_limit):
            res_map_a = {r['id']: r['coord'] for r in chain_a}
            res_map_b = {r['id']: r['coord'] for r in chain_b}
            common_ids = set(res_map_a.keys()).intersection(set(res_map_b.keys()))
            if len(common_ids) < 3: return 0
            
            z_diffs = []
            for rid in common_ids:
                pos_a, pos_b = res_map_a[rid], res_map_b[rid]
                dx, dy, dz = pos_b[0] - pos_a[0], pos_b[1] - pos_a[1], pos_b[2] - pos_a[2]
                if math.sqrt(dx*dx + dy*dy) > xy_limit: return 0
                z_diffs.append(dz)

            avg_dz = np.mean(z_diffs)
            abs_dz = abs(avg_dz)
            
            if z_min <= abs_dz <= z_max:
                if np.std(z_diffs) > 1.5: return 0 
                return 1 if avg_dz > 0 else -1
            return 0

        def get_expansion_order(self, n_layers):
            mid = n_layers // 2; order = [mid]; offset = 1
            while True:
                added = False
                up = mid + offset
                if up < n_layers: order.append(up); added = True
                down = mid - offset
                if down >= 0: order.append(down); added = True
                if not added: break
                offset += 1
            return order

        def generate_label(self, index):
            if self.chk_only_numbers.isChecked(): return str(index)
            use_numbers = self.chk_numbers_first.isChecked()
            if use_numbers:
                if index < 26: return chr(ord('A') + index)
                elif index < 36: return str(index - 26)
                else:
                    temp_index = index - 10; label = ""; temp_index += 1
                    while temp_index > 0:
                        temp_index -= 1; label = chr(ord('A') + (temp_index % 26)) + label; temp_index //= 26
                    return label
            else:
                label = ""; index += 1
                while index > 0: 
                    index -= 1; label = chr(ord('A') + (index % 26)) + label; index //= 26
                return label

        def create_renaming_mapping(self, sandwiches):
            n_layers = self.detected_layers
            layer_order = self.get_expansion_order(n_layers)
            
            skip_chars = set()
            if self.working_file_path:
                het_types = set()
                try:
                    is_cif = self.working_file_path.lower().endswith(('.cif', '.mmcif'))
                    with open(self.working_file_path, 'r') as f:
                        if is_cif:
                            in_atom_site = False
                            headers = []
                            for line in f:
                                s = line.strip()
                                if s == "loop_":
                                    in_atom_site = False
                                    headers = []
                                elif s.startswith("_atom_site."):
                                    in_atom_site = True
                                    headers.append(s.split('.')[1])
                                elif in_atom_site and s and not s.startswith("_") and not s.startswith("#"):
                                    parts = s.split()
                                    if len(parts) >= len(headers):
                                        try:
                                            col_group = headers.index("group_PDB")
                                            col_res = headers.index("label_comp_id") if "label_comp_id" in headers else headers.index("auth_comp_id")
                                            if parts[col_group] == "HETATM":
                                                het_types.add(parts[col_res])
                                        except ValueError:
                                            pass
                        else:
                            for line in f:
                                if line.startswith("HETATM") and len(line) >= 20:
                                    het_types.add(line[17:20].strip())
                                    
                    alphabet_backwards = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
                    for i, htype in enumerate(sorted(list(het_types))):
                        if i < len(alphabet_backwards):
                            skip_chars.add(alphabet_backwards[i])
                except: pass

            mapping = {}; global_index = 0
            for layer_idx in layer_order:
                for sandwich in sandwiches:
                    if layer_idx < len(sandwich):
                        chain_id = sandwich[layer_idx]
                        while True:
                            new_label = self.generate_label(global_index)
                            global_index += 1
                            if new_label not in skip_chars:
                                break
                        mapping[chain_id] = new_label
            return mapping
        
        def write_renamed_cif(self, input_filename, output_filename, sandwiches):
            import shlex
            prot_mapping = self.create_renaming_mapping(sandwiches)
            
            het_types = set()
            try:
                with open(input_filename, 'r') as f:
                    in_atom_site = False
                    headers = []
                    for line in f:
                        s = line.strip()
                        if s == "loop_": in_atom_site = False; headers = []
                        elif s.startswith("_atom_site."): in_atom_site = True; headers.append(s.split('.')[1])
                        elif in_atom_site and s and not s.startswith("_") and not s.startswith("#"):
                            parts = s.split()
                            if len(parts) >= len(headers):
                                col_g = headers.index("group_PDB") if "group_PDB" in headers else -1
                                col_r = headers.index("label_comp_id") if "label_comp_id" in headers else (headers.index("auth_comp_id") if "auth_comp_id" in headers else -1)
                                if col_g != -1 and col_r != -1 and parts[col_g] == "HETATM":
                                    het_types.add(parts[col_r])
            except: pass
            
            het_type_map = {}
            alphabet_backwards = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
            for i, htype in enumerate(sorted(list(het_types))):
                if i < len(alphabet_backwards):
                    het_type_map[htype] = alphabet_backwards[i]

            with open(input_filename, 'r') as fin, open(output_filename, 'w') as fout:
                in_loop = False
                loop_headers = []
                loop_lines = []
                chain_cols = []
                
                def process_and_flush_loop():
                    if not loop_headers: return
                    
                    if not chain_cols:
                        for h in loop_headers: fout.write(h + "\n")
                        for l in loop_lines: fout.write(l + "\n")
                        return
                    
                    parsed_rows = []
                    for line in loop_lines:
                        try: parts = shlex.split(line)
                        except ValueError: parts = line.split()
                        
                        if len(parts) >= len(loop_headers):
                            col_group = loop_headers.index("group_PDB") if "group_PDB" in loop_headers else -1
                            col_res = loop_headers.index("label_comp_id") if "label_comp_id" in loop_headers else (loop_headers.index("auth_comp_id") if "auth_comp_id" in loop_headers else -1)
                            is_hetatm = col_group != -1 and parts[col_group] == "HETATM"
                            res_name = parts[col_res] if col_res != -1 else ""

                            for c_idx in chain_cols:
                                if c_idx < len(parts):
                                    old_val = parts[c_idx]
                                    if old_val in prot_mapping:
                                        parts[c_idx] = prot_mapping[old_val]
                                    elif is_hetatm and res_name in het_type_map:
                                        parts[c_idx] = het_type_map[res_name]
                            
                            for i in range(len(parts)):
                                if ' ' in parts[i] and not (parts[i].startswith("'") or parts[i].startswith('"')):
                                    parts[i] = f"'{parts[i]}'"
                            parsed_rows.append(parts)
                        else:
                            parsed_rows.append([line.strip()])
                            
                    col_widths = [0] * len(loop_headers)
                    for row in parsed_rows:
                        if len(row) > 1:
                            for i, val in enumerate(row):
                                if i < len(col_widths):
                                    col_widths[i] = max(col_widths[i], len(val))
                                    
                    for h in loop_headers: fout.write(h + "\n")
                    for row in parsed_rows:
                        if len(row) == 1:
                            fout.write(row[0] + "\n")
                        else:
                            formatted = []
                            for i, val in enumerate(row):
                                if i < len(col_widths):
                                    formatted.append(val.ljust(col_widths[i]))
                                else:
                                    formatted.append(val)
                            fout.write(" ".join(formatted) + "\n")

                for line in fin:
                    s = line.strip()
                    if s == "loop_":
                        if in_loop: process_and_flush_loop()
                        in_loop = True
                        loop_headers = []
                        loop_lines = []
                        chain_cols = []
                        fout.write(line)
                        continue
                    
                    if in_loop and s.startswith("_"):
                        loop_headers.append(s)
                        if "asym_id" in s or s == "_struct_asym.id" or "pdb_strand_id" in s: 
                            chain_cols.append(len(loop_headers) - 1)
                        continue
                    
                    if in_loop and s and not s.startswith("#") and not s.startswith("_"):
                        loop_lines.append(line.rstrip('\r\n'))
                        continue
                        
                    if s.startswith("#"):
                        if in_loop:
                            process_and_flush_loop()
                            in_loop = False
                        fout.write(line)
                        continue
                        
                    if in_loop:
                        process_and_flush_loop()
                        in_loop = False
                    fout.write(line)
                    
                if in_loop:
                    process_and_flush_loop()

        def write_trimmed_cif(self, input_path, output_path, keep_chains, keep_het_lines):
            import shlex
            with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
                in_loop = False
                loop_headers = []
                chain_cols = []
                is_atom_site = False
                header_buffer = []
                wrote_headers = False
                
                for i, line in enumerate(fin):
                    s = line.strip()
                    if s == "loop_":
                        in_loop = True; loop_headers = []; chain_cols = []; is_atom_site = False
                        header_buffer = [line]; wrote_headers = False
                        continue
                        
                    if in_loop and s.startswith("_"):
                        loop_headers.append(s)
                        if "asym_id" in s: chain_cols.append(len(loop_headers) - 1)
                        if "_atom_site." in s: is_atom_site = True
                        header_buffer.append(line)
                        continue
                        
                    if in_loop and s and not s.startswith("#") and not s.startswith("_"):
                        keep_line = True
                        if is_atom_site:
                            parts = s.split()
                            if len(parts) >= len(loop_headers):
                                target_col = -1
                                for idx, h in enumerate(loop_headers):
                                    if "auth_asym_id" in h: target_col = idx; break
                                    elif "label_asym_id" in h: target_col = idx
                                if target_col != -1 and target_col < len(parts):
                                    if parts[target_col] not in keep_chains and i not in keep_het_lines:
                                        keep_line = False
                        elif chain_cols:
                            try: parts = shlex.split(s)
                            except: parts = s.split()
                            if len(parts) >= len(loop_headers):
                                for c_idx in chain_cols:
                                    if c_idx < len(parts):
                                        c_val = parts[c_idx]
                                        if c_val not in ["?", "."] and c_val not in keep_chains:
                                            keep_line = False; break
                        
                        if keep_line:
                            if not wrote_headers:
                                fout.writelines(header_buffer)
                                wrote_headers = True
                            fout.write(line)
                        continue
                            
                    if s.startswith("#"): 
                        in_loop = False
                        if wrote_headers or not header_buffer:
                            fout.write(line)
                        header_buffer = []
                        continue
                        
                    if not in_loop or wrote_headers:
                        fout.write(line)

        def write_expanded_cif(self, input_path, output_path, layers_to_add, use_auto, manual_twist, manual_rise, water_z_limit=4.0, use_alt=False, use_computed_axis=True):
            import numpy as np
            import math
            top_layers_to_add = layers_to_add // 2
            bottom_layers_to_add = layers_to_add - top_layers_to_add

            used_chain_ids = set()
            headers = []
            pre_lines, post_lines, atom_lines = [], [], []
            in_atom_site = False

            with open(input_path, 'r') as fin:
                for line in fin:
                    s = line.strip()
                    if s == "loop_":
                        if not in_atom_site and not headers: pre_lines.append(line)
                        else: post_lines.append(line)
                        continue
                    if s.startswith("_atom_site."):
                        in_atom_site = True
                        headers.append(s.split('.')[1])
                        pre_lines.append(line)
                        continue
                    if in_atom_site and s and not s.startswith("_") and not s.startswith("#"):
                        parts = line.split()
                        if len(parts) >= len(headers):
                            atom_lines.append(parts)
                            col_c = headers.index("auth_asym_id") if "auth_asym_id" in headers else headers.index("label_asym_id")
                            used_chain_ids.add(parts[col_c])
                        continue
                    elif s.startswith("#") and in_atom_site:
                        in_atom_site = False
                        post_lines.append(line)
                    else:
                        if not in_atom_site and not headers: pre_lines.append(line)
                        else: post_lines.append(line)

            col_x = headers.index("Cartn_x")
            col_y = headers.index("Cartn_y")
            col_z = headers.index("Cartn_z")
            col_c = headers.index("auth_asym_id") if "auth_asym_id" in headers else headers.index("label_asym_id")
            col_group = headers.index("group_PDB")
            col_res = headers.index("label_comp_id") if "label_comp_id" in headers else headers.index("auth_comp_id")

            het_types = set()
            for parts in atom_lines:
                if parts[col_group] == "HETATM":
                    het_types.add(parts[col_res])

            reserved_het_chains = {}
            alphabet_backwards = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
            for i, htype in enumerate(sorted(list(het_types))):
                if i < len(alphabet_backwards):
                    c = alphabet_backwards[i]
                    reserved_het_chains[htype] = c
                    used_chain_ids.add(c)
                else:
                    idx = 0
                    while True:
                        label = self.generate_label(idx)
                        if label not in used_chain_ids and label not in reserved_het_chains.values():
                            reserved_het_chains[htype] = label
                            used_chain_ids.add(label)
                            break
                        idx += 1

            def get_new_chain_id():
                idx = 0
                while True:
                    label = self.generate_label(idx)
                    if label not in used_chain_ids:
                        used_chain_ids.add(label)
                        return label
                    idx += 1

            core_chains = set()
            for s in self.final_sandwiches: core_chains.update(s)

            chain_atoms, chain_ca_coords, chain_centroids = {}, {}, {}
            associated_atoms = {cid: [] for cid in core_chains}
            
            for parts in atom_lines:
                cid, group = parts[col_c], parts[col_group]
                try: x, y, z = float(parts[col_x]), float(parts[col_y]), float(parts[col_z])
                except ValueError: continue
                
                if group == "ATOM" and cid in core_chains:
                    if cid not in chain_atoms:
                        chain_atoms[cid] = []
                        chain_ca_coords[cid] = []
                    chain_atoms[cid].append(parts)
                    if "CA" in parts[headers.index("label_atom_id")]: chain_ca_coords[cid].append([x, y, z])

            for cid, coords in chain_ca_coords.items():
                if coords: chain_centroids[cid] = np.mean(coords, axis=0)

            core_ca_coords = []
            for cid in core_chains:
                if cid in chain_ca_coords: core_ca_coords.extend(chain_ca_coords[cid])
            global_centroid = np.mean(core_ca_coords, axis=0) if core_ca_coords else np.zeros(3)

            for parts in atom_lines:
                group = parts[col_group]
                if group == "HETATM" or (group == "ATOM" and parts[col_c] not in core_chains):
                    try:
                        x, y, z = float(parts[col_x]), float(parts[col_y]), float(parts[col_z])
                        best_chain = None
                        min_dist = float('inf')
                        for cid, centroid in chain_centroids.items():
                            if abs(z - centroid[2]) <= water_z_limit:
                                dist = math.sqrt((x - centroid[0])**2 + (y - centroid[1])**2 + (z - centroid[2])**2)
                                if dist < min_dist:
                                    min_dist, best_chain = dist, cid
                        if best_chain: associated_atoms[best_chain].append(parts)
                    except ValueError: pass

            expansions = []
            reported_twist, reported_rise = manual_twist, manual_rise
            first_auto_calc = False

            all_layer_centroids = []
            for i in range(len(self.final_sandwiches[0])):
                layer_chains = [s[i] for s in self.final_sandwiches if len(s) > i]
                if layer_chains:
                    layer_com = np.mean([chain_centroids[c] for c in layer_chains if c in chain_centroids], axis=0)
                    all_layer_centroids.append(layer_com)
            
            all_layer_centroids = np.array(all_layer_centroids)
            
            if use_computed_axis:
                mean_centroid = np.mean(all_layer_centroids, axis=0)
                centered_points = all_layer_centroids - mean_centroid
                
                _, _, Vt = np.linalg.svd(centered_points)
                
                axis_vec = Vt[0]
                
                rough_vec = all_layer_centroids[-1] - all_layer_centroids[0]
                if np.dot(axis_vec, rough_vec) < 0:
                    axis_vec = -axis_vec
                    
                axis_len = np.linalg.norm(axis_vec)
                axis_u = axis_vec / axis_len if axis_len > 0 else np.array([0.0, 0.0, 1.0])
            else:
                axis_u = np.array([0.0, 0.0, 1.0])
                
            z_axis = np.array([0.0, 0.0, 1.0])
            dot_prod = np.clip(np.dot(axis_u, z_axis), -1.0, 1.0)
            tilt_deg = math.degrees(math.acos(dot_prod))
            if tilt_deg > 90.0:
                tilt_deg = 180.0 - tilt_deg
                
            # --- DYNAMIC SNAP TO Z THRESHOLD FOR CIF ---
            max_decimals = 3
            for parts in atom_lines[:100]:
                try:
                    x_str = parts[col_x]
                    if '.' in x_str:

                        dec_len = len(x_str.split('.')[1].rstrip('0'))
                        if dec_len > max_decimals:
                            max_decimals = dec_len
                except Exception:
                    pass
                    
            # Scales threshold: 3 dec = 0.1°, 4 dec = 0.01°, 5 dec = 0.001°, etc.
            snap_threshold = 0.1 / (10 ** max(0, max_decimals - 3))
            
            if tilt_deg < snap_threshold:
                axis_u = np.array([0.0, 0.0, 1.0])
                tilt_deg = 0.0
                
            z_axis = np.array([0.0, 0.0, 1.0])
            v = np.cross(axis_u, z_axis)
            c = np.dot(axis_u, z_axis)
            if c < -0.999999:
                R_align = -np.eye(3); R_align[2,2] = 1.0
            else:
                vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                R_align = np.eye(3) + vx + (vx @ vx) * (1.0 / (1.0 + c))
            R_align_inv = R_align.T

            for sandwich in self.final_sandwiches:
                if len(sandwich) == 0: continue
                
                bottom_chain = sandwich[0]
                top_chain = sandwich[-1]
                
                if use_auto:
                    twists = []
                    rises = []
                    step = 2 if use_alt else 1
                    for i in range(len(sandwich) - step):
                        coords_1 = (np.array(chain_ca_coords[sandwich[i]]) - global_centroid) @ R_align.T
                        coords_2 = (np.array(chain_ca_coords[sandwich[i+step]]) - global_centroid) @ R_align.T
                        R_step, t_step = self.get_transform(coords_1, coords_2)
                        twists.append(math.degrees(math.atan2(R_step[1, 0], R_step[0, 0])))
                        rises.append(t_step[2])
                    
                    if not twists and len(sandwich) == 2 and use_alt:
                        coords_1 = (np.array(chain_ca_coords[sandwich[0]]) - global_centroid) @ R_align.T
                        coords_2 = (np.array(chain_ca_coords[sandwich[1]]) - global_centroid) @ R_align.T
                        R_step, t_step = self.get_transform(coords_1, coords_2)
                        twists.append(math.degrees(math.atan2(R_step[1, 0], R_step[0, 0])) * 2.0)
                        rises.append(t_step[2] * 2.0)

                    if twists:
                        avg_twist = sum(twists) / len(twists)
                        avg_rise = sum(rises) / len(rises)
                    else:
                        avg_twist, avg_rise = 0.0, 0.0

                    calc_twist = avg_twist
                    calc_rise = avg_rise
                    
                    if not first_auto_calc:
                        reported_twist = avg_twist / 2.0 if use_alt else avg_twist
                        reported_rise = avg_rise / 2.0 if use_alt else avg_rise
                        first_auto_calc = True
                else:
                    calc_twist = manual_twist * 2.0 if use_alt else manual_twist
                    calc_rise = manual_rise * 2.0 if use_alt else manual_rise

                rad = math.radians(calc_twist)
                cos_t, sin_t = math.cos(rad), math.sin(rad)
                
                R_top_local = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
                t_top_local = np.array([0.0, 0.0, calc_rise])
                
                R_bottom_local = np.array([[cos_t, sin_t, 0], [-sin_t, cos_t, 0], [0, 0, 1]])
                t_bottom_local = np.array([0.0, 0.0, -calc_rise])

                R_top = R_align_inv @ R_top_local @ R_align
                t_top = global_centroid - R_top @ global_centroid + (R_align_inv @ t_top_local)

                R_bottom = R_align_inv @ R_bottom_local @ R_align
                t_bottom = global_centroid - R_bottom @ global_centroid + (R_align_inv @ t_bottom_local)

                if use_alt:
                    if len(sandwich) >= 2:
                        prev_b1 = sandwich[1]
                        prev_b2 = sandwich[0]
                    else:
                        prev_b1 = prev_b2 = sandwich[0]
                        
                    for _ in range(bottom_layers_to_add):
                        new_chain_id = get_new_chain_id()
                        expansions.append((prev_b1, new_chain_id, R_bottom, t_bottom))
                        prev_b1 = prev_b2
                        prev_b2 = new_chain_id

                    if len(sandwich) >= 2:
                        prev_t1 = sandwich[-2]
                        prev_t2 = sandwich[-1]
                    else:
                        prev_t1 = prev_t2 = sandwich[-1]
                        
                    for _ in range(top_layers_to_add):
                        new_chain_id = get_new_chain_id()
                        expansions.append((prev_t1, new_chain_id, R_top, t_top))
                        prev_t1 = prev_t2
                        prev_t2 = new_chain_id

                else:
                    current_ref_chain = bottom_chain
                    for _ in range(bottom_layers_to_add):
                        new_chain_id = get_new_chain_id()
                        expansions.append((current_ref_chain, new_chain_id, R_bottom, t_bottom))
                        current_ref_chain = new_chain_id
                        
                    current_ref_chain = top_chain
                    for _ in range(top_layers_to_add):
                        new_chain_id = get_new_chain_id()
                        expansions.append((current_ref_chain, new_chain_id, R_top, t_top))
                        current_ref_chain = new_chain_id

            col_auth_seq = headers.index("auth_seq_id") if "auth_seq_id" in headers else -1
            col_label_seq = headers.index("label_seq_id") if "label_seq_id" in headers else -1

            new_atom_lines = []
            het_seq_counters = {c: 1 for c in reserved_het_chains.values()}
            max_seq = {} 
            original_het_map = {}
            
            for parts in atom_lines:
                new_parts = parts[:]
                orig_c = new_parts[col_c]
                cid = orig_c
                
                if new_parts[col_group] == "HETATM":
                    res_name = new_parts[col_res]
                    if res_name in reserved_het_chains:
                        target_chain = reserved_het_chains[res_name]
                        new_parts[col_c] = target_chain
                        if "label_asym_id" in headers and "auth_asym_id" in headers:
                            new_parts[headers.index("label_asym_id")] = target_chain
                        
                        orig_seq = parts[col_auth_seq] if col_auth_seq != -1 else (parts[col_label_seq] if col_label_seq != -1 else "")
                        mol_key = (orig_c, orig_seq, res_name)
                        
                        if mol_key not in original_het_map:
                            original_het_map[mol_key] = het_seq_counters[target_chain]
                            het_seq_counters[target_chain] += 1
                            
                        new_seq = original_het_map[mol_key]
                        if col_auth_seq != -1: new_parts[col_auth_seq] = str(new_seq)
                        if col_label_seq != -1: new_parts[col_label_seq] = str(new_seq)
                        cid = target_chain 
                
                seq = -1
                if col_auth_seq != -1 and new_parts[col_auth_seq].isdigit(): seq = int(new_parts[col_auth_seq])
                elif col_label_seq != -1 and new_parts[col_label_seq].isdigit(): seq = int(new_parts[col_label_seq])
                if seq > max_seq.get(cid, 0): max_seq[cid] = seq
                
                new_atom_lines.append(new_parts)

            for ref_chain, new_chain, R, t in expansions:
                new_chain_parts = []
                for parts in chain_atoms.get(ref_chain, []):
                    new_parts = parts[:]
                    new_vec = R @ np.array([float(new_parts[col_x]), float(new_parts[col_y]), float(new_parts[col_z])]) + t
                    new_parts[col_c] = new_chain
                    if "label_asym_id" in headers and "auth_asym_id" in headers: new_parts[headers.index("label_asym_id")] = new_chain
                    new_parts[col_x], new_parts[col_y], new_parts[col_z] = f"{new_vec[0]:.3f}", f"{new_vec[1]:.3f}", f"{new_vec[2]:.3f}"
                    
                    seq = -1
                    if col_auth_seq != -1 and new_parts[col_auth_seq].isdigit(): seq = int(new_parts[col_auth_seq])
                    elif col_label_seq != -1 and new_parts[col_label_seq].isdigit(): seq = int(new_parts[col_label_seq])
                    if seq > max_seq.get(new_chain, 0): max_seq[new_chain] = seq
                    
                    new_atom_lines.append(new_parts)
                    new_chain_parts.append(new_parts)
                chain_atoms[new_chain] = new_chain_parts
                    
                new_het_lines_for_next = []
                het_res_map = {} 
                for parts in associated_atoms.get(ref_chain, []):
                    new_parts = parts[:]
                    new_vec = R @ np.array([float(new_parts[col_x]), float(new_parts[col_y]), float(new_parts[col_z])]) + t
                    
                    res_name = new_parts[col_res]
                    is_hetatm = new_parts[col_group] == "HETATM"
                    
                    if is_hetatm and res_name in reserved_het_chains:
                        target_chain = reserved_het_chains[res_name]
                        new_parts[col_c] = target_chain
                        if "label_asym_id" in headers and "auth_asym_id" in headers: new_parts[headers.index("label_asym_id")] = target_chain
                        new_parts[col_x], new_parts[col_y], new_parts[col_z] = f"{new_vec[0]:.3f}", f"{new_vec[1]:.3f}", f"{new_vec[2]:.3f}"
                        
                        orig_seq_str = parts[col_auth_seq] if col_auth_seq != -1 else (parts[col_label_seq] if col_label_seq != -1 else "")
                        orig_c_str = parts[col_c]
                        res_key = (orig_c_str, orig_seq_str, res_name)
                        
                        if res_key not in het_res_map:
                            het_res_map[res_key] = het_seq_counters[target_chain]
                            het_seq_counters[target_chain] += 1
                            
                        new_seq_str = str(het_res_map[res_key])
                        if col_auth_seq != -1: new_parts[col_auth_seq] = new_seq_str
                        if col_label_seq != -1: new_parts[col_label_seq] = new_seq_str
                    else:
                        target_chain = new_chain
                        new_parts[col_c] = target_chain
                        if "label_asym_id" in headers and "auth_asym_id" in headers: new_parts[headers.index("label_asym_id")] = target_chain
                        new_parts[col_x], new_parts[col_y], new_parts[col_z] = f"{new_vec[0]:.3f}", f"{new_vec[1]:.3f}", f"{new_vec[2]:.3f}"
                        
                        orig_seq_str = parts[col_auth_seq] if col_auth_seq != -1 else (parts[col_label_seq] if col_label_seq != -1 else "")
                        orig_c_str = parts[col_c]
                        res_key = (orig_c_str, orig_seq_str, res_name)
                        
                        if res_key not in het_res_map:
                            max_seq[target_chain] = max_seq.get(target_chain, 0) + 1
                            het_res_map[res_key] = max_seq[target_chain]
                            
                        new_seq_str = str(het_res_map[res_key])
                        if col_auth_seq != -1: new_parts[col_auth_seq] = new_seq_str
                        if col_label_seq != -1: new_parts[col_label_seq] = new_seq_str

                    new_atom_lines.append(new_parts)
                    new_het_lines_for_next.append(new_parts)
                associated_atoms[new_chain] = new_het_lines_for_next

            col_id = headers.index("id") if "id" in headers else -1
            if col_id != -1:
                for i, parts in enumerate(new_atom_lines): parts[col_id] = str(i + 1)

            import shlex
            chain_exp_map = {}
            for ref_c, new_c, _, _ in expansions:
                if ref_c not in chain_exp_map: chain_exp_map[ref_c] = []
                chain_exp_map[ref_c].append(new_c)

            def expand_cif_blocks(lines_list):
                expanded_list = []
                in_loop = False
                loop_headers = []
                chain_cols = []
                loop_data_rows = []

                def quote_cif_value(value):
                    """Return a valid single-line CIF token after shlex decoding."""
                    if value == "":
                        return "''"

                    lower_value = value.lower()
                    needs_quotes = (
                        any(ch.isspace() for ch in value)
                        or value[0] in "_#$;"
                        or lower_value in {"loop_", "stop_", "global_"}
                        or lower_value.startswith(("data_", "save_"))
                    )
                    if not needs_quotes:
                        return value
                    if "'" not in value:
                        return f"'{value}'"
                    if '"' not in value:
                        return f'"{value}"'

                    # Values requiring semicolon text fields are outside the
                    # single-line loops generated by ChimeraX.  Preserve the
                    # decoded value rather than silently deleting it.
                    return value
                
                def flush_loop():
                    if not loop_data_rows: return
                    col_widths = [0] * len(loop_headers) if loop_headers else []
                    
                    for row in loop_data_rows:
                        if isinstance(row, list) and len(row) > 1:
                            for i, val in enumerate(row):
                                if i < len(col_widths):
                                    col_widths[i] = max(col_widths[i], len(str(val)))
                    
                    for row in loop_data_rows:
                        if isinstance(row, list) and len(row) > 1:
                            formatted = []
                            for i, val in enumerate(row):
                                if i < len(col_widths):
                                    formatted.append(str(val).ljust(col_widths[i]))
                                else:
                                    formatted.append(str(val))
                            expanded_list.append(" ".join(formatted) + " \n")
                        elif isinstance(row, list):
                            expanded_list.append(str(row[0]) + " \n")
                        else:
                            expanded_list.append(row)
                    loop_data_rows.clear()

                for line in lines_list:
                    s = line.strip()
                    if s == "loop_":
                        if in_loop: flush_loop()
                        in_loop = True; loop_headers = []; chain_cols = []
                        expanded_list.append(line)
                        continue
                    if in_loop and s.startswith("_"):
                        loop_headers.append(s)
                        if "asym_id" in s or "pdb_strand_id" in s: 
                            chain_cols.append(len(loop_headers) - 1)
                        expanded_list.append(line)
                        continue
                    if in_loop and s and not s.startswith("#") and not s.startswith("_"):
                        # Expansion only needs to rewrite loops that reference
                        # chain IDs.  Keeping all other loop rows byte-for-byte
                        # avoids damaging valid CIF quoting (notably empty '').
                        if not chain_cols:
                            expanded_list.append(line)
                            continue

                        try: parts = shlex.split(s)
                        except: parts = s.split()
                        
                        parts_quoted = [quote_cif_value(p) for p in parts]
                        loop_data_rows.append(parts_quoted)
                        
                        if chain_cols and len(parts) >= len(loop_headers):
                            row_chains = []
                            for c_idx in chain_cols:
                                if c_idx < len(parts):
                                    val = parts[c_idx]
                                    if val not in ["?", "."]:
                                        row_chains.append(val)
                            
                            if row_chains and all(c in chain_exp_map for c in row_chains):
                                num_expansions = len(chain_exp_map[row_chains[0]])
                                if all(len(chain_exp_map[c]) == num_expansions for c in row_chains):
                                    for i in range(num_expansions):
                                        new_parts = parts_quoted[:]
                                        for c_idx in chain_cols:
                                            if c_idx < len(new_parts):
                                                old_c = new_parts[c_idx].strip("'\"")
                                                if old_c in chain_exp_map:
                                                    new_parts[c_idx] = chain_exp_map[old_c][i]
                                        loop_data_rows.append(new_parts)
                        continue
                    if s.startswith("#"):
                        if in_loop: flush_loop()
                        in_loop = False
                        expanded_list.append(line)
                        continue
                    if in_loop and not s:
                        loop_data_rows.append(line)
                        continue
                    if in_loop:
                        flush_loop()
                        in_loop = False
                    
                    expanded_list.append(line)
                    
                if in_loop: flush_loop()
                return expanded_list

            pre_lines = expand_cif_blocks(pre_lines)
            post_lines = expand_cif_blocks(post_lines)

            col_widths = []
            if new_atom_lines:
                num_cols = max(len(parts) for parts in new_atom_lines)
                col_widths = [0] * num_cols
                for parts in new_atom_lines:
                    for i, part in enumerate(parts):
                        str_part = f"'{part}'" if ' ' in str(part) and not str(part).startswith("'") and not str(part).startswith('"') else str(part)
                        col_widths[i] = max(col_widths[i], len(str_part))

            with open(output_path, 'w') as fout:
                for line in pre_lines: fout.write(line)
                for parts in new_atom_lines:
                    formatted_parts = []
                    for i, part in enumerate(parts):
                        str_part = f"'{part}'" if ' ' in str(part) and not str(part).startswith("'") and not str(part).startswith('"') else str(part)
                        if i < len(col_widths):
                            formatted_parts.append(str_part.ljust(col_widths[i]))
                        else:
                            formatted_parts.append(str_part)
                    
                    fout.write(" ".join(formatted_parts) + " \n")
                
                for line in post_lines: fout.write(line)

            return reported_twist, reported_rise, axis_u, tilt_deg

        def write_renamed_pdb(self, input_filename, output_filename, sandwiches):
            prot_mapping = self.create_renaming_mapping(sandwiches)
            het_types = set()
            with open(input_filename, 'r') as f:
                for line in f:
                    if line.startswith("HETATM") and len(line) >= 20:
                        het_types.add(line[17:20].strip())
            
            het_type_map = {}
            alphabet_backwards = "ZYXWVUTSRQPONMLKJIHGFEDCBA"
            for i, htype in enumerate(sorted(list(het_types))):
                if i < len(alphabet_backwards):
                    het_type_map[htype] = alphabet_backwards[i]

            het_counters = {cid: 1 for cid in het_type_map.values()}
            last_seen_residue = {cid: (None, None) for cid in het_type_map.values()}

            with open(input_filename, 'r') as fin, open(output_filename, 'w') as fout:
                for line in fin:
                    record = line[0:6].strip()
                    is_het = (record == "HETATM")
                    is_anisou = (record == "ANISOU")
                    res_name = line[17:20].strip() if len(line) >= 20 else ""
                    orig_chain = line[20:22].strip() if len(line) >= 22 else ""

                    if orig_chain not in prot_mapping and ((is_het) or (is_anisou and res_name in het_type_map)):
                        if res_name in het_type_map:
                            new_chain = het_type_map[res_name]
                            current_input_key = line[20:27] 
                            last_key, last_seq = last_seen_residue[new_chain]
                            if current_input_key == last_key: new_seq = last_seq
                            else:
                                new_seq = het_counters[new_chain]; het_counters[new_chain] += 1
                                last_seen_residue[new_chain] = (current_input_key, new_seq)
                            
                            l = list(line)
                            if len(new_chain) == 1: l[21] = new_chain; l[20] = ' ' 
                            elif len(new_chain) >= 2: l[20] = new_chain[0]; l[21] = new_chain[1]
                            new_seq_str = f"{new_seq:>4}"[-4:] 
                            l[22:26] = list(new_seq_str)
                            fout.write("".join(l))
                            continue 
                    
                    if record in self.CHAIN_RECORD_SPECS:
                        line_chars = list(line)
                        slice_list = self.CHAIN_RECORD_SPECS[record]
                        for start, end in slice_list:
                            if end <= len(line):
                                old_id = line[start:end].strip()
                                if old_id in prot_mapping:
                                    new_id = prot_mapping[old_id]
                                    width = end - start
                                    if len(new_id) <= width:
                                        formatted = f"{new_id:>{width}}"
                                        line_chars[start:end] = list(formatted)
                        fout.write("".join(line_chars))
                    else:
                        fout.write(line)

        def write_trimmed_pdb(self, input_path, output_path, keep_chains, keep_atom_indices=None):
            if keep_atom_indices is None: keep_atom_indices = set()
            serial_map = {}; new_serial_counter = 1
            master_counts = {
                "numRemark": 0, "numHet": 0, "numHelix": 0, "numSheet": 0,
                "numTurn": 0, "numSite": 0, "numXform": 0, "numCoord": 0,
                "numTer": 0, "numConect": 0, "numSeq": 0
            }
            sheet_buffer = []; helix_counter = 1

            def update_master_count(record_name):
                if record_name == "REMARK": master_counts["numRemark"] += 1
                elif record_name == "HET":    master_counts["numHet"] += 1
                elif record_name == "HELIX":  master_counts["numHelix"] += 1
                elif record_name == "SHEET":  master_counts["numSheet"] += 1
                elif record_name == "TURN":   master_counts["numTurn"] += 1
                elif record_name == "SITE":   master_counts["numSite"] += 1
                elif record_name in ["ATOM", "HETATM"]: master_counts["numCoord"] += 1
                elif record_name == "TER":    master_counts["numTer"] += 1
                elif record_name == "CONECT": master_counts["numConect"] += 1
                elif record_name == "SEQRES": master_counts["numSeq"] += 1
                elif record_name in ["ORIGX1", "ORIGX2", "ORIGX3", "SCALE1", "SCALE2", "SCALE3", "MTRIX1", "MTRIX2", "MTRIX3"]:
                    master_counts["numXform"] += 1

            with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
                for i, line in enumerate(fin):
                    record = line[0:6].strip()
                    if record in ["MASTER", "END"]: continue

                    if record == "SHEET":
                        sheet_buffer.append(line)
                        continue
                    if sheet_buffer and record != "SHEET":
                        self.process_and_write_sheets(sheet_buffer, keep_chains, fout, master_counts)
                        sheet_buffer = []

                    if record == "HELIX":
                        if len(line) > 31:
                            c1, c2 = line[19:21].strip(), line[31:33].strip()
                            if c1 in keep_chains and c2 in keep_chains:
                                new_id_str = f"{helix_counter:>3}"
                                new_line = line[:7] + new_id_str + line[10:]
                                fout.write(new_line)
                                update_master_count("HELIX")
                                helix_counter += 1
                        continue

                    if record in ["ATOM", "HETATM", "TER", "ANISOU"]:
                        if len(line) < 22: continue
                        chain_id = line[20:22].strip() 
                        is_kept_protein = chain_id in keep_chains
                        is_kept_water = i in keep_atom_indices
                        
                        if is_kept_protein or is_kept_water:
                            try:
                                old_serial = int(line[6:11])
                                if record == "ANISOU" and old_serial in serial_map: current_new_id = serial_map[old_serial]
                                else:
                                    current_new_id = new_serial_counter
                                    serial_map[old_serial] = current_new_id
                                    if record != "ANISOU": new_serial_counter += 1
                                new_line = line[:6] + f"{current_new_id:>5}" + line[11:]
                                fout.write(new_line)
                                update_master_count(record)
                            except ValueError:
                                fout.write(line)
                                update_master_count(record)
                        continue

                    if record == "CONECT":
                        try:
                            parts = line.split() 
                            if len(parts) < 2: continue
                            old_source = int(parts[1])
                            if old_source not in serial_map: continue
                            new_source = serial_map[old_source]
                            
                            valid_targets = []
                            for p in parts[2:]:
                                try:
                                    if int(p) in serial_map: valid_targets.append(serial_map[int(p)])
                                except: pass
                            
                            if not valid_targets: continue
                                
                            out_line = "CONECT" + f"{new_source:>5}"
                            for tgt in valid_targets: out_line += f"{tgt:>5}"
                            fout.write(out_line + "\n")
                            update_master_count("CONECT")
                        except: pass
                        continue

                    if record == "LINK":
                        try:
                            if len(line) > 51:
                                if line[21] in keep_chains and line[51] in keep_chains: fout.write(line)
                        except: pass
                        continue

                    should_write = True
                    if record in self.CHAIN_RECORD_SPECS:
                        slice_list = self.CHAIN_RECORD_SPECS[record]
                        chains_in_line = []
                        for start, end in slice_list:
                            if end <= len(line):
                                c = line[start:end].strip()
                                if c: chains_in_line.append(c)
                        if chains_in_line:
                            if not all(c in keep_chains for c in chains_in_line): should_write = False
                    
                    if should_write:
                        fout.write(line)
                        update_master_count(record)

                if sheet_buffer: self.process_and_write_sheets(sheet_buffer, keep_chains, fout, master_counts)

                master_line = "MASTER    {:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}{:>5}".format(
                    master_counts["numRemark"], "0", master_counts["numHet"], master_counts["numHelix"],
                    master_counts["numSheet"], master_counts["numTurn"], master_counts["numSite"],
                    master_counts["numXform"], master_counts["numCoord"], master_counts["numTer"],
                    master_counts["numConect"], master_counts["numSeq"]
                )
                fout.write(master_line + "\n")
                fout.write("END   \n")

        def process_and_write_sheets(self, sheet_lines, keep_chains, fout, master_counts):
            from collections import OrderedDict
            grouped_sheets = OrderedDict()
            for line in sheet_lines:
                if len(line) < 22: continue
                chain_id = line[20:22].strip() 
                if chain_id not in keep_chains: continue
                try: sheet_id = line.split()[2]
                except: sheet_id = line[11:14].strip()
                
                if sheet_id not in grouped_sheets: grouped_sheets[sheet_id] = []
                grouped_sheets[sheet_id].append(line)
            
            new_sheet_id_counter = 1
            for old_id, lines in grouped_sheets.items():
                total_strands = len(lines)
                if total_strands == 0: continue
                new_sheet_id_str = f"{new_sheet_id_counter:>3}"
                new_num_strands_str = f"{total_strands:>2}"
                current_strand_id = 1 
                for line in lines:
                    new_strand_id_str = f"{current_strand_id:>3}"
                    chars = list(line)
                    chars[7:10] = list(new_strand_id_str)
                    chars[11:14] = list(new_sheet_id_str)
                    chars[14:16] = list(new_num_strands_str)
                    
                    if current_strand_id == 1:
                        if len(chars) > 40: chars[38:40] = list(" 0")
                        if len(chars) > 41:
                            limit = min(len(chars), 70)
                            for i in range(41, limit): chars[i] = ' '
                    
                    fout.write("".join(chars))
                    master_counts["numSheet"] += 1
                    current_strand_id += 1
                new_sheet_id_counter += 1


    class ModifierLayerViewerTabs(QTabWidget):
        def __init__(self):
            super().__init__()
            self.pdb_widget = PDBLayerIdentifier()
            self.cif_widget = CIFLayerIdentifier()
            self.addTab(self.pdb_widget, "Modifier")
            self.addTab(self.cif_widget, "Layer Viewer")

    # Tests and embedding callers can obtain the exact components without
    # creating windows or entering QApplication.exec().
    if not launch:
        return {
            "OpenGLChainCanvas": OpenGLChainCanvas,
            "SoftwareChainCanvas": SoftwareChainCanvas,
            "ChainCanvas": ChainCanvasBase,
            "CIFLayerIdentifier": CIFLayerIdentifier,
            "PDBLayerIdentifier": PDBLayerIdentifier,
            "ModifierLayerViewerTabs": ModifierLayerViewerTabs
        }

    qt_app = QApplication.instance()
    if not qt_app:
        qt_app = QApplication(sys.argv)
    
    qt_app.setStyleSheet("""
        QMainWindow, QDialog, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Arial; font-size: 8pt; }
        
        QPushButton { background-color: #3e3e42; color: #d4d4d4; border: 1px solid #3e3e42; padding: 5px 15px; border-radius: 4px; }
        QPushButton:hover { background-color: #4e4e52; border: 1px solid #98c379; }
        QPushButton:pressed, QPushButton:checked { background-color: #98c379; color: #1e1e1e; border: 1px solid #98c379; }

        QPushButton#submitValidationButton,
        QPushButton#runExtractionButton,
        QPushButton#iterateDesignsButton {
            min-height: 28px;
            padding: 5px 12px;
            font-weight: normal;
        }
        QLineEdit, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit, QTextBrowser, QTableWidget, QTableView, QComboBox { 
            background-color: #1e1e1e; border: 1px solid #3e3e42; color: #cccccc; padding: 2px; border-radius: 2px;
            selection-background-color: #98c379; selection-color: #1e1e1e;
        }
        QTableView { alternate-background-color: #252526; gridline-color: #3e3e42; }
        
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 20px; 
        }

        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { 
            background-color: #252526; color: #d4d4d4; border: 1px solid #3e3e42; outline: none;
            selection-background-color: #98c379; selection-color: #1e1e1e;
        }
        QComboBox QAbstractItemView::item { padding: 2px 5px; min-height: 16px; border: none !important; }
        QComboBox QAbstractItemView::item:hover,
        QComboBox QAbstractItemView::item:selected { 
            background-color: #98c379; color: #1e1e1e; border: none !important; outline: none !important;
        }

        QTreeWidget, QListWidget, QTreeView { 
            background-color: #252526; border: 1px solid #3e3e42; color: #cccccc; outline: none; 
        }
        QTreeWidget::item, QListWidget::item { padding: 5px; }
        QTreeWidget::item:selected, QListWidget::item:selected { 
            background-color: #3e3e42; color: #ffffff; border-left: 3px solid #98c379; 
        }

        QHeaderView::section { background-color: #252526; color: #98c379; border: 1px solid #3e3e42; padding: 4px; font-weight: bold; }
        QTableCornerButton::section { background-color: #252526; border: 1px solid #3e3e42; }

        QMenu { background-color: #252526; color: #d4d4d4; border: 1px solid #3e3e42; }
        QMenu::item:selected { background-color: #3e3e42; color: #98c379; }

        QProgressBar { 
            border: 1px solid #3e3e42; border-radius: 2px; 
            background-color: #1e1e1e; text-align: center; color: #d4d4d4; 
        }
        QProgressBar::chunk { background-color: #98c379; border-radius: 2px; color: #1e1e1e; }

        QGroupBox { border: 1px solid #3e3e42; border-radius: 4px; margin-top: 1.0em; font-weight: bold; color: #98c379; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; border-radius: 2px; background-color: #1e1e1e; }
        QCheckBox::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        QRadioButton::indicator { width: 12px; height: 12px; border: 1px solid #555; border-radius: 7px; background-color: #1e1e1e; }
        QRadioButton::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        
        QSlider:vertical { min-width: 20px; }
        QSlider::groove:vertical { background: #3c3c3c; width: 6px; border-radius: 3px; }
        QSlider::handle:vertical { background: #98c379; height: 14px; width: 14px; margin: 0 -4px; border-radius: 7px; }
        QSlider::handle:vertical:hover { background: #b5e890; }
        
        QSlider:horizontal { min-height: 20px; }
        QSlider::groove:horizontal { background: #3c3c3c; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #98c379; height: 14px; width: 14px; margin: -4px 0; border-radius: 7px; }
        QSlider::handle:horizontal:hover { background: #b5e890; }
        
        QTabWidget::tab-bar { alignment: left; }
        QTabWidget::pane { border: 1px solid #3e3e42; top: -1px; }
        QTabBar::tab { background: #252526; border: 1px solid #3e3e42; padding: 6px 12px; min-width: 100px; color: #d4d4d4; }
        QTabBar::tab:selected { background: #3e3e42; color: #98c379; font-weight: bold; border-bottom: 1px solid #3e3e42; }
        QTabBar::tab:hover { background: #4e4e52; }
        
        QScrollBar:vertical { background: transparent; width: 12px; margin: 0px; }
        QScrollBar::handle:vertical { background-color: #4e4e52; min-height: 20px; border-radius: 5px; margin: 2px; }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed { background-color: #98c379; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; background: transparent; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        
        QScrollBar:horizontal { background: transparent; height: 12px; margin: 0px; }
        QScrollBar::handle:horizontal { background-color: #4e4e52; min-width: 20px; border-radius: 5px; margin: 2px; }
        QScrollBar::handle:horizontal:hover, QScrollBar::handle:horizontal:pressed { background-color: #98c379; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; background: transparent; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
    """)


    win = ModifierLayerViewerTabs()
    renderer_suffix = " (Software Renderer)" if use_software_renderer else ""
    win.setWindowTitle(f"Modifier + Layer Viewer{renderer_suffix}")
    win.resize(1250, 800)
    win.show()

    sys.exit(qt_app.exec())

if __name__ == "__main__":
    open_modifier_layer_viewer()
