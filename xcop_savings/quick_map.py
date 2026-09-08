import sys
import os
import re
import numpy as np
import mrcfile
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QGraphicsView, QGraphicsScene, 
    QWidget, QVBoxLayout, QPushButton, QInputDialog, QMessageBox, 
    QHBoxLayout, QLabel, QSlider, QSpinBox, QGraphicsPixmapItem,
    QFrame, QLineEdit, QGridLayout, QScrollArea, QSizePolicy,
    QGraphicsOpacityEffect, QCheckBox, QGraphicsRectItem, QGraphicsTextItem,
    QMenu, QFileDialog, QComboBox
)
from PyQt6.QtCore import Qt, QMimeData, QPoint, QRectF, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPalette, QColor, QPainter, QDrag, QFont, QPdfWriter, QPen, QAction
from PyQt6.QtSvg import QSvgGenerator

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
    dark_palette.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(152, 195, 121)) # #98c379 Green
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)
    
    app.setStyleSheet("""
        QWidget { background-color: #2b2b2b; color: #FFFFFF; }
        QLabel { background-color: transparent; }
        QPushButton { background-color: #3e3e3e; border: 1px solid #3e3e3e; border-radius: 5px; padding: 8px; color: white; }
        QPushButton:hover { background-color: #3e3e3e; border: 1px solid #98c379; }
        QPushButton:pressed { background-color: #98c379; color: #2b2b2b; }
        QLineEdit { background-color: #1e1e1e; border: 1px solid #444; color: white; padding: 4px; border-radius: 3px; }
        QLineEdit:focus { border: 1px solid #98c379; }
        QScrollArea { background-color: transparent; }
        
        /* Vertical ScrollBar */
        QScrollBar:vertical {
            border: none;
            background: transparent;
            width: 12px;
            margin: 0px;
        }
        QScrollBar::handle:vertical {
            background: #555;
            min-height: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:vertical:hover {
            background: #777;
        }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            border: none;
            background: none;
            height: 0px;
        }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
            background: transparent;
        }

        /* Horizontal ScrollBar */
        QScrollBar:horizontal {
            border: none;
            background: transparent;
            height: 12px;
            margin: 0px;
        }
        QScrollBar::handle:horizontal {
            background: #555;
            min-width: 20px;
            border-radius: 6px;
        }
        QScrollBar::handle:horizontal:hover {
            background: #777;
        }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            border: none;
            background: none;
            width: 0px;
        }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
            background: transparent;
        }
    """)

# --- The Graphics View ---
class MapViewer(QGraphicsView):
    def __init__(self, scene, parent_card=None):
        super().__init__(scene)
        self.parent_card = parent_card
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setBackgroundBrush(QColor(30, 30, 30))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.min_scale = None # Track the smallest allowed zoom level

    # ZOOM: Scroll Wheel
    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        
        # Limit zoom out so it cannot be smaller than the initial view
        if self.min_scale is not None:
            current_scale = self.transform().m11()
            if factor < 1.0 and (current_scale * factor) < self.min_scale:
                factor = self.min_scale / current_scale
                # If we are basically already at the minimum scale, don't do anything
                if factor >= 0.999: 
                    event.accept()
                    return

        old_pos = self.mapToScene(event.position().toPoint())
        self.scale(factor, factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
        # Accept the event so it doesn't propagate to the main scroll area
        event.accept()

    def contextMenuEvent(self, event):
        if not self.parent_card:
            return
        
        # Allow if it's a manual file OR if a job number is entered
        has_manual = hasattr(self.parent_card, 'manual_filepath') and self.parent_card.manual_filepath
        has_job = bool(self.parent_card.job_input.text().strip())
        
        if not (has_manual or has_job):
            return
            
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background-color: #333; color: white; border: 1px solid #555; }
            QMenu::item:selected { background-color: #98c379; color: #2b2b2b; }
        """)
        
        action_png = QAction("Save as PNG", self)
        action_svg = QAction("Save as Vector (SVG)", self)
        action_pdf = QAction("Save as Vector (PDF)", self)
        
        action_png.triggered.connect(self.parent_card.save_as_png)
        action_svg.triggered.connect(self.parent_card.save_as_svg)
        action_pdf.triggered.connect(self.parent_card.save_as_pdf)
        
        menu.addAction(action_png)
        menu.addAction(action_svg)
        menu.addAction(action_pdf)
        menu.exec(event.globalPos())

# --- The Job Card Widget ---
class JobCard(QFrame):
    def __init__(self, parent=None, delete_callback=None):
        super().__init__(parent)
        self.delete_callback = delete_callback
        self.setAcceptDrops(True)
        self.drag_start_pos = None
        
        # Card Styling
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("JobCard { border: 1px solid #555; border-radius: 8px; background-color: #333; }")
        self.setFixedSize(300, 460) # Increased height to allow for bottom padding 
        
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop) # Forces all visible elements to pack at the top
        
        # Top row: Header on left, Close button on right
        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lbl_header = QLabel("")
        self.lbl_header.setStyleSheet("color: #98c379; font-weight: bold;")
        top_layout.addWidget(self.lbl_header)
        
        top_layout.addStretch()
        
        btn_close = QPushButton("×")
        btn_close.setFixedSize(16, 16)
        btn_close.setStyleSheet("""
            QPushButton { background-color: transparent; color: #888; font-weight: bold; border: none; font-size: 14px; padding: 0px; margin: 0px; } 
            QPushButton:hover { color: #e74c3c; } 
        """)
        btn_close.clicked.connect(self.close_card)
        top_layout.addWidget(btn_close)
        layout.addLayout(top_layout)
        
        # Input row
        self.input_container = QWidget()
        self.input_container.setStyleSheet("background-color: transparent;")
        input_layout = QHBoxLayout(self.input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        
        # Add the label first so it appears on the left
        self.lbl_job_input = QLabel("Job Number:")
        input_layout.addWidget(self.lbl_job_input)
        
        self.job_input = QLineEdit()
        self.job_input.setPlaceholderText("001")
        self.job_input.returnPressed.connect(self.load_map)
        input_layout.addWidget(self.job_input)
        
        btn_show = QPushButton("Show")
        btn_show.clicked.connect(self.load_map)
        input_layout.addWidget(btn_show)
        layout.addWidget(self.input_container)
        
        self.manual_filepath = None
        
        # Info section
        self.lbl_alias = QLabel("")
        self.lbl_alias.setStyleSheet("color: #98c379; font-weight: bold;")
        self.lbl_alias.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_alias.setWordWrap(True)
        layout.addWidget(self.lbl_alias)
        
        self.lbl_details = QLabel("Box Size: -\nLowpass: -\nBlush: -\nTwist: -\nRise: -\nResolution: -")
        self.lbl_details.setStyleSheet("color: white; font-size: 11px; padding-left: 10px;")
        self.lbl_details.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.lbl_details.setWordWrap(True)
        layout.addWidget(self.lbl_details)
        
        # --- Common Properties (Scale Bar Length) ---
        common_prop_layout = QHBoxLayout()
        common_prop_layout.setContentsMargins(10, 0, 0, 0)
        lbl_scale_len = QLabel("Scale Bar (Å):")
        lbl_scale_len.setStyleSheet("color: white; font-size: 11px; background: transparent;")
        common_prop_layout.addWidget(lbl_scale_len)
        
        self.input_scale_len = QLineEdit("50")
        self.input_scale_len.setStyleSheet("QLineEdit { background-color: #1e1e1e; color: white; border: 1px solid #444; border-radius: 3px; padding: 2px; font-size: 11px; } QLineEdit:focus { border: 1px solid #98c379; }")
        self.input_scale_len.setFixedWidth(50)
        
        # Store the value to prevent double-reloads and connect to editingFinished
        self.last_scale_val = self.input_scale_len.text().strip()
        self.input_scale_len.editingFinished.connect(self.on_scale_len_changed)
        
        common_prop_layout.addWidget(self.input_scale_len)
        
        # New label to display the MRC pixel size
        self.lbl_mrc_pix = QLabel("Pixel size: N/A")
        self.lbl_mrc_pix.setStyleSheet("color: white; font-size: 11px; background: transparent; padding-left: 10px;")
        common_prop_layout.addWidget(self.lbl_mrc_pix)
        
        common_prop_layout.addStretch()
        
        self.common_prop_widget = QWidget()
        self.common_prop_widget.setStyleSheet("background: transparent;")
        self.common_prop_widget.setLayout(common_prop_layout)
        layout.addWidget(self.common_prop_widget)

        # --- Manual File Properties (Pixel Size & Box Size - Vertical) ---
        self.spi_pixel_layout = QVBoxLayout()
        self.spi_pixel_layout.setContentsMargins(10, 0, 0, 0)
        self.spi_pixel_layout.setSpacing(4)
        
        # Row 1: Box Size
        box_layout = QHBoxLayout()
        self.lbl_box = QLabel("Box Size:\n(px)")
        self.lbl_box.setStyleSheet("color: white; font-size: 11px; background: transparent;")
        self.lbl_box.setFixedWidth(95) # Wider to fit the w x h text
        box_layout.addWidget(self.lbl_box)
        
        self.input_box_size = QLineEdit()
        self.input_box_size.setPlaceholderText("e.g. 256")
        self.input_box_size.setStyleSheet("QLineEdit { background-color: #1e1e1e; color: white; border: 1px solid #444; border-radius: 3px; padding: 2px; font-size: 11px; } QLineEdit:focus { border: 1px solid #98c379; }")
        self.input_box_size.setFixedWidth(60)
        
        self.last_box_val = self.input_box_size.text().strip()
        self.input_box_size.editingFinished.connect(self.on_box_size_changed)
        box_layout.addWidget(self.input_box_size)
        box_layout.addStretch()
        self.spi_pixel_layout.addLayout(box_layout)

        # Row 2: Pixel Size
        self.pix_widget = QWidget()
        pix_layout = QHBoxLayout(self.pix_widget)
        pix_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_pixel = QLabel("Pixel Size (Å):")
        self.lbl_pixel.setStyleSheet("color: white; font-size: 11px; background: transparent;")
        self.lbl_pixel.setFixedWidth(95)
        pix_layout.addWidget(self.lbl_pixel)
        
        self.combo_pixel_size = QComboBox()
        self.combo_pixel_size.setStyleSheet("QComboBox { background-color: #1e1e1e; color: white; border: 1px solid #444; border-radius: 3px; padding: 2px; font-size: 11px; }")
        self.combo_pixel_size.addItems(["0.9557", "1.9114", "2.8671", "3.8228", "4.7785", "Custom..."])
        self.combo_pixel_size.activated.connect(self.on_pixel_size_changed)
        pix_layout.addWidget(self.combo_pixel_size)
        pix_layout.addStretch()
        self.spi_pixel_layout.addWidget(self.pix_widget)

        # Row 3: Crossover (For Rectangular SPI)
        self.cross_widget = QWidget()
        cross_layout = QHBoxLayout(self.cross_widget)
        cross_layout.setContentsMargins(0, 0, 0, 0)
        self.lbl_cross = QLabel("Crossover (Å):")
        self.lbl_cross.setStyleSheet("color: white; font-size: 11px; background: transparent;")
        self.lbl_cross.setFixedWidth(95)
        cross_layout.addWidget(self.lbl_cross)
        
        self.input_crossover = QLineEdit("350")
        self.input_crossover.setStyleSheet("QLineEdit { background-color: #1e1e1e; color: white; border: 1px solid #444; border-radius: 3px; padding: 2px; font-size: 11px; } QLineEdit:focus { border: 1px solid #98c379; }")
        self.input_crossover.setFixedWidth(60)
        self.last_crossover_val = self.input_crossover.text().strip()
        self.input_crossover.editingFinished.connect(self.on_crossover_changed)
        cross_layout.addWidget(self.input_crossover)
        cross_layout.addStretch()
        
        self.spi_pixel_layout.addWidget(self.cross_widget)
        self.cross_widget.hide() # Hidden by default unless a rectangular SPI is loaded
        
        self.spi_pixel_widget = QWidget()
        self.spi_pixel_widget.setStyleSheet("background: transparent;") 
        self.spi_pixel_widget.setLayout(self.spi_pixel_layout)
        self.spi_pixel_widget.hide() # Hidden by default
        layout.addWidget(self.spi_pixel_widget)
        
        # Canvas (Square display)
        self.scene = QGraphicsScene()
        self.view = MapViewer(self.scene, self)
        self.view.setFixedSize(250, 250)
        
        # Center the view in the layout
        view_layout = QHBoxLayout()
        view_layout.addStretch()
        view_layout.addWidget(self.view)
        view_layout.addStretch()
        layout.addLayout(view_layout)
        
        # Add a stretch at the bottom to act as padding below the canvas
        layout.addSpacing(15)
        
        self.pixmap_item = None
        self.scale_rect = None
        self.scale_text = None

    # --- DRAG AND DROP LOGIC ---
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton) or not self.drag_start_pos:
            return
        if (event.position().toPoint() - self.drag_start_pos).manhattanLength() < QApplication.startDragDistance():
            return
            
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setData("application/x-job-card", str(id(self)).encode())
        drag.setMimeData(mime_data)
        
        # Visual feedback for the floating drag icon
        pixmap = self.grab()
        drag.setPixmap(pixmap.scaledToWidth(150))
        drag.setHotSpot(QPoint(event.position().toPoint().x() // 2, event.position().toPoint().y() // 2))
        
        # Dim the card in the dashboard so it acts like a placeholder
        effect = QGraphicsOpacityEffect(self)
        effect.setOpacity(0.3)
        self.setGraphicsEffect(effect)
        
        drag.exec(Qt.DropAction.MoveAction)
        
        # Restore the card opacity when you release the mouse
        self.setGraphicsEffect(None)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat("application/x-job-card"):
            event.acceptProposedAction()
            
            # LIVE REORDER: Trigger the shift immediately when hovering over another card
            source_id = event.mimeData().data("application/x-job-card").data().decode()
            if source_id.isdigit():
                source_id_int = int(source_id)
                if id(self) != source_id_int and self.window() and hasattr(self.window(), 'reorder_cards'):
                    self.window().reorder_cards(source_id_int, id(self))

    def dropEvent(self, event):
        # The reordering is already handled visually during dragEnterEvent, so just accept
        event.acceptProposedAction()
        
    def close_card(self):
        if self.delete_callback:
            self.delete_callback(self)
        self.deleteLater()

    def on_scale_len_changed(self):
        current_val = self.input_scale_len.text().strip()
        if current_val != getattr(self, 'last_scale_val', ''):
            self.last_scale_val = current_val
            self.load_map()

    def on_box_size_changed(self):
        current_val = self.input_box_size.text().strip()
        if current_val != getattr(self, 'last_box_val', ''):
            self.last_box_val = current_val
            self.load_map()

    def on_crossover_changed(self):
        current_val = self.input_crossover.text().strip()
        if current_val != getattr(self, 'last_crossover_val', ''):
            self.last_crossover_val = current_val
            self.load_map()

    def on_pixel_size_changed(self, index):
        if self.combo_pixel_size.currentText() == "Custom...":
            val, ok = QInputDialog.getDouble(self, "Custom Pixel Size", "Enter pixel size (Å/px):", 1.0, 0.01, 100.0, 4)
            if ok:
                # Add custom value to the top of the list and select it
                self.combo_pixel_size.insertItem(0, str(val))
                self.combo_pixel_size.setCurrentIndex(0)
            else:
                # Revert selection to default if cancelled
                self.combo_pixel_size.setCurrentIndex(0)
        
        # Reload the map so the scale bar redraws with the new pixel size
        if hasattr(self, 'manual_filepath') and self.manual_filepath:
            self.load_map()
        
    def load_manual_file(self, filepath):
        self.manual_filepath = filepath
        self.input_container.hide()
        self.lbl_alias.hide()
        self.lbl_details.hide()
        
        if filepath.lower().endswith('.spi'):
            self.spi_pixel_widget.show()
        
        filename = os.path.basename(filepath)
        self.lbl_header.setText(f"{filename}")
        self.lbl_header.setStyleSheet("color: #61afef; font-weight: bold;") # Blue for manual files
        self.load_map()

    def load_map(self):
        if hasattr(self, 'manual_filepath') and self.manual_filepath:
            mrc_path = self.manual_filepath
        else:
            job_num = self.job_input.text().strip()
            if not job_num: return
            
            if job_num.isdigit() and len(job_num) < 3:
                job_num = job_num.zfill(3)
                self.job_input.setText(job_num)
                
            cwd = os.getcwd()
            job_type = "Refine3D"
            job_dir = os.path.join(cwd, "Refine3D", f"job{job_num}")
            mrc_path = os.path.join(job_dir, "run_class001.mrc")
                
            if not os.path.exists(mrc_path):
                job_type = "PostProcess"
                job_dir = os.path.join(cwd, "PostProcess", f"job{job_num}")
                mrc_path = os.path.join(job_dir, "postprocess.mrc")
                if not os.path.exists(mrc_path):
                    job_dir = os.path.join(cwd, "Postprocess", f"job{job_num}")
                    mrc_path = os.path.join(job_dir, "postprocess.mrc")

            if not os.path.exists(mrc_path):
                QMessageBox.warning(self, "Error", f"Could not find Refine3D or PostProcess map for job {job_num}")
                return
                
            box_size = "N/A"
            try:
                with mrcfile.open(mrc_path, header_only=True, permissive=True) as mrc:
                    box_size = str(int(mrc.header.nx))
            except Exception: pass
                
            if job_type == "Refine3D":
                self.lbl_header.setText("Refine3D")
                self.lbl_header.setStyleSheet("color: #98c379; font-weight: bold;")
                self.lbl_alias.setStyleSheet("color: #98c379; font-weight: bold;")
            else:
                self.lbl_header.setText("PostProcess")
                self.lbl_header.setStyleSheet("color: #e5c07b; font-weight: bold;")
                self.lbl_alias.setStyleSheet("color: #e5c07b; font-weight: bold;")

            alias = None
            star_path = os.path.join(cwd, "default_pipeline.star")
            if os.path.exists(star_path):
                try:
                    with open(star_path, 'r') as f:
                        for line in f:
                            if f"job{job_num}/" in line:
                                parts = line.strip().split()
                                if len(parts) >= 2 and parts[1] != "None":
                                    alias_parts = parts[1].strip('/').split('/')
                                    if len(alias_parts) > 1:
                                        alias = alias_parts[-1]
                                break
                except Exception: pass
                
            self.lbl_alias.setText(alias if alias else f"job{job_num}")
            
            run_out_path = os.path.join(job_dir, "run.out")
            if job_type == "Refine3D":
                filter_val, blush_val, twist_val, rise_val, res_val = "N/A", "N/A", "N/A", "N/A", "N/A"
                if os.path.exists(run_out_path):
                    try:
                        with open(run_out_path, 'r') as f:
                            content = f.read()
                            blush_val = "Yes" if "--blush" in content else "No"
                            match_filter = re.search(r'--ini_high\s+([\d\.]+)', content)
                            if match_filter: filter_val = match_filter.group(1)
                            matches_helix = re.findall(r'helical twist = ([-+]?[\d\.]+) degrees, rise = ([-+]?[\d\.]+) Angstroms', content)
                            if matches_helix:
                                twist_val, rise_val = matches_helix[-1][0], matches_helix[-1][1]
                            matches_res = re.findall(r'Final resolution.*?is:\s*([\d\.]+)', content)
                            if matches_res: res_val = matches_res[-1]
                    except Exception: pass
                self.lbl_details.setText(f"Box Size: {box_size} px\nLowpass Filter: {filter_val}\nBlush: {blush_val}\nTwist: {twist_val}\nRise: {rise_val}\nResolution: {res_val}")
            else:
                b_factor, res_val = "N/A", "N/A"
                if os.path.exists(run_out_path):
                    try:
                        with open(run_out_path, 'r') as f:
                            content = f.read()
                            matches_bfactor = re.findall(r'\+\s+apply b-factor of:\s+([-+]?[\d\.]+)', content)
                            if matches_bfactor: b_factor = matches_bfactor[-1]
                            matches_res = re.findall(r'\+\s+FINAL RESOLUTION:\s+([\d\.]+)', content)
                            if matches_res: res_val = matches_res[-1]
                    except Exception: pass
                self.lbl_details.setText(f"Box Size: {box_size} px\nB-factor: {b_factor}\nResolution: {res_val}\n\n\n")
            
        # 3. Process Map
        try:
            voxel_size_mrc = None
            if mrc_path.lower().endswith('.spi'):
                with open(mrc_path, 'rb') as f:
                    # Quick standard SPIDER read
                    head = np.fromfile(f, dtype='<f4', count=256)
                    nslice, nrow, nsam = int(head[0]), int(head[1]), int(head[11])
                    f.seek(0, os.SEEK_END)
                    filesize = f.tell()
                    f.seek(filesize - (nslice * nrow * nsam * 4))
                    map_data = np.fromfile(f, dtype='<f4').reshape(nslice, nrow, nsam)
            else:
                with mrcfile.open(mrc_path, permissive=True) as mrc:
                    map_data = mrc.data
                    if mrc.voxel_size.x > 0:
                        voxel_size_mrc = float(mrc.voxel_size.x)
                
            if map_data.ndim != 3: 
                QMessageBox.warning(self, "Error", "Map is not 3D.")
                return
            
            z_slices = map_data.shape[0]
            center_z = z_slices // 2
            
            # Fetch slice count dynamically from the parent window's toolbar
            try:
                num_slices = int(self.window().slice_input.text())
            except (ValueError, AttributeError):
                num_slices = 5
                
            half_k = num_slices // 2
            
            # Calculate dynamic start and end boundaries
            if num_slices % 2 == 0:
                start_z = max(0, center_z - half_k)
                end_z = min(z_slices, center_z + half_k)
            else:
                start_z = max(0, center_z - half_k)
                end_z = min(z_slices, center_z + half_k + 1)
                
            if start_z >= end_z:
                end_z = start_z + 1
            
            slice_block = map_data[start_z:end_z, :, :]
            avg_slice = np.mean(slice_block, axis=0)
            
            # Normalize to grayscale
            f_min, f_max = avg_slice.min(), avg_slice.max()
            if f_max > f_min:
                norm_frame = 255.0 * (avg_slice - f_min) / (f_max - f_min)
            else:
                norm_frame = np.zeros_like(avg_slice)
                
            img_uint8 = np.require(norm_frame.astype(np.uint8), np.uint8, 'C')
            h, w = img_uint8.shape
            
            # Update the Box label to show the true dimensions read from the file
            if hasattr(self, 'lbl_box'):
                if 'projection' in os.path.basename(mrc_path).lower():
                    self.lbl_box.setText(f"Box Length:\n({w}x{h})")
                else:
                    self.lbl_box.setText(f"Box Size:\n({w}x{h})")
                
            qimg = QImage(img_uint8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
            pixmap = QPixmap.fromImage(qimg)
            
            if self.pixmap_item is None:
                self.pixmap_item = QGraphicsPixmapItem(pixmap)
                self.scene.addItem(self.pixmap_item)
            else:
                self.pixmap_item.setPixmap(pixmap)
            
            # Force smooth blending (bilinear filtering) to avoid blocky pixels
            self.pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
            self.pixmap_item.setPos(-w/2, -h/2)
            
            # --- SCALE BAR LOGIC ---
            voxel_size = 1.0 # fallback
            mrc_pix_detected = False
            
            try:
                if mrc_path.lower().endswith('.spi'):
                    try:
                        text_val = self.combo_pixel_size.currentText()
                        if text_val != "Custom...":
                            voxel_size = float(text_val)
                    except ValueError:
                        pass
                    
                    # If user provided a custom box size, adjust voxel_size ratio
                    is_proj = 'projection' in os.path.basename(mrc_path).lower()
                    custom_box = self.input_box_size.text().strip()
                    
                    if custom_box and custom_box.isdigit():
                        override_box = float(custom_box)
                        if is_proj:
                            # Scale against width (length) for projections
                            if override_box > 0 and override_box != w:
                                voxel_size = (override_box * voxel_size) / w
                        else:
                            # Scale against height for standard square boxes
                            if override_box > 0 and override_box != h:
                                voxel_size = (override_box * voxel_size) / h
                    elif not custom_box:
                        if is_proj:
                            self.input_box_size.setText(str(w))
                        else:
                            self.input_box_size.setText(str(max(w, h)))
                        
                elif voxel_size_mrc is not None:
                    voxel_size = voxel_size_mrc
                    mrc_pix_detected = True
                    
            except Exception: pass
            
            # Global MRC Pixel Size Label Logic
            if hasattr(self, 'lbl_mrc_pix'):
                if mrc_path.lower().endswith('.spi'):
                    self.lbl_mrc_pix.hide()
                else:
                    self.lbl_mrc_pix.show()
                    if mrc_pix_detected:
                        self.lbl_mrc_pix.setText(f"Pixel size: {voxel_size:.4f} Å")
                    else:
                        self.lbl_mrc_pix.setText("Pixel size: N/A")
            
            # Read custom scale bar length
            try:
                angstroms = float(self.input_scale_len.text().strip())
                if angstroms <= 0: angstroms = 50.0
            except ValueError:
                angstroms = 50.0
                
            # --- Calculate Bar Length in Pixels ---
            if mrc_path.lower().endswith('.spi') and w != h:
                # RECTANGULAR SPI LOGIC
                if hasattr(self, 'cross_widget'): self.cross_widget.show()
                if hasattr(self, 'pix_widget'): self.pix_widget.hide()
                try:
                    crossover = float(self.input_crossover.text().strip())
                    if crossover <= 0: crossover = 350.0
                except ValueError:
                    crossover = 350.0
                
                bar_length_px = (angstroms * w) / (2.0 * crossover)
            else:
                # SQUARE SPI OR STANDARD MRC LOGIC
                if hasattr(self, 'cross_widget'): self.cross_widget.hide()
                if hasattr(self, 'pix_widget'): self.pix_widget.show()
                bar_length_px = angstroms / voxel_size
            
            if not self.scale_rect:
                self.scale_rect = QGraphicsRectItem()
                self.scale_rect.setBrush(QColor(255, 255, 255))
                self.scale_rect.setPen(QPen(Qt.PenStyle.NoPen))
                self.scene.addItem(self.scale_rect)
                
                self.scale_text = QGraphicsTextItem()
                self.scale_text.setDefaultTextColor(QColor(255, 255, 255))
                font = QFont("Helvetica", max(6, int(h * 0.04)), QFont.Weight.Bold)
                font.setStyleHint(QFont.StyleHint.SansSerif)
                self.scale_text.setFont(font)
                self.scene.addItem(self.scale_text)
                
            # Update text to match user input
            display_text = f"{int(angstroms)} Å" if angstroms.is_integer() else f"{angstroms} Å"
            self.scale_text.setPlainText(display_text)
            
            bar_height = max(2, h * 0.015)
            margin_x = max(2, w * 0.05)
            margin_y = max(2, h * 0.05)
            
            # Position at bottom left of the image
            rect_x = -w/2 + margin_x
            rect_y = h/2 - margin_y - bar_height
            self.scale_rect.setRect(rect_x, rect_y, bar_length_px, bar_height)
            self.scale_text.setPos(rect_x - 2, rect_y - self.scale_text.boundingRect().height() + 2)
            
            # Sync visibility with main window
            main_win = self.window()
            if main_win and hasattr(main_win, 'chk_scale_bar'):
                self.set_scale_bar_visible(main_win.chk_scale_bar.isChecked())
            
            self.view.resetTransform()
            
            scale_factor = min(240/w, 240/h)
            self.view.scale(scale_factor, scale_factor)
            self.view.min_scale = scale_factor # Save the baseline size as our zoom limit
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load MRCS:\n{str(e)}")

    def set_scale_bar_visible(self, visible):
        if self.scale_rect: self.scale_rect.setVisible(visible)
        if self.scale_text: self.scale_text.setVisible(visible)

    def save_as_png(self):
        if hasattr(self, 'manual_filepath') and self.manual_filepath:
            default_name = f"{os.path.splitext(os.path.basename(self.manual_filepath))[0]}.png"
        else:
            job_num = self.job_input.text().strip()
            default_name = f"Job{job_num}.png"
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Image", default_name, "PNG Images (*.png)")
        if path:
            self.scene.clearSelection()
            rect = self.scene.itemsBoundingRect()
            
            # Multiply the resolution by 4x for a high-quality export
            scale_factor = 4.0
            new_w = int(rect.width() * scale_factor)
            new_h = int(rect.height() * scale_factor)
            
            image = QImage(new_w, new_h, QImage.Format.Format_ARGB32)
            image.fill(Qt.GlobalColor.transparent)
            
            painter = QPainter(image)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            
            target_rect = QRectF(0.0, 0.0, float(new_w), float(new_h))
            self.scene.render(painter, target=target_rect, source=rect)
            painter.end()
            image.save(path)

    def save_as_svg(self):
        if hasattr(self, 'manual_filepath') and self.manual_filepath:
            default_name = f"{os.path.splitext(os.path.basename(self.manual_filepath))[0]}.svg"
        else:
            job_num = self.job_input.text().strip()
            default_name = f"Job{job_num}.svg"
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Vector", default_name, "SVG Files (*.svg)")
        if path:
            rect = self.scene.itemsBoundingRect()
            generator = QSvgGenerator()
            generator.setFileName(path)
            generator.setSize(rect.size().toSize())
            generator.setViewBox(rect)
            painter = QPainter(generator)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            self.scene.render(painter, target=QRectF(rect), source=rect)
            painter.end()

    def save_as_pdf(self):
        if hasattr(self, 'manual_filepath') and self.manual_filepath:
            default_name = f"{os.path.splitext(os.path.basename(self.manual_filepath))[0]}.pdf"
        else:
            job_num = self.job_input.text().strip()
            default_name = f"Job{job_num}.pdf"
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Vector", default_name, "PDF Files (*.pdf)")
        if path:
            rect = self.scene.itemsBoundingRect()
            writer = QPdfWriter(path)
            writer.setResolution(300) # High resolution for crisp vector parsing
            painter = QPainter(writer)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
            
            # Maintain aspect ratio to prevent the image from stretching to fill the page
            aspect_ratio = rect.width() / max(1, rect.height())
            dev_w = painter.device().width()
            dev_h = painter.device().height()
            
            new_w = dev_w
            new_h = dev_w / aspect_ratio
            if new_h > dev_h:
                new_h = dev_h
                new_w = dev_h * aspect_ratio
                
            target_rect = QRectF(0, 0, new_w, new_h)
            self.scene.render(painter, target=target_rect, source=rect)
            painter.end()

# --- Main Window (Dashboard) ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Quick Map Visualizer Dashboard")
        self.resize(1310, 535)
        
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        
        # Toolbar
        toolbar = QHBoxLayout()
        btn_add = QPushButton("+ Add Job Card")
        btn_add.setFixedWidth(120)
        btn_add.clicked.connect(self.add_card)
        toolbar.addWidget(btn_add)
        
        btn_browse = QPushButton("Browse Files...")
        btn_browse.setFixedWidth(120)
        btn_browse.clicked.connect(self.browse_files)
        toolbar.addWidget(btn_browse)
        
        # Slices Input (Moved between the button and the instruction)
        lbl_slices = QLabel("  Averaging Slices:")
        lbl_slices.setStyleSheet("color: white;")
        toolbar.addWidget(lbl_slices)
        
        self.slice_input = QLineEdit("5")
        self.slice_input.setFixedWidth(40)
        self.slice_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slice_input.textChanged.connect(self.update_all_cards)
        toolbar.addWidget(self.slice_input)

        toolbar.addSpacing(20)
        self.chk_scale_bar = QCheckBox("Scale Bar")
        self.chk_scale_bar.setStyleSheet("QCheckBox { color: white;}")
        self.chk_scale_bar.setChecked(True)
        self.chk_scale_bar.stateChanged.connect(self.toggle_scale_bars)
        toolbar.addWidget(self.chk_scale_bar)
        
        toolbar.addStretch()
        self.lbl_dashboard_info = QLabel("Compare multiple Refine3D/Postprocess jobs side-by-side")
        self.lbl_dashboard_info.setStyleSheet("color: #aaa;")
        toolbar.addWidget(self.lbl_dashboard_info)
        
        main_layout.addLayout(toolbar)
        
        # Scroll Area for the Dashboard Grid
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        
        self.dashboard_widget = QWidget()
        self.dashboard_layout = QGridLayout(self.dashboard_widget)
        self.dashboard_layout.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.dashboard_layout.setSpacing(20)
        self.scroll_area.setWidget(self.dashboard_widget)
        
        main_layout.addWidget(self.scroll_area)
        
        self.cards = []
        
        # Start the app with one empty card
        self.add_card()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.refresh_layout()
        
    def add_card(self):
        card = JobCard(delete_callback=self.remove_card)
        self.cards.append(card)
        self.refresh_layout()
        
    def browse_files(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select Map Files", "", "Map Files (*.mrc *.mrcs *.spi);;All Files (*)")
        for file_path in files:
            card = JobCard(delete_callback=self.remove_card)
            self.cards.append(card)
            card.load_manual_file(file_path)
        self.refresh_layout()

    def remove_card(self, card):
        if card in self.cards:
            self.cards.remove(card)
            self.refresh_layout()
            
    def reorder_cards(self, source_id, target_id):
        source_card = next((c for c in self.cards if id(c) == source_id), None)
        target_card = next((c for c in self.cards if id(c) == target_id), None)
        
        if source_card and target_card and source_card != target_card:
            # Re-insert source_card at target_card's position
            target_idx = self.cards.index(target_card)
            self.cards.remove(source_card)
            self.cards.insert(target_idx, source_card)
            self.refresh_layout()
            
    def update_all_cards(self):
        try:
            slices = int(self.slice_input.text())
            if slices > 0:
                for card in self.cards:
                    # Only reload cards that already have a job number inputted
                    if card.job_input.text().strip():
                        card.load_map()
        except ValueError:
            pass # Ignore invalid inputs (like an empty string) while the user is typing

    def toggle_scale_bars(self):
        show_bar = self.chk_scale_bar.isChecked()
        for card in self.cards:
            card.set_scale_bar_visible(show_bar)

    def refresh_layout(self):
        # Clear the current grid
        for i in reversed(range(self.dashboard_layout.count())): 
            item = self.dashboard_layout.itemAt(i)
            if item.widget():
                self.dashboard_layout.removeWidget(item.widget())
            
        # Re-populate the grid dynamically based on window width
        # Card width is 300, spacing is 20
        available_width = self.scroll_area.viewport().width()
        columns = max(1, available_width // 320)
        
        for i, card in enumerate(self.cards):
            row = i // columns
            col = i % columns
            self.dashboard_layout.addWidget(card, row, col)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    set_dark_theme(app)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())