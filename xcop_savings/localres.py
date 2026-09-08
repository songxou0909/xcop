import os
import sys
import subprocess
import argparse
import numpy as np

from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QLineEdit, 
                             QPushButton, QMessageBox, QFileDialog, 
                             QGridLayout, QVBoxLayout, QHBoxLayout,
                             QCheckBox, QSlider, QSplitter)
from PyQt6.QtCore import Qt

try:
    import mrcfile
    import scipy.ndimage
    import matplotlib
    matplotlib.use('QtAgg')
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure
    import matplotlib.colors as mcolors
    import matplotlib.patches as patches
    HAS_PREVIEW_LIBS = True
except ImportError:
    HAS_PREVIEW_LIBS = False

# Catch the arguments passed (if any)
parser = argparse.ArgumentParser()
parser.add_argument("--chimera-ver", type=str, default="chimera")
args, unknown = parser.parse_known_args()
CHIMERA_VER = args.chimera_ver

def show_error_message(parent, message):
    msg = QMessageBox(parent)
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Error")
    msg.setText(str(message))
    msg.exec()

def center_window(window, width, height):
    # PyQt6 no longer uses QApplication.desktop()
    screen_geometry = window.screen().availableGeometry()
    x = (screen_geometry.width() - width) // 2
    y = (screen_geometry.height() - height) // 2
    window.setGeometry(x, y, width, height)

def write_to_log(command):
    print(f"Log: {command}")

# --- REFACTORED STANDALONE WINDOW CLASS ---
class LocalResWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Local Resolution")
        center_window(self, 1100, 500)
        
        self.data_map = None
        self.data_locres = None
        self.map_min = 0
        self.map_max = 1
        self.map_rms = 1.0
        self._loaded_map_path = None
        self._loaded_locres_path = None
        self._updating_sliders = False
        
        self.init_ui()
        
    def init_ui(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)
        left_layout.setContentsMargins(10, 10, 10, 10)
        
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        
        # --- Helper for Row Creation ---
        def add_browse_row(row_idx, label_txt, filter_str, default_txt=""):
            grid_layout.addWidget(QLabel(label_txt), row_idx, 0)
            
            entry = QLineEdit(default_txt)
            grid_layout.addWidget(entry, row_idx, 1)
            
            btn = QPushButton("Browse")
            
            def browse():
                current_text = entry.text().strip()
                start_path = current_text if current_text else os.getcwd()
                if start_path and not os.path.isabs(start_path):
                    start_path = os.path.join(os.getcwd(), start_path)

                if "Output" in label_txt:
                    path, _ = QFileDialog.getSaveFileName(self, f"Select {label_txt}", start_path, filter_str)
                else:
                    path, _ = QFileDialog.getOpenFileName(self, f"Select {label_txt}", start_path, filter_str)
                
                if path:
                    entry.setText(os.path.abspath(path))
                    self.check_auto_preview()
                    
            btn.clicked.connect(browse)
            grid_layout.addWidget(btn, row_idx, 2)
            entry.editingFinished.connect(self.check_auto_preview)
            return entry

        # --- Input Rows ---
        self.entry_map = add_browse_row(0, "PostProcess Map (.mrc):", "Map Files (*.mrc);;All Files (*)")
        self.entry_locres = add_browse_row(1, "RELION LocRes Map (.mrc):", "Map Files (*.mrc);;All Files (*)")
        self.entry_pdb = add_browse_row(2, "Single Layer Model (.pdb):", "PDB Files (*.pdb);;All Files (*)")
        
        # --- Resolution & Range Row ---
        grid_layout.addWidget(QLabel("Color Thresholds:"), 3, 0)
        
        res_container = QWidget()
        rl = QHBoxLayout(res_container)
        rl.setContentsMargins(0, 0, 0, 0)
        
        rl.addWidget(QLabel("Resolution (Å):"))
        self.entry_res = QLineEdit("3.0")
        self.entry_res.setFixedWidth(60)
        rl.addWidget(self.entry_res)
        
        rl.addWidget(QLabel("Range:"))
        self.entry_range = QLineEdit("0.2")
        self.entry_range.setFixedWidth(60)
        rl.addWidget(self.entry_range)
        rl.addStretch()
        grid_layout.addWidget(res_container, 3, 1, 1, 2)

        self.entry_out = add_browse_row(4, "Output Image (.png):", "PNG Files (*.png)", "LocalRes.png")
        
        self.chk_cxc = QCheckBox("Export as ChimeraX CXC File")
        self.chk_cxc.setChecked(True)
        grid_layout.addWidget(self.chk_cxc, 5, 1)

        left_layout.addLayout(grid_layout)

        # --- Preview Controls ---
        self.chk_overlay = QCheckBox("Overlay LocRes on Map")
        self.chk_overlay.setChecked(True)
        left_layout.addWidget(self.chk_overlay)
        
        level_layout = QHBoxLayout()
        self.lbl_level = QLabel("Level: ")
        self.lbl_level.setFixedWidth(120)
        self.slider_level = QSlider(Qt.Orientation.Horizontal)
        self.slider_level.setRange(0, 1000)
        self.slider_level.setEnabled(False)
        level_layout.addWidget(self.lbl_level)
        level_layout.addWidget(self.slider_level)
        left_layout.addLayout(level_layout)
        
        rms_layout = QHBoxLayout()
        self.lbl_rms = QLabel("RMS Level: 5.00")
        self.lbl_rms.setFixedWidth(120)
        self.slider_rms = QSlider(Qt.Orientation.Horizontal)
        self.slider_rms.setRange(0, 400)
        self.slider_rms.setEnabled(False)
        rms_layout.addWidget(self.lbl_rms)
        rms_layout.addWidget(self.slider_rms)
        left_layout.addLayout(rms_layout)

        left_layout.addStretch()

        btn_run = QPushButton("Generate / Run")
        btn_run.setStyleSheet("font-weight: bold; padding: 10px;")
        btn_run.clicked.connect(self.generate_script)
        left_layout.addWidget(btn_run)

        # --- Right Panel (Canvas) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        if HAS_PREVIEW_LIBS:
            self.fig = Figure(figsize=(4, 4), dpi=100)
            self.fig.patch.set_facecolor('#1e1e1e')
            self.ax = self.fig.add_subplot(111)
            self.ax.set_facecolor('#1e1e1e')
            self.ax.axis('off')
            self.ax.text(0.5, 0.5, "Select Maps to Auto-Preview", color='white', 
                         fontsize=12, ha='center', va='center', transform=self.ax.transAxes)
            
            self.canvas = FigureCanvas(self.fig)
            right_layout.addWidget(self.canvas)
            
            self.slider_level.valueChanged.connect(self.on_level_changed)
            self.slider_rms.valueChanged.connect(self.on_rms_changed)
            self.chk_overlay.toggled.connect(self.draw_plot)
            self.entry_res.textChanged.connect(self.draw_plot)
            self.entry_range.textChanged.connect(self.draw_plot)
        else:
            lbl_no_libs = QLabel("Preview disabled.\nPlease install 'mrcfile', 'scipy', and 'matplotlib'.")
            lbl_no_libs.setAlignment(Qt.AlignmentFlag.AlignCenter)
            right_layout.addWidget(lbl_no_libs)

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)
        
        main_layout.addWidget(splitter)

    # --- SLIDER LOGIC ---
    def on_level_changed(self, value):
        if self._updating_sliders or not self.data_map is not None: return
        self._updating_sliders = True
        
        if self.map_rms > 0:
            slider_frac = value / 1000.0
            current_threshold = self.map_min + slider_frac * (self.map_max - self.map_min)
            current_rms_level = current_threshold / self.map_rms
            
            rms_val_int = int(current_rms_level * 40)
            self.slider_rms.blockSignals(True)
            self.slider_rms.setValue(max(0, min(400, rms_val_int)))
            self.slider_rms.blockSignals(False)
            self.lbl_rms.setText(f"RMS Level: {current_rms_level:.2f}")

        self.draw_plot()
        self._updating_sliders = False

    def on_rms_changed(self, value):
        if self._updating_sliders or not self.data_map is not None: return
        self._updating_sliders = True
        
        if self.map_rms > 0:
            snapped_value = round(value / 10) * 10
            if snapped_value != value:
                self.slider_rms.blockSignals(True)
                self.slider_rms.setValue(snapped_value)
                self.slider_rms.blockSignals(False)
                
            rms_level = snapped_value / 40.0
            current_threshold = rms_level * self.map_rms
            
            if self.map_max > self.map_min:
                slider_frac = (current_threshold - self.map_min) / (self.map_max - self.map_min)
                level_val_int = int(slider_frac * 1000)
                self.slider_level.blockSignals(True)
                self.slider_level.setValue(max(0, min(1000, level_val_int)))
                self.slider_level.blockSignals(False)
                
            self.lbl_rms.setText(f"RMS Level: {rms_level:.2f}")

        self.draw_plot()
        self._updating_sliders = False

    # --- AUTO PREVIEW LOGIC ---
    def check_auto_preview(self):
        if not HAS_PREVIEW_LIBS: return
        
        map_path = self.entry_map.text().strip()
        locres_path = self.entry_locres.text().strip()
        
        if map_path and locres_path and os.path.exists(map_path) and os.path.exists(locres_path):
            if map_path != self._loaded_map_path or locres_path != self._loaded_locres_path:
                self.load_data(map_path, locres_path)

    def load_data(self, map_path, locres_path):
        try:
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Loading maps...", color='white', 
                         fontsize=12, ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw_idle()
            QApplication.processEvents()

            with mrcfile.open(map_path, permissive=True) as mrc:
                self.data_map = mrc.data.astype(np.float32)
            with mrcfile.open(locres_path, permissive=True) as mrc:
                self.data_locres = mrc.data.astype(np.float32)
                
            self.map_min = float(np.min(self.data_map))
            self.map_max = float(np.max(self.data_map))
            if self.map_max == self.map_min:
                self.map_max = self.map_min + 1.0
                
            self.map_rms = float(np.sqrt(np.mean(np.square(self.data_map))))
            
            default_rms_level = 5.0
            default_thresh = float(default_rms_level * self.map_rms)
            
            default_slider_val = int(1000 * (default_thresh - self.map_min) / (self.map_max - self.map_min))
            default_slider_val = max(0, min(1000, default_slider_val))
            
            self.slider_level.blockSignals(True)
            self.slider_level.setValue(default_slider_val)
            self.slider_level.setEnabled(True)
            self.slider_level.blockSignals(False)
            
            self.slider_rms.blockSignals(True)
            self.slider_rms.setValue(int(default_rms_level * 40))
            self.slider_rms.setEnabled(True)
            self.slider_rms.blockSignals(False)
            self.lbl_rms.setText(f"RMS Level: {default_rms_level:.2f}")
            
            if self.data_map.shape == self.data_locres.shape:
                mask = self.data_map > default_thresh
                valid_locres = self.data_locres[mask]
                if len(valid_locres) > 0:
                    optimal_res = round(float(np.percentile(valid_locres, 30)), 1)
                    self.entry_res.blockSignals(True)
                    self.entry_res.setText(f"{optimal_res:.1f}")
                    self.entry_res.blockSignals(False)

            self._loaded_map_path = map_path
            self._loaded_locres_path = locres_path
            self.draw_plot()

        except Exception as e:
            self.ax.clear()
            self.ax.axis('off')
            self.ax.text(0.5, 0.5, f"Error loading maps:\n{e}", color='red', 
                         fontsize=10, ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw_idle()

    def draw_plot(self):
        if not HAS_PREVIEW_LIBS or self.data_map is None or self.data_locres is None: return
        
        try:
            r = float(self.entry_res.text().strip())
            rg = float(self.entry_range.text().strip())
        except ValueError:
            return

        max_z = self.data_map.shape[0]
        current_z = max_z // 2
        slice_thickness = 5
        
        slider_frac = self.slider_level.value() / 1000.0
        current_threshold = self.map_min + slider_frac * (self.map_max - self.map_min)
        self.lbl_level.setText(f"Level: {current_threshold:.4f}")

        z_start = max(0, current_z - slice_thickness // 2)
        z_end = min(max_z, current_z + slice_thickness // 2 + 1)

        slice_map = np.mean(self.data_map[z_start:z_end, :, :], axis=0)
        
        if self.data_map.shape == self.data_locres.shape:
            slice_locres = np.mean(self.data_locres[z_start:z_end, :, :], axis=0)
        else:
            self.ax.clear()
            self.ax.axis('off')
            self.ax.text(0.5, 0.5, "Map shapes do not match!", color='red', 
                         fontsize=12, ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        self.ax.clear()
        self.ax.axis('off')
        
        center_y, center_x = slice_map.shape[0] / 2, slice_map.shape[1] / 2
        radius = min(slice_map.shape) * 0.3 
        
        clip_circle = patches.Circle((center_x, center_y), radius, transform=self.ax.transData)
        
        img_base = self.ax.imshow(slice_map, cmap='gray', origin='lower')
        img_base.set_clip_path(clip_circle)

        if self.chk_overlay.isChecked():
            cmap = mcolors.LinearSegmentedColormap.from_list("rwb", ["red", "white", "blue"])
            vmin = r - rg
            vmax = r + rg
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

            base_mask = np.where(slice_map > current_threshold, 1.0, 0.0)
            smooth_mask = scipy.ndimage.gaussian_filter(base_mask, sigma=0.8)
            alpha_mask = np.clip(smooth_mask * 1.5, 0.0, 0.9) 
            
            rgba_img = cmap(norm(slice_locres))
            rgba_img[..., 3] = alpha_mask 

            img_overlay = self.ax.imshow(rgba_img, origin='lower', interpolation='bilinear')
            img_overlay.set_clip_path(clip_circle)

        self.ax.set_xlim(center_x - radius * 1.05, center_x + radius * 1.05)
        self.ax.set_ylim(center_y - radius * 1.05, center_y + radius * 1.05)

        self.canvas.draw_idle()

    def generate_script(self):
        f_map = self.entry_map.text().strip()
        f_locres = self.entry_locres.text().strip()
        f_pdb = self.entry_pdb.text().strip()

        if not all([f_map, f_locres, f_pdb]):
            show_error_message(self, "Please select Map, LocRes, and PDB files")
            return

        f_map_base = os.path.basename(f_map)
        f_locres_base = os.path.basename(f_locres)
        f_pdb_base = os.path.basename(f_pdb)

        f_map_name = os.path.splitext(f_map_base)[0]
        default_cxc_name = f"LocalRes_{f_map_name}.cxc"

        try:
            res_val = float(self.entry_res.text().strip())
            range_val = float(self.entry_range.text().strip())
        except ValueError:
            show_error_message(self, "Resolution and Range must be numbers")
            return

        low = res_val - range_val
        mid = res_val
        high = res_val + range_val
        rms_export_val = self.slider_rms.value() / 40.0 if self.slider_rms.isEnabled() else 5.0

        # EXPORT CHIMERAX CXC
        if self.chk_cxc.isChecked():
            start_dir = os.path.dirname(f_map) if os.path.dirname(f_map) else os.getcwd()
            cxc_path, _ = QFileDialog.getSaveFileName(
                self,
                "Export ChimeraX Script",
                os.path.join(start_dir, default_cxc_name),
                "ChimeraX Script (*.cxc)"
            )
            if not cxc_path:
                return

            cxc_content = f"""open "{f_map_base}" id #1
open "{f_locres_base}" id #2
open "{f_pdb_base}" id #3
color sample #1 map #2 palette {low:.1f},#ff0000:{mid:.1f},#ffffff:{high:.1f},#0000ff 
surface dust #1
set bgColor #ffffff00
hide #2 models 
graphics silhouettes true
graphics silhouette width 20
ui tool show "Surface Color"
key red-white-blue :{low:.1f}Å :{mid:.1f}Å :{high:.1f}Å
key fontSize 14
key pos 0.8000,0.06000
key size 0.15000,0.03000
key borderWidth 3.0
surface zone #1 near #3
hide #3 models 
hide #2 models
view orient
zoom pixelSize 0.18
scalebar 10 xpos 0.01 ypos 0.01
2dlabels create scalebar_legend text "10 Å" xpos 0.01 ypos 0.02 size 20
volume #1 step 1
volume #1 rmsLevel {rms_export_val:.2f}
lighting soft
#save LocalRes_{f_map_name}.png width 5000 height 3849 supersample 4 transparentBackground true"""

            try:
                with open(cxc_path, "w", encoding='utf-8') as f:
                    f.write(cxc_content.strip())
                QMessageBox.information(self, "Success", f"ChimeraX script exported to:\n{cxc_path}\n\nAll you need to do is drag this .cxc file into ChimeraX.")
            except Exception as e:
                show_error_message(self, f"Failed to save CXC file: {e}")
            return

        # RUN LEGACY CHIMERA
        output_path = self.entry_out.text().strip()
        
        chimera_script_content = f"""# -*- coding: utf-8 -*-
import chimera
from chimera import runCommand as rc

rc("open #1 {f_map}")
rc("open #2 {f_locres}")
rc("open #3 {f_pdb}")

cmd_color = "scolor #1 volume #2 cmap {low:.1f},red:{mid:.1f},white:{high:.1f},blue"
rc(cmd_color)

rc("set bgColor white")
rc("volume #2 hide")
rc("sop hideDust #1 size 10")
rc("set silhouette")
rc("set silhouetteWidth 10")
rc("sop zone #1 #3 2")

labeltext1 = u'colorkey 0.7,0.08 0.9,0.12 fontSize 16 "{low:.1f}\u00C5" red "{mid:.1f}\u00C5" white "{high:.1f}\u00C5" blue'
rc(labeltext1.encode('utf-8'))

rc("close #3")
rc("volume #1 step 1")
rc("volume #1 rmsLevel {rms_export_val:.2f}")
rc("focus #1")
rc("scale 0.8")
rc("copy file {output_path} width 1772 height 1149 supersample 4")
rc("close session")
rc("stop")
"""
        script_filename = "chimera_localres_script.py"
        try:
            with open(script_filename, "w") as f:
                f.write(chimera_script_content)
            
            full_command = f"module load {CHIMERA_VER} && chimera --script \"{script_filename}\" && rm -f {script_filename}"
            QMessageBox.information(self, "Status", "Command Sent\n\nRunning Chimera script locally, please wait...")
            write_to_log(full_command)
            subprocess.Popen(
                full_command, shell=True, executable='/bin/bash',
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True 
            )
        except Exception as e:
            show_error_message(self, f"Execution failed: {e}")

# --- GLOBAL DARK THEME ---
def set_dark_theme(app):
    app.setStyleSheet("""
        QMainWindow, QDialog, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Arial; font-size: 10pt; }
        QPushButton { background-color: #3e3e42; color: #d4d4d4; border: 1px solid #3e3e42; padding: 5px 15px; border-radius: 4px; }
        QPushButton:hover { background-color: #4e4e52; border: 1px solid #98c379; }
        QPushButton:pressed, QPushButton:checked { background-color: #98c379; color: #1e1e1e; border: 1px solid #98c379; }
        QLineEdit { background-color: #3c3c3c; border: 1px solid #3c3c3c; color: #cccccc; padding: 4px; selection-background-color: #98c379; selection-color: #1e1e1e; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; border-radius: 2px; background-color: #1e1e1e; }
        QCheckBox::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        QSplitter::handle { background-color: #3e3e42; width: 2px; }
        QSlider:vertical { min-width: 20px; }
        QSlider::groove:vertical { background: #3c3c3c; width: 6px; border-radius: 3px; }
        QSlider::handle:vertical { background: #98c379; height: 14px; width: 14px; margin: 0 -4px; border-radius: 7px; }
        QSlider::handle:vertical:hover { background: #b5e890; }
        QSlider::add-page:vertical, QSlider::sub-page:vertical { background: transparent; }
        QSlider:horizontal { min-height: 20px; }
        QSlider::groove:horizontal { background: #3c3c3c; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #98c379; width: 14px; height: 14px; margin: -4px 0; border-radius: 7px; }
        QSlider::handle:horizontal:hover { background: #b5e890; }
        QSlider::add-page:horizontal, QSlider::sub-page:horizontal { background: transparent; }
    """)

# === MAIN LAUNCHER ===
if __name__ == "__main__":
    qt_app = QApplication(sys.argv)
    set_dark_theme(qt_app)
    window = LocalResWindow()
    window.show()
    sys.exit(qt_app.exec())