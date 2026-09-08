import os
import sys
import re
import subprocess
import threading
import shutil
import json
import queue
import csv
import math
import numpy as np

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel, QLineEdit, 
                             QPushButton, QComboBox, QMessageBox, QGridLayout, 
                             QVBoxLayout, QHBoxLayout, QGroupBox, QFileDialog,
                             QSplitter, QSlider, QCheckBox, QTextEdit, QSpinBox,
                             QColorDialog, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QMetaObject, Q_ARG, QThread

import matplotlib
matplotlib.use('qtagg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# --- Path and Configuration Setup ---
script_dir = os.path.dirname(__file__)
xcop_path = os.path.join(os.path.dirname(script_dir), "xcop")

RELION_VER = "relion"
CHIMERA_VER = "chimera"
try:
    if os.path.exists(xcop_path):
        with open(xcop_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'^RELION_VER\s*=\s*["\']([^"\']+)["\']', content, flags=re.MULTILINE)
            if match: RELION_VER = match.group(1)
            match_chi = re.search(r'^CHIMERA_VER\s*=\s*["\']([^"\']+)["\']', content, flags=re.MULTILINE)
            if match_chi: CHIMERA_VER = match_chi.group(1)
except Exception as e:
    print(f"Error reading config from xcop: {e}")

# --- Global State for Manual Code Compatibility ---
class GlobalState:
    def __init__(self):
        self.current_model_folder = ""
        self.output_text = None  # Will hold a QTextEdit for logs
        self.current_theme = {'select_bg': '#98c379', 'fg': '#d4d4d4', 'entry_bg': '#3c3c3c'}
        self.theme_aware_optionmenus = []
        self.current_fsc_files = []
        self.current_fsc_model = ""

global_app = GlobalState()
toplevel_windows = []


# --- Missing Manual Helper Functions ---
def center_window(win, width, height, name=""):
    win.resize(width, height)
    qr = win.frameGeometry()
    cp = win.screen().availableGeometry().center()
    qr.moveCenter(cp)
    win.move(qr.topLeft())

def show_error_message(msg):
    QMessageBox.critical(None, "Error", str(msg))

def write_to_log(msg):
    if global_app.output_text:
        global_app.output_text.append(f"> {msg}")
    else:
        print(msg)

def close_toplevel_windows():
    for w in toplevel_windows:
        try: w.close()
        except: pass
    toplevel_windows.clear()



# --- Core FSC Functions ---
def safe_read_fsc(filepath):
    x_vals, y_vals = [], []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(('_', 'data_', 'loop_', '#')): continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x_vals.append(float(parts[1]))
                    y_vals.append(float(parts[3]))
                except ValueError: continue
    return np.column_stack((x_vals, y_vals))

def update_fsc_log_file(model_folder, section_color, new_data):
    try:
        log_path = os.path.join(os.getcwd(), "FSC", model_folder, "xcop_log.txt")
        other_lines = []
        data = { "Red": {}, "Blue": {} }
        
        if os.path.exists(log_path):
            current_section = None
            with open(log_path, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if stripped == "Red:": current_section = "Red"
                    elif stripped == "Blue:": current_section = "Blue"
                    elif stripped in ["Symmetry Expansion:", "Black:", "Yellow:"]:
                        current_section = stripped.replace(":", "")
                        other_lines.append(line)
                    elif current_section in ["Red", "Blue"] and stripped.startswith("- "):
                        parts = stripped[2:].split(":", 1)
                        if len(parts) == 2:
                            data[current_section][parts[0].strip()] = parts[1].strip()
                    elif current_section not in ["Red", "Blue"]:
                        other_lines.append(line)

        if section_color in data: data[section_color].update(new_data)
            
        with open(log_path, "w") as f:
            for line in other_lines:
                f.write(line)
                if not line.endswith('\n'): f.write('\n')
            for section in ["Red", "Blue"]:
                f.write(f"{section}:\n")
                keys_order = ["Model", "Resolution", "Initial Threshold", "Extend Initial Mask", "Soft Edge Width", "Low Pass Filter"]
                for k in keys_order:
                    val = data[section].get(k, "N/A")
                    f.write(f"- {k}: {val}\n")
                f.write("\n")
    except Exception as e:
        print(f"Failed to update log: {e}")

class ComprehensivePlotWindow(QWidget):
    def __init__(self, comp_dir, log_file):
        super().__init__()
        self.setWindowTitle("Simplified FSC Visualizer")
        self.resize(950, 600)
        self.setStyleSheet(QApplication.instance().styleSheet())
        
        self.comp_dir = comp_dir
        with open(log_file, 'r') as f:
            self.params_map = json.load(f)
            
        job_scores = []
        target_x = np.arange(0.1, 0.51, 0.05)
        for jid in self.params_map.keys():
            path = f"{self.comp_dir}/{jid}.txt"
            if not os.path.exists(path): continue
            try:
                data = safe_read_fsc(path)
                if data.ndim == 2 and data.shape[0] > 1:
                    x, y = data[:, 0], data[:, 1]
                    sort_idx = np.argsort(x)
                    x_sorted, y_sorted = x[sort_idx], y[sort_idx]
                    target_y = np.interp(target_x, x_sorted, y_sorted)
                    score = np.mean(target_y)
                else: score = 0.0
                job_scores.append((score, jid))
            except: job_scores.append((-1.0, jid))
        
        job_scores.sort(key=lambda x: x[0], reverse=True)
        self.job_ids = [j[1] for j in job_scores]
        
        main_layout = QHBoxLayout(self)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        main_layout.addWidget(self.canvas, 1) 
        
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(5, 5, 5, 5)
        right_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        
        self.slider = QSlider(Qt.Orientation.Vertical)
        self.slider.setRange(0, max(0, len(self.job_ids) - 1))
        self.slider.setInvertedAppearance(True) 
        self.slider.valueChanged.connect(self.on_slider_change)
        self.slider.setFixedWidth(40)
        self.slider.setStyleSheet("""
            QSlider::groove:vertical { background: #555; width: 6px; border-radius: 3px; }
            QSlider::handle:vertical { background: #98c379; height: 12px; margin: 0 -4px; border-radius: 6px; }
            QSlider::add-page:vertical { background: #555; }
            QSlider::sub-page:vertical { background: #555; }
        """)
        right_layout.addWidget(self.slider, 1, Qt.AlignmentFlag.AlignHCenter) 
        
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)
        btn_up = QPushButton("↑")
        btn_up.setFixedWidth(60)
        btn_up.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        btn_layout.addWidget(btn_up)
        btn_down = QPushButton("↓")
        btn_down.setFixedWidth(60)
        btn_down.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        btn_layout.addWidget(btn_down)
        right_layout.addLayout(btn_layout)
        
        self.info_label = QLabel("Hover or Slide\nfor details")
        self.info_label.setFixedWidth(160)
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("font-size: 11px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        right_layout.addWidget(self.info_label)
        
        self.btn_promote = QPushButton("Promote\nCurrent Curve")
        self.btn_promote.setToolTip("Automatically re-run the standard process in the main folder")
        self.btn_promote.setStyleSheet("""
            QPushButton { background-color: #98c379; color: #1e1e1e; font-weight: bold; padding: 5px; }
            QToolTip { background-color: #2C2C2C; color: white; border: 1px solid #777; font-weight: normal; }
        """)
        self.btn_promote.clicked.connect(self.promote_current_curve)
        right_layout.addWidget(self.btn_promote)
        main_layout.addWidget(right_panel)
        
        self.lines = {} 
        self.current_highlight = None
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(50) 
        self.debounce_timer.timeout.connect(self.perform_highlight)
        
        self.plot_all_data()
        self.canvas.mpl_connect("motion_notify_event", self.on_hover)
        if self.job_ids:
            self.slider.setValue(0)
            self.highlight_line(self.job_ids[0])

    def promote_current_curve(self):
        idx = self.slider.value()
        if not (0 <= idx < len(self.job_ids)): return
        job_id = self.job_ids[idx]
        raw_params = self.params_map.get(job_id, "")
        
        is_red = "red" in job_id.lower()
        is_blue = "blue" in job_id.lower()
        if not (is_red or is_blue): return QMessageBox.warning(self, "Error", "Could not determine color")

        parent_dir = os.path.dirname(self.comp_dir)
        required_molmap = os.path.join(parent_dir, "molmap_post.mrc") if is_red else os.path.join(parent_dir, "molmap_halfmap1.mrc")
        if not os.path.exists(required_molmap): return QMessageBox.warning(self, "Missing", "Molmap not detected")

        params = {}
        for p in raw_params.split():
            if ":" in p:
                k, v = p.split(":", 1)
                params[k] = v
        
        ini, ext, soft, low = params.get("Ini"), params.get("Ext"), params.get("Soft"), params.get("Low")
        color_str = "Red" if is_red else "Blue"
        msg = f"Promote {job_id}?\n\nThis re-runs the {color_str} mask process in the main folder."
        if is_blue: msg += "\nAlso runs Yellow curve."
        if QMessageBox.question(self, "Promote", msg, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No) != QMessageBox.StandardButton.Yes: return

        model_name = os.path.basename(parent_dir)
        cmds = []
        p_dir = parent_dir.replace("\\", "/") # Ensures bash compatibility on all OS
        if is_red:
            cmds.append(f"module load {RELION_VER} && relion_mask_create --i {p_dir}/molmap_post.mrc --o {p_dir}/molmap_post_masked.mrc --ini_threshold {ini} --extend_inimask {ext} --width_soft_edge {soft} --lowpass {low}")
            cmds.append(f"relion_image_handler --i {p_dir}/hh_postprocess.mrc --multiply {p_dir}/molmap_post_masked.mrc --o {p_dir}/postprocess_multiplied.mrc")
            cmds.append(f"relion_image_handler --i {p_dir}/postprocess_multiplied.mrc --fsc {p_dir}/molmap_post.mrc > {p_dir}/red.txt")
        elif is_blue:
            cmds.append(f"module load {RELION_VER} && relion_mask_create --i {p_dir}/molmap_halfmap1.mrc --o {p_dir}/molmap_halfmap1_masked.mrc --ini_threshold {ini} --extend_inimask {ext} --width_soft_edge {soft} --lowpass {low}")
            cmds.append(f"relion_image_handler --i {p_dir}/symEx_halfmap1.mrc --multiply {p_dir}/molmap_halfmap1_masked.mrc --o {p_dir}/halfmap1_multiplied.mrc")
            cmds.append(f"relion_image_handler --i {p_dir}/halfmap1_multiplied.mrc --fsc {p_dir}/molmap_halfmap1.mrc > {p_dir}/blue.txt")
            cmds.append(f"relion_image_handler --i {p_dir}/symEx_halfmap2.mrc --multiply {p_dir}/molmap_halfmap1_masked.mrc --o {p_dir}/halfmap2_multiplied.mrc")
            cmds.append(f"relion_image_handler --i {p_dir}/halfmap2_multiplied.mrc --fsc {p_dir}/molmap_halfmap1.mrc > {p_dir}/yellow.txt")

        try:
            full_cmd = " && ".join(cmds)
            write_to_log(f"Promoting {job_id} ({color_str}):\n{full_cmd}")
            subprocess.call(full_cmd, shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            update_fsc_log_file(model_name, color_str, {"Initial Threshold": ini, "Extend Initial Mask": ext, "Soft Edge Width": soft, "Low Pass Filter": low})
            QMessageBox.information(self, "Success", f"Promoted {job_id}\nProcesses re-run.")
        except Exception as e: QMessageBox.critical(self, "Error", str(e))

    def plot_all_data(self):
        self.ax.clear()
        for job_id in self.job_ids:
            path = f"{self.comp_dir}/{job_id}.txt"
            if not os.path.exists(path): continue
            try:
                data = safe_read_fsc(path)
                line, = self.ax.plot(data[:, 0], data[:, 1], color="grey", alpha=0.1, linewidth=1)
                self.lines[job_id] = line
            except: pass
        self.ax.set_ylim(0, 1.05); self.ax.set_xlim(0, 0.6)
        self.ax.set_xlabel("Spatial Frequency (1/A)"); self.ax.set_ylabel("FSC")
        self.canvas.draw()

    def on_slider_change(self): self.debounce_timer.start()

    def perform_highlight(self):
        idx = self.slider.value()
        if 0 <= idx < len(self.job_ids): self.highlight_line(self.job_ids[idx])

    def on_hover(self, event):
        if event.inaxes != self.ax: return
        best_line = None
        for job_id, line in self.lines.items():
            contains, _ = line.contains(event)
            if contains:
                best_line = job_id
                break 
        if best_line:
            try:
                self.slider.blockSignals(True)
                self.slider.setValue(self.job_ids.index(best_line))
                self.slider.blockSignals(False)
            except: pass
            self.highlight_line(best_line)

    def highlight_line(self, job_id):
        if job_id == self.current_highlight: return
        if self.current_highlight and self.current_highlight in self.lines:
            old = self.lines[self.current_highlight]
            old.set_alpha(0.1); old.set_color("grey"); old.set_linewidth(1); old.set_zorder(1)
        if job_id in self.lines:
            line = self.lines[job_id]
            line.set_alpha(1.0)
            col = "red" if "red" in job_id.lower() else "blue"
            line.set_color(col); line.set_linewidth(2); line.set_zorder(10)
            self.current_highlight = job_id
            disp = self.params_map.get(job_id, "").replace("Ini:", "Ini Threshold: ").replace(" Ext:", "\nExtend Mask: ").replace(" Soft:", "\nSoft Edge: ").replace(" Low:", "\nLow-pass Filter: ")
            self.info_label.setText(f"{job_id}.txt\n{disp}")
            self.canvas.draw_idle()

# --- Manual Utilities ---
def create_model_name_dropdown(layout, row, col, default_value=""):
    model_combo = QComboBox()
    model_combo.setMaxVisibleItems(15)
    model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    model_combo.setMinimumContentsLength(12)
    model_combo.view().setMinimumWidth(400)
    def refresh_model_options():
        folder_path = "FSC" 
        options = []
        try:
            full_path = os.path.abspath(folder_path)
            if os.path.isdir(full_path):
                options = sorted([d for d in os.listdir(full_path) if os.path.isdir(os.path.join(full_path, d))])
        except Exception as e: pass
        current = model_combo.currentText()
        model_combo.blockSignals(True)
        model_combo.clear()
        model_combo.addItems(options)
        if current in options: model_combo.setCurrentText(current)
        elif options: model_combo.setCurrentText(options[0])
        elif default_value:
             model_combo.addItem(default_value)
             model_combo.setCurrentText(default_value)
        model_combo.blockSignals(False)
    refresh_model_options()
    layout.addWidget(model_combo, row, col)
    global_app.theme_aware_optionmenus.append(model_combo)
    return model_combo, refresh_model_options



# --- Plotting Visualizer with Draggable Math ---
def generate_fsc_curves(auto_files=None, auto_model=None, auto_export_dir=None):
    def _manual_input_window():
        man_win = QWidget(); man_win.setWindowTitle("Select FSC Files Manually"); center_window(man_win, 500, 250, "fsc_manual_select"); toplevel_windows.append(man_win); man_win.setStyleSheet(QApplication.instance().styleSheet())
        layout = QGridLayout(man_win)
        labels = ["Black FSC:", "Red FSC:", "Blue FSC:", "Yellow FSC:"]
        entries = []
        for i, lbl in enumerate(labels):
            layout.addWidget(QLabel(lbl), i, 0); entry = QLineEdit(); layout.addWidget(entry, i, 1); entries.append(entry)
            btn = QPushButton("Browse")
            def browse(checked=False, e=entry):
                f, _ = QFileDialog.getOpenFileName(man_win, "Select FSC txt", os.getcwd(), "Text Files (*.txt);;All Files (*)")
                if f:
                    e.setText(f)
                    folder = os.path.dirname(f)
                    for j, color in enumerate(["black.txt", "red.txt", "blue.txt", "yellow.txt"]):
                        if entries[j] != e and not entries[j].text():
                            p = os.path.join(folder, color)
                            if os.path.exists(p): entries[j].setText(p)
            btn.clicked.connect(browse); layout.addWidget(btn, i, 2)
        def submit_manual():
            files = [e.text().strip() for e in entries]
            if not any(files): return show_error_message("Please select at least one file.")
            global_app.current_fsc_files = files; global_app.current_fsc_model = "Manual_Selection"; man_win.close(); _launch_plot_window()
        submit = QPushButton("Plot"); submit.clicked.connect(submit_manual); layout.addWidget(submit, 4, 0, 1, 3); man_win.show()

    def _select_and_plot():
        sel_win = QWidget(); sel_win.setWindowTitle("Select FSC Model Folder"); center_window(sel_win, 350, 160, "fsc_model_select_plot"); toplevel_windows.append(sel_win); sel_win.setStyleSheet(QApplication.instance().styleSheet())
        layout = QGridLayout(sel_win); layout.addWidget(QLabel("Model Name:"), 0, 0)
        model_combo, _ = create_model_name_dropdown(layout, 0, 1)
        def proceed():
            mdl = model_combo.currentText().strip()
            if not mdl: return show_error_message("Please select a Model Name.")
            global_app.current_fsc_model = mdl
            if global_app.current_fsc_files: global_app.current_fsc_files.clear()
            sel_win.close(); _launch_plot_window()
        btn = QPushButton("OK (Auto-detect from Folder)"); btn.clicked.connect(proceed); layout.addWidget(btn, 1, 0, 1, 2)
        btn_manual = QPushButton("or Browse Files Manually..."); btn_manual.clicked.connect(lambda: [sel_win.close(), _manual_input_window()]); layout.addWidget(btn_manual, 2, 0, 1, 2)
        sel_win.show()

    def _launch_plot_window():
        from PyQt6.QtGui import QCursor

        class TooltipSlider(QSlider):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.setTickPosition(QSlider.TickPosition.TicksBelow)
                self.setTickInterval(10)
                self.setSingleStep(10)
                self.setPageStep(10)
                
                # Custom Frameless Popup Label
                self._popup = QLabel(self, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
                self._popup.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
                self._popup.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._popup.setStyleSheet("""
                    QLabel {
                        background-color: rgba(40, 40, 40, 230);
                        color: #ffffff;
                        border: 1px solid #555555;
                        border-radius: 8px;
                        padding: 4px 10px;
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        font-size: 12px;
                        font-weight: bold;
                    }
                """)
                self._popup.hide()
                
                self.sliderPressed.connect(self._show_popup)
                self.sliderReleased.connect(self._popup.hide)
                self.valueChanged.connect(self._update_popup)

            def _show_popup(self):
                self._update_popup(self.value())
                self._popup.show()

            def _update_popup(self, value):
                if not self._popup.isVisible() and not self.isSliderDown():
                    return
                self._popup.setText(f"{value/10.0:g}")
                self._popup.adjustSize()
                
                # Center the bubble horizontally over the cursor and push it up slightly
                pos = QCursor.pos()
                x = pos.x() - self._popup.width() // 2
                y = pos.y() - self._popup.height() - 15
                self._popup.move(x, y)

            def mousePressEvent(self, event):
                super().mousePressEvent(event)
                if event.button() == Qt.MouseButton.LeftButton:
                    self._show_popup()
                    
            def mouseReleaseEvent(self, event):
                super().mouseReleaseEvent(event)
                self._popup.hide()

        fsc_data_cache = {}; coord_user_positions = []; arrow_pos_dict = {}
        if auto_files:
            files = auto_files; model_folder = auto_model or "Automated_Plot"
        elif global_app.current_fsc_files: files = global_app.current_fsc_files; model_folder = "Manual_Selection"
        else: model_folder = global_app.current_fsc_model; files = [f"FSC/{model_folder}/{c}.txt" for c in ("black", "red", "blue", "yellow")]
        
        plot_win = QWidget(); plot_win.setWindowTitle(f"FSC Curves: {model_folder}")
        center_window(plot_win, 1100, 780, "fsc_plot_window"); toplevel_windows.append(plot_win)

        def label_arrow_offset(arrowhead_x, arrowhead_y, label_cx, label_cy):
            MAX_X, MAX_Y = 0.025, 0.035
            dx, dy = arrowhead_x - label_cx, arrowhead_y - label_cy
            denom = math.sqrt(dx*dx/(MAX_X*MAX_X) + dy*dy/(MAX_Y*MAX_Y))
            if denom == 0: return label_cx, label_cy
            return label_cx + dx/denom, label_cy + dy/denom

        class DraggableElement:
            def __init__(self, element, canvas):
                self.element, self.canvas, self.press = element, canvas, None
                self.cids = [canvas.mpl_connect('button_press_event', self.on_press), canvas.mpl_connect('button_release_event', self.on_release), canvas.mpl_connect('motion_notify_event', self.on_motion)]
            def on_press(self, event):
                if event.inaxes != self.element.axes: return
                contains, _ = self.element.contains(event)
                if contains: self.press = (self.element.get_position(), event.xdata, event.ydata)
            def on_motion(self, event):
                if self.press is None or event.inaxes != self.element.axes: return
                (x0, y0), xp, yp = self.press
                self.update_position(x0 + event.xdata - xp, y0 + event.ydata - yp)
                self.canvas.draw_idle()
            def on_release(self, event):
                self.press = None; self.save_position(); self.canvas.draw_idle()
            def update_position(self, x, y): self.element.set_position((x, y))
            def save_position(self): pass
            def disconnect(self):
                for cid in self.cids: self.canvas.mpl_disconnect(cid)

        class DraggableCoord(DraggableElement):
            def __init__(self, text_obj, canvas, index, storage_list):
                super().__init__(text_obj, canvas); self.index, self.storage_list = index, storage_list
            def save_position(self):
                x, y = self.element.get_position()
                if self.index < len(self.storage_list): self.storage_list[self.index]['x'], self.storage_list[self.index]['y'] = x, y

        class DraggableArrow(DraggableElement):
            def __init__(self, text_obj, arrow_obj, canvas, key, storage_dict):
                super().__init__(text_obj, canvas); self.arrow, self.key, self.storage_dict = arrow_obj, key, storage_dict
            def update_position(self, x, y):
                self.element.set_position((x, y)); self.arrow.xytext = (x, y)
            def save_position(self):
                x, y = self.element.get_position(); self.storage_dict[self.key]['x'], self.storage_dict[self.key]['y'] = x, y

        chk_col, fg_col, ent_col = global_app.current_theme['select_bg'], global_app.current_theme['fg'], global_app.current_theme['entry_bg']
        plot_win.setStyleSheet(QApplication.instance().styleSheet() + f" QCheckBox {{ spacing: 5px; color: {fg_col}; }} QCheckBox::indicator {{ width: 18px; height: 18px; border: 2px solid #555; border-radius: 3px; background: {ent_col}; }} QCheckBox::indicator:checked {{ background-color: {chk_col}; border: 2px solid {chk_col}; }}")

        main_layout = QVBoxLayout(plot_win)
        splitter = QSplitter(Qt.Orientation.Horizontal); main_layout.addWidget(splitter)
        canvas_widget = QWidget(); canvas_layout = QVBoxLayout(canvas_widget)
        fig = Figure(figsize=(6, 5), dpi=100); ax = fig.add_subplot(111); canvas = FigureCanvas(fig); canvas_layout.addWidget(canvas); splitter.addWidget(canvas_widget)
        
        scroll = QScrollArea(); scroll.setWidgetResizable(True); right_widget = QWidget(); right_layout = QGridLayout(right_widget)
        ui_refs = {}; line_colors = {"black": "#212121", "red": "#D32F2F", "blue": "#2188A3", "yellow": "#F5D017"}; file_keys = ["black", "red", "blue", "yellow"]

        for i, filename in enumerate(files):
            if not filename or not os.path.exists(filename): continue
            key = file_keys[i]
            right_layout.addWidget(QLabel(f"{key.capitalize()} Legend:"), i*3, 0)
            default_legends = {
                "black": "FSC Map", 
                "red": "FSC Full", 
                "blue": "FSC Work", 
                "yellow": "FSC Free"
            }
            entry = QLineEdit(default_legends.get(key, key.capitalize())); entry.returnPressed.connect(lambda: update_plot()); right_layout.addWidget(entry, i*3, 1); ui_refs[f"leg_{key}"] = entry
            color_btn = QPushButton("Color")
            def pick_color(checked=False, k=key):
                col = QColorDialog.getColor()
                if col.isValid(): line_colors[k] = col.name(); update_plot()
            color_btn.clicked.connect(pick_color); right_layout.addWidget(color_btn, i*3, 2)
            cb1 = QCheckBox(f"{key.capitalize()} vs Line 1"); cb2 = QCheckBox(f"{key.capitalize()} vs Line 2")
            if key == "black": cb2.setChecked(True)
            if key == "red": cb1.setChecked(True)
            cb1.toggled.connect(lambda: update_plot()); cb2.toggled.connect(lambda: update_plot())
            right_layout.addWidget(cb1, i*3+1, 0, 1, 3); right_layout.addWidget(cb2, i*3+2, 0, 1, 3)
            ui_refs[f"{key}_hline1"] = cb1; ui_refs[f"{key}_hline2"] = cb2

        ro = 12
        def add_setting(label, default, key, row):
            right_layout.addWidget(QLabel(label), row, 0); inp = QLineEdit(str(default)); inp.returnPressed.connect(lambda: update_plot()); right_layout.addWidget(inp, row, 1); ui_refs[key] = inp
        add_setting("Legend Font:", "12", "leg_font", ro)
        ui_refs["show_xaxis"] = QCheckBox("Show X-Axis Line"); right_layout.addWidget(ui_refs["show_xaxis"], ro+1, 0, 1, 2)
        ui_refs["show_yaxis"] = QCheckBox("Show Y-Axis Line"); right_layout.addWidget(ui_refs["show_yaxis"], ro+2, 0, 1, 2)
        ui_refs["show_xaxis"].toggled.connect(lambda: update_plot()); ui_refs["show_yaxis"].toggled.connect(lambda: update_plot())
        
        right_layout.addWidget(QLabel("Fig Width:"), ro+3, 0); sl_w = TooltipSlider(Qt.Orientation.Horizontal); sl_w.setRange(10, 150); sl_w.setValue(80); sl_w.valueChanged.connect(lambda v: sl_w.setValue((v + 5) // 10 * 10) if v % 10 != 0 else update_plot()); right_layout.addWidget(sl_w, ro+3, 1, 1, 2); ui_refs["fig_width"] = sl_w
        right_layout.addWidget(QLabel("Fig Height:"), ro+4, 0); sl_h = TooltipSlider(Qt.Orientation.Horizontal); sl_h.setRange(10, 120); sl_h.setValue(60); sl_h.valueChanged.connect(lambda v: sl_h.setValue((v + 5) // 10 * 10) if v % 10 != 0 else update_plot()); right_layout.addWidget(sl_h, ro+4, 1, 1, 2); ui_refs["fig_height"] = sl_h
        scroll.setWidget(right_widget); splitter.addWidget(scroll); scroll.setMinimumWidth(300); scroll.setMinimumHeight(600); splitter.setStretchFactor(0, 4); splitter.setStretchFactor(1, 1)
        
        bottom_widget = QWidget(); bot_layout = QGridLayout(bottom_widget); main_layout.addWidget(bottom_widget)
        bot_entries = [("Plot Title:", "title", ""), ("X-Axis Title:", "xtitle", "Spatial Frequency (1/Å)"), ("Y-Axis Title:", "ytitle", "Fourier Shell Correlation"), ("Title Font:", "title_font", "14"), ("Line Width:", "lw", "1.5"), ("Tick Font:", "tick_font", "10"), ("Spine Width:", "spine_w", "2"), ("Tick Len:", "tick_len", "8"), ("HLine 1:", "hline1_y", "0.5"), ("HLine 2:", "hline2_y", "0.143"), ("Coord Font:", "coord_font", "12"), ("Export Scale:", "exp_scale", "2")]
        for i, (lbl, k, v) in enumerate(bot_entries):
            r, c = i % 4, (i // 4) * 2
            bot_layout.addWidget(QLabel(lbl), r, c); le = QLineEdit(v); le.returnPressed.connect(lambda: update_plot()); bot_layout.addWidget(le, r, c+1); ui_refs[k] = le
            
        bot_layout.addWidget(QLabel("Res Digits:"), 0, 6)
        spin_res_digits = QSpinBox(); spin_res_digits.setRange(0, 5); spin_res_digits.setValue(2); bot_layout.addWidget(spin_res_digits, 0, 7); ui_refs["res_digits"] = spin_res_digits
        checks = [("Show All", "show_all", True), ("Show Legend", "show_leg", False), ("Calc Res (Title)", "title_res", False), ("Show Coords", "show_coords", False), ("Convert Res (Å)", "conv_res", False), ("Show Arrow", "show_arrow", True), ("Show Intersection", "show_int", False), ("Show Frame", "show_frame", False), ("HLine 1 Visible", "show_h1", True), ("HLine 2 Visible", "show_h2", True), ("X-Axis @ 0", "x_at_0", True), ("Y-Axis @ 0", "y_at_0", True)]
        for i, (lbl, k, v) in enumerate(checks):
            cb = QCheckBox(lbl); cb.setChecked(v); cb.toggled.connect(lambda: update_plot()); bot_layout.addWidget(cb, i % 3, 8 + (i//3)); ui_refs[k] = cb
        upd_btn = QPushButton("Update Plot"); upd_btn.clicked.connect(lambda: update_plot(True)); bot_layout.addWidget(upd_btn, 4, 0, 1, 2)
        
        def export(fmt):
            try:
                scale = float(ui_refs["exp_scale"].text())
                w, h = (ui_refs["fig_width"].value()/10.0)*scale, (ui_refs["fig_height"].value()/10.0)*scale
                exp_fig = Figure(figsize=(w, h), dpi=300); exp_ax = exp_fig.add_subplot(111); _plot_logic(exp_ax, True)
                
                path = ""
                if auto_export_dir:
                    path, _ = QFileDialog.getSaveFileName(plot_win, f"Save {fmt.upper()}", os.path.join(auto_export_dir, f"FSC_curve.{fmt}"), f"{fmt.upper()} Files (*.{fmt});;All Files (*)")
                    if not path: return
                elif auto_files or global_app.current_fsc_files:
                    path, _ = QFileDialog.getSaveFileName(plot_win, f"Save {fmt.upper()}", f"FSC_curve.{fmt}", f"{fmt.upper()} Files (*.{fmt});;All Files (*)")
                    if not path: return
                else:
                    path = f"FSC/{model_folder}/FSC_curve.{fmt}"
                    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

                if fmt == 'svg': exp_fig.savefig(path, format='svg', bbox_inches='tight', transparent=True)
                else: exp_fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300, facecolor='white')
                QMessageBox.information(plot_win, "Export", f"Saved {fmt.upper()} to {path}")
            except Exception as e: show_error_message(f"Export failed: {e}")
            
        btn_png = QPushButton("PNG"); btn_png.clicked.connect(lambda: export('png'))
        btn_svg = QPushButton("SVG"); btn_svg.clicked.connect(lambda: export('svg'))
        btn_pdf = QPushButton("PDF"); btn_pdf.clicked.connect(lambda: export('pdf'))
        btn_ps = QPushButton("PS"); btn_ps.clicked.connect(lambda: export('ps'))
        bot_layout.addWidget(btn_png, 4, 8); bot_layout.addWidget(btn_svg, 4, 9); bot_layout.addWidget(btn_pdf, 4, 10); bot_layout.addWidget(btn_ps, 4, 11)

        def _plot_logic(ax_target, is_export=False):
            ax_target.clear()
            if not is_export and hasattr(ax_target, 'draggables'):
                for d in ax_target.draggables: d.disconnect()
            ax_target.draggables = []

            lw = float(ui_refs["lw"].text())
            res_digits = ui_refs["res_digits"].value()
            target_y1 = float(ui_refs["hline1_y"].text())
            target_y2 = float(ui_refs["hline2_y"].text())
            is_res_mode = ui_refs["conv_res"].isChecked()
            intercept_points = []; fsc_res_val = None

            for i, filename in enumerate(files):
                if not filename or not os.path.exists(filename): continue 
                key = file_keys[i]; cache_id = filename 
                
                if cache_id not in fsc_data_cache:
                    try: fsc_data_cache[cache_id] = safe_read_fsc(filename)
                    except: continue
                data = fsc_data_cache[cache_id]
                x, y = data[:, 0], data[:, 1]
                
                label = ui_refs[f"leg_{key}"].text(); color = line_colors[key]
                ax_target.plot(x, y, color=color, linewidth=lw, label=label)
                
                check_h1 = ui_refs[f"{key}_hline1"].isChecked()
                check_h2 = ui_refs[f"{key}_hline2"].isChecked()
                found_h1, found_h2 = False, False

                if check_h1 or check_h2:
                    for j in range(len(y)-1):
                        if check_h1 and not found_h1 and ((y[j] <= target_y1 <= y[j+1]) or (y[j+1] <= target_y1 <= y[j])):
                            denom = y[j+1] - y[j]
                            x_int = x[j] + (x[j+1]-x[j]) * (target_y1 - y[j]) / denom if denom != 0 else x[j]
                            intercept_points.append((x_int, target_y1, color, key, 'hline1'))
                            if ui_refs["show_int"].isChecked(): ax_target.vlines(x_int, -0.1, target_y1, colors=color, linestyles='dotted')
                            found_h1 = True 
                        if check_h2 and not found_h2 and ((y[j] <= target_y2 <= y[j+1]) or (y[j+1] <= target_y2 <= y[j])):
                            denom = y[j+1] - y[j]
                            x_int = x[j] + (x[j+1]-x[j]) * (target_y2 - y[j]) / denom if denom != 0 else x[j]
                            intercept_points.append((x_int, target_y2, color, key, 'hline2'))
                            if ui_refs["show_int"].isChecked(): ax_target.vlines(x_int, -0.1, target_y2, colors=color, linestyles='dotted')
                            if key == "black": fsc_res_val = x_int
                            found_h2 = True 
                        if (not check_h1 or found_h1) and (not check_h2 or found_h2): break

            xmin = 0.0 if ui_refs["x_at_0"].isChecked() else -0.05; ymin = 0.0 if ui_refs["y_at_0"].isChecked() else -0.1
            ax_target.set_xlim(xmin, 0.6); ax_target.set_ylim(ymin, 1.05)
            ticks = [0.0, 0.2, 0.4, 0.6]; ax_target.set_xticks(ticks)
            if is_res_mode and ui_refs["show_all"].isChecked(): ax_target.set_xticklabels(["DC" if t==0 else f"{1/t:.2f} Å" for t in ticks])
            if ui_refs["show_all"].isChecked():
                t_fs = float(ui_refs["title_font"].text())
                ax_target.set_xlabel(ui_refs["xtitle"].text(), fontsize=t_fs); ax_target.set_ylabel(ui_refs["ytitle"].text(), fontsize=t_fs)
                tt = ui_refs["title"].text()
                if ui_refs["title_res"].isChecked() and fsc_res_val: tt += f" {1/fsc_res_val:.{res_digits}f}Å"
                ax_target.set_title(tt, fontsize=t_fs, weight='bold'); ax_target.tick_params(labelsize=float(ui_refs["tick_font"].text()), length=float(ui_refs["tick_len"].text()))
            else: ax_target.set_xticklabels([]); ax_target.set_yticklabels([])
            
            for spine in ax_target.spines.values(): spine.set_linewidth(float(ui_refs["spine_w"].text()))
            show_fr = ui_refs["show_frame"].isChecked()
            ax_target.spines['top'].set_visible(show_fr); ax_target.spines['right'].set_visible(show_fr)
            if ui_refs["show_h1"].isChecked(): ax_target.hlines(target_y1, xmin, 0.6, 'black', '--')
            if ui_refs["show_h2"].isChecked(): ax_target.hlines(target_y2, xmin, 0.6, 'black', '--')
            if ui_refs["show_xaxis"].isChecked(): ax_target.vlines(0, ymin, 1.05, 'black', '-')
            if ui_refs["show_yaxis"].isChecked(): ax_target.hlines(0, xmin, 0.6, 'black', '-')
            if ui_refs["show_leg"].isChecked():
                handles, labels = ax_target.get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                ax_target.legend(by_label.values(), by_label.keys(), fontsize=float(ui_refs["leg_font"].text()), frameon=False, loc='upper right')

            if intercept_points:
                coord_fs = float(ui_refs["coord_font"].text())
                while len(coord_user_positions) < len(intercept_points): coord_user_positions.append({})
                for i, (px, py, col, key, hline) in enumerate(intercept_points):
                    txt = f"({1/px:.{res_digits}f}Å, {py:.3f})" if is_res_mode and px>0 else f"({px:.3f}, {py:.3f})"
                    if ui_refs["show_coords"].isChecked():
                        pos_x = coord_user_positions[i].get('x', px + 0.01)
                        pos_y = coord_user_positions[i].get('y', py + 0.02)
                        t = ax_target.text(pos_x, pos_y, txt, color=col, fontsize=coord_fs)
                        if not is_export: ax_target.draggables.append(DraggableCoord(t, canvas, i, coord_user_positions))

                    if ui_refs["show_arrow"].isChecked():
                        dict_key = f"{key}_{hline}"
                        if dict_key not in arrow_pos_dict:
                            apx, apy = (px - 0.1, py + 0.1) if px > 0.5 else (px + 0.05, py + 0.1)
                            arrow_pos_dict[dict_key] = {'x': apx, 'y': apy}
                        apos = arrow_pos_dict[dict_key]
                        start_x, start_y = label_arrow_offset(px, py, apos['x'], apos['y'])
                        ann = ax_target.annotate("", xy=(px, py), xytext=(start_x, start_y), arrowprops=dict(arrowstyle="->", color="black", lw=2))
                        res_val = 1.0 / px if px > 1e-6 else float('inf')
                        res_txt = "infinity Å" if res_val == float('inf') else f"{res_val:.{res_digits}f} Å"
                        at = ax_target.text(apos['x'], apos['y'], res_txt, ha='center', va='center', fontsize=coord_fs, color='black', bbox=dict(boxstyle="round,pad=0.4", facecolor='none', edgecolor='none', alpha=0.0))
                        if not is_export: ax_target.draggables.append(DraggableArrow(at, ann, canvas, dict_key, arrow_pos_dict))

        def update_plot(print_intercepts=False):
            w, h = ui_refs["fig_width"].value() / 10.0, ui_refs["fig_height"].value() / 10.0
            _plot_logic(ax)
            
            # 1. Lock the aspect ratio
            ax.set_box_aspect(h / w)
            
            # 2. Rigidly anchor the margins (completely replacing tight_layout)
            # The left and bottom margins are slightly larger to ensure labels always fit
            fig.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.90)
            
            # Draw the canvas
            canvas.draw()

        update_plot(); plot_win.show()

    if auto_files is not None: _launch_plot_window()
    else: _select_and_plot()

def generate_fsc_excel_file():
    def _manual_export_window():
        man_win = QWidget()
        man_win.setWindowTitle("Select FSC Files Manually")
        center_window(man_win, 500, 250, "fsc_manual_export")
        toplevel_windows.append(man_win)
        man_win.setStyleSheet(QApplication.instance().styleSheet())
        
        layout = QGridLayout(man_win)
        labels = ["Black FSC:", "Red FSC:", "Blue FSC:", "Yellow FSC:"]
        entries = []
        for i, lbl in enumerate(labels):
            layout.addWidget(QLabel(lbl), i, 0)
            entry = QLineEdit()
            layout.addWidget(entry, i, 1)
            entries.append(entry)
            btn = QPushButton("Browse")
            def browse(checked=False, e=entry):
                f, _ = QFileDialog.getOpenFileName(man_win, "Select FSC txt", os.getcwd(), "Text Files (*.txt);;All Files (*)")
                if f:
                    e.setText(f)
                    folder = os.path.dirname(f)
                    for j, color in enumerate(["black.txt", "red.txt", "blue.txt", "yellow.txt"]):
                        if entries[j] != e and not entries[j].text():
                            p = os.path.join(folder, color)
                            if os.path.exists(p): entries[j].setText(p)
            btn.clicked.connect(browse)
            layout.addWidget(btn, i, 2)
            
        def submit_manual_export():
            files = [e.text().strip() for e in entries]
            valid_files = [f for f in files if f and os.path.exists(f)]
            if not valid_files:
                return show_error_message("Please select at least one valid file.")
            
            out_path, _ = QFileDialog.getSaveFileName(man_win, "Save Combined FSC", os.path.join(os.getcwd(), "manual_fsc_combined.txt"), "Text Files (*.txt);;All Files (*)")
            if not out_path: return
            
            combined = []
            headers = ["Freq"]
            colors = ["Black", "Red", "Blue", "Yellow"]
            
            try:
                first_col = True
                for i, fname in enumerate(files):
                    if not fname or not os.path.exists(fname): continue
                    data = safe_read_fsc(fname)
                    if first_col: 
                        combined.append(data[:, 0].tolist())
                        first_col = False
                    combined.append(data[:, 1].tolist())
                    headers.append(colors[i])
                    
                with open(out_path, 'w', newline='') as f:
                    writer = csv.writer(f, delimiter='\t')
                    writer.writerow(headers)
                    writer.writerows(list(zip(*combined)))
                QMessageBox.information(man_win, "Success", f"Saved to {out_path}")
                man_win.close()
            except Exception as e:
                show_error_message(f"Error: {e}")

        submit = QPushButton("Export")
        submit.clicked.connect(submit_manual_export)
        layout.addWidget(submit, 4, 0, 1, 3)
        man_win.show()

    export_win = QWidget()
    export_win.setWindowTitle("Export Combined FSC")
    center_window(export_win, 350, 160, "export_fsc_model_select")
    toplevel_windows.append(export_win)
    export_win.setStyleSheet(QApplication.instance().styleSheet())

    layout = QGridLayout(export_win)
    layout.addWidget(QLabel("Model Name:"), 0, 0)
    model_combo, _ = create_model_name_dropdown(layout, 0, 1)

    def submit_export():
        model_folder = model_combo.currentText().strip()
        if not model_folder: return show_error_message("Please select a Model Name.")
        
        colors = ("black", "red", "blue", "yellow")
        files = [f"FSC/{model_folder}/{c}.txt" for c in colors]
        missing = [f for f in files if not os.path.exists(f)]
        if missing: return show_error_message(f"Missing files: {missing}")
        
        combined = []
        try:
            for i, fname in enumerate(files):
                data = safe_read_fsc(fname)
                if i == 0: combined.append(data[:, 0].tolist())
                combined.append(data[:, 1].tolist())
            out_path = f"FSC/{model_folder}/{model_folder}_fsc_combined.txt"
            with open(out_path, 'w', newline='') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow(["Freq", "Black", "Red", "Blue", "Yellow"])
                writer.writerows(list(zip(*combined)))
            QMessageBox.information(export_win, "Success", f"Saved to {out_path}")
            export_win.close()
        except Exception as e:
            show_error_message(f"Error: {e}")

    submit_btn = QPushButton("Export (Auto-detect from Folder)")
    submit_btn.clicked.connect(submit_export)
    layout.addWidget(submit_btn, 1, 0, 1, 2)
    
    btn_manual = QPushButton("or Select Manually...")
    btn_manual.clicked.connect(lambda: [export_win.close(), _manual_export_window()])
    layout.addWidget(btn_manual, 2, 0, 1, 2)
    export_win.show()

# --- Fully Automated GUI Component ---
class PreviewWorker(QThread):
    result_ready = pyqtSignal(object, object)

    def __init__(self, mrc_path, target_label):
        super().__init__()
        self.mrc_path = mrc_path
        self.target_label = target_label

    def run(self):
        try:
            import mrcfile
            import numpy as np
            import os
            if not os.path.exists(self.mrc_path):
                self.result_ready.emit(None, self.target_label)
                return
            with mrcfile.open(self.mrc_path, permissive=True) as mrc:
                map_data = mrc.data
            if map_data.ndim != 3:
                self.result_ready.emit(None, self.target_label)
                return
            z_slices = map_data.shape[0]
            center_z = z_slices // 2
            start_z = max(0, center_z - 2)
            end_z = min(z_slices, center_z + 3)
            slice_block = map_data[start_z:end_z, :, :]
            avg_slice = np.mean(slice_block, axis=0)
            f_min, f_max = avg_slice.min(), avg_slice.max()
            if f_max > f_min:
                norm_frame = 255.0 * (avg_slice - f_min) / (f_max - f_min)
            else:
                norm_frame = np.zeros_like(avg_slice)
            img_uint8 = np.require(norm_frame.astype(np.uint8), np.uint8, 'C')
            self.result_ready.emit(img_uint8, self.target_label)
        except Exception as e:
            print(f"Preview background error: {e}")
            self.result_ready.emit(None, self.target_label)

class HoverImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.full_pixmap = None
        self.setMouseTracking(True)
        self._popup = QLabel(self, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self._popup.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._popup.setStyleSheet("border: 2px solid #98c379; background-color: #1e1e1e;")
        self._popup.hide()

    def set_full_image(self, pixmap):
        self.full_pixmap = pixmap
        if pixmap:
            self._popup.setPixmap(pixmap)
            self._popup.adjustSize()

    def clear_image(self):
        self.clear()
        self.full_pixmap = None
        self._popup.hide()

    def enterEvent(self, event):
        super().enterEvent(event)
        if self.full_pixmap:
            from PyQt6.QtGui import QCursor
            pos = QCursor.pos()
            self._popup.move(pos.x() + 15, pos.y() + 15)
            self._popup.show()

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self._popup.hide()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.full_pixmap and self._popup.isVisible():
            from PyQt6.QtGui import QCursor
            pos = QCursor.pos()
            self._popup.move(pos.x() + 15, pos.y() + 15)


class BaseAutoApp(QMainWindow):
    pipeline_finished = pyqtSignal()
    step_updated = pyqtSignal(str)

    def request_mrc_preview(self, mrc_path, preview_label):
        preview_label.clear_image()
        if not mrc_path or not os.path.exists(mrc_path) or not mrc_path.lower().endswith('.mrc'): return
        if not hasattr(self, '_preview_threads'): self._preview_threads = []
        self._preview_threads = [t for t in self._preview_threads if t.isRunning()]
        
        worker = PreviewWorker(mrc_path, preview_label)
        worker.result_ready.connect(self._on_preview_ready)
        self._preview_threads.append(worker)
        worker.start()

    def _on_preview_ready(self, img_uint8, preview_label):
        if img_uint8 is None: return
        from PyQt6.QtGui import QImage, QPixmap
        from PyQt6.QtCore import Qt
        h, w = img_uint8.shape
        qimg = QImage(img_uint8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimg)
        
        preview_label.set_full_image(pixmap)
        zoom_factor = 2.4
        target_size = preview_label.size()
        zoomed_w = int(target_size.width() * zoom_factor)
        zoomed_h = int(target_size.height() * zoom_factor)
        
        zoomed_pixmap = pixmap.scaled(zoomed_w, zoomed_h, Qt.AspectRatioMode.KeepAspectRatioByExpanding, Qt.TransformationMode.SmoothTransformation)
        crop_x = (zoomed_pixmap.width() - target_size.width()) // 2
        crop_y = (zoomed_pixmap.height() - target_size.height()) // 2
        cropped_pixmap = zoomed_pixmap.copy(crop_x, crop_y, target_size.width(), target_size.height())
        
        preview_label.setPixmap(cropped_pixmap)

    def append_log_step(self, text):
        self.log_panel.append(f'<span style="color: #d4d4d4;">> {text}</span><br>')
        sb = self.log_panel.verticalScrollBar(); sb.setValue(sb.maximum())

    def _reset_run_button(self):
        self.btn_run.setText("RUN"); self.btn_run.setEnabled(True); self.check_visualizer_availability()

    def create_fsc_params(self, layout, defaults, row=1, color="Red"):
        chk_multi = QCheckBox("Run Multiple to Find the Best Parameters")
        layout.addWidget(chk_multi, row, 0, 1, 5)

        labels = ["Ini threshold:", "Extend inimask:", "Soft edge width:", "Low-pass filter:"]
        low_end = "5" if color == "Red" else "10"
        multi_defaults = {"Ini threshold:": ("0.1", "0.01"), "Extend inimask:": ("5", "1"), "Soft edge width:": ("2", "1"), "Low-pass filter:": (low_end, "1")}

        entries = {}; extra_widgets = []
        for i, (lbl, default) in enumerate(zip(labels, defaults)):
            r = row + 1 + i; layout.addWidget(QLabel(lbl), r, 0)
            e_start = QLineEdit(default); e_start.setFixedWidth(80); layout.addWidget(e_start, r, 1)
            lbl_tilde = QLabel("~"); lbl_tilde.setAlignment(Qt.AlignmentFlag.AlignCenter); layout.addWidget(lbl_tilde, r, 2)
            e_end = QLineEdit(); e_end.setFixedWidth(80); layout.addWidget(e_end, r, 3)
            lbl_left = QLabel("Step:"); lbl_left.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            e_step = QLineEdit(); e_step.setFixedWidth(80); e_step.setAlignment(Qt.AlignmentFlag.AlignLeft)
            lbl_right = QLabel(""); lbl_right.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            step_layout = QHBoxLayout(); step_layout.setContentsMargins(0, 0, 0, 0); step_layout.setSpacing(2)
            step_layout.addWidget(lbl_left); step_layout.addWidget(e_step); step_layout.addWidget(lbl_right)
            step_widget = QWidget(); step_widget.setLayout(step_layout); layout.addWidget(step_widget, r, 4)
            entries[lbl] = (e_start, e_end, e_step); extra_widgets.extend([lbl_tilde, e_end, step_widget])

        for w in extra_widgets:
            sp = w.sizePolicy()
            sp.setRetainSizeWhenHidden(True)
            w.setSizePolicy(sp)

        lbl_jobs = QLabel("Total Jobs: 1"); layout.addWidget(lbl_jobs, row+5, 0, 1, 2)
        btn_vis = QPushButton("View All Curves"); btn_vis.setVisible(False); btn_vis.clicked.connect(lambda: self.open_visualizer(color)); layout.addWidget(btn_vis, row+5, 3, 1, 2)

        def toggle_multi(checked):
            for w in extra_widgets: w.setVisible(checked)
            if checked:
                for lbl in labels:
                    _, e_end, e_step = entries[lbl]; e_end.setText(multi_defaults[lbl][0]); e_step.setText(multi_defaults[lbl][1])
            else:
                for lbl in labels:
                    _, e_end, e_step = entries[lbl]; e_end.clear(); e_step.clear()
                    
        chk_multi.toggled.connect(toggle_multi); toggle_multi(False)

        def get_range(s, e, st, is_float):
            try:
                start, end, step = float(s.text()), float(e.text()), float(st.text())
                if step <= 0: return []
                vals = np.arange(start, end + (step/1000.0), step)
                return [round(x, 3) for x in vals] if is_float else [int(round(x)) for x in vals]
            except: return []

        def recalc_jobs():
            l_ini = get_range(*entries["Ini threshold:"], True) or [entries["Ini threshold:"][0].text()]
            l_ext = get_range(*entries["Extend inimask:"], False) or [entries["Extend inimask:"][0].text()]
            l_soft = get_range(*entries["Soft edge width:"], False) or [entries["Soft edge width:"][0].text()]
            l_low = get_range(*entries["Low-pass filter:"], True) or [entries["Low-pass filter:"][0].text()]
            lbl_jobs.setText(f"Total Jobs: {len(l_ini) * len(l_ext) * len(l_soft) * len(l_low)}")

        for key in entries:
            for widget in entries[key]: widget.textChanged.connect(recalc_jobs)
        return entries, lbl_jobs, btn_vis, chk_multi

    def get_pixel_size(self):
        if self.pixel_size_combo.currentText() == "Custom": return self.pixel_size_custom.text().strip()
        return self.pixel_size_combo.currentText()

class RelionAutoApp(BaseAutoApp):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fully Automated FSC Curve")
        self.pipeline_finished.connect(self._reset_run_button)
        self.step_updated.connect(self.append_log_step)
        self.resize(1100, 700)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_layout.addWidget(splitter)
        
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(15, 15, 10, 15); left_layout.setSpacing(10)
        
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 15, 15, 15); right_layout.setSpacing(10)
        
        splitter.addWidget(left_widget); splitter.addWidget(right_widget); splitter.setSizes([600, 500])

        def create_aligned_label(text, color=None):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if color: lbl.setStyleSheet(f"color: {color};")
            return lbl

        def create_grey_label(text):
            lbl = QLabel(text); lbl.setStyleSheet("color: #8a8a8a;"); return lbl

        input_group = QGroupBox("Core Parameters"); grid = QGridLayout(input_group); grid.setSpacing(10)

        grid.addWidget(create_aligned_label("<span style='color: #98c379;'><b>Refine 3D</b></span> - Job Number:"), 0, 0, 1, 2)
        self.refine_job_entry = QLineEdit(); self.refine_job_entry.setPlaceholderText("e.g. 99")
        self.refine_job_entry.setFixedWidth(80)
        self.lbl_refine_alias = QLabel("")
        self.lbl_refine_alias.setStyleSheet("color: #98c379; font-weight: bold; padding-left: 10px;")
        
        refine_job_layout = QHBoxLayout(); refine_job_layout.setContentsMargins(0, 0, 0, 0)
        refine_job_layout.addWidget(self.refine_job_entry)
        refine_job_layout.addWidget(self.lbl_refine_alias)
        refine_job_layout.addStretch()
        grid.addLayout(refine_job_layout, 0, 2)
        
        self.refine_job_entry.textChanged.connect(self.load_refine_params)

        self.lbl_refine_preview = HoverImageLabel()
        self.lbl_refine_preview.setFixedSize(55, 55)
        self.lbl_refine_preview.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px;")
        self.lbl_refine_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_refine_preview, 1, 0, 2, 1, Qt.AlignmentFlag.AlignCenter)

        grid.addWidget(create_aligned_label("<span style='color: #98c379;'><b>PostProcess</b></span> - Job Number:"), 3, 0, 1, 2)
        self.post_job_entry = QLineEdit(); self.post_job_entry.setPlaceholderText("e.g. 100")
        self.post_job_entry.setFixedWidth(80)
        self.lbl_post_alias = QLabel("")
        self.lbl_post_alias.setStyleSheet("color: #98c379; font-weight: bold; padding-left: 10px;")
        
        post_job_layout = QHBoxLayout(); post_job_layout.setContentsMargins(0, 0, 0, 0)
        post_job_layout.addWidget(self.post_job_entry)
        post_job_layout.addWidget(self.lbl_post_alias)
        post_job_layout.addStretch()
        grid.addLayout(post_job_layout, 3, 2)
        
        self.post_job_entry.textChanged.connect(self.load_post_params)

        self.lbl_post_preview = HoverImageLabel()
        self.lbl_post_preview.setFixedSize(55, 55)
        self.lbl_post_preview.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px;")
        self.lbl_post_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_post_preview, 4, 0, 1, 1, Qt.AlignmentFlag.AlignCenter)

        grid.addWidget(create_aligned_label("Helical Twist & Rise:"), 5, 0, 1, 2)
        twist_rise_layout = QHBoxLayout(); twist_rise_layout.setContentsMargins(0, 0, 0, 0)
        twist_rise_layout.addWidget(create_grey_label("Twist:")); self.global_twist_entry = QLineEdit(); twist_rise_layout.addWidget(self.global_twist_entry)
        twist_rise_layout.addWidget(create_grey_label("Rise:")); self.global_rise_entry = QLineEdit(); twist_rise_layout.addWidget(self.global_rise_entry)
        grid.addLayout(twist_rise_layout, 5, 2)

        grid.addWidget(create_aligned_label("Resolution (Å):"), 6, 0, 1, 2)
        self.global_res_entry = QLineEdit()
        grid.addWidget(self.global_res_entry, 6, 2)

        grid.addWidget(create_aligned_label("Pixel Size (angpix):"), 7, 0, 1, 2)
        self.pixel_size_combo = QComboBox(); self.pixel_size_combo.addItems(["0.9557", "1.9114", "2.8671", "3.8228", "4.7785", "Custom"])
        self.pixel_size_custom = QLineEdit(); self.pixel_size_custom.setVisible(False)
        pix_layout = QHBoxLayout(); pix_layout.setContentsMargins(0, 0, 0, 0); pix_layout.addWidget(self.pixel_size_combo, 1); pix_layout.addWidget(self.pixel_size_custom, 2)
        grid.addLayout(pix_layout, 7, 2)
        self.pixel_size_combo.currentTextChanged.connect(lambda text: self.pixel_size_custom.setVisible(text == "Custom"))

        grid.addWidget(create_aligned_label("Tube Diameter (Å):"), 8, 0, 1, 2); self.tube_diam_entry = QLineEdit("200"); grid.addWidget(self.tube_diam_entry, 8, 2)
        grid.addWidget(create_aligned_label("Z Percentage (%):"), 9, 0, 1, 2); self.z_perc_entry = QLineEdit("30"); grid.addWidget(self.z_perc_entry, 9, 2)

        grid.addWidget(create_aligned_label("Model Name:"), 10, 0, 1, 2)
        self.model_combo = QComboBox(); self.model_custom = QLineEdit(); self.model_custom.setPlaceholderText("Custom Name"); self.model_custom.setVisible(False)
        self.model_combo.setMaxVisibleItems(15)
        self.model_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_combo.setMinimumContentsLength(12)
        self.model_combo.view().setMinimumWidth(400)
        self.model_combo.currentTextChanged.connect(self.on_model_changed); self.model_custom.textChanged.connect(lambda text: self.auto_fill_pdbs())
        model_layout = QHBoxLayout(); model_layout.setContentsMargins(0, 0, 0, 0); model_layout.addWidget(self.model_combo, 1); model_layout.addWidget(self.model_custom, 2)
        grid.addLayout(model_layout, 10, 2); left_layout.addWidget(input_group)

        lbl_log = QLabel("Live Execution Log"); lbl_log.setStyleSheet("font-weight: bold; color: #98c379; margin-top: 5px;")
        left_layout.addWidget(lbl_log)
        self.log_panel = QTextEdit(); self.log_panel.setReadOnly(True); self.log_panel.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.log_panel.setStyleSheet("""
            QTextEdit { background-color: #1e1e1e; font-family: monospace; border: 1px solid #3e3e42; border-radius: 4px; padding: 5px; selection-background-color: #98c379; selection-color: #1e1e1e; }
            QTextEdit:focus { outline: none; }
            QScrollBar:vertical { border: none; background: transparent; width: 10px; margin: 0px; }
            QScrollBar::handle:vertical { background: #4a4a4a; border-radius: 5px; min-height: 30px; }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed { background: #98c379; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; background: none; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        """)
        left_layout.addWidget(self.log_panel, stretch=1)

        from PyQt6.QtWidgets import QRadioButton
        self.black_group = QGroupBox("Black Curve (Map) - Halfmap 1 vs Halfmap 2"); self.black_group.setCheckable(True); self.black_group.setChecked(True)
        black_grid = QGridLayout(self.black_group)
        self.rb_black_helical = QRadioButton("Use Original Unfiltered Halfmaps")
        self.rb_black_helical.setChecked(True)
        self.rb_black_helical_exp = QRadioButton("Use Helical Expanded Maps")
        self.rb_black_star = QRadioButton("Use Postprocess.star from RELION")
        black_grid.addWidget(self.rb_black_helical, 0, 0, 1, 2)
        black_grid.addWidget(self.rb_black_helical_exp, 1, 0, 1, 2)
        black_grid.addWidget(self.rb_black_star, 2, 0, 1, 2)
        
        self.black_star_entry = QLineEdit()
        self.black_star_entry.setPlaceholderText("Select postprocess.star...")
        self.black_star_entry.setEnabled(False)
        self.btn_browse_black_star = QPushButton("Browse")
        self.btn_browse_black_star.setEnabled(False)
        
        def browse_star_file():
            path, _ = QFileDialog.getOpenFileName(self, "Select postprocess.star", os.getcwd(), "STAR Files (*.star);;All Files (*)")
            if path: self.black_star_entry.setText(path)
        self.btn_browse_black_star.clicked.connect(browse_star_file)
        
        black_grid.addWidget(self.black_star_entry, 3, 0)
        black_grid.addWidget(self.btn_browse_black_star, 3, 1)
        
        def toggle_black_star(checked):
            self.black_star_entry.setEnabled(checked)
            self.btn_browse_black_star.setEnabled(checked)
            if checked and not self.black_star_entry.text().strip():
                post_job = self.post_job_entry.text().strip().zfill(3)
                if post_job and post_job != "000":
                    cwd = os.getcwd()
                    star_path = os.path.join(cwd, "PostProcess", f"job{post_job}", "postprocess.star")
                    if not os.path.exists(star_path):
                        star_path = os.path.join(cwd, "Postprocess", f"job{post_job}", "postprocess.star")
                    if os.path.exists(star_path):
                        self.black_star_entry.setText(star_path)
            self.update_section_states()
        self.rb_black_star.toggled.connect(toggle_black_star)
        
        right_layout.addWidget(self.black_group, stretch=1)

        self.red_group = QGroupBox("Red Curve (Full) - Model vs PostProcess"); self.red_group.setCheckable(True); self.red_group.setChecked(True)
        red_grid = QGridLayout(self.red_group)
        red_grid.addWidget(QLabel("Final Model (Red):"), 0, 0); self.red_pdb_entry = QLineEdit(); self.red_pdb_entry.setPlaceholderText("e.g. model.pdb")
        red_grid.addWidget(self.red_pdb_entry, 0, 1, 1, 3)
        self.btn_browse_red = QPushButton("Browse"); self.btn_browse_red.setFixedWidth(160); self.btn_browse_red.clicked.connect(lambda: self.browse_pdb(self.red_pdb_entry)); red_grid.addWidget(self.btn_browse_red, 0, 4, 1, 1)
        self.red_params, self.lbl_red_jobs, self.btn_red_vis, self.chk_red_multi = self.create_fsc_params(red_grid, defaults=["0.1", "1", "1", "5"], row=1, color="Red")
        right_layout.addWidget(self.red_group, stretch=1) 

        self.blue_group = QGroupBox("Blue Curve (Work) - Model vs Halfmap 1"); self.blue_group.setCheckable(True); self.blue_group.setChecked(True)
        blue_grid = QGridLayout(self.blue_group)
        blue_grid.addWidget(QLabel("Final Model (Blue):"), 0, 0); self.blue_pdb_entry = QLineEdit(); self.blue_pdb_entry.setPlaceholderText("e.g. model_blue.pdb")
        blue_grid.addWidget(self.blue_pdb_entry, 0, 1, 1, 3)
        self.btn_browse_blue = QPushButton("Browse"); self.btn_browse_blue.setFixedWidth(160); self.btn_browse_blue.clicked.connect(lambda: self.browse_pdb(self.blue_pdb_entry)); blue_grid.addWidget(self.btn_browse_blue, 0, 4, 1, 1)
        self.blue_params, self.lbl_blue_jobs, self.btn_blue_vis, self.chk_blue_multi = self.create_fsc_params(blue_grid, defaults=["0.1", "1", "1", "10"], row=1, color="Blue")
        right_layout.addWidget(self.blue_group, stretch=1) 

        self.yellow_group = QGroupBox("Yellow Curve (Free) - Halfmap 1 vs Halfmap 2"); self.yellow_group.setCheckable(True); self.yellow_group.setChecked(True)
        yellow_grid = QGridLayout(self.yellow_group)
        lbl_yellow_desc = QLabel("Use Halfmap 2 and the Mask Generated by the Blue Curve")
        lbl_yellow_desc.setStyleSheet("color: #8a8a8a;")
        yellow_grid.addWidget(lbl_yellow_desc, 0, 0)
        right_layout.addWidget(self.yellow_group, stretch=1) 

        # Thread control to prevent HPC CPU policy violations
        self.thread_widget = QWidget()
        thread_layout = QHBoxLayout(self.thread_widget)
        thread_layout.setContentsMargins(0, 0, 0, 0)
        thread_layout.addWidget(QLabel("Parallel Threads:"))
        self.spin_threads = QSpinBox()
        self.spin_threads.setRange(1, 32)
        self.spin_threads.setValue(3) # Matches manual default
        thread_layout.addWidget(self.spin_threads)
        
        lbl_thread_warn = QLabel("More threads may exceed HPC CPU policy, but xcop will handle it smartly")
        lbl_thread_warn.setStyleSheet("color: #FF5555; font-size: 9pt;")
        lbl_thread_warn.setVisible(False)
        thread_layout.addWidget(lbl_thread_warn)
        thread_layout.addStretch()
        right_layout.addWidget(self.thread_widget)
        self.thread_widget.setVisible(False)

        def check_thread_warn(val): lbl_thread_warn.setVisible(val > 5)
        self.spin_threads.valueChanged.connect(check_thread_warn)
        check_thread_warn(self.spin_threads.value())

        def update_thread_visibility():
            self.thread_widget.setVisible(self.chk_red_multi.isChecked() or self.chk_blue_multi.isChecked())
        
        self.chk_red_multi.toggled.connect(lambda _: update_thread_visibility())
        self.chk_blue_multi.toggled.connect(lambda _: update_thread_visibility())

        self.btn_run = QPushButton("RUN"); self.btn_run.setFixedHeight(40); self.btn_run.setStyleSheet("font-weight: bold; background-color: #98c379; color: #1e1e1e;")
        self.btn_run.clicked.connect(self.run_expansion)
        right_layout.addWidget(self.btn_run)
        
        self.btn_main_vis = QPushButton("Visualize Curves"); self.btn_main_vis.setFixedHeight(40)
        self.btn_main_vis.setEnabled(False)
        self.btn_main_vis.clicked.connect(self.launch_main_vis)
        right_layout.addWidget(self.btn_main_vis)
        
        from PyQt6.QtWidgets import QSizePolicy
        for group in [self.black_group, self.red_group, self.blue_group, self.yellow_group]:
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            
        self.black_group.toggled.connect(lambda _: self.update_section_states())
        self.red_group.toggled.connect(lambda _: self.update_section_states())
        self.blue_group.toggled.connect(lambda _: self.update_section_states())
        self.yellow_group.toggled.connect(lambda _: self.update_section_states())
        
        self.update_section_states()
        self.refresh_model_options()

    def update_section_states(self):
        run_yellow = self.yellow_group.isChecked()
        run_blue = self.blue_group.isChecked()
        
        # Enforce Yellow -> Blue dependency
        if run_yellow and not run_blue:
            model_name = self.get_model_name()
            mask_path = os.path.join(os.getcwd(), "FSC", model_name, "molmap_halfmap1_masked.mrc")
            if not os.path.exists(mask_path):
                self.blue_group.blockSignals(True)
                self.blue_group.setChecked(True)
                self.blue_group.blockSignals(False)
                run_blue = True
                self.step_updated.emit("Required mask for Yellow not found. Auto-selecting Blue Curve.")

        run_black = self.black_group.isChecked()
        use_star = self.rb_black_star.isChecked()
        run_red = self.red_group.isChecked()

        if not run_red: self.chk_red_multi.setChecked(False)
        if not run_blue: self.chk_blue_multi.setChecked(False)

        need_refine = run_blue or run_yellow or (run_black and not use_star)
        need_post = run_red

        for w in [self.refine_job_entry, self.lbl_refine_preview]:
            w.setEnabled(need_refine)

        for w in [self.post_job_entry, self.lbl_post_preview]:
            w.setEnabled(need_post)
            
        self.global_res_entry.setEnabled(need_refine or need_post)
        self.global_twist_entry.setEnabled(need_refine or need_post)
        self.global_rise_entry.setEnabled(need_refine or need_post)

    def mousePressEvent(self, event):
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, QLineEdit): focused_widget.clearFocus()
        super().mousePressEvent(event)

    def load_refine_params(self):
        raw_num = self.refine_job_entry.text().strip()
        self.lbl_refine_alias.clear()
        if not raw_num: return
        job_num = raw_num.zfill(3)
        if job_num == "000": return
        
        star_path = os.path.join(os.getcwd(), "default_pipeline.star")
        if os.path.exists(star_path):
            try:
                with open(star_path, 'r') as f:
                    for line in f:
                        if f"Refine3D/job{job_num}/" in line:
                            parts = line.strip().split()
                            if len(parts) >= 2 and parts[1] != "None":
                                alias_parts = parts[1].strip('/').split('/')
                                if len(alias_parts) > 1: self.lbl_refine_alias.setText(alias_parts[-1])
                            break
            except: pass

        refine_out = os.path.join(os.getcwd(), "Refine3D", f"job{job_num}", "run.out")
        self.global_twist_entry.clear(); self.global_rise_entry.clear()
        if os.path.exists(refine_out):
            try:
                with open(refine_out, 'r', encoding='utf-8', errors='ignore') as f: content = f.read()
                avg = re.search(r'Averaged helical twist =\s*(-?[\d\.]+)\s*degrees,\s*rise =\s*(-?[\d\.]+)\s*Angstroms', content)
                if avg:
                    self.global_twist_entry.setText(avg.group(1)); self.global_rise_entry.setText(avg.group(2))
            except Exception as e: print(f"Error reading Refine parameters: {e}")

        if job_num != "000":
            mrc_path = os.path.join(os.getcwd(), "Refine3D", f"job{job_num}", "run_class001.mrc")
            self.request_mrc_preview(mrc_path, self.lbl_refine_preview)
        else:
            self.lbl_refine_preview.clear_image()

    def load_post_params(self):
        raw_post_num = self.post_job_entry.text().strip()
        self.lbl_post_alias.clear()
        if raw_post_num:
            post_job_num = raw_post_num.zfill(3)
            if post_job_num != "000":
                star_path = os.path.join(os.getcwd(), "default_pipeline.star")
                if os.path.exists(star_path):
                    try:
                        with open(star_path, 'r') as f:
                            for line in f:
                                if f"PostProcess/job{post_job_num}/" in line:
                                    parts = line.strip().split()
                                    if len(parts) >= 2 and parts[1] != "None":
                                        alias_parts = parts[1].strip('/').split('/')
                                        if len(alias_parts) > 1: self.lbl_post_alias.setText(alias_parts[-1])
                                    break
                    except: pass
        
        job_num = self.refine_job_entry.text().strip().zfill(3)
        self.global_res_entry.clear()
        if job_num and job_num != "000":
            refine_out = os.path.join(os.getcwd(), "Refine3D", f"job{job_num}", "run.out")
            if os.path.exists(refine_out):
                try:
                    with open(refine_out, 'r') as f: content = f.read()
                    avg = re.search(r'Averaged helical twist =\s*(-?[\d\.]+)\s*degrees,\s*rise =\s*(-?[\d\.]+)\s*Angstroms', content)
                    if avg and not self.global_twist_entry.text():
                        self.global_twist_entry.setText(avg.group(1)); self.global_rise_entry.setText(avg.group(2))
                except: pass

        if raw_post_num and raw_post_num.zfill(3) != "000":
            post_job_num = raw_post_num.zfill(3)
            
            post_out = os.path.join(os.getcwd(), "PostProcess", f"job{post_job_num}", "run.out")
            if not os.path.exists(post_out):
                post_out = os.path.join(os.getcwd(), "Postprocess", f"job{post_job_num}", "run.out")
            if os.path.exists(post_out):
                try:
                    with open(post_out, 'r') as f: post_content = f.read()
                    res_red_match = re.search(r'\+\s*FINAL RESOLUTION:\s*([\d\.]+)', post_content)
                    if res_red_match:
                        res_val = res_red_match.group(1)
                        self.global_res_entry.setText(res_val)
                except Exception as e: print(f"Error reading PostProcess resolution: {e}")
                
            mrc_path = os.path.join(os.getcwd(), "PostProcess", f"job{post_job_num}", "postprocess.mrc")
            if not os.path.exists(mrc_path): # Fallback to alternate folder naming
                mrc_path = os.path.join(os.getcwd(), "Postprocess", f"job{post_job_num}", "postprocess.mrc")
            self.request_mrc_preview(mrc_path, self.lbl_post_preview)
            
            if self.rb_black_star.isChecked() and not self.black_star_entry.text().strip():
                star_path = os.path.join(os.getcwd(), "PostProcess", f"job{post_job_num}", "postprocess.star")
                if not os.path.exists(star_path):
                    star_path = os.path.join(os.getcwd(), "Postprocess", f"job{post_job_num}", "postprocess.star")
                if os.path.exists(star_path):
                    self.black_star_entry.setText(star_path)
        else:
            self.lbl_post_preview.clear_image()

    def check_visualizer_availability(self):
        model_name = self.get_model_name()
        base_dir = os.path.join(os.getcwd(), "FSC", model_name)
        red_log = os.path.join(base_dir, "Comprehensive_Red", "params_map.json")
        self.btn_red_vis.setVisible(os.path.exists(red_log)); self.btn_red_vis.setEnabled(os.path.exists(red_log))
        blue_log = os.path.join(base_dir, "Comprehensive_Blue", "params_map.json")
        self.btn_blue_vis.setVisible(os.path.exists(blue_log)); self.btn_blue_vis.setEnabled(os.path.exists(blue_log))
        
        has_txt = any(os.path.exists(os.path.join(base_dir, f"{c}.txt")) for c in ("black", "red", "blue", "yellow"))
        self.btn_main_vis.setEnabled(has_txt)

    def launch_main_vis(self):
        model_name = self.get_model_name()
        base_dir = os.path.join(os.getcwd(), "FSC", model_name)
        files = [os.path.join(base_dir, f"{c}.txt") for c in ("black", "red", "blue", "yellow")]
        if any(os.path.exists(f) for f in files):
            generate_fsc_curves(auto_files=files, auto_model=model_name)

    def open_visualizer(self, color):
        model_name = self.get_model_name()
        comp_dir = os.path.join(os.getcwd(), "FSC", model_name, f"Comprehensive_{color}")
        log_file = os.path.join(comp_dir, "params_map.json")
        if not os.path.exists(log_file): return QMessageBox.critical(self, "Error", "No comprehensive data found. Run pipeline first.")
        self.vis_win = ComprehensivePlotWindow(comp_dir, log_file); self.vis_win.show()

    def browse_pdb(self, entry_widget):
        model_name = self.get_model_name()
        start_dir = os.path.join(os.getcwd(), "FSC", model_name) if model_name else os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Select PDB File", start_dir, "PDB Files (*.pdb *.cif);;All Files (*)")
        if path: entry_widget.setText(path)

    def on_model_changed(self, text):
        self.model_custom.setVisible(text == "Custom"); self.auto_fill_pdbs()

    def auto_fill_pdbs(self):
        fsc_model_path = os.path.join(os.getcwd(), "FSC", self.get_model_name())
        if os.path.isdir(fsc_model_path):
            pdbs = [f for f in os.listdir(fsc_model_path) if f.endswith(".pdb")]
            if len(pdbs) == 1: self.red_pdb_entry.setText(pdbs[0]); self.blue_pdb_entry.setText(pdbs[0])
            elif len(pdbs) >= 2: self.red_pdb_entry.setText(pdbs[0]); self.blue_pdb_entry.setText(pdbs[1])
            else: self.red_pdb_entry.clear(); self.blue_pdb_entry.clear()
        else: self.red_pdb_entry.clear(); self.blue_pdb_entry.clear()
        self.check_visualizer_availability()

    def refresh_model_options(self):
        options = ["Custom"]
        fsc_path = os.path.join(os.getcwd(), "FSC")
        if os.path.isdir(fsc_path):
            try: options = sorted([d for d in os.listdir(fsc_path) if os.path.isdir(os.path.join(fsc_path, d))]) + ["Custom"]
            except Exception as e: print(f"Error scanning FSC folder: {e}")
        self.model_combo.clear(); self.model_combo.addItems(options)
        if len(options) > 1: self.model_combo.setCurrentText(options[0])
        self.auto_fill_pdbs()

    def get_model_name(self):
        if self.model_combo.currentText() == "Custom": return self.model_custom.text().strip()
        return self.model_combo.currentText().strip()

    def run_expansion(self):
        refine_job, post_job, pix = self.refine_job_entry.text().strip().zfill(3), self.post_job_entry.text().strip().zfill(3), self.get_pixel_size()
        diam, z_perc_str = self.tube_diam_entry.text().strip(), self.z_perc_entry.text().strip()
        model_name, red_pdb_raw, blue_pdb_raw = self.get_model_name(), self.red_pdb_entry.text().strip(), self.blue_pdb_entry.text().strip()
        use_star_for_black = self.rb_black_star.isChecked()
        use_helical_exp_for_black = self.rb_black_helical_exp.isChecked()
        black_star_path = self.black_star_entry.text().strip()

        run_black = self.black_group.isChecked()
        run_red = self.red_group.isChecked()
        run_blue = self.blue_group.isChecked()
        run_yellow = self.yellow_group.isChecked()

        run_blue_masking = run_blue
        if run_yellow and not run_blue:
            mask_path = os.path.join(os.getcwd(), "FSC", model_name, "molmap_halfmap1_masked.mrc")
            if not os.path.exists(mask_path):
                run_blue_masking = True
                self.step_updated.emit("Required mask for Yellow not found. Will generate Blue mask automatically.")

        need_refine = run_blue or run_yellow or (run_black and not use_star_for_black)
        need_post = run_red

        required_fields = [pix, diam, z_perc_str, model_name]
        if need_refine: required_fields.append(refine_job)
        if need_post: required_fields.append(post_job)

        if not all(required_fields): return QMessageBox.critical(self, "Error", "Please fill in all required core parameters for the selected curves.")
        if run_red:
            if not red_pdb_raw: return QMessageBox.critical(self, "Error", "Please fill in the Red PDB parameter.")
            pdb_check = red_pdb_raw if os.path.isabs(red_pdb_raw) else os.path.join(os.getcwd(), "FSC", model_name, red_pdb_raw)
            if not os.path.exists(pdb_check): return QMessageBox.critical(self, "Error", f"Red PDB file not found:\n{pdb_check}")
        if run_blue_masking:
            if not blue_pdb_raw: return QMessageBox.critical(self, "Error", "Please fill in the Blue PDB parameter (required for mask generation).")
            pdb_check = blue_pdb_raw if os.path.isabs(blue_pdb_raw) else os.path.join(os.getcwd(), "FSC", model_name, blue_pdb_raw)
            if not os.path.exists(pdb_check): return QMessageBox.critical(self, "Error", f"Blue PDB file not found:\n{pdb_check}")
        if run_black and use_star_for_black and not os.path.exists(black_star_path): return QMessageBox.critical(self, "Error", "Please select a valid postprocess.star file for the Black curve.")
        try: z_perc = float(z_perc_str) / 100.0
        except ValueError: return QMessageBox.critical(self, "Error", "Z Percentage must be a valid number.")

        def parse_ranges(entries_dict):
            def get_vals(s, e, st, is_float):
                try:
                    start, end, step = float(s.text()), float(e.text()), float(st.text())
                    if step <= 0: raise ValueError
                    vals = np.arange(start, end + (step/1000.0), step)
                    return [str(round(x, 3)) if is_float else str(int(round(x))) for x in vals]
                except: return [s.text().strip()]
            return (get_vals(*entries_dict["Ini threshold:"], True), get_vals(*entries_dict["Extend inimask:"], False), get_vals(*entries_dict["Soft edge width:"], False), get_vals(*entries_dict["Low-pass filter:"], True))

        red_ini, red_ext, red_soft, red_low = parse_ranges(self.red_params)
        blue_ini, blue_ext, blue_soft, blue_low = parse_ranges(self.blue_params)
        red_jobs = len(red_ini) * len(red_ext) * len(red_soft) * len(red_low)
        blue_jobs = len(blue_ini) * len(blue_ext) * len(blue_soft) * len(blue_low)
        red_p, blue_p = [red_ini[0], red_ext[0], red_soft[0], red_low[0]], [blue_ini[0], blue_ext[0], blue_soft[0], blue_low[0]]

        cwd = os.getcwd(); out_dir = os.path.join(cwd, "FSC", model_name); os.makedirs(out_dir, exist_ok=True)
        def prepare_pdb(pdb_path):
            if not pdb_path: return ""
            abs_src = pdb_path if os.path.isabs(pdb_path) else os.path.join(cwd, "FSC", model_name, pdb_path)
            if not os.path.exists(abs_src) and os.path.exists(os.path.join(cwd, pdb_path)):
                abs_src = os.path.join(cwd, pdb_path)
                
            basename = os.path.basename(abs_src)
            target_path = os.path.join(out_dir, basename)
            if os.path.abspath(abs_src) != os.path.abspath(target_path) and os.path.exists(abs_src):
                try: shutil.copy2(abs_src, target_path)
                except Exception as e: print(f"Failed to copy {basename}: {e}")
            return basename

        red_pdb, blue_pdb = prepare_pdb(red_pdb_raw), prepare_pdb(blue_pdb_raw)

        refine_out, post_out = os.path.join(cwd, "Refine3D", f"job{refine_job}", "run.out"), os.path.join(cwd, "PostProcess", f"job{post_job}", "run.out")
        half1_path, half2_path = os.path.join(cwd, "Refine3D", f"job{refine_job}", "run_half1_class001_unfil.mrc"), os.path.join(cwd, "Refine3D", f"job{refine_job}", "run_half2_class001_unfil.mrc")
        post_path = os.path.join(cwd, "PostProcess", f"job{post_job}", "postprocess.mrc")

        missing_files = []
        if need_refine: missing_files.extend([half1_path, half2_path])
        if need_post: missing_files.append(post_path)
        missing_files = [f for f in missing_files if not os.path.exists(f)]
        if missing_files: return QMessageBox.critical(self, "Error", f"Missing required files in working directory:\n{chr(10).join(missing_files)}")

        try:
            res_blue, res_red = "N/A", "N/A"
            twist_h1, rise_h1, twist_h2, rise_h2, twist_avg, rise_avg = "", "", "", "", "", ""
            global_twist, global_rise = self.global_twist_entry.text().strip(), self.global_rise_entry.text().strip()
            
            if need_refine or need_post:
                if not (global_twist and global_rise): 
                    return QMessageBox.critical(self, "Missing Values", "Global Twist and Rise values missing.")
                twist_h1 = twist_h2 = twist_avg = global_twist
                rise_h1 = rise_h2 = rise_avg = global_rise

            res_global = self.global_res_entry.text().strip()
            res_blue, res_red = res_global, res_global
            
            if (need_refine or need_post) and not res_global:
                return QMessageBox.critical(self, "Missing Values", "Global Resolution is missing.")
        except Exception as e: return QMessageBox.critical(self, "Input Error", f"Error with inputs: {str(e)}")

        out_dir_unix = out_dir.replace("\\", "/")
        self.btn_run.setText("..."); self.log_panel.setFocus(); self.btn_run.setEnabled(False); self.log_panel.clear(); QApplication.processEvents()

        def pipeline_worker():
            try:
                self.step_updated.emit("Starting Pipeline...")
                base_cmd = f"module load {RELION_VER} && relion_helix_toolbox --impose"
                out_h1 = f"{out_dir_unix}/symEx_halfmap1.mrc"
                cmd_h1 = f'cp "{half1_path}" "{out_dir_unix}/" && {base_cmd} --i "{half1_path}" --o "{out_h1}" --angpix {pix} --twist {twist_h1} --rise {rise_h1} --cyl_outer_diameter {diam} --z_percentage {z_perc:.2f}'
                out_h2 = f"{out_dir_unix}/symEx_halfmap2.mrc"
                cmd_h2 = f'cp "{half2_path}" "{out_dir_unix}/" && {base_cmd} --i "{half2_path}" --o "{out_h2}" --angpix {pix} --twist {twist_h2} --rise {rise_h2} --cyl_outer_diameter {diam} --z_percentage {z_perc:.2f}'
                out_post = f"{out_dir_unix}/hh_postprocess.mrc"
                cmd_post = f'cp "{post_path}" "{out_dir_unix}/" && {base_cmd} --i "{post_path}" --o "{out_post}" --angpix {pix} --twist {twist_avg} --rise {rise_avg} --cyl_outer_diameter {diam} --z_percentage {z_perc:.2f}'

                if need_refine:
                    self.step_updated.emit(f"Symmetry Expanding: {os.path.basename(half1_path)}")
                    subprocess.run(cmd_h1, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.step_updated.emit(f"Symmetry Expanding: {os.path.basename(half2_path)}")
                    subprocess.run(cmd_h2, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if need_post:
                    self.step_updated.emit(f"Symmetry Expanding: {os.path.basename(post_path)}")
                    subprocess.run(cmd_post, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                parsed_z_percentage = "N/A"
                if run_black:
                    self.step_updated.emit("Calculate Black Curve...")
                    if use_star_for_black:
                        self.step_updated.emit(f"Extracting Black Curve from {os.path.basename(black_star_path)}")
                        out_txt = f"{out_dir_unix}/black.txt"
                        mask_name_path = None
                        
                        with open(black_star_path, 'r') as f_in, open(out_txt, 'w') as f_out:
                            in_fsc = False
                            for line in f_in:
                                stripped = line.strip()
                                if stripped.startswith("_rlnMaskName"):
                                    m_parts = stripped.split()
                                    if len(m_parts) >= 2:
                                        mask_name_path = m_parts[1].strip()
                                if stripped.startswith("data_fsc"):
                                    in_fsc = True
                                elif stripped.startswith("data_") and stripped != "data_fsc":
                                    in_fsc = False
                                elif in_fsc:
                                    parts = stripped.split()
                                    if len(parts) >= 4 and parts[0].isdigit():
                                        f_out.write(line)
                                        
                        if mask_name_path:
                            try:
                                # Safely extract directory path from MaskCreate relative strings
                                mask_dir = os.path.dirname(mask_name_path)
                                job_star_path = os.path.join(cwd, mask_dir, "job.star")
                                if os.path.exists(job_star_path):
                                    with open(job_star_path, 'r') as f_job:
                                        for line in f_job:
                                            if "helical_z_percentage" in line:
                                                z_parts = line.strip().split()
                                                if len(z_parts) >= 2:
                                                    parsed_z_percentage = f"{z_parts[1]}%"
                                                    break
                            except Exception as e:
                                print(f"Failed parsing helical_z_percentage: {e}")
                    else:
                        if use_helical_exp_for_black:
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_h1}" --fsc "{out_h2}" > "{out_dir_unix}/black.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{half1_path}" --fsc "{half2_path}" > "{out_dir_unix}/black.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if run_red:
                    red_script = f"{out_dir_unix}/chimera_red.py"
                    with open(red_script, "w") as f: f.write(f'import chimera\nfrom chimera import runCommand as rc\nrc("open #1 {out_dir_unix}/{red_pdb}")\nrc("open #2 {out_dir_unix}/hh_postprocess.mrc")\nrc("molmap #1 {res_red} modelId #3")\nrc("vop resample #3 onGrid #2 modelId #4")\nrc("volume #4 save {out_dir_unix}/molmap_post.mrc")\nrc("close session")\n')
                    self.step_updated.emit(f"Calculate Red Curve: Molmap - Res: {res_red}")
                    subprocess.run(f'module load {CHIMERA_VER} && chimera --nogui --script "{red_script}" && rm -f "{red_script}"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if run_blue_masking:
                    blue_script = f"{out_dir_unix}/chimera_blue.py"
                    with open(blue_script, "w") as f: f.write(f'import chimera\nfrom chimera import runCommand as rc\nrc("open #1 {out_dir_unix}/{blue_pdb}")\nrc("open #2 {out_dir_unix}/symEx_halfmap1.mrc")\nrc("molmap #1 {res_blue} modelId #3")\nrc("vop resample #3 onGrid #2 modelId #4")\nrc("volume #4 save {out_dir_unix}/molmap_halfmap1.mrc")\nrc("close session")\n')
                    self.step_updated.emit(f"Calculate Blue/Yellow Curve: Molmap - Res: {res_blue}")
                    subprocess.run(f'module load {CHIMERA_VER} && chimera --nogui --script "{blue_script}" && rm -f "{blue_script}"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                def run_comprehensive(color, in_molmap, in_map, l_ini, l_ext, l_soft, l_low):
                    comp_dir = f"{out_dir_unix}/Comprehensive_{color}"
                    os.makedirs(comp_dir, exist_ok=True)
                    log_file, job_q, params_log, j_count = f"{comp_dir}/params_map.json", queue.Queue(), {}, 0
                    
                    for ini in l_ini:
                        for ext in l_ext:
                            for soft in l_soft:
                                for low in l_low:
                                    j_count += 1; jid = f"{color.lower()}_{j_count}"
                                    params_log[jid] = f"Ini:{ini} Ext:{ext} Soft:{soft} Low:{low}"
                                    job_q.put((jid, ini, ext, soft, low))
                    
                    total_jobs = j_count
                    with open(log_file, "w") as fw: json.dump(params_log, fw)
                    
                    target_threads = [self.spin_threads.value()]
                    active_threads = [0]
                    jobs_done = [0]
                    lock = threading.Lock()
                    
                    def w(thread_idx):
                        import time
                        with lock: active_threads[0] += 1
                        t_mask, t_mult = f"{comp_dir}/temp_mask_t{thread_idx}.mrc", f"{comp_dir}/temp_mult_t{thread_idx}.mrc"
                        
                        while True:
                            with lock:
                                if active_threads[0] > target_threads[0]:
                                    active_threads[0] -= 1
                                    return # Self-terminate to step down thread count
                            
                            try: jid, ini, ext, soft, low = job_q.get_nowait()
                            except queue.Empty: break
                            
                            out_txt = f"{comp_dir}/{jid}.txt"
                            c1 = f'module load {RELION_VER} && relion_mask_create --i "{in_molmap}" --o "{t_mask}" --ini_threshold {ini} --extend_inimask {ext} --width_soft_edge {soft} --lowpass {low}'
                            c2 = f'relion_image_handler --i "{in_map}" --multiply "{t_mask}" --o "{t_mult}"'
                            c3 = f'relion_image_handler --i "{t_mult}" --fsc "{in_molmap}" > "{out_txt}"'
                            ret = subprocess.call(f"{c1} && {c2} && {c3}", shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                            if ret == 0 and os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
                                with lock:
                                    jobs_done[0] += 1
                                    pct = int((jobs_done[0] / total_jobs) * 100)
                                    if jobs_done[0] % 5 == 0 or jobs_done[0] == total_jobs:
                                        self.step_updated.emit(f"    -> {color} Progress: {jobs_done[0]}/{total_jobs} ({pct}%) [Threads: {active_threads[0]}]")
                            else:
                                try:
                                    if os.path.exists(out_txt): os.remove(out_txt)
                                except: pass
                                
                                with lock:
                                    if target_threads[0] > 1:
                                        target_threads[0] -= 1
                                        self.step_updated.emit(f'<span style="color: #FF5555;">Job {jid} terminated by cluster. Reducing active threads to {target_threads[0]}...</span>')
                                
                                job_q.put((jid, ini, ext, soft, low))
                                time.sleep(3.0) 
                                
                            try: os.remove(t_mask); os.remove(t_mult)
                            except: pass
                            job_q.task_done()
                            
                        with lock: active_threads[0] -= 1

                    th = []
                    init_threads = self.spin_threads.value()
                    for i in range(init_threads):
                        t = threading.Thread(target=w, args=(i,)); t.start(); th.append(t)
                    job_q.join()
                    for t in th: t.join()

                def auto_promote(color):
                    comp_dir = f"{out_dir_unix}/Comprehensive_{color}"; log_file = f"{comp_dir}/params_map.json"
                    if not os.path.exists(log_file): return None
                    with open(log_file, 'r') as f: params_map = json.load(f)
                    
                    job_scores, target_x = [], np.arange(0.1, 0.51, 0.05)
                    for jid in params_map.keys():
                        path = f"{comp_dir}/{jid}.txt"
                        if not os.path.exists(path): continue
                        try:
                            data = safe_read_fsc(path)
                            if data.ndim == 2 and data.shape[0] > 1:
                                x, y = data[:, 0], data[:, 1]
                                sort_idx = np.argsort(x); target_y = np.interp(target_x, x[sort_idx], y[sort_idx])
                                score = np.mean(target_y)
                            else: score = 0.0
                            job_scores.append((score, jid))
                        except: pass
                    
                    if not job_scores: return None
                    job_scores.sort(key=lambda x: x[0], reverse=True)
                    best_params_str = params_map[job_scores[0][1]]; params = {}
                    for p in best_params_str.split():
                        if ":" in p: k, v = p.split(":", 1); params[k] = v
                    ini, ext, soft, low = params.get("Ini"), params.get("Ext"), params.get("Soft"), params.get("Low")
                    
                    self.step_updated.emit(f"Auto-Promoting Best {color} Curve: Ini {ini}, Ext {ext}, Soft {soft}, Low {low}")
                    if color == "Red":
                        c1 = f'module load {RELION_VER} && relion_mask_create --i "{out_dir_unix}/molmap_post.mrc" --o "{out_dir_unix}/molmap_post_masked.mrc" --ini_threshold {ini} --extend_inimask {ext} --width_soft_edge {soft} --lowpass {low}'
                        subprocess.run(f'{c1} && relion_image_handler --i "{out_dir_unix}/hh_postprocess.mrc" --multiply "{out_dir_unix}/molmap_post_masked.mrc" --o "{out_dir_unix}/postprocess_multiplied.mrc" && relion_image_handler --i "{out_dir_unix}/postprocess_multiplied.mrc" --fsc "{out_dir_unix}/molmap_post.mrc" > "{out_dir_unix}/red.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        c1 = f'module load {RELION_VER} && relion_mask_create --i "{out_dir_unix}/molmap_halfmap1.mrc" --o "{out_dir_unix}/molmap_halfmap1_masked.mrc" --ini_threshold {ini} --extend_inimask {ext} --width_soft_edge {soft} --lowpass {low}'
                        cmds = [c1]
                        if run_blue:
                            cmds.append(f'relion_image_handler --i "{out_dir_unix}/symEx_halfmap1.mrc" --multiply "{out_dir_unix}/molmap_halfmap1_masked.mrc" --o "{out_dir_unix}/halfmap1_multiplied.mrc"')
                            cmds.append(f'relion_image_handler --i "{out_dir_unix}/halfmap1_multiplied.mrc" --fsc "{out_dir_unix}/molmap_halfmap1.mrc" > "{out_dir_unix}/blue.txt"')
                        if run_yellow:
                            cmds.append(f'relion_image_handler --i "{out_dir_unix}/symEx_halfmap2.mrc" --multiply "{out_dir_unix}/molmap_halfmap1_masked.mrc" --o "{out_dir_unix}/halfmap2_multiplied.mrc"')
                            cmds.append(f'relion_image_handler --i "{out_dir_unix}/halfmap2_multiplied.mrc" --fsc "{out_dir_unix}/molmap_halfmap1.mrc" > "{out_dir_unix}/yellow.txt"')
                        if cmds:
                            subprocess.run(" && ".join(cmds), shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return {"Initial Threshold": ini, "Extend Initial Mask": ext, "Soft Edge Width": soft, "Low Pass Filter": low}

                promoted_red, promoted_blue = None, None
                if run_red:
                    if red_jobs == 1:
                        self.step_updated.emit(f"Red: Creating Mask (Ini: {red_p[0]}, Ext: {red_p[1]}, Soft: {red_p[2]}, Low: {red_p[3]})...")
                        cmd_red_mask = f'module load {RELION_VER} && relion_mask_create --i "{out_dir_unix}/molmap_post.mrc" --o "{out_dir_unix}/molmap_post_masked.mrc" --ini_threshold {red_p[0]} --extend_inimask {red_p[1]} --width_soft_edge {red_p[2]} --lowpass {red_p[3]}'
                        subprocess.run(cmd_red_mask, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.step_updated.emit(f"Red: Multiplying PostProcess Map by Mask...")
                        subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/hh_postprocess.mrc" --multiply "{out_dir_unix}/molmap_post_masked.mrc" --o "{out_dir_unix}/postprocess_multiplied.mrc"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.step_updated.emit(f"Red: Calculating FSC Curve...")
                        subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/postprocess_multiplied.mrc" --fsc "{out_dir_unix}/molmap_post.mrc" > "{out_dir_unix}/red.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        self.step_updated.emit(f"Running Comprehensive Red Pipeline ({red_jobs} jobs)...")
                        run_comprehensive("Red", f"{out_dir_unix}/molmap_post.mrc", f"{out_dir_unix}/hh_postprocess.mrc", red_ini, red_ext, red_soft, red_low)
                        promoted_red = auto_promote("Red"); QMetaObject.invokeMethod(self.btn_red_vis, "setEnabled", Qt.ConnectionType.QueuedConnection, Q_ARG(bool, True))

                if run_blue_masking or run_yellow:
                    if run_blue_masking:
                        if blue_jobs == 1:
                            self.step_updated.emit(f"Blue/Yellow: Creating Mask (Ini: {blue_p[0]}, Ext: {blue_p[1]}, Soft: {blue_p[2]}, Low: {blue_p[3]})...")
                            cmd_blue_mask = f'module load {RELION_VER} && relion_mask_create --i "{out_dir_unix}/molmap_halfmap1.mrc" --o "{out_dir_unix}/molmap_halfmap1_masked.mrc" --ini_threshold {blue_p[0]} --extend_inimask {blue_p[1]} --width_soft_edge {blue_p[2]} --lowpass {blue_p[3]}'
                            subprocess.run(cmd_blue_mask, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            self.step_updated.emit(f"Running Comprehensive Blue Pipeline ({blue_jobs} jobs)...")
                            run_comprehensive("Blue", f"{out_dir_unix}/molmap_halfmap1.mrc", f"{out_dir_unix}/symEx_halfmap1.mrc", blue_ini, blue_ext, blue_soft, blue_low)
                            promoted_blue = auto_promote("Blue"); QMetaObject.invokeMethod(self.btn_blue_vis, "setEnabled", Qt.ConnectionType.QueuedConnection, Q_ARG(bool, True))
                            
                    if blue_jobs == 1 or not run_blue_masking:
                        if run_blue:
                            self.step_updated.emit(f"Blue: Multiplying Halfmap 1 by Mask...")
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/symEx_halfmap1.mrc" --multiply "{out_dir_unix}/molmap_halfmap1_masked.mrc" --o "{out_dir_unix}/halfmap1_multiplied.mrc"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            self.step_updated.emit(f"Blue: Calculating Blue FSC Curve...")
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/halfmap1_multiplied.mrc" --fsc "{out_dir_unix}/molmap_halfmap1.mrc" > "{out_dir_unix}/blue.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if run_yellow:
                            self.step_updated.emit(f"Yellow: Multiplying Halfmap 2 by Mask...")
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/symEx_halfmap2.mrc" --multiply "{out_dir_unix}/molmap_halfmap1_masked.mrc" --o "{out_dir_unix}/halfmap2_multiplied.mrc"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            self.step_updated.emit(f"Yellow: Calculating Yellow FSC Curve...")
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/halfmap2_multiplied.mrc" --fsc "{out_dir_unix}/molmap_halfmap1.mrc" > "{out_dir_unix}/yellow.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                log_path = os.path.join(out_dir, "xcop_log.txt")
                existing_lines = []; skip = False
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f: existing_lines = f.readlines()
                new_lines = []
                for line in existing_lines:
                    if line.strip() in ["Symmetry Expansion:", "Red:", "Blue:", "Black:", "Yellow:"]: skip = True
                    if skip and line.strip() == "":
                        skip = False; continue
                    if not skip: new_lines.append(line)

                log_content = ["Symmetry Expansion:\n", f"- Pixel Size: {pix}\n", f"- Tube Diameter: {diam}\n", f"- Z Percentage: {z_perc_str}\n"]
                if need_refine:
                    log_content.extend([f"- Half 1 Map: {half1_path}\n", f"- Half 1 Twist: {twist_h1}\n", f"- Half 1 Rise: {rise_h1}\n",
                                        f"- Half 2 Map: {half2_path}\n", f"- Half 2 Twist: {twist_h2}\n", f"- Half 2 Rise: {rise_h2}\n"])
                if need_post:
                    log_content.extend([f"- PostProcess Map: {post_path}\n", f"- PostProcess Twist: {twist_avg}\n", f"- PostProcess Rise: {rise_avg}\n\n"])
                
                if run_red:
                    log_content.extend(["Red:\n", f"- Model: {red_pdb}\n", f"- Resolution: {res_red}\n"])
                    if red_jobs == 1: log_content.extend([f"- Initial Threshold: {red_p[0]}\n", f"- Extend Initial Mask: {red_p[1]}\n", f"- Soft Edge Width: {red_p[2]}\n", f"- Low Pass Filter: {red_p[3]}\n\n"])
                    elif promoted_red: log_content.extend([f"- Initial Threshold: {promoted_red['Initial Threshold']}\n", f"- Extend Initial Mask: {promoted_red['Extend Initial Mask']}\n", f"- Soft Edge Width: {promoted_red['Soft Edge Width']}\n", f"- Low Pass Filter: {promoted_red['Low Pass Filter']}\n\n"])
                    else: log_content.extend([f"- Initial Threshold: (Comprehensive mode pending Promotion)\n", f"- Extend Initial Mask: -\n", f"- Soft Edge Width: -\n", f"- Low Pass Filter: -\n\n"])

                if run_blue or run_yellow:
                    log_content.extend(["Blue (Mask Source):\n", f"- Model: {blue_pdb}\n", f"- Resolution: {res_blue}\n"])
                    if blue_jobs == 1: log_content.extend([f"- Initial Threshold: {blue_p[0]}\n", f"- Extend Initial Mask: {blue_p[1]}\n", f"- Soft Edge Width: {blue_p[2]}\n", f"- Low Pass Filter: {blue_p[3]}\n\n"])
                    elif promoted_blue: log_content.extend([f"- Initial Threshold: {promoted_blue['Initial Threshold']}\n", f"- Extend Initial Mask: {promoted_blue['Extend Initial Mask']}\n", f"- Soft Edge Width: {promoted_blue['Soft Edge Width']}\n", f"- Low Pass Filter: {promoted_blue['Low Pass Filter']}\n\n"])
                    else: log_content.extend([f"- Initial Threshold: (Comprehensive mode pending Promotion)\n", f"- Extend Initial Mask: -\n", f"- Soft Edge Width: -\n", f"- Low Pass Filter: -\n\n"])

                if run_black:
                    log_content.extend(["Black:\n"])
                    if use_star_for_black:
                        log_content.extend([
                            f"- Input Map: {os.path.abspath(black_star_path)}\n",
                            f"- Z percentage: {parsed_z_percentage}\n\n"
                        ])
                    else:
                        if use_helical_exp_for_black:
                            log_content.extend([
                                f"- Input Map 1: {out_dir_unix}/symEx_halfmap1.mrc\n", 
                                f"- Input Map 2: {out_dir_unix}/symEx_halfmap2.mrc\n\n"
                            ])
                        else:
                            log_content.extend([
                                f"- Input Map 1: {half1_path}\n", 
                                f"- Input Map 2: {half2_path}\n\n"
                            ])

                if run_yellow:
                    log_content.extend([
                        "Yellow:\n", 
                        f"- Input Map: {out_dir_unix}/symEx_halfmap2.mrc\n", 
                        f"- Mask Used: {out_dir_unix}/molmap_halfmap1_masked.mrc\n\n"
                    ])

                with open(log_path, 'w') as f: f.writelines(log_content); f.writelines(new_lines)
                self.step_updated.emit("Pipeline Finished Successfully.")
            except Exception as e: self.step_updated.emit(f'<span style="color: red;">Pipeline Error: {e}</span>')
            finally: self.pipeline_finished.emit()

        threading.Thread(target=pipeline_worker, daemon=True).start()


class NonRelionAutoApp(BaseAutoApp):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fully Automated FSC Curve for Non-RELION Directory")
        self.pipeline_finished.connect(self._reset_run_button)
        self.step_updated.connect(self.append_log_step)
        self.resize(1150, 750)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        outer_layout = QHBoxLayout(central_widget)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer_layout.addWidget(splitter)
        
        left_widget = QWidget(); left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(15, 15, 10, 15); left_layout.setSpacing(10)
        
        right_widget = QWidget(); right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(10, 15, 15, 15); right_layout.setSpacing(10)
        
        splitter.addWidget(left_widget); splitter.addWidget(right_widget); splitter.setSizes([650, 500])

        def create_aligned_label(text, color=None):
            lbl = QLabel(text)
            lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            if color: lbl.setStyleSheet(f"color: {color};")
            return lbl

        def create_grey_label(text):
            lbl = QLabel(text); lbl.setStyleSheet("color: #8a8a8a;"); return lbl

        input_group = QGroupBox("Core Inputs"); grid = QGridLayout(input_group); grid.setSpacing(8)

        # Output Directory
        grid.addWidget(create_aligned_label("<span style='color: #56b6c2;'><b>Output Dir:</b></span>"), 0, 1)
        self.out_dir_entry = QLineEdit(); self.out_dir_entry.setPlaceholderText("Select directory to save all outputs...")
        grid.addWidget(self.out_dir_entry, 0, 2, 1, 3)
        btn_out = QPushButton("Browse"); btn_out.clicked.connect(lambda: self.browse_path(self.out_dir_entry, is_dir=True))
        grid.addWidget(btn_out, 0, 5)

        from PyQt6.QtWidgets import QRadioButton, QButtonGroup

        def create_separator():
            sep = QLabel()
            sep.setFixedHeight(1)
            sep.setStyleSheet("background-color: #3e3e42; margin-top: 4px; margin-bottom: 4px;")
            return sep

        # Halfmap 1
        self.lbl_h1_preview = HoverImageLabel()
        self.lbl_h1_preview.setFixedSize(55, 55)
        self.lbl_h1_preview.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px;")
        self.lbl_h1_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_h1_preview, 1, 0, 3, 1, Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_h1_title = create_aligned_label("Halfmap1:")
        grid.addWidget(self.lbl_h1_title, 1, 1)
        self.h1_entry = QLineEdit(); grid.addWidget(self.h1_entry, 1, 2, 1, 3)
        self.btn_h1 = QPushButton("Browse"); self.btn_h1.clicked.connect(lambda: self.browse_path(self.h1_entry))
        grid.addWidget(self.btn_h1, 1, 5)

        self.rb_h1_hexp = QRadioButton("Non-expanded")
        self.rb_h1_already = QRadioButton("Already Helical Expanded")
        self.bg_h1 = QButtonGroup(self)
        self.bg_h1.addButton(self.rb_h1_hexp)
        self.bg_h1.addButton(self.rb_h1_already)
        self.rb_h1_hexp.setChecked(True)
        
        h1_rb_layout = QHBoxLayout(); h1_rb_layout.setContentsMargins(0, 0, 0, 0)
        h1_rb_layout.addWidget(self.rb_h1_hexp); h1_rb_layout.addWidget(self.rb_h1_already); h1_rb_layout.addStretch()
        grid.addLayout(h1_rb_layout, 2, 2, 1, 4)
        
        grid.addWidget(create_separator(), 4, 0, 1, 6)

        # Halfmap 2
        self.lbl_h2_preview = HoverImageLabel()
        self.lbl_h2_preview.setFixedSize(55, 55)
        self.lbl_h2_preview.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px;")
        self.lbl_h2_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_h2_preview, 5, 0, 3, 1, Qt.AlignmentFlag.AlignCenter)
        
        self.lbl_h2_title = create_aligned_label("Halfmap2:")
        grid.addWidget(self.lbl_h2_title, 5, 1)
        self.h2_entry = QLineEdit(); grid.addWidget(self.h2_entry, 5, 2, 1, 3)
        self.btn_h2 = QPushButton("Browse"); self.btn_h2.clicked.connect(lambda: self.browse_path(self.h2_entry))
        grid.addWidget(self.btn_h2, 5, 5)

        self.rb_h2_hexp = QRadioButton("Non-expanded")
        self.rb_h2_already = QRadioButton("Already Helical Expanded")
        self.bg_h2 = QButtonGroup(self)
        self.bg_h2.addButton(self.rb_h2_hexp)
        self.bg_h2.addButton(self.rb_h2_already)
        self.rb_h2_hexp.setChecked(True)
        
        h2_rb_layout = QHBoxLayout(); h2_rb_layout.setContentsMargins(0, 0, 0, 0)
        h2_rb_layout.addWidget(self.rb_h2_hexp); h2_rb_layout.addWidget(self.rb_h2_already); h2_rb_layout.addStretch()
        grid.addLayout(h2_rb_layout, 6, 2, 1, 4)

        grid.addWidget(create_separator(), 8, 0, 1, 6)

        # PostProcess Map
        self.lbl_post_preview = HoverImageLabel()
        self.lbl_post_preview.setFixedSize(55, 55)
        self.lbl_post_preview.setStyleSheet("background-color: #1e1e1e; border: 1px solid #444; border-radius: 4px;")
        self.lbl_post_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        grid.addWidget(self.lbl_post_preview, 9, 0, 3, 1, Qt.AlignmentFlag.AlignCenter)
        
        grid.addWidget(create_aligned_label("PostProcess:"), 9, 1)
        self.post_entry = QLineEdit(); grid.addWidget(self.post_entry, 9, 2, 1, 3)
        self.btn_post = QPushButton("Browse"); self.btn_post.clicked.connect(lambda: self.browse_path(self.post_entry))
        grid.addWidget(self.btn_post, 9, 5)

        self.rb_post_hexp = QRadioButton("Non-expanded")
        self.rb_post_already = QRadioButton("Already Helical Expanded")
        self.bg_post = QButtonGroup(self)
        self.bg_post.addButton(self.rb_post_hexp)
        self.bg_post.addButton(self.rb_post_already)
        self.rb_post_hexp.setChecked(True)
        
        post_rb_layout = QHBoxLayout(); post_rb_layout.setContentsMargins(0, 0, 0, 0)
        post_rb_layout.addWidget(self.rb_post_hexp); post_rb_layout.addWidget(self.rb_post_already); post_rb_layout.addStretch()
        grid.addLayout(post_rb_layout, 10, 2, 1, 4)

        grid.addWidget(create_separator(), 12, 0, 1, 6)

        # Other Parameters
        self.lbl_global_twist = create_aligned_label("Helical Twist:")
        grid.addWidget(self.lbl_global_twist, 13, 1)
        self.global_twist_entry = QLineEdit()
        self.lbl_global_rise = create_aligned_label("Rise (Å):")
        self.global_rise_entry = QLineEdit()
        twist_rise_layout = QHBoxLayout(); twist_rise_layout.setContentsMargins(0,0,0,0)
        twist_rise_layout.addWidget(self.global_twist_entry); twist_rise_layout.addWidget(self.lbl_global_rise); twist_rise_layout.addWidget(self.global_rise_entry)
        grid.addLayout(twist_rise_layout, 13, 2, 1, 4)

        self.lbl_global_res = create_aligned_label("Resolution (Å):")
        grid.addWidget(self.lbl_global_res, 14, 1)
        self.global_res_entry = QLineEdit()
        grid.addWidget(self.global_res_entry, 14, 2, 1, 4)

        grid.addWidget(create_aligned_label("Pixel Size (angpix):"), 15, 1)
        self.pixel_size_combo = QComboBox(); self.pixel_size_combo.addItems(["0.9557", "1.9114", "2.8671", "3.8228", "4.7785", "Custom"])
        self.pixel_size_custom = QLineEdit(); self.pixel_size_custom.setVisible(False)
        pix_layout = QHBoxLayout(); pix_layout.setContentsMargins(0, 0, 0, 0); pix_layout.addWidget(self.pixel_size_combo, 1); pix_layout.addWidget(self.pixel_size_custom, 2)
        grid.addLayout(pix_layout, 15, 2, 1, 4)
        self.pixel_size_combo.currentTextChanged.connect(lambda text: self.pixel_size_custom.setVisible(text == "Custom"))

        self.lbl_tube_diam = create_aligned_label("Tube Diameter (Å):"); grid.addWidget(self.lbl_tube_diam, 16, 1)
        self.tube_diam_entry = QLineEdit("200"); grid.addWidget(self.tube_diam_entry, 16, 2, 1, 4)
        
        self.lbl_z_perc = create_aligned_label("Z Percentage (%):"); grid.addWidget(self.lbl_z_perc, 17, 1)
        self.z_perc_entry = QLineEdit("30"); grid.addWidget(self.z_perc_entry, 17, 2, 1, 4)
        left_layout.addWidget(input_group)
        
        # Connect map entry signals to trigger preview updates
        self.h1_entry.textChanged.connect(lambda: self.update_preview(self.h1_entry, self.lbl_h1_preview))
        self.h2_entry.textChanged.connect(lambda: self.update_preview(self.h2_entry, self.lbl_h2_preview))
        self.post_entry.textChanged.connect(lambda: self.update_preview(self.post_entry, self.lbl_post_preview))

        def update_hexp_visibility():
            h1_hexp = self.rb_h1_hexp.isChecked()
            h2_hexp = self.rb_h2_hexp.isChecked()
            post_hexp = self.rb_post_hexp.isChecked()
            
            any_hexp = h1_hexp or h2_hexp or post_hexp
            
            self.global_twist_entry.setVisible(any_hexp); self.lbl_global_twist.setVisible(any_hexp)
            self.global_rise_entry.setVisible(any_hexp); self.lbl_global_rise.setVisible(any_hexp)
            self.tube_diam_entry.setVisible(any_hexp); self.lbl_tube_diam.setVisible(any_hexp)
            self.z_perc_entry.setVisible(any_hexp); self.lbl_z_perc.setVisible(any_hexp)

        self.rb_h1_hexp.toggled.connect(update_hexp_visibility)
        self.rb_h2_hexp.toggled.connect(update_hexp_visibility)
        self.rb_post_hexp.toggled.connect(update_hexp_visibility)
        
        update_hexp_visibility()

        # Log Panel
        lbl_log = QLabel("Live Execution Log"); lbl_log.setStyleSheet("font-weight: bold; color: #56b6c2; margin-top: 5px;")
        left_layout.addWidget(lbl_log)
        self.log_panel = QTextEdit(); self.log_panel.setReadOnly(True); self.log_panel.setAttribute(Qt.WidgetAttribute.WA_MacShowFocusRect, False)
        self.log_panel.setStyleSheet("QTextEdit { background-color: #1e1e1e; font-family: monospace; border: 1px solid #3e3e42; border-radius: 4px; padding: 5px; selection-background-color: #56b6c2; selection-color: #1e1e1e; } QTextEdit:focus { outline: none; }")
        left_layout.addWidget(self.log_panel, stretch=1)

        from PyQt6.QtWidgets import QRadioButton
        self.black_group = QGroupBox("Black Curve (Map) - Halfmap 1 vs Halfmap 2"); self.black_group.setCheckable(True); self.black_group.setChecked(True)
        black_grid = QGridLayout(self.black_group)
        self.rb_black_helical = QRadioButton("Use Original Unfiltered Halfmaps")
        self.rb_black_helical.setChecked(True)
        self.rb_black_helical_exp = QRadioButton("Use Helical Expanded Maps")
        self.rb_black_star = QRadioButton("Use Postprocess.star from RELION")
        black_grid.addWidget(self.rb_black_helical, 0, 0, 1, 2)
        black_grid.addWidget(self.rb_black_helical_exp, 1, 0, 1, 2)
        black_grid.addWidget(self.rb_black_star, 2, 0, 1, 2)
        
        self.black_star_entry = QLineEdit(); self.black_star_entry.setPlaceholderText("Select postprocess.star..."); self.black_star_entry.setEnabled(False)
        self.btn_browse_black_star = QPushButton("Browse"); self.btn_browse_black_star.setEnabled(False)
        self.btn_browse_black_star.clicked.connect(lambda: self.browse_path(self.black_star_entry, filt="STAR Files (*.star);;All Files (*)"))
        black_grid.addWidget(self.black_star_entry, 3, 0); black_grid.addWidget(self.btn_browse_black_star, 3, 1)
        def toggle_black_star_nr(checked):
            self.black_star_entry.setEnabled(checked)
            self.btn_browse_black_star.setEnabled(checked)
            self.update_section_states()
        self.rb_black_star.toggled.connect(toggle_black_star_nr)
        
        right_layout.addWidget(self.black_group, stretch=1)

        self.red_group = QGroupBox("Red Curve (Full) - Model vs PostProcess"); self.red_group.setCheckable(True); self.red_group.setChecked(True)
        red_grid = QGridLayout(self.red_group)
        red_grid.addWidget(QLabel("Final Model (Red):"), 0, 0); self.red_pdb_entry = QLineEdit(); self.red_pdb_entry.setPlaceholderText("e.g. model.pdb")
        red_grid.addWidget(self.red_pdb_entry, 0, 1, 1, 3)
        self.btn_browse_red = QPushButton("Browse"); self.btn_browse_red.setFixedWidth(160); self.btn_browse_red.clicked.connect(lambda: self.browse_path(self.red_pdb_entry, filt="PDB Files (*.pdb *.cif);;All Files (*)"))
        red_grid.addWidget(self.btn_browse_red, 0, 4, 1, 1)
        self.red_params, self.lbl_red_jobs, self.btn_red_vis, self.chk_red_multi = self.create_fsc_params(red_grid, defaults=["0.1", "1", "1", "5"], row=1, color="Red")
        right_layout.addWidget(self.red_group, stretch=1) 

        self.blue_group = QGroupBox("Blue Curve (Work) - Model vs Halfmap 1"); self.blue_group.setCheckable(True); self.blue_group.setChecked(True)
        blue_grid = QGridLayout(self.blue_group)
        blue_grid.addWidget(QLabel("Final Model (Blue):"), 0, 0); self.blue_pdb_entry = QLineEdit(); self.blue_pdb_entry.setPlaceholderText("e.g. model_blue.pdb")
        blue_grid.addWidget(self.blue_pdb_entry, 0, 1, 1, 3)
        self.btn_browse_blue = QPushButton("Browse"); self.btn_browse_blue.setFixedWidth(160); self.btn_browse_blue.clicked.connect(lambda: self.browse_path(self.blue_pdb_entry, filt="PDB Files (*.pdb *.cif);;All Files (*)"))
        blue_grid.addWidget(self.btn_browse_blue, 0, 4, 1, 1)
        self.blue_params, self.lbl_blue_jobs, self.btn_blue_vis, self.chk_blue_multi = self.create_fsc_params(blue_grid, defaults=["0.1", "1", "1", "10"], row=1, color="Blue")
        right_layout.addWidget(self.blue_group, stretch=1) 

        self.yellow_group = QGroupBox("Yellow Curve (Free) - Halfmap 1 vs Halfmap 2"); self.yellow_group.setCheckable(True); self.yellow_group.setChecked(True)
        yellow_grid = QGridLayout(self.yellow_group)
        lbl_yellow_desc = QLabel("Use Halfmap 2 and the Mask Generated by the Blue Curve")
        lbl_yellow_desc.setStyleSheet("color: #8a8a8a;")
        yellow_grid.addWidget(lbl_yellow_desc, 0, 0)
        right_layout.addWidget(self.yellow_group, stretch=1) 

        self.thread_widget = QWidget()
        thread_layout = QHBoxLayout(self.thread_widget)
        thread_layout.setContentsMargins(0, 0, 0, 0)
        thread_layout.addWidget(QLabel("Parallel Threads:"))
        self.spin_threads = QSpinBox(); self.spin_threads.setRange(1, 32); self.spin_threads.setValue(3)
        thread_layout.addWidget(self.spin_threads)

        lbl_thread_warn = QLabel("More threads may exceed HPC CPU policy, but xcop will handle it smartly")
        lbl_thread_warn.setStyleSheet("color: #FF5555; font-size: 9pt;")
        lbl_thread_warn.setVisible(False)
        thread_layout.addWidget(lbl_thread_warn)
        thread_layout.addStretch()
        
        right_layout.addWidget(self.thread_widget)
        self.thread_widget.setVisible(False)

        def check_thread_warn(val): lbl_thread_warn.setVisible(val > 5)
        self.spin_threads.valueChanged.connect(check_thread_warn)
        check_thread_warn(self.spin_threads.value())

        def update_thread_visibility():
            self.thread_widget.setVisible(self.chk_red_multi.isChecked() or self.chk_blue_multi.isChecked())
        
        self.chk_red_multi.toggled.connect(lambda _: update_thread_visibility())
        self.chk_blue_multi.toggled.connect(lambda _: update_thread_visibility())

        self.btn_run = QPushButton("RUN"); self.btn_run.setFixedHeight(40); self.btn_run.setStyleSheet("font-weight: bold; background-color: #56b6c2; color: #1e1e1e;")
        self.btn_run.clicked.connect(self.run_expansion)
        right_layout.addWidget(self.btn_run)

        self.btn_main_vis = QPushButton("Visualize Curves"); self.btn_main_vis.setFixedHeight(40)
        self.btn_main_vis.setEnabled(False)
        self.btn_main_vis.clicked.connect(self.launch_main_vis)
        right_layout.addWidget(self.btn_main_vis)
        
        self.out_dir_entry.textChanged.connect(lambda text: self.check_visualizer_availability())

        from PyQt6.QtWidgets import QSizePolicy
        for group in [self.black_group, self.red_group, self.blue_group, self.yellow_group]:
            group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        self.black_group.toggled.connect(lambda _: self.update_section_states())
        self.red_group.toggled.connect(lambda _: self.update_section_states())
        self.blue_group.toggled.connect(lambda _: self.update_section_states())
        self.yellow_group.toggled.connect(lambda _: self.update_section_states())
        
        self.rb_h1_already.clicked.connect(lambda _: self.update_section_states())
        self.rb_h2_already.clicked.connect(lambda _: self.update_section_states())
        self.rb_h1_hexp.clicked.connect(lambda _: self.update_section_states())
        self.rb_h2_hexp.clicked.connect(lambda _: self.update_section_states())
        self.rb_black_helical.clicked.connect(lambda _: self.update_section_states())
        self.rb_black_helical_exp.clicked.connect(lambda _: self.update_section_states())
        
        self.update_section_states()

        # Apply cyan theme strictly to Non-RELION mode widgets
        self.setStyleSheet(QApplication.instance().styleSheet().replace('#98c379', '#56b6c2'))

    def update_section_states(self):
        run_yellow = self.yellow_group.isChecked()
        run_blue = self.blue_group.isChecked()
        
        # Enforce Yellow -> Blue dependency
        if run_yellow and not run_blue:
            out_dir = self.out_dir_entry.text().strip()
            mask_path = os.path.join(out_dir, "molmap_halfmap1_masked.mrc") if out_dir else ""
            
            if not out_dir or not os.path.exists(mask_path):
                self.blue_group.blockSignals(True)
                self.blue_group.setChecked(True)
                self.blue_group.blockSignals(False)
                run_blue = True
                self.step_updated.emit("Required mask for Yellow not found. Auto-selecting Blue Curve.")

        run_black = self.black_group.isChecked()
        use_star = self.rb_black_star.isChecked()
        use_helical = self.rb_black_helical.isChecked()
        use_helical_exp = self.rb_black_helical_exp.isChecked()
        run_red = self.red_group.isChecked()

        if self.rb_h1_already.isChecked() or self.rb_h2_already.isChecked():
            if self.rb_black_helical.isChecked():
                self.rb_black_helical_exp.setChecked(True)
            self.rb_black_helical.setEnabled(False)
        else:
            self.rb_black_helical.setEnabled(True)

        if run_black and self.rb_black_helical.isChecked():
            self.h1_entry.setPlaceholderText("Non-helical expanded unfiltered halfmaps")
            self.h2_entry.setPlaceholderText("Non-helical expanded unfiltered halfmaps")
            if hasattr(self, 'lbl_h1_title'): self.lbl_h1_title.setText("Unfiltered Halfmap1:")
            if hasattr(self, 'lbl_h2_title'): self.lbl_h2_title.setText("Unfiltered Halfmap2:")
            
            if run_blue: self.rb_h1_hexp.setChecked(True)
            if run_yellow: self.rb_h2_hexp.setChecked(True)
        elif run_black and self.rb_black_helical_exp.isChecked():
            self.h1_entry.setPlaceholderText("Helical expanded halfmap 1")
            self.h2_entry.setPlaceholderText("Helical expanded halfmap 2")
            if hasattr(self, 'lbl_h1_title'): self.lbl_h1_title.setText("Expanded Halfmap1:")
            if hasattr(self, 'lbl_h2_title'): self.lbl_h2_title.setText("Expanded Halfmap2:")
        else:
            self.h1_entry.setPlaceholderText("")
            self.h2_entry.setPlaceholderText("")
            if hasattr(self, 'lbl_h1_title'): self.lbl_h1_title.setText("Halfmap1:")
            if hasattr(self, 'lbl_h2_title'): self.lbl_h2_title.setText("Halfmap2:")

        if not run_red: self.chk_red_multi.setChecked(False)
        if not run_blue: self.chk_blue_multi.setChecked(False)

        need_h1 = run_blue or run_yellow or (run_black and not use_star)
        need_h2 = run_yellow or (run_black and not use_star)
        need_post = run_red

        for w in [self.h1_entry, self.btn_h1, self.rb_h1_hexp, self.rb_h1_already, 
                  self.lbl_h1_preview]:
            w.setEnabled(need_h1)

        for w in [self.h2_entry, self.btn_h2, self.rb_h2_hexp, self.rb_h2_already, 
                  self.lbl_h2_preview]:
            w.setEnabled(need_h2)

        for w in [self.post_entry, self.btn_post, self.rb_post_hexp, self.rb_post_already, 
                  self.lbl_post_preview]:
            w.setEnabled(need_post)
            
        any_needed = need_h1 or need_h2 or need_post
        self.global_res_entry.setEnabled(any_needed)
        self.global_twist_entry.setEnabled(any_needed)
        self.global_rise_entry.setEnabled(any_needed)

    def mousePressEvent(self, event):
        focused_widget = QApplication.focusWidget()
        if isinstance(focused_widget, QLineEdit): focused_widget.clearFocus()
        super().mousePressEvent(event)

    def browse_path(self, entry_widget, is_dir=False, filt="All Files (*)"):
        path = ""
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Select Directory", os.getcwd())
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Select File", os.getcwd(), filt)
        if path: entry_widget.setText(path)

    def update_preview(self, entry_widget, preview_label):
        mrc_path = entry_widget.text().strip()
        self.request_mrc_preview(mrc_path, preview_label)

    def check_visualizer_availability(self):
        out_dir = self.out_dir_entry.text().strip()
        if not out_dir:
            self.btn_red_vis.setVisible(False); self.btn_blue_vis.setVisible(False)
            self.btn_main_vis.setEnabled(False)
            return
            
        red_log = os.path.join(out_dir, "Comprehensive_Red", "params_map.json")
        self.btn_red_vis.setVisible(os.path.exists(red_log)); self.btn_red_vis.setEnabled(os.path.exists(red_log))
        blue_log = os.path.join(out_dir, "Comprehensive_Blue", "params_map.json")
        self.btn_blue_vis.setVisible(os.path.exists(blue_log)); self.btn_blue_vis.setEnabled(os.path.exists(blue_log))
        
        has_txt = any(os.path.exists(os.path.join(out_dir, f"{c}.txt")) for c in ("black", "red", "blue", "yellow"))
        self.btn_main_vis.setEnabled(has_txt)

    def launch_main_vis(self):
        out_dir = self.out_dir_entry.text().strip()
        if out_dir:
            files = [os.path.join(out_dir, f"{c}.txt") for c in ("black", "red", "blue", "yellow")]
            if any(os.path.exists(f) for f in files):
                generate_fsc_curves(auto_files=files, auto_model=os.path.basename(out_dir), auto_export_dir=out_dir)

    def open_visualizer(self, color):
        out_dir = self.out_dir_entry.text().strip()
        comp_dir = os.path.join(out_dir, f"Comprehensive_{color}")
        log_file = os.path.join(comp_dir, "params_map.json")
        if not os.path.exists(log_file): return QMessageBox.critical(self, "Error", "No comprehensive data found. Run pipeline first.")
        self.vis_win = ComprehensivePlotWindow(comp_dir, log_file); self.vis_win.show()

    def run_expansion(self):
        out_dir = self.out_dir_entry.text().strip()
        h1_path, h2_path, post_path = self.h1_entry.text().strip(), self.h2_entry.text().strip(), self.post_entry.text().strip()
        res_global = self.global_res_entry.text().strip()
        res_blue, res_red = res_global, res_global
        red_pdb_raw, blue_pdb_raw = self.red_pdb_entry.text().strip(), self.blue_pdb_entry.text().strip()
        use_star_for_black = self.rb_black_star.isChecked()
        use_helical_exp_for_black = self.rb_black_helical_exp.isChecked()
        black_star_path = self.black_star_entry.text().strip()
        
        do_h1_hexp = self.rb_h1_hexp.isChecked()
        do_h2_hexp = self.rb_h2_hexp.isChecked()
        do_post_hexp = self.rb_post_hexp.isChecked()

        twist_h1, rise_h1, twist_h2, rise_h2, twist_avg, rise_avg, pix, diam, z_perc_str = "", "", "", "", "", "", self.get_pixel_size(), "", ""
        z_perc = 0.0

        run_black = self.black_group.isChecked()
        run_red = self.red_group.isChecked()
        run_blue = self.blue_group.isChecked()
        run_yellow = self.yellow_group.isChecked()

        run_blue_masking = run_blue
        if run_yellow and not run_blue:
            mask_path = os.path.join(out_dir, "molmap_halfmap1_masked.mrc")
            if not os.path.exists(mask_path):
                run_blue_masking = True
                self.step_updated.emit("Required mask for Yellow not found. Will generate Blue mask automatically.")

        required_fields = [out_dir, pix]
        need_h1 = run_blue or run_yellow or (run_black and not use_star_for_black)
        need_h2 = run_yellow or (run_black and not use_star_for_black)
        need_post = run_red

        any_hexp = (need_h1 and do_h1_hexp) or (need_h2 and do_h2_hexp) or (need_post and do_post_hexp)
        
        if any_hexp:
            global_twist = self.global_twist_entry.text().strip()
            global_rise = self.global_rise_entry.text().strip()
            twist_h1 = twist_h2 = twist_avg = global_twist
            rise_h1 = rise_h2 = rise_avg = global_rise
            
            diam, z_perc_str = self.tube_diam_entry.text().strip(), self.z_perc_entry.text().strip()
            required_fields.extend([global_twist, global_rise, diam, z_perc_str])
            
        if need_h1:
            required_fields.append(h1_path)
            if run_blue_masking: required_fields.extend([res_blue, blue_pdb_raw])
        if need_h2:
            required_fields.append(h2_path)
        if need_post:
            required_fields.extend([post_path, res_red, red_pdb_raw])
            try: z_perc = float(z_perc_str) / 100.0
            except ValueError: return QMessageBox.critical(self, "Error", "Z Percentage must be a valid number.")

        if not all(required_fields): return QMessageBox.critical(self, "Error", "Please fill in all required parameters for the selected curves.")
        if run_red:
            pdb_check = red_pdb_raw if os.path.isabs(red_pdb_raw) else os.path.join(out_dir, red_pdb_raw)
            if not os.path.exists(pdb_check) and not os.path.exists(red_pdb_raw): return QMessageBox.critical(self, "Error", f"Red PDB file not found:\n{red_pdb_raw}")
        if run_blue_masking:
            pdb_check = blue_pdb_raw if os.path.isabs(blue_pdb_raw) else os.path.join(out_dir, blue_pdb_raw)
            if not os.path.exists(pdb_check) and not os.path.exists(blue_pdb_raw): return QMessageBox.critical(self, "Error", f"Blue PDB file not found:\n{blue_pdb_raw}")
        if run_black and use_star_for_black and not os.path.exists(black_star_path): return QMessageBox.critical(self, "Error", "Please select a valid postprocess.star file for the Black curve.")

        def parse_ranges(entries_dict):
            def get_vals(s, e, st, is_float):
                try:
                    start, end, step = float(s.text()), float(e.text()), float(st.text())
                    if step <= 0: raise ValueError
                    vals = np.arange(start, end + (step/1000.0), step)
                    return [str(round(x, 3)) if is_float else str(int(round(x))) for x in vals]
                except: return [s.text().strip()]
            return (get_vals(*entries_dict["Ini threshold:"], True), get_vals(*entries_dict["Extend inimask:"], False), get_vals(*entries_dict["Soft edge width:"], False), get_vals(*entries_dict["Low-pass filter:"], True))

        red_ini, red_ext, red_soft, red_low = parse_ranges(self.red_params)
        blue_ini, blue_ext, blue_soft, blue_low = parse_ranges(self.blue_params)
        red_jobs = len(red_ini) * len(red_ext) * len(red_soft) * len(red_low)
        blue_jobs = len(blue_ini) * len(blue_ext) * len(blue_soft) * len(blue_low)
        red_p, blue_p = [red_ini[0], red_ext[0], red_soft[0], red_low[0]], [blue_ini[0], blue_ext[0], blue_soft[0], blue_low[0]]

        os.makedirs(out_dir, exist_ok=True)
        def prepare_pdb(pdb_path):
            if not pdb_path: return ""
            abs_src = pdb_path if os.path.isabs(pdb_path) else os.path.join(out_dir, pdb_path)
            if not os.path.exists(abs_src) and os.path.exists(os.path.join(os.getcwd(), pdb_path)):
                abs_src = os.path.join(os.getcwd(), pdb_path)
                
            basename = os.path.basename(abs_src)
            target_path = os.path.join(out_dir, basename)
            if os.path.abspath(abs_src) != os.path.abspath(target_path) and os.path.exists(abs_src):
                try: shutil.copy2(abs_src, target_path)
                except Exception as e: print(f"Failed to copy {basename}: {e}")
            return basename

        red_pdb, blue_pdb = prepare_pdb(red_pdb_raw), prepare_pdb(blue_pdb_raw)

        files_to_check = []
        if need_h1: files_to_check.append(h1_path)
        if need_h2: files_to_check.append(h2_path)
        if need_post: files_to_check.append(post_path)
        missing_files = [f for f in files_to_check if not os.path.exists(f)]
        if missing_files: return QMessageBox.critical(self, "Error", f"Missing required input MRC files:\n{chr(10).join(missing_files)}")

        out_dir_unix = out_dir.replace("\\", "/")
        self.btn_run.setText("..."); self.log_panel.setFocus(); self.btn_run.setEnabled(False); self.log_panel.clear(); QApplication.processEvents()

        def pipeline_worker():
            try:
                self.step_updated.emit("Starting Pipeline...")
                out_h1 = f"{out_dir_unix}/symEx_halfmap1.mrc"
                out_h2 = f"{out_dir_unix}/symEx_halfmap2.mrc"
                out_post = f"{out_dir_unix}/hh_postprocess.mrc"
                
                base_cmd = f"module load {RELION_VER} && relion_helix_toolbox --impose"

                if need_h1:
                    if do_h1_hexp:
                        cp_h1 = f'cp "{h1_path}" "{out_dir_unix}/" && ' if os.path.abspath(os.path.dirname(h1_path)) != os.path.abspath(out_dir) else ''
                        cmd_h1 = f'{cp_h1}{base_cmd} --i "{h1_path}" --o "{out_h1}" --angpix {pix} --twist {twist_h1} --rise {rise_h1} --cyl_outer_diameter {diam} --z_percentage {z_perc:.2f}'
                        self.step_updated.emit(f"Symmetry Expanding: {os.path.basename(h1_path)}")
                        subprocess.run(cmd_h1, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        self.step_updated.emit("Skipping Helical Expansion for Halfmap 1. Copying directly...")
                        if os.path.abspath(h1_path) != os.path.abspath(out_h1):
                            subprocess.run(f'cp "{h1_path}" "{out_h1}"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if need_h2:
                    if do_h2_hexp:
                        cp_h2 = f'cp "{h2_path}" "{out_dir_unix}/" && ' if os.path.abspath(os.path.dirname(h2_path)) != os.path.abspath(out_dir) else ''
                        cmd_h2 = f'{cp_h2}{base_cmd} --i "{h2_path}" --o "{out_h2}" --angpix {pix} --twist {twist_h2} --rise {rise_h2} --cyl_outer_diameter {diam} --z_percentage {z_perc:.2f}'
                        self.step_updated.emit(f"Symmetry Expanding: {os.path.basename(h2_path)}")
                        subprocess.run(cmd_h2, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        self.step_updated.emit("Skipping Helical Expansion for Halfmap 2. Copying directly...")
                        if os.path.abspath(h2_path) != os.path.abspath(out_h2):
                            subprocess.run(f'cp "{h2_path}" "{out_h2}"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if need_post:
                    if do_post_hexp:
                        cp_post = f'cp "{post_path}" "{out_dir_unix}/" && ' if os.path.abspath(os.path.dirname(post_path)) != os.path.abspath(out_dir) else ''
                        cmd_post = f'{cp_post}{base_cmd} --i "{post_path}" --o "{out_post}" --angpix {pix} --twist {twist_avg} --rise {rise_avg} --cyl_outer_diameter {diam} --z_percentage {z_perc:.2f}'
                        self.step_updated.emit(f"Symmetry Expanding: {os.path.basename(post_path)}")
                        subprocess.run(cmd_post, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        self.step_updated.emit("Skipping Helical Expansion for PostProcess. Copying directly...")
                        if os.path.abspath(post_path) != os.path.abspath(out_post):
                            subprocess.run(f'cp "{post_path}" "{out_post}"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                parsed_z_percentage = "N/A"
                if run_black:
                    self.step_updated.emit("Calculate Black Curve...")
                    if use_star_for_black:
                        self.step_updated.emit(f"Extracting Black Curve from {os.path.basename(black_star_path)}")
                        out_txt = f"{out_dir_unix}/black.txt"
                        mask_name_path = None
                        with open(black_star_path, 'r') as f_in, open(out_txt, 'w') as f_out:
                            in_fsc = False
                            for line in f_in:
                                stripped = line.strip()
                                if stripped.startswith("_rlnMaskName"):
                                    m_parts = stripped.split()
                                    if len(m_parts) >= 2: mask_name_path = m_parts[1].strip()
                                if stripped.startswith("data_fsc"): in_fsc = True
                                elif stripped.startswith("data_") and stripped != "data_fsc": in_fsc = False
                                elif in_fsc:
                                    parts = stripped.split()
                                    if len(parts) >= 4 and parts[0].isdigit(): f_out.write(line)
                                        
                        if mask_name_path:
                            try:
                                mask_dir = os.path.dirname(mask_name_path)
                                job_star_path = os.path.join(os.path.dirname(black_star_path), mask_dir, "job.star")
                                if os.path.exists(job_star_path):
                                    with open(job_star_path, 'r') as f_job:
                                        for line in f_job:
                                            if "helical_z_percentage" in line:
                                                z_parts = line.strip().split()
                                                if len(z_parts) >= 2:
                                                    parsed_z_percentage = f"{z_parts[1]}%"
                                                    break
                            except Exception as e: print(f"Failed parsing helical_z_percentage: {e}")
                    else:
                        if use_helical_exp_for_black:
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_h1}" --fsc "{out_h2}" > "{out_dir_unix}/black.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{h1_path}" --fsc "{h2_path}" > "{out_dir_unix}/black.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                if run_red:
                    red_script = os.path.join(out_dir, "chimera_red.py")
                    with open(red_script, "w") as f: f.write(f'import chimera\nfrom chimera import runCommand as rc\nrc("open #1 {out_dir_unix}/{red_pdb}")\nrc("open #2 {out_dir_unix}/hh_postprocess.mrc")\nrc("molmap #1 {res_red} modelId #3")\nrc("vop resample #3 onGrid #2 modelId #4")\nrc("volume #4 save {out_dir_unix}/molmap_post.mrc")\nrc("close session")\n')
                    self.step_updated.emit(f"Calculate Red Curve: Molmap - Res: {res_red}")
                    subprocess.run(f'module load {CHIMERA_VER} && chimera --nogui --script "{red_script}" && rm -f "{red_script}"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if run_blue_masking:
                    blue_script = os.path.join(out_dir, "chimera_blue.py")
                    with open(blue_script, "w") as f: f.write(f'import chimera\nfrom chimera import runCommand as rc\nrc("open #1 {out_dir_unix}/{blue_pdb}")\nrc("open #2 {out_dir_unix}/symEx_halfmap1.mrc")\nrc("molmap #1 {res_blue} modelId #3")\nrc("vop resample #3 onGrid #2 modelId #4")\nrc("volume #4 save {out_dir_unix}/molmap_halfmap1.mrc")\nrc("close session")\n')
                    self.step_updated.emit(f"Calculate Blue/Yellow Curve: Molmap - Res: {res_blue}")
                    subprocess.run(f'module load {CHIMERA_VER} && chimera --nogui --script "{blue_script}" && rm -f "{blue_script}"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                def run_comprehensive(color, in_molmap, in_map, l_ini, l_ext, l_soft, l_low):
                    comp_dir = f"{out_dir_unix}/Comprehensive_{color}"
                    os.makedirs(comp_dir, exist_ok=True)
                    log_file, job_q, params_log, j_count = f"{comp_dir}/params_map.json", queue.Queue(), {}, 0
                    
                    for ini in l_ini:
                        for ext in l_ext:
                            for soft in l_soft:
                                for low in l_low:
                                    j_count += 1; jid = f"{color.lower()}_{j_count}"
                                    params_log[jid] = f"Ini:{ini} Ext:{ext} Soft:{soft} Low:{low}"
                                    job_q.put((jid, ini, ext, soft, low))
                    
                    total_jobs = j_count
                    with open(log_file, "w") as fw: json.dump(params_log, fw)
                    target_threads = [self.spin_threads.value()]
                    active_threads = [0]
                    jobs_done = [0]
                    lock = threading.Lock()
                    
                    def w(thread_idx):
                        import time
                        with lock: active_threads[0] += 1
                        t_mask, t_mult = f"{comp_dir}/temp_mask_t{thread_idx}.mrc", f"{comp_dir}/temp_mult_t{thread_idx}.mrc"
                        
                        while True:
                            with lock:
                                if active_threads[0] > target_threads[0]:
                                    active_threads[0] -= 1
                                    return
                            try: jid, ini, ext, soft, low = job_q.get_nowait()
                            except queue.Empty: break
                            
                            out_txt = f"{comp_dir}/{jid}.txt"
                            c1 = f'module load {RELION_VER} && relion_mask_create --i "{in_molmap}" --o "{t_mask}" --ini_threshold {ini} --extend_inimask {ext} --width_soft_edge {soft} --lowpass {low}'
                            c2 = f'relion_image_handler --i "{in_map}" --multiply "{t_mask}" --o "{t_mult}"'
                            c3 = f'relion_image_handler --i "{t_mult}" --fsc "{in_molmap}" > "{out_txt}"'
                            ret = subprocess.call(f"{c1} && {c2} && {c3}", shell=True, executable="/bin/bash", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            
                            if ret == 0 and os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
                                with lock:
                                    jobs_done[0] += 1
                                    pct = int((jobs_done[0] / total_jobs) * 100)
                                    if jobs_done[0] % 5 == 0 or jobs_done[0] == total_jobs:
                                        self.step_updated.emit(f"    -> {color} Progress: {jobs_done[0]}/{total_jobs} ({pct}%) [Threads: {active_threads[0]}]")
                            else:
                                try:
                                    if os.path.exists(out_txt): os.remove(out_txt)
                                except: pass
                                with lock:
                                    if target_threads[0] > 1:
                                        target_threads[0] -= 1
                                        self.step_updated.emit(f'<span style="color: #FF5555;">Job {jid} terminated by cluster. Reducing active threads to {target_threads[0]}...</span>')
                                job_q.put((jid, ini, ext, soft, low))
                                time.sleep(3.0) 
                            try: os.remove(t_mask); os.remove(t_mult)
                            except: pass
                            job_q.task_done()
                        with lock: active_threads[0] -= 1

                    th = []
                    init_threads = self.spin_threads.value()
                    for i in range(init_threads):
                        t = threading.Thread(target=w, args=(i,)); t.start(); th.append(t)
                    job_q.join()
                    for t in th: t.join()

                def auto_promote(color):
                    comp_dir = f"{out_dir_unix}/Comprehensive_{color}"; log_file = f"{comp_dir}/params_map.json"
                    if not os.path.exists(log_file): return None
                    with open(log_file, 'r') as f: params_map = json.load(f)
                    
                    job_scores, target_x = [], np.arange(0.1, 0.51, 0.05)
                    for jid in params_map.keys():
                        path = f"{comp_dir}/{jid}.txt"
                        if not os.path.exists(path): continue
                        try:
                            data = safe_read_fsc(path)
                            if data.ndim == 2 and data.shape[0] > 1:
                                x, y = data[:, 0], data[:, 1]
                                sort_idx = np.argsort(x); target_y = np.interp(target_x, x[sort_idx], y[sort_idx])
                                score = np.mean(target_y)
                            else: score = 0.0
                            job_scores.append((score, jid))
                        except: pass
                    
                    if not job_scores: return None
                    job_scores.sort(key=lambda x: x[0], reverse=True)
                    best_params_str = params_map[job_scores[0][1]]; params = {}
                    for p in best_params_str.split():
                        if ":" in p: k, v = p.split(":", 1); params[k] = v
                    ini, ext, soft, low = params.get("Ini"), params.get("Ext"), params.get("Soft"), params.get("Low")
                    
                    self.step_updated.emit(f"Auto-Promoting Best {color} Curve: Ini {ini}, Ext {ext}, Soft {soft}, Low {low}")
                    if color == "Red":
                        c1 = f'module load {RELION_VER} && relion_mask_create --i "{out_dir_unix}/molmap_post.mrc" --o "{out_dir_unix}/molmap_post_masked.mrc" --ini_threshold {ini} --extend_inimask {ext} --width_soft_edge {soft} --lowpass {low}'
                        subprocess.run(f'{c1} && relion_image_handler --i "{out_dir_unix}/hh_postprocess.mrc" --multiply "{out_dir_unix}/molmap_post_masked.mrc" --o "{out_dir_unix}/postprocess_multiplied.mrc" && relion_image_handler --i "{out_dir_unix}/postprocess_multiplied.mrc" --fsc "{out_dir_unix}/molmap_post.mrc" > "{out_dir_unix}/red.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        c1 = f'module load {RELION_VER} && relion_mask_create --i "{out_dir_unix}/molmap_halfmap1.mrc" --o "{out_dir_unix}/molmap_halfmap1_masked.mrc" --ini_threshold {ini} --extend_inimask {ext} --width_soft_edge {soft} --lowpass {low}'
                        cmds = [c1]
                        if run_blue:
                            cmds.append(f'relion_image_handler --i "{out_dir_unix}/symEx_halfmap1.mrc" --multiply "{out_dir_unix}/molmap_halfmap1_masked.mrc" --o "{out_dir_unix}/halfmap1_multiplied.mrc"')
                            cmds.append(f'relion_image_handler --i "{out_dir_unix}/halfmap1_multiplied.mrc" --fsc "{out_dir_unix}/molmap_halfmap1.mrc" > "{out_dir_unix}/blue.txt"')
                        if run_yellow:
                            cmds.append(f'relion_image_handler --i "{out_dir_unix}/symEx_halfmap2.mrc" --multiply "{out_dir_unix}/molmap_halfmap1_masked.mrc" --o "{out_dir_unix}/halfmap2_multiplied.mrc"')
                            cmds.append(f'relion_image_handler --i "{out_dir_unix}/halfmap2_multiplied.mrc" --fsc "{out_dir_unix}/molmap_halfmap1.mrc" > "{out_dir_unix}/yellow.txt"')
                        if cmds:
                            subprocess.run(" && ".join(cmds), shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return {"Initial Threshold": ini, "Extend Initial Mask": ext, "Soft Edge Width": soft, "Low Pass Filter": low}

                promoted_red, promoted_blue = None, None
                if run_red:
                    if red_jobs == 1:
                        self.step_updated.emit(f"Red: Creating Mask (Ini: {red_p[0]}, Ext: {red_p[1]}, Soft: {red_p[2]}, Low: {red_p[3]})...")
                        cmd_red_mask = f'module load {RELION_VER} && relion_mask_create --i "{out_dir_unix}/molmap_post.mrc" --o "{out_dir_unix}/molmap_post_masked.mrc" --ini_threshold {red_p[0]} --extend_inimask {red_p[1]} --width_soft_edge {red_p[2]} --lowpass {red_p[3]}'
                        subprocess.run(cmd_red_mask, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.step_updated.emit(f"Red: Multiplying PostProcess Map by Mask...")
                        subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/hh_postprocess.mrc" --multiply "{out_dir_unix}/molmap_post_masked.mrc" --o "{out_dir_unix}/postprocess_multiplied.mrc"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        self.step_updated.emit(f"Red: Calculating FSC Curve...")
                        subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/postprocess_multiplied.mrc" --fsc "{out_dir_unix}/molmap_post.mrc" > "{out_dir_unix}/red.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        self.step_updated.emit(f"Running Comprehensive Red Pipeline ({red_jobs} jobs)...")
                        run_comprehensive("Red", f"{out_dir_unix}/molmap_post.mrc", f"{out_dir_unix}/hh_postprocess.mrc", red_ini, red_ext, red_soft, red_low)
                        promoted_red = auto_promote("Red"); QMetaObject.invokeMethod(self.btn_red_vis, "setEnabled", Qt.ConnectionType.QueuedConnection, Q_ARG(bool, True))

                if run_blue_masking or run_yellow:
                    if run_blue_masking:
                        if blue_jobs == 1:
                            self.step_updated.emit(f"Blue/Yellow: Creating Mask (Ini: {blue_p[0]}, Ext: {blue_p[1]}, Soft: {blue_p[2]}, Low: {blue_p[3]})...")
                            cmd_blue_mask = f'module load {RELION_VER} && relion_mask_create --i "{out_dir_unix}/molmap_halfmap1.mrc" --o "{out_dir_unix}/molmap_halfmap1_masked.mrc" --ini_threshold {blue_p[0]} --extend_inimask {blue_p[1]} --width_soft_edge {blue_p[2]} --lowpass {blue_p[3]}'
                            subprocess.run(cmd_blue_mask, shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            self.step_updated.emit(f"Running Comprehensive Blue Pipeline ({blue_jobs} jobs)...")
                            run_comprehensive("Blue", f"{out_dir_unix}/molmap_halfmap1.mrc", f"{out_dir_unix}/symEx_halfmap1.mrc", blue_ini, blue_ext, blue_soft, blue_low)
                            promoted_blue = auto_promote("Blue"); QMetaObject.invokeMethod(self.btn_blue_vis, "setEnabled", Qt.ConnectionType.QueuedConnection, Q_ARG(bool, True))
                            
                    if blue_jobs == 1 or not run_blue_masking:
                        if run_blue:
                            self.step_updated.emit(f"Blue: Multiplying Halfmap 1 by Mask...")
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/symEx_halfmap1.mrc" --multiply "{out_dir_unix}/molmap_halfmap1_masked.mrc" --o "{out_dir_unix}/halfmap1_multiplied.mrc"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            self.step_updated.emit(f"Blue: Calculating Blue FSC Curve...")
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/halfmap1_multiplied.mrc" --fsc "{out_dir_unix}/molmap_halfmap1.mrc" > "{out_dir_unix}/blue.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if run_yellow:
                            self.step_updated.emit(f"Yellow: Multiplying Halfmap 2 by Mask...")
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/symEx_halfmap2.mrc" --multiply "{out_dir_unix}/molmap_halfmap1_masked.mrc" --o "{out_dir_unix}/halfmap2_multiplied.mrc"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            self.step_updated.emit(f"Yellow: Calculating Yellow FSC Curve...")
                            subprocess.run(f'module load {RELION_VER} && relion_image_handler --i "{out_dir_unix}/halfmap2_multiplied.mrc" --fsc "{out_dir_unix}/molmap_halfmap1.mrc" > "{out_dir_unix}/yellow.txt"', shell=True, executable="/bin/bash", check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

                log_path = os.path.join(out_dir, "xcop_log.txt")
                existing_lines = []; skip = False
                if os.path.exists(log_path):
                    with open(log_path, 'r') as f: existing_lines = f.readlines()
                new_lines = []
                for line in existing_lines:
                    if line.strip() in ["Symmetry Expansion:", "Red:", "Blue:", "Black:", "Yellow:"]: skip = True
                    if skip and line.strip() == "":
                        skip = False; continue
                    if not skip: new_lines.append(line)

                log_content = ["Symmetry Expansion:\n", f"- Pixel Size: {pix}\n"]
                if any_hexp:
                    log_content.extend([f"- Tube Diameter: {diam}\n", f"- Z Percentage: {z_perc_str}\n"])
                
                if need_h1:
                    log_content.append(f"- Half 1 Map: {h1_path}\n")
                    if do_h1_hexp: log_content.extend([f"  - Mode: Performed Symmetry Expansion\n", f"  - Twist: {twist_h1}\n", f"  - Rise: {rise_h1}\n"])
                    else: log_content.append(f"  - Mode: Already Expanded\n")

                if need_h2:
                    log_content.append(f"- Half 2 Map: {h2_path}\n")
                    if do_h2_hexp: log_content.extend([f"  - Mode: Performed Symmetry Expansion\n", f"  - Twist: {twist_h2}\n", f"  - Rise: {rise_h2}\n"])
                    else: log_content.append(f"  - Mode: Already Expanded\n")

                if need_post:
                    log_content.append(f"- PostProcess Map: {post_path}\n")
                    if do_post_hexp: log_content.extend([f"  - Mode: Performed Symmetry Expansion\n", f"  - Twist: {twist_avg}\n", f"  - Rise: {rise_avg}\n\n"])
                    else: log_content.append(f"  - Mode: Already Expanded\n\n")

                if run_red:
                    log_content.extend(["Red:\n", f"- Model: {red_pdb}\n", f"- Resolution: {res_red}\n"])
                    if red_jobs == 1: log_content.extend([f"- Initial Threshold: {red_p[0]}\n", f"- Extend Initial Mask: {red_p[1]}\n", f"- Soft Edge Width: {red_p[2]}\n", f"- Low Pass Filter: {red_p[3]}\n\n"])
                    elif promoted_red: log_content.extend([f"- Initial Threshold: {promoted_red['Initial Threshold']}\n", f"- Extend Initial Mask: {promoted_red['Extend Initial Mask']}\n", f"- Soft Edge Width: {promoted_red['Soft Edge Width']}\n", f"- Low Pass Filter: {promoted_red['Low Pass Filter']}\n\n"])
                    else: log_content.extend([f"- Initial Threshold: (Comprehensive mode pending Promotion)\n", f"- Extend Initial Mask: -\n", f"- Soft Edge Width: -\n", f"- Low Pass Filter: -\n\n"])

                if run_blue or run_yellow:
                    log_content.extend(["Blue (Mask Source):\n", f"- Model: {blue_pdb}\n", f"- Resolution: {res_blue}\n"])
                    if blue_jobs == 1: log_content.extend([f"- Initial Threshold: {blue_p[0]}\n", f"- Extend Initial Mask: {blue_p[1]}\n", f"- Soft Edge Width: {blue_p[2]}\n", f"- Low Pass Filter: {blue_p[3]}\n\n"])
                    elif promoted_blue: log_content.extend([f"- Initial Threshold: {promoted_blue['Initial Threshold']}\n", f"- Extend Initial Mask: {promoted_blue['Extend Initial Mask']}\n", f"- Soft Edge Width: {promoted_blue['Soft Edge Width']}\n", f"- Low Pass Filter: {promoted_blue['Low Pass Filter']}\n\n"])
                    else: log_content.extend([f"- Initial Threshold: (Comprehensive mode pending Promotion)\n", f"- Extend Initial Mask: -\n", f"- Soft Edge Width: -\n", f"- Low Pass Filter: -\n\n"])

                if run_black:
                    log_content.extend(["Black:\n"])
                    if use_star_for_black:
                        log_content.extend([f"- Input Map: {os.path.abspath(black_star_path)}\n", f"- Z percentage: {parsed_z_percentage}\n\n"])
                    else:
                        if use_helical_exp_for_black:
                            log_content.extend([f"- Input Map 1: {out_dir_unix}/symEx_halfmap1.mrc\n", f"- Input Map 2: {out_dir_unix}/symEx_halfmap2.mrc\n\n"])
                        else:
                            log_content.extend([f"- Input Map 1: {h1_path}\n", f"- Input Map 2: {h2_path}\n\n"])

                if run_yellow:
                    log_content.extend(["Yellow:\n", f"- Input Map: {out_dir_unix}/symEx_halfmap2.mrc\n", f"- Mask Used: {out_dir_unix}/molmap_halfmap1_masked.mrc\n\n"])

                with open(log_path, 'w') as f: f.writelines(log_content); f.writelines(new_lines)
                self.step_updated.emit("Pipeline Finished Successfully.")
            except Exception as e: self.step_updated.emit(f'<span style="color: red;">Pipeline Error: {e}</span>')
            finally: self.pipeline_finished.emit()

        threading.Thread(target=pipeline_worker, daemon=True).start()


# --- Application Entry Interface ---
class FSCMainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FSC Curve Panel")
        center_window(self, 500, 240, "fsc_central_window")
        
        layout = QVBoxLayout(self)
        
        btn_auto = QPushButton("Fully Automated FSC Curves")
        btn_auto.setStyleSheet("font-weight: bold; background-color: #98c379; color: #1e1e1e; padding: 10px; font-size: 14px;")
        btn_auto.clicked.connect(self.open_autofsc)
        layout.addWidget(btn_auto)

        btn_non_relion = QPushButton("Automated FSC Curves for Non-RELION Project Directory")
        btn_non_relion.setStyleSheet("font-weight: bold; background-color: #56b6c2; color: #1e1e1e; padding: 10px; font-size: 14px;")
        btn_non_relion.clicked.connect(self.open_non_relion_autofsc)
        layout.addWidget(btn_non_relion)
        
        sep = QLabel()
        sep.setFixedHeight(2)
        sep.setStyleSheet("background-color: #3e3e42; margin-top: 8px; margin-bottom: 4px;")
        layout.addWidget(sep)
        
        btn_vis = QPushButton("Visualize FSC Curves")
        btn_vis.clicked.connect(lambda checked=False: generate_fsc_curves())
        layout.addWidget(btn_vis)
        
        btn_exp = QPushButton("Export a Combined txt File")
        btn_exp.clicked.connect(generate_fsc_excel_file)
        layout.addWidget(btn_exp)

    def open_autofsc(self):
        self.auto_win = RelionAutoApp()
        toplevel_windows.append(self.auto_win)
        self.auto_win.show()

    def open_non_relion_autofsc(self):
        self.non_relion_win = NonRelionAutoApp()
        toplevel_windows.append(self.non_relion_win)
        self.non_relion_win.show()


def apply_global_theme(app_instance):
    app_instance.setStyle("Fusion")
    app_instance.setStyleSheet("""
        QMainWindow, QDialog, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Arial; font-size: 10pt; }
        QPushButton { background-color: #3e3e42; color: #d4d4d4; border: 1px solid #3e3e42; padding: 5px 15px; border-radius: 4px; outline: none; }
        QPushButton:hover { background-color: #4e4e52; border: 1px solid #98c379; }
        QPushButton:pressed { background-color: #98c379; color: #1e1e1e; border: 1px solid #98c379; }
        QPushButton:focus { outline: none; }
        QLineEdit, QComboBox, QSpinBox { background-color: #3c3c3c; border: 1px solid #3c3c3c; color: #cccccc; padding: 4px; border-radius: 2px; selection-background-color: #98c379; selection-color: #1e1e1e; combobox-popup: 0; }
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { background-color: #252526; color: #d4d4d4; border: 1px solid #3e3e42; outline: none; selection-background-color: #98c379; selection-color: #1e1e1e; }
        QComboBox QAbstractItemView::item { padding: 5px; min-height: 25px; border: none !important; }
        QComboBox QAbstractItemView::item:hover, QComboBox QAbstractItemView::item:selected { background-color: #98c379; color: #1e1e1e; border: none !important; outline: none !important; }
        QGroupBox { border: 1px solid #3e3e42; border-radius: 4px; margin-top: 24px; padding-top: 8px; font-weight: bold; color: #98c379; outline: none; }
        QGroupBox:focus { outline: none; border: 1px solid #3e3e42; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 2px 5px; outline: none; }
        QGroupBox::title:focus { outline: none; }
        QGroupBox::indicator { width: 14px; height: 14px; border: 1px solid #555; background-color: #1e1e1e; border-radius: 2px; margin-top: 1px; outline: none; }
        QGroupBox::indicator:unchecked { background-color: #1e1e1e; }
        QGroupBox::indicator:checked { background-color: #98c379; border: 1px solid #98c379; image: none; }
        QGroupBox::indicator:hover { border: 1px solid #98c379; }
        QCheckBox { color: #d4d4d4; spacing: 8px; outline: none; }
        QCheckBox:focus { outline: none; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; background-color: #1e1e1e; margin: 1px; border-radius: 2px; outline: none; }
        QCheckBox::indicator:unchecked { background-color: #1e1e1e; }
        QCheckBox::indicator:checked { background-color: #98c379; border: 1px solid #98c379; image: none; }
        QCheckBox::indicator:hover { border: 1px solid #98c379; }
        
        QRadioButton { color: #d4d4d4; spacing: 8px; }
        QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid #555; background-color: #1e1e1e; margin: 1px; border-radius: 8px; }
        QRadioButton::indicator:unchecked { background-color: #1e1e1e; }
        QRadioButton::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        QRadioButton::indicator:hover { border: 1px solid #98c379; }
        
        QSlider::groove:horizontal { background: #555; height: 6px; border-radius: 3px; }
        QSlider::handle:horizontal { background: #98c379; width: 12px; margin: -4px 0; border-radius: 6px; }
        QSlider::add-page:horizontal { background: #555; }
        QSlider::sub-page:horizontal { background: #98c379; }
        
        QSpinBox { padding-right: 15px; }
        QSpinBox::up-button { subcontrol-origin: border; subcontrol-position: top right; width: 16px; background: #3e3e42; border-left: 1px solid #1e1e1e; border-bottom: 1px solid #1e1e1e; border-top-right-radius: 2px; }
        QSpinBox::up-button:hover { background: #4e4e52; }
        QSpinBox::up-arrow { image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'><path d='M7 14l5-5 5 5z'/></svg>"); width: 10px; height: 10px; }
        
        QSpinBox::down-button { subcontrol-origin: border; subcontrol-position: bottom right; width: 16px; background: #3e3e42; border-left: 1px solid #1e1e1e; border-bottom-right-radius: 2px; }
        QSpinBox::down-button:hover { background: #4e4e52; }
        QSpinBox::down-arrow { image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='%23ffffff'><path d='M7 10l5 5 5-5z'/></svg>"); width: 10px; height: 10px; }

        QScrollBar:vertical { border: none; background: transparent; width: 10px; margin: 0px; }
        QScrollBar::handle:vertical { background: #4a4a4a; border-radius: 5px; min-height: 30px; }
        QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed { background: #98c379; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; background: none; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }

        QProgressBar { 
            border: 1px solid #3e3e42; border-radius: 2px; 
            background-color: #1e1e1e; text-align: center; 
            color: #d4d4d4; 
        }
        QProgressBar::chunk { background-color: #98c379; border-radius: 2px; color: #1e1e1e; }
    """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_global_theme(app)
    main_window = FSCMainWindow()
    main_window.show()
    sys.exit(app.exec())