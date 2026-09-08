import sys
import subprocess
import os
import re
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QScrollArea, 
                             QFrame, QMessageBox, QListWidget, QListWidgetItem,
                             QAbstractItemView, QMenu, QDialog, QDialogButtonBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer
from PyQt6.QtGui import QDrag, QPixmap, QPainter, QCursor, QColor

# ---------------------------------------------------------
# STYLESHEET (One Dark aesthetic from autofsc)
# ---------------------------------------------------------
STYLE_SHEET = """
QMainWindow {
    background-color: #1e1e1e;
}
QWidget {
    font-family: Arial, -apple-system, sans-serif;
    color: #d4d4d4;
}
QScrollArea {
    border: none;
    background-color: transparent;
}
QScrollArea > QWidget > QWidget {
    background-color: transparent;
}
QFrame#JobCard {
    background-color: #252526;
    border-radius: 4px;
    border: 1px solid #3e3e42;
}
QFrame#JobCard:hover {
    background-color: #2d2d30;
    border: 1px solid #98c379;
}
QLabel#JobId {
    color: #d4d4d4;
    font-size: 16px;
    font-weight: bold;
}
QLabel#StateRunning {
    color: #98c379;
    font-size: 13px;
    font-weight: bold;
    margin-left: 8px;
}
QLabel#StatePending {
    color: #e5c07b;
    font-size: 13px;
    font-weight: bold;
    margin-left: 8px;
}
QLabel#StateOther {
    color: #8a8a8a;
    font-size: 13px;
    font-weight: bold;
    margin-left: 8px;
}
QLabel#NiceTag {
    color: #e5c07b;
    background-color: rgba(229, 192, 123, 0.15);
    border: 1px solid #e5c07b;
    border-radius: 10px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: bold;
}
QListWidget {
    background-color: transparent;
    border: none;
    padding-right: 5px;
}
QListWidget::item {
    background-color: transparent;
}
QListWidget::item:selected {
    background-color: transparent;
    border: none;
}
QListView::drop-indicator {
    background-color: #0A84FF;
    height: 3px;
    border-radius: 1px;
}
QMenu {
    background-color: #252526;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
}
QMenu::item:selected {
    background-color: #3e3e42;
}
QLabel#TagLabel {
    color: #d4d4d4;
    background-color: #3a3a3c;
    border: none;
    border-radius: 10px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: bold;
}
QLabel#DirLabel {
    color: #8a8a8a;
    font-size: 12px;
}
QLabel#HeaderTitle {
    color: #d4d4d4;
    font-size: 20px;
    font-weight: bold;
}
QPushButton#RefreshBtn {
    background-color: #3e3e42;
    color: #d4d4d4;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: bold;
}
QPushButton#RefreshBtn:hover {
    background-color: #4e4e52;
    border: 1px solid #98c379;
}
QPushButton#RefreshBtn:pressed {
    background-color: #98c379;
    color: #1e1e1e;
    border: 1px solid #98c379;
}
QPushButton#RefreshBtn:disabled {
    background-color: #2d2d30;
    color: #555555;
    border: 1px solid #3e3e42;
}
QPushButton#RemoveNiceBtn {
    background-color: #3e3e42;
    color: #e5c07b;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: bold;
}
QPushButton#RemoveNiceBtn:hover {
    background-color: #4e4e52;
    border: 1px solid #e5c07b;
}
QPushButton#RemoveNiceBtn:pressed {
    background-color: #e5c07b;
    color: #1e1e1e;
    border: 1px solid #e5c07b;
}
QPushButton#RemoveNiceBtn:disabled {
    background-color: #2d2d30;
    color: #555555;
    border: 1px solid #3e3e42;
}
QPushButton#TopBtn {
    background-color: #3e3e42;
    color: #98c379;
    border: 1px solid #3e3e42;
    border-radius: 4px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: bold;
}
QPushButton#TopBtn:hover {
    background-color: #4e4e52;
    border: 1px solid #98c379;
}
QPushButton#TopBtn:pressed {
    background-color: #98c379;
    color: #1e1e1e;
    border: 1px solid #98c379;
}
QPushButton#TopBtn:disabled {
    background-color: #2d2d30;
    color: #555555;
    border: 1px solid #3e3e42;
}
QPushButton#CancelBtn {
    background-color: transparent;
    color: #FF5555;
    font-size: 16px;
    font-weight: bold;
    border: none;
    padding: 0px 4px;
    margin-left: 10px;
}
QPushButton#CancelBtn:hover {
    color: #ff7777;
}
QScrollBar:vertical {
    border: none;
    background: transparent;
    width: 10px;
    margin: 0px;
}
QScrollBar::handle:vertical {
    background: #4a4a4a;
    min-height: 20px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:vertical:pressed {
    background: #98c379;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QMessageBox {
    background-color: #252526;
}
QMessageBox QLabel {
    color: #d4d4d4;
    background-color: transparent;
}
QMessageBox QPushButton {
    background-color: #3e3e42;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px 15px;
    min-width: 60px;
}
QMessageBox QPushButton:hover {
    background-color: #4e4e52;
    border: 1px solid #98c379;
}
QDialog {
    background-color: #1e1e1e;
}
QDialog QLabel {
    background-color: transparent;
}
QDialog QPushButton {
    background-color: #3e3e42;
    color: #d4d4d4;
    border: 1px solid #555555;
    border-radius: 4px;
    padding: 5px 15px;
    min-width: 60px;
}
QDialog QPushButton:hover {
    background-color: #4e4e52;
    border: 1px solid #98c379;
}
"""

# ---------------------------------------------------------
# JOB TYPE COLORS (Monet "Giverny" palette, keyed by RELION folder name)
# ---------------------------------------------------------
JOB_TYPE_COLORS = {
    # --- Light Greens ---
    "Import": "#A5D6A7",          # Pale Emerald
    "Select": "#DCE775",          # Light Lime
    "MaskCreate": "#81C784",      # Soft Green
    "ModelAngelo": "#4DB6AC",     # Muted Teal/Verdigris

    # --- Light Blues ---
    "MotionCorr": "#81D4FA",      # Light Sky Blue
    "Extract": "#80CBC4",         # Soft Aqua
    "Class3D": "#64B5F6",         # Cornflower Blue
    "Multibody": "#90A4AE",       # Blue-Grey Mist
    "Class2D": "#4DD0E1",       # Cyan/Turquoise
    "PostProcess": "#E0F7FA",     # Ice White/Blue
    "External": "#B0BEC5",        # Light Grey

    # --- Light Purples & Pinks ---
    "DynaMight": "#CE93D8",         # Light Orchid/Lilac
    "Refine3D": "#9575CD",       # Soft Violet
    "CtfRefine": "#F8BBD0",        # Rose Pink
    "Substract": "#F48FB1",       # Soft Flamingo
    "LocalRes": "#F06292",        # Pale Pink

    # --- Light Warm Tones ---
    "CtfFind": "#FFB74D",         # Soft Apricot/Orange
    "ManualPick": "#E57373",      # Light Red/Coral
    "AutoPick": "#D7CCC8",        # Beige/Sand
    "InitialModel": "#FFD54F",    # Sunlight Yellow
    "Polish": "#FF8A65",          # Salmon
    "JoinStar": "#FFF176",        # Goldenrod/Wheat
}

# ---------------------------------------------------------
# BACKGROUND WORKER (Prevents GUI Freezing)
# ---------------------------------------------------------
class JobFetcherThread(QThread):
    jobs_fetched = pyqtSignal(list)
    status_msg = pyqtSignal(str)

    def run(self):
        user = os.getenv('USER')
        if not user:
            self.status_msg.emit("Error: Could not determine current user.")
            self.jobs_fetched.emit([])
            return

        self.status_msg.emit("Fetching...")
        
        try:
            # Fetch Job IDs
            squeue_output = subprocess.check_output(
                ["squeue", "-u", user, "-h", "-o", "%A"],
                universal_newlines=True
            )
        except FileNotFoundError:
            self.status_msg.emit("Error: 'squeue' not found. Not on HPC?")
            self.jobs_fetched.emit([])
            return
        except subprocess.CalledProcessError as e:
            self.status_msg.emit(f"Error running squeue: {e}")
            self.jobs_fetched.emit([])
            return

        job_ids = [jid.strip() for jid in squeue_output.strip().split('\n') if jid.strip()]

        if not job_ids:
            self.status_msg.emit("No active or queued jobs.")
            self.jobs_fetched.emit([])
            return

        # Fetch currently prioritized jobs waiting on the xcop_Qrst auto-reset trigger
        prioritized_ids = []
        try:
            xcop_out = subprocess.check_output(
                ["squeue", "-u", user, "--name=xcop_Qrst", "-h", "-o", "%E"], 
                universal_newlines=True
            ).strip()
            if xcop_out:
                prioritized_ids = re.findall(r'\d+', xcop_out)
        except Exception:
            pass

        self.status_msg.emit(f"Fetching...")
        jobs_list = []
        
        for job_id in job_ids:
            try:
                scontrol_output = subprocess.check_output(
                    ["scontrol", "show", "job", job_id],
                    universal_newlines=True
                )
                
                job_name_match = re.search(r'JobName=([^\s]+)', scontrol_output)
                job_name = job_name_match.group(1) if job_name_match else "Unknown"
                
                # --- NEW: Secretly ignore our background watcher jobs ---
                if job_name == "xcop_Qrst":
                    continue
                
                work_dir_match = re.search(r'WorkDir=([^\s]+)', scontrol_output)
                state_match = re.search(r'JobState=([^\s]+)', scontrol_output)
                nice_match = re.search(r'Nice=([^\s]+)', scontrol_output)
                dep_match = re.search(r'Dependency=([^\s]+)', scontrol_output)
                runtime_match = re.search(r'RunTime=([^\s]+)', scontrol_output)
                priority_match = re.search(r'Priority=(\d+)', scontrol_output)
                
                work_dir = work_dir_match.group(1) if work_dir_match else "Unknown"
                job_state = state_match.group(1) if state_match else "UNKNOWN"
                nice_val = nice_match.group(1) if nice_match else "0"
                priority_val = int(priority_match.group(1)) if priority_match else 0
                
                raw_dep = dep_match.group(1) if dep_match else "(null)"
                dependency_val = None if raw_dep == "(null)" else raw_dep
                
                runtime_val = runtime_match.group(1) if runtime_match else "00:00:00"
                
                parts = job_name.split('/')
                job_type = parts[0] if len(parts) > 0 else ""
                job_number = parts[1] if len(parts) > 1 else ""
                
                jobs_list.append({
                    "id": job_id,
                    "dir": work_dir,
                    "type": job_type,
                    "number": job_number,
                    "state": job_state,
                    "nice": nice_val,
                    "priority": priority_val,
                    "is_prioritized": job_id in prioritized_ids,
                    "dependency": dependency_val,
                    "runtime": runtime_val
                })
            except Exception:
                pass
        
        # Sort jobs: 
        # 1. RUNNING jobs first
        # 2. Prioritized jobs second (xcop_Qrst targets)
        # 3. Job ID ascending (Lower ID = Older job. These will naturally run next once nice values are cleared)
        jobs_list.sort(key=lambda j: (
            0 if j["state"] == "RUNNING" else 1,
            0 if j.get("is_prioritized") else 1,
            int(j["id"].split('_')[0]) if j.get("id", "").split('_')[0].isdigit() else float('inf')
        ))
        
        self.status_msg.emit(f"{len(jobs_list)} jobs")
        self.jobs_fetched.emit(jobs_list)


# ---------------------------------------------------------
# CUSTOM LIST WIDGET
# ---------------------------------------------------------
class JobQueueList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # MultiSelection allows click-to-toggle without holding Ctrl/Shift
        self.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        self.setSpacing(12)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Force scrollbar space to always be reserved so cards don't dynamically expand
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

from PyQt6.QtWidgets import QFrame, QSizePolicy, QLabel
from PyQt6.QtCore import Qt, QTimer, QSize
from PyQt6.QtGui import QPainter, QColor, QFontMetrics

class HoverScrollLabel(QLabel):
    def __init__(self, text, max_width, elide_mode=Qt.TextElideMode.ElideMiddle, is_path=False):
        super().__init__(text)
        self.full_text = text
        self.elide_mode = elide_mode
        self.is_path = is_path
        self.setMaximumWidth(max_width)
        # REMOVED setMinimumWidth so small tags can physically shrink-wrap their text
        
        self.is_scrolling = False
        self.scroll_offset = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_scroll)

    def get_full_width(self):
        font_normal = self.font()
        fm_normal = QFontMetrics(font_normal)
        if self.is_path and '/' in self.full_text:
            font_bold = self.font()
            font_bold.setBold(True)
            fm_bold = QFontMetrics(font_bold)
            idx = self.full_text.rfind('/')
            return fm_normal.horizontalAdvance(self.full_text[:idx+1]) + fm_bold.horizontalAdvance(self.full_text[idx+1:])
        return fm_normal.horizontalAdvance(self.full_text)

    def sizeHint(self):
        # Tags have 20px total horizontal padding (10px left/right). The path label has 0.
        padding = 0 if self.is_path else 20
        required_width = self.get_full_width() + padding
        return QSize(min(required_width, self.maximumWidth()), super().sizeHint().height())
        
    def paintEvent(self, event):
        QFrame.paintEvent(self, event)
        painter = QPainter(self)
        rect = self.contentsRect()
        
        font_normal = self.font()
        font_bold = self.font()
        font_bold.setBold(True)
        
        fm_normal = QFontMetrics(font_normal)
        fm_bold = QFontMetrics(font_bold)
        
        base_str = self.full_text
        tail_str = ""
        full_width = self.get_full_width()
        
        if self.is_path and '/' in self.full_text:
            idx = self.full_text.rfind('/')
            base_str = self.full_text[:idx+1]
            tail_str = self.full_text[idx+1:]
            
        painter.setClipRect(rect)
        base_color = self.palette().color(self.foregroundRole())
        highlight_color = QColor("#ffffff") # Pure white for the last folder
        
        def draw_styled_text(x_pos, y_pos, height):
            painter.setFont(font_normal)
            painter.setPen(base_color)
            painter.drawText(x_pos, y_pos, 9999, height, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, base_str)
            
            if tail_str:
                base_w = fm_normal.horizontalAdvance(base_str)
                painter.setFont(font_bold)
                painter.setPen(highlight_color)
                painter.drawText(x_pos + base_w, y_pos, 9999, height, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tail_str)

        if full_width <= rect.width():
            if self.is_path:
                draw_styled_text(rect.x(), rect.y(), rect.height())
            else:
                painter.setFont(font_normal)
                painter.setPen(base_color)
                # Drawing AlignCenter distributes any microscopic pixel math perfectly
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.full_text)
        elif self.is_scrolling:
            x = rect.x() - self.scroll_offset
            y = rect.y()
            h = rect.height()
            draw_styled_text(x, y, h)
            draw_styled_text(x + full_width + 30, y, h)
        else:
            if self.is_path and tail_str:
                w_tail = fm_bold.horizontalAdvance(tail_str)
                prefix = ".../"
                w_prefix = fm_normal.horizontalAdvance(prefix)
                
                # Scenario A: The tail fits completely, so we aggressively truncate the base path
                if w_tail <= rect.width() - w_prefix:
                    avail_base = rect.width() - w_tail
                    # ElideLeft naturally turns "/home/lab/..." into ".../lab/"
                    base_elided = fm_normal.elidedText(base_str, Qt.TextElideMode.ElideLeft, avail_base)
                    
                    painter.setFont(font_normal)
                    painter.setPen(base_color)
                    painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, base_elided)
                    
                    actual_base_w = fm_normal.horizontalAdvance(base_elided)
                    tail_rect = rect.adjusted(actual_base_w, 0, 0, 0)
                    
                    painter.setFont(font_bold)
                    painter.setPen(highlight_color)
                    painter.drawText(tail_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tail_str)
                
                # Scenario B: The tail itself is too massive. Force prefix and ElideMiddle the tail.
                else:
                    avail_tail = rect.width() - w_prefix
                    if avail_tail < 20: 
                        avail_tail = rect.width()
                        prefix = ""
                        w_prefix = 0
                        
                    # ElideMiddle naturally creates "final_fol...xxxxxxxxx"
                    tail_elided = fm_bold.elidedText(tail_str, Qt.TextElideMode.ElideMiddle, avail_tail)
                    
                    if prefix:
                        painter.setFont(font_normal)
                        painter.setPen(base_color)
                        painter.drawText(rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, prefix)
                        
                    tail_rect = rect.adjusted(w_prefix, 0, 0, 0)
                    painter.setFont(font_bold)
                    painter.setPen(highlight_color)
                    painter.drawText(tail_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, tail_elided)
            else:
                elided = fm_normal.elidedText(self.full_text, self.elide_mode, rect.width())
                painter.setFont(font_normal)
                painter.setPen(base_color)
                # Keep tags perfectly centered even when chopped with '...'
                painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, elided)

    def enterEvent(self, event):
        if self.get_full_width() > self.contentsRect().width():
            self.is_scrolling = True
            self.scroll_offset = 0
            # Lower number = faster update. 15ms is twice as fast and silky smooth.
            self.timer.start(15) 
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.is_scrolling = False
        self.timer.stop()
        self.scroll_offset = 0
        self.update()
        super().leaveEvent(event)

    def update_scroll(self):
        w = self.get_full_width()
        self.scroll_offset += 1
        if self.scroll_offset > w + 30:
            self.scroll_offset = 0
        self.update()

class ActiveTimeLabel(QLabel):
    def __init__(self, d, h, m, s, parent=None):
        super().__init__(parent)
        # Convert the starting Slurm time into pure seconds
        self.total_seconds = d * 86400 + h * 3600 + m * 60 + s
        self.update_display()
        
        # Monospace font keeps the text from jittering as the seconds tick by
        self.setStyleSheet("color: #8a8a8a; font-size: 12px; font-weight: bold; font-family: monospace;")
        
        # Start the internal ticker
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(1000)

    def tick(self):
        self.total_seconds += 1
        self.update_display()

    def update_display(self):
        ts = self.total_seconds
        d = ts // 86400
        ts %= 86400
        h = ts // 3600
        ts %= 3600
        m = ts // 60
        s = ts % 60
        
        if d > 0:
            time_str = f"{d:02d}-{h:02d}:{m:02d}:{s:02d}"
        elif h > 0:
            time_str = f"{h:02d}:{m:02d}:{s:02d}"
        else:
            time_str = f"{m:02d}:{s:02d}"
            
        self.setText(time_str)

# ---------------------------------------------------------
# CUSTOM DIALOGS
# ---------------------------------------------------------
class OverrideWarningDialog(QDialog):
    def __init__(self, parent, new_jobs, old_jobs):
        super().__init__(parent)
        self.setWindowTitle("Confirm Priority Override")
        self.setMinimumWidth(550)
        
        layout = QVBoxLayout(self)
        
        # Scroll area in case they top a massive amount of jobs
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setStyleSheet("border: none; background: transparent;")
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(0, 0, 15, 0)
        content_layout.setSpacing(10)
        
        # 1. Prioritizing Section
        lbl1 = QLabel("<b>Prioritizing:</b>")
        lbl1.setStyleSheet("color: #d4d4d4; font-size: 14px;")
        content_layout.addWidget(lbl1)
        
        for job in new_jobs:
            card = parent.create_job_card(job)
            card.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu) # Disable right-click
            cancel = card.findChild(QPushButton, "CancelBtn")
            if cancel: cancel.hide() # Disable canceling from inside the dialog
            content_layout.addWidget(card)
            
        content_layout.addSpacing(15)
        
        # 2. Losing Priority Section
        lbl2 = QLabel("<b>Will remove the priority of:</b>")
        lbl2.setStyleSheet("color: #FF5555; font-size: 14px;")
        content_layout.addWidget(lbl2)
        
        for job in old_jobs:
            card = parent.create_job_card(job)
            card.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            cancel = card.findChild(QPushButton, "CancelBtn")
            if cancel: cancel.hide()
            content_layout.addWidget(card)
            
        content_layout.addStretch()
        scroll.setWidget(content_widget)
        
        # Cap height so it doesn't overflow screen
        scroll.setMinimumHeight(min(content_widget.sizeHint().height() + 20, 500))
        
        layout.addWidget(scroll)
        
        # Standard Yes/No Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Yes | QDialogButtonBox.StandardButton.No)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

# ---------------------------------------------------------
# MAIN WINDOW
# ---------------------------------------------------------
class SlurmMonitor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Job Monitor")
        # Enforce a strict fixed width. The scrollbar and right edge can NEVER be cut off.
        self.setFixedWidth(600)
        self.setMinimumHeight(700)
        
        # Central Widget & Main Layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(20, 20, 20, 20)
        self.main_layout.setSpacing(15)

        # Header Area
        self.header_layout = QHBoxLayout()
        
        self.title_label = QLabel("Slurm Queue")
        self.title_label.setObjectName("HeaderTitle")
        
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #8E8E93; font-size: 13px;")
        
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("RefreshBtn")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh_jobs)

        self.remove_nice_btn = QPushButton("Reset Queue")
        self.remove_nice_btn.setObjectName("RemoveNiceBtn")
        self.remove_nice_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.remove_nice_btn.clicked.connect(self.remove_all_nice)

        self.top_btn = QPushButton("Top")
        self.top_btn.setToolTip("Top the selected one job")
        self.top_btn.setObjectName("TopBtn")
        self.top_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.top_btn.clicked.connect(self.top_selected_job)

        self.header_layout.addWidget(self.title_label)
        self.header_layout.addStretch()
        self.header_layout.addWidget(self.status_label)
        self.header_layout.addSpacing(10)
        self.header_layout.addWidget(self.top_btn)
        self.header_layout.addWidget(self.remove_nice_btn)
        self.header_layout.addWidget(self.refresh_btn)
        
        self.main_layout.addLayout(self.header_layout)

        # List Widget for Job Cards
        self.job_list = JobQueueList()
        self.job_list.itemSelectionChanged.connect(self.highlight_selected_card)
        self.main_layout.addWidget(self.job_list)

        # Initialize Thread
        self.fetcher_thread = JobFetcherThread()
        self.fetcher_thread.jobs_fetched.connect(self.populate_jobs)
        self.fetcher_thread.status_msg.connect(self.update_status)

        # Auto-fetch on startup
        self.refresh_jobs()

    def refresh_jobs(self):
        self.refresh_btn.setEnabled(False)
        self.remove_nice_btn.setEnabled(False)
        self.top_btn.setEnabled(False)
        self.refresh_btn.setText("Updating...")
        
        # DO NOT clear the list here! Leave the old items visible 
        # while the background thread communicates with the HPC.
        self.fetcher_thread.start()

    def highlight_selected_card(self):
        # Iterate through all items and apply a blue highlight to the selected one
        for i in range(self.job_list.count()):
            item = self.job_list.item(i)
            widget = self.job_list.itemWidget(item)
            if widget:
                if item.isSelected():
                    # Apply the blue border and slightly lighter background
                    widget.setStyleSheet("""
                        QFrame#JobCard {
                            background-color: #2d2d30;
                            border: 2px solid #98c379;
                        }
                    """)
                else:
                    # Clear the inline style so it reverts back to the global stylesheet
                    widget.setStyleSheet("")

    def update_status(self, msg):
        self.status_label.setText(msg)

    def populate_jobs(self, jobs):
        self.refresh_btn.setEnabled(True)
        self.remove_nice_btn.setEnabled(True)
        self.top_btn.setEnabled(True)
        self.refresh_btn.setText("Refresh")
        
        # 1. Disable UI repainting to prevent screen flicker
        self.job_list.setUpdatesEnabled(False)
        
        # 2. Save scroll position right before the swap
        saved_scroll_position = self.job_list.verticalScrollBar().value()
        
        # 3. Now wipe the old items out
        self.job_list.clear()
        
        if not jobs:
            item = QListWidgetItem()
            empty_label = QLabel("Queue is empty or unavailable.")
            empty_label.setStyleSheet("color: #8E8E93; font-size: 14px; padding: 20px;")
            empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            item.setSizeHint(empty_label.sizeHint())
            self.job_list.addItem(item)
            self.job_list.setItemWidget(item, empty_label)
            self.job_list.setUpdatesEnabled(True)
            return

        for job in jobs:
            card = self.create_job_card(job)

            item = QListWidgetItem()
            # Store job data in the item for reordering logic later
            item.setData(Qt.ItemDataRole.UserRole, job)
            item.setSizeHint(card.sizeHint())
            
            # Make only PENDING jobs selectable
            if job["state"] == "PENDING":
                item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            else:
                # Running/Other jobs cannot be selected or dragged
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                
            self.job_list.addItem(item)
            self.job_list.setItemWidget(item, card)
            
        # 4. Re-enable UI repainting (this makes the swap visually instantaneous)
        self.job_list.setUpdatesEnabled(True)
            
        # 5. Restore scroll position
        QTimer.singleShot(0, lambda: self.job_list.verticalScrollBar().setValue(saved_scroll_position))

    def create_job_card(self, job):
        # The Card Container
        card = QFrame()
        card.setObjectName("JobCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(15, 15, 15, 15)
        card_layout.setSpacing(8)

        # Top Row: Job ID, State, Tags, and Cancel Button
        top_row = QHBoxLayout()
        
        # Job ID
        job_id_label = QLabel(job["id"])
        job_id_label.setObjectName("JobId")
        # Pass mouse clicks through the text so you can drag the card from the ID
        job_id_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        top_row.addWidget(job_id_label)
        
        # State Label
        state_label = QLabel(job["state"].capitalize())
        if job["state"] == "RUNNING":
            state_label.setObjectName("StateRunning")
        elif job["state"] == "PENDING":
            state_label.setObjectName("StatePending")
        else:
            state_label.setObjectName("StateOther")
        top_row.addWidget(state_label)
        
        # Always create the tag to act as a physical spacer
        nice_val = job.get("nice", "0")
        is_prioritized = job.get("is_prioritized", False)
        
        # Give it a 6-digit placeholder text so it measures the width correctly
        if is_prioritized:
            display_text = "Priority"
        else:
            display_text = f"Nice: {nice_val}" if nice_val not in ["0", "Unknown"] else "Nice: 888888"
        
        nice_tag = QLabel(display_text)
        nice_tag.setObjectName("NiceTag")
        nice_tag.setMinimumWidth(100)  # Lock the width to accommodate 6 digits safely
        nice_tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        if is_prioritized:
            # Overwrite stylesheet properties inline for solid emphasis
            nice_tag.setStyleSheet("color: #1e1e1e; background-color: #e5c07b; border: 1px solid #e5c07b; border-radius: 10px; padding: 4px 10px; font-weight: bold; font-size: 11px;")
        elif nice_val == "0" or nice_val == "Unknown":
            # Make it completely invisible, but it still props open the layout!
            nice_tag.setStyleSheet("color: transparent; background-color: transparent; border: none;")
            
        top_row.addWidget(nice_tag)

        top_row.addStretch()

        # Job Type Tag
        if job["type"]:
            # max_width=115 cleanly fits ~10-12 characters before cutting to "..."
            type_tag = HoverScrollLabel(job["type"], max_width=100, elide_mode=Qt.TextElideMode.ElideRight)
            type_tag.setObjectName("TagLabel")
            # Color the tag by its RELION folder name; dark text for contrast on pastels
            bg_color = JOB_TYPE_COLORS.get(job["type"])
            if bg_color:
                type_tag.setStyleSheet(
                    f"color: #1e1e1e; background-color: {bg_color}; "
                    f"border-radius: 10px; padding: 4px 10px;"
                )
            top_row.addWidget(type_tag)

        # Job Number Tag
        if job["number"]:
            # max_width=95 cleanly fits ~8-10 characters before cutting to "..."
            num_tag = HoverScrollLabel(job["number"], max_width=70, elide_mode=Qt.TextElideMode.ElideRight)
            num_tag.setObjectName("TagLabel")
            # Alternate color for the second tag
            num_tag.setStyleSheet("color: #32D74B; background-color: rgba(50, 215, 75, 0.15); border-radius: 10px; padding: 4px 10px;")
            top_row.addWidget(num_tag)
            
        # Cancel Button (Standard X to fix Linux font rendering issues)
        cancel_btn = QPushButton("×")
        cancel_btn.setObjectName("CancelBtn")
        cancel_btn.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        cancel_btn.setToolTip("Cancel this job")
        cancel_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        # We capture the specific job id using default argument in lambda
        cancel_btn.clicked.connect(lambda checked, jid=job["id"]: self.cancel_job(jid))
        top_row.addWidget(cancel_btn)

        card_layout.addLayout(top_row)

        # Bottom Row: Directory Path and RunTime
        bottom_row = QHBoxLayout()
        
        import os
        # normpath automatically cleans up trailing slashes (/) and dot folders (/./)
        clean_dir = os.path.normpath(job['dir'])
        dir_text = f"Directory: {clean_dir}"
        
        # max_width=250 physically forces it down to ~25 visible characters.
        # is_path=True tells it to activate the dual-color text render.
        dir_label = HoverScrollLabel(dir_text, max_width=250, elide_mode=Qt.TextElideMode.ElideMiddle, is_path=True)
        dir_label.setObjectName("DirLabel")
        # Pass mouse clicks through the text so you can drag the card from the path
        dir_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        bottom_row.addWidget(dir_label)
        
        bottom_row.addStretch()
        
        # RunTime Label
        raw_rt = job.get("runtime", "00:00:00")
        if job["state"] == "RUNNING" and raw_rt not in ["00:00:00", "00:00", "Unknown"]:
            # Slurm outputs time as DD-HH:MM:SS or HH:MM:SS or MM:SS
            rt_clean = raw_rt.replace(":", "-")
            parts = rt_clean.split("-")
            
            d, h, m, s = 0, 0, 0, 0
            if len(parts) == 4:
                d, h, m, s = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
            elif len(parts) == 3:
                h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
            elif len(parts) == 2:
                m, s = int(parts[0]), int(parts[1])
                
            # Pass the parsed time into our live ticking widget
            time_label = ActiveTimeLabel(d, h, m, s)
            bottom_row.addWidget(time_label)
            
        card_layout.addLayout(bottom_row)
        
        # Right-click context menu setup
        card.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        card.customContextMenuRequested.connect(
            lambda pos, jid=job["id"], n=job.get("nice"): self.show_context_menu(card, pos, jid, n)
        )

        return card

    def cancel_job(self, job_id):
        # Native OS confirmation dialog
        reply = QMessageBox.question(
            self, 
            "Confirm Cancel", 
            f"Are you sure you want to cancel Job ID {job_id}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Issue the scancel command
                subprocess.run(["scancel", job_id], check=True)
                self.status_label.setText(f"Job {job_id} cancelled.")
                
                # Auto-refresh the list
                self.refresh_jobs()
                
            except subprocess.CalledProcessError as e:
                QMessageBox.warning(self, "Error", f"Failed to cancel job {job_id}.\nIt may have already finished.")
            except FileNotFoundError:
                QMessageBox.warning(self, "Error", "'scancel' command not found. Are you running this on the HPC node?")
                
    def show_context_menu(self, card_widget, pos, job_id, nice_val):
        menu = QMenu(self)
        
        # Action: Remove Nice
        remove_nice_action = menu.addAction("Remove Nice (Set to 0)")
        if nice_val == "0" or not nice_val:
            remove_nice_action.setEnabled(False)
            
        action = menu.exec(card_widget.mapToGlobal(pos))
        
        if action == remove_nice_action:
            self.update_job_nice(job_id, 0)

    def update_job_nice(self, job_id, nice_value):
        try:
            subprocess.run(["scontrol", "update", f"jobid={job_id}", f"nice={nice_value}"], check=True)
            self.status_label.setText(f"Job {job_id} nice updated to {nice_value}.")
            self.refresh_jobs()
        except subprocess.CalledProcessError as e:
            QMessageBox.warning(self, "Error", f"Failed to update nice for job {job_id}.")
        except FileNotFoundError:
            QMessageBox.warning(self, "Error", "'scontrol' command not found.")

    def remove_all_nice(self):
        pending_jobs_to_reset = []
        for i in range(self.job_list.count()):
            item = self.job_list.item(i)
            job_data = item.data(Qt.ItemDataRole.UserRole)
            
            # Only reset if the job is PENDING and has a nice value > 0
            if job_data and job_data.get("state") == "PENDING" and job_data.get("nice", "0") not in ["0", "Unknown"]:
                pending_jobs_to_reset.append(job_data["id"])

        if not pending_jobs_to_reset:
            QMessageBox.information(self, "Info", "No pending jobs currently have a modified nice value.")
            return

        reply = QMessageBox.question(
            self,
            "Confirm Remove All",
            f"Are you sure you want to reset the nice values for {len(pending_jobs_to_reset)} pending jobs?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.status_label.setText("Resetting nice values...")
            self.refresh_btn.setEnabled(False)
            self.remove_nice_btn.setEnabled(False)
            
            # --- NEW: Kill the auto-reset watcher since we are manually aborting ---
            try:
                user = os.getenv('USER')
                subprocess.run(["scancel", "-u", user, "--name=xcop_Qrst"], check=False, stderr=subprocess.DEVNULL)
            except Exception:
                pass
            
            for jid in pending_jobs_to_reset:
                try:
                    subprocess.run(["scontrol", "update", f"jobid={jid}", "nice=0"], check=True)
                except Exception as e:
                    print(f"Silently passing nice update failure for {jid}: {e}")
            
            # Refresh to fetch the newly clean queue state
            self.refresh_jobs()
            
    def get_currently_prioritized_jobs(self):
        try:
            user = os.getenv('USER')
            # Look at the 'xcop_Qrst' job and extract its dependency string (%E)
            out = subprocess.check_output(
                ["squeue", "-u", user, "--name=xcop_Qrst", "-h", "-o", "%E"], 
                universal_newlines=True
            ).strip()
            
            prioritized_ids = []
            if out:
                # Safely extract any job IDs from the dependency string using regex
                prioritized_ids = re.findall(r'\d+', out)
            return prioritized_ids
        except Exception:
            pass
        return []

    def get_job_data_by_id(self, jid):
        for i in range(self.job_list.count()):
            item = self.job_list.item(i)
            job_data = item.data(Qt.ItemDataRole.UserRole)
            if job_data and job_data.get("id") == jid:
                return job_data
        return None

    def top_selected_job(self):
        selected_items = self.job_list.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "Info", "Please select at least one pending job to top.")
            return

        # 1. Extract all valid PENDING job IDs from the current selection
        top_job_ids = []
        dependent_jobs_warn = []
        
        for item in selected_items:
            job_data = item.data(Qt.ItemDataRole.UserRole)
            if job_data and job_data.get("state") == "PENDING":
                top_job_ids.append(job_data["id"])
                
                # Check if this job has a native Slurm dependency
                dep = job_data.get("dependency")
                if dep:
                    dependent_jobs_warn.append((job_data["id"], dep))

        if not top_job_ids:
            QMessageBox.information(self, "Info", "Only pending jobs can be topped.")
            return

        # --- NEW: Intercept and warn if topping a dependent job ---
        if dependent_jobs_warn:
            warning_msg = "You are attempting to top jobs that have a native dependency:\n\n"
            for jid, dep in dependent_jobs_warn:
                warning_msg += f"• Job {jid} is waiting for: {dep}\n"
                
            warning_msg += "\nTopping a dependent job will push the job it is waiting for to the bottom of the queue. This will cause MASSIVE delays for both jobs instead of prioritizing them.\n\nAre you absolutely sure you want to do this?"
            
            reply = QMessageBox.warning(
                self,
                "Dependency Warning",
                warning_msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            
            # If the user clicks "No", instantly abort the function
            if reply == QMessageBox.StandardButton.No:
                return

        # --- NEW: Priority Override Warning Dialog ---
        old_prioritized_ids = self.get_currently_prioritized_jobs()
        actual_old_jobs = []
        
        for jid in old_prioritized_ids:
            if jid not in top_job_ids:  # Don't warn about jobs they are currently re-topping
                jdata = self.get_job_data_by_id(jid)
                # Only warn if the old job is STILL pending in the queue
                if jdata and jdata.get("state") == "PENDING":
                    actual_old_jobs.append(jdata)
                    
        if actual_old_jobs:
            new_jobs = [self.get_job_data_by_id(jid) for jid in top_job_ids]
            dialog = OverrideWarningDialog(self, new_jobs, actual_old_jobs)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return # Abort if they click No
        # ---------------------------------------------

        # 2. Gather all OTHER pending jobs in their current visual order
        other_pending_job_ids = []
        for i in range(self.job_list.count()):
            item = self.job_list.item(i)
            job_data = item.data(Qt.ItemDataRole.UserRole)
            if job_data and job_data.get("state") == "PENDING" and job_data["id"] not in top_job_ids:
                other_pending_job_ids.append(job_data["id"])

        # 3. Construct the new sequence: Topped jobs first, then the rest
        new_order = top_job_ids + other_pending_job_ids

        self.status_label.setText(f"Topping {len(top_job_ids)} jobs...")
        self.refresh_btn.setEnabled(False)
        self.remove_nice_btn.setEnabled(False)
        self.top_btn.setEnabled(False)

        # 4. Apply new nice values to enforce this order 
        for index, jid in enumerate(new_order):
            new_nice = index * 1000
            try:
                subprocess.run(["scontrol", "update", f"jobid={jid}", f"nice={new_nice}"], check=True)
            except Exception as e:
                print(f"Silently passing nice update failure for {jid}: {e}")

        # --- AUTO-RESET HELPER ---
        current_user = os.getenv('USER')
        try:
            # Removed the incompatible -W flag and explicitly added the -u user flag
            subprocess.run(["scancel", "-u", current_user, "--name=xcop_Qrst"], check=False, stderr=subprocess.DEVNULL)
        except Exception:
            pass
        
        # 5. Build the multi-job dependency string (e.g., after:1234:5678)
        # Slurm will not execute the reset script until ALL of these jobs have started running!
        dependency_str = ":".join(top_job_ids)
            
        reset_script = f"sleep 5; for j in $(squeue -u {current_user} -t PENDING -h -o '%A'); do scontrol update jobid=$j nice=0; done"
        
        helper_cmd = [
            "sbatch",
            "--job-name=xcop_Qrst",
            "--output=/dev/null",
            "--time=00:01:00",
            "--ntasks=1",
            "--mem=100M",
            f"--dependency=after:{dependency_str}",
            "--wrap", reset_script
        ]
        
        try:
            subprocess.run(helper_cmd, check=True, stdout=subprocess.DEVNULL)
            print(f"Spawned auto-reset watcher waiting for jobs: {dependency_str}")
        except Exception as e:
            print(f"Failed to spawn auto-reset watcher: {e}")
                
        # Refresh to redraw the widgets accurately
        self.refresh_jobs()

# ---------------------------------------------------------
# APPLICATION ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE_SHEET)
    
    window = SlurmMonitor()
    window.show()
    
    sys.exit(app.exec())
