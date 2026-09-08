import os
import sys
import math
import subprocess

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QPushButton, QCheckBox, QFileDialog, QMessageBox, QLineEdit)
from PyQt6.QtCore import Qt

# 1. Recreate required global variables so the script doesn't crash
toplevel_windows = []

THEME = {
    'bg': '#1e1e1e', 'fg': '#d4d4d4', 'button_bg': '#3e3e42', 
    'entry_bg': '#3c3c3c', 'select_bg': '#98c379'
}

def show_error_message(message):
    msg = QMessageBox()
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setWindowTitle("Error")
    msg.setText(message)
    msg.exec()

def center_window(window, width, height, window_type):
    # Center the window safely on the screen
    screen_geometry = QApplication.primaryScreen().availableGeometry()
    x = (screen_geometry.width() - width) // 2
    y = (screen_geometry.height() - height) // 2
    window.setGeometry(x, y, width, height)

# ====================== atom2svg (Ported to PyQt6) ======================
def open_atom2svg():
    # Add to imports section
    try:
        import numpy as np
        from numpy.linalg import eigh
        from PyQt6.QtGui import QPainter, QPen, QColor
        from PyQt6.QtCore import QPointF
        from PyQt6.QtWidgets import QCheckBox
    except ImportError:
        print("Numpy or PyQt6 GUI modules missing.")

    # Constants
    RADIUS = 10
    MARGIN = 100
    CB_FUDGE = 1.5
    EXPORT_SCALE = 8

    one_letter_code = {
        'ARG': 'R', 'HIS': 'H', 'LYS': 'K', 'ASP': 'D', 'GLU': 'E',
        'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'CYS': 'C',
        'GLY': 'G', 'PRO': 'P', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I',
        'LEU': 'L', 'MET': 'M', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W'
    }
    residue_class = {
        'ARG': 'b', 'HIS': 'b', 'LYS': 'b', 'ASP': 'a', 'GLU': 'a',
        'SER': 'w', 'THR': 'w', 'ASN': 'w', 'GLN': 'w', 'CYS': 's',
        'GLY': 'g', 'PRO': 'p', 'ALA': 'n', 'VAL': 'n', 'ILE': 'n',
        'LEU': 'n', 'MET': 's', 'PHE': 'n', 'TYR': 'n', 'TRP': 'n'
    }

    # Math Helpers
    def compute_face_on_rotation_matrix(c_alpha_dict):
        points = np.array([coord[:3] for coord in c_alpha_dict.values()])
        if len(points) < 3:
            return np.eye(3)
        
        points_centered = points - points.mean(axis=0)
        cov = np.cov(points_centered.T)
        eigenvals, eigenvecs = eigh(cov)
        idx = eigenvals.argsort()[::-1]
        
        major_axis = eigenvecs[:, idx[0]]
        medium_axis = eigenvecs[:, idx[1]]
        minor_axis = eigenvecs[:, idx[2]]
        
        R = np.array([major_axis, minor_axis, medium_axis])
        
        rotated_points = points_centered @ R.T
        if np.max(rotated_points[:, 0]) < -np.min(rotated_points[:, 0]): R[0, :] *= -1
        if np.max(rotated_points[:, 2]) < -np.min(rotated_points[:, 2]): R[2, :] *= -1
        if np.linalg.det(R) < 0: R[1, :] *= -1
        return R

    def get_axis_rotation(axis, theta):
        axis = np.asarray(axis)
        axis = axis / math.sqrt(np.dot(axis, axis))
        a = math.cos(theta / 2.0)
        b, c, d = -axis * math.sin(theta / 2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d
        return np.array([
            [aa+bb-cc-dd, 2*(bc+ad), 2*(bd-ac)],
            [2*(bc-ad), aa+cc-bb-dd, 2*(cd+ab)],
            [2*(bd+ac), 2*(cd-ab), aa+dd-bb-cc]
        ])

    # File Selection
    current_dir = os.getcwd()
    pdb_path, _ = QFileDialog.getOpenFileName(None, "Select PDB file", current_dir, "PDB files (*.pdb);;All files (*.*)")
    if not pdb_path:
        return

    # Parse PDB
    c_alpha = {}
    c_beta = {}
    coords = []
    
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    try:
                        atom = line[12:16].strip()
                        resn = line[17:20].strip()
                        chain = line[21]
                        resSeq = int(line[22:26])
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        key = (chain, resSeq)
                        if atom == "CA":
                            c_alpha[key] = (x, y, z, resn)
                            coords.append((x, y, z))
                        if atom == "CB" or (atom == "CA" and resn == "GLY"):
                            c_beta[key] = (x, y, z, resn)
                    except: continue
    except Exception as e:
        show_error_message(f"Failed to parse PDB: {e}")
        return

    if not c_alpha:
        show_error_message("No Cα atoms found")
        return

    # Center Coordinates
    ca_coords = [(x,y,z) for (x,y,z,_) in c_alpha.values()]
    xs, ys, zs = np.array(ca_coords).T
    cx, cy, cz = np.median(xs), np.median(ys), np.median(zs)

    for k in c_alpha:
        x, y, z, r = c_alpha[k]
        c_alpha[k] = (x - cx, y - cy, z - cz, r)
    for k in c_beta:
        if k in c_beta:
            x, y, z, r = c_beta[k]
            c_beta[k] = (x - cx, y - cy, z - cz, r)

    # Custom Drawing Widget
    class MoleculeViewer(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setMouseTracking(True)
            self.zoom = 6.0
            self.rot_matrix = get_axis_rotation([1, 0, 0], math.pi / 2)
            
            # Interaction State
            self.drag = False
            self.last_pos = QPointF()
            self.right_drag = False
            self.axis_hit_zones = []  # Store coordinates of drawn axes for hit detection
            
            # Constraints
            self.lock_x = False
            self.lock_y = False
            self.lock_z = False
            
            # --- Angle Editing State ---
            from PyQt6.QtWidgets import QLineEdit
            from PyQt6.QtGui import QDoubleValidator
            self.active_edit = None
            self.current_angles = (90.0, 0.0, 0.0)
            
            self.edit_x = QLineEdit(self)
            self.edit_y = QLineEdit(self)
            self.edit_z = QLineEdit(self)
            
            for ed in (self.edit_x, self.edit_y, self.edit_z):
                ed.setValidator(QDoubleValidator())
                ed.hide()
                ed.returnPressed.connect(self.confirm_angle_edits)
            
        def align_to_axis(self, axis_name):
            """
            Aligns the view so the clicked local axis points towards the screen.
            Global Coords: X=Right, Z=Up, -Y=Towards Viewer
            """
            if axis_name == 'X':
                # Map Local X -> Global -Y (Towards)
                # Map Local Y -> Global X  (Right)
                # Map Local Z -> Global Z  (Up)
                self.rot_matrix = np.array([
                    [0.0, 1.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0]
                ])
            elif axis_name == 'Y':
                # Map Local Y -> Global -Y (Towards)
                # Map Local X -> Global -X (Left)
                # Map Local Z -> Global Z  (Up)
                self.rot_matrix = np.array([
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ])
            elif axis_name == 'Z':
                # Map Local Z -> Global -Y (Towards)
                # Map Local X -> Global X  (Right)
                # Map Local Y -> Global Z  (Up)
                self.rot_matrix = np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0],
                    [0.0, 1.0, 0.0]
                ])
            self.update()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Background
            painter.fillRect(self.rect(), QColor(THEME['bg']))
            
            w, h = self.width(), self.height()
            cx, cy = w / 2, h / 2
            
            # Calculate bounding sphere radius for consistent depth effect
            if not hasattr(self, 'max_radius'):
                if c_alpha:
                    self.max_radius = max([np.linalg.norm([x, y, z]) for x, y, z, _ in c_alpha.values()])
                    if self.max_radius == 0: self.max_radius = 1.0
                else:
                    self.max_radius = 1.0

            # Project Points
            proj = {}
            
            # Projection loop
            for key, (x, y, z, _) in c_alpha.items():
                vec = np.array([x, y, z])
                rotated = self.rot_matrix @ vec
                # Map to screen: Global X -> Screen X, Global Z -> Screen -Y (Up)
                px = rotated[0] * self.zoom + cx
                py = -rotated[2] * self.zoom + cy
                pz = rotated[1] # depth (rotated Y axis corresponds to depth)
                proj[key] = (px, py, pz)

            # Global depth bounds based on the molecule's bounding sphere
            # This ensures the depth effect doesn't pop or scale wildly depending on the viewing angle
            min_z = -self.max_radius
            z_range = 2.0 * self.max_radius

            # Collect and sort lines for depth effect (Painter's algorithm + Alpha fading)
            lines_to_draw = []
            keys = sorted(c_alpha.keys())
            for (chain, resSeq) in keys:
                prev = (chain, resSeq - 1)
                if prev in proj and (chain, resSeq) in proj:
                    p1 = proj[prev]
                    p2 = proj[(chain, resSeq)]
                    avg_z = (p1[2] + p2[2]) / 2.0
                    lines_to_draw.append((avg_z, p1, p2))
            
            # Sort by depth descending (furthest first, assuming +Y points away)
            lines_to_draw.sort(key=lambda item: item[0], reverse=True)

            base_color = QColor(THEME['fg'])
            r, g, b = base_color.red(), base_color.green(), base_color.blue()

            for avg_z, p1, p2 in lines_to_draw:
                norm_z = (avg_z - min_z) / z_range
                
                # Stronger Alpha drop-off (Non-linear for enhanced depth perception)
                depth_factor = norm_z ** 1.5 
                
                # Closer -> alpha ~ 255; Furthest -> alpha ~ 25
                alpha = int(255 - (depth_factor * 230))
                alpha = max(25, min(255, alpha))
                
                # Size attenuation: Closer -> 5.0 px; Furthest -> 1.0 px
                thickness = 5.0 - (norm_z * 4.0)
                thickness = max(1.0, thickness)
                
                pen = QPen(QColor(r, g, b, alpha), thickness)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap) # Keeps the joints looking smooth with varying thickness
                painter.setPen(pen)
                painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))

            # Draw Axis Widget (Bottom Right)
            self.draw_axis_widget(painter, w, h)

        def draw_axis_widget(self, painter, w, h):
            self.axis_hit_zones = [] # Reset hit zones
            size = 40
            padding = 60
            ox, oy = w - padding, h - padding
            
            axes = [
                (np.array([1.0, 0.0, 0.0]), QColor("#FF4444"), "X"), 
                (np.array([0.0, 1.0, 0.0]), QColor("#44FF44"), "Y"), 
                (np.array([0.0, 0.0, 1.0]), QColor("#4488FF"), "Z") 
            ]
            
            # Sort by depth (Y axis in rotated space corresponds to depth)
            draw_list = []
            for vec, col, label in axes:
                rotated = self.rot_matrix @ vec
                px = ox + rotated[0] * size
                py = oy - rotated[2] * size
                depth = rotated[1]
                draw_list.append((depth, ox, oy, px, py, col, label))
            
            # Painter's algorithm: draw furthest first
            draw_list.sort(key=lambda x: x[0], reverse=True) 
            
            font = painter.font()
            font.setBold(True)
            painter.setFont(font)
            
            for _, x1, y1, x2, y2, col, lbl in draw_list:
                pen = QPen(col, 3)
                painter.setPen(pen)
                painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
                
                # Store coordinates for click detection
                self.axis_hit_zones.append((lbl, x1, y1, x2, y2))
                
                # Text
                painter.setPen(col)
                painter.drawText(int(x2), int(y2), lbl)

            # --- Extract and Draw Euler Angles ---
            R = self.rot_matrix
            sy = math.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
            singular = sy < 1e-6
            if not singular:
                rx = math.degrees(math.atan2(R[2,1], R[2,2]))
                ry = math.degrees(math.atan2(-R[2,0], sy))
                rz = math.degrees(math.atan2(R[1,0], R[0,0]))
            else:
                rx = math.degrees(math.atan2(-R[1,2], R[1,1]))
                ry = math.degrees(math.atan2(-R[2,0], sy))
                rz = 0

            # Save current angles to state for editing
            self.current_angles = (rx, ry, rz)
            
            # Draw text below the axis widget
            painter.setPen(QColor(THEME['fg']))
            font.setBold(False)
            painter.setFont(font)
            
            # Move text up slightly to prevent it from getting cut off at the bottom edge
            text_x = ox - 145
            text_y = oy + 45 
            
            active = getattr(self, 'active_edit', None)
            
            # Draw X part
            painter.drawText(int(text_x), int(text_y), "X:")
            if active != 'x':
                painter.drawText(int(text_x + 15), int(text_y), f"{rx:5.1f}°")
                
            # Draw Y part
            painter.drawText(int(text_x + 65), int(text_y), "Y:")
            if active != 'y':
                painter.drawText(int(text_x + 80), int(text_y), f"{ry:5.1f}°")
                
            # Draw Z part
            painter.drawText(int(text_x + 130), int(text_y), "Z:")
            if active != 'z':
                painter.drawText(int(text_x + 145), int(text_y), f"{rz:5.1f}°")

        def mousePressEvent(self, event):
            # If the user clicks away while editing, confirm the angle
            if getattr(self, 'active_edit', None) is not None:
                self.confirm_angle_edits()
                
            # PyQt6: event.position().toPoint()
            pos = event.position().toPoint()
            
            # 1. Check for Axis Click (Left Click only)
            # PyQt6: Qt.MouseButton.LeftButton
            if event.button() == Qt.MouseButton.LeftButton:
                p = np.array([pos.x(), pos.y()])
                for lbl, x1, y1, x2, y2 in self.axis_hit_zones:
                    # Distance from point p to line segment a-b
                    a = np.array([x1, y1])
                    b = np.array([x2, y2])
                    ab = b - a
                    length_sq = np.dot(ab, ab)
                    
                    if length_sq == 0:
                        dist = np.linalg.norm(p - a)
                    else:
                        # Projection factor t
                        t = max(0, min(1, np.dot(p - a, ab) / length_sq))
                        projection = a + t * ab
                        dist = np.linalg.norm(p - projection)
                    
                    if dist < 10:  # 10px tolerance
                        self.align_to_axis(lbl)
                        return  # Stop processing, do not drag

            # 2. Normal Rotation Drag
            self.drag = True
            self.last_pos = pos
            # PyQt6: Qt.MouseButton.RightButton
            self.right_drag = (event.button() == Qt.MouseButton.RightButton)

        def mouseDoubleClickEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton:
                w, h = self.width(), self.height()
                ox, oy = w - 60, h - 60
                text_x = ox - 145
                text_y = oy + 45

                ex, ey = event.position().toPoint().x(), event.position().toPoint().y()
                
                # Check vertical bounds
                if text_y - 20 <= ey <= text_y + 10:
                    # Check which axis was clicked horizontally
                    if text_x - 10 <= ex < text_x + 60:
                        self.show_angle_edits(text_x, text_y, 'x')
                    elif text_x + 60 <= ex < text_x + 125:
                        self.show_angle_edits(text_x, text_y, 'y')
                    elif text_x + 125 <= ex <= text_x + 190:
                        self.show_angle_edits(text_x, text_y, 'z')

        def show_angle_edits(self, x, y, axis):
            rx, ry, rz = self.current_angles
            self.edit_x.setText(f"{rx:.1f}")
            self.edit_y.setText(f"{ry:.1f}")
            self.edit_z.setText(f"{rz:.1f}")

            box_y = int(y - 15)
            
            # Use specific font size and 0 padding to prevent the text from shrinking
            style = f"background-color: {THEME['entry_bg']}; color: {THEME['fg']}; border: 1px solid #666; font-size: 13px; padding: 0px;"

            self.active_edit = axis

            if axis == 'x':
                self.edit_x.setGeometry(int(x + 15), box_y, 45, 20)
                self.edit_x.setStyleSheet(style)
                self.edit_x.show()
                self.edit_x.setFocus()
                self.edit_x.selectAll()
            elif axis == 'y':
                self.edit_y.setGeometry(int(x + 80), box_y, 45, 20)
                self.edit_y.setStyleSheet(style)
                self.edit_y.show()
                self.edit_y.setFocus()
                self.edit_y.selectAll()
            elif axis == 'z':
                self.edit_z.setGeometry(int(x + 145), box_y, 45, 20)
                self.edit_z.setStyleSheet(style)
                self.edit_z.show()
                self.edit_z.setFocus()
                self.edit_z.selectAll()

            self.update()

        def confirm_angle_edits(self):
            if not getattr(self, 'active_edit', None): return
            
            # Grab the current painted values for axes we aren't editing, and the textbox value for the one we are
            try:
                rx = float(self.edit_x.text()) if self.active_edit == 'x' else self.current_angles[0]
                ry = float(self.edit_y.text()) if self.active_edit == 'y' else self.current_angles[1]
                rz = float(self.edit_z.text()) if self.active_edit == 'z' else self.current_angles[2]

                # Convert degrees to radians and construct the new rotation matrix
                rad_x, rad_y, rad_z = math.radians(rx), math.radians(ry), math.radians(rz)
                mat_x = get_axis_rotation([1, 0, 0], rad_x)
                mat_y = get_axis_rotation([0, 1, 0], rad_y)
                mat_z = get_axis_rotation([0, 0, 1], rad_z)
                
                # Apply Z * Y * X matrix multiplication
                self.rot_matrix = mat_z @ mat_y @ mat_x
            except ValueError:
                pass # Ignore if invalid float entered

            # Hide text boxes
            for ed in (self.edit_x, self.edit_y, self.edit_z):
                ed.hide()
            self.active_edit = None
            self.update()

        def mouseMoveEvent(self, event):
            if not self.drag: return
            
            # PyQt6: event.position().toPoint()
            curr_pos = event.position().toPoint()
            dx = curr_pos.x() - self.last_pos.x()
            dy = curr_pos.y() - self.last_pos.y()
            self.last_pos = curr_pos
            
            is_x, is_y, is_z = self.lock_x, self.lock_y, self.lock_z
            if self.right_drag: is_y = True
            
            count = sum([is_x, is_y, is_z])
            if count >= 2: return
            
            sens = 0.01
            R_delta = np.eye(3)
            
            if count == 0:
                rx = get_axis_rotation([1,0,0], dy * sens)
                rz = get_axis_rotation([0,0,1], dx * sens)
                R_delta = rx @ rz
            elif is_x:
                R_delta = get_axis_rotation([1,0,0], dy * sens)
            elif is_y:
                R_delta = get_axis_rotation([0,1,0], dx * sens)
            elif is_z:
                R_delta = get_axis_rotation([0,0,1], dx * sens)
                
            self.rot_matrix = R_delta @ self.rot_matrix
            self.update()

        def mouseReleaseEvent(self, event):
            self.drag = False

        def wheelEvent(self, event):
            angle = event.angleDelta().y()
            factor = 1.15 if angle > 0 else 0.87
            self.zoom *= factor
            self.update()
            
        def reset_view(self):
            self.rot_matrix = get_axis_rotation([1, 0, 0], math.pi / 2)
            self.zoom = 6.0
            self.update()
            
        def flip_axis(self, axis_vec):
            self.rot_matrix = get_axis_rotation(axis_vec, math.pi) @ self.rot_matrix
            self.update()

        def rotate_90(self, axis_vec):
            self.rot_matrix = get_axis_rotation(axis_vec, math.pi/2) @ self.rot_matrix
            self.update()
            
        def auto_orient(self):
            self.rot_matrix = compute_face_on_rotation_matrix(c_alpha)
            self.update()

    # GUI Setup
    win = QWidget()
    win.setWindowTitle(f"atom2svg - {os.path.basename(pdb_path)}")
    center_window(win, 1200, 720, "atom2svg")
    toplevel_windows.append(win)
    qt_app = QApplication.instance()
    if qt_app: 
        win.setStyleSheet(qt_app.styleSheet())
    
    layout = QVBoxLayout(win)
    viewer = MoleculeViewer()
    layout.addWidget(viewer)
    
    # Controls
    controls_layout = QHBoxLayout()
    
    def btn(txt, func):
        b = QPushButton(txt)
        b.clicked.connect(func)
        controls_layout.addWidget(b)
        return b
        
    btn("Reset View", viewer.reset_view)
    btn("Flip X", lambda: viewer.flip_axis([1,0,0]))
    btn("Flip Y", lambda: viewer.flip_axis([0,1,0]))
    btn("Flip Z", lambda: viewer.flip_axis([0,0,1]))
    btn("X 90°", lambda: viewer.rotate_90([1,0,0]))
    btn("Y 90°", lambda: viewer.rotate_90([0,1,0]))
    btn("Z 90°", lambda: viewer.rotate_90([0,0,1]))
    
    btn_auto = btn("Auto Orient", viewer.auto_orient)
    btn_auto.setToolTip("Automatically orients in the direction that has the largest projection surface area to you, may not be symmetrical")
    
    # Checkboxes
    cb_layout = QHBoxLayout()
    
    def cb_toggled(state, attr):
        setattr(viewer, attr, state)
        
    cb_x = QCheckBox("Lock X"); cb_x.toggled.connect(lambda s: cb_toggled(s, 'lock_x'))
    cb_y = QCheckBox("Lock Y"); cb_y.toggled.connect(lambda s: cb_toggled(s, 'lock_y'))
    cb_z = QCheckBox("Lock Z"); cb_z.toggled.connect(lambda s: cb_toggled(s, 'lock_z'))
    
    # Styling checkboxes to match theme
    chk_color = THEME['select_bg']
    style = f"""
        QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid #666; border-radius: 2px; background-color: {THEME['entry_bg']}; }}
        QCheckBox::indicator:checked {{ background-color: {chk_color}; border: 1px solid {chk_color}; }}
    """
    for cb in [cb_x, cb_y, cb_z]:
        cb.setStyleSheet(style)
        cb_layout.addWidget(cb)
        
    controls_layout.addLayout(cb_layout)
    layout.addLayout(controls_layout)

    # Export Logic
    def export_vector(fmt="svg"):
        if fmt == "svg":
            path, _ = QFileDialog.getSaveFileName(win, "Save SVG", "schematic.svg", "SVG files (*.svg)")
        elif fmt == "pdf":
            path, _ = QFileDialog.getSaveFileName(win, "Save PDF", "schematic.pdf", "PDF files (*.pdf)")
        else:
            path, _ = QFileDialog.getSaveFileName(win, "Save PS", "schematic.ps", "PostScript files (*.ps)")
            
        if not path: return
        
        actual_path = path if fmt != "pdf" else path + ".ps"
        sc = EXPORT_SCALE
        
        # Parse Sheets for Export
        beta_sheet_residues = set()
        beta_sheet_ends = {}
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith("SHEET"):
                    try:
                        chain = line[21]
                        start = int(line[22:26])
                        end = int(line[33:37])
                        for r in range(start, end + 1): beta_sheet_residues.add((chain, r))
                        beta_sheet_ends[(chain, end)] = True
                    except: continue

        def rot(x, y, z):
            vec = np.array([x, y, z])
            rotated = viewer.rot_matrix @ vec
            return rotated[0]*sc, -rotated[2]*sc
            
        # Collect points
        all_x, all_y = [], []
        sidechain_pos = {}
        
        for key in c_alpha:
            x, y, z, resn = c_alpha[key]
            px, py = rot(x,y,z)
            all_x.append(px); all_y.append(py)
            
            if key in c_beta:
                bx, by, bz, _ = c_beta[key]
                sx, sy = rot(bx, by, bz)
                pos_x = CB_FUDGE * sx + (1 - CB_FUDGE) * px
                pos_y = CB_FUDGE * sy + (1 - CB_FUDGE) * py
                sidechain_pos[key] = (pos_x, pos_y)
                all_x.append(pos_x); all_y.append(pos_y)
            else:
                sidechain_pos[key] = (px, py)

        if not all_x: return 
        
        min_x, min_y = min(all_x) - MARGIN, min(all_y) - MARGIN
        max_x, max_y = max(all_x) + MARGIN, max(all_y) + MARGIN
        width, height = max_x - min_x, max_y - min_y
        
        with open(actual_path, "w") as f:
            if fmt == "svg":
                f.write(f'''<?xml version="1.0" encoding="UTF-8"?>
<svg width="{int(width)}px" height="{int(height)}px" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
<style>
.mainchain {{stroke: black; stroke-width: 2;}}
.sidechain {{stroke: black; stroke-width: 2;}}
circle {{stroke: black; stroke-width: 2;}}
.a {{fill: red;}} .b {{fill: skyblue;}} .w {{fill: green;}} .g {{fill: pink;}}
.n {{fill: white;}} .s {{fill: yellow;}} .p {{fill: purple;}}
text {{font-family: Arial, sans-serif; font-size: 16px; text-anchor: middle; dominant-baseline: central;}}
</style>
<g transform="translate({-min_x},{ -min_y})">
''')
            else:
                f.write(f'''%!PS-Adobe-3.0 EPSF-3.0
%%BoundingBox: 0 0 {int(width)} {int(height)}
/Helvetica findfont 16 scalefont setfont
''')

            # Draw Backbone
            for (chain, resSeq) in sorted(c_alpha):
                prev = (chain, resSeq - 1)
                if prev not in c_alpha: continue
                
                curr_key = (chain, resSeq)
                x2, y2 = rot(*c_alpha[curr_key][:3])
                x1, y1 = rot(*c_alpha[prev][:3])
                
                is_beta = curr_key in beta_sheet_residues and prev in beta_sheet_residues
                is_end = curr_key in beta_sheet_ends
                
                # Calculate vector for all segments to allow extension
                dx, dy = x2 - x1, y2 - y1
                length = (dx**2 + dy**2)**0.5
                if length < 1e-6: continue
                ux, uy = dx/length, dy/length
                
                # Extend segments by half the thinnest width (1.2) to close gaps
                ext = 1.2
                ex1, ey1 = x1 - ux * ext, y1 - uy * ext
                ex2, ey2 = x2 + ux * ext, y2 + uy * ext
                
                if is_beta:
                    lx2, ly2 = ex2, ey2
                    if is_end:
                        arrow_len, arrow_w = 14, 8
                        lx2 = x2 - arrow_len * ux
                        ly2 = y2 - arrow_len * uy
                        ax1 = x2 - arrow_len * ux + arrow_w * uy
                        ay1 = y2 - arrow_len * uy - arrow_w * ux
                        ax2 = x2 - arrow_len * ux - arrow_w * uy
                        ay2 = y2 - arrow_len * uy + arrow_w * ux
                        
                        if fmt == "svg":
                            f.write(f'  <polygon points="{x2:.1f},{y2:.1f} {ax1:.1f},{ay1:.1f} {ax2:.1f},{ay2:.1f}" fill="black"/>\n')
                        else:
                            f.write(f'  newpath {x2-min_x:.1f} {height-(y2-min_y):.1f} moveto {ax1-min_x:.1f} {height-(ay1-min_y):.1f} lineto {ax2-min_x:.1f} {height-(ay2-min_y):.1f} lineto closepath 0 0 0 setrgbcolor fill\n')
                    
                    if fmt == "svg":
                        f.write(f'  <line x1="{ex1:.1f}" y1="{ey1:.1f}" x2="{lx2:.1f}" y2="{ly2:.1f}" stroke="black" stroke-width="7"/>\n')
                    else:
                        f.write(f'  newpath {ex1-min_x:.1f} {height-(ey1-min_y):.1f} moveto {lx2-min_x:.1f} {height-(ly2-min_y):.1f} lineto 0 0 0 setrgbcolor 7 setlinewidth stroke\n')
                else:
                    if fmt == "svg":
                        f.write(f'  <line x1="{ex1:.1f}" y1="{ey1:.1f}" x2="{ex2:.1f}" y2="{ey2:.1f}" class="mainchain"/>\n')
                    else:
                        f.write(f'  newpath {ex1-min_x:.1f} {height-(ey1-min_y):.1f} moveto {ex2-min_x:.1f} {height-(ey2-min_y):.1f} lineto 0 0 0 setrgbcolor 2 setlinewidth stroke\n')

            # Draw Sidechains
            ps_colors = {'a': "1 0 0", 'b': "0.53 0.81 0.92", 'w': "0 0.5 0", 'g': "1 0.75 0.8", 'n': "1 1 1", 's': "1 1 0", 'p': "0.5 0 0.5"}
            for key, (cx, cy, cz, resn) in c_alpha.items():
                pos_x, pos_y = sidechain_pos[key]
                cax, cay = rot(cx, cy, cz)
                
                if resn != "GLY":
                    if fmt == "svg":
                        f.write(f'  <line x1="{pos_x:.1f}" y1="{pos_y:.1f}" x2="{cax:.1f}" y2="{cay:.1f}" class="sidechain"/>\n')
                    else:
                        f.write(f'  newpath {pos_x-min_x:.1f} {height-(pos_y-min_y):.1f} moveto {cax-min_x:.1f} {height-(cay-min_y):.1f} lineto 0 0 0 setrgbcolor 2 setlinewidth stroke\n')
                
                cls = residue_class.get(resn, "n")
                letter = one_letter_code.get(resn, "?")
                
                # Use white text for dark backgrounds: 'a' (red), 'w' (green), 'p' (purple)
                svg_tc = "white" if cls in ['a', 'w', 'p'] else "black"
                ps_tc = "1 1 1" if cls in ['a', 'w', 'p'] else "0 0 0"

                if fmt == "svg":
                    f.write(f'  <circle cx="{pos_x:.1f}" cy="{pos_y:.1f}" r="{RADIUS}" class="{cls}"/>\n')
                    f.write(f'  <text x="{pos_x:.1f}" y="{pos_y:.1f}" fill="{svg_tc}">{letter}</text>\n')
                else:
                    col = ps_colors.get(cls, "1 1 1")
                    f.write(f'  newpath {pos_x-min_x:.1f} {height-(pos_y-min_y):.1f} {RADIUS} 0 360 arc {col} setrgbcolor fill\n')
                    f.write(f'  newpath {pos_x-min_x:.1f} {height-(pos_y-min_y):.1f} {RADIUS} 0 360 arc 0 0 0 setrgbcolor 2 setlinewidth stroke\n')
                    f.write(f'  {pos_x-min_x:.1f} {height-(pos_y-min_y)-5.5:.1f} moveto ({letter}) dup stringwidth pop 2 div neg 0 rmoveto {ps_tc} setrgbcolor show\n')

            if fmt == "svg":
                f.write('</g>\n</svg>')
                
        if fmt == "pdf":
            try:
                subprocess.run(f"ps2pdf -dEPSCrop -dAutoRotatePages=/None '{actual_path}' '{path}'", shell=True, check=True)
                os.remove(actual_path)
            except Exception as e:
                show_error_message(f"PDF conversion failed (is ps2pdf installed?): {e}")
                return

        QMessageBox.information(win, "Success", f"{fmt.upper()} exported!")

    btn_svg = btn("SVG", lambda: export_vector("svg"))
    btn_pdf = btn("PDF", lambda: export_vector("pdf"))
    btn_ps = btn("PS", lambda: export_vector("ps"))
    btn_pdf.setToolTip("Use this option if you want to paste the schematic in Adobe Illustrator")
    btn_ps.setToolTip("Use this option if you want to paste the schematic in Adobe Illustrator")
    
    win.show()

# === OUTSIDE THE FUNCTION (Standalone Execution) ===
if __name__ == "__main__":
    qt_app = QApplication(sys.argv)
    
    # Inject the exact xcop global stylesheet so it matches the aesthetic
    qt_app.setStyleSheet("""
        QMainWindow, QDialog, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Arial; font-size: 10pt; }
        QPushButton { background-color: #3e3e42; color: #d4d4d4; border: 1px solid #3e3e42; padding: 5px 15px; border-radius: 4px; }
        QPushButton:hover { background-color: #4e4e52; border: 1px solid #98c379; }
        QPushButton:pressed, QPushButton:checked { background-color: #98c379; color: #1e1e1e; border: 1px solid #98c379; }
        QLineEdit, QSpinBox, QTextEdit, QTextBrowser, QTableWidget { background-color: #3c3c3c; border: 1px solid #3c3c3c; color: #cccccc; padding: 4px;}
        QGroupBox { border: 1px solid #3e3e42; border-radius: 4px; margin-top: 1.5em; font-weight: bold; color: #98c379; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; border-radius: 2px; background-color: #1e1e1e; }
        QCheckBox::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        QHeaderView::section { background-color: #252526; color: #98c379; border: 1px solid #3e3e42; padding: 4px; font-weight: bold; }
        QTableCornerButton::section { background-color: #252526; border: 1px solid #3e3e42; }
        
        /* === Scrollbar Styling === */
        QScrollBar:vertical { background: #1e1e1e; width: 12px; margin: 0px 0 0px 0; border-left: 1px solid #3e3e42; }
        QScrollBar::handle:vertical { background: #3e3e42; min-height: 20px; border-radius: 4px; margin: 2px; }
        QScrollBar::handle:vertical:hover { background: #98c379; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        
        QScrollBar:horizontal { background: #1e1e1e; height: 12px; margin: 0px 0 0px 0; border-top: 1px solid #3e3e42; }
        QScrollBar::handle:horizontal { background: #3e3e42; min-width: 20px; border-radius: 4px; margin: 2px; }
        QScrollBar::handle:horizontal:hover { background: #98c379; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }

        /* === Slider Styling === */
        QSlider:vertical { min-width: 20px; }
        QSlider::groove:vertical { background: #3c3c3c; width: 6px; border-radius: 3px; }
        QSlider::handle:vertical { background: #98c379; height: 14px; width: 14px; margin: 0 -4px; border-radius: 7px; }
        QSlider::handle:vertical:hover { background: #b5e890; }
        QSlider::add-page:vertical, QSlider::sub-page:vertical { background: transparent; }
    """)
    
    open_atom2svg()
    
    sys.exit(qt_app.exec())