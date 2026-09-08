import sys
import os
import shutil
import stat
import subprocess
import re
import uuid
import numpy as np
import threading

from PyQt6.QtWidgets import (QApplication, QWidget, QHBoxLayout, QVBoxLayout, QGridLayout, 
                             QPushButton, QLabel, QLineEdit, QTextEdit, QFileDialog, 
                             QSplitter, QGroupBox, QListWidget, QAbstractItemView, QMessageBox,
                             QMenu, QRadioButton, QFormLayout)
from PyQt6.QtCore import Qt, pyqtSignal, QObject, QEvent, QTimer, QMimeData
from PyQt6.QtGui import QColor, QDrag, QPixmap, QAction

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

# --- CUSTOM WIDGETS ---
class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)
    swapped = pyqtSignal(str, str) # Emits (source_key, target_key)

    def __init__(self, text, key, parent=None):
        super().__init__(text, parent)
        self.key = key
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAcceptDrops(True) # Enable dropping on this label
        
        self.drag_start_pos = None

        # Default styling
        self.default_style = "color: #AAAAAA; font-weight: normal; padding: 4px; border: 1px solid transparent;"
        self.active_style = "color: #98c379; font-weight: bold; padding: 4px; border: 1px solid #98c379; background-color: #3e3e3e; border-radius: 4px;"
        self.setStyleSheet(self.default_style)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = event.pos()

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return
        if not self.drag_start_pos:
            return
        
        # Determine if drag threshold is met
        if (event.pos() - self.drag_start_pos).manhattanLength() < QApplication.styleHints().startDragDistance():
            return

        # Start Drag
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.key) # Pass the group key as data
        drag.setMimeData(mime_data)
        
        # Create a visual ghost of the label
        pixmap = self.grab()
        drag.setPixmap(pixmap)
        drag.setHotSpot(event.pos())

        # Executing drag (blocks until drop)
        drag.exec(Qt.DropAction.MoveAction)
        
        # Reset after drag
        self.drag_start_pos = None

    def mouseReleaseEvent(self, event):
        # If we reached here, no drag happened, so it's a click
        if self.drag_start_pos and self.rect().contains(event.pos()):
            self.clicked.emit(self.key)
        self.drag_start_pos = None

    def dragEnterEvent(self, event):
        # Accept drag only if it carries text (our key)
        if event.mimeData().hasText():
            event.acceptProposedAction()

    def dropEvent(self, event):
        source_key = event.mimeData().text()
        target_key = self.key
        
        if source_key != target_key:
            self.swapped.emit(source_key, target_key)
            event.acceptProposedAction()

    def set_active(self, is_active):
        self.setStyleSheet(self.active_style if is_active else self.default_style)

class DraggableListWidget(QListWidget):
    # Signal emitted when an item is dropped into this list
    contentChanged = pyqtSignal()

    def __init__(self, parent=None, is_group_list=False):
        super().__init__(parent)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAcceptDrops(True)

        if is_group_list:
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            self.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos):
        if self.count() == 0: return
        
        menu = QMenu(self)
        
        action_reset = QAction("Move All to Stock", self)
        action_reset.triggered.connect(self.move_all_to_stock)
        menu.addAction(action_reset)
        
        menu.exec(self.mapToGlobal(pos))

    def move_all_to_stock(self):
        self.clear()
        self.contentChanged.emit()

    def dragEnterEvent(self, event):
        # Force MoveAction (prevents copying)
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        # Force MoveAction during the drag hover
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.setDropAction(Qt.DropAction.MoveAction)
            event.accept()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        # Force MoveAction on drop
        if event.mimeData().hasFormat('application/x-qabstractitemmodeldatalist'):
            event.setDropAction(Qt.DropAction.MoveAction)
        
        super().dropEvent(event)
        self.contentChanged.emit()

# --- WORKER THREAD ---
class WorkerSignals(QObject):
    log = pyqtSignal(str)
    result = pyqtSignal(str)
    raw_data = pyqtSignal(str)
    finished = pyqtSignal()
    error = pyqtSignal(str)

class FoldXWorker(QObject):
    def __init__(self, pdb_path, group1, group2, source_dir, is_seq_detail=False, operations=None):
        super().__init__()
        self.pdb_path = pdb_path
        self.group1 = group1 # List of strings
        self.group2 = group2 # List of strings
        self.source_dir = source_dir
        self.is_seq_detail = is_seq_detail
        self.operations = operations or []
        self.signals = WorkerSignals()

    def run(self):
        try:
            # target_dir is now the unique folder created during loading
            target_dir = os.path.dirname(os.path.abspath(self.pdb_path))
            pdb_filename = os.path.basename(self.pdb_path)
            
            # Parent directory for full cleanup later
            parent_dir = os.path.dirname(target_dir)
            
            # Format: Group1,Group2 (e.g., "ABC,EFG")
            str_g1 = "".join(self.group1)
            str_g2 = "".join(self.group2)
            chains_arg = f"{str_g1},{str_g2}"

            self.signals.log.emit(f"Source: {self.source_dir}")
            self.signals.log.emit(f"Target: {target_dir}")
            if not self.is_seq_detail:
                self.signals.log.emit(f"Interface: [{str_g1}] vs [{str_g2}]")
            else:
                self.signals.log.emit(f"Sequence Detail on: [{str_g1}]")
            self.signals.log.emit("-" * 40)

            executable_path = None
            
            if not os.path.exists(self.source_dir):
                 raise Exception(f"FoldX source folder not found: {self.source_dir}")

            # Find the FoldX executable directly in the source directory
            for item_name in os.listdir(self.source_dir):
                if "foldx" in item_name.lower() and not item_name.endswith(".txt"):
                    p = os.path.join(self.source_dir, item_name)
                    if os.path.isfile(p):
                        executable_path = p
                        break
            
            if not executable_path:
                raise Exception(
                    "Could not find a FoldX executable in the FoldX folder. "
                    "FoldX cannot be redistributed with xcop. Obtain your own licensed "
                    "copy from the official FoldX provider and put its distribution files "
                    f"inside: {self.source_dir}"
                )

            st = os.stat(executable_path)
            os.chmod(executable_path, st.st_mode | stat.S_IEXEC)

            # --- RUN PRE-OPERATIONS (RepairPDB / Optimize) ---
            current_pdb_filename = pdb_filename
            cumulative_ops = []
            
            for op in self.operations:
                cumulative_ops.append(op)
                self.signals.log.emit(f"\n--- Running Operation: {op} ---")
                op_cmd = [
                    executable_path,
                    f"--command={op}",
                    f"--pdb-dir={target_dir}/",
                    f"--pdb={current_pdb_filename}"
                ]
                
                # Snapshot PDBs before running
                pdbs_before = set([f for f in os.listdir(target_dir) if f.endswith('.pdb')]) | \
                              set([f for f in os.listdir(self.source_dir) if f.endswith('.pdb')])
                
                self.signals.log.emit(f"Running: {' '.join(op_cmd)}")
                op_process = subprocess.run(op_cmd, cwd=self.source_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                if op_process.stdout: self.signals.log.emit(op_process.stdout)
                if op_process.stderr: self.signals.log.emit(f"STDERR:\n{op_process.stderr}")
                
                # Snapshot PDBs after running to find the newly generated file
                pdbs_after = set([f for f in os.listdir(target_dir) if f.endswith('.pdb')]) | \
                             set([f for f in os.listdir(self.source_dir) if f.endswith('.pdb')])
                
                new_pdbs = list(pdbs_after - pdbs_before)
                if new_pdbs:
                    new_pdb = None
                    # Try to find the specific operation file
                    for f in new_pdbs:
                        if "Repair" in f or "Optimize" in f:
                            new_pdb = f
                            break
                    if not new_pdb:
                        new_pdb = new_pdbs[0] # Fallback to whatever new PDB was created
                    
                    # If FoldX dumped it in source_dir, move it to target_dir where it belongs
                    if os.path.exists(os.path.join(self.source_dir, new_pdb)):
                        shutil.move(os.path.join(self.source_dir, new_pdb), os.path.join(target_dir, new_pdb))
                    
                    current_pdb_filename = new_pdb
                    self.signals.log.emit(f">>> {op} generated new PDB: {current_pdb_filename}")
                    
                    # Copy out to the original folder
                    orig_base = os.path.splitext(pdb_filename)[0].replace("_renamed", "")
                    op_suffix = "_".join(cumulative_ops)
                    export_name = f"{orig_base}_{op_suffix}.pdb"
                    export_path = os.path.join(parent_dir, export_name)
                    
                    try:
                        shutil.copy2(os.path.join(target_dir, current_pdb_filename), export_path)
                        self.signals.log.emit(f">>> Saved intermediate PDB to: {export_path}")
                    except Exception as e:
                        self.signals.log.emit(f">>> Failed to copy PDB to original directory: {e}")
                        
                else:
                    self.signals.log.emit(f">>> WARNING: {op} did not generate a new PDB. Proceeding with {current_pdb_filename}")
                
                # Clean up intermediate logs and unrecognized files by moving them to the parent directory
                for trash in os.listdir(self.source_dir):
                    if trash.endswith(".fxout") or trash == "unrecognized_molecules.txt" or trash.endswith(".rot"):
                        try: 
                            shutil.move(os.path.join(self.source_dir, trash), os.path.join(parent_dir, f"{op}_{trash}"))
                        except: pass
            
            # Lock in the final PDB filename for the SequenceDetail/AnalyseComplex calculation
            pdb_filename = current_pdb_filename

            # Record files in the FoldX working directory before running to clean up outputs later
            files_before = set(os.listdir(self.source_dir))

            if self.is_seq_detail:
                cmd = [
                    executable_path,
                    "--command=SequenceDetail",
                    f"--pdb-dir={target_dir}/",
                    f"--pdb={pdb_filename}"
                ]
            else:
                cmd = [
                    executable_path,
                    "--command=AnalyseComplex",
                    f"--pdb-dir={target_dir}/",
                    f"--pdb={pdb_filename}",
                    f"--analyseComplexChains={chains_arg}"
                ]
            
            self.signals.log.emit(f"Running: {' '.join(cmd)}")
            
            # Run the command from the source_dir where the executable and resources reside
            process = subprocess.run(
                cmd, cwd=self.source_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            output = process.stdout
            self.signals.log.emit(output)
            if process.stderr: self.signals.log.emit(f"STDERR:\n{process.stderr}")

            if self.is_seq_detail:
                fxout_file = None
                for f in os.listdir(self.source_dir):
                    if f.endswith(".fxout") and "SD_" in f:
                        fxout_file = os.path.join(self.source_dir, f)
                        break
                if fxout_file:
                    result_data = {}
                    with open(fxout_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split('\t')
                            if len(parts) >= 9:
                                chain_id = parts[2]
                                try:
                                    energy = float(parts[8])
                                    if chain_id not in result_data:
                                        result_data[chain_id] = {'energy': 0.0, 'residues': 0}
                                    result_data[chain_id]['energy'] += energy
                                    result_data[chain_id]['residues'] += 1
                                except ValueError:
                                    pass
                    import json
                    self.signals.result.emit("SD_RESULT:" + json.dumps(result_data))
                    self.signals.log.emit(f"\n>>> SUCCESS: Sequence Detail calculated.")
                else:
                    self.signals.result.emit("Not Found")
            else:
                captured_energy = None
                interface_block_found = False
                for line in output.splitlines():
                    if "interaction between" in line: interface_block_found = True
                    if interface_block_found:
                        match = re.search(r'Total\s+=\s+([\d\.-]+)', line)
                        if match: captured_energy = match.group(1)
                
                if captured_energy:
                    self.signals.result.emit(f"{captured_energy} kcal/mol")
                    self.signals.log.emit(f"\n>>> SUCCESS: {captured_energy} kcal/mol")
                else:
                    self.signals.result.emit("Not Found")

            # Capture and Cleanup: Read generated .fxout files and move them to the parent directory
            files_after = set(os.listdir(self.source_dir))
            raw_content = []
            for f in (files_after - files_before):
                f_path = os.path.join(self.source_dir, f)
                dest_path = os.path.join(parent_dir, f)
                if f.endswith(".fxout"):
                    try: 
                        with open(f_path, 'r') as rfile:
                            raw_content.append(rfile.read())
                        shutil.move(f_path, dest_path)
                    except: pass
                elif f == "unrecognized_molecules.txt" or f.endswith(".rot"):
                    try: shutil.move(f_path, dest_path)
                    except: pass
            
            # Final sweep just to be safe in case files existed before the snapshot
            if os.path.exists(os.path.join(self.source_dir, "unrecognized_molecules.txt")):
                try: shutil.move(os.path.join(self.source_dir, "unrecognized_molecules.txt"), os.path.join(target_dir, "unrecognized_molecules_final.txt"))
                except: pass
            
            if raw_content:
                self.signals.raw_data.emit("\n\n".join(raw_content))

            self.signals.finished.emit()

        except Exception as e:
            self.signals.error.emit(str(e))

# --- 3D VIEWER ---
class PDBViewerCanvas(FigureCanvas):
    # Signal to emit the Chain ID when clicked in 3D
    chainClicked = pyqtSignal(str)

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), dpi=100)
        self.fig.set_facecolor('#2C2C2C')
        self.axes = self.fig.add_subplot(111, projection='3d')
        self.axes.set_facecolor('#2C2C2C')
        self.axes.axis('off')
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        
        super().__init__(self.fig)
        self.setParent(parent)
        self.updateGeometry()
        
        self.chains_data = {} 
        self.chain1 = []
        self.chain2 = []
        self.chain3 = [] # Hidden Group
        
        # Hover State
        self.hover_enabled = False
        self.chain_artists = {} # {chain_id: line_artist}
        self.hovered_chain = None
        
        # --- Custom Interaction Logic ---
        # 1. Disable Default Matplotlib 3D interaction
        self.axes.mouse_init = lambda: None

        # 2. Bind Custom Events
        self.mpl_connect('scroll_event', self.on_scroll)
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_move)
        self.mpl_connect('pick_event', self.on_pick)

        self.zoom = 50
        self.max_range = 10.0
        self.center = (0,0,0)
        
        # Rotation State
        self.dragging = False
        self.last_mouse = (0, 0)

    def load_pdb(self, path):
        self.chains_data = {}
        all_coords = []
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") and line[12:16].strip() == "CA":
                        chain = line[20:22].strip()
                        x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                        if chain not in self.chains_data: self.chains_data[chain] = []
                        self.chains_data[chain].append((x,y,z))
                        all_coords.append((x,y,z))
            
            if all_coords:
                xs, ys, zs = zip(*all_coords)
                self.center = (np.mean(xs), np.mean(ys), np.mean(zs))
                self.max_range = max(np.max(xs)-np.min(xs), np.max(ys)-np.min(ys), np.max(zs)-np.min(zs)) / 2.0
            
            self.draw_chains()
            return list(self.chains_data.keys())
        except Exception as e:
            print(f"Error parsing PDB: {e}")
            return []

    def update_selection(self, group1_list, group2_list, group3_list):
        self.chain1 = group1_list if group1_list else []
        self.chain2 = group2_list if group2_list else []
        self.chain3 = group3_list if group3_list else []
        self.draw_chains()

    def draw_chains(self):
        self.axes.clear()
        self.axes.axis('off')
        self.chain_artists = {} # Reset artists
        
        if not self.chains_data:
            self.draw()
            return

        for chain, coords in self.chains_data.items():
            if not coords: continue
            
            # Group 3 Logic: Skip drawing if in Group 3
            if chain in self.chain3:
                continue

            xs, ys, zs = zip(*coords)
            
            if chain in self.chain1:
                color = "#FF4444" # Red (Group 1)
                lw = 3.0
                alpha = 1.0
                zorder = 10
            elif chain in self.chain2:
                color = "#4488FF" # Blue (Group 2)
                lw = 3.0
                alpha = 1.0
                zorder = 10
            else:
                color = "#888888" # Grey (Stock/Unused)
                lw = 1.0
                alpha = 0.6 
                zorder = 1
            
            # Enable picking with a tolerance of 5 points
            lines = self.axes.plot(xs, ys, zs, c=color, linewidth=lw, alpha=alpha, zorder=zorder, picker=5.0)
            line = lines[0]
            line.set_gid(chain)
            
            # Store artist and its base width for hover restoration
            self.chain_artists[chain] = {'artist': line, 'base_lw': lw}
            
            cx, cy, cz = np.mean(xs), np.mean(ys), np.mean(zs)
            self.axes.text(cx, cy, cz, chain, color='white', fontsize=10, weight='bold', zorder=zorder+1)

        self.apply_zoom()

    def on_scroll(self, event):
        # Zoom only on Scroll
        step = 5 if event.button == 'up' else -5
        self.zoom = max(1, min(100, self.zoom + step))
        self.apply_zoom()

    def on_press(self, event):
        # Trigger on Right Click (Button 3 in MPL)
        if event.button == 3: 
            self.dragging = True
            self.last_mouse = (event.x, event.y)

    def on_release(self, event):
        if event.button == 3:
            self.dragging = False

    def on_move(self, event):
        # 1. Handle Rotation
        if self.dragging and event.x is not None and event.y is not None:
            dx = event.x - self.last_mouse[0]
            dy = event.y - self.last_mouse[1]
            self.axes.azim -= dx * 0.3
            self.axes.elev += dy * 0.3
            self.last_mouse = (event.x, event.y)
            self.draw()
            return

        # 2. Handle Hover (Only if enabled and not dragging)
        if self.hover_enabled and event.x is not None and event.y is not None:
            found_chain = None
            
            # Check intersection with all chain lines
            for chain, data in self.chain_artists.items():
                line = data['artist']
                contains, _ = line.contains(event)
                if contains:
                    found_chain = chain
                    break
            
            # State Change Detection
            if found_chain != self.hovered_chain:
                # Restore old
                if self.hovered_chain and self.hovered_chain in self.chain_artists:
                    old_data = self.chain_artists[self.hovered_chain]
                    old_data['artist'].set_linewidth(old_data['base_lw'])
                
                # Highlight new
                if found_chain and found_chain in self.chain_artists:
                    new_data = self.chain_artists[found_chain]
                    # Thicken the line (base + 3)
                    new_data['artist'].set_linewidth(new_data['base_lw'] + 3.0)
                
                self.hovered_chain = found_chain
                self.draw() # Redraw to show linewidth change

    def apply_zoom(self):
        if self.max_range == 0: return
        factor = 2.0 - ((self.zoom / 100.0) * 1.9)
        if factor < 0.05: factor = 0.05
        r = self.max_range * factor
        gx, gy, gz = self.center
        self.axes.set_xlim(gx-r, gx+r)
        self.axes.set_ylim(gy-r, gy+r)
        self.axes.set_zlim(gz-r, gz+r)
        self.draw()

    def on_pick(self, event):
        # Fix: Filter out Scroll (Zoom) and Right Click (Rotate)
        # Only accept strictly Left Click (Button 1)
        if not hasattr(event, 'mouseevent') or event.mouseevent.button != 1:
            return

        # Handle 3D click on a chain line
        if event.artist and event.artist.get_gid():
            self.chainClicked.emit(event.artist.get_gid())

# --- MAIN APP ---
class FoldXApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoldX Interface Energy Calculator")
        self.resize(1200, 750)
        
        # Dynamically locate the FoldX folder inside xcop_savings
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.source_dir = os.path.join(current_dir, "FoldX")
        
        self.pdb_path = None
        
        # State
        self.stock_chains = []
        self.chain1 = None
        self.chain2 = None
        self.temp_pdb_path = None
        
        # UI State for Click-to-Assign
        self.active_category_key = None # 'stock', 'g1', 'g2', 'g3' or None
        
        self.initUI()
        self.apply_dark_theme()
        self.toggle_mode()
        self.seq_detail_data = {}

    def swap_groups(self, key1, key2):
        # 1. Get the ListWidgets
        list1 = self.category_map[key1]['list']
        list2 = self.category_map[key2]['list']

        # 2. Extract Items
        items1 = self.get_list_items(list1)
        items2 = self.get_list_items(list2)

        # 3. Swap Content
        list1.clear()
        list2.clear()
        list1.addItems(items2)
        list2.addItems(items1)

        # 4. Refresh State
        self.refresh_ui_state()

    def on_category_label_clicked(self, key):
        # Toggle Logic
        if self.active_category_key == key:
            # Deactivate
            self.active_category_key = None
            self.category_map[key]['label'].set_active(False)
            self.viewer.hover_enabled = False # Disable hover
        else:
            # Deactivate old if exists
            if self.active_category_key:
                self.category_map[self.active_category_key]['label'].set_active(False)
            
            # Activate new
            self.active_category_key = key
            self.category_map[key]['label'].set_active(True)
            self.viewer.hover_enabled = True # Enable hover

    def on_3d_chain_clicked(self, chain_id):
        # Only proceed if a category is active
        if not self.active_category_key:
            return

        active_key = self.active_category_key
        target_list_widget = self.category_map[active_key]['list']
        stock_list_widget = self.category_map['stock']['list']
        
        # 1. Find where the chain currently is
        source_list_widget = None
        item_to_move = None
        
        for key, data in self.category_map.items():
            lw = data['list']
            items = lw.findItems(chain_id, Qt.MatchFlag.MatchExactly)
            if items:
                source_list_widget = lw
                item_to_move = items[0]
                break
        
        if not source_list_widget: return

        # 2. Determine Destination
        destination_widget = target_list_widget
        
        # LOGIC: If the active group is NOT stock, and the chain is ALREADY in the active group,
        # we treat this click as a "Deselect" -> Move back to Stock.
        if active_key != 'stock' and source_list_widget == target_list_widget:
            destination_widget = stock_list_widget

        # 3. Move it (if destination is different from source)
        if source_list_widget != destination_widget:
            row = source_list_widget.row(item_to_move)
            source_list_widget.takeItem(row)
            destination_widget.addItem(chain_id)
            
            # 4. Trigger updates
            self.refresh_ui_state()

    def keyPressEvent(self, event):
        """Jump to chain matching the pressed key (Case Sensitive)."""
        text = event.text()
        if not text:
            super().keyPressEvent(event)
            return

        # Search Stock, Group 1, Group 2, and Group 3
        for lw in [self.list_stock, self.list_g1, self.list_g2, self.list_g3]:
            # Find items matching text exactly and case-sensitively
            matches = lw.findItems(text, Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive)
            if matches:
                item = matches[0]
                lw.scrollToItem(item)      # Ensure visible
                lw.setCurrentItem(item)    # Highlight/Select
                lw.setFocus()              # Move focus to that list
                return                     # Stop searching (chains are unique)

        super().keyPressEvent(event)

    def eventFilter(self, source, event):
        """Intercept key presses to perform global, case-sensitive navigation."""
        if event.type() == QEvent.Type.KeyPress:
            text = event.text()
            
            # Check if key is a valid single alphanumeric character
            if text and len(text) == 1 and text.isalnum():
                
                # Check all lists: Stock, Group 1, Group 2, Group 3
                for lw in [self.list_stock, self.list_g1, self.list_g2, self.list_g3]:
                    # Find EXACT match with CASE SENSITIVITY
                    matches = lw.findItems(text, Qt.MatchFlag.MatchExactly | Qt.MatchFlag.MatchCaseSensitive)
                    
                    if matches:
                        item = matches[0]
                        
                        # 1. Clear selection in others to avoid confusion
                        for other_lw in [self.list_stock, self.list_g1, self.list_g2, self.list_g3]:
                            if other_lw != lw:
                                other_lw.clearSelection()
                        
                        # 2. Focus and Select the found item
                        lw.setFocus()
                        lw.scrollToItem(item)
                        lw.setCurrentItem(item)
                        return True # Stop event propagation (we handled it)

        return super().eventFilter(source, event)

    def closeEvent(self, event):
        """Handle application closure: Clean up temp files."""
        self.cleanup_temp_file()
        event.accept()

    def cleanup_temp_file(self):
        """Deletes the temporary workspace folder and all its contents."""
        if hasattr(self, 'temp_workspace') and self.temp_workspace and os.path.exists(self.temp_workspace):
            try:
                shutil.rmtree(self.temp_workspace)
                print(f"Cleaned up workspace: {self.temp_workspace}")
            except Exception as e:
                print(f"Failed to delete workspace: {e}")
        self.temp_pdb_path = None
        self.temp_workspace = None

    def toggle_mode(self):
        if self.radio_seq_detail.isChecked():
            self.btn_browse.setText("Browse PDB / Results")
            self.lbl_g1.setText("Active")
            self.v3_widget.hide()
            self.v4_widget.hide()
            self.lbl_result.hide()
            self.lbl_sd_result.show()
            self.sd_options_widget.show()
            self.btn_export.show()
            self.move_all_to_stock_except_active()
        else:
            self.btn_browse.setText("Browse PDB")
            self.lbl_g1.setText("Group 1")
            self.v3_widget.show()
            self.v4_widget.show()
            self.lbl_result.show()
            self.lbl_sd_result.hide()
            self.sd_options_widget.hide()
            self.btn_export.hide()
        self.update_sd_output()

    def move_all_to_stock_except_active(self):
        items_to_move = []
        for lw in [self.list_g2, self.list_g3]:
            for i in range(lw.count()):
                items_to_move.append(lw.item(i).text())
            lw.clear()
        if items_to_move:
            self.list_stock.addItems(items_to_move)
        self.refresh_ui_state()

    def auto_select_middle_layers(self):
        try:
            target_layers = int(self.layer_input.text())
        except ValueError:
            return
            
        if not hasattr(self, 'all_chains') or not self.all_chains:
            return

        chains = self.all_chains
        
        try: Z_MIN = float(self.edit_z_min.text())
        except ValueError: Z_MIN = 4.6
        try: Z_MAX = float(self.edit_z_max.text())
        except ValueError: Z_MAX = 5.0
        try: XY_SHIFT_LIMIT = float(self.edit_xy_limit.text())
        except ValueError: XY_SHIFT_LIMIT = 3.0
        
        cents = {}
        for c in chains:
            if c in self.viewer.chains_data and self.viewer.chains_data[c]:
                cents[c] = np.mean(self.viewer.chains_data[c], axis=0)
        
        neighbors = {c: {'up': None, 'down': None} for c in cents}
        for c1 in cents:
            min_up, min_down = float('inf'), float('inf')
            for c2 in cents:
                if c1 == c2: continue
                dz = cents[c2][2] - cents[c1][2]
                dxy = np.hypot(cents[c2][0]-cents[c1][0], cents[c2][1]-cents[c1][1])
                if dxy > XY_SHIFT_LIMIT: continue
                if Z_MIN <= dz <= Z_MAX:
                    if dz < min_up: min_up = dz; neighbors[c1]['up'] = c2
                elif Z_MIN <= -dz <= Z_MAX:
                    if -dz < min_down: min_down = -dz; neighbors[c1]['down'] = c2
        
        protofilaments = []
        visited = set()
        for c in cents:
            if c in visited: continue
            curr = c
            while neighbors[curr]['down'] and neighbors[curr]['down'] not in visited:
                curr = neighbors[curr]['down']
            pf = []
            while curr:
                pf.append(curr)
                visited.add(curr)
                curr = neighbors[curr]['up']
            protofilaments.append(pf)
            
        if not protofilaments: return
        
        selected_chains = set()
        for pf in protofilaments:
            total = len(pf)
            if target_layers >= total:
                selected_chains.update(pf)
            else:
                start_idx = (total - target_layers) // 2
                end_idx = start_idx + target_layers
                selected_chains.update(pf[start_idx:end_idx])
                
        self.list_g1.clear()
        self.list_g2.clear()
        self.list_g3.clear()
        
        valid_selected = [c for c in self.all_chains if c in selected_chains]
        self.list_g1.addItems(valid_selected)
        
        self.refresh_ui_state()

    def apply_dark_theme(self):
        self.setStyleSheet("""
            QWidget { background-color: #2b2b2b; color: #FFFFFF; font-family: Arial; font-size: 10pt; }
            QLineEdit, QTextEdit, QListWidget { 
                background-color: #1e1e1e; border: 1px solid #444; color: white; padding: 4px; border-radius: 2px; 
                selection-background-color: #98c379; selection-color: #1e1e1e;
            }
            
            QRadioButton::indicator { width: 12px; height: 12px; border: 1px solid #555; border-radius: 7px; background-color: #1e1e1e; }
            QRadioButton::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
            
            QPushButton { background-color: #3e3e3e; border: 1px solid #3e3e3e; border-radius: 5px; padding: 8px; color: white; outline: none; } 
            QPushButton:hover { background-color: #3e3e3e; border: 1px solid #98c379; }
            QPushButton:pressed { background-color: #2b2b2b; border: 1px solid #98c379; }
            QPushButton:disabled { background-color: #555; border: 1px solid #555; color: #888; }
            
            QGroupBox { border: 1px solid #555; margin-top: 10px; font-weight: bold; color: #98c379; }
            QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
            
            QSplitter::handle { background-color: #444; }
            QListWidget::item { padding: 5px; font-size: 14px; }
            QListWidget::item:hover { background-color: #444; }
            QListWidget::item:selected { background-color: #3e3e3e; border: 1px solid #98c379; color: white; }
            
            QMenu { background-color: #1e1e1e; color: white; border: 1px solid #555; }
            QMenu::item:selected { background-color: #3e3e3e; color: #98c379; }
            
            QLabel#ResultLabel { font-size: 24px; font-weight: bold; color: #98c379; border: 2px solid #98c379; border-radius: 5px; padding: 10px; background-color: #1e1e1e; }
            
            /* Apple-style Scrollbar */
            QScrollBar:vertical { border: none; background: #2C2C2C; width: 10px; margin: 0; border-radius: 0; }
            QScrollBar::handle:vertical { background: #555; min-height: 20px; border-radius: 5px; }
            QScrollBar::handle:vertical:hover { background: #777; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { border: none; background: none; height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }

            QScrollBar:horizontal { border: none; background: #2C2C2C; height: 10px; margin: 0; border-radius: 0; }
            QScrollBar::handle:horizontal { background: #555; min-width: 20px; border-radius: 5px; }
            QScrollBar::handle:horizontal:hover { background: #777; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { border: none; background: none; width: 0; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
        """)

    def initUI(self):
        main_layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # === LEFT PANEL ===
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        # 1. Header & Load
        hb_load = QHBoxLayout()
        self.path_entry = QLineEdit()
        self.path_entry.setReadOnly(True)
        self.path_entry.setPlaceholderText("No file loaded")
        self.btn_browse = QPushButton("Browse PDB")
        self.btn_browse.clicked.connect(self.browse_pdb)
        hb_load.addWidget(self.path_entry)
        hb_load.addWidget(self.btn_browse)
        left_layout.addLayout(hb_load)

        # 1.5 Mode Selection
        hb_mode = QHBoxLayout()
        self.radio_seq_detail = QRadioButton("Sequence Detail")
        self.radio_analyse_complex = QRadioButton("Analyse Complex")
        self.radio_seq_detail.setChecked(True)
        hb_mode.addWidget(self.radio_seq_detail)
        hb_mode.addWidget(self.radio_analyse_complex)
        left_layout.addLayout(hb_mode)

        self.radio_seq_detail.toggled.connect(self.toggle_mode)
        self.radio_analyse_complex.toggled.connect(self.toggle_mode)

        # 1.8 Operations Selection
        from PyQt6.QtWidgets import QCheckBox
        self.grp_ops = QGroupBox("Operations")
        
        ops_layout = QHBoxLayout(self.grp_ops)
        ops_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.chk_op1 = QCheckBox("RepairPDB")
        self.chk_op2 = QCheckBox("Optimize")
        
        self.btn_swap_ops = QPushButton("Swap Order ⇄")
        self.btn_swap_ops.setStyleSheet("QPushButton { color: #98c379; background-color: #3e3e3e; border: 1px solid #555; padding: 4px 10px; border-radius: 4px; } QPushButton:hover { border: 1px solid #98c379; }")
        self.btn_swap_ops.clicked.connect(self.swap_operations)
        
        ops_layout.addWidget(QLabel("➔"))
        ops_layout.addWidget(self.chk_op1)
        ops_layout.addWidget(QLabel("➔"))
        ops_layout.addWidget(self.chk_op2)
        ops_layout.addSpacing(15)
        ops_layout.addWidget(self.btn_swap_ops)
        ops_layout.addStretch() # Push everything to the left

        left_layout.addWidget(self.grp_ops)

        # 2. Chain Selection (Drag & Drop)
        grp_select = QGroupBox("Chain Selection (Drag to Move)")
        layout_select = QHBoxLayout()
        layout_select.setSpacing(10)

        # -- Dictionary to map keys to lists and labels --
        self.category_map = {} 

        # -- Col 1: Stock --
        v1 = QVBoxLayout()
        self.lbl_stock = ClickableLabel("Stock", "stock")
        self.lbl_stock.clicked.connect(self.on_category_label_clicked)
        self.lbl_stock.swapped.connect(self.swap_groups)
        v1.addWidget(self.lbl_stock)
        self.list_stock = DraggableListWidget(is_group_list=False)
        v1.addWidget(self.list_stock)
        layout_select.addLayout(v1)
        self.category_map['stock'] = {'label': self.lbl_stock, 'list': self.list_stock}

        # -- Col 2: Group 1 --
        v2 = QVBoxLayout()
        self.lbl_g1 = ClickableLabel("Group 1", "g1")
        self.lbl_g1.clicked.connect(self.on_category_label_clicked)
        self.lbl_g1.swapped.connect(self.swap_groups)
        v2.addWidget(self.lbl_g1)
        self.list_g1 = DraggableListWidget(is_group_list=True)
        v2.addWidget(self.list_g1)
        layout_select.addLayout(v2)
        self.category_map['g1'] = {'label': self.lbl_g1, 'list': self.list_g1}

        # -- Col 3: Group 2 --
        v3 = QVBoxLayout()
        v3.setContentsMargins(0, 0, 0, 0)
        self.lbl_g2 = ClickableLabel("Group 2", "g2")
        self.lbl_g2.clicked.connect(self.on_category_label_clicked)
        self.lbl_g2.swapped.connect(self.swap_groups)
        v3.addWidget(self.lbl_g2)
        self.list_g2 = DraggableListWidget(is_group_list=True)
        v3.addWidget(self.list_g2)
        self.v3_widget = QWidget()
        self.v3_widget.setLayout(v3)
        layout_select.addWidget(self.v3_widget)
        self.category_map['g2'] = {'label': self.lbl_g2, 'list': self.list_g2}

        # -- Col 4: Group 3 (Hidden) --
        v4 = QVBoxLayout()
        v4.setContentsMargins(0, 0, 0, 0)
        self.lbl_g3 = ClickableLabel("Group 3 (Hidden)", "g3")
        self.lbl_g3.clicked.connect(self.on_category_label_clicked)
        self.lbl_g3.swapped.connect(self.swap_groups)
        v4.addWidget(self.lbl_g3)
        self.list_g3 = DraggableListWidget(is_group_list=True)
        v4.addWidget(self.list_g3)
        self.v4_widget = QWidget()
        self.v4_widget.setLayout(v4)
        layout_select.addWidget(self.v4_widget)
        self.category_map['g3'] = {'label': self.lbl_g3, 'list': self.list_g3}

        # Connect signals (Delayed to allow drag-drop to finish)
        self.list_stock.contentChanged.connect(lambda: QTimer.singleShot(10, self.refresh_ui_state))
        self.list_g1.contentChanged.connect(lambda: QTimer.singleShot(10, self.refresh_ui_state))
        self.list_g2.contentChanged.connect(lambda: QTimer.singleShot(10, self.refresh_ui_state))
        self.list_g3.contentChanged.connect(lambda: QTimer.singleShot(10, self.refresh_ui_state))

        grp_select.setLayout(layout_select)
        left_layout.addWidget(grp_select)

        # 2.5 Sequence Detail Options (Auto Select + Parameters)
        self.sd_options_widget = QWidget()
        sd_options_layout = QVBoxLayout(self.sd_options_widget)
        sd_options_layout.setContentsMargins(0, 0, 0, 0)
        sd_options_layout.setSpacing(5)

        # Auto Select Row
        sd_layer_layout = QHBoxLayout()
        sd_layer_layout.addWidget(QLabel("Auto-select middle layers:"))
        self.layer_input = QLineEdit()
        self.layer_input.setPlaceholderText("#")
        self.layer_input.setFixedWidth(40)
        sd_layer_layout.addWidget(self.layer_input)
        
        self.btn_apply_layers = QPushButton("Apply")
        self.btn_apply_layers.setFixedWidth(60)
        self.btn_apply_layers.clicked.connect(self.auto_select_middle_layers)
        sd_layer_layout.addWidget(self.btn_apply_layers)
        sd_layer_layout.addStretch()
        sd_options_layout.addLayout(sd_layer_layout)

        # Detection Parameters Group
        from PyQt6.QtGui import QDoubleValidator
        self.grp_params = QGroupBox("Detection Parameters")
        params_form = QFormLayout(self.grp_params)
        
        self.edit_z_min = QLineEdit("4.6")
        self.edit_z_min.setValidator(QDoubleValidator())
        params_form.addRow("Z Shift Min (Å):", self.edit_z_min)
        
        self.edit_z_max = QLineEdit("5.0")
        self.edit_z_max.setValidator(QDoubleValidator())
        params_form.addRow("Z Shift Max (Å):", self.edit_z_max)
        
        self.edit_xy_limit = QLineEdit("3.0")
        self.edit_xy_limit.setValidator(QDoubleValidator())
        params_form.addRow("XY Shift Limit (Å):", self.edit_xy_limit)
        
        sd_options_layout.addWidget(self.grp_params)
        left_layout.addWidget(self.sd_options_widget)

        # 3. Action & Result
        self.btn_run = QPushButton("CALCULATE INTERFACE ENERGY")
        self.btn_run.setFixedHeight(50)
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_run.setEnabled(False)
        left_layout.addWidget(self.btn_run)

        self.lbl_result = QLabel("Not Calculated")
        self.lbl_result.setObjectName("ResultLabel")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.lbl_result)

        self.lbl_sd_result = QLabel("")
        self.lbl_sd_result.setStyleSheet("font-size: 12pt; color: #98c379; padding: 5px;")
        self.lbl_sd_result.hide()
        left_layout.addWidget(self.lbl_sd_result)

        self.btn_export = QPushButton("Export Raw Data")
        self.btn_export.setFixedHeight(40)
        self.btn_export.clicked.connect(self.export_raw_data)
        self.btn_export.setEnabled(False)
        left_layout.addWidget(self.btn_export)

        # 4. Log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFontFamily("Courier New")
        left_layout.addWidget(self.log_text)

        # === RIGHT PANEL ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        self.viewer = PDBViewerCanvas()
        self.viewer.chainClicked.connect(self.on_3d_chain_clicked)
        right_layout.addWidget(self.viewer)

        # === SPLITTER SETUP ===
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        # Reduced left panel width (approx half of previous size)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        main_layout.addWidget(splitter)

        # Install Global Event Filter to catch keys regardless of focus
        self.list_stock.installEventFilter(self)
        self.list_g1.installEventFilter(self)
        self.list_g2.installEventFilter(self)
        self.list_g3.installEventFilter(self)

    # --- HELPER FUNCTIONS ---
    def swap_operations(self):
        text1 = self.chk_op1.text()
        state1 = self.chk_op1.isChecked()
        
        text2 = self.chk_op2.text()
        state2 = self.chk_op2.isChecked()
        
        self.chk_op1.setText(text2)
        self.chk_op1.setChecked(state2)
        
        self.chk_op2.setText(text1)
        self.chk_op2.setChecked(state1)

    def refresh_ui_state(self):
        g1 = self.get_list_items(self.list_g1)
        g2 = self.get_list_items(self.list_g2)
        g3 = self.get_list_items(self.list_g3)
        self.update_sd_output()
        
        # Calculate what should be in Stock (Master List - Selected)
        # This automatically removes duplicates from Stock
        used_chains = set(g1 + g2 + g3)
        if hasattr(self, 'all_chains'):
            stock_items = [c for c in self.all_chains if c not in used_chains]
        else:
            stock_items = [] # Fallback if pdb not loaded yet

        # Update Stock List
        self.list_stock.blockSignals(True)
        self.list_stock.clear()
        self.list_stock.addItems(stock_items)
        self.list_stock.blockSignals(False)

        # Update 3D Viewer Highlighting (Pass g3 so it can hide them)
        self.viewer.update_selection(g1, g2, g3)

    # --- LOGIC ---
    def browse_pdb(self):
        options = QFileDialog.Option.DontUseNativeDialog
        if self.radio_seq_detail.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Open File", os.getcwd(), "PDB or Result Files (*.pdb *.csv *.fxout *.txt)", options=options)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Open PDB", os.getcwd(), "PDB Files (*.pdb)", options=options)
            
        if path:
            if path.lower().endswith(('.csv', '.fxout', '.txt')):
                self.load_result_file(path)
            else:
                self.load_pdb(path)

    def load_result_file(self, path):
        self.path_entry.setText(os.path.basename(path))
        self.log_text.setText(f"Loaded Result File: {path}\n")
        result_data = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    # Support both standard tab-separated .fxout and exported .csv files
                    parts = line.split('\t') if '\t' in line else line.split(',')
                    parts = [p.strip(' "') for p in parts]
                    
                    if len(parts) >= 9:
                        chain_id = parts[2]
                        try:
                            energy = float(parts[8])
                            if chain_id not in result_data:
                                result_data[chain_id] = {'energy': 0.0, 'residues': 0}
                            result_data[chain_id]['energy'] += energy
                            result_data[chain_id]['residues'] += 1
                        except ValueError:
                            pass
                            
            self.seq_detail_data = result_data
            
            # Clear UI and populate 'Active' list with found chains
            self.list_g1.clear()
            self.list_g2.clear()
            self.list_g3.clear()
            self.list_stock.clear()
            
            chains = sorted(list(result_data.keys()))
            self.all_chains = chains.copy()  # Fix: Keep a master record so chains don't vanish on drop
            self.list_g1.addItems(chains)
            
            self.log_text.append(f"Successfully parsed {len(chains)} chains from result file.")
            self.update_sd_output()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse result file:\n{e}")

    def load_pdb(self, path):
        # If loading a NEW file (not the current temp one), clean up the old temp file
        if self.temp_pdb_path and path != self.temp_pdb_path:
            self.cleanup_temp_file()

        # --- Pre-scan for Chain ID analysis ---
        unique_chains = set()
        has_two_char = False
        
        try:
            with open(path, 'r') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        if len(line) > 22:
                            # Robust extraction (Cols 21-22 in PDB standard/hybrid)
                            c_id = line[20:22].strip()
                            if c_id:
                                unique_chains.add(c_id)
                                if len(c_id) > 1:
                                    has_two_char = True
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read file: {e}")
            return

        sorted_chains = sorted(list(unique_chains))

        is_seq_detail = self.radio_seq_detail.isChecked()

        # 0. Limit Check
        if not is_seq_detail and len(sorted_chains) > 62:
            QMessageBox.warning(self, "Too Many Chains", 
                f"Detected {len(sorted_chains)} chains.\n"
                "FoldX supports a maximum of 62 chains (A-Z, a-z, 0-9).\n"
                "Please trim the PDB file before proceeding.")
            return

        # Create a unique workspace folder for this specific session
        dirname = os.path.dirname(path)
        filename = os.path.basename(path)
        name_part, ext_part = os.path.splitext(filename)
        unique_id = uuid.uuid4().hex[:8]
        workspace_dir = os.path.join(dirname, f"{name_part}_{unique_id}")

        try:
            if not os.path.exists(workspace_dir):
                os.makedirs(workspace_dir)
        except Exception as e:
            QMessageBox.critical(self, "Folder Error", f"Could not create workspace:\n{e}")
            return

        # 1. & 2. Two-char Detection & Prompt
        renamed_success = False
        if not is_seq_detail and has_two_char:
            reply = QMessageBox.question(self, "Incompatible Chain IDs",
                "Detected chain IDs with 2 characters.\n"
                "FoldX AnalyseComplex requires single-character Chain IDs.\n\n"
                "A renamed copy will be created in the workspace folder.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.Yes)
            
            if reply == QMessageBox.StandardButton.Yes:
                # 3. Rename Logic (Pool: 62 chars)
                pool = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
                mapping = {}
                for i, old_id in enumerate(sorted_chains):
                    mapping[old_id] = pool[i]

                # Create Temp File inside the workspace folder
                new_path = os.path.join(workspace_dir, f"{name_part}_renamed{ext_part}")

                try:
                    with open(path, 'r') as fin, open(new_path, 'w') as fout:
                        for line in fin:
                            if line.startswith("ATOM") or line.startswith("HETATM"):
                                curr_id = line[20:22].strip()
                                if curr_id in mapping:
                                    new_id = mapping[curr_id]
                                    # Replace Chain ID at col 22 (index 21), clear col 21 (index 20)
                                    l_chars = list(line)
                                    if len(l_chars) > 21:
                                        l_chars[21] = new_id
                                        l_chars[20] = ' '
                                    fout.write("".join(l_chars))
                                else:
                                    fout.write(line)
                            else:
                                fout.write(line)
                    
                    # 4. Update Path (No Recursion)
                    self.temp_workspace = workspace_dir 
                    self.temp_pdb_path = new_path
                    path = new_path # Continue using the renamed file
                    renamed_success = True
                    
                except Exception as e:
                    QMessageBox.critical(self, "Rename Error", f"Could not create temp file:\n{e}")
                    return

        # If no renaming was performed (either not needed or declined), copy the original PDB
        if not renamed_success:
            self.temp_workspace = workspace_dir
            self.temp_pdb_path = os.path.join(workspace_dir, filename)
            shutil.copy2(path, self.temp_pdb_path)
            path = self.temp_pdb_path 

        self.pdb_path = path
        self.path_entry.setText(os.path.basename(path))
        
        # Reset UI
        self.list_stock.clear()
        self.list_g1.clear()
        self.list_g2.clear()
        self.list_g3.clear()
        self.lbl_result.setText("Not Calculated")
        self.btn_run.setEnabled(True)
        self.log_text.setText(f"Loaded: {path}")
        
        # Load Chains
        raw_chains = self.viewer.load_pdb(path)
        
        # Sorting: Numerical then Alphabetical
        def chain_sort_key(s):
            if s.isdigit(): return (0, int(s))
            return (1, s)
            
        self.all_chains = sorted(raw_chains, key=chain_sort_key) # Save Master List
        self.list_stock.addItems(self.all_chains)
        self.viewer.update_selection(None, None, None)

    def get_list_items(self, list_widget):
        items = []
        for index in range(list_widget.count()):
            items.append(list_widget.item(index).text())
        return items

    def run_analysis(self):
        g1 = self.get_list_items(self.list_g1)
        g2 = self.get_list_items(self.list_g2)

        is_seq_detail = self.radio_seq_detail.isChecked()

        if not is_seq_detail and (not g1 or not g2):
            QMessageBox.warning(self, "Selection Error", "Please drag chains into both Group 1 and Group 2.")
            return
        if is_seq_detail and not g1:
            QMessageBox.warning(self, "Selection Error", "Please drag chains into Active.")
            return

        # Extract selected operations in their current left-to-right order
        operations = []
        if self.chk_op1.isChecked():
            operations.append(self.chk_op1.text())
        if self.chk_op2.isChecked():
            operations.append(self.chk_op2.text())

        self.btn_run.setEnabled(False)
        self.log_text.clear()
        
        if is_seq_detail:
            self.lbl_sd_result.setText("Calculating Sequence Detail...")
        else:
            self.lbl_result.setText("Calculating...")
        
        self.thread = threading.Thread(target=self._worker_wrapper, args=(g1, g2, is_seq_detail, operations))
        self.thread.start()

    def _worker_wrapper(self, g1, g2, is_seq_detail, operations):
        self.raw_foldx_data = None
        self.btn_export.setEnabled(False)
        worker = FoldXWorker(self.pdb_path, g1, g2, self.source_dir, is_seq_detail, operations)
        worker.signals.log.connect(self.append_log)
        worker.signals.result.connect(self.show_result)
        worker.signals.raw_data.connect(self.store_raw_data)
        worker.signals.finished.connect(self.on_finished)
        worker.signals.error.connect(self.on_error)
        worker.run()

    def append_log(self, text):
        self.log_text.append(text)

    def show_result(self, text):
        if text.startswith("SD_RESULT:"):
            import json
            self.seq_detail_data = json.loads(text[10:])
            self.update_sd_output()
        else:
            self.lbl_result.setText(text)

    def get_number_of_layers(self, chains):
        if not chains: return 0
        
        try: Z_MIN = float(self.edit_z_min.text())
        except ValueError: Z_MIN = 4.6
        try: Z_MAX = float(self.edit_z_max.text())
        except ValueError: Z_MAX = 5.0
        try: XY_SHIFT_LIMIT = float(self.edit_xy_limit.text())
        except ValueError: XY_SHIFT_LIMIT = 3.0
        
        cents = {}
        for c in chains:
            if c in self.viewer.chains_data and self.viewer.chains_data[c]:
                cents[c] = np.mean(self.viewer.chains_data[c], axis=0)
        
        neighbors = {c: {'up': None, 'down': None} for c in cents}
        for c1 in cents:
            min_up, min_down = float('inf'), float('inf')
            for c2 in cents:
                if c1 == c2: continue
                dz = cents[c2][2] - cents[c1][2]
                dxy = np.hypot(cents[c2][0]-cents[c1][0], cents[c2][1]-cents[c1][1])
                if dxy > XY_SHIFT_LIMIT: continue
                if Z_MIN <= dz <= Z_MAX:
                    if dz < min_up: min_up = dz; neighbors[c1]['up'] = c2
                elif Z_MIN <= -dz <= Z_MAX:
                    if -dz < min_down: min_down = -dz; neighbors[c1]['down'] = c2
        
        # build protofilaments
        protofilaments = []
        visited = set()
        for c in cents:
            if c in visited: continue
            curr = c
            while neighbors[curr]['down'] and neighbors[curr]['down'] not in visited:
                curr = neighbors[curr]['down']
            pf = []
            while curr:
                pf.append(curr)
                visited.add(curr)
                curr = neighbors[curr]['up']
            protofilaments.append(pf)
            
        if not protofilaments: return 0
        return max(len(pf) for pf in protofilaments)

    def update_sd_output(self):
        if not self.radio_seq_detail.isChecked(): return
        
        g1 = self.get_list_items(self.list_g1)
        if not g1 or not self.seq_detail_data:
            self.lbl_sd_result.setText("Total Energy: N/A\nEnergy per Layer: N/A\nEnergy per Molecule: N/A\nEnergy per Residue: N/A")
            return
        
        total_energy = 0.0
        total_residues = 0
        for chain in g1:
            if chain in self.seq_detail_data:
                total_energy += self.seq_detail_data[chain]['energy']
                total_residues += self.seq_detail_data[chain]['residues']
                
        layers = self.get_number_of_layers(g1)
        num_molecules = len(g1)
        
        layer_text = f"{layers} layers" if layers > 0 else "Unknown layers"
        
        # 1. Total Energy
        str_total = f"Total Energy: {total_energy:.4f} kcal/mol ({layer_text})"
        
        # 2. Energy per Layer & Residue Context
        if layers > 0:
            str_layer = f"Energy per Layer: {total_energy / layers:.4f} kcal/mol ({layer_text})"
            
            # Calculate residues per layer, formatting to a clean integer if perfectly divisible
            res_per_layer = total_residues / layers
            res_per_layer_str = f"{int(res_per_layer)}" if res_per_layer.is_integer() else f"{res_per_layer:.1f}"
            res_info = f"({total_residues} residues in total, {res_per_layer_str} residues per layer)"
        else:
            str_layer = "Energy per Layer: N/A (Requires PDB)"
            res_info = f"({total_residues} residues in total)"
            
        # 3. Energy per Molecule
        str_mol = f"Energy per Molecule: {total_energy / num_molecules:.4f} kcal/mol ({num_molecules} molecules)"
        
        # 4. Energy per Residue
        if total_residues > 0:
            str_res = f"Energy per Residue: {total_energy / total_residues:.4f} kcal/mol {res_info}"
        else:
            str_res = "Energy per Residue: N/A"
            
        # Set Final Text
        self.lbl_sd_result.setText(f"{str_total}\n{str_layer}\n{str_mol}\n{str_res}")

    def store_raw_data(self, data):
        self.raw_foldx_data = data
        self.btn_export.setEnabled(True)

    def export_raw_data(self):
        if not hasattr(self, 'raw_foldx_data') or not self.raw_foldx_data:
            QMessageBox.warning(self, "No Data", "No raw data available to export.")
            return
            
        path, _ = QFileDialog.getSaveFileName(self, "Save Raw Data", "foldx_raw.csv", "CSV Files (*.csv)", options=QFileDialog.Option.DontUseNativeDialog)
        if not path:
            return
            
        try:
            import csv
            with open(path, 'w', newline='') as f:
                writer = csv.writer(f)
                for line in self.raw_foldx_data.splitlines():
                    if line.strip(): # Skip empty lines
                        # .fxout files are typically tab-separated
                        writer.writerow(line.split('\t'))
            QMessageBox.information(self, "Success", f"Raw data exported to:\n{path}")
            self.log_text.append(f"\n>>> Exported raw data to {path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export data:\n{e}")

    def on_finished(self):
        self.btn_run.setEnabled(True)

    def on_error(self, err):
        self.btn_run.setEnabled(True)
        self.lbl_result.setText("Error")
        QMessageBox.critical(self, "Error", f"Failed:\n{err}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FoldXApp()
    window.show()
    sys.exit(app.exec())
