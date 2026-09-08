import sys
import os
import shutil
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import subprocess
import re
import platform
import argparse
import time
import datetime
import json

from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QProgressBar,
                             QPushButton, QLabel, QLineEdit, QComboBox, 
                             QTextEdit, QFileDialog, QMessageBox, QDialog,
                             QTabWidget, QListWidget, QListWidgetItem, QMenu,
                             QAbstractItemView, QSplitter)
from PyQt6.QtCore import QThread, pyqtSignal, QObject, Qt, QSize, QEvent
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QPen, QColor

# --- PyTorch Core Functions ---
def get_transforms():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

class JPGDataset(Dataset):
    def __init__(self, target_dir, filenames, transform=None):
        self.target_dir = target_dir
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.target_dir, filename)
        try:
            image = Image.open(filepath).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, filename, True
        except Exception as e:
            # Return dummy tensor on failure but flag it as False so we can skip it
            dummy = torch.zeros((3, 384, 384))
            return dummy, filename, False

def execute_model(target_dir, model_path, force_device="Auto", batch_size=64, progress_callback=None):
    if force_device == "GPU (NVIDIA)":
        device = torch.device("cuda")
    elif force_device == "CPU":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    print(f"\n======================================")
    print(f" INFO: Job is running on {str(device).upper()} ")
    print(f"======================================\n")

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        class_names = checkpoint['classes']
        
        model = models.efficientnet_v2_s(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()

        transform = get_transforms()
        
        valid_extensions = ('.jpg', '.jpeg')
        
        # Get list of valid JPG files (implicitly ignores .eer and other extensions)
        all_jpgs = [f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)]
        
        non_data_jpgs = [f for f in all_jpgs if "Data" not in f]
        files_to_process = [f for f in all_jpgs if "Data" in f]
        
        # 1. Move non-data JPGs out of the way first
        if non_data_jpgs:
            non_data_dir = os.path.join(target_dir, "Non-data_JPG")
            os.makedirs(non_data_dir, exist_ok=True)
            for filename in non_data_jpgs:
                src = os.path.join(target_dir, filename)
                dst = os.path.join(non_data_dir, filename)
                if os.path.exists(dst):
                    os.remove(dst)
                shutil.move(src, dst)
            print(f"Moved {len(non_data_jpgs)} non-data JPGs to Non-data_JPG/.")

        total_files = len(files_to_process)
        
        if total_files == 0:
            print("No Data JPG files found in the target directory to process.")
            return

        # 2. Setup bad and good directories for the inference results
        bad_dir = os.path.join(target_dir, "bad_jpg")
        good_dir = os.path.join(target_dir, "good_jpg")
        os.makedirs(bad_dir, exist_ok=True)
        os.makedirs(good_dir, exist_ok=True)

        moved_bad_count = 0
        moved_good_count = 0
        good_filenames = []

        print(f"Scanning {total_files} images using DataLoader (Batching enabled)...")
        
        # Setup DataLoader for fast parallel disk reading and batched GPU processing
        num_workers = min(4, os.cpu_count() or 1)
        dataset = JPGDataset(target_dir, files_to_process, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        processed_count = 0
        confidence_dict = {}
        
        with torch.no_grad():
            for inputs, filenames, valids in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                # Find the index for "good" to standardize the 0-100 score
                good_idx = class_names.index("good") if "good" in class_names else 1

                for i in range(len(filenames)):
                    filename = filenames[i]
                    is_valid = valids[i].item()
                    filepath = os.path.join(target_dir, filename)

                    if not is_valid:
                        print(f"Could not read/process {filename}, skipping.")
                        continue

                    predicted_class = class_names[preds[i]]
                    
                    # Calculate score (0 to 100, where 100 means 100% confident it's "good")
                    score = probs[i][good_idx].item() * 100
                    confidence_dict[filename] = round(score, 2)
                    
                    if predicted_class == "bad":
                        dest_path = os.path.join(bad_dir, filename)
                        if os.path.exists(dest_path):
                            os.remove(dest_path) 
                        shutil.move(filepath, dest_path)
                        moved_bad_count += 1
                        print(f"Moved: {filename} -> bad_jpg/")
                    else:
                        dest_path = os.path.join(good_dir, filename)
                        if os.path.exists(dest_path):
                            os.remove(dest_path)
                        shutil.move(filepath, dest_path)
                        moved_good_count += 1
                        good_filenames.append(filename)
                        print(f"Moved: {filename} -> good_jpg/")

                processed_count += len(filenames)
                # Update progress bar if running locally
                if progress_callback:
                    progress_pct = int((processed_count / total_files) * 100)
                    progress_callback(progress_pct)

        # Write the filelist_goodJPG.txt (saving only the filenames, not the paths)
        list_path = os.path.join(target_dir, "filelist_goodJPG.txt")
        with open(list_path, 'w') as f:
            for name in sorted(good_filenames):
                f.write(f"{name}\n")
                
        # Write the confident_level.json
        conf_path = os.path.join(target_dir, "confident_level.json")
        try:
            with open(conf_path, 'w') as f:
                json.dump(confidence_dict, f, indent=4)
        except Exception as e:
            print(f"Could not save confidence levels: {e}")

        print(f"\nExecution complete.")
        if non_data_jpgs:
            print(f"- Moved {len(non_data_jpgs)} non-data JPGs to Non-data_JPG/.")
        print(f"- Moved {moved_bad_count} bad micrographs to bad_jpg/.")
        print(f"- Moved {moved_good_count} good micrographs to good_jpg/.")
        print(f"- Saved {len(good_filenames)} good filenames to {list_path}.")
        
        return list_path

    except Exception as e:
        print(f"Execution failed: {e}")
        return None

# --- PyQt6 GUI Implementation ---

class EmittingStream(QObject):
    textWritten = pyqtSignal(str)
    
    def __init__(self, textWritten=None):
        super().__init__()
        if textWritten:
            self.textWritten.connect(textWritten)

    def write(self, text):
        self.textWritten.emit(str(text))
        
    def flush(self): pass

class WorkerTask(QThread):
    finished = pyqtSignal(str)
    progress_update = pyqtSignal(int)

    def __init__(self, kwargs):
        super().__init__()
        self.kwargs = kwargs

    def run(self):
        result_path = execute_model(
            self.kwargs['target_dir'], 
            self.kwargs['model_path'], 
            force_device=self.kwargs['force_device'],
            batch_size=self.kwargs.get('batch_size', 64),
            progress_callback=self.progress_update.emit
        )
        self.finished.emit(str(result_path) if result_path else "")

class SlurmMonitorThread(QThread):
    progress_update = pyqtSignal(int)
    finished = pyqtSignal(str)

    def __init__(self, target_dir, total_files):
        super().__init__()
        self.target_dir = target_dir
        self.total_files = total_files
        self._is_running = True

    def run(self):
        valid_extensions = ('.jpg', '.jpeg')
        while self._is_running:
            start_time = time.time()
            try:
                all_files = os.listdir(self.target_dir)
                remaining = len([f for f in all_files if "Data" in f and f.lower().endswith(valid_extensions)])
            except Exception:
                remaining = self.total_files

            check_duration = time.time() - start_time
            
            if self.total_files > 0:
                processed = self.total_files - remaining
                pct = int((processed / self.total_files) * 100)
                self.progress_update.emit(pct)
            
            if remaining == 0:
                self.progress_update.emit(100)
                list_path = os.path.join(self.target_dir, "filelist_goodJPG.txt")
                self.finished.emit(list_path)
                break
            
            # Sleep based on how long the directory check took + 3 seconds
            time.sleep(check_duration + 3.0)

    def stop(self):
        self._is_running = False

class ThumbnailLoader(QThread):
    thumbnail_ready = pyqtSignal(str, QImage, str)
    finished_loading = pyqtSignal()

    def __init__(self, target_dir):
        super().__init__()
        self.target_dir = target_dir
        self._is_running = True

    def run(self):
        for cat in ["good", "bad"]:
            cat_dir = os.path.join(self.target_dir, f"{cat}_jpg")
            if not os.path.exists(cat_dir): continue
            
            files = [f for f in os.listdir(cat_dir) if f.lower().endswith(('.jpg', '.jpeg'))]
            for f in files:
                if not self._is_running: break
                path = os.path.join(cat_dir, f)
                img = QImage(path)
                if not img.isNull():
                    # Load a fast, smaller thumbnail to fit 8 per row
                    thumb = img.scaled(125, 125, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
                    self.thumbnail_ready.emit(f, thumb, cat)
        self.finished_loading.emit()

    def stop(self):
        self._is_running = False


from PyQt6.QtWidgets import QSizePolicy
from PyQt6.QtCore import QSize

class PreviewLabel(QLabel):
    def __init__(self, text=""):
        super().__init__(text)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet("border: 2px solid #3e3e42; background-color: #252526; color: #d4d4d4; font-size: 14pt;")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(200, 200)
        self._pixmap = None

    def sizeHint(self):
        # Override the size hint so large images don't block the panel from shrinking
        return QSize(300, 300)

    def set_custom_pixmap(self, pixmap):
        self._pixmap = pixmap
        super().clear()  # Erase the text so it doesn't render under the image
        self.setStyleSheet("border: 2px solid #98c379; background-color: #1e1e1e;")
        self.update()  # Instantly triggers paintEvent without triggering a layout shift
        
    def clear_custom(self):
        self._pixmap = None
        self.setText("Hover over an image\nto see preview")
        self.setStyleSheet("border: 2px solid #3e3e42; background-color: #252526; color: #d4d4d4; font-size: 14pt;")
        self.update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update() # Force an instant repaint whenever the boundary is dragged

    def paintEvent(self, event):
        # Let Qt draw the stylesheet border, background, and any text first
        super().paintEvent(event) 
        
        # Then, manually paint the image on top perfectly fitted to the current panel size
        if self._pixmap and not self._pixmap.isNull():
            painter = QPainter(self)
            # Guarantee fast HPC rendering (No bilinear lag)
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
            
            # Get the exact canvas space available inside the border
            rect = self.contentsRect()
            
            # Calculate the exact size needed based on current panel dimensions
            scaled_size = self._pixmap.size().scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio)
            
            # Center the image dynamically
            x = rect.x() + (rect.width() - scaled_size.width()) // 2
            y = rect.y() + (rect.height() - scaled_size.height()) // 2
            
            # Draw it directly to the screen
            painter.drawPixmap(x, y, scaled_size.width(), scaled_size.height(), self._pixmap)

class ExamineWindow(QDialog):
    def __init__(self, target_dir, parent=None):
        super().__init__(parent)
        self.target_dir = target_dir
        self.setWindowTitle("Examine Segregated Images")
        self.resize(1480, 800) # 400px for left panel + 1040px for 8 items + 40px for padding/scrollbar
        main_layout = QHBoxLayout(self)
        
        # Create a Splitter for resizable panels
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setStyleSheet("QSplitter::handle { background-color: #3e3e42; width: 4px; margin: 0px 5px; } QSplitter::handle:hover { background-color: #98c379; }")
        main_layout.addWidget(self.splitter)

        # Left panel for dynamically resizing preview
        self.left_panel_widget = QWidget()
        self.preview_layout = QVBoxLayout(self.left_panel_widget)
        
        # Use our new dynamically resizing label instead
        self.hover_label = PreviewLabel("Hover over an image\nto see preview")
        # Give it a stretch factor of 1 so it aggressively consumes all vertical space!
        self.preview_layout.addWidget(self.hover_label, 1)
        
        # Add filename label under preview
        self.filename_label = QLabel("Filename: None")
        self.filename_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.filename_label.setStyleSheet("color: #98c379; font-size: 7pt; padding: 5px;")
        self.filename_label.setWordWrap(True)
        # Allow the text to span across the expanding layout
        self.filename_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.preview_layout.addWidget(self.filename_label)
        
        # Add tips section
        self.tips_label = QLabel(
            "<h3>How to use:</h3>"
            "<ul style='line-height: 1.5; margin-left: 0px;'>"
            "<li><b>Hover:</b> View a larger 400x400 preview above.</li>"
            "<li><b>Right-Click or Press 'S':</b> Move the hovered image to the other category (Good ↔ Bad).</li>"
            "<li><b>Ctrl+V/Cmd+V:</b> Jump to the image having the same filename you pasted.</li>"
            "<li><b>Ctrl+Z/Cmd+Z:</b> Undo moving categories.</li>"
            "</ul>"
        )
        self.tips_label.setWordWrap(True)
        self.tips_label.setStyleSheet("""
            background-color: #252526; 
            color: #d4d4d4; 
            font-size: 9pt; 
            padding: 15px; 
            border: 1px solid #3e3e42; 
            border-radius: 4px; 
            margin-top: 15px;
        """)
        self.tips_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        
        self.preview_layout.addWidget(self.tips_label)
        
        # Add status label to the left panel
        self.status_label = QLabel("Good: 0  |  Bad: 0")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #d4d4d4; font-weight: bold; font-size: 8pt; padding: 10px; background-color: #252526; border: 1px solid #3e3e42; border-radius: 4px; margin-top: 10px;")
        self.preview_layout.addWidget(self.status_label)
        
        # addStretch removed so the image is allowed to expand downward
        self.splitter.addWidget(self.left_panel_widget)

        self.tabs = QTabWidget()
        self.good_list = self.create_list_widget()
        self.bad_list = self.create_list_widget()

        self.tabs.addTab(self.good_list, "Good")
        self.tabs.addTab(self.bad_list, "Bad")
        
        # Add the Execute Separation button to the top right corner of the tabs
        self.btn_execute_sep = QPushButton("Execute Separation")
        self.btn_execute_sep.setStyleSheet("background-color: #98c379; color: #1e1e1e; font-weight: bold; padding: 4px 15px; margin: 2px;")
        self.btn_execute_sep.clicked.connect(self.execute_separation)
        self.tabs.setCornerWidget(self.btn_execute_sep, Qt.Corner.TopRightCorner)
        
        self.splitter.addWidget(self.tabs)
        
        # Set default split ratios (left panel tight to the 400px fixed preview)
        self.splitter.setSizes([430, 1050])

        self.load_state()
        self._menu_open = False
        self.undo_stack = []  # Track up to 100 move actions

        # Count total files for progress tracking
        self.total_good = 0
        self.total_bad = 0
        self.loaded_good = 0
        self.loaded_bad = 0
        self._is_loading = True
        
        good_dir = os.path.join(self.target_dir, "good_jpg")
        if os.path.exists(good_dir):
            self.total_good = len([f for f in os.listdir(good_dir) if f.lower().endswith(('.jpg', '.jpeg'))])
            
        bad_dir = os.path.join(self.target_dir, "bad_jpg")
        if os.path.exists(bad_dir):
            self.total_bad = len([f for f in os.listdir(bad_dir) if f.lower().endswith(('.jpg', '.jpeg'))])

        # Load confidence data
        self.confidence_data = {}
        conf_path = os.path.join(self.target_dir, "confident_level.json")
        if os.path.exists(conf_path):
            try:
                with open(conf_path, 'r') as f:
                    self.confidence_data = json.load(f)
            except Exception as e:
                print(f"Could not load confidence data: {e}")

        # Start asynchronous thumbnail loading
        self.loader = ThumbnailLoader(self.target_dir)
        self.loader.thumbnail_ready.connect(self.add_thumbnail)
        self.loader.finished_loading.connect(self.restore_state)
        self.loader.start()

    def load_state(self):
        self.saved_target_name = None
        list_path = os.path.join(self.target_dir, "filelist_goodJPG.txt")
        if os.path.exists(list_path):
            try:
                with open(list_path, "r") as f:
                    lines = f.read().splitlines()
                    for line in lines:
                        line = line.strip()
                        # Check if the line is a reversed .jpg or .jpeg filename
                        if line.startswith("gpj.") or line.startswith("gepj."):
                            self.saved_target_name = line[::-1]
            except: pass

    def get_current_visible_filename(self):
        # Get the top-left most visible item for whichever tab is currently active
        active_list = self.good_list if self.tabs.currentIndex() == 0 else self.bad_list
        item = active_list.itemAt(5, 5)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def save_state(self):
        # Automatically updates the filelist and appends the reversed filename
        self.update_filelist()

    def find_item_by_filename(self, list_widget, filename):
        # Helper function to search the hidden UserRole data instead of display text
        for i in range(list_widget.count()):
            item = list_widget.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == filename:
                return item
        return None

    def update_status_counts(self):
        if getattr(self, '_is_loading', False):
            # Calculate percentages (default to 100 if empty folder)
            good_pct = int((self.loaded_good / self.total_good * 100)) if self.total_good > 0 else 100
            bad_pct = int((self.loaded_bad / self.total_bad * 100)) if self.total_bad > 0 else 100
            self.status_label.setText(f"Good: {good_pct}%  |  Bad: {bad_pct}%")
        else:
            # Show actual item counts
            good_cnt = self.good_list.count()
            bad_cnt = self.bad_list.count()
            self.status_label.setText(f"Good: {good_cnt}  |  Bad: {bad_cnt}")

    def restore_state(self):
        self._is_loading = False
        self.update_status_counts()
        if hasattr(self, 'saved_target_name') and self.saved_target_name:
            # Look in the Good list first
            item = self.find_item_by_filename(self.good_list, self.saved_target_name)
            if item:
                self.tabs.setCurrentIndex(0)
                self.good_list.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtTop)
                self.show_hover_image(item)  # Auto-load preview
                return
                
            # Otherwise, look in the Bad list
            item = self.find_item_by_filename(self.bad_list, self.saved_target_name)
            if item:
                self.tabs.setCurrentIndex(1)
                self.bad_list.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtTop)
                self.show_hover_image(item)  # Auto-load preview

    def create_list_widget(self):
        lw = QListWidget()
        lw.setViewMode(QListWidget.ViewMode.IconMode)
        lw.setIconSize(QSize(125, 125))
        lw.setGridSize(QSize(130, 130)) 
        lw.setResizeMode(QListWidget.ResizeMode.Adjust)
        lw.setMovement(QListWidget.Movement.Static)
        lw.setSpacing(0)
        
        # Force disable selection, focus rings, and override the global style
        lw.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        lw.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        lw.setStyleSheet("QListWidget::item:selected { background-color: transparent; border: none; }")
        
        lw.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        lw.customContextMenuRequested.connect(self.show_context_menu)
        lw.setMouseTracking(True)
        lw.itemEntered.connect(self.show_hover_image)
        lw.viewport().installEventFilter(self)
        return lw

    def clear_preview(self):
        self.current_hovered_item = None
        self.hover_label.clear_custom()
        if hasattr(self, 'filename_label'):
            self.filename_label.setText("Filename: None")

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Type.Wheel:
            self.clear_preview()
            # Determine which list widget is being scrolled
            lw = self.good_list if obj == self.good_list.viewport() else self.bad_list
            delta = event.angleDelta().y()
            sb = lw.verticalScrollBar()
            # Force exactly 1 row (130px) per scroll tick
            if delta > 0:
                sb.setValue(sb.value() - 130)
            else:
                sb.setValue(sb.value() + 130)
            return True # Consume the event so it doesn't double-scroll
        elif event.type() == QEvent.Type.Leave:
            if not getattr(self, '_menu_open', False):
                self.clear_preview()
        return super().eventFilter(obj, event)

    def apply_border_to_image(self, qimage, filename):
        conf = self.confidence_data.get(filename)
        if conf is None:
            return qimage
            
        color = None
        if conf == "manual":
            color = QColor(86, 156, 214) # Blue
        elif isinstance(conf, (int, float)) and 35 <= conf <= 65:
            color = QColor(220, 220, 170) # Yellow
            
        if color:
            from PyQt6.QtGui import QImage
            img_copy = qimage.convertToFormat(QImage.Format.Format_RGB32)
            
            painter = QPainter(img_copy)
            pen = QPen(color)
            pen.setWidth(8) # 4px thick border visible on all sides
            painter.setPen(pen)
            # Draw rect slightly inside the bounds to ensure it isn't cut off
            painter.drawRect(2, 2, img_copy.width() - 4, img_copy.height() - 4)
            painter.end()
            return img_copy
        return qimage

    def add_thumbnail(self, filename, qimage, category):
        processed_img = self.apply_border_to_image(qimage, filename)
        icon = QIcon(QPixmap.fromImage(processed_img))
        item = QListWidgetItem(icon, "")  # Must pass an empty string so PyQt knows which constructor to use
        item.setData(Qt.ItemDataRole.UserRole, filename)
        
        if category == "good":
            self.good_list.addItem(item)
            self.loaded_good += 1
        else:
            self.bad_list.addItem(item)
            self.loaded_bad += 1
            
        # Update the label roughly every 10 images to prevent UI lag during fast loading
        if (self.loaded_good + self.loaded_bad) % 10 == 0:
            self.update_status_counts()

    def show_hover_image(self, item):
        self.current_hovered_item = item
        filename = item.data(Qt.ItemDataRole.UserRole)
        # Determine actual current folder based on which list widget holds the item
        is_in_good = (item.listWidget() == self.good_list)
        cat_dir = "good_jpg" if is_in_good else "bad_jpg"
        path = os.path.join(self.target_dir, cat_dir, filename)
        
        img = QPixmap(path)
        if not img.isNull():
            # Send the original pixmap to our custom label to handle dynamic scaling
            self.hover_label.set_custom_pixmap(img)
            self.filename_label.setText(f"Filename: {filename}")

    def show_context_menu(self, pos):
        sender_list = self.sender()
        item = sender_list.itemAt(pos)
        if not item: return

        # Ensure preview updates to the item being right-clicked
        self.show_hover_image(item)
        self._menu_open = True

        menu = QMenu(self)
        is_good_tab = (sender_list == self.good_list)
        
        action_move = menu.addAction("Send to Bad" if is_good_tab else "Send to Good")
        menu.addSeparator()
        action_copy = menu.addAction("Copy Filename")
        
        action_open = None
        sys_name = platform.system()
        if sys_name == "Windows":
            action_open = menu.addAction("Open in File Explorer")
        elif sys_name == "Darwin":
            action_open = menu.addAction("Open in Finder")

        selected_action = menu.exec(sender_list.mapToGlobal(pos))
        self._menu_open = False
        
        if not selected_action: 
            # Optionally clear preview if they clicked away without selecting anything
            self.clear_preview()
            return

        filename = item.data(Qt.ItemDataRole.UserRole)
        
        if selected_action == action_move:
            self.move_image(item, is_good_tab)
        elif selected_action == action_copy:
            QApplication.clipboard().setText(filename)
        elif action_open and selected_action == action_open:
            cat_dir = "good_jpg" if is_good_tab else "bad_jpg"
            filepath = os.path.join(self.target_dir, cat_dir, filename)
            
            # Select/highlight the file natively depending on OS
            if sys_name == "Windows":
                subprocess.Popen(f'explorer /select,"{os.path.normpath(filepath)}"')
            elif sys_name == "Darwin":
                subprocess.Popen(['open', '-R', filepath])

    def move_image(self, item, from_good, is_undo=False, insert_row=None, original_conf=None):
        filename = item.data(Qt.ItemDataRole.UserRole)
        src_dir = os.path.join(self.target_dir, "good_jpg" if from_good else "bad_jpg")
        dst_dir = os.path.join(self.target_dir, "bad_jpg" if from_good else "good_jpg")

        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)

        try:
            # Physically move the file
            if os.path.exists(dst_path):
                os.remove(dst_path)
            shutil.move(src_path, dst_path)
            
            # Move the item in the UI
            src_list = self.good_list if from_good else self.bad_list
            dst_list = self.bad_list if from_good else self.good_list

            row = src_list.row(item)
            taken_item = src_list.takeItem(row)
            
            # Update confidence based on undo state and save JSON
            if not is_undo:
                original_conf = self.confidence_data.get(filename)
                self.confidence_data[filename] = "manual"
            else:
                if original_conf is not None:
                    self.confidence_data[filename] = original_conf
                else:
                    self.confidence_data.pop(filename, None)
            conf_path = os.path.join(self.target_dir, "confident_level.json")
            try:
                with open(conf_path, 'w') as f:
                    json.dump(self.confidence_data, f, indent=4)
            except: pass
            
            # Redraw the thumbnail icon so it gets the correct border instantly
            img = QImage(dst_path)
            if not img.isNull():
                thumb = img.scaled(125, 125, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
                processed_thumb = self.apply_border_to_image(thumb, filename)
                taken_item.setIcon(QIcon(QPixmap.fromImage(processed_thumb)))
                
            if is_undo and insert_row is not None:
                dst_list.insertItem(insert_row, taken_item)
            else:
                dst_list.addItem(taken_item)
            
            self.clear_preview()
            self.update_filelist()
            self.update_status_counts()
            
            if not is_undo:
                self.undo_stack.append((filename, from_good, row, original_conf))
                if len(self.undo_stack) > 100:
                    self.undo_stack.pop(0)
            
        except Exception as e:
            print(f"Error moving file: {e}")

    def update_filelist(self):
        list_path = os.path.join(self.target_dir, "filelist_goodJPG.txt")
        good_items = []
        for i in range(self.good_list.count()):
            good_items.append(self.good_list.item(i).data(Qt.ItemDataRole.UserRole))
            
        with open(list_path, 'w') as f:
            for name in sorted(good_items):
                f.write(f"{name}\n")
                
            # Append reversed filename of current view as the tracker
            current_file = self.get_current_visible_filename()
            if current_file:
                f.write(f"{current_file[::-1]}\n")

    def execute_separation(self):
        # Force a filelist save just to ensure tracking is up-to-date
        self.update_filelist()
        
        # Create a small popup dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Execute Separation")
        dialog.resize(500, 150)
        layout = QVBoxLayout(dialog)

        label = QLabel("Select Movies Folder\n\n- The folder containing all EER files corresponding to the current JPG files")
        layout.addWidget(label)

        input_layout = QHBoxLayout()
        path_input = QLineEdit()
        path_input.setPlaceholderText("Browse or paste folder path...")
        browse_btn = QPushButton("Browse")
        
        def do_browse():
            folder = QFileDialog.getExistingDirectory(dialog, "Select Movies Folder", options=QFileDialog.Option.DontUseNativeDialog)
            if folder:
                path_input.setText(folder)
                
        browse_btn.clicked.connect(do_browse)
        input_layout.addWidget(path_input)
        input_layout.addWidget(browse_btn)
        layout.addLayout(input_layout)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

        # Show dialog and wait for user action
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        movies_folder = path_input.text().strip()
        if not movies_folder or not os.path.isdir(movies_folder):
            QMessageBox.warning(self, "Warning", "Please provide a valid folder path.")
            return
            
        # Build the set of expected .eer filenames straight from the Good list
        eer_filenames = set()
        for i in range(self.good_list.count()):
            filename = self.good_list.item(i).data(Qt.ItemDataRole.UserRole)
            name, _ = os.path.splitext(filename)
            eer_filenames.add(f"{name}_EER.eer")
            
        if not eer_filenames:
            QMessageBox.warning(self, "Warning", "There are no images in the Good list to process.")
            return
            
        try:
            parent_dir = os.path.dirname(os.path.abspath(movies_folder))
            bad_folder = os.path.join(parent_dir, "bAD_mOviEs")
            os.makedirs(bad_folder, exist_ok=True)
            
            eer_files = [f for f in os.listdir(movies_folder) if f.endswith('_EER.eer')]
            initial_count = len(eer_files)
            
            moved_count = 0
            for filename in eer_files:
                if filename not in eer_filenames:
                    src = os.path.join(movies_folder, filename)
                    dst = os.path.join(bad_folder, filename)
                    shutil.move(src, dst)
                    moved_count += 1
                    
            remaining_count = len([f for f in os.listdir(movies_folder) if f.endswith('_EER.eer')])
            
            msg = (f"Initial eer files in {os.path.basename(movies_folder)}: {initial_count}\n"
                   f"{moved_count} eer files moved to <bAD_mOviEs>\n"
                   f"{remaining_count} eer files remain in {os.path.basename(movies_folder)}")
                   
            QMessageBox.information(self, "Separation Complete", msg)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to process files:\n{e}")

    def keyPressEvent(self, event):
        from PyQt6.QtGui import QKeySequence
        
        # Check for "S" key to move hovered image
        if event.key() == Qt.Key.Key_S and getattr(self, 'current_hovered_item', None):
            is_in_good = (self.current_hovered_item.listWidget() == self.good_list)
            self.move_image(self.current_hovered_item, is_in_good)
            return

        # Check for Undo command (Ctrl+Z / Cmd+Z)
        if event.matches(QKeySequence.StandardKey.Undo):
            if self.undo_stack:
                filename, was_from_good, orig_row, orig_conf = self.undo_stack.pop()
                # If originally moved from Good to Bad, it is currently in Bad
                current_list = self.bad_list if was_from_good else self.good_list
                item = self.find_item_by_filename(current_list, filename)
                
                if item:
                    # Reverse the move direction, pass the original row and confidence
                    self.move_image(item, not was_from_good, is_undo=True, insert_row=orig_row, original_conf=orig_conf)
                    
                    # Prevent losing progress track by jumping back to the restored item
                    target_list = self.good_list if was_from_good else self.bad_list
                    target_tab_idx = 0 if was_from_good else 1
                    
                    self.tabs.setCurrentIndex(target_tab_idx)
                    # Use EnsureVisible instead of PositionAtTop to minimize jarring jumps if it's already on screen
                    target_list.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
                    self.show_hover_image(item)
            return

        # Check if the user triggered the system paste command (Ctrl+V / Cmd+V)
        if event.matches(QKeySequence.StandardKey.Paste):
            clipboard_text = QApplication.clipboard().text().strip()
            if clipboard_text:
                # Search in Good list
                item = self.find_item_by_filename(self.good_list, clipboard_text)
                if item:
                    self.tabs.setCurrentIndex(0)
                    self.good_list.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtTop)
                    self.show_hover_image(item)
                    return
                # Search in Bad list
                item = self.find_item_by_filename(self.bad_list, clipboard_text)
                if item:
                    self.tabs.setCurrentIndex(1)
                    self.bad_list.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtTop)
                    self.show_hover_image(item)
                    return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.save_state()
        self.loader.stop()
        self.loader.wait()
        event.accept()

class ExecGUI(QWidget):
    def __init__(self, model_path="", ipc_file="", parent_pid=""):
        super().__init__()
        self.ipc_file = ipc_file
        self.parent_pid = parent_pid
        self.initUI()
        sys.stdout = EmittingStream(textWritten=self.normalOutputWritten)
        sys.stderr = EmittingStream(textWritten=self.normalOutputWritten)
        
        if model_path:
            self.load_model_input.setText(model_path)

    def initUI(self):
        self.setWindowTitle('AutoJPG')
        self.resize(600, 450)

        main_layout = QVBoxLayout()

        self.is_linux = (platform.system() == "Linux")

        # --- Settings Row ---
        settings_layout = QHBoxLayout()
        settings_layout.addWidget(QLabel("<b>Environment:</b>"))
        self.env_label = QLabel("Slurm Cluster" if self.is_linux else "Local Computer")
        self.env_label.setStyleSheet("color: #98c379; font-weight: bold;")
        settings_layout.addWidget(self.env_label)
        
        settings_layout.addWidget(QLabel("<b>  Device:</b>"))
        self.device_label = QLabel("Auto")
        self.device_label.setStyleSheet("color: #98c379; font-weight: bold;")
        settings_layout.addWidget(self.device_label)
        settings_layout.addStretch()
        main_layout.addLayout(settings_layout)

        # --- Inputs ---
        self.load_model_input, btn1 = self.create_browse_row("Trained Model (.pth):")
        btn1.clicked.connect(self.load_model_dialog)
        main_layout.addLayout(self.load_model_input.parent_layout)

        self.target_dir_input, btn2 = self.create_browse_row("Target JPG Folder:")
        btn2.clicked.connect(self.browse_folder)
        main_layout.addLayout(self.target_dir_input.parent_layout)

        # --- Batch Size Input ---
        batch_layout = QHBoxLayout()
        batch_label = QLabel("Batch Size:")
        batch_label.setMinimumWidth(130)
        default_batch = "64" if self.is_linux else "16"
        self.batch_input = QLineEdit(default_batch)
        batch_layout.addWidget(batch_label)
        batch_layout.addWidget(self.batch_input)
        main_layout.addLayout(batch_layout)

        # --- Slurm Controls ---
        if self.is_linux:
            slurm_layout = QHBoxLayout()
            
            slurm_layout.addWidget(QLabel("Partition:"))
            self.partition_combo = QComboBox()
            self.partition_combo.setEditable(True)
            self.partition_combo.setMaximumWidth(120)
            
            partitions, default_part = self.get_slurm_partitions()
            if partitions:
                self.partition_combo.addItems(partitions)
                gpu_parts = [p for p in partitions if 'gpu' in p.lower()]
                if gpu_parts:
                    self.partition_combo.setCurrentText(gpu_parts[0])
                elif default_part:
                    self.partition_combo.setCurrentText(default_part)
            else:
                self.partition_combo.addItem("gpu")
                
            slurm_layout.addWidget(self.partition_combo)

            self.slurm_status_label = QLabel("  <b>Slurm Job:</b> Checking...")
            self.btn_check_slurm = QPushButton("Check Slurm")
            self.btn_cancel_slurm = QPushButton("Stop Slurm")
            self.btn_cancel_slurm.setStyleSheet("background-color: #D32F2F; color: white; border: 1px solid #D32F2F;")
            self.btn_cancel_slurm.setEnabled(False)

            self.btn_check_slurm.clicked.connect(self.check_slurm_status)
            self.btn_cancel_slurm.clicked.connect(self.cancel_slurm_job)

            slurm_layout.addWidget(self.slurm_status_label)
            slurm_layout.addWidget(self.btn_check_slurm)
            slurm_layout.addWidget(self.btn_cancel_slurm)
            slurm_layout.addStretch()
            main_layout.addLayout(slurm_layout)

        # --- Bottom ---
        self.run_btn = QPushButton("Execute Model")
        self.run_btn.setStyleSheet("background-color: #98c379; color: #1e1e1e; font-weight: bold; padding: 10px; border: 1px solid #98c379;")
        self.run_btn.clicked.connect(self.start_task)
        main_layout.addWidget(self.run_btn)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        main_layout.addWidget(QLabel("Console Output:"))
        main_layout.addWidget(self.log_box)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        main_layout.addWidget(self.progress_bar)

        self.btn_direct_examine = QPushButton("Examine Existing Folder")
        self.btn_direct_examine.setStyleSheet("background-color: #3e3e42; color: #d4d4d4; font-weight: bold; padding: 8px; margin-top: 5px;")
        self.btn_direct_examine.clicked.connect(self.open_direct_examine)
        main_layout.addWidget(self.btn_direct_examine)

        self.action_layout = QHBoxLayout()
        self.btn_examine = QPushButton("Examine")
        self.btn_examine.clicked.connect(self.open_examine_window)
        self.btn_proceed = QPushButton("Send Filelist to xcop")
        self.btn_proceed.setStyleSheet("background-color: #98c379; color: #1e1e1e; font-weight: bold;")
        self.btn_proceed.clicked.connect(self.proceed_to_xcop)
        
        self.action_layout.addWidget(self.btn_examine)
        self.action_layout.addWidget(self.btn_proceed)

        self.ok_layout = QHBoxLayout()
        self.btn_ok = QPushButton("OK")
        self.btn_ok.clicked.connect(self.reset_ui)
        self.ok_layout.addWidget(self.btn_ok)
        
        self.btn_examine.hide()
        self.btn_proceed.hide()
        self.btn_ok.hide()
        
        main_layout.addLayout(self.action_layout)
        main_layout.addLayout(self.ok_layout)

        self.setLayout(main_layout)
        self.check_slurm_status()

    def reset_ui(self):
        self.btn_examine.hide()
        self.btn_ok.hide()
        self.btn_proceed.hide()
        self.run_btn.setEnabled(True)
        
    def open_examine_window(self):
        target_dir = self.target_dir_input.text().strip()
        if os.path.exists(target_dir):
            self.examine_win = ExamineWindow(target_dir, self)
            self.examine_win.show()

    def open_direct_examine(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Processed Folder to Examine", options=QFileDialog.Option.DontUseNativeDialog)
        if folder:
            good_dir = os.path.join(folder, "good_jpg")
            bad_dir = os.path.join(folder, "bad_jpg")
            list_file = os.path.join(folder, "filelist_goodJPG.txt")

            # Check if all 3 required elements exist
            if os.path.exists(good_dir) and os.path.exists(bad_dir) and os.path.exists(list_file):
                self.examine_win = ExamineWindow(folder, self)
                self.examine_win.show()
            else:
                reply = QMessageBox.question(
                    self, 
                    "Setup Manual Separation", 
                    "The selected folder has not been separated yet.\n\n"
                    "Would you like to set up manual separation?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    valid_extensions = ('.jpg', '.jpeg')
                    good_filenames = []
                    
                    # Check if the folders already exist
                    if os.path.exists(good_dir) and os.path.exists(bad_dir):
                        # Folders exist, just index what is currently inside good_jpg
                        all_files = os.listdir(good_dir)
                        for f in all_files:
                            if f.lower().endswith(valid_extensions):
                                good_filenames.append(f)
                    else:
                        # Folders do not exist, do the full separation process
                        os.makedirs(good_dir, exist_ok=True)
                        os.makedirs(bad_dir, exist_ok=True)
                        
                        all_files = os.listdir(folder)
                        all_jpgs = [f for f in all_files if f.lower().endswith(valid_extensions)]
                        
                        # Mirror the filtering from your execution model
                        non_data_jpgs = [f for f in all_jpgs if "Data" not in f]
                        files_to_process = [f for f in all_jpgs if "Data" in f]
                        
                        # Move non-data JPGs out of the way
                        if non_data_jpgs:
                            non_data_dir = os.path.join(folder, "Non-data_JPG")
                            os.makedirs(non_data_dir, exist_ok=True)
                            for filename in non_data_jpgs:
                                shutil.move(os.path.join(folder, filename), os.path.join(non_data_dir, filename))
                        
                        # Move target files to good_jpg
                        for filename in files_to_process:
                            src = os.path.join(folder, filename)
                            dst = os.path.join(good_dir, filename)
                            try:
                                shutil.move(src, dst)
                                good_filenames.append(filename)
                            except Exception as e:
                                print(f"Error moving {filename}: {e}")
                                
                    # Generate the filelist
                    with open(list_file, 'w') as f:
                        for name in sorted(good_filenames):
                            f.write(f"{name}\n")
                            
                    # Create an empty confidence file
                    conf_path = os.path.join(folder, "confident_level.json")
                    with open(conf_path, 'w') as f:
                        json.dump({}, f)
                            
                    # Now open the examine window
                    self.examine_win = ExamineWindow(folder, self)
                    self.examine_win.show()
        
    def is_parent_alive(self):
        if not self.parent_pid: return False
        try:
            pid = int(self.parent_pid)
            if platform.system() == "Windows":
                import subprocess
                output = subprocess.check_output(f'tasklist /FI "PID eq {pid}"', shell=True).decode()
                return str(pid) in output
            else:
                import os
                os.kill(pid, 0)
                return True
        except Exception:
            return False
            
    def proceed_to_xcop(self):
        if self.ipc_file and hasattr(self, 'last_list_path'):
            try:
                with open(self.ipc_file, "w") as f:
                    f.write(self.last_list_path)
            except Exception as e:
                print(f"Error writing IPC file: {e}")
        self.close()

    def create_browse_row(self, label_text):
        layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setMinimumWidth(130)
        layout.addWidget(label)
        line_edit = QLineEdit()
        line_edit.setReadOnly(True)
        line_edit.parent_layout = layout 
        btn = QPushButton("Browse")
        layout.addWidget(line_edit)
        layout.addWidget(btn)
        return line_edit, btn

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Target Folder", options=QFileDialog.Option.DontUseNativeDialog)
        if folder:
            self.target_dir_input.setText(folder)

    def load_model_dialog(self):
        file, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PyTorch Models (*.pth)", options=QFileDialog.Option.DontUseNativeDialog)
        if file: 
            self.load_model_input.setText(file)

    def normalOutputWritten(self, text):
        from PyQt6.QtGui import QTextCursor
        cursor = self.log_box.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(text)
        self.log_box.setTextCursor(cursor)
        self.log_box.ensureCursorVisible()

    def get_slurm_partitions(self):
        try:
            res = subprocess.run(['sinfo', '-h', '-o', '%P'], capture_output=True, text=True)
            if res.returncode == 0:
                partitions = []
                default_part = None
                for line in res.stdout.strip().split('\n'):
                    p = line.strip()
                    if not p: continue
                    if p.endswith('*'):  
                        p = p[:-1]
                        default_part = p
                    if p not in partitions:
                        partitions.append(p)
                return partitions, default_part
        except Exception:
            pass
        return [], None

    # --- Slurm Tracking ---
    def get_saved_job_id(self):
        if os.path.exists("active_exec_slurm.txt"):
            with open("active_exec_slurm.txt", "r") as f:
                return f.read().strip()
        return None

    def check_slurm_status(self):
        if not hasattr(self, 'slurm_status_label'):
            return  # Skip checking if not on Linux (UI doesn't exist)
        job_id = self.get_saved_job_id()
        if not job_id:
            self.slurm_status_label.setText("<b>Slurm Job:</b> None")
            self.btn_cancel_slurm.setEnabled(False)
            return

        try:
            res = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
            if job_id in res.stdout:
                self.slurm_status_label.setText(f"<b>Slurm Job {job_id}:</b> <span style='color:#98c379;'>ACTIVE</span>")
                self.btn_cancel_slurm.setEnabled(True)
            else:
                self.slurm_status_label.setText(f"<b>Slurm Job {job_id}:</b> DONE")
                self.btn_cancel_slurm.setEnabled(False)
                if os.path.exists("active_exec_slurm.txt"):
                    os.remove("active_exec_slurm.txt") # Clear it if done
        except Exception:
            self.slurm_status_label.setText("<b>Slurm Job:</b> Local only")

    def cancel_slurm_job(self):
        job_id = self.get_saved_job_id()
        if job_id:
            reply = QMessageBox.question(self, 'Confirm', f"Kill Slurm Job {job_id}?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if reply == QMessageBox.StandardButton.Yes:
                try:
                    subprocess.run(['scancel', job_id])
                    print(f"\n[Slurm] Killed job {job_id}.")
                    self.check_slurm_status()
                except Exception as e:
                    print(f"\n[Slurm] Error: {e}")

    # --- Execution Logic ---
    def start_task(self):
        model_path = self.load_model_input.text().strip()
        target_dir = self.target_dir_input.text().strip()
        env = self.env_label.text()
        device_arg = "Auto"

        if not model_path or not target_dir:
            QMessageBox.warning(self, "Warning", "Please select a model and target directory.")
            return

        try:
            batch_size_val = int(self.batch_input.text())
        except ValueError:
            QMessageBox.warning(self, "Warning", "Batch Size must be a whole number.")
            return

        self.log_box.clear()
        self.run_btn.setEnabled(False)

        if env == "Local Computer":
            self.progress_bar.setValue(0)
            print("Starting execution locally...\n")
            kwargs = {'model_path': model_path, 'target_dir': target_dir, 'force_device': device_arg, 'batch_size': batch_size_val}
            self.worker = WorkerTask(kwargs)
            self.worker.progress_update.connect(self.update_progress)
            self.worker.finished.connect(self.task_finished)
            self.worker.start()

        elif env == "Slurm Cluster":
            partition = self.partition_combo.currentText().strip()
            if not partition:
                QMessageBox.warning(self, "Warning", "Please specify a Slurm partition.")
                self.run_btn.setEnabled(True)
                return
            self.submit_to_slurm(target_dir, model_path, device_arg, partition, batch_size_val)

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def submit_to_slurm(self, target_dir, model_path, device_arg, partition, batch_size_val):
        self.btn_examine.hide()
        self.btn_ok.hide()
        self.btn_proceed.hide()
        self.progress_bar.setValue(0)
        
        # Pre-count files for the monitor
        valid_extensions = ('.jpg', '.jpeg')
        try:
            all_files = os.listdir(target_dir)
            total_files = len([f for f in all_files if "Data" in f and f.lower().endswith(valid_extensions)])
        except Exception:
            total_files = 0
            
        if total_files == 0:
            print("No Data JPG files found in the target directory to process.")
            self.run_btn.setEnabled(True)
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        py_script_path = os.path.join(target_dir, f"AutoJPG_{timestamp}.py")
        
        # 1. Generate Headless Python Script
        with open(py_script_path, "w") as f:
            f.write(f"""import sys
import os

sys.path.append(r"{os.path.dirname(os.path.abspath(__file__))}")
from {os.path.basename(__file__).replace('.py', '')} import execute_model

print("Starting Slurm Execution...")
execute_model(
    target_dir=r"{target_dir}",
    model_path=r"{model_path}",
    force_device="{device_arg}",
    batch_size={batch_size_val}
)
print("Slurm Execution Complete.")
""")

        # --- Dynamic GPU Allocation ---
        needs_gpu = False
        if device_arg == "GPU (NVIDIA)":
            needs_gpu = True
        elif device_arg == "Auto" and torch.cuda.is_available():
            needs_gpu = True
            
        gpu_sbatch = "#SBATCH --gres=gpu:1\n" if needs_gpu else ""

        # 2. Generate Bash Script
        sh_script_path = os.path.join(target_dir, f"AutoJPG_{timestamp}.sh")
        with open(sh_script_path, "w", newline='\n') as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name=AutoJPG
#SBATCH --partition={partition}
{gpu_sbatch}#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output={os.path.join(target_dir, 'AutoJPG_%j.out')}
#SBATCH --error={os.path.join(target_dir, 'AutoJPG_%j.err')}

"{sys.executable}" -B "{py_script_path}"
""")

        # 3. Submit
        print("Generated scripts. Submitting to cluster...")
        try:
            result = subprocess.run(['sbatch', sh_script_path], capture_output=True, text=True, check=True)
            print(f"\nSUCCESS! {result.stdout}")
            print("Monitoring directory for progress...")
            
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                job_id = match.group(1)
                with open("active_exec_slurm.txt", "w") as f:
                    f.write(job_id)
            self.check_slurm_status()
            
            # Start monitor
            self.monitor = SlurmMonitorThread(target_dir, total_files)
            self.monitor.progress_update.connect(self.update_progress)
            self.monitor.finished.connect(self.task_finished)
            self.monitor.start()
            
        except subprocess.CalledProcessError as e:
            print(f"\nERROR submitting job: {e}")
            print(f"Slurm says: {e.stderr}") 
            self.run_btn.setEnabled(True)
            
        except Exception as e:
            print(f"\nUnexpected ERROR submitting job: {e}")
            self.run_btn.setEnabled(True)

    def task_finished(self, result_path=""):
        self.run_btn.setEnabled(True)
        if result_path:
            self.last_list_path = result_path
            self.btn_examine.show()
            self.btn_ok.show()
            print(f"\nFinished! Good JPG list saved at: {result_path}")
            if self.is_parent_alive():
                self.btn_proceed.show()
                self.btn_proceed.setEnabled(True)
            else:
                self.btn_proceed.show()
                self.btn_proceed.setEnabled(False)
                self.btn_proceed.setStyleSheet("background-color: grey; color: lightgrey;")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="", help="Path to the trained model .pth file")
    parser.add_argument("--ipc-file", default="", help="IPC file to communicate with xcop")
    parser.add_argument("--parent-pid", default="", help="PID of the parent xcop process")
    parser.add_argument("--ready-file", default="", help="File to signal parent that GUI is ready")
    args, unknown = parser.parse_known_args()

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Apply Global Stylesheet from xcop.py
    app.setStyleSheet("""
        QMainWindow, QDialog { background-color: #1e1e1e; color: #ffffff; }
        QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Arial; font-size: 10pt; }
        
        QTreeWidget, QListWidget, QTableWidget, QTreeView { 
            background-color: #252526; border: 1px solid #3e3e42; color: #cccccc; font-size: 10pt; outline: none; 
        }
        QTreeWidget::item, QListWidget::item { padding: 5px; }
        QTreeWidget::item:selected, QListWidget::item:selected { 
            background-color: #3e3e42; color: #ffffff; border-left: 3px solid #98c379; 
        }
        
        QTextEdit, QPlainTextEdit { 
            background-color: #1e1e1e; color: #d4d4d4; border: 1px solid #3e3e42; padding: 10px; 
            selection-background-color: #98c379; selection-color: #1e1e1e; 
        }
        
        QPushButton {
            background-color: #3e3e42; color: #d4d4d4; border: 1px solid #3e3e42;
            padding: 5px 15px; border-radius: 4px;
        }
        QPushButton:hover { background-color: #4e4e52; border: 1px solid #98c379; }
        QPushButton:pressed, QPushButton:checked { background-color: #98c379; color: #1e1e1e; border: 1px solid #98c379; }
        
        QLineEdit, QComboBox { 
            background-color: #3c3c3c; border: 1px solid #3c3c3c; color: #cccccc; 
            padding: 4px; border-radius: 2px; selection-background-color: #98c379; selection-color: #1e1e1e; 
        }
        
        QSpinBox, QDoubleSpinBox { 
            background-color: #3c3c3c; border: 1px solid #3c3c3c; color: #cccccc; 
            padding: 4px; padding-right: 24px; border-radius: 2px; 
            selection-background-color: #98c379; selection-color: #1e1e1e; 
        }
        
        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            width: 20px; 
        }
        
        QComboBox::drop-down { border: none; }
        QComboBox QAbstractItemView { 
            background-color: #252526; color: #d4d4d4; border: 1px solid #3e3e42; outline: none;
            selection-background-color: #98c379; selection-color: #1e1e1e;
        }
        QComboBox QAbstractItemView::item {
            padding: 5px; min-height: 25px; border: none !important;
        }
        QComboBox QAbstractItemView::item:hover,
        QComboBox QAbstractItemView::item:selected { 
            background-color: #98c379; color: #1e1e1e; border: none !important; outline: none !important;
        }

        /* Scrollbars (Apple Style) */
        QScrollBar:vertical { border: none; background: transparent; width: 12px; margin: 0px; }
        QScrollBar::handle:vertical { background-color: #424242; min-height: 20px; border-radius: 6px; margin: 2px; }
        QScrollBar::handle:vertical:hover { background-color: #606060; }
        QScrollBar::handle:vertical:pressed { background-color: #98c379; }
        QScrollBar::sub-line:vertical, QScrollBar::add-line:vertical { border: none; background: none; height: 0; }
        QScrollBar::up-arrow:vertical, QScrollBar::down-arrow:vertical, QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { border: none; background: none; }

        QScrollBar:horizontal { border: none; background: transparent; height: 12px; margin: 0px; }
        QScrollBar::handle:horizontal { background-color: #424242; min-width: 20px; border-radius: 6px; margin: 2px; }
        QScrollBar::handle:horizontal:hover { background-color: #606060; }
        QScrollBar::handle:horizontal:pressed { background-color: #98c379; }
        QScrollBar::sub-line:horizontal, QScrollBar::add-line:horizontal { border: none; background: none; width: 0; }
        QScrollBar::left-arrow:horizontal, QScrollBar::right-arrow:horizontal, QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { border: none; background: none; }

        /* Checkbox & Radio */
        QCheckBox::indicator, QRadioButton::indicator { 
            width: 14px; height: 14px; border: 1px solid #555; border-radius: 2px; background-color: #1e1e1e; margin: 1px;
        }
        QRadioButton::indicator { border-radius: 7px; }
        QCheckBox::indicator:checked, QRadioButton::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        
        /* Slider */
        QSlider::groove:horizontal { border: 1px solid #3e3e42; height: 6px; background: #252526; border-radius: 3px; }
        QSlider::handle:horizontal { background: #98c379; border: 1px solid #98c379; width: 12px; height: 12px; margin: -4px 0; border-radius: 6px; }
        QSlider::groove:vertical { border: 1px solid #3e3e42; width: 6px; background: #252526; border-radius: 3px; }
        QSlider::handle:vertical { background: #98c379; border: 1px solid #98c379; width: 12px; height: 12px; margin: 0 -4px; border-radius: 6px; }
        
        /* Progress Bar */
        QProgressBar { 
            border: 1px solid #3e3e42; border-radius: 2px; 
            background-color: #1e1e1e; text-align: center; 
            color: #d4d4d4; 
        }
        QProgressBar::chunk { background-color: #98c379; border-radius: 2px; color: #1e1e1e; }

        /* GroupBox & Tabs */
        QGroupBox { border: 1px solid #3e3e42; border-radius: 4px; margin-top: 1.5em; font-weight: bold; color: #98c379; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        
        QTabWidget::pane { border: 1px solid #3e3e42; }
        QTabBar::tab { background-color: #252526; color: #d4d4d4; padding: 8px 15px; border: 1px solid #3e3e42; border-bottom: none; }
        QTabBar::tab:selected { background-color: #1e1e1e; color: #98c379; font-weight: bold; }
        
        QHeaderView::section { background-color: #252526; color: #98c379; border: 1px solid #3e3e42; padding: 4px; font-weight: bold; }
        QTableCornerButton::section { background-color: #252526; border: 1px solid #3e3e42; }
        
        QMenu { background-color: #252526; color: #d4d4d4; border: 1px solid #3e3e42; }
        QMenu::item:selected { background-color: #3e3e42; color: #98c379; }
    """)
    
    gui = ExecGUI(model_path=args.model_path, ipc_file=args.ipc_file, parent_pid=args.parent_pid)
    gui.show()
    
    # Signal xcop.py that the UI is fully loaded and displayed
    if args.ready_file:
        try:
            with open(args.ready_file, "w") as f:
                f.write("ready")
        except Exception:
            pass
            
    sys.exit(app.exec())