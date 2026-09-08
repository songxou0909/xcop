import os
import sys
import math
import csv
import shutil
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


class ScoreTableModel(QAbstractTableModel):
    """Virtualized CSV model that keeps large score files responsive."""

    CRITERIA = {
        "rmsd": ("max", "RMSD"),
        "binder_ptm": ("min", "Binder pTM"),
        "pae_min": ("max", "PAE Min"),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.headers = []
        self.rows = []
        self.row_order = []
        self.criteria = {
            "rmsd": 5.0,
            "binder_ptm": 0.7,
            "pae_min": 5.0,
        }
        self.column_by_key = {}
        self.numeric_values = {}
        self.cell_passes = {}
        self.all_match = []
        self.match_count = 0
        self.show_seeds = False
        self.seed_rows = []
        self.ignore_cyclic_rmsd = True
        self.rmsd_ignored = []

    @staticmethod
    def _header_key(header):
        return str(header).strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _number(value):
        try:
            result = float(value)
            return result if math.isfinite(result) else None
        except (TypeError, ValueError):
            return None

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.row_order)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.headers)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.headers):
                return self.headers[section]
            return None
        return section + 1

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self.row_order):
            return None

        source_row = self.row_order[index.row()]
        column = index.column()
        if not 0 <= column < len(self.headers):
            return None

        if role == Qt.ItemDataRole.DisplayRole:
            return self.rows[source_row][column]

        key = self._header_key(self.headers[column])
        if role == Qt.ItemDataRole.TextAlignmentRole and key in {
            "rmsd", "binder_ptm", "pae_min", "iptm", "plddt"
        }:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        if role == Qt.ItemDataRole.BackgroundRole:
            passes = self.cell_passes.get(key, [])
            if key in self.CRITERIA and source_row < len(passes) and passes[source_row]:
                return QColor("#315c3a")
            if (
                key == "rmsd"
                and source_row < len(self.rmsd_ignored)
                and self.rmsd_ignored[source_row]
            ):
                return QColor("#5a4a23")
            if self.all_match and self.all_match[source_row]:
                return QColor("#263f2b")

        if role == Qt.ItemDataRole.ForegroundRole:
            if key in self.CRITERIA and self.cell_passes.get(key, [])[source_row]:
                return QColor("#f1fff3")

        if (
            role == Qt.ItemDataRole.ToolTipRole
            and key == "rmsd"
            and source_row < len(self.rmsd_ignored)
            and self.rmsd_ignored[source_row]
        ):
            return (
                "RMSD is reported for this linear-reference to cyclic "
                "prediction, but is not used as a pass/fail criterion."
            )

        if role == Qt.ItemDataRole.ToolTipRole and key in self.CRITERIA:
            direction, label = self.CRITERIA[key]
            relation = "at most" if direction == "max" else "at least"
            return f"{label} must be {relation} {self.criteria[key]:g}."

        return None

    def clear(self):
        self.beginResetModel()
        self.headers = []
        self.rows = []
        self.row_order = []
        self.column_by_key = {}
        self.numeric_values = {}
        self.cell_passes = {}
        self.all_match = []
        self.match_count = 0
        self.seed_rows = []
        self.rmsd_ignored = []
        self.endResetModel()

    def load_csv_files(self, paths):
        """Load one master CSV, or combine compatible batch CSVs in memory."""
        headers = []
        rows = []

        for path in paths:
            with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    continue
                if not headers:
                    headers = [str(field) for field in reader.fieldnames]
                for record in reader:
                    rows.append([record.get(header, "") for header in headers])

        self.beginResetModel()
        self.headers = headers
        self.rows = rows
        self.row_order = list(range(len(rows)))
        self.column_by_key = {
            self._header_key(header): column
            for column, header in enumerate(headers)
        }
        design_column = self.column_by_key.get("design_name")
        self.seed_rows = [
            self._is_seed_design(row[design_column])
            if design_column is not None else False
            for row in rows
        ]
        self.numeric_values = {}
        for key in self.CRITERIA:
            column = self.column_by_key.get(key)
            self.numeric_values[key] = (
                [self._number(row[column]) for row in rows]
                if column is not None else [None] * len(rows)
            )
        self.cell_passes = {
            key: [False] * len(rows) for key in self.CRITERIA
        }
        self.all_match = [False] * len(rows)
        self.match_count = 0
        self.endResetModel()
        self.set_criteria(
            self.criteria["rmsd"],
            self.criteria["binder_ptm"],
            self.criteria["pae_min"],
            self.ignore_cyclic_rmsd,
        )

    def set_criteria(
        self, rmsd, binder_ptm, pae_min, ignore_cyclic_rmsd=True
    ):
        self.criteria = {
            "rmsd": float(rmsd),
            "binder_ptm": float(binder_ptm),
            "pae_min": float(pae_min),
        }
        self.ignore_cyclic_rmsd = bool(ignore_cyclic_rmsd)

        if not self.rows:
            self.match_count = 0
            return

        topology_column = self.column_by_key.get("topology")
        reference_topology_column = self.column_by_key.get(
            "reference_topology"
        )
        self.rmsd_ignored = [
            self.ignore_cyclic_rmsd
            and topology_column is not None
            and str(row[topology_column]).strip().lower() == "cyclic"
            and (
                reference_topology_column is None
                or str(row[reference_topology_column]).strip().lower()
                != "cyclic"
            )
            for row in self.rows
        ]

        cell_passes = {}
        for key, (direction, _label) in self.CRITERIA.items():
            cutoff = self.criteria[key]
            if key not in self.column_by_key:
                cell_passes[key] = [False] * len(self.rows)
                continue
            values = self.numeric_values.get(key, [])
            if direction == "max":
                cell_passes[key] = [
                    value is not None and value <= cutoff for value in values
                ]
            else:
                cell_passes[key] = [
                    value is not None and value >= cutoff for value in values
                ]

        all_match = []
        for row in range(len(self.rows)):
            rmsd_passes = (
                cell_passes["rmsd"][row] or self.rmsd_ignored[row]
            )
            all_match.append(
                rmsd_passes
                and cell_passes["binder_ptm"][row]
                and cell_passes["pae_min"][row]
            )
        self.cell_passes = cell_passes
        self.all_match = all_match
        self._rebuild_row_order()

    @staticmethod
    def _is_seed_design(design_name):
        normalized = str(design_name or "").strip().lower()
        seed_position = normalized.rfind("_seed-")
        sample_position = normalized.rfind("_sample-")
        return seed_position >= 0 and sample_position > seed_position

    def _rebuild_row_order(self):
        visible_rows = [
            row for row in range(len(self.rows))
            if self.show_seeds
            or row >= len(self.seed_rows)
            or not self.seed_rows[row]
        ]
        matching = [row for row in visible_rows if self.all_match[row]]
        remaining = [row for row in visible_rows if not self.all_match[row]]

        self.beginResetModel()
        self.row_order = matching + remaining
        self.match_count = len(matching)
        self.endResetModel()

    def set_show_seeds(self, show_seeds):
        show_seeds = bool(show_seeds)
        if show_seeds == self.show_seeds:
            return
        self.show_seeds = show_seeds
        self._rebuild_row_order()

    def seed_row_count(self):
        return sum(self.seed_rows)

    def record_for_view_row(self, view_row):
        """Return a score row as a header-keyed dictionary."""
        if not 0 <= view_row < len(self.row_order):
            return {}
        source_row = self.row_order[view_row]
        if not 0 <= source_row < len(self.rows):
            return {}
        return {
            self._header_key(header): self.rows[source_row][column]
            for column, header in enumerate(self.headers)
        }


def merge_structure_chains_with_biopython(input_path, output_path):
    """Merge the first model into chain A and number its polymers from 1.

    Polymer residues are written first with consecutive residue numbers so an
    RF Diffusion contig describes the merged target as one continuous block.
    Ligands and waters are retained after the polymer and cannot introduce
    artificial gaps into that block.
    """
    try:
        from Bio.PDB import MMCIFIO, MMCIFParser, PDBIO, PDBParser
        from Bio.PDB.Chain import Chain
        from Bio.PDB.Model import Model
        from Bio.PDB.Polypeptide import is_aa, is_nucleic
        from Bio.PDB.Structure import Structure
    except ImportError as error:
        raise RuntimeError(
            "Biopython is required to merge RF Diffusion structures. "
            "Install it with 'python -m pip install biopython'."
        ) from error

    is_cif = input_path.lower().endswith((".cif", ".mmcif"))
    parser_class = MMCIFParser if is_cif else PDBParser
    parser = parser_class(QUIET=True)
    structure = parser.get_structure("rf_diffusion_target", input_path)
    source_model = next(structure.get_models(), None)
    if source_model is None:
        raise ValueError("No model was found in the structure file.")

    source_chains = [
        chain for chain in source_model
        if any(True for _atom in chain.get_atoms())
    ]
    if not source_chains:
        raise ValueError("No chains were found in the structure file.")

    def chain_anchor(chain):
        first_atom = next(chain.get_atoms(), None)
        if first_atom is None:
            return (0.0, 0.0, -9999.0)
        coordinates = first_atom.get_coord()
        return tuple(float(coordinate) for coordinate in coordinates[:3])

    # Preserve the previous layer-aware ordering: start at the highest chain,
    # then walk to nearby lower layers before beginning another stack.
    anchors = {chain.id: chain_anchor(chain) for chain in source_chains}
    remaining = {chain.id: chain for chain in source_chains}
    ordered_chains = []
    while remaining:
        active_id = max(
            remaining,
            key=lambda chain_id: (anchors[chain_id][2], str(chain_id)),
        )
        ordered_chains.append(remaining.pop(active_id))

        while True:
            active_x, active_y, active_z = anchors[active_id]
            candidates = []
            for chain_id in remaining:
                candidate_x, candidate_y, candidate_z = anchors[chain_id]
                z_drop = active_z - candidate_z
                xy_separation = math.hypot(
                    active_x - candidate_x,
                    active_y - candidate_y,
                )
                if z_drop > -2.0 and xy_separation < 25.0:
                    candidates.append(chain_id)

            if not candidates:
                break

            active_id = min(
                candidates,
                key=lambda chain_id: (
                    abs(active_z - anchors[chain_id][2]),
                    -anchors[chain_id][2],
                    str(chain_id),
                ),
            )
            ordered_chains.append(remaining.pop(active_id))

    polymer_residues = []
    non_polymer_residues = []
    for chain in ordered_chains:
        for residue in chain.get_residues():
            if not any(True for _atom in residue.get_atoms()):
                continue
            hetero_flag = str(residue.id[0]).strip()
            if not hetero_flag or is_aa(residue, standard=False) or is_nucleic(
                residue, standard=False
            ):
                polymer_residues.append(residue)
            else:
                non_polymer_residues.append(residue)

    if not polymer_residues:
        raise ValueError("No polymer residues were found in the structure file.")

    merged_structure = Structure("rf_diffusion_target")
    merged_structure.header = dict(getattr(structure, "header", {}) or {})
    merged_model = Model(0)
    merged_chain = Chain("A")
    merged_structure.add(merged_model)
    merged_model.add(merged_chain)

    merged_residue_number = 0
    for residue in polymer_residues:
        merged_residue_number += 1
        merged_residue = residue.copy()
        # Modified amino/nucleic acids can be stored as HETATM records. Treat
        # them as part of the polymer so the target numbering remains intact.
        merged_residue.id = (" ", merged_residue_number, " ")
        merged_chain.add(merged_residue)

    next_non_polymer_number = merged_residue_number
    for residue in non_polymer_residues:
        next_non_polymer_number += 1
        merged_residue = residue.copy()
        hetero_flag = residue.id[0]
        if not str(hetero_flag).strip():
            hetero_flag = f"H_{residue.resname.strip() or 'UNK'}"
        merged_residue.id = (hetero_flag, next_non_polymer_number, " ")
        merged_chain.add(merged_residue)

    temporary_path = output_path + ".merge.tmp"
    try:
        structure_io = MMCIFIO() if is_cif else PDBIO()
        structure_io.set_structure(merged_structure)
        structure_io.save(temporary_path)
        os.replace(temporary_path, output_path)
    except Exception:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
        raise

    return [str(chain.id) for chain in ordered_chains], merged_residue_number


def parse_slurm_job_id(output):
    """Extract a numeric job ID from normal or --parsable sbatch output."""
    import re

    match = re.search(
        r"(?:Submitted\s+batch\s+job\s+)?(\d+)(?:;[^\s]+)?",
        str(output or "").strip(),
    )
    if not match:
        raise ValueError(f"Could not parse a SLURM job ID from: {output!r}")
    return match.group(1)


def build_slurm_submit_command(submit_script, dependency_job_id=None):
    """Build an sbatch command that cancels unsatisfiable dependencies."""
    command = ["sbatch", "--parsable"]
    if dependency_job_id is not None:
        command.extend([
            f"--dependency=afterok:{dependency_job_id}",
            "--kill-on-invalid-dep=yes",
        ])
    command.append(submit_script)
    return command


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

def open_rf_diffusion(launch=True):
    try:
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
    print(f"RF Diffusion preview renderer: {renderer}")
    
    class _DetectorValue:
        def __init__(self, value):
            self.value = str(value)

        def text(self):
            return self.value

    class _DetectorFlag:
        def isChecked(self):
            return False

    class _DetectorLog:
        def __init__(self):
            self.messages = []

        def append(self, message):
            self.messages.append(str(message))

    class _DetectorButton:
        def __init__(self):
            self.enabled = False

        def setEnabled(self, enabled):
            self.enabled = bool(enabled)

    class _DetectorCombo:
        def __init__(self):
            self.items = []

        def blockSignals(self, _blocked):
            pass

        def clear(self):
            self.items.clear()

        def addItem(self, label, userData=None):
            self.items.append((label, userData))

    class _DetectorCanvas:
        def plot_chains(self, *_args, **_kwargs):
            pass

    class RFStructureDetector(QWidget):
        """Private, UI-free adapter for RF's shared structure detection."""

        def __init__(self):
            super().__init__()
            self.final_sandwiches = []
            self.detected_layers = 0
            self.results = {}
            self.working_file_path = None
            self.working_file_is_mutable = False
            self.original_filename_display = ""
            self.edit_z_min = _DetectorValue(4.6)
            self.edit_z_max = _DetectorValue(5.0)
            self.edit_xy_limit = _DetectorValue(3.0)
            self.edit_neighbor_count = _DetectorValue(6)
            self.chk_anti = _DetectorFlag()
            self.text_area = _DetectorLog()
            self.btn_beta = _DetectorButton()
            self.cmb_protofilaments = _DetectorCombo()
            self.mol_canvas = _DetectorCanvas()

        def browse_file(self, filename=None, mutable=False):
            if not filename or isinstance(filename, bool):
                filename, _ = QFileDialog.getOpenFileName(
                    self,
                    "Open PDB/CIF File",
                    "",
                    "Model Files (*.pdb *.cif *.mmcif);;All Files (*)",
                )
            if not filename:
                return

            filename = os.path.abspath(filename)
            self.original_filename_display = os.path.basename(filename)
            self.working_file_path = filename
            self.working_file_is_mutable = bool(mutable)
            self.text_area.messages.clear()
            # Detection is read-only: parse the selected structure directly and
            # keep any duplicate-chain correction in memory.  A hard crash can
            # therefore never leave a detector copy beside the source file.
            pdb_lines = self.check_spatial_and_id_duplicates(filename)
            self.process_pdb(filename, pdb_lines=pdb_lines)

        def get_param(self, widget, default):
            try:
                val = float(widget.text())
                return val
            except ValueError:
                return default


        def action_evaluate_beta(self):
            if not self.working_file_path: return

            # Chimera writes the evaluated structure back to disk.  Only allow
            # that for an explicitly generated working structure (normally
            # <source>_merged.pdb); never overwrite a structure loaded by the
            # user for read-only detection.
            if not self.working_file_is_mutable:
                self.text_area.append(
                    ">>> Skipped beta-sheet evaluation to preserve the original file."
                )
                return
            
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
                return None
            split_line_indices = set()
            segments_per_chain_id = {} # e.g., {'A': 1, 'B': 1}

            last_chain_id = None

            # 1. SCAN PASS: Detect Splits
            with open(file_path, 'r') as f:
                pdb_lines = f.readlines()

            for i, line in enumerate(pdb_lines):
                # Ignore HETATM records: standard PDBs often append them after
                # the polymer and they must not create false chain segments.
                if not line.startswith("ATOM"):
                    continue

                current_chain_id = line[20:22].strip()
                if current_chain_id == last_chain_id:
                    continue

                split_line_indices.add(i)
                segments_per_chain_id[current_chain_id] = (
                    segments_per_chain_id.get(current_chain_id, 0) + 1
                )
                last_chain_id = current_chain_id

            # 2. DECISION: Do we have any ID that appeared as multiple segments?
            has_duplicates = any(count > 1 for count in segments_per_chain_id.values())

            if has_duplicates:
                reply = QMessageBox.question(
                    self, 
                    "Broken Chains / Duplicates Detected", 
                    "Detected multiple segments using the same Chain ID.\n"
                    "(Based on non-contiguous chain records in file order.)\n\n"
                    "Rename all segments to unique IDs to proceed?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, 
                    QMessageBox.StandardButton.Yes
                )
                
                if reply == QMessageBox.StandardButton.Yes:
                    self.text_area.append(
                        ">>> Auto-corrected duplicate/broken chains in memory; "
                        "the original file was not changed."
                    )
                    return self.rewrite_split_chains_in_memory(
                        pdb_lines, split_line_indices
                    )

            return None


        def rewrite_split_chains_in_memory(self, pdb_lines, split_indices):
            """Return corrected PDB records without creating or changing a file."""
            # Counter starts at -1 so the first split (index 0 or later) bumps it to 0 (Chain A)
            current_segment_idx = -1
            current_label = "A"
            pdb_chain_labels = (
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
            )
            corrected_lines = []

            for i, line in enumerate(pdb_lines):
                if i in split_indices:
                    current_segment_idx += 1
                    if current_segment_idx >= len(pdb_chain_labels):
                        raise ValueError(
                            "The PDB contains more chain segments than the "
                            "single-character PDB format can represent."
                        )
                    current_label = pdb_chain_labels[current_segment_idx]

                if line.startswith(("ATOM", "HETATM")):
                    line_list = list(line)
                    if len(line_list) > 21:
                        line_list[20] = " "
                        line_list[21] = current_label
                    corrected_lines.append("".join(line_list))
                else:
                    corrected_lines.append(line)

            return corrected_lines


        def process_pdb(
            self,
            filename,
            old_sandwiches=None,
            rename_map=None,
            pdb_lines=None,
        ):
            Z_MIN = self.get_param(self.edit_z_min, 4.6)
            Z_MAX = self.get_param(self.edit_z_max, 5.0)
            XY_SHIFT_LIMIT = self.get_param(self.edit_xy_limit, 3.0)
            NEIGHBOR_SEARCH_COUNT = int(self.get_param(self.edit_neighbor_count, 6))
            
            try:
                if filename.lower().endswith(('.cif', '.mmcif')):
                    raw_chains, waters, ions = self.parse_cif_manual_logic(filename)
                else:
                    raw_chains, waters, ions = self.parse_pdb_manual_logic(
                        filename, lines=pdb_lines
                    )
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


        def parse_pdb_manual_logic(self, filepath, lines=None):
            chains = {}; waters = []; ions = []
            if lines is None:
                with open(filepath, 'r') as pdb_file:
                    lines = pdb_file.readlines()

            for line in lines:
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


    class AtomOutlineItem(ContextSafeScatterPlotItem):
        def paint(self):
            glDepthFunc(GL_LESS)
            super().paint()


    class AtomCoreItem(gl.GLScatterPlotItem):
        _programs_by_share_group = {}

        @staticmethod
        def getShaderProgram():
            """Render point sprites as lit spheres instead of flat discs."""
            from OpenGL import GL
            from OpenGL.GL import shaders

            context = QOpenGLContext.currentContext()
            if context is None:
                return gl.GLScatterPlotItem.getShaderProgram()

            share_group = context.shareGroup()
            cache_key = share_group if share_group is not None else context
            cached_program = AtomCoreItem._programs_by_share_group.get(
                cache_key
            )
            if cached_program is not None:
                return cached_program

            shader_format = context.format()
            use_core = (
                (context.isOpenGLES() and shader_format.version() >= (3, 0))
                or (
                    not context.isOpenGLES()
                    and shader_format.version() >= (3, 1)
                )
            )
            if context.isOpenGLES():
                version = "#version 300 es\n" if use_core else "#version 100\n"
            else:
                version = "#version 140\n" if use_core else "#version 120\n"

            if use_core:
                vertex_source = """
                    uniform float u_scale;
                    uniform mat4 u_modelview;
                    uniform mat4 u_mvp;
                    in vec4 a_position;
                    in vec4 a_color;
                    in float a_size;
                    out vec4 v_color;
                    void main() {
                        gl_Position = u_mvp * a_position;
                        v_color = a_color;
                        gl_PointSize = a_size;
                        if (u_scale != 0.0) {
                            vec4 cpos = u_modelview * a_position;
                            float px_size = length(cpos.xyz) * u_scale;
                            gl_PointSize /= px_size;
                        }
                    }
                """
                fragment_source = """
                    #ifdef GL_ES
                    precision mediump float;
                    #endif
                    in vec4 v_color;
                    out vec4 fragColor;
                    void main() {
                        vec2 xy = (gl_PointCoord - 0.5) * 2.0;
                        float radius_squared = dot(xy, xy);
                        if (radius_squared > 1.0) discard;
                        vec3 normal = normalize(vec3(
                            xy.x, -xy.y, sqrt(1.0 - radius_squared)
                        ));
                        vec3 key_direction = normalize(
                            vec3(-0.55, 0.65, 0.85)
                        );
                        vec3 fill_direction = normalize(
                            vec3(0.55, -0.35, 0.75)
                        );
                        float key = max(
                            dot(normal, key_direction), 0.0
                        );
                        float fill = max(
                            dot(normal, fill_direction), 0.0
                        );
                        float front = max(normal.z, 0.0);
                        float shade = 0.56 + 0.27 * key + 0.10 * fill
                            + 0.07 * front;
                        vec3 half_vector = normalize(
                            key_direction + vec3(0.0, 0.0, 1.0)
                        );
                        float highlight = 0.12 * pow(
                            max(dot(normal, half_vector), 0.0), 12.0
                        );
                        vec3 lit_color = v_color.rgb * shade
                            + vec3(1.0, 0.94, 0.82) * highlight;
                        fragColor = vec4(lit_color, v_color.a);
                    }
                """
            else:
                vertex_source = """
                    uniform float u_scale;
                    uniform mat4 u_modelview;
                    uniform mat4 u_mvp;
                    attribute vec4 a_position;
                    attribute vec4 a_color;
                    attribute float a_size;
                    varying vec4 v_color;
                    void main() {
                        gl_Position = u_mvp * a_position;
                        v_color = a_color;
                        gl_PointSize = a_size;
                        if (u_scale != 0.0) {
                            vec4 cpos = u_modelview * a_position;
                            float px_size = length(cpos.xyz) * u_scale;
                            gl_PointSize /= px_size;
                        }
                    }
                """
                fragment_source = """
                    #ifdef GL_ES
                    precision mediump float;
                    #endif
                    varying vec4 v_color;
                    void main() {
                        vec2 xy = (gl_PointCoord - 0.5) * 2.0;
                        float radius_squared = dot(xy, xy);
                        if (radius_squared > 1.0) discard;
                        vec3 normal = normalize(vec3(
                            xy.x, -xy.y, sqrt(1.0 - radius_squared)
                        ));
                        vec3 key_direction = normalize(
                            vec3(-0.55, 0.65, 0.85)
                        );
                        vec3 fill_direction = normalize(
                            vec3(0.55, -0.35, 0.75)
                        );
                        float key = max(
                            dot(normal, key_direction), 0.0
                        );
                        float fill = max(
                            dot(normal, fill_direction), 0.0
                        );
                        float front = max(normal.z, 0.0);
                        float shade = 0.56 + 0.27 * key + 0.10 * fill
                            + 0.07 * front;
                        vec3 half_vector = normalize(
                            key_direction + vec3(0.0, 0.0, 1.0)
                        );
                        float highlight = 0.12 * pow(
                            max(dot(normal, half_vector), 0.0), 12.0
                        );
                        vec3 lit_color = v_color.rgb * shade
                            + vec3(1.0, 0.94, 0.82) * highlight;
                        gl_FragColor = vec4(lit_color, v_color.a);
                    }
                """

            compiled_vertex = shaders.compileShader(
                [version, vertex_source], GL.GL_VERTEX_SHADER
            )
            compiled_fragment = shaders.compileShader(
                [version, fragment_source], GL.GL_FRAGMENT_SHADER
            )
            program = shaders.compileProgram(
                compiled_vertex, compiled_fragment
            )
            GL.glBindAttribLocation(program, 0, "a_position")
            GL.glBindAttribLocation(program, 1, "a_color")
            GL.glBindAttribLocation(program, 2, "a_size")
            GL.glLinkProgram(program)

            AtomCoreItem._programs_by_share_group[cache_key] = program
            try:
                cache_key.destroyed.connect(
                    lambda _object=None, key=cache_key:
                    AtomCoreItem._programs_by_share_group.pop(key, None)
                )
            except (AttributeError, TypeError):
                pass
            return program

        def paint(self):
            glDepthFunc(GL_LEQUAL)
            super().paint()
            glDepthFunc(GL_LESS)


    class GPUAcceleratedPreviewCanvas(gl.GLViewWidget):
        ELEMENT_COLORS = {
            # ChimeraX-inspired palette requested for the RF Diffusion preview.
            "C": (0.82, 0.70, 0.51, 1.0),
            "N": (0.12, 0.30, 0.88, 1.0),
            "O": (0.92, 0.05, 0.08, 1.0),
            "S": (0.96, 0.76, 0.08, 1.0),
        }
        FALLBACK_COLOR = (0.72, 0.72, 0.76, 1.0)
        SELECTION_COLOR = (0.00, 0.90, 0.95, 1.0)
        SELECTABLE_ELEMENTS = {"C", "N", "O", "S"}
        COVALENT_RADII = {
            "C": 0.76, "N": 0.71, "O": 0.66, "S": 1.05,
            "P": 1.07, "F": 0.57, "CL": 1.02, "BR": 1.20,
            "I": 1.39, "FE": 1.24, "ZN": 1.22, "MG": 1.41,
        }

        def __init__(self, parent=None):
            # Quaternion rotation has no +/-90 degree elevation clamp and
            # avoids the one-plane lock of the default Euler camera.
            super().__init__(parent, rotationMethod='quaternion')
            self.setBackgroundColor('#1e1e1e')
            self.setCameraPosition(
                distance=100,
                elevation=30,
                azimuth=45
            )
            self.setMouseTracking(True) # Required to detect hover without clicking
            
            self.max_atom_size = 28.0 
            self.max_stroke_size = 9.0
            
            self.current_atom_size = self.max_atom_size * 0.65
            self.current_stroke_size = self.max_stroke_size * 0.30
            
            self.balls = None
            self.outlines = None
            self.bonds = None
            self._custom_mouse_pos = None

            # Used to distinguish a click from a rotation drag.
            self._press_pos = None
            self._pressed_atom = None
            self._is_dragging = False
            
            self.atoms = []
            self.visible_atom_indices = []
            self.selectable_atoms_data = []
            self.default_colors = []
            self.atom_coords = []
            self.bond_pairs = []
            self.bond_positions = np.empty((0, 3), dtype=float)
            self.selected_indices = set()
            self.hide_layers = False
            self.hide_non_carbons = True
            self.pdb_widget = None

        def update_atom_radius(self, percentage):
            self.current_atom_size = self.max_atom_size * (percentage / 100.0)
            if self.balls is not None:
                self.balls.setData(size=self.current_atom_size)
            if self.outlines is not None:
                self.outlines.setData(size=self.current_atom_size + self.current_stroke_size)

        def update_stroke_size(self, percentage):
            self.current_stroke_size = self.max_stroke_size * (percentage / 100.0)
            if self.outlines is not None:
                self.outlines.setData(size=self.current_atom_size + self.current_stroke_size)

        def get_screen_pos(self, pt3d):
            from PyQt6.QtGui import QVector4D
            w, h = self.width(), self.height()
            if w == 0 or h == 0: 
                return None

            # Add the homogeneous coordinate (1.0) to make it a 4D vector
            vec = QVector4D(*pt3d, 1.0)
            view = self.viewMatrix()
            
            # Newer pyqtgraph versions require explicit region and viewport dimensions
            region = (0, 0, w, h)
            viewport = (0, 0, w, h)
            proj = self.projectionMatrix(region, viewport)
            
            # 4x4 Matrix * 4D Vector preserves the 'w' (clip) value
            clip = proj * view * vec
            if clip.w() == 0: 
                return None
                
            ndc_x = clip.x() / clip.w()
            ndc_y = clip.y() / clip.w()
            
            x = (ndc_x + 1.0) * w / 2.0
            y = (1.0 - ndc_y) * h / 2.0
            return x, y

        def get_hovered_atom(self, pos2d):
            if self.balls is None or not self.selectable_atoms_data: return None
            min_dist = 20 # Hitbox radius in pixels
            best_atom = None
            for atom in self.selectable_atoms_data:
                sp = self.get_screen_pos((atom['x'], atom['y'], atom['z']))
                if sp is None: continue
                dist = ((sp[0] - pos2d.x())**2 + (sp[1] - pos2d.y())**2)**0.5
                if dist < min_dist:
                    min_dist = dist
                    best_atom = atom
            return best_atom

        def mousePressEvent(self, ev):
            current_pos = ev.position().toPoint()
            self._custom_mouse_pos = current_pos

            if ev.button() == Qt.MouseButton.LeftButton:
                # Do not select yet. Wait for release so we can distinguish
                # a click from a rotation drag.
                self._press_pos = current_pos
                self._pressed_atom = self.get_hovered_atom(current_pos)
                self._is_dragging = False

            super().mousePressEvent(ev)

        def mouseMoveEvent(self, ev):
            current_pos = ev.position().toPoint()

            if ev.buttons() == Qt.MouseButton.NoButton:
                atom = self.get_hovered_atom(current_pos)

                if atom:
                    tooltip = (
                        f"Chain {atom.get('chain_id', '')} | "
                        f"Res {atom['res_seq']} {atom['res_name']} | "
                        f"Atom {atom['name'].strip()}"
                    )
                    QToolTip.showText(
                        ev.globalPosition().toPoint(),
                        tooltip,
                        self
                    )
                else:
                    QToolTip.hideText()

            elif ev.buttons() == Qt.MouseButton.LeftButton:
                if self._press_pos is None:
                    self._press_pos = current_pos

                total_movement = (
                    current_pos - self._press_pos
                ).manhattanLength()

                drag_threshold = QApplication.startDragDistance()

                if not self._is_dragging:
                    if total_movement < drag_threshold:
                        # Small hand movement still counts as a click.
                        return

                    self._is_dragging = True

                    # Include movement accumulated before reaching the
                    # drag threshold.
                    diff = current_pos - self._press_pos
                else:
                    diff = current_pos - self._custom_mouse_pos

                self.orbit(-diff.x(), diff.y())
                self._custom_mouse_pos = current_pos
                ev.accept()

            elif ev.buttons() == Qt.MouseButton.RightButton:
                diff = current_pos - self._custom_mouse_pos
                self.pan(
                    diff.x(),
                    diff.y(),
                    0,
                    relative='view'
                )
                self._custom_mouse_pos = current_pos
                ev.accept()

            else:
                super().mouseMoveEvent(ev)

        def mouseReleaseEvent(self, ev):
            should_select = (
                ev.button() == Qt.MouseButton.LeftButton
                and not self._is_dragging
                and self._pressed_atom is not None
            )

            clicked_atom = self._pressed_atom

            self._custom_mouse_pos = None
            self._press_pos = None
            self._pressed_atom = None
            self._is_dragging = False

            if should_select:
                self.on_atom_clicked(clicked_atom)

            super().mouseReleaseEvent(ev)

        @staticmethod
        def _normalized_element(atom):
            element = str(atom.get('element', '') or '').strip().upper()
            if element:
                return element
            return QuickLookStructureReader._guess_element(atom.get('name', ''))

        def _atom_color(self, atom):
            return self.ELEMENT_COLORS.get(
                self._normalized_element(atom),
                self.FALLBACK_COLOR,
            )

        def _selection_colors(self):
            colors = np.array(self.default_colors, copy=True)
            for rendered_index, atom_index in enumerate(
                self.visible_atom_indices
            ):
                if atom_index in self.selected_indices:
                    colors[rendered_index] = self.SELECTION_COLOR
            return colors

        def _bond_colors(self):
            colors = []
            for first_index, second_index in self.bond_pairs:
                colors.extend((
                    self.SELECTION_COLOR
                    if first_index in self.selected_indices
                    else self._atom_color(self.atoms[first_index]),
                    self.SELECTION_COLOR
                    if second_index in self.selected_indices
                    else self._atom_color(self.atoms[second_index]),
                ))
            return np.asarray(colors, dtype=float)

        def _refresh_selection_colors(self):
            colors = self._selection_colors()
            if self.balls is not None:
                self.balls.setData(color=colors)
            if self.bonds is not None and self.bond_pairs:
                self.bonds.setData(color=self._bond_colors())

        def _update_selection_count(self):
            parent = self.parent()
            while parent:
                if hasattr(parent, 'lbl_selection_count'):
                    parent.lbl_selection_count.setText(
                        f"Selected atoms: {len(self.selected_indices)}"
                    )
                    if hasattr(parent, 'schedule_project_state_save'):
                        parent.schedule_project_state_save()
                    break
                parent = parent.parent()

        def _middle_layer_chain_ids(self):
            sandwiches = (
                getattr(self.pdb_widget, 'final_sandwiches', [])
                if self.pdb_widget else []
            )
            if not sandwiches:
                return set()

            detected_layers = int(
                getattr(self.pdb_widget, 'detected_layers', 0) or 0
            )
            middle_index = (
                detected_layers // 2
                if detected_layers > 0
                else max(len(group) for group in sandwiches) // 2
            )
            middle_chains = {
                group[middle_index]
                for group in sandwiches
                if middle_index < len(group)
            }
            # A partially detected protofilament should still contribute its
            # own middle chain rather than disappearing from the preview.
            for group in sandwiches:
                if group and not any(chain in middle_chains for chain in group):
                    middle_chains.add(group[len(group) // 2])
            return middle_chains

        def set_hide_layers(self, hidden):
            hidden = bool(hidden)
            if hidden == self.hide_layers:
                return
            self.hide_layers = hidden
            self._render_structure(fit_view=False)

        def set_hide_non_carbons(self, hidden):
            hidden = bool(hidden)
            if hidden == self.hide_non_carbons:
                return
            self.hide_non_carbons = hidden
            self._render_structure(fit_view=False)

        def plot_structure(self, atoms):
            # Selection indices always refer to this complete heavy-atom list.
            # Visibility filters only change the rendered subset, so selecting
            # a visible atom can still select its hidden layer matches.
            self.atoms = [
                atom for atom in atoms
                if self._normalized_element(atom) not in {'H', 'D'}
            ]
            self.selected_indices.clear()
            self._render_structure(fit_view=True)

        def _render_structure(self, fit_view=False):
            self.clear()
            self.balls = None
            self.outlines = None
            self.bonds = None

            middle_chains = (
                self._middle_layer_chain_ids() if self.hide_layers else set()
            )
            if self.hide_layers and middle_chains:
                bond_atom_indices = [
                    index for index, atom in enumerate(self.atoms)
                    if atom.get('chain_id', '') in middle_chains
                ]
            else:
                bond_atom_indices = list(range(len(self.atoms)))

            visible_atom_indices = bond_atom_indices
            if self.hide_non_carbons:
                visible_atom_indices = [
                    index for index in visible_atom_indices
                    if self._normalized_element(self.atoms[index]) == 'C'
                ]
            self.visible_atom_indices = visible_atom_indices

            self.selectable_atoms_data = []
            self.default_colors = []
            self.atom_coords = []
            self.bond_pairs = []
            self.bond_positions = np.empty((0, 3), dtype=float)

            if not bond_atom_indices:
                self._update_selection_count()
                return

            visible_atoms = [
                self.atoms[index] for index in self.visible_atom_indices
            ]
            for atom in visible_atoms:
                self.atom_coords.append([atom['x'], atom['y'], atom['z']])
                self.default_colors.append(self._atom_color(atom))
                if self._normalized_element(atom) in self.SELECTABLE_ELEMENTS:
                    self.selectable_atoms_data.append(atom)

            # Reapply selections while rebuilding the scene. Visibility
            # toggles must never make an existing selection look unselected.
            colors_arr = self._selection_colors()

            # Bond topology is calculated from every heavy atom in the visible
            # layers. Hide Non-carbons suppresses only spheres; sticks through
            # nitrogen, oxygen, or sulfur remain so the backbone stays intact.
            bond_atoms = [self.atoms[index] for index in bond_atom_indices]
            bond_coords = np.asarray([
                [atom['x'], atom['y'], atom['z']] for atom in bond_atoms
            ], dtype=float)
            tree = cKDTree(bond_coords)
            bond_positions = []
            for first, second in sorted(tree.query_pairs(r=2.65)):
                distance = np.linalg.norm(
                    bond_coords[first] - bond_coords[second]
                )
                if distance < 0.35:
                    continue
                first_element = self._normalized_element(bond_atoms[first])
                second_element = self._normalized_element(bond_atoms[second])
                first_radius = self.COVALENT_RADII.get(first_element, 0.77)
                second_radius = self.COVALENT_RADII.get(second_element, 0.77)
                if distance > 1.22 * (first_radius + second_radius) + 0.12:
                    continue
                self.bond_pairs.append((
                    bond_atom_indices[first], bond_atom_indices[second]
                ))
                bond_positions.extend((
                    bond_coords[first], bond_coords[second]
                ))

            if bond_positions:
                self.bond_positions = np.asarray(bond_positions, dtype=float)
                self.bonds = ContextSafeLinePlotItem(
                    pos=self.bond_positions,
                    mode='lines',
                    color=self._bond_colors(),
                    width=6.0,
                    antialias=True,
                    glOptions='opaque',
                )
                self.addItem(self.bonds)

            # A subtle dark rim keeps neighboring balls legible on the black
            # background and echoes the depth separation in ChimeraX.
            if self.visible_atom_indices:
                coords = np.asarray(self.atom_coords, dtype=float)
                self.outlines = AtomOutlineItem(
                    pos=coords, color=(0.025, 0.025, 0.025, 1.0),
                    size=self.current_atom_size + self.current_stroke_size,
                    pxMode=True, glOptions='opaque'
                )
                self.addItem(self.outlines)

                self.balls = AtomCoreItem(
                    pos=coords, color=colors_arr,
                    size=self.current_atom_size,
                    pxMode=True, glOptions='opaque'
                )
                self.addItem(self.balls)
            
            # Loading a different structure gets an initial fit. Visibility
            # toggles rebuild only the scene items and preserve the user's
            # current pan, zoom, and rotation.
            if fit_view:
                centroid = np.mean(bond_coords, axis=0)
                self.opts['center'] = QVector3D(
                    centroid[0], centroid[1], centroid[2]
                )
                bounding_radius = float(np.max(
                    np.linalg.norm(bond_coords - centroid, axis=1)
                ))
                self.setCameraPosition(
                    distance=max(10.0, bounding_radius * 1.75)
                )
            self._update_selection_count()

        # Compatibility for older saved callers and plugin integrations.
        def plot_carbons(self, atoms):
            self.plot_structure(atoms)

        def restore_selected_atoms(self, saved_atoms):
            """Restore hotspot selections after the same PDB is reloaded."""
            def atom_key(atom):
                try:
                    return (
                        str(atom.get('chain_id', '')),
                        str(atom.get('res_seq', '')),
                        str(atom.get('name', '')).strip(),
                        round(float(atom.get('x', 0.0)), 3),
                        round(float(atom.get('y', 0.0)), 3),
                        round(float(atom.get('z', 0.0)), 3),
                    )
                except (TypeError, ValueError):
                    return None

            saved_keys = {
                key for key in (atom_key(atom) for atom in (saved_atoms or []))
                if key is not None
            }

            self.selected_indices = {
                i for i, atom in enumerate(self.atoms)
                if atom_key(atom) in saved_keys
            }

            self._refresh_selection_colors()
            self._update_selection_count()

        def on_atom_clicked(self, picked_atom):
            target_seq = picked_atom['res_seq']
            target_chain = picked_atom.get('chain_id', '')
            target_atom_name = picked_atom.get('name', '')
            
            # Check if 'Select as Layers' is enabled via parent container lookup
            select_layers = True
            p = self.parent()
            while p:
                if hasattr(p, 'chk_select_layers'):
                    select_layers = p.chk_select_layers.isChecked()
                    break
                p = p.parent()

            pf_sandwiches = getattr(self.pdb_widget, 'final_sandwiches', []) if self.pdb_widget else []
            
            if select_layers:
                target_pf_chains = {target_chain}
                for pf in pf_sandwiches:
                    if target_chain in pf:
                        target_pf_chains = set(pf)
                        break
                matching_indices = [
                    i for i, a in enumerate(self.atoms) 
                    if a['res_seq'] == target_seq 
                    and a.get('chain_id') in target_pf_chains
                    and a.get('name') == target_atom_name
                ]
            else:
                # Find the exact index of the clicked atom in self.atoms
                matching_indices = []
                for i, a in enumerate(self.atoms):
                    if a is picked_atom:
                        matching_indices = [i]
                        break
            
            was_selected = any(
                i in self.selected_indices for i in matching_indices
            )

            # Each exact atom name is an independent toggle. This permits a
            # residue to emit values such as "CE,CB" in select_hotspots.
            if was_selected:
                self.selected_indices.difference_update(matching_indices)
            else:
                self.selected_indices.update(matching_indices)

            self._refresh_selection_colors()
            self._update_selection_count()


    class SoftwarePreviewCanvas(SoftwareChainCanvas):
        """Interactive CPU atom/bond preview that avoids QOpenGLWidget."""

        ELEMENT_COLORS = GPUAcceleratedPreviewCanvas.ELEMENT_COLORS
        FALLBACK_COLOR = GPUAcceleratedPreviewCanvas.FALLBACK_COLOR
        SELECTION_COLOR = GPUAcceleratedPreviewCanvas.SELECTION_COLOR
        SELECTABLE_ELEMENTS = GPUAcceleratedPreviewCanvas.SELECTABLE_ELEMENTS
        COVALENT_RADII = GPUAcceleratedPreviewCanvas.COVALENT_RADII

        def __init__(self, parent=None):
            super().__init__(parent, width=8, height=8, dpi=75)
            self.setMouseTracking(True)
            self.max_atom_size = 28.0
            self.max_stroke_size = 9.0
            self.current_atom_size = self.max_atom_size * 0.65
            self.current_stroke_size = self.max_stroke_size * 0.30
            self.balls = None
            self.outlines = None
            self.bonds = None
            self._custom_mouse_pos = None
            self._press_pos = None
            self._pressed_atom = None
            self._is_dragging = False
            self.atoms = []
            self.visible_atom_indices = []
            self.selectable_atoms_data = []
            self.default_colors = []
            self.atom_coords = []
            self.bond_pairs = []
            self.bond_positions = np.empty((0, 3), dtype=float)
            self.selected_indices = set()
            self.hide_layers = False
            self.hide_non_carbons = True
            self.pdb_widget = None

        def update_atom_radius(self, percentage):
            self.current_atom_size = self.max_atom_size * (percentage / 100.0)
            self.update()

        def update_stroke_size(self, percentage):
            self.current_stroke_size = self.max_stroke_size * (percentage / 100.0)
            self.update()

        def get_screen_pos(self, point):
            return self._screen_position(point)

        def get_hovered_atom(self, position):
            if not self.selectable_atoms_data:
                return None
            closest_atom = None
            closest_distance = max(12.0, self.current_atom_size)
            for atom in self.selectable_atoms_data:
                screen_position = self.get_screen_pos((atom["x"], atom["y"], atom["z"]))
                if screen_position is None:
                    continue
                distance = math.hypot(
                    screen_position[0] - position.x(),
                    screen_position[1] - position.y(),
                )
                if distance < closest_distance:
                    closest_distance = distance
                    closest_atom = atom
            return closest_atom

        def mousePressEvent(self, event):
            current = event.position().toPoint()
            self._custom_mouse_pos = current
            if event.button() == Qt.MouseButton.LeftButton:
                self._press_pos = current
                self._pressed_atom = self.get_hovered_atom(current)
                self._is_dragging = False
            event.accept()

        def mouseMoveEvent(self, event):
            current = event.position().toPoint()
            if event.buttons() == Qt.MouseButton.NoButton:
                atom = self.get_hovered_atom(current)
                if atom:
                    QToolTip.showText(
                        event.globalPosition().toPoint(),
                        f"Chain {atom.get('chain_id', '')} | "
                        f"Res {atom.get('res_seq', '')} {atom.get('res_name', '')} | "
                        f"Atom {str(atom.get('name', '')).strip()}",
                        self,
                    )
                else:
                    QToolTip.hideText()
                event.accept()
                return

            if self._custom_mouse_pos is None:
                self._custom_mouse_pos = current

            if event.buttons() & Qt.MouseButton.LeftButton:
                if self._press_pos is None:
                    self._press_pos = current
                total_movement = (current - self._press_pos).manhattanLength()
                if not self._is_dragging:
                    if total_movement < QApplication.startDragDistance():
                        event.accept()
                        return
                    self._is_dragging = True
                    difference = current - self._press_pos
                else:
                    difference = current - self._custom_mouse_pos
                self.orbit(-difference.x(), difference.y())
            elif event.buttons() & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton):
                difference = current - self._custom_mouse_pos
                self.pan(difference.x(), difference.y())

            self._custom_mouse_pos = current
            event.accept()

        def mouseReleaseEvent(self, event):
            should_select = (
                event.button() == Qt.MouseButton.LeftButton
                and not self._is_dragging
                and self._pressed_atom is not None
            )
            clicked_atom = self._pressed_atom
            self._custom_mouse_pos = None
            self._press_pos = None
            self._pressed_atom = None
            self._is_dragging = False
            if should_select:
                self.on_atom_clicked(clicked_atom)
            event.accept()

        @staticmethod
        def _normalized_element(atom):
            element = str(atom.get("element", "") or "").strip().upper()
            if element:
                return element
            return QuickLookStructureReader._guess_element(atom.get("name", ""))

        def _atom_color(self, atom):
            return self.ELEMENT_COLORS.get(
                self._normalized_element(atom), self.FALLBACK_COLOR
            )

        def _selection_colors(self):
            colors = np.array(self.default_colors, copy=True)
            for rendered_index, atom_index in enumerate(self.visible_atom_indices):
                if atom_index in self.selected_indices:
                    colors[rendered_index] = self.SELECTION_COLOR
            return colors

        def _bond_colors(self):
            colors = []
            for first_index, second_index in self.bond_pairs:
                colors.extend((
                    self.SELECTION_COLOR
                    if first_index in self.selected_indices
                    else self._atom_color(self.atoms[first_index]),
                    self.SELECTION_COLOR
                    if second_index in self.selected_indices
                    else self._atom_color(self.atoms[second_index]),
                ))
            return np.asarray(colors, dtype=float)

        def _refresh_selection_colors(self):
            self.update()

        def _update_selection_count(self):
            parent = self.parent()
            while parent:
                if hasattr(parent, "lbl_selection_count"):
                    parent.lbl_selection_count.setText(
                        f"Selected atoms: {len(self.selected_indices)}"
                    )
                    if hasattr(parent, "schedule_project_state_save"):
                        parent.schedule_project_state_save()
                    break
                parent = parent.parent()

        def _middle_layer_chain_ids(self):
            sandwiches = (
                getattr(self.pdb_widget, "final_sandwiches", [])
                if self.pdb_widget else []
            )
            if not sandwiches:
                return set()
            detected_layers = int(
                getattr(self.pdb_widget, "detected_layers", 0) or 0
            )
            middle_index = (
                detected_layers // 2
                if detected_layers > 0
                else max(len(group) for group in sandwiches) // 2
            )
            middle_chains = {
                group[middle_index]
                for group in sandwiches
                if middle_index < len(group)
            }
            for group in sandwiches:
                if group and not any(chain in middle_chains for chain in group):
                    middle_chains.add(group[len(group) // 2])
            return middle_chains

        def set_hide_layers(self, hidden):
            hidden = bool(hidden)
            if hidden != self.hide_layers:
                self.hide_layers = hidden
                self._render_structure(fit_view=False)

        def set_hide_non_carbons(self, hidden):
            hidden = bool(hidden)
            if hidden != self.hide_non_carbons:
                self.hide_non_carbons = hidden
                self._render_structure(fit_view=False)

        def plot_structure(self, atoms):
            self.atoms = [
                atom for atom in atoms
                if self._normalized_element(atom) not in {"H", "D"}
            ]
            self.selected_indices.clear()
            self._render_structure(fit_view=True)

        def _render_structure(self, fit_view=False):
            self.balls = None
            self.outlines = None
            self.bonds = None
            middle_chains = self._middle_layer_chain_ids() if self.hide_layers else set()
            if self.hide_layers and middle_chains:
                bond_atom_indices = [
                    index for index, atom in enumerate(self.atoms)
                    if atom.get("chain_id", "") in middle_chains
                ]
            else:
                bond_atom_indices = list(range(len(self.atoms)))

            self.visible_atom_indices = bond_atom_indices
            if self.hide_non_carbons:
                self.visible_atom_indices = [
                    index for index in self.visible_atom_indices
                    if self._normalized_element(self.atoms[index]) == "C"
                ]

            self.selectable_atoms_data = []
            self.default_colors = []
            self.atom_coords = []
            self.bond_pairs = []
            self.bond_positions = np.empty((0, 3), dtype=float)

            if not bond_atom_indices:
                self.update()
                self._update_selection_count()
                return

            for index in self.visible_atom_indices:
                atom = self.atoms[index]
                self.atom_coords.append([atom["x"], atom["y"], atom["z"]])
                self.default_colors.append(self._atom_color(atom))
                if self._normalized_element(atom) in self.SELECTABLE_ELEMENTS:
                    self.selectable_atoms_data.append(atom)

            bond_atoms = [self.atoms[index] for index in bond_atom_indices]
            bond_coords = np.asarray([
                [atom["x"], atom["y"], atom["z"]] for atom in bond_atoms
            ], dtype=float)
            tree = cKDTree(bond_coords)
            bond_positions = []
            for first, second in sorted(tree.query_pairs(r=2.65)):
                distance = np.linalg.norm(bond_coords[first] - bond_coords[second])
                if distance < 0.35:
                    continue
                first_element = self._normalized_element(bond_atoms[first])
                second_element = self._normalized_element(bond_atoms[second])
                first_radius = self.COVALENT_RADII.get(first_element, 0.77)
                second_radius = self.COVALENT_RADII.get(second_element, 0.77)
                if distance > 1.22 * (first_radius + second_radius) + 0.12:
                    continue
                self.bond_pairs.append((
                    bond_atom_indices[first], bond_atom_indices[second]
                ))
                bond_positions.extend((bond_coords[first], bond_coords[second]))

            if bond_positions:
                self.bond_positions = np.asarray(bond_positions, dtype=float)
                self.bonds = True
            if self.visible_atom_indices:
                self.outlines = True
                self.balls = True

            if fit_view:
                self.global_center = bond_coords.mean(axis=0)
                distances = np.linalg.norm(bond_coords - self.global_center, axis=1)
                self.max_range = float(distances.max()) or 10.0
                self.zoom_level = 50
                self._pan[:] = 0
            self.update()
            self._update_selection_count()

        def plot_carbons(self, atoms):
            self.plot_structure(atoms)

        def restore_selected_atoms(self, saved_atoms):
            def atom_key(atom):
                try:
                    return (
                        str(atom.get("chain_id", "")),
                        str(atom.get("res_seq", "")),
                        str(atom.get("name", "")).strip(),
                        round(float(atom.get("x", 0.0)), 3),
                        round(float(atom.get("y", 0.0)), 3),
                        round(float(atom.get("z", 0.0)), 3),
                    )
                except (TypeError, ValueError):
                    return None

            saved_keys = {
                key for key in (atom_key(atom) for atom in (saved_atoms or []))
                if key is not None
            }
            self.selected_indices = {
                index for index, atom in enumerate(self.atoms)
                if atom_key(atom) in saved_keys
            }
            self._refresh_selection_colors()
            self._update_selection_count()

        def on_atom_clicked(self, picked_atom):
            target_sequence = picked_atom["res_seq"]
            target_chain = picked_atom.get("chain_id", "")
            target_atom_name = picked_atom.get("name", "")
            select_layers = True
            parent = self.parent()
            while parent:
                if hasattr(parent, "chk_select_layers"):
                    select_layers = parent.chk_select_layers.isChecked()
                    break
                parent = parent.parent()

            sandwiches = (
                getattr(self.pdb_widget, "final_sandwiches", [])
                if self.pdb_widget else []
            )
            if select_layers:
                target_chains = {target_chain}
                for sandwich in sandwiches:
                    if target_chain in sandwich:
                        target_chains = set(sandwich)
                        break
                matching_indices = [
                    index for index, atom in enumerate(self.atoms)
                    if atom["res_seq"] == target_sequence
                    and atom.get("chain_id") in target_chains
                    and atom.get("name") == target_atom_name
                ]
            else:
                matching_indices = [
                    index for index, atom in enumerate(self.atoms)
                    if atom is picked_atom
                ][:1]

            if any(index in self.selected_indices for index in matching_indices):
                self.selected_indices.difference_update(matching_indices)
            else:
                self.selected_indices.update(matching_indices)
            self._refresh_selection_colors()
            self._update_selection_count()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
            painter.fillRect(self.rect(), QColor("#1e1e1e"))
            if not self.atoms:
                painter.setPen(QColor("#8a8a8a"))
                painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No structure loaded")
                return

            if len(self.bond_positions):
                projected_bonds, bond_depths = self._project(self.bond_positions)
                bond_colors = self._bond_colors()
                segments = []
                for offset in range(0, len(projected_bonds), 2):
                    segments.append((
                        float((bond_depths[offset] + bond_depths[offset + 1]) / 2.0),
                        offset,
                    ))
                for _depth, offset in sorted(segments):
                    first = projected_bonds[offset]
                    second = projected_bonds[offset + 1]
                    midpoint = (first + second) / 2.0
                    painter.setPen(QPen(self._qcolor(bond_colors[offset]), 5.0))
                    painter.drawLine(QPointF(*first), QPointF(*midpoint))
                    painter.setPen(QPen(self._qcolor(bond_colors[offset + 1]), 5.0))
                    painter.drawLine(QPointF(*midpoint), QPointF(*second))

            if self.atom_coords:
                projected_atoms, atom_depths = self._project(self.atom_coords)
                colors = self._selection_colors()
                radius = max(2.0, self.current_atom_size / 2.0)
                outline_radius = radius + max(0.5, self.current_stroke_size / 2.0)
                for rendered_index in np.argsort(atom_depths):
                    point = projected_atoms[rendered_index]
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QBrush(QColor("#080808")))
                    painter.drawEllipse(QPointF(*point), outline_radius, outline_radius)
                    painter.setBrush(QBrush(self._qcolor(colors[rendered_index])))
                    painter.drawEllipse(QPointF(*point), radius, radius)


    PreviewCanvas = (
        SoftwarePreviewCanvas if use_software_renderer
        else GPUAcceleratedPreviewCanvas
    )


    class QuickLookStructureReader:
        """Small dependency-free PDB/mmCIF reader for score quick looks."""

        @staticmethod
        def _clean_cif_value(value):
            value = str(value or "").strip()
            return "" if value in {".", "?"} else value

        @staticmethod
        def _guess_element(atom_name):
            letters = "".join(
                character for character in str(atom_name).upper()
                if character.isalpha()
            )
            if not letters:
                return "C"
            if letters[:2] in {"CL", "BR", "FE", "ZN", "MG", "MN", "CU"}:
                return letters[:2]
            return letters[0]

        @classmethod
        def _iter_cif_rows(cls, filepath):
            import shlex

            with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()

            index = 0
            line_count = len(lines)
            relevant_prefixes = (
                "_atom_site.",
                "_struct_conf.",
                "_struct_sheet_range.",
            )

            while index < line_count:
                if lines[index].strip().lower() != "loop_":
                    index += 1
                    continue

                index += 1
                headers = []
                while index < line_count:
                    stripped = lines[index].strip()
                    if not stripped.startswith("_"):
                        break
                    headers.append(stripped.split()[0])
                    index += 1

                if not headers:
                    continue

                is_relevant = any(
                    headers[0].lower().startswith(prefix)
                    for prefix in relevant_prefixes
                )
                token_buffer = []
                field_count = len(headers)

                while index < line_count:
                    stripped = lines[index].strip()
                    lowered = stripped.lower()
                    if not stripped:
                        index += 1
                        continue
                    if stripped == "#":
                        index += 1
                        break
                    if (
                        lowered == "loop_"
                        or lowered.startswith("data_")
                        or lowered.startswith("save_")
                        or stripped.startswith("_")
                    ) and not token_buffer:
                        break

                    if is_relevant:
                        try:
                            token_buffer.extend(
                                shlex.split(stripped, comments=False, posix=True)
                            )
                        except ValueError:
                            token_buffer.extend(stripped.split())

                        while len(token_buffer) >= field_count:
                            row = token_buffer[:field_count]
                            del token_buffer[:field_count]
                            yield headers, row
                    index += 1

        @classmethod
        def _parse_cif(cls, filepath):
            atoms = []
            secondary_ranges = []
            first_model = None
            cached_headers = None
            header_map = {}

            for headers, row in cls._iter_cif_rows(filepath):
                if headers is not cached_headers:
                    cached_headers = headers
                    header_map = {
                        header.lower(): position
                        for position, header in enumerate(headers)
                    }

                def field(*names):
                    for name in names:
                        position = header_map.get(name.lower())
                        if position is not None and position < len(row):
                            return cls._clean_cif_value(row[position])
                    return ""

                first_header = headers[0].lower()
                if first_header.startswith("_atom_site."):
                    model_number = field("_atom_site.pdbx_PDB_model_num") or "1"
                    if first_model is None:
                        first_model = model_number
                    if model_number != first_model:
                        continue

                    alt_loc = field(
                        "_atom_site.label_alt_id",
                        "_atom_site.pdbx_PDB_alt_id",
                    )
                    if alt_loc not in {"", "A", "1"}:
                        continue

                    try:
                        x = float(field("_atom_site.Cartn_x"))
                        y = float(field("_atom_site.Cartn_y"))
                        z = float(field("_atom_site.Cartn_z"))
                    except (TypeError, ValueError):
                        continue

                    atom_name = field(
                        "_atom_site.auth_atom_id",
                        "_atom_site.label_atom_id",
                    )
                    element = field("_atom_site.type_symbol").upper()
                    atoms.append({
                        "name": atom_name,
                        "element": element or cls._guess_element(atom_name),
                        "record_type": (
                            field("_atom_site.group_PDB").upper() or "ATOM"
                        ),
                        "res_name": field(
                            "_atom_site.auth_comp_id",
                            "_atom_site.label_comp_id",
                        ).upper(),
                        "res_seq": field(
                            "_atom_site.auth_seq_id",
                            "_atom_site.label_seq_id",
                        ),
                        "chain_id": field(
                            "_atom_site.auth_asym_id",
                            "_atom_site.label_asym_id",
                        ),
                        "x": x,
                        "y": y,
                        "z": z,
                        "secondary": "coil",
                    })

                elif first_header.startswith("_struct_conf."):
                    conf_type = field("_struct_conf.conf_type_id").upper()
                    if "HELX" in conf_type or "HELIX" in conf_type:
                        secondary_ranges.append((
                            "helix",
                            field(
                                "_struct_conf.beg_auth_asym_id",
                                "_struct_conf.beg_label_asym_id",
                            ),
                            field(
                                "_struct_conf.beg_auth_seq_id",
                                "_struct_conf.beg_label_seq_id",
                            ),
                            field(
                                "_struct_conf.end_auth_seq_id",
                                "_struct_conf.end_label_seq_id",
                            ),
                        ))

                elif first_header.startswith("_struct_sheet_range."):
                    secondary_ranges.append((
                        "sheet",
                        field(
                            "_struct_sheet_range.beg_auth_asym_id",
                            "_struct_sheet_range.beg_label_asym_id",
                        ),
                        field(
                            "_struct_sheet_range.beg_auth_seq_id",
                            "_struct_sheet_range.beg_label_seq_id",
                        ),
                        field(
                            "_struct_sheet_range.end_auth_seq_id",
                            "_struct_sheet_range.end_label_seq_id",
                        ),
                    ))

            cls._assign_secondary(atoms, secondary_ranges)
            return atoms

        @classmethod
        def _parse_pdb(cls, filepath):
            atoms = []
            secondary_ranges = []
            active_model = True
            saw_model = False

            with open(filepath, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    record = line[:6].strip().upper()
                    if record == "MODEL":
                        if saw_model:
                            active_model = False
                        else:
                            saw_model = True
                            active_model = True
                        continue
                    if record == "ENDMDL":
                        if saw_model:
                            active_model = False
                        continue
                    if record == "HELIX":
                        secondary_ranges.append((
                            "helix", line[19:20].strip(),
                            line[21:25].strip(), line[33:37].strip(),
                        ))
                        continue
                    if record == "SHEET":
                        secondary_ranges.append((
                            "sheet", line[21:22].strip(),
                            line[22:26].strip(), line[33:37].strip(),
                        ))
                        continue
                    if record not in {"ATOM", "HETATM"} or not active_model:
                        continue
                    if len(line) < 54:
                        continue
                    alt_loc = line[16:17]
                    if alt_loc not in {" ", "A", "1"}:
                        continue
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                    except ValueError:
                        continue

                    atom_name = line[12:16].strip()
                    element = line[76:78].strip().upper() if len(line) >= 78 else ""
                    atoms.append({
                        "name": atom_name,
                        "element": element or cls._guess_element(atom_name),
                        "record_type": record,
                        "res_name": line[17:20].strip().upper(),
                        "res_seq": (line[22:26] + line[26:27]).strip(),
                        "chain_id": line[21:22].strip(),
                        "x": x,
                        "y": y,
                        "z": z,
                        "secondary": "coil",
                    })

            cls._assign_secondary(atoms, secondary_ranges)
            return atoms

        @staticmethod
        def _residue_number(value):
            import re

            match = re.search(r"-?\d+", str(value))
            return int(match.group(0)) if match else None

        @classmethod
        def _assign_secondary(cls, atoms, ranges):
            for secondary_type, chain_id, start_value, end_value in ranges:
                start = cls._residue_number(start_value)
                end = cls._residue_number(end_value)
                if start is None or end is None:
                    continue
                lower, upper = sorted((start, end))
                for atom in atoms:
                    residue = cls._residue_number(atom.get("res_seq", ""))
                    if (
                        atom.get("chain_id", "") == chain_id
                        and residue is not None
                        and lower <= residue <= upper
                    ):
                        atom["secondary"] = secondary_type

            # RF3 files do not always contain DSSP records. Give the quick-look
            # ribbon a conservative CA-geometry fallback in that case.
            if ranges or not atoms:
                return

            chains = {}
            for atom in atoms:
                if atom.get("name", "").strip().upper() == "CA":
                    chains.setdefault(atom.get("chain_id", ""), []).append(atom)

            for chain_atoms in chains.values():
                if len(chain_atoms) < 4:
                    continue
                coordinates = np.array([
                    [atom["x"], atom["y"], atom["z"]]
                    for atom in chain_atoms
                ])
                assignments = ["coil"] * len(chain_atoms)

                for position in range(len(chain_atoms) - 4):
                    distance_i3 = np.linalg.norm(
                        coordinates[position] - coordinates[position + 3]
                    )
                    distance_i4 = np.linalg.norm(
                        coordinates[position] - coordinates[position + 4]
                    )
                    if 4.4 <= distance_i3 <= 6.4 and 5.0 <= distance_i4 <= 7.3:
                        for offset in range(5):
                            assignments[position + offset] = "helix"

                for position in range(len(chain_atoms) - 3):
                    if any(
                        assignments[position + offset] == "helix"
                        for offset in range(4)
                    ):
                        continue
                    distance_i2 = np.linalg.norm(
                        coordinates[position] - coordinates[position + 2]
                    )
                    distance_i3 = np.linalg.norm(
                        coordinates[position] - coordinates[position + 3]
                    )
                    if 5.8 <= distance_i2 <= 7.4 and 8.0 <= distance_i3 <= 11.2:
                        for offset in range(4):
                            if assignments[position + offset] == "coil":
                                assignments[position + offset] = "sheet"

                secondary_by_residue = {
                    (atom.get("chain_id", ""), str(atom.get("res_seq", ""))): assignment
                    for atom, assignment in zip(chain_atoms, assignments)
                }
                for atom in atoms:
                    key = (
                        atom.get("chain_id", ""),
                        str(atom.get("res_seq", "")),
                    )
                    if key in secondary_by_residue:
                        atom["secondary"] = secondary_by_residue[key]

        @classmethod
        def read(cls, filepath):
            if filepath.lower().endswith(('.cif', '.mmcif')):
                return cls._parse_cif(filepath)
            return cls._parse_pdb(filepath)

        @staticmethod
        def map_selected_hotspots(atoms, selected_coordinates, tolerance=0.01):
            """Map selected coordinates to exact atom names in a structure."""
            hotspot_atom_names = {}
            for atom in atoms:
                atom_name = str(atom.get('name', '')).strip()
                for x, y, z, selected_name in selected_coordinates:
                    if (
                        selected_name == atom_name
                        and math.isclose(atom['x'], x, abs_tol=tolerance)
                        and math.isclose(atom['y'], y, abs_tol=tolerance)
                        and math.isclose(atom['z'], z, abs_tol=tolerance)
                    ):
                        hotspot_key = (
                            f"{atom.get('chain_id', '')}"
                            f"{atom.get('res_seq', '')}"
                        )
                        names = hotspot_atom_names.setdefault(hotspot_key, [])
                        if atom_name not in names:
                            names.append(atom_name)
                        break
            return {
                key: ",".join(atom_names)
                for key, atom_names in hotspot_atom_names.items()
            }

        @classmethod
        def build_protein_contig(cls, atoms):
            """Build RFD3 target contigs using the structure's real chains."""
            protein_atoms = [
                atom for atom in atoms
                if atom.get('record_type', 'ATOM').upper() == 'ATOM'
            ]
            residue_atoms = [
                atom for atom in protein_atoms
                if str(atom.get('name', '')).strip().upper() == 'CA'
            ] or protein_atoms

            residues_by_chain = {}
            for atom in residue_atoms:
                residue_number = cls._residue_number(atom.get('res_seq', ''))
                if residue_number is None:
                    continue
                chain_id = str(atom.get('chain_id', '') or 'A')
                residues_by_chain.setdefault(chain_id, set()).add(
                    residue_number
                )

            contig_blocks = []
            for chain_id, residue_numbers in residues_by_chain.items():
                sorted_sequences = sorted(residue_numbers)
                if not sorted_sequences:
                    continue
                start = previous = sorted_sequences[0]
                for sequence in sorted_sequences[1:]:
                    if sequence != previous + 1:
                        contig_blocks.append(
                            f"{chain_id}{start}-{previous}"
                        )
                        start = sequence
                    previous = sequence
                contig_blocks.append(f"{chain_id}{start}-{previous}")

            return ",".join(contig_blocks) if contig_blocks else "A1-1"


    class QuickLookCanvas(ChainCanvasBase):
        """Layer Viewer-style CA traces for RF Diffusion score quick looks."""

        DEFAULT_CHAIN_STYLE = ((1.0, 1.0, 1.0, 0.6), 1.5)
        CHAIN_STYLES = {
            "A": DEFAULT_CHAIN_STYLE,
            "B": ((0.596, 0.765, 0.475, 0.9), 3.0),
        }

        def __init__(self, parent=None):
            super().__init__(
                parent,
                width=8,
                height=5,
                dpi=50,
                highlight_selected=False,
                center_on_labels=False,
                centered_labels=False,
            )
            self.setMouseTracking(True)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

            self.atoms = []
            self.chain_data = {}
            self.hover_atoms = []
            self._last_mouse_pos = None

            self.hover_label = QLabel(self)
            self.hover_label.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents
            )
            self.hover_label.setStyleSheet(
                "QLabel { background: rgba(20, 23, 28, 235); "
                "color: #f1f3f5; border: 1px solid #6f7782; "
                "border-radius: 4px; padding: 4px 7px; font-weight: bold; }"
            )
            self.hover_label.hide()

        def set_structure(self, atoms):
            self.atoms = [
                atom for atom in atoms
                if atom.get("record_type", "ATOM").upper() == "ATOM"
            ]
            self.atoms = self.atoms or list(atoms)
            if not self.atoms:
                raise ValueError("The structure contains no displayable atoms.")

            residues = {}
            for atom in self.atoms:
                key = (
                    atom.get("chain_id", ""),
                    str(atom.get("res_seq", "")),
                )
                residues.setdefault(key, atom)
                if atom.get("name", "").strip().upper() == "CA":
                    residues[key] = atom
            self.hover_atoms = list(residues.values())

            self.chain_data = {}
            for atom in self.hover_atoms:
                chain_id = str(atom.get("chain_id", "") or "A")
                self.chain_data.setdefault(chain_id, []).append((
                    atom["x"], atom["y"], atom["z"],
                ))

            chain_styles = {
                chain_id: self.CHAIN_STYLES.get(
                    chain_id.upper(), self.DEFAULT_CHAIN_STYLE
                )
                for chain_id in self.chain_data
            }
            self.zoom_level = 50
            self.plot_chains(
                self.chain_data,
                chain_styles=chain_styles,
            )

        def fit_structure(self):
            if use_software_renderer:
                super().fit_structure()
            else:
                self.zoom_level = 50
                self.apply_zoom()

        def get_screen_pos(self, point):
            return self._screen_position(point)

        def update_hover_label(self, position):
            closest = None
            closest_distance = 18.0
            for atom in self.hover_atoms:
                screen_position = self.get_screen_pos(
                    (atom["x"], atom["y"], atom["z"])
                )
                if screen_position is None:
                    continue
                distance = math.hypot(
                    screen_position[0] - position.x(),
                    screen_position[1] - position.y(),
                )
                if distance < closest_distance:
                    closest_distance = distance
                    closest = atom

            if closest is None:
                self.hover_label.hide()
                return

            residue_name = closest.get("res_name", "UNK") or "UNK"
            residue_number = closest.get("res_seq", "?") or "?"
            chain = closest.get("chain_id", "")
            chain_text = f"  ·  Chain {chain}" if chain else ""
            self.hover_label.setText(
                f"{residue_name} {residue_number}{chain_text}"
            )
            self.hover_label.adjustSize()
            label_x = min(
                position.x() + 14,
                max(4, self.width() - self.hover_label.width() - 5),
            )
            label_y = min(
                position.y() + 14,
                max(4, self.height() - self.hover_label.height() - 5),
            )
            self.hover_label.move(max(4, label_x), max(4, label_y))
            self.hover_label.show()
            self.hover_label.raise_()

        def mousePressEvent(self, event):
            self.setFocus()
            self._last_mouse_pos = event.position().toPoint()
            self.hover_label.hide()
            event.accept()

        def mouseMoveEvent(self, event):
            current_position = event.position().toPoint()
            buttons = event.buttons()
            if buttons == Qt.MouseButton.NoButton:
                self.update_hover_label(current_position)
                event.accept()
                return

            if self._last_mouse_pos is None:
                self._last_mouse_pos = current_position
            difference = current_position - self._last_mouse_pos
            if buttons & Qt.MouseButton.LeftButton:
                self.orbit(-difference.x(), difference.y())
            elif buttons & (
                Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton
            ):
                self.pan(
                    difference.x(), difference.y(), 0, relative='view'
                )
            self._last_mouse_pos = current_position
            event.accept()

        def mouseReleaseEvent(self, event):
            self._last_mouse_pos = None
            event.accept()

        def leaveEvent(self, event):
            self.hover_label.hide()
            super().leaveEvent(event)


    class ScoreQuickLookWindow(QWidget):
        def __init__(self, design_name, structure_path, atoms, parent=None):
            super().__init__(parent, Qt.WindowType.Window)
            self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
            self.setWindowTitle(f"{design_name} — RF Diffusion Quick Look")
            self.resize(1040, 760)
            self.setMinimumSize(660, 480)

            layout = QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.setSpacing(6)

            toolbar = QHBoxLayout()
            toolbar.setSpacing(6)
            title = QLabel(f"<b>{design_name}</b>")
            title.setToolTip(structure_path)
            toolbar.addWidget(title)
            toolbar.addSpacing(14)

            toolbar.addStretch()

            fit_button = QPushButton("Fit")
            fit_button.setToolTip("Center the complete structure in the window.")
            toolbar.addWidget(fit_button)
            layout.addLayout(toolbar)

            self.canvas = QuickLookCanvas(self)
            self.canvas.set_structure(atoms)
            layout.addWidget(self.canvas, 1)

            instructions = QLabel(
                "Left-drag rotate   ·   Right/middle-drag pan   ·   "
                "Wheel zoom   ·   Hover a residue for its amino-acid label"
            )
            instructions.setStyleSheet("color: #9aa3ad; padding: 2px 4px;")
            layout.addWidget(instructions)

            fit_button.clicked.connect(self.canvas.fit_structure)


    class RFDiffusionTab(QWidget):
        def __init__(self, pdb_widget_instance, parent=None):
            super().__init__(parent)
            self.pdb_widget = pdb_widget_instance
            self._restoring_project_state = False
            self._current_project_dir = ""
            self.loaded_score_signature = None
            self.loaded_score_description = ""
            self.score_currency_warning = ""
            self.local_extraction_process = None
            self.local_extraction_output = []
            self.quick_view_windows = []
            self.init_ui()

            self.state_save_timer = QTimer(self)
            self.state_save_timer.setSingleShot(True)
            self.state_save_timer.setInterval(250)
            self.state_save_timer.timeout.connect(self.save_project_state)

            self.criteria_refresh_timer = QTimer(self)
            self.criteria_refresh_timer.setSingleShot(True)
            self.criteria_refresh_timer.setInterval(100)
            self.criteria_refresh_timer.timeout.connect(
                self.apply_score_criteria
            )

            self.connect_project_state_signals()

            # Refresh SLURM states while this window remains open.
            self.job_status_timer = QTimer(self)
            self.job_status_timer.setInterval(15000)
            self.job_status_timer.timeout.connect(self.refresh_job_statuses)
            self.job_status_timer.start()

            app = QApplication.instance()
            if app is not None:
                app.aboutToQuit.connect(self.save_project_state)

        def init_ui(self):
            main_layout = QHBoxLayout(self)
            main_layout.setContentsMargins(10, 10, 10, 10)

            from PyQt6.QtWidgets import QScrollArea, QDoubleSpinBox
            splitter = QSplitter(Qt.Orientation.Horizontal)
            
            left_scroll = QScrollArea()
            left_scroll.setWidgetResizable(True)
            left_scroll.setMinimumWidth(520)
            
            left_widget = QWidget()
            left_layout = QVBoxLayout(left_widget)
            left_layout.setSpacing(15)
            left_layout.setContentsMargins(10, 10, 20, 10)

            # Workspace
            project_group = QGroupBox("Project Workspace")
            project_layout = QHBoxLayout(project_group)
            self.project_dir_input = QLineEdit()
            self.project_dir_input.setPlaceholderText("Select project workspace folder...")
            self.project_dir_input.editingFinished.connect(self.on_project_dir_changed)
            browse_proj_btn = QPushButton("Browse")
            browse_proj_btn.clicked.connect(self.browse_project_folder)
            project_layout.addWidget(self.project_dir_input)
            project_layout.addWidget(browse_proj_btn)
            left_layout.addWidget(project_group)

            # Section 1
            rfd3_group = QGroupBox("Section 1: RFD3 Backbone Generation (Auto-JSON)")
            rfd3_main_layout = QVBoxLayout(rfd3_group)
            
            pdb_layout = QHBoxLayout()
            self.input_pdb_input = QLineEdit()
            self.input_pdb_input.setPlaceholderText("Select target PDB or CIF...")
            browse_pdb_btn = QPushButton("Browse Structure")
            browse_pdb_btn.clicked.connect(self.browse_and_load_pdb)
            pdb_layout.addWidget(self.input_pdb_input)
            pdb_layout.addWidget(browse_pdb_btn)
            rfd3_main_layout.addLayout(pdb_layout)
            
            self.generated_json_path = ""
            rfd3_form = QFormLayout()
            self.binder_length_input = QLineEdit()
            self.binder_length_input.setText("15-20")
            rfd3_form.addRow("Binder Length (AA):", self.binder_length_input)
            
            chk_layout = QHBoxLayout()
            self.non_loopy_chk = QCheckBox(
                "Favor structured/helical backbones"
            )
            self.non_loopy_chk.setChecked(True)
            self.non_loopy_chk.setToolTip(
                "Bias RFD3 toward backbones with more regular secondary "
                "structure, especially alpha helices, and fewer irregular "
                "loop/coil regions. This does not control peptide "
                "cyclization."
            )
            self.run_dssp_chk = QCheckBox("Run DSSP")
            self.run_dssp_chk.setChecked(False)
            chk_layout.addWidget(self.non_loopy_chk)
            chk_layout.addWidget(self.run_dssp_chk)
            rfd3_form.addRow("", chk_layout)

            # 1. Designs per SLURM Job (Maps to NUM_DESIGNS in notebook)
            self.num_designs_spin = QSpinBox()
            self.num_designs_spin.setRange(1, 100000)
            self.num_designs_spin.setValue(100)
            rfd3_form.addRow("Designs per SLURM Job:", self.num_designs_spin)
            
            # 2. Number of SLURM Jobs (Maps to the Array size in notebook)
            self.num_batches_spin = QSpinBox()
            self.num_batches_spin.setRange(1, 1000)
            self.num_batches_spin.setValue(20)
            rfd3_form.addRow("Number of SLURM Jobs:", self.num_batches_spin)

            self.btn_run_rfd3 = QPushButton("Run RFD3")
            self.btn_run_rfd3.clicked.connect(self.run_rfd3)
            rfd3_form.addRow("", self.btn_run_rfd3)
            rfd3_main_layout.addLayout(rfd3_form)
            left_layout.addWidget(rfd3_group)

            # Section 2
            mpnn_group = QGroupBox("Section 2: MPNN Sequence Design")
            mpnn_layout = QFormLayout(mpnn_group)
            self.seqs_per_struct_spin = QSpinBox()
            self.seqs_per_struct_spin.setRange(1, 1000)
            self.seqs_per_struct_spin.setValue(4)
            mpnn_layout.addRow("Sequences per Backbone:", self.seqs_per_struct_spin)
            
            self.btn_run_mpnn = QPushButton("Run MPNN")
            self.btn_run_mpnn.clicked.connect(self.run_mpnn)
            mpnn_layout.addRow("", self.btn_run_mpnn)
            left_layout.addWidget(mpnn_group)

            # Section 3
            rf3_group = QGroupBox("Section 3: RF3 Structure Validation")
            rf3_layout = QVBoxLayout(rf3_group)
            topology_label = QLabel("Binder topology to validate:")
            topology_layout = QHBoxLayout()
            self.rf3_linear_chk = QCheckBox("Linear")
            self.rf3_linear_chk.setChecked(True)
            self.rf3_linear_chk.setToolTip(
                "Predict the binder with free N- and C-termini."
            )
            self.rf3_cyclic_chk = QCheckBox("Head-to-tail cyclic")
            self.rf3_cyclic_chk.setChecked(False)
            self.rf3_cyclic_chk.setToolTip(
                "Predict binder chain B with a peptide bond connecting the "
                "last residue to the first. Select both topologies to run "
                "and compare both predictions."
            )
            topology_layout.addWidget(self.rf3_linear_chk)
            topology_layout.addWidget(self.rf3_cyclic_chk)
            topology_layout.addStretch()
            rf3_layout.addWidget(topology_label)
            rf3_layout.addLayout(topology_layout)
            self.btn_run_rf3 = QPushButton("Run RF3")
            self.btn_run_rf3.clicked.connect(self.run_rf3)
            rf3_layout.addWidget(self.btn_run_rf3)
            left_layout.addWidget(rf3_group)

            # Section 4
            val_group = QGroupBox("Section 4: Data Consolidation & Extraction")
            val_layout = QFormLayout(val_group)

            self.btn_submit_val = QPushButton("Run CPU Validation Array")
            self.btn_submit_val.setObjectName("submitValidationButton")
            self.btn_submit_val.clicked.connect(self.run_validation_slurm)
            self.btn_submit_val.setToolTip(
                "Generate the raw score CSV files used by the thresholds below."
            )
            val_layout.addRow(self.btn_submit_val)
            
            self.rmsd_cutoff_spin = QDoubleSpinBox()
            self.rmsd_cutoff_spin.setRange(0.0, 50.0)
            self.rmsd_cutoff_spin.setValue(5.0)
            self.rmsd_cutoff_spin.setSingleStep(0.5)
            self.rmsd_cutoff_spin.setToolTip(
                "Maximum binder-backbone RMSD after aligning the RF3 model "
                "to its original RFD3 backbone, measured in angstroms.\n"
                "Lower is better. A design passes when RMSD is less than or "
                "equal to this value. Decrease it to be stricter."
            )
            val_layout.addRow("Max RMSD (Å):", self.rmsd_cutoff_spin)

            self.ignore_cyclic_rmsd_chk = QCheckBox(
                "Ignore RMSD cutoff to cyclic predictions"
            )
            self.ignore_cyclic_rmsd_chk.setChecked(True)
            self.ignore_cyclic_rmsd_chk.setToolTip(
                "RFD3 generates the first-round reference as a linear "
                "backbone. A cyclic RF3 prediction may therefore rearrange "
                "when its termini are joined. When enabled, cyclic RMSD is "
                "still calculated and displayed, but it is not used to "
                "decide whether a first-round cyclic result passes "
                "extraction. Linear predictions and cyclic predictions "
                "iterated from a cyclic reference always use the RMSD "
                "cutoff."
            )
            val_layout.addRow("", self.ignore_cyclic_rmsd_chk)
            
            self.ptm_cutoff_spin = QDoubleSpinBox()
            self.ptm_cutoff_spin.setRange(0.0, 1.0)
            self.ptm_cutoff_spin.setValue(0.7)
            self.ptm_cutoff_spin.setSingleStep(0.05)
            self.ptm_cutoff_spin.setToolTip(
                "Minimum RF3 predicted TM-score confidence for the binder "
                "chain, ranging from 0 to 1.\n"
                "Higher is better. A design passes when Binder pTM is greater "
                "than or equal to this value. Increase it to be stricter."
            )
            val_layout.addRow("Min Binder pTM:", self.ptm_cutoff_spin)
            
            self.pae_cutoff_spin = QDoubleSpinBox()
            self.pae_cutoff_spin.setRange(0.0, 50.0)
            self.pae_cutoff_spin.setValue(5.0)
            self.pae_cutoff_spin.setSingleStep(0.5)
            self.pae_cutoff_spin.setToolTip(
                "PAE shows the interactions of the binder's amino acids to "
                "the target protein, measured as errors, so the less the "
                "better. Values are measured in angstroms.\n"
                "A design passes when PAE Min is less than or equal to this "
                "value. Decrease it to be stricter."
            )
            val_layout.addRow("Max PAE Min:", self.pae_cutoff_spin)

            for criteria_widget in (
                self.rmsd_cutoff_spin,
                self.ptm_cutoff_spin,
                self.pae_cutoff_spin,
            ):
                criteria_label = val_layout.labelForField(criteria_widget)
                if criteria_label is not None:
                    criteria_label.setToolTip(criteria_widget.toolTip())
            
            post_validation_buttons_layout = QVBoxLayout()
            post_validation_buttons_layout.setSpacing(7)
            self.btn_run_consolidation = QPushButton("Run Local Extraction")
            self.btn_run_consolidation.setObjectName("runExtractionButton")
            self.btn_run_consolidation.clicked.connect(self.run_consolidation_local)
            self.btn_iterate_designs = QPushButton("Iterate designs (Optional)")
            self.btn_iterate_designs.setObjectName("iterateDesignsButton")
            self.btn_iterate_designs.setToolTip(
                "Create and load a new workspace that uses the current "
                "workspace's passed designs as MPNN backbones."
            )
            self.btn_iterate_designs.clicked.connect(self.iterate_designs)

            post_validation_buttons_layout.addWidget(
                self.btn_run_consolidation
            )
            post_validation_buttons_layout.addWidget(
                self.btn_iterate_designs
            )
            val_layout.addRow(post_validation_buttons_layout)

            left_layout.addWidget(val_group)

            # Separate automation cell below Section 4.
            automation_group = QGroupBox()
            automation_layout = QVBoxLayout(automation_group)
            automation_layout.setSpacing(7)
            self.btn_auto_run = QPushButton("Auto Run")
            self.btn_auto_run.setObjectName("autoRunButton")
            self.btn_auto_run.setToolTip(
                "Submit RFD3, MPNN, RF3, and CPU validation using the "
                "currently displayed parameters. Each stage starts only "
                "after the preceding SLURM array succeeds."
            )
            self.btn_auto_run.clicked.connect(self.auto_run_pipeline)
            self.btn_cancel_jobs = QPushButton("Cancel Running Jobs")
            self.btn_cancel_jobs.setObjectName("cancelJobsButton")
            self.btn_cancel_jobs.setToolTip(
                "Cancel all active or scheduled RFD3, MPNN, RF3, and CPU "
                "validation jobs tracked for the current workspace."
            )
            self.btn_cancel_jobs.clicked.connect(self.cancel_running_jobs)
            self.btn_cancel_jobs.setEnabled(False)

            automation_layout.addWidget(self.btn_auto_run)
            automation_layout.addWidget(self.btn_cancel_jobs)
            left_layout.addWidget(automation_group)
            left_layout.addStretch()
            left_scroll.setWidget(left_widget)
            
            # Right Panel
            right_widget = QWidget()
            right_layout = QVBoxLayout(right_widget)
            right_layout.setContentsMargins(0, 0, 0, 0)

            self.right_tabs = QTabWidget()
            pdb_preview_tab = QWidget()
            pdb_preview_layout = QVBoxLayout(pdb_preview_tab)
            pdb_preview_layout.setContentsMargins(0, 0, 0, 0)

            toolbar_layout = QVBoxLayout()
            toolbar_layout.setContentsMargins(10, 5, 10, 5)
            toolbar_layout.setSpacing(2)

            appearance_controls = QHBoxLayout()
            appearance_controls.setSpacing(6)

            selection_controls = QHBoxLayout()
            selection_controls.setSpacing(10)
            
            self.slider_size = QSlider(Qt.Orientation.Horizontal)
            self.slider_size.setRange(1, 100)
            self.slider_size.setValue(65)
            self.slider_size.setFixedWidth(120)
            self.slider_size.valueChanged.connect(self.on_slider_changed)
            
            self.slider_stroke = QSlider(Qt.Orientation.Horizontal)
            self.slider_stroke.setRange(0, 100)
            self.slider_stroke.setValue(30)
            self.slider_stroke.setFixedWidth(80)
            self.slider_stroke.valueChanged.connect(self.on_stroke_slider_changed)
            
            self.lbl_atom_val = QLabel("65%")
            self.lbl_stroke_val = QLabel("30%")

            self.chk_select_layers = QCheckBox("Select as Layers")
            self.chk_select_layers.setChecked(True)
            self.chk_select_layers.setToolTip(
                "When checked, clicking an atom selects the same exact atom "
                "name across all layers of the protofilament."
            )
            self.chk_select_layers.setMinimumWidth(
                self.chk_select_layers.sizeHint().width()
            )
            self.chk_select_layers.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Preferred,
            )

            self.chk_hide_layers = QCheckBox("Hide Layers")
            self.chk_hide_layers.setChecked(False)
            self.chk_hide_layers.setToolTip(
                "Show only the detected middle layer. Hidden layers remain "
                "available to Select as Layers and JSON generation."
            )
            self.chk_hide_layers.toggled.connect(
                self.on_hide_layers_changed
            )
            self.chk_hide_layers.setMinimumWidth(
                self.chk_hide_layers.sizeHint().width()
            )
            self.chk_hide_layers.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Preferred,
            )

            self.chk_hide_non_carbons = QCheckBox("Hide Non-carbons")
            self.chk_hide_non_carbons.setChecked(True)
            self.chk_hide_non_carbons.setToolTip(
                "Show only carbon atoms in the structure preview. Hidden "
                "atoms keep any existing hotspot selections."
            )
            self.chk_hide_non_carbons.toggled.connect(
                self.on_hide_non_carbons_changed
            )
            self.chk_hide_non_carbons.setMinimumWidth(
                self.chk_hide_non_carbons.sizeHint().width()
            )
            self.chk_hide_non_carbons.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Preferred,
            )

            self.lbl_selection_count = QLabel("Selected atoms: 0")
            self.lbl_selection_count.setStyleSheet(
                "color: #98c379; font-weight: bold;"
            )
            self.lbl_selection_count.setMinimumWidth(
                self.lbl_selection_count.fontMetrics().horizontalAdvance(
                    "Selected atoms: 99999"
                ) + 8
            )
            self.lbl_selection_count.setSizePolicy(
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Preferred,
            )

            appearance_controls.addWidget(QLabel("Atom Size:"))
            appearance_controls.addWidget(self.slider_size)
            appearance_controls.addWidget(self.lbl_atom_val)
            appearance_controls.addSpacing(10)
            appearance_controls.addWidget(QLabel("Stroke:"))
            appearance_controls.addWidget(self.slider_stroke)
            appearance_controls.addWidget(self.lbl_stroke_val)
            appearance_controls.addStretch()

            selection_controls.addWidget(self.chk_hide_layers)
            selection_controls.addWidget(self.chk_hide_non_carbons)
            selection_controls.addWidget(self.chk_select_layers)
            selection_controls.addStretch()
            selection_controls.addWidget(self.lbl_selection_count)

            toolbar_layout.addLayout(appearance_controls)
            toolbar_layout.addLayout(selection_controls)

            pdb_preview_layout.addLayout(toolbar_layout, 0)
            
            self.mol_canvas = PreviewCanvas(self)
            self.mol_canvas.pdb_widget = self.pdb_widget
            pdb_preview_layout.addWidget(self.mol_canvas, 1)
            self.right_tabs.addTab(pdb_preview_tab, "Structure Preview")

            scores_tab = QWidget()
            scores_layout = QVBoxLayout(scores_tab)
            scores_layout.setContentsMargins(8, 8, 8, 8)

            scores_header = QHBoxLayout()
            self.scores_status_label = QLabel(
                "Select a workspace to load score CSV files."
            )
            self.scores_status_label.setWordWrap(True)
            reload_scores_btn = QPushButton("Reload Scores")
            reload_scores_btn.clicked.connect(
                lambda: self.refresh_scores(force=True)
            )
            self.show_seeds_checkbox = QCheckBox("Show Seeds")
            self.show_seeds_checkbox.setChecked(False)
            self.show_seeds_checkbox.setToolTip(
                "Show RF3 seed/sample variants such as "
                "design_seed-0_sample-0. Hidden by default."
            )
            self.show_seeds_checkbox.toggled.connect(
                self.on_show_seeds_changed
            )
            scores_header.addWidget(self.scores_status_label, 1)
            scores_header.addWidget(self.show_seeds_checkbox)
            scores_header.addWidget(reload_scores_btn)
            scores_layout.addLayout(scores_header)

            self.score_model = ScoreTableModel(self)
            self.score_table = QTableView()
            self.score_table.setModel(self.score_model)
            self.score_table.setEditTriggers(
                QAbstractItemView.EditTrigger.NoEditTriggers
            )
            self.score_table.setSelectionBehavior(
                QAbstractItemView.SelectionBehavior.SelectRows
            )
            self.score_table.setAlternatingRowColors(True)
            self.score_table.setSortingEnabled(False)
            self.score_table.verticalHeader().setDefaultSectionSize(22)
            self.score_table.horizontalHeader().setDefaultSectionSize(145)
            self.score_table.horizontalHeader().setMinimumSectionSize(70)
            self.score_table.horizontalHeader().setStretchLastSection(True)
            self.score_table.doubleClicked.connect(
                self.open_score_quick_view
            )
            self.score_table.setToolTip(
                "Double-click a score row to open its RF3 model in a "
                "separate 3D quick-look window."
            )
            scores_layout.addWidget(self.score_table, 1)
            self.right_tabs.addTab(scores_tab, "Scores")

            right_layout.addWidget(self.right_tabs, 1)
            
            splitter.addWidget(left_scroll)
            splitter.addWidget(right_widget)
            splitter.setSizes([520, 930]) # Forces the initial pixel width
            splitter.setStretchFactor(0, 4) # Reduces the left panel's growth ratio
            splitter.setStretchFactor(1, 6) # Increases the right panel's growth ratio
            
            main_layout.addWidget(splitter)

        def connect_project_state_signals(self):
            for line_edit in (
                self.input_pdb_input,
                self.binder_length_input,
            ):
                line_edit.editingFinished.connect(
                    self.schedule_project_state_save
                )

            for checkbox in (
                self.non_loopy_chk,
                self.run_dssp_chk,
                self.rf3_linear_chk,
                self.rf3_cyclic_chk,
                self.ignore_cyclic_rmsd_chk,
                self.chk_hide_layers,
                self.chk_hide_non_carbons,
                self.chk_select_layers,
            ):
                checkbox.toggled.connect(
                    self.schedule_project_state_save
                )

            self.ignore_cyclic_rmsd_chk.toggled.connect(
                lambda _checked: self.criteria_refresh_timer.start()
            )
            self.rf3_linear_chk.toggled.connect(
                self.enforce_rf3_topology_selection
            )
            self.rf3_cyclic_chk.toggled.connect(
                self.enforce_rf3_topology_selection
            )

            for widget in (
                self.num_designs_spin,
                self.num_batches_spin,
                self.seqs_per_struct_spin,
                self.rmsd_cutoff_spin,
                self.ptm_cutoff_spin,
                self.pae_cutoff_spin,
                self.slider_size,
                self.slider_stroke,
            ):
                widget.valueChanged.connect(
                    self.schedule_project_state_save
                )

            for widget in (
                self.rmsd_cutoff_spin,
                self.ptm_cutoff_spin,
                self.pae_cutoff_spin,
            ):
                widget.valueChanged.connect(
                    lambda _value: self.criteria_refresh_timer.start()
                )

        def get_project_state_path(self, pwd=None):
            pwd = pwd or self._current_project_dir
            return (
                os.path.join(pwd, "scripts", "project_state.json")
                if pwd else ""
            )

        def schedule_project_state_save(self, *_args):
            if (
                self._restoring_project_state
                or not self._current_project_dir
                or not os.path.isdir(self._current_project_dir)
            ):
                return
            self.state_save_timer.start()

        def save_project_state(self):
            pwd = self._current_project_dir
            if (
                self._restoring_project_state
                or not pwd
                or not os.path.isdir(pwd)
            ):
                return

            try:
                import json, time

                scripts_dir = os.path.join(pwd, "scripts")
                os.makedirs(scripts_dir, exist_ok=True)
                state_path = self.get_project_state_path(pwd)
                temporary_path = state_path + ".tmp"
                state = {
                    "schema_version": 1,
                    "saved_at": time.time(),
                    "parameters": self.get_current_job_parameters(),
                }
                with open(temporary_path, "w") as handle:
                    json.dump(state, handle, indent=4)
                os.replace(temporary_path, state_path)
            except Exception as error:
                print(f"Failed to save project state: {error}")

        def clear_scores(self, message="No score CSV files found in this workspace."):
            self.loaded_score_signature = None
            self.loaded_score_description = ""
            self.score_model.clear()
            self.scores_status_label.setText(message)

        def find_score_csvs(self, pwd):
            scores_dir = os.path.join(pwd, "scores")
            if not os.path.isdir(scores_dir):
                return []

            csv_paths = [
                os.path.join(scores_dir, name)
                for name in os.listdir(scores_dir)
                if name.lower().endswith(".csv")
            ]
            batch_paths = sorted(
                path for path in csv_paths
                if os.path.basename(path).lower().startswith("scores_batch_")
            )
            master_paths = [
                path for path in csv_paths
                if os.path.basename(path).lower().startswith("master_scores")
            ]

            # One batch is authoritative on its own. For multiple batches,
            # prefer the consolidated master. Combining raw files is a useful
            # fallback while that master is still being generated.
            if len(batch_paths) == 1:
                return batch_paths
            if len(batch_paths) > 1 and master_paths:
                return [max(master_paths, key=os.path.getmtime)]
            if len(batch_paths) > 1:
                return batch_paths
            if master_paths:
                return [max(master_paths, key=os.path.getmtime)]
            if len(csv_paths) == 1:
                return csv_paths
            return sorted(csv_paths)

        def refresh_scores(self, force=False, activate=False):
            pwd = self._current_project_dir
            if not pwd or not os.path.isdir(pwd):
                self.score_currency_warning = ""
                self.clear_scores("Select a workspace to load score CSV files.")
                return

            validation_status = self.get_step_status(pwd, "validation")
            if validation_status and validation_status != "COMPLETED":
                self.score_currency_warning = (
                    "Scores may not be current; showing the existing CSV "
                    "results because the tracked CPU validation step is not "
                    "completed."
                )
            else:
                self.score_currency_warning = ""

            paths = self.find_score_csvs(pwd)
            if not paths:
                if self.score_currency_warning:
                    self.clear_scores(
                        f"{self.score_currency_warning} No score CSV files "
                        "are currently available."
                    )
                else:
                    self.clear_scores()
                return

            try:
                signature = tuple(
                    (path, os.path.getmtime(path), os.path.getsize(path))
                    for path in paths
                )
            except OSError as error:
                self.clear_scores(f"Unable to inspect score CSV files: {error}")
                return

            if not force and signature == self.loaded_score_signature:
                self.update_scores_status()
                return

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                self.score_model.load_csv_files(paths)
                self.loaded_score_signature = signature
                if len(paths) == 1:
                    self.loaded_score_description = os.path.basename(paths[0])
                else:
                    self.loaded_score_description = (
                        f"{len(paths)} batch CSV files (combined)"
                    )
                self.update_scores_status()
                if activate:
                    self.right_tabs.setCurrentIndex(1)
            except Exception as error:
                self.clear_scores(f"Failed to load score CSV: {error}")
            finally:
                QApplication.restoreOverrideCursor()

        def apply_score_criteria(self):
            self.score_model.set_criteria(
                self.rmsd_cutoff_spin.value(),
                self.ptm_cutoff_spin.value(),
                self.pae_cutoff_spin.value(),
                self.ignore_cyclic_rmsd_chk.isChecked(),
            )
            self.update_scores_status()

        def on_show_seeds_changed(self, checked):
            self.score_model.set_show_seeds(checked)
            self.update_scores_status()

        def update_scores_status(self):
            if not self.score_model.rows:
                return
            total_rows = len(self.score_model.rows)
            shown_rows = self.score_model.rowCount()
            row_summary = (
                f"{shown_rows:,} shown of {total_rows:,} rows"
                if shown_rows != total_rows
                else f"{total_rows:,} rows"
            )
            self.scores_status_label.setText(
                (
                    f"{self.score_currency_warning}  |  "
                    if self.score_currency_warning else ""
                )
                + f"{self.loaded_score_description}  |  "
                f"{row_summary}  |  "
                f"{self.score_model.match_count:,} meet all criteria  |  "
                "Double-click a row for 3D quick look"
            )

        @staticmethod
        def _score_sample_type(record):
            import re

            sample_type = str(record.get("sample_type", "") or "").strip()
            if sample_type:
                return sample_type
            subpath = str(record.get("rf3_subpath", "") or "").replace(
                "\\", "/"
            ).rstrip("/")
            design_name = str(record.get("design_name", "") or "")
            if subpath == design_name:
                return "Top-Level"
            if subpath:
                return subpath.rsplit("/", 1)[-1]
            seed_match = re.search(
                r"_(seed-\d+_sample-\d+)$", design_name, re.IGNORECASE
            )
            return seed_match.group(1) if seed_match else ""

        def find_score_structure(self, record):
            """Resolve a score row to the RF3 model that produced it."""
            import re

            workspace = self._current_project_dir
            design_name = str(record.get("design_name", "") or "").strip()
            batch = str(record.get("batch", "") or "").strip()
            if not workspace or not design_name:
                return "", []

            rf3_root = os.path.join(workspace, "rf3_outputs")
            batch_dir = os.path.join(rf3_root, batch) if batch else rf3_root
            folders = []
            raw_subpath = str(record.get("rf3_subpath", "") or "").replace(
                "\\", "/"
            )
            if raw_subpath:
                safe_parts = [
                    part for part in raw_subpath.split("/")
                    if part not in {"", ".", ".."}
                    and ":" not in part
                ]
                if safe_parts:
                    folders.append(os.path.join(batch_dir, *safe_parts))

            sample_type = self._score_sample_type(record)
            sample_type_lower = sample_type.lower()
            seed_match = re.fullmatch(
                r"seed-\d+_sample-\d+", sample_type, re.IGNORECASE
            )
            if sample_type_lower in {"top-level", "top_level", "top level"}:
                # RF3 commonly writes top-level models directly in the batch,
                # though older outputs used a design-named subdirectory.
                folders.append(batch_dir)
                folders.append(os.path.join(batch_dir, design_name))
            elif seed_match:
                # Current RF3 layout:
                #   batch/seed-N_sample-M/<full-design>_model.cif
                # Older score batches used:
                #   batch/<parent-design>/seed-N_sample-M/<full-design>_model.cif
                seed_folder = seed_match.group(0)
                folders.append(os.path.join(batch_dir, seed_folder))
                suffix = "_" + sample_type
                parent_design = (
                    design_name[:-len(suffix)]
                    if design_name.lower().endswith(suffix.lower())
                    else design_name
                )
                folders.append(
                    os.path.join(batch_dir, parent_design, seed_folder)
                )
            elif sample_type:
                suffix = "_" + sample_type
                parent_design = (
                    design_name[:-len(suffix)]
                    if design_name.lower().endswith(suffix.lower())
                    else design_name
                )
                folders.append(
                    os.path.join(batch_dir, parent_design, sample_type)
                )
            folders.append(os.path.join(batch_dir, design_name))
            folders.append(batch_dir)

            filenames = (
                design_name + "_model.cif",
                design_name + ".cif",
                design_name + "_model.pdb",
                design_name + ".pdb",
            )
            wanted_names = {name.lower() for name in filenames}
            checked_paths = []
            seen = set()
            for folder in folders:
                normalized_folder = os.path.normpath(folder)
                folder_key = os.path.normcase(normalized_folder)
                if folder_key in seen:
                    continue
                seen.add(folder_key)
                for filename in filenames:
                    candidate = os.path.join(normalized_folder, filename)
                    checked_paths.append(candidate)
                    if os.path.isfile(candidate):
                        return os.path.abspath(candidate), checked_paths

                # Preserve case-insensitive exact matching on case-sensitive
                # filesystems, but never substitute an unrelated model.
                if os.path.isdir(normalized_folder):
                    try:
                        for entry in os.scandir(normalized_folder):
                            if (
                                entry.is_file()
                                and entry.name.lower() in wanted_names
                            ):
                                return os.path.abspath(entry.path), checked_paths
                    except OSError:
                        pass

            # Older master score files omit rf3_subpath. Search the affected
            # batch only as a last resort and require an exact design filename.
            if os.path.isdir(batch_dir):
                try:
                    for root, _directories, files in os.walk(batch_dir):
                        for filename in files:
                            if filename.lower() in wanted_names:
                                return os.path.abspath(
                                    os.path.join(root, filename)
                                ), checked_paths
                except OSError:
                    pass

            successful_dir = os.path.join(workspace, "successful_designs")
            if os.path.isdir(successful_dir):
                try:
                    for entry in os.scandir(successful_dir):
                        if (
                            entry.is_file()
                            and design_name in entry.name
                            and entry.name.lower().endswith(
                                (".cif", ".mmcif", ".pdb")
                            )
                        ):
                            return os.path.abspath(entry.path), checked_paths
                except OSError:
                    pass

            return "", checked_paths

        def _forget_quick_view(self, window):
            try:
                self.quick_view_windows.remove(window)
            except ValueError:
                pass

        def open_score_quick_view(self, index):
            if not index.isValid():
                return
            record = self.score_model.record_for_view_row(index.row())
            design_name = str(record.get("design_name", "") or "").strip()
            structure_path, checked_paths = self.find_score_structure(record)
            if not structure_path:
                expected = checked_paths[0] if checked_paths else os.path.join(
                    self._current_project_dir or "<workspace>",
                    "rf3_outputs",
                )
                QMessageBox.warning(
                    self,
                    "Quick Look Model Not Found",
                    "The RF3 structure for this score row could not be found.\n\n"
                    f"Design: {design_name or '(unnamed)'}\n"
                    f"First expected path:\n{expected}",
                )
                return

            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                atoms = QuickLookStructureReader.read(structure_path)
                if not atoms:
                    raise ValueError("No atoms were found in the model file.")
                window = ScoreQuickLookWindow(
                    design_name or os.path.basename(structure_path),
                    structure_path,
                    atoms,
                    self,
                )
            except Exception as error:
                QMessageBox.critical(
                    self,
                    "Unable to Open 3D Quick Look",
                    f"Could not load:\n{structure_path}\n\n{error}",
                )
                return
            finally:
                QApplication.restoreOverrideCursor()

            self.quick_view_windows.append(window)
            window.destroyed.connect(
                lambda _object=None, target=window:
                self._forget_quick_view(target)
            )
            window.show()
            window.raise_()
            window.activateWindow()

        def on_slider_changed(self, value):
            if hasattr(self, 'lbl_atom_val'):
                self.lbl_atom_val.setText(f"{value}%")
            if hasattr(self, 'mol_canvas'):
                self.mol_canvas.update_atom_radius(value)

        def on_stroke_slider_changed(self, value):
            if hasattr(self, 'lbl_stroke_val'):
                self.lbl_stroke_val.setText(f"{value}%")
            if hasattr(self, 'mol_canvas'):
                self.mol_canvas.update_stroke_size(value)

        def on_hide_layers_changed(self, checked):
            if hasattr(self, 'mol_canvas'):
                self.mol_canvas.set_hide_layers(checked)

        def on_hide_non_carbons_changed(self, checked):
            if hasattr(self, 'mol_canvas'):
                self.mol_canvas.set_hide_non_carbons(checked)

        def browse_project_folder(self):
            folder = QFileDialog.getExistingDirectory(
                self, "Select Project Folder"
            )
            if folder:
                self.project_dir_input.setText(folder)
                self.on_project_dir_changed()

        def on_project_dir_changed(self):
            pwd = self.project_dir_input.text().strip()
            if not pwd or not os.path.isdir(pwd):
                return

            pwd = os.path.abspath(pwd)
            self.project_dir_input.setText(pwd)

            if self._current_project_dir and self._current_project_dir != pwd:
                self.state_save_timer.stop()
                self.save_project_state()

            self._current_project_dir = pwd

            # Prevent states from the previously selected project remaining visible.
            for button, original_text in (
                (self.btn_run_rfd3, "Run RFD3"),
                (self.btn_run_mpnn, "Run MPNN"),
                (self.btn_run_rf3, "Run RF3"),
                (self.btn_submit_val, "Run CPU Validation Array"),
            ):
                self.reset_button_style(button, original_text)

            self._restoring_project_state = True
            try:
                self.input_pdb_input.clear()
                self.generated_json_path = ""
                self.binder_length_input.setText("15-20")
                self.non_loopy_chk.setChecked(True)
                self.run_dssp_chk.setChecked(False)
                self.rf3_linear_chk.setChecked(True)
                self.rf3_cyclic_chk.setChecked(False)
                self.ignore_cyclic_rmsd_chk.setChecked(True)
                self.num_designs_spin.setValue(100)
                self.num_batches_spin.setValue(20)
                self.seqs_per_struct_spin.setValue(4)
                self.rmsd_cutoff_spin.setValue(5.0)
                self.ptm_cutoff_spin.setValue(0.7)
                self.pae_cutoff_spin.setValue(5.0)
                self.chk_hide_layers.setChecked(False)
                self.chk_hide_non_carbons.setChecked(True)
                self.chk_select_layers.setChecked(True)
                self.slider_size.setValue(65)
                self.slider_stroke.setValue(30)
                self.mol_canvas.plot_structure([])
                self.lbl_selection_count.setText("Selected atoms: 0")
            finally:
                self._restoring_project_state = False

            restored = self.restore_project_state(pwd)
            self.check_all_job_statuses(pwd)
            self.refresh_scores(force=True)
            if not restored:
                self.save_project_state()

        def refresh_job_statuses(self):
            pwd = self._current_project_dir
            if pwd and os.path.isdir(pwd):
                self.check_all_job_statuses(os.path.abspath(pwd))
                self.refresh_scores(force=False)

        def get_current_job_parameters(self):
            selected_atoms = []

            for index in sorted(
                getattr(self.mol_canvas, 'selected_indices', set())
            ):
                if not 0 <= index < len(self.mol_canvas.atoms):
                    continue

                atom = self.mol_canvas.atoms[index]
                selected_atoms.append({
                    "chain_id": atom.get("chain_id", ""),
                    "res_seq": atom.get("res_seq", ""),
                    "name": atom.get("name", ""),
                    "x": atom.get("x", 0.0),
                    "y": atom.get("y", 0.0),
                    "z": atom.get("z", 0.0),
                })

            return {
                "input_pdb": self.input_pdb_input.text().strip(),
                "generated_json_path": getattr(
                    self, 'generated_json_path', ''
                ),
                "binder_length": self.binder_length_input.text().strip(),
                "non_loopy": self.non_loopy_chk.isChecked(),
                "run_dssp": self.run_dssp_chk.isChecked(),
                "rf3_linear": self.rf3_linear_chk.isChecked(),
                "rf3_cyclic": self.rf3_cyclic_chk.isChecked(),
                "ignore_cyclic_rmsd": (
                    self.ignore_cyclic_rmsd_chk.isChecked()
                ),
                "num_designs": self.num_designs_spin.value(),
                "num_batches": self.num_batches_spin.value(),
                "seqs_per_struct": self.seqs_per_struct_spin.value(),
                "rmsd_cutoff": self.rmsd_cutoff_spin.value(),
                "ptm_cutoff": self.ptm_cutoff_spin.value(),
                "pae_cutoff": self.pae_cutoff_spin.value(),
                "hide_layers": self.chk_hide_layers.isChecked(),
                "hide_non_carbons": self.chk_hide_non_carbons.isChecked(),
                "select_as_layers": self.chk_select_layers.isChecked(),
                "atom_size": self.slider_size.value(),
                "stroke_size": self.slider_stroke.value(),
                "selected_atoms": selected_atoms,
            }

        def restore_job_parameters(self, parameters):
            if not isinstance(parameters, dict):
                return

            if "binder_length" in parameters:
                self.binder_length_input.setText(
                    str(parameters["binder_length"])
                )
            if "non_loopy" in parameters:
                self.non_loopy_chk.setChecked(
                    bool(parameters["non_loopy"])
                )
            if "run_dssp" in parameters:
                self.run_dssp_chk.setChecked(
                    bool(parameters["run_dssp"])
                )
            if "rf3_linear" in parameters:
                self.rf3_linear_chk.setChecked(
                    bool(parameters["rf3_linear"])
                )
            if "rf3_cyclic" in parameters:
                self.rf3_cyclic_chk.setChecked(
                    bool(parameters["rf3_cyclic"])
                )
            if not (
                self.rf3_linear_chk.isChecked()
                or self.rf3_cyclic_chk.isChecked()
            ):
                self.rf3_linear_chk.setChecked(True)
            if "ignore_cyclic_rmsd" in parameters:
                self.ignore_cyclic_rmsd_chk.setChecked(
                    bool(parameters["ignore_cyclic_rmsd"])
                )
            if "select_as_layers" in parameters:
                self.chk_select_layers.setChecked(
                    bool(parameters["select_as_layers"])
                )
            if "hide_layers" in parameters:
                self.chk_hide_layers.setChecked(
                    bool(parameters["hide_layers"])
                )
            if "hide_non_carbons" in parameters:
                self.chk_hide_non_carbons.setChecked(
                    bool(parameters["hide_non_carbons"])
                )

            numeric_values = (
                ("num_designs", self.num_designs_spin, int),
                ("num_batches", self.num_batches_spin, int),
                ("seqs_per_struct", self.seqs_per_struct_spin, int),
                ("rmsd_cutoff", self.rmsd_cutoff_spin, float),
                ("ptm_cutoff", self.ptm_cutoff_spin, float),
                ("pae_cutoff", self.pae_cutoff_spin, float),
                ("atom_size", self.slider_size, int),
                ("stroke_size", self.slider_stroke, int),
            )

            for key, widget, converter in numeric_values:
                if key not in parameters:
                    continue
                try:
                    widget.setValue(converter(parameters[key]))
                except (TypeError, ValueError):
                    pass

            self.generated_json_path = str(
                parameters.get("generated_json_path", "") or ""
            )

            pdb_path = str(parameters.get("input_pdb", "") or "")
            if pdb_path:
                if not os.path.exists(pdb_path):
                    workspace_copy = os.path.join(
                        self._current_project_dir,
                        os.path.basename(pdb_path),
                    )
                    if os.path.exists(workspace_copy):
                        pdb_path = workspace_copy

                self.input_pdb_input.setText(pdb_path)

                if os.path.exists(pdb_path):
                    try:
                        self.pdb_widget.browse_file(pdb_path)
                        self.load_pdb_to_canvas(pdb_path)
                        self.mol_canvas.restore_selected_atoms(
                            parameters.get("selected_atoms", [])
                        )
                    except Exception as e:
                        print(f"Failed to restore tracked PDB: {e}")

        def restore_project_state(self, pwd):
            try:
                import json

                parameters = None
                state_file = self.get_project_state_path(pwd)
                if os.path.exists(state_file):
                    with open(state_file, "r") as handle:
                        state = json.load(handle)
                    if isinstance(state, dict):
                        candidate = state.get("parameters", state)
                        if isinstance(candidate, dict):
                            parameters = candidate

                # Backward-compatible fallback for workspaces created before
                # project_state.json existed.
                tracker_file = self.get_tracker_path(pwd)
                if parameters is None and os.path.exists(tracker_file):
                    with open(tracker_file, 'r') as f:
                        tracker = json.load(f)

                    candidates = []
                    for order, data in enumerate(tracker.values()):
                        if (
                            isinstance(data, dict)
                            and isinstance(data.get("parameters"), dict)
                        ):
                            try:
                                submitted_at = float(
                                    data.get("submitted_at", 0)
                                )
                            except (TypeError, ValueError):
                                submitted_at = 0.0
                            candidates.append(
                                (submitted_at, order, data["parameters"])
                            )

                    if candidates:
                        parameters = max(candidates)[2]

                if parameters is None:
                    return False

                self._restoring_project_state = True
                try:
                    self.restore_job_parameters(parameters)
                finally:
                    self._restoring_project_state = False

                # Upgrade older workspaces to the independent state file.
                self.save_project_state()
                return True

            except Exception as e:
                print(f"Failed to restore job parameters: {e}")
                return False

        def get_tracker_path(self, pwd=None):
            if not pwd: pwd = self.project_dir_input.text()
            return os.path.join(pwd, "scripts", "job_tracker.json") if pwd else ""

        def load_job_tracker(self, pwd):
            tracker_file = self.get_tracker_path(pwd)
            if not tracker_file or not os.path.exists(tracker_file):
                return {}
            try:
                import json
                with open(tracker_file, "r") as handle:
                    tracker = json.load(handle)
                return tracker if isinstance(tracker, dict) else {}
            except Exception as error:
                print(f"Failed to load job tracker: {error}")
                return {}

        def write_job_tracker(self, pwd, tracker):
            import json
            tracker_file = self.get_tracker_path(pwd)
            os.makedirs(os.path.dirname(tracker_file), exist_ok=True)
            temporary_tracker = tracker_file + ".tmp"
            with open(temporary_tracker, "w") as handle:
                json.dump(tracker, handle, indent=4)
            os.replace(temporary_tracker, tracker_file)

        def get_step_status(self, pwd, step_name, tracker=None):
            tracker = tracker if tracker is not None else self.load_job_tracker(pwd)
            data = tracker.get(step_name, {})
            if not isinstance(data, dict):
                return ""
            status = str(data.get("status", "")).strip()
            if not status:
                return ""
            return (
                status.split("|", 1)[0]
                .strip()
                .split()[0]
                .split("+", 1)[0]
                .upper()
            )

        def invalidate_downstream_jobs(self, tracker, step_name):
            import time

            workflow = ["rfd3", "mpnn", "rf3", "validation"]
            try:
                step_index = workflow.index(step_name)
            except ValueError:
                return []

            invalidated = []
            for downstream in workflow[step_index + 1:]:
                data = tracker.get(downstream)
                if not isinstance(data, dict):
                    continue
                data["status"] = "STALE"
                data["invalidated_by"] = step_name
                data["invalidated_at"] = time.time()
                invalidated.append(downstream)
            return invalidated

        def apply_workflow_availability(self, pwd, tracker=None):
            # Workflow state affects styling only. Sections are never locked
            # merely because an earlier section is unfinished.
            for button in (
                self.btn_run_rfd3,
                self.btn_run_mpnn,
                self.btn_run_rf3,
                self.btn_submit_val,
            ):
                button.setToolTip("")
                if button.property("job_state") not in {
                    "RUNNING", "SCHEDULED"
                }:
                    button.setEnabled(True)

            tracker = (
                tracker if tracker is not None else self.load_job_tracker(pwd)
            )
            active_states = {
                "PENDING", "RUNNING", "SCHEDULED", "CONFIGURING",
                "COMPLETING", "RESIZING", "REQUEUED", "REQUEUE_FED",
                "SUSPENDED",
            }
            pipeline_active = any(
                self.get_step_status(pwd, step_name, tracker) in active_states
                for step_name in ("rfd3", "mpnn", "rf3", "validation")
            )
            extraction_active = (
                self.local_extraction_process is not None
                and self.local_extraction_process.state()
                != QProcess.ProcessState.NotRunning
            )
            self.btn_auto_run.setEnabled(
                not pipeline_active and not extraction_active
            )
            self.btn_cancel_jobs.setEnabled(pipeline_active)

        def prepare_step_rerun(self, button, original_text):
            step_map = {
                self.btn_run_rfd3: "rfd3",
                self.btn_run_mpnn: "mpnn",
                self.btn_run_rf3: "rf3",
                self.btn_submit_val: "validation",
            }
            button_map = {
                "rfd3": (self.btn_run_rfd3, "Run RFD3"),
                "mpnn": (
                    self.btn_run_mpnn, "Run MPNN"
                ),
                "rf3": (
                    self.btn_run_rf3, "Run RF3"
                ),
                "validation": (
                    self.btn_submit_val,
                    "Run CPU Validation Array",
                ),
            }
            step_name = step_map.get(button)
            pwd = self._current_project_dir
            if not step_name or not pwd:
                self.reset_button_style(button, original_text)
                return

            tracker = self.load_job_tracker(pwd)
            current_data = tracker.setdefault(step_name, {})
            if isinstance(current_data, dict):
                current_data["status"] = "READY"
            invalidated = self.invalidate_downstream_jobs(
                tracker, step_name
            )
            self.write_job_tracker(pwd, tracker)

            self.reset_button_style(button, original_text)
            for downstream in invalidated:
                downstream_button = button_map.get(downstream)
                if downstream_button:
                    self.reset_button_style(
                        downstream_button[0], downstream_button[1]
                    )

            # Resetting an earlier step changes the currency warning and button
            # styling, but existing score CSVs remain useful and viewable.
            self.refresh_scores(force=False)

        def update_button_state(self, button, state, original_text):
            try: button.clicked.disconnect() 
            except TypeError: pass

            button.setProperty("job_state", state)
            font_weight = "bold"
            if state == "RUNNING":
                button.setText("Running")
                button.setStyleSheet(
                    "background-color: #5c5c5c; color: #aaaaaa; "
                    f"font-weight: {font_weight};"
                )
                button.setEnabled(False)
            elif state == "SCHEDULED":
                button.setText("Scheduled")
                button.setStyleSheet(
                    "background-color: #e5c07b; color: #1e1e1e; "
                    f"font-weight: {font_weight};"
                )
                button.setEnabled(False)
            elif state in {"FAILED", "CANCELLED"}:
                button.setText(
                    "Cancelled" if state == "CANCELLED" else "Failed"
                )
                button.setStyleSheet(
                    "background-color: #e06c75; color: #1e1e1e; "
                    f"font-weight: {font_weight};"
                )
                button.setEnabled(True)
                button.clicked.connect(
                    lambda _checked=False: self.prepare_step_rerun(
                        button, original_text
                    )
                )
            elif state == "COMPLETED":
                button.setText("Finished")
                button.setStyleSheet(
                    "background-color: #98c379; color: #1e1e1e; "
                    f"font-weight: {font_weight};"
                )
                button.setEnabled(True)
                button.clicked.connect(
                    lambda _checked=False: self.prepare_step_rerun(
                        button, original_text
                    )
                )
            else:
                self.reset_button_style(button, original_text)
                
        def reset_button_style(self, button, original_text):
            button.setText(original_text)
            button.setStyleSheet("")
            button.setEnabled(True)
            button.setProperty("job_state", "")
            try: button.clicked.disconnect() 
            except TypeError: pass
            
            # Map buttons back to their target run functions
            if button == self.btn_run_rfd3: button.clicked.connect(self.run_rfd3)
            elif button == self.btn_run_mpnn: button.clicked.connect(self.run_mpnn)
            elif button == self.btn_run_rf3: button.clicked.connect(self.run_rf3)
            elif button == self.btn_submit_val: button.clicked.connect(self.run_validation_slurm)

        def check_all_job_statuses(self, pwd):
            tracker_file = self.get_tracker_path(pwd)
            if not os.path.exists(tracker_file):
                self.apply_workflow_availability(pwd, {})
                return

            try:
                import json, subprocess

                with open(tracker_file, 'r') as f:
                    tracker = json.load(f)

                btn_map = {
                    "rfd3": (
                        self.btn_run_rfd3,
                        "Run RFD3"
                    ),
                    "mpnn": (
                        self.btn_run_mpnn,
                        "Run MPNN"
                    ),
                    "rf3": (
                        self.btn_run_rf3,
                        "Run RF3"
                    ),
                    "validation": (
                        self.btn_submit_val,
                        "Run CPU Validation Array"
                    ),
                }

                active_states = {
                    "PENDING",
                    "RUNNING",
                    "SCHEDULED",
                    "CONFIGURING",
                    "COMPLETING",
                    "RESIZING",
                    "REQUEUED",
                    "REQUEUE_FED",
                    "SUSPENDED",
                }

                failed_states = {
                    "BOOT_FAIL",
                    "CANCELLED",
                    "DEADLINE",
                    "FAILED",
                    "NODE_FAIL",
                    "OUT_OF_MEMORY",
                    "PREEMPTED",
                    "REVOKED",
                    "TIMEOUT",
                }

                def normalize_state(raw_state):
                    state = raw_state.strip().split('|', 1)[0].strip()
                    if not state:
                        return ""
                    return state.split()[0].split('+', 1)[0].upper()

                tracker_changed = False
                validation_completed = False

                for step, data in tracker.items():
                    if step not in btn_map or not isinstance(data, dict):
                        continue

                    button, original_text = btn_map[step]

                    saved_state = normalize_state(
                        str(data.get("status", ""))
                    )

                    # Terminal and invalidated states no longer need repeated
                    # scheduler queries. In particular, never let an old
                    # completed SLURM allocation revive a stale downstream step.
                    if saved_state in {
                        "COMPLETED", "FAILED", "CANCELLED", "STALE", "READY"
                    }:
                        self.update_button_state(
                            button, saved_state, original_text
                        )
                        continue

                    if saved_state in active_states:
                        self.update_button_state(
                            button,
                            (
                                "SCHEDULED"
                                if saved_state == "SCHEDULED" else "RUNNING"
                            ),
                            original_text,
                        )

                    job_id = data.get("job_id")
                    if not job_id:
                        continue

                    resolved_state = None

                    try:
                        # Any result in squeue means at least part of the
                        # job or array is still active.
                        sq_result = subprocess.run(
                            [
                                "squeue",
                                "-j", str(job_id),
                                "-h",
                                "-o", "%T",
                            ],
                            capture_output=True,
                            text=True,
                            timeout=10,
                        )

                        queue_states = [
                            normalize_state(line)
                            for line in sq_result.stdout.splitlines()
                            if normalize_state(line)
                        ]

                        if queue_states:
                            if (
                                data.get("dependency")
                                and all(
                                    state == "PENDING"
                                    for state in queue_states
                                )
                            ):
                                resolved_state = "SCHEDULED"
                            else:
                                resolved_state = "RUNNING"
                        else:
                            # Check every allocation record. This prevents a
                            # failed array element being hidden by a completed
                            # first element.
                            account_result = subprocess.run(
                                [
                                    "sacct",
                                    "-j", str(job_id),
                                    "-X",
                                    "-n",
                                    "-P",
                                    "-o", "State%30",
                                ],
                                capture_output=True,
                                text=True,
                                timeout=10,
                            )

                            states = [
                                normalize_state(line)
                                for line in account_result.stdout.splitlines()
                                if normalize_state(line)
                            ]

                            if any(
                                state in active_states for state in states
                            ):
                                if (
                                    data.get("dependency")
                                    and all(
                                        state in {"PENDING", "SCHEDULED"}
                                        for state in states
                                    )
                                ):
                                    resolved_state = "SCHEDULED"
                                else:
                                    resolved_state = "RUNNING"
                            elif any(
                                state in failed_states - {"CANCELLED"}
                                for state in states
                            ):
                                resolved_state = "FAILED"
                            elif any(
                                state == "CANCELLED" for state in states
                            ):
                                resolved_state = "CANCELLED"
                            elif (
                                states
                                and all(
                                    state == "COMPLETED"
                                    for state in states
                                )
                            ):
                                resolved_state = "COMPLETED"

                    except subprocess.TimeoutExpired:
                        # Preserve the current state and retry in 15 seconds.
                        continue

                    if resolved_state:
                        self.update_button_state(
                            button, resolved_state, original_text
                        )

                        if data.get("status") != resolved_state:
                            if (
                                step == "validation"
                                and resolved_state == "COMPLETED"
                            ):
                                validation_completed = True
                            data["status"] = resolved_state
                            tracker_changed = True

                if tracker_changed:
                    self.write_job_tracker(pwd, tracker)

                self.apply_workflow_availability(pwd, tracker)
                if validation_completed:
                    self.refresh_scores(force=True, activate=True)

            except Exception as e:
                print(f"Failed to check job status: {e}")

        def browse_and_load_pdb(self):
            file, _ = QFileDialog.getOpenFileName(
                self,
                "Select Target Structure",
                "",
                "Structure Files (*.pdb *.cif *.mmcif);;"
                "PDB Files (*.pdb);;mmCIF Files (*.cif *.mmcif);;"
                "All Files (*)",
            )
            if file: 
                file = os.path.abspath(file)
                self.input_pdb_input.setText(file)
                # The modifier computes protofilaments/layers for both formats.
                self.pdb_widget.browse_file(file)
                self.load_pdb_to_canvas(file)
                self.schedule_project_state_save()

        def load_pdb_to_canvas(self, filepath):
            try:
                atoms = QuickLookStructureReader.read(filepath)
                if not atoms:
                    raise ValueError("No atoms were found in the structure file.")
                self.mol_canvas.plot_structure(atoms)
            except Exception as e:
                self.mol_canvas.plot_structure([])
                QMessageBox.critical(
                    self, "Error", f"Failed to parse structure file: {e}"
                )

        def merge_chains_for_rf_diffusion(self, input_path, output_path):
            """Write a Biopython-built, continuous single-chain target."""
            return merge_structure_chains_with_biopython(input_path, output_path)

        def generate_json_file(self, show_success=True):
            if not self.mol_canvas.atoms:
                QMessageBox.warning(
                    self, "Warning", "Please load a PDB or CIF structure first."
                )
                return
            
            pwd = self.project_dir_input.text()
            if not pwd or not os.path.isdir(pwd):
                QMessageBox.warning(self, "Warning", "Please select a valid project workspace folder first.")
                return

            original_structure = self.input_pdb_input.text()
            if not os.path.exists(original_structure):
                QMessageBox.warning(
                    self, "Warning", "Original structure file not found."
                )
                return

            # 1. Store accurate absolute coordinates to trace structural changes
            selected_coords = []
            captured_types = set()
            for idx in getattr(self.mol_canvas, 'selected_indices', set()):
                atom = self.mol_canvas.atoms[idx]
                selected_coords.append( (atom['x'], atom['y'], atom['z'], atom['name']) )
                captured_types.add(atom['name'])

            # 2. Check the loaded structure directly. RF Diffusion owns its
            # merge implementation, so this workflow remains movable without
            # depending on the Modifier tab's Connect Chains action.
            structure_chain_ids = {
                str(atom.get("chain_id", ""))
                for atom in self.mol_canvas.atoms
                if atom.get("record_type", "ATOM") == "ATOM"
            }
            needs_merging = len(structure_chain_ids) > 1
            
            if needs_merging:
                base_name = os.path.splitext(
                    os.path.basename(original_structure)
                )[0]
                target_extension = (
                    ".cif"
                    if original_structure.lower().endswith((".cif", ".mmcif"))
                    else ".pdb"
                )
                final_target_path = os.path.join(
                    pwd, f"{base_name}_merged{target_extension}"
                )
                try:
                    self.merge_chains_for_rf_diffusion(
                        original_structure, final_target_path
                    )
                    if self.run_dssp_chk.isChecked():
                        self.pdb_widget.browse_file(
                            final_target_path, mutable=True
                        )
                        self.pdb_widget.action_evaluate_beta()
                        evaluated_path = self.pdb_widget.working_file_path
                        if not evaluated_path or not os.path.exists(evaluated_path):
                            raise RuntimeError(
                                "The beta-sheet evaluation produced no file."
                            )
                        if os.path.abspath(evaluated_path) != os.path.abspath(
                            final_target_path
                        ):
                            shutil.copy2(evaluated_path, final_target_path)
                except Exception as error:
                    QMessageBox.critical(
                        self,
                        "Error",
                        f"Failed to merge chains or re-evaluate beta sheet: {error}",
                    )
                    return
            else:
                # Use the original single-chain structure directly.
                base_name = os.path.splitext(
                    os.path.basename(original_structure)
                )[0]
                final_target_path = original_structure

            # 5. Spatially identify the new residue sequences in the updated single-chain topology
            try:
                final_atoms = QuickLookStructureReader.read(final_target_path)
                if not final_atoms:
                    raise ValueError("No atoms were found after structure processing.")
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to parse the processed structure: {e}",
                )
                return

            hotspots = QuickLookStructureReader.map_selected_hotspots(
                final_atoms, selected_coords
            )
            protein_contig = QuickLookStructureReader.build_protein_contig(
                final_atoms
            )
            
            # 6. JSON Assembly
            binder_len = self.binder_length_input.text().strip()
            if not binder_len: binder_len = "15-20"
            
            json_data = {
                "Binder": {
                    "dialect": 2,
                    "infer_ori_strategy": "hotspots",
                    "input": final_target_path,
                    # /0 is only the break between the fixed target and binder.
                    "contig": f"{protein_contig},/0,{binder_len}",
                    "select_hotspots": hotspots,
                    "is_non_loopy": self.non_loopy_chk.isChecked()
                }
            }
            
            out_path = os.path.join(pwd, f"{base_name}.json")
            try:
                with open(out_path, 'w') as f:
                    import json
                    json.dump(json_data, f, indent=4)
                self.generated_json_path = out_path
                self.schedule_project_state_save()
                
                msg = (
                    f"JSON generated with {len(hotspots)} mapped hotspot "
                    f"residues!\n"
                )
                msg += (
                    "Captured atom names: "
                    f"{', '.join(sorted(captured_types)) or 'None'}\n\n"
                )
                msg += f"Saved to:\n{out_path}\n\n"
                msg += f"Target structure used:\n{final_target_path}"
                if show_success:
                    QMessageBox.information(self, "Success", msg)
                return out_path
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save JSON: {e}")
                return None

        def _check_workspace(self):
            import os
            pwd = self.project_dir_input.text()
            if not pwd or not os.path.isdir(pwd):
                QMessageBox.critical(self, "Error", "Please select a valid project workspace folder.")
                return None
            os.makedirs(os.path.join(pwd, "logs"), exist_ok=True)
            os.makedirs(os.path.join(pwd, "rfd3_outputs"), exist_ok=True)
            os.makedirs(os.path.join(pwd, "mpnn_outputs"), exist_ok=True)
            os.makedirs(os.path.join(pwd, "rf3_outputs"), exist_ok=True)
            os.makedirs(os.path.join(pwd, "scores"), exist_ok=True)
            os.makedirs(os.path.join(pwd, "scripts"), exist_ok=True)
            return pwd

        def execute_command(self, cmd_list, cwd, success_msg, step_name=None, button=None, orig_text=None):
            try:
                import subprocess, re, time
                result = subprocess.run(cmd_list, cwd=cwd, capture_output=True, text=True, check=True)
                output = result.stdout.strip()
                QMessageBox.information(self, "Success", f"{success_msg}\n\nOutput:\n{output}")
                
                # Check for SLURM submission response
                if cmd_list[0] == "sbatch" and step_name and button:
                    match = re.search(r"Submitted batch job (\d+)", output)
                    if match:
                        job_id = match.group(1)
                        tracker = self.load_job_tracker(cwd)
                        invalidated = self.invalidate_downstream_jobs(
                            tracker, step_name
                        )
                        script_path = os.path.join(cwd, cmd_list[1])
                        try:
                            with open(script_path, 'r') as script_file:
                                script_text = script_file.read()
                        except OSError:
                            script_text = ""

                        runner_name = {
                            "rfd3": "run_rfd3.py",
                            "mpnn": "run_mpnn.py",
                            "rf3": "run_rf3.py",
                            "validation": "run_validation.py",
                        }.get(step_name, "")

                        runner_path = (
                            os.path.join(cwd, "scripts", runner_name)
                            if runner_name else ""
                        )

                        try:
                            with open(runner_path, 'r') as runner_file:
                                runner_text = runner_file.read()
                        except OSError:
                            runner_text = ""

                        tracker[step_name] = {
                            "job_id": job_id,
                            "status": "RUNNING",
                            "submitted_at": time.time(),
                            "command": list(cmd_list),
                            "submit_script": script_path,
                            "submit_script_text": script_text,
                            "runner_script": runner_path,
                            "runner_script_text": runner_text,
                            "parameters": self.get_current_job_parameters(),
                        }
                        self.write_job_tracker(cwd, tracker)

                        self.update_button_state(button, "RUNNING", orig_text)
                        button_map = {
                            "mpnn": (
                                self.btn_run_mpnn,
                                "Run MPNN",
                            ),
                            "rf3": (
                                self.btn_run_rf3,
                                "Run RF3",
                            ),
                            "validation": (
                                self.btn_submit_val,
                                "Run CPU Validation Array",
                            ),
                        }
                        for invalidated_step in invalidated:
                            invalidated_button = button_map.get(
                                invalidated_step
                            )
                            if invalidated_button:
                                self.update_button_state(
                                    invalidated_button[0],
                                    "STALE",
                                    invalidated_button[1],
                                )

                        self.apply_workflow_availability(cwd, tracker)
                        self.save_project_state()
                        # Keep already-produced scores visible during reruns.
                        # The score header clearly marks them as potentially
                        # out of date until validation completes again.
                        self.refresh_scores(force=False)
                return True

            except subprocess.CalledProcessError as e:
                QMessageBox.critical(self, "Execution Error", f"Command failed with exit code {e.returncode}\n\nError:\n{e.stderr.strip()}")
                return False
            except FileNotFoundError:
                QMessageBox.critical(self, "System Error", f"Command not found: {cmd_list[0]}\nAre you running this on the HPC login node?")
                return False

        def run_rfd3(self):
            if not getattr(self.mol_canvas, 'selected_indices', set()):
                self.right_tabs.setCurrentIndex(0)
                QMessageBox.warning(
                    self,
                    "Select Hotspot Atoms",
                    "Click atoms in the Structure Preview to select hotspot "
                    "atoms before running RFD3.\n\nUsually, select only "
                    "carbon atoms.",
                )
                return

            pwd = self._check_workspace()
            if not pwd: return
            
            input_json = self.generate_json_file()
            if not input_json or not os.path.exists(input_json):
                QMessageBox.critical(self, "Error", "Failed to generate JSON configuration.")
                return
            
            # Fetch the 2 variables from the UI
            num_designs = self.num_designs_spin.value()
            num_jobs = self.num_batches_spin.value()
            
            # Calculate the SLURM array limit (e.g., 20 jobs = array 0-19)
            array_limit = max(0, num_jobs - 1)
            
            # Hardcode the GPU inference batch size to 1 (prevents GPU memory crashes)
            internal_gpu_batch_size = 1 
            
            self._write_rfd3_scripts(pwd, input_json, num_designs, internal_gpu_batch_size, array_limit)
            self.execute_command(["sbatch", "scripts/submit_rfd3.sh"], pwd, "RFD3 Job Submitted Successfully.", "rfd3", self.btn_run_rfd3, "Run RFD3")

        def auto_run_pipeline(self):
            """Submit all four compute stages as an afterok dependency chain."""
            if not getattr(self.mol_canvas, 'selected_indices', set()):
                self.right_tabs.setCurrentIndex(0)
                QMessageBox.warning(
                    self,
                    "Select Hotspot Atoms",
                    "Click atoms in the Structure Preview to select hotspot "
                    "atoms before starting Auto Run.\n\nUsually, select only "
                    "carbon atoms.",
                )
                return

            pwd = self._check_workspace()
            if not pwd:
                return

            rf3_topologies = self.selected_rf3_topologies(show_warning=True)
            if not rf3_topologies:
                return

            self.btn_auto_run.setText("Submitting...")
            self.btn_auto_run.setEnabled(False)
            self.btn_cancel_jobs.setEnabled(False)
            QApplication.processEvents()

            submitted = []
            try:
                import subprocess
                import time

                input_json = self.generate_json_file(show_success=False)
                if not input_json or not os.path.exists(input_json):
                    return

                array_limit = max(0, self.num_batches_spin.value() - 1)
                self._write_rfd3_scripts(
                    pwd,
                    input_json,
                    self.num_designs_spin.value(),
                    1,
                    array_limit,
                )
                self._write_mpnn_scripts(
                    pwd, self.seqs_per_struct_spin.value(), array_limit
                )
                self._write_rf3_scripts(
                    pwd, array_limit, rf3_topologies
                )
                self._write_validation_scripts(pwd, array_limit)

                stages = (
                    (
                        "rfd3", "scripts/submit_rfd3.sh", "run_rfd3.py",
                        self.btn_run_rfd3, "Run RFD3",
                    ),
                    (
                        "mpnn", "scripts/submit_mpnn.sh", "run_mpnn.py",
                        self.btn_run_mpnn, "Run MPNN",
                    ),
                    (
                        "rf3", "scripts/submit_rf3.sh", "run_rf3.py",
                        self.btn_run_rf3, "Run RF3",
                    ),
                    (
                        "validation", "scripts/submit_validation_cpu.sh",
                        "run_validation.py", self.btn_submit_val,
                        "Run CPU Validation Array",
                    ),
                )

                tracker = self.load_job_tracker(pwd)
                parameters = self.get_current_job_parameters()
                dependency_job_id = None

                for step_name, submit_script, runner_name, button, label in stages:
                    command = build_slurm_submit_command(
                        submit_script, dependency_job_id
                    )
                    result = subprocess.run(
                        command,
                        cwd=pwd,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    job_id = parse_slurm_job_id(result.stdout)

                    if step_name == "rfd3":
                        self.invalidate_downstream_jobs(tracker, step_name)

                    submit_path = os.path.join(pwd, submit_script)
                    runner_path = os.path.join(pwd, "scripts", runner_name)
                    try:
                        with open(submit_path, "r") as handle:
                            submit_text = handle.read()
                    except OSError:
                        submit_text = ""
                    try:
                        with open(runner_path, "r") as handle:
                            runner_text = handle.read()
                    except OSError:
                        runner_text = ""

                    display_state = (
                        "RUNNING"
                        if dependency_job_id is None else "SCHEDULED"
                    )
                    tracker[step_name] = {
                        "job_id": job_id,
                        "status": display_state,
                        "submitted_at": time.time(),
                        "command": list(command),
                        "dependency": (
                            f"afterok:{dependency_job_id}"
                            if dependency_job_id is not None else ""
                        ),
                        "submission_mode": "auto",
                        "submit_script": submit_path,
                        "submit_script_text": submit_text,
                        "runner_script": runner_path,
                        "runner_script_text": runner_text,
                        "parameters": parameters,
                    }
                    self.write_job_tracker(pwd, tracker)
                    self.update_button_state(button, display_state, label)
                    submitted.append((step_name, job_id))
                    dependency_job_id = job_id

                self.apply_workflow_availability(pwd, tracker)
                self.save_project_state()
                self.refresh_scores(force=False)
                job_summary = "\n".join(
                    f"{step_name.upper()}: {job_id}"
                    for step_name, job_id in submitted
                )
                QMessageBox.information(
                    self,
                    "Auto Run Submitted",
                    "RFD3 is running. MPNN, RF3, and CPU validation are "
                    "scheduled with SLURM afterok dependencies. If a stage "
                    "fails, its dependent jobs are cancelled.\n\n"
                    f"{job_summary}",
                )
            except subprocess.CalledProcessError as error:
                stderr = (error.stderr or "").strip()
                already_submitted = "\n".join(
                    f"{step_name.upper()}: {job_id}"
                    for step_name, job_id in submitted
                )
                message = (
                    "Pipeline submission stopped because sbatch exited with "
                    f"code {error.returncode}.\n\n"
                    f"{stderr or 'No error text was returned.'}"
                )
                if already_submitted:
                    message += (
                        "\n\nAlready submitted jobs remain tracked:\n"
                        f"{already_submitted}"
                    )
                QMessageBox.critical(self, "Auto Run Submission Error", message)
            except FileNotFoundError:
                QMessageBox.critical(
                    self,
                    "System Error",
                    "Command not found: sbatch\nAre you running this on the "
                    "HPC login node?",
                )
            except Exception as error:
                QMessageBox.critical(
                    self,
                    "Auto Run Submission Error",
                    f"Could not submit the complete pipeline:\n{error}",
                )
            finally:
                self.btn_auto_run.setText("Auto Run")
                tracker = self.load_job_tracker(pwd)
                self.apply_workflow_availability(pwd, tracker)

        def cancel_running_jobs(self):
            """Cancel every active tracked SLURM job in this workspace."""
            pwd = self._check_workspace()
            if not pwd:
                return

            active_states = {
                "PENDING", "RUNNING", "SCHEDULED", "CONFIGURING",
                "COMPLETING", "RESIZING", "REQUEUED", "REQUEUE_FED",
                "SUSPENDED",
            }
            tracker = self.load_job_tracker(pwd)
            candidates = []
            for step_name in ("rfd3", "mpnn", "rf3", "validation"):
                data = tracker.get(step_name, {})
                if not isinstance(data, dict):
                    continue
                job_id = str(data.get("job_id", "")).strip()
                if (
                    job_id
                    and self.get_step_status(pwd, step_name, tracker)
                    in active_states
                ):
                    candidates.append((step_name, job_id))

            if not candidates:
                QMessageBox.information(
                    self,
                    "Cancel Running Jobs",
                    "No active or scheduled pipeline jobs are tracked for "
                    "this workspace.",
                )
                self.apply_workflow_availability(pwd, tracker)
                return

            unique_job_ids = list(dict.fromkeys(
                job_id for _step_name, job_id in candidates
            ))
            job_summary = "\n".join(
                f"{step_name.upper()}: {job_id}"
                for step_name, job_id in candidates
            )
            answer = QMessageBox.question(
                self,
                "Cancel Running Jobs",
                "Cancel all of these active or scheduled SLURM jobs?\n\n"
                f"{job_summary}",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

            try:
                import subprocess
                import time

                result = subprocess.run(
                    ["scancel", *unique_job_ids],
                    cwd=pwd,
                    capture_output=True,
                    text=True,
                    check=True,
                )
                cancelled_ids = set(unique_job_ids)
                button_map = {
                    "rfd3": (self.btn_run_rfd3, "Run RFD3"),
                    "mpnn": (self.btn_run_mpnn, "Run MPNN"),
                    "rf3": (self.btn_run_rf3, "Run RF3"),
                    "validation": (
                        self.btn_submit_val, "Run CPU Validation Array"
                    ),
                }
                for step_name, data in tracker.items():
                    if (
                        step_name not in button_map
                        or not isinstance(data, dict)
                        or str(data.get("job_id", "")) not in cancelled_ids
                    ):
                        continue
                    data["status"] = "CANCELLED"
                    data["cancelled_at"] = time.time()
                    button, label = button_map[step_name]
                    self.update_button_state(button, "CANCELLED", label)

                self.write_job_tracker(pwd, tracker)
                self.apply_workflow_availability(pwd, tracker)
                self.refresh_scores(force=False)
                output = (result.stdout or result.stderr or "").strip()
                message = f"Cancellation requested for:\n{job_summary}"
                if output:
                    message += f"\n\nScheduler output:\n{output}"
                QMessageBox.information(
                    self, "Jobs Cancelled", message
                )
            except subprocess.CalledProcessError as error:
                QMessageBox.critical(
                    self,
                    "Cancellation Error",
                    "scancel failed with exit code "
                    f"{error.returncode}.\n\n{(error.stderr or '').strip()}",
                )
                self.check_all_job_statuses(pwd)
            except FileNotFoundError:
                QMessageBox.critical(
                    self,
                    "System Error",
                    "Command not found: scancel\nAre you running this on the "
                    "HPC login node?",
                )

        def run_mpnn(self):
            pwd = self._check_workspace()
            if not pwd: return
            rfd3_dir = os.path.join(pwd, "rfd3_outputs")
            batches = [
                d for d in os.listdir(rfd3_dir)
                if d.startswith("batch_")
                and os.path.isdir(os.path.join(rfd3_dir, d))
            ]
            if not batches:
                QMessageBox.critical(
                    self, "Dependency Error",
                    "No RFD3 batch outputs were found for Section 2 to use.",
                )
                return
            array_limit = len(batches) - 1
            seqs = self.seqs_per_struct_spin.value()
            self._write_mpnn_scripts(pwd, seqs, array_limit)
            self.execute_command(["sbatch", "scripts/submit_mpnn.sh"], pwd, "MPNN Job Submitted Successfully.", "mpnn", self.btn_run_mpnn, "Run MPNN")

        def selected_rf3_topologies(self, show_warning=False):
            topologies = []
            if self.rf3_linear_chk.isChecked():
                topologies.append("linear")
            if self.rf3_cyclic_chk.isChecked():
                topologies.append("cyclic")
            if not topologies and show_warning:
                QMessageBox.warning(
                    self,
                    "Select RF3 Topology",
                    "Select at least one binder topology: Linear or "
                    "Head-to-tail cyclic.",
                )
            return topologies

        def enforce_rf3_topology_selection(self, _checked=False):
            if self._restoring_project_state:
                return
            if self.selected_rf3_topologies():
                return

            changed_checkbox = self.sender()
            if changed_checkbox in (
                self.rf3_linear_chk, self.rf3_cyclic_chk
            ):
                changed_checkbox.setChecked(True)
            else:
                self.rf3_linear_chk.setChecked(True)

        def run_rf3(self):
            pwd = self._check_workspace()
            if not pwd: return
            topologies = self.selected_rf3_topologies(show_warning=True)
            if not topologies:
                return
            mpnn_dir = os.path.join(pwd, "mpnn_outputs")
            batches = [
                d for d in os.listdir(mpnn_dir)
                if d.startswith("batch_")
                and os.path.isdir(os.path.join(mpnn_dir, d))
            ]
            if not batches:
                QMessageBox.critical(
                    self, "Dependency Error",
                    "No MPNN batch outputs were found for Section 3 to use.",
                )
                return
            array_limit = len(batches) - 1
            self._write_rf3_scripts(pwd, array_limit, topologies)
            self.execute_command(["sbatch", "scripts/submit_rf3.sh"], pwd, "RF3 Validation Job Submitted Successfully.", "rf3", self.btn_run_rf3, "Run RF3")

        def run_validation_slurm(self):
            pwd = self._check_workspace()
            if not pwd: return
            rf3_dir = os.path.join(pwd, "rf3_outputs")
            batches = [
                d for d in os.listdir(rf3_dir)
                if d.startswith("batch_")
                and os.path.isdir(os.path.join(rf3_dir, d))
            ]
            if not batches:
                QMessageBox.critical(
                    self, "Dependency Error",
                    "No RF3 batch outputs were found for Section 4 to use.",
                )
                return
            array_limit = len(batches) - 1
            self._write_validation_scripts(pwd, array_limit)
            self.execute_command(["sbatch", "scripts/submit_validation_cpu.sh"], pwd, "Validation Calculation Job Submitted.", "validation", self.btn_submit_val, "Run CPU Validation Array")

        def run_consolidation_local(self):
            pwd = self._check_workspace()
            if not pwd: return
            scores_dir = os.path.join(pwd, "scores")
            if not os.path.exists(scores_dir) or not [f for f in os.listdir(scores_dir) if f.endswith(".csv")]:
                QMessageBox.warning(self, "Warning", "No score CSVs found. Ensure the CPU validation array has completed first.")
                return
            rmsd = self.rmsd_cutoff_spin.value()
            ptm = self.ptm_cutoff_spin.value()
            pae = self.pae_cutoff_spin.value()
            self._write_consolidation_script(
                pwd,
                rmsd,
                ptm,
                pae,
                self.ignore_cyclic_rmsd_chk.isChecked(),
            )
            self.start_local_extraction(pwd)

        def _next_iteration_workspace(self, pwd):
            """Return a new sibling workspace path with the next iteration ID."""
            import re

            current_workspace = os.path.normpath(os.path.abspath(pwd))
            parent_dir = os.path.dirname(current_workspace)
            workspace_name = os.path.basename(current_workspace)
            iteration_prefix = (
                f"{workspace_name}_Passed_Binders_Iteration"
            )
            iteration_pattern = re.compile(
                rf"^{re.escape(iteration_prefix)}(\d+)$"
            )

            existing_numbers = []
            for entry in os.scandir(parent_dir):
                match = iteration_pattern.match(entry.name)
                if match and entry.is_dir():
                    existing_numbers.append(int(match.group(1)))

            iteration_number = max(existing_numbers, default=0) + 1
            candidate = os.path.join(
                parent_dir, f"{iteration_prefix}{iteration_number}"
            )
            while os.path.exists(candidate):
                iteration_number += 1
                candidate = os.path.join(
                    parent_dir, f"{iteration_prefix}{iteration_number}"
                )
            return candidate

        def iterate_designs(self):
            pwd = self._check_workspace()
            if not pwd:
                return

            reminder = (
                "This will create a new workspace and allow you to use the "
                "passed designs as backbones to run another iteration of "
                "designs"
            )
            response = QMessageBox.question(
                self,
                "Iterate designs (Optional)",
                reminder,
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if response != QMessageBox.StandardButton.Yes:
                return

            import re

            source_dir = os.path.join(pwd, "successful_designs")
            if not os.path.isdir(source_dir):
                QMessageBox.warning(
                    self,
                    "No Passed Designs",
                    "The 'successful_designs' folder was not found. Run "
                    "Local Extraction in Section 4 first.",
                )
                return

            filename_pattern = re.compile(
                r"^batch_(\d+)_(.+)\.cif$", re.IGNORECASE
            )
            topology_pattern = re.compile(
                r"(?:^|_)(linear|cyclic)(?:_|$)", re.IGNORECASE
            )
            designs = []
            try:
                with os.scandir(source_dir) as entries:
                    source_entries = sorted(
                        entries, key=lambda entry: entry.name.lower()
                    )
                for entry in source_entries:
                    if (
                        not entry.is_file()
                        or not entry.name.lower().endswith(".cif")
                    ):
                        continue
                    match = filename_pattern.match(entry.name)
                    if not match:
                        continue
                    topology_matches = list(
                        topology_pattern.finditer(match.group(2))
                    )
                    topology = (
                        topology_matches[-1].group(1).lower()
                        if topology_matches else "linear"
                    )
                    designs.append({
                        "source": entry.path,
                        "batch": int(match.group(1)),
                        "topology": topology,
                    })
            except OSError as error:
                QMessageBox.critical(
                    self,
                    "Iteration Error",
                    f"Could not read passed designs:\n{error}",
                )
                return

            if not designs:
                QMessageBox.warning(
                    self,
                    "No Passed Designs",
                    "No matching CIF files were found in 'successful_designs'. "
                    "Run Local Extraction in Section 4 first.",
                )
                return

            unique_batches = sorted({design["batch"] for design in designs})
            batch_map = {
                old_batch: new_batch
                for new_batch, old_batch in enumerate(unique_batches)
            }

            try:
                new_workspace = self._next_iteration_workspace(pwd)
            except OSError as error:
                QMessageBox.critical(
                    self,
                    "Iteration Error",
                    f"Could not inspect the workspace folder:\n{error}",
                )
                return

            rfd3_output_dir = os.path.join(new_workspace, "rfd3_outputs")
            created_workspace = False
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            try:
                os.makedirs(new_workspace, exist_ok=False)
                created_workspace = True
                os.makedirs(rfd3_output_dir, exist_ok=False)

                copied_destinations = set()
                for design_number, design in enumerate(designs):
                    batch_dir = os.path.join(
                        rfd3_output_dir,
                        f"batch_{batch_map[design['batch']]}",
                    )
                    os.makedirs(batch_dir, exist_ok=True)
                    destination = os.path.join(
                        batch_dir,
                        f"Passed_Binder_{design['topology']}_"
                        f"{design_number:06d}_model_0.cif",
                    )
                    shutil.copy2(design["source"], destination)
                    copied_destinations.add(destination)
            except (OSError, shutil.Error) as error:
                cleanup_failed = False
                if created_workspace:
                    try:
                        shutil.rmtree(new_workspace)
                    except OSError:
                        cleanup_failed = True
                message = f"Could not create the iteration workspace:\n{error}"
                if cleanup_failed:
                    message += (
                        "\n\nA partial workspace may remain at:\n"
                        f"{new_workspace}"
                    )
                QMessageBox.critical(self, "Iteration Error", message)
                return
            finally:
                QApplication.restoreOverrideCursor()

            self.project_dir_input.setText(new_workspace)
            self.on_project_dir_changed()
            copied_topologies = {
                design["topology"] for design in designs
            }
            self._restoring_project_state = True
            try:
                self.rf3_linear_chk.setChecked(
                    "linear" in copied_topologies
                )
                self.rf3_cyclic_chk.setChecked(
                    "cyclic" in copied_topologies
                )
            finally:
                self._restoring_project_state = False
            self.save_project_state()
            QMessageBox.information(
                self,
                "Iteration Workspace Ready",
                f"Created and loaded:\n{new_workspace}\n\n"
                f"Prepared {len(copied_destinations)} passed backbone(s) from "
                f"{len(designs)} successful design(s) in "
                f"{len(unique_batches)} RFD3 batch folder(s).\n\n"
                "Continue from Section 2: MPNN Sequence Design.",
            )

        def start_local_extraction(self, pwd):
            if (
                self.local_extraction_process is not None
                and self.local_extraction_process.state()
                != QProcess.ProcessState.NotRunning
            ):
                QMessageBox.information(
                    self,
                    "Local Extraction",
                    "Local extraction is already running.",
                )
                return

            process = QProcess(self)
            self.local_extraction_process = process
            self.local_extraction_output = []

            process.setWorkingDirectory(pwd)
            process.setProgram("conda")
            process.setArguments([
                "run", "-n", "rfdxcop", "python",
                "scripts/run_consolidation.py",
            ])
            process.setProcessChannelMode(
                QProcess.ProcessChannelMode.MergedChannels
            )
            process.readyReadStandardOutput.connect(
                lambda: self.collect_local_extraction_output(process)
            )
            process.finished.connect(
                lambda exit_code, exit_status: self.finish_local_extraction(
                    process, pwd, exit_code, exit_status
                )
            )
            process.errorOccurred.connect(
                lambda error: self.handle_local_extraction_error(
                    process, error
                )
            )

            self.btn_run_consolidation.setText("Extracting...")
            self.btn_run_consolidation.setStyleSheet(
                "background-color: #5c5c5c; color: #aaaaaa;"
            )
            self.btn_run_consolidation.setEnabled(False)
            self.btn_iterate_designs.setEnabled(False)
            self.btn_auto_run.setEnabled(False)
            process.start()

        def collect_local_extraction_output(self, process):
            if process is None:
                return
            output = bytes(process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            )
            if output:
                self.local_extraction_output.append(output)

        def restore_local_extraction_button(self):
            self.btn_run_consolidation.setText("Run Local Extraction")
            self.btn_run_consolidation.setStyleSheet("")
            self.btn_run_consolidation.setEnabled(True)
            self.btn_iterate_designs.setEnabled(True)
            self.apply_workflow_availability(
                self._current_project_dir,
                self.load_job_tracker(self._current_project_dir),
            )

        def finish_local_extraction(
            self, process, pwd, exit_code, exit_status
        ):
            if process is not self.local_extraction_process:
                return

            self.collect_local_extraction_output(process)
            output = "".join(self.local_extraction_output).strip()
            self.local_extraction_process = None
            self.restore_local_extraction_button()
            process.deleteLater()

            succeeded = (
                exit_code == 0
                and exit_status == QProcess.ExitStatus.NormalExit
            )
            if succeeded:
                message = (
                    "Data consolidation, sequence deduplication, and "
                    "extraction complete! "
                    "Check the 'successful_designs' folder."
                )
                if output:
                    message += f"\n\nOutput:\n{output[-12000:]}"
                QMessageBox.information(self, "Success", message)
                if self._current_project_dir == pwd:
                    self.refresh_scores(force=True, activate=True)
            else:
                details = output or "The extraction process exited unexpectedly."
                QMessageBox.critical(
                    self,
                    "Execution Error",
                    f"Local extraction failed with exit code {exit_code}."
                    f"\n\nOutput:\n{details[-12000:]}",
                )

        def handle_local_extraction_error(self, process, error):
            if (
                process is not self.local_extraction_process
                or error != QProcess.ProcessError.FailedToStart
            ):
                return

            error_text = process.errorString()
            self.local_extraction_process = None
            self.restore_local_extraction_button()
            process.deleteLater()
            QMessageBox.critical(
                self,
                "System Error",
                "Could not start Local Extraction. Ensure 'conda' is "
                f"available on PATH.\n\n{error_text}",
            )

        def _write_rfd3_scripts(self, base_dir, input_json, num_designs, batch_size, array_limit):
            run_rfd3_code = '''import argparse\nimport os\nfrom lightning.fabric import seed_everything\nfrom rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine\n\nparser = argparse.ArgumentParser()\nparser.add_argument("--input_json", type=str, required=True)\nparser.add_argument("--num_designs", type=int, default=10)\nparser.add_argument("--batch_size", type=int, default=1)\nparser.add_argument("--out_dir", type=str, default="./rfd3_outputs")\nargs = parser.parse_args()\n\nnum_batches = (args.num_designs + args.batch_size - 1) // args.batch_size\nos.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\nseed_everything(0)\n\nconfig = RFD3InferenceConfig(specification={'extra': {}}, diffusion_batch_size=args.batch_size)\nmodel = RFD3InferenceEngine(**config)\nmodel.run(inputs=args.input_json, out_dir=args.out_dir, n_batches=num_batches)\n'''
            with open(os.path.join(base_dir, "scripts", "run_rfd3.py"), "w") as f: f.write(run_rfd3_code)
            slurm_rfd3_code = f'''#!/bin/bash\n#SBATCH --job-name=rfd3_gen\n#SBATCH --output=logs/%x_%A_%a.out\n#SBATCH --error=logs/%x_%A_%a.err\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=32G\n#SBATCH --time=24:00:00\n#SBATCH --partition=normal\n#SBATCH --gres=gpu:1\n#SBATCH --array=0-{array_limit}%8\n#SBATCH --export=NONE\n\nset -eo pipefail\n\nmkdir -p logs\neval "$(conda shell.bash hook)"\nconda activate rfdxcop\n\npython scripts/run_rfd3.py \\\n        --input_json "{input_json}" \\\n        --num_designs {num_designs} \\\n        --batch_size {batch_size} \\\n        --out_dir ./rfd3_outputs/batch_${{SLURM_ARRAY_TASK_ID}}\n\nfind ./rfd3_outputs/batch_${{SLURM_ARRAY_TASK_ID}} -name "*.gz" -exec gunzip -f {{}} +\n'''
            with open(os.path.join(base_dir, "scripts", "submit_rfd3.sh"), "w") as f: f.write(slurm_rfd3_code)

        def _write_mpnn_scripts(self, base_dir, seqs_per_struct, array_limit):
            run_mpnn_code = '''import argparse
import glob
import os

import biotite.structure as struc
import atomworks.io.utils.io_utils as atomworks_io
from mpnn.inference_engines.mpnn import MPNNInferenceEngine


# AtomWorks' entity_poly writer calculates residue-start indices on a
# chain-filtered AtomArray, but uses them to index the full AtomArray.  This
# makes every chain after the first advertise a sequence taken from chain A.
# Calling the original builder on one chain at a time keeps the indices local;
# rows that belong to the same entity are then merged back together.
_original_build_entity_poly = atomworks_io._build_entity_poly


def _build_entity_poly_with_chain_local_indices(atom_array):
    structure = (
        atom_array[0]
        if isinstance(atom_array, struc.AtomArrayStack)
        else atom_array
    )
    chain_starts = struc.get_chain_starts(structure)
    rows_by_entity = {}
    column_names = None

    for chain_iid in structure.chain_iid[chain_starts]:
        chain_array = structure[structure.chain_iid == chain_iid]
        built = _original_build_entity_poly(chain_array)
        category = built.get("entity_poly") if built else None
        if not category:
            continue

        if column_names is None:
            column_names = tuple(category.keys())
        for row_index in range(len(category["entity_id"])):
            row = {
                column: category[column][row_index]
                for column in column_names
            }
            entity_id = row["entity_id"]
            if entity_id not in rows_by_entity:
                rows_by_entity[entity_id] = row
                continue

            existing = rows_by_entity[entity_id]
            strand_ids = [
                strand_id
                for strand_id in (
                    str(existing["pdbx_strand_id"]) + "," +
                    str(row["pdbx_strand_id"])
                ).split(",")
                if strand_id
            ]
            existing["pdbx_strand_id"] = ",".join(
                dict.fromkeys(strand_ids)
            )

    if not rows_by_entity or column_names is None:
        return {}
    rows = list(rows_by_entity.values())
    return {
        "entity_poly": {
            column: [row[column] for row in rows]
            for column in column_names
        }
    }


atomworks_io._build_entity_poly = _build_entity_poly_with_chain_local_indices

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--seqs_per_struct", type=int, default=4)
args = parser.parse_args()

cif_files = sorted(glob.glob(os.path.join(args.input_dir, "*.cif")))
if not cif_files:
    raise SystemExit(f"No RFD3 CIF files found in {args.input_dir}")

engine_config = {
    "model_type": "protein_mpnn",
    "is_legacy_weights": True,
    "out_directory": args.out_dir,
    "write_structures": True,
    "write_fasta": False,
}
input_configs = [
    {
        "batch_size": args.seqs_per_struct,
        "remove_waters": True,
        "structure_path": path,
        "fixed_chains": ["A"],
    }
    for path in cif_files
]

MPNNInferenceEngine(**engine_config).run(input_dicts=input_configs)
'''
            with open(os.path.join(base_dir, "scripts", "run_mpnn.py"), "w") as f: f.write(run_mpnn_code)
            slurm_mpnn_code = f'''#!/bin/bash\n#SBATCH --job-name=mpnn_seq\n#SBATCH --output=logs/%x_%A_%a.out\n#SBATCH --error=logs/%x_%A_%a.err\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=2\n#SBATCH --mem=16G\n#SBATCH --time=04:00:00\n#SBATCH --partition=normal\n#SBATCH --gres=gpu:1\n#SBATCH --array=0-{array_limit}%8\n#SBATCH --export=NONE\n\nset -eo pipefail\n\neval "$(conda shell.bash hook)"\nconda activate rfdxcop\n\nCURRENT_INPUT_DIR="./rfd3_outputs/batch_${{SLURM_ARRAY_TASK_ID}}"\nCURRENT_OUTPUT_DIR="./mpnn_outputs/batch_${{SLURM_ARRAY_TASK_ID}}"\nmkdir -p $CURRENT_OUTPUT_DIR\n\npython scripts/run_mpnn.py --input_dir $CURRENT_INPUT_DIR --out_dir $CURRENT_OUTPUT_DIR --seqs_per_struct {seqs_per_struct}\n\nsed -i "s/ \\[\\]$/ '[]'/g" $CURRENT_OUTPUT_DIR/*.cif\n'''
            with open(os.path.join(base_dir, "scripts", "submit_mpnn.sh"), "w") as f: f.write(slurm_mpnn_code)

        def _write_rf3_scripts(
            self, base_dir, array_limit, topologies=("linear",)
        ):
            selected_topologies = tuple(topologies or ())
            invalid_topologies = set(selected_topologies) - {
                "linear", "cyclic"
            }
            if invalid_topologies or not selected_topologies:
                raise ValueError(
                    "RF3 requires at least one valid topology: linear or "
                    "cyclic."
                )
            topology_flags = " ".join(
                f"--{topology}" for topology in selected_topologies
            )

            run_rf3_code = '''import argparse
import glob
import os
import shutil
import tempfile

from rf3.inference_engines.rf3 import RF3InferenceEngine


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)
parser.add_argument("--out_dir", type=str, required=True)
parser.add_argument("--linear", action="store_true")
parser.add_argument("--cyclic", action="store_true")
args = parser.parse_args()

topologies = [
    topology
    for topology, enabled in (
        ("linear", args.linear),
        ("cyclic", args.cyclic),
    )
    if enabled
]
if not topologies:
    parser.error("select at least one of --linear or --cyclic")

cif_files = sorted(glob.glob(os.path.join(args.input_dir, "*.cif")))
if not cif_files:
    raise SystemExit(f"No MPNN CIF files found in {args.input_dir}")

engine = RF3InferenceEngine(ckpt_path="rf3", verbose=False)
for topology in topologies:
    # RF3 names its outputs after the input stem. Topology-labelled temporary
    # copies keep linear and cyclic results in the same output directory
    # without collisions or an extra directory level.
    with tempfile.TemporaryDirectory(
        prefix=f"xrfd_rf3_{topology}_"
    ) as labeled_input_dir:
        for source_path in cif_files:
            source_name = os.path.basename(source_path)
            source_stem, source_extension = os.path.splitext(source_name)
            labeled_name = f"{source_stem}_{topology}{source_extension}"
            shutil.copy2(
                source_path,
                os.path.join(labeled_input_dir, labeled_name),
            )

        run_options = {
            "inputs": labeled_input_dir,
            "out_dir": args.out_dir,
            "ground_truth_conformer_selection": ["A"],
            "template_selection": ["A"],
        }
        if topology == "cyclic":
            run_options["cyclic_chains"] = ["B"]
        engine.run(**run_options)
'''
            with open(
                os.path.join(base_dir, "scripts", "run_rf3.py"), "w"
            ) as f:
                f.write(run_rf3_code)

            slurm_rf3_code = f'''#!/bin/bash
#SBATCH --job-name=rf3_fold
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --array=0-{array_limit}%8
#SBATCH --export=NONE

set -eo pipefail

eval "$(conda shell.bash hook)"
conda activate rfdxcop

CURRENT_INPUT_DIR="./mpnn_outputs/batch_${{SLURM_ARRAY_TASK_ID}}"
CURRENT_OUTPUT_DIR="./rf3_outputs/batch_${{SLURM_ARRAY_TASK_ID}}"
mkdir -p $CURRENT_OUTPUT_DIR
python scripts/run_rf3.py --input_dir $CURRENT_INPUT_DIR --out_dir $CURRENT_OUTPUT_DIR {topology_flags}
'''
            with open(
                os.path.join(base_dir, "scripts", "submit_rf3.sh"), "w"
            ) as f:
                f.write(slurm_rf3_code)

        def _write_validation_scripts(self, base_dir, array_limit):
            run_val_code = '''import argparse, os, csv, json, re
import numpy as np
import biotite.structure.io.pdbx as pdbx
from biotite.structure import rmsd, superimpose
from atomworks.constants import PROTEIN_BACKBONE_ATOM_NAMES

AA_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

parser = argparse.ArgumentParser()
parser.add_argument("--rf3_dir", type=str, required=True)
parser.add_argument("--rfd3_base_dir", type=str, required=True)
parser.add_argument("--output_csv", type=str, required=True)
args = parser.parse_args()


def get_backbone(arr, chain):
    mask = np.isin(arr.atom_name, PROTEIN_BACKBONE_ATOM_NAMES)
    return arr[mask][arr[mask].chain_id == chain]


def extract_chain_sequence(structure_path, chain):
    try:
        structure = pdbx.get_structure(
            pdbx.CIFFile.read(structure_path), model=1
        )
        ca_atoms = structure[
            (structure.chain_id == chain) & (structure.atom_name == "CA")
        ]
        if len(ca_atoms) == 0:
            return "Error"
        return "".join(
            AA_THREE_TO_ONE.get(res_name, "X")
            for res_name in ca_atoms.res_name
        )
    except Exception:
        return "Error"


def topology_for_name(name):
    without_seed = re.sub(
        r"_seed-\\d+_sample-\\d+$", "", name, flags=re.IGNORECASE
    )
    match = re.search(
        r"_(linear|cyclic)$", without_seed, flags=re.IGNORECASE
    )
    return match.group(1).lower() if match else "linear"


def reference_topology_for_name(name):
    match = re.fullmatch(
        r"Passed_Binder_(linear|cyclic)_\\d+_model_\\d+",
        name,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else "linear"


results = []
current_batch = os.path.basename(os.path.normpath(args.rf3_dir))

for root, _, files in os.walk(args.rf3_dir):
    for f in files:
        if not f.endswith("summary_confidences.json"):
            continue

        json_path = os.path.join(root, f)
        base_name = f.replace("_summary_confidences.json", "")
        model_path = os.path.join(root, base_name + "_model.cif")
        if not os.path.exists(model_path):
            continue

        original_match = re.search(r"(.*?_model_\\d+)", base_name)
        if original_match is None:
            continue
        orig_id = original_match.group(1)
        topology = topology_for_name(base_name)
        reference_topology = reference_topology_for_name(orig_id)
        rfd3_path = os.path.join(
            args.rfd3_base_dir, current_batch, orig_id + ".cif"
        )

        val_rmsd = "Error"
        if os.path.exists(rfd3_path):
            try:
                reference = pdbx.get_structure(
                    pdbx.CIFFile.read(rfd3_path), model=1
                )
                mobile = pdbx.get_structure(
                    pdbx.CIFFile.read(model_path), model=1
                )
                reference_target = get_backbone(reference, "A")
                mobile_target = get_backbone(mobile, "A")
                reference_binder = get_backbone(reference, "B")
                mobile_binder = get_backbone(mobile, "B")

                if (
                    len(reference_target) == len(mobile_target)
                    and len(reference_binder) == len(mobile_binder)
                    and len(reference_target) > 0
                    and len(reference_binder) > 0
                ):
                    _, target_transform = superimpose(
                        reference_target, mobile_target
                    )
                    target_aligned_binder = target_transform.apply(
                        mobile_binder
                    )
                    val_rmsd = rmsd(
                        reference_binder, target_aligned_binder
                    )
            except Exception:
                pass

        try:
            with open(json_path, "r") as jp:
                data = json.load(jp)
            b_ptm = data.get("chain_ptm", [0, 0])[1]
            pae = data.get("chain_pair_pae_min", [[0, 0], [0, 0]])
            pae_min = pae[0][1] if len(pae) > 1 else None
        except Exception:
            b_ptm, pae_min, data = None, None, {}

        results.append({
            "design_name": base_name,
            "topology": topology,
            "reference_topology": reference_topology,
            "batch": current_batch,
            "original_backbone": orig_id,
            "rmsd": val_rmsd,
            "binder_ptm": b_ptm,
            "pae_min": pae_min,
            "iptm": data.get("iptm"),
            "plddt": data.get("overall_plddt"),
            "rf3_subpath": os.path.relpath(root, args.rf3_dir),
            "sequence": extract_chain_sequence(model_path, "B"),
        })

if results:
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
'''
            with open(os.path.join(base_dir, "scripts", "run_validation.py"), "w") as f: f.write(run_val_code)
            slurm_val_code = f'''#!/bin/bash\n#SBATCH --job-name=validate\n#SBATCH --output=logs/%x_%A_%a.out\n#SBATCH --error=logs/%x_%A_%a.err\n#SBATCH --ntasks=1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=8G\n#SBATCH --time=04:00:00\n#SBATCH --partition=cpu\n#SBATCH --array=0-{array_limit}\n#SBATCH --export=NONE\n\nset -eo pipefail\n\neval "$(conda shell.bash hook)"\nconda activate rfdxcop\n\nmkdir -p ./scores\npython scripts/run_validation.py --rf3_dir ./rf3_outputs/batch_${{SLURM_ARRAY_TASK_ID}} --rfd3_base_dir ./rfd3_outputs --output_csv ./scores/scores_batch_${{SLURM_ARRAY_TASK_ID}}.csv\n'''
            with open(os.path.join(base_dir, "scripts", "submit_validation_cpu.sh"), "w") as f: f.write(slurm_val_code)

        def _write_consolidation_script(
            self,
            base_dir,
            rmsd_cutoff,
            ptm_cutoff,
            pae_cutoff,
            ignore_cyclic_rmsd=True,
        ):
            code = '''import csv
import glob
import os
import re
import shutil
import warnings

import biotite.structure.io.pdbx as pdbx

warnings.filterwarnings("ignore")

SCORES_DIR = "./scores"
RF3_ROOT = "./rf3_outputs"
OUTPUT_DIR = "./successful_designs"
MASTER_CSV = os.path.join(SCORES_DIR, "MASTER_SCORES_CLEAN.csv")
MASTER_READABLE = os.path.join(SCORES_DIR, "MASTER_SCORES_READABLE.txt")
UNIQUE_CSV = os.path.join(SCORES_DIR, "Unique_designs.csv")
UNIQUE_TXT = os.path.join(SCORES_DIR, "Unique_designs.txt")
TARGET_CHAIN = "B"

RMSD_CUTOFF = __RMSD_CUTOFF__
PTM_CUTOFF = __PTM_CUTOFF__
PAE_CUTOFF = __PAE_CUTOFF__
IGNORE_CYCLIC_RMSD = __IGNORE_CYCLIC_RMSD__

MASTER_COLUMNS = [
    "design_name", "topology", "reference_topology", "sample_type",
    "rmsd", "binder_ptm", "pae_min", "iptm", "plddt", "batch", "sequence",
]

NUMERIC_COLUMNS = {"rmsd", "binder_ptm", "pae_min", "iptm", "plddt"}

AA_MAP = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def batch_sort_key(path):
    name = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, int(name.rsplit("_", 1)[1]))
    except (IndexError, ValueError):
        return (1, name)


def sample_type_for(row):
    existing = str(row.get("sample_type", "") or "").strip()
    if existing:
        return existing

    subpath = str(row.get("rf3_subpath", "") or "").replace("\\\\", "/")
    design_name = str(row.get("design_name", "") or "")
    if subpath == design_name:
        return "Top-Level"
    return subpath.rstrip("/").rsplit("/", 1)[-1] or "Unknown"


def topology_for(row):
    existing = str(row.get("topology", "") or "").strip().lower()
    if existing in {"linear", "cyclic"}:
        return existing

    design_name = str(row.get("design_name", "") or "")
    without_seed = re.sub(
        r"_seed-\\d+_sample-\\d+$", "", design_name, flags=re.IGNORECASE
    )
    match = re.search(
        r"_(linear|cyclic)$", without_seed, flags=re.IGNORECASE
    )
    return match.group(1).lower() if match else "linear"


def reference_topology_for(row):
    existing = str(
        row.get("reference_topology", "") or ""
    ).strip().lower()
    if existing in {"linear", "cyclic"}:
        return existing

    original_backbone = str(row.get("original_backbone", "") or "")
    match = re.fullmatch(
        r"Passed_Binder_(linear|cyclic)_\\d+_model_\\d+",
        original_backbone,
        flags=re.IGNORECASE,
    )
    return match.group(1).lower() if match else "linear"


def normalized_master_row(row):
    normalized = {
        "design_name": row.get("design_name", ""),
        "topology": topology_for(row),
        "reference_topology": reference_topology_for(row),
        "sample_type": sample_type_for(row),
        "rmsd": row.get("rmsd", ""),
        "binder_ptm": row.get("binder_ptm", ""),
        "pae_min": row.get("pae_min", ""),
        "iptm": row.get("iptm", ""),
        "plddt": row.get("plddt", ""),
        "batch": row.get("batch", ""),
        "sequence": row.get("sequence", ""),
    }
    if not normalized["sequence"]:
        cif_path = find_cif(row)
        normalized["sequence"] = (
            extract_sequence_from_cif(cif_path, chain_id=TARGET_CHAIN)
            if cif_path and os.path.exists(cif_path)
            else "FileNotFound"
        )
    for column in NUMERIC_COLUMNS:
        numeric_value = number(normalized.get(column))
        normalized[column] = (
            numeric_value if numeric_value is not None else ""
        )
    return normalized


def formatted_value(row, column):
    value = row.get(column, "")
    if column in NUMERIC_COLUMNS:
        numeric_value = number(value)
        return f"{numeric_value:.3f}" if numeric_value is not None else "-"
    return str(value)


def write_readable_table(rows, path, columns):
    formatted_rows = [
        [formatted_value(row, column) for column in columns]
        for row in rows
    ]
    widths = [len(column) for column in columns]
    for values in formatted_rows:
        widths = [
            max(width, len(value))
            for width, value in zip(widths, values)
        ]

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("  ".join(
            column.ljust(width)
            for column, width in zip(columns, widths)
        ) + "\\n")
        for values in formatted_rows:
            handle.write("  ".join(
                value.ljust(width)
                for value, width in zip(values, widths)
            ) + "\\n")


def master_needs_rebuild(batch_files):
    if not os.path.exists(MASTER_CSV) or not os.path.exists(MASTER_READABLE):
        return True
    if os.path.getmtime(MASTER_CSV) < max(os.path.getmtime(p) for p in batch_files):
        return True
    try:
        with open(MASTER_CSV, "r", newline="", encoding="utf-8-sig") as handle:
            fields = csv.DictReader(handle).fieldnames or []
        return any(column not in fields for column in MASTER_COLUMNS)
    except OSError:
        return True


def rebuild_master(batch_files):
    temporary_path = MASTER_CSV + ".tmp"
    rows = []
    for path in batch_files:
        with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as source:
            rows.extend(
                normalized_master_row(row)
                for row in csv.DictReader(source)
            )

    rows.sort(
        key=lambda row: (
            number(row.get("binder_ptm"))
            if number(row.get("binder_ptm")) is not None
            else float("-inf")
        ),
        reverse=True,
    )
    with open(temporary_path, "w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=MASTER_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, MASTER_CSV)
    write_readable_table(rows, MASTER_READABLE, MASTER_COLUMNS)
    print(f"Rebuilt master score cache: {len(rows)} rows")


def iter_score_rows(paths):
    for path in paths:
        with open(path, "r", newline="", encoding="utf-8-sig", errors="replace") as handle:
            for row in csv.DictReader(handle):
                yield normalized_master_row(row)


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


batch_indexes = {}


def index_batch_models(batch):
    if batch in batch_indexes:
        return batch_indexes[batch]

    index = {}
    batch_dir = os.path.join(RF3_ROOT, batch)
    if os.path.isdir(batch_dir):
        for root, _dirs, files in os.walk(batch_dir):
            for filename in files:
                if filename.endswith("_model.cif"):
                    design_name = filename[:-len("_model.cif")]
                    index.setdefault(design_name, os.path.join(root, filename))
    batch_indexes[batch] = index
    return index


def direct_result_folder(row):
    batch = str(row.get("batch", ""))
    design_name = str(row.get("design_name", ""))
    raw_subpath = str(row.get("rf3_subpath", "") or "").replace("\\\\", "/")

    if raw_subpath:
        parts = [part for part in raw_subpath.split("/") if part not in {"", ".", ".."}]
        return os.path.join(RF3_ROOT, batch, *parts)

    sample_type = sample_type_for(row)
    if sample_type == "Top-Level":
        return os.path.join(RF3_ROOT, batch, design_name)

    suffix = "_" + sample_type
    parent_design = (
        design_name[:-len(suffix)]
        if design_name.endswith(suffix) else design_name
    )
    return os.path.join(RF3_ROOT, batch, parent_design, sample_type)


def find_cif(row):
    batch = str(row.get("batch", ""))
    design_name = str(row.get("design_name", ""))
    folder = direct_result_folder(row)
    exact_path = os.path.join(folder, design_name + "_model.cif")
    if os.path.isfile(exact_path):
        return exact_path

    if os.path.isdir(folder):
        try:
            for entry in os.scandir(folder):
                if entry.is_file() and entry.name.endswith("_model.cif"):
                    return entry.path
        except OSError:
            pass

    # Build at most one recursive index per affected batch, only when the
    # deterministic path does not exist.
    return index_batch_models(batch).get(design_name)


def extract_sequence_from_cif(filepath, chain_id=TARGET_CHAIN):
    """Extract the binder sequence exactly as in the reference notebook."""
    try:
        cif_file = pdbx.CIFFile.read(filepath)
        structure = pdbx.get_structure(cif_file, model=1)
        chains = list(set(structure.chain_id))
        target_chain = chain_id if chain_id in chains else sorted(chains)[-1]

        mask = (
            (structure.chain_id == target_chain)
            & (structure.atom_name == "CA")
        )
        ca_atoms = structure[mask]
        if len(ca_atoms) == 0:
            return "EmptyChain"
        return "".join(AA_MAP.get(residue, "X") for residue in ca_atoms.res_name)
    except Exception as error:
        return f"Err:{str(error)[:20]}"


batch_files = sorted(
    glob.glob(os.path.join(SCORES_DIR, "scores_batch_*.csv")),
    key=batch_sort_key,
)

if len(batch_files) > 1:
    if master_needs_rebuild(batch_files):
        rebuild_master(batch_files)
    else:
        print("Using unchanged master score cache")
    score_sources = [MASTER_CSV]
elif len(batch_files) == 1:
    score_sources = batch_files
elif os.path.exists(MASTER_CSV):
    score_sources = [MASTER_CSV]
else:
    raise SystemExit("No score CSV files were found")

passed_rows = []
total_rows = 0
for row in iter_score_rows(score_sources):
    total_rows += 1
    rmsd = number(row.get("rmsd"))
    ptm = number(row.get("binder_ptm"))
    pae = number(row.get("pae_min"))
    rmsd_is_ignored = (
        IGNORE_CYCLIC_RMSD
        and row.get("topology") == "cyclic"
        and row.get("reference_topology") != "cyclic"
    )
    rmsd_passes = rmsd_is_ignored or (
        rmsd is not None and rmsd <= RMSD_CUTOFF
    )
    if (
        rmsd_passes
        and ptm is not None and ptm >= PTM_CUTOFF
        and pae is not None and pae <= PAE_CUTOFF
    ):
        row["_rmsd"] = rmsd if rmsd is not None else float("inf")
        row["_ptm"] = ptm
        row["_pae"] = pae
        passed_rows.append(row)

print(f"Filtered {total_rows} rows: {len(passed_rows)} passed")
if not passed_rows:
    print("No designs passed the filters; stopping extraction")
    raise SystemExit(0)

# The notebook extracts sequences only after filtering, then retains the
# highest-pTM / lowest-PAE / lowest-RMSD representative of each sequence and
# topology. The same sequence may be retained once as linear and once as cyclic.
sequence_cache = {}
resolved_rows = []
missing_count = 0
for row in passed_rows:
    cif_path = find_cif(row)
    if cif_path and os.path.exists(cif_path):
        if cif_path not in sequence_cache:
            sequence_cache[cif_path] = extract_sequence_from_cif(
                cif_path, chain_id=TARGET_CHAIN
            )
        row["sequence"] = sequence_cache[cif_path]
        row["_cif_path"] = cif_path
    else:
        row["sequence"] = "FileNotFound"
        row["_cif_path"] = "NotFound"
        missing_count += 1
    resolved_rows.append(row)

resolved_rows.sort(
    key=lambda row: (-row["_ptm"], row["_pae"], row["_rmsd"])
)
unique_by_sequence = {}
for row in resolved_rows:
    sequence_key = (row.get("topology", "linear"), row["sequence"])
    unique_by_sequence.setdefault(sequence_key, row)
unique_rows = list(unique_by_sequence.values())

print(
    f"Sequence deduplication retained {len(unique_rows)} unique designs "
    f"from {len(passed_rows)} passing rows"
)

# Match the notebook's clean-start behavior for successful_designs.
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

copied_count = 0
for row in unique_rows:
    cif_path = row["_cif_path"]
    if cif_path == "NotFound" or not os.path.exists(cif_path):
        continue

    sample_type = sample_type_for(row).replace("/", "_").replace("\\\\", "_")
    filename = f"{row.get('batch', '')}_{row.get('design_name', '')}_{sample_type}.cif"
    destination = os.path.join(OUTPUT_DIR, filename)
    try:
        shutil.copy2(cif_path, destination)
        copied_count += 1
    except OSError as error:
        print(f"Could not copy {cif_path}: {error}")

unique_columns = MASTER_COLUMNS
with open(UNIQUE_CSV, "w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=unique_columns)
    writer.writeheader()
    writer.writerows(
        {column: row.get(column, "") for column in unique_columns}
        for row in unique_rows
    )
write_readable_table(unique_rows, UNIQUE_TXT, unique_columns)

print(
    f"Copied {copied_count} sequence-unique CIF files; "
    f"missing source files: {missing_count}"
)
print(f"Saved unique design lists to {UNIQUE_CSV} and {UNIQUE_TXT}")
'''
            code = code.replace(
                "__RMSD_CUTOFF__", repr(float(rmsd_cutoff))
            ).replace(
                "__PTM_CUTOFF__", repr(float(ptm_cutoff))
            ).replace(
                "__PAE_CUTOFF__", repr(float(pae_cutoff))
            ).replace(
                "__IGNORE_CYCLIC_RMSD__",
                repr(bool(ignore_cyclic_rmsd)),
            )
            with open(os.path.join(base_dir, "scripts", "run_consolidation.py"), "w") as f: f.write(code)


    class RFDiffusionWindow(RFDiffusionTab):
        """Standalone RF Diffusion UI with a private structure detector."""

        def __init__(self, parent=None):
            detector = RFStructureDetector()
            super().__init__(detector, parent)
            self.structure_detector = detector
            # Keep the detector alive but hidden: it supplies layer and
            # protofilament state to the RF preview without exposing Modifier.
            self.structure_detector.setObjectName("rfStructureDetector")

        def closeEvent(self, event):
            self.structure_detector.close()
            super().closeEvent(event)

    # Tests and embedding callers can obtain the exact components without
    # creating windows or entering QApplication.exec().
    if not launch:
        return {
            "ScoreTableModel": ScoreTableModel,
            "RFStructureDetector": RFStructureDetector,
            "GPUAcceleratedPreviewCanvas": GPUAcceleratedPreviewCanvas,
            "SoftwarePreviewCanvas": SoftwarePreviewCanvas,
            "PreviewCanvas": PreviewCanvas,
            "SoftwareChainCanvas": SoftwareChainCanvas,
            "ChainCanvas": ChainCanvasBase,
            "QuickLookStructureReader": QuickLookStructureReader,
            "ScoreQuickLookWindow": ScoreQuickLookWindow,
            "RFDiffusionTab": RFDiffusionTab,
            "RFDiffusionWindow": RFDiffusionWindow
        }

    qt_app = QApplication.instance()
    if not qt_app:
        qt_app = QApplication(sys.argv)
    
    qt_app.setStyleSheet("""
        QMainWindow, QDialog, QWidget { background-color: #1e1e1e; color: #d4d4d4; font-family: Arial; font-size: 8pt; }
        
        QPushButton { background-color: #3e3e42; color: #d4d4d4; border: 1px solid #3e3e42; padding: 5px 15px; border-radius: 4px; }
        QPushButton:hover { background-color: #4e4e52; border: 1px solid #98c379; }
        QPushButton:pressed, QPushButton:checked { background-color: #98c379; color: #1e1e1e; border: 1px solid #98c379; }

        QPushButton#autoRunButton,
        QPushButton#cancelJobsButton {
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


    win = RFDiffusionWindow()
    renderer_suffix = " (Software Renderer)" if use_software_renderer else ""
    win.setWindowTitle(f"RF Diffusion{renderer_suffix}")
    win.resize(1450, 850)
    win.show()

    sys.exit(qt_app.exec())

if __name__ == "__main__":
    open_rf_diffusion()
