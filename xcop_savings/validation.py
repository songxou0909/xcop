import sys
import os
import re
import math
import subprocess
import gc
import traceback
import json
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QFileDialog, QGroupBox, 
    QGridLayout, QTableWidget, QTableWidgetItem, QHeaderView, 
    QMessageBox, QAbstractItemView, QCheckBox, 
    QDialog, QRadioButton, QDialogButtonBox, QFileIconProvider
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QFileInfo, QTimer, QLineF
from PyQt6.QtGui import QColor, QPalette, QFont, QPainter, QPixmap, QIcon, QPen
import numpy as np

# ==============================================================================
# 1. SETUP & LOGGING
# ==============================================================================
import argparse

parser = argparse.ArgumentParser(description="MolProbity+")
parser.add_argument("--phenix-ver", type=str, default="phenix", help="Phenix module version to load")
args, unknown = parser.parse_known_args()

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Set Phenix version dynamically from the caller
PHENIX_VER = args.phenix_ver

class PDBVisualizer(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.coords = []
        self.max_dist = 1.0
        self.angle_y = 0.0
        self.angle_x = 0.35  # Tilted product angle showcase
        
        # Drive smooth auto-rotation animation natively
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_rotation)
        self.timer.start(30)  # ~33 FPS

    def load_pdb(self, filepath):
        if not filepath or not os.path.exists(filepath) or not filepath.lower().endswith('.pdb'):
            return
        coords = []
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Trace alpha-carbon (CA) backbone chain
                    if line.startswith("ATOM  ") and line[12:16].strip() == "CA":
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            coords.append([x, y, z])
                        except ValueError:
                            continue
        except Exception as e:
            print(f"Error loading structure: {e}")
            return

        if not coords:
            self.coords = []
            self.update()
            return

        # Convert to NumPy for extreme high-performance C-level matrix math
        self.coords_np = np.array(coords, dtype=np.float32)
        
        # Normalize and center
        center = np.mean(self.coords_np, axis=0)
        self.coords_np -= center
        self.max_dist = float(np.max(np.linalg.norm(self.coords_np, axis=1)))
        
        # PRE-CALCULATE: Valid connections (CA-CA dist < 5.0 Å) to fix spiderwebs
        diffs = self.coords_np[1:] - self.coords_np[:-1]
        dists_sq = np.sum(diffs**2, axis=1)
        self.valid_edges = np.where(dists_sq <= 25.0)[0]
        
        # PRE-CALCULATE: Hues for the pastel N-to-C terminal color spectrum
        num_points = len(self.coords_np)
        self.hues = (np.arange(num_points) / max(1, num_points - 1)) * 0.85
        
        self.update()

    def update_rotation(self):
        if hasattr(self, 'coords_np') and len(self.coords_np) > 0:
            self.angle_y += 0.015  # Moderate, elegant presentation speed
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#2b2b2b"))
        
        if not hasattr(self, 'coords_np') or len(self.coords_np) == 0:
            painter.setPen(QColor("#666666"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Active PDB Loaded")
            return

        w, h = self.width(), self.height()
        scale = min(w, h) / (2.0 * self.max_dist) * 0.85
        
        # Create rotation matrices
        cos_y, sin_y = np.cos(self.angle_y), np.sin(self.angle_y)
        cos_x, sin_x = np.cos(self.angle_x), np.sin(self.angle_x)
        
        Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]], dtype=np.float32)
        Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]], dtype=np.float32)
        R = Rx @ Ry
        
        # INSTANT VECTORIZED PROJECTION (Replaces the slow Python math loop)
        rotated = self.coords_np @ R.T
        u = w / 2.0 + rotated[:, 0] * scale
        v = h / 2.0 - rotated[:, 1] * scale
        z = rotated[:, 2]

        # DYNAMIC LOD (Level of Detail): 
        # If structure is massive (>15,000 bonds), dynamically stride to preserve GUI frame rate.
        # At this density, the skipped lines blend seamlessly into the cloud visually.
        stride = 1
        if len(self.valid_edges) > 15000:
            stride = max(1, len(self.valid_edges) // 15000)

        # Extract only valid, pre-computed segments
        i1 = self.valid_edges[::stride]
        i2 = i1 + 1

        u1, v1, z1 = u[i1], v[i1], z[i1]
        u2, v2, z2 = u[i2], v[i2], z[i2]
        
        # Vectorized depth calculations
        avg_z = (z1 + z2) / 2.0
        depth_factors = np.clip((avg_z + self.max_dist) / (2.0 * self.max_dist), 0.25, 1.0)
        hues = self.hues[i1]

        # Draw the batched arrays
        for idx in range(len(i1)):
            df = float(depth_factors[idx])
            color = QColor.fromHsvF(float(hues[idx]), 0.40, float(0.95 * df))
            pen_width = max(1, int(4.5 * df))
            
            painter.setPen(QPen(color, pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(QLineF(float(u1[idx]), float(v1[idx]), float(u2[idx]), float(v2[idx])))

class Logger:
    def __init__(self):
        self.log_paths = []

    def setup(self, input_pdb_path):
        self.log_paths = [] 

    def log(self, message):
        try:
            print(message[:200] + "..." if len(message) > 200 else message)
        except: 
            pass 

logger = Logger()

# ==============================================================================
# 2. BACKEND
# ==============================================================================
def run_tool(tool_name, args, redirect_output_path=None):
    cmd_str = f"{tool_name} {args}"
    full_cmd = f"export OMP_NUM_THREADS=1 && module load {PHENIX_VER} && {cmd_str}"

    logger.log(f"--- RUNNING: {tool_name} ---")

    try:
        # FIX: capture as BINARY (text=False) to prevent UnicodeDecodeError on tool output
        result = subprocess.run(full_cmd, shell=True, executable="/bin/bash", capture_output=True, text=False)
        
        # Manually decode with error replacement
        stdout_str = result.stdout.decode('utf-8', errors='replace') if result.stdout else ""
        stderr_str = result.stderr.decode('utf-8', errors='replace') if result.stderr else ""

        # Log truncated output
        if stdout_str: 
            if len(stdout_str) > 5000:
                logger.log(f">>> STDOUT ({tool_name}): [Truncated large output: {len(stdout_str)} chars]")
            else:
                logger.log(f">>> STDOUT ({tool_name}):\n{stdout_str}")

        if stderr_str: logger.log(f">>> STDERR ({tool_name}):\n{stderr_str}")
        
        # 2. Write file if requested
        if redirect_output_path:
            if stdout_str and len(stdout_str) > 50:
                try:
                    # FIX: Write with explicit encoding
                    with open(redirect_output_path, "w", encoding='utf-8') as f:
                        f.write(stdout_str)
                except Exception as e:
                    logger.log(f"ERROR writing file: {e}")
            return ""

        return stdout_str

    except Exception as e:
        logger.log(f"CRITICAL ERROR executing {tool_name}: {e}")
        return ""

def parse_val(regex, text, default=0.0):
    if not text: return default 
    match = re.search(regex, text)
    if match: return float(match.group(1))
    return default

class AnalysisBackend:
    def __init__(self):
        self.residue_data = {} 
        self.summary_data = {}
        self.db_file = ""

    def load_db(self):
        """Load data from the text file into memory."""
        try:
            if os.path.exists(self.db_file):
                with open(self.db_file, 'r', encoding='utf-8', errors='ignore') as f:
                    data = json.load(f)
                    self.residue_data = data.get("residues", {})
                    if "summary" in data and data["summary"]:
                        self.summary_data = data["summary"]
            else:
                self.residue_data = {}
        except:
            self.residue_data = {}

    def save_db(self):
        """Save memory to text file, then CLEAR memory."""
        try:
            full_data = {
                "summary": self.summary_data,
                "residues": self.residue_data
            }
            with open(self.db_file, 'w', encoding='utf-8') as f:
                json.dump(full_data, f, indent=2)
            
            self.residue_data.clear() 
            gc.collect()              
        except Exception as e:
            logger.log(f"Error saving DB: {e}")

    def get_entry(self, chain, res):
        chain, res = chain.strip(), res.strip()
        key = f"{chain}|{res}"
        if key not in self.residue_data:
            self.residue_data[key] = {
                "Chain": chain, "Res": res, "Type": "???",
                "Bfactor": 0.0, "Clash": "", "Rama": "", 
                "Rotamer": "", "Cbeta": "", "CaBLAM": "", 
                "Bonds": "", "Angles": "", "Cis": ""
            }
        return self.residue_data[key]

    def run_full_analysis(self, input_pdb, status_callback, do_renumber=True, do_cleanup=False):
        logger.setup(input_pdb)
        self.residue_data = {}
        self.summary_data = {
            'clashscore': 0.0, 'rota_out_pct': 0.0, 'rota_fav_pct': 0.0,
            'rama_out_pct': 0.0, 'rama_fav_pct': 0.0, 'rama_z': "N/A", 'mp_score': 0.0,
            'cb_out_cnt': 0, 'cis_pro_cnt': 0, 'cis_pro_total': 0,
            'twisted_cnt': 0, 'twisted_total': 0, 'cab_out_cnt': 0, 'cab_total': 0,
            'ca_geo_cnt': 0, 'ca_geo_total': 0, 'b_out_cnt': 0, 'b_total': 0, 
            'a_out_cnt': 0, 'a_total': 0
        }
        
        abs_input = os.path.abspath(input_pdb)
        work_dir = os.path.dirname(abs_input)
        base_name = os.path.splitext(os.path.basename(abs_input))[0]
        
        stripped_pdb = os.path.join(work_dir, f"{base_name}_cleanH.pdb")
        final_pdb = os.path.join(work_dir, f"{base_name}_withH.pdb")
        generated_files = [stripped_pdb, final_pdb]
        
        self.db_file = os.path.join(work_dir, f"{base_name}_Molprobity.txt")
        with open(self.db_file, 'w', encoding='utf-8') as f: json.dump({}, f)

        # 1. STRIP H
        status_callback("Removing Hydrogens...")
        run_tool("phenix.pdbtools", f"\"{abs_input}\" remove=\"element H\" output.file_name=\"{stripped_pdb}\"")
        if not os.path.exists(stripped_pdb): stripped_pdb = abs_input

        # 2. ADD H
        status_callback("Adding Hydrogens...")
        run_tool("phenix.reduce", f"\"{stripped_pdb}\" -NoFlip", redirect_output_path=final_pdb)
        if not os.path.exists(final_pdb):
             run_tool("phenix.reduce", f"\"{stripped_pdb}\" flip=False", redirect_output_path=final_pdb)
        
        # 2b. RENUMBER
        if os.path.exists(final_pdb) and do_renumber:
            status_callback("Renumbering atoms...")
            renum_pdb = os.path.join(work_dir, f"{base_name}_renumH.pdb")
            try:
                # FIX: Use errors='ignore' to handle potential non-UTF8 chars from Phenix
                with open(final_pdb, 'r', encoding='utf-8', errors='ignore') as fin, \
                     open(renum_pdb, 'w', encoding='utf-8') as fout:
                    serial = 1
                    for line in fin:
                        if line.startswith("ATOM") or line.startswith("HETATM"):
                            # FIX: Handle serial > 99999 gracefully (modulo 100000 or truncate)
                            # Standard PDB practice for >99999 is Hybrid-36 or wrapping. 
                            # Here we simply wrap modulo 100000 to keep alignment.
                            serial_val = serial % 100000 
                            new_serial = f"{serial_val:5d}" 
                            line = line[:6] + new_serial + line[11:]
                            serial += 1
                        fout.write(line)
                final_pdb = renum_pdb
            except Exception as e:
                logger.log(f"WARNING: Renumbering failed: {e}")

        if not os.path.exists(final_pdb): final = stripped_pdb
        else: final = final_pdb

        # 3. READ RESIDUES
        self.load_db() # LOAD
        status_callback("Reading Residues...")
        try:
            # FIX: Use errors='ignore' to prevent crash on reading large files with potential noise
            with open(final, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if line.startswith("ATOM") or line.startswith("HETATM"):
                        chain = line[20:22].strip()
                        resnum = line[22:26].strip()
                        icode = line[26].strip()
                        resname = line[17:20].strip()
                        
                        entry = self.get_entry(chain, resnum + icode)
                        entry["Type"] = resname
                        try:
                            b = float(line[60:66])
                            if b > entry["Bfactor"]: entry["Bfactor"] = b
                        except: pass
        except Exception as e:
            # FIX: Print traceback to identify why reading fails
            logger.log(f"CRITICAL ERROR Reading Residues in Step 3: {e}")
            traceback.print_exc()

        self.save_db() # SAVE & CLEAR

        # 4. CLASHSCORE
        self.load_db() # LOAD
        status_callback("Running Clashscore...")
        out_clash = run_tool("phenix.clashscore", f"\"{final}\" keep_hydrogens=True verbose=True")
        self.summary_data['clashscore'] = parse_val(r"clashscore\s*=\s*([\d\.]+)", out_clash)
        
        for line in out_clash.splitlines():
            if ":" in line and "Bad Clashes" not in line:
                parts = line.split(":")
                if len(parts) >= 2:
                    try:
                        overlap = float(parts[-1].strip())
                        if overlap >= 0.4:
                            tokens = parts[0].strip().split()
                            if len(tokens) >= 8:
                                e1 = self.get_entry(tokens[0], tokens[1])
                                if not e1["Clash"] or float(e1["Clash"]) < overlap: e1["Clash"] = f"{overlap:.2f}"
                                e2 = self.get_entry(tokens[4], tokens[5])
                                if not e2["Clash"] or float(e2["Clash"]) < overlap: e2["Clash"] = f"{overlap:.2f}"
                    except: pass
        
        del out_clash 
        self.save_db() 

        # 5. RAMALYZE
        self.load_db() 
        status_callback("Running Ramachandran...")
        out_rama = run_tool("phenix.ramalyze", f"\"{final}\"")
        rama_sum = re.search(r"SUMMARY:\s*(\d+)\s*Favored,\s*(\d+)\s*Allowed,\s*(\d+)\s*Outlier\s*out of\s*(\d+)", out_rama)
        if rama_sum:
            self.summary_data['rama_fav_cnt'] = int(rama_sum.group(1))
            self.summary_data['rama_out_cnt'] = int(rama_sum.group(3))
            self.summary_data['rama_tot'] = int(rama_sum.group(4))
            if self.summary_data['rama_tot'] > 0:
                self.summary_data['rama_fav_pct'] = (self.summary_data['rama_fav_cnt'] / self.summary_data['rama_tot']) * 100.0
                self.summary_data['rama_out_pct'] = (self.summary_data['rama_out_cnt'] / self.summary_data['rama_tot']) * 100.0
        else:
            self.summary_data['rama_fav_pct'] = parse_val(r"SUMMARY:\s*([\d\.]+)%\s*favored", out_rama)
            self.summary_data['rama_out_pct'] = parse_val(r"SUMMARY:\s*([\d\.]+)%\s*outliers", out_rama)
        
        for line in out_rama.splitlines():
            if ":" in line and "SUMMARY" not in line and not line.strip().startswith("residue"):
                parts = line.split(":")
                if len(parts) >= 6:
                    status = parts[4].strip().upper()
                    score = parts[1].strip()
                    res_info = parts[0].strip().split()
                    if len(res_info) >= 2:
                        e = self.get_entry(res_info[0], res_info[1])
                        if "OUTLIER" in status: e["Rama"] = f"OUTLIER\n({score}%)"
                        elif "ALLOWED" in status: e["Rama"] = f"Allowed\n({score}%)"
                        elif "FAVORED" in status: e["Rama"] = f"Favored\n({score}%)"
        
        del out_rama
        self.save_db()

        # 6. RAMA-Z
        self.load_db()
        out_z = run_tool("mmtbx.rama_z", f"\"{final}\"")
        z_match = re.search(r"whole:\s*([\-\d\.]+)\s*\(([\d\.]+)\)", out_z)
        if z_match: self.summary_data['rama_z'] = f"{z_match.group(1)} ± {z_match.group(2)}"
        self.save_db() 

        # 7. ROTALYZE
        self.load_db()
        status_callback("Running Rotalyze...")
        out_rota = run_tool("phenix.rotalyze", f"\"{final}\"")
        rota_cnts = {"Favored":0, "Allowed":0, "OUTLIER":0, "Total":0}
        for line in out_rota.splitlines():
            if ":" in line and "SUMMARY" not in line and not line.strip().startswith("residue"):
                parts = line.split(":")
                if len(parts) >= 8:
                    status = parts[-2].strip()
                    rota_cnts["Total"] += 1
                    if "Favored" in status: rota_cnts["Favored"] += 1
                    elif "Allowed" in status: rota_cnts["Allowed"] += 1
                    elif "OUTLIER" in status: rota_cnts["OUTLIER"] += 1
                    
                    score = parts[2].strip()
                    res_info = parts[0].strip().split()
                    if len(res_info) >= 2:
                        e = self.get_entry(res_info[0], res_info[1])
                        if "OUTLIER" in status: e["Rotamer"] = f"OUTLIER\n({score}%)"
                        elif "Allowed" in status: e["Rotamer"] = f"Allowed\n({score}%)"
                        elif "Favored" in status: e["Rotamer"] = f"Favored\n({score}%)"

        self.summary_data['rota_tot'] = rota_cnts["Total"]
        self.summary_data['rota_fav_cnt'] = rota_cnts["Favored"]
        self.summary_data['rota_out_cnt'] = rota_cnts["OUTLIER"]
        if rota_cnts["Total"] > 0:
            self.summary_data['rota_fav_pct'] = (rota_cnts["Favored"] / rota_cnts["Total"]) * 100.0
            self.summary_data['rota_out_pct'] = (rota_cnts["OUTLIER"] / rota_cnts["Total"]) * 100.0
        
        del out_rota
        self.save_db()

        # 8. CBETA
        self.load_db()
        status_callback("Running C-Beta...")
        out_cb = run_tool("phenix.cbetadev", f"\"{final}\"")
        cb_match = re.search(r"SUMMARY:\s*(\d+)\s*C-beta deviations", out_cb)
        if cb_match: self.summary_data['cb_out_cnt'] = int(cb_match.group(1))

        for line in out_cb.splitlines():
            if line.count(":") >= 5 and not line.strip().startswith("pdb"):
                parts = line.split(":")
                try:
                    dev = float(parts[5].strip())
                    e = self.get_entry(parts[3].strip(), parts[4].strip())
                    e["Cbeta"] = f"{dev:.2f}Å"
                except: pass
        
        del out_cb
        self.save_db()

        # 9. CABLAM
        self.load_db()
        status_callback("Running CaBLAM...")
        out_cab = run_tool("phenix.cablam", f"\"{final}\"")
        
        cab_match = re.search(r"SUMMARY:\s*(\d+)\s*residues.*have outlier conformations", out_cab)
        if cab_match: self.summary_data['cab_out_cnt'] = int(cab_match.group(1))
        
        ca_geo_match = re.search(r"SUMMARY:\s*(\d+)\s*residues.*severe CA geometry outliers", out_cab)
        if ca_geo_match: self.summary_data['ca_geo_cnt'] = int(ca_geo_match.group(1))
        
        cab_tot_match = re.search(r"SUMMARY: CaBLAM found\s*(\d+)\s*full protein residues", out_cab)
        if cab_tot_match: self.summary_data['ca_geo_total'] = int(cab_tot_match.group(1))

        for line in out_cab.splitlines():
            if ":" in line and not line.strip().startswith("residue"):
                parts = line.split(":")
                if len(parts) >= 4:
                    outlier_type = parts[1].strip()

                    res_info = parts[0].strip().split()
                    if len(res_info) >= 2:
                        e = self.get_entry(res_info[0], res_info[1])
                        if "CA Geom" in outlier_type:
                            try:
                                score = float(parts[3].strip()) * 100
                                e["CaBLAM"] = f"CA Geom Outlier ({score:.3f}%)"
                            except: e["CaBLAM"] = "CA Geom Outlier"
                        elif "CaBLAM" in outlier_type:
                            if "Geom" in e["CaBLAM"]: continue
                            try:
                                score = float(parts[2].strip()) * 100
                                lbl = "Outlier" if "Outlier" in outlier_type else "Disfavored"
                                e["CaBLAM"] = f"CaBLAM {lbl} ({score:.3f}%)"
                            except: e["CaBLAM"] = outlier_type
                        elif not outlier_type:
                            if e["CaBLAM"] and "Geom" in e["CaBLAM"]: continue
                            try:
                                score = float(parts[2].strip()) * 100
                                e["CaBLAM"] = f"CaBLAM Favored ({score:.3f}%)"
                            except: pass
        
        del out_cab
        self.save_db()

        # 10. GEOMETRY
        self.load_db()
        status_callback("Running Geometry...")
        out_stats = run_tool("phenix.model_statistics", f"\"{final}\"")
        b_tot = parse_val(r"Bond\s*:\s*[\d\.]+\s+[\d\.]+\s+(\d+)", out_stats)
        a_tot = parse_val(r"Angle\s*:\s*[\d\.]+\s+[\d\.]+\s+(\d+)", out_stats)
        self.summary_data['b_total'] = int(b_tot)
        self.summary_data['a_total'] = int(a_tot)

        out_geo = run_tool("mmtbx.mp_geo", f"pdb=\"{final}\" bonds_and_angles=True")
        
        out_cab = None
        out_stats = None
        gc.collect()

        b_cnt = 0
        a_cnt = 0
        BOND_SIGMA_LIMIT = 4.0
        ANGLE_SIGMA_LIMIT = 4.0

        for line in out_geo.splitlines():
            line = line.strip()
            if not line: continue
            try:
                parts = line.rsplit(":", 3)
                if len(parts) < 4: continue
                sigma_str = parts[2].strip()
                sub_parts = parts[0].rsplit(":", 1)
                if len(sub_parts) < 2: continue
                prefix_residue = sub_parts[0]
                atoms_str = sub_parts[1].strip()
                res_parts = prefix_residue.rsplit(":", 5) 
                if len(res_parts) < 6: continue
                chain_id = res_parts[1].strip()
                res_num = res_parts[2].strip()
                sigma = float(sigma_str)
                num_atoms = len([x for x in atoms_str.replace("--", "-").split("-") if x])
                is_bond = (num_atoms == 2)
                is_angle = (num_atoms >= 3)
                
                def get_existing_sigma(txt):
                    if not txt: return -1.0
                    try: return float(txt.split("(")[1].split("σ")[0])
                    except: return -1.0

                if is_bond and sigma >= BOND_SIGMA_LIMIT:
                    b_cnt += 1
                    out_txt = f"OUTLIER ({sigma:.1f}σ)"
                    e = self.get_entry(chain_id, res_num)
                    if sigma > get_existing_sigma(e["Bonds"]): e["Bonds"] = out_txt
                elif is_angle and sigma >= ANGLE_SIGMA_LIMIT:
                    a_cnt += 1
                    out_txt = f"OUTLIER ({sigma:.1f}σ)"
                    e = self.get_entry(chain_id, res_num)
                    if sigma > get_existing_sigma(e["Angles"]): e["Angles"] = out_txt
            except: pass

        self.summary_data['b_out_cnt'] = b_cnt
        self.summary_data['a_out_cnt'] = a_cnt
        
        del out_geo
        self.save_db()

        # 11. CIS PROLINES
        self.load_db()
        status_callback("Analyzing Cis Prolines...")
        out_omega = run_tool("phenix.omegalyze", f"\"{final}\"")
        
        cis_pro = parse_val(r"SUMMARY:\s*(\d+)\s*cis prolines", out_omega)
        total_pro = parse_val(r"cis prolines out of\s+(\d+)\s+PRO", out_omega)
        twist_pro = parse_val(r"SUMMARY:\s*(\d+)\s*twisted prolines", out_omega)
        twist_non = parse_val(r"SUMMARY:\s*(\d+)\s*other twisted", out_omega)
        total_non = parse_val(r"twisted residues out of\s+(\d+)\s+nonPRO", out_omega)

        self.summary_data['cis_pro_cnt'] = int(cis_pro)
        self.summary_data['cis_pro_total'] = int(total_pro)
        self.summary_data['twisted_cnt'] = int(twist_pro + twist_non)
        self.summary_data['twisted_total'] = int(total_pro + total_non)
        
        for line in out_omega.splitlines():
             if ":" in line:
                 parts = line.split(":")
                 res_info = parts[0].strip().split()
                 if len(res_info) >= 2:
                     e = self.get_entry(res_info[0], res_info[1])
                     if "cis" in line.lower(): e["Cis"] = "Cis"
                     elif "twisted" in line.lower(): e["Cis"] = "Twisted"
        
        del out_omega
        self.save_db() # FINAL SAVE

        # 12. MOLPROBITY SCORE
        status_callback("Calculating MolProbity Score...")

        self.load_db()
        try:
            clash = self.summary_data['clashscore']
            rot_out = self.summary_data['rota_out_pct']
            rama_fav = self.summary_data['rama_fav_pct']
            term1 = 0.42574 * math.log(1 + clash)
            term2 = 0.32996 * math.log(1 + max(0, rot_out - 1))
            term3 = 0.24979 * math.log(1 + max(0, (100 - rama_fav) - 2))
            self.summary_data['mp_score'] = term1 + term2 + term3 + 0.5
        except:
            self.summary_data['mp_score'] = 0.0
        
        self.save_db()

        logger.log("Analysis Complete")
        return True, generated_files

    def cleanup_files(self, files_list, original_input):
        abs_input = os.path.abspath(original_input)
        for f in files_list:
            if os.path.exists(f) and os.path.abspath(f) != abs_input:
                try: os.remove(f)
                except: pass

# ==============================================================================
# 3. PYQT5 WORKER
# ==============================================================================
class AnalysisWorker(QObject):
    finished = pyqtSignal(bool)
    results_ready = pyqtSignal()
    status_update = pyqtSignal(str)

    def __init__(self, backend, pdb_path, renumber, cleanup):
        super().__init__()
        self.backend = backend
        self.pdb_path = pdb_path
        self.renumber = renumber
        self.cleanup = cleanup

    def run(self):
        try:
            success, generated_files = self.backend.run_full_analysis(
                self.pdb_path, self.emit_status, self.renumber
            )
            
            if success:
                self.emit_status("Rendering Table...")
                self.results_ready.emit()
                
            if self.cleanup and success:
                self.backend.cleanup_files(generated_files, self.pdb_path)
            
            self.finished.emit(success)

        except Exception as e:
            print(f"CRITICAL WORKER ERROR: {e}")
            traceback.print_exc()
            self.emit_status(f"Error: {e}")
            self.finished.emit(False)

    def emit_status(self, msg):
        self.status_update.emit(msg)

# ==============================================================================
# 4. CUSTOM TABLE ITEM & GUI (PRESERVED)
# ==============================================================================
class SortableTableWidgetItem(QTableWidgetItem):
    def __init__(self, text):
        super().__init__(text) 
        self.setData(Qt.ItemDataRole.UserRole, text)
    
    def __lt__(self, other):
        txt1 = self.data(Qt.ItemDataRole.UserRole) or ""
        txt2 = other.data(Qt.ItemDataRole.UserRole) or ""
        txt1, txt2 = txt1.strip(), txt2.strip()

        is_empty1 = (not txt1 or txt1 == "-")
        is_empty2 = (not txt2 or txt2 == "-")

        table = self.tableWidget()
        is_ascending = True
        if table:
            is_ascending = (table.horizontalHeader().sortIndicatorOrder() == Qt.SortOrder.AscendingOrder)

        if is_empty1 and not is_empty2: 
            return False if is_ascending else True
        if not is_empty1 and is_empty2: 
            return True if is_ascending else False
        if is_empty1 and is_empty2: 
            return False

        val1 = self.get_val(txt1)
        val2 = self.get_val(txt2)
        
        primary_less = False
        primary_equal = False
        
        if val1 is not None and val2 is not None:
            if val1 != val2: primary_less = val1 < val2
            else: primary_equal = True
        else:
            if txt1 != txt2: primary_less = txt1 < txt2
            else: primary_equal = True

        if not primary_equal:
            if self.column() in [3, 6, 8, 9]:
                return not primary_less
            return primary_less

        row1, row2 = self.row(), other.row()
        item1_chain = table.item(row1, 0).data(Qt.ItemDataRole.UserRole) or ""
        item2_chain = table.item(row2, 0).data(Qt.ItemDataRole.UserRole) or ""
        
        if item1_chain != item2_chain:
            return (item1_chain < item2_chain) if is_ascending else (item1_chain > item2_chain)

        item1_res = table.item(row1, 1).data(Qt.ItemDataRole.UserRole) or "0"
        item2_res = table.item(row2, 1).data(Qt.ItemDataRole.UserRole) or "0"
        
        def extract_num(s):
            m = re.search(r"(\d+)", s)
            return int(m.group(1)) if m else -1
            
        r1_num = extract_num(item1_res)
        r2_num = extract_num(item2_res)
        
        if r1_num != r2_num:
             return (r1_num < r2_num) if is_ascending else (r1_num > r2_num)
             
        return False

    def get_val(self, txt):
        m_pct = re.search(r"\(([\d\.]+)%\)", txt)
        if m_pct: return float(m_pct.group(1))
        m_num = re.search(r"([\d\.]+)", txt)
        if m_num: return float(m_num.group(1))
        return None

class CustomIconProvider(QFileIconProvider):
    def icon(self, arg):
        if not isinstance(arg, QFileInfo):
            return super().icon(arg)

        if arg.isDir():
            return super().icon(arg)

        ext = arg.suffix().lower()
        
        if ext == "pdb":
            return self.generate_icon("☣", "#98c379", "black") 
        
        elif ext == "txt":
            return self.generate_icon("✎", "#ffffff", "black")

        return super().icon(arg)

    def generate_icon(self, text, bg_hex, text_color):
        pixmap = QPixmap(64, 64)
        pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        bg_color = QColor(bg_hex)
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(4, 4, 56, 56, 8, 8)
        
        painter.setPen(QColor(text_color))
        font = QFont("Arial", 32, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, text)
        
        painter.end()
        return QIcon(pixmap)

class ExportDialog(QDialog):
    def __init__(self, parent=None, has_chart=True):
        super().__init__(parent)
        self.setWindowTitle("Export Options")
        layout = QVBoxLayout(self)
        
        self.chk_summary = QCheckBox("Export Summary")
        self.chk_summary.setChecked(True)
        layout.addWidget(self.chk_summary)
        
        self.has_chart = has_chart
        if has_chart:
            self.chk_chart = QCheckBox("Export Chart")
            self.chk_chart.setChecked(True)
            layout.addWidget(self.chk_chart)
            
            self.rad_static = QRadioButton("Static Chart")
            self.rad_interactive = QRadioButton("Interactive (Sortable) Chart")
            self.rad_interactive.setChecked(True)
            
            rad_layout = QVBoxLayout()
            rad_layout.setContentsMargins(25, 0, 0, 10)
            rad_layout.addWidget(self.rad_static)
            rad_layout.addWidget(self.rad_interactive)
            layout.addLayout(rad_layout)
            
            self.chk_chart.toggled.connect(self.rad_static.setEnabled)
            self.chk_chart.toggled.connect(self.rad_interactive.setEnabled)
        
        self.btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.btn_box.accepted.connect(self.accept)
        self.btn_box.rejected.connect(self.reject)
        self.btn_box.setStyleSheet("QPushButton { min-width: 80px; }")
        layout.addWidget(self.btn_box)

    def get_options(self):
        opts = {"summary": self.chk_summary.isChecked()}
        if self.has_chart:
            opts["chart"] = self.chk_chart.isChecked()
            opts["interactive"] = self.rad_interactive.isChecked()
        return opts

class MolProbityGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MolProbity")
        self.resize(1300, 900)
        self.backend = AnalysisBackend()
        self.worker_thread = None
        self.worker = None
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout()
        lbl_pdb = QLabel("PDB/Molprobity File:")
        self.txt_path = QLineEdit()
        self.txt_path.setPlaceholderText("Select a file...")
        btn_browse = QPushButton("Browse")
        btn_browse.setToolTip("Browse a PDB file or PDB_Molprobity.txt to perform analysis")
        btn_browse.clicked.connect(self.browse_file)
        self.btn_run = QPushButton("Run Analysis")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_export = QPushButton("Export HTML")
        self.btn_export.clicked.connect(self.export_html)
        self.chk_renumber = QCheckBox("Renumbering H")
        self.chk_renumber.setToolTip("***Renumber the hydrogen for the PDB generated by Phenix\n\nHydrogen added by Phenix will have only zeros for the atom serial numbers\nYou will get warnings if open in ChimeraX")
        self.chk_renumber.setChecked(False)
        self.chk_clean = QCheckBox("Remove Intermediate Files")
        self.chk_clean.setToolTip("***This allows auto cleaning of these intermediate files:\n\nPDB_cleanH.pdb - Generated when removing hydrogens\nPDB_withH.pdb - Generated when adding hydrogens\n\nNote that Phenix adds hydrogen with Nuclear x-H, but when computing clashes, it will use the bond length of that in Electron-cloud x-H")
        self.chk_clean.setChecked(True)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: white; font-weight: bold;")
        
        # Row 1: File selection
        row1 = QHBoxLayout()
        row1.addWidget(lbl_pdb)
        row1.addWidget(self.txt_path, 1)
        row1.addWidget(btn_browse)
        
        self.lbl_back = QLabel("↺")
        self.lbl_back.setStyleSheet("color: #98c379; font-weight: bold; font-size: 26px; padding-right: 5px;")
        self.lbl_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_back.setToolTip("Back to Mode Selection")
        self.lbl_back.mousePressEvent = self.go_back

        # Row 2: Options, Action, and Status
        row2 = QHBoxLayout()
        row2.addWidget(self.chk_renumber)
        row2.addWidget(self.chk_clean)
        row2.addWidget(self.btn_run)
        row2.addWidget(self.btn_export)
        row2.addWidget(self.lbl_status)
        row2.addStretch() # Pushes status to the left so it always has room to expand on the right
        row2.addWidget(self.lbl_back) # Placed rightmost inside the input box
        
        input_layout.addLayout(row1)
        input_layout.addLayout(row2)
        input_group.setLayout(input_layout)
        
        main_layout.addWidget(input_group)

        sum_group = QGroupBox("Summary Statistics")
        self.sum_layout = QGridLayout()
        self.sum_layout.setColumnStretch(3, 1)
        headers = ["Metric", "Value", "Goal"]
        for i, h in enumerate(headers):
            l = QLabel(h)
            l.setStyleSheet("font-weight: bold;")
            self.sum_layout.addWidget(l, 0, i)

        self.metrics_labels = {}
        self.metric_defs = [
            ("Clashscore", "Good: <12; Caution: 12 - 22; Warning: >22"), ("Poor rotamers", "< 0.3%"),
            ("Favored rotamers", "> 98%"), ("Ramachandran outliers", "< 0.05%"),
            ("Ramachandran favored", "> 98%"), ("Rama distribution Z-score", "abs(Z) < 2"),
            ("MolProbity score", "Good: <2.8; Caution: 2.8 - 3.2; Warning: >3.2"), ("Cβ deviations >0.25Å", "0"),
            ("Bad bonds", "0"), ("Bad angles", "< 0.1%"), ("Cis Prolines", "0"),
            ("Twisted Peptides", "0"), ("CaBLAM outliers", "< 1.0%"),
            ("CA Geometry outliers", "< 0.5%")
        ]

        mp_tip = (
            "<html>"
            "For Clashscores and Molprobity scores:<br><br>"
            "xcop gives the same values as Molprobity, but does not contain the database to rank the percentiles<br>"
            "Therefore, xcop may not color the same as Molprobity<br><br>"
            "***This coloring assumes the final density map has a resolution of 3Å<br>"
            "Generally, if Molprobity score equals to the final map resolution, that means the PDB is at the 50th percentile, Caution (Yellow)<br><br>"
            "Molprobity cutoffs:<br>"
            "<span style='color:#4b9f49'>Good: &gt;66 percentile</span> <br>"
            "<span style='color:#dbd05f'>Caution: 66 - 33 percentile</span> <br>"
            "<span style='color:#cd5c5c'>Warning: &lt;33 percentile</span>"
            "</html>"
        )

        for i, (name, goal) in enumerate(self.metric_defs, 1):
            name_lbl = QLabel(name)
            goal_lbl = QLabel(goal)
            
            val_lbl = QLabel("-")
            val_lbl.setFixedWidth(300)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            val_lbl.setStyleSheet("background-color: #333; border: 1px solid #555; padding: 2px 5px; border-radius: 3px;")
            
            if "Clashscore" in name or "MolProbity" in name:
                name_lbl.setToolTip(mp_tip)
                val_lbl.setToolTip(mp_tip)
                goal_lbl.setToolTip(mp_tip)

            self.sum_layout.addWidget(name_lbl, i, 0)
            self.sum_layout.addWidget(val_lbl, i, 1)
            self.sum_layout.addWidget(goal_lbl, i, 2)
            self.metrics_labels[name] = val_lbl
            
        # Instantiate native 3D module on the right side
        self.viewer = PDBVisualizer()
        self.viewer.setMinimumWidth(380)
        self.sum_layout.addWidget(self.viewer, 0, 3, len(self.metric_defs) + 1, 1)

        sum_group.setLayout(self.sum_layout)
        main_layout.addWidget(sum_group)

        chart_group = QGroupBox("Multi-Criterion Chart")
        chart_layout = QVBoxLayout()
        self.table = QTableWidget()
        cols = ["Chain", "Res", "Type", "Clash > 0.4Å", "Ramachandran", "Rotamer", "Cβ dev", "CaBLAM", "Bond lengths", "Bond angles", "Cis Peptides"]
        self.table.setColumnCount(len(cols))
        self.table.setHorizontalHeaderLabels(cols)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.table.setSortingEnabled(True)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        for i in [0, 1, 2]: header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.cellDoubleClicked.connect(self.show_details)
        chart_layout.addWidget(self.table)
        chart_group.setLayout(chart_layout)
        main_layout.addWidget(chart_group, 1)

    def go_back(self, event):
        self.launcher = LauncherWindow()
        self.launcher.show()
        self.close()

    def browse_file(self):
        dialog = QFileDialog(self, "Open File", "")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setIconProvider(CustomIconProvider())
        dialog.setNameFilters(["Files (*.pdb *.txt)", "All Files (*)"])
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setStyleSheet("""
            QFileDialog { background-color: #2b2b2b; color: white; }
            QListView { background-color: #1e1e1e; color: white; }
            QTreeView { background-color: #1e1e1e; color: white; }
            QLabel { color: white; }
            QPushButton { background-color: #3e3e3e; color: white; }
        """)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                self.txt_path.setText(selected_files[0])
                self.viewer.load_pdb(selected_files[0])

    def run_analysis(self):
        self.txt_path.deselect()
        self.table.setFocus()
        path = self.txt_path.text()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Please select a valid file.")
            return

        self.viewer.load_pdb(path)

        if path.endswith(".txt"):
            self.lbl_status.setText("Loading data from file...")
            self.backend.db_file = path
            self.backend.load_db() 
            self.populate_metrics()
            self.populate_table()
            self.lbl_status.setText("Data Loaded from Text File")
            return

        self.btn_run.setEnabled(False)
        self.table.setRowCount(0)
        for lbl in self.metrics_labels.values():
            lbl.setText("-")
            lbl.setStyleSheet("background-color: #333; border: 1px solid #555; padding: 2px; border-radius: 3px;")

        self.worker_thread = QThread()
        renum = self.chk_renumber.isChecked()
        clean = self.chk_clean.isChecked()
        self.worker = AnalysisWorker(self.backend, path, renum, clean)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.status_update.connect(self.update_status)
        self.worker.results_ready.connect(self.populate_metrics)
        self.worker.results_ready.connect(self.populate_table)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def update_status(self, msg):
        self.lbl_status.setText(msg)

    def analysis_finished(self, success):
        self.btn_run.setEnabled(True)

    def populate_metrics(self):
        d = self.backend.summary_data
        C_GOOD, C_CAUTION, C_WARN, C_NONE = "#4b9f49", "#dbd05f", "#cd5c5c", "#333333"
        def set_m(key, txt, color):
            lbl = self.metrics_labels[key]
            lbl.setText(str(txt))
            lbl.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
            lbl.setStyleSheet(f"""
                QLabel {{ background-color: {color}; color: white; border: 1px solid #777; padding: 2px 5px; border-radius: 3px; font-weight: bold; }}
                QToolTip {{ color: #ffffff; background-color: #2a2a2a; border: 1px solid #555; }}
            """)

        cs = d.get('clashscore', 0.0)
        if cs < 12: c_clash = C_GOOD
        elif cs <= 22: c_clash = C_CAUTION
        else: c_clash = C_WARN
        set_m("Clashscore", f"{cs:.2f}", c_clash)
        rota_out = d.get('rota_out_pct', 0.0)
        c_rota = C_GOOD if rota_out <= 0.3 else (C_CAUTION if rota_out <= 1.5 else C_WARN)
        set_m("Poor rotamers", f"{d.get('rota_out_cnt',0)} ({rota_out:.2f}%)", c_rota)
        rota_fav = d.get('rota_fav_pct', 0.0)
        c_fav = C_GOOD if rota_fav >= 98 else (C_CAUTION if rota_fav >= 95 else C_WARN)
        set_m("Favored rotamers", f"{d.get('rota_fav_cnt',0)} ({rota_fav:.2f}%)", c_fav)
        rama_out_pct = d.get('rama_out_pct', 0.0)
        rama_out_cnt = d.get('rama_out_cnt', 0)
        if rama_out_pct <= 0.05: c_rama_out = C_GOOD
        elif rama_out_pct <= 0.5: c_rama_out = C_CAUTION
        else: c_rama_out = C_CAUTION if rama_out_cnt <= 1 else C_WARN
        set_m("Ramachandran outliers", f"{rama_out_cnt} ({rama_out_pct:.2f}%)", c_rama_out)
        rama_fav = d.get('rama_fav_pct', 0.0)
        c_rama_fav = C_GOOD if rama_fav >= 98 else (C_CAUTION if rama_fav >= 95 else C_WARN)
        set_m("Ramachandran favored", f"{d.get('rama_fav_cnt',0)} ({rama_fav:.2f}%)", c_rama_fav)
        rama_z_str = d.get('rama_z', "N/A")
        c_z = C_NONE
        if "N/A" not in rama_z_str:
            try:
                z_val = abs(float(rama_z_str.split()[0]))
                if z_val <= 2: c_z = C_GOOD
                elif z_val <= 3: c_z = C_CAUTION
                else: c_z = C_WARN
            except: pass
        set_m("Rama distribution Z-score", rama_z_str, c_z)
        mp = d.get('mp_score', 0.0)
        if mp < 2.8: c_mp = C_GOOD
        elif mp <= 3.2: c_mp = C_CAUTION
        else: c_mp = C_WARN
        set_m("MolProbity score", f"{mp:.2f}", c_mp)
        cb_cnt = d.get('cb_out_cnt', 0)
        res_tot = d.get('rama_tot', 1) 
        cb_pct = (cb_cnt / res_tot * 100) if res_tot else 0
        if cb_cnt == 0: c_cb = C_GOOD
        elif cb_pct < 5: c_cb = C_CAUTION
        else: c_cb = C_WARN
        set_m("Cβ deviations >0.25Å", f"{cb_cnt} ({cb_pct:.2f}%)", c_cb)
        b_cnt = d.get('b_out_cnt', 0)
        b_tot = d.get('b_total', 1)
        b_pct = (b_cnt / b_tot * 100) if b_tot > 0 else 0
        c_bond = C_GOOD if b_pct < 0.01 else (C_CAUTION if b_pct < 0.2 else C_WARN)
        set_m("Bad bonds", f"{b_cnt} / {b_tot} ({b_pct:.4f}%)", c_bond)
        a_cnt = d.get('a_out_cnt', 0)
        a_tot = d.get('a_total', 1)
        a_pct = (a_cnt / a_tot * 100) if a_tot > 0 else 0
        c_ang = C_GOOD if a_pct < 0.1 else (C_CAUTION if a_pct < 0.5 else C_WARN)
        set_m("Bad angles", f"{a_cnt} / {a_tot} ({a_pct:.4f}%)", c_ang)
        c_cnt = d.get('cis_pro_cnt', 0)
        c_tot = d.get('cis_pro_total', 0)
        set_m("Cis Prolines", f"{c_cnt} / {c_tot}", C_NONE)
        t_cnt = d.get('twisted_cnt', 0)
        t_tot = d.get('twisted_total', 0)
        t_pct = (t_cnt/t_tot*100) if t_tot else 0
        if t_cnt == 0: c_twist = C_GOOD
        elif t_pct <= 0.1: c_twist = C_CAUTION
        else: c_twist = C_WARN
        set_m("Twisted Peptides", f"{t_cnt} / {t_tot} ({t_pct:.3f}%)", c_twist)
        cab_cnt = d.get('cab_out_cnt', 0)
        cab_tot = d.get('ca_geo_total', 1) 
        cab_pct = (cab_cnt / cab_tot * 100) if cab_tot else 0
        c_cab = C_GOOD if cab_pct <= 1.0 else (C_CAUTION if cab_pct <= 5.0 else C_WARN)
        set_m("CaBLAM outliers", f"{cab_cnt} ({cab_pct:.2f}%)", c_cab)
        geo_cnt = d.get('ca_geo_cnt', 0)
        geo_pct = (geo_cnt / cab_tot * 100) if cab_tot else 0
        c_geo = C_GOOD if geo_pct <= 0.5 else (C_CAUTION if geo_pct < 1.0 else C_WARN)
        set_m("CA Geometry outliers", f"{geo_cnt} ({geo_pct:.2f}%)", c_geo)

    def populate_table(self):
        self.lbl_status.setText("Rendering Table...")
        QApplication.processEvents()
        
        self.backend.load_db()

        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)

        try:
            self.table.setRowCount(0)
            
            C_RED = QColor("#f66969")
            C_PINK = QColor("#e886b9")
            C_WHITE = QColor("white")
            
            for key in sorted(self.backend.residue_data.keys()):
                r = self.backend.residue_data[key]
                vals = [
                    r.get("Chain",""), r.get("Res",""), r.get("Type",""),
                    r.get("Clash",""), r.get("Rama",""), r.get("Rotamer",""),
                    r.get("Cbeta",""), r.get("CaBLAM",""), 
                    r.get("Bonds",""), r.get("Angles",""), r.get("Cis","")
                ]
                
                row = self.table.rowCount()
                self.table.insertRow(row)

                for col, text in enumerate(vals):
                    display_text = text
                    text_color = C_WHITE
                    txt_upper = text.upper() if text else ""

                    if col == 3 and text:
                        try:
                            val = float(text)
                            if val >= 1.0: text_color = C_RED
                            elif val > 0.4: text_color = C_PINK
                            display_text = f"{text} Å"
                        except: pass

                    elif "OUTLIER" in txt_upper:
                        text_color = C_RED
                    elif any(x in txt_upper for x in ["DISFAVORED", "ALLOWED", "TWISTED"]):
                        text_color = C_PINK

                    item = SortableTableWidgetItem(display_text)
                    item.setForeground(text_color)
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.table.setItem(row, col, item)

        finally:
            self.table.setSortingEnabled(True)
            self.table.resizeRowsToContents()
            self.table.setUpdatesEnabled(True)
            self.lbl_status.setText("Analysis Complete")

    def show_details(self, row, col):
        chain = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
        res = self.table.item(row, 1).data(Qt.ItemDataRole.UserRole)
        typ = self.table.item(row, 2).data(Qt.ItemDataRole.UserRole)
        msg = f"Residue: {res} Chain: {chain}\nType: {typ}\n\n"
        headers = ["Clash", "Rama", "Rotamer", "Cbeta", "CaBLAM", "Bonds", "Angles", "Cis"]
        for i, header in enumerate(headers):
            val = self.table.item(row, i+3).data(Qt.ItemDataRole.UserRole)
            if val and val != "-" and val != "": msg += f"{header}: {val}\n"
        QMessageBox.information(self, "Residue Details", msg)

    def export_html(self):
        input_path = self.txt_path.text()
        default_name = ""
        if input_path and os.path.exists(input_path):
            base = os.path.splitext(os.path.basename(input_path))[0]
            default_name = base + "_MolProbity.html"

        dialog = ExportDialog(self, has_chart=True)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        opts = dialog.get_options()
        if not opts.get("summary") and not opts.get("chart"):
            QMessageBox.warning(self, "Warning", "No sections selected for export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save HTML", default_name, "HTML Files (*.html)")
        if not path: return
        
        html = [
            "<html><head><style>",
            "body { background-color: #2b2b2b; color: white; font-family: sans-serif; padding: 20px; }",
            "h2 { color: #98c379; border-bottom: 1px solid #555; padding-bottom: 5px; }",
            ".summary { display: grid; grid-template-columns: auto auto auto; gap: 10px; margin-bottom: 20px; }",
            ".sum-hdr { font-weight: bold; padding: 5px; font-size: 16px; }",
            ".sum-val { background-color: #333; border: 1px solid #777; padding: 5px; border-radius: 3px; font-weight: bold; color: white; }",
            "table { width: 100%; border-collapse: collapse; background-color: #1e1e1e; color: #eee; border: 1px solid #444; margin-top: 10px; }",
            "th { background-color: #333; color: white; padding: 8px; border: 1px solid #444; font-weight: bold; cursor: pointer; }",
            "th:hover { background-color: #444; }",
            "td { padding: 8px; border: 1px solid #444; text-align: center; }",
            "tr:nth-child(even) { background-color: #2a2a2a; }",
            "</style>"
        ]

        if opts.get("chart") and opts.get("interactive"):
            html.append("""
            <script>
            function sortTable(n) {
                var table = document.getElementById("molTable");
                var rows = Array.prototype.slice.call(table.rows, 1);
                var dir = table.getAttribute("data-dir") === "asc" ? "desc" : "asc";
                table.setAttribute("data-dir", dir);

                rows.sort(function(a, b) {
                    var x = a.cells[n].innerText || a.cells[n].textContent;
                    var y = b.cells[n].innerText || b.cells[n].textContent;
                    x = x.trim(); y = y.trim();
                    
                    if (x === "" && y !== "") return 1;
                    if (x !== "" && y === "") return -1;
                    if (x === "" && y === "") return 0;
                    
                    var mX = x.match(/-?[\\d\\.]+/); var nX = mX ? parseFloat(mX[0]) : NaN;
                    var mY = y.match(/-?[\\d\\.]+/); var nY = mY ? parseFloat(mY[0]) : NaN;
                    
                    var cmpX = isNaN(nX) ? x.toLowerCase() : nX;
                    var cmpY = isNaN(nY) ? y.toLowerCase() : nY;
                    
                    if (cmpX < cmpY) return dir === "asc" ? -1 : 1;
                    if (cmpX > cmpY) return dir === "asc" ? 1 : -1;
                    return 0;
                });
                for (var i = 0; i < rows.length; i++) {
                    table.appendChild(rows[i]);
                }
            }
            </script>
            """)

        html.append("</head><body>")

        if opts.get("summary"):
            html.append("<h2>MolProbity Summary Statistics</h2>")
            html.append("<div class='summary'>")
            html.extend(["<div class='sum-hdr'>Metric</div>", "<div class='sum-hdr'>Value</div>", "<div class='sum-hdr'>Goal</div>"])
            for name, goal in self.metric_defs:
                lbl = self.metrics_labels.get(name)
                val_txt = lbl.text() if lbl else "-"
                # Extract background color to match UI conditional coloring
                color = "#333"
                if lbl and lbl.styleSheet():
                    m = re.search(r'background-color:\s*(#[0-9a-fA-F]+)', lbl.styleSheet())
                    if m: color = m.group(1)
                
                html.append(f"<div>{name}</div>")
                html.append(f"<div class='sum-val' style='background-color: {color};'>{val_txt}</div>")
                html.append(f"<div>{goal}</div>")
            html.append("</div>")
            
        if opts.get("chart"):
            html.append("<h2>Multi-Criterion Chart</h2>")
            table_id = " id='molTable'" if opts.get("interactive") else ""
            html.append(f"<table{table_id}><tr>")
            
            for c in range(self.table.columnCount()):
                click_attr = f" onclick='sortTable({c})'" if opts.get("interactive") else ""
                html.append(f"<th{click_attr}>{self.table.horizontalHeaderItem(c).text()}</th>")
            html.append("</tr>")
            
            for r in range(self.table.rowCount()):
                html.append("<tr>")
                for c in range(self.table.columnCount()):
                    item = self.table.item(r, c)
                    txt = item.text() if item else ""
                    txt = txt.replace("\n", "<br>") # Safely maintain row spacing for HTML
                    color = item.foreground().color().name() if item and item.foreground().color().isValid() else "white"
                    html.append(f"<td style='color: {color};'>{txt}</td>")
                html.append("</tr>")
            html.append("</table>")
            
        html.append("</body></html>")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        QMessageBox.information(self, "Success", "HTML exported successfully!")

class CryoEMWorker(QObject):
    finished = pyqtSignal(dict)
    status_update = pyqtSignal(str)

    def __init__(self, pdb_path, map_path, resolution):
        super().__init__()
        self.pdb_path = pdb_path
        self.map_path = map_path
        self.resolution = resolution

    def run(self):
        # 1. Run CryoEM Validation
        self.status_update.emit("Running phenix.validation_cryoem... This may take a moment.")
        args = f'"{self.pdb_path}" "{self.map_path}" resolution={self.resolution}'
        out = run_tool("phenix.validation_cryoem", args)
        
        # 2. Run CaBLAM explicitly since validation_cryoem omits it
        self.status_update.emit("Running phenix.cablam...")
        cablam_out = run_tool("phenix.cablam", f'"{self.pdb_path}"')
        
        self.status_update.emit("Parsing results...")
        
        def p_val(regex, text, default="N/A"):
            match = re.search(regex, text)
            return float(match.group(1)) if match else default
            
        def p_int(regex, text, default=0):
            match = re.search(regex, text)
            return int(match.group(1)) if match else default

        # --- Parse validation_cryoem output ---
        chains = p_int(r"chains\s*:\s*(\d+)", out)
        all_atoms = p_int(r"all atoms\s*:\s*(\d+)", out)
        h_atoms = p_int(r"H or D atoms\s*:\s*(\d+)", out)
        non_h_atoms = all_atoms - h_atoms if all_atoms else "N/A"
        protein_res = p_int(r"a\.a\.\s*residues\s*:\s*(\d+)", out)
        water_res = p_int(r"water\s*:\s*(\d+)", out)

        cc_mask = p_val(r"CC_mask\s*:\s*([\d\.]+)", out)

        # B factors (capturing the FIRST number, which is 'min')
        b_prot = p_val(r"Protein:\s+([\d\.]+)", out)
        b_water = p_val(r"Water:\s+([\d\.]+)", out)

        rmsd_bond = p_val(r"BOND\s*:\s*([\d\.]+)", out)
        rmsd_angle = p_val(r"ANGLE\s*:\s*([\d\.]+)", out)

        clash = p_val(r"ALL-ATOM CLASHSCORE\s*:\s*([\d\.]+)", out, 0.0)
        rota_out = p_val(r"ROTAMER OUTLIERS\s*:\s*([\d\.]+)", out, 0.0)
        cbeta = p_val(r"CBETA DEVIATIONS\s*:\s*([\d\.]+)", out, 0.0)

        rama_fav = p_val(r"FAVORED\s*:\s*([\d\.]+)", out, 0.0)
        rama_allow = p_val(r"ALLOWED\s*:\s*([\d\.]+)", out, 0.0)
        rama_out = p_val(r"RAMACHANDRAN PLOT:\s*OUTLIERS\s*:\s*([\d\.]+)", out, 0.0)

        # --- Parse CaBLAM output ---
        cab_cnt = p_int(r"SUMMARY:\s*(\d+)\s*residues.*have outlier conformations", cablam_out)
        cab_tot = p_int(r"SUMMARY: CaBLAM found\s*(\d+)\s*full protein residues", cablam_out, 1) # Default to 1 to prevent div/0
        
        cablam_pct = (cab_cnt / cab_tot * 100.0) if cab_tot > 1 else 0.0
        cablam_str = f"{cablam_pct:.2f}" if cab_tot > 1 else "N/A"

        # Calculate Molprobity Score manually
        try:
            term1 = 0.42574 * math.log(1 + clash)
            term2 = 0.32996 * math.log(1 + max(0, rota_out - 1))
            term3 = 0.24979 * math.log(1 + max(0, (100 - rama_fav) - 2))
            mp_score = term1 + term2 + term3 + 0.5
            mp_score_str = f"{mp_score:.2f}"
        except:
            mp_score_str = "N/A"

        results = {
            "Chains": str(chains),
            "Non-hydrogen atoms": str(non_h_atoms),
            "Protein residues": str(protein_res),
            "Water": str(water_res),
            "Resolution": str(self.resolution),
            "CC (mask)": str(cc_mask),
            "Protein": str(b_prot),
            "Water (B-factor)": str(b_water),
            "Bond lengths": str(rmsd_bond),
            "Bond angles": str(rmsd_angle),
            "Molprobity score": mp_score_str,
            "Clashscore": f"{clash:.2f}",
            "Poor rotamers (%)": f"{rota_out:.2f}",
            "CaBLAM outliers (%)": cablam_str,
            "Cβ outleirs (%)": f"{cbeta:.2f}",
            "Favored (%)": f"{rama_fav:.2f}",
            "Allowed (%)": f"{rama_allow:.2f}",
            "Disallowed (%)": f"{rama_out:.2f}",
        }
        
        self.status_update.emit("Analysis Complete")
        self.finished.emit(results)


class CryoEMValidationGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CryoEM Validation")
        self.resize(800, 750)
        self.worker_thread = None
        self.worker = None
        self.result_labels = {}
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        # --- INPUT GROUP ---
        input_group = QGroupBox("Inputs")
        input_layout = QGridLayout()

        self.txt_pdb = QLineEdit()
        self.txt_pdb.setPlaceholderText("Select PDB file...")
        btn_pdb = QPushButton("Browse PDB")
        btn_pdb.clicked.connect(lambda: self.browse_file(self.txt_pdb, "*.pdb"))

        self.txt_map = QLineEdit()
        self.txt_map.setPlaceholderText("Select Map (MRC) file...")
        btn_map = QPushButton("Browse Map")
        btn_map.clicked.connect(lambda: self.browse_file(self.txt_map, "*.mrc *.map"))

        self.txt_res = QLineEdit("3.0")
        self.txt_res.setFixedWidth(80)

        input_layout.addWidget(QLabel("PDB File:"), 0, 0)
        input_layout.addWidget(self.txt_pdb, 0, 1)
        input_layout.addWidget(btn_pdb, 0, 2)

        input_layout.addWidget(QLabel("Map File:"), 1, 0)
        input_layout.addWidget(self.txt_map, 1, 1)
        input_layout.addWidget(btn_map, 1, 2)

        self.lbl_back = QLabel("↺")
        self.lbl_back.setStyleSheet("color: #98c379; font-weight: bold; font-size: 26px;")
        self.lbl_back.setCursor(Qt.CursorShape.PointingHandCursor)
        self.lbl_back.setToolTip("Back to Mode Selection")
        self.lbl_back.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_back.mousePressEvent = self.go_back

        input_layout.addWidget(QLabel("Resolution (Å):"), 2, 0)
        input_layout.addWidget(self.txt_res, 2, 1)
        input_layout.addWidget(self.lbl_back, 2, 2) # Placed in grid directly under the Browse Map button

        input_group.setLayout(input_layout)
        main_layout.addWidget(input_group)

        # --- RUN BUTTON & STATUS ---
        run_layout = QHBoxLayout()
        self.btn_run = QPushButton("Run Validation")
        self.btn_run.clicked.connect(self.run_analysis)
        self.btn_export = QPushButton("Export HTML")
        self.btn_export.clicked.connect(self.export_html)
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: #98c379; font-weight: bold;")
        run_layout.addWidget(self.btn_run)
        run_layout.addWidget(self.btn_export)
        run_layout.addWidget(self.lbl_status)
        run_layout.addStretch()
        main_layout.addLayout(run_layout)

        # --- RESULTS GROUP ---
        res_group = QGroupBox("Required Info")
        res_layout = QVBoxLayout()
        
        categories = {
            "Model composition": ["Chains", "Non-hydrogen atoms", "Protein residues", "Water"],
            "Refinement": ["Resolution", "CC (mask)"],
            "B factors": ["Protein", "Water (B-factor)"],
            "RMSD deviations": ["Bond lengths", "Bond angles"],
            "Validation": ["Molprobity score", "Clashscore", "Poor rotamers (%)", "CaBLAM outliers (%)", "Cβ outleirs (%)"],
            "Ramachandran plot": ["Favored (%)", "Allowed (%)", "Disallowed (%)"]
        }

        for cat, fields in categories.items():
            cat_label = QLabel(cat)
            cat_label.setStyleSheet("font-weight: bold; font-size: 13px; color: #98c379; margin-top: 5px;")
            res_layout.addWidget(cat_label)
            
            grid = QGridLayout()
            grid.setContentsMargins(15, 0, 0, 5) # Indent items
            for i, field in enumerate(fields):
                lbl_name = QLabel(f"{field}:")
                lbl_name.setFixedWidth(150)
                
                lbl_val = QLabel("-")
                lbl_val.setStyleSheet("background-color: #333; border: 1px solid #555; padding: 2px 5px; border-radius: 3px;")
                lbl_val.setFixedWidth(100)
                
                self.result_labels[field] = lbl_val
                grid.addWidget(lbl_name, i, 0)
                grid.addWidget(lbl_val, i, 1)
                
                if field == "Resolution":
                    lbl_note = QLabel("Use the resolution from FSC curve calculation if different")
                    lbl_note.setStyleSheet("color: grey;")
                    grid.addWidget(lbl_note, i, 2)
                
                grid.setColumnStretch(3, 1) # Push everything left
            
            res_layout.addLayout(grid)

        res_group.setLayout(res_layout)
        main_layout.addWidget(res_group)
        main_layout.addStretch()

    def go_back(self, event):
        self.launcher = LauncherWindow()
        self.launcher.show()
        self.close()

    def browse_file(self, line_edit, filter_str):
        dialog = QFileDialog(self, "Select File", "")
        dialog.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        dialog.setNameFilters([f"Files ({filter_str})", "All Files (*)"])
        dialog.setFileMode(QFileDialog.FileMode.ExistingFile)
        dialog.setStyleSheet("""
            QFileDialog { background-color: #2b2b2b; color: white; }
            QListView { background-color: #1e1e1e; color: white; }
            QTreeView { background-color: #1e1e1e; color: white; }
            QLabel { color: white; }
            QPushButton { background-color: #3e3e3e; color: white; }
        """)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_files = dialog.selectedFiles()
            if selected_files:
                line_edit.setText(selected_files[0])

    def run_analysis(self):
        self.txt_pdb.deselect()
        self.txt_map.deselect()
        self.txt_res.deselect()
        self.setFocus() # Shifts focus away from the text boxes so they don't highlight
        
        pdb = self.txt_pdb.text().strip()
        mrc = self.txt_map.text().strip()
        res = self.txt_res.text().strip()

        if not os.path.exists(pdb) or not os.path.exists(mrc):
            QMessageBox.warning(self, "Error", "Please provide valid paths for both PDB and Map files.")
            return

        self.btn_run.setEnabled(False)
        for lbl in self.result_labels.values():
            lbl.setText("-")

        self.worker_thread = QThread()
        self.worker = CryoEMWorker(pdb, mrc, res)
        self.worker.moveToThread(self.worker_thread)
        
        self.worker_thread.started.connect(self.worker.run)
        self.worker.status_update.connect(self.lbl_status.setText)
        self.worker.finished.connect(self.populate_results)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(lambda: self.btn_run.setEnabled(True))
        
        self.worker_thread.start()

    def populate_results(self, data):
        for key, val in data.items():
            if key in self.result_labels:
                self.result_labels[key].setText(str(val))

    def export_html(self):
        input_path = self.txt_pdb.text()
        default_name = ""
        if input_path and os.path.exists(input_path):
            base = os.path.splitext(os.path.basename(input_path))[0]
            default_name = base + "_CryoEM.html"

        dialog = ExportDialog(self, has_chart=False)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
            
        opts = dialog.get_options()
        if not opts.get("summary"):
            QMessageBox.warning(self, "Warning", "No sections selected for export.")
            return

        path, _ = QFileDialog.getSaveFileName(self, "Save HTML", default_name, "HTML Files (*.html)")
        if not path: return
        
        html = [
            "<html><head><style>",
            "body { background-color: #2b2b2b; color: white; font-family: sans-serif; padding: 20px; }",
            "h2 { color: #98c379; border-bottom: 1px solid #555; padding-bottom: 5px; margin-top: 30px; }",
            "h3 { color: #98c379; font-size: 14px; margin-top: 15px; }",
            ".grid { display: grid; grid-template-columns: 200px 150px auto; gap: 10px; margin-bottom: 10px; }",
            ".val { background-color: #333; border: 1px solid #555; padding: 5px; border-radius: 3px; font-weight: bold; }",
            "</style></head><body>",
            "<h2>CryoEM Validation Results</h2>",
            f"<p><strong>PDB File:</strong> {self.txt_pdb.text()}</p>",
            f"<p><strong>Map File:</strong> {self.txt_map.text()}</p>",
            f"<p><strong>Resolution:</strong> {self.txt_res.text()} &Aring;</p>"
        ]
        
        categories = {
            "Model composition": ["Chains", "Non-hydrogen atoms", "Protein residues", "Water"],
            "Refinement": ["Resolution", "CC (mask)"],
            "B factors": ["Protein", "Water (B-factor)"],
            "RMSD deviations": ["Bond lengths", "Bond angles"],
            "Validation": ["Molprobity score", "Clashscore", "Poor rotamers (%)", "CaBLAM outliers (%)", "Cβ outleirs (%)"],
            "Ramachandran plot": ["Favored (%)", "Allowed (%)", "Disallowed (%)"]
        }
        
        for cat, fields in categories.items():
            html.append(f"<h3>{cat}</h3><div class='grid'>")
            for field in fields:
                val = self.result_labels[field].text() if field in self.result_labels else "-"
                note = "Use the resolution from FSC curve calculation if different" if field == "Resolution" else ""
                html.append(f"<div>{field}:</div><div class='val'>{val}</div><div style='color:grey;'>{note}</div>")
            html.append("</div>")
            
        html.append("</body></html>")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(html))
        QMessageBox.information(self, "Success", "HTML exported successfully!")

class LauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Select Mode")
        self.resize(350, 150)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        lbl = QLabel("Select Mode:")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("color: white; font-weight: bold; font-size: 14px;")
        layout.addWidget(lbl)
        
        self.btn_cryoem = QPushButton("CryoEM Validation")
        self.btn_molprobity = QPushButton("Molprobity")
        
        layout.addWidget(self.btn_cryoem)
        layout.addWidget(self.btn_molprobity)
        
        self.btn_cryoem.clicked.connect(self.open_cryoem)
        self.btn_molprobity.clicked.connect(self.open_molprobity)
        
    def open_cryoem(self):
        self.cryoem_window = CryoEMValidationGUI()
        self.cryoem_window.show()
        self.close()
        
    def open_molprobity(self):
        self.molprobity_window = MolProbityGUI()
        self.molprobity_window.show()
        self.close()

def apply_global_theme(app):
    # 1. Global Palette
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
    dark_palette.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
    dark_palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)
    app.setPalette(dark_palette)

    # 2. Global Stylesheet
    app.setStyleSheet("""
        QToolTip { color: #ffffff; background-color: #2a2a2a; border: 1px solid #555; }
        QMainWindow, LauncherWindow { background-color: #2b2b2b; }
        QGroupBox { border: 1px solid #555; margin-top: 10px; font-weight: bold; color: #98c379; }
        QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; padding: 0 5px; }
        QPushButton { background-color: #3e3e3e; border: 1px solid #3e3e3e; border-radius: 5px; padding: 8px 12px; color: white; font-weight: bold; outline: none; }
        QPushButton:hover { background-color: #3e3e3e; border: 1px solid #98c379; }
        QPushButton:pressed { background-color: #98c379; border: 1px solid #98c379; color: #2b2b2b; }
        QPushButton:disabled { color: #777; border: 1px solid #333; }
        QLineEdit { background-color: #1e1e1e; border: 1px solid #444; padding: 4px; color: #fff; border-radius: 2px; }
        QTableWidget { background-color: #1e1e1e; gridline-color: #333; color: #eee; border: 1px solid #444; outline: 0; }
        QTableWidget::item { border: none; }
        QHeaderView::section { background-color: #333; color: white; padding: 4px; border: 1px solid #444; font-weight: bold; }
        QLabel { color: #eee; }
        QCheckBox { color: #eee; spacing: 8px; }
        QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #555; border-radius: 2px; background-color: #1e1e1e; }
        QCheckBox::indicator:unchecked { background-color: #1e1e1e; }
        QCheckBox::indicator:checked { background-color: #98c379; border: 1px solid #98c379; image: none; }
        QCheckBox::indicator:hover { border: 1px solid #98c379; }
        
        QRadioButton { color: #eee; spacing: 8px; }
        QRadioButton::indicator { width: 14px; height: 14px; border: 1px solid #555; border-radius: 7px; background-color: #1e1e1e; }
        QRadioButton::indicator:unchecked { background-color: #1e1e1e; }
        QRadioButton::indicator:checked { background-color: #98c379; border: 1px solid #98c379; }
        QRadioButton::indicator:hover { border: 1px solid #98c379; }
    """)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion") 
    apply_global_theme(app)
    
    launcher = LauncherWindow()
    launcher.show()
    sys.exit(app.exec())