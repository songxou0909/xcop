import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from datetime import datetime, timezone
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

class ToolTip:
    def __init__(self, widget):
        self.widget = widget
        self.tipwindow = None

    def showtip(self, text):
        if self.tipwindow or not text:
            return
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(1)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=text, justify=tk.LEFT,
                         background="#2d2d2d", foreground="#d4d4d4", 
                         relief=tk.SOLID, borderwidth=1, font=("Arial", 9))
        label.pack(ipadx=5, ipady=2)

    def hidetip(self):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None

# ==========================================
# GLOBAL CONFIGURATION
# ==========================================
def get_optimal_pytorch_setup(target_packages=None, log_callback=None, cancel_check=None):
    """Dynamically finds the newest compatible CUDA version and streams logs to the GUI."""
    if target_packages is None:
        target_packages = []
        
    def log(msg):
        if log_callback:
            log_callback(msg)
            
    # Force conda to strictly pull PyTorch from its own channel to prevent CPU downgrades
    gpu_base_pkgs = ["pytorch::pytorch", "pytorch::torchvision"]
    cpu_base_pkgs = ["pytorch", "torchvision"]
    
    if sys.platform.startswith('darwin'):
        return cpu_base_pkgs, None

    try:
        log("    [Search] Checking nvidia-smi for local GPU limits...")
        smi_res = subprocess.run(["nvidia-smi"], stdout=subprocess.PIPE, text=True, check=True)
        match = re.search(r"CUDA Version:\s*(\d+\.\d+)", smi_res.stdout)
        
        if not match:
            log("    [Warning] Could not read max CUDA version. Falling back to CPU.")
            return cpu_base_pkgs + ["cpuonly"], None
            
        max_cuda = float(match.group(1))
        log(f"    [Search] System supports up to CUDA {max_cuda}. Fetching online Conda catalog (this may take a few minutes)...")

        # This is the step that takes the longest over the internet!
        search_res = subprocess.run(
            ["conda", "search", "pytorch-cuda", "-c", "pytorch", "--json"], 
            stdout=subprocess.PIPE, text=True, check=True
        )
        
        search_data = json.loads(search_res.stdout)
        
        available_versions = []
        if "pytorch-cuda" in search_data:
            for entry in search_data["pytorch-cuda"]:
                try:
                    available_versions.append(float(entry["version"]))
                except ValueError:
                    continue
                    
        available_versions = sorted(list(set(available_versions)), reverse=True)
        valid_versions = [v for v in available_versions if v <= max_cuda]
        
        log(f"    [Search] Online fetch complete! Testing targets in batches of 2: {valid_versions}")

        import concurrent.futures

        # Define the function that the threads will run
        def test_cuda_version(version):
            # The core GPU packages we want to install
            core_gpu_pkgs = gpu_base_pkgs + [f"pytorch::pytorch-cuda={version}"]
            
            # We MUST test them alongside target_packages (pyqt6, etc) to catch Python version conflicts!
            test_pkgs = core_gpu_pkgs + target_packages
            
            # UNIQUE NAME is critical so conda threads don't lock each other out
            unique_env = f"_xcop_dryrun_{str(version).replace('.', '_')}_"
            
            cmd = [
                "conda", "create", "--name", unique_env, "--dry-run", "-v", 
                "-c", "pytorch", "-c", "conda-forge", 
                "--solver=libmamba"
            ] + test_pkgs 
            
            # Force unbuffered output so Conda doesn't hold back the text
            import os
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            # Use Popen to stream the output in real-time instead of waiting for it to finish
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            
            for line in process.stdout:
                # By splitting on \r we catch the internal solver steps printed by verbose mode
                clean_line = line.split('\r')[-1].strip()
                
                # We removed the strict filters so you can see the live verbose output!
                if clean_line and not clean_line.startswith(('Channels:', 'Platform:')):
                    # Truncate extremely long internal solver math lines so they don't lag the Tkinter UI
                    if len(clean_line) > 120:
                        clean_line = clean_line[:117] + "..."
                    log(f"      [CUDA {version}] {clean_line}")
                    
            process.wait()
            
            # Catch mid-run cancellations so the thread dies gracefully
            if cancel_check and cancel_check():
                return False, version, None
                
            # We return ONLY the core GPU packages. Target packages are already handled by Step 1.
            return process.returncode == 0, version, core_gpu_pkgs

        # 3. SEQUENTIAL BATCH TESTING
        # Step through the list 1 version at a time for maximum safety on HPC login nodes
        for i in range(0, len(valid_versions), 1):
            # Break out immediately if the user clicked Skip
            if cancel_check and cancel_check():
                log("    [Skip] CUDA scan aborted by user.")
                return cpu_base_pkgs + ["cpuonly"], None
            batch = valid_versions[i:i+1]
            log(f"\n    [Solver] Dry-running CUDA version: {batch[0]}...")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # executor.map keeps the results in the exact same order as 'batch' (highest version first)
                results = list(executor.map(test_cuda_version, batch))
                
            # Check results from highest to lowest
            for success, version, test_gpu_pkgs in results:
                if success:
                    log(f"    [Success] Solver guarantees CUDA {version} works!")
                    # Return the packages directly. Python version will be dynamically resolved.
                    return test_gpu_pkgs, str(version)
                else:
                    log(f"    [Fail] CUDA {version} failed dependency check.")

        log("    [Warning] No GPU versions resolved across all attempts. Falling back to CPU.")
        return cpu_base_pkgs + ["cpuonly"], None 
        
    except Exception as e:
        log(f"    [Error] Exception occurred: {e}")
        return cpu_base_pkgs + ["cpuonly"], None

# We start these empty, they will be populated by the background thread!
PT_PKGS = []
OPTIMAL_CUDA_VER = None
CURRENT_CUDA_VER = None
NEEDS_PYTORCH = False  # Dynamic flag to track if we need to install it

XCOP_ROOT = Path(__file__).resolve().parent

ENV_NAME = "xcop"
SCRIPT_NAME = str(XCOP_ROOT / "xcop")
PACKAGES = ["pyqt6", "matplotlib", "numpy", "mrcfile", "scikit-learn", "pyqtgraph", "PyOpenGL", "biopython"]

DEFAULT_ENV = "xcop"
DEFAULT_SCRIPT = str(XCOP_ROOT / "xcop")
DEFAULT_PACKAGES = ["pyqt6", "matplotlib", "numpy", "mrcfile", "scikit-learn", "pyqtgraph", "PyOpenGL", "biopython"]

UPDATE_REPOSITORY = "songxou0909/xcop"
UPDATE_BRANCH = "main"
UPDATE_REPOSITORY_URL = f"https://github.com/{UPDATE_REPOSITORY}"
UPDATE_RAW_BASE_URL = f"https://raw.githubusercontent.com/{UPDATE_REPOSITORY}/{UPDATE_BRANCH}/"
UPDATE_MANIFEST_PATH = "xcop_savings/update_manifest.json"
UPDATE_TIMEOUT_SECONDS = 20
UPDATE_MAX_FILE_BYTES = 5 * 1024 * 1024
UPDATE_VERSION_VARIABLES = (
    "RELION_VER",
    "XMIPP_VER",
    "SERVALCAT_VER",
    "PHENIX_VER",
    "CHIMERA_VER",
)
UPDATE_EXCLUDED_DIRECTORIES = {
    "AutoIllustrate",
    "FoldX",
    "__pycache__",
    "foundry",
}

FOUNDRY_ENV_NAME = "rfdxcop"
FOUNDRY_CHECKPOINT_FILENAMES = (
    "rfd3_latest.ckpt",
    "proteinmpnn_v_48_020.pt",
    "ligandmpnn_v_32_010_25.pt",
    "rf3_foundry_01_24_latest_remapped.ckpt",
)


def build_foundry_bashrc_entry(checkpoint_dir):
    """Return the managed comment/export pair for an absolute checkpoint path."""
    checkpoint_dir = os.path.abspath(os.fspath(checkpoint_dir))
    bashrc_checkpoint_dir = (
        checkpoint_dir.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("$", "\\$")
        .replace("`", "\\`")
    )
    export_line = f'export FOUNDRY_CHECKPOINT_DIRS="{bashrc_checkpoint_dir}"'
    comment = f"# Added by rfdxcop setup wizard: {export_line}"
    return comment, export_line


def build_xcop_bashrc_entries(script_name, env_name, include_foundry=True):
    """Build every managed ~/.bashrc entry from the same absolute XCOP root."""
    script_dir = os.path.dirname(os.path.abspath(script_name))
    export_line = f"export PATH={script_dir}:$PATH"
    queue_monitor_path = os.path.join(script_dir, "xcop_savings", "queue_monitor.py")
    alias_line = f"alias xq='{queue_monitor_path} &'"
    entries = [
        (f"# Added by {env_name} setup wizard: {export_line}", export_line),
        (f"# Added by {env_name} setup wizard: {alias_line}", alias_line),
    ]
    if include_foundry:
        checkpoint_dir = os.path.join(
            script_dir, "xcop_savings", "foundry", "checkpoints"
        )
        entries.append(build_foundry_bashrc_entry(checkpoint_dir))
    return entries


def conda_environment_exists(env_name):
    """Return whether Conda has an environment whose final path component is env_name."""
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return False
        environment_paths = json.loads(result.stdout).get("envs", [])
        return any(Path(path).name == env_name for path in environment_paths)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_foundry_install_slurm_script(checkpoint_dir):
    """Build a fail-fast Slurm installer for Foundry and its base checkpoints."""
    checkpoint_dir = os.path.abspath(os.fspath(checkpoint_dir))
    quoted_checkpoint_dir = shlex.quote(checkpoint_dir)
    bashrc_export_comment, bashrc_export_line = build_foundry_bashrc_entry(
        checkpoint_dir
    )
    required_checkpoints = "\n".join(
        f"    {shlex.quote(filename)}" for filename in FOUNDRY_CHECKPOINT_FILENAMES
    )

    return f'''#!/bin/bash
#SBATCH --job-name=install_rfd
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=install_foundry_%j.log

set -Eeuo pipefail
trap 'echo "Foundry installation failed at line $LINENO." >&2' ERR

CHECKPOINT_DIR={quoted_checkpoint_dir}
mkdir -p "$CHECKPOINT_DIR"

eval "$(conda shell.bash hook)"
conda activate {FOUNDRY_ENV_NAME}

export FOUNDRY_CHECKPOINT_DIRS="$CHECKPOINT_DIR"

echo "Installing Foundry into the {FOUNDRY_ENV_NAME} environment..."
python -m pip install "rc-foundry[all]"

required_checkpoints=(
{required_checkpoints}
)
for checkpoint_name in "${{required_checkpoints[@]}}"; do
    # Foundry skips any destination that already exists, including an empty file.
    # Remove only empty checkpoint files so an interrupted download can be repaired.
    if [[ -e "$CHECKPOINT_DIR/$checkpoint_name" && ! -s "$CHECKPOINT_DIR/$checkpoint_name" ]]; then
        rm -f "$CHECKPOINT_DIR/$checkpoint_name"
    fi
done

echo "Downloading Foundry base models to $CHECKPOINT_DIR..."
foundry install base-models --checkpoint-dir "$CHECKPOINT_DIR"

for checkpoint_name in "${{required_checkpoints[@]}}"; do
    if [[ ! -s "$CHECKPOINT_DIR/$checkpoint_name" ]]; then
        echo "Missing or empty checkpoint: $CHECKPOINT_DIR/$checkpoint_name" >&2
        exit 1
    fi
done

# Persist the verified absolute path both for interactive shells and for XCOP's
# --export=NONE Slurm jobs, which all activate the rfdxcop Conda environment.
FOUNDRY_BASHRC_EXPORT={shlex.quote(bashrc_export_line)}
FOUNDRY_BASHRC_COMMENT={shlex.quote(bashrc_export_comment)}
BASHRC_PATH="$HOME/.bashrc"
touch "$BASHRC_PATH"
if ! grep -Fqx -- "$FOUNDRY_BASHRC_EXPORT" "$BASHRC_PATH"; then
    if [[ -s "$BASHRC_PATH" ]]; then
        printf '\n' >> "$BASHRC_PATH"
    fi
    printf '%s\n%s\n' "$FOUNDRY_BASHRC_COMMENT" "$FOUNDRY_BASHRC_EXPORT" >> "$BASHRC_PATH"
fi

mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/xcop_foundry_checkpoints.sh" <<'XCOP_FOUNDRY_ENV'
export FOUNDRY_CHECKPOINT_DIRS={quoted_checkpoint_dir}
XCOP_FOUNDRY_ENV

echo "Foundry installation complete. Verified ${{#required_checkpoints[@]}} checkpoints in $CHECKPOINT_DIR."
rm -f "install_foundry_${{SLURM_JOB_ID}}.log"
'''


def append_missing_bashrc_entries(bashrc_path, entries):
    """Append independently managed shell entries and return commands that were added."""
    content = ""
    if os.path.exists(bashrc_path):
        with open(bashrc_path, "r") as handle:
            content = handle.read()

    active_lines = {
        line.strip()
        for line in content.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    missing_entries = [entry for entry in entries if entry[1] not in active_lines]
    if not missing_entries:
        return []

    with open(bashrc_path, "a") as handle:
        if content and not content.endswith("\n"):
            handle.write("\n")
        for comment, command in missing_entries:
            handle.write(f"{comment}\n{command}\n")
    return [command for _comment, command in missing_entries]


def remove_managed_bashrc_entries(bashrc_path, entries):
    """Remove all copies of managed comment/command pairs and return the line count."""
    with open(bashrc_path, "r") as handle:
        lines = handle.readlines()

    stripped_lines = {line.strip() for line in lines}
    comments_to_remove = {
        comment for comment, _command in entries if comment in stripped_lines
    }
    commands_to_remove = {
        command for comment, command in entries if comment in comments_to_remove
    }
    targets = comments_to_remove | commands_to_remove
    if not targets:
        return 0

    lines_to_keep = [line for line in lines if line.strip() not in targets]
    removed_count = len(lines) - len(lines_to_keep)
    with open(bashrc_path, "w") as handle:
        handle.writelines(lines_to_keep)
    return removed_count


class UpdateError(Exception):
    """Raised when an update cannot be verified or safely installed."""


def get_xcop_root():
    """Return the folder containing this setup script."""
    return XCOP_ROOT


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def normalize_update_script_bytes(data):
    """Return script bytes with Unix LF line endings."""
    return data.replace(b"\r\n", b"\n")


def canonicalize_update_script_for_hash(data, preserve_rules=None):
    """Remove installation-specific values before hashing an update script."""
    canonical_data = normalize_update_script_bytes(data)
    preserve_rules = set(preserve_rules or [])
    if not preserve_rules:
        return canonical_data

    try:
        script_text = canonical_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UpdateError("An updateable script is not valid UTF-8 text.") from exc

    if "shebang" in preserve_rules:
        lines = script_text.splitlines(keepends=True)
        if lines and lines[0].startswith("#!"):
            lines.pop(0)
        script_text = "".join(lines)

    if "software_versions" in preserve_rules:
        for name in UPDATE_VERSION_VARIABLES:
            assignment_pattern = re.compile(
                rf'^{re.escape(name)}\s*=\s*(?:"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')',
                flags=re.MULTILINE,
            )
            script_text, replacement_count = assignment_pattern.subn(
                f'{name} = "__XCOP_PRESERVED_VALUE__"',
                script_text,
                count=1,
            )
            if replacement_count != 1:
                raise UpdateError(
                    f"The updateable script does not contain the expected {name} setting."
                )

    return script_text.encode("utf-8")


def update_script_sha256(data, preserve_rules=None):
    """Hash script code while ignoring portable installation settings."""
    return sha256_bytes(canonicalize_update_script_for_hash(data, preserve_rules))


def script_has_shebang(script_path):
    """Return whether a script starts with a Unix shebang."""
    with Path(script_path).open("rb") as script_file:
        return script_file.read(2) == b"#!"


def discover_update_scripts(root=None):
    """Discover public Python scripts and attach their installation rules."""
    root = Path(root) if root else get_xcop_root()
    scripts = []

    main_script = root / "xcop"
    if main_script.is_file():
        scripts.append({
            "path": "xcop",
            "preserve": ["shebang", "software_versions"],
            "executable": True,
        })

    savings_dir = root / "xcop_savings"
    if savings_dir.is_dir():
        for script_path in sorted(savings_dir.rglob("*.py"), key=lambda p: p.as_posix().lower()):
            relative_parts = script_path.relative_to(savings_dir).parts
            if any(part in UPDATE_EXCLUDED_DIRECTORIES for part in relative_parts[:-1]):
                continue

            relative_path = script_path.relative_to(root).as_posix()
            entry = {"path": relative_path}
            preserve_rules = []
            if script_has_shebang(script_path):
                preserve_rules.append("shebang")
            if (
                relative_path == "xcop_savings/queue_monitor.py"
                and "shebang" not in preserve_rules
            ):
                preserve_rules.append("shebang")
            if preserve_rules:
                entry["preserve"] = preserve_rules
            if relative_path == "xcop_savings/queue_monitor.py":
                entry["executable"] = True
            scripts.append(entry)

    setup_script = root / "xcop_setup.py"
    if setup_script.is_file():
        entry = {"path": "xcop_setup.py"}
        if script_has_shebang(setup_script):
            entry["preserve"] = ["shebang"]
        scripts.append(entry)

    return scripts


def generate_update_manifest(root=None):
    """Regenerate the checked update manifest after publishing script changes."""
    root = Path(root) if root else get_xcop_root()
    scripts = []
    for rule in discover_update_scripts(root):
        entry = dict(rule)
        script_path = root / PurePosixPath(rule["path"])
        script_data = script_path.read_bytes()
        lf_script_data = normalize_update_script_bytes(script_data)
        if lf_script_data != script_data:
            script_path.write_bytes(lf_script_data)
        entry["sha256"] = update_script_sha256(
            lf_script_data,
            rule.get("preserve", []),
        )
        scripts.append(entry)

    manifest = {
        "schema_version": 1,
        "repository": UPDATE_REPOSITORY,
        "branch": UPDATE_BRANCH,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "scripts": scripts,
    }
    manifest_path = root / PurePosixPath(UPDATE_MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    return manifest_path


def extract_top_level_string_assignments(script_text, names):
    """Read selected top-level string constants without executing the script."""
    try:
        tree = ast.parse(script_text)
    except SyntaxError as exc:
        raise UpdateError(f"The installed script cannot be parsed: {exc}") from exc

    wanted = set(names)
    values = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in wanted:
                values[target.id] = node.value.value
    return values


def apply_installation_settings(remote_data, local_data, preserve_rules):
    """Apply the installed cluster's shebang and software selections to new code."""
    if not local_data or not preserve_rules:
        return remote_data

    try:
        remote_text = remote_data.decode("utf-8")
        local_text = local_data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise UpdateError("An updateable script is not valid UTF-8 text.") from exc

    if "shebang" in preserve_rules:
        local_lines = local_text.splitlines()
        local_shebang = local_lines[0] if local_lines and local_lines[0].startswith("#!") else None
        remote_lines = remote_text.splitlines(keepends=True)
        newline = "\r\n" if "\r\n" in remote_text else "\n"
        if remote_lines and remote_lines[0].startswith("#!"):
            if local_shebang:
                remote_lines[0] = local_shebang + newline
            else:
                remote_lines.pop(0)
        elif local_shebang:
            remote_lines.insert(0, local_shebang + newline)
        remote_text = "".join(remote_lines)

    if "software_versions" in preserve_rules:
        values = extract_top_level_string_assignments(local_text, UPDATE_VERSION_VARIABLES)
        missing = [name for name in UPDATE_VERSION_VARIABLES if name not in values]
        if missing:
            raise UpdateError(
                "Cannot preserve the installed software selections because these values are missing: "
                + ", ".join(missing)
            )

        for name in UPDATE_VERSION_VARIABLES:
            assignment_pattern = re.compile(
                rf'^{re.escape(name)}\s*=\s*(?:"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')',
                flags=re.MULTILINE,
            )
            remote_text, replacement_count = assignment_pattern.subn(
                f"{name} = {json.dumps(values[name], ensure_ascii=False)}",
                remote_text,
                count=1,
            )
            if replacement_count != 1:
                raise UpdateError(
                    f"The downloaded xcop script does not contain the expected {name} setting."
                )

    return remote_text.encode("utf-8")

# THEME COLORS
BG_COLOR = "#1e1e1e"
FG_COLOR = "#d4d4d4"
ACCENT_COLOR = "#98c379"
BTN_BG = "#3e3e42"
ENTRY_BG = "#252526"
# ==========================================

class SetupWizard:
    def __init__(self, root):
        self.root = root
        self.root.title(f"Setup {SCRIPT_NAME}")
        self.root.geometry("850x650")
        self.root.configure(bg=BG_COLOR)
        
        # Forcefully handle the window close event to kill background threads
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        self.step = 1
        self.mode = tk.StringVar(value="setup")
        self.selected_relion = tk.StringVar(value="")
        self.update_window = None

        # UI Elements: Mode Selection
        self.mode_frame = tk.Frame(root, bg=BG_COLOR)
        self.mode_frame.pack(pady=10)
        
        radio_kwargs = {
            "bg": BG_COLOR, "fg": FG_COLOR, "selectcolor": ENTRY_BG, 
            "activebackground": BG_COLOR, "activeforeground": ACCENT_COLOR, 
            "font": ("Arial", 10, "bold"), "bd": 0, "highlightthickness": 0
        }
        tk.Radiobutton(self.mode_frame, text="Setup Mode", variable=self.mode, value="setup", command=self.update_mode, **radio_kwargs).pack(side=tk.LEFT, padx=15)
        tk.Radiobutton(self.mode_frame, text="Uninstall Mode", variable=self.mode, value="uninstall", command=self.update_mode, **radio_kwargs).pack(side=tk.LEFT, padx=15)

        # Environment Dropdown for Uninstall (Hidden by default)
        self.env_combo_frame = tk.Frame(root, bg=BG_COLOR)
        tk.Label(self.env_combo_frame, text="Target Environment:", font=("Arial", 9, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(side=tk.LEFT, padx=(5, 2))
        self.uninstall_env_var = tk.StringVar(value=ENV_NAME)
        self.uninstall_env_combo = ttk.Combobox(self.env_combo_frame, textvariable=self.uninstall_env_var, state="readonly", width=80)
        self.uninstall_env_combo.pack(side=tk.LEFT)
        self.uninstall_env_combo.bind("<<ComboboxSelected>>", self.on_uninstall_env_selected)

        self.env_tooltip_text = "Select an environment..."
        self.combo_tooltip = ToolTip(self.uninstall_env_combo)
        self.uninstall_env_combo.bind("<Enter>", lambda e: self.combo_tooltip.showtip(self.env_tooltip_text))
        self.uninstall_env_combo.bind("<Leave>", lambda e: self.combo_tooltip.hidetip())

        btn_kwargs = {
            "bg": BTN_BG, "fg": FG_COLOR, "activebackground": ACCENT_COLOR, 
            "activeforeground": BG_COLOR, "relief": tk.FLAT, "bd": 0, "padx": 10, "pady": 4,
            "font": ("Arial", 9, "bold")
        }
        
        self.customize_btn = tk.Button(self.mode_frame, text="Customize Environment", command=self.show_customize_window, **btn_kwargs)
        self.customize_btn.pack(side=tk.LEFT, padx=(5, 5))
        
        # Tooltip binding
        self.customize_tooltip = ToolTip(self.customize_btn)
        self.customize_btn.bind("<Enter>", lambda e: self.customize_tooltip.showtip(self.get_customize_tooltip_text()))
        self.customize_btn.bind("<Leave>", lambda e: self.customize_tooltip.hidetip())

        # External Library Management Button
        self.manage_lib_btn = tk.Button(self.mode_frame, text="Manage Libraries", command=self.show_lib_management_window, **btn_kwargs)
        self.manage_lib_btn.pack(side=tk.LEFT, padx=(10, 5))

        # "Setup Manually" Button moved to the rightmost side
        self.help_button = tk.Button(self.mode_frame, text="Setup Manually", command=self.show_help_window, **btn_kwargs)
        self.help_button.pack(side=tk.RIGHT, padx=(5, 5))
        
        self.status_label = tk.Label(root, text="Checking existing environment...", font=("Arial", 12, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR)
        self.status_label.pack(pady=5)

        # Split panels container
        self.panel_frame = tk.Frame(root, bg=BG_COLOR)
        self.panel_frame.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        # Left Panel Container
        self.left_panel = tk.Frame(self.panel_frame, bg=BG_COLOR, width=220)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self.left_panel.pack_propagate(False)

        # Top Left Panel: Checklist
        self.checklist_frame = tk.Frame(self.left_panel, bg=ENTRY_BG)
        self.checklist_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=(0, 5))
        self.checklist_frame.pack_propagate(False)

        tk.Label(self.checklist_frame, text="Checklist:", font=("Arial", 10, "bold"), bg=ENTRY_BG, fg=ACCENT_COLOR).pack(pady=10)
        
        self.setup_steps_text = ["Install Environment", "Change Permissions", "Get Path", "Update ~/.bashrc", "Select Software Versions"]
        self.uninstall_steps_text = ["Clean ~/.bashrc", "Remove Environment"]
        
        # Define descriptions for the tooltips
        self.setup_tooltips = {
            1: "Creates Conda Environment\nInstalls the specified packages for your script",
            2: "Grants executable permissions (chmod 750) to the target script",
            3: "Finds the Conda Environment's Python path and adds it as a shebang header",
            4: "This allows you to execute your script anywhere by typing the script name directly, without [conda activate]\nYou can remove the .py from the name so you don't have to type .py every time",
            5: "Scans for available software versions (e.g., RELION) to inject into the script (xcop)\nThis is required for xcop to function properly"
        }
        
        self.uninstall_tooltips = {
            1: "Removes the script's PATH export line from your ~/.bashrc\nThis can ONLY remove the content added by Setup.py",
            2: "Deletes the target Conda Environment and all its packages"
        }
        
        self.completed_steps = {"setup": set(), "uninstall": set()}
        self.check_labels = {}
        self.step_tooltips = [] # Keep references to prevent garbage collection
        
        for i, text in enumerate(self.setup_steps_text, 1):
            lbl = tk.Label(self.checklist_frame, text=f"  [ ] {i}. {text}", font=("Arial", 10), bg=ENTRY_BG, fg=FG_COLOR, anchor="w", cursor="hand2")
            lbl.pack(fill=tk.X, padx=10, pady=4)
            lbl.bind("<Button-1>", lambda e, step_num=i: self.on_step_click(step_num, "setup"))
            
            # Add Tooltip
            tip = ToolTip(lbl)
            self.step_tooltips.append(tip)
            tip_text = self.setup_tooltips.get(i, "")
            lbl.bind("<Enter>", lambda e, t=tip, txt=tip_text: t.showtip(txt))
            lbl.bind("<Leave>", lambda e, t=tip: t.hidetip())
            
            self.check_labels[f"setup_{i}"] = lbl
            
        for i, text in enumerate(self.uninstall_steps_text, 1):
            lbl = tk.Label(self.checklist_frame, text=f"  [ ] {i}. {text}", font=("Arial", 10), bg=ENTRY_BG, fg=FG_COLOR, anchor="w", cursor="hand2")
            lbl.bind("<Button-1>", lambda e, step_num=i: self.on_step_click(step_num, "uninstall"))
            
            # Add Tooltip
            tip = ToolTip(lbl)
            self.step_tooltips.append(tip)
            tip_text = self.uninstall_tooltips.get(i, "")
            lbl.bind("<Enter>", lambda e, t=tip, txt=tip_text: t.showtip(txt))
            lbl.bind("<Leave>", lambda e, t=tip: t.hidetip())
            
            self.check_labels[f"uninstall_{i}"] = lbl

        # Bottom Left Panel: Information
        self.info_frame = tk.Frame(self.left_panel, bg=ENTRY_BG)
        self.info_frame.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, pady=(5, 0))
        self.info_frame.pack_propagate(False)

        self.info_text = scrolledtext.ScrolledText(
            self.info_frame, width=20, height=10, 
            bg=ENTRY_BG, fg=FG_COLOR, 
            insertbackground=FG_COLOR, 
            highlightthickness=0,
            font=("Consolas", 9), bd=0, wrap=tk.WORD
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Configure text tags for styling
        self.info_text.tag_config("header", font=("Arial", 9, "bold"), foreground="#ffffff")
        self.info_text.tag_config("value", foreground=ACCENT_COLOR)
        self.info_text.tag_config("warning", foreground="#e5c07b") # Yellow for pending installs
        
        self.update_info_panel()

        # Right Panel: Log Area
        self.log_area = scrolledtext.ScrolledText(
            self.panel_frame, width=65, height=22, 
            bg=BG_COLOR, fg=FG_COLOR, 
            insertbackground=FG_COLOR, 
            highlightthickness=0,
            font=("Consolas", 10), bd=0
        )
        self.log_area.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.log_area.config(state=tk.DISABLED)

        # Options Frame for Checkboxes
        self.options_frame = tk.Frame(root, bg=BG_COLOR)
        self.options_frame.pack(pady=5)
        
        self.autojpg_var = tk.BooleanVar(value=True)
        self.autojpg_chk = tk.Checkbutton(
            self.options_frame, text="Install Deep Learning (AutoJPG)", variable=self.autojpg_var,
            font=("Arial", 10, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR, selectcolor=ENTRY_BG,
            activebackground=BG_COLOR, activeforeground=ACCENT_COLOR, highlightthickness=0, bd=0
        )
        self.autojpg_chk.pack(side=tk.LEFT, padx=10)

        self.rfd_var = tk.BooleanVar(value=True)
        self.rfd_chk = tk.Checkbutton(
            self.options_frame, text="Install RFdiffusion (rfdxcop)", variable=self.rfd_var,
            font=("Arial", 10, "bold"), bg=BG_COLOR, fg=ACCENT_COLOR, selectcolor=ENTRY_BG,
            activebackground=BG_COLOR, activeforeground=ACCENT_COLOR, highlightthickness=0, bd=0
        )
        self.rfd_chk.pack(side=tk.LEFT, padx=10)

        # Action Frame (Button + Checkbox)
        self.action_frame = tk.Frame(root, bg=BG_COLOR)
        self.action_frame.pack(pady=15)

        # Main Button
        self.btn = tk.Button(
            self.action_frame, text="Proceed", 
            font=("Arial", 10, "bold"), 
            bg=BTN_BG, fg=FG_COLOR, 
            activebackground=ACCENT_COLOR, activeforeground=BG_COLOR,
            relief=tk.FLAT, bd=0, padx=20, pady=8,
            command=self.run_step
        )
        self.btn.pack(side=tk.LEFT, padx=10)
        self.btn.config(state=tk.DISABLED)

        # Skip CUDA Button (Hidden by default)
        self.skip_cuda_btn = tk.Button(
            self.action_frame, text="Skip CUDA Installation", 
            font=("Arial", 10, "bold"), 
            bg="#e06c75", fg="#1e1e1e", # Reddish warning color to stand out
            activebackground=BTN_BG, activeforeground=FG_COLOR,
            relief=tk.FLAT, bd=0, padx=20, pady=8,
            command=self.skip_cuda_check
        )

        # Backup Checkbox
        self.backup_var = tk.BooleanVar(value=False)
        self.backup_chk = tk.Checkbutton(
            self.action_frame, text="Backup ~/.bashrc File", variable=self.backup_var,
            font=("Arial", 10), bg=BG_COLOR, fg=FG_COLOR, selectcolor=ENTRY_BG,
            activebackground=BG_COLOR, activeforeground=ACCENT_COLOR,
            highlightthickness=0, bd=0
        )
        # Checkbox is initially hidden; it will be shown when needed.

        # Copy Log Button
        self.copy_log_btn = tk.Button(
            self.action_frame, text="Copy Log", 
            font=("Arial", 10, "bold"), 
            bg=BTN_BG, fg=FG_COLOR, 
            activebackground=ACCENT_COLOR, activeforeground=BG_COLOR,
            relief=tk.FLAT, bd=0, padx=15, pady=8,
            command=self.copy_log_to_clipboard
        )
        self.copy_log_btn.pack(side=tk.RIGHT, padx=10)

        # Script updater. This remains available even when Conda is unavailable.
        self.update_btn = tk.Button(
            self.action_frame, text="Update",
            font=("Arial", 10, "bold"),
            bg=BTN_BG, fg=FG_COLOR,
            activebackground=ACCENT_COLOR, activeforeground=BG_COLOR,
            relief=tk.FLAT, bd=0, padx=15, pady=8,
            command=self.show_update_window
        )
        self.update_btn.pack(side=tk.RIGHT, padx=10)

        # Define safe buttons that should remain available during setup work.
        self.safe_buttons = [
            self.customize_btn,
            self.manage_lib_btn,
            self.help_button,
            self.update_btn,
        ]
        
        # Pre-Flight Conda Check
        if not shutil.which("conda"):
            self.status_label.config(
                text="Error: 'conda' not found. Please load or install Conda first.",
                fg="#e06c75"
            )
            self.set_widget_state(
                self.mode_frame, tk.DISABLED,
                exclude=[self.help_button, self.update_btn]
            )
            self.set_widget_state(self.env_combo_frame, tk.DISABLED)
            return

        # Set initial startup state
        self.has_started = False
        self.btn.config(state=tk.NORMAL, text="Start")
        self.status_label.config(text="Ready. Click 'Start' to scan environment.", fg=FG_COLOR)
        # We do NOT disable the UI or auto-trigger the scan anymore so the user can customize first.

    def prompt_cuda_scan(self):
        """Asks the user if they want to run the slow CUDA dry-run scan."""
        if ENV_NAME == "xcop":
            perform_scan = self.autojpg_var.get()
        else:
            perform_scan = False

        threading.Thread(target=self.run_smart_fetch, args=(perform_scan,), daemon=True).start()

    def skip_cuda_check(self):
        """Triggered by the Skip button to abort the dry-run."""
        self.cancel_dry_run = True
        self.skip_cuda_btn.pack_forget()
        self.log("\n>>> Aborting CUDA scan... switching to CPU fallback.")

    def on_closing(self):
        """Catches the 'X' button click and forces Python to terminate immediately."""
        self.cancel_dry_run = True # Signal any running threads to stop
        self.root.destroy()
        os._exit(0) # Bypass the ThreadPoolExecutor lock and kill the program instantly
        
    def run_smart_fetch(self, perform_scan=True):
        """Runs the internet scraping function in the background so the UI doesn't freeze."""
        global PT_PKGS, OPTIMAL_CUDA_VER
        self.cancel_dry_run = False
        
        # Only ping the internet for PyTorch versions if the target is exactly 'xcop' AND user consented
        if ENV_NAME == "xcop" and perform_scan:
            self.root.after(0, lambda: self.skip_cuda_btn.pack(side=tk.LEFT, padx=10))
            self.root.after(0, lambda: self.status_label.config(text="Scanning online repositories for PyTorch updates... (Please wait)", fg="#e5c07b"))
            self.log(">>> Testing PyTorch-CUDA compatibility with local packages via dry-runs...")
            
            # Pass the cancel checker lambda into the function
            PT_PKGS, OPTIMAL_CUDA_VER = get_optimal_pytorch_setup(
                PACKAGES, 
                log_callback=self.log, 
                cancel_check=lambda: getattr(self, 'cancel_dry_run', False)
            )
            
            self.root.after(0, lambda: self.skip_cuda_btn.pack_forget())
            
        elif ENV_NAME == "xcop" and not perform_scan:
            self.log(">>> Skipped CUDA detection for AutoJPG per user request.")
            # Provide base fallback so PyTorch still installs natively if environment is missing
            PT_PKGS = ["pytorch", "torchvision"]
            OPTIMAL_CUDA_VER = None
        else:
            # Instantly bypass the heavy internet check for bonus/custom environments
            PT_PKGS = []
            OPTIMAL_CUDA_VER = None
        
        # Move on to checking the local environment
        self.root.after(0, lambda: self.status_label.config(text="Checking existing environment...", fg=ACCENT_COLOR))
        self.initial_env_check()

    def set_widget_state(self, widget, state, exclude=None):
        """Recursively sets the state of a widget and its children safely."""
        if exclude is None:
            exclude = []
            
        try:
            if widget in exclude:
                return  # Skip widgets in the exclude list
            if isinstance(widget, ttk.Combobox) and state == tk.NORMAL:
                widget.config(state="readonly")
            else:
                widget.config(state=state)
        except tk.TclError:
            pass
            
        for child in widget.winfo_children():
            self.set_widget_state(child, state, exclude)

    def update_info_panel(self):
        """Updates the configuration info panel with current global variables."""
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        
        # Insert Headers and Values with specific tags
        self.info_text.insert(tk.END, "Current Status:\n\n", "header")
        
        self.info_text.insert(tk.END, "Environment Name:\n", "header")
        self.info_text.insert(tk.END, f"{ENV_NAME}\n\n", "value")
        
        self.info_text.insert(tk.END, "Target Script Path:\n", "header")
        self.info_text.insert(tk.END, f"{SCRIPT_NAME}\n\n", "value")
        
        self.info_text.insert(tk.END, "Base Packages:\n", "header")
        pkg_str = ', '.join(PACKAGES) if PACKAGES else 'None'
        self.info_text.insert(tk.END, f"{pkg_str}\n\n", "value")
        
        if ENV_NAME == "xcop":
            self.info_text.insert(tk.END, "Deep Learning (AutoJPG):\n", "header")
            if NEEDS_PYTORCH:
                self.info_text.insert(tk.END, f"PENDING: {', '.join(PT_PKGS)}\n", "warning")
            else:
                self.info_text.insert(tk.END, "READY / SATISFIED\n", "value")
                
        self.info_text.config(state=tk.DISABLED)

    def update_checklist_ui(self):
        """Redraws the checklist to show completions and the active '>' pointer."""
        mode_str = self.mode.get()
        steps_text = self.setup_steps_text if mode_str == "setup" else self.uninstall_steps_text
        
        for i, text in enumerate(steps_text, 1):
            lbl = self.check_labels[f"{mode_str}_{i}"]
            prefix = ">" if i == self.step else " "
            checked = " √ " if i in self.completed_steps[mode_str] else " "
            color = ACCENT_COLOR if i in self.completed_steps[mode_str] else FG_COLOR
            font_weight = "bold" if i == self.step else "normal"
            
            lbl.config(text=f"{prefix} [{checked}] {i}. {text}", fg=color, font=("Arial", 10, font_weight))
            
        # Show/Hide the Backup Checkbox depending on whether bashrc is about to be modified
        if (mode_str == "setup" and self.step == 4) or (mode_str == "uninstall" and self.step == 1):
            self.backup_chk.pack(side=tk.LEFT, padx=10)
        else:
            self.backup_chk.pack_forget()

    def on_step_click(self, step_num, mode):
        """Allows clicking green (completed) steps to redo them."""
        if mode != self.mode.get(): return
        # Allow clicking if it's already completed or if it's the target step
        if step_num in self.completed_steps[mode] or step_num <= len(self.completed_steps[mode]) + 1:
            self.step = step_num
            self.update_checklist_ui()
            self.status_label.config(text=f"Selected Step {step_num} to run next.", fg=FG_COLOR)

    def initial_env_check(self):
        self.log(">>> Checking current setup status...")
        
        # 1. Check Conda Env & Packages
        check_process = subprocess.run("conda env list", shell=True, stdout=subprocess.PIPE, text=True)
        env_actually_exists = ENV_NAME in check_process.stdout
        base_env_exists = env_actually_exists
        
        if env_actually_exists:
            list_process = subprocess.run(f"conda list -n {ENV_NAME}", shell=True, stdout=subprocess.PIPE, text=True)
            installed_output = list_process.stdout.lower()
            for pkg in PACKAGES:
                # Strip out version tags (e.g. pyqt=6 -> pyqt)
                pkg_name = pkg.split('=')[0].split('<')[0].split('>')[0].strip().lower()
                if not re.search(rf'^{pkg_name}\s+', installed_output, re.MULTILINE):
                    base_env_exists = False
                    self.log(f">>> Package '{pkg_name}' missing from '{ENV_NAME}'. Step 1 marked incomplete.")
                    break
                    
        # --- PyTorch GPU/CPU Check ---
        global NEEDS_PYTORCH
        global PT_STATUS
        NEEDS_PYTORCH = False
        PT_STATUS = "none"
        
        if ENV_NAME == "xcop":
            if env_actually_exists:
                self.log(">>> Checking PyTorch installation status...")
                pt_installed_check = subprocess.run(f"conda run -n {ENV_NAME} python -c \"import torch\"", shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                
                if pt_installed_check.returncode == 0:
                    pt_gpu_check = subprocess.run(f"conda run -n {ENV_NAME} python -c \"import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)\"", shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                    if pt_gpu_check.returncode == 0:
                        PT_STATUS = "gpu"
                        # Detect the exactly installed PyTorch CUDA version
                        cuda_ver_check = subprocess.run(f"conda run -n {ENV_NAME} python -c \"import torch; print(torch.version.cuda)\"", shell=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
                        global CURRENT_CUDA_VER
                        CURRENT_CUDA_VER = cuda_ver_check.stdout.strip()
                        self.log(f">>> PyTorch (GPU version) detected (CUDA {CURRENT_CUDA_VER}). Checking against driver updates...")
                    else:
                        PT_STATUS = "cpu"
                        self.log(">>> PyTorch (CPU version) detected. Pending user confirmation for GPU switch.")
                else:
                    PT_STATUS = "none"
                    self.log(">>> PyTorch not detected. Pending user confirmation.")
            else:
                PT_STATUS = "none"

        # 2. Check Script Permissions
        perm_ok = False
        if os.path.exists(SCRIPT_NAME):
            perm_ok = os.access(SCRIPT_NAME, os.X_OK)
            
        # 3. Check Shebang
        shebang_ok = False
        if os.path.exists(SCRIPT_NAME):
            try:
                with open(SCRIPT_NAME, 'r') as f:
                    first_line = f.readline().strip()
                if first_line.startswith("#!"):
                    res = subprocess.run(f"conda run -n {ENV_NAME} which python", shell=True, stdout=subprocess.PIPE, text=True)
                    output_lines = [line.strip() for line in res.stdout.strip().split('\n') if line.strip()]
                    if output_lines:
                        expected_python = output_lines[-1]
                        if first_line == f"#!{expected_python}":
                            shebang_ok = True
            except Exception:
                pass
                
        # 4. Check ~/.bashrc (Edge cases: Safe Read, ignoring commented lines)
        bashrc_ok = False
        script_dir = os.path.dirname(SCRIPT_NAME)
        bashrc_path = os.path.expanduser("~/.bashrc")
        export_line = f'export PATH={script_dir}:$PATH'
        if os.path.exists(bashrc_path):
            try:
                with open(bashrc_path, "r") as f:
                    for line in f:
                        # Ensures the match isn't on a commented-out line
                        if export_line in line and not line.lstrip().startswith('#'):
                            bashrc_ok = True
                            break
            except Exception:
                pass

        self.root.after(0, self.apply_initial_checks, env_actually_exists, base_env_exists, PT_STATUS, perm_ok, shebang_ok, bashrc_ok)

    def apply_initial_checks(self, env_actually_exists, base_env_exists, pt_status, perm_ok, shebang_ok, bashrc_ok):
        global NEEDS_PYTORCH
        
        env_exists = base_env_exists
        
        # --- Prompt user for PyTorch CPU to GPU Switch & Updates ---
        if ENV_NAME == "xcop":
            if self.autojpg_var.get():
                NEEDS_PYTORCH = True
                if env_actually_exists and pt_status == "gpu" and OPTIMAL_CUDA_VER and CURRENT_CUDA_VER and CURRENT_CUDA_VER == OPTIMAL_CUDA_VER:
                    NEEDS_PYTORCH = False
                    self.log(f">>> PyTorch is already optimal for current GPU driver (CUDA {CURRENT_CUDA_VER}).")
                else:
                    env_exists = False # Force Step 1 to run
                    self.log(">>> Deep Learning modules required. Step 1 marked incomplete.")
            else:
                NEEDS_PYTORCH = False
                env_exists = base_env_exists
                self.log(">>> AutoJPG deep learning packages skipped by user.")
        
        self.update_info_panel()

        self.completed_steps["setup"].clear()
        self.step = 1
        
        if env_exists:
            self.completed_steps["setup"].add(1)
            self.step = 2
            self.log(">>> Environment found.")
            if perm_ok:
                self.completed_steps["setup"].add(2)
                self.step = 3
                self.log(">>> Executable permissions found.")
                if shebang_ok:
                    self.completed_steps["setup"].add(3)
                    self.step = 4
                    self.log(">>> Script shebang matches environment.")
                    if bashrc_ok:
                        self.completed_steps["setup"].add(4)
                        self.step = 5
                        self.log(">>> PATH already in ~/.bashrc.")
        
        if len(self.completed_steps["setup"]) == 0:
            self.log(">>> No existing setup found. Ready to install.")
            
        self.update_checklist_ui()
        
        if self.step > len(self.setup_steps_text):
            self.btn.config(state=tk.DISABLED)
            self.status_label.config(text="All steps completed successfully!", fg=ACCENT_COLOR)
        else:
            if self.mode.get() == "setup" and self.step == 2 and not os.path.exists(SCRIPT_NAME):
                self.btn.config(state=tk.DISABLED)
                self.status_label.config(text=f"Target script missing! Check 'Customize Environment'.", fg="#e06c75")
            else:
                self.btn.config(state=tk.NORMAL)
                self.status_label.config(text=f"Checks complete. Ready to proceed with Step {self.step}.", fg=ACCENT_COLOR)
                
        # Re-enable the UI components now that the check is finished
        self.set_widget_state(self.mode_frame, tk.NORMAL)
        self.set_widget_state(self.env_combo_frame, tk.NORMAL)

    def check_uninstall_status(self):
        self.log(">>> Checking current uninstall status...")
        
        # 1. Check ~/.bashrc
        bashrc_path = os.path.expanduser("~/.bashrc")
        script_dir = os.path.dirname(SCRIPT_NAME)
        exact_comment = f"# Added by {ENV_NAME} setup wizard: export PATH={script_dir}:$PATH"
        bashrc_has_path = False
        
        if os.path.exists(bashrc_path):
            try:
                with open(bashrc_path, "r") as f:
                    for line in f:
                        if exact_comment in line:
                            bashrc_has_path = True
                            break
            except Exception:
                pass
                
        # 2. Check Conda Env
        check_process = subprocess.run("conda env list", shell=True, stdout=subprocess.PIPE, text=True)
        env_exists = ENV_NAME in check_process.stdout
        
        env_list = []
        for line in check_process.stdout.split('\n'):
            if line.strip() and not line.startswith('#'):
                parts = line.split()
                if parts:
                    env_name = parts[0]
                    if env_name.lower() != 'base':
                        env_list.append(env_name)
        
        self.root.after(0, self.apply_uninstall_checks, bashrc_has_path, env_exists, env_list)

    def apply_uninstall_checks(self, bashrc_has_path, env_exists, env_list):
        self.step = 1
        
        # Populate the combobox and ensure our target is selected
        self.uninstall_env_combo['values'] = env_list
        if ENV_NAME in env_list:
            self.uninstall_env_var.set(ENV_NAME)
        elif env_list:
            self.uninstall_env_var.set(env_list[0])
        else:
            self.uninstall_env_var.set("")
            
        # Start fetching packages for the default selected environment
        if self.uninstall_env_var.get():
            self.env_tooltip_text = "Fetching packages..."
            threading.Thread(target=self.fetch_env_packages, args=(self.uninstall_env_var.get(),), daemon=True).start()
        
        if not bashrc_has_path:
            self.completed_steps["uninstall"].add(1)
            self.step = 2
            self.log(">>> ~/.bashrc is already clean for the current target script.")
            
        if not env_exists:
            self.completed_steps["uninstall"].add(2)
            self.log(f">>> Conda environment '{ENV_NAME}' is not present.")
            
        self.update_checklist_ui()
        
        if not bashrc_has_path and not env_exists:
            self.log(">>> Nothing to uninstall. System is already clean.")
            self.log(">>> Uninstall Complete.")
            self.finish_step(3, "Uninstall Complete", True)
        else:
            self.status_label.config(text=f"Ready to proceed with Step {self.step}.", fg=ACCENT_COLOR)
            self.btn.config(state=tk.NORMAL)

    def show_help_window(self):
        top = tk.Toplevel(self.root)
        top.title("Manual Installation Logic")
        top.geometry("700x650")
        top.configure(bg=BG_COLOR)
        
        txt = scrolledtext.ScrolledText(top, bg=ENTRY_BG, fg=FG_COLOR, font=("Consolas", 10), padx=15, pady=15, bd=0)
        txt.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        txt.tag_config("bold", font=("Arial", 10, "bold"), foreground=FG_COLOR)
        txt.tag_config("italic", font=("Arial", 10, "italic"), foreground=FG_COLOR)
        txt.tag_config("code", foreground=ACCENT_COLOR)
        
        content = [
            ("1. Create environment:\n", ""),
            ("conda create --name xcop\nconda activate xcop\nconda install -c conda-forge pyqt6 matplotlib numpy mrcfile scikit-learn pyqtgraph PyOpenGL\n\n", "code"),
            
            ("2. Change permission of the script (xcop): \n", ""),
            ("chmod 750 xcop\n\n", "code"),
            
            ("3. Check where is the python environment and update to the script:\n", ""),
            ("which python\n\n", "code"),
            
            ("If the directory contains ", ""),
            ("~, \n", "italic"),
            ("echo $HOME\n\n", "code"),
            
            ("Replace the ", ""),
            ("~ ", "italic"),
            ("with whatever “echo $HOME” shows, and add a header (shebang) in the first line of your script (xcop): \n", ""),
            ("#! ", "code"),
            ("followed by the \"echo $HOME\" output (without any space)\n", ""),
            ("For example:\n", ""),
            ("#!/home/username/.conda/envs/xcop/bin/python\n\n", "code"),
            
            
            ("4. Add to bash, which means every time you open a new session, auto run the contained commands\n", ""),
            ("vim ~/.bashrc\n\n", "code"),
            
            ("Go to the next line:\n", ""),
            ("press ", ""),
            ("o\n\n", "bold"),
            
            ("Paste the command that should be run at the start of the session, PATH is the directory where your script locates:\n", ""),
            ("paste ", ""),
            ("export PATH=/home/username/path/to/xcop:$PATH\n\n", "code"),
            
            ("Prepare to save:\n", ""),
            ("press ", ""),
            ("esc\n\n", "bold"),
            
            ("Save and exit:\n", ""),
            ("type ", ""),
            (":x\n\n", "code"),
            
            ("Confirm save and exit:\n", ""),
            ("press ", ""),
            ("enter \n\n", "bold"),
            
            ("5. To remove environment path in ", ""),
            ("~", "italic"),
            ("/.bashrc:\n", ""),
            ("vim ~/.bashrc\n\n", "code"),
            
            ("press ", ""),
            ("i", "bold"),
            (" to insert\n\n", ""),
            
            ("delete the added PATH\n\n", ""),
            ("press ", ""),
            ("esc\n\n", "bold"),
            
            ("Save and exit:\n", ""),
            ("type ", ""),
            (":x\n\n", "code"),
            ("press ", ""),
            ("enter \n\n", "bold"),
            
            ("6. Remove the conda environment:\n", ""),
            ("conda deactivate\nconda remove -n xcop --all\ny\n\n", "code"),

            # --- Added Manual CUDA Setup Instructions ---
            ("Manual CUDA Installation Steps:\n", "bold"), # Ensure you have a "bold" tag configured, or use ""
            ("If you prefer to configure CUDA manually inside your environment, open your terminal and run the following commands sequentially:\n\n", ""),
            
            ("1. Activate your conda environment:\n", ""),
            ("conda activate xcop\n\n", "code"),
            
            ("2. Install the required CUDA Toolkit via conda:\n", ""),
            ("conda install -c nvidia cuda-toolkit -y\n\n", "code"),
            
            ("3. Install PyTorch with CUDA support via pip:\n", ""),
            ("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n\n", "code"),
            
            ("4. Test if the system successfully detects your GPU:\n", ""),
            ("python -c \"import torch; print('CUDA Available:', torch.cuda.is_available())\"\n\n", "code"),

            ("Other Useful instructions:\n", "bold"),
            ("1. If need to module load relion (same for other softwares):\n", ""),
            ("module load relion/5.0_gpu_ompi4_cuda111\n\n", "code"),
            
            ("2. To check available software versions:\n", ""),
            ("module avail\n\n", "code"),

            ("3. To add extra external packages (to add <numpy> to xcop):\n", ""),
            ("conda install -n xcop -c conda-forge numpy\n\n", "code"),
            
            ("4. To remove external packages for an environment (to remove <numpy> from xcop):\n", ""),
            ("conda remove -n xcop numpy\n\n", "code")
        ]
        
        for text, tag in content:
            if tag:
                txt.insert(tk.END, text, tag)
            else:
                txt.insert(tk.END, text)
                
        txt.config(state=tk.DISABLED)

    def show_customize_window(self):
        top = tk.Toplevel(self.root)
        top.title("Customize Environment")
        top.geometry("450x260")
        top.configure(bg=BG_COLOR)
        top.transient(self.root)
        top.grab_set()

        tk.Label(top, text="Environment Name:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 10, "bold")).pack(pady=(15, 2))
        env_entry = tk.Entry(top, bg=ENTRY_BG, fg=FG_COLOR, insertbackground=FG_COLOR, width=40, font=("Consolas", 10))
        env_entry.insert(0, ENV_NAME)
        env_entry.pack()

        tk.Label(top, text="Target Script Path:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 10, "bold")).pack(pady=(10, 2))
        
        script_frame = tk.Frame(top, bg=BG_COLOR)
        script_frame.pack()
        
        script_entry = tk.Entry(script_frame, bg=ENTRY_BG, fg=FG_COLOR, insertbackground=FG_COLOR, width=32, font=("Consolas", 10))
        
        # Always prefill so the user can verify the target path
        script_entry.insert(0, SCRIPT_NAME)
            
        script_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        def browse_script():
            filepath = filedialog.askopenfilename(title="Select Script")
            if filepath:
                script_entry.delete(0, tk.END)
                script_entry.insert(0, filepath)
                
        tk.Button(script_frame, text="Browse", command=browse_script, bg=BTN_BG, fg=FG_COLOR, activebackground=ACCENT_COLOR, activeforeground=BG_COLOR, relief=tk.FLAT, bd=0).pack(side=tk.LEFT)

        tk.Label(top, text="Packages (separated by ;):", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 10, "bold")).pack(pady=(10, 2))
        pkg_entry = tk.Entry(top, bg=ENTRY_BG, fg=FG_COLOR, insertbackground=FG_COLOR, width=40, font=("Consolas", 10))
        pkg_entry.insert(0, "; ".join(PACKAGES))
        pkg_entry.pack()

        def submit():
            global ENV_NAME, SCRIPT_NAME, PACKAGES
            new_env = env_entry.get().strip()
            
            # Validate Environment Name
            if not re.match(r"^[a-zA-Z0-9_-]+$", new_env):
                messagebox.showwarning("Invalid Name", "Environment name can only contain letters, numbers, underscores, and hyphens.")
                return
                
            # --- NEW: HPC Global Failsafe ---
            if new_env.lower() in ["base", "root", ""]:
                messagebox.showwarning("Restricted Environment", "You cannot use 'base', 'root', or leave this blank.\n\nTo protect the cluster's global packages, you must use a dedicated custom environment.")
                return
                
            new_script = script_entry.get().strip()
            raw_pkgs = pkg_entry.get().split(';')
            new_pkgs = [p.strip() for p in raw_pkgs if p.strip()]
            
            # --- SMART CHECK: Only update and restart if something actually changed ---
            if new_env == ENV_NAME and new_script == SCRIPT_NAME and new_pkgs == PACKAGES:
                self.log(">>> Configuration unchanged. Continuing current session.")
                top.destroy()
                return
                
            self.log(f"\n>>> Customization applied! Restarting environment checks for '{new_env}'...")
            ENV_NAME = new_env
            SCRIPT_NAME = new_script
            PACKAGES = new_pkgs
            
            self.root.title(f"Setup {SCRIPT_NAME}")
            self.update_customize_btn_visuals()
            self.update_info_panel()
            top.destroy()
            self.update_mode()

        def reset_to_default():
            global ENV_NAME, SCRIPT_NAME, PACKAGES
            
            # --- SMART CHECK: Only restart if we aren't already on defaults ---
            if ENV_NAME == DEFAULT_ENV and SCRIPT_NAME == DEFAULT_SCRIPT and PACKAGES == DEFAULT_PACKAGES:
                self.log(">>> Already using default configuration.")
                top.destroy()
                return
                
            self.log(f"\n>>> Reset to defaults applied! Restarting environment checks for '{DEFAULT_ENV}'...")
            ENV_NAME = DEFAULT_ENV
            SCRIPT_NAME = DEFAULT_SCRIPT
            PACKAGES = list(DEFAULT_PACKAGES)
            
            self.root.title(f"Setup {SCRIPT_NAME}")
            self.update_customize_btn_visuals()
            self.update_info_panel()
            top.destroy()
            self.update_mode()

        btn_frame = tk.Frame(top, bg=BG_COLOR)
        btn_frame.pack(pady=20)

        tk.Button(btn_frame, text="Reset", command=reset_to_default, bg=BTN_BG, fg=FG_COLOR, activebackground="#e06c75", activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=15, pady=6, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=10)
        tk.Button(btn_frame, text="Submit", command=submit, bg=BTN_BG, fg=FG_COLOR, activebackground=ACCENT_COLOR, activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=25, pady=6, font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=10)

    def get_customize_tooltip_text(self):
        if ENV_NAME == DEFAULT_ENV and SCRIPT_NAME == DEFAULT_SCRIPT and PACKAGES == DEFAULT_PACKAGES:
            return ""  # Don't show tooltip if everything is default
        return f"Current Custom Setup:\nEnvironment Name: {ENV_NAME}\nTarget Script Path: {SCRIPT_NAME}\nPackages: {'; '.join(PACKAGES)}"

    def update_customize_btn_visuals(self):
        if ENV_NAME == DEFAULT_ENV and SCRIPT_NAME == DEFAULT_SCRIPT and PACKAGES == DEFAULT_PACKAGES:
            # Revert to default theme colors
            self.customize_btn.config(bg=BTN_BG, fg=FG_COLOR)
        else:
            # Turn yellow to indicate customized state (using a dark-theme friendly yellow)
            self.customize_btn.config(bg="#e5c07b", fg="#1e1e1e")

    def on_uninstall_env_selected(self, event):
        global ENV_NAME
        selected_env = self.uninstall_env_var.get()
        if selected_env and selected_env != ENV_NAME:
            ENV_NAME = selected_env
            self.log(f"\n>>> Target uninstall environment changed to: {ENV_NAME}")
            self.update_customize_btn_visuals()
            self.update_info_panel()
            
            # Fetch packages for the newly selected environment
            self.env_tooltip_text = "Fetching packages..."
            threading.Thread(target=self.fetch_env_packages, args=(ENV_NAME,), daemon=True).start()
            
            # Re-trigger uninstall check for the newly selected env
            self.completed_steps["uninstall"].clear()
            self.log_area.config(state=tk.NORMAL)
            self.log_area.delete(1.0, tk.END)
            self.log_area.config(state=tk.DISABLED)
            self.btn.config(state=tk.DISABLED)
            self.status_label.config(text=f"Checking uninstall status for {ENV_NAME}...", fg=ACCENT_COLOR)
            threading.Thread(target=self.check_uninstall_status, daemon=True).start()

    def fetch_env_packages(self, env_name):
        """Fetches explicitly installed external packages in the background."""
        try:
            # --from-history outputs only packages explicitly requested by the user, skipping default dependencies
            cmd = f"conda env export -n {env_name} --from-history"
            process = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            packages = []
            in_deps = False
            for line in process.stdout.split('\n'):
                line = line.strip()
                if line.startswith('dependencies:'):
                    in_deps = True
                    continue
                if in_deps and line.startswith('-'):
                    pkg = line.lstrip('- ').strip()
                    # Extra filter to ignore python itself if it pops up
                    if not pkg.startswith(('python', 'conda')):
                        packages.append(pkg)
            
            if packages:
                self.env_tooltip_text = f"External Packages in {env_name}:\n  • " + "\n  • ".join(packages)
            else:
                self.env_tooltip_text = f"No external packages found in {env_name}."
        except Exception:
            self.env_tooltip_text = "Failed to load packages."

    def show_lib_management_window(self):
        top = tk.Toplevel(self.root)
        top.title("External Library Management")
        top.geometry("500x400")
        top.configure(bg=BG_COLOR)
        top.transient(self.root)
        
        # --- Environment Section ---
        tk.Label(top, text="Target Environment:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 10, "bold")).pack(pady=(15, 2))
        env_var = tk.StringVar()
        env_combo = ttk.Combobox(top, textvariable=env_var, state="readonly", width=45)
        env_combo.pack(pady=(0, 15))
        
        tk.Frame(top, bg=BTN_BG, height=1).pack(fill=tk.X, padx=30, pady=5)
        
        # --- Add Packages Section ---
        tk.Label(top, text="Add Packages (separated by ;):", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 10, "bold")).pack(pady=(10, 2))
        add_entry = tk.Entry(top, bg=ENTRY_BG, fg=FG_COLOR, insertbackground=FG_COLOR, width=48, font=("Consolas", 10))
        add_entry.pack(pady=(0, 10))
        
        add_btn = tk.Button(top, text="Add", bg=BTN_BG, fg=FG_COLOR, activebackground=ACCENT_COLOR, activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=25, pady=6, font=("Arial", 9, "bold"))
        add_btn.pack(pady=(0, 15))

        tk.Frame(top, bg=BTN_BG, height=1).pack(fill=tk.X, padx=30, pady=5)

        # --- Remove Packages Section ---
        tk.Label(top, text="Remove Installed Package:", bg=BG_COLOR, fg=FG_COLOR, font=("Arial", 10, "bold")).pack(pady=(10, 2))
        pkg_var = tk.StringVar()
        pkg_combo = ttk.Combobox(top, textvariable=pkg_var, state="readonly", width=45)
        pkg_combo.pack(pady=(0, 10))

        remove_btn = tk.Button(top, text="Remove", bg=BTN_BG, fg=FG_COLOR, activebackground="#e06c75", activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=25, pady=6, font=("Arial", 9, "bold"))
        remove_btn.pack(pady=(0, 15))
        
        tk.Frame(top, bg=BTN_BG, height=1).pack(fill=tk.X, padx=30, pady=5)

        status_lbl = tk.Label(top, text="Loading environments...", bg=BG_COLOR, fg=ACCENT_COLOR, font=("Arial", 9))
        status_lbl.pack(pady=10)

        # Logic functions
        def load_envs():
            try:
                process = subprocess.run("conda env list", shell=True, stdout=subprocess.PIPE, text=True)
                envs = [
                    line.split()[0] for line in process.stdout.split('\n') 
                    if line.strip() and not line.startswith('#') and line.split() and line.split()[0].lower() != 'base'
                ]
                # Check if window still exists before scheduling UI update
                if top.winfo_exists():
                    top.after(0, lambda: update_env_combo(envs))
            except Exception:
                if top.winfo_exists():
                    top.after(0, lambda: status_lbl.config(text="Failed to load environments.", fg="#e06c75"))

        def update_env_combo(envs):
            # Double-check existence inside the main thread callback
            if not top.winfo_exists(): return
            env_combo['values'] = envs
            if ENV_NAME in envs:
                env_combo.set(ENV_NAME)
            elif envs:
                env_combo.set(envs[0])
            status_lbl.config(text="Environments loaded. Ready.", fg=FG_COLOR)
            load_packages(env_combo.get())

        threading.Thread(target=load_envs, daemon=True).start()

        def load_packages(env_name):
            if not top.winfo_exists(): return
            pkg_combo.set('')
            pkg_combo['values'] = []
            if not env_name: return
            status_lbl.config(text=f"Loading installed packages for {env_name}...", fg=ACCENT_COLOR)
            
            def _fetch():
                try:
                    process = subprocess.run(f"conda list -n {env_name}", shell=True, stdout=subprocess.PIPE, text=True)
                    pkgs = [line.split()[0] for line in process.stdout.split('\n') if line.strip() and not line.startswith('#')]
                    if top.winfo_exists():
                        top.after(0, lambda: update_pkg_combo(pkgs))
                except Exception:
                    if top.winfo_exists():
                        top.after(0, lambda: status_lbl.config(text="Failed to load packages.", fg="#e06c75"))
            threading.Thread(target=_fetch, daemon=True).start()

        def update_pkg_combo(pkgs):
            if not top.winfo_exists(): return
            pkg_combo['values'] = pkgs
            if pkgs:
                pkg_combo.set(pkgs[0])
            status_lbl.config(text="Ready.", fg=FG_COLOR)

        env_combo.bind("<<ComboboxSelected>>", lambda e: load_packages(env_combo.get()))

        def add_pkgs():
            env = env_combo.get()
            raw_pkgs = add_entry.get().split(';')
            requested_pkgs = [p.strip() for p in raw_pkgs if p.strip()]
            
            if not env or not requested_pkgs: return
            
            # --- Smart Install Check ---
            installed_pkgs = pkg_combo['values']
            pkgs_to_install = []
            skipped_pkgs = []
            
            for pkg in requested_pkgs:
                if pkg in installed_pkgs:
                    skipped_pkgs.append(pkg)
                else:
                    pkgs_to_install.append(pkg)
                    
            if not pkgs_to_install:
                status_lbl.config(text="All requested packages are already installed.", fg=ACCENT_COLOR)
                add_entry.delete(0, tk.END)
                return
                
            pkg_str = " ".join(pkgs_to_install)
            
            # --- NEW: Dynamic Channel & Size Warning for Manual Installs ---
            channels = "-c conda-forge --solver=libmamba"
            needs_heavy_warning = False
            
            for p in pkgs_to_install:
                if "pytorch" in p.lower() or "torchvision" in p.lower() or "cuda" in p.lower():
                    channels = "-c pytorch -c conda-forge --solver=libmamba"
                    needs_heavy_warning = True
                    break

            if needs_heavy_warning:
                proceed = messagebox.askokcancel(
                    "Large Download Required",
                    f"You are requesting deep learning packages manually:\n"
                    f"{pkg_str}\n\n"
                    f"This may involve a large download (~2GB+). Do you want to proceed?"
                )
                if not proceed:
                    status_lbl.config(text="Installation cancelled by user.", fg="#e5c07b")
                    return
            # -------------------------------------------------------------
            
            msg = f"Installing {pkg_str}..."
            if skipped_pkgs:
                msg = f"Skipped {len(skipped_pkgs)} existing. " + msg
                
            status_lbl.config(text=msg, fg=ACCENT_COLOR)
            
            def _install():
                cmd = f"conda install -n {env} {channels} {pkg_str} -y"
                res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode == 0:
                    top.after(0, lambda: status_lbl.config(text="Installation successful!", fg=ACCENT_COLOR))
                    top.after(0, lambda: add_entry.delete(0, tk.END))
                    top.after(0, lambda: load_packages(env))
                else:
                    top.after(0, lambda: status_lbl.config(text="Installation failed! Check console.", fg="#e06c75"))
                    err_msg = res.stderr.decode('utf-8')
                    print(err_msg)
                    # Trigger the copyable manual run popup instead of a standard generic error
                    top.after(0, lambda: self.show_manual_install_popup(cmd))
            threading.Thread(target=_install, daemon=True).start()

        def remove_pkg():
            env = env_combo.get()
            pkg = pkg_combo.get()
            
            if not env or not pkg: return
            
            # --- Smart Removal Logic ---
            pkgs_to_remove = [pkg]
            installed_pkgs = pkg_combo['values']
            
            # If user selected the main package, check for its -base counterpart
            if f"{pkg}-base" in installed_pkgs:
                pkgs_to_remove.append(f"{pkg}-base")
            # If user selected the -base package, check for the main wrapper
            elif pkg.endswith("-base"):
                main_pkg = pkg[:-5]
                if main_pkg in installed_pkgs:
                    pkgs_to_remove.append(main_pkg)
                    
            pkg_str = " ".join(pkgs_to_remove)
            # ---------------------------
            
            status_lbl.config(text=f"Removing {pkg_str}...", fg=ACCENT_COLOR)
            
            def _remove():
                cmd = f"conda remove -n {env} {pkg_str} -y"
                res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if res.returncode == 0:
                    top.after(0, lambda: status_lbl.config(text="Removal successful!", fg=ACCENT_COLOR))
                    top.after(0, lambda: load_packages(env))
                else:
                    top.after(0, lambda: status_lbl.config(text="Removal failed! Check console.", fg="#e06c75"))
                    err_msg = res.stderr.decode('utf-8')
                    print(err_msg)
                    top.after(0, lambda: messagebox.showerror("Removal Failed", f"Failed to remove packages from '{env}'.\n\nPlease check the terminal/console for the exact Conda error."))
            threading.Thread(target=_remove, daemon=True).start()

        add_btn.config(command=add_pkgs)
        remove_btn.config(command=remove_pkg)

    def check_script_needs_versions(self):
        if not os.path.exists(SCRIPT_NAME):
            return True
        try:
            with open(SCRIPT_NAME, 'r', encoding='utf-8') as f:
                content = f.read()
            # Check for at least one of the target global variables
            patterns = ['RELION_VER', 'XMIPP_VER', 'SERVALCAT_VER', 'PHENIX_VER', 'CHIMERA_VER']
            for p in patterns:
                if re.search(rf'^{p}\s*=', content, flags=re.MULTILINE):
                    return True
            return False
        except Exception:
            return True

    def update_mode(self):
        """Resets the UI when switching between Setup and Uninstall modes."""
        if self.mode.get() == "setup":
            is_customized = not (ENV_NAME == DEFAULT_ENV and SCRIPT_NAME == DEFAULT_SCRIPT and PACKAGES == DEFAULT_PACKAGES)
            if is_customized and not self.check_script_needs_versions():
                self.setup_steps_text = ["Install Environment", "Change Permissions", "Get Path", "Update ~/.bashrc"]
            else:
                self.setup_steps_text = ["Install Environment", "Change Permissions", "Get Path", "Update ~/.bashrc", "Select Software Versions"]

        self.has_started = False
        self.step = 1
        self.log_area.config(state=tk.NORMAL)
        self.log_area.delete(1.0, tk.END)
        self.log_area.config(state=tk.DISABLED)
        self.btn.config(state=tk.NORMAL, bg=BTN_BG, text="Start")
        self.status_label.config(text="Ready. Click 'Start' to scan environment.", fg=FG_COLOR)
        
        for lbl in self.check_labels.values():
            lbl.pack_forget()
            
        if self.mode.get() == "setup":
            self.env_combo_frame.pack_forget()
            self.options_frame.pack(pady=5, before=self.action_frame)
            self.autojpg_chk.config(state=tk.NORMAL)
            self.rfd_chk.config(state=tk.NORMAL)
            for i in range(1, len(self.setup_steps_text) + 1):
                self.check_labels[f"setup_{i}"].pack(fill=tk.X, padx=10, pady=4)
        else:
            self.options_frame.pack_forget()
            self.env_combo_frame.pack(pady=(0, 10), before=self.status_label)
            self.completed_steps["uninstall"].clear()
            for i in range(1, len(self.uninstall_steps_text) + 1):
                self.check_labels[f"uninstall_{i}"].pack(fill=tk.X, padx=10, pady=4)

    def log(self, message, replace_last=False):
        """Safely update the text area from a background thread."""
        def _update_ui():
            self.log_area.config(state=tk.NORMAL)
            if replace_last:
                # Delete the previously inserted line so the progress bar updates in place
                self.log_area.delete("end-2l linestart", "end-1l linestart")
            self.log_area.insert(tk.END, message + "\n")
            self.log_area.see(tk.END)
            self.log_area.config(state=tk.DISABLED)
            
        # Schedule the UI update to run on the main thread immediately
        self.root.after(0, _update_ui)

    def copy_log_to_clipboard(self):
        """Copies the contents of the log area to the system clipboard."""
        try:
            raw_content = self.log_area.get(1.0, tk.END).strip()
            
            # Strip out invisible terminal control characters (like backspaces and null bytes from Conda's progress bars)
            # that cause the system clipboard to truncate the string prematurely.
            clean_log = "".join(c for c in raw_content if c.isprintable() or c in "\n\t")
            
            self.root.clipboard_clear()
            self.root.clipboard_append(clean_log)
            self.root.update()
            
            # Visual feedback: flash the button to the accent color
            self.copy_log_btn.config(text="Copied!", bg=ACCENT_COLOR, fg=BG_COLOR)
            self.root.after(2000, lambda: self.copy_log_btn.config(text="Copy Log", bg=BTN_BG, fg=FG_COLOR))
        except Exception as e:
            self.log(f">>> Failed to copy log to clipboard: {e}")

    @staticmethod
    def _read_update_url(url, max_bytes=UPDATE_MAX_FILE_BYTES):
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "xcop-script-updater/1",
                "Accept": "application/octet-stream",
                "Cache-Control": "no-cache",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=UPDATE_TIMEOUT_SECONDS) as response:
                data = response.read(max_bytes + 1)
        except urllib.error.HTTPError as exc:
            raise UpdateError(f"GitHub returned HTTP {exc.code} for {url}") from exc
        except urllib.error.URLError as exc:
            raise UpdateError(f"Could not connect to GitHub: {exc.reason}") from exc

        if len(data) > max_bytes:
            raise UpdateError(f"The downloaded file exceeds the {max_bytes}-byte safety limit.")
        return data

    @staticmethod
    def _validate_manifest(manifest):
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
            raise UpdateError("GitHub returned an unsupported update manifest.")
        if manifest.get("repository") != UPDATE_REPOSITORY:
            raise UpdateError("The update manifest points to an unexpected repository.")
        if manifest.get("branch") != UPDATE_BRANCH:
            raise UpdateError("The update manifest points to an unexpected branch.")

        scripts = manifest.get("scripts")
        if not isinstance(scripts, list) or not scripts:
            raise UpdateError("The update manifest does not contain any scripts.")

        allowed_preserve_rules = {"shebang", "software_versions"}
        seen_paths = set()
        validated = []
        for raw_entry in scripts:
            if not isinstance(raw_entry, dict):
                raise UpdateError("The update manifest contains an invalid script entry.")

            relative_path = raw_entry.get("path")
            if not isinstance(relative_path, str):
                raise UpdateError("An update entry is missing its script path.")
            posix_path = PurePosixPath(relative_path)
            if (
                posix_path.is_absolute()
                or ".." in posix_path.parts
                or relative_path.startswith(("/", "\\"))
                or "\\" in relative_path
            ):
                raise UpdateError(f"Unsafe update path rejected: {relative_path}")

            is_main_script = relative_path == "xcop"
            is_setup_script = relative_path == "xcop_setup.py"
            is_savings_script = (
                len(posix_path.parts) >= 2
                and posix_path.parts[0] == "xcop_savings"
                and posix_path.suffix == ".py"
                and not any(part in UPDATE_EXCLUDED_DIRECTORIES for part in posix_path.parts[1:-1])
            )
            if not (is_main_script or is_setup_script or is_savings_script):
                raise UpdateError(f"Non-script update path rejected: {relative_path}")
            if relative_path in seen_paths:
                raise UpdateError(f"Duplicate update path rejected: {relative_path}")
            seen_paths.add(relative_path)

            expected_hash = raw_entry.get("sha256")
            if not isinstance(expected_hash, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
                raise UpdateError(f"Invalid SHA-256 checksum for {relative_path}")

            preserve_rules = raw_entry.get("preserve", [])
            if (
                not isinstance(preserve_rules, list)
                or any(rule not in allowed_preserve_rules for rule in preserve_rules)
            ):
                raise UpdateError(f"Invalid preservation rule for {relative_path}")

            validated.append({
                "path": relative_path,
                "sha256": expected_hash.lower(),
                "preserve": list(preserve_rules),
                "executable": bool(raw_entry.get("executable", False)),
            })

        manifest = dict(manifest)
        manifest["scripts"] = validated
        return manifest

    def _load_remote_update_manifest(self):
        manifest_url = UPDATE_RAW_BASE_URL + urllib.parse.quote(
            UPDATE_MANIFEST_PATH, safe="/"
        )
        raw_manifest = self._read_update_url(manifest_url, max_bytes=1024 * 1024)
        try:
            manifest = json.loads(raw_manifest.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UpdateError("GitHub returned an unreadable update manifest.") from exc
        return self._validate_manifest(manifest)

    @staticmethod
    def _local_update_path(relative_path):
        root = get_xcop_root()
        target = (root / PurePosixPath(relative_path)).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise UpdateError(f"Unsafe local update path rejected: {relative_path}") from exc
        return target

    def _download_verified_script(self, entry):
        relative_path = entry["path"]
        download_url = UPDATE_RAW_BASE_URL + urllib.parse.quote(relative_path, safe="/")
        remote_data = self._read_update_url(download_url)
        actual_hash = update_script_sha256(remote_data, entry.get("preserve", []))
        if actual_hash != entry["sha256"]:
            raise UpdateError(
                f"Checksum verification failed for {relative_path}. Nothing was replaced."
            )
        return normalize_update_script_bytes(remote_data)

    def _prepare_remote_script(self, entry):
        target = self._local_update_path(entry["path"])
        remote_data = self._download_verified_script(entry)
        local_data = target.read_bytes() if target.is_file() else None
        return target, apply_installation_settings(
            remote_data, local_data, entry.get("preserve", [])
        )

    def _collect_update_statuses(self):
        manifest = self._load_remote_update_manifest()
        statuses = []
        for entry in manifest["scripts"]:
            target = self._local_update_path(entry["path"])
            try:
                if not target.is_file():
                    needs_update = True
                    status_text = "Not installed"
                else:
                    _, prepared_data = self._prepare_remote_script(entry)
                    needs_update = target.read_bytes() != prepared_data
                    status_text = "Update available" if needs_update else "Up to date"
                statuses.append({
                    "entry": entry,
                    "needs_update": needs_update,
                    "status": status_text,
                    "error": None,
                })
            except Exception as exc:
                statuses.append({
                    "entry": entry,
                    "needs_update": False,
                    "status": "Cannot check",
                    "error": str(exc),
                })
        return manifest, statuses

    @staticmethod
    def _validate_downloaded_script(relative_path, script_data):
        try:
            script_text = script_data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise UpdateError(f"{relative_path} is not valid UTF-8 text.") from exc
        try:
            compile(script_text, relative_path, "exec")
        except SyntaxError as exc:
            raise UpdateError(
                f"Python syntax validation failed for {relative_path}: {exc}"
            ) from exc

    @staticmethod
    def _replace_script_atomically(target, script_data, executable=False):
        target.parent.mkdir(parents=True, exist_ok=True)
        original_mode = stat.S_IMODE(target.stat().st_mode) if target.exists() else None
        new_fd, new_name = tempfile.mkstemp(
            prefix=f".{target.name}.", suffix=".xcop-update", dir=str(target.parent)
        )
        backup_name = None
        try:
            with os.fdopen(new_fd, "wb") as new_file:
                new_file.write(script_data)
                new_file.flush()
                os.fsync(new_file.fileno())

            if target.exists():
                backup_fd, backup_name = tempfile.mkstemp(
                    prefix=f".{target.name}.", suffix=".xcop-update-backup",
                    dir=str(target.parent),
                )
                os.close(backup_fd)
                shutil.copy2(target, backup_name)

            os.replace(new_name, target)
            new_name = None
            os.chmod(target, original_mode if original_mode is not None else (0o750 if executable else 0o644))
        except Exception:
            if backup_name and os.path.exists(backup_name):
                os.replace(backup_name, target)
                backup_name = None
            elif original_mode is None and target.exists():
                target.unlink()
            raise
        finally:
            if new_name and os.path.exists(new_name):
                os.unlink(new_name)
            if backup_name and os.path.exists(backup_name):
                os.unlink(backup_name)

    def _start_script_update(self, top, entry, status_label, update_button):
        update_button.config(state=tk.DISABLED, text="Updating...")
        status_label.config(text="Downloading...", fg="#e5c07b")

        def worker():
            try:
                target, prepared_data = self._prepare_remote_script(entry)
                self._validate_downloaded_script(entry["path"], prepared_data)
                self._replace_script_atomically(
                    target, prepared_data, executable=entry.get("executable", False)
                )
            except Exception as exc:
                error_message = str(exc)

                def show_error():
                    if not top.winfo_exists():
                        return
                    status_label.config(text="Update failed", fg="#e06c75")
                    update_button.config(state=tk.NORMAL, text="Update")
                    messagebox.showerror("Update Failed", error_message, parent=top)

                self.root.after(0, show_error)
                return

            def show_success():
                if not top.winfo_exists():
                    return
                status_label.config(text="Up to date", fg=ACCENT_COLOR)
                update_button.destroy()
                self.log(f">>> Updated {entry['path']} from {UPDATE_REPOSITORY_URL}")
                if entry["path"] == "xcop_setup.py":
                    messagebox.showinfo(
                        "Setup Updated",
                        "xcop_setup.py was updated successfully. Close and restart this setup program to use the new version.",
                        parent=top,
                    )

            self.root.after(0, show_success)

        threading.Thread(target=worker, daemon=True).start()

    def generate_update_manifest_from_ui(self, parent=None):
        """Generate the publishing manifest without requiring a shell command."""
        try:
            manifest_path = generate_update_manifest()
        except Exception as exc:
            error_message = str(exc)
            self.log(f">>> Failed to generate update manifest: {error_message}")
            messagebox.showerror(
                "Manifest Generation Failed",
                f"Could not generate the update manifest.\n\n{error_message}",
                parent=parent,
            )
            return

        self.log(f">>> Generated update manifest: {manifest_path}")
        messagebox.showinfo(
            "Update Manifest Generated",
            "The update manifest was generated successfully.\n\n"
            f"Saved to:\n{manifest_path}\n\n"
            "All managed scripts were converted to Unix LF line endings. This button "
            "does not upload anything to GitHub. Publish this manifest together with "
            "every changed or newly added script.",
            parent=parent,
        )

    def show_update_window(self):
        """Show GitHub-backed updates and buttons only for scripts that differ."""
        if self.update_window is not None and self.update_window.winfo_exists():
            self.update_window.lift()
            self.update_window.focus_force()
            return

        top = tk.Toplevel(self.root)
        self.update_window = top
        top.title("Update xcop Scripts")
        top.geometry("820x580")
        top.minsize(700, 420)
        top.configure(bg=BG_COLOR)
        top.transient(self.root)

        def close_window():
            self.update_window = None
            top.destroy()

        top.protocol("WM_DELETE_WINDOW", close_window)

        tk.Label(
            top, text="xcop Script Updates", font=("Arial", 14, "bold"),
            bg=BG_COLOR, fg=ACCENT_COLOR
        ).pack(pady=(15, 4))
        repository_link = tk.Label(
            top, text=UPDATE_REPOSITORY_URL, font=("Arial", 10, "underline"),
            bg=BG_COLOR, fg="#61afef", cursor="hand2"
        )
        repository_link.pack(pady=(0, 4))
        repository_link.bind(
            "<Button-1>", lambda _event: webbrowser.open_new_tab(UPDATE_REPOSITORY_URL)
        )
        tk.Label(
            top,
            text="Only scripts that differ from the verified GitHub snapshot show an Update button.",
            font=("Arial", 9), bg=BG_COLOR, fg=FG_COLOR
        ).pack(pady=(0, 8))

        status_var = tk.StringVar(value="Checking GitHub for script updates...")
        status_label = tk.Label(
            top, textvariable=status_var, font=("Arial", 9, "bold"),
            bg=BG_COLOR, fg="#e5c07b"
        )
        status_label.pack(pady=(0, 8))

        list_container = tk.Frame(top, bg=ENTRY_BG)
        list_container.pack(fill=tk.BOTH, expand=True, padx=18, pady=5)
        canvas = tk.Canvas(list_container, bg=ENTRY_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(list_container, orient=tk.VERTICAL, command=canvas.yview)
        rows_frame = tk.Frame(canvas, bg=ENTRY_BG)
        rows_window = canvas.create_window((0, 0), window=rows_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        rows_frame.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.bind(
            "<Configure>",
            lambda event: canvas.itemconfigure(rows_window, width=event.width),
        )

        footer = tk.Frame(top, bg=BG_COLOR)
        footer.pack(fill=tk.X, padx=18, pady=(8, 14))
        refresh_button = tk.Button(
            footer, text="Check Again", font=("Arial", 9, "bold"),
            bg=BTN_BG, fg=FG_COLOR, activebackground=ACCENT_COLOR,
            activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=15, pady=6
        )
        refresh_button.pack(side=tk.LEFT)
        tk.Button(
            footer, text="Generate Update Manifest",
            command=lambda: self.generate_update_manifest_from_ui(top),
            font=("Arial", 9, "bold"), bg=BTN_BG, fg=FG_COLOR,
            activebackground=ACCENT_COLOR, activeforeground=BG_COLOR,
            relief=tk.FLAT, bd=0, padx=15, pady=6
        ).pack(side=tk.LEFT, padx=(10, 0))
        tk.Button(
            footer, text="Close", command=close_window,
            font=("Arial", 9, "bold"), bg=BTN_BG, fg=FG_COLOR,
            activebackground="#e06c75", activeforeground=BG_COLOR,
            relief=tk.FLAT, bd=0, padx=15, pady=6
        ).pack(side=tk.RIGHT)

        def render_results(manifest, statuses):
            if not top.winfo_exists():
                return
            for child in rows_frame.winfo_children():
                child.destroy()

            headers = (("Script", 0), ("Status", 1), ("Action", 2))
            for title, column in headers:
                tk.Label(
                    rows_frame, text=title, font=("Arial", 9, "bold"),
                    bg=ENTRY_BG, fg="#ffffff", anchor="w"
                ).grid(row=0, column=column, sticky="ew", padx=10, pady=(8, 6))
            rows_frame.grid_columnconfigure(0, weight=1)
            rows_frame.grid_columnconfigure(1, minsize=130)
            rows_frame.grid_columnconfigure(2, minsize=100)

            updates_found = 0
            errors_found = 0
            for row_number, result in enumerate(statuses, 1):
                entry = result["entry"]
                relative_path = entry["path"]
                file_url = (
                    f"{UPDATE_REPOSITORY_URL}/blob/{UPDATE_BRANCH}/"
                    + urllib.parse.quote(relative_path, safe="/")
                )
                script_link = tk.Label(
                    rows_frame, text=relative_path, font=("Consolas", 9, "underline"),
                    bg=ENTRY_BG, fg="#61afef", cursor="hand2", anchor="w"
                )
                script_link.grid(row=row_number, column=0, sticky="ew", padx=10, pady=5)
                script_link.bind(
                    "<Button-1>",
                    lambda _event, url=file_url: webbrowser.open_new_tab(url),
                )

                color = ACCENT_COLOR
                if result["error"]:
                    color = "#e06c75"
                    errors_found += 1
                elif result["needs_update"]:
                    color = "#e5c07b"
                    updates_found += 1
                item_status = tk.Label(
                    rows_frame, text=result["status"], font=("Arial", 9),
                    bg=ENTRY_BG, fg=color, anchor="w"
                )
                item_status.grid(row=row_number, column=1, sticky="w", padx=10, pady=5)

                if result["needs_update"]:
                    item_button = tk.Button(
                        rows_frame, text="Update", font=("Arial", 8, "bold"),
                        bg=BTN_BG, fg=FG_COLOR, activebackground=ACCENT_COLOR,
                        activeforeground=BG_COLOR, relief=tk.FLAT, bd=0,
                        padx=12, pady=4
                    )
                    item_button.config(
                        command=lambda e=entry, s=item_status, b=item_button:
                            self._start_script_update(top, e, s, b)
                    )
                    item_button.grid(row=row_number, column=2, padx=10, pady=5)
                elif result["error"]:
                    error_text = result["error"]
                    error_label = tk.Label(
                        rows_frame, text="Details", font=("Arial", 8, "underline"),
                        bg=ENTRY_BG, fg="#e06c75", cursor="hand2"
                    )
                    error_label.grid(row=row_number, column=2, padx=10, pady=5)
                    error_label.bind(
                        "<Button-1>",
                        lambda _event, text=error_text: messagebox.showerror(
                            "Update Check Failed", text, parent=top
                        ),
                    )

            generated_at = manifest.get("generated_at", "unknown time")
            if errors_found:
                status_var.set(
                    f"Check incomplete: {errors_found} script(s) could not be verified. "
                    f"Repository snapshot: {generated_at}"
                )
                status_label.config(fg="#e06c75")
            elif updates_found:
                status_var.set(
                    f"{updates_found} script update(s) available. Repository snapshot: {generated_at}"
                )
                status_label.config(fg="#e5c07b")
            else:
                status_var.set(f"All scripts are up to date. Repository snapshot: {generated_at}")
                status_label.config(fg=ACCENT_COLOR)
            refresh_button.config(state=tk.NORMAL)

        def show_check_error(error_message):
            if not top.winfo_exists():
                return
            status_var.set(f"Update check failed: {error_message}")
            status_label.config(fg="#e06c75")
            refresh_button.config(state=tk.NORMAL)

        def check_for_updates():
            if not top.winfo_exists():
                return
            refresh_button.config(state=tk.DISABLED)
            status_var.set("Checking GitHub for script updates...")
            status_label.config(fg="#e5c07b")
            for child in rows_frame.winfo_children():
                child.destroy()

            def worker():
                try:
                    manifest, statuses = self._collect_update_statuses()
                except Exception as exc:
                    self.root.after(0, lambda message=str(exc): show_check_error(message))
                    return
                self.root.after(0, lambda: render_results(manifest, statuses))

            threading.Thread(target=worker, daemon=True).start()

        refresh_button.config(command=check_for_updates)
        check_for_updates()

    def run_command(self, cmd, cwd=None):
        """Executes a shell command and logs its output continuously."""
        self.log(f"\n>>> Executing: {cmd}")
        if cwd:
            self.log(f">>> Working directory: {cwd}")
        
        # Force unbuffered output so Conda flushes text immediately
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=cwd,
        )
        
        current_line = ""
        last_was_cr = False
        
        while True:
            char = process.stdout.read(1).decode('utf-8', errors='replace')
            if not char and process.poll() is not None:
                if current_line:
                    self.log(current_line, replace_last=last_was_cr)
                break
                
            # Conda uses \r for progress bars and \b for the ...working... spinner
            if char in ['\r', '\b']: 
                if current_line.strip():
                    self.log(current_line, replace_last=last_was_cr)
                    last_was_cr = True
                current_line = ""
            elif char == '\n': 
                if current_line.strip():
                    self.log(current_line, replace_last=last_was_cr)
                last_was_cr = False
                current_line = ""
            else:
                current_line += char
                
        process.wait()
        return process.returncode == 0

    def fail_setup_step(self, title, message):
        """Stop the current setup step and restore controls after a hard failure."""
        self.log(f">>> ERROR: {message}")

        def restore_controls():
            messagebox.showerror(title, message)
            self.btn.config(state=tk.NORMAL, bg=BTN_BG)
            self.backup_chk.config(state=tk.NORMAL)
            self.autojpg_chk.config(state=tk.NORMAL)
            self.rfd_chk.config(state=tk.NORMAL)
            self.set_widget_state(self.mode_frame, tk.NORMAL)
            self.set_widget_state(self.env_combo_frame, tk.NORMAL)
            self.status_label.config(text=title, fg="#e06c75")

        self.root.after(0, restore_controls)

    # ==================== SETUP MODE ====================

    def show_manual_install_popup(self, failed_cmd):
        """Displays a copyable text box when HPC kills a heavy installation."""
        top = tk.Toplevel(self.root)
        top.title("Installation Terminated (HPC Policy)")
        top.geometry("600x250")
        top.configure(bg=BG_COLOR)
        top.transient(self.root)
        top.grab_set()

        msg = (
            "The HPC cluster policy likely terminated the installation due to CPU/time limits.\n\n"
            "Please copy the exact command below, run it manually in your terminal, "
            "and then restart this setup wizard once it finishes."
        )
        tk.Label(top, text=msg, bg=BG_COLOR, fg="#e5c07b", font=("Arial", 10, "bold"), justify=tk.LEFT, wraplength=550).pack(pady=(15, 10), padx=20)

        text_frame = tk.Frame(top, bg=BG_COLOR)
        text_frame.pack(fill=tk.X, padx=20)

        cmd_text = tk.Text(text_frame, height=4, bg=ENTRY_BG, fg=ACCENT_COLOR, font=("Consolas", 10), wrap=tk.WORD, bd=1, relief=tk.SOLID)
        cmd_text.insert(tk.END, failed_cmd)
        cmd_text.config(state=tk.DISABLED)
        cmd_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        def copy_to_clipboard():
            self.root.clipboard_clear()
            self.root.clipboard_append(failed_cmd)
            self.root.update()
            copy_btn.config(text="Copied!", bg=ACCENT_COLOR, fg=BG_COLOR)
            top.after(2000, lambda: copy_btn.config(text="Copy", bg=BTN_BG, fg=FG_COLOR))

        copy_btn = tk.Button(text_frame, text="Copy", command=copy_to_clipboard, bg=BTN_BG, fg=FG_COLOR, activebackground=ACCENT_COLOR, activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, font=("Arial", 9, "bold"))
        copy_btn.pack(side=tk.LEFT, padx=(10, 0), fill=tk.Y)

        tk.Button(top, text="Close", command=top.destroy, bg=BTN_BG, fg=FG_COLOR, activebackground="#e06c75", activeforeground=BG_COLOR, relief=tk.FLAT, bd=0, padx=20, pady=5, font=("Arial", 9, "bold")).pack(pady=20)

    def execute_setup_step_1(self):
        global NEEDS_PYTORCH
        self.log(f"--- Setup Step 1: Checking and installing conda environment '{ENV_NAME}' ---")
        
        # --- Deep Learning Package Selection ---
        if NEEDS_PYTORCH and ENV_NAME == "xcop":
            active_pkgs = PACKAGES + PT_PKGS
            channels = "-c pytorch -c conda-forge --solver=libmamba"
        else:
            active_pkgs = PACKAGES + ["python"]
            channels = "-c conda-forge --solver=libmamba"

        install_success = True

        self.log(f"Checking for existing '{ENV_NAME}' environment...")
        check_process = subprocess.run("conda env list", shell=True, stdout=subprocess.PIPE, text=True)
        
        if ENV_NAME in check_process.stdout:
            self.log(f"\n>>> Environment '{ENV_NAME}' already exists! Skipping creation.")
            self.log(">>> Checking which packages are missing to avoid unnecessary solver overhead...")
            
            # Get currently installed packages
            list_process = subprocess.run(f"conda list -n {ENV_NAME}", shell=True, stdout=subprocess.PIPE, text=True)
            installed_output = list_process.stdout.lower()
            
            pkgs_to_install = []
            for pkg in active_pkgs:
                pkg_name = pkg.split('=')[0].split('<')[0].split('>')[0].strip().lower()
                
                # FORCE PyTorch packages to re-install if we are explicitly switching from CPU to GPU
                if NEEDS_PYTORCH and pkg in (PT_PKGS + ["python"]):
                    pkgs_to_install.append(pkg)
                    continue

                if not re.search(rf'^{pkg_name}\s+', installed_output, re.MULTILINE):
                    pkgs_to_install.append(pkg)
                    
            install_cmd = ""
            if not pkgs_to_install:
                self.log("\n>>> All required packages are already installed.")
                install_success = True
            else:
                # Wrap any packages with wildcards in quotes to protect them from bash expansion
                safe_pkgs = [f"'{p}'" if '*' in p else p for p in pkgs_to_install]
                pkg_str = " ".join(safe_pkgs)
                install_cmd = f"conda install -n {ENV_NAME} {channels} {pkg_str} -y"
                self.log(f"\n>>> Installing missing packages: {pkg_str}")
                install_success = self.run_command(install_cmd)
        else:
            self.log(f"\n>>> Environment not found. Creating '{ENV_NAME}' and solving dependencies...")
            safe_pkgs = [f"'{p}'" if '*' in p else p for p in active_pkgs]
            pkg_str = " ".join(safe_pkgs)
            
            if pkg_str:
                install_cmd = f"conda create --name {ENV_NAME} {channels} {pkg_str} -y"
            else:
                install_cmd = f"conda create --name {ENV_NAME} python -y"
                
            install_success = self.run_command(install_cmd)
                
        if not install_success:
            self.log(">>> ERROR: Process terminated. Likely killed by HPC CPU/time policy.")
            self.root.after(0, lambda: self.show_manual_install_popup(install_cmd))
            # Safely re-enable UI elements so the user isn't stuck
            self.root.after(0, lambda: self.btn.config(state=tk.NORMAL, bg=BTN_BG))
            self.root.after(0, lambda: self.set_widget_state(self.mode_frame, tk.NORMAL))
            self.root.after(0, lambda: self.set_widget_state(self.env_combo_frame, tk.NORMAL))
            return # Halt step execution

        # Install RFdiffusion if selected
        if self.rfd_var.get():
            self.log(f"\n>>> Installing Foundry environment ({FOUNDRY_ENV_NAME})...")
            if conda_environment_exists(FOUNDRY_ENV_NAME):
                self.log(
                    f">>> Conda environment '{FOUNDRY_ENV_NAME}' already exists; "
                    "reusing it and repairing any missing installation files."
                )
            else:
                rfd_cmd = f"conda create -n {FOUNDRY_ENV_NAME} python=3.12 -y"
                if not self.run_command(rfd_cmd):
                    self.fail_setup_step(
                        "Foundry Environment Creation Failed",
                        f"Could not create the '{FOUNDRY_ENV_NAME}' Conda environment. "
                        "The checkpoint installation was not submitted.",
                    )
                    return

            script_dir = os.path.dirname(SCRIPT_NAME)
            xcop_savings_dir = os.path.abspath(os.path.join(script_dir, "xcop_savings"))
            checkpoint_dir = os.path.abspath(
                os.path.join(xcop_savings_dir, "foundry", "checkpoints")
            )
            os.makedirs(checkpoint_dir, exist_ok=True)

            slurm_script_path = os.path.join(xcop_savings_dir, "install_foundry.sh")
            slurm_content = build_foundry_install_slurm_script(checkpoint_dir)
            with open(slurm_script_path, "w", newline="\n") as handle:
                handle.write(slurm_content)

            self.log(">>> Submitting Foundry installation to Slurm and waiting for verification...")
            self.log(
                ">>> The Foundry package installation and checkpoint downloads run "
                "inside the Slurm job; once submitted, they continue independently "
                "if this setup window is closed."
            )
            install_success = self.run_command(
                "sbatch --wait install_foundry.sh",
                cwd=xcop_savings_dir,
            )
            if not install_success:
                self.fail_setup_step(
                    "Foundry Installation Failed",
                    "The Foundry Slurm job failed. XCOP kept install_foundry.sh and "
                    f"install_foundry_<job-id>.log in {xcop_savings_dir} for diagnosis.",
                )
                return

            missing_checkpoints = [
                filename
                for filename in FOUNDRY_CHECKPOINT_FILENAMES
                if not os.path.isfile(os.path.join(checkpoint_dir, filename))
                or os.path.getsize(os.path.join(checkpoint_dir, filename)) == 0
            ]
            if missing_checkpoints:
                self.fail_setup_step(
                    "Foundry Checkpoint Verification Failed",
                    f"The Slurm job finished, but these checkpoints are missing from "
                    f"{checkpoint_dir}: {', '.join(missing_checkpoints)}",
                )
                return

            self.log(
                f">>> Foundry installation verified successfully in {checkpoint_dir}."
            )

        # Tell the script the PyTorch requirement is satisfied and refresh the UI panel
        NEEDS_PYTORCH = False
        self.root.after(0, self.update_info_panel)

        self.root.after(0, self.finish_step, 2, "Start Step 2: Change Permissions")

    def execute_setup_step_2(self):
        self.log(f"--- Setup Step 2: Changing permissions for {SCRIPT_NAME} ---")
        
        # Real-time existence check
        if not os.path.exists(SCRIPT_NAME):
            self.log(f">>> ERROR: Target script '{SCRIPT_NAME}' not found.")
            self.root.after(0, lambda: messagebox.showerror("File Not Found", f"The target script '{SCRIPT_NAME}' could not be found.\n\nPlease fix the path in 'Customize Environment'."))
            self.root.after(0, lambda: self.btn.config(state=tk.NORMAL, bg=BTN_BG))
            self.root.after(0, lambda: self.set_widget_state(self.mode_frame, tk.NORMAL))
            self.root.after(0, lambda: self.set_widget_state(self.env_combo_frame, tk.NORMAL))
            self.root.after(0, lambda: self.backup_chk.config(state=tk.NORMAL))
            return
            
        self.run_command(f"chmod 750 {SCRIPT_NAME}")
        
        qm_path = os.path.join(os.path.dirname(SCRIPT_NAME), "xcop_savings", "queue_monitor.py")
        if os.path.exists(qm_path):
            self.run_command(f"chmod +x {qm_path}")
        else:
            self.log(f">>> Warning: '{qm_path}' not found, skipping permissions.")
            
        self.root.after(0, self.finish_step, 3, "Start Step 3: Get Path")

    def execute_setup_step_3(self):
        self.log("--- Setup Step 3: Updating Script Interpreter ---")
        self.log(f">>> Fetching Python path for '{ENV_NAME}'...")
        
        try:
            # Capture the path instead of just printing it
            result = subprocess.run(f"conda run -n {ENV_NAME} which python", shell=True, stdout=subprocess.PIPE, text=True)
            
            # Extract the actual path (taking the last line in case conda prints warnings first)
            output_lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            
            if output_lines and os.path.exists(SCRIPT_NAME):
                python_path = output_lines[-1]
                self.log(f">>> Found Python path: {python_path}")
                self.log(f">>> Updating shebang in {SCRIPT_NAME}...")
                
                with open(SCRIPT_NAME, 'r') as f:
                    lines = f.readlines()
                
                new_shebang = f"#!{python_path}\n"
                
                # If the file already has a shebang, replace it. Otherwise, insert it at the top.
                if lines and lines[0].startswith("#!"):
                    lines[0] = new_shebang
                else:
                    lines.insert(0, new_shebang)
                    
                with open(SCRIPT_NAME, 'w') as f:
                    f.writelines(lines)
                    
                self.log(">>> Successfully updated script interpreter for main script.")
                
                # Apply same shebang logic to queue_monitor.py
                qm_path = os.path.join(os.path.dirname(SCRIPT_NAME), "xcop_savings", "queue_monitor.py")
                if os.path.exists(qm_path):
                    self.log(f">>> Updating shebang in {qm_path}...")
                    with open(qm_path, 'r') as f:
                        qm_lines = f.readlines()
                        
                    if qm_lines and qm_lines[0].startswith("#!"):
                        qm_lines[0] = new_shebang
                    else:
                        qm_lines.insert(0, new_shebang)
                        
                    with open(qm_path, 'w') as f:
                        f.writelines(qm_lines)
                    self.log(">>> Successfully updated script interpreter for queue_monitor.py.")
                else:
                    self.log(f">>> Warning: '{qm_path}' not found, skipping shebang update.")
            else:
                self.log(f">>> Warning: Could not find Python path or {SCRIPT_NAME} does not exist.")
        except Exception as e:
            self.log(f">>> Error updating interpreter: {e}")

        self.root.after(0, self.finish_step, 4, "Start Step 4: Update ~/.bashrc")

    def backup_bashrc(self, bashrc_path, target_dir):
        """Creates a timestamped backup of ~/.bashrc in the setup script's directory."""
        import shutil
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        backup_name = f".bashrc_{timestamp}"
        backup_path = os.path.join(target_dir, backup_name)
        try:
            shutil.copy2(bashrc_path, backup_path)
            self.log(f">>> Checkbox selected: Backed up ~/.bashrc to: {backup_name}")
        except Exception as e:
            self.log(f">>> Warning: Failed to backup ~/.bashrc: {e}")

    def execute_setup_step_4(self, do_backup):
        self.log("--- Setup Step 4: Adding script directory to ~/.bashrc ---")
        
        script_dir = os.path.dirname(SCRIPT_NAME)
        bashrc_path = os.path.expanduser("~/.bashrc")

        # 1. Backup if requested and file exists
        if do_backup and os.path.exists(bashrc_path):
            self.backup_bashrc(bashrc_path, script_dir)

        include_foundry = self.rfd_var.get() or os.path.exists(
            os.path.join(script_dir, "xcop_savings", "foundry")
        )
        entries = build_xcop_bashrc_entries(
            SCRIPT_NAME,
            ENV_NAME,
            include_foundry=include_foundry,
        )

        try:
            added_commands = append_missing_bashrc_entries(bashrc_path, entries)
            if added_commands:
                self.log(f"\n>>> Successfully appended to {bashrc_path}:")
                for command in added_commands:
                    self.log(command)
                self.log(
                    "\nNOTE: Run 'source ~/.bashrc' or restart your terminal "
                    "for changes to take effect."
                )
            else:
                self.log(f"\n>>> All XCOP shell entries are already active in {bashrc_path}.")
        except Exception as e:
            self.log(f"Error writing to {bashrc_path}: {e}")

        if len(self.setup_steps_text) == 5:
            self.root.after(0, self.finish_step, 5, "Start Step 5: Select Software Versions")
        else:
            self.root.after(0, self.finish_step, 5, "Setup Complete", True)

    def execute_setup_step_5(self):
        self.log("--- Setup Step 5: Fetching Available Software Versions ---")
        self.log(">>> Running 'module avail'...")
        
        # Using bash -l -c to ensure the module system aliases are loaded
        cmd = "bash -l -c 'module avail 2>&1'"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, text=True)
        output, _ = process.communicate()
        
        software_modules = {
            'relion': [],
            'xmipp': [],
            'servalcat': [],
            'phenix': [],
            'chimera': []
        }
        
        for line in output.split('\n'):
            for word in line.split():
                # Strip out common Lmod artifact markers like (default) or (D)
                clean_word = word.replace('(default)', '').replace('(D)', '').strip()
                word_lower = clean_word.lower()
                
                for sw in software_modules.keys():
                    if sw in word_lower and clean_word not in software_modules[sw]:
                        software_modules[sw].append(clean_word)
        
        # Ensure we have a fallback if a module isn't found on the system
        for sw in software_modules:
            if not software_modules[sw]:
                software_modules[sw].append(f"Not Found: Ask IT support to install {sw}")
                
        self.log(">>> Module search complete.")
        self.root.after(0, self.prompt_software_selection, software_modules)

    def prompt_software_selection(self, software_modules):
        """Displays a popup window for the user to select the desired software versions."""
        top = tk.Toplevel(self.root)
        top.title("Select Software Versions")
        top.geometry("550x320")
        top.configure(bg=BG_COLOR)
        top.transient(self.root) # Keeps popup on top of main window
        top.grab_set()           # Prevents interacting with main window until closed

        tk.Label(top, text="Select versions for your environment:", font=("Arial", 10, "bold"), bg=BG_COLOR, fg=FG_COLOR).pack(pady=10)
        
        combos = {}
        # Create a dropdown for each software package
        for sw, modules in software_modules.items():
            frame = tk.Frame(top, bg=BG_COLOR)
            frame.pack(fill=tk.X, padx=20, pady=5)
            
            tk.Label(frame, text=f"{sw.capitalize()}:", font=("Arial", 10), bg=BG_COLOR, fg=FG_COLOR, width=12, anchor="e").pack(side=tk.LEFT, padx=5)

            combo = ttk.Combobox(frame, values=modules, state="readonly", width=50)
            if modules:
                combo.current(0)
            combo.pack(side=tk.LEFT, padx=5)
            combos[sw] = combo
            
        def on_send_to_script():
            selections = {sw: combo.get() for sw, combo in combos.items()}
            self.log(f">>> User selected versions: {selections}")
            
            # Update the target script using regex to safely target the variables
            if os.path.exists(SCRIPT_NAME):
                try:
                    with open(SCRIPT_NAME, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Determine values: use an empty string if the software was "Not Found..."
                    val_relion = "" if selections['relion'].startswith("Not Found") else selections['relion']
                    val_xmipp = "" if selections['xmipp'].startswith("Not Found") else selections['xmipp']
                    val_servalcat = "" if selections['servalcat'].startswith("Not Found") else selections['servalcat']
                    val_phenix = "" if selections['phenix'].startswith("Not Found") else selections['phenix']
                    val_chimera = "" if selections['chimera'].startswith("Not Found") else selections['chimera']

                    # Apply updates safely to global variables only
                    content = re.sub(r'^RELION_VER\s*=\s*["\'].*?["\']', f'RELION_VER = "{val_relion}"', content, flags=re.MULTILINE)
                    content = re.sub(r'^XMIPP_VER\s*=\s*["\'].*?["\']', f'XMIPP_VER = "{val_xmipp}"', content, flags=re.MULTILINE)
                    content = re.sub(r'^SERVALCAT_VER\s*=\s*["\'].*?["\']', f'SERVALCAT_VER = "{val_servalcat}"', content, flags=re.MULTILINE)
                    content = re.sub(r'^PHENIX_VER\s*=\s*["\'].*?["\']', f'PHENIX_VER = "{val_phenix}"', content, flags=re.MULTILINE)
                    content = re.sub(r'^CHIMERA_VER\s*=\s*["\'].*?["\']', f'CHIMERA_VER = "{val_chimera}"', content, flags=re.MULTILINE)
                        
                    with open(SCRIPT_NAME, 'w', encoding='utf-8') as f:
                        f.write(content)
                        
                    self.log(f">>> Successfully updated {SCRIPT_NAME} with selected versions.")
                except Exception as e:
                    self.log(f">>> Error updating {SCRIPT_NAME}: {e}")
            else:
                self.log(f">>> Target script {SCRIPT_NAME} not found. Skipping file modification.")
            
            top.destroy()
            self.finish_step(6, "Setup Complete", True)

        tk.Button(
            top, text=f"Send to {SCRIPT_NAME}", 
            bg=BTN_BG, fg=FG_COLOR, 
            activebackground=ACCENT_COLOR, activeforeground=BG_COLOR,
            relief=tk.FLAT, bd=0, padx=15, pady=5,
            command=on_send_to_script
        ).pack(pady=20)

    # ==================== UNINSTALL MODE ====================

    def execute_uninstall_step_1(self, do_backup):
        self.log("--- Uninstall Step 1: Removing PATH from ~/.bashrc ---")
        
        bashrc_path = os.path.expanduser("~/.bashrc")
        entries = build_xcop_bashrc_entries(
            SCRIPT_NAME,
            ENV_NAME,
            include_foundry=True,
        )
        
        if not os.path.exists(bashrc_path):
            self.log(f">>> {bashrc_path} does not exist. Nothing to remove.")
            self.root.after(0, self.finish_step, 2, "Start Step 2: Remove Environment")
            return
            
        if do_backup:
            self.backup_bashrc(bashrc_path, os.path.dirname(os.path.abspath(__file__)))
            
        try:
            removed_count = remove_managed_bashrc_entries(bashrc_path, entries)
            if removed_count > 0:
                self.log(f">>> Successfully removed {removed_count} lines from {bashrc_path} for '{ENV_NAME}'.")
            else:
                self.log(f">>> Target environment '{ENV_NAME}' not found in {bashrc_path}. Nothing to remove.")
        except Exception as e:
            self.log(f"Error modifying {bashrc_path}: {e}")

        self.root.after(0, self.finish_step, 2, "Start Step 2: Remove Environment")

    def execute_uninstall_step_2(self):
        self.log(f"--- Uninstall Step 2: Removing Conda Environment '{ENV_NAME}' ---")
        self.run_command(f"conda remove -n {ENV_NAME} --all -y")
        if ENV_NAME == "xcop":
            self.log(f"--- Removing Conda Environment 'rfdxcop' ---")
            self.run_command(f"conda remove -n rfdxcop --all -y")
        self.root.after(0, self.finish_step, 3, "Uninstall Complete", True)
        self.root.after(100, self.check_uninstall_status)

    # ==================== CONTROLLER ====================

    def finish_step(self, next_step, btn_text, disable_btn=False):
        """Updates the UI after a step finishes."""
        mode_str = self.mode.get()
        
        # Mark current step as completed
        self.completed_steps[mode_str].add(self.step)
        
        # Auto-advance to the next uncompleted step, unless completion forces it
        self.step = next_step
        while self.step in self.completed_steps[mode_str] and not disable_btn:
            self.step += 1
            if (mode_str == "setup" and self.step > len(self.setup_steps_text)) or \
               (mode_str == "uninstall" and self.step > len(self.uninstall_steps_text)):
                disable_btn = True
                break
                
        self.update_checklist_ui()
        self.btn.config(text="Proceed", state=tk.DISABLED if disable_btn else tk.NORMAL, bg=BTN_BG if not disable_btn else ENTRY_BG)
        
        if disable_btn:
            self.status_label.config(text="All steps completed successfully!", fg=ACCENT_COLOR)
            self.set_widget_state(self.mode_frame, tk.NORMAL)
            self.set_widget_state(self.env_combo_frame, tk.NORMAL)
            
            # Show Completion Popup for Setup Mode
            if self.mode.get() == "setup":
                messagebox.showinfo("Setup Complete", f"Setup finished, please re-start your session manually to enable the feature to call {SCRIPT_NAME} in any directory")
        else:
            if self.mode.get() == "uninstall":
                self.status_label.config(text=f"Waiting to proceed to Step {next_step}... (Do NOT close the program!)", fg="#e06c75") 
            elif self.mode.get() == "setup" and self.step == 2 and not os.path.exists(SCRIPT_NAME):
                self.btn.config(state=tk.DISABLED, bg=ENTRY_BG)
                self.status_label.config(text=f"Target script missing! Cannot proceed to Step 2.", fg="#e06c75")
            else:
                self.status_label.config(text=f"Waiting to proceed to Step {next_step}...", fg=FG_COLOR)
            
            # --- "ALL OR NONE" LOCK ---
            # Only unlock the toolbar if they are completely stuck at Step 2 due to a missing file.
            # Otherwise, keep it locked between successful steps to prevent midway tampering!
            if self.mode.get() == "setup" and self.step == 2 and not os.path.exists(SCRIPT_NAME):
                self.set_widget_state(self.mode_frame, tk.NORMAL)
                self.set_widget_state(self.env_combo_frame, tk.NORMAL)

    def run_step(self):
        """Triggers the appropriate background thread based on mode and current step."""
        # --- NEW: Intercept the first click to run the initial environment scans ---
        if not getattr(self, 'has_started', False):
            self.has_started = True
            self.btn.config(text="Proceed", state=tk.DISABLED, bg=ENTRY_BG)
            self.autojpg_chk.config(state=tk.DISABLED)
            self.rfd_chk.config(state=tk.DISABLED)
            self.set_widget_state(self.mode_frame, tk.DISABLED, exclude=self.safe_buttons)
            self.set_widget_state(self.env_combo_frame, tk.DISABLED)
            
            if self.mode.get() == "setup":
                self.prompt_cuda_scan()
            else:
                self.status_label.config(text="Checking uninstall status...", fg=ACCENT_COLOR)
                threading.Thread(target=self.check_uninstall_status, daemon=True).start()
            return

        # Pre-flight check for Step 2 to ensure the target script actually exists
        if self.mode.get() == "setup" and self.step == 2:
            if not os.path.exists(SCRIPT_NAME):
                self.log(f">>> ERROR: Target script '{SCRIPT_NAME}' not found.")
                messagebox.showerror("File Not Found", f"The target script '{SCRIPT_NAME}' could not be found.\n\nPlease fix the path in 'Customize Environment'.")
                return  # Exit early so the UI doesn't get disabled and stuck

        self.btn.config(state=tk.DISABLED, bg=ENTRY_BG)
        self.backup_chk.config(state=tk.DISABLED)
        self.set_widget_state(self.mode_frame, tk.DISABLED)
        self.set_widget_state(self.env_combo_frame, tk.DISABLED)
            
        do_backup = self.backup_var.get()
        
        if self.mode.get() == "setup":
            if self.step == 1:
                self.status_label.config(text="Setup Step 1: Checking Environment...", fg=FG_COLOR)
                threading.Thread(target=self.execute_setup_step_1, daemon=True).start()
            elif self.step == 2:
                self.status_label.config(text="Setup Step 2: Changing Permissions...", fg=FG_COLOR)
                threading.Thread(target=self.execute_setup_step_2, daemon=True).start()
            elif self.step == 3:
                self.status_label.config(text="Setup Step 3: Fetching Path...", fg=FG_COLOR)
                threading.Thread(target=self.execute_setup_step_3, daemon=True).start()
            elif self.step == 4:
                self.status_label.config(text="Setup Step 4: Updating ~/.bashrc...", fg=FG_COLOR)
                threading.Thread(target=self.execute_setup_step_4, args=(do_backup,), daemon=True).start()
            elif self.step == 5:
                self.status_label.config(text="Setup Step 5: Fetching Relion Versions...", fg=FG_COLOR)
                threading.Thread(target=self.execute_setup_step_5, daemon=True).start()
                
        elif self.mode.get() == "uninstall":
            self.log("\n==================================================================")
            self.log("!!! WARNING: Do NOT close the program until uninstall finishes !!!")
            self.log("==================================================================\n")
            
            if self.step == 1:
                self.status_label.config(text="Uninstall Step 1: Cleaning ~/.bashrc... (Do NOT close the program!)", fg="#e06c75")
                threading.Thread(target=self.execute_uninstall_step_1, args=(do_backup,), daemon=True).start()
            elif self.step == 2:
                self.status_label.config(text="Uninstall Step 2: Removing Environment... (Do NOT close the program!)", fg="#e06c75")
                threading.Thread(target=self.execute_uninstall_step_2, daemon=True).start()

if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--generate-update-manifest":
        manifest_path = generate_update_manifest()
        print(f"Updated {manifest_path}")
        raise SystemExit(0)

    root = tk.Tk()
    app = SetupWizard(root)
    root.mainloop()
