"""
app.py — Amodal Instance Segmentation Analysis Tool

A simple desktop GUI that:
  1. Shows hardware info (GPU, VRAM, disk, RAM) with pass/warn indicators
  2. Runs the full analysis pipeline on one click
  3. Streams live log output to both the GUI and the console
  4. Opens the output folder when done

Usage:
    python app.py
"""

import os
import sys
import shutil
import threading
import traceback
import subprocess
import platform
from datetime import datetime

import customtkinter as ctk

import config as C

# ── Theme ──────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

VERSION = "1.0.0"
TITLE   = "Amodal Segmentation Analysis Tool"


# ================================================================
#  Hardware probing helpers
# ================================================================

def _gpu_info():
    """Return (name, vram_gb) or (None, 0) if no CUDA GPU."""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            return name, round(vram, 1)
    except Exception:
        pass
    return None, 0.0


def _disk_free_gb(path="."):
    total, used, free = shutil.disk_usage(os.path.abspath(path))
    return round(free / (1024 ** 3), 1)


def _ram_gb():
    try:
        import psutil
        return round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        # Fallback for Windows without psutil
        if sys.platform == "win32":
            try:
                r = subprocess.run(
                    ["wmic", "ComputerSystem", "get", "TotalPhysicalMemory"],
                    capture_output=True, text=True)
                for line in r.stdout.strip().splitlines():
                    line = line.strip()
                    if line.isdigit():
                        return round(int(line) / 1e9, 1)
            except Exception:
                pass
        return 0.0


def _python_info():
    return f"{platform.python_version()} ({platform.architecture()[0]})"


def _pytorch_info():
    try:
        import torch
        cuda = torch.version.cuda or "N/A"
        return f"{torch.__version__}  (CUDA {cuda})"
    except ImportError:
        return "NOT INSTALLED"


# ================================================================
#  Main Application Window
# ================================================================

class App(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(TITLE)
        self.geometry("980x720")
        self.minsize(860, 620)

        # ── State ──
        self._running = False

        # ── Build UI ──
        self._build_header()
        self._build_hardware_panel()
        self._build_log_panel()
        self._build_footer()

        # ── Initial hardware check ──
        self.after(200, self._check_hardware)

    # ────────────────────────────────────────────────────────────
    #  UI construction
    # ────────────────────────────────────────────────────────────

    def _build_header(self):
        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=16, pady=(12, 4))

        ctk.CTkLabel(hdr, text=TITLE,
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        ctk.CTkLabel(hdr, text=f"v{VERSION}",
                     font=ctk.CTkFont(size=12),
                     text_color="gray60").pack(side="left", padx=(10, 0))

    def _build_hardware_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(fill="x", padx=16, pady=(4, 4))

        ctk.CTkLabel(frame, text="System Check",
                     font=ctk.CTkFont(size=14, weight="bold")
                     ).grid(row=0, column=0, columnspan=4,
                            sticky="w", padx=12, pady=(8, 4))

        self._hw_labels = {}
        items = [
            ("gpu",     "GPU"),
            ("vram",    "VRAM"),
            ("disk",    "Disk Free"),
            ("ram",     "RAM"),
            ("python",  "Python"),
            ("pytorch", "PyTorch"),
        ]
        for i, (key, label) in enumerate(items):
            r, c = divmod(i, 3)
            r += 1  # row 0 is the title

            lbl = ctk.CTkLabel(frame, text=f"{label}:", anchor="e", width=80,
                               font=ctk.CTkFont(size=12))
            lbl.grid(row=r, column=c * 2, sticky="e", padx=(12, 4), pady=2)

            val = ctk.CTkLabel(frame, text="checking …", anchor="w",
                               font=ctk.CTkFont(size=12))
            val.grid(row=r, column=c * 2 + 1, sticky="w", padx=(0, 20), pady=2)
            self._hw_labels[key] = val

        # Status summary
        self._status = ctk.CTkLabel(frame, text="", anchor="w",
                                    font=ctk.CTkFont(size=13, weight="bold"))
        self._status.grid(row=3, column=0, columnspan=6,
                          sticky="w", padx=12, pady=(6, 10))

    def _build_log_panel(self):
        frame = ctk.CTkFrame(self)
        frame.pack(fill="both", expand=True, padx=16, pady=4)
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

        self._log_box = ctk.CTkTextbox(frame, wrap="word",
                                       font=ctk.CTkFont(family="Consolas",
                                                         size=12),
                                       state="disabled")
        self._log_box.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

    def _build_footer(self):
        foot = ctk.CTkFrame(self, fg_color="transparent")
        foot.pack(fill="x", padx=16, pady=(4, 12))

        self._progress = ctk.CTkProgressBar(foot, mode="indeterminate",
                                            height=6)
        self._progress.pack(fill="x", pady=(0, 8))
        self._progress.set(0)

        btn_frame = ctk.CTkFrame(foot, fg_color="transparent")
        btn_frame.pack(fill="x")

        self._btn_run = ctk.CTkButton(
            btn_frame, text="▶  Run Full Pipeline", width=220, height=40,
            font=ctk.CTkFont(size=14, weight="bold"),
            command=self._on_run)
        self._btn_run.pack(side="left")

        self._btn_open = ctk.CTkButton(
            btn_frame, text="📂  Open Output Folder", width=200, height=40,
            font=ctk.CTkFont(size=13),
            state="disabled",
            command=self._on_open_output)
        self._btn_open.pack(side="left", padx=(12, 0))

        self._btn_quit = ctk.CTkButton(
            btn_frame, text="Quit", width=80, height=40,
            fg_color="gray30", hover_color="gray40",
            command=self._on_quit)
        self._btn_quit.pack(side="right")

    # ────────────────────────────────────────────────────────────
    #  Hardware check
    # ────────────────────────────────────────────────────────────

    def _check_hardware(self):
        gpu_name, vram = _gpu_info()
        disk  = _disk_free_gb(C.ROOT)
        ram   = _ram_gb()
        pyver = _python_info()
        ptver = _pytorch_info()

        issues = []

        # GPU
        if gpu_name:
            self._hw_labels["gpu"].configure(text=gpu_name, text_color="green")
        else:
            self._hw_labels["gpu"].configure(text="None (CPU only)",
                                             text_color="orange")
            issues.append("No GPU — inference will be slow")

        # VRAM
        if vram >= C.RECOMMENDED_VRAM_GB:
            self._hw_labels["vram"].configure(text=f"{vram} GB", text_color="green")
        elif vram >= C.MIN_VRAM_GB:
            self._hw_labels["vram"].configure(text=f"{vram} GB (minimum met)",
                                              text_color="yellow")
        elif gpu_name:
            self._hw_labels["vram"].configure(text=f"{vram} GB (LOW)",
                                              text_color="red")
            issues.append(f"Low VRAM ({vram} GB, need {C.MIN_VRAM_GB}+)")
        else:
            self._hw_labels["vram"].configure(text="N/A", text_color="gray60")

        # Disk
        if disk >= C.MIN_DISK_GB:
            self._hw_labels["disk"].configure(text=f"{disk} GB", text_color="green")
        else:
            self._hw_labels["disk"].configure(text=f"{disk} GB (NEED {C.MIN_DISK_GB}+)",
                                              text_color="red")
            issues.append(f"Need {C.MIN_DISK_GB} GB disk, only {disk} available")

        # RAM
        if ram >= 8:
            self._hw_labels["ram"].configure(text=f"{ram} GB", text_color="green")
        else:
            self._hw_labels["ram"].configure(text=f"{ram} GB", text_color="orange")

        # Python / PyTorch
        self._hw_labels["python"].configure(text=pyver, text_color="white")
        if "NOT INSTALLED" in ptver:
            self._hw_labels["pytorch"].configure(text=ptver, text_color="red")
            issues.append("PyTorch not installed")
        else:
            self._hw_labels["pytorch"].configure(text=ptver, text_color="green")

        # Summary
        if not issues:
            self._status.configure(text="✅  All checks passed — ready to run.",
                                   text_color="green")
        else:
            self._status.configure(
                text="⚠️  Warnings: " + " | ".join(issues),
                text_color="orange")

    # ────────────────────────────────────────────────────────────
    #  Logging
    # ────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        """Thread-safe log to both the GUI textbox and stdout."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}]  {msg}\n"

        # Console
        sys.stdout.write(line)
        sys.stdout.flush()

        # GUI (must be called from main thread)
        self._log_box.configure(state="normal")
        self._log_box.insert("end", line)
        self._log_box.see("end")
        self._log_box.configure(state="disabled")

    def _log_safe(self, msg: str):
        """Log from a worker thread by scheduling on the main loop."""
        self.after(0, self._log, msg)

    # ────────────────────────────────────────────────────────────
    #  Pipeline execution
    # ────────────────────────────────────────────────────────────

    def _on_run(self):
        if self._running:
            return
        self._running = True
        self._btn_run.configure(state="disabled", text="⏳  Running …")
        self._btn_open.configure(state="disabled")
        self._progress.start()
        self._log_box.configure(state="normal")
        self._log_box.delete("1.0", "end")
        self._log_box.configure(state="disabled")
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        import pipeline as P

        steps = [
            ("Setup & Install",       P.step_setup),
            ("Download Datasets",     P.step_download),
            ("Ground-Truth Vis",      P.step_ground_truth),
            ("Baseline Inference",    P.step_baseline),
            ("Side-by-Side Compare",  P.step_comparisons),
            ("Quantitative Analysis", P.step_quantitative),
            ("Failure Deep Dive",     P.step_deep_dive),
            ("AISFormer Charts",      P.step_aisformer_charts),
            ("Package Results",       P.step_package),
        ]

        log = self._log_safe
        log(f"Pipeline started  ({len(steps)} steps)")
        log(f"Output folder: {C.OUTPUT_DIR}\n")

        t0 = __import__("time").time()

        for i, (name, fn) in enumerate(steps, 1):
            log(f"\n{'─' * 60}")
            log(f"  [{i}/{len(steps)}]  {name}")
            log(f"{'─' * 60}\n")
            try:
                fn(log)
            except Exception:
                log(f"\n❌  STEP FAILED: {name}")
                log(traceback.format_exc())
                self.after(0, self._pipeline_done, False)
                return

        elapsed = __import__("time").time() - t0
        m, s = divmod(int(elapsed), 60)
        log(f"\n{'=' * 60}")
        log(f"✅  Pipeline complete!  ({m}m {s}s)")
        log(f"    Output → {C.OUTPUT_DIR}")
        log(f"    ZIP    → {C.RESULTS_ZIP}")
        log(f"{'=' * 60}")

        self.after(0, self._pipeline_done, True)

    def _pipeline_done(self, success: bool):
        self._running = False
        self._progress.stop()
        self._progress.set(1 if success else 0)
        self._btn_run.configure(
            state="normal",
            text="▶  Run Full Pipeline" if success else "▶  Retry Pipeline")
        if success:
            self._btn_open.configure(state="normal")

    # ────────────────────────────────────────────────────────────
    #  Buttons
    # ────────────────────────────────────────────────────────────

    def _on_open_output(self):
        p = C.OUTPUT_DIR
        if not os.path.isdir(p):
            os.makedirs(p, exist_ok=True)
        if sys.platform == "win32":
            os.startfile(p)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", p])
        else:
            subprocess.Popen(["xdg-open", p])

    def _on_quit(self):
        self.destroy()


# ================================================================
#  Entry point
# ================================================================

if __name__ == "__main__":
    app = App()
    app.mainloop()
