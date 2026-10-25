#!/usr/bin/env python3
"""
Video Recorder & Editor — Screen recording + timeline editing app.
Records a screen region with system audio, then provides a visual timeline
editor for trimming, cutting, splitting, and exporting.

Requirements:
    pip install PySide6
    ffmpeg must be installed and available in PATH (or set FFMPEG_PATH env var)

Usage:
    python video_recorder_editor.py

Author: Built for ParisNeo (Saifeddine ALOUI)
"""

import sys
import os
import subprocess
import platform
import json
import tempfile
import shutil
import re
import uuid
from pathlib import Path
from datetime import datetime

from PySide6.QtCore import (
    Qt, QRect, QRectF, QPoint, QSize, QTimer, QThread, Signal, QObject,
    QUrl, QPropertyAnimation, QEasingCurve, QPointF
)
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QImage,
    QMouseEvent, QKeyEvent, QAction, QShortcut, QKeySequence,
    QIcon, QLinearGradient, QPainterPath, QPolygonF, QPalette,
    QDesktopServices
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QFileDialog, QComboBox, QSlider,
    QGroupBox, QLineEdit, QInputDialog, QCheckBox, QDialog,
    QFormLayout, QDialogButtonBox, QProgressBar, QScrollArea,
    QFrame, QSizePolicy, QToolButton, QMessageBox, QColorDialog,
    QGridLayout, QTextEdit, QSplitter, QStackedWidget, QListWidget,
    QListWidgetItem, QMenu, QTabWidget
)
from PySide6.QtCore import QMimeData
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput, QVideoSink
from PySide6.QtMultimediaWidgets import QVideoWidget

import ascii_colors as logging
logger = logging.getLogger("video_recorder")

class TrackPlayer(QObject):
    frame_ready = Signal(str)

    def __init__(self, track_name, parent=None):
        super().__init__(parent)
        self.track_name = track_name
        # CRITICAL: Do NOT pass parent to QMediaPlayer. 
        # Passing parent (QMainWindow) causes Qt Multimedia to render a phantom 
        # stretched hardware overlay directly onto the main window on Windows.
        self.player = QMediaPlayer() 
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.sink = QVideoSink()
        self.player.setVideoOutput(self.sink)
        self.sink.videoFrameChanged.connect(self._on_frame)
        self.current_image = None
        self.active_clip = None

    def _on_frame(self, frame):
        if frame.isValid():
            self.current_image = frame.toImage()
            self.frame_ready.emit(self.track_name)

_APP_DIR = Path(__file__).resolve().parent

def init_ffmpeg_path():
    """Locate or bind local portable FFmpeg binaries into PATH."""
    candidate_dirs = [
        _APP_DIR / "ffmpeg" / "bin",
        _APP_DIR / "ffmpeg",
    ]
    if (_APP_DIR / "ffmpeg").is_dir():
        for sub in (_APP_DIR / "ffmpeg").glob("**/bin"):
            if sub.is_dir() and sub not in candidate_dirs:
                candidate_dirs.append(sub)
        for sub in (_APP_DIR / "ffmpeg").glob("**"):
            if sub.is_dir() and sub not in candidate_dirs:
                candidate_dirs.append(sub)

    for candidate in candidate_dirs:
        ffmpeg_exe = candidate / ("ffmpeg.exe" if platform.system() == "Windows" else "ffmpeg")
        if ffmpeg_exe.is_file():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            os.environ["FFMPEG_BINARY"] = str(ffmpeg_exe)
            os.environ["FFMPEG_PATH"] = str(ffmpeg_exe)
            logger.info(f"Using local FFmpeg from: {candidate}")
            return str(candidate)
    return None

init_ffmpeg_path()

import ascii_colors as logging
logger = logging.getLogger("video_recorder")


class Command:
    """Base class for undoable operations."""
    def execute(self): pass
    def undo(self): pass
    def redo(self): pass


class AddTextCommand(Command):
    def __init__(self, editor, overlay):
        self.editor = editor
        self.overlay = overlay

    def execute(self):
        self.editor.text_overlays.append(self.overlay)
        self.editor.timeline.add_text_overlay(self.overlay)
        self.editor.timeline.update()

    def undo(self):
        if self.overlay in self.editor.text_overlays:
            self.editor.text_overlays.remove(self.overlay)
            if "text_overlays" in self.editor.timeline._tracks:
                track = self.editor.timeline._tracks["text_overlays"]
                track.clips = [c for c in track.clips if c.text_content != self.overlay.get("text")]
            self.editor.timeline.update()

    def redo(self):
        self.execute()


class SplitCommand(Command):
    def __init__(self, editor, track_name, pos):
        self.editor = editor
        self.primary_track = track_name
        self.pos = pos
        self.state_before = {}  # track_name -> list of Clips
        self.state_after = {}

    def execute(self):
        if not self.primary_track or self.primary_track not in self.editor.timeline._tracks:
            return

        # Determine target tracks (primary + linked track if present)
        target_tracks = [self.primary_track]
        linked = self.editor.timeline._tracks[self.primary_track].linked_to
        if linked and linked in self.editor.timeline._tracks:
            target_tracks.append(linked)

        if not self.state_before:
            for t_name in target_tracks:
                track = self.editor.timeline._tracks[t_name]
                self.state_before[t_name] = [c.clone() for c in track.clips]

                new_clips = []
                for clip in track.clips:
                    if clip.timeline_start < self.pos < clip.timeline_end:
                        offset = self.pos - clip.timeline_start
                        left = clip.clone()
                        left.duration = offset

                        right = clip.clone(new_start=self.pos)
                        right.duration = clip.duration - offset
                        right.source_start = clip.source_start + offset

                        new_clips.extend([left, right])
                    else:
                        new_clips.append(clip)
                self.state_after[t_name] = new_clips

        for t_name, clips in self.state_after.items():
            if t_name in self.editor.timeline._tracks:
                self.editor.timeline._tracks[t_name].clips = [c.clone() for c in clips]

        self.editor.timeline.segments_changed.emit()
        self.editor.timeline._update_layout()
        self.editor.timeline.update()

    def undo(self):
        for t_name, clips in self.state_before.items():
            if t_name in self.editor.timeline._tracks:
                self.editor.timeline._tracks[t_name].clips = [c.clone() for c in clips]
        self.editor.timeline.segments_changed.emit()
        self.editor.timeline._update_layout()
        self.editor.timeline.update()

    def redo(self):
        self.execute()


class DeleteSegmentCommand(Command):
    def __init__(self, editor, track_name, seg_idx, ripple=False):
        self.editor = editor
        self.primary_track = track_name
        self.seg_idx = seg_idx
        self.ripple = ripple
        self.state_before = {}
        self.state_after = {}

    def execute(self):
        if not self.primary_track or self.primary_track not in self.editor.timeline._tracks:
            return

        target_tracks = [self.primary_track]
        linked = self.editor.timeline._tracks[self.primary_track].linked_to
        if linked and linked in self.editor.timeline._tracks:
            target_tracks.append(linked)

        if not self.state_before:
            for t_name in target_tracks:
                track = self.editor.timeline._tracks[t_name]
                self.state_before[t_name] = [c.clone() for c in track.clips]

                if 0 <= self.seg_idx < len(track.clips):
                    del_clip = track.clips[self.seg_idx]
                    dur = del_clip.duration
                    del_start = del_clip.timeline_start

                    new_clips = [c.clone() for i, c in enumerate(track.clips) if i != self.seg_idx]

                    if self.ripple:
                        for c in new_clips:
                            if c.timeline_start >= del_start:
                                c.timeline_start = max(0.0, c.timeline_start - dur)

                    self.state_after[t_name] = new_clips

        for t_name, clips in self.state_after.items():
            if t_name in self.editor.timeline._tracks:
                self.editor.timeline._tracks[t_name].clips = [c.clone() for c in clips]

        self.editor.timeline._selected_segment = -1
        self.editor.timeline.segments_changed.emit()
        self.editor.timeline._update_layout()
        self.editor.timeline.update()

    def undo(self):
        for t_name, clips in self.state_before.items():
            if t_name in self.editor.timeline._tracks:
                self.editor.timeline._tracks[t_name].clips = [c.clone() for c in clips]
        self.editor.timeline.segments_changed.emit()
        self.editor.timeline._update_layout()
        self.editor.timeline.update()

    def redo(self):
        self.execute()


class NewProjectDialog(QDialog):
    """Dialog for creating a new project or recording."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(400)
        self.setStyleSheet(f"""
            QDialog {{ background: {CLR_PANEL}; color: {CLR_TEXT}; }}
            QLabel {{ color: {CLR_TEXT}; }}
            QLineEdit, QTextEdit {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; padding: 4px 8px; border-radius: 4px;
            }}
            QPushButton {{
                background: {CLR_ACCENT}; color: white; border: none;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {CLR_ACCENT_2}; }}
        """)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("My Awesome Video")
        form.addRow("Project Name:", self.name_edit)

        self.desc_edit = QTextEdit()
        self.desc_edit.setPlaceholderText("Describe your project...")
        self.desc_edit.setMaximumHeight(80)
        form.addRow("Description:", self.desc_edit)

        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["16:9", "9:16", "1:1", "4:3", "21:9"])
        self.ratio_combo.setCurrentIndex(0)
        form.addRow("Aspect Ratio:", self.ratio_combo)

        layout.addLayout(form)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_data(self):
        return {
            "name": self.name_edit.text().strip() or "Untitled Project",
            "description": self.desc_edit.toPlainText().strip(),
            "aspect_ratio": self.ratio_combo.currentText()
        }

# ═══════════════════════════════════════════════════════════════
#  PROJECT MANAGER (Persistence)
# ═══════════════════════════════════════════════════════════════

class ProjectManager:
    """Handles saving and loading project states."""

    PROJECTS_DIR = os.path.join(os.path.expanduser("~"), "Documents", "ParisNeo_VideoProjects")

    @staticmethod
    def get_projects():
        """Retrieve all saved projects."""
        os.makedirs(ProjectManager.PROJECTS_DIR, exist_ok=True)
        projects = []
        for f in os.listdir(ProjectManager.PROJECTS_DIR):
            if f.endswith(".json"):
                path = os.path.join(ProjectManager.PROJECTS_DIR, f)
                try:
                    with open(path, 'r') as file:
                        data = json.load(file)
                        data['file_path'] = path
                        projects.append(data)
                except Exception:
                    pass
        return sorted(projects, key=lambda x: x.get('last_modified', ''), reverse=True)

    @staticmethod
    def save_project(project_data):
        """Save or update a project."""
        os.makedirs(ProjectManager.PROJECTS_DIR, exist_ok=True)
        proj_id = project_data.get('id', f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}")
        project_data['id'] = proj_id
        project_data['last_modified'] = datetime.now().isoformat()

        path = os.path.join(ProjectManager.PROJECTS_DIR, f"{proj_id}.json")
        with open(path, 'w') as file:
            json.dump(project_data, file, indent=4)
        return path

    @staticmethod
    def delete_project(proj_id):
        path = os.path.join(ProjectManager.PROJECTS_DIR, f"{proj_id}.json")
        if os.path.exists(path):
            os.remove(path)

# ═══════════════════════════════════════════════════════════════
#  RECORDING SETTINGS MANAGER (Session Persistence)
# ═══════════════════════════════════════════════════════════════

class RecordingSettingsManager:
    """Persists recording configuration (region, devices, mode, fps) across sessions."""

    SETTINGS_DIR = os.path.join(os.path.expanduser("~"), "Documents", "ParisNeo_VideoProjects")
    SETTINGS_FILE = os.path.join(SETTINGS_DIR, "recording_settings.json")

    @staticmethod
    def load_settings():
        """Return persisted recording settings dict, or an empty dict if none/invalid."""
        try:
            if os.path.exists(RecordingSettingsManager.SETTINGS_FILE):
                with open(RecordingSettingsManager.SETTINGS_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        return data
        except Exception as e:
            logger.warning(f"Failed to load recording settings: {e}")
        return {}

    @staticmethod
    def save_settings(settings_dict):
        """Persist recording settings dict to disk."""
        try:
            os.makedirs(RecordingSettingsManager.SETTINGS_DIR, exist_ok=True)
            with open(RecordingSettingsManager.SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings_dict, f, indent=4)
        except Exception as e:
            logger.warning(f"Failed to save recording settings: {e}")


# ═══════════════════════════════════════════════════════════════
#  HOME DASHBOARD (Project Selection)
# ═══════════════════════════════════════════════════════════════

class ProjectCard(QFrame):
    """A visual card representing a saved project with direct recording and open actions."""
    open_requested = Signal(str)  # project_id
    record_requested = Signal(str) # project_id
    delete_requested = Signal(str)

    def __init__(self, project_data, parent=None):
        super().__init__(parent)
        self.project_data = project_data
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(f"""
            ProjectCard {{
                background: {CLR_PANEL_LIGHT};
                border: 1px solid {CLR_BORDER};
                border-radius: 12px;
                padding: 10px;
            }}
            ProjectCard:hover {{
                border: 2px solid {CLR_ACCENT};
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # Thumbnail
        self.thumb_label = QLabel()
        self.thumb_label.setFixedSize(240, 135)
        self.thumb_label.setStyleSheet(f"background: {CLR_BG}; border-radius: 6px;")
        self.thumb_label.setAlignment(Qt.AlignCenter)
        if project_data.get('thumbnail_path') and os.path.exists(project_data['thumbnail_path']):
            pixmap = QPixmap(project_data['thumbnail_path'])
            self.thumb_label.setPixmap(pixmap.scaled(240, 135, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation))
        else:
            self.thumb_label.setText("🎬")
            self.thumb_label.setStyleSheet(f"background: {CLR_BG}; border-radius: 6px; font-size: 40px; color: {CLR_TEXT_DIM};")
        layout.addWidget(self.thumb_label)

        # Title
        self.title_label = QLabel(project_data.get('name', 'Untitled Project'))
        self.title_label.setStyleSheet(f"color: {CLR_TEXT}; font-weight: bold; font-size: 14px;")
        layout.addWidget(self.title_label)

        # Date
        date_str = project_data.get('last_modified', '')[:10]
        self.date_label = QLabel(f"Modified: {date_str}")
        self.date_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        layout.addWidget(self.date_label)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_open = QPushButton("Open")
        btn_open.setStyleSheet(f"background: {CLR_ACCENT}; color: white; border-radius: 4px; padding: 6px; font-weight: bold;")
        btn_open.clicked.connect(lambda: self.open_requested.emit(project_data['id']))
        btn_layout.addWidget(btn_open)

        btn_rec = QPushButton("⏺ Record")
        btn_rec.setToolTip("Record directly into this project")
        btn_rec.setStyleSheet(f"background: {CLR_DANGER}; color: white; border-radius: 4px; padding: 6px; font-weight: bold;")
        btn_rec.clicked.connect(lambda: self.record_requested.emit(project_data['id']))
        btn_layout.addWidget(btn_rec)

        btn_del = QPushButton("🗑")
        btn_del.setStyleSheet(f"background: {CLR_PANEL}; color: {CLR_TEXT_DIM}; border: 1px solid {CLR_BORDER}; border-radius: 4px; padding: 6px;")
        btn_del.clicked.connect(lambda: self.delete_requested.emit(project_data['id']))
        btn_layout.addWidget(btn_del)

        layout.addLayout(btn_layout)

class ProjectDashboard(QWidget):
    """Home screen showing all projects and fast creation/recording options."""

    open_project = Signal(str)      # project_id
    record_in_project = Signal(str) # project_id
    new_project = Signal()
    record_new = Signal()
    fast_record = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background: {CLR_BG};")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(40, 40, 40, 40)

        # Header
        header = QHBoxLayout()
        title = QLabel("Video Studio")
        title.setStyleSheet(f"color: {CLR_TEXT}; font-size: 32px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        btn_fast_rec = QPushButton("⚡ Fast Record")
        btn_fast_rec.setToolTip("Start recording immediately with one click")
        btn_fast_rec.setStyleSheet(f"background: {CLR_DANGER}; color: white; padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: bold;")
        btn_fast_rec.clicked.connect(self.fast_record.emit)
        header.addWidget(btn_fast_rec)

        btn_new = QPushButton("✨ New Project")
        btn_new.setStyleSheet(f"background: {CLR_ACCENT}; color: white; padding: 10px 20px; border-radius: 6px; font-size: 14px; font-weight: bold;")
        btn_new.clicked.connect(self.new_project.emit)
        header.addWidget(btn_new)

        main_layout.addLayout(header)
        main_layout.addSpacing(20)

        # Scroll Area for Projects
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("border: none;")

        self.scroll_content = QWidget()
        self.grid_layout = QGridLayout(self.scroll_content)
        self.grid_layout.setSpacing(20)
        self.grid_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.scroll_area.setWidget(self.scroll_content)
        main_layout.addWidget(self.scroll_area)

        self.refresh_projects()

    def refresh_projects(self):
        # Clear existing
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        projects = ProjectManager.get_projects()
        if not projects:
            empty_label = QLabel("No projects yet. Create a new one to get started!")
            empty_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 16px;")
            empty_label.setAlignment(Qt.AlignCenter)
            self.grid_layout.addWidget(empty_label, 0, 0, 1, 5, Qt.AlignCenter)
            return

        for i, proj in enumerate(projects):
            row = i // 4
            col = i % 4
            card = ProjectCard(proj)
            card.open_requested.connect(self.open_project.emit)
            card.record_requested.connect(self.record_in_project.emit)
            card.delete_requested.connect(self._handle_delete)
            self.grid_layout.addWidget(card, row, col)

    def _handle_delete(self, proj_id):
        reply = QMessageBox.question(self, "Delete Project", "Are you sure you want to delete this project?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            ProjectManager.delete_project(proj_id)
            self.refresh_projects()
# ═══════════════════════════════════════════════════════════════
#  CONSTANTS & THEME
# ═══════════════════════════════════════════════════════════════

APP_NAME = "ParisNeo Video Recorder & Editor"
APP_VERSION = "1.0"

# Dark theme colors (Pro NLE Slate Theme)
CLR_BG          = "#0d0d12"
CLR_PANEL       = "#14141d"
CLR_PANEL_LIGHT = "#1e1e2d"
CLR_ACCENT      = "#8b5cf6"   # Purple
CLR_ACCENT_2    = "#06b6d4"   # Cyan
CLR_DANGER      = "#ef4444"   # Red
CLR_SUCCESS     = "#10b981"   # Green
CLR_WARN        = "#f59e0b"   # Amber/Gold
CLR_TEXT        = "#f3f4f6"
CLR_TEXT_DIM    = "#9ca3af"
CLR_BORDER      = "#27273a"
CLR_TIMELINE_BG = "#09090e"
CLR_SEGMENT     = "#3b206e"
CLR_SEGMENT_ALT = "#164e63"
CLR_PLAYHEAD    = "#ef4444"
CLR_THUMB_BG    = "#181824"

RECORD_FPS_DEFAULT = 30


def get_ffmpeg_path():
    """Find ffmpeg executable."""
    env_path = os.environ.get("FFMPEG_PATH")
    if env_path and os.path.isfile(env_path):
        return env_path
    return shutil.which("ffmpeg") or "ffmpeg"


def get_ffprobe_path():
    """Find ffprobe executable."""
    ffmpeg_path = get_ffmpeg_path()
    if ffmpeg_path:
        p = Path(ffmpeg_path)
        ffprobe_exe = p.parent / "ffprobe.exe"
        if ffprobe_exe.exists():
            return str(ffprobe_exe)
        ffprobe = p.parent / "ffprobe"
        if ffprobe.exists():
            return str(ffprobe)
    return shutil.which("ffprobe") or "ffprobe"


def get_platform_name():
    return platform.system()  # 'Windows', 'Linux', 'Darwin'


# ═══════════════════════════════════════════════════════════════
#  FFMPEG HELPER
# ═══════════════════════════════════════════════════════════════

class FFmpegHelper:
    """Utility class for ffmpeg operations."""

    @staticmethod
    def get_version():
        try:
            result = subprocess.run(
                [get_ffmpeg_path(), "-version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                return lines[0] if lines else "Unknown"
            return None
        except Exception:
            return None

    @staticmethod
    def list_audio_devices():
        """List available audio input devices for recording."""
        plat = get_platform_name()
        devices = []

        if plat == "Windows":
            try:
                proc = subprocess.Popen(
                    [get_ffmpeg_path(), "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                stdout, stderr = proc.communicate(timeout=10)
                output = stderr if stderr else stdout

                logger.info("--- Raw DirectShow Output ---")
                logger.info(output)
                logger.info("------------------------------")

                lines = output.split('\n')
                for line in lines:
                    if '" (audio)' in line and 'Alternative name' not in line:
                        matches = re.findall(r'"([^"]+)"', line)
                        if matches:
                            dev_name = matches[0]
                            logger.info(f"Found Audio Device: {dev_name}")
                            if dev_name not in devices:
                                devices.append(dev_name)

                logger.info(f"--- Parsed Audio Devices: {devices} ---")
            except Exception as e:
                logger.error(f"Error listing audio devices: {e}")
                import traceback
                logger.error(traceback.format_exc())

        elif plat == "Linux":
            try:
                result = subprocess.run(
                    ["pactl", "list", "short", "sources"],
                    capture_output=True, text=True, timeout=5
                )
                for line in result.stdout.split('\n'):
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        name = parts[1]
                        if '.monitor' in name or 'loopback' in name.lower():
                            devices.append(f"[Monitor] {name}")
                        devices.append(name)
            except Exception:
                pass
            if not devices:
                devices.append("default")

        elif plat == "Darwin":
            try:
                result = subprocess.run(
                    [get_ffmpeg_path(), "-list_devices", "true", "-f", "avfoundation", "-i", "dummy"],
                    capture_output=True, text=True, timeout=10
                )
                in_audio = False
                for line in result.stderr.split('\n'):
                    if 'AVFoundation audio devices' in line:
                        in_audio = True
                        continue
                    if 'AVFoundation video devices' in line:
                        in_audio = False
                        continue
                    if in_audio and '[' in line:
                        match = re.search(r'\[(\d+)\]\s+(.+)', line)
                        if match:
                            devices.append(f"[{match.group(1)}] {match.group(2)}")
            except Exception:
                pass

        if not devices:
            devices.append("Default Audio Input")

        return devices

    @staticmethod
    def is_nvenc_available():
        """Active runtime probe to verify if NVIDIA NVENC hardware encoding works on this GPU."""
        try:
            cmd = [
                get_ffmpeg_path(), "-y",
                "-f", "lavfi", "-i", "nullsrc=s=64x64:d=0.1",
                "-c:v", "h264_nvenc",
                "-f", "null", "-"
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=3)
            return res.returncode == 0
        except Exception:
            return False

    @staticmethod
    def list_video_devices():
        """List available video input devices (cameras) for recording."""
        plat = get_platform_name()
        devices = []

        if plat == "Windows":
            try:
                proc = subprocess.Popen(
                    [get_ffmpeg_path(), "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                    stderr=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    text=True,
                    encoding='utf-8',
                    errors='replace'
                )
                stdout, stderr = proc.communicate(timeout=10)
                output = stderr if stderr else stdout

                lines = output.split('\n')
                for line in lines:
                    if '" (video)' in line and 'Alternative name' not in line:
                        matches = re.findall(r'"([^"]+)"', line)
                        if matches:
                            dev_name = matches[0]
                            logger.info(f"Found Video Device: {dev_name}")
                            if dev_name not in devices:
                                devices.append(dev_name)
            except Exception as e:
                logger.error(f"Error listing video devices: {e}")
                import traceback
                logger.error(traceback.format_exc())

        elif plat == "Linux":
            try:
                result = subprocess.run(
                    ["v4l2-ctl", "--list-devices"],
                    capture_output=True, text=True, timeout=5
                )
                current_name = None
                for line in result.stdout.split('\n'):
                    if line and not line.startswith('\t') and not line.startswith(' '):
                        current_name = line.split('(')[0].strip()
                    elif line.strip().startswith('/dev/video'):
                        dev_path = line.strip()
                        label = f"{current_name} ({dev_path})" if current_name else dev_path
                        devices.append(label)
            except FileNotFoundError:
                logger.warning("v4l2-ctl not found; falling back to /dev/video* enumeration")
            except Exception as e:
                logger.error(f"Error listing video devices: {e}")

            if not devices:
                # Fallback: just enumerate /dev/video* nodes directly
                try:
                    for name in sorted(os.listdir("/dev")):
                        if name.startswith("video"):
                            devices.append(f"/dev/{name}")
                except Exception:
                    pass

        elif plat == "Darwin":
            try:
                result = subprocess.run(
                    [get_ffmpeg_path(), "-list_devices", "true", "-f", "avfoundation", "-i", "dummy"],
                    capture_output=True, text=True, timeout=10
                )
                in_video = False
                for line in result.stderr.split('\n'):
                    if 'AVFoundation video devices' in line:
                        in_video = True
                        continue
                    if 'AVFoundation audio devices' in line:
                        in_video = False
                        continue
                    if in_video and '[' in line:
                        match = re.search(r'\[(\d+)\]\s+(.+)', line)
                        if match:
                            devices.append(f"[{match.group(1)}] {match.group(2)}")
            except Exception:
                pass

        if not devices:
            devices.append("Default Video Input")

        return devices


    @staticmethod
    def get_hwaccel_args():
        """Returns platform hardware acceleration flags for FFmpeg."""
        plat = get_platform_name()
        if plat == "Windows":
            return ["-hwaccel", "auto", "-hwaccel_output_format", "yuv420p"]
        elif plat == "Linux":
            return ["-hwaccel", "auto"]
        elif plat == "Darwin":
            return ["-hwaccel", "videotoolbox"]
        return []

    @staticmethod
    def build_record_command(output_path, x, y, w, h, fps, audio_device=None, mic_device=None, video_device=None, pip=False, video_mode="screen_only", pip_position="bottom-right", force_gdi=False):
        """Build dual-audio capable hardware-accelerated screen recording command."""
        plat = get_platform_name()
        ff = get_ffmpeg_path()

        if video_mode != "camera_only":
            w = max(2, w - (w % 2))
            h = max(2, h - (h % 2))
            x = max(0, x)
            y = max(0, y)

        if plat == "Windows":
            cmd = [ff, "-y"]
            has_nvenc = FFmpegHelper.is_nvenc_available()

            # High Quality capture opts
            dshow_audio_opts = [
                "-thread_queue_size", "4096",
                "-rtbufsize", "500M",
                "-f", "dshow",
            ]

            input_idx = 0
            audio_inputs = []

            # 1. Primary Video Capture Stream
            if video_mode == "camera_only":
                cmd += [
                    "-thread_queue_size", "4096",
                    "-rtbufsize", "1000M",
                    "-f", "dshow",
                    "-framerate", str(fps),
                    "-video_size", f"{w}x{h}",
                    "-i", f"video={video_device}"
                ]
                video_input_idx = input_idx
                input_idx += 1

            elif video_mode == "screen_pip" and video_device and video_device != "None":
                if not force_gdi:
                    dda_filter = f"ddagrab=framerate={fps}:draw_mouse=1:offset_x={x}:offset_y={y}:video_size={w}x{h},hwdownload,format=bgra"
                    cmd += ["-f", "lavfi", "-thread_queue_size", "4096", "-i", dda_filter]
                else:
                    cmd += [
                        "-thread_queue_size", "4096", "-rtbufsize", "1500M",
                        "-f", "gdigrab", "-framerate", str(fps),
                        "-offset_x", str(x), "-offset_y", str(y),
                        "-video_size", f"{w}x{h}", "-draw_mouse", "1",
                        "-i", "desktop"
                    ]
                bg_idx = input_idx
                input_idx += 1

                cmd += [
                    "-thread_queue_size", "4096",
                    "-rtbufsize", "500M",
                    "-f", "dshow",
                    "-video_size", "640x480",
                    "-i", f"video={video_device}"
                ]
                cam_idx = input_idx
                input_idx += 1

            else:
                if not force_gdi:
                    dda_filter = f"ddagrab=framerate={fps}:draw_mouse=1:offset_x={x}:offset_y={y}:video_size={w}x{h},hwdownload,format=bgra"
                    cmd += ["-f", "lavfi", "-thread_queue_size", "4096", "-i", dda_filter]
                else:
                    cmd += [
                        "-thread_queue_size", "4096", "-rtbufsize", "1500M",
                        "-f", "gdigrab", "-framerate", str(fps),
                        "-offset_x", str(x), "-offset_y", str(y),
                        "-video_size", f"{w}x{h}", "-draw_mouse", "1",
                        "-i", "desktop"
                    ]
                video_input_idx = input_idx
                input_idx += 1

            # 2. Audio Capture Streams (PC Output + Microphone)
            if audio_device and audio_device != "No Audio":
                cmd += dshow_audio_opts + ["-i", f"audio={audio_device}"]
                audio_inputs.append(input_idx)
                input_idx += 1

            if mic_device and mic_device not in ("No Audio", "No Microphone", audio_device):
                cmd += dshow_audio_opts + ["-i", f"audio={mic_device}"]
                audio_inputs.append(input_idx)
                input_idx += 1

            # 3. Filter Complex (Video PIP + Audio Mixing)
            filter_parts = []
            final_v_map = f"{video_input_idx}:v" if video_mode != "screen_pip" else "[vout]"
            final_a_map = None

            if video_mode == "screen_pip" and video_device and video_device != "None":
                pip_w = max(150, int(w * 0.25))
                pip_h = max(90, int(h * 0.25))

                if pip_position == "top-left":
                    pip_x, pip_y = 20, 20
                elif pip_position == "top-right":
                    pip_x, pip_y = w - pip_w - 20, 20
                elif pip_position == "bottom-left":
                    pip_x, pip_y = 20, h - pip_h - 20
                else:
                    pip_x, pip_y = w - pip_w - 20, h - pip_h - 20

                filter_parts.append(
                    f"[{bg_idx}:v]scale={w}:{h}[bg];"
                    f"[{cam_idx}:v]scale={pip_w}:{pip_h}[cam];"
                    f"[bg][cam]overlay={pip_x}:{pip_y}[vout]"
                )

            if len(audio_inputs) == 1:
                # Single Audio Stream
                filter_parts.append(f"[{audio_inputs[0]}:a]aresample=async=1000:first_pts=0[aout]")
                final_a_map = "[aout]"
            elif len(audio_inputs) >= 2:
                # Dual Audio Mix (PC Audio + Mic Commentary)
                filter_parts.append(
                    f"[{audio_inputs[0]}:a]aresample=async=1000:first_pts=0[a1];"
                    f"[{audio_inputs[1]}:a]aresample=async=1000:first_pts=0,volume=1.2[a2];"
                    f"[a1][a2]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
                )
                final_a_map = "[aout]"

            if filter_parts:
                cmd += ["-filter_complex", ";".join(filter_parts)]

            cmd += ["-map", final_v_map]
            if final_a_map:
                cmd += ["-map", final_a_map]

            # Video Encoding Engine: Active Tested NVENC or Multi-Core x264
            if has_nvenc:
                cmd += [
                    "-c:v", "h264_nvenc",
                    "-preset", "p7",
                    "-rc:v", "vbr",
                    "-cq:v", "14",
                    "-b:v", "0",
                    "-profile:v", "high444p",
                    "-pix_fmt", "yuv444p",
                    "-r", str(fps),
                    "-fps_mode", "cfr",
                ]
            else:
                cmd += [
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "12",
                    "-tune", "animation",
                    "-profile:v", "high444",
                    "-pix_fmt", "yuv444p",
                    "-r", str(fps),
                    "-fps_mode", "cfr",
                    "-threads", "0",
                    "-x264-params", "aq-mode=2:no-fast-pskip=1",
                ]

            if final_a_map or audio_inputs:
                cmd += [
                    "-c:a", "aac",
                    "-b:a", "192k"
                ]

            cmd.append(output_path)

        elif plat == "Linux":
            cmd = [ff, "-y"]
            input_idx = 0
            audio_inputs = []

            if video_mode == "camera_only" and video_device:
                dev_path = video_device.split('(')[-1].rstrip(')') if '(' in video_device else video_device
                if not dev_path.startswith('/dev/'):
                    dev_path = "/dev/video0"
                cmd += [
                    "-f", "v4l2",
                    "-framerate", str(fps),
                    "-video_size", f"{w}x{h}",
                    "-thread_queue_size", "1024",
                    "-i", dev_path,
                ]
                video_input_idx = input_idx
                input_idx += 1

            elif video_mode == "screen_pip" and video_device and video_device != "None":
                cmd += [
                    "-f", "x11grab",
                    "-framerate", str(fps),
                    "-video_size", f"{w}x{h}",
                    "-i", f":0.0+{x},{y}",
                ]
                bg_idx = input_idx
                input_idx += 1

                dev_path = video_device.split('(')[-1].rstrip(')') if '(' in video_device else video_device
                if not dev_path.startswith('/dev/'):
                    dev_path = "/dev/video0"
                cmd += [
                    "-f", "v4l2",
                    "-video_size", "640x480",
                    "-thread_queue_size", "1024",
                    "-i", dev_path,
                ]
                cam_idx = input_idx
                input_idx += 1

            else:
                cmd += [
                    "-f", "x11grab",
                    "-framerate", str(fps),
                    "-video_size", f"{w}x{h}",
                    "-i", f":0.0+{x},{y}",
                ]
                video_input_idx = input_idx
                input_idx += 1

            if audio_device and audio_device != "No Audio":
                dev = audio_device.replace("[Monitor] ", "") if audio_device.startswith("[Monitor] ") else audio_device
                cmd += ["-f", "pulse", "-i", dev]
                audio_inputs.append(input_idx)
                input_idx += 1

            if mic_device and mic_device not in ("No Audio", "No Microphone", audio_device):
                dev_mic = mic_device.replace("[Monitor] ", "") if mic_device.startswith("[Monitor] ") else mic_device
                cmd += ["-f", "pulse", "-i", dev_mic]
                audio_inputs.append(input_idx)
                input_idx += 1

            filter_parts = []
            final_v_map = f"{video_input_idx}:v" if video_mode != "screen_pip" else "[vout]"
            final_a_map = None

            if video_mode == "screen_pip" and video_device and video_device != "None":
                pip_w = max(150, int(w * 0.25))
                pip_h = max(90, int(h * 0.25))
                pip_x = w - pip_w - 20 if pip_position in ("top-right", "bottom-right") else 20
                pip_y = h - pip_h - 20 if pip_position in ("bottom-left", "bottom-right") else 20
                filter_parts.append(
                    f"[{bg_idx}:v]scale={w}:{h}[bg];"
                    f"[{cam_idx}:v]scale={pip_w}:{pip_h}[cam];"
                    f"[bg][cam]overlay={pip_x}:{pip_y}[vout]"
                )

            if len(audio_inputs) == 1:
                filter_parts.append(f"[{audio_inputs[0]}:a]aresample=async=1000:first_pts=0[aout]")
                final_a_map = "[aout]"
            elif len(audio_inputs) >= 2:
                filter_parts.append(
                    f"[{audio_inputs[0]}:a]aresample=async=1000:first_pts=0[a1];"
                    f"[{audio_inputs[1]}:a]aresample=async=1000:first_pts=0,volume=1.2[a2];"
                    f"[a1][a2]amix=inputs=2:duration=longest:dropout_transition=0[aout]"
                )
                final_a_map = "[aout]"

            if filter_parts:
                cmd += ["-filter_complex", ";".join(filter_parts)]

            cmd += ["-map", final_v_map]
            if final_a_map:
                cmd += ["-map", final_a_map]

            cmd += [
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "12",
                "-tune", "animation",
                "-profile:v", "high444",
                "-pix_fmt", "yuv444p",
                "-fps_mode", "cfr",
                "-r", str(fps),
            ]
            if final_a_map or audio_inputs:
                cmd += ["-c:a", "aac", "-b:a", "192k"]

            cmd.append(output_path)

        elif plat == "Darwin":
            video_idx = "0"
            audio_idx = ""
            if audio_device:
                match = re.match(r'\[(\d+)\]', audio_device)
                if match:
                    audio_idx = match.group(1)
            cmd = [
                ff, "-y",
                "-f", "avfoundation",
                "-framerate", str(fps),
                "-capture_cursor", "1",
                "-i", f"{video_idx}:{audio_idx}" if audio_idx else f"{video_idx}:none",
                "-c:v", "libx264", "-preset", "veryfast", "-tune", "animation",
                "-crf", "12",
                "-profile:v", "high444",
                "-pix_fmt", "yuv444p",
                "-fps_mode", "cfr",
                "-r", str(fps),
                "-c:a", "aac", "-b:a", "192k",
                output_path
            ]
        else:
            raise RuntimeError(f"Unsupported platform: {plat}")

        return cmd


    @staticmethod
    def has_audio(video_path):
        """Check if a video file has an audio stream using ffprobe."""
        try:
            ffprobe = get_ffprobe_path()
            result = subprocess.run(
                [
                    ffprobe, "-v", "error",
                    "-show_entries", "stream=codec_type",
                    "-of", "json", video_path
                ],
                capture_output=True, text=True, timeout=5
            )
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    return True
            return False
        except Exception:
            return False

    @staticmethod
    def get_video_dimensions(video_path):
        """Get width and height of a video or image using ffprobe."""
        try:
            result = subprocess.run(
                [
                    get_ffprobe_path(), "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0", video_path
                ],
                capture_output=True, text=True, timeout=5
            )
            w, h = map(int, result.stdout.strip().split('x'))
            return w, h
        except Exception:
            return 1920, 1080

    @staticmethod
    def get_video_duration(video_path):
        """Get video duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    get_ffprobe_path(), "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "json", video_path
                ],
                capture_output=True, text=True, timeout=10
            )
            data = json.loads(result.stdout)
            return float(data["format"]["duration"])
        except Exception:
            return 0.0

    @staticmethod
    def generate_thumbnails(video_path, count=30, thumb_width=120, output_dir=None):
        """Generate thumbnail frames from the video."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="vidthumbs_")

        duration = FFmpegHelper.get_video_duration(video_path)
        if duration <= 0:
            return [], output_dir

        interval = max(duration / count, 0.1)
        thumb_paths = []

        for i in range(count):
            timestamp = min(i * interval, duration - 0.1)
            if timestamp < 0:
                break
            thumb_path = os.path.join(output_dir, f"thumb_{i:04d}.png")
            try:
                subprocess.run(
                    [
                        get_ffmpeg_path(), "-y",
                        "-ss", f"{timestamp:.3f}",
                        "-i", video_path,
                        "-frames:v", "1",
                        "-vf", f"scale={thumb_width}:-1",
                        thumb_path
                    ],
                    capture_output=True, timeout=10
                )
                if os.path.exists(thumb_path):
                    thumb_paths.append((timestamp, thumb_path))
            except Exception:
                pass

        return thumb_paths, output_dir

    @staticmethod
    def generate_waveform(audio_path, width=1000, height=60, output_dir=None):
        """Generate a waveform image for an audio file using FFmpeg showwavespic."""
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="vidwave_")

        wave_path = os.path.join(output_dir, f"wave_{Path(audio_path).stem}.png")

        try:
            cmd = [
                get_ffmpeg_path(), "-y",
                "-i", audio_path,
                "-filter_complex", f"showwavespic=s={width}x{height}:colors=#00d4ff|#7b2ff7",
                "-frames:v", "1",
                wave_path
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            if os.path.exists(wave_path):
                return wave_path
        except Exception:
            pass
        return None

    @staticmethod
    def build_export_command(segments, input_path, output_path, quality="medium",
                              re_encode=True, resolution=None, fps=None, audio_tracks=None):
        """
        Build ffmpeg commands for exporting trimmed/segmented video.
        Supports multi-track audio mixing via audio_tracks list.
        audio_tracks: list of dicts [{"path": str, "volume": float, "start_offset": float}]
        """
        ff = get_ffmpeg_path()
        commands = []
        temp_dir = tempfile.mkdtemp(prefix="videxport_")
        seg_files = []

        crf_map = {"high": "18", "medium": "23", "low": "28"}
        crf = crf_map.get(quality, "23")

        vf_filters = []
        if resolution:
            # ffmpeg's scale filter takes w:h, not w x h
            vf_filters.append(f"scale={resolution.replace('x', ':').replace('X', ':')}")
        if fps:
            vf_filters.append(f"fps={fps}")
        vf_str = ",".join(vf_filters) if vf_filters else None

        has_main_audio = FFmpegHelper.has_audio(input_path)

        valid_audio_tracks = []
        if audio_tracks:
            for track in audio_tracks:
                if FFmpegHelper.has_audio(track["path"]):
                    valid_audio_tracks.append(track)

        for i, (start, end) in enumerate(segments):
            seg_path = os.path.join(temp_dir, f"seg_{i:04d}.mp4")
            seg_files.append(seg_path)

            cmd = [ff, "-y", "-i", input_path]
            input_count = 1

            audio_filter_labels = []
            mix_labels = []

            # Only prepare the main audio for mixing if there are actually other tracks to mix it with
            if has_main_audio and valid_audio_tracks:
                audio_filter_labels.append("[0:a]anull[a0]")
                mix_labels.append("[a0]")

            for track in valid_audio_tracks:
                cmd += ["-i", track["path"]]
                vol = track.get("volume", 1.0)
                offset = track.get("start_offset", 0.0)
                delay_ms = int(offset * 1000)
                audio_filter_labels.append(
                    f"[{input_count}:a]volume={vol},adelay={delay_ms}|{delay_ms}[a{input_count}]"
                )
                mix_labels.append(f"[a{input_count}]")
                input_count += 1

            cmd += ["-ss", f"{start:.3f}", "-to", f"{end:.3f}"]

            # Video encoding parameters
            if re_encode:
                cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", crf,
                        "-pix_fmt", "yuv420p"]
                if vf_str:
                    cmd += ["-vf", vf_str]
            else:
                cmd += ["-c:v", "copy"]

            # Audio encoding and mix mapping
            if mix_labels:
                if len(mix_labels) == 1:
                    # Single audio stream, no need to amix
                    amix_str = ";".join(audio_filter_labels)
                    cmd += ["-filter_complex", amix_str, "-map", "0:v", "-map", mix_labels[0]]
                else:
                    amix_str = ";".join(audio_filter_labels)
                    amix_str += ";" + "".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=longest[aout]"
                    cmd += ["-filter_complex", amix_str, "-map", "0:v", "-map", "[aout]"]
                cmd += ["-c:a", "aac", "-b:a", "192k"]
            else:
                # Fallback: No tracks needed mixing (either main only or fully silent)
                if has_main_audio:
                    cmd += ["-map", "0:v", "-map", "0:a"]
                    if re_encode:
                        cmd += ["-c:a", "aac", "-b:a", "192k"]
                    else:
                        cmd += ["-c:a", "copy"]
                else:
                    cmd += ["-map", "0:v"]

            cmd.append(seg_path)
            commands.append(cmd)

        list_path = os.path.join(temp_dir, "concat_list.txt")
        with open(list_path, "w", encoding='utf-8') as f:
            for seg_path in seg_files:
                f.write(f"file '{seg_path.replace(chr(92), '/')}'\n")

        concat_cmd = [
            ff, "-y",
            "-f", "concat", "-safe", "0",
            "-i", list_path,
            "-c", "copy"
        ]
        concat_cmd.append(output_path)
        commands.append(concat_cmd)

        return commands, temp_dir

    _available_encoders = None

    @staticmethod
    def get_available_encoders():
        """Retrieve and cache list of available encoders in FFmpeg."""
        if FFmpegHelper._available_encoders is not None:
            return FFmpegHelper._available_encoders
        try:
            result = subprocess.run(
                [get_ffmpeg_path(), "-encoders"],
                capture_output=True, text=True, timeout=5
            )
            FFmpegHelper._available_encoders = result.stdout
            return result.stdout
        except Exception:
            return ""

    @staticmethod
    def get_export_encoder_args(quality="medium"):
        """Select fastest hardware encoder if available, else standard fallback."""
        encoders_str = FFmpegHelper.get_available_encoders()
        crf_map = {"high": "18", "medium": "23", "low": "28"}
        crf = crf_map.get(quality, "23")

        if "h264_nvenc" in encoders_str:
            cq_map = {"high": "19", "medium": "24", "low": "29"}
            cq = cq_map.get(quality, "24")
            logger.info("Using NVIDIA NVENC hardware acceleration for export")
            return ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", cq]
        elif "h264_videotoolbox" in encoders_str:
            q_map = {"high": "75", "medium": "55", "low": "35"}
            q = q_map.get(quality, "55")
            logger.info("Using Apple VideoToolbox hardware acceleration for export")
            return ["-c:v", "h264_videotoolbox", "-q:v", q]
        elif "h264_qsv" in encoders_str:
            gq_map = {"high": "20", "medium": "25", "low": "30"}
            gq = gq_map.get(quality, "25")
            logger.info("Using Intel QSV hardware acceleration for export")
            return ["-c:v", "h264_qsv", "-global_quality", gq]
        elif "h264_amf" in encoders_str:
            logger.info("Using AMD AMF hardware acceleration for export")
            return ["-c:v", "h264_amf"]
        else:
            logger.info("Using software libx264 for export")
            return ["-c:v", "libx264", "-preset", "medium", "-crf", crf]

    @staticmethod
    def build_timeline_export_command(tracks, total_duration, output_path, quality="medium", width=1920, height=1080, fps=30):
        """Compiles the entire multi-track timeline using high-fidelity filter graphs."""
        ff = get_ffmpeg_path()
        cmd = [ff, "-y"]

        # Background color source generator
        cmd += ["-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={total_duration}:r={fps}"]

        input_map = {}
        unique_inputs = []

        # Map unique video and audio source filepaths
        for name, track in tracks.items():
            for clip in track.clips:
                if clip.source_path:
                    path = os.path.normpath(clip.source_path)
                    if path not in input_map:
                        input_map[path] = len(input_map) + 1
                        unique_inputs.append(clip.source_path)

        for path in unique_inputs:
            cmd += ["-i", path]

        filter_parts = []

        # Prioritize tracks: standard track layers first, drawing overlays next, text overlays on top
        def get_track_priority(name):
            name_lower = name.lower()
            if "main_video" in name_lower or "video" in name_lower:
                return 1
            elif "drawing" in name_lower:
                return 2
            elif "text" in name_lower:
                return 3
            return 4

        sorted_track_names = sorted(tracks.keys(), key=get_track_priority)

        # Collect and stack video clips sequentially
        video_clips_to_overlay = []
        for t_name in sorted_track_names:
            track = tracks[t_name]
            if track.type == Track.VIDEO or t_name in ("text_overlays", "drawings"):
                sorted_clips = sorted(track.clips, key=lambda c: c.timeline_start)
                for clip in sorted_clips:
                    video_clips_to_overlay.append(clip)

        v_acc_idx = 0
        v_acc_name = "[0:v]"

        for k, clip in enumerate(video_clips_to_overlay):
            if not clip.source_path:
                continue
            path = os.path.normpath(clip.source_path)
            input_idx = input_map.get(path)
            if input_idx is None:
                continue

            v_out_name = f"[v_acc_{v_acc_idx + 1}]"
            is_overlay_png = clip.source_path.lower().endswith('.png')

            if is_overlay_png:
                # Transparent full-resolution drawing or text layer
                v_fade_label = f"[v_fade_png_{v_acc_idx}]"
                filters = ["format=rgba"]

                op = getattr(clip, 'opacity', 1.0)
                if op < 1.0:
                    filters.append(f"colorchannelmixer=aa={op:.2f}")

                fi = getattr(clip, 'fade_in', 0.0)
                fo = getattr(clip, 'fade_out', 0.0)
                if fi > 0:
                    filters.append(f"fade=t=in:st=0:d={fi:.2f}")
                if fo > 0:
                    filters.append(f"fade=t=out:st={clip.duration - fo:.2f}:d={fo:.2f}")

                filter_complex_chain = ",".join(filters)
                filter_parts.append(
                    f"[{input_idx}:v]{filter_complex_chain}{v_fade_label}"
                )
                filter_parts.append(
                    f"{v_acc_name}{v_fade_label}overlay=enable='between(t,{clip.timeline_start:.3f},{clip.timeline_end:.3f})'{v_out_name}"
                )
            else:
                # Actual media video track segment
                v_trim_label = f"[v_trim_{k}]"
                v_filter_label = f"[v_filter_{k}]"
                v_scaled_label = f"[v_scaled_{k}]"

                filter_parts.append(
                    f"[{input_idx}:v]trim=start={clip.source_start:.3f}:duration={clip.duration:.3f},setpts=PTS-STARTPTS{v_trim_label}"
                )

                # Composite properties (opacity, fade)
                filters = ["format=rgba"]
                op = getattr(clip, 'opacity', 1.0)
                if op < 1.0:
                    filters.append(f"colorchannelmixer=aa={op:.2f}")

                fi = getattr(clip, 'fade_in', 0.0)
                fo = getattr(clip, 'fade_out', 0.0)
                if fi > 0:
                    filters.append(f"fade=t=in:st=0:d={fi:.2f}")
                if fo > 0:
                    filters.append(f"fade=t=out:st={clip.duration - fo:.2f}:d={fo:.2f}")

                filter_complex_chain = ",".join(filters)
                filter_parts.append(
                    f"{v_trim_label}{filter_complex_chain}{v_filter_label}"
                )

                w = int(round(clip.video_scale_x * width)) // 2 * 2
                h = int(round(clip.video_scale_y * height)) // 2 * 2
                w = max(2, w)
                h = max(2, h)
                filter_parts.append(
                    f"{v_filter_label}scale={w}:{h}{v_scaled_label}"
                )

                x = int(round(clip.video_x * width))
                y = int(round(clip.video_y * height))
                filter_parts.append(
                    f"{v_acc_name}{v_scaled_label}overlay=x={x}:y={y}:enable='between(t,{clip.timeline_start:.3f},{clip.timeline_end:.3f})'{v_out_name}"
                )

            v_acc_name = v_out_name
            v_acc_idx += 1

        # Collect and stack audio tracks
        audio_clips_to_mix = []
        for t_name, track in tracks.items():
            if track.type == Track.AUDIO or t_name == "main_audio":
                for clip in track.clips:
                    if clip.source_path and FFmpegHelper.has_audio(clip.source_path):
                        audio_clips_to_mix.append(clip)

        a_delayed_labels = []
        for j, clip in enumerate(audio_clips_to_mix):
            path = os.path.normpath(clip.source_path)
            input_idx = input_map.get(path)
            if input_idx is None:
                continue

            a_trim_label = f"[a_trim_{j}]"
            a_filter_label = f"[a_filt_{j}]"
            a_vol_label = f"[a_vol_{j}]"
            a_delayed_label = f"[a_delay_{j}]"

            filter_parts.append(
                f"[{input_idx}:a]atrim=start={clip.source_start:.3f}:duration={clip.duration:.3f},asetpts=PTS-STARTPTS{a_trim_label}"
            )

            # Smooth transitions (afade)
            a_filters = []
            fi = getattr(clip, 'fade_in', 0.0)
            fo = getattr(clip, 'fade_out', 0.0)
            if fi > 0:
                a_filters.append(f"afade=t=in:st=0:d={fi:.2f}")
            if fo > 0:
                a_filters.append(f"afade=t=out:st={clip.duration - fo:.2f}:d={fo:.2f}")

            if a_filters:
                a_chain = ",".join(a_filters)
                filter_parts.append(f"{a_trim_label}{a_chain}{a_filter_label}")
                current_audio_source = a_filter_label
            else:
                current_audio_source = a_trim_label

            vol = getattr(clip, 'volume', 1.0)
            filter_parts.append(
                f"{current_audio_source}volume={vol:.2f}{a_vol_label}"
            )

            delay_ms = int(round(clip.timeline_start * 1000))
            if delay_ms > 0:
                filter_parts.append(
                    f"{a_vol_label}adelay={delay_ms}|{delay_ms}{a_delayed_label}"
                )
                a_delayed_labels.append(a_delayed_label)
            else:
                a_delayed_labels.append(a_vol_label)

        if a_delayed_labels:
            if len(a_delayed_labels) == 1:
                filter_parts.append(f"{a_delayed_labels[0]}anull[aout]")
            else:
                inputs_str = "".join(a_delayed_labels)
                filter_parts.append(f"{inputs_str}amix=inputs={len(a_delayed_labels)}:duration=longest[aout]")
        else:
            filter_parts.append("anullsrc=r=44100:cl=stereo:d=0.1,anull[aout]")

        filter_complex_str = ";".join(filter_parts)
        cmd += ["-filter_complex", filter_complex_str]
        cmd += ["-map", v_acc_name, "-map", "[aout]"]

        # Accelerate encoding
        cmd += FFmpegHelper.get_export_encoder_args(quality)
        cmd += ["-c:a", "aac", "-b:a", "192k"]
        cmd.append(output_path)

        return cmd

# ═══════════════════════════════════════════════════════════════
#  REGION SELECTOR OVERLAY
# ═══════════════════════════════════════════════════════════════

class RegionSelector(QWidget):
    """Fullscreen transparent overlay for selecting a screen region."""

    region_selected = Signal(int, int, int, int)  # x, y, w, h
    selection_cancelled = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)

        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)
        self.full_screen = screen

        self._origin = QPoint()
        self._current = QPoint()
        self._selecting = False

        # Instructions
        self._instruction_text = "Click and drag to select recording region. ESC to cancel."

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Dark overlay
        painter.fillRect(self.rect(), QColor(0, 0, 0, 120))

        if self._selecting:
            rect = QRect(self._origin, self._current).normalized()

            # Clear the selected region (transparent)
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            painter.fillRect(rect, Qt.transparent)
            painter.setCompositionMode(QPainter.CompositionMode_SourceOver)

            # Draw border
            pen = QPen(QColor(CLR_ACCENT), 2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(rect)

            # Corner handles
            handle_size = 8
            painter.setBrush(QColor(CLR_ACCENT))
            for corner in [
                rect.topLeft(), rect.topRight(),
                rect.bottomLeft(), rect.bottomRight()
            ]:
                painter.drawRect(QRect(
                    corner.x() - handle_size // 2,
                    corner.y() - handle_size // 2,
                    handle_size, handle_size
                ))

            # Size label
            size_text = f"{rect.width()} × {rect.height()}"
            painter.setPen(QColor(CLR_TEXT))
            font = QFont("Segoe UI", 10)
            font.setBold(True)
            painter.setFont(font)
            label_y = rect.bottom() + 20
            if label_y > self.height() - 30:
                label_y = rect.top() - 20
            painter.drawText(
                QRect(rect.x(), label_y, rect.width(), 20),
                Qt.AlignCenter, size_text
            )

            # Dim outside region more
            # Top
            painter.fillRect(QRect(0, 0, self.width(), rect.y()), QColor(0, 0, 0, 80))
            # Bottom
            painter.fillRect(QRect(0, rect.bottom() + 1, self.width(),
                          self.height() - rect.bottom() - 1), QColor(0, 0, 0, 80))
            # Left
            painter.fillRect(QRect(0, rect.y(), rect.x(), rect.height()), QColor(0, 0, 0, 80))
            # Right
            painter.fillRect(QRect(rect.right() + 1, rect.y(),
                          self.width() - rect.right() - 1, rect.height()), QColor(0, 0, 0, 80))
        else:
            # Instructions
            painter.setPen(QColor(CLR_TEXT))
            font = QFont("Segoe UI", 14)
            font.setBold(True)
            painter.setFont(font)
            painter.drawText(self.rect(), Qt.AlignCenter, self._instruction_text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._origin = event.position().toPoint()
            self._current = self._origin
            self._selecting = True
            self.update()

    def mouseMoveEvent(self, event):
        if self._selecting:
            self._current = event.position().toPoint()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._selecting:
            self._current = event.position().toPoint()
            rect = QRect(self._origin, self._current).normalized()

            if rect.width() > 20 and rect.height() > 20:
                # Multiply logical screen coordinates by Windows High-DPI ratio (e.g. 1.25, 1.5, 2.0)
                screen = QApplication.primaryScreen()
                dpr = screen.devicePixelRatio() if screen else 1.0
                screen_geo = screen.geometry() if screen else QRect(0, 0, 1920, 1080)
                max_w = int(screen_geo.width() * dpr)
                max_h = int(screen_geo.height() * dpr)

                phys_x = max(0, min(int(rect.x() * dpr), max_w - 20))
                phys_y = max(0, min(int(rect.y() * dpr), max_h - 20))
                phys_w = max(20, min(int(rect.width() * dpr), max_w - phys_x))
                phys_h = max(20, min(int(rect.height() * dpr), max_h - phys_y))

                # Ensure dimensions are strictly even numbers for video codecs
                phys_w = phys_w - (phys_w % 2)
                phys_h = phys_h - (phys_h % 2)

                self.region_selected.emit(
                    phys_x, phys_y, phys_w, phys_h
                )
            else:
                self.selection_cancelled.emit()

            self._selecting = False
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.selection_cancelled.emit()
            self.close()


# ═══════════════════════════════════════════════════════════════
#  RECORDING OVERLAY
# ═══════════════════════════════════════════════════════════════

def open_in_file_manager(file_path):
    """Open folder in OS file manager with the target file highlighted."""
    if not file_path or not os.path.exists(file_path):
        return
    norm_path = os.path.normpath(file_path)
    if platform.system() == "Windows":
        try:
            subprocess.Popen(f'explorer /select,"{norm_path}"')
        except Exception:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(norm_path)))
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", "-R", norm_path])
    else:
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(norm_path)))


def play_in_default_player(file_path):
    """Launch the video file in the user's default system media player."""
    if file_path and os.path.exists(file_path):
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.normpath(file_path)))


class MediaCompleteDialog(QDialog):
    """Modern completion modal offering Editor, Player, and Explorer actions."""
    def __init__(self, title, heading, file_path, on_open_editor=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.setStyleSheet(f"""
            QDialog {{ background: {CLR_PANEL}; color: {CLR_TEXT}; }}
            QLabel {{ color: {CLR_TEXT}; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        head_lbl = QLabel(f"🎉 {heading}")
        head_lbl.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {CLR_SUCCESS};")
        layout.addWidget(head_lbl)

        path_box = QLineEdit(file_path)
        path_box.setReadOnly(True)
        path_box.setStyleSheet(f"""
            QLineEdit {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT_DIM};
                border: 1px solid {CLR_BORDER}; padding: 6px 10px; border-radius: 4px;
                font-family: Consolas; font-size: 11px;
            }}
        """)
        layout.addWidget(path_box)

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        if on_open_editor:
            btn_editor = QPushButton("🎬 Open in Video Editor")
            btn_editor.setStyleSheet(f"background: {CLR_ACCENT}; color: white; padding: 9px 16px; border-radius: 5px; font-weight: bold;")
            def _open_edit():
                self.accept()
                on_open_editor(file_path)
            btn_editor.clicked.connect(_open_edit)
            btn_layout.addWidget(btn_editor)

        btn_play = QPushButton("▶️ Play in Player")
        btn_play.setStyleSheet(f"background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT}; border: 1px solid {CLR_BORDER}; padding: 9px 16px; border-radius: 5px; font-weight: bold;")
        btn_play.clicked.connect(lambda: play_in_default_player(file_path))
        btn_layout.addWidget(btn_play)

        btn_folder = QPushButton("📁 Open Output Folder")
        btn_folder.setStyleSheet(f"background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT}; border: 1px solid {CLR_BORDER}; padding: 9px 16px; border-radius: 5px; font-weight: bold;")
        btn_folder.clicked.connect(lambda: open_in_file_manager(file_path))
        btn_layout.addWidget(btn_folder)

        layout.addLayout(btn_layout)

        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(f"background: transparent; color: {CLR_TEXT_DIM}; border: none; padding: 6px;")
        btn_close.clicked.connect(self.reject)
        layout.addWidget(btn_close, alignment=Qt.AlignRight)


class RecordingOverlay(QWidget):
    """Small floating overlay showing recording status and timer."""

    stop_requested = Signal()

    def __init__(self, region_rect):
        super().__init__()
        self.region_rect = region_rect
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)

        # Position at top-right of screen
        screen = QApplication.primaryScreen().geometry()
        self.fixed_x = screen.width() - 220
        self.fixed_y = 20
        self.setGeometry(self.fixed_x, self.fixed_y, 200, 64)

        self._elapsed = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

        self._recording = True

    def start(self):
        self._timer.start(1000)
        self.show()

    def stop(self):
        self._recording = False
        self._timer.stop()

    def _tick(self):
        self._elapsed += 1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.setBrush(QColor(20, 20, 30, 240))
        painter.setPen(QPen(QColor(CLR_BORDER), 1))
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 10, 10)

        # Recording dot (pulsing)
        pulse = (self._elapsed % 2 == 0) if self._recording else False
        dot_color = QColor(CLR_DANGER) if pulse else QColor(120, 30, 30)
        painter.setBrush(dot_color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(QPoint(22, 32), 8, 8)

        # Timer text
        mins = self._elapsed // 60
        secs = self._elapsed % 60
        time_str = f"{mins:02d}:{secs:02d}"
        painter.setPen(QColor(CLR_TEXT))
        font = QFont("Consolas", 16)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRect(40, 18, 100, 28), Qt.AlignLeft | Qt.AlignVCenter, time_str)

        # "REC" label
        font2 = QFont("Segoe UI", 8)
        font2.setBold(True)
        painter.setFont(font2)
        painter.setPen(QColor(CLR_DANGER))
        painter.drawText(QRect(40, 40, 60, 16), Qt.AlignLeft, "RECORDING")

        # Stop button
        btn_rect = QRect(self.width() - 52, 16, 36, 36)
        painter.setBrush(QColor(CLR_DANGER))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(btn_rect, 6, 6)
        # Square stop icon
        painter.setBrush(QColor(255, 255, 255))
        square_size = 10
        cx, cy = btn_rect.center().x(), btn_rect.center().y()
        painter.drawRect(QRect(cx - square_size//2, cy - square_size//2, square_size, square_size))

        self._stop_btn_rect = btn_rect

    def mousePressEvent(self, event):
        if hasattr(self, '_stop_btn_rect') and self._stop_btn_rect.contains(event.position().toPoint()):
            self.stop_requested.emit()


# ═══════════════════════════════════════════════════════════════
#  TIMELINE WIDGET
# ═══════════════════════════════════════════════════════════════

class Clip:
    """Represents a discrete piece of media on a track with transforms, transitions, and link metadata."""
    def __init__(self, source_path, timeline_start, duration, source_start=0.0, clip_type="video", name=""):
        self.source_path = source_path
        self.timeline_start = timeline_start
        self.duration = duration
        self.source_start = source_start
        self.type = clip_type  # "video", "audio", "text", "drawing"
        self.name = name or os.path.basename(source_path)

        self.volume = 1.0
        self.video_scale_x = 1.0
        self.video_scale_y = 1.0
        self.video_x = 0.0  # Normalized X offset (0.0 to 1.0)
        self.video_y = 0.0  # Normalized Y offset (0.0 to 1.0)
        self.opacity = 1.0

        # Transitions
        self.fade_in = 0.0   # seconds
        self.fade_out = 0.0  # seconds
        self.transition_type = "Fade"  # "Fade", "Dip to Black", "Wipe Left", "Wipe Right"

        self.thumbnails = []
        self.waveform_path = None
        self.text_content = ""
        self.drawing_paths = []  # List of (QPainterPath, QColor, pen_size)

        self.linked_to = None  # Reference to linked Clip object or track name (if any)
        self.group_id = None

    @property
    def timeline_end(self):
        return self.timeline_start + self.duration

    def intersects(self, start, end):
        return self.timeline_start < end and self.timeline_end > start

    def clone(self, new_start=None):
        start = self.timeline_start if new_start is None else new_start
        c = Clip(self.source_path, start, self.duration, self.source_start, self.type, self.name)
        c.volume = self.volume
        c.video_scale_x = self.video_scale_x
        c.video_scale_y = self.video_scale_y
        c.video_x = self.video_x
        c.video_y = self.video_y
        c.opacity = self.opacity
        c.fade_in = self.fade_in
        c.fade_out = self.fade_out
        c.transition_type = self.transition_type
        c.text_content = self.text_content
        c.thumbnails = list(self.thumbnails)
        c.waveform_path = self.waveform_path
        c.group_id = self.group_id
        c.drawing_paths = [(QPainterPath(p), QColor(col), w) for p, col, w in self.drawing_paths]
        return c


class Track:
    """Represents a media track (Video or Audio)."""
    VIDEO = 0
    AUDIO = 1

    def __init__(self, track_type, name, path=None, duration=0.0):
        self.type = track_type
        self.name = name
        self.path = path
        self.duration = duration
        self.volume = 1.0
        self.video_scale_x = 1.0
        self.video_scale_y = 1.0
        self.height = 64
        self.waveform_path = None
        self.thumbnails = []
        self.clips = []
        self.linked_to = None  # Name of the linked track (e.g., main_audio links to main_video)

class TimelineWidget(QWidget):
    """
    Custom pro timeline widget showing:
    - Multiple tracks (Video + Audio + Text)
    - Integrated left track controls header
    - Thumbnail strip / Waveform rendering
    - Precision time ruler
    - Segments with trim handles and selection outlines
    - Playhead with timecode display
    - Wheel zoom & pan support
    """

    position_changed = Signal(float)
    segments_changed = Signal()
    media_dropped = Signal(str)
    track_selected = Signal(str)
    split_requested = Signal()
    delete_requested = Signal()
    reset_requested = Signal()
    zoom_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAcceptDrops(True)

        self._duration = 0.0
        self._tracks = {}  # name -> Track
        self._playhead = 0.0

        self._header_width = 120
        self._padding_right = 30
        self._ruler_height = 26
        self._track_height = 60
        self._track_gap = 6
        self._min_pixels_per_second = 1.0
        self._max_pixels_per_second = 200.0
        self._pixels_per_second = 15.0

        self._drag_mode = None
        self._drag_track = None
        self._drag_segment_idx = -1
        self._selected_track = None
        self._selected_segment = -1
        self._selected_group = None
        self._drag_orig_starts = {}
        self._drag_orig_durs = {}

    def dragEnterEvent(self, event):
        if event.mimeData().hasText() or event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        path = None
        if event.mimeData().hasText():
            path = event.mimeData().text()
        elif event.mimeData().hasUrls():
            path = event.mimeData().urls()[0].toLocalFile()

        if path and path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
            playhead_pos = self._playhead
            if "main_video" in self._tracks:
                track = self._tracks["main_video"]

                thumbs, _ = FFmpegHelper.generate_thumbnails(path, count=1, thumb_width=120)
                duration = FFmpegHelper.get_video_duration(path)

                clip = Clip(path, playhead_pos, duration, clip_type="video")
                if thumbs:
                    clip.thumbnails = thumbs
                track.clips.append(clip)

                self.segments_changed.emit()
                self._update_layout()
                self.update()
            else:
                self.media_dropped.emit(path)
            event.acceptProposedAction()

    def set_duration(self, duration):
        self._duration = duration
        if "main_video" not in self._tracks and duration > 0:
            self._tracks["main_video"] = Track(Track.VIDEO, "Video", duration=duration)
        self._update_layout()
        self.update()

    def add_track(self, track_type, name, path=None, duration=None):
        """Add a new empty track (video or audio) to the timeline."""
        if duration is None:
            duration = self._duration
        key = name
        suffix = 1
        while key in self._tracks:
            suffix += 1
            key = f"{name} {suffix}"
        track = Track(track_type, name, path=path, duration=duration or 0.0)
        self._tracks[key] = track
        self._update_layout()
        self.update()
        return key

    def add_audio_track(self, name, path, duration, waveform_path):
        track = Track(Track.AUDIO, name, path, duration)
        track.waveform_path = waveform_path
        self._tracks[name] = track
        if duration > self._duration:
            self.set_duration(duration)
        self.update()

    def add_text_overlay(self, overlay):
        if "text_overlays" not in self._tracks:
            track = Track(Track.VIDEO, "Text Overlays", duration=self._duration)
            self._tracks["text_overlays"] = track

        clip = Clip("", overlay["start"], overlay["end"] - overlay["start"], clip_type="text", name=overlay.get("text", "Text"))
        clip.text_content = overlay.get("text", "")
        self._tracks["text_overlays"].clips.append(clip)
        self.update()

    def set_thumbnails(self, thumbs):
        if "main_video" in self._tracks and self._tracks["main_video"].clips:
            self._tracks["main_video"].clips[0].thumbnails = thumbs
        self.update()

    def get_segments(self):
        if "main_video" in self._tracks:
            return [(c.timeline_start, c.timeline_end) for c in self._tracks["main_video"].clips]
        return []

    def set_segments(self, segments):
        if "main_video" in self._tracks:
            track = self._tracks["main_video"]
            track.clips.clear()
            for start, end in segments:
                track.clips.append(Clip(track.path, start, end - start, clip_type="video"))
            self.segments_changed.emit()
            self.update()

    def set_playhead(self, pos):
        self._playhead = max(0, min(pos, self._duration))
        self.update()

    def get_playhead(self):
        return self._playhead

    def split_at_playhead(self):
        pos = self._playhead
        if self._selected_track and self._selected_track in self._tracks:
            tracks_to_split = [self._selected_track]
            linked = self._tracks[self._selected_track].linked_to
            if linked and linked in self._tracks:
                tracks_to_split.append(linked)

            for t_name in tracks_to_split:
                track = self._tracks[t_name]
                new_clips = []
                for clip in track.clips:
                    if clip.timeline_start < pos < clip.timeline_end:
                        left = Clip(clip.source_path, clip.timeline_start, pos - clip.timeline_start, clip.source_start, clip.type, clip.name)
                        right = Clip(clip.source_path, pos, clip.timeline_end - pos, clip.source_start + (pos - clip.timeline_start), clip.type, clip.name)
                        new_clips.extend([left, right])
                    else:
                        new_clips.append(clip)
                track.clips = new_clips
            self.segments_changed.emit()
            self.update()

    def delete_selected_segment(self):
        if self._selected_track and self._selected_segment >= 0:
            tracks_to_delete = [self._selected_track]
            linked = self._tracks[self._selected_track].linked_to
            if linked and linked in self._tracks:
                tracks_to_delete.append(linked)

            for t_name in tracks_to_delete:
                track = self._tracks[t_name]
                if 0 <= self._selected_segment < len(track.clips):
                    del track.clips[self._selected_segment]
            self._selected_segment = -1
            self.segments_changed.emit()
            self.update()

    def reset_segments(self):
        if self._selected_track and self._selected_track in self._tracks:
            track = self._tracks[self._selected_track]
            track.clips = [Clip(track.path, 0.0, track.duration, clip_type="video" if track.type == Track.VIDEO else "audio")]
            self._selected_segment = -1
            self.segments_changed.emit()
            self.update()

    def set_zoom(self, pps):
        self._pixels_per_second = max(self._min_pixels_per_second, min(pps, self._max_pixels_per_second))
        self._update_layout()
        self.zoom_changed.emit(self._pixels_per_second)
        self.update()

    def zoom_in(self):
        self.set_zoom(self._pixels_per_second * 1.25)

    def zoom_out(self):
        self.set_zoom(self._pixels_per_second / 1.25)

    def zoom_fit(self):
        parent_w = self.parentWidget().width() if self.parentWidget() else 800
        available_w = max(100, parent_w - self._header_width - self._padding_right)
        if self._duration > 0:
            self.set_zoom(available_w / self._duration)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier:
            angle = event.angleDelta().y()
            if angle > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def _update_layout(self):
        parent_w = self.parentWidget().width() if self.parentWidget() else 800
        content_w = int(self._header_width + self._duration * self._pixels_per_second + self._padding_right)
        total_w = max(parent_w, content_w)
        num_tracks = max(1, len(self._tracks))
        total_h = self._ruler_height + (self._track_height + self._track_gap) * num_tracks + 20
        self.setMinimumWidth(total_w)
        self.setMinimumHeight(total_h)

    def resizeEvent(self, event):
        self._update_layout()
        super().resizeEvent(event)

    def _time_to_x(self, t):
        return self._header_width + int(t * self._pixels_per_second)

    def _x_to_time(self, x):
        return max(0.0, min((x - self._header_width) / self._pixels_per_second, self._duration))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(CLR_TIMELINE_BG))

        if self._duration <= 0:
            painter.setPen(QColor(CLR_TEXT_DIM))
            font_msg = QFont("Segoe UI", 12)
            painter.setFont(font_msg)
            painter.drawText(self.rect(), Qt.AlignCenter, "📹 Import or record media to begin editing")
            return

        # Header Track Labels Bar (Top Left Corner)
        header_corner = QRect(0, 0, self._header_width, self._ruler_height)
        painter.fillRect(header_corner, QColor(CLR_PANEL))
        painter.setPen(QPen(QColor(CLR_BORDER), 1))
        painter.drawLine(self._header_width, 0, self._header_width, self.height())

        # Time Ruler
        ruler_rect = QRect(self._header_width, 0, self.width() - self._header_width, self._ruler_height)
        painter.fillRect(ruler_rect, QColor(CLR_PANEL_LIGHT))
        painter.setPen(QPen(QColor(CLR_BORDER), 1))
        painter.drawLine(0, self._ruler_height - 1, self.width(), self._ruler_height - 1)

        interval = self._choose_ruler_interval()
        font_small = QFont("Consolas", 8)
        painter.setFont(font_small)

        t = 0.0
        while t <= self._duration + interval:
            x = self._time_to_x(t)
            if self._header_width <= x <= self.width():
                painter.setPen(QPen(QColor(CLR_TEXT_DIM), 1))
                painter.drawLine(x, self._ruler_height - 8, x, self._ruler_height)
                mins = int(t) // 60
                secs = int(t) % 60
                label = f"{mins:02d}:{secs:02d}"
                painter.drawText(QRect(x - 20, 2, 40, 14), Qt.AlignCenter, label)

                # Minor tick marks
                sub_step = interval / 5.0
                for s in range(1, 5):
                    sub_x = self._time_to_x(t + s * sub_step)
                    if self._header_width <= sub_x <= self.width():
                        painter.drawLine(sub_x, self._ruler_height - 4, sub_x, self._ruler_height)
            t += interval

        # Tracks & Clips
        track_y = self._ruler_height + 4
        for track_name, track in self._tracks.items():
            # Track Left Header Block
            hdr_rect = QRect(0, track_y, self._header_width, self._track_height)
            is_selected = (track_name == self._selected_track)
            hdr_bg = QColor(CLR_PANEL_LIGHT).lighter(120) if is_selected else QColor(CLR_PANEL)
            painter.fillRect(hdr_rect, hdr_bg)
            painter.setPen(QPen(QColor(CLR_BORDER), 1))
            painter.drawRect(hdr_rect)

            # Header Track Text & Icon
            painter.setPen(QColor(CLR_TEXT))
            font_hdr = QFont("Segoe UI", 9)
            font_hdr.setBold(is_selected)
            painter.setFont(font_hdr)

            icon = "🎬" if track.type == Track.VIDEO else ("🎵" if track.type == Track.AUDIO else "✏️")
            painter.drawText(hdr_rect.adjusted(8, 6, -4, -20), Qt.AlignLeft | Qt.AlignTop, f"{icon} {track.name}")

            font_sub = QFont("Segoe UI", 7)
            painter.setFont(font_sub)
            painter.setPen(QColor(CLR_TEXT_DIM))
            type_str = "Video Track" if track.type == Track.VIDEO else ("Audio Track" if track.type == Track.AUDIO else "Overlay")
            painter.drawText(hdr_rect.adjusted(8, 26, -4, -4), Qt.AlignLeft | Qt.AlignTop, type_str)

            # Track Timeline Content Area
            track_content_rect = QRect(self._header_width, track_y, self.width() - self._header_width, self._track_height)
            painter.fillRect(track_content_rect, QColor(CLR_TIMELINE_BG))
            painter.setPen(QPen(QColor(CLR_BORDER), 1))
            painter.drawLine(self._header_width, track_y + self._track_height, self.width(), track_y + self._track_height)

            # Clips inside track
            for i, clip in enumerate(track.clips):
                x1 = self._time_to_x(clip.timeline_start)
                x2 = self._time_to_x(clip.timeline_end)
                clip_w = max(6, x2 - x1)
                clip_rect = QRectF(x1, track_y + 3, clip_w, self._track_height - 6)

                # Clip Colors
                if clip.type == "text":
                    base_col = QColor("#d97706")  # Amber
                    border_col = QColor("#f59e0b")
                elif clip.type == "drawing":
                    base_col = QColor("#db2777")  # Pink/Magenta
                    border_col = QColor("#f43f5e")
                elif track.type == Track.VIDEO:
                    base_col = QColor("#4c1d95")  # Dark Violet
                    border_col = QColor("#8b5cf6")
                else:
                    base_col = QColor("#0891b2")  # Dark Cyan
                    border_col = QColor("#06b6d4")

                clip_is_selected = (
                    (hasattr(clip, 'group_id') and clip.group_id == self._selected_group and self._selected_group) or 
                    (track_name == self._selected_track and i == self._selected_segment)
                )

                # Draw Clip Body with Rounded Rect
                path = QPainterPath()
                path.addRoundedRect(clip_rect, 5, 5)

                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(base_col))
                painter.drawPath(path)

                # Clip Content (Thumbnails / Waveforms)
                painter.save()
                painter.setClipPath(path)

                if clip.type == "video" and clip.thumbnails:
                    for timestamp, thumb_path in clip.thumbnails:
                        thumb_x = x1 + (timestamp - clip.timeline_start) * self._pixels_per_second
                        if thumb_x > self.width() or thumb_x < x1 - 120:
                            continue
                        if os.path.exists(thumb_path):
                            pixmap = QPixmap(thumb_path)
                            if not pixmap.isNull():
                                painter.drawPixmap(int(thumb_x), int(track_y + 16), pixmap.scaledToHeight(int(self._track_height - 20), Qt.SmoothTransformation))

                elif clip.type == "audio" and clip.waveform_path:
                    if os.path.exists(clip.waveform_path):
                        wave_pixmap = QPixmap(clip.waveform_path)
                        if not wave_pixmap.isNull():
                            painter.drawPixmap(clip_rect.toRect(), wave_pixmap.scaled(
                                int(clip_rect.width()), int(clip_rect.height()), Qt.IgnoreAspectRatio, Qt.SmoothTransformation
                            ))

                painter.restore()

                # Clip Title Header Bar
                title_bar = QRectF(clip_rect.x(), clip_rect.y(), clip_rect.width(), 16)
                painter.fillRect(title_bar, QColor(0, 0, 0, 90))
                painter.setPen(QColor(CLR_TEXT))
                font_clip = QFont("Segoe UI", 8)
                font_clip.setBold(True)
                painter.setFont(font_clip)
                painter.drawText(title_bar.adjusted(6, 0, -4, 0), Qt.AlignLeft | Qt.AlignVCenter, clip.name or "Clip")

                # Render Transition Wedges (Fade In / Fade Out Visuals)
                fade_in = getattr(clip, 'fade_in', 0.0)
                fade_out = getattr(clip, 'fade_out', 0.0)
                if fade_in > 0:
                    fi_w = min(clip_w * 0.45, fade_in * self._pixels_per_second)
                    fi_poly = QPolygonF([
                        QPointF(x1, track_y + 3),
                        QPointF(x1 + fi_w, track_y + 3),
                        QPointF(x1, track_y + self._track_height - 3)
                    ])
                    painter.setBrush(QColor(255, 255, 255, 75))
                    painter.setPen(Qt.NoPen)
                    painter.drawPolygon(fi_poly)

                if fade_out > 0:
                    fo_w = min(clip_w * 0.45, fade_out * self._pixels_per_second)
                    fo_poly = QPolygonF([
                        QPointF(x2, track_y + 3),
                        QPointF(x2 - fo_w, track_y + 3),
                        QPointF(x2, track_y + self._track_height - 3)
                    ])
                    painter.setBrush(QColor(255, 255, 255, 75))
                    painter.setPen(Qt.NoPen)
                    painter.drawPolygon(fo_poly)

                # Selection Highlight Border
                if clip_is_selected:
                    painter.setPen(QPen(QColor("#00d4ff"), 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path)
                else:
                    painter.setPen(QPen(border_col, 1))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawPath(path)

                # Trim handles
                handle_w = 5
                hl = QRectF(x1, track_y + 3, handle_w, self._track_height - 6)
                hr = QRectF(x2 - handle_w, track_y + 3, handle_w, self._track_height - 6)
                painter.setBrush(QColor("#f59e0b"))
                painter.setPen(Qt.NoPen)
                painter.drawRoundedRect(hl, 2, 2)
                painter.drawRoundedRect(hr, 2, 2)

            track_y += self._track_height + self._track_gap

        # Playhead
        px = self._time_to_x(self._playhead)
        if px >= self._header_width:
            painter.setPen(QPen(QColor(CLR_PLAYHEAD), 2))
            painter.drawLine(px, 0, px, self.height())

            # Head Cap
            tri = QPolygonF([QPointF(px - 7, 0), QPointF(px + 7, 0), QPointF(px + 7, 10), QPointF(px, 17), QPointF(px - 7, 10)])
            painter.setBrush(QColor(CLR_PLAYHEAD))
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.drawPolygon(tri)

            label = self._format_time(self._playhead)
            painter.setPen(QColor(255, 255, 255))
            font_ph = QFont("Consolas", 8)
            font_ph.setBold(True)
            painter.setFont(font_ph)
            painter.drawText(QRect(px + 9, 2, 65, 14), Qt.AlignLeft, label)

    def _choose_ruler_interval(self):
        pps = self._pixels_per_second
        if pps > 100: return 0.2
        elif pps > 50: return 0.5
        elif pps > 25: return 1.0
        elif pps > 10: return 2.0
        elif pps > 5: return 5.0
        elif pps > 2: return 10.0
        else: return 30.0

    @staticmethod
    def _format_time(seconds):
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        millis = int((seconds % 1) * 10)
        return f"{mins:02d}:{secs:02d}.{millis}"

    def _track_at_y(self, y):
        """Find track key corresponding to vertical Y pixel position."""
        track_y = self._ruler_height + 4
        for track_name, track in self._tracks.items():
            if track_y <= y <= track_y + self._track_height:
                return track_name
            track_y += self._track_height + self._track_gap
        return None

    def _hit_test(self, pos):
        x, y = pos.x(), pos.y()
        px = self._time_to_x(self._playhead)
        if abs(x - px) < 8 and y <= self._ruler_height + 5:
            return ('playhead', None, -1)

        track_y = self._ruler_height + 4
        for track_name, track in self._tracks.items():
            if track_y <= y <= track_y + self._track_height:
                if x < self._header_width:
                    return ('track_header', track_name, -1)

                handle_w = 8
                for i, clip in enumerate(track.clips):
                    x1 = self._time_to_x(clip.timeline_start)
                    x2 = self._time_to_x(clip.timeline_end)
                    if abs(x - x1) < handle_w: return ('trim_left', track_name, i)
                    if abs(x - x2) < handle_w: return ('trim_right', track_name, i)
                    if x1 < x < x2: return ('segment', track_name, i)
                return ('track', track_name, -1)
            track_y += self._track_height + self._track_gap

        if y < self._ruler_height:
            t = self._x_to_time(x)
            if 0 <= t <= self._duration: return ('seek', None, -1)
        return (None, None, -1)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            hit_type, track_name, seg_idx = self._hit_test(event.position().toPoint())

            if track_name is not None:
                self._selected_track = track_name
                self.track_selected.emit(track_name)
            self._selected_segment = seg_idx

            if hit_type in ('segment', 'trim_left', 'trim_right'):
                clip = self._tracks[track_name].clips[seg_idx]
                self._selected_group = getattr(clip, 'group_id', None)
            else:
                self._selected_group = None

            self._drag_orig_starts.clear()
            self._drag_orig_durs.clear()
            if self._selected_group:
                for t_name, t_obj in self._tracks.items():
                    for i, c_obj in enumerate(t_obj.clips):
                        if getattr(c_obj, 'group_id', None) == self._selected_group:
                            self._drag_orig_starts[(t_name, i)] = c_obj.timeline_start
                            self._drag_orig_durs[(t_name, i)] = c_obj.duration
            else:
                if track_name and 0 <= seg_idx < len(self._tracks[track_name].clips):
                    c_obj = self._tracks[track_name].clips[seg_idx]
                    self._drag_orig_starts[(track_name, seg_idx)] = c_obj.timeline_start
                    self._drag_orig_durs[(track_name, seg_idx)] = c_obj.duration

            if hit_type == 'playhead': self._drag_mode = 'playhead'
            elif hit_type == 'trim_left': self._drag_mode = 'trim_left'
            elif hit_type == 'trim_right': self._drag_mode = 'trim_right'
            elif hit_type == 'segment':
                self._drag_mode = 'segment'
                self._drag_start_x = event.position().toPoint().x()
                self._drag_orig_start = self._tracks[track_name].clips[seg_idx].timeline_start
            elif hit_type == 'seek':
                t = self._x_to_time(event.position().toPoint().x())
                self._playhead = t
                self.position_changed.emit(t)
                self._drag_mode = 'seek'

            self._drag_track = track_name
            self._drag_segment_idx = seg_idx
            self.update()
        elif event.button() == Qt.RightButton:
            hit_type, track_name, seg_idx = self._hit_test(event.position().toPoint())
            if track_name is not None:
                self._selected_track = track_name
                self.track_selected.emit(track_name)
            self._selected_segment = seg_idx

            if hit_type in ('segment', 'trim_left', 'trim_right'):
                clip = self._tracks[track_name].clips[seg_idx]
                self._selected_group = getattr(clip, 'group_id', None)
            else:
                self._selected_group = None

            self._show_context_menu(event.position().toPoint())

    def mouseMoveEvent(self, event):
        pos = event.position().toPoint()
        if self._drag_mode in ('playhead', 'seek'):
            t = self._x_to_time(pos.x())
            self._playhead = t
            self.position_changed.emit(t)
            self.update()
        elif self._drag_mode in ('trim_left', 'trim_right') and self._drag_track:
            track = self._tracks[self._drag_track]
            if 0 <= self._drag_segment_idx < len(track.clips):
                t = self._x_to_time(pos.x())
                clip = track.clips[self._drag_segment_idx]

                if self._drag_mode == 'trim_left':
                    min_t = track.clips[self._drag_segment_idx-1].timeline_end if self._drag_segment_idx > 0 else 0
                    max_t = clip.timeline_end - 0.1
                    t = max(min_t, min(t, max_t))
                    delta = t - clip.timeline_start

                    for (t_name, i), orig_st in self._drag_orig_starts.items():
                        if t_name in self._tracks and i < len(self._tracks[t_name].clips):
                            c_item = self._tracks[t_name].clips[i]
                            c_item.timeline_start = orig_st + delta
                            c_item.duration = self._drag_orig_durs[(t_name, i)] - delta
                            c_item.source_start += delta
                else:
                    min_t = clip.timeline_start + 0.1
                    max_t = track.clips[self._drag_segment_idx+1].timeline_start if self._drag_segment_idx < len(track.clips)-1 else self._duration
                    t = max(min_t, min(t, max_t))
                    delta_dur = t - (clip.timeline_start + clip.duration)

                    for (t_name, i), orig_dur in self._drag_orig_durs.items():
                        if t_name in self._tracks and i < len(self._tracks[t_name].clips):
                            self._tracks[t_name].clips[i].duration = orig_dur + delta_dur

                self.segments_changed.emit()
                self.update()

        elif self._drag_mode == 'segment' and self._drag_track:
            primary_track = self._tracks[self._drag_track]
            if 0 <= self._drag_segment_idx < len(primary_track.clips):
                delta_t = (pos.x() - self._drag_start_x) / self._pixels_per_second
                clip = primary_track.clips[self._drag_segment_idx]

                raw_new_start = max(0.0, self._drag_orig_start + delta_t)
                new_start = raw_new_start

                # Magnetic Snapping
                snap_threshold_sec = 8.0 / max(1.0, self._pixels_per_second)
                snap_targets = [self._playhead, 0.0]
                for t_k, t_v in self._tracks.items():
                    for c_idx, c_obj in enumerate(t_v.clips):
                        if getattr(c_obj, 'group_id', None) == self._selected_group and self._selected_group:
                            continue
                        snap_targets.extend([c_obj.timeline_start, c_obj.timeline_end])

                candidate_end = new_start + clip.duration
                for target in snap_targets:
                    if abs(new_start - target) <= snap_threshold_sec:
                        new_start = target
                        break
                    elif abs(candidate_end - target) <= snap_threshold_sec:
                        new_start = max(0.0, target - clip.duration)
                        break

                new_start = max(0.0, new_start)
                shift_delta = new_start - self._drag_orig_start

                for (t_name, i), orig_st in self._drag_orig_starts.items():
                    if t_name in self._tracks and i < len(self._tracks[t_name].clips):
                        self._tracks[t_name].clips[i].timeline_start = max(0.0, orig_st + shift_delta)

                # Vertical cross-track hover tracking
                hover_track = self._track_at_y(pos.y())
                if hover_track and hover_track in self._tracks:
                    if self._tracks[hover_track].type == primary_track.type:
                        self._drag_dest_track = hover_track
                    else:
                        self._drag_dest_track = None
                else:
                    self._drag_dest_track = None

                self.segments_changed.emit()
                self.update()
        else:
            hit_type, _, _ = self._hit_test(pos)
            if hit_type in ('playhead', 'trim_left', 'trim_right'): self.setCursor(Qt.SizeHorCursor)
            elif hit_type == 'segment': self.setCursor(Qt.SizeAllCursor)
            else: self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        if self._drag_mode == 'segment' and getattr(self, '_drag_dest_track', None) and self._drag_dest_track != self._drag_track:
            src_t = self._drag_track
            dst_t = self._drag_dest_track
            if src_t in self._tracks and dst_t in self._tracks and 0 <= self._drag_segment_idx < len(self._tracks[src_t].clips):
                clip_to_move = self._tracks[src_t].clips.pop(self._drag_segment_idx)
                self._tracks[dst_t].clips.append(clip_to_move)
                self._tracks[dst_t].clips.sort(key=lambda c: c.timeline_start)
                self._selected_track = dst_t
                self._selected_segment = self._tracks[dst_t].clips.index(clip_to_move)
                self.track_selected.emit(dst_t)

        if self._drag_mode in ('segment', 'trim_left', 'trim_right'):
            for (t_name, i) in self._drag_orig_starts.keys():
                if t_name in self._tracks:
                    self._tracks[t_name].clips.sort(key=lambda c: c.timeline_start)

        self._drag_dest_track = None
        self._drag_mode = None
        self._drag_track = None
        self._drag_segment_idx = -1
        self.segments_changed.emit()
        self._update_layout()
        self.update()

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        t = self._x_to_time(pos.x())
        self._playhead = t
        self.position_changed.emit(t)

        action_split = menu.addAction("✂  Split at Playhead (S)")
        action_split.triggered.connect(self.split_requested.emit)

        if self._selected_segment >= 0 and self._selected_track and self._selected_track in self._tracks:
            curr_track = self._tracks[self._selected_track]
            curr_clip = curr_track.clips[self._selected_segment]

            menu.addSeparator()
            # Move to other channel submenu
            move_menu = menu.addMenu("➡️  Move Clip to Channel...")
            for t_name, t_obj in self._tracks.items():
                if t_obj.type == curr_track.type and t_name != self._selected_track:
                    act_move = move_menu.addAction(f"Move to '{t_obj.name}'")
                    def _do_move(target_k=t_name):
                        if 0 <= self._selected_segment < len(curr_track.clips):
                            c = curr_track.clips.pop(self._selected_segment)
                            self._tracks[target_k].clips.append(c)
                            self._tracks[target_k].clips.sort(key=lambda item: item.timeline_start)
                            self._selected_track = target_k
                            self._selected_segment = self._tracks[target_k].clips.index(c)
                            self.segments_changed.emit()
                            self.track_selected.emit(target_k)
                            self.update()
                    act_move.triggered.connect(_do_move)

            # Transitions quick preset
            trans_menu = menu.addMenu("✨  Add Transition Preset...")
            for dur, label in [(0.5, "0.5s Fade"), (1.0, "1.0s Crossfade"), (1.5, "1.5s Dissolve"), (2.0, "2.0s Cinematic Fade")]:
                act_trans = trans_menu.addAction(label)
                def _do_trans(d=dur):
                    curr_clip.fade_in = d
                    curr_clip.fade_out = d
                    self.segments_changed.emit()
                    self.update()
                act_trans.triggered.connect(_do_trans)

            menu.addSeparator()
            action_del = menu.addAction("🗑  Delete Segment (Del)")
            action_del.triggered.connect(self.delete_requested.emit)

        menu.addSeparator()
        action_reset = menu.addAction("↺  Reset All Segments")
        action_reset.triggered.connect(self.reset_requested.emit)
        menu.exec(self.mapToGlobal(pos))

    def keyPressEvent(self, event):
        # Split/Delete are routed through the editor's undo-aware handlers
        # (see VideoEditor._setup_shortcuts) so we don't duplicate them here
        # and risk double-applying an action.
        super().keyPressEvent(event)


# ═══════════════════════════════════════════════════════════════
#  EXPORT DIALOG
# ═══════════════════════════════════════════════════════════════

class ExportDialog(QDialog):
    """Dialog for configuring export settings."""

    def __init__(self, parent=None, duration=0.0, segment_count=1):
        super().__init__(parent)
        self.setWindowTitle("Export Video")
        self.setMinimumWidth(400)
        self.setStyleSheet(self._stylesheet())

        layout = QVBoxLayout(self)

        # Info
        info = QLabel(f"Duration: {duration:.1f}s  |  Segments: {segment_count}")
        info.setStyleSheet("color: #888890; font-size: 11px;")
        layout.addWidget(info)

        # Form
        form = QFormLayout()

        self.quality_combo = QComboBox()
        self.quality_combo.addItems(["High (CRF 18)", "Medium (CRF 23)", "Low (CRF 28)"])
        self.quality_combo.setCurrentIndex(1)
        form.addRow("Quality:", self.quality_combo)

        self.reencode_check = QCheckBox("Re-encode for frame-accurate cuts")
        self.reencode_check.setChecked(True)
        self.reencode_check.setToolTip("If unchecked, uses fast stream copy (may not be frame-accurate)")
        form.addRow("", self.reencode_check)

        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems(["Original", "1920x1080", "1280x720", "854x480", "640x360"])
        form.addRow("Resolution:", self.resolution_combo)

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["Original", "60", "30", "24"])
        form.addRow("Frame Rate:", self.fps_combo)

        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP4 (H.264)", "WebM (VP9)", "MKV (H.264)"])
        form.addRow("Format:", self.format_combo)

        layout.addLayout(form)

        # Output file
        file_layout = QHBoxLayout()
        self.file_label = QLabel("output.mp4")
        self.file_label.setStyleSheet("color: #e0e0e0; padding: 4px;")
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self._browse)
        file_layout.addWidget(self.file_label)
        file_layout.addWidget(browse_btn)
        layout.addLayout(file_layout)

        # Buttons
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

        self._output_path = None

        # Default output path
        default_dir = os.path.join(os.path.expanduser("~"), "Videos")
        os.makedirs(default_dir, exist_ok=True)
        self._output_path = os.path.join(default_dir, f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        self.file_label.setText(os.path.basename(self._output_path))

    def _browse(self):
        ext_map = {0: "mp4", 1: "webm", 2: "mkv"}
        ext = ext_map.get(self.format_combo.currentIndex(), "mp4")
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video",
            self._output_path or "",
            f"Video (*.{ext})"
        )
        if path:
            self._output_path = path
            self.file_label.setText(os.path.basename(path))

    def _on_accept(self):
        if not self._output_path:
            QMessageBox.warning(self, "No Output", "Please select an output file.")
            return
        self.accept()

    def get_settings(self):
        quality_map = {0: "high", 1: "medium", 2: "low"}
        resolution = None
        res_idx = self.resolution_combo.currentIndex()
        if res_idx > 0:
            resolution = self.resolution_combo.currentText()

        fps = None
        fps_idx = self.fps_combo.currentIndex()
        if fps_idx > 0:
            fps = self.fps_combo.currentText()

        format_map = {0: "mp4", 1: "webm", 2: "mkv"}
        fmt = format_map.get(self.format_combo.currentIndex(), "mp4")

        # Ensure extension matches format
        path = self._output_path
        if not path.endswith(f".{fmt}"):
            path = os.path.splitext(path)[0] + f".{fmt}"

        return {
            "quality": quality_map.get(self.quality_combo.currentIndex(), "medium"),
            "re_encode": self.reencode_check.isChecked(),
            "resolution": resolution,
            "fps": fps,
            "format": fmt,
            "output_path": path,
        }

    def _stylesheet(self):
        return f"""
            QDialog {{ background: {CLR_PANEL}; color: {CLR_TEXT}; }}
            QLabel {{ color: {CLR_TEXT}; }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; padding: 4px 8px; border-radius: 4px;
            }}
            QCheckBox {{ color: {CLR_TEXT}; }}
            QPushButton {{
                background: {CLR_ACCENT}; color: white; border: none;
                padding: 6px 16px; border-radius: 4px; font-weight: bold;
            }}
            QPushButton:hover {{ background: {CLR_ACCENT_2}; }}
        """


# ═══════════════════════════════════════════════════════════════
#  EXPORT WORKER THREAD
# ═══════════════════════════════════════════════════════════════

class ExportWorker(QThread):
    progress = Signal(str, int)  # message, percent
    finished_signal = Signal(bool, str)  # success, message

    def __init__(self, command, temp_dir, output_path, total_duration=0.0):
        super().__init__()
        self.command = command
        self.temp_dir = temp_dir
        self.output_path = output_path
        self.total_duration = total_duration

    def run(self):
        try:
            self.progress.emit("Pre-rendering timeline overlays...", 5)
            logger.info("Executing timeline export: " + " ".join(self.command))

            creationflags = 0
            if platform.system() == "Windows":
                creationflags = subprocess.CREATE_NO_WINDOW

            process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                bufsize=1,
                creationflags=creationflags
            )

            while True:
                line = process.stderr.readline()
                if not line:
                    break
                logger.info(f"[FFMPEG] {line.strip()}")

                # Real-time frame rendering percentages
                time_match = re.search(r"time=(\d+):(\d+):(\d+\.\d+)", line)
                if time_match and self.total_duration > 0:
                    hours, minutes, seconds = time_match.groups()
                    elapsed = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                    pct = int(min(99, (elapsed / self.total_duration) * 100))
                    self.progress.emit(f"Rendering multi-channel timeline... {pct}%", pct)

            process.wait()
            ret_code = process.returncode

            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)

            if ret_code == 0:
                self.progress.emit("Done!", 100)
                self.finished_signal.emit(True, self.output_path)
            else:
                self.finished_signal.emit(False, f"ffmpeg exited with error code {ret_code}.")

        except Exception as e:
            import traceback
            logger.error(f"Timeline export crashed:\n{traceback.format_exc()}")
            self.finished_signal.emit(False, str(e))


# ═══════════════════════════════════════════════════════════════
#  VIDEO EDITOR WINDOW
# ═══════════════════════════════════════════════════════════════

class WaveformThread(QThread):
    waveform_ready = Signal(str, str, float, str)  # track_name, path, duration, wave_path

    def __init__(self, track_name, audio_path, duration):
        super().__init__()
        self.track_name = track_name
        self.audio_path = audio_path
        self.duration = duration

    def run(self):
        wave_path = FFmpegHelper.generate_waveform(self.audio_path)
        self.waveform_ready.emit(self.track_name, self.audio_path, self.duration, wave_path)


class CompositionViewportWidget(QWidget):
    """
    Interactive Viewport Canvas for multi-track timeline preview:
    - Direct-manipulation click-and-drag position & corner resizing of clips
    - Top-layer Text Overlay rendering
    - Freehand Canvas Drawing annotation tool
    - Spline/Motion path visualization
    """
    clip_transformed = Signal(str, int, float, float, float)  # track_name, seg_idx, x, y, scale

    def __init__(self, editor_ref, parent=None):
        super().__init__(parent)
        self.editor = editor_ref
        self.setMinimumHeight(320)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background: {CLR_BG}; border-radius: 8px;")

        self.draw_mode = False
        self.draw_color = QColor(255, 78, 80)
        self.draw_pen_size = 4

        self._active_drawing_clip = None
        self._active_drawing_path = None

        self._active_handles = []  # List of dicts: {clip, track_name, seg_idx, rect, handle_rects}
        self._selected_viewport_clip = None
        self._drag_action = None  # 'move', 'resize_br', 'resize_tl', etc.
        self._drag_start_pos = QPoint()
        self._orig_clip_x = 0.0
        self._orig_clip_y = 0.0
        self._orig_clip_scale_x = 1.0
        self._orig_clip_scale_y = 1.0

    def _get_canvas_rect(self):
        proj_ratio_str = self.editor.project_data.get('aspect_ratio', '16:9')
        try:
            rw, rh = map(float, proj_ratio_str.split(':'))
            aspect = rw / rh
        except Exception:
            aspect = 16.0 / 9.0

        widget_aspect = self.width() / max(1, self.height())
        if widget_aspect > aspect:
            ch = self.height()
            cw = int(ch * aspect)
            cx = (self.width() - cw) // 2
            cy = 0
        else:
            cw = self.width()
            ch = int(cw / aspect)
            cx = 0
            cy = (self.height() - ch) // 2

        return QRectF(cx, cy, cw, ch)

    def clear_drawings(self):
        t_name = self.editor.timeline._selected_track
        s_idx = self.editor.timeline._selected_segment
        if t_name and s_idx >= 0 and t_name in self.editor.timeline._tracks:
            clip = self.editor.timeline._tracks[t_name].clips[s_idx]
            if clip.type == "drawing":
                clip.drawing_paths.clear()
                self.update()
                self.editor._save_project_state()

    def set_draw_mode(self, enabled):
        self.draw_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        canvas_rect = self._get_canvas_rect()

        # Draw Widget Background (Letterbox Bounds)
        painter.fillRect(self.rect(), QColor(10, 10, 15))

        # Draw Real Project Canvas Area
        painter.fillRect(canvas_rect, QColor(0, 0, 0))

        if not self.editor or not hasattr(self.editor, 'timeline'):
            return

        playhead_t = self.editor.timeline.get_playhead()
        tracks = self.editor.timeline._tracks

        # Evaluate and render video/image tracks from bottom to top
        video_tracks = [t for t in tracks.values() if t.type == Track.VIDEO and t.name != "text_overlays"]
        self._active_handles.clear()

        for track_name, track in tracks.items():
            if track.type != Track.VIDEO or track_name == "text_overlays":
                continue

            for idx, clip in enumerate(track.clips):
                if clip.timeline_start <= playhead_t < clip.timeline_end:
                    scale_x = getattr(clip, 'video_scale_x', 1.0)
                    scale_y = getattr(clip, 'video_scale_y', 1.0)
                    vw = canvas_rect.width() * scale_x
                    vh = canvas_rect.height() * scale_y
                    vx = canvas_rect.x() + clip.video_x * canvas_rect.width()
                    vy = canvas_rect.y() + clip.video_y * canvas_rect.height()

                    clip_rect = QRectF(vx, vy, vw, vh)
                    is_selected = (track_name == self.editor.timeline._selected_track and idx == self.editor.timeline._selected_segment)

                    # Real-time Transitions & Opacity Blending
                    local_pos = playhead_t - clip.timeline_start
                    base_opacity = getattr(clip, 'opacity', 1.0)
                    fade_in_dur = getattr(clip, 'fade_in', 0.0)
                    fade_out_dur = getattr(clip, 'fade_out', 0.0)

                    if fade_in_dur > 0 and local_pos < fade_in_dur:
                        base_opacity *= max(0.0, min(1.0, local_pos / fade_in_dur))
                    if fade_out_dur > 0 and (clip.duration - local_pos) < fade_out_dur:
                        base_opacity *= max(0.0, min(1.0, (clip.duration - local_pos) / fade_out_dur))

                    painter.save()
                    painter.setOpacity(max(0.0, min(1.0, base_opacity)))

                    if clip.type == "drawing":
                        if is_selected:
                            painter.fillRect(clip_rect, QColor(255, 255, 255, 20))
                        
                        painter.save()
                        painter.translate(clip_rect.x(), clip_rect.y())
                        if clip_rect.width() > 0 and clip_rect.height() > 0:
                            painter.scale(clip_rect.width() / 1000.0, clip_rect.height() / 1000.0)
                        
                        for p, col, w in getattr(clip, 'drawing_paths', []):
                            pen = QPen(col, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                            pen.setCosmetic(True)
                            painter.setPen(pen)
                            painter.drawPath(p)
                        
                        if getattr(self, '_active_drawing_path', None) and getattr(self, '_active_drawing_clip', None) == clip:
                            pen = QPen(self.draw_color, self.draw_pen_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                            pen.setCosmetic(True)
                            painter.setPen(pen)
                            painter.drawPath(self._active_drawing_path)
                            
                        painter.restore()

                    else:
                        image = None
                        if clip.source_path.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                            image = QImage(clip.source_path)
                        elif track_name in self.editor.track_players:
                            image = self.editor.track_players[track_name].current_image
                        
                        if image and not image.isNull():
                            painter.drawImage(clip_rect, image)
                        else:
                            painter.fillRect(clip_rect, QColor(42, 27, 78, 200))

                    painter.restore()

                    # Frame outline & Handles
                    painter.setPen(QPen(QColor(CLR_ACCENT_2), 1, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(clip_rect)

                    handle_size = 10
                    br_handle = QRectF(clip_rect.right() - handle_size, clip_rect.bottom() - handle_size, handle_size, handle_size)
                    painter.setBrush(QColor(CLR_ACCENT_2))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(br_handle)

                    self._active_handles.append({
                        "clip": clip,
                        "track_name": track_name,
                        "seg_idx": idx,
                        "rect": clip_rect,
                        "br_handle": br_handle
                    })

        # RENDER TEXT OVERLAYS ON TOP OF ALL VIDEO TRACKS WITHIN STRICT TIME BOUNDS
        if "text_overlays" in tracks:
            text_track = tracks["text_overlays"]
            for idx, clip in enumerate(text_track.clips):
                if clip.timeline_start <= playhead_t < clip.timeline_end:
                    text_str = getattr(clip, 'text_content', clip.name)
                    if not text_str:
                        continue

                    scale_x = getattr(clip, 'video_scale_x', 1.0)
                    scale_y = getattr(clip, 'video_scale_y', 1.0)

                    font_family = getattr(clip, 'text_font_family', 'Segoe UI')
                    font_size = getattr(clip, 'text_font_size', 28)

                    # Safely handle QColor conversion if text_color was loaded from JSON dict
                    tc_val = getattr(clip, 'text_color', QColor(255, 255, 255))
                    bg_val = getattr(clip, 'text_bg_color', QColor(0, 0, 0, 150))
                    text_color = QColor(*tc_val) if isinstance(tc_val, (list, tuple)) else (tc_val if isinstance(tc_val, QColor) else QColor(255, 255, 255))
                    bg_color = QColor(*bg_val) if isinstance(bg_val, (list, tuple)) else (bg_val if isinstance(bg_val, QColor) else QColor(0, 0, 0, 150))

                    font = QFont(font_family, font_size, QFont.Bold)
                    painter.setFont(font)

                    fm = painter.fontMetrics()
                    text_rect = fm.boundingRect(text_str)
                    padding = 12

                    base_w = text_rect.width() + padding * 2
                    base_h = text_rect.height() + padding * 2
                    scaled_w = base_w * scale_x
                    scaled_h = base_h * scale_y

                    cx = int(canvas_rect.x() + clip.video_x * canvas_rect.width()) if clip.video_x > 0 else int(canvas_rect.x() + (canvas_rect.width() - scaled_w) / 2)
                    cy = int(canvas_rect.y() + clip.video_y * canvas_rect.height()) if clip.video_y > 0 else int(canvas_rect.y() + (canvas_rect.height() - scaled_h) / 2)

                    clip_rect = QRectF(cx, cy, scaled_w, scaled_h)

                    painter.save()
                    painter.translate(cx, cy)
                    painter.scale(scale_x, scale_y)

                    draw_box = QRectF(0, 0, base_w, base_h)

                    if bg_color.alpha() > 0:
                        painter.fillRect(draw_box, bg_color)
                        painter.setPen(QPen(QColor(CLR_BORDER), 1))
                        painter.drawRect(draw_box)

                    painter.setPen(text_color)
                    painter.drawText(draw_box, Qt.AlignCenter, text_str)
                    painter.restore()

                    # Interactive Handles for Text
                    painter.setPen(QPen(QColor(CLR_ACCENT_2), 1, Qt.DashLine))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawRect(clip_rect)

                    handle_size = 10
                    br_handle = QRectF(clip_rect.right() - handle_size, clip_rect.bottom() - handle_size, handle_size, handle_size)
                    painter.setBrush(QColor(CLR_ACCENT_2))
                    painter.setPen(Qt.NoPen)
                    painter.drawRect(br_handle)

                    self._active_handles.append({
                        "clip": clip,
                        "track_name": "text_overlays",
                        "seg_idx": idx,
                        "rect": clip_rect,
                        "br_handle": br_handle
                    })

    def _map_to_logical(self, pos, clip):
        canvas_rect = self._get_canvas_rect()
        clip_w = canvas_rect.width() * getattr(clip, 'video_scale_x', 1.0)
        clip_h = canvas_rect.height() * getattr(clip, 'video_scale_y', 1.0)
        clip_x = canvas_rect.x() + clip.video_x * canvas_rect.width()
        clip_y = canvas_rect.y() + clip.video_y * canvas_rect.height()
        if clip_w == 0 or clip_h == 0:
            return 0.0, 0.0
        lx = (pos.x() - clip_x) / clip_w * 1000.0
        ly = (pos.y() - clip_y) / clip_h * 1000.0
        return lx, ly

    def mousePressEvent(self, event):
        if self.draw_mode:
            t_name = self.editor.timeline._selected_track
            s_idx = self.editor.timeline._selected_segment
            if t_name and s_idx >= 0 and t_name in self.editor.timeline._tracks:
                clip = self.editor.timeline._tracks[t_name].clips[s_idx]
                if clip.type == "drawing":
                    self._active_drawing_clip = clip
                    self._active_drawing_path = QPainterPath()
                    lx, ly = self._map_to_logical(event.position(), clip)
                    self._active_drawing_path.moveTo(lx, ly)
                    self.update()
                    return
            return

        pos = event.position()
        self._drag_action = None

        for item in reversed(self._active_handles):
            if item["br_handle"].contains(pos):
                self._drag_action = "resize"
                self._selected_viewport_clip = item
                self._drag_start_pos = pos.toPoint()
                self._orig_clip_scale_x = getattr(item["clip"], 'video_scale_x', 1.0)
                self._orig_clip_scale_y = getattr(item["clip"], 'video_scale_y', 1.0)
                break
            elif item["rect"].contains(pos):
                self._drag_action = "move"
                self._selected_viewport_clip = item
                self._drag_start_pos = pos.toPoint()
                self._orig_clip_x = item["clip"].video_x
                self._orig_clip_y = item["clip"].video_y
                break


    def mouseMoveEvent(self, event):
        if self.draw_mode and getattr(self, '_active_drawing_path', None):
            lx, ly = self._map_to_logical(event.position(), self._active_drawing_clip)
            self._active_drawing_path.lineTo(lx, ly)
            self.update()
            return

        pos = event.position()

        canvas_rect = self._get_canvas_rect()

        if self._drag_action == "move" and self._selected_viewport_clip:
            clip = self._selected_viewport_clip["clip"]
            dx = (pos.x() - self._drag_start_pos.x()) / float(canvas_rect.width())
            dy = (pos.y() - self._drag_start_pos.y()) / float(canvas_rect.height())
            clip.video_x = max(0.0, min(1.0, self._orig_clip_x + dx))
            clip.video_y = max(0.0, min(1.0, self._orig_clip_y + dy))
            
            # Auto-update inspector if the dragged clip is selected in the timeline
            if getattr(self.editor, 'props_x_slider', None) and self.editor.timeline._selected_track == self._selected_viewport_clip["track_name"] and self.editor.timeline._selected_segment == self._selected_viewport_clip["seg_idx"]:
                self.editor.props_x_slider.blockSignals(True)
                self.editor.props_x_slider.setValue(int(clip.video_x * 100))
                self.editor.props_x_slider.blockSignals(False)
                self.editor.props_y_slider.blockSignals(True)
                self.editor.props_y_slider.setValue(int(clip.video_y * 100))
                self.editor.props_y_slider.blockSignals(False)

            self.update()
            self.editor._save_project_state()

        elif self._drag_action == "resize" and self._selected_viewport_clip:
            clip = self._selected_viewport_clip["clip"]
            dx = (pos.x() - self._drag_start_pos.x()) / float(canvas_rect.width())
            dy = (pos.y() - self._drag_start_pos.y()) / float(canvas_rect.height())

            clip.video_scale_x = max(0.05, min(10.0, self._orig_clip_scale_x + dx * 2.0))
            clip.video_scale_y = max(0.05, min(10.0, self._orig_clip_scale_y + dy * 2.0))

            if getattr(self.editor, 'props_scale_x_slider', None) and self.editor.timeline._selected_track == self._selected_viewport_clip["track_name"] and self.editor.timeline._selected_segment == self._selected_viewport_clip["seg_idx"]:
                self.editor.props_scale_x_slider.blockSignals(True)
                self.editor.props_scale_x_slider.setValue(int(clip.video_scale_x * 100))
                self.editor.props_scale_x_slider.blockSignals(False)
                self.editor.props_scale_y_slider.blockSignals(True)
                self.editor.props_scale_y_slider.setValue(int(clip.video_scale_y * 100))
                self.editor.props_scale_y_slider.blockSignals(False)

            self.update()
            self.editor._save_project_state()

    def mouseReleaseEvent(self, event):
        if self.draw_mode and getattr(self, '_active_drawing_path', None):
            self._active_drawing_clip.drawing_paths.append(
                (self._active_drawing_path, self.draw_color, self.draw_pen_size)
            )
            self._active_drawing_path = None
            self._active_drawing_clip = None
            self.update()
            self.editor._save_project_state()
            return

        self._drag_action = None
        self._selected_viewport_clip = None

        
class VideoEditor(QMainWindow):
    """Main video editing window with preview, timeline, mixer, and export."""

    def __init__(self, project_data=None, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.project_data = project_data or {}
        self.video_path = self.project_data.get("video_path")
        self.project_id = self.project_data.get("id")

        self.audio_mixer = {}
        self.text_overlays = []  # List of dicts: {"text": str, "x": float, "y": float, "start": float, "end": float}
        self.undo_stack = []
        self.redo_stack = []

        if self.video_path:
            self.setWindowTitle(f"Editor — {self.project_data.get('name', os.path.basename(self.video_path))}")
            self.duration = FFmpegHelper.get_video_duration(self.video_path)
        else:
            self.setWindowTitle(f"{APP_NAME} — Editor")
            self.duration = 0.0

        self.setMinimumSize(1000, 700)
        self.setStyleSheet(self._stylesheet())

        self.thumb_dir = None
        self._export_worker = None
        self._thumb_thread = None
        self._wave_threads = []

        self.track_players = {}
        self.playhead = 0.0
        self.is_playing = False

        self.playback_monitor = QTimer(self)
        self.playback_monitor.setInterval(33)  # ~30fps NLE central clock
        self.playback_monitor.timeout.connect(self._check_playback_position)

        self._build_ui()

        if self.video_path:
            self._load_video()
            self._load_project_state()
        else:
            self._show_empty_state()

        self._setup_shortcuts()


    def _fit_clip_to_canvas(self, clip, w, h):
        """Calculates scale and offset so the clip fits inside the project's aspect ratio without deformation."""
        media_aspect = w / max(1, h)
        proj_ratio_str = self.project_data.get('aspect_ratio', '16:9')
        try:
            rw, rh = map(float, proj_ratio_str.split(':'))
            proj_aspect = rw / rh
        except Exception:
            proj_aspect = 16.0 / 9.0

        if media_aspect > proj_aspect:
            # Media is wider than project; fit to width
            scale_x = 1.0
            scale_y = (1.0 / media_aspect) * proj_aspect
        else:
            # Media is taller than project; fit to height
            scale_y = 1.0
            scale_x = media_aspect / proj_aspect

        clip.video_scale_x = scale_x
        clip.video_scale_y = scale_y
        clip.video_x = (1.0 - scale_x) / 2.0
        clip.video_y = (1.0 - scale_y) / 2.0

    def _pick_draw_color(self):
        col = QColorDialog.getColor(self.viewport_widget.draw_color, self, "Select Drawing Color")
        if col.isValid():
            self.viewport_widget.draw_color = col

    def _move_selected_track_up(self):
        t_name = self.timeline._selected_track
        if not t_name or t_name not in self.timeline._tracks:
            return
        keys = list(self.timeline._tracks.keys())
        idx = keys.index(t_name)
        if idx > 0:
            keys[idx], keys[idx-1] = keys[idx-1], keys[idx]
            self.timeline._tracks = {k: self.timeline._tracks[k] for k in keys}
            self.timeline.update()
            self.viewport_widget.update()
            self._save_project_state()

    def _move_selected_track_down(self):
        t_name = self.timeline._selected_track
        if not t_name or t_name not in self.timeline._tracks:
            return
        keys = list(self.timeline._tracks.keys())
        idx = keys.index(t_name)
        if idx < len(keys) - 1:
            keys[idx], keys[idx+1] = keys[idx+1], keys[idx]
            self.timeline._tracks = {k: self.timeline._tracks[k] for k in keys}
            self.timeline.update()
            self.viewport_widget.update()
            self._save_project_state()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Top Section: Horizontal Splitter (Left Full Vertical Media Panel + Center Video Viewport)
        main_h_splitter = QSplitter(Qt.Horizontal)
        main_h_splitter.setHandleWidth(3)

        self.media_panel = MediaLibraryPanel(self)
        self.media_panel.setStyleSheet(f"background: {CLR_PANEL};")
        self.btn_add_media = self.media_panel.btn_add_media
        self.btn_add_to_timeline = self.media_panel.btn_add_to_timeline
        self.media_list = self.media_panel.media_list

        if self.project_data.get('media_library'):
            for item_path in self.project_data['media_library']:
                self.media_panel.add_media_item(item_path)

        self.media_panel.item_selection_changed.connect(self._on_media_selection_changed)

        main_h_splitter.addWidget(self.media_panel)

        # Center Interactive Composition Viewport Area
        center_widget = QWidget()
        center_layout = QVBoxLayout(center_widget)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(6)

        # Canvas Draw & Overlay Control Bar
        viewport_tools = QHBoxLayout()
        viewport_tools.setSpacing(6)

        self.btn_draw_mode = QPushButton("🎨 Draw")
        self.btn_draw_mode.setCheckable(True)
        self.btn_draw_mode.setToolTip("Enable Freehand Canvas Drawing Mode")
        self.btn_draw_mode.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_draw_mode.toggled.connect(lambda chk: self.viewport_widget.set_draw_mode(chk))
        viewport_tools.addWidget(self.btn_draw_mode)

        self.btn_color_pick = QPushButton("🖌️ Color")
        self.btn_color_pick.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_color_pick.clicked.connect(self._pick_draw_color)
        viewport_tools.addWidget(self.btn_color_pick)

        self.btn_clear_draw = QPushButton("🧹 Clear Drawings")
        self.btn_clear_draw.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_clear_draw.clicked.connect(lambda: self.viewport_widget.clear_drawings())
        viewport_tools.addWidget(self.btn_clear_draw)

        self.btn_reorder_track_up = QPushButton("⬆️ Move Channel Up")
        self.btn_reorder_track_up.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_reorder_track_up.clicked.connect(self._move_selected_track_up)
        viewport_tools.addWidget(self.btn_reorder_track_up)

        self.btn_reorder_track_down = QPushButton("⬇️ Move Channel Down")
        self.btn_reorder_track_down.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_reorder_track_down.clicked.connect(self._move_selected_track_down)
        viewport_tools.addWidget(self.btn_reorder_track_down)

        viewport_tools.addStretch()
        center_layout.addLayout(viewport_tools)

        self.viewport_widget = CompositionViewportWidget(self)
        center_layout.addWidget(self.viewport_widget, stretch=1)

        main_h_splitter.addWidget(center_widget)
        main_h_splitter.setSizes([240, 760])
        main_h_splitter.setStretchFactor(0, 0)
        main_h_splitter.setStretchFactor(1, 1)

        layout.addWidget(main_h_splitter, stretch=1)

        # Streamlined modern Transport Bar with step triggers
        controls = QHBoxLayout()
        controls.setSpacing(6)

        self.btn_to_start = QPushButton("⏮")
        self.btn_to_start.setToolTip("Go to Start (Home)")
        self.btn_to_start.setFixedSize(38, 38)
        self.btn_to_start.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_to_start.clicked.connect(lambda: (self.timeline.set_playhead(0.0), self._on_timeline_seek(0.0)))
        controls.addWidget(self.btn_to_start)

        self.btn_rewind = QPushButton("⏪")
        self.btn_rewind.setToolTip("Rewind / Shuttle Reverse (J)")
        self.btn_rewind.setFixedSize(38, 38)
        self.btn_rewind.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_rewind.clicked.connect(self._shuttle_reverse)
        controls.addWidget(self.btn_rewind)

        self.btn_play = QPushButton("▶")
        self.btn_play.setToolTip("Play / Pause (Space)")
        self.btn_play.setFixedSize(65, 38)
        self.btn_play.setStyleSheet(self._btn_style(CLR_ACCENT))
        self.btn_play.clicked.connect(self._toggle_playback)
        controls.addWidget(self.btn_play)

        self.btn_fwd = QPushButton("⏩")
        self.btn_fwd.setToolTip("Fast Forward / Shuttle Forward (L)")
        self.btn_fwd.setFixedSize(38, 38)
        self.btn_fwd.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_fwd.clicked.connect(self._shuttle_forward)
        controls.addWidget(self.btn_fwd)

        self.btn_to_end = QPushButton("⏭")
        self.btn_to_end.setToolTip("Go to End (End)")
        self.btn_to_end.setFixedSize(38, 38)
        self.btn_to_end.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_to_end.clicked.connect(lambda: (self.timeline.set_playhead(self.duration), self._on_timeline_seek(self.duration)))
        controls.addWidget(self.btn_to_end)

        self.time_label = QLabel("00:00:00 / 00:00:00")
        self.time_label.setStyleSheet(f"color: {CLR_TEXT}; font-family: Consolas; font-size: 13px; font-weight: bold;")
        self.time_label.setMinimumWidth(160)
        controls.addWidget(self.time_label)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(90)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.valueChanged.connect(self._set_master_volume)
        controls.addWidget(QLabel("🔊"))
        controls.addWidget(self.volume_slider)

        controls.addStretch()

        self.btn_split = QPushButton("✂ Split (S)")
        self.btn_split.setStyleSheet(self._btn_style(CLR_WARN))
        self.btn_split.clicked.connect(self._split_at_playhead)
        controls.addWidget(self.btn_split)

        self.btn_delete_seg = QPushButton("🗑 Delete")
        self.btn_delete_seg.setToolTip("Delete Clip (Delete)")
        self.btn_delete_seg.setStyleSheet(self._btn_style(CLR_DANGER))
        self.btn_delete_seg.clicked.connect(self._delete_segment)
        controls.addWidget(self.btn_delete_seg)

        self.btn_ripple_del = QPushButton("🌊 Ripple Del")
        self.btn_ripple_del.setToolTip("Ripple Delete - Shift Clip Gap (Shift+Delete)")
        self.btn_ripple_del.setStyleSheet(self._btn_style(CLR_ACCENT_2))
        self.btn_ripple_del.clicked.connect(self._ripple_delete_segment)
        controls.addWidget(self.btn_ripple_del)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.setStyleSheet(self._btn_style(CLR_TEXT_DIM))
        self.btn_reset.clicked.connect(self._reset_segments)
        controls.addWidget(self.btn_reset)

        layout.addLayout(controls)

        # Main Splitter (Timeline + Track Properties)
        main_splitter = QSplitter(Qt.Horizontal)

        # Timeline Container
        timeline_container = QWidget()
        tl_layout = QVBoxLayout(timeline_container)
        tl_layout.setContentsMargins(0, 0, 0, 0)
        tl_layout.setSpacing(4)

        tl_header = QHBoxLayout()
        tl_header.setSpacing(4)

        self.media_toggle_btn = QPushButton("📁 Media")
        self.media_toggle_btn.setToolTip("Toggle Media Library Dock")
        self.media_toggle_btn.setStyleSheet(self._btn_style(CLR_ACCENT_2))
        self.media_toggle_btn.clicked.connect(self._toggle_media_library)
        tl_header.addWidget(self.media_toggle_btn)

        self.btn_import_video = QPushButton("📹 +Video")
        self.btn_import_video.setToolTip("Import Video Clip onto Active Track")
        self.btn_import_video.setStyleSheet(self._btn_style(CLR_ACCENT))
        self.btn_import_video.clicked.connect(self._import_video)
        tl_header.addWidget(self.btn_import_video)

        self.btn_add_audio = QPushButton("🎵 +Audio")
        self.btn_add_audio.setToolTip("Import Audio Track (.mp3, .wav)")
        self.btn_add_audio.setStyleSheet(self._btn_style(CLR_ACCENT_2))
        self.btn_add_audio.clicked.connect(self._import_audio)
        tl_header.addWidget(self.btn_add_audio)

        self.btn_add_video_track = QPushButton("🎞️ +Track")
        self.btn_add_video_track.setToolTip("Add Empty Video Channel")
        self.btn_add_video_track.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_add_video_track.clicked.connect(
            lambda: (self.timeline.add_track(Track.VIDEO, f"Video {len([t for t in self.timeline._tracks.values() if t.type == Track.VIDEO])+1}"), self._save_project_state())
        )
        tl_header.addWidget(self.btn_add_video_track)

        self.btn_add_text = QPushButton("✏ +Text")
        self.btn_add_text.setToolTip("Add Text Overlay Clip")
        self.btn_add_text.setStyleSheet(self._btn_style(CLR_WARN))
        self.btn_add_text.clicked.connect(self._add_text_overlay)
        tl_header.addWidget(self.btn_add_text)

        self.btn_add_draw_clip = QPushButton("🎨 +Draw")
        self.btn_add_draw_clip.setToolTip("Add Drawing Overlay Clip")
        self.btn_add_draw_clip.setStyleSheet(self._btn_style(CLR_WARN))
        self.btn_add_draw_clip.clicked.connect(self._add_drawing_overlay)
        tl_header.addWidget(self.btn_add_draw_clip)

        self.btn_toggle_link = QPushButton("🔗 Link")
        self.btn_toggle_link.setToolTip("Link / Unlink Audio and Video")
        self.btn_toggle_link.setStyleSheet(self._btn_style(CLR_TEXT_DIM))
        self.btn_toggle_link.clicked.connect(self._toggle_track_link)
        tl_header.addWidget(self.btn_toggle_link)

        self.btn_save = QPushButton("💾 Save")
        self.btn_save.setStyleSheet(self._btn_style(CLR_SUCCESS))
        self.btn_save.clicked.connect(self._save_project_state)
        tl_header.addWidget(self.btn_save)

        self.btn_home = QPushButton("🏠")
        self.btn_home.setToolTip("Return to Home Studio")
        self.btn_home.setStyleSheet(self._btn_style(CLR_TEXT_DIM))
        self.btn_home.clicked.connect(self._go_home)
        tl_header.addWidget(self.btn_home)

        tl_header.addStretch()

        # Zoom Controls
        self.btn_zoom_out = QPushButton("🔍-")
        self.btn_zoom_out.setToolTip("Zoom Out (Ctrl+Minus)")
        self.btn_zoom_out.setFixedSize(32, 28)
        self.btn_zoom_out.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_zoom_out.clicked.connect(lambda: self.timeline.zoom_out())
        tl_header.addWidget(self.btn_zoom_out)

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 100)
        self.zoom_slider.setValue(15)
        self.zoom_slider.setFixedWidth(90)
        self.zoom_slider.setToolTip("Timeline Zoom Level")
        self.zoom_slider.valueChanged.connect(lambda v: self.timeline.set_zoom(float(v)))
        tl_header.addWidget(self.zoom_slider)

        self.btn_zoom_in = QPushButton("🔍+")
        self.btn_zoom_in.setToolTip("Zoom In (Ctrl+Plus)")
        self.btn_zoom_in.setFixedSize(32, 28)
        self.btn_zoom_in.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_zoom_in.clicked.connect(lambda: self.timeline.zoom_in())
        tl_header.addWidget(self.btn_zoom_in)

        self.btn_zoom_fit = QPushButton("↔ Fit")
        self.btn_zoom_fit.setToolTip("Zoom to Fit Timeline")
        self.btn_zoom_fit.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_zoom_fit.clicked.connect(lambda: self.timeline.zoom_fit())
        tl_header.addWidget(self.btn_zoom_fit)

        tl_layout.addLayout(tl_header)

        # Timeline inside QScrollArea
        self.timeline = TimelineWidget()
        self.timeline.set_duration(self.duration)
        self.timeline.position_changed.connect(self._on_timeline_seek)
        self.timeline.media_dropped.connect(self._on_media_dropped_to_timeline)
        self.timeline.split_requested.connect(self._split_at_playhead)
        self.timeline.delete_requested.connect(self._delete_segment)
        self.timeline.reset_requested.connect(self._reset_segments)
        self.timeline.zoom_changed.connect(lambda z: self.zoom_slider.setValue(int(z)))

        self.timeline_scroll = QScrollArea()
        self.timeline_scroll.setWidgetResizable(True)
        self.timeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.timeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.timeline_scroll.setStyleSheet(f"QScrollArea {{ background: {CLR_TIMELINE_BG}; border: 1px solid {CLR_BORDER}; border-radius: 6px; }}")
        self.timeline_scroll.setWidget(self.timeline)

        tl_layout.addWidget(self.timeline_scroll, stretch=1)
        main_splitter.addWidget(timeline_container)

        # Inspector (Track & Clip Properties)
        props_group = QGroupBox("Inspector / Clip Properties")
        props_layout = QVBoxLayout(props_group)
        props_layout.setSpacing(5)

        self.props_track_label = QLabel("No clip or track selected")
        self.props_track_label.setStyleSheet(f"color: {CLR_TEXT}; font-weight: bold;")
        props_layout.addWidget(self.props_track_label)

        # Channel / Track Reassignment Combo & Snap Button
        reassign_layout = QHBoxLayout()
        self.clip_track_combo = QComboBox()
        self.clip_track_combo.setToolTip("Reassign Clip to Channel / Track")
        self.clip_track_combo.setEnabled(False)
        self.clip_track_combo.currentIndexChanged.connect(self._on_reassign_clip_track)
        reassign_layout.addWidget(QLabel("Channel:"))
        reassign_layout.addWidget(self.clip_track_combo)

        self.btn_snap_prev = QPushButton("🧲 Snap to Prev")
        self.btn_snap_prev.setToolTip("Snap this clip directly after the previous clip on this track")
        self.btn_snap_prev.setStyleSheet(self._btn_style(CLR_ACCENT_2))
        self.btn_snap_prev.clicked.connect(self._snap_clip_to_previous)
        reassign_layout.addWidget(self.btn_snap_prev)
        props_layout.addLayout(reassign_layout)

        # Opacity & Transition Controls
        self.props_opacity_label = QLabel("Layer Opacity (100%):")
        self.props_opacity_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(self.props_opacity_label)
        self.props_opacity_slider = QSlider(Qt.Horizontal)
        self.props_opacity_slider.setRange(0, 100)
        self.props_opacity_slider.setValue(100)
        self.props_opacity_slider.setEnabled(False)
        self.props_opacity_slider.valueChanged.connect(self._on_clip_opacity_changed)
        props_layout.addWidget(self.props_opacity_slider)

        self.props_fade_in_label = QLabel("Fade In Transition (0.0s):")
        self.props_fade_in_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(self.props_fade_in_label)
        self.props_fade_in_slider = QSlider(Qt.Horizontal)
        self.props_fade_in_slider.setRange(0, 50)
        self.props_fade_in_slider.setValue(0)
        self.props_fade_in_slider.setEnabled(False)
        self.props_fade_in_slider.valueChanged.connect(self._on_clip_fade_in_changed)
        props_layout.addWidget(self.props_fade_in_slider)

        self.props_fade_out_label = QLabel("Fade Out Transition (0.0s):")
        self.props_fade_out_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(self.props_fade_out_label)
        self.props_fade_out_slider = QSlider(Qt.Horizontal)
        self.props_fade_out_slider.setRange(0, 50)
        self.props_fade_out_slider.setValue(0)
        self.props_fade_out_slider.setEnabled(False)
        self.props_fade_out_slider.valueChanged.connect(self._on_clip_fade_out_changed)
        props_layout.addWidget(self.props_fade_out_slider)

        # Position Preset Combo (Picture-in-Picture / Corners)
        preset_label = QLabel("PIP / Position Preset:")
        preset_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(preset_label)

        self.pip_preset_combo = QComboBox()
        self.pip_preset_combo.addItems([
            "Full Screen (100%)",
            "Top-Left Corner (PIP)",
            "Top-Right Corner (PIP)",
            "Bottom-Left Corner (PIP)",
            "Bottom-Right Corner (PIP)",
            "Center Overlay (50%)"
        ])
        self.pip_preset_combo.setEnabled(False)
        self.pip_preset_combo.currentIndexChanged.connect(self._apply_pip_preset)
        props_layout.addWidget(self.pip_preset_combo)

        # Scale X & Y Controls
        self.props_scale_x_label = QLabel("Scale X (Width):")
        self.props_scale_x_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(self.props_scale_x_label)
        self.props_scale_x_slider = QSlider(Qt.Horizontal)
        self.props_scale_x_slider.setRange(5, 1000)
        self.props_scale_x_slider.setValue(100)
        self.props_scale_x_slider.setEnabled(False)
        self.props_scale_x_slider.valueChanged.connect(self._on_clip_scale_x_changed)
        props_layout.addWidget(self.props_scale_x_slider)

        self.props_scale_y_label = QLabel("Scale Y (Height):")
        self.props_scale_y_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(self.props_scale_y_label)
        self.props_scale_y_slider = QSlider(Qt.Horizontal)
        self.props_scale_y_slider.setRange(5, 1000)
        self.props_scale_y_slider.setValue(100)
        self.props_scale_y_slider.setEnabled(False)
        self.props_scale_y_slider.valueChanged.connect(self._on_clip_scale_y_changed)
        props_layout.addWidget(self.props_scale_y_slider)

        # Position X & Y Controls
        pos_x_label = QLabel("Position X:")
        pos_x_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(pos_x_label)
        self.props_x_slider = QSlider(Qt.Horizontal)
        self.props_x_slider.setRange(0, 100)
        self.props_x_slider.setValue(0)
        self.props_x_slider.setEnabled(False)
        self.props_x_slider.valueChanged.connect(self._on_clip_x_changed)
        props_layout.addWidget(self.props_x_slider)

        pos_y_label = QLabel("Position Y:")
        pos_y_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(pos_y_label)
        self.props_y_slider = QSlider(Qt.Horizontal)
        self.props_y_slider.setRange(0, 100)
        self.props_y_slider.setValue(0)
        self.props_y_slider.setEnabled(False)
        self.props_y_slider.valueChanged.connect(self._on_clip_y_changed)
        props_layout.addWidget(self.props_y_slider)

        # Volume Control
        vol_label = QLabel("Audio Volume:")
        vol_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        props_layout.addWidget(vol_label)
        self.props_vol_slider = QSlider(Qt.Horizontal)
        self.props_vol_slider.setRange(0, 200)
        self.props_vol_slider.setValue(100)
        self.props_vol_slider.setEnabled(False)
        self.props_vol_slider.valueChanged.connect(self._on_track_volume_changed)
        props_layout.addWidget(self.props_vol_slider)

        props_layout.addStretch()
        main_splitter.addWidget(props_group)

        main_splitter.setSizes([700, 300])
        layout.addWidget(main_splitter, stretch=1)

        # Bottom Bar
        bottom = QHBoxLayout()
        self.info_label = QLabel(f"Duration: {self.duration:.1f}s  |  Segments: 1")
        self.info_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        bottom.addWidget(self.info_label)
        bottom.addStretch()

        self.btn_export = QPushButton("📤 Export Video")
        self.btn_export.setStyleSheet(self._btn_style(CLR_ACCENT))
        self.btn_export.setMinimumWidth(140)
        self.btn_export.clicked.connect(self._export)
        bottom.addWidget(self.btn_export)

        self.btn_new = QPushButton("⏺ Record New Media")
        self.btn_new.setStyleSheet(self._btn_style(CLR_DANGER))
        self.btn_new.setMinimumWidth(140)
        self.btn_new.clicked.connect(self._record_new_media)
        bottom.addWidget(self.btn_new)
        layout.addLayout(bottom)

        # Progress
        self.progress_widget = QWidget()
        prog_layout = QHBoxLayout(self.progress_widget)
        self.progress_bar = QProgressBar()
        self.progress_label = QLabel("Ready")
        prog_layout.addWidget(self.progress_label)
        prog_layout.addWidget(self.progress_bar)
        self.progress_widget.setVisible(False)
        layout.addWidget(self.progress_widget)

        self.timeline.segments_changed.connect(self._update_info)
        self.timeline.track_selected.connect(self._on_track_selected)

    def _set_master_volume(self, val):
        vol = val / 100.0
        for tp in self.track_players.values():
            tp.audio.setVolume(vol)

    def _show_empty_state(self):
        """Display a prompt to import media when the project is empty."""
        if not hasattr(self, 'empty_state_label'):
            self.empty_state_label = QLabel("📹 Drop a video file here or click 'Import Video' to begin.")
            self.empty_state_label.setAlignment(Qt.AlignCenter)
            self.empty_state_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 18px; background: {CLR_BG}; border-radius: 8px;")
            self.empty_state_label.setAcceptDrops(True)

            def dragEnterEvent(e):
                if e.mimeData().hasUrls():
                    e.acceptProposedAction()
            def dropEvent(e):
                for url in e.mimeData().urls():
                    path = url.toLocalFile()
                    if path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
                        self.video_path = path
                        self.project_data['video_path'] = path
                        self.project_data['name'] = os.path.basename(path)
                        self.empty_state_label.hide()
                        self._load_video()
                        self._save_project_state()
                        break

            self.empty_state_label.dragEnterEvent = dragEnterEvent
            self.empty_state_label.dropEvent = dropEvent

        if self.centralWidget().layout().indexOf(self.empty_state_label) == -1:
            self.centralWidget().layout().insertWidget(1, self.empty_state_label, stretch=1)
        self.empty_state_label.show()

    def _import_video(self):
        """Open a file dialog to import a video into the current project."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Import Video",
            os.path.join(os.path.expanduser("~"), "Videos"),
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*.*)"
        )
        if path:
            self.video_path = path
            self.project_data['video_path'] = path
            self.project_data['name'] = os.path.basename(path)
            if hasattr(self, 'empty_state_label'):
                self.empty_state_label.hide()
            if "main_video" in self.timeline._tracks:
                del self.timeline._tracks["main_video"]
            if "main_audio" in self.timeline._tracks:
                del self.timeline._tracks["main_audio"]
            self._load_video()
            self._save_project_state()

    def _add_drawing_overlay(self):
        """Add a drawing overlay clip at the current playhead position."""
        pos = self.timeline.get_playhead()
        duration = 5.0

        if "drawings" not in self.timeline._tracks:
            track = Track(Track.VIDEO, "Drawings", duration=self.duration)
            self.timeline._tracks["drawings"] = track

        clip = Clip("", pos, duration, clip_type="drawing", name="Drawing")

        self.timeline._tracks["drawings"].clips.append(clip)
        self.timeline._tracks["drawings"].clips.sort(key=lambda c: c.timeline_start)

        self.timeline.segments_changed.emit()
        self.timeline._update_layout()
        self.timeline.update()
        self.viewport_widget.update()
        self._save_project_state()

    def _add_text_overlay(self):
        """Add a customizable text overlay at the current playhead position."""
        text, ok = QInputDialog.getText(self, "Add Text Overlay", "Enter overlay text:")
        if ok and text:
            pos = self.timeline.get_playhead()
            duration = 5.0

            overlay = {
                "text": text,
                "start": pos,
                "end": pos + duration
            }

            if "text_overlays" not in self.timeline._tracks:
                track = Track(Track.VIDEO, "Text Overlays", duration=self.duration)
                self.timeline._tracks["text_overlays"] = track

            clip = Clip("", pos, duration, clip_type="text", name=text)
            clip.text_content = text
            clip.text_font_family = "Segoe UI"
            clip.text_font_size = 28
            clip.text_color = QColor(255, 255, 255)
            clip.text_bg_color = QColor(0, 0, 0, 150)

            self.timeline._tracks["text_overlays"].clips.append(clip)
            self.timeline._tracks["text_overlays"].clips.sort(key=lambda c: c.timeline_start)

            self.timeline.segments_changed.emit()
            self.timeline._update_layout()
            self.timeline.update()
            self.viewport_widget.update()
            self._save_project_state()

    def _save_project_state(self):
        """Save the current project configuration with rich compositing metadata."""
        if not self.project_id:
            self.project_id = f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        self.project_data['id'] = self.project_id
        self.project_data['video_path'] = self.video_path

        tracks_data = []
        for name, track in self.timeline._tracks.items():
            clips_serialized = []
            for c in track.clips:
                clips_serialized.append({
                    "source_path": c.source_path,
                    "timeline_start": c.timeline_start,
                    "duration": c.duration,
                    "source_start": c.source_start,
                    "type": c.type,
                    "name": c.name,
                    "volume": c.volume,
                    "video_scale_x": c.video_scale_x,
                    "video_scale_y": c.video_scale_y,
                    "video_x": c.video_x,
                    "video_y": c.video_y,
                    "opacity": getattr(c, "opacity", 1.0),
                    "fade_in": getattr(c, "fade_in", 0.0),
                    "fade_out": getattr(c, "fade_out", 0.0),
                    "text_content": getattr(c, "text_content", ""),
                })

            tracks_data.append({
                "name": name,
                "display_name": track.name,
                "type": track.type,
                "path": track.path,
                "volume": track.volume,
                "video_scale_x": track.video_scale_x,
                "video_scale_y": track.video_scale_y,
                "clips": clips_serialized
            })
        self.project_data['tracks'] = tracks_data

        if not self.project_data.get('name'):
            self.project_data['name'] = "Untitled Project"

        if not self.project_data.get('thumbnail_path') and self.video_path:
            thumb_dir = tempfile.mkdtemp()
            thumb_path = os.path.join(thumb_dir, "thumb.png")
            subprocess.run([get_ffmpeg_path(), "-y", "-ss", "1", "-i", self.video_path, "-frames:v", "1", "-vf", "scale=240:135", thumb_path], capture_output=True)
            self.project_data['thumbnail_path'] = thumb_path

        ProjectManager.save_project(self.project_data)
        self.statusBar().showMessage("Project Saved!", 3000)

    def _load_project_state(self):
        """Load clips and tracks with rich compositing and transition settings."""
        if not self.project_data:
            return

        for track_data in self.project_data.get('tracks', []):
            name = track_data['name']
            if name not in self.timeline._tracks:
                track_path = track_data.get('path') or ""
                track = Track(
                    track_data['type'],
                    track_data['display_name'],
                    path=track_path,
                    duration=FFmpegHelper.get_video_duration(track_path) if track_path else 0.0
                )
                track.volume = track_data.get('volume', 1.0)
                track.video_scale_x = track_data.get('video_scale_x', 1.0)
                track.video_scale_y = track_data.get('video_scale_y', 1.0)

                # Support rich layout loading with robust fallback
                if 'clips' in track_data:
                    for clip_serialized in track_data['clips']:
                        c = Clip(
                            clip_serialized["source_path"],
                            clip_serialized["timeline_start"],
                            clip_serialized["duration"],
                            source_start=clip_serialized["source_start"],
                            clip_type=clip_serialized["type"],
                            name=clip_serialized["name"]
                        )
                        c.volume = clip_serialized.get("volume", 1.0)
                        c.video_scale_x = clip_serialized.get("video_scale_x", 1.0)
                        c.video_scale_y = clip_serialized.get("video_scale_y", 1.0)
                        c.video_x = clip_serialized.get("video_x", 0.0)
                        c.video_y = clip_serialized.get("video_y", 0.0)
                        c.opacity = clip_serialized.get("opacity", 1.0)
                        c.fade_in = clip_serialized.get("fade_in", 0.0)
                        c.fade_out = clip_serialized.get("fade_out", 0.0)
                        c.text_content = clip_serialized.get("text_content", "")
                        track.clips.append(c)
                else:
                    # Fallback backward compatibility loading
                    for seg in track_data.get('segments', []):
                        start = seg[0]
                        end = seg[1]
                        source_st = seg[2] if len(seg) > 2 else 0.0
                        track.clips.append(Clip(track.path, start, end - start, source_start=source_st, clip_type="video" if track.type == Track.VIDEO else "audio"))

                self.timeline._tracks[name] = track
                if track.type == Track.AUDIO and track.path:
                    self._generate_waveform_for_track(name, track.path, track.duration)

        self.timeline._update_layout()
        self.timeline.update()

    def _go_home(self):
        """Save and return to dashboard."""
        self._save_project_state()
        self.close()
        if self.parent_window:
            self.parent_window.show_dashboard()

    def _import_audio(self, path=None, volume=1.0, track_name=None):
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Import Audio Track", "",
                "Audio Files (*.mp3 *.wav *.aac *.flac *.ogg);;All Files (*.*)"
            )
        if path:
            if not track_name:
                track_name = f"Audio {len(self.timeline._tracks) + 1}"
            duration = FFmpegHelper.get_video_duration(path)

            self._generate_waveform_for_track(track_name, path, duration)

            if track_name not in self.timeline._tracks:
                self.timeline._tracks[track_name] = Track(Track.AUDIO, track_name, path=path, duration=duration)
                self.timeline._update_layout()
                self.timeline.update()

            self._save_project_state()

    def _on_waveform_ready(self, track_name, path, duration, wave_path):
        if track_name in self.timeline._tracks and self.timeline._tracks[track_name].clips:
            self.timeline._tracks[track_name].clips[0].waveform_path = wave_path
        else:
            self.timeline.add_audio_track(track_name, path, duration, wave_path)
        self.timeline.update()

    def _on_thumbnails_ready(self, thumbs, thumb_dir):
        self.thumb_dir = thumb_dir
        if "main_video" in self.timeline._tracks and self.timeline._tracks["main_video"].clips:
            self.timeline._tracks["main_video"].clips[0].thumbnails = thumbs
            self.timeline.update()

    def _generate_waveform_for_track(self, track_name, path, duration):
        thread = WaveformThread(track_name, path, duration)
        thread.waveform_ready.connect(self._on_waveform_ready)
        self._wave_threads.append(thread)
        thread.finished.connect(lambda t=thread: self._wave_threads.remove(t) if t in self._wave_threads else None)
        thread.start()

    def _load_video(self):
        if not self.video_path or not os.path.exists(self.video_path):
            return

        dur = FFmpegHelper.get_video_duration(self.video_path)
        if dur > 0:
            self.duration = dur
            self.timeline.set_duration(self.duration)

        # Setup/Update main video track
        if "main_video" not in self.timeline._tracks:
            vid_track = Track(Track.VIDEO, "Video 1", path=self.video_path, duration=self.duration)
            vid_clip = Clip(self.video_path, 0.0, self.duration, clip_type="video", name="Main Video")

            w, h = FFmpegHelper.get_video_dimensions(self.video_path)
            self._fit_clip_to_canvas(vid_clip, w, h)

            vid_track.clips.append(vid_clip)
            self.timeline._tracks["main_video"] = vid_track
        else:
            vid_track = self.timeline._tracks["main_video"]
            vid_track.path = self.video_path
            vid_track.duration = self.duration
            if not vid_track.clips:
                vid_clip = Clip(self.video_path, 0.0, self.duration, clip_type="video", name="Main Video")

                w, h = FFmpegHelper.get_video_dimensions(self.video_path)
                self._fit_clip_to_canvas(vid_clip, w, h)

                vid_track.clips.append(vid_clip)
            else:
                vid_track.clips[0].source_path = self.video_path
                vid_track.clips[0].duration = self.duration

        # Setup/Update main audio track
        if "main_audio" not in self.timeline._tracks:
            aud_track = Track(Track.AUDIO, "Audio 1", path=self.video_path, duration=self.duration)
            aud_clip = Clip(self.video_path, 0.0, self.duration, clip_type="audio", name="Main Audio")
            aud_track.clips.append(aud_clip)
            self.timeline._tracks["main_audio"] = aud_track
            self._generate_waveform_for_track("main_audio", self.video_path, self.duration)
        else:
            aud_track = self.timeline._tracks["main_audio"]
            aud_track.path = self.video_path
            aud_track.duration = self.duration
            if not aud_track.clips:
                aud_track.clips.append(Clip(self.video_path, 0.0, self.duration, clip_type="audio", name="Main Audio"))
            else:
                aud_track.clips[0].source_path = self.video_path
                aud_track.clips[0].duration = self.duration

        # Link the main video and audio clips
        if self.timeline._tracks["main_video"].clips and self.timeline._tracks["main_audio"].clips:
            self.timeline._tracks["main_video"].clips[0].linked_to = self.timeline._tracks["main_audio"].clips[0]
            self.timeline._tracks["main_audio"].clips[0].linked_to = self.timeline._tracks["main_video"].clips[0]

        self.timeline._tracks["main_video"].linked_to = "main_audio"
        self.timeline._tracks["main_audio"].linked_to = "main_video"

        self.timeline._update_layout()
        self.timeline.update()

        self._thumb_thread = ThumbnailThread(self.video_path)
        self._thumb_thread.thumbnails_ready.connect(self._on_thumbnails_ready)
        self._thumb_thread.start()
        
        
        
    def _toggle_playback(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            self.shuttle_speed = 1.0
            for tp in self.track_players.values():
                tp.player.setPlaybackRate(1.0)
            self.btn_play.setText("⏸  Pause")
            self.playback_monitor.start()
        else:
            self.btn_play.setText("▶  Play")
            self.playback_monitor.stop()
            for tp in self.track_players.values():
                tp.player.pause()

    def _stop_playback(self):
        self.is_playing = False
        self.shuttle_speed = 1.0
        self.btn_play.setText("▶  Play")
        self.playback_monitor.stop()
        self.playhead = 0.0
        self.timeline.set_playhead(0.0)
        self.time_label.setText(f"{self._fmt_time(0)} / {self._fmt_time(self.duration)}")
        for tp in self.track_players.values():
            tp.player.stop()
            tp.active_clip = None
            tp.current_image = None
        self.viewport_widget.update()

    def _check_playback_position(self):
        if not self.is_playing:
            return

        speed = getattr(self, 'shuttle_speed', 1.0)
        self.playhead += 0.033 * speed

        if self.playhead < 0.0:
            self.playhead = 0.0
            self._stop_playback()
            return
        elif self.playhead > self.duration:
            self.playhead = self.duration
            self._stop_playback()
            return

        self.timeline.set_playhead(self.playhead)
        self.time_label.setText(f"{self._fmt_time(self.playhead)} / {self._fmt_time(self.duration)}")

        for t_name, track in self.timeline._tracks.items():
            if t_name not in self.track_players:
                self.track_players[t_name] = TrackPlayer(t_name, self)
                self.track_players[t_name].frame_ready.connect(lambda name: self.viewport_widget.update())
            player_wrap = self.track_players[t_name]

            active_clip = None
            for clip in track.clips:
                if clip.timeline_start <= self.playhead < clip.timeline_end:
                    active_clip = clip
                    break

            if active_clip:
                if player_wrap.active_clip != active_clip:
                    player_wrap.player.setSource(QUrl.fromLocalFile(active_clip.source_path))
                    player_wrap.active_clip = active_clip
                    player_wrap.player.play()

                expected_src = active_clip.source_start + (self.playhead - active_clip.timeline_start)
                actual_src = player_wrap.player.position() / 1000.0

                if abs(actual_src - expected_src) > 0.15:
                    player_wrap.player.setPosition(int(expected_src * 1000))

                if player_wrap.player.playbackState() != QMediaPlayer.PlayingState:
                    player_wrap.player.play()
            else:
                if player_wrap.active_clip is not None:
                    player_wrap.player.stop()
                    player_wrap.active_clip = None
                    player_wrap.current_image = None

        self.viewport_widget.update()

    def _sync_players_to_playhead(self):
        self.time_label.setText(f"{self._fmt_time(self.playhead)} / {self._fmt_time(self.duration)}")
        for t_name, track in self.timeline._tracks.items():
            if t_name not in self.track_players:
                self.track_players[t_name] = TrackPlayer(t_name, self)
                self.track_players[t_name].frame_ready.connect(lambda name: self.viewport_widget.update())
            player_wrap = self.track_players[t_name]

            active_clip = None
            for clip in track.clips:
                if clip.timeline_start <= self.playhead < clip.timeline_end:
                    active_clip = clip
                    break

            if active_clip:
                if player_wrap.active_clip != active_clip:
                    player_wrap.player.setSource(QUrl.fromLocalFile(active_clip.source_path))
                    player_wrap.active_clip = active_clip

                expected_src = active_clip.source_start + (self.playhead - active_clip.timeline_start)
                player_wrap.player.setPosition(int(expected_src * 1000))

                if self.is_playing and player_wrap.player.playbackState() != QMediaPlayer.PlayingState:
                    player_wrap.player.play()
                elif not self.is_playing:
                    player_wrap.player.pause()
            else:
                if player_wrap.active_clip is not None:
                    player_wrap.player.stop()
                    player_wrap.active_clip = None
                    player_wrap.current_image = None
        self.viewport_widget.update()

    def _seek_relative(self, delta):
        self.playhead = max(0.0, min(self.playhead + delta, self.duration))
        self.timeline.set_playhead(self.playhead)
        self._sync_players_to_playhead()

    def _on_timeline_seek(self, pos_sec):
        self.playhead = pos_sec
        self._sync_players_to_playhead()


    def _on_reassign_clip_track(self, index):
        target_track = self.clip_track_combo.currentData()
        src_track = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if target_track and src_track and target_track != src_track:
            if src_track in self.timeline._tracks and target_track in self.timeline._tracks:
                if 0 <= s_idx < len(self.timeline._tracks[src_track].clips):
                    clip = self.timeline._tracks[src_track].clips.pop(s_idx)
                    self.timeline._tracks[target_track].clips.append(clip)
                    self.timeline._tracks[target_track].clips.sort(key=lambda c: c.timeline_start)
                    self.timeline._selected_track = target_track
                    self.timeline._selected_segment = self.timeline._tracks[target_track].clips.index(clip)
                    self.timeline.segments_changed.emit()
                    self.timeline.update()
                    self.viewport_widget.update()
                    self._save_project_state()
                    self.statusBar().showMessage(f"Moved clip to '{self.timeline._tracks[target_track].name}'", 3000)

    def _snap_clip_to_previous(self):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                clip = track.clips[s_idx]
                if s_idx > 0:
                    prev_clip = track.clips[s_idx - 1]
                    clip.timeline_start = prev_clip.timeline_end
                else:
                    clip.timeline_start = 0.0
                track.clips.sort(key=lambda c: c.timeline_start)
                self.timeline.segments_changed.emit()
                self.timeline.update()
                self.viewport_widget.update()
                self._save_project_state()

    def _on_clip_opacity_changed(self, val):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                track.clips[s_idx].opacity = val / 100.0
                self.props_opacity_label.setText(f"Layer Opacity ({val}%):")
                self.viewport_widget.update()
                self._save_project_state()

    def _on_clip_fade_in_changed(self, val):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                sec = val / 10.0
                track.clips[s_idx].fade_in = sec
                self.props_fade_in_label.setText(f"Fade In Transition ({sec:.1f}s):")
                self.timeline.update()
                self.viewport_widget.update()
                self._save_project_state()

    def _on_clip_fade_out_changed(self, val):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                sec = val / 10.0
                track.clips[s_idx].fade_out = sec
                self.props_fade_out_label.setText(f"Fade Out Transition ({sec:.1f}s):")
                self.timeline.update()
                self.viewport_widget.update()
                self._save_project_state()
    def _copy_selected_clip(self):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                self._copied_clip = track.clips[s_idx].clone()
                self.statusBar().showMessage(f"Copied clip '{self._copied_clip.name}' to clipboard", 3000)

    def _paste_copied_clip(self):
        if not hasattr(self, '_copied_clip') or not self._copied_clip:
            self.statusBar().showMessage("Clipboard is empty! Select a clip and click Copy.", 3000)
            return

        target_track_name = self.timeline._selected_track or "main_video"
        if target_track_name not in self.timeline._tracks:
            return

        track = self.timeline._tracks[target_track_name]
        pos = self.timeline.get_playhead()

        pasted = self._copied_clip.clone(new_start=pos)
        track.clips.append(pasted)
        track.clips.sort(key=lambda c: c.timeline_start)

        self.timeline.segments_changed.emit()
        self.timeline._update_layout()
        self.timeline.update()
        self._save_project_state()
        self.statusBar().showMessage(f"Pasted '{pasted.name}' at {pos:.1f}s", 3000)

    def _apply_pip_preset(self, index):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if not t_name or t_name not in self.timeline._tracks or s_idx < 0:
            return

        clip = self.timeline._tracks[t_name].clips[s_idx]
        if clip.type != "video":
            return

        if index == 0:  # Full Screen
            clip.video_scale_x, clip.video_scale_y, clip.video_x, clip.video_y = 1.0, 1.0, 0.0, 0.0
        elif index == 1:  # Top-Left
            clip.video_scale_x, clip.video_scale_y, clip.video_x, clip.video_y = 0.35, 0.35, 0.02, 0.02
        elif index == 2:  # Top-Right
            clip.video_scale_x, clip.video_scale_y, clip.video_x, clip.video_y = 0.35, 0.35, 0.63, 0.02
        elif index == 3:  # Bottom-Left
            clip.video_scale_x, clip.video_scale_y, clip.video_x, clip.video_y = 0.35, 0.35, 0.02, 0.63
        elif index == 4:  # Bottom-Right
            clip.video_scale_x, clip.video_scale_y, clip.video_x, clip.video_y = 0.35, 0.35, 0.63, 0.63
        elif index == 5:  # Center
            clip.video_scale_x, clip.video_scale_y, clip.video_x, clip.video_y = 0.50, 0.50, 0.25, 0.25

        self.props_scale_x_slider.blockSignals(True)
        self.props_scale_x_slider.setValue(int(clip.video_scale_x * 100))
        self.props_scale_x_slider.blockSignals(False)

        self.props_scale_y_slider.blockSignals(True)
        self.props_scale_y_slider.setValue(int(clip.video_scale_y * 100))
        self.props_scale_y_slider.blockSignals(False)

        self.props_x_slider.blockSignals(True)
        self.props_x_slider.setValue(int(clip.video_x * 100))
        self.props_x_slider.blockSignals(False)

        self.props_y_slider.blockSignals(True)
        self.props_y_slider.setValue(int(clip.video_y * 100))
        self.props_y_slider.blockSignals(False)

        self.timeline.update()
        self._save_project_state()

    def _on_clip_x_changed(self, val):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                track.clips[s_idx].video_x = val / 100.0
                self.timeline.update()

    def _on_clip_y_changed(self, val):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                track.clips[s_idx].video_y = val / 100.0
                self.timeline.update()

    def _toggle_track_link(self):
        if self.timeline._selected_track in ("main_video", "main_audio"):
            vid = self.timeline._tracks.get("main_video")
            aud = self.timeline._tracks.get("main_audio")
            if vid and aud:
                if vid.linked_to:
                    vid.linked_to = None
                    aud.linked_to = None
                    self.btn_toggle_link.setText("🔗 Link Audio/Video")
                else:
                    vid.linked_to = "main_audio"
                    aud.linked_to = "main_video"
                    self.btn_toggle_link.setText("🔗 Unlink Audio/Video")

    def _on_track_volume_changed(self, val):
        if self.timeline._selected_track in self.timeline._tracks:
            self.timeline._tracks[self.timeline._selected_track].volume = val / 100.0
            self.timeline.update()

    def _on_clip_scale_x_changed(self, val):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                track.clips[s_idx].video_scale_x = val / 100.0
                self.viewport_widget.update()

    def _on_clip_scale_y_changed(self, val):
        t_name = self.timeline._selected_track
        s_idx = self.timeline._selected_segment
        if t_name and t_name in self.timeline._tracks and s_idx >= 0:
            track = self.timeline._tracks[t_name]
            if 0 <= s_idx < len(track.clips):
                track.clips[s_idx].video_scale_y = val / 100.0
                self.viewport_widget.update()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if path.lower().endswith(('.mp4', '.avi', '.mkv', '.mov', '.webm')):
                self.video_path = path
                self.project_data['video_path'] = path
                self.project_data['name'] = os.path.basename(path)
                if hasattr(self, 'empty_state_label'):
                    self.empty_state_label.hide()
                self._load_video()
                self._save_project_state()
                break

    def _split_at_playhead(self):
        pos = self.timeline.get_playhead()
        target_group = self.timeline._selected_group
        target_track = self.timeline._selected_track
        target_linked = None
        if target_track and target_track in self.timeline._tracks:
            target_linked = self.timeline._tracks[target_track].linked_to

        for t_name, track in self.timeline._tracks.items():
            new_clips = []
            for clip in track.clips:
                if clip.timeline_start < pos < clip.timeline_end:
                    should_split = False
                    if target_group:
                        if getattr(clip, 'group_id', None) == target_group:
                            should_split = True
                    elif target_track:
                        if t_name == target_track or t_name == target_linked:
                            should_split = True
                    else:
                        should_split = True

                    if should_split:
                        offset = pos - clip.timeline_start
                        left = clip.clone()
                        left.duration = offset

                        right = clip.clone(new_start=pos)
                        right.duration = clip.duration - offset
                        right.source_start = clip.source_start + offset

                        new_clips.extend([left, right])
                    else:
                        new_clips.append(clip)
                else:
                    new_clips.append(clip)
            track.clips = new_clips

        self.timeline.segments_changed.emit()
        self.timeline.update()
        self._save_project_state()
    def _on_track_selected(self, track_name):
        if track_name and track_name in self.timeline._tracks:
            track = self.timeline._tracks[track_name]
            seg_idx = self.timeline._selected_segment

            # Populate Track Reassign Dropdown with compatible channels
            self.clip_track_combo.blockSignals(True)
            self.clip_track_combo.clear()
            for t_k, t_v in self.timeline._tracks.items():
                if t_v.type == track.type:
                    self.clip_track_combo.addItem(t_v.name, userData=t_k)
            idx = self.clip_track_combo.findData(track_name)
            if idx >= 0:
                self.clip_track_combo.setCurrentIndex(idx)
            self.clip_track_combo.blockSignals(False)

            if seg_idx >= 0 and seg_idx < len(track.clips):
                clip = track.clips[seg_idx]
                self.props_track_label.setText(f"Editing Clip: '{clip.name}' ({track.name})")
                self.clip_track_combo.setEnabled(True)
                self.btn_snap_prev.setEnabled(True)

                self.props_opacity_slider.blockSignals(True)
                self.props_opacity_slider.setValue(int(getattr(clip, 'opacity', 1.0) * 100))
                self.props_opacity_slider.blockSignals(False)
                self.props_opacity_slider.setEnabled(True)
                self.props_opacity_label.setText(f"Layer Opacity ({int(getattr(clip, 'opacity', 1.0)*100)}%):")

                self.props_fade_in_slider.blockSignals(True)
                self.props_fade_in_slider.setValue(int(getattr(clip, 'fade_in', 0.0) * 10))
                self.props_fade_in_slider.blockSignals(False)
                self.props_fade_in_slider.setEnabled(True)
                self.props_fade_in_label.setText(f"Fade In Transition ({getattr(clip, 'fade_in', 0.0):.1f}s):")

                self.props_fade_out_slider.blockSignals(True)
                self.props_fade_out_slider.setValue(int(getattr(clip, 'fade_out', 0.0) * 10))
                self.props_fade_out_slider.blockSignals(False)
                self.props_fade_out_slider.setEnabled(True)
                self.props_fade_out_label.setText(f"Fade Out Transition ({getattr(clip, 'fade_out', 0.0):.1f}s):")

                self.props_scale_x_slider.blockSignals(True)
                self.props_scale_x_slider.setValue(int(clip.video_scale_x * 100))
                self.props_scale_x_slider.blockSignals(False)
                self.props_scale_x_slider.setEnabled(clip.type in ("video", "drawing", "text"))

                self.props_scale_y_slider.blockSignals(True)
                self.props_scale_y_slider.setValue(int(clip.video_scale_y * 100))
                self.props_scale_y_slider.blockSignals(False)
                self.props_scale_y_slider.setEnabled(clip.type in ("video", "drawing", "text"))

                self.props_x_slider.blockSignals(True)
                self.props_x_slider.setValue(int(clip.video_x * 100))
                self.props_x_slider.blockSignals(False)
                self.props_x_slider.setEnabled(clip.type in ("video", "drawing", "text"))

                self.props_y_slider.blockSignals(True)
                self.props_y_slider.setValue(int(clip.video_y * 100))
                self.props_y_slider.blockSignals(False)
                self.props_y_slider.setEnabled(clip.type in ("video", "drawing", "text"))

                self.pip_preset_combo.setEnabled(clip.type in ("video", "drawing", "text"))

                self.props_vol_slider.blockSignals(True)
                self.props_vol_slider.setValue(int(clip.volume * 100))
                self.props_vol_slider.blockSignals(False)
                self.props_vol_slider.setEnabled(clip.type in ("audio", "video"))
            else:
                self.props_track_label.setText(f"Editing Track: {track.name}")
                self.clip_track_combo.setEnabled(False)
                self.btn_snap_prev.setEnabled(False)
                self.props_opacity_slider.setEnabled(False)
                self.props_fade_in_slider.setEnabled(False)
                self.props_fade_out_slider.setEnabled(False)

                self.props_vol_slider.blockSignals(True)
                self.props_vol_slider.setValue(int(track.volume * 100))
                self.props_vol_slider.blockSignals(False)
                self.props_vol_slider.setEnabled(track.type == Track.AUDIO)

                self.props_scale_x_slider.blockSignals(True)
                self.props_scale_x_slider.setValue(int(track.video_scale_x * 100))
                self.props_scale_x_slider.blockSignals(False)
                self.props_scale_x_slider.setEnabled(track.type == Track.VIDEO)

                self.props_scale_y_slider.blockSignals(True)
                self.props_scale_y_slider.setValue(int(track.video_scale_y * 100))
                self.props_scale_y_slider.blockSignals(False)
                self.props_scale_y_slider.setEnabled(track.type == Track.VIDEO)

                self.props_x_slider.setEnabled(False)
                self.props_y_slider.setEnabled(False)
                self.pip_preset_combo.setEnabled(False)
    def _delete_segment(self):
        target_group = self.timeline._selected_group
        if target_group:
            for t_name, track in self.timeline._tracks.items():
                track.clips = [c for c in track.clips if getattr(c, 'group_id', None) != target_group]
        else:
            t_name = self.timeline._selected_track
            s_idx = self.timeline._selected_segment
            if t_name and s_idx >= 0 and t_name in self.timeline._tracks:
                track = self.timeline._tracks[t_name]
                if 0 <= s_idx < len(track.clips):
                    del track.clips[s_idx]

                linked = track.linked_to
                if linked and linked in self.timeline._tracks:
                    l_track = self.timeline._tracks[linked]
                    if 0 <= s_idx < len(l_track.clips):
                        del l_track.clips[s_idx]

        self.timeline._selected_group = None
        self.timeline._selected_segment = -1
        self.timeline.segments_changed.emit()
        self.timeline.update()
        self._save_project_state()

    def _ripple_delete_segment(self):
        target_group = self.timeline._selected_group
        deleted_dur = 0
        deleted_start = -1

        if target_group:
            for t_name, track in self.timeline._tracks.items():
                for c in track.clips:
                    if getattr(c, 'group_id', None) == target_group:
                        deleted_dur = max(deleted_dur, c.duration)
                        deleted_start = c.timeline_start if deleted_start == -1 else min(deleted_start, c.timeline_start)
                        break

            for t_name, track in self.timeline._tracks.items():
                new_clips = []
                for c in track.clips:
                    if getattr(c, 'group_id', None) != target_group:
                        if c.timeline_start >= deleted_start:
                            c.timeline_start = max(0.0, c.timeline_start - deleted_dur)
                        new_clips.append(c)
                track.clips = new_clips
        else:
            t_name = self.timeline._selected_track
            s_idx = self.timeline._selected_segment
            if t_name and s_idx >= 0 and t_name in self.timeline._tracks:
                track = self.timeline._tracks[t_name]
                if 0 <= s_idx < len(track.clips):
                    deleted_start = track.clips[s_idx].timeline_start
                    deleted_dur = track.clips[s_idx].duration
                    del track.clips[s_idx]

                linked = track.linked_to
                if linked and linked in self.timeline._tracks:
                    l_track = self.timeline._tracks[linked]
                    if 0 <= s_idx < len(l_track.clips):
                        if deleted_start == -1:
                            deleted_start = l_track.clips[s_idx].timeline_start
                        deleted_dur = max(deleted_dur, l_track.clips[s_idx].duration)
                        del l_track.clips[s_idx]

            if deleted_start != -1:
                for t_name, track in self.timeline._tracks.items():
                    for c in track.clips:
                        if c.timeline_start >= deleted_start:
                            c.timeline_start = max(0.0, c.timeline_start - deleted_dur)

        self.timeline._selected_group = None
        self.timeline._selected_segment = -1
        self.timeline.segments_changed.emit()
        self.timeline.update()
        self._save_project_state()

    def _shuttle_reverse(self):
        if not hasattr(self, 'shuttle_speed'):
            self.shuttle_speed = -1.0
        else:
            if self.shuttle_speed > 0:
                self.shuttle_speed = -1.0
            else:
                self.shuttle_speed = max(-4.0, self.shuttle_speed * 2.0)

        for tp in self.track_players.values():
            tp.player.setPlaybackRate(abs(self.shuttle_speed))
        if not self.is_playing:
            self._toggle_playback()

    def _shuttle_forward(self):
        if not hasattr(self, 'shuttle_speed'):
            self.shuttle_speed = 1.0
        else:
            if self.shuttle_speed < 0:
                self.shuttle_speed = 1.0
            else:
                self.shuttle_speed = min(4.0, self.shuttle_speed * 2.0)

        for tp in self.track_players.values():
            tp.player.setPlaybackRate(self.shuttle_speed)
        if not self.is_playing:
            self._toggle_playback()

    def _execute_command(self, cmd):
        cmd.execute()
        self.undo_stack.append(cmd)
        self.redo_stack.clear()

    def _undo(self):
        if self.undo_stack:
            cmd = self.undo_stack.pop()
            cmd.undo()
            self.redo_stack.append(cmd)
            self._update_info()
            self._save_project_state()

    def _redo(self):
        if self.redo_stack:
            cmd = self.redo_stack.pop()
            cmd.redo()
            self.undo_stack.append(cmd)
            self._update_info()
            self._save_project_state()

    def _reset_segments(self):
        self.timeline.reset_segments()
        self._update_info()
        self._save_project_state()

    def _update_info(self):
        segments = self.timeline.get_segments()
        total = sum(e - s for s, e in segments)
        self.info_label.setText(f"Total: {self.duration:.1f}s  |  Kept: {total:.1f}s  |  Segments: {len(segments)}")

    def _render_overlay_clip_to_png(self, clip, width, height, output_path):
        """Generates a high-fidelity transparent PNG of text or drawings to preserve visual alignment on export."""
        image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        image.fill(Qt.transparent)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        canvas_rect = QRectF(0, 0, width, height)

        if clip.type == "drawing":
            scale_x = getattr(clip, 'video_scale_x', 1.0)
            scale_y = getattr(clip, 'video_scale_y', 1.0)
            vw = canvas_rect.width() * scale_x
            vh = canvas_rect.height() * scale_y
            vx = canvas_rect.x() + clip.video_x * canvas_rect.width()
            vy = canvas_rect.y() + clip.video_y * canvas_rect.height()
            clip_rect = QRectF(vx, vy, vw, vh)

            painter.save()
            painter.translate(clip_rect.x(), clip_rect.y())
            if clip_rect.width() > 0 and clip_rect.height() > 0:
                painter.scale(clip_rect.width() / 1000.0, clip_rect.height() / 1000.0)

            for p, col, w in getattr(clip, 'drawing_paths', []):
                pen = QPen(col, w, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                pen.setCosmetic(True)
                painter.setPen(pen)
                painter.drawPath(p)

            painter.restore()

        elif clip.type == "text":
            text_str = getattr(clip, 'text_content', clip.name)
            if text_str:
                scale_x = getattr(clip, 'video_scale_x', 1.0)
                scale_y = getattr(clip, 'video_scale_y', 1.0)
                font_family = getattr(clip, 'text_font_family', 'Segoe UI')
                font_size = getattr(clip, 'text_font_size', 28)

                tc_val = getattr(clip, 'text_color', QColor(255, 255, 255))
                bg_val = getattr(clip, 'text_bg_color', QColor(0, 0, 0, 150))
                text_color = QColor(*tc_val) if isinstance(tc_val, (list, tuple)) else (tc_val if isinstance(tc_val, QColor) else QColor(255, 255, 255))
                bg_color = QColor(*bg_val) if isinstance(bg_val, (list, tuple)) else (bg_val if isinstance(bg_val, QColor) else QColor(0, 0, 0, 150))

                font = QFont(font_family, font_size, QFont.Bold)
                painter.setFont(font)

                fm = painter.fontMetrics()
                text_rect = fm.boundingRect(text_str)
                padding = 12

                base_w = text_rect.width() + padding * 2
                base_h = text_rect.height() + padding * 2
                scaled_w = base_w * scale_x
                scaled_h = base_h * scale_y

                cx = int(canvas_rect.x() + clip.video_x * canvas_rect.width()) if clip.video_x > 0 else int(canvas_rect.x() + (canvas_rect.width() - scaled_w) / 2)
                cy = int(canvas_rect.y() + clip.video_y * canvas_rect.height()) if clip.video_y > 0 else int(canvas_rect.y() + (canvas_rect.height() - scaled_h) / 2)

                painter.save()
                painter.translate(cx, cy)
                painter.scale(scale_x, scale_y)

                draw_box = QRectF(0, 0, base_w, base_h)
                if bg_color.alpha() > 0:
                    painter.fillRect(draw_box, bg_color)
                    painter.setPen(QPen(QColor(CLR_BORDER), 1))
                    painter.drawRect(draw_box)

                painter.setPen(text_color)
                painter.drawText(draw_box, Qt.AlignCenter, text_str)
                painter.restore()

        painter.end()
        image.save(output_path)

    def _export(self):
        """Prepares timeline assets, maps ratios, and renders the multi-channel timeline."""
        has_video_clips = False
        for track in self.timeline._tracks.values():
            if track.clips:
                has_video_clips = True
                break

        if not has_video_clips:
            QMessageBox.warning(self, "No Segments", "There are no segments on the timeline to export.")
            return

        dialog = ExportDialog(self, duration=self.duration, segment_count=len(self.timeline.get_segments()))
        if dialog.exec() != QDialog.Accepted:
            return

        settings = dialog.get_settings()

        # Map project aspect ratios to baseline dimensions
        ratio_str = self.project_data.get('aspect_ratio', '16:9')
        if ratio_str == "9:16":
            width, height = 1080, 1920
        elif ratio_str == "1:1":
            width, height = 1080, 1080
        elif ratio_str == "4:3":
            width, height = 1440, 1080
        elif ratio_str == "21:9":
            width, height = 2560, 1080
        else:
            width, height = 1920, 1080

        # Scale based on target export request while maintaining project aspect ratio
        res_str = settings.get("resolution")
        if res_str and "x" in res_str:
            try:
                req_w, req_h = map(int, res_str.split('x'))
                proj_aspect = width / height
                if width >= height:
                    width = req_w
                    height = int(round(width / proj_aspect)) // 2 * 2
                else:
                    height = req_h
                    width = int(round(height * proj_aspect)) // 2 * 2
            except Exception:
                pass

        # Create clean scratch folder
        temp_dir = tempfile.mkdtemp(prefix="timeline_export_")

        # Copy track model structure to keep export isolated and safe
        export_tracks = {}
        for name, track in self.timeline._tracks.items():
            new_track = Track(track.type, track.name, track.path, track.duration)
            new_track.volume = track.volume
            new_track.clips = [clip.clone() for clip in track.clips]
            export_tracks[name] = new_track

        # Pre-render drawings & texts into high-res PNG overlays
        for name, track in export_tracks.items():
            for clip in track.clips:
                if clip.type in ("text", "drawing"):
                    png_path = os.path.join(temp_dir, f"overlay_{uuid.uuid4().hex}.png")
                    self._render_overlay_clip_to_png(clip, width, height, png_path)
                    clip.source_path = png_path
                    # Promote overlay into static full-frame video stream for FFmpeg overlay composition
                    clip.video_scale_x = 1.0
                    clip.video_scale_y = 1.0
                    clip.video_x = 0.0
                    clip.video_y = 0.0

        fps_val = 30
        if settings.get("fps") and settings["fps"].isdigit():
            fps_val = int(settings["fps"])

        # Compile rendering pipeline command
        cmd = FFmpegHelper.build_timeline_export_command(
            tracks=export_tracks,
            total_duration=self.duration,
            output_path=settings["output_path"],
            quality=settings["quality"],
            width=width,
            height=height,
            fps=fps_val
        )

        self.progress_widget.setVisible(True)
        self.btn_export.setEnabled(False)

        self._export_worker = ExportWorker(cmd, temp_dir, settings["output_path"], total_duration=self.duration)
        self._export_worker.progress.connect(
            lambda msg, pct: (self.progress_label.setText(msg), self.progress_bar.setValue(pct))
        )
        self._export_worker.finished_signal.connect(
            lambda success, msg: self._on_export_finished(success, msg)
        )
        self._export_worker.start()

    def _on_export_finished(self, success, message):
        self.progress_widget.setVisible(False)
        self.btn_export.setEnabled(True)
        if success:
            logger.info(f"Export completed successfully: {message}")
            dialog = MediaCompleteDialog(
                title="Export Complete",
                heading="Video Exported Successfully!",
                file_path=message,
                on_open_editor=None,
                parent=self
            )
            dialog.exec()
        else:
            logger.error(f"Export failed error dialog shown to user:\n{message}")
            QMessageBox.critical(self, "Export Failed", f"Error:\n{message}")

    def _record_new_media(self):
        self.recording_window = MainWindow(editor_ref=self)
        self.recording_window.show()

    def _add_media_item_to_ui(self, path):
        thumb_dir = tempfile.mkdtemp()
        thumb_path = os.path.join(thumb_dir, "thumb.png")
        subprocess.run([get_ffmpeg_path(), "-y", "-ss", "1", "-i", path, "-frames:v", "1", "-vf", "scale=160:90", thumb_path], capture_output=True)

        item = QListWidgetItem()
        item.setIcon(QIcon(thumb_path))
        item.setText(os.path.basename(path))
        item.setData(Qt.UserRole, path)
        item.setToolTip(path)
        self.media_list.addItem(item)

    def add_media_to_library(self, file_path):
        if not file_path or not os.path.exists(file_path):
            return

        media_dir = os.path.join(os.path.expanduser("~"), "Documents", "ParisNeo_VideoProjects", self.project_id, "media")
        os.makedirs(media_dir, exist_ok=True)

        dest_path = os.path.join(media_dir, os.path.basename(file_path))
        try:
            if os.path.normpath(file_path) != os.path.normpath(dest_path):
                shutil.copy2(file_path, dest_path)
        except Exception as e:
            QMessageBox.warning(self, "Media Error", f"Failed to copy media: {e}")
            return

        if 'media_library' not in self.project_data:
            self.project_data['media_library'] = []

        if dest_path not in self.project_data['media_library']:
            self.project_data['media_library'].append(dest_path)
            self._save_project_state()
            self._add_media_item_to_ui(dest_path)

    def _toggle_media_library(self):
        """Show/hide the full vertical media library panel to maximize preview workspace."""
        if self.media_panel.isVisible():
            self.media_panel.hide()
            self.media_toggle_btn.setText("▶ Media")
        else:
            self.media_panel.show()
            self.media_toggle_btn.setText("◀ Media")
            self.media_panel.updateGeometry()

    def _add_media_item_to_ui(self, path):
        """Deprecated: Delegated to MediaLibraryPanel."""
        self.media_panel.add_media_item(path)

    def _show_media_context_menu(self, pos):
        item = self.media_list.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        action_rename = menu.addAction("✏️ Rename")
        action_delete = menu.addAction("🗑️ Delete")

        action = menu.exec(self.media_list.mapToGlobal(pos))

        if action == action_rename:
            self._rename_media_item(item)
        elif action == action_delete:
            self._delete_media_item(item)

    def _rename_media_item(self, item):
        old_path = item.data(Qt.UserRole)
        old_name = os.path.basename(old_path)

        new_name, ok = QInputDialog.getText(self, "Rename Media", "New name:", text=old_name)
        if ok and new_name and new_name != old_name:
            media_dir = os.path.dirname(old_path)
            new_path = os.path.join(media_dir, new_name)

            try:
                os.rename(old_path, new_path)

                if old_path in self.project_data['media_library']:
                    idx = self.project_data['media_library'].index(old_path)
                    self.project_data['media_library'][idx] = new_path

                item.setData(Qt.UserRole, new_path)
                item.setText(new_name)
                item.setToolTip(new_path)
                self._save_project_state()

            except OSError as e:
                QMessageBox.warning(self, "Rename Error", f"Failed to rename file: {e}")

    def _delete_media_item(self, item):
        path = item.data(Qt.UserRole)
        reply = QMessageBox.question(self, "Delete Media", f"Are you sure you want to delete {os.path.basename(path)} permanently?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            try:
                os.remove(path)
                if path in self.project_data['media_library']:
                    self.project_data['media_library'].remove(path)

                self.media_list.takeItem(self.media_list.row(item))
                self._save_project_state()
            except OSError as e:
                QMessageBox.warning(self, "Delete Error", f"Failed to delete file: {e}")

    def _on_media_selection_changed(self):
        selected = self.media_list.currentItem()
        self.btn_add_to_timeline.setEnabled(selected is not None)

    def _add_selected_media_to_timeline(self):
        selected = self.media_list.currentItem()
        if selected:
            path = selected.data(Qt.UserRole)
            if path:
                self._add_media_to_timeline_pos(path)

    def _on_media_dropped_to_timeline(self, path):
        self._add_media_to_timeline_pos(path)

    def _add_media_to_timeline_pos(self, path, target_track_name=None, start_time=None):
        """Append media sequentially or place at playhead on selected channel."""
        if not path or not os.path.exists(path):
            return

        if hasattr(self, 'empty_state_label'):
            self.empty_state_label.hide()

        dur = FFmpegHelper.get_video_duration(path)
        if dur <= 0:
            dur = 5.0

        pos = self.timeline.get_playhead() if start_time is None else start_time

        if not self.video_path:
            self.video_path = path
            self.project_data['video_path'] = path
            self.project_data['name'] = os.path.basename(path)

        selected_tr = target_track_name or self.timeline._selected_track

        if selected_tr and selected_tr in self.timeline._tracks and self.timeline._tracks[selected_tr].type == Track.VIDEO:
            v_track_name = selected_tr
        else:
            v_track_name = "main_video"

        if v_track_name not in self.timeline._tracks:
            self.timeline._tracks[v_track_name] = Track(Track.VIDEO, "Video 1", path=path, duration=dur)

        thumbs, _ = FFmpegHelper.generate_thumbnails(path, count=10, thumb_width=120)

        clip_name = os.path.basename(path)
        grp_id = str(uuid.uuid4())

        v_clip = Clip(path, pos, dur, clip_type="video", name=clip_name)
        v_clip.thumbnails = thumbs
        v_clip.group_id = grp_id

        w, h = FFmpegHelper.get_video_dimensions(path)
        self._fit_clip_to_canvas(v_clip, w, h)

        self.timeline._tracks[v_track_name].clips.append(v_clip)
        self.timeline._tracks[v_track_name].clips.sort(key=lambda c: c.timeline_start)

        a_track_name = self.timeline._tracks[v_track_name].linked_to
        if not a_track_name:
            a_track_name = f"Audio {len([t for t in self.timeline._tracks.values() if t.type == Track.AUDIO]) + 1}"
            self.timeline._tracks[a_track_name] = Track(Track.AUDIO, a_track_name, path=path, duration=dur)
            self.timeline._tracks[v_track_name].linked_to = a_track_name
            self.timeline._tracks[a_track_name].linked_to = v_track_name

        a_clip = Clip(path, pos, dur, clip_type="audio", name=f"Audio - {clip_name}")
        a_clip.group_id = grp_id
        self.timeline._tracks[a_track_name].clips.append(a_clip)
        self.timeline._tracks[a_track_name].clips.sort(key=lambda c: c.timeline_start)

        v_clip.linked_to = a_clip
        a_clip.linked_to = v_clip

        self._generate_waveform_for_track(a_track_name, path, dur)

        max_end = max([c.timeline_end for t in self.timeline._tracks.values() for c in t.clips] + [self.duration])
        self.duration = max_end
        self.timeline.set_duration(self.duration)

        self.timeline._update_layout()
        self.timeline.update()
        self._update_info()
        self._save_project_state()
        self.statusBar().showMessage(f"Added '{clip_name}' to timeline at {pos:.1f}s", 3000)

    def _import_media_to_library(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Import Media to Library",
            os.path.join(os.path.expanduser("~"), "Videos"),
            "Media Files (*.mp4 *.avi *.mkv *.mov *.webm *.mp3 *.wav *.aac *.png *.jpg *.jpeg *.webp);;Image Files (*.png *.jpg *.jpeg *.webp);;All Files (*.*)"
        )
        if paths:
            for path in paths:
                self.add_media_to_library(path)

    def _on_media_dropped_to_timeline(self, path):
        self.video_path = path
        self.project_data['video_path'] = path
        self.project_data['name'] = os.path.basename(path)
        if hasattr(self, 'empty_state_label'):
            self.empty_state_label.hide()
        if "main_video" in self.timeline._tracks:
            del self.timeline._tracks["main_video"]
        if "main_audio" in self.timeline._tracks:
            del self.timeline._tracks["main_audio"]
        self._load_video()
        self._save_project_state()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_playback)
        QShortcut(QKeySequence("J"), self, self._shuttle_reverse)
        QShortcut(QKeySequence("K"), self, self._stop_playback)
        QShortcut(QKeySequence("L"), self, self._shuttle_forward)
        QShortcut(QKeySequence("S"), self, self._split_at_playhead)
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self._seek_relative(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._seek_relative(1))
        QShortcut(QKeySequence(Qt.Key_Delete), self, self._delete_segment)
        QShortcut(QKeySequence("Shift+Delete"), self, self._ripple_delete_segment)
        QShortcut(QKeySequence.StandardKey.Undo, self, self._undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, self._redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)
        QShortcut(QKeySequence("Ctrl+C"), self, self._copy_selected_clip)
        QShortcut(QKeySequence("Ctrl+V"), self, self._paste_copied_clip)
        QShortcut(QKeySequence("Ctrl+="), self, lambda: self.timeline.zoom_in())
        QShortcut(QKeySequence("Ctrl+-"), self, lambda: self.timeline.zoom_out())
        QShortcut(QKeySequence("Ctrl+0"), self, lambda: self.timeline.zoom_fit())

    @staticmethod
    def _fmt_time(seconds):
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{
                background: {color}; color: white; border: none;
                padding: 6px 14px; border-radius: 4px; font-weight: bold; font-size: 12px;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background: #333; color: #666; }}
        """

    def _stylesheet(self):
        return f"""
            QMainWindow {{ background: {CLR_BG}; }}
            QWidget {{ color: {CLR_TEXT}; font-family: 'Segoe UI', sans-serif; font-size: 12px; }}
            QGroupBox {{ border: 1px solid {CLR_BORDER}; border-radius: 8px; margin-top: 12px; padding-top: 16px; font-weight: bold; color: {CLR_TEXT_DIM}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; }}
            QPushButton {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; border-radius: 4px;
                padding: 6px 12px; font-size: 12px;
            }}
            QPushButton:hover {{ background: {CLR_BORDER}; }}
            QLabel {{ color: {CLR_TEXT}; }}
            QSlider::groove:horizontal {{
                background: {CLR_PANEL_LIGHT}; height: 4px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {CLR_ACCENT}; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }}
            QSlider::groove:vertical {{
                background: {CLR_PANEL_LIGHT}; width: 4px; border-radius: 2px;
            }}
            QSlider::handle:vertical {{
                background: {CLR_ACCENT}; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }}
        """

    def load_new_video(self, video_path):
        self.video_path = video_path
        self.setWindowTitle(f"Editor — {os.path.basename(video_path)}")
        self.duration = FFmpegHelper.get_video_duration(video_path)
        self.timeline.set_duration(self.duration)
        self._update_info()
        self._load_video()

    def closeEvent(self, event):
        if self.thumb_dir and os.path.isdir(self.thumb_dir):
            shutil.rmtree(self.thumb_dir, ignore_errors=True)

        if hasattr(self, '_wave_threads'):
            for t in self._wave_threads:
                if t.isRunning():
                    t.wait(2000)

        if hasattr(self, 'track_players'):
            for tp in self.track_players.values():
                tp.player.stop()
        super().closeEvent(event)

# ═══════════════════════════════════════════════════════════════
#  MEDIA LIBRARY DOCK PANEL
# ═══════════════════════════════════════════════════════════════

class MediaLibraryPanel(QWidget):
    """Premiere-style collapsible media dock with folders, search, and grid."""
    item_selection_changed = Signal()

    def __init__(self, editor_ref, parent=None):
        super().__init__(parent)
        self.editor = editor_ref
        self.setMinimumWidth(180)
        self.setMaximumWidth(500)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # Search Bar
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍 Search media...")
        self.search_edit.setStyleSheet(f"""
            QLineEdit {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; padding: 6px; border-radius: 4px;
            }}
        """)
        self.search_edit.textChanged.connect(self._filter_items)
        layout.addWidget(self.search_edit)

        # Folder Dropdown
        self.folder_combo = QComboBox()
        self.folder_combo.setStyleSheet(f"""
            QComboBox {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; padding: 4px; border-radius: 4px;
            }}
        """)
        self.folder_combo.addItem("📁 All Media")
        self.folder_combo.currentIndexChanged.connect(self._filter_items)
        layout.addWidget(self.folder_combo)

        # Action Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(4)

        self.btn_add_media = QPushButton("➕")
        self.btn_add_media.setToolTip("Add Media")
        self.btn_add_media.setStyleSheet(self.editor._btn_style(CLR_ACCENT_2))
        self.btn_add_media.clicked.connect(self.editor._import_media_to_library)
        btn_layout.addWidget(self.btn_add_media)

        self.btn_add_folder = QPushButton("📂")
        self.btn_add_folder.setToolTip("New Folder")
        self.btn_add_folder.setStyleSheet(self.editor._btn_style(CLR_PANEL_LIGHT))
        self.btn_add_folder.clicked.connect(self._create_folder)
        btn_layout.addWidget(self.btn_add_folder)

        self.btn_add_to_timeline = QPushButton("➡️")
        self.btn_add_to_timeline.setToolTip("Add to Timeline")
        self.btn_add_to_timeline.setStyleSheet(self.editor._btn_style(CLR_ACCENT))
        self.btn_add_to_timeline.setEnabled(False)
        self.btn_add_to_timeline.clicked.connect(self.editor._add_selected_media_to_timeline)
        btn_layout.addWidget(self.btn_add_to_timeline)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Thumbnail Grid
        self.media_list = QListWidget()
        self.media_list.setStyleSheet(f"""
            QListWidget {{
                background: {CLR_BG}; border: 1px solid {CLR_BORDER}; border-radius: 4px;
                outline: none;
            }}
            QListWidget::item {{
                background: {CLR_PANEL_LIGHT}; border-radius: 4px; margin: 4px;
            }}
            QListWidget::item:selected {{
                background: {CLR_ACCENT};
            }}
        """)
        self.media_list.setDragEnabled(True)
        self.media_list.setViewMode(QListWidget.IconMode)
        self.media_list.setIconSize(QSize(120, 68))
        self.media_list.setResizeMode(QListWidget.Adjust)
        self.media_list.setMovement(QListWidget.Static)
        self.media_list.setSpacing(4)
        self.media_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.media_list.customContextMenuRequested.connect(self.editor._show_media_context_menu)
        self.media_list.itemSelectionChanged.connect(self.item_selection_changed.emit)
        layout.addWidget(self.media_list)

    def add_media_item(self, path, folder="📁 All Media"):
        thumb_dir = tempfile.mkdtemp()
        thumb_path = os.path.join(thumb_dir, "thumb.png")
        subprocess.run([get_ffmpeg_path(), "-y", "-ss", "1", "-i", path, "-frames:v", "1", "-vf", "scale=120:68", thumb_path], capture_output=True)

        item = QListWidgetItem()
        item.setIcon(QIcon(thumb_path))
        item.setText(os.path.basename(path))
        item.setData(Qt.UserRole, path)
        item.setData(Qt.UserRole + 1, folder)
        item.setToolTip(path)
        self.media_list.addItem(item)

    def _create_folder(self):
        folder_name, ok = QInputDialog.getText(self, "New Folder", "Folder name:")
        if ok and folder_name:
            folder_label = f"📁 {folder_name}"
            if self.folder_combo.findText(folder_label) == -1:
                self.folder_combo.addItem(folder_label)
                self.folder_combo.setCurrentText(folder_label)

    def _filter_items(self):
        search_text = self.search_edit.text().lower()
        folder_filter = self.folder_combo.currentText()

        for i in range(self.media_list.count()):
            item = self.media_list.item(i)
            path = item.data(Qt.UserRole)
            item_folder = item.data(Qt.UserRole + 1)

            matches_search = search_text in os.path.basename(path).lower()
            matches_folder = (folder_filter == "📁 All Media" or item_folder == folder_filter)

            item.setHidden(not (matches_search and matches_folder))

    def updateGeometry(self):
        super().updateGeometry()

    def _setup_shortcuts(self):
        QShortcut(QKeySequence(Qt.Key_Space), self, self._toggle_playback)
        QShortcut(QKeySequence("S"), self, self._split_at_playhead)
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self._seek_relative(-1))
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self._seek_relative(1))
        QShortcut(QKeySequence(Qt.Key_Delete), self, self._delete_segment)
        QShortcut(QKeySequence.StandardKey.Undo, self, self._undo)
        QShortcut(QKeySequence.StandardKey.Redo, self, self._redo)
        QShortcut(QKeySequence("Ctrl+Y"), self, self._redo)

    @staticmethod
    def _fmt_time(seconds):
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{
                background: {color}; color: white; border: none;
                padding: 6px 14px; border-radius: 4px; font-weight: bold; font-size: 12px;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background: #333; color: #666; }}
        """

    def _stylesheet(self):
        return f"""
            QMainWindow {{ background: {CLR_BG}; }}
            QWidget {{ color: {CLR_TEXT}; font-family: 'Segoe UI', sans-serif; }}
            QGroupBox {{ border: 1px solid {CLR_BORDER}; border-radius: 8px; margin-top: 12px; padding-top: 16px; font-weight: bold; color: {CLR_TEXT_DIM}; }}
            QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 4px; }}
            QPushButton {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; border-radius: 4px;
                padding: 6px 12px; font-size: 12px;
            }}
            QPushButton:hover {{ background: {CLR_BORDER}; }}
            QLabel {{ color: {CLR_TEXT}; }}
            QSlider::groove:horizontal {{
                background: {CLR_PANEL_LIGHT}; height: 4px; border-radius: 2px;
            }}
            QSlider::handle:horizontal {{
                background: {CLR_ACCENT}; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }}
            QSlider::groove:vertical {{
                background: {CLR_PANEL_LIGHT}; width: 4px; border-radius: 2px;
            }}
            QSlider::handle:vertical {{
                background: {CLR_ACCENT}; width: 12px; height: 12px;
                margin: -4px 0; border-radius: 6px;
            }}
        """

    def load_new_video(self, video_path):
        self.video_path = video_path
        self.setWindowTitle(f"Editor — {os.path.basename(video_path)}")
        self.duration = FFmpegHelper.get_video_duration(video_path)
        self.timeline.set_duration(self.duration)
        self._update_info()
        self._load_video()

    def closeEvent(self, event):
        if self.thumb_dir and os.path.isdir(self.thumb_dir):
            shutil.rmtree(self.thumb_dir, ignore_errors=True)
        self.media_player.stop()
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════
#  THUMBNAIL GENERATION THREAD
# ═══════════════════════════════════════════════════════════════

class ThumbnailThread(QThread):
    thumbnails_ready = Signal(list, str)  # list of (timestamp, path), dir

    def __init__(self, video_path, count=30):
        super().__init__()
        self.video_path = video_path
        self.count = count

    def run(self):
        thumbs, thumb_dir = FFmpegHelper.generate_thumbnails(
            self.video_path, count=self.count, thumb_width=120
        )
        self.thumbnails_ready.emit(thumbs, thumb_dir)


# ═══════════════════════════════════════════════════════════════
#  RECORDING THREAD
# ═══════════════════════════════════════════════════════════════

class FFmpegInstallWorker(QThread):
    """Asynchronous worker for downloading and installing portable FFmpeg."""
    progress_signal = Signal(int, str)  # percent, status
    finished_signal = Signal(bool, str) # success, message

    def __init__(self, target_dir):
        super().__init__()
        self.target_dir = target_dir

    def run(self):
        try:
            from install_ffmpeg import install_ffmpeg
            def cb(downloaded, total, status_text):
                pct = int(downloaded * 100 / max(1, total)) if total > 0 else 50
                self.progress_signal.emit(pct, status_text)

            bin_dir = install_ffmpeg(self.target_dir, progress_callback=cb)
            init_ffmpeg_path()
            self.finished_signal.emit(True, bin_dir)
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class FFmpegAutoInstallDialog(QDialog):
    """Modal prompt that offers 1-click download when FFmpeg is not found."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("FFmpeg Setup Required")
        self.setMinimumWidth(480)
        self.setStyleSheet(f"""
            QDialog {{ background: {CLR_PANEL}; color: {CLR_TEXT}; }}
            QLabel {{ color: {CLR_TEXT}; }}
            QProgressBar {{
                background: {CLR_PANEL_LIGHT}; border: 1px solid {CLR_BORDER};
                border-radius: 4px; text-align: center; color: white; height: 20px;
            }}
            QProgressBar::chunk {{ background: {CLR_ACCENT}; border-radius: 4px; }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        icon_lbl = QLabel("🎬 FFmpeg Not Detected")
        icon_lbl.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {CLR_WARN};")
        layout.addWidget(icon_lbl)

        info_lbl = QLabel(
            "ParisNeo Video Studio requires <b>FFmpeg</b> to record screen/audio and edit videos.<br><br>"
            "Would you like to automatically download and install a portable version into your local <code>./ffmpeg</code> folder?"
        )
        info_lbl.setWordWrap(True)
        info_lbl.setTextFormat(Qt.RichText)
        layout.addWidget(info_lbl)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        self.status_lbl = QLabel("")
        self.status_lbl.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 11px;")
        self.status_lbl.setVisible(False)
        layout.addWidget(self.status_lbl)

        self.btn_layout = QHBoxLayout()
        self.btn_install = QPushButton("⬇️ Install Portable FFmpeg (1-Click)")
        self.btn_install.setStyleSheet(f"background: {CLR_ACCENT}; color: white; padding: 10px 18px; border-radius: 6px; font-weight: bold;")
        self.btn_install.clicked.connect(self._start_install)
        self.btn_layout.addWidget(self.btn_install)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setStyleSheet(f"background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT}; border: 1px solid {CLR_BORDER}; padding: 10px 18px; border-radius: 6px;")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_layout.addWidget(self.btn_cancel)

        layout.addLayout(self.btn_layout)
        self.worker = None

    def _start_install(self):
        self.btn_install.setEnabled(False)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.status_lbl.setVisible(True)
        self.status_lbl.setText("Connecting to repository...")

        self.worker = FFmpegInstallWorker(str(_APP_DIR))
        self.worker.progress_signal.connect(self._on_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    def _on_progress(self, pct, status):
        self.progress_bar.setValue(pct)
        self.status_lbl.setText(status)

    def _on_finished(self, success, message):
        if success:
            self.status_lbl.setText("Installation Complete!")
            QMessageBox.information(self, "Success", f"FFmpeg portable has been installed into:\n{message}")
            self.accept()
        else:
            QMessageBox.critical(self, "Installation Failed", f"Could not install FFmpeg:\n{message}")
            self.btn_install.setEnabled(True)
            self.btn_cancel.setEnabled(True)


class RecordingThread(QThread):
    """Runs ffmpeg recording subprocess with zero-backpressure asynchronous I/O."""
    recording_finished = Signal(bool, str)  # success, output_path
    error_occurred = Signal(str)

    def __init__(self, command, output_path):
        super().__init__()
        self.command = command
        self.output_path = output_path
        self._process = None
        self._stop_requested = False

    def run(self):
        try:
            logger.info(f"Starting recording: {' '.join(self.command)}")

            creationflags = 0
            if platform.system() == "Windows":
                creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

            # Route stdout to DEVNULL to avoid OS pipe stalls on real-time carriage-return loops
            self._process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                creationflags=creationflags
            )

            # Collect only stderr logs for error reporting
            stderr_tail = []
            if self._process.stderr:
                for line in self._process.stderr:
                    stderr_tail.append(line)
                    if len(stderr_tail) > 40:
                        stderr_tail.pop(0)

            self._process.wait()
            ret_code = self._process.returncode

            if self._stop_requested and ret_code != 0:
                if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
                    logger.info(f"Recording stopped gracefully: {self.output_path}")
                    self.recording_finished.emit(True, self.output_path)
                    return

            if ret_code == 0 or ret_code == 255:
                if os.path.exists(self.output_path) and os.path.getsize(self.output_path) > 0:
                    logger.info(f"Recording finished successfully: {self.output_path}")
                    self.recording_finished.emit(True, self.output_path)
                else:
                    logger.error("Recording produced empty file.")
                    self.error_occurred.emit("Recording produced empty file.")
            else:
                err_msg = "".join(stderr_tail)[-500:] or f"ffmpeg exited with code {ret_code}."
                logger.error(f"Recording failed: {err_msg}")
                self.error_occurred.emit(err_msg)

        except Exception as e:
            logger.exception("Recording thread crashed")
            self.error_occurred.emit(str(e))

    def stop(self):
        """Gracefully stop the recording by sending 'q' to ffmpeg."""
        if self._process and self._process.poll() is None:
            self._stop_requested = True
            try:
                logger.info("Sending 'q' to stop recording gracefully...")
                self._process.stdin.write('q')
                self._process.stdin.flush()
            except Exception:
                logger.warning("Failed to send 'q', attempting safe termination.")
                try:
                    if platform.system() == "Windows":
                        self._process.send_signal(subprocess.CTRL_BREAK_EVENT)
                    else:
                        self._process.terminate()
                except Exception:
                    pass

# ═══════════════════════════════════════════════════════════════
#  SLEEP / LOCK INHIBITOR
# ═══════════════════════════════════════════════════════════════

class SleepInhibitor:
    """
    Cross-platform power-management inhibitor used while recording.

    While active, the OS will NOT:
      - dim / power off the display (screen going black)
      - start the screensaver (which also triggers session lock)
      - suspend the machine to idle sleep

    Backends:
      Windows : SetThreadExecutionState (ctypes, official API)
      macOS   : `caffeinate -disu` helper process
      Linux   : DBus Inhibit on freedesktop ScreenSaver + GNOME SessionManager,
                falling back to `systemd-inhibit` when available

    All failures are logged and swallowed — recording must never break
    because a power-management assertion could not be obtained.
    """

    ES_CONTINUOUS       = 0x80000000
    ES_SYSTEM_REQUIRED  = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002

    def __init__(self, app_name="ParisNeo Video Studio"):
        self._app_name = app_name
        self._active = False
        self._caffeinate_proc = None
        self._systemd_proc = None
        self._dbus_sessions = []  # list of (dest, path, interface, cookie)

    def __del__(self):
        try:
            if self._active:
                self.release()
        except Exception:
            pass

    def acquire(self, reason="Screen recording in progress"):
        if self._active:
            return
        plat = get_platform_name()
        try:
            if plat == "Windows":
                self._acquire_windows()
            elif plat == "Darwin":
                self._acquire_macos()
            elif plat == "Linux":
                self._acquire_linux(reason)
            self._active = True
            logger.info(f"Sleep/lock inhibitor ACQUIRED ({plat}): {reason}")
        except Exception as e:
            logger.warning(f"Could not acquire sleep inhibitor: {e}")

    def release(self):
        if not self._active:
            return
        plat = get_platform_name()
        try:
            if plat == "Windows":
                self._release_windows()
            elif plat == "Darwin":
                self._release_macos()
            elif plat == "Linux":
                self._release_linux()
        except Exception as e:
            logger.warning(f"Could not release sleep inhibitor: {e}")
        finally:
            self._active = False
            logger.info("Sleep/lock inhibitor RELEASED")

    def _acquire_windows(self):
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(
            self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_DISPLAY_REQUIRED
        )

    def _release_windows(self):
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)

    def _acquire_macos(self):
        if shutil.which("caffeinate"):
            self._caffeinate_proc = subprocess.Popen(
                ["caffeinate", "-disu"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    def _release_macos(self):
        if self._caffeinate_proc and self._caffeinate_proc.poll() is None:
            self._caffeinate_proc.terminate()
            try:
                self._caffeinate_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._caffeinate_proc.kill()
        self._caffeinate_proc = None

    def _dbus_call(self, args):
        if not shutil.which("dbus-send"):
            return None
        try:
            result = subprocess.run(
                ["dbus-send", "--session", "--print-reply"] + args,
                capture_output=True, text=True, timeout=3
            )
            match = re.search(r"uint32\s+(\d+)", result.stdout)
            if result.returncode == 0 and match:
                return int(match.group(1))
        except Exception:
            pass
        return None

    def _acquire_linux(self, reason):
        # freedesktop ScreenSaver (KDE, XFCE, Cinnamon, generic XSS)
        cookie = self._dbus_call([
            "--dest=org.freedesktop.ScreenSaver",
            "/org/freedesktop/ScreenSaver",
            "org.freedesktop.ScreenSaver.Inhibit",
            f"string:{self._app_name}", f"string:{reason}"
        ])
        if cookie is not None:
            self._dbus_sessions.append((
                "org.freedesktop.ScreenSaver", "/org/freedesktop/ScreenSaver",
                "org.freedesktop.ScreenSaver", cookie
            ))

        # GNOME SessionManager (flags 4|8 = suspend + idle/lock)
        cookie = self._dbus_call([
            "--dest=org.gnome.SessionManager",
            "/org/gnome/SessionManager",
            "org.gnome.SessionManager.Inhibit",
            f"string:{self._app_name}", "uint32:0", f"string:{reason}", "uint32:12"
        ])
        if cookie is not None:
            self._dbus_sessions.append((
                "org.gnome.SessionManager", "/org/gnome/SessionManager",
                "org.gnome.SessionManager", cookie
            ))

        # Fallback: logind-level suspend/idle inhibition
        if not self._dbus_sessions and shutil.which("systemd-inhibit"):
            self._systemd_proc = subprocess.Popen(
                [
                    "systemd-inhibit",
                    "--what=sleep:idle",
                    "--mode=block",
                    f"--why={reason}",
                    "tail", "-f", "/dev/null"
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )

    def _release_linux(self):
        for dest, path, iface, cookie in self._dbus_sessions:
            try:
                subprocess.run([
                    "dbus-send", "--session", "--print-reply", f"--dest={dest}",
                    path, f"{iface}.UnInhibit", f"uint32:{cookie}"
                ], capture_output=True, timeout=3)
            except Exception:
                pass
        self._dbus_sessions.clear()

        if self._systemd_proc and self._systemd_proc.poll() is None:
            self._systemd_proc.terminate()
            try:
                self._systemd_proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._systemd_proc.kill()
        self._systemd_proc = None


# ═══════════════════════════════════════════════════════════════
#  MAIN WINDOW (Recording Mode)
# ═══════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Main application window — recording mode."""

    def __init__(self, editor_ref=None):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(520, 400)
        self.setStyleSheet(self._stylesheet())

        # State
        self.region_rect = None  # (x, y, w, h)
        self.recording_thread = None
        self.recording_overlay = None
        self.recorded_file = None
        self.editor_window = editor_ref
        self.countdown_label = None
        self.sleep_inhibitor = SleepInhibitor(APP_NAME)
        self.recording_settings = RecordingSettingsManager.load_settings()

        # Check ffmpeg
        self._check_ffmpeg()

        # Build UI
        self._build_ui()

        # Apply persisted settings to widgets now that they exist
        self._apply_loaded_settings()

        # Connect persistence signals once all widgets exist
        self._connect_persistence_signals()

    def _connect_persistence_signals(self):
        """Wire all recording widget change signals to the settings serializer (once)."""
        if not getattr(self, '_persistence_signals_connected', False):
            self.video_mode_combo.currentIndexChanged.connect(self._on_video_mode_changed)
            self.pip_pos_combo.currentIndexChanged.connect(self._save_recording_settings)
            self.fps_combo.currentIndexChanged.connect(self._save_recording_settings)
            self.audio_combo.currentIndexChanged.connect(self._save_recording_settings)
            if hasattr(self, 'mic_combo'):
                self.mic_combo.currentIndexChanged.connect(self._save_recording_settings)
            self.video_combo.currentIndexChanged.connect(self._save_recording_settings)
            self._persistence_signals_connected = True

    def _check_ffmpeg(self):
        version = FFmpegHelper.get_version()
        if version:
            logger.info(f"ffmpeg found: {version}")
        else:
            logger.warning("FFmpeg not detected! Launching 1-click portable installer dialog...")
            dialog = FFmpegAutoInstallDialog(self)
            if dialog.exec() == QDialog.Accepted:
                init_ffmpeg_path()
                new_ver = FFmpegHelper.get_version()
                if new_ver:
                    logger.info(f"FFmpeg installed successfully: {new_ver}")
                    self.statusBar().showMessage(f"FFmpeg ready: {new_ver}", 4000)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        title = QLabel("🎬 Screen Recorder & Editor")
        title.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {CLR_ACCENT};")
        layout.addWidget(title)

        subtitle = QLabel("Record a screen region with audio, then trim, cut, and export.")
        subtitle.setStyleSheet(f"color: {CLR_TEXT_DIM}; font-size: 12px;")
        layout.addWidget(subtitle)

        region_group = QGroupBox("Recording Region")
        region_layout = QVBoxLayout(region_group)

        self.region_label = QLabel("No region selected — click below to select")
        self.region_label.setStyleSheet(f"color: {CLR_TEXT_DIM}; padding: 8px;")
        region_layout.addWidget(self.region_label)

        btn_select = QPushButton("📐 Select Screen Region")
        btn_select.setMinimumHeight(40)
        btn_select.setStyleSheet(self._btn_style(CLR_ACCENT))
        btn_select.clicked.connect(self._select_region)
        region_layout.addWidget(btn_select)

        layout.addWidget(region_group)

        settings_group = QGroupBox("Recording Settings")
        settings_layout = QFormLayout(settings_group)

        self.audio_combo = QComboBox()
        self.audio_combo.setMinimumWidth(300)
        self.audio_combo.addItem("No Audio", userData=None)
        settings_layout.addRow("🔊 PC Audio / TTS:", self.audio_combo)

        self.mic_combo = QComboBox()
        self.mic_combo.setMinimumWidth(300)
        self.mic_combo.addItem("No Microphone (Off)", userData=None)
        settings_layout.addRow("🎙️ Mic / Commentary:", self.mic_combo)

        self._refresh_audio_devices()

        self.btn_audio_help = QPushButton("❓ Audio & Mic Help")
        self.btn_audio_help.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        self.btn_audio_help.clicked.connect(self._show_audio_help)
        settings_layout.addRow("", self.btn_audio_help)

        self.video_combo = QComboBox()
        self.video_combo.setMinimumWidth(300)
        self._refresh_video_devices()
        settings_layout.addRow("Video Source (Camera):", self.video_combo)

        self.video_mode_combo = QComboBox()
        self.video_mode_combo.addItem("Screen Capture Only", "screen_only")
        self.video_mode_combo.addItem("Screen Capture + Camera (PIP)", "screen_pip")
        self.video_mode_combo.addItem("Camera Only", "camera_only")
        self.video_mode_combo.currentIndexChanged.connect(self._on_video_mode_changed)
        settings_layout.addRow("Video Source Mode:", self.video_mode_combo)

        self.pip_pos_combo = QComboBox()
        self.pip_pos_combo.addItem("Top-Left", "top-left")
        self.pip_pos_combo.addItem("Top-Right", "top-right")
        self.pip_pos_combo.addItem("Bottom-Left", "bottom-left")
        self.pip_pos_combo.addItem("Bottom-Right", "bottom-right")
        self.pip_pos_combo.setCurrentIndex(3)
        settings_layout.addRow("PIP Position:", self.pip_pos_combo)
        self.pip_pos_combo.setEnabled(False)

        btn_refresh = QPushButton("🔄 Refresh Devices")
        btn_refresh.clicked.connect(self._refresh_audio_devices)
        btn_refresh.clicked.connect(self._refresh_video_devices)
        settings_layout.addRow("", btn_refresh)

        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["15", "24", "30", "60"])
        self.fps_combo.setCurrentText("30")
        settings_layout.addRow("Frame Rate:", self.fps_combo)

        layout.addWidget(settings_group)

        self.btn_record = QPushButton("⏺ Start Recording")
        self.btn_record.setMinimumHeight(50)
        self.btn_record.setStyleSheet(f"""
            QPushButton {{
                background: {CLR_DANGER}; color: white; border: none;
                border-radius: 8px; font-size: 16px; font-weight: bold;
            }}
            QPushButton:hover {{ background: #ff6b6d; }}
            QPushButton:disabled {{ background: #333; color: #666; }}
        """)
        self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self._start_recording)
        layout.addWidget(self.btn_record)

        btn_open = QPushButton("📂 Open Existing Video for Editing")
        btn_open.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
        btn_open.clicked.connect(self._open_existing)
        layout.addWidget(btn_open)

        layout.addStretch()

        self.statusBar().showMessage("Ready")

    def _save_recording_settings(self):
        """Serialize current recording widget selections to disk."""
        settings = {
            "region_rect": list(self.region_rect) if self.region_rect else None,
            "audio_device": self.audio_combo.currentData() if self.audio_combo.currentIndex() > 0 else None,
            "mic_device": self.mic_combo.currentData() if hasattr(self, 'mic_combo') and self.mic_combo.currentIndex() > 0 else None,
            "video_device": self.video_combo.currentText() if self.video_combo.count() > 0 else None,
            "video_mode": self.video_mode_combo.currentData(),
            "pip_position": self.pip_pos_combo.currentData(),
            "fps": self.fps_combo.currentText(),
        }
        RecordingSettingsManager.save_settings(settings)

    def _apply_loaded_settings(self):
        """Restore widget selections from persisted settings with graceful fallback."""
        s = self.recording_settings
        if not s:
            return

        # Restore recording region
        region = s.get("region_rect")
        if region and isinstance(region, (list, tuple)) and len(region) == 4:
            try:
                x, y, w, h = int(region[0]), int(region[1]), int(region[2]), int(region[3])
                if w >= 20 and h >= 20:
                    self.region_rect = (x, y, w, h)
                    self.region_label.setText(f"Region: {w} × {h} at ({x}, {y})")
                    self.region_label.setStyleSheet(f"color: {CLR_SUCCESS}; padding: 8px; font-weight: bold;")
                    self.btn_record.setEnabled(self.video_mode_combo.currentData() != "camera_only" or True)
            except Exception:
                pass

        # Restore FPS
        fps = s.get("fps")
        if fps:
            idx = self.fps_combo.findText(str(fps))
            if idx >= 0:
                self.fps_combo.setCurrentIndex(idx)

        # Restore video mode (triggers _on_video_mode_changed for PIP enable state)
        mode = s.get("video_mode")
        if mode:
            idx = self.video_mode_combo.findData(mode)
            if idx >= 0:
                self.video_mode_combo.setCurrentIndex(idx)

        # Restore PIP position
        pip_pos = s.get("pip_position")
        if pip_pos:
            idx = self.pip_pos_combo.findData(pip_pos)
            if idx >= 0:
                self.pip_pos_combo.setCurrentIndex(idx)

        # Restore audio device by userData match (graceful fallback if unplugged)
        audio_dev = s.get("audio_device")
        if audio_dev and hasattr(self, 'audio_combo'):
            idx = self.audio_combo.findData(audio_dev)
            if idx >= 0:
                self.audio_combo.setCurrentIndex(idx)

        # Restore mic device
        mic_dev = s.get("mic_device")
        if mic_dev and hasattr(self, 'mic_combo'):
            idx = self.mic_combo.findData(mic_dev)
            if idx >= 0:
                self.mic_combo.setCurrentIndex(idx)

        # Restore video device (match by text, fallback to index 0)
        video_dev = s.get("video_device")
        if video_dev and self.video_combo.count() > 0:
            idx = self.video_combo.findText(video_dev)
            if idx >= 0:
                self.video_combo.setCurrentIndex(idx)

        # Enable record button if a valid region or camera-only mode is available
        if self.region_rect or self.video_mode_combo.currentData() == "camera_only":
            self.btn_record.setEnabled(True)

    def _on_video_mode_changed(self, index):
        mode = self.video_mode_combo.currentData()
        if mode == "screen_pip":
            self.pip_pos_combo.setEnabled(True)
        else:
            self.pip_pos_combo.setEnabled(False)
        self._save_recording_settings()


    def _refresh_audio_devices(self):
        self.audio_combo.clear()
        self.audio_combo.addItem("No Audio", userData=None)

        if hasattr(self, 'mic_combo'):
            self.mic_combo.clear()
            self.mic_combo.addItem("No Microphone (Off)", userData=None)

        devices = FFmpegHelper.list_audio_devices()

        sys_keywords = [
            "stereo mix", "mixage stéréo", "what u hear", "wave out",
            "loopback", "cable output", "voicemeeter", "virtual",
            ".monitor", "blackhole", "soundflower"
        ]

        system_audio_idx = -1
        mic_audio_idx = -1

        for dev in devices:
            is_sys = any(kw in dev.lower() for kw in sys_keywords)
            if is_sys:
                display_label = f"🔊 [PC Audio / TTS] {dev}"
            else:
                display_label = f"🎙️ {dev}"

            self.audio_combo.addItem(display_label, userData=dev)
            if hasattr(self, 'mic_combo'):
                self.mic_combo.addItem(display_label, userData=dev)

            if is_sys and system_audio_idx == -1:
                system_audio_idx = self.audio_combo.count() - 1
            elif not is_sys and mic_audio_idx == -1:
                mic_audio_idx = self.audio_combo.count() - 1

        if system_audio_idx > 0:
            self.audio_combo.setCurrentIndex(system_audio_idx)

        if hasattr(self, 'mic_combo') and mic_audio_idx > 0:
            # Leave mic default as 'No Microphone' unless user desires or select first physical mic
            pass

        # Persist on device selection change (connect once)
        if not getattr(self, '_audio_signals_connected', False):
            self.audio_combo.currentIndexChanged.connect(self._save_recording_settings)
            if hasattr(self, 'mic_combo'):
                self.mic_combo.currentIndexChanged.connect(self._save_recording_settings)
            self._audio_signals_connected = True

        self.statusBar().showMessage(f"Found {len(devices)} audio/mic device(s)")

    def _refresh_video_devices(self):
        self.video_combo.clear()
        devices = FFmpegHelper.list_video_devices()
        for dev in devices:
            self.video_combo.addItem(dev)

        # Persist on video device selection change (connect once)
        if not getattr(self, '_video_signals_connected', False):
            self.video_combo.currentIndexChanged.connect(self._save_recording_settings)
            self._video_signals_connected = True

        self.statusBar().showMessage(f"Found {len(devices)} video device(s)")

    def _show_audio_help(self):
        """Comprehensive Cross-Platform Guide for System Audio & Text-to-Speech Recording."""
        dialog = QDialog(self)
        dialog.setWindowTitle("🔊 Cross-Platform Audio Capture & TTS Setup Guide")
        dialog.setMinimumSize(700, 580)
        dialog.setStyleSheet(f"""
            QDialog {{ background: {CLR_PANEL}; color: {CLR_TEXT}; }}
            QTabWidget::pane {{ border: 1px solid {CLR_BORDER}; border-radius: 6px; background: {CLR_BG}; padding: 12px; }}
            QTabBar::tab {{ background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT_DIM}; padding: 9px 16px; margin-right: 4px; border-top-left-radius: 4px; border-top-right-radius: 4px; font-weight: bold; }}
            QTabBar::tab:selected {{ background: {CLR_ACCENT}; color: white; }}
            QLabel {{ color: {CLR_TEXT}; font-size: 12px; line-height: 18px; }}
            QScrollArea {{ border: none; background: transparent; }}
        """)

        layout = QVBoxLayout(dialog)
        layout.setSpacing(12)

        header = QLabel("Audio Capture & Text-to-Speech Setup (Windows / Linux / macOS)")
        header.setStyleSheet(f"font-size: 17px; font-weight: bold; color: {CLR_ACCENT};")
        layout.addWidget(header)

        tabs = QTabWidget()
        os_name = platform.system()

        # ==========================================
        # TAB 1: WINDOWS (VB-CABLE & STEREO MIX)
        # ==========================================
        tab_win = QWidget()
        l_win = QVBoxLayout(tab_win)
        scroll_win = QScrollArea()
        scroll_win.setWidgetResizable(True)
        w_win = QWidget()
        lw_win = QVBoxLayout(w_win)

        t_win = QLabel(
            "<h3>🪟 Windows Setup Guide</h3>"
            "Windows isolates audio playback from recording endpoints. Choose the solution that matches your setup:<br><br>"
            "<b>Option 1: VB-Audio Virtual Cable (Recommended for Bluetooth, USB, HDMI & Generic Drivers)</b><br>"
            "• <b>Step 1 (Send PC sound into Cable):</b> Click the Speaker icon on your Windows taskbar and set output to <b><code>CABLE Input (VB-Audio)</code></b>.<br>"
            "• <b>Step 2 (Hear sound in your ears while recording):</b> Open Sound Settings (click the button below) &rarr; Go to <b>Enregistrement (Recording)</b> tab &rarr; Double-click <b><code>CABLE Output</code></b> &rarr; <b>Écouter (Listen)</b> tab &rarr; Check <b>'Écouter ce périphérique' (Listen to this device)</b> and select your <b>Speakers or Bluetooth Headphones</b>.<br>"
            "• <b>Step 3:</b> In this app, select <b><code>🔊 [PC Audio / TTS] CABLE Output</code></b> and record.<br>"
            "<i>💡 Troubleshooting: If you don't hear your recorded video during playback, ensure the 'Listen' box is checked or switch the taskbar speaker back to your physical output.</i><br><br>"
            "<b>Option 2: Native 'Mixage Stéréo' (Stereo Mix)</b><br>"
            "• Works on motherboard speakers with the official <b>Realtek Audio Driver</b> installed.<br>"
            "• Right-click in the Recording tab &rarr; Check <i>'Show Disabled Devices'</i> &rarr; Right-click <b>Mixage Stéréo</b> &rarr; <b>Activer (Enable)</b>."
        )
        t_win.setWordWrap(True)
        t_win.setTextFormat(Qt.RichText)
        lw_win.addWidget(t_win)
        scroll_win.setWidget(w_win)
        l_win.addWidget(scroll_win)
        tabs.addTab(tab_win, "🪟 Windows")

        # ==========================================
        # TAB 2: LINUX (PULSEAUDIO & PIPEWIRE)
        # ==========================================
        tab_linux = QWidget()
        l_linux = QVBoxLayout(tab_linux)
        scroll_linux = QScrollArea()
        scroll_linux.setWidgetResizable(True)
        w_linux = QWidget()
        lw_linux = QVBoxLayout(w_linux)

        t_linux = QLabel(
            "<h3>🐧 Linux Setup Guide (PulseAudio & PipeWire)</h3>"
            "Linux natively supports system audio loopback without installing third-party virtual cables.<br><br>"
            "<b>Native Monitor Sources:</b><br>"
            "1. PulseAudio and PipeWire automatically create a <code>.monitor</code> source for your active sound card.<br>"
            "2. In this recorder's <b>Audio Source</b> dropdown, select any entry marked <b><code>[Monitor] ...</code></b> (e.g., <i>[Monitor] alsa_output.pci-0000_00_1b.0.analog-stereo.monitor</i>).<br>"
            "3. Any audio generated on your desktop (TTS synthesis, browser, game) will be captured with zero latency.<br><br>"
            "<b>Dynamic Routing with <code>pavucontrol</code> (Volume Control):</b><br>"
            "• Install if missing: <code>sudo apt install pavucontrol</code> (or <code>sudo dnf install pavucontrol</code>)<br>"
            "• Open <b>pavucontrol</b> &rarr; Go to the <b>Recording</b> tab while capturing.<br>"
            "• Change the capture source dropdown of FFmpeg to <i>'Monitor of Built-in Audio'</i> or directly to a specific running application!"
        )
        t_linux.setWordWrap(True)
        t_linux.setTextFormat(Qt.RichText)
        lw_linux.addWidget(t_linux)
        scroll_linux.setWidget(w_linux)
        l_linux.addWidget(scroll_linux)
        tabs.addTab(tab_linux, "🐧 Linux")

        # ==========================================
        # TAB 3: MACOS (BLACKHOLE & COREAUDIO)
        # ==========================================
        tab_mac = QWidget()
        l_mac = QVBoxLayout(tab_mac)
        scroll_mac = QScrollArea()
        scroll_mac.setWidgetResizable(True)
        w_mac = QWidget()
        lw_mac = QVBoxLayout(w_mac)

        t_mac = QLabel(
            "<h3>🍎 macOS Setup Guide (CoreAudio)</h3>"
            "macOS security restricts direct internal audio capture. To capture internal Mac sound and hear it simultaneously:<br><br>"
            "<b>Step 1: Install BlackHole (Free, Open-Source Virtual Audio Driver)</b><br>"
            "• Via Homebrew: <code>brew install blackhole-2ch</code><br>"
            "• Or download the installer directly from: <b>existential.audio/blackhole</b><br><br>"
            "<b>Step 2: Create a Multi-Output Device (Hear audio in your Mac speakers/AirPods too)</b><br>"
            "1. Open <b>Audio MIDI Setup</b> (Press <code>Cmd + Space</code>, type <i>Audio MIDI Setup</i>).<br>"
            "2. Click the <b>+</b> icon in the bottom-left corner &rarr; <b>Create Multi-Output Device</b>.<br>"
            "3. Check both your <b>Mac Speakers / Headphones</b> AND <b>BlackHole 2ch</b>.<br>"
            "4. Right-click the Multi-Output Device &rarr; select <b>'Use This Device For Sound Output'</b>.<br><br>"
            "<b>Step 3: In ParisNeo Video Recorder</b><br>"
            "• In the Audio Source dropdown, select <b><code>[0] BlackHole 2ch</code></b> and record."
        )
        t_mac.setWordWrap(True)
        t_mac.setTextFormat(Qt.RichText)
        lw_mac.addWidget(t_mac)
        scroll_mac.setWidget(w_mac)
        l_mac.addWidget(scroll_mac)
        tabs.addTab(tab_mac, "🍎 macOS")

        # ==========================================
        # TAB 4: DIRECT FILE IMPORT (ALL PLATFORMS)
        # ==========================================
        tab_import = QWidget()
        l_import = QVBoxLayout(tab_import)
        t_import = QLabel(
            "<h3>🎵 Direct TTS / Audio File Import (Studio Fidelity)</h3>"
            "<b>Zero live recording or device routing required!</b><br><br>"
            "If you generate Text-to-Speech (TTS) using Python scripts, ElevenLabs, OpenAI, Piper, or Edge-TTS:<br><br>"
            "1. Save your synthesized speech directly as a <code>.wav</code>, <code>.mp3</code>, or <code>.aac</code> file.<br>"
            "2. In ParisNeo Video Studio editor, click <b>🎵 +Audio</b> on the timeline toolbar.<br>"
            "3. Import your audio file directly onto any audio channel.<br>"
            "4. Position and slice your voice clips with frame-accurate timeline synchronization and zero background noise."
        )
        t_import.setWordWrap(True)
        t_import.setTextFormat(Qt.RichText)
        l_import.addWidget(t_import)
        l_import.addStretch()
        tabs.addTab(tab_import, "🎵 Direct File Import")

        # Auto-focus the tab matching the user's OS
        if os_name == "Linux":
            tabs.setCurrentIndex(1)
        elif os_name == "Darwin":
            tabs.setCurrentIndex(2)
        else:
            tabs.setCurrentIndex(0)

        layout.addWidget(tabs)

        # ==========================================
        # INTERACTIVE ACTION BUTTONS
        # ==========================================
        btn_actions = QHBoxLayout()
        btn_actions.setSpacing(8)

        if os_name == "Windows":
            btn_sound = QPushButton("⚙️ Sound Recording Settings")
            btn_sound.setToolTip("Open Windows Sound Control Panel directly to Recording tab")
            btn_sound.setStyleSheet(self._btn_style(CLR_ACCENT_2))
            btn_sound.clicked.connect(lambda: subprocess.Popen(["rundll32.exe", "shell32.dll,Control_RunDLL", "mmsys.cpl,,1"]))
            btn_actions.addWidget(btn_sound)

            btn_site = QPushButton("🌐 VB-Audio Website")
            btn_site.setToolTip("Open official download page for VB-Audio Cable")
            btn_site.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
            btn_site.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://vb-audio.com/Cable/")))
            btn_actions.addWidget(btn_site)

        elif os_name == "Linux":
            btn_pavu = QPushButton("🎛️ Open pavucontrol")
            btn_pavu.setToolTip("Open PulseAudio Volume Control to inspect monitor streams")
            btn_pavu.setStyleSheet(self._btn_style(CLR_ACCENT_2))
            btn_pavu.clicked.connect(lambda: subprocess.Popen(["pavucontrol"]))
            btn_actions.addWidget(btn_pavu)

        elif os_name == "Darwin":
            btn_midi = QPushButton("🎛️ Open Audio MIDI Setup")
            btn_midi.setToolTip("Open macOS Audio MIDI Setup to configure Multi-Output device")
            btn_midi.setStyleSheet(self._btn_style(CLR_ACCENT_2))
            btn_midi.clicked.connect(lambda: subprocess.Popen(["open", "-a", "Audio MIDI Setup"]))
            btn_actions.addWidget(btn_midi)

            btn_bh = QPushButton("🌐 BlackHole Website")
            btn_bh.setStyleSheet(self._btn_style(CLR_PANEL_LIGHT))
            btn_bh.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://existential.audio/blackhole/")))
            btn_actions.addWidget(btn_bh)

        btn_actions.addStretch()

        btn_close = QPushButton("Close")
        btn_close.setStyleSheet(self._btn_style(CLR_ACCENT))
        btn_close.clicked.connect(dialog.accept)
        btn_actions.addWidget(btn_close)

        layout.addLayout(btn_actions)
        dialog.exec()

    def _select_region(self):
        self.hide()
        QTimer.singleShot(300, self._show_region_selector)

    def _show_region_selector(self):
        self.selector = RegionSelector()
        self.selector.region_selected.connect(self._on_region_selected)
        self.selector.selection_cancelled.connect(self._on_region_cancelled)
        self.selector.show()

    def _on_region_selected(self, x, y, w, h):
        self.region_rect = (x, y, w, h)
        self.region_label.setText(f"Region: {w} × {h} at ({x}, {y})")
        self.region_label.setStyleSheet(f"color: {CLR_SUCCESS}; padding: 8px; font-weight: bold;")
        self.btn_record.setEnabled(True)
        self._save_recording_settings()
        self.show()

    def _on_region_cancelled(self):
        self.show()

    def _start_recording(self):
        if not self.region_rect and self.video_mode_combo.currentData() != "camera_only":
            QMessageBox.warning(self, "No Region", "Please select a recording region first.")
            return

        x, y, w, h = self.region_rect if self.region_rect else (0, 0, 640, 480)
        fps = int(self.fps_combo.currentText())
        audio_device = None
        if self.audio_combo.currentIndex() > 0:
            audio_device = self.audio_combo.currentData() or self.audio_combo.currentText()

        mic_device = None
        if hasattr(self, 'mic_combo') and self.mic_combo.currentIndex() > 0:
            mic_device = self.mic_combo.currentData() or self.mic_combo.currentText()

        video_device = self.video_combo.currentText()
        video_mode = self.video_mode_combo.currentData()
        pip_position = self.pip_pos_combo.currentData()

        output_dir = os.path.join(os.path.expanduser("~"), "Videos", "ParisNeo_Recordings")
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.recorded_file = os.path.join(output_dir, f"recording_{timestamp}.mp4")

        try:
            cmd = FFmpegHelper.build_record_command(
                self.recorded_file, x, y, w, h, fps,
                audio_device=audio_device,
                mic_device=mic_device,
                video_device=video_device,
                pip=False,
                video_mode=video_mode,
                pip_position=pip_position
            )
        except Exception as e:
            logger.exception("Failed to build recording command")
            QMessageBox.critical(self, "Error", f"Failed to build recording command:\n{e}")
            return

        print(f"🎬 Recording command: {' '.join(cmd)}")

        self.hide()
        self._show_countdown(lambda: self._begin_recording(cmd))


    def _show_countdown(self, callback):
        """Show 3-2-1 countdown overlay."""
        self.show()
        self.raise_()
        self.activateWindow()
        self.countdown_label = QLabel(self)
        self.countdown_label.setAlignment(Qt.AlignCenter)
        self.countdown_label.setStyleSheet(f"""
            font-size: 120px; font-weight: bold; color: {CLR_ACCENT};
            background: rgba(10, 10, 15, 0.8); border-radius: 20px;
        """)
        self.countdown_label.setGeometry(self.geometry())
        self.countdown_label.show()

        count = 3

        def tick():
            nonlocal count
            if count > 0:
                self.countdown_label.setText(str(count))
                self.countdown_label.adjustSize()
                self.countdown_label.setGeometry(
                    (self.width() - self.countdown_label.width()) // 2,
                    (self.height() - self.countdown_label.height()) // 2,
                    self.countdown_label.width(),
                    self.countdown_label.height()
                )
                count -= 1
                QTimer.singleShot(800, tick)
            else:
                self.countdown_label.setText("GO!")
                self.countdown_label.setStyleSheet(f"""
                    font-size: 80px; font-weight: bold; color: {CLR_SUCCESS};
                    background: rgba(10, 10, 15, 0.8); border-radius: 20px;
                """)
                self.countdown_label.adjustSize()
                self.countdown_label.setGeometry(
                    (self.width() - self.countdown_label.width()) // 2,
                    (self.height() - self.countdown_label.height()) // 2,
                    self.countdown_label.width(),
                    self.countdown_label.height()
                )
                QTimer.singleShot(400, lambda: (self.countdown_label.close(), callback()))

        tick()

    def _begin_recording(self, cmd):
        self.hide()
        self.sleep_inhibitor.acquire("Screen recording in progress")

        # Start recording thread
        self.recording_thread = RecordingThread(cmd, self.recorded_file)
        self.recording_thread.recording_finished.connect(self._on_recording_finished)
        self.recording_thread.error_occurred.connect(self._on_recording_error)
        self.recording_thread.start()

        # Show recording overlay
        self.recording_overlay = RecordingOverlay(QRect(*self.region_rect)) if self.region_rect else RecordingOverlay(QRect(0, 0, 640, 480))
        self.recording_overlay.stop_requested.connect(self._stop_recording)
        self.recording_overlay.start()

        self.statusBar().showMessage("Recording (display sleep & auto-lock disabled)... Click Stop or press Ctrl+R to finish.")

    def _stop_recording(self):
        self.sleep_inhibitor.release()

        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()

        if self.recording_overlay:
            self.recording_overlay.stop()
            self.recording_overlay.close()
            self.recording_overlay = None

        self.show()
        self.raise_()
        self.activateWindow()

        self.btn_record.setText("⏺ Start Recording")
        self.btn_record.setEnabled(False)
        self.statusBar().showMessage("Processing recording...")

    def _on_recording_finished(self, success, output_path):
        self.sleep_inhibitor.release()
        self.btn_record.setEnabled(True)
        self.btn_record.setText("⏺ Start Recording")
        try:
            self.btn_record.clicked.disconnect()
            self.btn_record.clicked.connect(self._start_recording)
        except RuntimeError:
            pass

        if success:
            self.statusBar().showMessage(f"Recording saved: {output_path}")
            dialog = MediaCompleteDialog(
                title="Recording Complete",
                heading="Recording Saved Successfully!",
                file_path=output_path,
                on_open_editor=self._open_editor,
                parent=self
            )
            dialog.exec()
        else:
            self.statusBar().showMessage("Recording failed.")

    def _on_recording_error(self, error_msg):
        self.sleep_inhibitor.release()
        self.btn_record.setEnabled(True)
        self.btn_record.setText("⏺ Start Recording")
        try:
            self.btn_record.clicked.disconnect()
            self.btn_record.clicked.connect(self._start_recording)
        except Exception:
            pass

        self.statusBar().showMessage("Recording error!")
        QMessageBox.critical(self, "Recording Error", error_msg)

    def _open_editor(self, video_path):
        """Open the video editor window."""
        if self.editor_window and isinstance(self.editor_window, VideoEditor):
            self.editor_window.add_media_to_library(video_path)
            self.close()
        elif self.editor_window and isinstance(self.editor_window, StudioMainWindow):
            parent_studio = self.editor_window
            proj_data = parent_studio.pending_project_data
            proj_data['video_path'] = video_path
            proj_data['name'] = proj_data.get('name', os.path.basename(video_path))

            editor = VideoEditor(proj_data, parent=parent_studio)
            editor._save_project_state()

            parent_studio.stacked_widget.addWidget(editor)
            parent_studio.stacked_widget.setCurrentWidget(editor)
            parent_studio.dashboard.refresh_projects()
            self.close()
        elif self.editor_window and hasattr(self.editor_window, 'load_new_video'):
            self.editor_window.load_new_video(video_path)
            self.editor_window.show()
            self.editor_window.raise_()
            self.editor_window.activateWindow()
            self.close()
        else:
            self.editor_window = VideoEditor({"video_path": video_path}, parent=None)
            self.editor_window.show()
            self.close()

    def show_recording_mode(self):
        """Show the recording main window (called from editor)."""
        if not self.editor_window:
            self.editor_window = StudioMainWindow()
        if hasattr(self.editor_window, 'show_recording_mode'):
            self.editor_window.show_recording_mode()
        else:
            self.show()
        self.show()
        self.raise_()
        self.activateWindow()

    def _open_existing(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Video File",
            os.path.join(os.path.expanduser("~"), "Videos"),
            "Video Files (*.mp4 *.avi *.mkv *.mov *.webm);;All Files (*.*)"
        )
        if path:
            self._open_editor(path)

    @staticmethod
    def _btn_style(color):
        return f"""
            QPushButton {{
                background: {color}; color: white; border: none;
                padding: 8px 16px; border-radius: 6px; font-weight: bold;
            }}
            QPushButton:hover {{ opacity: 0.85; }}
            QPushButton:disabled {{ background: #333; color: #666; }}
        """

    def _stylesheet(self):
        return f"""
            QMainWindow {{ background: {CLR_BG}; }}
            QWidget {{ color: {CLR_TEXT}; font-family: 'Segoe UI', sans-serif; font-size: 13px; }}
            QGroupBox {{
                border: 1px solid {CLR_BORDER}; border-radius: 8px;
                margin-top: 12px; padding-top: 16px;
                font-weight: bold; color: {CLR_TEXT_DIM};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 12px; padding: 0 4px;
            }}
            QComboBox {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; padding: 6px 10px; border-radius: 4px;
            }}
            QComboBox::drop-down {{ border: none; }}
            QComboBox QAbstractItemView {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                selection-background-color: {CLR_ACCENT};
            }}
            QPushButton {{
                background: {CLR_PANEL_LIGHT}; color: {CLR_TEXT};
                border: 1px solid {CLR_BORDER}; border-radius: 4px;
                padding: 8px 14px;
            }}
            QPushButton:hover {{ background: {CLR_BORDER}; }}
            QLabel {{ color: {CLR_TEXT}; }}
            QStatusBar {{ background: {CLR_PANEL}; color: {CLR_TEXT_DIM}; }}
        """

    def keyPressEvent(self, event):
        # Ctrl+R to stop recording
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_R:
            if self.recording_thread and self.recording_thread.isRunning():
                self._stop_recording()
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.sleep_inhibitor.release()

        # Stop recording if active
        if self.recording_thread and self.recording_thread.isRunning():
            self.recording_thread.stop()
            self.recording_thread.wait(5000)
        super().closeEvent(event)


# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════

class StudioMainWindow(QMainWindow):
    """Main shell that manages the Dashboard and Editor views."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1200, 800)
        self.setStyleSheet(f"background: {CLR_BG};")

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.dashboard = ProjectDashboard()
        self.dashboard.open_project.connect(self._open_project)
        self.dashboard.record_in_project.connect(self._record_in_project)
        self.dashboard.new_project.connect(self._new_project)
        self.dashboard.record_new.connect(self._record_new)
        self.dashboard.fast_record.connect(self._fast_record)

        self.stacked_widget.addWidget(self.dashboard)

        self.editor = None
        self.recorder = None

    def show_dashboard(self):
        self.dashboard.refresh_projects()
        self.stacked_widget.setCurrentWidget(self.dashboard)

    def _open_project(self, proj_id):
        projects = ProjectManager.get_projects()
        proj_data = next((p for p in projects if p['id'] == proj_id), None)
        if proj_data:
            self.editor = VideoEditor(proj_data, parent=self)
            self.stacked_widget.addWidget(self.editor)
            self.stacked_widget.setCurrentWidget(self.editor)

    def _new_project(self):
        dialog = NewProjectDialog(self)
        if dialog.exec() == QDialog.Accepted:
            data = dialog.get_data()
            proj_data = {
                "id": f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "name": data["name"],
                "description": data["description"]
            }
            ProjectManager.save_project(proj_data)
            self.dashboard.refresh_projects()

            self.editor = VideoEditor(proj_data, parent=self)
            self.stacked_widget.addWidget(self.editor)
            self.stacked_widget.setCurrentWidget(self.editor)

    def _fast_record(self):
        """Instant record: auto-generates a timestamped project and launches recorder."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        proj_data = {
            "id": f"proj_{timestamp}",
            "name": f"Recording {timestamp}",
            "description": "Quick screen recording"
        }
        ProjectManager.save_project(proj_data)
        self.dashboard.refresh_projects()

        self.pending_project_data = proj_data
        self.recorder = MainWindow(editor_ref=self)
        self.recorder.show()

    def _record_in_project(self, proj_id):
        """Directly record new media and import it into an existing project."""
        projects = ProjectManager.get_projects()
        proj_data = next((p for p in projects if p['id'] == proj_id), None)
        if proj_data:
            self.pending_project_data = proj_data
            self.recorder = MainWindow(editor_ref=self)
            self.recorder.show()

    def _record_new(self):
        dialog = NewProjectDialog(self)
        if dialog.exec() == QDialog.Accepted:
            data = dialog.get_data()
            proj_data = {
                "id": f"proj_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "name": data["name"],
                "description": data["description"]
            }
            ProjectManager.save_project(proj_data)
            self.dashboard.refresh_projects()

            self.pending_project_data = proj_data
            self.recorder = MainWindow(editor_ref=self)
            self.recorder.show()

    def show_recording_mode(self):
        self._record_new()

def main():
    # Force hardware-level DPI awareness to prevent OS scaling blurring
    if hasattr(Qt.HighDpiScaleFactorRoundingPolicy, 'PassPrefer'):
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassPrefer)
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    palette = app.palette()
    palette.setColor(QPalette.Window, QColor(CLR_BG))
    palette.setColor(QPalette.WindowText, QColor(CLR_TEXT))
    palette.setColor(QPalette.Base, QColor(CLR_PANEL))
    palette.setColor(QPalette.Text, QColor(CLR_TEXT))
    palette.setColor(QPalette.Button, QColor(CLR_PANEL_LIGHT))
    palette.setColor(QPalette.ButtonText, QColor(CLR_TEXT))
    app.setPalette(palette)

    # Check FFmpeg availability before dashboard launch
    if not FFmpegHelper.get_version():
        dlg = FFmpegAutoInstallDialog()
        dlg.exec()
        init_ffmpeg_path()

    # Launch the Studio Shell
    studio = StudioMainWindow()
    studio.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()