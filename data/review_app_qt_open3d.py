import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import open3d as o3d
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QAction, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QTextEdit,
    QLineEdit,
    QGridLayout,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QGroupBox,
    QMessageBox,
    QRadioButton,
    QButtonGroup,
    QScrollArea,
    QSizePolicy,
    QFrame,
    QSplitter,
)

REJECT_REASONS = {
    0: "pytorch3d_render_error",
    1: "multi_object",
    2: "scene",
    3: "figure",
    4: "transparent",
    5: "single_color",
    6: "single_layer_sheet_like_surface",
    7: "malformed_or_weird_structure",
    8: "structure_causes_unreliable_difficulty",
    9: "other",
}

PREVIEW_FILE_MAP = {
    "Front": "front.png",
    "Side": "side.png",
    "Top": "top.png",
    "Iso": "iso.png",
}


@dataclass
class Paths:
    input_json: Path
    output_json: Path
    state_json: Path
    preview_root: Path
    order_by_output: bool = False


class ReviewStore:
    def __init__(self, paths: Paths, reviewer: int):
        self.paths = paths
        self.reviewer = reviewer
        self.records: List[Dict[str, Any]] = self._load_input(paths.input_json)

        self.reviews: Dict[str, Dict[str, Any]] = self._load_json(paths.output_json, default={})
        if isinstance(self.reviews, list):
            self.reviews = {x["uid"]: x for x in self.reviews if isinstance(x, dict) and "uid" in x}

        if getattr(paths, "order_by_output", False) and isinstance(self.reviews, dict) and len(self.reviews) > 0:
            self.records = self._reorder_by_review_file_order(self.records, self.reviews)

        self.by_uid = {r["uid"]: r for r in self.records}
        self.state = self._load_json(paths.state_json, default={})
        self.current_index = int(self.state.get("current_index", self._first_unreviewed_index()))
        self.current_index = max(0, min(self.current_index, max(0, len(self.records) - 1)))

    @staticmethod
    def _load_input(path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"Expected list in input JSON: {path}")
        return data

    @staticmethod
    def _load_json(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    
    def _reorder_by_review_file_order(
        self,
        current_records: List[Dict[str, Any]],
        review_dict: Dict[str, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Reorder current_records following the key order in review json.
        """
        ordered_uids = [str(uid) for uid in review_dict.keys()]
        current_by_uid = {str(r["uid"]): r for r in current_records}

        reordered: List[Dict[str, Any]] = []
        for uid in ordered_uids:
            if uid in current_by_uid:
                reordered.append(current_by_uid[uid])

        used_uids = {str(r["uid"]) for r in reordered}
        for rec in current_records:
            uid = str(rec["uid"])
            if uid not in used_uids:
                reordered.append(rec)

        return reordered

    def _first_unreviewed_index(self) -> int:
        for i, rec in enumerate(self.records):
            if rec["uid"] not in self.reviews:
                return i
        return 0

    def get_record(self, index: int) -> Dict[str, Any]:
        return self.records[index]

    def get_review(self, uid: str) -> Optional[Dict[str, Any]]:
        return self.reviews.get(uid)

    def save_review(self, uid: str, manual_label: str, reject_reason: Optional[str], note: str) -> None:
        self.reviews[uid] = {
            "uid": uid,
            "reviewer": self.reviewer,
            "manual_label": manual_label,
            "manual_reject_reason": reject_reason,
            "manual_note": note,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        }
        self._flush()

    def set_current_index(self, index: int) -> None:
        self.current_index = index
        self._flush_state()

    def _flush(self) -> None:
        self.paths.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.output_json.open("w", encoding="utf-8") as f:
            json.dump(self.reviews, f, ensure_ascii=False, indent=2)
        self._flush_state()

    def _flush_state(self) -> None:
        payload = {
            "reviewer": self.reviewer,
            "current_index": self.current_index,
            "reviewed_count": len(self.reviews),
            "total_count": len(self.records),
        }
        self.paths.state_json.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.state_json.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)


class ImagePane(QLabel):
    def __init__(self, title: str):
        super().__init__()
        self.title = title
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setFrameShape(QFrame.Shape.Box)
        self.setText(title)
        self._pixmap: Optional[QPixmap] = None

    def set_pixmap(self, pixmap: Optional[QPixmap], fallback_text: str = "") -> None:
        self._pixmap = pixmap
        if pixmap is None:
            self.setText(f"{self.title}\n{fallback_text}".strip())
            self.setPixmap(QPixmap())
        else:
            self._update_scaled_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._pixmap is not None:
            self._update_scaled_pixmap()

    def _update_scaled_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(scaled)
        self.setText("")


class ReviewApp(QMainWindow):
    def __init__(self, store: ReviewStore):
        super().__init__()
        self.store = store
        self.mesh_cache: Dict[str, Optional[o3d.geometry.TriangleMesh]] = {}
        self.view_cache: Dict[str, Dict[str, Optional[QPixmap]]] = {}

        self.setWindowTitle(f"ObjView Manual Review - reviewer {store.reviewer}")
        self.resize(1800, 1050)

        self.uid_label = QLabel("uid")
        self.progress_label = QLabel("")
        self.viewer_hint = QLabel("3D interactive viewer uses Open3D and opens in a separate window.")
        self.viewer_hint.setWordWrap(True)

        self.meta_labels: Dict[str, QLabel] = {}
        self.risk_flags_label = QLabel("")
        self.risk_flags_label.setWordWrap(True)
        self.risk_flags_label.setTextFormat(Qt.TextFormat.RichText)
        self.reviewed_label = QLabel("")
        self.reviewed_label.setTextFormat(Qt.TextFormat.RichText)

        self.image_panes = {name: ImagePane(name) for name in PREVIEW_FILE_MAP.keys()}
        self.reject_group = QButtonGroup(self)
        self.reject_radios: Dict[int, QRadioButton] = {}
        self.reject_panel = QWidget()
        self.other_note = QTextEdit()
        self.other_note.setPlaceholderText("Note required for reason 0 or 9.")
        self.other_note.setMaximumHeight(80)
        self.other_note.hide()

        self._build_ui()
        self._bind_shortcuts()
        self.load_current_record()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)

        header = QHBoxLayout()
        header.addWidget(self.uid_label, 3)
        header.addWidget(self.progress_label, 2)
        outer.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, 1)

        left = QWidget()
        left_layout = QGridLayout(left)
        left_layout.setContentsMargins(4, 4, 4, 4)
        left_layout.setSpacing(6)
        left_layout.addWidget(self.image_panes["Front"], 0, 0)
        left_layout.addWidget(self.image_panes["Side"], 0, 1)
        left_layout.addWidget(self.image_panes["Top"], 1, 0)
        left_layout.addWidget(self.image_panes["Iso"], 1, 1)
        splitter.addWidget(left)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right = QWidget()
        right_scroll.setWidget(right)
        splitter.addWidget(right_scroll)
        splitter.setSizes([1100, 700])

        right_layout = QVBoxLayout(right)
        right_layout.setSpacing(10)

        meta_box = QGroupBox("Metadata")
        meta_form = QFormLayout(meta_box)
        fields = [
            "pool_type",
            "shape_type",
            "fill_bucket",
            "self_occlusion_attribute",
            "observation_saturation_view_num",
            "selected_view_count",
            "gt_surface_voxel_count",
            "bc_ratio",
            "manual_priority_score",
        ]
        for key in fields:
            label = QLabel("-")
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            self.meta_labels[key] = label
            meta_form.addRow(key, label)
        right_layout.addWidget(meta_box)

        risk_box = QGroupBox("Risk flags")
        risk_layout = QVBoxLayout(risk_box)
        risk_layout.addWidget(self.risk_flags_label)
        risk_layout.addWidget(self.reviewed_label)
        right_layout.addWidget(risk_box)

        viewer_box = QGroupBox("3D View")
        viewer_layout = QVBoxLayout(viewer_box)
        open_viewer_btn = QPushButton("Open interactive Open3D viewer")
        open_viewer_btn.clicked.connect(self.open_open3d_viewer)
        refresh_btn = QPushButton("Refresh preview images")
        refresh_btn.clicked.connect(self.refresh_current_views)
        viewer_layout.addWidget(open_viewer_btn)
        viewer_layout.addWidget(refresh_btn)
        viewer_layout.addWidget(self.viewer_hint)
        right_layout.addWidget(viewer_box)

        action_box = QGroupBox("Review")
        action_layout = QVBoxLayout(action_box)

        btn_row = QHBoxLayout()
        yes_btn = QPushButton("Yes [Y]")
        yes_btn.clicked.connect(self.mark_yes)
        maybe_btn = QPushButton("Maybe [M]")
        maybe_btn.clicked.connect(self.mark_maybe)
        reject_mode_btn = QPushButton("Reject mode [R]")
        reject_mode_btn.clicked.connect(self.toggle_reject_panel)
        btn_row.addWidget(yes_btn)
        btn_row.addWidget(maybe_btn)
        btn_row.addWidget(reject_mode_btn)
        action_layout.addLayout(btn_row)

        reject_layout = QVBoxLayout(self.reject_panel)
        reject_layout.addWidget(QLabel("Reject reason (0-9):"))
        for idx, name in REJECT_REASONS.items():
            rb = QRadioButton(f"{idx}. {name}")
            self.reject_group.addButton(rb, idx)
            rb.toggled.connect(self._on_reject_reason_toggled)
            reject_layout.addWidget(rb)
            self.reject_radios[idx] = rb
        reject_layout.addWidget(self.other_note)

        reject_btn_row = QHBoxLayout()
        confirm_reject_btn = QPushButton("Confirm reject")
        confirm_reject_btn.clicked.connect(self.confirm_reject)
        cancel_reject_btn = QPushButton("Hide reject panel")
        cancel_reject_btn.clicked.connect(self.hide_reject_panel)
        reject_btn_row.addWidget(confirm_reject_btn)
        reject_btn_row.addWidget(cancel_reject_btn)
        reject_layout.addLayout(reject_btn_row)

        self.reject_panel.hide()
        action_layout.addWidget(self.reject_panel)
        right_layout.addWidget(action_box)

        nav_box = QGroupBox("Navigation")
        nav_layout = QVBoxLayout(nav_box)
        nav_row = QHBoxLayout()
        prev_btn = QPushButton("Previous [P]")
        prev_btn.clicked.connect(self.go_prev)
        next_btn = QPushButton("Next [N]")
        next_btn.clicked.connect(self.go_next)
        nav_row.addWidget(prev_btn)
        nav_row.addWidget(next_btn)
        nav_layout.addLayout(nav_row)

        jump_row = QHBoxLayout()
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("Jump to uid or index")
        jump_btn = QPushButton("Jump")
        jump_btn.clicked.connect(self.jump_to)
        jump_row.addWidget(self.jump_input)
        jump_row.addWidget(jump_btn)
        nav_layout.addLayout(jump_row)
        right_layout.addWidget(nav_box)

        right_layout.addStretch(1)

        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        reload_action = QAction("Reload current", self)
        reload_action.triggered.connect(self.load_current_record)
        file_menu.addAction(reload_action)

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("Y"), self, activated=self.mark_yes)
        QShortcut(QKeySequence("M"), self, activated=self.mark_maybe)
        QShortcut(QKeySequence("R"), self, activated=self.toggle_reject_panel)
        QShortcut(QKeySequence("N"), self, activated=self.go_next)
        QShortcut(QKeySequence("P"), self, activated=self.go_prev)

        QShortcut(QKeySequence("0"), self, activated=lambda: self.select_reject_reason(0))
        for i in range(1, 10):
            QShortcut(QKeySequence(str(i)), self, activated=lambda i=i: self.select_reject_reason(i))

    def current_record(self) -> Dict[str, Any]:
        return self.store.get_record(self.store.current_index)
    
    def _format_colored_value(self, key: str, value: Any) -> str:
        if value is None:
            return "-"

        # Default text formatting
        if isinstance(value, float):
            text = f"{value:.6f}" if key in {"self_occlusion_attribute", "bc_ratio"} else f"{value:.3f}"
        else:
            text = str(value)

        color = None
        bold = False

        if key == "manual_priority_score":
            score = int(value)
            if score >= 5:
                color = "#c62828"   # red
                bold = True
            elif score >= 3:
                color = "#ef6c00"   # orange
                bold = True
            elif score >= 1:
                color = "#1565c0"   # blue
        elif key == "gt_surface_voxel_count":
            vox = float(value)
            if vox < 1000:
                color = "#c62828"
                bold = True
            elif vox < 1500:
                color = "#ef6c00"
        elif key == "observation_saturation_view_num":
            b = float(value)
            if b >= 120:
                color = "#6a1b9a"   # purple
                bold = True
            elif b >= 80:
                color = "#1565c0"   # blue
        elif key == "selected_view_count":
            c = float(value)
            if c >= 100:
                color = "#c62828"
                bold = True
            elif c >= 40:
                color = "#ef6c00"

        if color is None:
            return text

        weight = "700" if bold else "500"
        return f'<span style="color:{color}; font-weight:{weight};">{text}</span>'


    def _format_risk_flags_html(self, flags: List[str]) -> str:
        """
        Render risk flags as colored badges.
        """
        if not flags:
            return '<span style="color:#666666;">(none)</span>'

        def style_for_flag(flag: str) -> str:
            # True risk flags: red-ish palette
            if flag == "LOW_SURFACE_VOXEL":
                return "background:#ffebee; color:#c62828; border:1px solid #ef9a9a;"
            if flag == "C_EXCEEDS_B_SIGNIFICANTLY":
                return "background:#ffebee; color:#b71c1c; border:1px solid #e57373;"
            if flag == "HIGH_C_NEAR_VIEWSET_LIMIT":
                return "background:#fff3e0; color:#ef6c00; border:1px solid #ffb74d;"

            # Structural indicator flags: blue/purple/orange palette
            if flag == "LONG_TAIL_POOL":
                return "background:#f3e5f5; color:#6a1b9a; border:1px solid #ce93d8;"
            if flag == "HIGH_B_LOW_C":
                return "background:#e3f2fd; color:#1565c0; border:1px solid #90caf9;"
            if flag == "HIGH_C_LOW_B":
                return "background:#e0f7fa; color:#00838f; border:1px solid #80deea;"
            if flag == "HIGH_B_HIGH_C":
                return "background:#fff3e0; color:#ef6c00; border:1px solid #ffcc80;"

            return "background:#eeeeee; color:#424242; border:1px solid #bdbdbd;"

        chunks = []
        for flag in flags:
            style = style_for_flag(flag)
            chunks.append(
                f'<span style="display:inline-block; padding:6px 16px; margin:6px 12px 6px 3px; '
                f'border-radius:8px; {style}">{flag}</span>'
            )

        return "<div style='line-height: 1.8;'>" + "".join(chunks) + "</div>"

    def load_current_record(self) -> None:
        rec = self.current_record()
        uid = rec["uid"]
        self.uid_label.setText(f"UID: {uid}    Reviewer: {self.store.reviewer}")
        self.progress_label.setText(
            f"Index: {self.store.current_index + 1}/{len(self.store.records)}    "
            f"Reviewed: {len(self.store.reviews)}"
        )

        for key, lbl in self.meta_labels.items():
            value = rec.get(key)
            lbl.setText(self._format_colored_value(key, value))

        flags = rec.get("risk_flags", [])
        self.risk_flags_label.setText(self._format_risk_flags_html(flags))

        existing = self.store.get_review(uid)
        if existing is None:
            self.reviewed_label.setText('<span style="color:#666666;">Current review: not reviewed yet</span>')
        else:
            label = existing.get("manual_label")
            reason = existing.get("manual_reject_reason")

            if label == "yes":
                color = "#2e7d32"
            elif label == "maybe":
                color = "#ef6c00"
            else:
                color = "#c62828"

            text = f"Current review: {label}"
            if reason:
                text += f" / {reason}"

            self.reviewed_label.setText(f'<span style="color:{color}; font-weight:600;">{text}</span>')

        self.hide_reject_panel(clear_selection=True)
        self._load_views_for_record(rec)

    def _load_views_for_record(self, rec: Dict[str, Any]) -> None:
        uid = rec["uid"]
        if uid in self.view_cache:
            for name, pane in self.image_panes.items():
                pane.set_pixmap(self.view_cache[uid].get(name), fallback_text="preview unavailable")
            return

        pixmaps = self._load_preview_pixmaps(uid)
        self.view_cache[uid] = pixmaps
        for name, pane in self.image_panes.items():
            pane.set_pixmap(pixmaps.get(name), fallback_text="preview unavailable")

    def refresh_current_views(self) -> None:
        uid = self.current_record()["uid"]
        if uid in self.view_cache:
            del self.view_cache[uid]
        self._load_views_for_record(self.current_record())

    def _load_preview_pixmaps(self, uid: str) -> Dict[str, Optional[QPixmap]]:
        preview_dir = self.store.paths.preview_root / uid
        out: Dict[str, Optional[QPixmap]] = {}
        for pane_name, file_name in PREVIEW_FILE_MAP.items():
            png_path = preview_dir / file_name
            if png_path.exists():
                pixmap = QPixmap(str(png_path))
                out[pane_name] = pixmap if not pixmap.isNull() else None
            else:
                out[pane_name] = None
        return out

    def _load_mesh(self, obj_path: str) -> Optional[o3d.geometry.TriangleMesh]:
        if obj_path in self.mesh_cache:
            return self.mesh_cache[obj_path]

        mesh: Optional[o3d.geometry.TriangleMesh] = None
        try:
            mesh = o3d.io.read_triangle_mesh(obj_path, enable_post_processing=True)
            if mesh is None or mesh.is_empty():
                mesh = None
            else:
                if not mesh.has_triangle_normals():
                    mesh.compute_triangle_normals()
                if not mesh.has_vertex_normals():
                    mesh.compute_vertex_normals()
                if not mesh.has_vertex_colors() and not mesh.has_textures():
                    mesh.paint_uniform_color([0.78, 0.78, 0.80])
        except Exception:
            mesh = None

        self.mesh_cache[obj_path] = mesh
        return mesh

    def open_open3d_viewer(self) -> None:
        rec = self.current_record()
        mesh = self._load_mesh(rec.get("obj_path"))
        if mesh is None:
            QMessageBox.warning(self, "Open3D", "Failed to load mesh.")
            return
        try:
            o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, window_name=rec["uid"])
        except Exception as e:
            QMessageBox.critical(self, "Open3D", f"Failed to open viewer:\n{e}")

    def mark_yes(self) -> None:
        rec = self.current_record()
        self.store.save_review(rec["uid"], "yes", None, "")
        self.go_next(auto_saved=True)

    def mark_maybe(self) -> None:
        rec = self.current_record()
        self.store.save_review(rec["uid"], "maybe", None, "")
        self.go_next(auto_saved=True)

    def toggle_reject_panel(self) -> None:
        self.reject_panel.setVisible(not self.reject_panel.isVisible())

    def hide_reject_panel(self, clear_selection: bool = False) -> None:
        self.reject_panel.hide()
        if clear_selection:
            self.reject_group.setExclusive(False)
            for rb in self.reject_radios.values():
                rb.setChecked(False)
            self.reject_group.setExclusive(True)
            self.other_note.clear()
            self.other_note.hide()

    def _on_reject_reason_toggled(self) -> None:
        checked_id = self.reject_group.checkedId()
        self.other_note.setVisible(checked_id == 9)

    def select_reject_reason(self, idx: int) -> None:
        if idx not in self.reject_radios:
            return
        self.reject_panel.show()
        self.reject_radios[idx].setChecked(True)
        if idx != 9:
            self.confirm_reject()

    def confirm_reject(self) -> None:
        checked_id = self.reject_group.checkedId()
        if checked_id == -1:
            QMessageBox.information(self, "Reject", "Please choose a reject reason.")
            return
        reason = REJECT_REASONS[checked_id]
        note = ""
        if checked_id == 9:
            note = self.other_note.toPlainText().strip()
            if not note:
                QMessageBox.information(self, "Reject", "Reason 9 requires a note.")
                return
        rec = self.current_record()
        self.store.save_review(rec["uid"], "reject", reason, note)
        self.go_next(auto_saved=True)

    def go_next(self, auto_saved: bool = False) -> None:
        if self.store.current_index < len(self.store.records) - 1:
            self.store.set_current_index(self.store.current_index + 1)
            self.load_current_record()
        elif auto_saved:
            QMessageBox.information(self, "Done", "Reached the last record.")

    def go_prev(self) -> None:
        if self.store.current_index > 0:
            self.store.set_current_index(self.store.current_index - 1)
            self.load_current_record()

    def jump_to(self) -> None:
        text = self.jump_input.text().strip()
        if not text:
            return

        if text.isdigit():
            idx = int(text)
            idx0 = idx - 1 if idx > 0 else idx
            if 0 <= idx0 < len(self.store.records):
                self.store.set_current_index(idx0)
                self.load_current_record()
                return

        for i, rec in enumerate(self.store.records):
            if rec["uid"] == text:
                self.store.set_current_index(i)
                self.load_current_record()
                return

        QMessageBox.warning(self, "Jump", f"Could not find index/uid: {text}")

    def closeEvent(self, event):
        self.store.set_current_index(self.store.current_index)
        super().closeEvent(event)


def resolve_paths(args: argparse.Namespace) -> Paths:
    input_json = Path(args.input)
    output_json = Path(args.output) if args.output else input_json.parent / f"review_results_reviewer_{args.reviewer}.json"
    state_json = output_json.with_name(output_json.stem + "_state.json")
    preview_root = Path(args.preview_root)
    return Paths(
        input_json=input_json,
        output_json=output_json,
        state_json=state_json,
        preview_root=preview_root,
        order_by_output=args.order_by_output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual review app for ObjView benchmark candidates.")
    parser.add_argument("--reviewer", type=int, required=True, help="Required integer reviewer id, e.g. 0, 1, 2")
    parser.add_argument(
        "--input",
        type=str,
        default="geometry_sampled/manual_review_candidates_with_risk.json",
        help="Path to input candidate json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output review json path. Defaults to review_results_reviewer_<id>.json",
    )
    parser.add_argument(
        "--preview-root",
        type=str,
        default="geometry_sampled/previews",
        help="Root directory containing per-uid preview PNGs",
    )
    parser.add_argument(
        "--order-by-output",
        action="store_true",
        help="If set, display records in the uid order of the output review json.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = resolve_paths(args)

    if not paths.input_json.exists():
        print(f"Input JSON not found: {paths.input_json}", file=sys.stderr)
        return 1

    app = QApplication(sys.argv)
    store = ReviewStore(paths, reviewer=args.reviewer)
    win = ReviewApp(store)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

"""
python review_app_qt_open3d.py --reviewer 0 --input geometry_sampled/manual_review_candidates_with_risk.json --preview-root geometry_sampled/previews --output geometry_sampled/review_results_reviewer_0.json

python review_app_qt_open3d.py --reviewer 999 --input geometry_sampled/manual_review_candidates_with_risk.json --preview-root geometry_sampled/previews --output geometry_sampled/review_results_reviewer_999.json --order-by-output
"""
