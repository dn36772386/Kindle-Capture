# core/config.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, dataclasses, yaml
from dataclasses import dataclass, asdict
from typing import Dict, Any

def _local_appdata() -> str:
    return os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")

def _base_app_dir() -> str:
    return os.path.join(_local_appdata(), "SnapLite")

def _pictures_dir() -> str:
    return os.path.join(os.path.expanduser("~"), "Pictures", "SnapLite")

@dataclass
class _Paths:
    """app.py 互換の paths 属性。"""
    base_dir: str
    log_dir: str
    log_file: str
    tmp_dir: str
    output_dir: str
    settings_file: str

@dataclass
class Settings:
    # 主要パス
    output_dir: str = dataclasses.field(default_factory=_pictures_dir)
    settings_file: str = dataclasses.field(default_factory=lambda: os.path.join(_base_app_dir(), "settings.yaml"))
    # 画質/保存
    format: str = "png"
    raw_format: str = "png"
    raw_png_fast_level: int = 1
    final_png_fast_level: int = 1
    use_temp_raw_dir: bool = True
    temp_raw_root: str = dataclasses.field(default_factory=lambda: os.path.join(_base_app_dir(), "tmp"))
    legacy_raw_subdir: str = "_raw"
    # トリム（ZIP準拠の既定）
    trim_margin_top: int = 1
    trim_margin_bottom: int = 16
    trim_margin_left: int = 20
    trim_margin_right: int = 20
    bg_tolerance: int = 3
    auto_bottom_margin: bool = True
    bottom_margin_min: int = 16
    bottom_dark_threshold: int = 70
    bottom_dark_run_rows: int = 6
    split_double_page: bool = False
    # 変化検出
    change_downscale: int = 24
    change_distance_threshold: int = 4
    change_poll_interval: float = 0.05
    change_timeout: float = 7.0
    duplicate_distance_threshold: int = 2
    duplicate_end_streak: int = 3
    # クリーンフレーム待機
    clean_poll_interval: float = 0.06
    clean_required_streak: int = 1
    clean_extra_wait_ms: int = 80
    clean_timeout: float = 2.0
    clean_dark_threshold: int = 35
    clean_dark_min_rows: int = 8
    # UI
    progress_update_every_pages: int = 10
    # キー/フルスクリーン/カバー遷移
    use_fullscreen_toggle: bool = True
    enter_fullscreen_on_start: bool = True
    exit_fullscreen_on_finish: bool = True
    fullscreen_key: str = "f11"
    fullscreen_wait: float = 1.0
    next_page_key: str = "right"
    prev_page_key: str = "left"
    goto_cover_before_start: bool = True
    goto_cover_sequence: tuple = ("home",)
    # 後処理時のフォーカス
    minimize_kindle_on_postprocess: bool = True
    bring_app_front_on_postprocess: bool = True
    # 非同期保存
    save_queue_size: int = 6

    # --- 後方互換: app.py の settings.paths.* を提供 ---
    @property
    def paths(self) -> _Paths:
        base = _base_app_dir()
        log_dir = os.path.join(base, "logs")
        log_file = os.path.join(log_dir, "snaplite.log")
        tmp_dir = self.temp_raw_root
        return _Paths(
            base_dir=base,
            log_dir=log_dir,
            log_file=log_file,
            tmp_dir=tmp_dir,
            output_dir=self.output_dir,
            settings_file=self.settings_file,
        )

    def ensure_runtime_dirs(self) -> None:
        os.makedirs(os.path.dirname(self.settings_file), exist_ok=True)
        os.makedirs(self.paths.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_raw_root, exist_ok=True)

    # ユーティリティ
    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def load_or_create() -> "Settings":
        s = Settings()
        path = s.settings_file
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                for k, v in data.items():
                    if hasattr(s, k):
                        setattr(s, k, v)
            else:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(asdict(s), f, allow_unicode=True)
        except Exception:
            pass
        # 必要なディレクトリを用意
        try:
            s.ensure_runtime_dirs()
        except Exception:
            pass
        return s
