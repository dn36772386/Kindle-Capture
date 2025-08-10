# -*- coding: utf-8 -*-
import os, yaml, dataclasses, datetime, getpass
from dataclasses import dataclass, asdict

def _local_appdata():
    return os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")

@dataclass
class Paths:
    base_dir: str
    log_dir: str
    log_file: str
    output_dir: str
    settings_file: str

    @staticmethod
    def build():
        base = os.path.join(_local_appdata(), "SnapLite")
        log_dir = os.path.join(base, "logs")
        out_dir = os.path.join(os.path.expanduser("~/Pictures"), "SnapLite")
        os.makedirs(base, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        return Paths(
            base_dir=base,
            log_dir=log_dir,
            log_file=os.path.join(log_dir, "snaplite.log"),
            output_dir=out_dir,
            settings_file=os.path.join(base, "settings.yaml")
        )

@dataclass
class Settings:
    # UI / ホットキー
    hotkey_string: str = "ctrl+shift+s"  # 例: "ctrl+shift+s"
    theme: str = "auto"  # "auto"|"light"|"dark"

    # キャプチャ／操作
    wait_after_focus: float = 0.6
    # 固定待ち（差分待ちが無効な場合のフォールバック）
    wait_after_turn: float = 0.5
    next_key: str = "right"  # 'right'|'pagedown' など pyautoguiのキー名
    use_click: bool = False
    click_pos: tuple = dataclasses.field(default_factory=lambda: (1800, 1000))
    scroll_mode: bool = False
    scroll_pixels: int = 600
    max_pages: int = 2000

    # 自動トリミング
    auto_crop: bool = True
    min_contour_area_ratio: float = 0.1
    fallback_margin: int = 10
    split_double_page: bool = False

    # --- ストリーミング（差分待ち） ---
    change_polling: bool = True           # True なら差分待ちでページ描画完了を判定
    change_poll_interval: float = 0.15    # 監視間隔（秒）
    change_timeout: float = 5.0           # 最大待ち時間（秒）
    change_hamming_threshold: int = 4     # 変化判定のハミング閾値

    # --- 動的トリム（左右境界検出） ---
    dynamic_trim: bool = True             # True で左右境界を検出してトリム
    trim_margin_top: int = 1
    trim_margin_bottom: int = 16
    trim_margin_left: int = 20
    trim_margin_right: int = 20
    # --- 旧式一括トリム（レガシー最終再クロップ用フラグ群） ---
    legacy_final_recrop: bool = True
    legacy_keep_raw: bool = False
    legacy_raw_subdir: str = "_raw"
    dynamic_trim_method: str = "color"   # "color"|"variance" 将来的拡張用
    bg_tolerance: int = 3

    # --- 起動時の表紙移動とUI写り込み対策 ---
    go_to_cover_on_start: bool = True
    cover_seek_max_steps: int = 300
    cover_seek_stable_trials: int = 6
    ui_clean_check: bool = True
    ui_settle_timeout: float = 4.0
    ui_settle_check_interval: float = 0.15
    stable_frames_required: int = 3
    allow_fs_retoggle: bool = True
    after_fs_toggle_wait: float = 0.8
    overlay_dark_threshold: int = 70       # 0..255（暗さ閾値）
    overlay_center_ratio: float = 0.12     # 画面中央帯の高さ比
    overlay_bottom_ratio: float = 0.18     # 画面下部帯の高さ比
    overlay_pixel_ratio: float = 0.06      # 帯領域内の暗画素割合閾値

    # 旧式互換（最終一括トリム）
    legacy_final_recrop: bool = True       # True: 逐次保存せず、最後に全ページ同じ左右で一括トリム
    legacy_keep_raw: bool = False          # True: raw一時ファイルを残す
    legacy_raw_subdir: str = "_raw"        # raw保存のサブフォルダ名
    dynamic_trim_method: str = "color"  # "variance" or "color"
    bg_tolerance: int = 3                  # dynamic_trim_method="color" のときの背景色許容差


    # --- グレースケール判定 ---
    grayscale_detect: bool = True
    grayscale_threshold: int = 7          # RGB差の許容最大値（小さいほど厳しい）
    grayscale_margin_top: int = 0
    grayscale_margin_bottom: int = 0
    grayscale_margin_left: int = 0
    grayscale_margin_right: int = 0

    # --- 保存前リサイズ ---
    resize_width: int = 0                 # 0 なら無効。>0 で指定幅にリサイズ

    # --- フルスクリーン制御 ---
    use_fullscreen_toggle: bool = True
    fullscreen_key: str = "f11"
    fullscreen_wait: float = 1.0
    exit_fullscreen_on_finish: bool = True

    # --- ページジャンプ（任意、ZIP相当） ---
    pagejump_enabled: bool = False
    pagejump_combo: str = "ctrl+g"   # "ctrl+g" 形式
    pagejump_value: str = "1"
    pagejump_confirm_key: str = "enter"
    pagejump_wait: float = 0.6       # 送信後に待つ

    # --- 完了時の振る舞い ---
    open_folder_on_finish: bool = True    # 取得完了後に保存フォルダを開く

    # 重複判定（終端検知）
    dedup_hamming_threshold: int = 2
    dedup_patience: int = 3

    # 出力
    screenshot_format: str = "png"  # 'png' 推奨
    base_title: str = ""            # 空ならウィンドウタイトルを使う

    # パス
    paths: Paths = dataclasses.field(default_factory=Paths.build)

    def as_dict(self):
        d = asdict(self)
        # dataclassの入れ子を基本型に
        d["paths"] = asdict(self.paths)
        return d

    def update_from_dict(self, d: dict):
        for k, v in d.items():
            if not hasattr(self, k): 
                continue
            if k == "paths":
                for pk, pv in v.items():
                    if hasattr(self.paths, pk):
                        setattr(self.paths, pk, pv)
            else:
                setattr(self, k, v)

    def save(self):
        with open(self.paths.settings_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.as_dict(), f, allow_unicode=True)

    @staticmethod
    def load_or_create():
        s = Settings()
        p = s.paths.settings_file
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                s.update_from_dict(data)
            except Exception:
                pass
        else:
            s.save()
        return s
