"""
core.capture
------------
Kindle の画面をウィンドウ矩形でキャプチャし、raw を連番で保存します。
遅延対策として非同期保存（RawSaver）を内蔵します。
"""
from __future__ import annotations
import os, time, datetime, queue, threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import mss
import pyautogui

# ---- 例外 ------------------------------------------------------------------

class CaptureError(Exception):
    """Capture 処理の致命エラー用"""
    pass

# 依存（ユーザー環境の既存モジュール）
try:
    from .windows import get_window_rect, minimize_window, bring_to_front  # type: ignore
except Exception:
    def get_window_rect(hwnd:int)->Dict[str,int]:
        raise CaptureError("get_window_rect not available on this environment")
    def minimize_window(hwnd:int)->None:
        pass
    def bring_to_front(hwnd:int)->None:
        pass

try:
    from .logging_conf import get_logger  # type: ignore
    logger = get_logger()
except Exception:
    class _Dummy:
        def info(self,*a,**k): print(*a)
        def warning(self,*a,**k): print(*a)
        def error(self,*a,**k): print(*a)
        def debug(self,*a,**k): pass
    logger = _Dummy()


# ---- 非同期セーバー -----------------------------------------------------------

class RawSaver:
    """別スレッドでディスク保存。BMP/PNG/JPEG に対応。"""
    def __init__(self, fmt: str = "png", fast_level: int = 1, maxsize: int = 6):
        self.q: "queue.Queue[Tuple[Optional[Image.Image], Optional[str]]]" = queue.Queue(maxsize=maxsize)
        self.fmt = fmt.lower()
        self.fast_level = int(fast_level)
        self.dead = False
        self.t = threading.Thread(target=self._worker, daemon=True)
        self.t.start()

    def submit(self, img: Image.Image, path: str):
        # 呼び出し側が早く解放できるよう copy を投入
        self.q.put((img.copy(), path), timeout=10)

    def close(self):
        self.dead = True
        try:
            self.q.put((None, None), timeout=2)
        except Exception:
            pass
        self.t.join(timeout=30)

    def _worker(self):
        while True:
            img, path = self.q.get()
            if img is None or self.dead:
                break
            try:
                d = os.path.dirname(path); os.makedirs(d, exist_ok=True)
                if self.fmt == "png":
                    img.save(path, format="PNG", optimize=False, compress_level=self.fast_level)
                elif self.fmt in ("jpg","jpeg"):
                    img.save(path, format="JPEG", quality=92, optimize=True)
                elif self.fmt == "bmp":
                    img.save(path, format="BMP")
                else:
                    img.save(path)
            except Exception as e:
                logger.warning(f"save failed: {path}: {e}")
            finally:
                try: img.close()
                except: pass
                self.q.task_done()


# ---- キャプチャ基本 -----------------------------------------------------------

def _grab_rect(sct: mss.mss, rect: Dict[str, int]) -> Image.Image:
    """mss で指定矩形を取得して PIL.Image で返す"""
    mon = {"left": rect["left"], "top": rect["top"], "width": rect["width"], "height": rect["height"]}
    img = sct.grab(mon)
    # bgra → rgb
    arr = np.asarray(img)[:, :, :3][:, :, ::-1]
    return Image.fromarray(arr, "RGB")

def _safe_open_mss():
    try:
        return mss.mss()
    except Exception as e:
        raise CaptureError(f"failed to initialize mss: {e}")

def _sig_for_change(img: Image.Image, size: int = 32) -> np.ndarray:
    """
    変化検出用のシグネチャ。グレイスケール縮小のベクトル。
    高頻度でも軽い。
    """
    g = img.convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(g, dtype=np.int16).ravel()

def _dist(a: np.ndarray, b: np.ndarray) -> int:
    if a is None or b is None:
        return 9999
    # マンハッタン距離（小さいほど近い）
    return int(np.abs(a - b).mean())

def _wait_for_clean_frame(
    sct: mss.mss,
    rect: Dict[str, int],
    opt: Dict[str, Any],
    max_wait: float = 2.0,
) -> Image.Image:
    """
    初期の写り込み（メニュー/トースト等）が消えるのを待つ。
    暗い帯検知＋数フレーム安定で判定する簡易版。
    """
    interval = float(opt.get("clean_poll_interval", 0.06))
    streak_need = int(opt.get("clean_required_streak", 1))
    extra_ms = int(opt.get("clean_extra_wait_ms", 80))
    thr_dark = int(opt.get("clean_dark_threshold", 35))
    min_dark_rows = int(opt.get("clean_dark_min_rows", 8))

    streak = 0
    deadline = time.time() + float(opt.get("clean_timeout", max_wait))
    last = None
    while time.time() < deadline:
        im = _grab_rect(sct, rect)
        g = np.asarray(im.convert("L"))
        # 下端の暗帯（Kindle の UI）を検知
        dark_rows = (g < thr_dark).mean(axis=1)
        dark_len = (dark_rows[::-1] > 0.2).argmin() if (dark_rows[::-1] > 0.2).any() else 0
        # 小さな暗帯は許容
        ok = dark_len <= min_dark_rows
        # 直近との差分が小さいかも見る
        sig = _sig_for_change(im, size=24)
        if last is not None and _dist(sig, last) < 2 and ok:
            streak += 1
        else:
            streak = 1 if ok else 0
        last = sig
        if streak >= streak_need:
            time.sleep(extra_ms / 1000.0)
            return im
        time.sleep(interval)
    # タイムアウト時は最後のフレーム
    return im

def _wait_until_change(
    sct: mss.mss,
    rect: Dict[str, int],
    prev_sig: Optional[np.ndarray],
    opt: Dict[str, Any],
) -> Tuple[bool, Optional[Image.Image], Optional[np.ndarray], int]:
    """
    ページ送り後、画面が十分変わるまで待機。
    戻り値: (変化したか, 画像, シグネチャ, 距離)
    """
    if prev_sig is None:
        im = _grab_rect(sct, rect)
        return True, im, _sig_for_change(im, size=int(opt.get("change_downscale", 24))), 9999

    deadline = time.time() + float(opt.get("change_timeout", 7.0))
    interval = float(opt.get("change_poll_interval", 0.05))
    thr = int(opt.get("change_distance_threshold", 4))
    while time.time() < deadline:
        im = _grab_rect(sct, rect)
        sig = _sig_for_change(im, size=int(opt.get("change_downscale", 24)))
        d = _dist(sig, prev_sig)
        if d >= thr:
            return True, im, sig, d
        time.sleep(interval)
    # 変化しない場合は最後のものを返す
    return False, im, sig, _dist(sig, prev_sig)

def _toggle_fullscreen(opt: Dict[str,Any], enter: bool):
    if not bool(opt.get("use_fullscreen_toggle", True)):
        return
    key = str(opt.get("fullscreen_key", "f11"))
    try:
        pyautogui.press(key)
        time.sleep(max(0.2, float(opt.get("fullscreen_wait", 1.0))))
        logger.info(f"fullscreen toggled {'ON' if enter else 'OFF'}")
    except Exception as e:
        logger.warning(f"fullscreen toggle failed: {e}")

def _turn_page(opt: Dict[str,Any], prev: bool=False):
    key = str(opt.get("prev_page_key" if prev else "next_page_key", "right"))
    try:
        pyautogui.press(key)
    except Exception as e:
        logger.warning(f"turn page failed: {e}")

def _go_cover(opt: Dict[str,Any]):
    """任意：カバーに移動。設定によってはスキップ。"""
    if not bool(opt.get("goto_cover_before_start", True)):
        return
    seq = opt.get("goto_cover_sequence", ["home"])  # ["ctrl","home"] 等
    try:
        if isinstance(seq, (list, tuple)) and len(seq) > 1:
            pyautogui.hotkey(*[str(x) for x in seq])
        else:
            pyautogui.press(str(seq[0] if isinstance(seq, (list, tuple)) else str(seq)))
    except Exception as e:
        logger.warning(f"goto cover failed: {e}")

# ---- 入口 -----------------------------------------------------------------

def run_capture(hwnd: int, opt: Dict[str,Any], title: str, progress_cb=None) -> Dict[str,Any]:
    """
    Kindle ウィンドウ hwnd を対象に、連番 raw 画像を保存する。
    戻り値: {"captured_pages": int, "raw_paths": List[str], "output_dir": str}
    """
    # ウィンドウ矩形
    try:
        rect = get_window_rect(hwnd)
    except Exception as e:
        raise CaptureError(f"failed to get window rect: {e}")
    # 出力先決定
    out_dir = opt.get("output_dir") or os.path.join(os.path.expanduser("~"), "Pictures", "SnapLite")
    # raw 保存先
    session = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if bool(opt.get("use_temp_raw_dir", True)):
        raw_root = opt.get("temp_raw_root") or os.path.join(os.path.expanduser("~"), "AppData", "Local", "Temp", "SnapLite")
        raw_dir = os.path.join(raw_root, "_raw", session)
    else:
        raw_dir = os.path.join(out_dir, "_raw")

    raw_fmt = str(opt.get("raw_format", "png")).lower()
    fast_level = int(opt.get("raw_png_fast_level", 1))

    saver = RawSaver(fmt=raw_fmt, fast_level=fast_level, maxsize=int(opt.get("save_queue_size", 6)))

    raw_paths: List[str] = []
    base_title = title
    ts = session
    page = 1
    captured = 0
    duplicate_thr = int(opt.get("duplicate_end_streak", 3))
    duplicate_streak = 0
    last_sig: Optional[np.ndarray] = None

    # フルスクリーン
    if bool(opt.get("enter_fullscreen_on_start", True)):
        _toggle_fullscreen(opt, enter=True)

    # カバーへ移動（任意）
    _go_cover(opt)

    with _safe_open_mss() as sct:
        # 初回はクリーン待ち
        im = _wait_for_clean_frame(sct, rect, opt)
        # 保存
        raw_name = f"{base_title}_{ts}_{page:04}.{raw_fmt}"
        raw_path = os.path.join(raw_dir, raw_name)
        saver.submit(im, raw_path)
        raw_paths.append(raw_path); captured += 1
        w, h = im.size
        logger.info(f"saved raw: {raw_path} size=({w}, {h})")
        last_sig = _sig_for_change(im, size=int(opt.get("change_downscale", 24)))
        if progress_cb:
            progress_cb(captured)

        # 以降ループ
        while True:
            _turn_page(opt, prev=False)
            changed, im2, sig2, dist = _wait_until_change(sct, rect, last_sig, opt)
            if changed:
                logger.info(f"change detected: dist={dist} (thr={opt.get('change_distance_threshold', 4)})")
            else:
                logger.warning("change wait timeout; continue with last frame")

            # クリーン待ち（短め）
            im2 = _wait_for_clean_frame(sct, rect, opt)
            sig2 = _sig_for_change(im2, size=int(opt.get("change_downscale", 24)))
            # 重複チェック
            ddup = _dist(sig2, last_sig)
            if ddup <= int(opt.get("duplicate_distance_threshold", 2)):
                duplicate_streak += 1
                logger.info(f"duplicate? dist={ddup}, streak={duplicate_streak}/{duplicate_thr}")
                if duplicate_streak >= duplicate_thr:
                    logger.info("end detected by duplicate streak")
                    break
            else:
                duplicate_streak = 0

            page += 1
            raw_name = f"{base_title}_{ts}_{page:04}.{raw_fmt}"
            raw_path = os.path.join(raw_dir, raw_name)
            saver.submit(im2, raw_path)
            raw_paths.append(raw_path); captured += 1
            w, h = im2.size
            logger.info(f"saved raw: {raw_path} size=({w}, {h})")
            last_sig = sig2
            if progress_cb and (page % int(opt.get("progress_update_every_pages", 10) or 1) == 0):
                progress_cb(captured)

    # セーバーを終了（未処理があれば flush）
    saver.close()

    # フルスクリーン解除
    if bool(opt.get("exit_fullscreen_on_finish", True)):
        _toggle_fullscreen(opt, enter=False)

    return {"captured_pages": captured, "raw_paths": raw_paths, "output_dir": out_dir}
