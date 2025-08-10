# -*- coding: utf-8 -*-
import time
import ctypes
import ctypes.wintypes as wt
from typing import List, Tuple
import win32gui, win32con, win32api
from win32con import SW_SHOW, SW_RESTORE, SW_MAXIMIZE, SW_SHOWMAXIMIZED

def list_windows(query: str) -> List[Tuple[int, str]]:
    res = []
    q = (query or "").lower()
    def enum_cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title and (q in title.lower()):
                res.append((hwnd, title))
        return True
    win32gui.EnumWindows(enum_cb, None)
    return res

def get_window_rect(hwnd: int):
    l, t, r, b = win32gui.GetWindowRect(hwnd)
    return {"left": l, "top": t, "width": r-l, "height": b-t}

def bring_to_front_and_maximize(hwnd: int, wait_after_focus: float = 0.6):
    """前面化・最大化を確実化。ズーム状態を確認しつつリトライ。"""
    user32 = ctypes.windll.user32
    IsZoomed = user32.IsZoomed
    IsIconic = user32.IsIconic

    # 1) 復元 or 表示
    try:
        if IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, SW_RESTORE)
        else:
            win32gui.ShowWindow(hwnd, SW_SHOW)
    except Exception:
        pass

    # 2) フォアグラウンド許可と前面化
    try:
        user32.AllowSetForegroundWindow(-1)
    except Exception:
        pass
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

    # 3) 最大化（リトライ付き）
    for _ in range(3):
        try:
            win32gui.ShowWindow(hwnd, SW_MAXIMIZE)
        except Exception:
            pass
        time.sleep(0.15)
        try:
            if IsZoomed(hwnd):
                break
        except Exception:
            break

    time.sleep(max(0.0, wait_after_focus))

# 低レベルキー送信（必要に応じて使用）。簡易にはpyautoguiを優先。
def send_key_vk(vk: int, hold_ms: int = 30):
    KEYEVENTF_KEYUP = 0x0002
    win32api.keybd_event(vk, 0, 0, 0)
    time.sleep(hold_ms/1000.0)
    win32api.keybd_event(vk, 0, KEYEVENTF_KEYUP, 0)

def scroll_wheel(amount: int):
    # 正: 上スクロール、負: 下スクロール
    win32api.mouse_event(win32con.MOUSEEVENTF_WHEEL, 0, 0, amount, 0)
