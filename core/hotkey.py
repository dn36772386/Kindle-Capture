# -*- coding: utf-8 -*-
import threading
import win32con, win32api
import ctypes

# 文字列 "ctrl+shift+s" → RegisterHotKey用 修飾/キーに変換
def _parse_hotkey(hks: str):
    s = (hks or "").lower().replace(" ", "")
    mods = 0
    if "ctrl" in s:  mods |= win32con.MOD_CONTROL
    if "alt" in s:   mods |= win32con.MOD_ALT
    if "shift" in s: mods |= win32con.MOD_SHIFT
    if "win" in s:   mods |= win32con.MOD_WIN

    # 最後の"+"以降をキーとみなす簡易実装（例: ctrl+shift+s）
    key = s.split("+")[-1]
    # 英字・数字・機能キーをそこそこ扱う
    vk = None
    if len(key) == 1:
        vk = ord(key.upper())
    else:
        names = {
            "f1": win32con.VK_F1,  "f2": win32con.VK_F2,  "f3": win32con.VK_F3,
            "f4": win32con.VK_F4,  "f5": win32con.VK_F5,  "f6": win32con.VK_F6,
            "f7": win32con.VK_F7,  "f8": win32con.VK_F8,  "f9": win32con.VK_F9,
            "f10": win32con.VK_F10,"f11": win32con.VK_F11,"f12": win32con.VK_F12,
            "space": win32con.VK_SPACE, "tab": win32con.VK_TAB, "esc": win32con.VK_ESCAPE
        }
        vk = names.get(key)
    if vk is None:
        raise ValueError(f"ホットキー解析に失敗: {hks}")
    return mods, vk

class GlobalHotkey:
    def __init__(self, hotkey_string: str, on_trigger, logger=None):
        self.logger = logger
        self.on_trigger = on_trigger
        self.mods, self.vk = _parse_hotkey(hotkey_string)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()
        if self.logger: self.logger.info("global hotkey thread started")

    def stop(self):
        self._stop.set()

    def _loop(self):
        user32 = ctypes.windll.user32
        id = 1
        if not user32.RegisterHotKey(None, id, self.mods, self.vk):
            if self.logger: self.logger.error("RegisterHotKey failed")
            return
        try:
            msg = ctypes.wintypes.MSG()
            while not self._stop.is_set():
                if user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
                    if msg.message == win32con.WM_HOTKEY and msg.wParam == id:
                        try:
                            self.on_trigger()
                        except Exception:
                            if self.logger: self.logger.exception("hotkey callback error")
                else:
                    user32.MsgWaitForMultipleObjects(0, None, False, 50, 255)
        finally:
            user32.UnregisterHotKey(None, id)
            if self.logger: self.logger.info("global hotkey unregistered")
