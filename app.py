# -*- coding: utf-8 -*-
import os, sys, threading, time, traceback, json
import webview
import ctypes
import win32gui
from importlib.metadata import version as pkg_version, PackageNotFoundError
from core.config import Settings
from core.logging_conf import setup_logging, get_logger
from core.windows import list_windows, bring_to_front_and_maximize
from core.capture import run_capture, CaptureError
from core.toast import show_toast

APP_TITLE = "SnapLite"
logger = None
settings = None
ui_window = None

class Api:
    """UI(JS) ⇔ Pythonの橋渡し。pywebviewが自動で公開する。"""

    def get_settings(self):
        return settings.as_dict()

    def save_settings(self, new_settings: dict):
        try:
            settings.update_from_dict(new_settings)
            settings.save()
            logger.info("settings updated")
            return {"ok": True}
        except Exception as e:
            logger.exception("failed to save settings")
            return {"ok": False, "message": str(e)}

    def list_windows(self, query: str):
        try:
            wins = list_windows(query)
            return {"ok": True, "items": [{"hwnd": int(h), "title": t} for h, t in wins[:50]]}
        except Exception as e:
            logger.exception("list_windows error")
            return {"ok": False, "message": str(e)}

    def start_capture(self, req: dict):
        """
        req = {
           "hwnd": 123456,
           "options": { ... }   # UIから一時的に上書きする設定（任意）
        }
        """
        try:
            hwnd = int(req.get("hwnd", 0))
            if hwnd == 0:
                return {"ok": False, "message": "対象ウィンドウが選択されていません。"}
            # 一時オーバーライド（UIオプション）
            opt = dict(settings.as_dict())
            opt.update(req.get("options", {}))

            # 保存名にウィンドウタイトルを採用（base_title が空のとき）
            try:
                actual_title = win32gui.GetWindowText(hwnd) or ""
                opt["actual_title"] = actual_title
            except Exception:
                pass

            # 前面化・最大化（安定のため）
            bring_to_front_and_maximize(hwnd, wait_after_focus=opt.get("wait_after_focus", 0.6))

            # 非同期でキャプチャ実行（UIはすぐ閉じる）
            threading.Thread(target=self._capture_thread, args=(hwnd, opt), daemon=True).start()
            return {"ok": True}
        except Exception as e:
            logger.exception("start_capture error")
            return {"ok": False, "message": str(e)}

    def _capture_thread(self, hwnd: int, opt: dict):
        start = time.time()
        try:
            result = run_capture(hwnd, opt, logger)
            dur = time.time() - start
            msg = f"完了: {result['captured_pages']} ページ（{dur:.1f}秒）\n保存先: {result['output_dir']}"
            logger.info(msg)
            show_toast(APP_TITLE, msg)
            # 完了後に保存フォルダを開く（設定でONのとき）
            try:
                if settings.open_folder_on_finish:
                    os.startfile(result['output_dir'])
            except Exception:
                pass
        except CaptureError as ce:
            logger.warning(f"CaptureError: {ce}")
            show_toast(APP_TITLE, f"中断: {ce}")
        except Exception as e:
            logger.exception("capture fatal error")
            show_toast(APP_TITLE, f"失敗: {e}")

    def hide(self):  # 互換用（将来削除予定）
        return self.quit()

    def quit(self):
        """アプリを終了する（UIの『閉じる』ボタンから呼ぶ）"""
        try:
            if ui_window:
                try:
                    ui_window.destroy()  # pywebview 4.x
                except Exception:
                    webview.destroy_window(ui_window)
            return {"ok": True}
        except Exception as e:
            logger.exception("quit error; forcing exit")
            os._exit(0)

def main():
    global logger, settings, ui_window

    settings = Settings.load_or_create()
    log_path = settings.paths.log_file
    setup_logging(log_path)
    logger = get_logger()
    logger.info("=== SnapLite start ===")

    api = Api()
    index_path = os.path.join(os.path.dirname(__file__), "ui", "index.html")
    if not os.path.exists(index_path):
        raise RuntimeError("ui/index.html が見つかりません。")

    # 画面中央に初期配置（create_window の x,y を利用）
    user32 = ctypes.windll.user32
    sw, sh = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    w, h = 720, 300
    x, y = max(0, (sw - w)//2), max(0, (sh - h)//2)

    ui_window = webview.create_window(
        APP_TITLE, url=index_path,
        width=w, height=h, x=x, y=y,
        frameless=False, resizable=True,  # 枠あり & リサイズ可
        on_top=False, easy_drag=False,    # 通常ウィンドウ
        js_api=api
    )
    try:
        pv = pkg_version('pywebview')
    except Exception:
        pv = 'unknown'
    logger.info(f"pywebview version: {pv} (Edge/WebView2を利用予定)")

    # func を使わず直接開始（KeyError 'master' 回避）
    try:
        webview.start(gui='edgechromium', http_server=True, debug=False)
    except Exception as e:
        logger.warning(f"edgechromium GUI failed: {e}; fallback to default GUI")
        webview.start(http_server=True, debug=False)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # 起動直後の例外はコンソールにも出す
        traceback.print_exc()
        if logger:
            logger.exception("fatal on main")
        os._exit(1)
