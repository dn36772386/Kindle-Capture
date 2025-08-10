"""
core.app_ui
-----------
pywebview 経由の JS 評価を安全に呼び出すヘルパ。
"""
from __future__ import annotations
import time
from typing import Optional

try:
    import webview  # type: ignore
except Exception:
    webview = None

def safe_eval_js(window, script: str, *, retry: int = 2, wait: float = 0.2):
    """
    window.evaluate_js を、ウィンドウがまだ初期化されていない場合にも安全に呼ぶ。
    失敗しても例外にしない。
    """
    if window is None or webview is None:
        return
    for _ in range(max(1, int(retry))):
        try:
            return window.evaluate_js(script)
        except Exception:
            time.sleep(wait)
    return None

def update_progress(window, step: str, current: int, total: int):
    """
    進捗 UI 更新の JS 呼び出し。window 側で handleProgress(step, current, total) を用意しておく。
    """
    if not window:
        return
    script = f'try{{window.handleProgress && window.handleProgress("{step}", {int(current)}, {int(total)})}}catch(e){{}}'
    safe_eval_js(window, script)
