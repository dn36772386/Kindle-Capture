# -*- coding: utf-8 -*-
import os, time, datetime, math
import numpy as np
from PIL import Image
import mss
import cv2
import pyautogui
from .windows import get_window_rect
import re

class CaptureError(Exception):
    pass

def _clamp_rect(rect):
    """仮想スクリーン内にウィンドウ矩形を収める"""
    with mss.mss() as sct:
        mon = sct.monitors[0]  # 仮想全体
        L = max(mon['left'], rect['left'])
        T = max(mon['top'], rect['top'])
        R = min(mon['left'] + mon['width'], rect['left'] + rect['width'])
        B = min(mon['top'] + mon['height'], rect['top'] + rect['height'])
        if R <= L or B <= T:
            raise CaptureError("ウィンドウが画面外です。")
        return {'left': L, 'top': T, 'width': R-L, 'height': B-T}

def sanitize(s: str):
    bad = '<>:"/\\|?*'
    return "".join('_' if c in bad else c for c in s).strip()

def average_hash(pil_img, hash_size=8):
    img = pil_img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    mean = arr.mean()
    return (arr >= mean).flatten()

def hamming(a, b):
    return int(np.count_nonzero(a != b))

def fallback_crop(pil_img, margin):
    w, h = pil_img.size
    m = max(0, int(margin))
    return pil_img.crop((m, m, w-m, h-m))

def _sig_for_change(pil_img):
    """変化検出用シグネチャ（中央領域の aHash）"""
    w, h = pil_img.size
    cw, ch = max(32, w//2), max(24, h//2)
    l = (w - cw)//2; t = (h - ch)//2
    roi = pil_img.crop((l, t, l+cw, t+ch))
    return average_hash(roi, hash_size=8)

def _wait_until_change(rect, prev_sig, opt, logger):
    """ページ送り後、画面が変わるまで待機し、変化検知したフレームを返す"""
    if not opt.get("change_polling", True) or prev_sig is None:
        return grab_window_image(rect), None
    deadline = time.time() + float(opt.get("change_timeout", 5.0))
    interval = max(0.05, float(opt.get("change_poll_interval", 0.15)))
    thr = int(opt.get("change_hamming_threshold", 4))
    last = None
    while time.time() < deadline:
        time.sleep(interval)
        cur = grab_window_image(rect)
        sig = _sig_for_change(cur)
        dist = hamming(prev_sig, sig)
        if dist >= thr:
            logger.info(f"change detected: dist={dist} (thr={thr})")
            return cur, sig
        last = (cur, sig)
    logger.warning("change wait timeout; continue with last frame")
    return (last[0] if last else grab_window_image(rect)), (last[1] if last else None)

def _detect_lr_bounds(np_bgr, mtop, mbot, mleft, mright):
    """左右境界を列分散で検出し (left,right) を返す"""
    h, w = np_bgr.shape[:2]
    y0, y1 = max(0, mtop), max(0, h - mbot)
    x0, x1 = max(0, mleft), max(0, w - mright)
    crop = np_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return 0, w
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    col_var = gray.var(axis=0)
    med = float(np.median(col_var)) + 1e-6
    mask = col_var > (2.5 * med)
    if not mask.any():
        return 0, w
    cols = np.where(mask)[0]
    L = int(cols.min()) + x0
    R = int(cols.max()) + x0 + 1
    L = max(0, L - 2)
    R = min(w, R + 2)
    return L, R


def _detect_lr_bounds_color(np_bgr, mtop, mbot, mleft, mright, bg_tol=3, min_width=10):
    """左右境界を背景色比較で検出し (left,right) を返す（旧ロジック互換）
    ROI 左上近傍の色を背景とみなし、そこから有意に変化する最初/最後の列を境界とする。
    """
    h, w = np_bgr.shape[:2]
    y0, y1 = max(0, mtop), max(0, h - mbot)
    x0, x1 = max(0, mleft), max(0, w - mright)
    if (y1 - y0) <= 0 or (x1 - x0) <= 0:
        return 0, w
    roi = np_bgr[y0:y1, x0:x1, :]
    # BGR -> use small offset to avoid corner noise
    yy = min(1, roi.shape[0]-1); xx = min(1, roi.shape[1]-1)
    ref = roi[yy, xx, :].astype(np.int16)
    diff = np.any(np.any(np.abs(roi.astype(np.int16) - ref) > int(bg_tol), axis=2), axis=0)
    cols = np.where(diff)[0]
    if cols.size == 0:
        return 0, w
    L = int(cols.min()) + x0
    R = int(cols.max()) + x0 + 1  # 右は排他的
    if (R - L) < int(min_width):
        return 0, w
    L = max(0, L - 2)
    R = min(w, R + 2)
    return L, R


def auto_crop_content(pil_img, min_ratio=0.1, fallback_margin=10):
    np_img = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 35, 15)
    kernel = np.ones((3,3), np.uint8)
    thr = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return fallback_crop(pil_img, fallback_margin)

    h, w = gray.shape[:2]
    area_total = w * h
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area / area_total < float(min_ratio):
            continue
        x, y, cw, ch = cv2.boundingRect(cnt)
        x0 = max(0, x - 2); y0 = max(0, y - 2)
        x1 = min(w, x + cw + 2); y1 = min(h, y + ch + 2)
        return pil_img.crop((x0, y0, x1, y1))
    return fallback_crop(pil_img, fallback_margin)

def grab_window_image(rect):
    with mss.mss() as sct:
        shot = sct.grab(rect)  # BGRA
        arr = np.array(shot)
        arr = arr[:, :, :3][:, :, ::-1]  # BGRA->RGB
        return Image.fromarray(arr)

def _maybe_to_grayscale(pil_img, opt):
    if not opt.get("grayscale_detect", True):
        return pil_img
    th = int(opt.get("grayscale_threshold", 7))
    gm = (
        int(opt.get("grayscale_margin_top", 0)),
        int(opt.get("grayscale_margin_bottom", 0)),
        int(opt.get("grayscale_margin_left", 0)),
        int(opt.get("grayscale_margin_right", 0)),
    )
    arr = np.array(pil_img)
    h, w = arr.shape[:2]
    y0, y1 = max(0, gm[0]), max(0, h - gm[1])
    x0, x1 = max(0, gm[2]), max(0, w - gm[3])
    roi = arr[y0:y1, x0:x1, :3]
    if roi.size == 0:
        return pil_img
    b, g, r = roi[:, :, 2], roi[:, :, 1], roi[:, :, 0]
    maxdiff = max(np.abs(b-g).max(), np.abs(g-r).max(), np.abs(r-b).max())
    if int(maxdiff) <= th:
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY), mode='L')
    return pil_img

# ================= オーバーレイ検出 / クリーンフレーム待機 =================
def _overlay_score(pil_img: Image.Image, opt) -> tuple[float, float]:
    """中央帯と下部帯の暗画素比率（0..1, 0=明,1=暗）を返す"""
    try:
        g = pil_img.convert("L")
        arr = np.asarray(g, dtype=np.uint8)
        h, w = arr.shape
        center_ratio = float(opt.get("overlay_center_ratio", 0.12))
        bottom_ratio = float(opt.get("overlay_bottom_ratio", 0.18))
        ch = max(1, int(h * center_ratio))
        bh = max(1, int(h * bottom_ratio))
        # 中央帯（横は中央 50%）
        cw = max(1, int(w * 0.5))
        cx0 = (w - cw)//2; cx1 = cx0 + cw
        cy0 = (h - ch)//2; cy1 = cy0 + ch
        center = arr[cy0:cy1, cx0:cx1]
        bottom = arr[h-bh:h, :]
        thr = int(opt.get("overlay_dark_threshold", 70))
        c_dark = float((center < thr).mean()) if center.size else 0.0
        b_dark = float((bottom < thr).mean()) if bottom.size else 0.0
        return c_dark, b_dark
    except Exception:
        return 0.0, 0.0

def _has_overlay(pil_img: Image.Image, opt) -> bool:
    pr = float(opt.get("overlay_pixel_ratio", 0.06))
    c_dark, b_dark = _overlay_score(pil_img, opt)
    return (c_dark > pr) or (b_dark > pr)

def _wait_for_clean_frame(rect, opt, logger, allow_toggle=False):
    """オーバーレイ（中央メッセージ/下部バー）が消えかつ数フレーム安定するまで待つ"""
    if not bool(opt.get("ui_clean_check", True)):
        return grab_window_image(rect)
    stable_need = int(opt.get("stable_frames_required", 3))
    timeout = float(opt.get("ui_settle_timeout", 4.0))
    interval = float(opt.get("ui_settle_check_interval", 0.15))
    allow_retoggle = allow_toggle and bool(opt.get("allow_fs_retoggle", True)) and bool(opt.get("use_fullscreen_toggle", True))
    t0 = time.time()
    prev_hash = None
    stable = 0
    retoggled = False
    while True:
        img = grab_window_image(rect)
        if not _has_overlay(img, opt):
            hsh = average_hash(img)
            if prev_hash is not None and hamming(prev_hash, hsh) <= 1:
                stable += 1
            else:
                stable = 1
            prev_hash = hsh
            if stable >= stable_need:
                return img
        else:
            stable = 0

        if time.time() - t0 > timeout:
            if allow_retoggle and not retoggled:
                try:
                    pyautogui.press(str(opt.get("fullscreen_key", "f11")))
                    time.sleep(float(opt.get("after_fs_toggle_wait", 0.8)))
                    pyautogui.press(str(opt.get("fullscreen_key", "f11")))
                    time.sleep(float(opt.get("after_fs_toggle_wait", 0.8)))
                    retoggled = True
                    t0 = time.time()  # タイムアウトリセット
                    logger.info("fullscreen re-toggle to clear overlay")
                    continue
                except Exception as e:
                    logger.warning(f"fullscreen re-toggle failed: {e}")
            # これ以上待たない
            logger.info("overlay wait timeout; continue with current frame")
            return img
        time.sleep(interval)

def _opposite_key(key: str) -> str:
    k = (key or "").lower()
    mapping = {
        "right": "left",
        "left": "right",
        "pagedown": "pageup",
        "pageup": "pagedown",
    }
    return mapping.get(k, "left")

def _navigate_to_cover(rect, opt, logger):
    if not bool(opt.get("go_to_cover_on_start", True)):
        return
    max_steps = int(opt.get("cover_seek_max_steps", 300))
    stable_need = int(opt.get("cover_seek_stable_trials", 6))
    back_key = _opposite_key(opt.get("next_key", "right"))
    prev_img = grab_window_image(rect)
    prev_hash = average_hash(prev_img)
    stable = 0
    for i in range(max_steps):
        pyautogui.press(back_key)
        time.sleep(max(0.05, float(opt.get("wait_after_turn", 0.3))))
        cur_img = grab_window_image(rect)
        cur_hash = average_hash(cur_img)
        dist = hamming(prev_hash, cur_hash)
        if dist < int(opt.get("change_hamming_threshold", 4)):
            stable += 1
            if stable >= stable_need:
                logger.info(f"cover reached after {i+1} steps (stable streak)")
                break
        else:
            stable = 0
        prev_hash = cur_hash
    # クリーン化待機（UI写り込み除去）
    try:
        _wait_for_clean_frame(rect, opt, logger, allow_toggle=True)
    except Exception as e:
        logger.warning(f"initial clean wait failed: {e}")

def save_image(img: Image.Image, path: str, fmt: str = "png"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if fmt.lower() == "png":
        img.save(path, format="PNG", optimize=True)
    else:
        img.save(path, format=fmt.upper())

def turn_page(opt: dict):
    time.sleep(max(0.0, float(opt.get("wait_after_turn", 0.5))))
    if bool(opt.get("scroll_mode", False)):
        pixels = int(opt.get("scroll_pixels", 600))
        pyautogui.scroll(-pixels)
        return
    if bool(opt.get("use_click", False)):
        x, y = opt.get("click_pos", (0,0))
        if isinstance(x, (list, tuple)):
            x, y = x
        pyautogui.click(int(x), int(y))
        return
    # キー送信
    key = opt.get("next_key", "right")
    pyautogui.press(key)


def run_capture(hwnd: int, opt: dict, logger):
    """
    旧式互換を統合：
    - legacy_final_recrop=True の場合、各ページは raw を一時保存し、最後に左右境界を全ページ同一に揃えて一括トリムして保存。
    - legacy_final_recrop=False の場合は従来通り逐次トリムして保存。
    """
    rect_raw = get_window_rect(hwnd)
    rect = _clamp_rect(rect_raw)
    logger.info(f"window rect raw={rect_raw} clamped={rect}")
    if rect["width"] <= 0 or rect["height"] <= 0:
        raise CaptureError("対象ウィンドウのサイズが不正です。")

    out_dir = opt.get("paths", {}).get("output_dir") or opt.get("output_dir") or None
    if not out_dir:
        raise CaptureError("出力先フォルダの解決に失敗しました。設定を確認してください。")

    base_title = opt.get("base_title") or ""
    actual_title = sanitize(str(opt.get("actual_title") or ""))  # UIが渡さない場合は空
    if not base_title:
        base_title = actual_title if actual_title else "SnapLite"

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    fmt = (opt.get("screenshot_format") or "png").lower()
    max_pages = int(opt.get("max_pages", 2000))

    auto_crop = bool(opt.get("auto_crop", True))
    min_ratio = float(opt.get("min_contour_area_ratio", 0.1))
    fb_margin = int(opt.get("fallback_margin", 10))
    split_double = bool(opt.get("split_double_page", False))

    dedup_thr = int(opt.get("dedup_hamming_threshold", 2))
    dedup_patience = int(opt.get("dedup_patience", 3))

    wait_after_focus = float(opt.get("wait_after_focus", 0.6))
    time.sleep(wait_after_focus)  # 表示安定待ち

    # --- フルスクリーン化（F11） ---
    if bool(opt.get("use_fullscreen_toggle", True)):
        try:
            pyautogui.press(str(opt.get("fullscreen_key", "f11")))
            time.sleep(max(0.2, float(opt.get("fullscreen_wait", 1.0))))
            logger.info("fullscreen toggled ON")
        except Exception as e:
            logger.warning(f"fullscreen toggle failed: {e}")

    # --- ページジャンプ（ctrl+g → 値 → enter）任意 ---
    if bool(opt.get("pagejump_enabled", False)):
        try:
            combo = str(opt.get("pagejump_combo", "ctrl+g")).lower().split("+")
            combo = [c.strip() for c in combo if c.strip()]
            if combo:
                pyautogui.hotkey(*combo)
                time.sleep(0.2)
            val = str(opt.get("pagejump_value", "1"))
            pyautogui.typewrite(val, interval=0.02)
            pyautogui.press(str(opt.get("pagejump_confirm_key", "enter")))
            time.sleep(max(0.2, float(opt.get("pagejump_wait", 0.6))))
            logger.info(f"pagejump sent: {val}")
        except Exception as e:
            logger.warning(f"pagejump failed: {e}")

    prev_hash = None
    same_count = 0
    page = 1
    captured = 0
    prev_sig = None          # 変化検出用
    stable_lr = None         # 左右境界（全ページでの minL / maxR）
    saved_paths = []         # 逐次保存したファイル（legacy=False のとき用）
    raw_paths = []           # legacy=True のときの raw 画像の保存先

    # 旧式互換設定
    legacy = bool(opt.get("legacy_final_recrop", True))
    raw_subdir = str(opt.get("legacy_raw_subdir", "_raw"))
    bg_tol = int(opt.get("bg_tolerance", 3))
    method = str(opt.get("dynamic_trim_method", "variance")).lower()
    raw_dir = os.path.join(out_dir, raw_subdir)
    if legacy:
        os.makedirs(raw_dir, exist_ok=True)

    logger.info(f"capture start: rect={rect}, out_dir={out_dir}, legacy_final_recrop={legacy}")

    while page <= max_pages:
        # ページ遷移後の変化待ち（初回は即撮り）
        if captured == 0:
            base_img = grab_window_image(rect)
            cur_sig = _sig_for_change(base_img)
        else:
            base_img, cur_sig = _wait_until_change(rect, prev_sig, opt, logger)

        img = base_img

        # --- 左右境界の検出（legacy でも毎回計算し、min/max を更新） ---
        if bool(opt.get("dynamic_trim", True)):
            np_bgr = cv2.cvtColor(np.array(base_img), cv2.COLOR_RGB2BGR)
            if method == "color":
                L, R = _detect_lr_bounds_color(
                    np_bgr,
                    int(opt.get("trim_margin_top", 1)),
                    int(opt.get("trim_margin_bottom", 16)),
                    int(opt.get("trim_margin_left", 20)),
                    int(opt.get("trim_margin_right", 20)),
                    bg_tol=bg_tol
                )
            else:
                L, R = _detect_lr_bounds(
                    np_bgr,
                    int(opt.get("trim_margin_top", 1)),
                    int(opt.get("trim_margin_bottom", 16)),
                    int(opt.get("trim_margin_left", 20)),
                    int(opt.get("trim_margin_right", 20)),
                )
            if stable_lr is None:
                stable_lr = (L, R)
            else:
                stable_lr = (min(stable_lr[0], L), max(stable_lr[1], R))

        # --- legacy: raw保存のみ。非legacy: ここでトリム・分割・保存 ---
        if legacy:
            # raw を PNG で保存（後で一括処理）
            p_raw = os.path.join(raw_dir, f"{base_title}_{ts}_{page:04}.png")
            save_image(base_img, p_raw, "png")
            raw_paths.append(p_raw)
            # 重複検出用ハッシュは raw で計算
            cur_hash = average_hash(base_img)
            logger.info(f"saved raw: {p_raw} size={base_img.size}")
        else:
            # --- 従来の逐次処理 ---
            if bool(opt.get("dynamic_trim", True)) and stable_lr is not None:
                w_img, h_img = img.size
                Lc, Rc = max(0, stable_lr[0]), min(w_img, stable_lr[1])
                if Rc - Lc >= 10:
                    img = img.crop((Lc, 0, Rc, h_img))
            elif auto_crop:
                try:
                    img = auto_crop_content(img, min_ratio=min_ratio, fallback_margin=fb_margin)
                except Exception as e:
                    logger.warning(f"auto_crop failed: {e}; fallback")
                    img = fallback_crop(img, fb_margin)

            # 保存名
            if split_double:
                w, h = img.size
                left = img.crop((0, 0, w//2, h))
                right = img.crop((w//2, 0, w, h))
                pL = os.path.join(out_dir, f"{base_title}_{ts}_{page:04}_L.{fmt}")
                pR = os.path.join(out_dir, f"{base_title}_{ts}_{page:04}_R.{fmt}")
                save_image(left, pL, fmt); save_image(right, pR, fmt)
                cur_hash = average_hash(right)  # 代表として右で判定
                saved_paths.extend([pL, pR])
                logger.info(f"saved: {pL}, {pR}")
            else:
                # リサイズ（幅基準）
                rzw = int(opt.get("resize_width", 0) or 0)
                if rzw > 0 and img.mode != 'L':
                    w0, h0 = img.size
                    if w0 > rzw:
                        nh = int(round(h0 * (rzw / w0)))
                        img = img.resize((rzw, nh), Image.LANCZOS)
                # グレースケール判定
                img_save = _maybe_to_grayscale(img, opt)
                p = os.path.join(out_dir, f"{base_title}_{ts}_{page:04}.{fmt}")
                save_image(img_save, p, fmt)
                cur_hash = average_hash(img_save)
                saved_paths.append(p)
                logger.info(f"saved: {p} size={img_save.size} mode={img_save.mode}")

        captured += 1
        prev_sig = cur_sig

        if prev_hash is not None:
            dist = hamming(prev_hash, cur_hash)
            if dist <= dedup_thr:
                same_count += 1
                logger.info(f"duplicate? dist={dist}, streak={same_count}/{dedup_patience}")
            else:
                same_count = 0

            if same_count >= dedup_patience:
                logger.info("end detected by duplicate streak")
                break

        prev_hash = cur_hash
        page += 1
        turn_page(opt)

    if captured == 0:
        raise CaptureError("1枚も保存できませんでした。設定と対象を確認してください。")

    # --- legacy: 最終一括トリム ---
    if legacy:
        if stable_lr is None:
            # 予備: 安全側で枠なし
            logger.warning("stable_lr is None; skip final recrop")
        else:
            Lc, Rc = int(max(0, stable_lr[0])), int(stable_lr[1])
            logger.info(f"final recrop with stable_lr=({Lc},{Rc}) on {len(raw_paths)} pages")
        # 出力
        for i, p_raw in enumerate(raw_paths, start=1):
            try:
                im = Image.open(p_raw)
                w0, h0 = im.size
                if stable_lr is not None:
                    Lb, Rb = max(0, Lc), min(w0, Rc)
                    if (Rb - Lb) >= 10:
                        im = im.crop((Lb, 0, Rb, h0))
                # 見開き分割
                if split_double:
                    w, h = im.size
                    left = im.crop((0, 0, w//2, h))
                    right = im.crop((w//2, 0, w, h))
                    pL = os.path.join(out_dir, f"{base_title}_{ts}_{i:04}_L.{fmt}")
                    pR = os.path.join(out_dir, f"{base_title}_{ts}_{i:04}_R.{fmt}")
                    # リサイズ・グレースケール（左右それぞれ）
                    if int(opt.get("resize_width", 0) or 0) > 0 and left.mode != 'L':
                        rzw = int(opt.get("resize_width", 0) or 0)
                        if left.size[0] > rzw:
                            nh = int(round(left.size[1] * (rzw / left.size[0])))
                            left = left.resize((rzw, nh), Image.LANCZOS)
                        if right.size[0] > rzw:
                            nh = int(round(right.size[1] * (rzw / right.size[0])))
                            right = right.resize((rzw, nh), Image.LANCZOS)
                    left = _maybe_to_grayscale(left, opt)
                    right = _maybe_to_grayscale(right, opt)
                    save_image(left, pL, fmt); save_image(right, pR, fmt)
                else:
                    # リサイズ
                    rzw = int(opt.get("resize_width", 0) or 0)
                    if rzw > 0 and im.mode != 'L':
                        w0, h0 = im.size
                        if w0 > rzw:
                            nh = int(round(h0 * (rzw / w0)))
                            im = im.resize((rzw, nh), Image.LANCZOS)
                    im = _maybe_to_grayscale(im, opt)
                    p = os.path.join(out_dir, f"{base_title}_{ts}_{i:04}.{fmt}")
                    save_image(im, p, fmt)
            except Exception as e:
                logger.warning(f"final recrop failed at page {i}: {e}")

        # raw削除
        if not bool(opt.get("legacy_keep_raw", False)):
            try:
                for p_raw in raw_paths:
                    try:
                        os.remove(p_raw)
                    except Exception:
                        pass
                # 空ならディレクトリ削除
                try:
                    os.rmdir(raw_dir)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"cleanup raw failed: {e}")

    # --- フルスクリーン解除 ---
    if bool(opt.get("use_fullscreen_toggle", True)) and bool(opt.get("exit_fullscreen_on_finish", True)):
        try:
            pyautogui.press(str(opt.get("fullscreen_key", "f11")))
            time.sleep(max(0.2, float(opt.get("fullscreen_wait", 1.0))))
            logger.info("fullscreen toggled OFF")
        except Exception as e:
            logger.warning(f"fullscreen toggle off failed: {e}")

    return {"captured_pages": captured, "output_dir": out_dir}
