"""
core.final_recrop
-----------------
一次キャプチャ（raw）を、左右だけ共通境界で一括トリミングして保存します。
- 上下は切りません（ZIP相当の挙動）。
- コミック用の見開き分割（中央2分割）に対応します。
- progress_cb(i:int) を渡すと進捗を通知します（1..2N を想定）。
"""
from __future__ import annotations
import os
from typing import Callable, List, Tuple, Any, Optional
import numpy as np
from PIL import Image

# logger は任意
try:
    from .logging_conf import get_logger  # type: ignore
    logger = get_logger()
except Exception:
    class _Dummy:
        def info(self, *a, **k): pass
        def warning(self, *a, **k): pass
        def error(self, *a, **k): pass
        def debug(self, *a, **k): pass
    logger = _Dummy()

# ---- ユーティリティ -----------------------------------------------------------

def _ctxget(ctx: Any, key: str, default=None):
    """ctx が dict/インスタンス/クラスでも安全に取り出す"""
    if isinstance(ctx, dict):
        return ctx.get(key, default)
    # クラスが渡っても getattr は動作する（存在しなければ default）
    return getattr(ctx, key, default)

def _derive_out_dir_from_raw(paths: List[str], raw_subdir: str) -> str:
    """
    raw パス群から出力先ディレクトリを推定する。
    通常は <out_dir>/_raw/xxx.png → <out_dir>/xxx.png に出す。
    """
    if not paths:
        return os.getcwd()
    p0 = os.path.abspath(paths[0])
    parent = os.path.dirname(p0)
    base = os.path.basename(parent)
    if base == raw_subdir:
        return os.path.dirname(parent)
    return parent  # _raw でなければ同じ階層へ

def _replace_dir_and_ext(raw_path: str, out_dir: str, new_ext: str) -> str:
    """raw のファイル名を保ったまま、ディレクトリと拡張子だけを差し替え"""
    stem = os.path.splitext(os.path.basename(raw_path))[0]
    return os.path.join(out_dir, f"{stem}.{new_ext}")

def _save_image(img: Image.Image, path: str, fmt: str, *, fast_png_level: int = 1):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = fmt.lower()
    if f == "png":
        img.save(path, format="PNG", optimize=False, compress_level=int(fast_png_level))
    elif f in ("jpg", "jpeg"):
        img.save(path, format="JPEG", quality=95, optimize=True)
    elif f == "bmp":
        img.save(path, format="BMP")
    else:
        img.save(path, format=f.upper())

# ---- LR検出（色差法 + 下面UI自動無視） ----------------------------------------

def _estimate_bottom_ui_rows(gray: np.ndarray, thr: int = 70, run_rows: int = 6) -> int:
    """
    画像下端から上向きに暗い行が run_rows 連続する部分を UI 帯とみなし行数を返す。なければ 0。
    """
    h, w = gray.shape
    dark = (gray < thr).mean(axis=1)  # 各行の暗画素率
    cnt = 0
    for y in range(h - 1, -1, -1):
        if dark[y] > 0.2:
            cnt += 1
            if cnt >= run_rows:
                return cnt
        else:
            if cnt > 0:
                break
    return 0

def detect_lr_bounds_color(
    im: Image.Image,
    mt: int,
    mb: int,
    ml: int,
    mr: int,
    bg_tol: int = 3,
    *,
    auto_bottom_margin: bool = True,
    bottom_margin_min: int = 16,
    bottom_dark_threshold: int = 70,
    bottom_dark_run_rows: int = 6,
    min_width: int = 10,
) -> Tuple[int, int]:
    """
    画像の左右境界を色差で検出する。上下は解析範囲を制限するだけで、出力では切らない。
    """
    w, h = im.size
    t = int(mt)
    if auto_bottom_margin:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
        est = _estimate_bottom_ui_rows(gray, thr=int(bottom_dark_threshold), run_rows=int(bottom_dark_run_rows))
        b = max(int(bottom_margin_min), int(mb), int(est))
    else:
        b = int(mb)

    l, r = int(ml), int(mr)
    x0, x1 = max(0, l), max(0, w - r)
    y0, y1 = max(0, t), max(0, h - b)
    if x1 <= x0 or y1 <= y0:
        return 0, w

    arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
    roi = arr[y0:y1, x0:x1, :]
    # 左上近傍の画素を背景の代表値とする（角のノイズを避け 1,1 を採用）
    yy = min(1, roi.shape[0] - 1)
    xx = min(1, roi.shape[1] - 1)
    ref = roi[yy, xx, :].astype(np.int16)
    diff = np.any(np.abs(roi.astype(np.int16) - ref) > int(bg_tol), axis=2)
    cols = np.where(diff.any(axis=0))[0]
    if cols.size == 0:
        return 0, w
    L = int(cols.min()) + x0
    R = int(cols.max()) + x0 + 1  # 右端は排他的
    if (R - L) < int(min_width):
        return 0, w
    # 少し余白を戻す（過剰検出の保険）
    L = max(0, L - 2)
    R = min(w, R + 2)
    return L, R

# ---- メイン ---------------------------------------------------------------

def run_final_recrop(ctx: Any, progress_cb: Optional[Callable[[int], None]] = None) -> List[str]:
    """
    ctx には少なくとも以下が入っている想定：
      - ctx.raw_paths: List[str]       … 一次キャプチャのファイル群
      - ctx.opt: dict                  … 設定
      - ctx.output_dir: str (任意)     … 出力先（無ければ raw から推定）
    progress_cb(i) は任意。i は 1..2N で増加（検出 N + 書き出し N）。
    """
    opt = _ctxget(ctx, "opt", {}) or {}
    paths = _ctxget(ctx, "raw_paths", []) or []
    out_dir = _ctxget(ctx, "output_dir", None)

    if not isinstance(paths, (list, tuple)) or len(paths) == 0:
        logger.warning("run_final_recrop: raw_paths が空です")
        return []

    raw_subdir = str(opt.get("legacy_raw_subdir", "_raw"))
    if not out_dir:
        out_dir = _derive_out_dir_from_raw(list(paths), raw_subdir)

    fmt = str(opt.get("format", "png")).lower()
    fast_png_level = int(opt.get("final_png_fast_level", opt.get("raw_png_fast_level", 1)))
    split_double = bool(opt.get("split_double_page", False))

    # 1) 全ページの LR を計測
    Ls: List[int] = []
    Rs: List[int] = []
    for i, p in enumerate(paths, start=1):
        try:
            im = Image.open(p).convert("RGB")
            L, R = detect_lr_bounds_color(
                im,
                int(opt.get("trim_margin_top", 1)),
                int(opt.get("trim_margin_bottom", 16)),
                int(opt.get("trim_margin_left", 20)),
                int(opt.get("trim_margin_right", 20)),
                int(opt.get("bg_tolerance", 3)),
                auto_bottom_margin=bool(opt.get("auto_bottom_margin", True)),
                bottom_margin_min=int(opt.get("bottom_margin_min", 16)),
                bottom_dark_threshold=int(opt.get("bottom_dark_threshold", 70)),
                bottom_dark_run_rows=int(opt.get("bottom_dark_run_rows", 6)),
            )
            Ls.append(L); Rs.append(R)
        except Exception as e:
            logger.warning(f"LR 検出に失敗: {p}: {e}")
        if progress_cb:
            progress_cb(i)

    if not Ls or not Rs:
        logger.warning("run_final_recrop: LR が検出できませんでした（フォールバック：無加工コピー）")
        out = []
        for i, p in enumerate(paths, start=1):
            dst = _replace_dir_and_ext(p, out_dir, fmt)
            try:
                im = Image.open(p)
                _save_image(im, dst, fmt, fast_png_level=fast_png_level)
                out.append(dst)
            except Exception as e:
                logger.warning(f"final write failed on {p}: {e}")
            if progress_cb:
                progress_cb(len(paths) + i)
        return out

    Lg, Rg = min(Ls), max(Rs)
    logger.info(f"final unified LR=({Lg},{Rg}) via color pass over {len(Ls)} pages")

    # 2) 書き出し
    out_paths: List[str] = []
    for i, p in enumerate(paths, start=1):
        try:
            im = Image.open(p)
            w, h = im.size
            Lc, Rc = max(0, Lg), min(w, Rg)
            if Rc - Lc >= 10:
                im = im.crop((Lc, 0, Rc, h))  # 左右のみ
            if split_double:
                w2 = im.size[0] // 2
                left = im.crop((0, 0, w2, h))
                right = im.crop((w2, 0, im.size[0], h))
                pL = _replace_dir_and_ext(p, out_dir, fmt).replace(f".{fmt}", f"_L.{fmt}")
                pR = _replace_dir_and_ext(p, out_dir, fmt).replace(f".{fmt}", f"_R.{fmt}")
                _save_image(left, pL, fmt, fast_png_level=fast_png_level)
                _save_image(right, pR, fmt, fast_png_level=fast_png_level)
                out_paths.extend([pL, pR])
            else:
                dst = _replace_dir_and_ext(p, out_dir, fmt)
                _save_image(im, dst, fmt, fast_png_level=fast_png_level)
                out_paths.append(dst)
        except Exception as e:
            logger.warning(f"final write failed on {p}: {e}")
        if progress_cb:
            progress_cb(len(paths) + i)

    return out_paths
