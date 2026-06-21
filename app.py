"""
Election Form OCR Pipeline — Streamlit Web App
===============================================
Deploy on Streamlit Cloud, Render, or run locally with:
    streamlit run app.py
"""

import io
import csv
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytesseract
import pandas as pd
import streamlit as st
from PIL import Image
from pdf2image import convert_from_path

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Election Form OCR",
    page_icon="🗳️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# SIDEBAR — Settings
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")

    render_dpi = st.slider("Render DPI", min_value=150, max_value=600, value=400, step=50,
                           help="Higher = better OCR accuracy but slower. 300–400 is ideal.")

    page_number = st.number_input("Page number", min_value=1, value=1, step=1,
                                  help="Which page of the PDF to process.")

    total_col_name = st.text_input("Total column keyword", value="total",
                                   help="Case-insensitive substring to identify the 'Total of valid votes' column.")

    save_debug_cells = st.checkbox("Save cell crops (debug)", value=False,
                                   help="Save every cell crop as a PNG for manual inspection.")

    st.divider()
    st.caption("Tesseract & Poppler must be installed on the server. "
               "On Streamlit Cloud they are pre-installed via packages.txt.")

# ─────────────────────────────────────────────
# TESSERACT CONFIG
# ─────────────────────────────────────────────

TESS_DIGITS = "--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789"
TESS_TEXT   = "--psm 7 --oem 3"

# On Windows set this; on Linux/Mac Tesseract is usually in PATH
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ─────────────────────────────────────────────
# PIPELINE FUNCTIONS
# ─────────────────────────────────────────────

def render_page(pdf_bytes: bytes, page: int, dpi: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(pdf_bytes)
        tmp_path = f.name
    images = convert_from_path(tmp_path, dpi=dpi, first_page=page, last_page=page)
    Path(tmp_path).unlink(missing_ok=True)
    return np.array(images[0].convert("RGB"))


def deskew(img: np.ndarray) -> tuple[np.ndarray, float]:
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180, threshold=200,
        minLineLength=img.shape[1] * 0.3, maxLineGap=20
    )
    if lines is None:
        return img, 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if abs(angle) < 5:
            angles.append(angle)

    if not angles:
        return img, 0.0

    skew_angle = float(np.median(angles))
    if abs(skew_angle) < 0.05:
        return img, skew_angle

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), skew_angle, 1.0)
    corrected = cv2.warpAffine(img, M, (w, h),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(255, 255, 255))
    return corrected, skew_angle


def detect_grid(img: np.ndarray) -> tuple[list[int], list[int]]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    h, w = binary.shape

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (w // 10, 1))
    h_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    h_proj   = np.sum(h_lines, axis=1)
    h_mask   = h_proj > h_proj.max() * 0.3

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, h // 10))
    v_lines  = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    v_proj   = np.sum(v_lines, axis=0)
    v_mask   = v_proj > v_proj.max() * 0.3

    def cluster(mask, min_gap=5):
        positions = np.where(mask)[0]
        if len(positions) == 0:
            return []
        clusters, current = [], [positions[0]]
        for p in positions[1:]:
            if p - current[-1] <= min_gap:
                current.append(p)
            else:
                clusters.append(int(np.mean(current)))
                current = [p]
        clusters.append(int(np.mean(current)))
        return clusters

    return cluster(h_mask), cluster(v_mask)


def preprocess_cell(cell_img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cell_img, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape
    if h < 60 or w < 60:
        scale = max(60 / h, 60 / w, 1)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.copyMakeBorder(binary, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=255)


def ocr_cell(cell_img: np.ndarray, digits_only: bool = True) -> str:
    processed = preprocess_cell(cell_img)
    config = TESS_DIGITS if digits_only else TESS_TEXT
    result = pytesseract.image_to_string(Image.fromarray(processed), config=config).strip()
    if digits_only:
        result = result.replace(" ", "").replace("\n", "").replace(",", "")
    return result


def extract_cells(img, row_ys, col_xs, padding=4, debug_dir=None, progress_bar=None):
    table = []
    total_cells = (len(row_ys) - 1) * (len(col_xs) - 1)
    done = 0

    for r in range(len(row_ys) - 1):
        y1 = row_ys[r] + padding
        y2 = row_ys[r + 1] - padding
        row_data = []

        for c in range(len(col_xs) - 1):
            x1 = col_xs[c] + padding
            x2 = col_xs[c + 1] - padding
            if y2 <= y1 or x2 <= x1:
                row_data.append("")
                continue

            cell = img[y1:y2, x1:x2]
            is_header = (r == 0)
            text = ocr_cell(cell, digits_only=not is_header)
            row_data.append(text)

            if debug_dir:
                safe = text[:10].replace("/", "_").replace("\\", "_")
                cv2.imwrite(
                    str(debug_dir / f"r{r:03d}_c{c:03d}_{safe}.png"),
                    cv2.cvtColor(cell, cv2.COLOR_RGB2BGR)
                )

            done += 1
            if progress_bar:
                progress_bar.progress(done / total_cells, text=f"OCR: {done}/{total_cells} cells")

        table.append(row_data)

    return table


def validate_rows(table, total_col_name="total"):
    if len(table) < 2:
        return [], None

    headers = table[0]
    total_col = next(
        (i for i, h in enumerate(headers) if total_col_name.lower() in h.lower()), None
    )

    results = []
    passed = failed = errors = 0

    for r, row in enumerate(table[1:], 1):
        candidate_cols = [
            i for i in range(len(row))
            if i != total_col and row[i].isdigit()
        ] if total_col is not None else []

        try:
            ocr_sum = sum(int(row[i]) for i in candidate_cols if row[i]) if candidate_cols else None
        except ValueError:
            ocr_sum = None

        try:
            printed_total = int(row[total_col]) if (total_col is not None and row[total_col]) else None
        except (ValueError, IndexError):
            printed_total = None

        if ocr_sum is not None and printed_total is not None:
            delta = ocr_sum - printed_total
            status = "✅ PASS" if delta == 0 else f"❌ FAIL (Δ={delta:+d})"
            if delta == 0:
                passed += 1
            else:
                failed += 1
        else:
            delta = None
            status = "⚠️ PARSE ERROR"
            errors += 1

        results.append({
            "row": r,
            "data": dict(zip(headers, row)),
            "status": status,
            "ocr_sum": ocr_sum,
            "printed_total": printed_total,
            "delta": delta,
        })

    return results, {"passed": passed, "failed": failed, "errors": errors}


def results_to_csv(results, headers) -> bytes:
    buf = io.StringIO()
    fieldnames = headers + ["ocr_sum", "printed_total", "delta", "status"]
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for res in results:
        row = res["data"].copy()
        row.update({
            "ocr_sum": res["ocr_sum"],
            "printed_total": res["printed_total"],
            "delta": res["delta"],
            "status": res["status"],
        })
        writer.writerow(row)
    return buf.getvalue().encode("utf-8")


def grid_overlay_image(img, row_ys, col_xs) -> np.ndarray:
    vis = img.copy()
    for y in row_ys:
        cv2.line(vis, (0, y), (vis.shape[1], y), (255, 0, 0), 2)
    for x in col_xs:
        cv2.line(vis, (x, 0), (x, vis.shape[0]), (0, 0, 255), 2)
    return vis


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

st.title("🗳️ Election Form OCR Pipeline")
st.caption("Upload a scanned election PDF → extract table data → validate checksums → download CSV")

uploaded = st.file_uploader("Upload scanned election PDF", type=["pdf"])

if not uploaded:
    st.info("Upload a PDF to get started. Configure settings in the sidebar.")
    st.stop()

# ── Run pipeline ──
run_btn = st.button("▶ Run OCR Pipeline", type="primary", use_container_width=True)

if run_btn:
    pdf_bytes = uploaded.read()

    with st.status("Running pipeline...", expanded=True) as status:

        # Step 1
        st.write("📄 Rendering page...")
        img = render_page(pdf_bytes, page=page_number, dpi=render_dpi)
        st.write(f"   Image: {img.shape[1]}×{img.shape[0]}px")

        # Step 2
        st.write("📐 Deskewing...")
        img, skew = deskew(img)
        st.write(f"   Skew detected: {skew:.3f}°")

        # Step 3
        st.write("🔲 Detecting grid...")
        row_ys, col_xs = detect_grid(img)
        st.write(f"   {len(row_ys)-1} rows × {len(col_xs)-1} columns")

        # Step 4
        st.write("🔍 Running OCR on cells...")
        progress = st.progress(0.0, text="Starting OCR...")

        debug_dir = None
        if save_debug_cells:
            debug_dir = Path(tempfile.mkdtemp())

        table = extract_cells(img, row_ys, col_xs,
                              debug_dir=debug_dir,
                              progress_bar=progress)
        progress.empty()

        # Step 5
        st.write("✅ Validating checksums...")
        results, summary = validate_rows(table, total_col_name=total_col_name)

        status.update(label="Pipeline complete!", state="complete")

    # Store in session state
    st.session_state["table"]    = table
    st.session_state["results"]  = results
    st.session_state["summary"]  = summary
    st.session_state["img"]      = img
    st.session_state["row_ys"]   = row_ys
    st.session_state["col_xs"]   = col_xs
    st.session_state["debug_dir"] = debug_dir

# ── Show results if available ──
if "results" in st.session_state:
    table   = st.session_state["table"]
    results = st.session_state["results"]
    summary = st.session_state["summary"]
    img     = st.session_state["img"]
    row_ys  = st.session_state["row_ys"]
    col_xs  = st.session_state["col_xs"]

    headers = table[0] if table else []

    st.divider()

    # ── Summary metrics ──
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total rows", len(results))
        c2.metric("✅ Pass", summary["passed"])
        c3.metric("❌ Fail", summary["failed"])
        c4.metric("⚠️ Errors", summary["errors"])

    # ── Tabs ──
    tab1, tab2, tab3 = st.tabs(["📊 Extracted Data", "🔲 Grid Preview", "📋 Raw Table"])

    with tab1:
        if results:
            df_rows = []
            for res in results:
                row = res["data"].copy()
                row["ocr_sum"]       = res["ocr_sum"]
                row["printed_total"] = res["printed_total"]
                row["delta"]         = res["delta"]
                row["status"]        = res["status"]
                df_rows.append(row)
            df = pd.DataFrame(df_rows)

            # Highlight failures
            def highlight_status(val):
                if "FAIL" in str(val):
                    return "background-color: #ffcccc"
                elif "PASS" in str(val):
                    return "background-color: #ccffcc"
                return ""

            st.dataframe(
                df.style.applymap(highlight_status, subset=["status"]),
                use_container_width=True,
                height=500
            )

            # Download CSV
            csv_bytes = results_to_csv(results, headers)
            st.download_button(
                label="⬇️ Download results.csv",
                data=csv_bytes,
                file_name="results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning("No results to display.")

    with tab2:
        st.caption("Red lines = horizontal grid, Blue lines = vertical grid")
        overlay = grid_overlay_image(img, row_ys, col_xs)
        # Downscale for display
        display_w = 1200
        scale = display_w / overlay.shape[1]
        display = cv2.resize(overlay, (display_w, int(overlay.shape[0] * scale)))
        st.image(display, channels="RGB", use_container_width=True)

        # Download debug images
        col1, col2 = st.columns(2)
        with col1:
            rendered_png = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))[1].tobytes()
            st.download_button("⬇️ Download deskewed image", rendered_png,
                               file_name="deskewed.png", mime="image/png")
        with col2:
            overlay_png = cv2.imencode(".png", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))[1].tobytes()
            st.download_button("⬇️ Download grid overlay", overlay_png,
                               file_name="grid_overlay.png", mime="image/png")

    with tab3:
        if table:
            raw_df = pd.DataFrame(table[1:], columns=table[0] if table else None)
            st.dataframe(raw_df, use_container_width=True, height=400)
        else:
            st.warning("No table data.")
