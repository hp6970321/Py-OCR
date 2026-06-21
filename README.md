# Election Form OCR Pipeline

Streamlit web app for extracting tabular data from scanned election PDFs using OCR, with automatic checksum validation.

## Run Locally

```bash
# 1. Install Python dependencies
pip install -r requirements.txt

# 2. Install system dependencies (Windows)
#    - Tesseract: https://github.com/UB-Mannheim/tesseract/wiki
#    - Poppler:   https://github.com/oschwartz10612/poppler-windows/releases

# 3. If Tesseract isn't in PATH, uncomment and set this line in app.py:
#    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# 4. Run
streamlit run app.py
```

## Deploy to Streamlit Cloud (Free)

1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Click **New app** → select your repo → set Main file: `app.py`
4. Click **Deploy**

Tesseract and Poppler are installed automatically via `packages.txt`.

## Deploy to Render (Free tier)

1. Push to GitHub
2. Go to https://render.com → New → Web Service
3. Connect repo, set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Add a `render.yaml` or use the dashboard

## Pipeline Steps

| Step | What it does |
|------|-------------|
| 1. Render | PDF page → high-res image (configurable DPI) |
| 2. Deskew | Hough line detection → rotate to fix scan skew |
| 3. Grid detection | Morphological ops → find all row/column lines |
| 4. Cell OCR | Crop + preprocess each cell → Tesseract (digits-only for data cells) |
| 5. Checksum | Sum candidate columns → compare to printed total → PASS/FAIL per row |
| 6. Export | Download results as CSV |
