#!/usr/bin/env python3
"""
Auto-scanning 1D Barcode Reader (Tkinter + OpenCV + pyzbar) + Selenium shop buttons

Shops supported:
  - REWE:  https://kartenwelt.rewe.de/rewe-geschenkkarte.html
           #card_number
  - DM:    https://www.dm.de/services/services-im-markt/geschenkkarten
           #credit-checker-printedCreditKey-input
  - ALDI:  https://www.helaba.com/de/aldi/
           #card > tbody > tr:nth-child(2) > td:nth-child(2) > input
  - LIDL:  https://www.lidl.at/c/geschenkkarte-guthabenabfrage/s10012116
           #card > tbody > tr:nth-child(2) > td:nth-child(2) > input
  - EDEKA: https://gutschein.avs.de/edeka-mh/home.htm
           #postform > div > div:nth-child(5) > div > div > input[type="text"]:nth-child(1)

Environment (conda):
  conda create -n voucher-scan -c conda-forge python=3.12 opencv pillow tk zbar pyzbar selenium
  conda activate voucher-scan
  python voucher_scanner_shops.py
"""

import os
import sys
import time
import threading
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---- Optional beep on success (noop on non-Windows) -------------------------
try:
    import winsound

    def beep(freq=1500, dur=120):
        try:
            winsound.Beep(freq, dur)
        except Exception:
            pass

except Exception:

    def beep(*args, **kwargs):
        pass


# ---- Barcode backend (pyzbar) -----------------------------------------------
try:
    from pyzbar.pyzbar import decode, ZBarSymbol
except Exception as e:
    print("ERROR: pyzbar not available:", e)
    print("Install in your conda env: conda install -c conda-forge pyzbar zbar")
    sys.exit(1)


def zbar_symbols(names):
    """Return available ZBarSymbol members (handles build differences)."""
    out = []
    for n in names:
        sym = getattr(ZBarSymbol, n, None)
        if sym is not None:
            out.append(sym)
    return out


LINEAR_SYMBOL_NAMES = ["EAN13", "EAN8", "UPCA", "UPCE", "CODE128", "CODE39", "I25"]
SYMBOLS = zbar_symbols(LINEAR_SYMBOL_NAMES + ["QRCODE"])

# ---- Selenium (multi-browser fallback) --------------------------------------
# REWE
REWE_URL = "https://kartenwelt.rewe.de/rewe-geschenkkarte.html"
REWE_SELECTOR = "#card_number"
# DM
DM_URL = "https://www.dm.de/services/services-im-markt/geschenkkarten"
DM_SELECTOR = "#credit-checker-printedCreditKey-input"
# ALDI
ALDI_URL = "https://www.helaba.com/de/aldi/"
ALDI_SELECTOR = "#card > tbody > tr:nth-child(2) > td:nth-child(2) > input"
# LIDL
LIDL_URL = "https://www.lidl.at/c/geschenkkarte-guthabenabfrage/s10012116"
LIDL_SELECTOR = "#card > tbody > tr:nth-child(2) > td:nth-child(2) > input"
# EDEKA
EDEKA_URL = "https://gutschein.avs.de/edeka-mh/home.htm"
EDEKA_SELECTOR = (
    '#postform > div > div:nth-child(5) > div > div > input[type="text"]:nth-child(1)'
)

SELENIUM_AVAILABLE = True
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except Exception:
    SELENIUM_AVAILABLE = False

# ---- Tuning parameters (scanner) --------------------------------------------
SCAN_EVERY_MS = 150  # scanning cadence (ms)
ROI_HEIGHT_FRAC = 0.35  # centered stripe height (0..1)
ROI_WIDTH_FRAC = 0.90  # width fraction (0..1)
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
UNSHARP_AMOUNT = 1.4
UNSHARP_SIGMA = 1.0
MORPH_KERNEL_W = 21  # width of horizontal kernel to bridge dashes
MORPH_KERNEL_H = 3
MORPH_ITER = 1
DRAW_DASH_GAP = 14  # pixels between dashes on the overlay
CORNER_LEN = 26  # length of corner brackets
OVERLAY_COLOR = (0, 200, 255)  # BGR
BOX_COLOR = (0, 220, 0)
TEXT_COLOR = (20, 20, 20)


class AutoBarcodeApp:
    def __init__(self, root: tk.Tk, camera_index: int = 0):
        self.root = root
        root.title("Auto 1D Barcode Scanner + REWE / DM / ALDI / LIDL / EDEKA")

        # Camera ----------------------------------------------------------------
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Could not open the camera. Check: System Settings → Privacy & Security → Camera.",
            )
            root.destroy()
            sys.exit(1)

        # Best-effort hints
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # UI --------------------------------------------------------------------
        self.label = ttk.Label(root)  # image area
        self.label.grid(row=0, column=0, columnspan=7, padx=10, pady=10)

        ttk.Label(root, text="Decoded:").grid(row=1, column=0, sticky="w", padx=10)
        self.code = tk.StringVar()
        ttk.Entry(root, textvariable=self.code, width=40).grid(
            row=1, column=1, padx=6, pady=2, sticky="we", columnspan=2
        )

        # Buttons (disabled until we have a code)
        self.rewe_btn = ttk.Button(root, text="🛒 REWE", command=self.open_rewe)
        self.dm_btn = ttk.Button(root, text="🛍️ DM", command=self.open_dm)
        self.aldi_btn = ttk.Button(root, text="🥫 ALDI", command=self.open_aldi)
        self.lidl_btn = ttk.Button(root, text="🛒 LIDL", command=self.open_lidl)
        self.edeka_btn = ttk.Button(root, text="🍎 EDEKA", command=self.open_edeka)

        # place buttons
        self.rewe_btn.grid(row=1, column=3, padx=4, pady=2, sticky="we")
        self.dm_btn.grid(row=1, column=4, padx=4, pady=2, sticky="we")
        self.aldi_btn.grid(row=1, column=5, padx=4, pady=2, sticky="we")
        self.lidl_btn.grid(row=1, column=6, padx=4, pady=2, sticky="we")
        self.edeka_btn.grid(row=1, column=7, padx=4, pady=2, sticky="we")

        # disable initially
        for b in (
            self.rewe_btn,
            self.dm_btn,
            self.aldi_btn,
            self.lidl_btn,
            self.edeka_btn,
        ):
            b.state(["disabled"])

        active_syms = [
            n
            for n in LINEAR_SYMBOL_NAMES + ["QRCODE"]
            if getattr(ZBarSymbol, n, None) is not None
        ]
        self.status = ttk.Label(
            root,
            text=f"Live autoscan · Symbols: {', '.join(active_syms)}",
            foreground="blue",
        )
        self.status.grid(row=2, column=0, columnspan=8, pady=4)

        # State -----------------------------------------------------------------
        self._last_scan_t = 0.0
        self._last_boxes = []  # list of polygons (np arrays) to draw
        self._last_label = ""  # text label to display next to polygon
        self._decoded_once = False

        # Selenium driver persistence (reuse same window across clicks)
        self._driver = None

        # Start loop
        self.update_frame()

    # ---------------------- Selenium integration ------------------------------
    def _status_async(self, text, color="blue"):
        def _apply():
            self.status.config(text=text, foreground=color)

        self.root.after(0, _apply)

    def _ensure_driver(self):
        """
        Create (or reuse) a WebDriver with robust fallbacks:
          1) Firefox (geckodriver)
          2) Chrome (Selenium Manager)
          3) Chrome (Homebrew chromedriver path)
          4) Safari (safaridriver)
        """
        if not SELENIUM_AVAILABLE:
            raise RuntimeError(
                "Selenium not installed. Run: conda install -c conda-forge selenium"
            )

        # If we already have a living driver, reuse it
        if self._driver is not None:
            try:
                _ = self._driver.current_url
                return self._driver
            except Exception:
                self._driver = None  # recreate

        # 1) Firefox
        try:
            firefox_opts = webdriver.FirefoxOptions()
            firefox_opts.add_argument("--width=1280")
            firefox_opts.add_argument("--height=900")
            self._driver = webdriver.Firefox(options=firefox_opts)
            return self._driver
        except Exception:
            self._driver = None

        # 2) Chrome via Selenium Manager
        try:
            chrome_opts = webdriver.ChromeOptions()
            chrome_opts.add_argument("--start-maximized")
            chrome_opts.add_argument("--disable-backgrounding-occluded-windows")
            self._driver = webdriver.Chrome(options=chrome_opts)
            return self._driver
        except Exception:
            self._driver = None

        # 3) Chrome via Homebrew chromedriver path
        try:
            brew_paths = [
                "/opt/homebrew/bin/chromedriver",  # Apple Silicon
                "/usr/local/bin/chromedriver",  # Intel mac
            ]
            for p in brew_paths:
                if os.path.exists(p):
                    from selenium.webdriver.chrome.service import (
                        Service as ChromeService,
                    )

                    chrome_opts = webdriver.ChromeOptions()
                    chrome_opts.add_argument("--start-maximized")
                    chrome_opts.add_argument("--disable-backgrounding-occluded-windows")
                    self._driver = webdriver.Chrome(
                        service=ChromeService(p), options=chrome_opts
                    )
                    return self._driver
        except Exception:
            self._driver = None

        # 4) Safari (one-time: sudo /usr/bin/safaridriver --enable)
        try:
            self._driver = webdriver.Safari()
            return self._driver
        except Exception as e:
            self._driver = None
            raise RuntimeError(
                "Could not start any browser. Try:\n"
                "  A) conda install -c conda-forge firefox geckodriver\n"
                "  B) Install Google Chrome (and optionally brew install chromedriver)\n"
                "  C) sudo /usr/bin/safaridriver --enable"
            ) from e

    def _open_and_fill(self, url: str, selector: str, code_text: str, site_name: str):
        """Navigate to URL, wait for input by selector, fill code, scroll it into view."""

        def worker():
            try:
                self._status_async(f"Launching browser for {site_name}…", "blue")
                driver = self._ensure_driver()

                if not driver.current_url.startswith(url):
                    driver.get(url)

                self._status_async(f"Waiting for {site_name} page…", "blue")
                wait = WebDriverWait(driver, 30)
                field = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )

                try:
                    field.clear()
                except Exception:
                    pass
                field.send_keys(code_text)

                try:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({behavior:'smooth',block:'center'});",
                        field,
                    )
                except Exception:
                    pass

                self._status_async(
                    f"Code pasted on {site_name}. Complete PIN/CAPTCHA manually.",
                    "green",
                )
            except Exception as e:
                self._status_async(f"Selenium error ({site_name}): {e}", "red")

        threading.Thread(target=worker, daemon=True).start()

    def open_rewe(self):
        code_text = self.code.get().strip()
        if not code_text:
            self.status.config(
                text="No code to send. Hold code in the scanner.", foreground="red"
            )
            return
        self._open_and_fill(REWE_URL, REWE_SELECTOR, code_text, "REWE")

    def open_dm(self):
        code_text = self.code.get().strip()
        if not code_text:
            self.status.config(
                text="No code to send. Hold code in the scanner.", foreground="red"
            )
            return
        self._open_and_fill(DM_URL, DM_SELECTOR, code_text, "DM")

    def open_aldi(self):
        code_text = self.code.get().strip()
        if not code_text:
            self.status.config(
                text="No code to send. Hold code in the scanner.", foreground="red"
            )
            return
        self._open_and_fill(ALDI_URL, ALDI_SELECTOR, code_text, "ALDI")

    def open_lidl(self):
        code_text = self.code.get().strip()
        if not code_text:
            self.status.config(
                text="No code to send. Hold code in the scanner.", foreground="red"
            )
            return
        self._open_and_fill(LIDL_URL, LIDL_SELECTOR, code_text, "LIDL")

    def open_edeka(self):
        code_text = self.code.get().strip()
        if not code_text:
            self.status.config(
                text="No code to send. Hold code in the scanner.", foreground="red"
            )
            return
        self._open_and_fill(EDEKA_URL, EDEKA_SELECTOR, code_text, "EDEKA")

    # ---------------------- Main loop ----------------------------------------
    def update_frame(self):
        ok, frame = self.cap.read()
        if not ok:
            self.status.config(
                text="Camera read failed - retrying...", foreground="red"
            )
            self.root.after(20, self.update_frame)
            return

        h, w = frame.shape[:2]
        roi = self._compute_roi_rect(w, h)
        x0, y0, x1, y1 = roi

        # scan cadence
        now = time.time()
        if (now - self._last_scan_t) * 1000.0 >= SCAN_EVERY_MS:
            self._last_scan_t = now
            crop = frame[y0:y1, x0:x1]

            decoded = self._scan_1d(crop)

            # rotated pass if nothing found
            if not decoded:
                crop_rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                decoded_rot = self._scan_1d(crop_rot)
                if decoded_rot:
                    decoded = []
                    H, W = crop_rot.shape[:2]
                    for txt, sym, poly in decoded_rot:
                        poly = np.array(poly, dtype=np.int32)
                        poly_back = np.stack([poly[:, 1], W - 1 - poly[:, 0]], axis=1)
                        decoded.append((txt, sym, poly_back.tolist()))

            if decoded:
                txt, sym, poly = decoded[0]
                self.code.set(txt)
                self._decoded_once = True
                self.status.config(text=f"✓ {sym}: {txt}", foreground="green")
                for b in (
                    self.rewe_btn,
                    self.dm_btn,
                    self.aldi_btn,
                    self.lidl_btn,
                    self.edeka_btn,
                ):
                    b.state(["!disabled"])
                beep()

                # translate polygon to full-frame coords
                poly_np = np.array(poly, dtype=np.int32)
                if poly_np.ndim == 2 and poly_np.shape[1] == 2:
                    poly_np[:, 0] += x0
                    poly_np[:, 1] += y0
                    self._last_boxes = [(poly_np, BOX_COLOR, f"{sym}")]
                else:
                    self._last_boxes = []
                self._last_label = txt[:60]
            else:
                if not self._decoded_once:
                    self.status.config(
                        text="Scanning… tip: fill the scanner area, keep bars horizontal, avoid glare",
                        foreground="orange",
                    )

        # draw overlays
        vis = frame.copy()
        self._draw_scanner_overlay(vis, roi)
        self._draw_boxes(vis, self._last_boxes, self._last_label)

        # Tk show
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=img)
        self.label.image = img

        self.root.after(20, self.update_frame)

    # ---------------------- Scanner core -------------------------------------
    def _scan_1d(self, bgr):
        """Preprocess for 1D (“dashed” tolerant) + pyzbar decode."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # CLAHE improves local contrast
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        enhanced = clahe.apply(gray)

        # light unsharp mask
        blurred = cv2.GaussianBlur(enhanced, (0, 0), UNSHARP_SIGMA)
        sharp = cv2.addWeighted(
            enhanced, 1.0 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0
        )

        # morphological close with horizontal kernel to bridge dashed bars
        kx = max(3, MORPH_KERNEL_W | 1)  # ensure odd
        ky = max(1, MORPH_KERNEL_H | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        closed = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)

        # try decode on both closed and sharp
        results = decode(closed, symbols=SYMBOLS)
        if not results:
            results = decode(sharp, symbols=SYMBOLS)

        out = []
        for r in results or []:
            txt = r.data.decode("utf-8", errors="replace").strip()
            poly = [(p.x, p.y) for p in getattr(r, "polygon", [])] or []
            out.append((txt, r.type, poly))
        return out

    # ---------------------- Overlay drawing ----------------------------------
    def _compute_roi_rect(self, w, h):
        roi_h = int(h * ROI_HEIGHT_FRAC)
        roi_w = int(w * ROI_WIDTH_FRAC)
        x0 = (w - roi_w) // 2
        y0 = (h - roi_h) // 2
        x1 = x0 + roi_w
        y1 = y0 + roi_h
        return x0, y0, x1, y1

    def _draw_scanner_overlay(self, img, roi):
        x0, y0, x1, y1 = roi
        # dashed rectangle
        self._draw_dashed_rect(img, (x0, y0), (x1, y1), OVERLAY_COLOR, 2, DRAW_DASH_GAP)
        # corner brackets
        c = OVERLAY_COLOR
        L = CORNER_LEN
        th = 3
        # TL
        cv2.line(img, (x0, y0), (x0 + L, y0), c, th)
        cv2.line(img, (x0, y0), (x0, y0 + L), c, th)
        # TR
        cv2.line(img, (x1, y0), (x1 - L, y0), c, th)
        cv2.line(img, (x1, y0), (x1, y0 + L), c, th)
        # BL
        cv2.line(img, (x0, y1), (x0 + L, y1), c, th)
        cv2.line(img, (x0, y1), (x0, y1 - L), c, th)
        # BR
        cv2.line(img, (x1, y1), (x1 - L, y1), c, th)
        cv2.line(img, (x1, y1), (x1, y1 - L), c, th)

        # faint darken outside ROI
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        alpha = 0.20
        img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _draw_dashed_rect(self, img, p0, p1, color, thickness, gap):
        (x0, y0), (x1, y1) = p0, p1
        # top edge
        for x in range(x0, x1, gap * 2):
            cv2.line(img, (x, y0), (min(x + gap, x1), y0), color, thickness)
        # bottom edge
        for x in range(x0, x1, gap * 2):
            cv2.line(img, (x, y1), (min(x + gap, x1), y1), color, thickness)
        # left edge
        for y in range(y0, y1, gap * 2):
            cv2.line(img, (x0, y), (x0, min(y + gap, y1)), color, thickness)
        # right edge
        for y in range(y0, y1, gap * 2):
            cv2.line(img, (x1, y), (x1, min(y + gap, y1)), color, thickness)

    def _draw_boxes(self, img, items, label):
        for poly, color, typ in items:
            if poly is not None and len(poly) >= 4:
                pts = poly.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=2)
        if label:
            pad = 8
            txt = f"{label}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(
                img,
                (10, 10),
                (10 + tw + 2 * pad, 10 + th + 2 * pad),
                (230, 255, 230),
                -1,
            )
            cv2.putText(
                img,
                txt,
                (10 + pad, 10 + th + pad - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                TEXT_COLOR,
                2,
                cv2.LINE_AA,
            )

    # ---------------------- Cleanup ------------------------------------------
    def close(self):
        try:
            if self.cap:
                self.cap.release()
        finally:
            # keep any browser open so you can finish PIN/CAPTCHA
            self.root.destroy()


def main():
    root = tk.Tk()
    app = AutoBarcodeApp(root, camera_index=0)
    try:
        root.mainloop()
    finally:
        try:
            app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
