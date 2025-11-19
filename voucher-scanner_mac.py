#!/usr/bin/env python3

"""
Auto-scanning Barcode/QR/OCR Reader (Tkinter + OpenCV + pyzbar) + Selenium shop buttons

Shops supported:
  - REWE:   https://kartenwelt.rewe.de/rewe-geschenkkarte.html
  - DM:     https://www.dm.de/services/services-im-markt/geschenkkarten
  - ALDI:   https://www.helaba.com/de/aldi/
  - LIDL:   https://www.lidl.at/c/geschenkkarte-guthabenabfrage/s10012116
  - EDEKA:  https://gutschein.avs.de/edeka-mh/home.htm

Environment (conda):

  conda create -n voucher-scan -c conda-forge python=3.12 opencv pillow tk zbar pyzbar selenium

  conda activate voucher-scan

  # For OCR fallback (optional):

  # conda install -c conda-forge pytesseract tesseract

  python voucher_scanner_shops.py

"""

import os
import platform
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# ---- Camera configuration (webcam or IP stream) ----------------------------

# Standard: "0" = erste lokale Webcam
# Für IP-Kamera: z. B. "http://192.168.0.23:8080/video"
# CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "0")
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "http://192.168.178.46:8080/video")


def create_capture(source: str) -> cv2.VideoCapture:
    """
    Erzeugt ein cv2.VideoCapture-Objekt.
    - Wenn 'source' eine Zahl ist (z.B. "0", "1"): lokale Kamera.
    - Andernfalls: IP-Stream-URL (z.B. "http://.../video").
    """
    try:
        idx = int(source)
        return cv2.VideoCapture(idx)
    except ValueError:
        # Keine Zahl -> als URL/String behandeln
        return cv2.VideoCapture(source)


# ---- Optional beep on success (cross-platform) --------------------------

try:
    # --- Windows ---

    import winsound

    def beep(freq=1500, dur=120):
        try:
            winsound.Beep(freq, dur)

        except Exception:
            pass  # Fail silently (e.g., no sound card)

except ImportError:
    # --- Not Windows ---

    if platform.system() == "Darwin":  # macOS

        def beep(*args, **kwargs):
            try:
                # Use 'afplay' on macOS. Non-blocking.

                subprocess.Popen(
                    ["afplay", "/System/Library/Sounds/Purr.aiff"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )

            except Exception:
                pass  # Fail silently

    else:
        # --- Linux or other ---

        def beep(*args, **kwargs):
            # No-op for other systems

            pass


# ---- Barcode backend (pyzbar) -----------------------------------------------

try:
    from pyzbar.pyzbar import ZBarSymbol, decode

except Exception as e:
    print("ERROR: pyzbar not available:", e)

    print("Install in your conda env: conda install -c conda-forge pyzbar zbar")

    sys.exit(1)


# ---- Optional OCR backend (pytesseract) --------------------------------------

try:
    import pytesseract

    TESSERACT_AVAILABLE = True

except Exception:
    TESSERACT_AVAILABLE = False

    print("WARNING: pytesseract not available, OCR fallback disabled.")


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

#

# !!! --- ACTION REQUIRED --- !!!

# You must find the CSS selector for the 4-digit PIN field for each shop

# and replace 'None' with the correct selector (e.g., "#pin-input").

#

SHOPS = {
    "REWE": {
        "url": "https://kartenwelt.rewe.de/rewe-geschenkkarte.html",
        "card_selector": "#card_number",
        "pin_selector": "#pin",  # <-- FIND AND REPLACE 'None'
        "emoji": "🛒",
    },
    "DM": {
        "url": "https://www.dm.de/services/services-im-markt/geschenkkarten",
        "card_selector": "#credit-checker-printedCreditKey-input",
        "pin_selector": "#credit-checker-verificationCode-input",  # <-- FIND AND REPLACE 'None'
        "emoji": "🛍️",
    },
    "ALDI": {
        "url": "https://www.helaba.com/de/aldi/",
        "iframe_selector": 'iframe[src*="balancechecks"]',  # <-- UPDATED: Find iframe by its URL
        "card_selector": ".cardnumberfield",
        "pin_selector": ".pin",  # <-- Please double-check this one on the ALDI page
        "emoji": "🥫",
    },
    "LIDL": {
        "url": "https://www.lidl.at/c/geschenkkarte-guthabenabfrage/s10012116",
        "iframe_selector": "#gift-card-balance-check-iframe",
        "card_selector": ".cardnumberfield",
        "pin_selector": ".pin",
        "emoji": "🍍",
    },
    "EDEKA": {
        "url": "https://gutschein.avs.de/edeka-mh/home.htm",
        "card_selector": '#postform > div > div:nth-child(5) > div > div > input[type="text"]:nth-child(1)',
        "pin_selector": None,  # <-- FIND AND REPLACE 'None'
        "emoji": "🍎",
    },
}


SELENIUM_AVAILABLE = True

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait

except Exception:
    SELENIUM_AVAILABLE = False


# ---- Tuning parameters (scanner) --------------------------------------------

SCAN_EVERY_MS = 150  # scanning cadence (ms)
STABLE_THRESHOLD = 3  # consecutive frames to be "sure"
MIN_OCR_DIGITS = 10  # Ignore OCR card numbers shorter than this
MAX_OCR_DIGITS = 24  # Ignore OCR card numbers longer than this
PIN_DIGITS = 4  # Look for a 4-digit PIN
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
SUCCESS_COLOR = (0, 220, 0)  # BGR for green


class AutoBarcodeApp:
    # MODIFIED: Added iframe_selector to button handler

    def __init__(self, root: tk.Tk, camera_source: str = CAMERA_SOURCE):
        self.root = root

        root.title("Auto 1D Barcode Scanner + REWE / DM / ALDI / LIDL / EDEKA")

        # Set default window size - more compact
        root.geometry("660x680")

        # Camera ----------------------------------------------------------------

        self.cap = create_capture(camera_source)

        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error",
                "Could not open the camera. Check: System Settings → Privacy & Security → Camera.",
            )

            root.destroy()

            sys.exit(1)

        # Best-effort hints

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280 / 2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720 / 2)
        self.cap.set(cv2.CAP_PROP_FPS, 60)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # UI --------------------------------------------------------------------

        # Video area at the top
        self.label = ttk.Label(root)  # image area
        self.label.grid(row=0, column=0, columnspan=7, padx=10, pady=0, sticky="nsew")

        # Make the window resizable - video area expands
        # root.grid_rowconfigure(0, weight=1)
        # for i in range(7):
        #     root.grid_columnconfigure(i, weight=1)

        controls_frame = ttk.LabelFrame(root, text="")

        # Place the *entire frame* on row 2 of the main window's grid.
        # columnspan=4 just in case your grid has other items. Adjust as needed.
        controls_frame.grid(row=2, column=0, columnspan=4, sticky="w", padx=12)

        # --- Add widgets INSIDE the controls_frame using .pack() ---

        # Card-Number Label
        ttk.Label(controls_frame, text="Card-Number:").pack(side="left")

        # Card-Number Entry
        self.code = tk.StringVar()
        ttk.Entry(controls_frame, textvariable=self.code, width=15).pack(
            side="left", padx=(2, 12)
        )

        # PIN Label
        ttk.Label(controls_frame, text="PIN:").pack(side="left")

        # PIN Entry
        self.pin = tk.StringVar()
        ttk.Entry(controls_frame, textvariable=self.pin, width=6).pack(
            side="left", padx=(2, 12)
        )

        # Reset Button
        self.reset_btn = ttk.Button(
            controls_frame, text="Reset 🔄", command=self.reset_scan, width=8
        )
        self.reset_btn.pack(side="left")  # Add padx=10 if you want space before it
        self.reset_btn.state(["disabled"])

        # Shop buttons below - second row
        shop_frame = ttk.LabelFrame(root, text="")
        shop_frame.grid(row=3, column=0, columnspan=4, sticky="w", padx=12)
        self.shop_buttons = []
        # We no longer need col_idx or complex row calculations
        for name, config in SHOPS.items():
            # Your handler logic remains the same
            handler = lambda n=name, cfg=config: self._open_shop(
                n,
                cfg["url"],
                cfg["card_selector"],
                cfg.get("pin_selector"),
                cfg.get("iframe_selector"),
            )

            # Create the button with 'shop_frame' as its parent
            btn = ttk.Button(
                shop_frame, text=f"{config['emoji']} {name}", command=handler, width=8
            )

            # Pack it to the left, right after the previous widget
            btn.pack(side="left", padx=2)  # padx=2 adds a small space between buttons

            btn.state(["disabled"])
            self.shop_buttons.append(btn)

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

        self.status.grid(row=1, column=0, columnspan=7, sticky="w", padx=12)

        # State -----------------------------------------------------------------

        self._last_scan_t = 0.0
        self._last_boxes = []
        self._last_label = ""
        self.scanning = True

        # --- Stability state ---

        self._potential_code = ""
        self._potential_code_type = ""
        self._potential_code_count = 0
        self._stable_code = ""
        self._potential_pin = ""
        self._potential_pin_count = 0
        self._stable_pin = ""
        self.STABLE_THRESHOLD = STABLE_THRESHOLD
        self._driver = None
        self.update_frame()

    def reset_scan(self):
        """Resets the scanner to its initial state."""

        self._potential_code = ""
        self._potential_code_count = 0
        self._stable_code = ""
        self.code.set("")
        self._potential_pin = ""
        self._potential_pin_count = 0
        self._stable_pin = ""
        self.pin.set("")
        self.reset_btn.state(["disabled"])

        for btn in self.shop_buttons:
            btn.state(["disabled"])

        self._last_boxes = []
        self._last_label = ""
        self.status.config(text="Scanning...", foreground="blue")
        self.scanning = True
        self.root.after(20, self.update_frame)

    # ---------------------- Selenium integration ------------------------------

    def _status_async(self, text, color="blue"):
        def _apply():
            self.status.config(text=text, foreground=color)

        self.root.after(0, _apply)

    def _ensure_driver(self):
        """

        Create (or reuse) a WebDriver with robust fallbacks.

        """

        if not SELENIUM_AVAILABLE:
            raise RuntimeError(
                "Selenium not installed. Run: conda install -c conda-forge selenium"
            )

        if self._driver is not None:
            try:
                _ = self._driver.current_url
                return self._driver

            except Exception:
                self._driver = None

        try:
            firefox_opts = webdriver.FirefoxOptions()
            firefox_opts.add_argument("--width=1280")
            firefox_opts.add_argument("--height=900")
            self._driver = webdriver.Firefox(options=firefox_opts)
            return self._driver

        except Exception:
            self._driver = None

        try:
            chrome_opts = webdriver.ChromeOptions()
            chrome_opts.add_argument("--start-maximized")
            chrome_opts.add_argument("--disable-backgrounding-occluded-windows")
            self._driver = webdriver.Chrome(options=chrome_opts)

            return self._driver

        except Exception:
            self._driver = None

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

        try:
            self._driver = webdriver.Safari()

            return self._driver

        except Exception as e:
            self._driver = None

            raise RuntimeError(
                "Could not start any browser. Try:\n"
                "  A) conda install -c conda-forge firefox geckodriver\n"
                "  B) Install Google Chrome (and optionally brew install chromedriver)\n"
                "  C) sudo /usr/bin.safaridriver --enable"
            ) from e

    # MODIFIED: Upgraded function to handle iframe_selector

    def _open_and_fill(
        self,
        url: str,
        card_selector: str,
        code_text: str,
        pin_selector: str,
        pin_text: str,
        site_name: str,
        iframe_selector: str = None,  # NEW
    ):
        """Navigate to URL, wait for inputs, fill code and PIN."""

        def worker():
            driver = None
            switched_to_iframe = False

            try:
                self._status_async(f"Launching browser for {site_name}…", "blue")

                driver = self._ensure_driver()

                if not driver.current_url.startswith(url):
                    driver.get(url)

                self._status_async(f"Waiting for {site_name} page…", "blue")

                wait = WebDriverWait(driver, 30)

                # --- NEW: IFRAME LOGIC ---

                if iframe_selector:
                    self._status_async(f"Switching to iframe on {site_name}…", "blue")

                    iframe = wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, iframe_selector)
                        )
                    )

                    driver.switch_to.frame(iframe)

                    switched_to_iframe = True

                # --- END IFRAME LOGIC ---

                # --- Fill Card Number ---

                # (This logic is now inside the iframe, if applicable)

                field_card = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, card_selector))
                )

                try:
                    field_card.clear()

                except Exception:
                    pass

                field_card.send_keys(code_text)

                # --- Fill PIN if provided ---

                if pin_selector and pin_text:
                    # (This logic is also inside the iframe, if applicable)

                    self._status_async(f"Waiting for {site_name} PIN field…", "blue")

                    field_pin = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, pin_selector))
                    )

                    try:
                        field_pin.clear()

                    except Exception:
                        pass

                    field_pin.send_keys(pin_text)

                # --- Scroll main field into view ---

                try:
                    driver.execute_script(
                        "arguments[0].scrollIntoView({behavior:'smooth',block:'center'});",
                        field_card,  # Scroll card field
                    )
                except Exception:
                    pass

                self._status_async(
                    f"Code and PIN pasted on {site_name}. Complete CAPTCHA manually.",
                    "green",
                )
            except Exception as e:
                self._status_async(f"Selenium error ({site_name}): {e}", "red")

            finally:
                # --- NEW: Robustly switch back ---
                if switched_to_iframe and driver:
                    try:
                        driver.switch_to.default_content()
                    except Exception as e:
                        print(f"Warning: could not switch back from iframe: {e}")

        threading.Thread(target=worker, daemon=True).start()

    # MODIFIED: Upgraded function to handle iframe_selector

    def _open_shop(
        self,
        site_name: str,
        url: str,
        card_selector: str,
        pin_selector: str = None,
        iframe_selector: str = None,  # NEW
    ):
        """Generic handler for all shop buttons."""

        code_text = self.code.get().strip()

        pin_text = self.pin.get().strip()

        if not code_text:
            self.status.config(
                text="No code to send. Hold code in the scanner.", foreground="red"
            )

            return

        if pin_selector and not pin_text:
            self.status.config(
                text=f"PIN required for {site_name} but not found. Please rescan.",
                foreground="red",
            )

            return

        # Pass all selectors to the fill function

        self._open_and_fill(
            url,
            card_selector,
            code_text,
            pin_selector,
            pin_text,
            site_name,
            iframe_selector,  # NEW
        )

    # ---------------------- Main loop ----------------------------------------

    def _scan_barcodes_only(self, bgr):
        """Performs only the pyzbar scan on a pre-processed image."""
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        enhanced = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced, (0, 0), UNSHARP_SIGMA)
        sharp = cv2.addWeighted(
            enhanced, 1.0 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0
        )
        kx = max(3, MORPH_KERNEL_W | 1)
        ky = max(1, MORPH_KERNEL_H | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        closed = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)
        results = decode(closed, symbols=SYMBOLS)

        if not results:
            results = decode(sharp, symbols=SYMBOLS)

        out = []

        for r in results or []:
            txt = r.data.decode("utf-8", errors="replace").strip()

            poly = [(p.x, p.y) for p in getattr(r, "polygon", [])] or []

            out.append((txt, r.type, poly))

        return out

    def update_frame(self):
        if not self.scanning:
            return

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

        vis = frame.copy()

        now = time.time()

        if (now - self._last_scan_t) * 1000.0 >= SCAN_EVERY_MS:
            self._last_scan_t = now
            crop = frame[y0:y1, x0:x1]

            decoded_dict = self._scan_1d(crop)
            card_info = decoded_dict.get("card")
            pin_info = decoded_dict.get("pin")

            # Rotated pass if *card* not found

            if not card_info:
                crop_rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
                decoded_rot = self._scan_barcodes_only(crop_rot)

                if decoded_rot:
                    txt, sym, poly = decoded_rot[0]
                    H, W = crop_rot.shape[:2]
                    poly = np.array(poly, dtype=np.int32)
                    poly_back = np.stack([poly[:, 1], W - 1 - poly[:, 0]], axis=1)
                    card_info = (txt, sym, poly_back.tolist())

            # --- *** STABILITY LOGIC *** ---

            # --- 1. Handle Card Stability ---

            if card_info:
                txt, sym, poly = card_info

                if txt == self._potential_code:
                    self._potential_code_count += 1
                else:
                    self._potential_code = txt
                    self._potential_code_type = sym
                    self._potential_code_count = 1

                poly_np = np.array(poly, dtype=np.int32)

                if poly_np.ndim == 2 and poly_np.shape[1] == 2:
                    poly_np[:, 0] += x0
                    poly_np[:, 1] += y0
                    self._last_boxes = [(poly_np, BOX_COLOR, f"{sym}")]
                else:
                    self._last_boxes = []

            else:
                self._potential_code = ""
                self._potential_code_count = 0
                self._last_boxes = []

            # --- 2. Handle PIN Stability ---

            if pin_info:
                pin_txt, _, _ = pin_info
                if pin_txt == self._potential_pin:
                    self._potential_pin_count += 1
                else:
                    self._potential_pin = pin_txt
                    self._potential_pin_count = 1

            else:
                self._potential_pin = ""
                self._potential_pin_count = 0

            # --- 3. Check for Lock-in ---

            if self._potential_code_count >= self.STABLE_THRESHOLD:
                is_new_lock = self._potential_code != self._stable_code

                self._stable_code = self._potential_code

                self.code.set(self._stable_code)

                self._last_label = self._stable_code

                status_text = (
                    f"✓ Locked: {self._potential_code_type}: {self._stable_code}"
                )

                if self._potential_pin_count >= self.STABLE_THRESHOLD:
                    self._stable_pin = self._potential_pin

                    self.pin.set(self._stable_pin)

                    self._last_label += f" | PIN: {self._stable_pin}"

                    status_text += f" | PIN: {self._stable_pin}"

                self.status.config(text=status_text, foreground="green")

                if is_new_lock:
                    for btn in self.shop_buttons:
                        btn.state(["!disabled"])
                    beep()
                    self.scanning = False
                    self.reset_btn.state(["!disabled"])

            elif self._potential_code_count > 0:
                status_text = f"Tracking ({self._potential_code_count}/{self.STABLE_THRESHOLD}): {self._potential_code}"
                self._last_label = self._potential_code

                if self._potential_pin_count > 0:
                    status_text += f" | PIN ({self._potential_pin_count}/{self.STABLE_THRESHOLD}): {self._potential_pin}"
                    self._last_label += f" | PIN: {self._potential_pin}"
                self.status.config(text=status_text, foreground="orange")

            else:
                self._last_label = ""

                if not self._stable_code:
                    self.status.config(
                        text="Scanning… tip: fill the scanner area, keep bars horizontal, avoid glare",
                        foreground="blue",
                    )

            # --- *** END OF STABILITY LOGIC *** ---
        self._draw_scanner_overlay(vis, roi, success=(not self.scanning))
        self._draw_boxes(vis, self._last_boxes, self._last_label)
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=img)
        self.label.image = img
        if self.scanning:
            self.root.after(20, self.update_frame)

    # ---------------------- Scanner core -------------------------------------

    def _scan_1d(self, bgr):
        """Scan for barcodes/QR, and OCR for PIN and card fallback."""

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)

        enhanced = clahe.apply(gray)

        blurred = cv2.GaussianBlur(enhanced, (0, 0), UNSHARP_SIGMA)

        sharp = cv2.addWeighted(
            enhanced, 1.0 + UNSHARP_AMOUNT, blurred, -UNSHARP_AMOUNT, 0
        )

        kx = max(3, MORPH_KERNEL_W | 1)
        ky = max(1, MORPH_KERNEL_H | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky))
        closed = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITER)

        # --- 1. Barcode Scan (for Card Number) ---
        results = decode(closed, symbols=SYMBOLS)
        if not results:
            results = decode(sharp, symbols=SYMBOLS)

        barcode_result = None

        if results:
            r = results[0]
            txt = r.data.decode("utf-8", errors="replace").strip()
            poly = [(p.x, p.y) for p in getattr(r, "polygon", [])] or []
            barcode_result = (txt, r.type, poly)

        # --- 2. OCR Scan (for PIN and Card Fallback) ---
        ocr_card_result = None
        ocr_pin_result = None

        try:
            raw_ocr_output = pytesseract.image_to_string(
                sharp,
                config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789",
            )

            all_numbers = raw_ocr_output.split()

            full_frame_poly = [
                (0, 0),
                (bgr.shape[1], 0),
                (bgr.shape[1], bgr.shape[0]),
                (0, bgr.shape[0]),
            ]

            # --- 2a. Find PIN (always) ---

            valid_pins = [n for n in all_numbers if len(n) == PIN_DIGITS]

            if valid_pins:
                ocr_pin_result = (valid_pins[0], "PIN", full_frame_poly)

            # --- 2b. Find Card Number (only if barcode failed) ---

            if not barcode_result:
                valid_card_numbers = [
                    n for n in all_numbers if MIN_OCR_DIGITS <= len(n) <= MAX_OCR_DIGITS
                ]
                valid_digit_numbers = [
                    n for n in valid_card_numbers if len(n) == [13, 24, 20]
                ]

                card_num_str = None

                if valid_digit_numbers:
                    card_num_str = valid_digit_numbers[0]
                elif valid_card_numbers:
                    card_num_str = max(valid_card_numbers, key=len)
                if card_num_str:
                    ocr_card_result = (card_num_str, "OCR", full_frame_poly)

        except Exception as e:
            print(f"WARNING: pytesseract fallback failed: {e}")

        # --- 3. Consolidate Results ---

        final_card = barcode_result or ocr_card_result

        return {"card": final_card, "pin": ocr_pin_result}

    # ---------------------- Overlay drawing ----------------------------------

    def _compute_roi_rect(self, w, h):
        roi_h = int(h * ROI_HEIGHT_FRAC)
        roi_w = int(w * ROI_WIDTH_FRAC)
        x0 = (w - roi_w) // 2
        y0 = (h - roi_h) // 2
        x1 = x0 + roi_w
        y1 = y0 + roi_h

        return x0, y0, x1, y1

    def _draw_scanner_overlay(self, img, roi, success=False):
        x0, y0, x1, y1 = roi

        if success:
            # Draw solid green border and tint
            cv2.rectangle(img, (x0, y0), (x1, y1), SUCCESS_COLOR, 4)
            overlay = img.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), SUCCESS_COLOR, -1)
            alpha = 0.25  # Transparency
            img[y0:y1, x0:x1] = cv2.addWeighted(
                overlay[y0:y1, x0:x1], alpha, img[y0:y1, x0:x1], 1 - alpha, 0
            )

        else:
            # Draw dashed lines and corners

            self._draw_dashed_rect(
                img, (x0, y0), (x1, y1), OVERLAY_COLOR, 2, DRAW_DASH_GAP
            )
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

        # Faint darken outside ROI

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)

        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)  # "punch hole"

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
            self.scanning = False
            if self.cap:
                self.cap.release()
        finally:
            # keep any browser open so you can finish PIN/CAPTCHA
            self.root.destroy()


def main():
    root = tk.Tk()
    app = AutoBarcodeApp(root, camera_source=CAMERA_SOURCE)
    try:
        root.mainloop()
    finally:
        try:
            app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
