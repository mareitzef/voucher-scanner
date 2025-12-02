#!/usr/bin/env python3

"""
Picture-based Barcode/QR/OCR Reader (Tkinter + OpenCV + pyzbar) + Selenium shop buttons

Shops supported:
  - REWE, DM, ALDI, LIDL, EDEKA

Usage:
  1. Adjust camera view with live video
  2. Click "Take Picture" when ready
  3. Code scans the frozen image
  4. Click shop buttons to fill forms
  5. Click "New Scan" to take another picture

  TODO:
  - set number of digits per shop
  - try freifunk which is closer to cam
  - bugfix, sometimes I have to press picture button several times
"""

import os, time
import platform
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
import pytesseract
from pytesseract import Output

import cv2
import numpy as np
from PIL import Image, ImageTk
from dotenv import load_dotenv
import os

load_dotenv()

# ---- Camera configuration ------------------------------------------------
IP_phone = os.getenv("IP_PHONE")
# IP_phone = "10.60.142.22"

CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", f"https://{IP_phone}:8080/video")

RES_PHONE_WIDTH = 1280
RES_PHONE_HEIGHT = 720


def create_capture(source: str) -> cv2.VideoCapture:
    """Create VideoCapture object from index or URL."""
    try:
        idx = int(source)
        return cv2.VideoCapture(idx)
    except ValueError:
        return cv2.VideoCapture(source)


# ---- Beep on success -----------------------------------------------------
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

except ImportError:
    if platform.system() == "Darwin":  # macOS

        def beep(*args, **kwargs):
            try:
                subprocess.Popen(
                    ["afplay", "/System/Library/Sounds/Purr.aiff"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass

    else:

        def beep(*args, **kwargs):
            pass


# ---- Barcode backend -----------------------------------------------------
try:
    from pyzbar.pyzbar import ZBarSymbol, decode
except Exception as e:
    print("ERROR: pyzbar not available:", e)
    print("Install: conda install -c conda-forge pyzbar zbar")
    sys.exit(1)


# ---- Optional OCR backend ------------------------------------------------
try:
    import pytesseract

    TESSERACT_AVAILABLE = True
except Exception:
    TESSERACT_AVAILABLE = False
    print("WARNING: pytesseract not available, OCR fallback disabled.")


def zbar_symbols(names):
    """Return available ZBarSymbol members."""
    out = []

    for n in names:
        sym = getattr(ZBarSymbol, n, None)

        if sym is not None:
            out.append(sym)

    return out


LINEAR_SYMBOL_NAMES = [
    "EAN13",
    "EAN8",
    "UPCA",
    "UPCE",
    "CODE128",
    "CODE39",
    "CODE93",
    "I25",  # Interleaved 2 of 5
    "CODABAR",
    "CODE32",  # Italian Pharmacode
    "DATABAR",
    "DATABAR_EXP",
]
SYMBOLS = zbar_symbols(LINEAR_SYMBOL_NAMES + ["QRCODE", "PDF417"])

# ---- Shop configurations -------------------------------------------------
SHOPS = {
    "REWE": {
        "url": "https://kartenwelt.rewe.de/rewe-geschenkkarte.html",
        "card_selector": "#card_number",
        "pin_selector": "#pin",
        "emoji": "🛒",
    },
    "DM": {
        "url": "https://www.dm.de/services/services-im-markt/geschenkkarten",
        "card_selector": "#credit-checker-printedCreditKey-input",
        "pin_selector": "#credit-checker-verificationCode-input",
        "emoji": "🛍️",
    },
    "ALDI": {
        "url": "https://www.helaba.com/de/aldi/",
        "iframe_selector": 'iframe[src*="balancechecks"]',
        "card_selector": ".cardnumberfield",
        "pin_selector": ".pin",
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
        "pin_selector": None,
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


# ---- Tuning parameters ---------------------------------------------------
MIN_OCR_DIGITS = 10
MAX_OCR_DIGITS = 24
PIN_DIGITS = 4
ROI_HEIGHT_FRAC = 0.35
ROI_WIDTH_FRAC = 0.90
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
UNSHARP_AMOUNT = 1.4
UNSHARP_SIGMA = 1.0
MORPH_KERNEL_W = 21
MORPH_KERNEL_H = 3
MORPH_ITER = 1
DRAW_DASH_GAP = 14
CORNER_LEN = 26
OVERLAY_COLOR = (0, 200, 255)  # BGR
BOX_COLOR = (0, 220, 0)
TEXT_COLOR = (20, 20, 20)
SUCCESS_COLOR = (0, 220, 0)


class VoucherScannerApp:

    def __init__(self, root: tk.Tk, camera_source: str = CAMERA_SOURCE):
        self.root = root
        self.camera_source = camera_source  # Store for reconnection
        root.title("Voucher Scanner - Picture Mode")
        geometry_width = round(RES_PHONE_WIDTH + RES_PHONE_WIDTH * 0.1)
        geometry_height = round(RES_PHONE_HEIGHT + RES_PHONE_HEIGHT * 0.3)
        root.geometry(f"{geometry_width}x{geometry_height}")

        # Camera setup
        self.cap = create_capture(camera_source)
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error", "Could not open the camera. Check camera settings."
            )

            root.destroy()

            sys.exit(1)

        # Camera hints
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_PHONE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_PHONE_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

        # UI - Video display
        self.label = ttk.Label(root)
        self.label.grid(row=0, column=0, columnspan=7, padx=10, pady=5, sticky="nsew")

        # Status label
        self.status = ttk.Label(
            root, text="📹 Live video - Ready to capture", foreground="blue", anchor="w"
        )
        self.status.grid(row=1, column=0, columnspan=7, sticky="we", padx=12, pady=5)
        # prevent the status row from expanding vertically when long messages arrive
        try:
            root.grid_rowconfigure(1, minsize=24)
        except Exception:
            pass

        # Controls frame
        controls_frame = ttk.LabelFrame(root, text="")
        controls_frame.grid(row=2, column=0, columnspan=4, sticky="w", padx=12, pady=5)

        ttk.Label(controls_frame, text="Card Number:").pack(side="left")
        self.code = tk.StringVar()
        # Wider entry for card number
        ttk.Entry(controls_frame, textvariable=self.code, width=30).pack(
            side="left", padx=(2, 12)
        )

        ttk.Label(controls_frame, text="PIN:").pack(side="left")
        self.pin = tk.StringVar()
        ttk.Entry(controls_frame, textvariable=self.pin, width=6).pack(
            side="left", padx=(2, 12)
        )

        # Start browser hint + button (new row above shops)
        start_hint = ttk.Label(
            root, text="Start browser first (open shop tabs if needed)"
        )
        start_hint.grid(row=3, column=0, columnspan=4, sticky="w", padx=12, pady=(8, 0))

        start_frame = ttk.Frame(root)
        start_frame.grid(
            row=4, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 6)
        )

        # Take picture row (take picture button alone)
        take_hint = ttk.Label(
            root, text="Take picture (shop auto-selected except ALDI/LIDL)"
        )
        take_hint.grid(row=5, column=0, columnspan=4, sticky="w", padx=12, pady=(4, 0))

        take_frame = ttk.Frame(root)
        take_frame.grid(row=6, column=0, columnspan=4, sticky="w", padx=12, pady=(0, 6))

        # Shop buttons hint (will be placed below start browser)
        shop_hint = ttk.Label(
            root, text="Then choose the Shop (choose ALDI or LIDL if ambiguous)"
        )
        shop_hint.grid(row=7, column=0, columnspan=4, sticky="w", padx=12, pady=(6, 0))

        # Shop buttons frame (placed below start browser)
        shop_frame = ttk.LabelFrame(root, text="")
        shop_frame.grid(row=8, column=0, columnspan=4, sticky="w", padx=12, pady=5)

        # Action area hint
        action_hint = ttk.Label(
            root, text="After scan: press 'Fill Selected' to fill the selected shop"
        )
        action_hint.grid(
            row=9, column=0, columnspan=4, sticky="w", padx=12, pady=(6, 0)
        )

        # Action buttons frame (below shop buttons)
        action_frame = ttk.LabelFrame(root, text="")
        action_frame.grid(row=10, column=0, columnspan=4, sticky="w", padx=12, pady=5)

        # Style for selected/pressed buttons
        self._style = ttk.Style()
        try:
            self._style.configure("Selected.TButton", background="#b7ebc6")
            self._style.configure("Pressed.TButton", background="#d6f0ff")
            self._style.configure("Ambiguous.TButton", background="#ffd59e")
        except Exception:
            pass

        # Action buttons (order will be added to action_frame; start button lives in start_frame)
        self.start_browser_btn = ttk.Button(
            start_frame,
            text="Start Browser",
            command=lambda: (
                self._flash_button(self.start_browser_btn),
                self._manual_open_browser(),
            ),
            width=20,
        )
        self.start_browser_btn.pack(side="left", padx=2)

        self.take_picture_btn = ttk.Button(
            take_frame,
            text="Take Picture",
            command=lambda: (
                self._flash_button(self.take_picture_btn),
                self._take_picture(),
            ),
            width=20,
        )
        self.take_picture_btn.pack(side="left", padx=2)

        self.fill_selected_btn = ttk.Button(
            action_frame,
            text="Fill Selected",
            command=lambda: (
                self._flash_button(self.fill_selected_btn),
                self._fill_selected_shop(),
            ),
            width=14,
        )
        self.fill_selected_btn.pack(side="left", padx=2)

        self.reset_camera_btn = ttk.Button(
            action_frame,
            text="Reset Camera",
            command=lambda: (
                self._flash_button(self.reset_camera_btn),
                self._reset_camera(),
            ),
            width=14,
        )
        self.reset_camera_btn.pack(side="left", padx=2)

        self.shop_buttons = []
        for name, config in SHOPS.items():
            text = f"{name}"
            btn = ttk.Button(shop_frame, text=text, width=12, style="TButton")
            btn._orig_text = text
            btn._shop_name = name
            # Selecting a shop (does not open browser yet)
            btn.config(command=lambda n=name, b=btn: self._select_shop(n, b))
            btn.pack(side="left", padx=2)
            self.shop_buttons.append(btn)

        # State
        self.picture_mode = False
        self.frozen_frame = None
        self.last_boxes = []
        self.last_label = ""
        self._driver = None
        self._shop_windows = {}

        # Start live video
        self.update_live_video()

    def _select_shop(self, name, button):
        """Mark a shop as selected. Visual toggle on the button."""
        # Deselect previous (remove selected style)
        try:
            prev = getattr(self, "selected_shop_button", None)
            if prev and prev is not button:
                prev.config(style="TButton")
        except Exception:
            pass

        # Toggle selection
        if getattr(self, "selected_shop", None) == name:
            # Deselect: clear selection and reset styles for all shop buttons
            self.selected_shop = None
            try:
                for b in self.shop_buttons:
                    b.config(style="TButton")
            except Exception:
                pass
            self.selected_shop_button = None
            self.status.config(text="Shop deselected", foreground="blue")
        else:
            # Select this shop and set style green; reset others to default
            self.selected_shop = name
            self.selected_shop_button = button
            try:
                for b in self.shop_buttons:
                    try:
                        if getattr(b, "_shop_name", None) == name:
                            b.config(style="Selected.TButton")
                        else:
                            b.config(style="TButton")
                    except Exception:
                        pass
            except Exception:
                pass
            self.status.config(text=f"Selected shop: {name}", foreground="blue")

    def _flash_button(self, button, ms: int = 300):
        """Temporarily apply Pressed style to a button then revert."""
        try:
            orig_style = button.cget("style") if "style" in button.keys() else "TButton"
            button.config(style="Pressed.TButton")
            self.root.after(ms, lambda: button.config(style=orig_style))
        except Exception:
            pass

    def _fill_selected_shop(self):
        """Fill the selected shop's form in the browser using captured IDs."""
        if not getattr(self, "selected_shop", None):
            messagebox.showinfo("No shop selected", "Please select a shop first.")
            return

        cfg = SHOPS.get(self.selected_shop)
        if not cfg:
            messagebox.showerror("Shop Error", "Selected shop config not found.")
            return

        # Validate card number length according to shop rules
        code_raw = self.code.get() or ""
        digits = "".join(ch for ch in code_raw if ch.isdigit())

        def _validate_for(shop, ds: str):
            n = len(ds)
            if shop == "REWE":
                return (n == 13, ds, n)
            if shop == "DM":
                return (n == 24, ds, n)
            if shop in ("ALDI", "LIDL"):
                if n == 20:
                    return (True, ds, n)
                if n == 38:
                    # drop first 18 digits to get 20
                    corrected = ds[18:]
                    return (True, corrected, n)
                return (False, ds, n)
            if shop == "EDEKA":
                return (n == 16, ds, n)
            # default: accept if between min/max
            return (MIN_OCR_DIGITS <= n <= MAX_OCR_DIGITS, ds, n)

        ok, corrected_code, count = _validate_for(self.selected_shop, digits)
        if not ok:
            messagebox.showwarning(
                "Invalid code length",
                f"Detected {count} digits for {self.selected_shop}. Try again.",
            )
            return
        # If corrected (e.g., ALDI/LIDL 38->20), update the field
        if corrected_code != digits:
            self.code.set(corrected_code)
            digits = corrected_code

        # Ensure browser is started
        try:
            driver = self._ensure_driver()
        except Exception:
            # try to start browser manually
            self._manual_open_browser()
            # mark start button as active if driver available
            try:
                self.start_browser_btn.config(style="Selected.TButton")
            except Exception:
                pass

        # Now fill the form for the selected shop
        # call in background so GUI doesn't block
        threading.Thread(
            target=self._open_shop,
            args=(
                self.selected_shop,
                cfg["url"],
                cfg["card_selector"],
                cfg.get("pin_selector"),
                cfg.get("iframe_selector"),
            ),
            daemon=True,
        ).start()

    # ==================== Camera Control Methods ====================

    def _reset_camera(self):
        """Reset/reconnect the camera connection."""
        self.status.config(text="🔌 Resetting camera...", foreground="orange")
        self.root.update()

        try:
            # Release current connection
            if self.cap:
                self.cap.release()

            # Reconnect
            self.cap = create_capture(self.camera_source)
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", "Could not reconnect to camera.")
                self.status.config(text="❌ Camera connection failed", foreground="red")
                return

            # Set camera properties again
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, RES_PHONE_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RES_PHONE_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 20)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

            self.status.config(
                text="✅ Camera reconnected - Ready to capture", foreground="green"
            )
            # Reset picture mode in case we were frozen
            self.picture_mode = False
            self.frozen_frame = None

        except Exception as e:
            messagebox.showerror("Camera Error", f"Failed to reset camera: {e}")
            self.status.config(text="❌ Camera reset failed", foreground="red")

    # ==================== Picture Capture Methods ====================

    def _take_picture(self):
        """Capture current frame and process it."""
        ok, frame = self.cap.read()
        if not ok:
            self.status.config(text="❌ Failed to capture picture", foreground="red")
            return

        # Store frozen frame
        self.frozen_frame = frame.copy()
        self.picture_mode = True

        # Update button states
        self.take_picture_btn.state(["disabled"])

        self.status.config(text="⏳ Processing image...", foreground="orange")

        # Process the image asynchronously so the GUI remains responsive
        threading.Thread(target=self._process_frozen_frame, daemon=True).start()
        # Ensure the UI doesn't stay frozen longer than 2 seconds
        try:
            self.root.after(2000, self._auto_unfreeze)
        except Exception:
            pass

    def _new_scan(self):
        """Clear results and return to live video mode."""
        self.picture_mode = False
        self.frozen_frame = None
        self.last_boxes = []
        self.last_label = ""

        # Clear text fields
        self.code.set("")
        self.pin.set("")

        # Update button states
        self.take_picture_btn.state(["!disabled"])
        for btn in self.shop_buttons:
            btn.state(["disabled"])

        self.status.config(text="📹 Live video - Ready to capture", foreground="blue")

        # Resume live video
        self.update_live_video()

    def _auto_unfreeze(self):
        """Automatically exit picture mode after a short timeout.

        If the user didn't resume or processing hasn't finished, force return
        to live preview to avoid a permanent freeze.
        """
        if not getattr(self, "picture_mode", False):
            return

        # Reset state and UI
        self.picture_mode = False
        self.frozen_frame = None
        try:
            self.take_picture_btn.state(["!disabled"])
        except Exception:
            pass

        self.status.config(text="📹 Live video - Ready to capture", foreground="blue")
        try:
            self.update_live_video()
        except Exception:
            pass

    def _process_frozen_frame(self):
        """Scan and extract codes from the frozen frame."""
        if self.frozen_frame is None:
            return

        h, w = self.frozen_frame.shape[:2]
        roi = self._compute_roi_rect(w, h)
        x0, y0, x1, y1 = roi
        crop = self.frozen_frame[y0:y1, x0:x1]

        # Scan the cropped region
        decoded_dict = self._scan_1d(crop)
        card_info = decoded_dict.get("card")
        pin_info = decoded_dict.get("pin")

        # Try rotated if card not found
        if not card_info:
            crop_rot = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
            decoded_rot = self._scan_barcodes_only(crop_rot)
            if decoded_rot:
                txt, sym, poly = decoded_rot[0]
                H, W = crop_rot.shape[:2]
                poly = np.array(poly, dtype=np.int32)
                poly_back = np.stack([poly[:, 1], W - 1 - poly[:, 0]], axis=1)
                card_info = (txt, sym, poly_back.tolist())

        # Update UI with results
        if card_info:
            txt, sym, poly = card_info
            self.code.set(txt)

            poly_np = np.array(poly, dtype=np.int32)
            if poly_np.ndim == 2 and poly_np.shape[1] == 2:
                poly_np[:, 0] += x0
                poly_np[:, 1] += y0
                self.last_boxes = [(poly_np, BOX_COLOR, f"{sym}")]

            status_text = f"✅ Found {sym}: {txt}"

            if pin_info:
                pin_txt, _, _ = pin_info
                self.pin.set(pin_txt)
                status_text += f" | PIN: {pin_txt}"
                self.last_label = f"{txt} | PIN: {pin_txt}"
            else:
                self.last_label = txt

            self.status.config(text=status_text, foreground="green")

            # Enable shop buttons
            for btn in self.shop_buttons:
                btn.state(["!disabled"])
            # Auto-select shop when unambiguous (except ALDI/LIDL require manual choice)
            try:
                # extract digits from code text
                digits_only = "".join(ch for ch in txt if ch.isdigit())
                n = len(digits_only)
                candidates = []
                if n == 13:
                    candidates = ["REWE"]
                elif n == 24:
                    candidates = ["DM"]
                elif n == 16:
                    candidates = ["EDEKA"]
                elif n == 20 or n == 38:
                    # ALDI and LIDL both accept 20 (or 38 -> drop first 18)
                    candidates = ["ALDI", "LIDL"]

                if len(candidates) == 1:
                    shop_to_select = candidates[0]
                    # set selection and style
                    self.selected_shop = shop_to_select
                    # update previous selection style
                    try:
                        prev = getattr(self, "selected_shop_button", None)
                        if prev and prev._shop_name != shop_to_select:
                            prev.config(style="TButton")
                    except Exception:
                        pass
                    for b in self.shop_buttons:
                        try:
                            if getattr(b, "_shop_name", None) == shop_to_select:
                                b.config(style="Selected.TButton")
                                self.selected_shop_button = b
                            else:
                                b.config(style="TButton")
                        except Exception:
                            pass
                    self.status.config(
                        text=f"Auto-selected shop: {shop_to_select}", foreground="green"
                    )
                elif len(candidates) > 1:
                    # ambiguous ALDI/LIDL - require manual choice
                    # mark both ALDI and LIDL buttons as ambiguous (orange)
                    try:
                        for b in self.shop_buttons:
                            if getattr(b, "_shop_name", None) in ("ALDI", "LIDL"):
                                b.config(style="Ambiguous.TButton")
                            else:
                                b.config(style="TButton")
                    except Exception:
                        pass
                    self.status.config(
                        text=f"Ambiguous shop (ALDI/LIDL). Please choose.",
                        foreground="orange",
                    )
                else:
                    # no match
                    self.status.config(
                        text=f"Detected {n} digits — no matching shop. Try again.",
                        foreground="red",
                    )
            except Exception:
                pass
            beep()
        else:
            self.status.config(
                text="❌ No barcode/code found in picture", foreground="red"
            )
            self.last_boxes = []
            self.last_label = ""

        # Display the processed frozen frame
        self._display_frozen_frame()

    def _display_frozen_frame(self):
        """Display the frozen frame with overlays."""
        if self.frozen_frame is None:
            return

        vis = self.frozen_frame.copy()
        h, w = vis.shape[:2]
        roi = self._compute_roi_rect(w, h)

        # Draw overlays
        success = bool(self.code.get())
        self._draw_scanner_overlay(vis, roi, success=success)
        self._draw_boxes(vis, self.last_boxes, self.last_label)

        # Convert and display
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=img)
        self.label.image = img

    # ==================== Live Video Methods ====================

    def update_live_video(self):
        """Update live video feed (no processing)."""
        if self.picture_mode:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self.update_live_video)
            return

        # Draw overlay (no scanning)
        vis = frame.copy()
        h, w = vis.shape[:2]
        roi = self._compute_roi_rect(w, h)
        self._draw_scanner_overlay(vis, roi, success=False)

        # Display
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        img = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.label.configure(image=img)
        self.label.image = img

        self.root.after(30, self.update_live_video)

    # ==================== Scanning Methods ====================
    def _scan_1d_old(self, bgr):
        import cv2, numpy as np
        from pyzbar.pyzbar import decode
        import pytesseract

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # -----------------------
        # 1) BARCODE first
        # -----------------------
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

        barcode_result = None
        if results:
            r = results[0]
            txt = r.data.decode("utf-8", errors="replace").strip()
            poly = [(p.x, p.y) for p in getattr(r, "polygon", [])] or []
            barcode_result = (txt, r.type, poly)

        if barcode_result:
            return {"card": barcode_result, "pin": None}

        # -----------------------
        # 2) OCR fallback for multi-line numeric ID
        # -----------------------
        h, w = gray.shape
        # Large crop because number layouts vary
        roi = gray[int(0.30 * h) : int(0.98 * h), int(0.03 * w) : int(0.97 * w)]

        # --- candidates for iterative OCR ---
        transforms = [
            lambda x: x,
            lambda x: cv2.bilateralFilter(x, 9, 40, 40),
            lambda x: cv2.GaussianBlur(x, (5, 5), 0),
            lambda x: cv2.adaptiveThreshold(
                x, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10
            ),
            lambda x: cv2.adaptiveThreshold(
                x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 8
            ),
            lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda x: cv2.medianBlur(x, 3),
            lambda x: cv2.resize(
                x, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC
            ),
        ]

        def run_ocr(img):
            txt = pytesseract.image_to_string(
                img, config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
            )
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            nums = ["".join(ch for ch in ln if ch.isdigit()) for ln in lines]
            nums = [n for n in nums if MIN_OCR_DIGITS <= len(n) <= MAX_OCR_DIGITS]
            if nums:
                return max(nums, key=len)
            return None

        # Try sequential transforms until one produces a valid ID
        best = None
        img = roi.copy()
        for t in transforms:
            try:
                mod = t(img)
                if mod.ndim == 3:
                    mod = cv2.cvtColor(mod, cv2.COLOR_BGR2GRAY)
                n = run_ocr(mod)
                if n:
                    best = n
                    break
            except:
                continue

        if best:
            poly = [(0, 0), (w, 0), (w, h), (0, h)]
            return {"card": (best, "OCR", poly), "pin": None}

        return {"card": None, "pin": None}

    def _scan_1d(self, bgr):
        import cv2, numpy as np
        from pyzbar.pyzbar import decode
        import pytesseract

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # -----------------------
        # 1) BARCODE first - Build preprocessing candidates
        # -----------------------
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

        # Build candidate list
        candidates = []
        candidates.append(("gray", gray))
        candidates.append(("sharp", sharp))
        candidates.append(("closed", closed))
        candidates.append(("enhanced", enhanced))

        # Binary thresholding variants
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        candidates.append(("binary", binary))

        _, inv_binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        candidates.append(("inv_binary", inv_binary))

        adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        candidates.append(("adaptive", adaptive))

        # FIRST PASS: Try all candidates normally
        for name, img in candidates:
            results = decode(img, symbols=SYMBOLS)
            if results:
                r = results[0]
                txt = r.data.decode("utf-8", errors="replace").strip()
                poly = [(p.x, p.y) for p in getattr(r, "polygon", [])] or []
                return {"card": (txt, r.type, poly), "pin": None}

        # SECOND PASS: Try scaling on best candidates
        best_for_scaling = ["binary", "sharp", "closed"]
        for name, img in candidates:
            if name in best_for_scaling:
                results = self._try_multiple_scales(img)
                if results:
                    r = results[0]
                    txt = r.data.decode("utf-8", errors="replace").strip()
                    poly = [(p.x, p.y) for p in getattr(r, "polygon", [])] or []
                    return {"card": (txt, r.type, poly), "pin": None}

        # THIRD PASS: Try rotation on binary variants
        best_for_rotation = ["binary", "inv_binary"]
        for name, img in candidates:
            if name in best_for_rotation:
                results = self._try_rotations(img)
                if results:
                    r = results[0]
                    txt = r.data.decode("utf-8", errors="replace").strip()
                    poly = [(p.x, p.y) for p in getattr(r, "polygon", [])] or []
                    return {"card": (txt, r.type, poly), "pin": None}

        # -----------------------
        # 2) OCR fallback for multi-line numeric ID
        # -----------------------
        h, w = gray.shape
        # Large crop because number layouts vary
        roi = gray[int(0.30 * h) : int(0.98 * h), int(0.03 * w) : int(0.97 * w)]

        # --- candidates for iterative OCR ---
        transforms = [
            lambda x: x,
            lambda x: cv2.bilateralFilter(x, 9, 40, 40),
            lambda x: cv2.GaussianBlur(x, (5, 5), 0),
            lambda x: cv2.adaptiveThreshold(
                x, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10
            ),
            lambda x: cv2.adaptiveThreshold(
                x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 8
            ),
            lambda x: cv2.threshold(x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            lambda x: cv2.medianBlur(x, 3),
            lambda x: cv2.resize(
                x, None, fx=2.2, fy=2.2, interpolation=cv2.INTER_CUBIC
            ),
        ]

        def run_ocr(img):
            txt = pytesseract.image_to_string(
                img, config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
            )
            lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
            nums = ["".join(ch for ch in ln if ch.isdigit()) for ln in lines]
            nums = [n for n in nums if MIN_OCR_DIGITS <= len(n) <= MAX_OCR_DIGITS]
            if nums:
                return max(nums, key=len)
            return None

        # Try sequential transforms until one produces a valid ID
        best = None
        img = roi.copy()
        for t in transforms:
            try:
                mod = t(img)
                if mod.ndim == 3:
                    mod = cv2.cvtColor(mod, cv2.COLOR_BGR2GRAY)
                n = run_ocr(mod)
                if n:
                    best = n
                    break
            except:
                continue

        if best:
            poly = [(0, 0), (w, 0), (w, h), (0, h)]
            return {"card": (best, "OCR", poly), "pin": None}

        return {"card": None, "pin": None}

    def _scan_1d_and_PIN(self, bgr):
        # ---------------------
        # 1) BARCODE BRANCH
        # ---------------------
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

        barcode_result = None
        if results:
            r = results[0]
            txt = r.data.decode("utf-8", errors="replace").strip()
            poly = [(p.x, p.y) for p in getattr(r, "polygon", [])] or []
            barcode_result = (txt, r.type, poly)

        # ---------------------
        # 2) OCR BRANCH
        # ---------------------
        # Crop lower half (numbers often below barcode, but layout varies)
        h, w = gray.shape
        roi = gray[int(0.45 * h) : int(0.98 * h), int(0.02 * w) : int(0.98 * w)]

        # Mild denoise + adaptive threshold
        roi_f = cv2.bilateralFilter(roi, 9, 40, 40)
        thr = cv2.adaptiveThreshold(
            roi_f, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 35, 10
        )

        raw = pytesseract.image_to_string(
            thr, config="--psm 6 --oem 3 -c tessedit_char_whitelist=0123456789"
        )

        lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]

        # ---------------------
        # Extract PIN (exact length)
        # ---------------------
        pin = None
        for ln in lines:
            tokens = ln.split()
            hits = [t for t in tokens if len(t) == PIN_DIGITS and t.isdigit()]
            if hits:
                pin = (hits[0], "PIN", [(0, 0), (w, 0), (w, h), (0, h)])
                break

        # ---------------------
        # Extract Card/ID number (multi-line)
        # ---------------------
        # Strategy: find all numeric tokens >= MIN_OCR_DIGITS
        # If spread across several lines, join contiguous lines.
        candidates = []

        for ln in lines:
            tokens = [t for t in ln.split() if t.isdigit()]
            longtok = [t for t in tokens if len(t) >= MIN_OCR_DIGITS]
            candidates.extend(longtok)

        # Additional rule: join adjacent lines if each line contains numeric blocks
        if not candidates:
            numlines = []
            for ln in lines:
                numeric = "".join(ch for ch in ln if ch.isdigit())
                if numeric:
                    numlines.append(numeric)
            if len(numlines) >= 2:
                joined = "".join(numlines)
                if MIN_OCR_DIGITS <= len(joined) <= MAX_OCR_DIGITS:
                    candidates.append(joined)

        card = None
        if candidates:
            best = max(candidates, key=len)
            card = (best, "OCR", [(0, 0), (w, 0), (w, h), (0, h)])

        # If barcode succeeded, prefer barcode result for card ID
        final_card = barcode_result if barcode_result else card

        return {"card": final_card, "pin": pin}

    # ==================== Selenium Methods ====================

    def _manual_open_browser(self):
        """Manually open browser with all shop tabs."""

        def worker():
            try:
                self._status_async("Opening browser with all shop tabs…", "blue")
                driver = self._ensure_driver()
                # if successful, mark start button
                try:
                    self.start_browser_btn.config(style="Selected.TButton")
                except Exception:
                    pass
            except Exception as e:
                self._status_async(f"Error opening browser: {e}", "red")

        threading.Thread(target=worker, daemon=True).start()

    def _status_async(self, text, color="blue"):
        def _apply():
            # keep status to a single concise line to avoid vertical layout shifts
            txt = text if text is not None else ""
            if len(txt) > 120:
                txt = txt[:117] + "..."
            self.status.config(text=txt, foreground=color)

        self.root.after(0, _apply)

    def _ensure_driver(self):
        """Create/reuse WebDriver and open all shop tabs."""
        if not SELENIUM_AVAILABLE:
            raise RuntimeError("Selenium not installed")

        if self._driver is not None:
            try:
                _ = self._driver.current_url
                return self._driver
            except Exception:
                self._driver = None

        # Try Chrome
        try:
            chrome_opts = webdriver.ChromeOptions()
            chrome_opts.add_argument("--start-maximized")
            self._driver = webdriver.Chrome(options=chrome_opts)
        except Exception:
            pass

        # Try Firefox
        if self._driver is None:
            try:
                firefox_opts = webdriver.FirefoxOptions()
                self._driver = webdriver.Firefox(options=firefox_opts)
            except Exception:
                pass

        # Try Safari
        if self._driver is None:
            try:
                self._driver = webdriver.Safari()
            except Exception as e:
                raise RuntimeError("Could not start any browser") from e

        # Open all shop tabs
        if self._driver is not None:
            self._open_all_shop_tabs()
            try:
                # mark start browser button as active
                self.start_browser_btn.config(style="Selected.TButton")
            except Exception:
                pass

        return self._driver

    def _open_all_shop_tabs(self):
        """Open all shop URLs in separate tabs."""
        if not self._driver:
            return

        self._shop_windows = {}
        try:
            first_shop = True
            for shop_name, config in SHOPS.items():
                if first_shop:
                    self._driver.get(config["url"])
                    self._shop_windows[shop_name] = self._driver.current_window_handle
                    first_shop = False
                else:
                    self._driver.execute_script("window.open('');")
                    self._driver.switch_to.window(self._driver.window_handles[-1])
                    self._driver.get(config["url"])
                    self._shop_windows[shop_name] = self._driver.current_window_handle

            self._driver.switch_to.window(self._driver.window_handles[0])
            self._status_async(f"✅ Opened {len(SHOPS)} shop tabs", "green")
        except Exception as e:
            self._status_async(f"Error opening tabs: {e}", "red")

    def _open_shop(
        self, site_name, url, card_selector, pin_selector=None, iframe_selector=None
    ):
        """Switch to shop tab and fill form."""
        code_text = self.code.get().strip()
        pin_text = self.pin.get().strip()

        if not code_text:
            self.status.config(text="❌ No code to send", foreground="red")
            return

        if pin_selector and not pin_text:
            pin_selector = None

        self._open_and_fill(
            url,
            card_selector,
            code_text,
            pin_selector,
            pin_text,
            site_name,
            iframe_selector,
        )

    def _open_and_fill(
        self,
        url,
        card_selector,
        code_text,
        pin_selector,
        pin_text,
        site_name,
        iframe_selector=None,
    ):
        """Fill form in existing tab."""

        def worker():
            driver = None
            switched_to_iframe = False

            try:
                driver = self._ensure_driver()

                # Switch to shop tab
                if site_name in self._shop_windows:
                    try:
                        driver.switch_to.window(self._shop_windows[site_name])
                        driver.refresh()
                    except Exception:
                        driver.execute_script("window.open('');")
                        driver.switch_to.window(driver.window_handles[-1])
                        driver.get(url)
                        self._shop_windows[site_name] = driver.current_window_handle
                else:
                    driver.execute_script("window.open('');")
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.get(url)
                    self._shop_windows[site_name] = driver.current_window_handle

                wait = WebDriverWait(driver, 30)

                # Switch to iframe if needed
                if iframe_selector:
                    iframe = wait.until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, iframe_selector)
                        )
                    )
                    driver.switch_to.frame(iframe)

                # Fill card number
                field_card = wait.until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, card_selector))
                )
                try:
                    field_card.clear()
                except Exception:
                    pass
                field_card.send_keys(code_text)

                # Fill PIN if available
                if pin_selector and pin_text:
                    field_pin = wait.until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, pin_selector))
                    )
                    try:
                        field_pin.clear()
                    except Exception:
                        pass
                    field_pin.send_keys(pin_text)

                self._status_async(f"✅ Filled {site_name} form", "green")

                if iframe_selector:
                    driver.switch_to.default_content()
            except Exception as e:
                self._status_async(f"❌ Error with {site_name}: {e}", "red")

            finally:
                # --- NEW: Robustly switch back ---
                if switched_to_iframe and driver:
                    try:
                        driver.switch_to.default_content()
                    except Exception as e:
                        print(f"Warning: could not switch back from iframe: {e}")

        threading.Thread(target=worker, daemon=True).start()

    # ==================== Drawing Methods ====================

    def _compute_roi_rect(self, w, h):
        roi_h = int(h * ROI_HEIGHT_FRAC)
        roi_w = int(w * ROI_WIDTH_FRAC)
        x0 = (w - roi_w) // 2
        y0 = (h - roi_h) // 2
        return x0, y0, x0 + roi_w, y0 + roi_h

    def _draw_scanner_overlay(self, img, roi, success=False):
        x0, y0, x1, y1 = roi

        if success:
            cv2.rectangle(img, (x0, y0), (x1, y1), SUCCESS_COLOR, 4)
            overlay = img.copy()
            cv2.rectangle(overlay, (x0, y0), (x1, y1), SUCCESS_COLOR, -1)
            alpha = 0.25
            img[y0:y1, x0:x1] = cv2.addWeighted(
                overlay[y0:y1, x0:x1], alpha, img[y0:y1, x0:x1], 1 - alpha, 0
            )
        else:
            self._draw_dashed_rect(
                img, (x0, y0), (x1, y1), OVERLAY_COLOR, 2, DRAW_DASH_GAP
            )
            c, L, th = OVERLAY_COLOR, CORNER_LEN, 3
            cv2.line(img, (x0, y0), (x0 + L, y0), c, th)
            cv2.line(img, (x0, y0), (x0, y0 + L), c, th)
            cv2.line(img, (x1, y0), (x1 - L, y0), c, th)
            cv2.line(img, (x1, y0), (x1, y0 + L), c, th)
            cv2.line(img, (x0, y1), (x0 + L, y1), c, th)
            cv2.line(img, (x0, y1), (x0, y1 - L), c, th)
            cv2.line(img, (x1, y1), (x1 - L, y1), c, th)
            cv2.line(img, (x1, y1), (x1, y1 - L), c, th)

        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (img.shape[1], img.shape[0]), (0, 0, 0), -1)

        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)  # "punch hole"

        alpha = 0.20
        img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    def _draw_dashed_rect(self, img, p0, p1, color, thickness, gap):
        (x0, y0), (x1, y1) = p0, p1
        for x in range(x0, x1, gap * 2):
            cv2.line(img, (x, y0), (min(x + gap, x1), y0), color, thickness)
        for x in range(x0, x1, gap * 2):
            cv2.line(img, (x, y1), (min(x + gap, x1), y1), color, thickness)
        for y in range(y0, y1, gap * 2):
            cv2.line(img, (x0, y), (x0, min(y + gap, y1)), color, thickness)
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
    app = VoucherScannerApp(root, camera_source=CAMERA_SOURCE)
    try:
        root.mainloop()
    finally:
        try:
            app.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
