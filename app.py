"""
RoadSense AI — Complete Flask Backend
All routes, AI detection, scoring, PDF, WhatsApp, Email, Chat Agent in one file.
"""

import os, json, base64, uuid, time, re, smtplib
from datetime import datetime
from io import BytesIO
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from flask import (Flask, render_template, request, jsonify,
                   redirect, url_for, session, send_file, Response)
from ultralytics import YOLO
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                 Table, TableStyle, Image as RLImage)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# ─── App Setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = "roadsense_secret_2024"

# Folders
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "static", "outputs")
REPORT_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# ─── Load YOLO Models ────────────────────────────────────────────────────────
MODEL1_PATH = r"D:\Pathhole_Detection\YOLOv8_Pothole_Segmentation_Road_Damage_Assessment\model\best.pt"
MODEL2_PATH = None

import torch
import torch.serialization

def _patch_pytorch_26():
    try:
        from ultralytics.nn.tasks import (
            DetectionModel, SegmentationModel, PoseModel, ClassificationModel
        )
        torch.serialization.add_safe_globals([
            DetectionModel, SegmentationModel, PoseModel, ClassificationModel
        ])
        print("[RoadSense] PyTorch 2.6 safe-globals patch applied ✓")
    except AttributeError:
        pass
    except ImportError:
        pass

_patch_pytorch_26()

print(f"[RoadSense] Loading Model 1 (Pothole) ...")
model1 = YOLO(MODEL1_PATH)
print(f"[RoadSense] Model 1 loaded ✓  classes: {list(model1.names.values())}")

model2 = None
if MODEL2_PATH and os.path.exists(MODEL2_PATH):
    print(f"[RoadSense] Loading Model 2 (Crack) ...")
    model2 = YOLO(MODEL2_PATH)
    print(f"[RoadSense] Model 2 loaded ✓  classes: {list(model2.names.values())}")
else:
    print("[RoadSense] Model 2 not configured — single-model mode")

model = model1

# ─── In-Memory Database ──────────────────────────────────────────────────────
defects_db      = []
road_records_db = []
sessions_db     = {}
officers_db     = {
    "admin":   {
        "password": "admin123",
        "name":     "Admin Officer",
        "phone":    "+91XXXXXXXXXX",   # ← change to real number (with country code)
        "email":    "officer@municipality.gov.in",  # ← change to real email
    },
    "officer1": {
        "password": "officer123",
        "name":     "Field Officer 1",
        "phone":    "+91XXXXXXXXXX",
        "email":    "officer1@municipality.gov.in",
    },
    "officer2": {
        "password": "officer456",
        "name":     "Field Officer 2",
        "phone":    "+91XXXXXXXXXX",
        "email":    "officer2@municipality.gov.in",
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT CONFIGURATION — FILL THESE IN
# ═══════════════════════════════════════════════════════════════════════════════
#
#  HOW TO GET FREE GMAIL SMTP:
#  1. Go to Google Account → Security → Enable 2-Step Verification
#  2. Go to Google Account → Security → App Passwords
#  3. Select "Mail" + "Windows Computer" → Generate
#  4. Copy the 16-character password shown (e.g. "abcd efgh ijkl mnop")
#  5. Paste it as ALERT_EMAIL_PASSWORD below (no spaces)
#
#  HOW TO GET FREE TWILIO WHATSAPP (Sandbox):
#  1. Go to twilio.com → Sign up (free)
#  2. Go to Messaging → Try it out → WhatsApp Sandbox
#  3. Send "join <your-sandbox-word>" to +1 415 523 8886 from officer's WhatsApp
#  4. Copy Account SID and Auth Token from twilio.com/console
#
# ═══════════════════════════════════════════════════════════════════════════════

ALERT_CONFIG = {
    # ── EMAIL (Gmail SMTP — Free) ──────────────────────────────────────────
    "email_enabled":   True,
    "smtp_server":     "smtp.gmail.com",
    "smtp_port":       587,
    "sender_email":    os.environ.get("ALERT_EMAIL",    "your_gmail@gmail.com"),
    "sender_password": os.environ.get("ALERT_PASSWORD", "your_16char_app_password"),
    "sender_name":     "RoadSense AI Alert System",

    # ── WHATSAPP (Twilio — Free Sandbox) ──────────────────────────────────
    "whatsapp_enabled": True,
    "twilio_sid":      os.environ.get("TWILIO_SID",   ""),
    "twilio_token":    os.environ.get("TWILIO_TOKEN", ""),
    "twilio_from":     os.environ.get("TWILIO_FROM",  "whatsapp:+14155238886"),

    # ── THRESHOLD ─────────────────────────────────────────────────────────
    "alert_threshold": 7.0,   # Send alert when score >= this value
}

# ─── Damage Constants ────────────────────────────────────────────────────────
DAMAGE_CONFIG = {
    "pothole":            {"base": 5.0, "label": "Pothole",            "repair_cost": 5500},
    "D40":                {"base": 5.0, "label": "Pothole",            "repair_cost": 5500},
    "alligator_crack":    {"base": 4.5, "label": "Alligator Crack",    "repair_cost": 16000},
    "D20":                {"base": 4.5, "label": "Alligator Crack",    "repair_cost": 16000},
    "transverse_crack":   {"base": 2.5, "label": "Transverse Crack",   "repair_cost": 900},
    "D10":                {"base": 2.5, "label": "Transverse Crack",   "repair_cost": 900},
    "longitudinal_crack": {"base": 1.5, "label": "Longitudinal Crack", "repair_cost": 500},
    "D00":                {"base": 1.5, "label": "Longitudinal Crack", "repair_cost": 500},
}

SEVERITY_LEVELS = [
    (0, 2,  "Good",      "success", "#28a745", 90),
    (2, 4,  "Moderate",  "warning", "#ffc107", 70),
    (4, 6,  "Severe",    "orange",  "#fd7e14", 50),
    (6, 8,  "Critical",  "danger",  "#dc3545", 30),
    (8, 10, "Emergency", "dark",    "#7b0000", 10),
]

# ─── Helper Functions ─────────────────────────────────────────────────────────

def get_level(score):
    for lo, hi, label, bs_class, color, _ in SEVERITY_LEVELS:
        if lo <= score <= hi:
            return label, bs_class, color
    return "Emergency", "dark", "#7b0000"

def get_repair_days(score):
    if score >= 8: return "Immediately — Today"
    if score >= 6: return "Within 7 days"
    if score >= 4: return "Within 30 days"
    if score >= 2: return "Within 90 days"
    return "Monitor monthly"

def calc_economic_impact(score):
    if score <= 1:
        return 0
    severity_factor = (score / 10) ** 1.5
    daily_vehicles  = 500
    cost_per_vehicle_per_pothole = 15
    days = 30
    impact = round(daily_vehicles * cost_per_vehicle_per_pothole * days * severity_factor)
    return max(0, impact)

def calc_score(detections, img_w, img_h):
    if not detections:
        return 0.0, []
    img_area = img_w * img_h
    scored = []
    for det in detections:
        cls_name = det["class"].lower().replace(" ", "_")
        cfg = DAMAGE_CONFIG.get(cls_name, {"base": 3.0, "label": cls_name, "repair_cost": 2000})
        base = cfg["base"]
        bx1, by1, bx2, by2 = det["bbox"]
        box_area   = max(1, (bx2 - bx1) * (by2 - by1))
        size_ratio = box_area / img_area if img_area > 0 else 0.01
        if   size_ratio > 0.10: size_mult = 2.0
        elif size_ratio > 0.06: size_mult = 1.7
        elif size_ratio > 0.03: size_mult = 1.4
        elif size_ratio > 0.01: size_mult = 1.0
        elif size_ratio > 0.005: size_mult = 0.75
        else:                   size_mult = 0.50
        conf = det.get("confidence", 0.8)
        if   conf >= 0.90: conf_w = 1.00
        elif conf >= 0.75: conf_w = 0.88
        elif conf >= 0.55: conf_w = 0.72
        elif conf >= 0.40: conf_w = 0.55
        else:              conf_w = 0.40
        single_score = min(10.0, base * size_mult * conf_w)
        scored.append({**det,
                        "single_score": round(single_score, 2),
                        "label":       cfg["label"],
                        "repair_cost": cfg["repair_cost"]})
    n         = len(scored)
    max_score = max(d["single_score"] for d in scored)
    avg_score = sum(d["single_score"] for d in scored) / n
    if n == 1:
        final = max_score
    elif n <= 3:
        final = max_score * 0.70 + avg_score * 0.30
        final = min(10.0, final * (1 + (n-1) * 0.08))
    else:
        final = max_score * 0.60 + avg_score * 0.40
        final = min(10.0, final * (1 + min(0.3, (n-3) * 0.05)))
    return round(final, 2), scored

def annotate_image(image_bytes, detections):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return image_bytes
    COLOR_MAP = {
        "Pothole":            (0,   0,   200),
        "Alligator Crack":    (0,   165, 255),
        "Transverse Crack":   (0,   200, 200),
        "Longitudinal Crack": (0,   200, 0),
    }
    for det in detections:
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
        label = det.get("label", det["class"])
        conf  = det.get("confidence", 0)
        color = COLOR_MAP.get(label, (100, 100, 255))
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

def _extract_boxes(yolo_model, img_cv2):
    results = yolo_model(img_cv2, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cls_id   = int(box.cls[0])
        cls_name = yolo_model.names[cls_id]
        conf     = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        dets.append({"class": cls_name, "confidence": round(conf, 3), "bbox": [x1,y1,x2,y2]})
    return dets

def _bbox_iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    if inter == 0:
        return 0.0
    return inter / ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)

def run_yolo(image_bytes):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        return [], 640, 480
    h, w = img.shape[:2]
    detections = _extract_boxes(model1, img)
    if model2 is not None:
        for cd in _extract_boxes(model2, img):
            if not any(_bbox_iou(d["bbox"], cd["bbox"]) > 0.4 for d in detections):
                detections.append(cd)
    return detections, w, h

def _save_image(annotated_bytes):
    if not annotated_bytes:
        return None
    fname = f"{uuid.uuid4().hex}.jpg"
    fpath = os.path.join(OUTPUT_DIR, fname)
    with open(fpath, "wb") as fh:
        fh.write(annotated_bytes)
    return f"/static/outputs/{fname}"

def save_frame_to_session(session_id, lat, lng, detections, score, annotated_bytes):
    sess = sessions_db.get(session_id)
    if not sess:
        return
    img_url = _save_image(annotated_bytes)
    sess["frames"].append({
        "lat": lat, "lng": lng,
        "detections": detections,
        "score": score,
        "img_url": img_url,
        "timestamp": datetime.now().isoformat(),
    })
    if score > sess["worst_score"]:
        sess["worst_score"] = score
        sess["worst_img"]   = img_url
        sess["worst_lat"]   = lat
        sess["worst_lng"]   = lng

def finalise_session(session_id):
    sess = sessions_db.get(session_id)
    if not sess:
        return None
    frames    = sess.get("frames", [])
    road_name = sess["road_name"]
    if not frames:
        level, bs_class, color = get_level(0)
        record = {
            "id": str(uuid.uuid4()), "road_name": road_name,
            "lat": sess.get("start_lat", 0), "lng": sess.get("start_lng", 0),
            "inspection_date": sess["start_time"][:10],
            "inspection_time": sess["start_time"][11:19],
            "timestamp": sess["start_time"],
            "frames_captured": sess["frames_captured"],
            "defect_frames": 0,
            "all_detections": [],
            "score": 0, "level": "Good", "bs_class": "success", "color": "#28a745",
            "annotated_img": None, "worst_img": None,
            "economic_impact": 0, "repair_action": "Monitor monthly",
            "source": "webcam", "status": "pending",
            "officer": sess.get("officer", ""),
            "route_points": [],
        }
    else:
        scores      = [f["score"] for f in frames]
        avg_score   = round(sum(scores) / len(scores), 2)
        worst       = sess["worst_score"]
        final_score = round(min(10.0, worst * 0.6 + avg_score * 0.4), 2)
        all_types   = {}
        for fr in frames:
            for d in fr["detections"]:
                lbl = d.get("label", d.get("class", "Unknown"))
                all_types[lbl] = all_types.get(lbl, 0) + 1
        level, bs_class, color = get_level(final_score)
        route_points = [{"lat": f["lat"], "lng": f["lng"], "score": f["score"]} for f in frames]
        record = {
            "id": str(uuid.uuid4()), "road_name": road_name,
            "lat": sess["worst_lat"], "lng": sess["worst_lng"],
            "inspection_date": sess["start_time"][:10],
            "inspection_time": sess["start_time"][11:19],
            "timestamp": sess["start_time"],
            "frames_captured": sess["frames_captured"],
            "defect_frames": len(frames),
            "all_detections": [{"label": k, "count": v} for k, v in all_types.items()],
            "score": final_score, "level": level, "bs_class": bs_class, "color": color,
            "annotated_img": sess["worst_img"],
            "worst_img": sess["worst_img"],
            "economic_impact": calc_economic_impact(final_score),
            "repair_action":   get_repair_days(final_score),
            "source": "webcam", "status": "pending",
            "officer": sess.get("officer", ""),
            "route_points": route_points,
        }
    road_records_db.append(record)
    defects_db.append(record)
    del sessions_db[session_id]
    return record

def save_record(road_name, lat, lng, detections, score, annotated_bytes, source="citizen"):
    img_url = _save_image(annotated_bytes)
    level, bs_class, color = get_level(score)
    all_types = {}
    for d in detections:
        lbl = d.get("label", d.get("class","Unknown"))
        all_types[lbl] = all_types.get(lbl, 0) + 1
    record = {
        "id": str(uuid.uuid4()), "road_name": road_name,
        "lat": lat, "lng": lng,
        "inspection_date": datetime.now().strftime("%Y-%m-%d"),
        "inspection_time": datetime.now().strftime("%H:%M:%S"),
        "timestamp": datetime.now().isoformat(),
        "frames_captured": 1, "defect_frames": 1 if detections else 0,
        "all_detections": [{"label": k, "count": v} for k, v in all_types.items()],
        "detections": detections,
        "score": score, "level": level, "bs_class": bs_class, "color": color,
        "annotated_img": img_url, "worst_img": img_url,
        "economic_impact": calc_economic_impact(score),
        "repair_action":   get_repair_days(score),
        "source": source, "status": "pending", "officer": "",
        "route_points": [{"lat": lat, "lng": lng, "score": score}],
    }
    road_records_db.append(record)
    defects_db.append(record)
    return record


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT CONFIGURATION — YOUR REAL CREDENTIALS
# ═══════════════════════════════════════════════════════════════════════════════

ALERT_CONFIG = {
    # ── EMAIL (Gmail SMTP — Free) ──────────────────────────────────────────────
    # Sender: your Gmail account (onkarkorale7@gmail.com)
    # HOW TO GET APP PASSWORD:
    #   1. Go to myaccount.google.com → Security → Enable 2-Step Verification
    #   2. Go to myaccount.google.com → Security → App Passwords
    #   3. Select "Mail" → Generate → copy the 16-char password (no spaces)
    #   4. Set env var:  set ALERT_PASSWORD=abcdefghijklmnop
    "email_enabled":   True,
    "smtp_server":     "smtp.gmail.com",
    "smtp_port":       587,
    "sender_email":    os.environ.get("ALERT_EMAIL",    "onkarkorale7@gmail.com"),
    "sender_password": os.environ.get("ALERT_PASSWORD", "YOUR_16_CHAR_APP_PASSWORD"),
    "sender_name":     "RoadSense AI Alert System",

    # ── WHATSAPP (Twilio — Free Sandbox) ──────────────────────────────────────
    # Sender number: +91 99228 18898 (your number — must join Twilio sandbox)
    # HOW TO SET UP:
    #   1. Go to twilio.com → Sign up free
    #   2. Go to Messaging → Try it out → WhatsApp Sandbox
    #   3. From +91 99228 18898, send "join <sandbox-word>" to +1 415 523 8886
    #   4. From +91 79721 86197 (officer), ALSO send the same join message
    #   5. Copy Account SID and Auth Token from twilio.com/console
    #   6. Set env vars:
    #        set TWILIO_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    #        set TWILIO_TOKEN=your_auth_token
    "whatsapp_enabled": True,
    "twilio_sid":      os.environ.get("TWILIO_SID",   ""),
    "twilio_token":    os.environ.get("TWILIO_TOKEN", ""),
    # This is YOUR number — the sender (Twilio sandbox number)
    "twilio_from":     "whatsapp:+14155238886",  # Twilio sandbox FROM number (always this)

    # ── THRESHOLD ─────────────────────────────────────────────────────────────
    "alert_threshold": 7.0,   # Send alert when score >= 7.0
}

# ─── Officers Database — with real contact info ────────────────────────────
officers_db = {
    "admin": {
        "password": "admin123",
        "name":     "Admin Officer",
        "email":    "onkarkorale7@gmail.com",      # Your email (system admin)
        "phone":    "+919922818898",               # Your number
    },
    "officer1": {
        "password": "officer123",
        "name":     "Arnav P (Field Officer)",
        "email":    "arnavp651@gmail.com",         # Officer's email
        "phone":    "+917972186197",               # Officer's WhatsApp number
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
#  ALERT SYSTEM — EMAIL + WHATSAPP (complete functions)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_email_html(record):
    """Beautiful HTML email — works in Gmail, Outlook, phone apps."""
    level  = record.get("level", "Critical")
    score  = record.get("score", 0)
    road   = record.get("road_name", "Unknown")
    action = record.get("repair_action", "Immediate action required")
    eco    = record.get("economic_impact", 0)
    lat    = record.get("lat", 0)
    lng    = record.get("lng", 0)
    ts     = record.get("timestamp", "")[:19].replace("T", " ")
    source = record.get("source", "").capitalize()

    # Detections rows
    det_rows = ""
    for d in record.get("all_detections", record.get("detections", [])):
        label = d.get("label", d.get("class", "Unknown"))
        count = d.get("count", 1)
        det_rows += (
            f"<tr>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #f0f0f0'>{label}</td>"
            f"<td style='padding:8px 12px;border-bottom:1px solid #f0f0f0;text-align:center'>{count}</td>"
            f"</tr>"
        )
    if not det_rows:
        det_rows = "<tr><td colspan='2' style='padding:8px 12px;color:#999'>No detail available</td></tr>"

    color_map = {
        "Good":      "#28a745",
        "Moderate":  "#ffc107",
        "Severe":    "#fd7e14",
        "Critical":  "#dc3545",
        "Emergency": "#7b0000",
    }
    alert_color = color_map.get(level, "#dc3545")
    maps_url    = f"https://www.google.com/maps?q={lat},{lng}"

    return f"""<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"/></head>
<body style="margin:0;padding:0;font-family:Arial,sans-serif;background:#f5f5f5">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#f5f5f5;padding:30px 0">
    <tr><td align="center">
      <table width="600" cellpadding="0" cellspacing="0"
             style="background:#ffffff;border-radius:12px;overflow:hidden;
                    box-shadow:0 4px 20px rgba(0,0,0,0.1)">

        <!-- HEADER -->
        <tr>
          <td style="background:{alert_color};padding:28px 32px;text-align:center">
            <div style="font-size:36px;margin-bottom:8px">🚨</div>
            <h1 style="color:#ffffff;margin:0;font-size:24px;font-weight:800;letter-spacing:1px">
              ROAD DAMAGE ALERT
            </h1>
            <p style="color:rgba(255,255,255,0.85);margin:8px 0 0;font-size:14px">
              RoadSense AI — Automated Detection System
            </p>
          </td>
        </tr>

        <!-- SCORE BANNER -->
        <tr>
          <td style="background:#1a1a2e;padding:20px 32px;text-align:center">
            <span style="font-size:52px;font-weight:900;color:{alert_color};line-height:1">{score}</span>
            <span style="font-size:24px;color:#aaaaaa">/10</span>
            <span style="display:inline-block;margin-left:16px;background:{alert_color};color:#fff;
                         padding:6px 18px;border-radius:100px;font-size:15px;
                         font-weight:700;vertical-align:middle">
              {level}
            </span>
          </td>
        </tr>

        <!-- ROAD INFO -->
        <tr>
          <td style="padding:28px 32px">
            <table width="100%" cellpadding="0" cellspacing="0">
              <tr>
                <td width="50%" style="vertical-align:top;padding-right:16px">
                  <p style="margin:0 0 4px;font-size:11px;color:#999;
                             text-transform:uppercase;letter-spacing:1px">Road Name</p>
                  <p style="margin:0 0 20px;font-size:17px;font-weight:700;color:#1a1a2e">{road}</p>

                  <p style="margin:0 0 4px;font-size:11px;color:#999;
                             text-transform:uppercase;letter-spacing:1px">Action Required</p>
                  <p style="margin:0 0 20px;font-size:15px;font-weight:700;color:{alert_color}">{action}</p>

                  <p style="margin:0 0 4px;font-size:11px;color:#999;
                             text-transform:uppercase;letter-spacing:1px">Detected At</p>
                  <p style="margin:0 0 20px;font-size:14px;color:#444">{ts}</p>
                </td>
                <td width="50%" style="vertical-align:top;padding-left:16px;
                                       border-left:1px solid #f0f0f0">
                  <p style="margin:0 0 4px;font-size:11px;color:#999;
                             text-transform:uppercase;letter-spacing:1px">Economic Risk (30 days)</p>
                  <p style="margin:0 0 20px;font-size:17px;font-weight:700;color:#dc3545">
                    ₹{eco:,}
                  </p>

                  <p style="margin:0 0 4px;font-size:11px;color:#999;
                             text-transform:uppercase;letter-spacing:1px">GPS Location</p>
                  <p style="margin:0 0 20px;font-size:13px;color:#444">{lat:.4f}, {lng:.4f}</p>

                  <p style="margin:0 0 4px;font-size:11px;color:#999;
                             text-transform:uppercase;letter-spacing:1px">Source</p>
                  <p style="margin:0;font-size:14px;color:#444">{source} Inspection</p>
                </td>
              </tr>
            </table>
          </td>
        </tr>

        <!-- DETECTIONS TABLE -->
        <tr>
          <td style="padding:0 32px 24px">
            <p style="font-size:13px;font-weight:700;color:#1a1a2e;text-transform:uppercase;
                      letter-spacing:1px;margin:0 0 12px">Defects Detected</p>
            <table width="100%"
                   style="border:1px solid #f0f0f0;border-radius:8px;
                          overflow:hidden;border-collapse:collapse">
              <tr style="background:#f8f8f8">
                <th style="padding:10px 12px;text-align:left;font-size:12px;
                           color:#666;text-transform:uppercase">Type</th>
                <th style="padding:10px 12px;text-align:center;font-size:12px;
                           color:#666;text-transform:uppercase">Count</th>
              </tr>
              {det_rows}
            </table>
          </td>
        </tr>

        <!-- BUTTONS -->
        <tr>
          <td style="padding:0 32px 32px;text-align:center">
            <a href="{maps_url}"
               style="display:inline-block;background:#1a1a2e;color:#ffffff;
                      padding:14px 32px;border-radius:8px;text-decoration:none;
                      font-weight:700;font-size:15px;margin-right:12px">
              📍 View on Google Maps
            </a>
            <a href="http://localhost:5000/city-map"
               style="display:inline-block;background:{alert_color};color:#ffffff;
                      padding:14px 32px;border-radius:8px;text-decoration:none;
                      font-weight:700;font-size:15px">
              🗺️ Open Dashboard
            </a>
          </td>
        </tr>

        <!-- FOOTER -->
        <tr>
          <td style="background:#f8f8f8;padding:16px 32px;text-align:center;
                     border-top:1px solid #eee">
            <p style="margin:0;font-size:11px;color:#999">
              Automated alert from <strong>RoadSense AI</strong> — Road Inspection System<br/>
              Report ID: {record.get('id','')[:8].upper()} &nbsp;|&nbsp;
              Sender: onkarkorale7@gmail.com
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""


def send_email_alert(record, to_email=None, officer_name="Officer"):
    """
    Send HTML email alert.
    Sender  : onkarkorale7@gmail.com  (needs Gmail App Password)
    Default recipient: arnavp651@gmail.com  (officer)
    """
    if not ALERT_CONFIG["email_enabled"]:
        print("[Email] Alerts disabled.")
        return False

    sender   = ALERT_CONFIG["sender_email"]     # onkarkorale7@gmail.com
    password = ALERT_CONFIG["sender_password"]

    if "YOUR_16_CHAR" in password:
        print("[Email] ⚠️  App Password not set. Run:  set ALERT_PASSWORD=your16charpassword")
        return False

    # Default recipient is the officer (arnavp651@gmail.com)
    recipient = to_email or "arnavp651@gmail.com"

    level = record.get("level", "Critical")
    road  = record.get("road_name", "Unknown Road")
    score = record.get("score", 0)
    eco   = record.get("economic_impact", 0)
    lat   = record.get("lat", 0)
    lng   = record.get("lng", 0)

    msg            = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 RoadSense ALERT — {level} Damage on {road} ({score}/10)"
    msg["From"]    = f"RoadSense AI <{sender}>"
    msg["To"]      = recipient

    plain_text = f"""
ROADSENSE AI — ROAD DAMAGE ALERT
==================================
Road      : {road}
Score     : {score}/10 — {level}
Action    : {record.get('repair_action', 'Immediate action required')}
Economic  : Rs.{eco:,} risk in 30 days
GPS       : {lat:.4f}, {lng:.4f}
Detected  : {record.get('timestamp','')[:19].replace('T', ' ')}
Source    : {record.get('source','').capitalize()} Inspection
Report ID : {record.get('id','')[:8].upper()}

Maps Link : https://www.google.com/maps?q={lat},{lng}
Dashboard : http://localhost:5000/city-map
==================================
Sent by RoadSense AI from onkarkorale7@gmail.com
"""
    html_body = _build_email_html(record)

    msg.attach(MIMEText(plain_text, "plain"))
    msg.attach(MIMEText(html_body,  "html"))

    # Attach annotated image if it exists on disk
    if record.get("annotated_img"):
        img_path = os.path.join(BASE_DIR, record["annotated_img"].lstrip("/"))
        if os.path.exists(img_path):
            with open(img_path, "rb") as fh:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(fh.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f"attachment; filename=road_detection_{record.get('id','')[:8]}.jpg"
                )
                msg.attach(part)

    try:
        server = smtplib.SMTP(ALERT_CONFIG["smtp_server"], ALERT_CONFIG["smtp_port"])
        server.ehlo()
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, msg.as_string())
        server.quit()
        print(f"[Email] ✅ Sent to {recipient}  |  {road} — {score}/10 {level}")
        return True
    except smtplib.SMTPAuthenticationError:
        print("[Email] ❌ Wrong Gmail App Password. Regenerate at myaccount.google.com → Security → App Passwords")
        return False
    except smtplib.SMTPException as e:
        print(f"[Email] ❌ SMTP error: {e}")
        return False
    except Exception as e:
        print(f"[Email] ❌ Unexpected: {e}")
        return False


def send_whatsapp_alert(record, to_number=None):
    """
    Send WhatsApp alert via Twilio.
    FROM (sandbox): +1 415 523 8886  (Twilio sandbox — always this number)
    TO (officer)  : +91 79721 86197  (arnavp651 / Arnav)

    IMPORTANT — before this works:
      1. Officer (+91 79721 86197) must WhatsApp "join <word>" to +1 415 523 8886
      2. Your number (+91 99228 18898) must also join if you want to receive too
      3. Set TWILIO_SID and TWILIO_TOKEN env vars
    """
    if not ALERT_CONFIG["whatsapp_enabled"]:
        print("[WhatsApp] Alerts disabled.")
        return False

    sid   = ALERT_CONFIG["twilio_sid"]
    token = ALERT_CONFIG["twilio_token"]

    if not sid or not token:
        print("[WhatsApp] ⚠️  Twilio SID/Token not set.")
        print("           Run:  set TWILIO_SID=ACxxxxx  and  set TWILIO_TOKEN=xxxxx")
        return False

    # Default: send to officer's number
    recipient = to_number or "whatsapp:+917972186197"

    level  = record.get("level", "Critical")
    road   = record.get("road_name", "Unknown")
    score  = record.get("score", 0)
    action = record.get("repair_action", "Immediate action required")
    eco    = record.get("economic_impact", 0)
    lat    = record.get("lat", 0)
    lng    = record.get("lng", 0)
    ts     = record.get("timestamp", "")[:19].replace("T", " ")

    emoji_map = {
        "Good": "✅", "Moderate": "🟡", "Severe": "🟠",
        "Critical": "🔴", "Emergency": "🚨"
    }
    emoji = emoji_map.get(level, "🚨")

    det_lines = ""
    for d in record.get("all_detections", record.get("detections", [])):
        label = d.get("label", d.get("class", "Unknown"))
        count = d.get("count", 1)
        det_lines += f"  • {label} × {count}\n"
    if not det_lines:
        det_lines = "  • Road damage detected\n"

    message = (
        f"{emoji} *RoadSense AI — ROAD ALERT*\n\n"
        f"*Road:* {road}\n"
        f"*Score:* {score}/10 — *{level}*\n"
        f"*Action:* {action}\n"
        f"*Economic Risk:* Rs.{eco:,}/30 days\n\n"
        f"*Defects Found:*\n{det_lines}\n"
        f"*GPS:* {lat:.4f}, {lng:.4f}\n"
        f"*Time:* {ts}\n\n"
        f"📍 Maps: https://maps.google.com?q={lat},{lng}\n"
        f"🗺️ Dashboard: http://localhost:5000/city-map\n\n"
        f"_RoadSense AI | Report: {record.get('id','')[:8].upper()}_"
    )

    try:
        from twilio.rest import Client
        client      = Client(sid, token)
        message_obj = client.messages.create(
            body=message,
            from_=ALERT_CONFIG["twilio_from"],   # whatsapp:+14155238886
            to=recipient                          # whatsapp:+917972186197
        )
        print(f"[WhatsApp] ✅ Sent to {recipient}  |  SID: {message_obj.sid}")
        return True
    except ImportError:
        print("[WhatsApp] ❌ Twilio not installed. Run:  pip install twilio")
        return False
    except Exception as e:
        print(f"[WhatsApp] ❌ Failed: {e}")
        return False


def send_all_alerts(record, officer_username=None):
    """
    Master alert function.
    Fires when score >= 7.0

    WHO GETS ALERTED:
    ─────────────────────────────────────────────────────
    Email  → arnavp651@gmail.com  (officer, Arnav)
             onkarkorale7@gmail.com  (you, admin — CC'd via broadcast)
    WA     → +91 79721 86197  (officer, Arnav)
    ─────────────────────────────────────────────────────
    """
    score = record.get("score", 0)
    if score < ALERT_CONFIG["alert_threshold"]:
        return   # Below threshold — no alert needed

    road  = record.get("road_name", "Unknown")
    level = record.get("level", "Critical")
    print(f"\n{'='*55}")
    print(f"[Alert] 🚨  Score {score}/10 — {level}")
    print(f"[Alert] Road: {road}")
    print(f"[Alert] Sending Email + WhatsApp...")
    print(f"{'='*55}")

    # ── Determine email recipient ──────────────────────────────────────────
    if officer_username and officer_username in officers_db:
        # Logged-in officer gets the alert
        odata         = officers_db[officer_username]
        officer_email = odata.get("email", "arnavp651@gmail.com")
        officer_phone = odata.get("phone", "+917972186197")
        officer_name  = odata.get("name", "Officer")

        send_email_alert(record,
                         to_email=officer_email,
                         officer_name=officer_name)

        # Also CC admin if the logged-in officer is not admin
        if officer_username != "admin":
            admin_email = officers_db.get("admin", {}).get("email", "")
            if admin_email and admin_email != officer_email:
                send_email_alert(record,
                                 to_email=admin_email,
                                 officer_name="Admin Officer")

        # WhatsApp to officer
        if officer_phone:
            wa_number = f"whatsapp:{officer_phone}" if not officer_phone.startswith("whatsapp:") else officer_phone
            send_whatsapp_alert(record, to_number=wa_number)

    else:
        # No officer logged in (citizen upload) → broadcast to ALL officers
        print("[Alert] No officer session — broadcasting to all officers")
        for uname, odata in officers_db.items():
            email = odata.get("email", "")
            phone = odata.get("phone", "")
            name  = odata.get("name", "Officer")
            if email:
                send_email_alert(record, to_email=email, officer_name=name)
            if phone:
                wa_num = f"whatsapp:{phone}" if not phone.startswith("whatsapp:") else phone
                send_whatsapp_alert(record, to_number=wa_num)

    print(f"[Alert] ✅ Done.\n")

# ─── City Health Score ────────────────────────────────────────────────────────
def get_city_health_score():
    if not road_records_db:
        return 100
    roads = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["score"] > roads[rn]["score"]:
            roads[rn] = r
    road_list = list(roads.values())
    n = len(road_list)
    if n == 0:
        return 100
    max_deduction = 70
    road_weight   = max_deduction / (n + 5)
    total_deduction = 0
    for r in road_list:
        severity = r["score"] / 10.0
        total_deduction += road_weight * severity
    health = max(30, min(100, 100 - total_deduction))
    return round(health, 1)

def get_stats():
    total  = len(road_records_db)
    crit   = sum(1 for r in road_records_db if r["score"] >= 6)
    emerg  = sum(1 for r in road_records_db if r["score"] >= 8)
    eco    = sum(r["economic_impact"] for r in road_records_db)
    return {"total": total, "critical": crit, "emergency": emerg,
            "economic_risk": eco, "city_health": get_city_health_score()}

# ─── Auth ─────────────────────────────────────────────────────────────────────
def login_required(f):
    from functools import wraps
    @wraps(f)
    def wrapper(*args, **kwargs):
        if "officer" not in session:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper

# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES — Pages
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    if "officer" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        uname = request.form.get("username", "").strip()
        pwd   = request.form.get("password", "").strip()
        if uname in officers_db and officers_db[uname]["password"] == pwd:
            session["officer"] = uname
            session["name"]    = officers_db[uname]["name"]
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", officer=session.get("name"))

@app.route("/city-map")
@login_required
def city_map():
    return render_template("city_map.html", officer=session.get("name"))

@app.route("/repair-queue")
@login_required
def repair_queue():
    return render_template("repair_queue.html", officer=session.get("name"))

@app.route("/history")
@login_required
def history():
    return render_template("history.html", officer=session.get("name"))

@app.route("/live-inspection")
@login_required
def live_inspection():
    return render_template("live_inspection.html", officer=session.get("name"))

@app.route("/video-upload")
@login_required
def video_upload():
    return render_template("video_upload.html", officer=session.get("name"))

@app.route("/report")
def citizen_report():
    return render_template("citizen_report.html")

@app.route("/predictive")
@login_required
def predictive_page():
    return render_template("predictive.html", officer=session.get("name"))

@app.route("/monsoon")
@login_required
def monsoon_page():
    return render_template("monsoon.html", officer=session.get("name"))

# ═══════════════════════════════════════════════════════════════════════════════
#  API ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/stats")
@login_required
def api_stats():
    s = get_stats()
    dist = {"Good": 0, "Moderate": 0, "Severe": 0, "Critical": 0, "Emergency": 0}
    for r in road_records_db:
        dist[r["level"]] = dist.get(r["level"], 0) + 1
    dtype_dist = {}
    for r in road_records_db:
        for d in r.get("all_detections", r.get("detections", [])):
            lbl = d.get("label", d.get("class", "Unknown"))
            cnt = d.get("count", 1)
            dtype_dist[lbl] = dtype_dist.get(lbl, 0) + cnt
    recent = sorted(road_records_db, key=lambda x: x["timestamp"], reverse=True)[:5]
    return jsonify({**s, "severity_dist": dist, "damage_types": dtype_dist, "recent": recent})

@app.route("/api/city-health-score")
def api_city_health():
    return jsonify({"score": get_city_health_score(), "total": len(defects_db)})

@app.route("/api/heatmap-data")
def api_heatmap():
    data = []
    for r in road_records_db:
        if r.get("lat") and r.get("lng"):
            data.append({
                "id": r["id"], "road_name": r["road_name"],
                "lat": r["lat"], "lng": r["lng"],
                "score": r["score"], "level": r["level"],
                "color": r["color"], "timestamp": r["timestamp"],
                "inspection_date": r.get("inspection_date",""),
                "inspection_time": r.get("inspection_time",""),
                "annotated_img": r.get("annotated_img") or r.get("worst_img"),
                "economic_impact": r["economic_impact"],
                "repair_action": r["repair_action"],
                "defect_frames": r.get("defect_frames", 0),
                "frames_captured": r.get("frames_captured", 1),
                "all_detections": r.get("all_detections", []),
                "officer": r.get("officer",""),
                "source": r.get("source", ""),
                "route_points": r.get("route_points", []),
            })
    return jsonify(data)

@app.route("/api/priority-queue")
@login_required
def api_priority_queue():
    roads = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["score"] > roads[rn]["score"]:
            roads[rn] = r
    sorted_roads = sorted(roads.values(), key=lambda x: x["score"], reverse=True)
    return jsonify(sorted_roads)

@app.route("/api/defects")
@login_required
def api_defects():
    q     = request.args.get("q", "").lower()
    level = request.args.get("level", "")
    data  = road_records_db
    if q:
        data = [r for r in data if q in r["road_name"].lower()]
    if level:
        data = [r for r in data if r["level"].lower() == level.lower()]
    return jsonify(sorted(data, key=lambda x: x["timestamp"], reverse=True))

@app.route("/api/defects/<record_id>/status", methods=["POST"])
@login_required
def api_update_status(record_id):
    status = request.json.get("status", "pending")
    for r in defects_db:
        if r["id"] == record_id:
            r["status"] = status
            return jsonify({"ok": True})
    return jsonify({"ok": False, "error": "Not found"}), 404

@app.route("/api/start-session", methods=["POST"])
@login_required
def api_start_session():
    data      = request.json
    road_name = data.get("road_name", "Unknown Road").strip()
    lat       = data.get("lat", 0.0)
    lng       = data.get("lng", 0.0)
    sess_id   = uuid.uuid4().hex
    sessions_db[sess_id] = {
        "road_name":       road_name,
        "start_time":      datetime.now().isoformat(),
        "start_lat":       lat,
        "start_lng":       lng,
        "officer":         session.get("name", ""),
        "officer_username": session.get("officer", ""),
        "frames_captured": 0,
        "frames":          [],
        "worst_score":     0,
        "worst_img":       None,
        "worst_lat":       lat,
        "worst_lng":       lng,
    }
    return jsonify({"session_id": sess_id, "road_name": road_name})

@app.route("/api/stop-session", methods=["POST"])
@login_required
def api_stop_session():
    sess_id = request.json.get("session_id", "")
    if sess_id not in sessions_db:
        return jsonify({"error": "Session not found"}), 404

    # Grab officer username before finalise deletes session
    officer_username = sessions_db[sess_id].get("officer_username", "")

    record = finalise_session(sess_id)
    if not record:
        return jsonify({"error": "Could not finalise session"}), 500

    # Send alerts if threshold met
    send_all_alerts(record, officer_username=officer_username)

    return jsonify({
        "road_name":       record["road_name"],
        "score":           record["score"],
        "level":           record["level"],
        "color":           record["color"],
        "frames_captured": record["frames_captured"],
        "defect_frames":   record["defect_frames"],
        "economic_impact": record["economic_impact"],
        "repair_action":   record["repair_action"],
        "annotated_img":   record["annotated_img"],
        "record_id":       record["id"],
        "inspection_date": record["inspection_date"],
        "inspection_time": record["inspection_time"],
        "all_detections":  record["all_detections"],
    })

@app.route("/api/process-frame", methods=["POST"])
@login_required
def api_process_frame():
    data    = request.json
    sess_id = data.get("session_id", "")
    lat     = data.get("lat", 0.0)
    lng     = data.get("lng", 0.0)
    img_b64 = data.get("image", "")
    if not img_b64:
        return jsonify({"error": "No image"}), 400
    sess = sessions_db.get(sess_id)
    if sess:
        sess["frames_captured"] += 1
    if "," in img_b64:
        img_b64 = img_b64.split(",")[1]
    image_bytes = base64.b64decode(img_b64)
    raw_dets, w, h = run_yolo(image_bytes)
    if not raw_dets:
        return jsonify({"detected": False,
                        "frames_captured": sess["frames_captured"] if sess else 0})
    score, scored_dets = calc_score(raw_dets, w, h)
    annotated          = annotate_image(image_bytes, scored_dets)
    if sess:
        save_frame_to_session(sess_id, lat, lng, scored_dets, score, annotated)
    return jsonify({
        "detected":        True,
        "score":           score,
        "level":           get_level(score)[0],
        "color":           get_level(score)[2],
        "detections":      scored_dets,
        "img_url":         sess["worst_img"] if sess else None,
        "frames_captured": sess["frames_captured"] if sess else 0,
        "defect_frames":   len(sess["frames"]) if sess else 0,
        "worst_score":     sess["worst_score"] if sess else score,
    })

@app.route("/api/upload-image", methods=["POST"])
def api_upload_image():
    road_name = request.form.get("road_name", "Unknown Road")
    lat       = float(request.form.get("lat",  17.6868))
    lng       = float(request.form.get("lng",  75.9079))
    f         = request.files.get("image")
    if not f:
        return jsonify({"error": "No file"}), 400
    image_bytes        = f.read()
    raw_dets, w, h     = run_yolo(image_bytes)
    score, scored_dets = calc_score(raw_dets, w, h)
    annotated          = annotate_image(image_bytes, scored_dets)
    record             = save_record(road_name, lat, lng, scored_dets, score, annotated, "citizen")

    # Alert — citizen uploads don't have a logged-in officer,
    # so broadcast to all officers
    send_all_alerts(record, officer_username=None)

    return jsonify({
        "detected":      len(scored_dets) > 0,
        "score":         score,
        "level":         record["level"],
        "color":         record["color"],
        "detections":    scored_dets,
        "img_url":       record["annotated_img"],
        "record_id":     record["id"],
        "economic":      record["economic_impact"],
        "repair_action": record["repair_action"],
    })

@app.route("/api/process-video", methods=["POST"])
@login_required
def api_process_video():
    road_name = request.form.get("road_name", "Unknown Road")
    lat       = float(request.form.get("lat",  17.6868))
    lng       = float(request.form.get("lng",  75.9079))
    f         = request.files.get("video")
    if not f:
        return jsonify({"error": "No video file"}), 400
    tmp_path = os.path.join(OUTPUT_DIR, f"tmp_{uuid.uuid4().hex}.mp4")
    f.save(tmp_path)
    cap      = cv2.VideoCapture(tmp_path)
    fps      = cap.get(cv2.CAP_PROP_FPS) or 30
    interval = int(fps * 2)
    frame_num = 0
    processed = 0
    results   = []
    officer_username = session.get("officer", "")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_num % interval == 0:
                _, buf         = cv2.imencode(".jpg", frame)
                image_bytes    = buf.tobytes()
                raw_dets, w, h = run_yolo(image_bytes)
                if raw_dets:
                    score, scored_dets = calc_score(raw_dets, w, h)
                    annotated          = annotate_image(image_bytes, scored_dets)
                    record             = save_record(road_name, lat, lng, scored_dets,
                                                     score, annotated, "video")
                    send_all_alerts(record, officer_username=officer_username)
                    results.append({"frame": frame_num, "score": score, "level": record["level"]})
                processed += 1
            frame_num += 1
    finally:
        cap.release()
        os.remove(tmp_path)
    return jsonify({
        "frames_analyzed": processed,
        "defects_found":   len(results),
        "results":         results,
        "city_health":     get_city_health_score(),
    })

# ─── PDF Report ───────────────────────────────────────────────────────────────
@app.route("/api/report/<record_id>")
@login_required
def api_report(record_id):
    record = next((r for r in road_records_db if r["id"] == record_id), None)
    if not record:
        return "Record not found", 404
    pdf_path = os.path.join(REPORT_DIR, f"report_{record_id}.pdf")
    _generate_pdf(record, pdf_path)
    return send_file(pdf_path, as_attachment=True,
                     download_name=f"RoadSense_{record['road_name'].replace(' ','_')}.pdf")

def _generate_pdf(record, pdf_path):
    doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                                leftMargin=2*cm, rightMargin=2*cm,
                                topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []
    title_style = ParagraphStyle("title", fontSize=20, fontName="Helvetica-Bold",
                                  textColor=colors.HexColor("#1a1a2e"), alignment=TA_CENTER, spaceAfter=6)
    sub_style   = ParagraphStyle("sub",   fontSize=11, fontName="Helvetica",
                                  textColor=colors.HexColor("#666666"),  alignment=TA_CENTER, spaceAfter=20)
    head_style  = ParagraphStyle("head",  fontSize=13, fontName="Helvetica-Bold",
                                  textColor=colors.HexColor("#1a1a2e"), spaceBefore=16, spaceAfter=8)
    story.append(Paragraph("🛣 RoadSense AI", title_style))
    story.append(Paragraph("Road Inspection Report", sub_style))
    story.append(Spacer(1, 0.3*cm))
    ts   = record["timestamp"][:19].replace("T", " ")
    meta = [
        ["Road Name", record["road_name"],  "Timestamp", ts],
        ["GPS",       f"{record['lat']}, {record['lng']}", "Source", record.get("source","").capitalize()],
        ["Severity",  f"{record['score']}/10", "Level",  record["level"]],
        ["Action",    record["repair_action"],  "Economic Risk", f"₹{record['economic_impact']:,}"],
    ]
    t = Table(meta, colWidths=[3.5*cm, 6*cm, 3.5*cm, 4.5*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#e8f4fd")),
        ("BACKGROUND", (2,0), (2,-1), colors.HexColor("#e8f4fd")),
        ("FONTNAME",   (0,0), (-1,-1), "Helvetica"),
        ("FONTNAME",   (0,0), (0,-1),  "Helvetica-Bold"),
        ("FONTNAME",   (2,0), (2,-1),  "Helvetica-Bold"),
        ("FONTSIZE",   (0,0), (-1,-1), 9),
        ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("PADDING",    (0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))
    if record.get("annotated_img"):
        img_path = os.path.join(BASE_DIR, record["annotated_img"].lstrip("/"))
        if os.path.exists(img_path):
            story.append(Paragraph("Annotated Detection Image", head_style))
            story.append(RLImage(img_path, width=15*cm, height=9*cm))
            story.append(Spacer(1, 0.3*cm))
    if record.get("detections"):
        story.append(Paragraph("Defect Breakdown", head_style))
        det_data = [["#", "Type", "Confidence", "Score", "Size"]]
        for i, d in enumerate(record["detections"], 1):
            bx = d.get("bbox", [0,0,0,0])
            w  = bx[2]-bx[0]; h = bx[3]-bx[1]
            det_data.append([str(i), d.get("label", d.get("class","")),
                              f"{d.get('confidence',0):.0%}", str(d.get("single_score","")),
                              f"{int(w)}×{int(h)}px"])
        dt = Table(det_data, colWidths=[1*cm, 5*cm, 3*cm, 3*cm, 4.5*cm])
        dt.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1a1a2e")),
            ("TEXTCOLOR",  (0,0), (-1,0), colors.white),
            ("FONTNAME",   (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",   (0,0), (-1,-1), 9),
            ("GRID",       (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f9f9f9")]),
            ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
            ("PADDING",    (0,0), (-1,-1), 6),
        ]))
        story.append(dt)
    story.append(Spacer(1, 1*cm))
    footer_style = ParagraphStyle("foot", fontSize=8, textColor=colors.HexColor("#999999"), alignment=TA_CENTER)
    story.append(Paragraph(
        f"Generated by RoadSense AI | {datetime.now().strftime('%d %b %Y %H:%M')} | "
        f"Report ID: {record['id'][:8].upper()}", footer_style))
    doc.build(story)

# ─── Budget Optimizer ─────────────────────────────────────────────────────────
@app.route("/api/budget-optimizer", methods=["POST"])
@login_required
def api_budget_optimizer():
    budget = float(request.json.get("budget", 0))
    roads  = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["score"] > roads[rn]["score"]:
            roads[rn] = r
    sorted_roads = sorted(roads.values(), key=lambda x: x["score"], reverse=True)
    def est_cost(record):
        base_cost = sum(d.get("count", 1) * 10000 for d in record.get("all_detections", record.get("detections",[])))
        return max(base_cost, 5000)
    selected  = []
    remaining = budget
    old_hs    = get_city_health_score()
    for r in sorted_roads:
        cost = est_cost(r)
        if cost <= remaining:
            selected.append({**r, "estimated_cost": cost})
            remaining -= cost
    return jsonify({
        "selected":           selected,
        "total_roads":        len(selected),
        "budget_used":        budget - remaining,
        "budget_left":        remaining,
        "old_health":         old_hs,
        "estimated_health":   round(min(100, old_hs + len(selected) * 2), 1),
    })

# ─── AI Chat Agent ────────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def api_chat():
    user_msg = request.json.get("message", "")
    if not user_msg:
        return jsonify({"reply": "Please type a message."})
    stats = get_stats()
    queue = sorted(defects_db, key=lambda x: x["score"], reverse=True)[:10]
    queue_summary = "\n".join(
        f"  - {r['road_name']}: {r['score']}/10 ({r['level']}) — {r['repair_action']} — ₹{r['economic_impact']:,} risk"
        for r in queue
    )
    context = f"""You are RoadSense AI, an intelligent road governance assistant for Indian municipalities.

LIVE DATABASE SUMMARY:
- Total detections: {stats['total']}
- Critical roads (score 6+): {stats['critical']}
- Emergency roads (score 8+): {stats['emergency']}
- Total economic risk (30 days): ₹{stats['economic_risk']:,}
- City Health Score: {stats['city_health']}/100

TOP PRIORITY ROADS:
{queue_summary if queue_summary else '  No roads detected yet.'}

Answer the officer's question using this live data. Be specific with road names, numbers, and rupee amounts. Keep response concise (under 200 words)."""
    try:
        import anthropic
        client   = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            system=context,
            messages=[{"role": "user", "content": user_msg}]
        )
        reply = response.content[0].text
    except Exception as e:
        reply = _rule_based_chat(user_msg, stats, queue)
    return jsonify({"reply": reply})

def _rule_based_chat(msg, stats, queue):
    msg = msg.lower()
    if "budget" in msg or "fix" in msg or "repair" in msg:
        if queue:
            ans = "Based on current data, I recommend fixing:\n"
            for i, r in enumerate(queue[:3], 1):
                ans += f"{i}. {r['road_name']} (Score {r['score']}/10) — ₹{r['economic_impact']:,} risk\n"
            return ans
    if "worst" in msg or "critical" in msg or "dangerous" in msg:
        if queue:
            r = queue[0]
            return (f"The most critical road is {r['road_name']} with score {r['score']}/10. "
                    f"Action required: {r['repair_action']}. "
                    f"Economic risk: ₹{r['economic_impact']:,}/30 days.")
    if "score" in msg or "health" in msg or "city" in msg:
        return (f"Current City Health Score is {stats['city_health']}/100. "
                f"There are {stats['critical']} critical roads and {stats['emergency']} emergency roads. "
                f"Total economic risk over 30 days: ₹{stats['economic_risk']:,}.")
    if "total" in msg or "how many" in msg:
        return (f"I have detected {stats['total']} road defects so far. "
                f"{stats['critical']} are critical (score 6+) and "
                f"{stats['emergency']} are emergency (score 8+).")
    return ("I can help you with repair recommendations, budget planning, and road priority analysis. "
            "Try asking: 'Which road should I fix first?' or 'What is the city health score?'")

# ─── Demo Data ────────────────────────────────────────────────────────────────
@app.route("/api/add-demo-data", methods=["POST"])
@login_required
def api_add_demo():
    demo_records = [
        {"road": "MG Road",       "lat": 17.6868, "lng": 75.9079, "type": "pothole",            "conf": 0.91},
        {"road": "Station Road",  "lat": 17.6820, "lng": 75.9010, "type": "alligator_crack",    "conf": 0.84},
        {"road": "College Road",  "lat": 17.6900, "lng": 75.9150, "type": "transverse_crack",   "conf": 0.78},
        {"road": "Market Road",   "lat": 17.6780, "lng": 75.9000, "type": "longitudinal_crack", "conf": 0.70},
        {"road": "Ring Road",     "lat": 17.6950, "lng": 75.9200, "type": "pothole",            "conf": 0.95},
        {"road": "Nehru Nagar",   "lat": 17.6750, "lng": 75.9050, "type": "alligator_crack",    "conf": 0.88},
        {"road": "Gandhi Chowk",  "lat": 17.6810, "lng": 75.9120, "type": "pothole",            "conf": 0.93},
        {"road": "Tilak Road",    "lat": 17.6840, "lng": 75.9080, "type": "transverse_crack",   "conf": 0.75},
    ]
    for d in demo_records:
        cfg       = DAMAGE_CONFIG.get(d["type"], {"base": 5, "repair_cost": 10000, "label": d["type"]})
        size_mult = 1.3
        conf_w    = 1.0 if d["conf"] >= 0.9 else (0.85 if d["conf"] >= 0.7 else 0.7)
        score     = min(10.0, cfg["base"] * size_mult * conf_w)
        det       = [{
            "class": d["type"], "confidence": d["conf"],
            "bbox":  [50, 50, 200, 200],
            "label": cfg["label"], "single_score": round(score, 2),
            "repair_cost": cfg["repair_cost"],
        }]
        save_record(d["road"], d["lat"], d["lng"], det, round(score, 2), None, "demo")
    return jsonify({"ok": True, "added": len(demo_records), "total": len(road_records_db)})

@app.route("/api/reset", methods=["POST"])
@login_required
def api_reset():
    defects_db.clear()
    road_records_db.clear()
    return jsonify({"ok": True})


# ─── Predictive Intelligence ──────────────────────────────────────────────────
def _days_since(ts_str):
    try:
        ts = datetime.fromisoformat(ts_str)
        return (datetime.now() - ts).days
    except Exception:
        return 0

def _monsoon_risk(record):
    score  = record.get("score", 0)
    days   = _days_since(record.get("timestamp", datetime.now().isoformat()))
    types  = [d.get("label","") for d in record.get("all_detections", record.get("detections", []))]
    crack_w = 0
    for t in types:
        tl = t.lower()
        if "transverse"  in tl: crack_w = max(crack_w, 0.9)
        elif "alligator" in tl: crack_w = max(crack_w, 0.85)
        elif "longitudinal" in tl: crack_w = max(crack_w, 0.5)
        elif "pothole"   in tl: crack_w = max(crack_w, 0.7)
    return round(min(100, (score/10)*40 + crack_w*30 + min(20,(days/90)*20) + 10), 1)

def _predict_deterioration(record):
    current  = record.get("score", 0)
    types    = [d.get("label","") for d in record.get("all_detections", record.get("detections",[]))]
    type_str = " ".join(types).lower()
    if "pothole"      in type_str: rate = 1.30
    elif "alligator"  in type_str: rate = 1.22
    elif "transverse" in type_str: rate = 1.15
    elif "longitudinal" in type_str: rate = 1.10
    else:                          rate = 1.05
    def _lv(s):
        if s >= 8: return "Emergency"
        if s >= 6: return "Critical"
        if s >= 4: return "Severe"
        if s >= 2: return "Moderate"
        return "Good"
    p30 = round(min(10.0, current * rate), 2)
    p60 = round(min(10.0, current * rate**2), 2)
    p90 = round(min(10.0, current * rate**3), 2)
    days_to_emergency = None
    if current < 8 and rate > 1.0:
        import math
        try: days_to_emergency = round((math.log(8/current)/math.log(rate))*30)
        except Exception: pass
    return {"current": current, "in_30_days": p30, "level_30": _lv(p30),
            "in_60_days": p60, "level_60": _lv(p60),
            "in_90_days": p90, "level_90": _lv(p90),
            "days_to_emergency": days_to_emergency, "deterioration_rate": rate}

@app.route("/api/predictive-analysis")
@login_required
def api_predictive():
    results = []
    roads = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["timestamp"] > roads[rn]["timestamp"]:
            roads[rn] = r
    for rn, r in roads.items():
        pred   = _predict_deterioration(r)
        m_risk = _monsoon_risk(r)
        days   = _days_since(r.get("timestamp", datetime.now().isoformat()))
        hidden_danger = (r["score"] < 6) and (pred["in_60_days"] >= 6)
        results.append({
            "id": r["id"], "road_name": rn,
            "current_score": r["score"], "current_level": r["level"], "color": r["color"],
            "prediction": pred, "monsoon_risk": m_risk, "days_since_inspection": days,
            "hidden_danger": hidden_danger,
            "economic_impact_now": r["economic_impact"],
            "economic_impact_90d": calc_economic_impact(pred["in_90_days"]),
            "inspection_date": r.get("inspection_date",""), "source": r.get("source",""),
            "annotated_img": r.get("annotated_img") or r.get("worst_img"),
        })
    results.sort(key=lambda x: (not x["hidden_danger"], -x["monsoon_risk"]))
    return jsonify(results)

@app.route("/api/monsoon-report")
@login_required
def api_monsoon_report():
    roads = {}
    for r in road_records_db:
        rn = r["road_name"]
        if rn not in roads or r["timestamp"] > roads[rn]["timestamp"]:
            roads[rn] = r
    risk_list  = []
    total_cost = 0
    for rn, r in roads.items():
        m_risk      = _monsoon_risk(r)
        pred        = _predict_deterioration(r)
        repair_cost = max(50000, int(r["score"] * 50000))
        if m_risk >= 50:
            total_cost += repair_cost
        risk_list.append({
            "road_name": rn, "monsoon_risk": m_risk,
            "current_score": r["score"], "predicted_score_monsoon": pred["in_60_days"],
            "color": r["color"], "repair_cost": repair_cost,
            "must_fix": m_risk >= 70, "watch": 50 <= m_risk < 70, "safe": m_risk < 50,
        })
    risk_list.sort(key=lambda x: -x["monsoon_risk"])
    must_fix   = [r for r in risk_list if r["must_fix"]]
    watch_list = [r for r in risk_list if r["watch"]]
    safe_list  = [r for r in risk_list if r["safe"]]
    overall_prep = 0
    if risk_list:
        avg_risk     = sum(r["monsoon_risk"] for r in risk_list) / len(risk_list)
        overall_prep = round(100 - avg_risk, 1)
    return jsonify({
        "overall_preparedness": overall_prep, "total_roads": len(risk_list),
        "must_fix_count": len(must_fix), "watch_count": len(watch_list),
        "safe_count": len(safe_list), "estimated_repair_cost": total_cost,
        "must_fix": must_fix, "watch_list": watch_list, "safe_list": safe_list,
        "generated_at": datetime.now().strftime("%d %b %Y %H:%M"),
    })

@app.route("/api/zone-intelligence")
@login_required
def api_zone_intelligence():
    zones = {}
    for r in road_records_db:
        if not r.get("lat") or not r.get("lng"):
            continue
        zone_lat = round(r["lat"], 2)
        zone_lng = round(r["lng"], 2)
        key = f"{zone_lat}_{zone_lng}"
        if key not in zones:
            zones[key] = {"zone_id": key, "center_lat": zone_lat, "center_lng": zone_lng,
                          "roads": [], "zone_name": f"Zone {len(zones)+1}"}
        zones[key]["roads"].append(r)
    result = []
    for key, z in zones.items():
        roads   = z["roads"]
        n       = len(roads)
        avg_sc  = round(sum(r["score"] for r in roads) / n, 2) if n else 0
        worst   = max(roads, key=lambda x: x["score"]) if roads else {}
        total_e = sum(r["economic_impact"] for r in roads)
        crit_n  = sum(1 for r in roads if r["score"] >= 6)
        m_risks = [_monsoon_risk(r) for r in roads]
        avg_m   = round(sum(m_risks)/len(m_risks), 1) if m_risks else 0
        level, bs, color = get_level(avg_sc)
        result.append({
            "zone_id": key, "zone_name": z["zone_name"],
            "center_lat": z["center_lat"], "center_lng": z["center_lng"],
            "road_count": n, "avg_score": avg_sc, "level": level, "color": color,
            "critical_roads": crit_n, "total_economic_risk": total_e,
            "avg_monsoon_risk": avg_m,
            "worst_road": worst.get("road_name",""), "worst_score": worst.get("score", 0),
        })
    result.sort(key=lambda x: -x["avg_score"])
    return jsonify(result)

# ─── Manual Alert Test Endpoint ───────────────────────────────────────────────
@app.route("/api/test-alerts", methods=["POST"])
@login_required
def api_test_alerts():
    """
    Test endpoint — sends a fake alert so you can verify
    email and WhatsApp are configured correctly before demo.
    Hit this from Postman or the browser console:
    fetch('/api/test-alerts', {method:'POST'})
    """
    fake_record = {
        "id":             "TEST0001",
        "road_name":      "TEST — MG Road Demo",
        "score":          8.5,
        "level":          "Emergency",
        "repair_action":  "Immediately — Today",
        "economic_impact": 157500,
        "lat":            17.6868,
        "lng":            75.9079,
        "timestamp":      datetime.now().isoformat(),
        "source":         "test",
        "all_detections": [
            {"label": "Pothole",         "count": 2},
            {"label": "Alligator Crack", "count": 1},
        ],
        "annotated_img":  None,
    }
    officer_username = session.get("officer", "admin")
    send_all_alerts(fake_record, officer_username=officer_username)
    return jsonify({"ok": True, "message": "Test alerts sent — check your email and WhatsApp"})


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  RoadSense AI — Starting Server")
    print(f"  Alert threshold : score >= {ALERT_CONFIG['alert_threshold']}")
    print(f"  Email alerts    : {'✓ Enabled' if ALERT_CONFIG['email_enabled'] else '✗ Disabled'}")
    print(f"  WhatsApp alerts : {'✓ Enabled' if ALERT_CONFIG['whatsapp_enabled'] else '✗ Disabled'}")
    print("  Dashboard : http://localhost:5000/dashboard")
    print("  City Map  : http://localhost:5000/city-map")
    print("  Citizen   : http://localhost:5000/report")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)