# ============================================================
# DISASTER DAMAGE DETECTOR
# Dark Disaster Response Command Center
# ============================================================

from pathlib import Path
from datetime import datetime
import json
import math
import hashlib
import time

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional MAVLink support
try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except Exception:
    MAVLINK_AVAILABLE = False

# Optional YOLO support
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Disaster Response Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# DARK THEME
# ============================================================
# IMPORTANT:
# This is ONLY CSS styling.
# All visible dashboard content below uses native Streamlit
# components. No HTML dashboard markup is used.
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: #07101d;
            color: #e8eef7;
        }

        [data-testid="stAppViewContainer"] {
            background: #07101d;
        }

        [data-testid="stHeader"] {
            background: #07101d;
        }

        [data-testid="stSidebar"] {
            background: #050c16;
            border-right: 1px solid #1d3349;
        }

        [data-testid="stSidebar"] * {
            color: #dce7f4;
        }

        h1, h2, h3, h4 {
            color: #f3f7fb !important;
        }

        p, label, span {
            color: #cbd7e5;
        }

        [data-testid="stMetric"] {
            background: #0d1b2a;
            border: 1px solid #1c3b55;
            border-radius: 14px;
            padding: 14px;
        }

        [data-testid="stMetricValue"] {
            color: #f3f7fb;
        }

        [data-testid="stMetricLabel"] {
            color: #91a7bc;
        }

        div[data-testid="stExpander"] {
            background: #0b1826;
            border: 1px solid #1c3b55;
            border-radius: 12px;
        }

        .stButton > button {
            border-radius: 9px;
            border: 1px solid #31516c;
            background: #102437;
            color: #eaf3fb;
            min-height: 42px;
        }

        .stButton > button:hover {
            border-color: #5d9ac5;
            background: #16344d;
        }

        .stTextInput input,
        .stNumberInput input,
        .stSelectbox div,
        .stMultiSelect div {
            background: #0d1a29 !important;
            color: #edf4fa !important;
        }

        [data-testid="stFileUploader"] {
            background: #0b1826;
            border: 1px dashed #31516c;
            border-radius: 12px;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        hr {
            border-color: #20384e;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid #1c3b55;
            border-radius: 10px;
        }

        .small-muted {
            color: #7f96aa;
            font-size: 0.85rem;
        }

        .status-online {
            color: #57d68d;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

HISTORY_DIR = BASE_DIR / "drone_history"
SCANS_DIR = HISTORY_DIR / "scans"
RESULTS_DIR = HISTORY_DIR / "results"

NOTIFICATIONS_FILE = HISTORY_DIR / "notifications.json"
TELEMETRY_FILE = HISTORY_DIR / "telemetry.json"

HISTORY_DIR.mkdir(exist_ok=True)
SCANS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULTS = {
    "gps": {
        "lat": 23.2599,
        "lon": 77.4126,
        "altitude": 80.0,
        "heading": 0.0,
        "source": "Manual",
        "timestamp": None,
    },
    "notifications": [],
    "telemetry": [],
    "last_drone_result": None,
    "last_satellite_result": None,
    "last_camera_result": None,
    "yolo_model": None,
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# BASIC HELPERS
# ============================================================

def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def make_scan_id(prefix="SCAN"):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"


def image_hash(image):
    if isinstance(image, Image.Image):
        image = np.array(image)

    return hashlib.sha256(
        image.tobytes()
    ).hexdigest()[:16]


def uploaded_to_cv(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv_to_pil(image):
    if image is None:
        return None

    if len(image.shape) == 2:
        return Image.fromarray(image)

    return Image.fromarray(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    )


def cv_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_same(image1, image2, width=1000):
    if image1 is None or image2 is None:
        return None, None

    h1, w1 = image1.shape[:2]

    if w1 > width:
        scale = width / w1
        image1 = cv2.resize(
            image1,
            (int(w1 * scale), int(h1 * scale)),
            interpolation=cv2.INTER_AREA,
        )

    h2, w2 = image2.shape[:2]

    if w2 > width:
        scale = width / w2
        image2 = cv2.resize(
            image2,
            (int(w2 * scale), int(h2 * scale)),
            interpolation=cv2.INTER_AREA,
        )

    target_h = min(image1.shape[0], image2.shape[0])
    target_w = min(image1.shape[1], image2.shape[1])

    image1 = cv2.resize(
        image1,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    image2 = cv2.resize(
        image2,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )

    return image1, image2


# ============================================================
# JSON STORAGE
# ============================================================

def load_json(path, default):
    try:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass

    return default


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as e:
        st.warning(f"Could not save {path.name}: {e}")


def load_notifications():
    return load_json(NOTIFICATIONS_FILE, [])


def save_notifications(data):
    save_json(NOTIFICATIONS_FILE, data)


def load_telemetry():
    return load_json(TELEMETRY_FILE, [])


def save_telemetry(data):
    save_json(TELEMETRY_FILE, data)


# ============================================================
# GPS FUNCTIONS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(p1)
        * math.cos(p2)
        * math.sin(dlon / 2) ** 2
    )

    return earth_radius * 2 * math.asin(math.sqrt(a))


def generate_demo_gps():
    base_lat = 23.2599
    base_lon = 77.4126

    t = time.time()

    lat = base_lat + math.sin(t / 15) * 0.002
    lon = base_lon + math.cos(t / 18) * 0.002

    altitude = 70 + math.sin(t / 10) * 15
    heading = (t * 8) % 360

    return {
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "altitude": round(altitude, 2),
        "heading": round(heading, 2),
        "source": "Demo Simulator",
        "timestamp": now_string(),
    }


def read_mavlink(endpoint, baud):
    if not MAVLINK_AVAILABLE:
        return None, "pymavlink is not installed."

    try:
        connection = mavutil.mavlink_connection(
            endpoint,
            baud=baud,
        )

        msg = connection.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=3,
        )

        if msg is None:
            return None, "No GPS telemetry received."

        gps = {
            "lat": round(msg.lat / 1e7, 6),
            "lon": round(msg.lon / 1e7, 6),
            "altitude": round(msg.relative_alt / 1000, 2),
            "heading": round(msg.hdg / 100, 2),
            "source": "MAVLink",
            "timestamp": now_string(),
        }

        return gps, None

    except Exception as e:
        return None, str(e)


def update_gps(source, lat=None, lon=None, altitude=None, heading=None,
               mavlink_endpoint=None, mavlink_baud=57600):

    if source == "Demo Simulator":
        gps = generate_demo_gps()

    elif source == "MAVLink":
        gps, error = read_mavlink(
            mavlink_endpoint,
            mavlink_baud,
        )

        if gps is None:
            return False, error

    else:
        gps = {
            "lat": float(lat),
            "lon": float(lon),
            "altitude": float(altitude),
            "heading": float(heading),
            "source": "Manual",
            "timestamp": now_string(),
        }

    st.session_state.gps = gps

    telemetry = load_telemetry()

    telemetry.append(gps)

    telemetry = telemetry[-500:]

    save_telemetry(telemetry)

    return True, None


# ============================================================
# DRONE SCAN HISTORY
# ============================================================

def load_scan_history():
    records = []

    for file in HISTORY_DIR.glob("scan_*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception:
            continue

    records.sort(
        key=lambda x: x.get("timestamp", ""),
        reverse=True,
    )

    return records


def save_scan_record(record):
    scan_id = record["scan_id"]

    path = HISTORY_DIR / f"scan_{scan_id}.json"

    save_json(path, record)


def find_nearest_scan(lat, lon, radius_km, scan_type=None):
    history = load_scan_history()

    candidates = []

    for scan in history:

        if scan_type and scan.get("scan_type") != scan_type:
            continue

        scan_lat = scan.get("lat")
        scan_lon = scan.get("lon")

        if scan_lat is None or scan_lon is None:
            continue

        distance = haversine_km(
            lat,
            lon,
            scan_lat,
            scan_lon,
        )

        if distance <= radius_km:
            candidates.append(
                (
                    distance,
                    scan,
                )
            )

    candidates.sort(
        key=lambda x: x[0]
    )

    if candidates:
        return candidates[0]

    return None


# ============================================================
# IMAGE ALIGNMENT
# ============================================================

def align_images(reference, current):
    ref_gray = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY,
    )

    cur_gray = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY,
    )

    orb = cv2.ORB_create(
        nfeatures=3000
    )

    key1, desc1 = orb.detectAndCompute(
        ref_gray,
        None,
    )

    key2, desc2 = orb.detectAndCompute(
        cur_gray,
        None,
    )

    if desc1 is None or desc2 is None:
        return current, False

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING
    )

    matches = matcher.knnMatch(
        desc2,
        desc1,
        k=2,
    )

    good = []

    for pair in matches:

        if len(pair) != 2:
            continue

        m, n = pair

        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 8:
        return current, False

    src_pts = np.float32(
        [
            key2[m.queryIdx].pt
            for m in good
        ]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [
            key1[m.trainIdx].pt
            for m in good
        ]
    ).reshape(-1, 1, 2)

    homography, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        5.0,
    )

    if homography is None:
        return current, False

    h, w = reference.shape[:2]

    aligned = cv2.warpPerspective(
        current,
        homography,
        (w, h),
    )

    return aligned, True


# ============================================================
# DAMAGE ANALYSIS
# ============================================================

def calculate_difference(
    reference,
    current,
    blur_size=5,
    threshold=35,
):

    reference, current = resize_same(
        reference,
        current,
    )

    if reference is None:
        return None, None

    aligned, aligned_ok = align_images(
        reference,
        current,
    )

    if not aligned_ok:
        aligned = current

    gray_ref = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY,
    )

    gray_current = cv2.cvtColor(
        aligned,
        cv2.COLOR_BGR2GRAY,
    )

    gray_ref = cv2.GaussianBlur(
        gray_ref,
        (blur_size, blur_size),
        0,
    )

    gray_current = cv2.GaussianBlur(
        gray_current,
        (blur_size, blur_size),
        0,
    )

    diff = cv2.absdiff(
        gray_ref,
        gray_current,
    )

    diff = cv2.normalize(
        diff,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
    )

    mask = np.where(
        diff >= threshold,
        255,
        0,
    ).astype(np.uint8)

    kernel = np.ones(
        (5, 5),
        np.uint8,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    return diff, mask


def damage_class(value, critical, moderate, low):
    if value >= critical:
        return "Critical"

    if value >= moderate:
        return "Moderate"

    if value >= low:
        return "Low"

    return "Safe"


def create_damage_map(
    reference,
    current,
    critical_threshold,
    moderate_threshold,
    low_threshold,
):

    diff, mask = calculate_difference(
        reference,
        current,
        threshold=low_threshold,
    )

    if diff is None:
        return None, None

    damage_map = np.zeros_like(
        current
    )

    critical = diff >= critical_threshold
    moderate = (
        (diff >= moderate_threshold)
        & (diff < critical_threshold)
    )
    low = (
        (diff >= low_threshold)
        & (diff < moderate_threshold)
    )

    damage_map[critical] = [
        0,
        0,
        255,
    ]

    damage_map[moderate] = [
        0,
        165,
        255,
    ]

    damage_map[low] = [
        0,
        255,
        255,
    ]

    safe = ~(
        critical
        | moderate
        | low
    )

    damage_map[safe] = current[safe] // 4

    return damage_map, diff


def damage_statistics(diff, critical, moderate, low):

    total = diff.size

    critical_pixels = np.sum(
        diff >= critical
    )

    moderate_pixels = np.sum(
        (diff >= moderate)
        & (diff < critical)
    )

    low_pixels = np.sum(
        (diff >= low)
        & (diff < moderate)
    )

    safe_pixels = total - (
        critical_pixels
        + moderate_pixels
        + low_pixels
    )

    return {
        "Critical": round(
            critical_pixels / total * 100,
            2,
        ),
        "Moderate": round(
            moderate_pixels / total * 100,
            2,
        ),
        "Low": round(
            low_pixels / total * 100,
            2,
        ),
        "Safe": round(
            safe_pixels / total * 100,
            2,
        ),
    }


# ============================================================
# RESPONDER / DAMAGE MARKERS
# ============================================================

def add_damage_markers(image, diff, threshold=45):
    output = image.copy()

    mask = np.where(
        diff >= threshold,
        255,
        0,
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    markers = []

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        if area < 150:
            continue

        x, y, w, h = cv2.boundingRect(
            contour
        )

        cx = x + w // 2
        cy = y + h // 2

        markers.append(
            {
                "x": cx,
                "y": cy,
                "area": round(area, 1),
            }
        )

        cv2.circle(
            output,
            (cx, cy),
            7,
            (255, 255, 255),
            -1,
        )

        cv2.circle(
            output,
            (cx, cy),
            9,
            (0, 0, 0),
            2,
        )

    return output, markers


# ============================================================
# YOLO
# ============================================================

@st.cache_resource
def load_yolo_model():

    if not YOLO_AVAILABLE:
        return None

    candidates = [
        BASE_DIR / "models" / "disaster_best.pt",
        BASE_DIR / "models" / "best.pt",
        BASE_DIR / "best.pt",
    ]

    for path in candidates:
        if path.exists():
            try:
                return YOLO(str(path))
            except Exception:
                pass

    try:
        return YOLO("yolo26n.pt")
    except Exception:
        try:
            return YOLO("yolo11n.pt")
        except Exception:
            try:
                return YOLO("yolov8n.pt")
            except Exception:
                return None


def get_detection_label(names, class_id):
    try:
        if isinstance(names, dict):
            return str(
                names.get(
                    int(class_id),
                    "object",
                )
            )

        return str(
            names[int(class_id)]
        )

    except Exception:
        return "object"


def run_yolo(image, confidence=0.35):

    model = load_yolo_model()

    if model is None:
        return None, []

    try:
        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )

        if not results:
            return None, []

        result = results[0]

        plotted = result.plot()

        detections = []

        if result.boxes is not None:

            for box in result.boxes:

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                    .tolist()
                )

                conf = float(
                    box.conf[0]
                    .cpu()
                    .numpy()
                )

                class_id = int(
                    box.cls[0]
                    .cpu()
                    .numpy()
                )

                label = get_detection_label(
                    result.names,
                    class_id,
                )

                detections.append(
                    {
                        "Class": label,
                        "Confidence": round(
                            conf,
                            3,
                        ),
                        "X1": round(
                            xyxy[0]
                        ),
                        "Y1": round(
                            xyxy[1]
                        ),
                        "X2": round(
                            xyxy[2]
                        ),
                        "Y2": round(
                            xyxy[3]
                        ),
                    }
                )

        return plotted, detections

    except Exception as e:
        st.error(
            f"YOLO inference failed: {e}"
        )

        return None, []


# ============================================================
# SURVIVOR LOCATION ESTIMATION
# ============================================================

def estimate_ground_position(
    gps,
    bbox,
    image_width,
    image_height,
    hfov,
    vfov,
):

    x1, y1, x2, y2 = bbox

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    normalized_x = (
        center_x / image_width
    ) - 0.5

    normalized_y = (
        center_y / image_height
    ) - 0.5

    horizontal_angle = (
        normalized_x * hfov
    )

    vertical_angle = (
        normalized_y * vfov
    )

    altitude = max(
        float(gps["altitude"]),
        1.0,
    )

    horizontal_distance = (
        altitude
        * math.tan(
            math.radians(
                horizontal_angle
            )
        )
    )

    forward_distance = (
        altitude
        * math.tan(
            math.radians(
                -vertical_angle
            )
        )
    )

    heading = math.radians(
        float(gps["heading"])
    )

    east = (
        horizontal_distance
        * math.sin(heading)
        + forward_distance
        * math.cos(heading)
    )

    north = (
        horizontal_distance
        * math.cos(heading)
        - forward_distance
        * math.sin(heading)
    )

    lat = (
        gps["lat"]
        + north / 111320
    )

    lon = (
        gps["lon"]
        + east
        / (
            111320
            * math.cos(
                math.radians(
                    gps["lat"]
                )
            )
        )
    )

    return round(lat, 6), round(lon, 6)


def create_survivor_alerts(
    detections,
    image_shape,
    gps,
    scan_id,
    estimate_position=True,
    hfov=70,
    vfov=50,
):

    alerts = []

    height, width = image_shape[:2]

    for index, detection in enumerate(
        detections,
        start=1,
    ):

        label = detection["Class"].lower()

        if label != "person":
            continue

        x1 = detection["X1"]
        y1 = detection["Y1"]
        x2 = detection["X2"]
        y2 = detection["Y2"]

        if estimate_position:

            lat, lon = estimate_ground_position(
                gps,
                (x1, y1, x2, y2),
                width,
                height,
                hfov,
                vfov,
            )

        else:
            lat = gps["lat"]
            lon = gps["lon"]

        alert = {
            "alert_id": make_scan_id("ALERT"),
            "type": "Possible Survivor",
            "scan_id": scan_id,
            "latitude": lat,
            "longitude": lon,
            "confidence": round(
                detection["Confidence"],
                3,
            ),
            "timestamp": now_string(),
            "gps_source": gps["source"],
        }

        alerts.append(alert)

    return alerts


# ============================================================
# ALERT STORAGE
# ============================================================

def add_alerts(alerts):

    if not alerts:
        return

    existing = load_notifications()

    existing.extend(alerts)

    existing = existing[-1000:]

    save_notifications(existing)

    st.session_state.notifications = existing


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛰️ Response Control")
    st.caption(
        "Disaster intelligence and field response system"
    )

    st.divider()

    st.subheader("⚙️ Analysis Settings")

    critical_threshold = st.slider(
        "Critical Threshold",
        min_value=40,
        max_value=200,
        value=100,
        step=5,
    )

    moderate_threshold = st.slider(
        "Moderate Threshold",
        min_value=20,
        max_value=120,
        value=65,
        step=5,
    )

    low_threshold = st.slider(
        "Low Threshold",
        min_value=5,
        max_value=80,
        value=35,
        step=5,
    )

    st.divider()

    st.subheader("📡 GPS Telemetry")

    gps_source = st.selectbox(
        "GPS Source",
        [
            "Manual",
            "Demo Simulator",
            "MAVLink",
        ],
    )

    manual_lat = st.number_input(
        "Latitude",
        value=float(
            st.session_state.gps["lat"]
        ),
        format="%.6f",
    )

    manual_lon = st.number_input(
        "Longitude",
        value=float(
            st.session_state.gps["lon"]
        ),
        format="%.6f",
    )

    manual_alt = st.number_input(
        "Altitude (m)",
        min_value=1.0,
        value=float(
            st.session_state.gps["altitude"]
        ),
    )

    manual_heading = st.number_input(
        "Heading (°)",
        min_value=0.0,
        max_value=360.0,
        value=float(
            st.session_state.gps["heading"]
        ),
    )

    mavlink_endpoint = "/dev/tty.usbmodem"

    mavlink_baud = 57600

    if gps_source == "MAVLink":

        mavlink_endpoint = st.text_input(
            "MAVLink Endpoint",
            value="/dev/tty.usbmodem",
        )

        mavlink_baud = st.number_input(
            "Baud Rate",
            value=57600,
            step=9600,
        )

    if st.button(
        "🔄 Update Telemetry",
        width="stretch",
    ):

        success, error = update_gps(
            gps_source,
            manual_lat,
            manual_lon,
            manual_alt,
            manual_heading,
            mavlink_endpoint,
            mavlink_baud,
        )

        if success:
            st.success(
                "Telemetry updated."
            )

        else:
            st.error(
                error
                or "Telemetry update failed."
            )

    st.divider()

    st.subheader("🤖 AI Detection")

    yolo_confidence = st.slider(
        "YOLO Confidence",
        min_value=0.10,
        max_value=0.95,
        value=0.35,
        step=0.05,
    )

    estimate_survivor_position = st.checkbox(
        "Estimate survivor coordinates",
        value=True,
    )

    hfov = st.slider(
        "Camera HFOV",
        30,
        120,
        70,
    )

    vfov = st.slider(
        "Camera VFOV",
        20,
        100,
        50,
    )

    st.divider()

    st.subheader("🟢 System Status")

    st.success("SYSTEM OPERATIONAL")

    st.caption(
        f"Last telemetry: "
        f"{st.session_state.gps.get('timestamp') or 'Not updated'}"
    )


# ============================================================
# LOAD PERSISTENT DATA
# ============================================================

notifications = load_notifications()
st.session_state.notifications = notifications

scan_history = load_scan_history()


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🛰️ Disaster Response Command Center"
)

st.caption(
    "AI-powered drone damage assessment • GPS intelligence • "
    "survivor detection • satellite comparison"
)

st.divider()


# ============================================================
# KPI ROW
# ============================================================

gps = st.session_state.gps

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "🛰️ Drone Status",
        "LIVE",
        gps["source"],
    )

with col2:
    st.metric(
        "📍 Current Latitude",
        f"{gps['lat']:.5f}",
        f"Lon {gps['lon']:.5f}",
    )

with col3:
    st.metric(
        "🚨 Active Alerts",
        len(notifications),
        "Persistent history",
    )

with col4:
    st.metric(
        "📊 Indexed Scans",
        len(scan_history),
        "GPS indexed",
    )


# ============================================================
# MAIN TABS
# ============================================================

tab_drone, tab_satellite, tab_camera, tab_alerts = st.tabs(
    [
        "🛰️ Drone Command",
        "🗺️ Satellite Analysis",
        "📷 Camera AI",
        "🔔 Alert History",
    ]
)


# ============================================================
# DRONE COMMAND
# ============================================================

with tab_drone:

    st.header("🛰️ Drone Mission Control")

    top_left, top_right = st.columns(
        [1.2, 1]
    )

    with top_left:

        st.subheader(
            "📍 Current Drone Position"
        )

        map_df = pd.DataFrame(
            [
                {
                    "lat": gps["lat"],
                    "lon": gps["lon"],
                }
            ]
        )

        st.map(
            map_df,
            zoom=14,
        )

    with top_right:

        st.subheader(
            "📡 Telemetry"
        )

        telemetry_data = {
            "Parameter": [
                "Latitude",
                "Longitude",
                "Altitude",
                "Heading",
                "GPS Source",
                "Timestamp",
            ],
            "Value": [
                f"{gps['lat']:.6f}",
                f"{gps['lon']:.6f}",
                f"{gps['altitude']:.2f} m",
                f"{gps['heading']:.2f}°",
                gps["source"],
                gps["timestamp"] or "N/A",
            ],
        }

        st.dataframe(
            pd.DataFrame(telemetry_data),
            width="stretch",
            hide_index=True,
        )

    st.divider()

    st.subheader(
        "📸 Coordinate-Based Drone Scan"
    )

    scan_col1, scan_col2 = st.columns(2)

    with scan_col1:

        coordinate_radius = st.number_input(
            "Matching Radius (km)",
            min_value=0.05,
            max_value=20.0,
            value=2.0,
            step=0.05,
        )

    with scan_col2:

        scan_type = st.selectbox(
            "Scan Type",
            [
                "Post-Disaster",
                "Pre-Disaster",
                "Monitoring",
            ],
        )

    drone_image_file = st.file_uploader(
        "Upload drone image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp",
        ],
        key="drone_upload",
    )

    if drone_image_file is not None:

        current_drone = uploaded_to_cv(
            drone_image_file
        )

        st.subheader(
            "Current Drone Image"
        )

        st.image(
            cv_to_rgb(current_drone),
            width="stretch",
        )

        nearest = None

        if scan_type == "Post-Disaster":

            nearest = find_nearest_scan(
                gps["lat"],
                gps["lon"],
                coordinate_radius,
                "Pre-Disaster",
            )

        else:

            nearest = find_nearest_scan(
                gps["lat"],
                gps["lon"],
                coordinate_radius,
            )

        if nearest:

            distance, previous = nearest

            st.info(
                f"Nearest GPS-matched scan: "
                f"{previous.get('scan_id')} "
                f"at {distance:.3f} km"
            )

            previous_image_path = Path(
                previous.get(
                    "image_path",
                    "",
                )
            )

            if previous_image_path.exists():

                previous_image = cv2.imread(
                    str(previous_image_path)
                )

                if previous_image is not None:

                    with st.expander(
                        "View matched reference scan"
                    ):
                        st.image(
                            cv_to_rgb(
                                previous_image
                            ),
                            width="stretch",
                        )

            else:
                previous_image = None

        else:

            previous_image = None

            if scan_type == "Post-Disaster":

                st.warning(
                    "No pre-disaster scan was found "
                    "within the selected GPS radius."
                )

            else:

                st.info(
                    "No nearby reference scan found."
                )

        process_drone = st.button(
            "🚀 PROCESS DRONE SCAN",
            type="primary",
            width="stretch",
        )

        if process_drone:

            scan_id = make_scan_id(
                "SCAN"
            )

            # ----------------------------------------
            # DAMAGE COMPARISON
            # ----------------------------------------

            damage_map = None
            damage_stats = None
            diff = None
            markers = []

            if previous_image is not None:

                reference, current = resize_same(
                    previous_image,
                    current_drone,
                )

                damage_map, diff = create_damage_map(
                    reference,
                    current,
                    critical_threshold,
                    moderate_threshold,
                    low_threshold,
                )

                if diff is not None:

                    damage_stats = damage_statistics(
                        diff,
                        critical_threshold,
                        moderate_threshold,
                        low_threshold,
                    )

                    damage_map, markers = add_damage_markers(
                        damage_map,
                        diff,
                        low_threshold,
                    )

            # ----------------------------------------
            # YOLO
            # ----------------------------------------

            yolo_image, detections = run_yolo(
                current_drone,
                yolo_confidence,
            )

            # ----------------------------------------
            # SURVIVOR ALERTS
            # ----------------------------------------

            alerts = create_survivor_alerts(
                detections,
                current_drone.shape,
                gps,
                scan_id,
                estimate_survivor_position,
                hfov,
                vfov,
            )

            add_alerts(alerts)

            # ----------------------------------------
            # SAVE IMAGE
            # ----------------------------------------

            image_path = (
                SCANS_DIR
                / f"{scan_id}.jpg"
            )

            cv2.imwrite(
                str(image_path),
                current_drone,
            )

            result_path = None

            if damage_map is not None:

                result_path = (
                    RESULTS_DIR
                    / f"{scan_id}_damage.jpg"
                )

                cv2.imwrite(
                    str(result_path),
                    damage_map,
                )

            # ----------------------------------------
            # SAVE SCAN RECORD
            # ----------------------------------------

            record = {
                "scan_id": scan_id,
                "timestamp": now_string(),
                "scan_type": scan_type,
                "lat": gps["lat"],
                "lon": gps["lon"],
                "altitude": gps["altitude"],
                "heading": gps["heading"],
                "gps_source": gps["source"],
                "image_path": str(
                    image_path
                ),
                "result_path": (
                    str(result_path)
                    if result_path
                    else None
                ),
                "reference_scan_id": (
                    previous.get("scan_id")
                    if nearest
                    else None
                ),
                "reference_distance_km": (
                    round(
                        nearest[0],
                        4,
                    )
                    if nearest
                    else None
                ),
                "damage_statistics": damage_stats,
                "detection_count": len(
                    detections
                ),
                "survivor_alert_count": len(
                    alerts
                ),
            }

            save_scan_record(record)

            # ----------------------------------------
            # STORE RESULT IN SESSION
            # ----------------------------------------

            st.session_state.last_drone_result = {
                "scan_id": scan_id,
                "timestamp": record["timestamp"],
                "damage_map": damage_map,
                "diff": diff,
                "damage_stats": damage_stats,
                "detections": detections,
                "alerts": alerts,
                "yolo_image": yolo_image,
                "markers": markers,
                "record": record,
            }

            st.success(
                f"Scan {scan_id} processed successfully."
            )

            if alerts:
                st.warning(
                    f"🚨 {len(alerts)} possible survivor/person "
                    f"detection(s) generated."
                )
            else:
                st.info(
                    "No person detections were generated."
                )

    # ========================================================
    # LATEST DRONE RESULT
    # ========================================================

    result = st.session_state.last_drone_result

    if result:

        st.divider()

        st.header(
            "📊 Latest Mission Result"
        )

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric(
                "Scan ID",
                result["scan_id"],
            )

        with r2:
            st.metric(
                "AI Detections",
                len(
                    result["detections"]
                ),
            )

        with r3:
            st.metric(
                "Possible Survivors",
                len(
                    result["alerts"]
                ),
            )

        if result["damage_stats"]:

            st.subheader(
                "🔥 Damage Distribution"
            )

            stats = result[
                "damage_stats"
            ]

            chart_df = pd.DataFrame(
                {
                    "Damage Class": list(
                        stats.keys()
                    ),
                    "Area (%)": list(
                        stats.values()
                    ),
                }
            )

            st.bar_chart(
                chart_df.set_index(
                    "Damage Class"
                )
            )

            d1, d2, d3, d4 = st.columns(4)

            d1.metric(
                "🔴 Critical",
                f"{stats['Critical']}%",
            )

            d2.metric(
                "🟠 Moderate",
                f"{stats['Moderate']}%",
            )

            d3.metric(
                "🟡 Low",
                f"{stats['Low']}%",
            )

            d4.metric(
                "🟢 Safe",
                f"{stats['Safe']}%",
            )

        else:

            st.info(
                "No GPS-matched reference image was "
                "available for damage comparison."
            )

        image_col1, image_col2 = st.columns(2)

        with image_col1:

            if result["damage_map"] is not None:

                st.subheader(
                    "🗺️ Damage Map"
                )

                st.image(
                    cv_to_rgb(
                        result["damage_map"]
                    ),
                    width="stretch",
                )

        with image_col2:

            if result["yolo_image"] is not None:

                st.subheader(
                    "🤖 AI Detection"
                )

                st.image(
                    cv_to_rgb(
                        result["yolo_image"]
                    ),
                    width="stretch",
                )

        if result["detections"]:

            st.subheader(
                "🤖 Detection Table"
            )

            st.dataframe(
                pd.DataFrame(
                    result["detections"]
                ),
                width="stretch",
                hide_index=True,
            )

        if result["alerts"]:

            st.subheader(
                "🚨 Survivor Alerts"
            )

            alert_df = pd.DataFrame(
                result["alerts"]
            )

            st.dataframe(
                alert_df,
                width="stretch",
                hide_index=True,
            )

        if result["markers"]:

            st.caption(
                f"{len(result['markers'])} damage regions "
                "identified for responder attention."
            )

    # ========================================================
    # SCAN HISTORY
    # ========================================================

    st.divider()

    st.subheader(
        "📚 GPS Scan History"
    )

    scan_history = load_scan_history()

    if scan_history:

        rows = []

        for scan in scan_history[:25]:

            rows.append(
                {
                    "Scan ID": scan.get(
                        "scan_id"
                    ),
                    "Type": scan.get(
                        "scan_type"
                    ),
                    "Latitude": scan.get(
                        "lat"
                    ),
                    "Longitude": scan.get(
                        "lon"
                    ),
                    "Reference": scan.get(
                        "reference_scan_id"
                    ),
                    "Alerts": scan.get(
                        "survivor_alert_count",
                        0,
                    ),
                    "Time": scan.get(
                        "timestamp"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "No scans have been indexed yet."
        )


# ============================================================
# SATELLITE ANALYSIS
# ============================================================

with tab_satellite:

    st.header(
        "🗺️ Satellite Pre/Post Analysis"
    )

    st.caption(
        "Compare two images from the same location "
        "to estimate disaster-related visual change."
    )

    sat_col1, sat_col2 = st.columns(2)

    with sat_col1:

        satellite_pre = st.file_uploader(
            "Pre-disaster satellite image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            key="sat_pre",
        )

    with sat_col2:

        satellite_post = st.file_uploader(
            "Post-disaster satellite image",
            type=[
                "jpg",
                "jpeg",
                "png",
                "webp",
            ],
            key="sat_post",
        )

    if satellite_pre and satellite_post:

        pre_image = uploaded_to_cv(
            satellite_pre
        )

        post_image = uploaded_to_cv(
            satellite_post
        )

        s1, s2 = st.columns(2)

        with s1:
            st.image(
                cv_to_rgb(pre_image),
                caption="Pre-Disaster",
                width="stretch",
            )

        with s2:
            st.image(
                cv_to_rgb(post_image),
                caption="Post-Disaster",
                width="stretch",
            )

        analyze_satellite = st.button(
            "🛰️ ANALYZE SATELLITE CHANGE",
            type="primary",
            width="stretch",
        )

        if analyze_satellite:

            reference, current = resize_same(
                pre_image,
                post_image,
            )

            damage_map, diff = create_damage_map(
                reference,
                current,
                critical_threshold,
                moderate_threshold,
                low_threshold,
            )

            if diff is not None:

                stats = damage_statistics(
                    diff,
                    critical_threshold,
                    moderate_threshold,
                    low_threshold,
                )

                st.session_state.last_satellite_result = {
                    "damage_map": damage_map,
                    "diff": diff,
                    "stats": stats,
                }

    satellite_result = (
        st.session_state.last_satellite_result
    )

    if satellite_result:

        st.divider()

        st.subheader(
            "📊 Satellite Damage Assessment"
        )

        stats = satellite_result[
            "stats"
        ]

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "🔴 Critical",
            f"{stats['Critical']}%",
        )

        c2.metric(
            "🟠 Moderate",
            f"{stats['Moderate']}%",
        )

        c3.metric(
            "🟡 Low",
            f"{stats['Low']}%",
        )

        c4.metric(
            "🟢 Safe",
            f"{stats['Safe']}%",
        )

        chart_df = pd.DataFrame(
            {
                "Class": list(
                    stats.keys()
                ),
                "Area (%)": list(
                    stats.values()
                ),
            }
        )

        st.bar_chart(
            chart_df.set_index(
                "Class"
            )
        )

        st.subheader(
            "🗺️ Generated Damage Map"
        )

        st.image(
            cv_to_rgb(
                satellite_result[
                    "damage_map"
                ]
            ),
            width="stretch",
        )


# ============================================================
# CAMERA AI
# ============================================================

with tab_camera:

    st.header(
        "📷 Camera AI"
    )

    st.caption(
        "Capture an image and run AI object/person detection."
    )

    camera_image = st.camera_input(
        "Capture disaster scene"
    )

    if camera_image:

        image = uploaded_to_cv(
            camera_image
        )

        st.subheader(
            "Captured Image"
        )

        st.image(
            cv_to_rgb(image),
            width="stretch",
        )

        analyze_camera = st.button(
            "🤖 RUN CAMERA AI",
            type="primary",
            width="stretch",
        )

        if analyze_camera:

            camera_id = make_scan_id(
                "CAM"
            )

            yolo_image, detections = run_yolo(
                image,
                yolo_confidence,
            )

            alerts = create_survivor_alerts(
                detections,
                image.shape,
                gps,
                camera_id,
                estimate_survivor_position,
                hfov,
                vfov,
            )

            add_alerts(alerts)

            st.session_state.last_camera_result = {
                "image": yolo_image,
                "detections": detections,
                "alerts": alerts,
            }

    camera_result = (
        st.session_state.last_camera_result
    )

    if camera_result:

        st.divider()

        if camera_result["image"] is not None:

            st.subheader(
                "🤖 AI Detection Result"
            )

            st.image(
                cv_to_rgb(
                    camera_result["image"]
                ),
                width="stretch",
            )

        detections = camera_result[
            "detections"
        ]

        alerts = camera_result[
            "alerts"
        ]

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "AI Detections",
                len(detections),
            )

        with c2:
            st.metric(
                "🚨 Possible Survivors",
                len(alerts),
            )

        if detections:

            st.subheader(
                "Detection Results"
            )

            st.dataframe(
                pd.DataFrame(detections),
                width="stretch",
                hide_index=True,
            )

        if alerts:

            st.error(
                f"🚨 {len(alerts)} possible survivor/person "
                "alert(s) generated."
            )

            st.dataframe(
                pd.DataFrame(alerts),
                width="stretch",
                hide_index=True,
            )

        else:

            st.success(
                "No person detection was generated."
            )


# ============================================================
# ALERT HISTORY
# ============================================================

with tab_alerts:

    st.header(
        "🔔 Mission Alert History"
    )

    st.caption(
        "Persistent AI-generated survivor/person alerts"
    )

    notifications = load_notifications()

    if not notifications:

        st.info(
            "🔔 No alerts have been generated yet."
        )

        st.caption(
            "Run a drone scan or Camera AI analysis "
            "with a person detection to generate an alert."
        )

    else:

        latest = notifications[-1]

        st.subheader(
            "🚨 Latest Alert"
        )

        l1, l2, l3, l4 = st.columns(4)

        l1.metric(
            "Alert Type",
            latest.get(
                "type",
                "Unknown",
            ),
        )

        l2.metric(
            "Confidence",
            f"{latest.get('confidence', 0) * 100:.1f}%",
        )

        l3.metric(
            "Latitude",
            f"{latest.get('latitude', 0):.6f}",
        )

        l4.metric(
            "Longitude",
            f"{latest.get('longitude', 0):.6f}",
        )

        st.divider()

        st.subheader(
            "🗺️ Alert Locations"
        )

        map_rows = []

        for alert in notifications:

            if (
                alert.get("latitude")
                is not None
                and alert.get("longitude")
                is not None
            ):

                map_rows.append(
                    {
                        "lat": alert[
                            "latitude"
                        ],
                        "lon": alert[
                            "longitude"
                        ],
                    }
                )

        if map_rows:

            st.map(
                pd.DataFrame(map_rows),
                zoom=12,
            )

        st.divider()

        st.subheader(
            "📋 Alert Log"
        )

        alert_rows = []

        for alert in reversed(
            notifications[-100:]
        ):

            alert_rows.append(
                {
                    "Alert ID": alert.get(
                        "alert_id"
                    ),
                    "Type": alert.get(
                        "type"
                    ),
                    "Confidence": (
                        f"{alert.get('confidence', 0) * 100:.1f}%"
                    ),
                    "Latitude": alert.get(
                        "latitude"
                    ),
                    "Longitude": alert.get(
                        "longitude"
                    ),
                    "Scan ID": alert.get(
                        "scan_id"
                    ),
                    "GPS Source": alert.get(
                        "gps_source"
                    ),
                    "Timestamp": alert.get(
                        "timestamp"
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(alert_rows),
            width="stretch",
            hide_index=True,
        )

        if st.button(
            "🗑️ Clear Alert History",
        ):

            save_notifications([])

            st.session_state.notifications = []

            st.success(
                "Alert history cleared."
            )

            st.rerun()


# ============================================================
# FOOTER / DISCLAIMER
# ============================================================

st.divider()

st.caption(
    "Prototype system • Damage classification is an AI/computer-vision "
    "estimate and should be validated by trained responders and "
    "appropriate geospatial/RF/field instrumentation before operational use."
)

st.caption(
    f"System time: {now_string()}  •  "
    f"Scans: {len(load_scan_history())}  •  "
    f"Alerts: {len(load_notifications())}"
)