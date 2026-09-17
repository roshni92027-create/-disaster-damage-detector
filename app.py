import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math
import os

# ============================================================
# OPTIONAL MAVLINK
# ============================================================

PYMAVLINK_AVAILABLE = False

try:
    from pymavlink import mavutil
    PYMAVLINK_AVAILABLE = True
except Exception:
    PYMAVLINK_AVAILABLE = False


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Disaster Response Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM UI
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- GLOBAL ---------- */

    .stApp {
        background: #f4f7fb;
    }

    .main .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* ---------- SIDEBAR ---------- */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #07111f 0%,
                #0b1628 50%,
                #101c30 100%
            );
        border-right: 1px solid #1f3047;
    }

    [data-testid="stSidebar"] * {
        color: #e8eef7 !important;
    }

    [data-testid="stSidebar"] label {
        color: #b9c6d8 !important;
    }

    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea,
    [data-testid="stSidebar"] select {
        background-color: #111e31 !important;
        color: #ffffff !important;
        border-color: #2c405a !important;
    }

    /* ---------- HEADER ---------- */

    .command-header {
        background:
            linear-gradient(
                135deg,
                #ffffff 0%,
                #f8fbff 55%,
                #eef5ff 100%
            );
        border: 1px solid #dce6f2;
        border-radius: 20px;
        padding: 22px 28px;
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(20, 45, 80, 0.07);
    }

    .header-title {
        font-size: 30px;
        font-weight: 800;
        color: #0c1b2e;
        letter-spacing: -0.8px;
    }

    .header-subtitle {
        color: #68788d;
        margin-top: 4px;
        font-size: 14px;
    }

    /* ---------- STATUS ---------- */

    .live-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #e9fbf0;
        color: #08753c;
        border: 1px solid #bceacb;
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
    }

    .live-dot {
        width: 8px;
        height: 8px;
        background: #13a857;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 5px rgba(19,168,87,.10);
        animation: pulse 1.7s infinite;
    }

    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(19,168,87,.35); }
        70% { box-shadow: 0 0 0 8px rgba(19,168,87,0); }
        100% { box-shadow: 0 0 0 0 rgba(19,168,87,0); }
    }

    /* ---------- KPI CARDS ---------- */

    .kpi-card {
        background: #ffffff;
        border: 1px solid #dfe7f0;
        border-radius: 17px;
        padding: 18px;
        min-height: 115px;
        box-shadow: 0 5px 18px rgba(20, 45, 80, 0.055);
        transition: all .2s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 26px rgba(20, 45, 80, 0.11);
        border-color: #b9cde4;
    }

    .kpi-label {
        font-size: 12px;
        color: #75869b;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: .6px;
    }

    .kpi-value {
        font-size: 25px;
        color: #11253e;
        font-weight: 800;
        margin-top: 8px;
    }

    .kpi-small {
        font-size: 11px;
        color: #8a98a9;
        margin-top: 3px;
    }

    /* ---------- SECTION ---------- */

    .section-title {
        font-size: 20px;
        font-weight: 800;
        color: #14263d;
        margin-top: 12px;
        margin-bottom: 12px;
    }

    .section-caption {
        font-size: 12px;
        color: #718197;
        margin-top: -7px;
        margin-bottom: 15px;
    }

    /* ---------- DARK PANEL ---------- */

    .dark-panel {
        background: linear-gradient(
            135deg,
            #0a1627,
            #102239
        );
        border: 1px solid #203853;
        border-radius: 18px;
        padding: 20px;
        color: #ffffff;
        box-shadow: 0 8px 25px rgba(5, 17, 32, .16);
    }

    .dark-label {
        color: #8fa5be;
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 700;
        letter-spacing: .8px;
    }

    .dark-value {
        color: #ffffff;
        font-size: 21px;
        font-weight: 800;
        margin-top: 5px;
    }

    /* ---------- ALERT ---------- */

    .survivor-alert {
        background:
            linear-gradient(
                135deg,
                #fff5f4,
                #ffffff
            );
        border: 1px solid #ffb9b2;
        border-left: 6px solid #e53935;
        border-radius: 17px;
        padding: 18px 20px;
        box-shadow: 0 8px 25px rgba(229,57,53,.10);
        animation: alertIn .35s ease-out;
    }

    @keyframes alertIn {
        from {
            opacity: 0;
            transform: translateY(-5px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .alert-title {
        color: #b71c1c;
        font-weight: 900;
        font-size: 18px;
    }

    .alert-text {
        color: #6f2c2c;
        font-size: 13px;
        margin-top: 5px;
    }

    /* ---------- INFO CARD ---------- */

    .info-card {
        background: #ffffff;
        border: 1px solid #dfe7f0;
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 5px 18px rgba(20,45,80,.05);
    }

    /* ---------- BADGES ---------- */

    .badge {
        display: inline-block;
        padding: 5px 9px;
        border-radius: 7px;
        font-size: 10px;
        font-weight: 800;
        margin-right: 5px;
    }

    .badge-green {
        background: #e8f8ef;
        color: #08753c;
    }

    .badge-red {
        background: #ffebeb;
        color: #b71c1c;
    }

    .badge-orange {
        background: #fff2df;
        color: #a55b00;
    }

    .badge-blue {
        background: #e9f2ff;
        color: #145da0;
    }

    /* ---------- DIVIDER ---------- */

    .soft-divider {
        height: 1px;
        background: #e1e8f0;
        margin: 22px 0;
    }

    /* ---------- STREAMLIT BUTTONS ---------- */

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #cdd9e7;
        font-weight: 700;
        transition: all .18s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: #6d9ed4;
        box-shadow: 0 5px 14px rgba(20,80,140,.10);
    }

    /* ---------- FILE UPLOADER ---------- */

    [data-testid="stFileUploader"] {
        background: #ffffff;
        border-radius: 14px;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ---------- MOBILE ---------- */

    @media (max-width: 800px) {
        .header-title {
            font-size: 23px;
        }

        .kpi-card {
            min-height: 100px;
        }
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DIRECTORIES
# ============================================================

DRONE_ROOT = Path("drone_history")
DRONE_SCANS = DRONE_ROOT / "scans"
DRONE_RESULTS = DRONE_ROOT / "results"

DRONE_ROOT.mkdir(exist_ok=True)
DRONE_SCANS.mkdir(exist_ok=True)
DRONE_RESULTS.mkdir(exist_ok=True)

NOTIFICATION_FILE = DRONE_ROOT / "notifications.json"
TELEMETRY_FILE = DRONE_ROOT / "telemetry.json"


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "satellite_result": None,
    "drone_history": None,
    "drone_last_hash": "",
    "drone_current_analysis": None,
    "notifications": None,
    "telemetry": None,
    "gps": {
        "latitude": 23.2599,
        "longitude": 77.4126,
        "altitude": 50.0,
        "heading": 0.0,
        "source": "Demo"
    },
    "demo_step": 0
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def uploaded_to_cv(uploaded_file):
    if uploaded_file is None:
        return None

    data = uploaded_file.getvalue()

    if not data:
        return None

    arr = np.frombuffer(data, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    return image


def cv_to_rgb(image):
    if image is None:
        return None

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_images(img1, img2, max_width=1000):
    if img1 is None or img2 is None:
        return img1, img2

    h1, w1 = img1.shape[:2]

    if w1 > max_width:
        scale = max_width / w1
        img1 = cv2.resize(
            img1,
            (int(w1 * scale), int(h1 * scale))
        )

    h2, w2 = img2.shape[:2]

    if w2 > max_width:
        scale = max_width / w2
        img2 = cv2.resize(
            img2,
            (int(w2 * scale), int(h2 * scale))
        )

    return img1, img2


def image_hash(uploaded_file):
    if uploaded_file is None:
        return ""

    return hashlib.sha256(
        uploaded_file.getvalue()
    ).hexdigest()


# ============================================================
# GPS
# ============================================================

def haversine_m(lat1, lon1, lat2, lon2):
    """
    Distance between two GPS coordinates in meters.
    """

    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except Exception:
        return float("inf")

    radius = 6371000

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(dlambda / 2) ** 2
    )

    return radius * 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )


def save_telemetry(point):
    history = load_json(TELEMETRY_FILE, [])

    history.append(point)

    history = history[-500:]

    save_json(
        TELEMETRY_FILE,
        history
    )


def load_json(path, default):
    try:
        if not path.exists():
            return default

        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return default


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2
        )


def read_mavlink_gps(endpoint, baud=57600):
    if not PYMAVLINK_AVAILABLE:
        raise RuntimeError(
            "pymavlink is not installed."
        )

    connection = None

    try:

        connection = mavutil.mavlink_connection(
            endpoint,
            baud=baud
        )

        msg = connection.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=5
        )

        if msg is None:
            raise RuntimeError(
                "No GPS telemetry received."
            )

        latitude = msg.lat / 1e7
        longitude = msg.lon / 1e7

        altitude = (
            msg.relative_alt / 1000
            if getattr(msg, "relative_alt", 0)
            else msg.alt / 1000
        )

        heading = (
            msg.hdg / 100
            if getattr(msg, "hdg", 65535) != 65535
            else 0
        )

        return {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "heading": heading,
            "source": "MAVLink",
            "timestamp": now_string()
        }

    finally:
        try:
            if connection:
                connection.close()
        except Exception:
            pass


# ============================================================
# DRONE HISTORY
# ============================================================

def load_drone_history():
    files = sorted(
        DRONE_ROOT.glob("scan_*.json"),
        key=lambda x: x.stat().st_mtime
    )

    records = []

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception:
            continue

    return records


def save_drone_record(record):
    scan_id = record["scan_id"]

    path = DRONE_ROOT / f"{scan_id}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            record,
            f,
            indent=2
        )


def next_scan_id():
    numbers = []

    for file in DRONE_ROOT.glob("scan_*.json"):

        try:
            number = int(
                file.stem.split("_")[1]
            )

            numbers.append(number)

        except Exception:
            pass

    next_number = max(numbers, default=0) + 1

    return f"scan_{next_number:04d}"


def save_drone_image(image, scan_id):
    path = DRONE_SCANS / f"{scan_id}.png"

    cv2.imwrite(
        str(path),
        image
    )

    return str(path)


def save_result_image(image, scan_id, suffix):
    path = DRONE_RESULTS / f"{scan_id}_{suffix}.png"

    cv2.imwrite(
        str(path),
        image
    )

    return str(path)


# ============================================================
# IMAGE ALIGNMENT
# ============================================================

def align_images(reference, current):
    if reference is None or current is None:
        return None, False, 0

    try:

        ref_gray = cv2.cvtColor(
            reference,
            cv2.COLOR_BGR2GRAY
        )

        cur_gray = cv2.cvtColor(
            current,
            cv2.COLOR_BGR2GRAY
        )

        orb = cv2.ORB_create(
            nfeatures=3000
        )

        kp1, des1 = orb.detectAndCompute(
            ref_gray,
            None
        )

        kp2, des2 = orb.detectAndCompute(
            cur_gray,
            None
        )

        if des1 is None or des2 is None:
            return None, False, 0

        matcher = cv2.BFMatcher(
            cv2.NORM_HAMMING,
            crossCheck=True
        )

        matches = matcher.match(
            des1,
            des2
        )

        matches = sorted(
            matches,
            key=lambda x: x.distance
        )

        if len(matches) < 10:
            return None, False, len(matches)

        good = matches[:min(80, len(matches))]

        src_pts = np.float32([
            kp2[m.trainIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)

        dst_pts = np.float32([
            kp1[m.queryIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if matrix is None:
            return None, False, len(good)

        h, w = reference.shape[:2]

        aligned = cv2.warpPerspective(
            current,
            matrix,
            (w, h)
        )

        return aligned, True, len(good)

    except Exception:
        return None, False, 0


def calculate_difference(reference, current):
    reference, current = resize_images(
        reference,
        current
    )

    if reference.shape[:2] != current.shape[:2]:

        current = cv2.resize(
            current,
            (
                reference.shape[1],
                reference.shape[0]
            )
        )

    gray1 = cv2.cvtColor(
        reference,
        cv2.COLOR_BGR2GRAY
    )

    gray2 = cv2.cvtColor(
        current,
        cv2.COLOR_BGR2GRAY
    )

    blur1 = cv2.GaussianBlur(
        gray1,
        (5, 5),
        0
    )

    blur2 = cv2.GaussianBlur(
        gray2,
        (5, 5),
        0
    )

    diff = cv2.absdiff(
        blur1,
        blur2
    )

    diff = cv2.normalize(
        diff,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    _, threshold = cv2.threshold(
        diff,
        35,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel
    )

    return diff, threshold


# ============================================================
# DAMAGE ANALYSIS
# ============================================================

def create_damage_map(
    difference,
    threshold,
    critical_threshold=180,
    moderate_threshold=100,
    low_threshold=45
):

    damage_map = np.zeros(
        (
            difference.shape[0],
            difference.shape[1],
            3
        ),
        dtype=np.uint8
    )

    safe = difference < low_threshold

    low = (
        (difference >= low_threshold)
        &
        (difference < moderate_threshold)
    )

    moderate = (
        (difference >= moderate_threshold)
        &
        (difference < critical_threshold)
    )

    critical = difference >= critical_threshold

    # BGR
    damage_map[safe] = [60, 180, 60]
    damage_map[low] = [0, 220, 255]
    damage_map[moderate] = [0, 150, 255]
    damage_map[critical] = [0, 0, 255]

    return damage_map


def damage_percentages(
    difference,
    critical_threshold=180,
    moderate_threshold=100,
    low_threshold=45
):

    total = difference.size

    if total == 0:
        return {
            "Critical": 0,
            "Moderate": 0,
            "Low": 0,
            "Safe": 0
        }

    critical = np.sum(
        difference >= critical_threshold
    )

    moderate = np.sum(
        (
            difference >= moderate_threshold
        )
        &
        (
            difference < critical_threshold
        )
    )

    low = np.sum(
        (
            difference >= low_threshold
        )
        &
        (
            difference < moderate_threshold
        )
    )

    safe = total - critical - moderate - low

    return {
        "Critical": round(
            critical / total * 100,
            2
        ),
        "Moderate": round(
            moderate / total * 100,
            2
        ),
        "Low": round(
            low / total * 100,
            2
        ),
        "Safe": round(
            safe / total * 100,
            2
        )
    }


def add_responder_markers(
    image,
    points
):

    output = image.copy()

    for x, y, label in points:

        x = int(x)
        y = int(y)

        cv2.circle(
            output,
            (x, y),
            12,
            (255, 255, 255),
            3
        )

        cv2.circle(
            output,
            (x, y),
            7,
            (0, 0, 255),
            -1
        )

        cv2.putText(
            output,
            label,
            (x + 15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2
        )

    return output


# ============================================================
# COORDINATE BASED COMPARISON
# ============================================================

def find_nearest_scan(
    history,
    latitude,
    longitude,
    radius_m=50,
    preferred_type=None
):

    candidates = []

    for record in history:

        rlat = record.get("latitude")
        rlon = record.get("longitude")

        if rlat is None or rlon is None:
            continue

        if preferred_type:
            if record.get("scan_type") != preferred_type:
                continue

        distance = haversine_m(
            latitude,
            longitude,
            rlat,
            rlon
        )

        candidates.append(
            (
                distance,
                record
            )
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    if not candidates:
        return None, None

    distance, record = candidates[0]

    if distance <= radius_m:
        return record, distance

    return None, distance


# ============================================================
# SURVIVOR POSITION ESTIMATION
# ============================================================

def estimate_detection_coordinate(
    drone_lat,
    drone_lon,
    altitude,
    heading,
    image_shape,
    bbox,
    hfov=60,
    vfov=45
):

    if altitude <= 0:
        return drone_lat, drone_lon

    image_h, image_w = image_shape[:2]

    x1, y1, x2, y2 = bbox

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    ground_width = (
        2
        * altitude
        * math.tan(
            math.radians(hfov / 2)
        )
    )

    ground_height = (
        2
        * altitude
        * math.tan(
            math.radians(vfov / 2)
        )
    )

    # Image coordinate → local ground coordinate
    east_img = (
        (cx / image_w) - 0.5
    ) * ground_width

    north_img = (
        0.5 - (cy / image_h)
    ) * ground_height

    heading_rad = math.radians(
        heading
    )

    east_world = (
        east_img * math.cos(heading_rad)
        +
        north_img * math.sin(heading_rad)
    )

    north_world = (
        -east_img * math.sin(heading_rad)
        +
        north_img * math.cos(heading_rad)
    )

    latitude = (
        drone_lat
        +
        north_world / 111320
    )

    longitude = (
        drone_lon
        +
        east_world
        /
        (
            111320
            *
            math.cos(
                math.radians(drone_lat)
            )
        )
    )

    return latitude, longitude


# ============================================================
# YOLO
# ============================================================

@st.cache_resource
def load_yolo_model(model_path):
    try:

        from ultralytics import YOLO

        return YOLO(
            model_path
        )

    except Exception as e:

        st.warning(
            f"YOLO model could not be loaded: {e}"
        )

        return None


def find_model_path():

    custom_candidates = [
        Path("models/disaster_best.pt"),
        Path("models/best.pt"),
        Path("best.pt")
    ]

    for path in custom_candidates:

        if path.exists():
            return str(path)

    return "yolo26n.pt"


def classify_detection(name):

    name = str(name).lower()

    if name in [
        "person",
        "survivor",
        "human"
    ]:
        return "Person / Possible Survivor"

    if name in [
        "car",
        "truck",
        "bus",
        "motorcycle"
    ]:
        return "Vehicle"

    if name in [
        "building",
        "house"
    ]:
        return "Building"

    return "Other"


def run_yolo(
    image,
    confidence=0.35
):

    model_path = find_model_path()

    model = load_yolo_model(
        model_path
    )

    if model is None:
        return image, []

    try:

        results = model.predict(
            image,
            conf=confidence,
            verbose=False
        )

        result = results[0]

        plotted = result.plot()

        detections = []

        names = result.names

        for box in result.boxes:

            cls_id = int(
                box.cls[0]
            )

            conf = float(
                box.conf[0]
            )

            if isinstance(
                names,
                dict
            ):
                class_name = names.get(
                    cls_id,
                    str(cls_id)
                )
            else:
                class_name = names[
                    cls_id
                ]

            coords = box.xyxy[0].tolist()

            detections.append({
                "class": class_name,
                "category": classify_detection(
                    class_name
                ),
                "confidence": round(
                    conf * 100,
                    2
                ),
                "x1": int(coords[0]),
                "y1": int(coords[1]),
                "x2": int(coords[2]),
                "y2": int(coords[3])
            })

        return plotted, detections

    except Exception as e:

        st.warning(
            f"YOLO detection failed: {e}"
        )

        return image, []


# ============================================================
# NOTIFICATIONS
# ============================================================

def load_notifications():
    return load_json(
        NOTIFICATION_FILE,
        []
    )


def save_notification(
    notification
):

    notifications = load_notifications()

    notifications.append(
        notification
    )

    notifications = notifications[-500:]

    save_json(
        NOTIFICATION_FILE,
        notifications
    )

    st.session_state.notifications = notifications


def create_survivor_alerts(
    detections,
    drone_lat,
    drone_lon,
    altitude,
    heading,
    image_shape,
    use_geometry=True,
    hfov=60,
    vfov=45,
    scan_id=""
):

    alerts = []

    for index, detection in enumerate(
        detections
    ):

        if (
            detection["category"]
            !=
            "Person / Possible Survivor"
        ):
            continue

        bbox = [
            detection["x1"],
            detection["y1"],
            detection["x2"],
            detection["y2"]
        ]

        if use_geometry:

            latitude, longitude = (
                estimate_detection_coordinate(
                    drone_lat,
                    drone_lon,
                    altitude,
                    heading,
                    image_shape,
                    bbox,
                    hfov,
                    vfov
                )
            )

        else:

            latitude = drone_lat
            longitude = drone_lon

        alert = {
            "notification_id":
                f"{scan_id}_person_{index+1}",

            "timestamp":
                now_string(),

            "type":
                "SURVIVOR_ALERT",

            "message":
                "Person / Possible Survivor detected",

            "latitude":
                round(latitude, 7),

            "longitude":
                round(longitude, 7),

            "altitude":
                round(float(altitude), 2),

            "heading":
                round(float(heading), 2),

            "confidence":
                detection["confidence"],

            "bbox":
                bbox,

            "scan_id":
                scan_id
        }

        alerts.append(
            alert
        )

    return alerts


# ============================================================
# SATELLITE ANALYSIS
# ============================================================

def analyze_satellite(
    pre_image,
    post_image,
    critical,
    moderate,
    low
):

    pre_image, post_image = resize_images(
        pre_image,
        post_image
    )

    aligned, success, matches = align_images(
        pre_image,
        post_image
    )

    if success:
        comparison_image = aligned
    else:
        comparison_image = cv2.resize(
            post_image,
            (
                pre_image.shape[1],
                pre_image.shape[0]
            )
        )

    difference, threshold = calculate_difference(
        pre_image,
        comparison_image
    )

    damage_map = create_damage_map(
        difference,
        threshold,
        critical,
        moderate,
        low
    )

    percentages = damage_percentages(
        difference,
        critical,
        moderate,
        low
    )

    return {
        "pre": pre_image,
        "post": comparison_image,
        "difference": difference,
        "threshold": threshold,
        "damage_map": damage_map,
        "percentages": percentages,
        "alignment_success": success,
        "matches": matches
    }


# ============================================================
# INITIAL DATA
# ============================================================

if st.session_state.drone_history is None:
    st.session_state.drone_history = (
        load_drone_history()
    )

if st.session_state.notifications is None:
    st.session_state.notifications = (
        load_notifications()
    )

if st.session_state.telemetry is None:
    st.session_state.telemetry = (
        load_json(
            TELEMETRY_FILE,
            []
        )
    )


# ============================================================
# SIDEBAR — CONTROL ZONE
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:22px;
            font-weight:900;
            margin-bottom:3px;
        ">
        🛰️ RESPONSE CONTROL
        </div>

        <div style="
            color:#8fa5be;
            font-size:11px;
            margin-bottom:20px;
        ">
        DISASTER INTELLIGENCE SYSTEM
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="dark-panel">

        <div class="dark-label">
        SYSTEM STATUS
        </div>

        <div class="dark-value">
        🟢 OPERATIONAL
        </div>

        <div style="
            color:#91a4bb;
            font-size:11px;
            margin-top:7px;
        ">
        AI + GPS + DAMAGE ANALYTICS
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### 🎛️ Analysis Controls")

    critical_threshold = st.slider(
        "Critical Threshold",
        100,
        255,
        180
    )

    moderate_threshold = st.slider(
        "Moderate Threshold",
        50,
        200,
        100
    )

    low_threshold = st.slider(
        "Low Threshold",
        10,
        100,
        45
    )

    st.markdown("---")

    st.markdown("### 🛰️ GPS Configuration")

    gps_mode = st.selectbox(
        "GPS Source",
        [
            "Manual",
            "Demo Simulator",
            "MAVLink"
        ]
    )

    if gps_mode == "Manual":

        manual_lat = st.number_input(
            "Latitude",
            value=float(
                st.session_state.gps["latitude"]
            ),
            format="%.7f"
        )

        manual_lon = st.number_input(
            "Longitude",
            value=float(
                st.session_state.gps["longitude"]
            ),
            format="%.7f"
        )

        manual_alt = st.number_input(
            "Altitude (m)",
            min_value=0.0,
            value=float(
                st.session_state.gps["altitude"]
            )
        )

        manual_heading = st.number_input(
            "Heading (°)",
            min_value=0.0,
            max_value=359.9,
            value=float(
                st.session_state.gps["heading"]
            )
        )

        st.session_state.gps = {
            "latitude": manual_lat,
            "longitude": manual_lon,
            "altitude": manual_alt,
            "heading": manual_heading,
            "source": "Manual"
        }

    elif gps_mode == "Demo Simulator":

        demo_lat = st.number_input(
            "Demo Start Latitude",
            value=23.2599,
            format="%.7f"
        )

        demo_lon = st.number_input(
            "Demo Start Longitude",
            value=77.4126,
            format="%.7f"
        )

        demo_step = st.slider(
            "Movement Step",
            1,
            20,
            5
        )

        demo_alt = st.number_input(
            "Demo Altitude (m)",
            5.0,
            500.0,
            50.0
        )

        demo_heading = st.number_input(
            "Demo Heading (°)",
            0.0,
            359.9,
            45.0
        )

        if st.button(
            "📡 Get Next GPS Position",
            use_container_width=True
        ):

            st.session_state.demo_step += 1

        step = (
            st.session_state.demo_step
            * demo_step
        )

        current_lat = (
            demo_lat
            +
            step * 0.00001
        )

        current_lon = (
            demo_lon
            +
            step * 0.000015
        )

        st.session_state.gps = {
            "latitude": current_lat,
            "longitude": current_lon,
            "altitude": demo_alt,
            "heading": demo_heading,
            "source": "Demo"
        }

    else:

        if not PYMAVLINK_AVAILABLE:

            st.warning(
                "Install pymavlink to use MAVLink GPS."
            )

        mav_endpoint = st.text_input(
            "MAVLink Endpoint",
            value="udp:127.0.0.1:14550"
        )

        mav_baud = st.number_input(
            "Baud Rate",
            9600,
            921600,
            57600
        )

        if st.button(
            "📡 Read Live GPS",
            use_container_width=True
        ):

            try:

                gps_data = read_mavlink_gps(
                    mav_endpoint,
                    int(mav_baud)
                )

                st.session_state.gps = (
                    gps_data
                )

                save_telemetry(
                    gps_data
                )

                st.success(
                    "GPS telemetry updated."
                )

            except Exception as e:

                st.error(
                    f"GPS read failed: {e}"
                )

    st.markdown("---")

    st.markdown("### 🤖 AI Detection")

    yolo_confidence = st.slider(
        "YOLO Confidence",
        0.10,
        0.95,
        0.35
    )

    estimate_survivor_position = st.checkbox(
        "Estimate Survivor GPS from Image",
        value=True
    )

    hfov = st.number_input(
        "Camera Horizontal FOV (°)",
        20.0,
        120.0,
        60.0
    )

    vfov = st.number_input(
        "Camera Vertical FOV (°)",
        20.0,
        100.0,
        45.0
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            color:#6f8198;
            font-size:10px;
            line-height:1.5;
        ">
        Prototype system. Person detection is treated
        as a possible-survivor indicator and requires
        responder verification.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# HEADER
# ============================================================

gps = st.session_state.gps

notifications = (
    st.session_state.notifications
    or []
)

unread_alerts = len(
    notifications
)

st.markdown(
    f"""
    <div class="command-header">

        <div style="
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:20px;
            flex-wrap:wrap;
        ">

            <div>
                <div class="header-title">
                    🛰️ Disaster Response Command Center
                </div>

                <div class="header-subtitle">
                    AI-powered damage intelligence •
                    Drone GPS tracking •
                    Possible survivor detection
                </div>
            </div>

            <div>
                <span class="live-pill">
                    <span class="live-dot"></span>
                    SYSTEM ONLINE
                </span>
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# KPI ROW
# ============================================================

history = st.session_state.drone_history or []

scan_count = len(history)

survivor_count = len(
    notifications
)

lat = gps["latitude"]
lon = gps["longitude"]
alt = gps["altitude"]

k1, k2, k3, k4 = st.columns(4)

with k1:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
            🛰️ Drone Status
            </div>

            <div class="kpi-value">
            LIVE
            </div>

            <div class="kpi-small">
            GPS source: {gps["source"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k2:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
            📍 Current Latitude
            </div>

            <div class="kpi-value">
            {lat:.5f}
            </div>

            <div class="kpi-small">
            Longitude {lon:.5f}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k3:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
            🔔 Survivor Alerts
            </div>

            <div class="kpi-value">
            {survivor_count}
            </div>

            <div class="kpi-small">
            Persistent notification history
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with k4:

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">
            📊 Drone Scans
            </div>

            <div class="kpi-value">
            {scan_count}
            </div>

            <div class="kpi-small">
            GPS-indexed scan history
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🛰️ Drone Command",
        "🗺️ Satellite Analysis",
        "📷 Camera AI",
        "🔔 Alert History"
    ]
)


# ============================================================
# TAB 1 — DRONE COMMAND
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">🛰️ Live Drone Mission</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-caption">
        GPS telemetry controls the mission map and determines
        which pre/post drone scans are compared.
        </div>
        """,
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.65, 1]
    )

    with left:

        st.markdown(
            """
            <div class="info-card">
            <b>📍 LIVE GPS TRACK</b>
            <br>
            <span style="color:#718197;font-size:12px;">
            Current drone position and previous scan locations
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        gps_rows = []

        for record in history:

            if (
                record.get("latitude")
                is not None
                and
                record.get("longitude")
                is not None
            ):

                gps_rows.append({
                    "latitude":
                        record["latitude"],

                    "longitude":
                        record["longitude"]
                })

        if gps_rows:

            gps_df = pd.DataFrame(
                gps_rows
            )

            current_df = pd.DataFrame([
                {
                    "latitude": lat,
                    "longitude": lon
                }
            ])

            st.map(
                pd.concat(
                    [
                        gps_df,
                        current_df
                    ],
                    ignore_index=True
                ),
                height=430
            )

        else:

            st.map(
                pd.DataFrame([
                    {
                        "latitude": lat,
                        "longitude": lon
                    }
                ]),
                height=430
            )

    with right:

        st.markdown(
            f"""
            <div class="dark-panel">

                <div class="dark-label">
                CURRENT DRONE POSITION
                </div>

                <div class="dark-value">
                📍 {lat:.7f}
                </div>

                <div style="
                    color:#8fa5be;
                    font-size:13px;
                    margin-top:3px;
                ">
                {lon:.7f}
                </div>

                <br>

                <div class="dark-label">
                ALTITUDE
                </div>

                <div class="dark-value">
                {alt:.1f} m
                </div>

                <br>

                <div class="dark-label">
                HEADING
                </div>

                <div class="dark-value">
                {gps["heading"]:.1f}°
                </div>

                <br>

                <div class="dark-label">
                TELEMETRY SOURCE
                </div>

                <div class="dark-value">
                {gps["source"]}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        st.markdown(
            """
            <div class="info-card">
            <b>🎯 Coordinate Matching</b>
            <br>
            <span style="
                color:#718197;
                font-size:12px;
            ">
            New drone images are compared with the nearest
            historical scan inside the GPS radius.
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )

        coordinate_radius = st.number_input(
            "Maximum comparison radius (meters)",
            min_value=5,
            max_value=1000,
            value=50
        )

        scan_type = st.selectbox(
            "Current Scan Type",
            [
                "Monitoring",
                "Pre-Disaster",
                "Post-Disaster"
            ]
        )


    # --------------------------------------------------------
    # SURVIVOR ALERTS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🚨 Survivor Detection Center</div>',
        unsafe_allow_html=True
    )

    recent_notifications = (
        notifications[-3:][::-1]
    )

    if recent_notifications:

        for alert in recent_notifications:

            st.markdown(
                f"""
                <div class="survivor-alert">

                    <div class="alert-title">
                    🚨 SURVIVOR / PERSON DETECTED
                    </div>

                    <div class="alert-text">
                    Possible survivor detected by AI.
                    Responder verification required.
                    </div>

                    <div style="
                        margin-top:12px;
                        display:flex;
                        gap:10px;
                        flex-wrap:wrap;
                    ">

                    <span class="badge badge-red">
                    CONFIDENCE {alert["confidence"]}%
                    </span>

                    <span class="badge badge-blue">
                    LAT {alert["latitude"]:.6f}
                    </span>

                    <span class="badge badge-blue">
                    LON {alert["longitude"]:.6f}
                    </span>

                    <span class="badge badge-orange">
                    {alert["timestamp"]}
                    </span>

                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown("")

    else:

        st.markdown(
            """
            <div class="info-card">
            🟢 No survivor alerts yet.
            <br>
            <span style="
                color:#718197;
                font-size:12px;
            ">
            Upload a drone scan to run AI detection.
            </span>
            </div>
            """,
            unsafe_allow_html=True
        )


    # --------------------------------------------------------
    # DRONE UPLOAD
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">📸 Drone Scan</div>',
        unsafe_allow_html=True
    )

    drone_file = st.file_uploader(
        "Upload current drone image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="drone_upload"
    )

    if drone_file:

        drone_image = uploaded_to_cv(
            drone_file
        )

        if drone_image is not None:

            col1, col2 = st.columns(
                [1, 1]
            )

            with col1:

                st.image(
                    cv_to_rgb(
                        drone_image
                    ),
                    caption="Current Drone Frame",
                    width="stretch"
                )

            with col2:

                st.markdown(
                    """
                    <div class="info-card">

                    <b>AI Scan Configuration</b>

                    <br><br>

                    <span style="
                        color:#718197;
                        font-size:12px;
                    ">
                    The system will:
                    <br>1. Match the scan using GPS
                    <br>2. Compare the correct geographic area
                    <br>3. Run YOLO person detection
                    <br>4. Estimate survivor coordinates
                    <br>5. Create dashboard alerts
                    <br>6. Save the scan + alert history
                    </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            if st.button(
                "🚀 PROCESS DRONE SCAN",
                type="primary",
                use_container_width=True
            ):

                current_hash = image_hash(
                    drone_file
                )

                if (
                    current_hash
                    ==
                    st.session_state.drone_last_hash
                ):

                    st.info(
                        "This scan has already been processed."
                    )

                else:

                    scan_id = next_scan_id()

                    image_path = save_drone_image(
                        drone_image,
                        scan_id
                    )

                    # ------------------------------------------------
                    # FIND GPS-MATCHED PREVIOUS SCAN
                    # ------------------------------------------------

                    previous_record = None
                    coordinate_distance = None

                    if scan_type == "Post-Disaster":

                        previous_record, coordinate_distance = (
                            find_nearest_scan(
                                history,
                                lat,
                                lon,
                                coordinate_radius,
                                preferred_type="Pre-Disaster"
                            )
                        )

                    else:

                        previous_record, coordinate_distance = (
                            find_nearest_scan(
                                history,
                                lat,
                                lon,
                                coordinate_radius
                            )
                        )

                    # ------------------------------------------------
                    # DAMAGE COMPARISON
                    # ------------------------------------------------

                    comparison_available = False
                    alignment_success = False
                    matches = 0
                    statistics = None
                    damage_map = None
                    previous_image = None

                    if previous_record:

                        previous_path = previous_record.get(
                            "image_path"
                        )

                        if previous_path and Path(
                            previous_path
                        ).exists():

                            previous_image = cv2.imread(
                                previous_path
                            )

                            if previous_image is not None:

                                aligned, alignment_success, matches = (
                                    align_images(
                                        previous_image,
                                        drone_image
                                    )
                                )

                                if alignment_success:

                                    compare_image = aligned

                                else:

                                    compare_image = cv2.resize(
                                        drone_image,
                                        (
                                            previous_image.shape[1],
                                            previous_image.shape[0]
                                        )
                                    )

                                difference, threshold = (
                                    calculate_difference(
                                        previous_image,
                                        compare_image
                                    )
                                )

                                damage_map = (
                                    create_damage_map(
                                        difference,
                                        threshold,
                                        critical_threshold,
                                        moderate_threshold,
                                        low_threshold
                                    )
                                )

                                statistics = (
                                    damage_percentages(
                                        difference,
                                        critical_threshold,
                                        moderate_threshold,
                                        low_threshold
                                    )
                                )

                                comparison_available = True

                    # ------------------------------------------------
                    # AI DETECTION
                    # ------------------------------------------------

                    plotted_image, detections = run_yolo(
                        drone_image,
                        yolo_confidence
                    )

                    # ------------------------------------------------
                    # SURVIVOR ALERTS
                    # ------------------------------------------------

                    survivor_alerts = (
                        create_survivor_alerts(
                            detections,
                            lat,
                            lon,
                            alt,
                            gps["heading"],
                            drone_image.shape,
                            estimate_survivor_position,
                            hfov,
                            vfov,
                            scan_id
                        )
                    )

                    # Save notifications
                    for alert in survivor_alerts:

                        save_notification(
                            alert
                        )

                    # ------------------------------------------------
                    # SAVE RESULT IMAGE
                    # ------------------------------------------------

                    detection_path = (
                        save_result_image(
                            plotted_image,
                            scan_id,
                            "ai"
                        )
                    )

                    damage_path = None

                    if damage_map is not None:

                        damage_path = (
                            save_result_image(
                                damage_map,
                                scan_id,
                                "damage"
                            )
                        )

                    # ------------------------------------------------
                    # RECORD
                    # ------------------------------------------------

                    record = {
                        "scan_id": scan_id,
                        "timestamp": now_string(),
                        "image_path": image_path,
                        "ai_image_path": detection_path,
                        "damage_map_path": damage_path,

                        "latitude": round(
                            float(lat),
                            7
                        ),

                        "longitude": round(
                            float(lon),
                            7
                        ),

                        "altitude": round(
                            float(alt),
                            2
                        ),

                        "heading": round(
                            float(gps["heading"]),
                            2
                        ),

                        "gps_source":
                            gps["source"],

                        "scan_type":
                            scan_type,

                        "compared_with":
                            previous_record["scan_id"]
                            if previous_record
                            else None,

                        "coordinate_distance_m":
                            round(
                                coordinate_distance,
                                2
                            )
                            if coordinate_distance
                            is not None
                            else None,

                        "coordinate_match":
                            previous_record is not None,

                        "comparison_available":
                            comparison_available,

                        "alignment_success":
                            alignment_success,

                        "feature_matches":
                            matches,

                        "statistics":
                            statistics,

                        "detections":
                            detections,

                        "survivor_alerts":
                            survivor_alerts,

                        "image_hash":
                            current_hash
                    }

                    save_drone_record(
                        record
                    )

                    st.session_state.drone_history = (
                        load_drone_history()
                    )

                    st.session_state.drone_last_hash = (
                        current_hash
                    )

                    st.session_state.drone_current_analysis = {
                        "record": record,
                        "current": drone_image,
                        "previous": previous_image,
                        "damage_map": damage_map,
                        "ai_image": plotted_image
                    }

                    st.success(
                        f"{scan_id} processed successfully."
                    )

                    if survivor_alerts:

                        st.error(
                            f"🚨 {len(survivor_alerts)} "
                            f"possible survivor/person alert(s) generated!"
                        )

                    else:

                        st.success(
                            "🟢 No person detected in this scan."
                        )

                    if comparison_available:

                        st.info(
                            f"📍 GPS matched with "
                            f"{previous_record['scan_id']} "
                            f"at {coordinate_distance:.2f} m."
                        )

                    else:

                        st.warning(
                            "📍 No previous scan found within "
                            f"{coordinate_radius} m. "
                            "Damage comparison skipped."
                        )


    # --------------------------------------------------------
    # LATEST RESULT
    # --------------------------------------------------------

    analysis = (
        st.session_state.drone_current_analysis
    )

    if analysis:

        st.markdown(
            '<div class="section-title">📊 Latest Mission Result</div>',
            unsafe_allow_html=True
        )

        result_record = analysis["record"]

        c1, c2, c3 = st.columns(
            [1, 1, 1]
        )

        with c1:

            st.image(
                cv_to_rgb(
                    analysis["current"]
                ),
                caption="Current Scan",
                width="stretch"
            )

        with c2:

            if analysis["previous"] is not None:

                st.image(
                    cv_to_rgb(
                        analysis["previous"]
                    ),
                    caption="GPS-Matched Previous Scan",
                    width="stretch"
                )

            else:

                st.markdown(
                    """
                    <div class="info-card">
                    <b>NO GPS-MATCHED PREVIOUS IMAGE</b>
                    <br><br>
                    <span style="
                        color:#718197;
                        font-size:12px;
                    ">
                    A comparison will become available
                    when a scan from the same geographic
                    area is available.
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with c3:

            if analysis["damage_map"] is not None:

                st.image(
                    cv_to_rgb(
                        analysis["damage_map"]
                    ),
                    caption="Damage Map",
                    width="stretch"
                )

            else:

                st.markdown(
                    """
                    <div class="info-card">
                    <b>GPS BASELINE</b>
                    <br><br>
                    <span style="
                        color:#718197;
                        font-size:12px;
                    ">
                    No damage comparison available
                    for this location yet.
                    </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ----------------------------------------------------
        # STATS
        # ----------------------------------------------------

        if result_record.get(
            "statistics"
        ):

            st.markdown(
                '<div class="section-title">📈 Damage Distribution</div>',
                unsafe_allow_html=True
            )

            stats = result_record[
                "statistics"
            ]

            a, b, c, d = st.columns(4)

            values = [
                ("Critical", stats["Critical"]),
                ("Moderate", stats["Moderate"]),
                ("Low", stats["Low"]),
                ("Safe", stats["Safe"])
            ]

            for column, (
                label,
                value
            ) in zip(
                [a, b, c, d],
                values
            ):

                with column:

                    st.markdown(
                        f"""
                        <div class="kpi-card">
                            <div class="kpi-label">
                            {label}
                            </div>
                            <div class="kpi-value">
                            {value}%
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            chart_df = pd.DataFrame({
                "Damage Level": list(
                    stats.keys()
                ),
                "Area %": list(
                    stats.values()
                )
            })

            st.bar_chart(
                chart_df.set_index(
                    "Damage Level"
                )
            )


        # ----------------------------------------------------
        # AI DETECTIONS
        # ----------------------------------------------------

        detections = result_record.get(
            "detections",
            []
        )

        if detections:

            st.markdown(
                '<div class="section-title">🤖 AI Detections</div>',
                unsafe_allow_html=True
            )

            detection_df = pd.DataFrame(
                detections
            )

            st.dataframe(
                detection_df,
                width="stretch",
                hide_index=True
            )


    # --------------------------------------------------------
    # DRONE HISTORY
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">🗂️ Coordinate-Based Scan History</div>',
        unsafe_allow_html=True
    )

    history = (
        st.session_state.drone_history
        or []
    )

    if history:

        rows = []

        for record in reversed(
            history
        ):

            rows.append({
                "Scan":
                    record.get(
                        "scan_id"
                    ),

                "Time":
                    record.get(
                        "timestamp"
                    ),

                "Type":
                    record.get(
                        "scan_type"
                    ),

                "Latitude":
                    record.get(
                        "latitude"
                    ),

                "Longitude":
                    record.get(
                        "longitude"
                    ),

                "Compared With":
                    record.get(
                        "compared_with"
                    )
                    or "—",

                "GPS Distance (m)":
                    record.get(
                        "coordinate_distance_m"
                    )
                    if record.get(
                        "coordinate_distance_m"
                    ) is not None
                    else "—",

                "GPS Match":
                    "YES"
                    if record.get(
                        "coordinate_match"
                    )
                    else "NO",

                "Survivors":
                    len(
                        record.get(
                            "survivor_alerts",
                            []
                        )
                    )
            })

        history_df = pd.DataFrame(
            rows
        )

        st.dataframe(
            history_df,
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No drone scans recorded yet."
        )


# ============================================================
# TAB 2 — SATELLITE ANALYSIS
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">🗺️ Pre-Disaster vs Post-Disaster Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-caption">
        Upload two images of the same geographic area.
        Optional GPS coordinates can be used to prevent
        comparison of different locations.
        </div>
        """,
        unsafe_allow_html=True
    )

    use_satellite_gps = st.checkbox(
        "📍 Enable coordinate validation",
        value=False
    )

    if use_satellite_gps:

        g1, g2 = st.columns(2)

        with g1:

            pre_lat = st.number_input(
                "Pre-disaster Latitude",
                value=23.2599,
                format="%.7f"
            )

            pre_lon = st.number_input(
                "Pre-disaster Longitude",
                value=77.4126,
                format="%.7f"
            )

        with g2:

            post_lat = st.number_input(
                "Post-disaster Latitude",
                value=23.2599,
                format="%.7f"
            )

            post_lon = st.number_input(
                "Post-disaster Longitude",
                value=77.4126,
                format="%.7f"
            )

        satellite_radius = st.number_input(
            "Maximum allowed GPS difference (m)",
            5,
            5000,
            100
        )

    s1, s2 = st.columns(2)

    with s1:

        pre_file = st.file_uploader(
            "Upload PRE-DISASTER image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="pre_upload"
        )

    with s2:

        post_file = st.file_uploader(
            "Upload POST-DISASTER image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ],
            key="post_upload"
        )

    if pre_file and post_file:

        pre_img = uploaded_to_cv(
            pre_file
        )

        post_img = uploaded_to_cv(
            post_file
        )

        if st.button(
            "🔎 ANALYZE DAMAGE",
            type="primary",
            use_container_width=True
        ):

            coordinate_distance = None

            if use_satellite_gps:

                coordinate_distance = haversine_m(
                    pre_lat,
                    pre_lon,
                    post_lat,
                    post_lon
                )

                if (
                    coordinate_distance
                    >
                    satellite_radius
                ):

                    st.error(
                        f"❌ Images are {coordinate_distance:.2f} m apart. "
                        f"Allowed radius is {satellite_radius} m. "
                        "Comparison stopped."
                    )

                    st.stop()

                st.success(
                    f"📍 GPS validated: "
                    f"{coordinate_distance:.2f} m apart."
                )

            result = analyze_satellite(
                pre_img,
                post_img,
                critical_threshold,
                moderate_threshold,
                low_threshold
            )

            st.session_state.satellite_result = (
                result
            )

    result = (
        st.session_state.satellite_result
    )

    if result:

        st.markdown(
            '<div class="section-title">🛰️ Analysis Output</div>',
            unsafe_allow_html=True
        )

        a, b, c = st.columns(3)

        with a:

            st.image(
                cv_to_rgb(
                    result["pre"]
                ),
                caption="Pre-Disaster",
                width="stretch"
            )

        with b:

            st.image(
                cv_to_rgb(
                    result["post"]
                ),
                caption="Post-Disaster",
                width="stretch"
            )

        with c:

            st.image(
                cv_to_rgb(
                    result["damage_map"]
                ),
                caption="Damage Classification",
                width="stretch"
            )

        if result["alignment_success"]:

            st.success(
                f"Image alignment successful • "
                f"{result['matches']} feature matches"
            )

        else:

            st.warning(
                "Automatic image alignment was not reliable."
            )

        stats = result[
            "percentages"
        ]

        st.markdown(
            '<div class="section-title">📊 Damage Distribution</div>',
            unsafe_allow_html=True
        )

        p1, p2, p3, p4 = st.columns(4)

        for col, label in zip(
            [p1, p2, p3, p4],
            [
                "Critical",
                "Moderate",
                "Low",
                "Safe"
            ]
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="kpi-card">
                        <div class="kpi-label">
                        {label}
                        </div>

                        <div class="kpi-value">
                        {stats[label]}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        chart_df = pd.DataFrame({
            "Damage Level":
                list(stats.keys()),

            "Area %":
                list(stats.values())
        })

        st.bar_chart(
            chart_df.set_index(
                "Damage Level"
            )
        )


# ============================================================
# TAB 3 — CAMERA AI
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">📷 Camera AI Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-caption">
        Capture an image using your camera and run AI detection.
        A detected person is treated as a possible-survivor signal.
        </div>
        """,
        unsafe_allow_html=True
    )

    camera_image = st.camera_input(
        "Capture disaster scene"
    )

    if camera_image:

        image = uploaded_to_cv(
            camera_image
        )

        if image is not None:

            st.image(
                cv_to_rgb(image),
                caption="Captured Scene",
                width="stretch"
            )

            if st.button(
                "🤖 RUN AI DETECTION",
                type="primary",
                use_container_width=True
            ):

                output, detections = run_yolo(
                    image,
                    yolo_confidence
                )

                st.image(
                    cv_to_rgb(output),
                    caption="AI Detection Result",
                    width="stretch"
                )

                if detections:

                    df = pd.DataFrame(
                        detections
                    )

                    st.dataframe(
                        df,
                        width="stretch",
                        hide_index=True
                    )

                    person_count = sum(
                        1
                        for d in detections
                        if d["category"]
                        ==
                        "Person / Possible Survivor"
                    )

                    if person_count:

                        st.warning(
                            f"🚨 {person_count} "
                            f"possible survivor/person "
                            f"detection(s) found."
                        )

                    else:

                        st.success(
                            "No person detected."
                        )

                else:

                    st.info(
                        "No objects detected."
                    )


# ============================================================
# TAB 4 — NOTIFICATION HISTORY
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">🔔 Mission Alert History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="section-caption">
        Persistent alerts generated by the AI drone scans.
        Each alert stores its estimated GPS position.
        </div>
        """,
        unsafe_allow_html=True
    )

    notifications = (
        load_notifications()
    )

    st.session_state.notifications = (
        notifications
    )

    if notifications:

        latest = notifications[-1]

        st.markdown(
            f"""
            <div class="survivor-alert">

                <div class="alert-title">
                🚨 LATEST ALERT
                </div>

                <div class="alert-text">
                Possible survivor/person detected
                </div>

                <div style="
                    margin-top:12px;
                    display:flex;
                    flex-wrap:wrap;
                    gap:8px;
                ">

                <span class="badge badge-red">
                {latest["confidence"]}% CONFIDENCE
                </span>

                <span class="badge badge-blue">
                📍 {latest["latitude"]:.6f},
                {latest["longitude"]:.6f}
                </span>

                <span class="badge badge-orange">
                {latest["timestamp"]}
                </span>

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("")

        notification_rows = []

        for n in reversed(
            notifications
        ):

            notification_rows.append({
                "Time":
                    n.get(
                        "timestamp"
                    ),

                "Alert":
                    n.get(
                        "message"
                    ),

                "Latitude":
                    n.get(
                        "latitude"
                    ),

                "Longitude":
                    n.get(
                        "longitude"
                    ),

                "Altitude (m)":
                    n.get(
                        "altitude"
                    ),

                "Confidence":
                    f'{n.get("confidence", 0)}%',

                "Scan":
                    n.get(
                        "scan_id"
                    )
            })

        notification_df = pd.DataFrame(
            notification_rows
        )

        st.dataframe(
            notification_df,
            width="stretch",
            hide_index=True
        )

        st.markdown(
            '<div class="section-title">📍 Alert Locations</div>',
            unsafe_allow_html=True
        )

        alert_map_df = pd.DataFrame([
            {
                "latitude":
                    n["latitude"],

                "longitude":
                    n["longitude"]
            }
            for n in notifications
            if n.get("latitude") is not None
            and n.get("longitude") is not None
        ])

        if not alert_map_df.empty:

            st.map(
                alert_map_df,
                height=450
            )

    else:

        st.info(
            "🔔 No notifications have been generated yet."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:30px;
        padding:15px 0;
        border-top:1px solid #dfe7f0;
        text-align:center;
        color:#8493a6;
        font-size:11px;
    ">
        DISASTER RESPONSE COMMAND CENTER
        &nbsp;•&nbsp;
        AI DAMAGE INTELLIGENCE
        &nbsp;•&nbsp;
        GPS RESPONSE TRACKING
        <br>
        Prototype for research & demonstration.
        AI person detection requires human verification.
    </div>
    """,
    unsafe_allow_html=True
)