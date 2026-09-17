import streamlit as st
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import hashlib
import json
import math

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
# DARK COMMAND CENTER UI
# ============================================================

st.markdown("""
<style>

/* =========================================================
   GLOBAL
   ========================================================= */

.stApp {
    background: #050b14;
    color: #e8eef7;
}

.main {
    background: #050b14;
}

.block-container {
    max-width: 1500px;
    padding-top: 1.2rem;
    padding-bottom: 2rem;
}


/* =========================================================
   SIDEBAR
   ========================================================= */

[data-testid="stSidebar"] {
    background: #07101d !important;
    border-right: 1px solid #1a2b40;
}

[data-testid="stSidebar"] > div {
    background: #07101d !important;
}

[data-testid="stSidebar"] * {
    color: #e8eef7 !important;
}

[data-testid="stSidebar"] label {
    color: #9aabc0 !important;
}

[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea {
    background: #0d1928 !important;
    color: #ffffff !important;
    border: 1px solid #263b54 !important;
}

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: #0d1928 !important;
    border-color: #263b54 !important;
    color: #ffffff !important;
}

[data-testid="stSidebar"] [data-baseweb="slider"] {
    color: #5aa9ff !important;
}


/* =========================================================
   SIDEBAR BRAND
   ========================================================= */

.sidebar-brand {
    padding: 6px 4px 22px 4px;
}

.sidebar-brand-title {
    font-size: 21px;
    font-weight: 900;
    letter-spacing: .5px;
    color: #ffffff;
}

.sidebar-brand-subtitle {
    margin-top: 4px;
    color: #6f8299;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1.2px;
}


/* =========================================================
   SIDEBAR PANELS
   ========================================================= */

.control-panel {
    background: #0b1726;
    border: 1px solid #1b3048;
    border-radius: 14px;
    padding: 15px;
    margin-bottom: 15px;
}

.panel-subtitle {
    color: #70859d;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 5px;
}

.status-online {
    color: #4ade80;
    font-size: 18px;
    font-weight: 900;
    margin-bottom: 4px;
}


/* =========================================================
   MAIN HEADER
   ========================================================= */

.command-header {
    background:
        linear-gradient(
            135deg,
            #0b1726,
            #0d1c2e
        );

    border: 1px solid #1c324b;
    border-radius: 20px;
    padding: 24px 28px;
    margin-bottom: 18px;

    box-shadow:
        0 12px 40px rgba(0,0,0,.28);
}

.header-title {
    color: #ffffff;
    font-size: 29px;
    font-weight: 900;
    letter-spacing: -.7px;
}

.header-subtitle {
    color: #8295aa;
    font-size: 13px;
    margin-top: 5px;
}

.header-meta {
    color: #5f748d;
    font-size: 10px;
    margin-top: 12px;
    letter-spacing: .5px;
}

.live-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;

    background: #092319;
    border: 1px solid #174c32;

    color: #4ade80;

    padding: 8px 13px;
    border-radius: 999px;

    font-size: 11px;
    font-weight: 800;
}

.live-dot {
    width: 8px;
    height: 8px;
    background: #4ade80;
    border-radius: 50%;
    box-shadow: 0 0 10px #4ade80;
}


/* =========================================================
   KPI
   ========================================================= */

.kpi-card {
    background: #0b1726;
    border: 1px solid #1b3048;
    border-radius: 16px;

    padding: 17px;
    min-height: 105px;

    box-shadow:
        0 7px 25px rgba(0,0,0,.22);

    transition: .2s ease;
}

.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #315271;
}

.kpi-label {
    color: #71869e;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: .7px;
    text-transform: uppercase;
}

.kpi-value {
    color: #ffffff;
    font-size: 25px;
    font-weight: 900;
    margin-top: 8px;
}

.kpi-small {
    color: #61758b;
    font-size: 10px;
    margin-top: 3px;
}


/* =========================================================
   SECTION
   ========================================================= */

.section-title {
    color: #ffffff;
    font-size: 19px;
    font-weight: 850;

    margin-top: 22px;
    margin-bottom: 5px;
}

.section-caption {
    color: #70849a;
    font-size: 11px;
    margin-bottom: 14px;
}


/* =========================================================
   CARDS
   ========================================================= */

.info-card {
    background: #0b1726;
    border: 1px solid #1b3048;
    border-radius: 15px;
    padding: 17px;

    color: #e8eef7;

    box-shadow:
        0 6px 22px rgba(0,0,0,.18);
}

.info-card-title {
    color: #ffffff;
    font-size: 13px;
    font-weight: 800;
}

.info-card-text {
    color: #74889e;
    font-size: 11px;
    line-height: 1.6;
    margin-top: 6px;
}


/* =========================================================
   DARK MISSION PANEL
   ========================================================= */

.mission-panel {
    background:
        linear-gradient(
            145deg,
            #0d1d30,
            #091522
        );

    border: 1px solid #1d3853;
    border-radius: 17px;

    padding: 20px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.02),
        0 10px 30px rgba(0,0,0,.25);
}

.mission-label {
    color: #6d829a;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}

.mission-value {
    color: #ffffff;
    font-size: 21px;
    font-weight: 900;
    margin-top: 4px;
}

.mission-coord {
    color: #58a6ff;
    font-size: 13px;
    margin-top: 2px;
}


/* =========================================================
   ALERT
   ========================================================= */

.survivor-alert {
    background:
        linear-gradient(
            135deg,
            #211015,
            #160d12
        );

    border: 1px solid #6e2730;
    border-left: 5px solid #ef4444;

    border-radius: 15px;

    padding: 17px 19px;

    margin-bottom: 10px;

    box-shadow:
        0 8px 25px rgba(239,68,68,.08);
}

.alert-title {
    color: #ff6262;
    font-size: 16px;
    font-weight: 900;
}

.alert-text {
    color: #b77b82;
    font-size: 11px;
    margin-top: 4px;
}

.alert-data {
    color: #dce4ed;
    font-size: 11px;
    margin-top: 10px;
}


/* =========================================================
   BADGES
   ========================================================= */

.badge {
    display: inline-block;
    padding: 5px 8px;

    border-radius: 6px;

    font-size: 9px;
    font-weight: 800;

    margin-right: 4px;
}

.badge-red {
    background: #321216;
    color: #ff6b6b;
    border: 1px solid #64262e;
}

.badge-blue {
    background: #0c2035;
    color: #58a6ff;
    border: 1px solid #20486b;
}

.badge-orange {
    background: #2b1d0c;
    color: #f59e0b;
    border: 1px solid #59401a;
}

.badge-green {
    background: #0b2518;
    color: #4ade80;
    border: 1px solid #1d5835;
}


/* =========================================================
   STREAMLIT ELEMENTS
   ========================================================= */

.stButton > button {
    background: #102338;
    color: #eaf2fb;

    border: 1px solid #28445f;
    border-radius: 9px;

    font-weight: 800;

    min-height: 40px;
}

.stButton > button:hover {
    background: #15304a;
    border-color: #3d6b92;
    color: #ffffff;
}

.stButton > button[kind="primary"] {
    background: #125a91;
    border-color: #237bb8;
    color: white;
}

.stButton > button[kind="primary"]:hover {
    background: #1670ad;
}


/* =========================================================
   TABS
   ========================================================= */

.stTabs [data-baseweb="tab-list"] {
    gap: 5px;
    background: #081321;
    padding: 6px;
    border: 1px solid #192d43;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    color: #72879e;
    font-size: 11px;
    font-weight: 800;
}

.stTabs [aria-selected="true"] {
    background: #10283e !important;
    color: #ffffff !important;
    border-radius: 8px;
}


/* =========================================================
   FILE UPLOADER
   ========================================================= */

[data-testid="stFileUploader"] {
    background: #0b1726;
    border: 1px solid #1b3048;
    border-radius: 13px;
    padding: 4px;
}

[data-testid="stFileUploader"] section {
    background: #0b1726 !important;
}


/* =========================================================
   DATAFRAME
   ========================================================= */

[data-testid="stDataFrame"] {
    border: 1px solid #1b3048;
    border-radius: 10px;
}


/* =========================================================
   MAP
   ========================================================= */

[data-testid="stMap"] {
    border: 1px solid #1b3048;
    border-radius: 14px;
    overflow: hidden;
}


/* =========================================================
   IMAGE
   ========================================================= */

[data-testid="stImage"] {
    border-radius: 12px;
    overflow: hidden;
}


/* =========================================================
   ALERT BOXES
   ========================================================= */

.stAlert {
    border-radius: 10px;
}


/* =========================================================
   FOOTER
   ========================================================= */

.footer {
    margin-top: 35px;
    padding: 18px;

    border-top: 1px solid #192c41;

    text-align: center;

    color: #566b81;
    font-size: 9px;
    letter-spacing: .5px;
}


/* =========================================================
   MOBILE
   ========================================================= */

@media(max-width: 900px) {

    .header-title {
        font-size: 22px;
    }

    .command-header {
        padding: 18px;
    }

}

</style>
""", unsafe_allow_html=True)


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

if "satellite_result" not in st.session_state:
    st.session_state.satellite_result = None

if "drone_history" not in st.session_state:
    st.session_state.drone_history = None

if "drone_last_hash" not in st.session_state:
    st.session_state.drone_last_hash = ""

if "drone_current_analysis" not in st.session_state:
    st.session_state.drone_current_analysis = None

if "notifications" not in st.session_state:
    st.session_state.notifications = None

if "telemetry" not in st.session_state:
    st.session_state.telemetry = None

if "demo_step" not in st.session_state:
    st.session_state.demo_step = 0

if "gps" not in st.session_state:
    st.session_state.gps = {
        "latitude": 23.2599,
        "longitude": 77.4126,
        "altitude": 50.0,
        "heading": 0.0,
        "source": "Demo"
    }


# ============================================================
# BASIC FUNCTIONS
# ============================================================

def now_string():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
        json.dump(data, f, indent=2)


def uploaded_to_cv(uploaded_file):

    if uploaded_file is None:
        return None

    data = uploaded_file.getvalue()

    if not data:
        return None

    array = np.frombuffer(data, np.uint8)

    return cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )


def cv_to_rgb(image):

    if image is None:
        return None

    return cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


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

    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except Exception:
        return float("inf")

    radius = 6371000

    p1 = math.radians(lat1)
    p2 = math.radians(lat2)

    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)

    a = (
        math.sin(dp / 2) ** 2
        +
        math.cos(p1)
        * math.cos(p2)
        * math.sin(dl / 2) ** 2
    )

    return radius * 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )


def save_telemetry(point):

    data = load_json(
        TELEMETRY_FILE,
        []
    )

    data.append(point)

    save_json(
        TELEMETRY_FILE,
        data[-500:]
    )


def read_mavlink_gps(endpoint, baud):

    if not PYMAVLINK_AVAILABLE:
        raise RuntimeError(
            "pymavlink is not installed."
        )

    connection = mavutil.mavlink_connection(
        endpoint,
        baud=baud
    )

    try:

        message = connection.recv_match(
            type="GLOBAL_POSITION_INT",
            blocking=True,
            timeout=5
        )

        if message is None:
            raise RuntimeError(
                "No GPS telemetry received."
            )

        altitude = (
            message.relative_alt / 1000
            if getattr(
                message,
                "relative_alt",
                0
            )
            else message.alt / 1000
        )

        heading = (
            message.hdg / 100
            if getattr(
                message,
                "hdg",
                65535
            ) != 65535
            else 0
        )

        return {
            "latitude": message.lat / 1e7,
            "longitude": message.lon / 1e7,
            "altitude": altitude,
            "heading": heading,
            "source": "MAVLink",
            "timestamp": now_string()
        }

    finally:

        try:
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

            with open(
                file,
                "r",
                encoding="utf-8"
            ) as f:

                records.append(
                    json.load(f)
                )

        except Exception:
            continue

    return records


def next_scan_id():

    numbers = []

    for file in DRONE_ROOT.glob(
        "scan_*.json"
    ):

        try:

            numbers.append(
                int(
                    file.stem.split("_")[1]
                )
            )

        except Exception:
            pass

    return f"scan_{max(numbers, default=0)+1:04d}"


def save_drone_record(record):

    path = (
        DRONE_ROOT
        /
        f'{record["scan_id"]}.json'
    )

    save_json(
        path,
        record
    )


def save_drone_image(
    image,
    scan_id
):

    path = (
        DRONE_SCANS
        /
        f"{scan_id}.png"
    )

    cv2.imwrite(
        str(path),
        image
    )

    return str(path)


def save_result_image(
    image,
    scan_id,
    suffix
):

    path = (
        DRONE_RESULTS
        /
        f"{scan_id}_{suffix}.png"
    )

    cv2.imwrite(
        str(path),
        image
    )

    return str(path)


# ============================================================
# IMAGE ALIGNMENT
# ============================================================

def align_images(
    reference,
    current
):

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

        good = matches[
            :min(80, len(matches))
        ]

        src = np.float32([
            kp2[m.trainIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)

        dst = np.float32([
            kp1[m.queryIdx].pt
            for m in good
        ]).reshape(-1, 1, 2)

        matrix, mask = cv2.findHomography(
            src,
            dst,
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


def calculate_difference(
    reference,
    current
):

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

    difference = cv2.absdiff(
        blur1,
        blur2
    )

    difference = cv2.normalize(
        difference,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    )

    _, threshold = cv2.threshold(
        difference,
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

    return difference, threshold


# ============================================================
# DAMAGE
# ============================================================

def create_damage_map(
    difference,
    critical,
    moderate,
    low
):

    result = np.zeros(
        (
            difference.shape[0],
            difference.shape[1],
            3
        ),
        dtype=np.uint8
    )

    result[
        difference < low
    ] = [60, 180, 60]

    result[
        (difference >= low)
        &
        (difference < moderate)
    ] = [0, 220, 255]

    result[
        (difference >= moderate)
        &
        (difference < critical)
    ] = [0, 150, 255]

    result[
        difference >= critical
    ] = [0, 0, 255]

    return result


def damage_percentages(
    difference,
    critical,
    moderate,
    low
):

    total = difference.size

    if total == 0:
        return {
            "Critical": 0,
            "Moderate": 0,
            "Low": 0,
            "Safe": 0
        }

    critical_count = np.sum(
        difference >= critical
    )

    moderate_count = np.sum(
        (difference >= moderate)
        &
        (difference < critical)
    )

    low_count = np.sum(
        (difference >= low)
        &
        (difference < moderate)
    )

    safe_count = (
        total
        -
        critical_count
        -
        moderate_count
        -
        low_count
    )

    return {
        "Critical": round(
            critical_count / total * 100,
            2
        ),
        "Moderate": round(
            moderate_count / total * 100,
            2
        ),
        "Low": round(
            low_count / total * 100,
            2
        ),
        "Safe": round(
            safe_count / total * 100,
            2
        )
    }


# ============================================================
# COORDINATE MATCHING
# ============================================================

def find_nearest_scan(
    history,
    latitude,
    longitude,
    radius,
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
            (distance, record)
        )

    candidates.sort(
        key=lambda x: x[0]
    )

    if not candidates:
        return None, None

    distance, record = candidates[0]

    if distance <= radius:
        return record, distance

    return None, distance


# ============================================================
# YOLO
# ============================================================

@st.cache_resource
def load_yolo_model(path):

    try:

        from ultralytics import YOLO

        return YOLO(path)

    except Exception:
        return None


def find_model_path():

    candidates = [
        Path("models/disaster_best.pt"),
        Path("models/best.pt"),
        Path("best.pt")
    ]

    for path in candidates:

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
        return "Possible Survivor"

    if name in [
        "car",
        "truck",
        "bus",
        "motorcycle"
    ]:
        return "Vehicle"

    return "Object"


def run_yolo(
    image,
    confidence
):

    model = load_yolo_model(
        find_model_path()
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

            if isinstance(names, dict):

                class_name = names.get(
                    cls_id,
                    str(cls_id)
                )

            else:

                class_name = names[
                    cls_id
                ]

            coords = (
                box.xyxy[0]
                .tolist()
            )

            detections.append({
                "class": class_name,
                "category":
                    classify_detection(
                        class_name
                    ),
                "confidence":
                    round(conf * 100, 2),
                "x1": int(coords[0]),
                "y1": int(coords[1]),
                "x2": int(coords[2]),
                "y2": int(coords[3])
            })

        return plotted, detections

    except Exception:

        return image, []


# ============================================================
# SURVIVOR GPS ESTIMATION
# ============================================================

def estimate_detection_coordinate(
    drone_lat,
    drone_lon,
    altitude,
    heading,
    image_shape,
    bbox,
    hfov,
    vfov
):

    if altitude <= 0:
        return drone_lat, drone_lon

    h, w = image_shape[:2]

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

    east = (
        (cx / w) - .5
    ) * ground_width

    north = (
        .5 - (cy / h)
    ) * ground_height

    angle = math.radians(
        heading
    )

    east_world = (
        east * math.cos(angle)
        +
        north * math.sin(angle)
    )

    north_world = (
        -east * math.sin(angle)
        +
        north * math.cos(angle)
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
                math.radians(
                    drone_lat
                )
            )
        )
    )

    return latitude, longitude


def create_survivor_alerts(
    detections,
    gps,
    image_shape,
    hfov,
    vfov,
    scan_id
):

    alerts = []

    for index, detection in enumerate(
        detections
    ):

        if detection["category"] != "Possible Survivor":
            continue

        bbox = [
            detection["x1"],
            detection["y1"],
            detection["x2"],
            detection["y2"]
        ]

        lat, lon = estimate_detection_coordinate(
            gps["latitude"],
            gps["longitude"],
            gps["altitude"],
            gps["heading"],
            image_shape,
            bbox,
            hfov,
            vfov
        )

        alerts.append({

            "notification_id":
                f"{scan_id}_person_{index+1}",

            "timestamp":
                now_string(),

            "type":
                "SURVIVOR_ALERT",

            "message":
                "Person / Possible Survivor detected",

            "latitude":
                round(lat, 7),

            "longitude":
                round(lon, 7),

            "altitude":
                round(
                    float(
                        gps["altitude"]
                    ),
                    2
                ),

            "confidence":
                detection["confidence"],

            "scan_id":
                scan_id,

            "bbox":
                bbox
        })

    return alerts


# ============================================================
# NOTIFICATIONS
# ============================================================

def load_notifications():

    return load_json(
        NOTIFICATION_FILE,
        []
    )


def save_notification(alert):

    notifications = load_notifications()

    notifications.append(alert)

    save_json(
        NOTIFICATION_FILE,
        notifications[-500:]
    )

    st.session_state.notifications = (
        notifications[-500:]
    )


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
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("""
    <div class="sidebar-brand">

        <div class="sidebar-brand-title">
            🛰️ RESPONSE CONTROL
        </div>

        <div class="sidebar-brand-subtitle">
            DISASTER INTELLIGENCE SYSTEM
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="control-panel">

        <div class="panel-subtitle">
            SYSTEM STATUS
        </div>

        <div class="status-online">
            🟢 OPERATIONAL
        </div>

        <div class="panel-subtitle">
            AI • GPS • DAMAGE ANALYTICS
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ ANALYSIS")

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

    st.divider()

    st.markdown("### 🛰️ GPS SOURCE")

    gps_mode = st.selectbox(
        "Telemetry Mode",
        [
            "Manual",
            "Demo Simulator",
            "MAVLink"
        ]
    )

    if gps_mode == "Manual":

        latitude = st.number_input(
            "Latitude",
            value=float(
                st.session_state.gps[
                    "latitude"
                ]
            ),
            format="%.7f"
        )

        longitude = st.number_input(
            "Longitude",
            value=float(
                st.session_state.gps[
                    "longitude"
                ]
            ),
            format="%.7f"
        )

        altitude = st.number_input(
            "Altitude (m)",
            min_value=0.0,
            value=float(
                st.session_state.gps[
                    "altitude"
                ]
            )
        )

        heading = st.number_input(
            "Heading (°)",
            0.0,
            359.9,
            value=float(
                st.session_state.gps[
                    "heading"
                ]
            )
        )

        st.session_state.gps = {
            "latitude": latitude,
            "longitude": longitude,
            "altitude": altitude,
            "heading": heading,
            "source": "Manual"
        }

    elif gps_mode == "Demo Simulator":

        base_lat = st.number_input(
            "Start Latitude",
            value=23.2599,
            format="%.7f"
        )

        base_lon = st.number_input(
            "Start Longitude",
            value=77.4126,
            format="%.7f"
        )

        step_size = st.slider(
            "Movement Step",
            1,
            20,
            5
        )

        demo_altitude = st.number_input(
            "Altitude (m)",
            5.0,
            500.0,
            50.0
        )

        demo_heading = st.number_input(
            "Heading (°)",
            0.0,
            359.9,
            45.0
        )

        if st.button(
            "📡 NEXT GPS POSITION",
            use_container_width=True
        ):

            st.session_state.demo_step += 1

        movement = (
            st.session_state.demo_step
            * step_size
        )

        st.session_state.gps = {
            "latitude":
                base_lat
                +
                movement * 0.00001,

            "longitude":
                base_lon
                +
                movement * 0.000015,

            "altitude":
                demo_altitude,

            "heading":
                demo_heading,

            "source":
                "Demo"
        }

    else:

        if not PYMAVLINK_AVAILABLE:

            st.warning(
                "Install pymavlink for MAVLink."
            )

        endpoint = st.text_input(
            "MAVLink Endpoint",
            "udp:127.0.0.1:14550"
        )

        baud = st.number_input(
            "Baud Rate",
            9600,
            921600,
            57600
        )

        if st.button(
            "📡 READ LIVE GPS",
            use_container_width=True
        ):

            try:

                gps_data = read_mavlink_gps(
                    endpoint,
                    int(baud)
                )

                st.session_state.gps = gps_data

                save_telemetry(
                    gps_data
                )

                st.success(
                    "GPS updated."
                )

            except Exception as error:

                st.error(
                    str(error)
                )

    st.divider()

    st.markdown("### 🤖 AI DETECTION")

    yolo_confidence = st.slider(
        "YOLO Confidence",
        0.10,
        0.95,
        0.35
    )

    hfov = st.number_input(
        "Camera HFOV (°)",
        20.0,
        120.0,
        60.0
    )

    vfov = st.number_input(
        "Camera VFOV (°)",
        20.0,
        100.0,
        45.0
    )

    st.divider()

    st.markdown("""
    <div class="control-panel">

        <div class="panel-subtitle">
            PROTOTYPE NOTICE
        </div>

        <div style="
            color:#7f92a8;
            font-size:10px;
            line-height:1.6;
        ">
            Person detection is an AI indicator only.
            Final survivor confirmation requires
            responder verification.
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# CURRENT VALUES
# ============================================================

gps = st.session_state.gps

history = (
    st.session_state.drone_history
    or []
)

notifications = (
    st.session_state.notifications
    or []
)


lat = float(
    gps["latitude"]
)

lon = float(
    gps["longitude"]
)

alt = float(
    gps["altitude"]
)

heading = float(
    gps["heading"]
)


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(f"""
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
                AI Damage Intelligence
                &nbsp;•&nbsp;
                Drone GPS Tracking
                &nbsp;•&nbsp;
                Possible Survivor Detection
            </div>

            <div class="header-meta">
                LAST TELEMETRY: {now_string()}
            </div>

        </div>

        <div class="live-pill">
            <span class="live-dot"></span>
            SYSTEM ONLINE
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# KPI ROW
# ============================================================

k1, k2, k3, k4 = st.columns(4)

with k1:

    st.markdown(f"""
    <div class="kpi-card">

        <div class="kpi-label">
            🛰️ DRONE STATUS
        </div>

        <div class="kpi-value">
            LIVE
        </div>

        <div class="kpi-small">
            Source: {gps["source"]}
        </div>

    </div>
    """, unsafe_allow_html=True)


with k2:

    st.markdown(f"""
    <div class="kpi-card">

        <div class="kpi-label">
            📍 CURRENT LOCATION
        </div>

        <div class="kpi-value">
            {lat:.5f}
        </div>

        <div class="kpi-small">
            {lon:.5f}
        </div>

    </div>
    """, unsafe_allow_html=True)


with k3:

    st.markdown(f"""
    <div class="kpi-card">

        <div class="kpi-label">
            🚨 ALERTS
        </div>

        <div class="kpi-value">
            {len(notifications)}
        </div>

        <div class="kpi-small">
            Possible survivor alerts
        </div>

    </div>
    """, unsafe_allow_html=True)


with k4:

    st.markdown(f"""
    <div class="kpi-card">

        <div class="kpi-label">
            📊 SCANS
        </div>

        <div class="kpi-value">
            {len(history)}
        </div>

        <div class="kpi-small">
            GPS indexed scans
        </div>

    </div>
    """, unsafe_allow_html=True)


# ============================================================
# TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🛰️ DRONE COMMAND",
    "🗺️ SATELLITE ANALYSIS",
    "📷 CAMERA AI",
    "🔔 ALERT HISTORY"
])


# ============================================================
# TAB 1
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">🛰️ Live Drone Mission</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Live position, scan matching and AI response monitoring'
        '</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns(
        [1.7, 1]
    )

    with left:

        st.markdown("""
        <div class="info-card">

            <div class="info-card-title">
                📍 LIVE GPS TRACK
            </div>

            <div class="info-card-text">
                Current drone position and
                historical scan locations.
            </div>

        </div>
        """, unsafe_allow_html=True)

        gps_points = []

        for record in history:

            if (
                record.get("latitude")
                is not None
                and
                record.get("longitude")
                is not None
            ):

                gps_points.append({
                    "latitude":
                        record["latitude"],

                    "longitude":
                        record["longitude"]
                })

        gps_points.append({
            "latitude": lat,
            "longitude": lon
        })

        st.map(
            pd.DataFrame(gps_points),
            height=420
        )

    with right:

        st.markdown(f"""
        <div class="mission-panel">

            <div class="mission-label">
                CURRENT DRONE POSITION
            </div>

            <div class="mission-value">
                GPS LOCKED
            </div>

            <div class="mission-coord">
                {lat:.7f}, {lon:.7f}
            </div>

            <br>

            <div class="mission-label">
                ALTITUDE
            </div>

            <div class="mission-value">
                {alt:.1f} m
            </div>

            <br>

            <div class="mission-label">
                HEADING
            </div>

            <div class="mission-value">
                {heading:.1f}°
            </div>

            <br>

            <div class="mission-label">
                TELEMETRY
            </div>

            <div class="mission-value">
                {gps["source"]}
            </div>

        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        st.markdown("""
        <div class="info-card">

            <div class="info-card-title">
                🎯 COORDINATE MATCHING
            </div>

            <div class="info-card-text">
                Drone scans are matched with the
                nearest historical image using
                geographic coordinates.
            </div>

        </div>
        """, unsafe_allow_html=True)

        coordinate_radius = st.number_input(
            "Comparison Radius (meters)",
            5,
            1000,
            50
        )

        scan_type = st.selectbox(
            "Current Scan Type",
            [
                "Monitoring",
                "Pre-Disaster",
                "Post-Disaster"
            ]
        )


    # ========================================================
    # ALERT CENTER
    # ========================================================

    st.markdown(
        '<div class="section-title">🚨 Survivor Detection Center</div>',
        unsafe_allow_html=True
    )

    if notifications:

        for alert in notifications[-3:][::-1]:

            st.markdown(f"""
            <div class="survivor-alert">

                <div class="alert-title">
                    🚨 POSSIBLE SURVIVOR DETECTED
                </div>

                <div class="alert-text">
                    AI detected a person in the drone frame.
                    Responder verification required.
                </div>

                <div class="alert-data">

                    <span class="badge badge-red">
                        CONFIDENCE {alert["confidence"]}%
                    </span>

                    <span class="badge badge-blue">
                        📍 {alert["latitude"]:.6f},
                        {alert["longitude"]:.6f}
                    </span>

                    <span class="badge badge-orange">
                        {alert["timestamp"]}
                    </span>

                </div>

            </div>
            """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="info-card">

            <div class="info-card-title">
                🟢 NO ACTIVE SURVIVOR ALERT
            </div>

            <div class="info-card-text">
                Upload a drone image to run AI detection.
            </div>

        </div>
        """, unsafe_allow_html=True)


    # ========================================================
    # DRONE SCAN
    # ========================================================

    st.markdown(
        '<div class="section-title">📸 Drone Scan</div>',
        unsafe_allow_html=True
    )

    drone_file = st.file_uploader(
        "Upload current drone image",
        type=["jpg", "jpeg", "png"],
        key="drone_upload"
    )

    if drone_file:

        drone_image = uploaded_to_cv(
            drone_file
        )

        if drone_image is not None:

            preview_col, info_col = st.columns(
                [1.4, 1]
            )

            with preview_col:

                st.image(
                    cv_to_rgb(
                        drone_image
                    ),
                    caption="CURRENT DRONE FRAME",
                    width="stretch"
                )

            with info_col:

                st.markdown("""
                <div class="info-card">

                    <div class="info-card-title">
                        AI MISSION PIPELINE
                    </div>

                    <div class="info-card-text">
                        01 — GPS coordinate lookup<br>
                        02 — Geographic scan matching<br>
                        03 — Image alignment<br>
                        04 — Damage comparison<br>
                        05 — YOLO object detection<br>
                        06 — Survivor coordinate estimation<br>
                        07 — Persistent alert generation
                    </div>

                </div>
                """, unsafe_allow_html=True)

            if st.button(
                "🚀 PROCESS DRONE SCAN",
                type="primary",
                use_container_width=True
            ):

                current_hash = image_hash(
                    drone_file
                )

                if current_hash == st.session_state.drone_last_hash:

                    st.info(
                        "This image has already been processed."
                    )

                else:

                    scan_id = next_scan_id()

                    image_path = save_drone_image(
                        drone_image,
                        scan_id
                    )

                    # ------------------------------------------
                    # FIND BASELINE
                    # ------------------------------------------

                    previous_record = None
                    coordinate_distance = None

                    if scan_type == "Post-Disaster":

                        previous_record, coordinate_distance = (
                            find_nearest_scan(
                                history,
                                lat,
                                lon,
                                coordinate_radius,
                                "Pre-Disaster"
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

                    # ------------------------------------------
                    # DAMAGE COMPARISON
                    # ------------------------------------------

                    previous_image = None
                    damage_map = None
                    statistics = None
                    comparison_available = False
                    alignment_success = False
                    matches = 0

                    if previous_record:

                        previous_path = previous_record.get(
                            "image_path"
                        )

                        if (
                            previous_path
                            and
                            Path(previous_path).exists()
                        ):

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

                    # ------------------------------------------
                    # AI
                    # ------------------------------------------

                    plotted_image, detections = run_yolo(
                        drone_image,
                        yolo_confidence
                    )

                    # ------------------------------------------
                    # SURVIVOR ALERTS
                    # ------------------------------------------

                    survivor_alerts = create_survivor_alerts(
                        detections,
                        gps,
                        drone_image.shape,
                        hfov,
                        vfov,
                        scan_id
                    )

                    for alert in survivor_alerts:

                        save_notification(
                            alert
                        )

                    # ------------------------------------------
                    # SAVE RESULTS
                    # ------------------------------------------

                    ai_path = save_result_image(
                        plotted_image,
                        scan_id,
                        "ai"
                    )

                    damage_path = None

                    if damage_map is not None:

                        damage_path = save_result_image(
                            damage_map,
                            scan_id,
                            "damage"
                        )

                    # ------------------------------------------
                    # RECORD
                    # ------------------------------------------

                    record = {

                        "scan_id":
                            scan_id,

                        "timestamp":
                            now_string(),

                        "image_path":
                            image_path,

                        "ai_image_path":
                            ai_path,

                        "damage_map_path":
                            damage_path,

                        "latitude":
                            round(lat, 7),

                        "longitude":
                            round(lon, 7),

                        "altitude":
                            round(alt, 2),

                        "heading":
                            round(heading, 2),

                        "gps_source":
                            gps["source"],

                        "scan_type":
                            scan_type,

                        "compared_with":
                            previous_record[
                                "scan_id"
                            ]
                            if previous_record
                            else None,

                        "coordinate_distance_m":
                            round(
                                coordinate_distance,
                                2
                            )
                            if coordinate_distance is not None
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

                        "record":
                            record,

                        "current":
                            drone_image,

                        "previous":
                            previous_image,

                        "damage_map":
                            damage_map,

                        "ai_image":
                            plotted_image
                    }

                    if survivor_alerts:

                        st.error(
                            f"🚨 {len(survivor_alerts)} "
                            "possible survivor alert(s) generated."
                        )

                    else:

                        st.success(
                            "🟢 No person detected."
                        )

                    if comparison_available:

                        st.success(
                            f"📍 GPS matched with "
                            f"{previous_record['scan_id']} "
                            f"({coordinate_distance:.2f} m)"
                        )

                    else:

                        st.warning(
                            f"📍 No baseline scan within "
                            f"{coordinate_radius} m."
                        )


    # ========================================================
    # LATEST RESULT
    # ========================================================

    analysis = (
        st.session_state.drone_current_analysis
    )

    if analysis:

        st.markdown(
            '<div class="section-title">📊 Latest Mission Result</div>',
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns(3)

        with c1:

            st.image(
                cv_to_rgb(
                    analysis["current"]
                ),
                caption="CURRENT SCAN",
                width="stretch"
            )

        with c2:

            if analysis["previous"] is not None:

                st.image(
                    cv_to_rgb(
                        analysis["previous"]
                    ),
                    caption="GPS MATCHED BASELINE",
                    width="stretch"
                )

            else:

                st.markdown("""
                <div class="info-card">

                    <div class="info-card-title">
                        NO BASELINE
                    </div>

                    <div class="info-card-text">
                        No previous scan was found
                        inside the selected GPS radius.
                    </div>

                </div>
                """, unsafe_allow_html=True)

        with c3:

            if analysis["damage_map"] is not None:

                st.image(
                    cv_to_rgb(
                        analysis["damage_map"]
                    ),
                    caption="DAMAGE MAP",
                    width="stretch"
                )

            else:

                st.markdown("""
                <div class="info-card">

                    <div class="info-card-title">
                        GPS BASELINE REQUIRED
                    </div>

                    <div class="info-card-text">
                        Damage comparison will appear
                        when a geographic baseline exists.
                    </div>

                </div>
                """, unsafe_allow_html=True)

        record = analysis["record"]

        if record.get("statistics"):

            stats = record["statistics"]

            st.markdown(
                '<div class="section-title">📈 Damage Distribution</div>',
                unsafe_allow_html=True
            )

            a, b, c, d = st.columns(4)

            for col, label in zip(
                [a, b, c, d],
                [
                    "Critical",
                    "Moderate",
                    "Low",
                    "Safe"
                ]
            ):

                with col:

                    st.markdown(f"""
                    <div class="kpi-card">

                        <div class="kpi-label">
                            {label}
                        </div>

                        <div class="kpi-value">
                            {stats[label]}%
                        </div>

                    </div>
                    """, unsafe_allow_html=True)

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

        detections = record.get(
            "detections",
            []
        )

        if detections:

            st.markdown(
                '<div class="section-title">🤖 AI Detections</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                pd.DataFrame(detections),
                width="stretch",
                hide_index=True
            )


    # ========================================================
    # SCAN HISTORY
    # ========================================================

    st.markdown(
        '<div class="section-title">🗂️ Coordinate-Based Scan History</div>',
        unsafe_allow_html=True
    )

    if history:

        rows = []

        for record in reversed(history):

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

                "GPS Distance":
                    record.get(
                        "coordinate_distance_m"
                    )
                    or "—",

                "GPS Match":
                    "YES"
                    if record.get(
                        "coordinate_match"
                    )
                    else "NO",

                "Alerts":
                    len(
                        record.get(
                            "survivor_alerts",
                            []
                        )
                    )
            })

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True
        )

    else:

        st.info(
            "No drone scans recorded yet."
        )


# ============================================================
# TAB 2 — SATELLITE
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">🗺️ Satellite Damage Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Compare pre-disaster and post-disaster imagery'
        '</div>',
        unsafe_allow_html=True
    )

    use_gps = st.checkbox(
        "📍 Validate image coordinates",
        False
    )

    if use_gps:

        g1, g2 = st.columns(2)

        with g1:

            pre_lat = st.number_input(
                "Pre Latitude",
                value=23.2599,
                format="%.7f"
            )

            pre_lon = st.number_input(
                "Pre Longitude",
                value=77.4126,
                format="%.7f"
            )

        with g2:

            post_lat = st.number_input(
                "Post Latitude",
                value=23.2599,
                format="%.7f"
            )

            post_lon = st.number_input(
                "Post Longitude",
                value=77.4126,
                format="%.7f"
            )

        satellite_radius = st.number_input(
            "Allowed GPS Difference (m)",
            5,
            5000,
            100
        )

    s1, s2 = st.columns(2)

    with s1:

        pre_file = st.file_uploader(
            "PRE-DISASTER IMAGE",
            type=["jpg", "jpeg", "png"],
            key="pre_upload"
        )

    with s2:

        post_file = st.file_uploader(
            "POST-DISASTER IMAGE",
            type=["jpg", "jpeg", "png"],
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

            if use_gps:

                distance = haversine_m(
                    pre_lat,
                    pre_lon,
                    post_lat,
                    post_lon
                )

                if distance > satellite_radius:

                    st.error(
                        f"Images are {distance:.2f} m apart."
                    )

                else:

                    st.success(
                        f"GPS validated — {distance:.2f} m"
                    )

            result = None

            aligned, success, matches = align_images(
                pre_img,
                post_img
            )

            if success:

                compare = aligned

            else:

                compare = cv2.resize(
                    post_img,
                    (
                        pre_img.shape[1],
                        pre_img.shape[0]
                    )
                )

            difference, threshold = (
                calculate_difference(
                    pre_img,
                    compare
                )
            )

            damage_map = create_damage_map(
                difference,
                critical_threshold,
                moderate_threshold,
                low_threshold
            )

            stats = damage_percentages(
                difference,
                critical_threshold,
                moderate_threshold,
                low_threshold
            )

            st.session_state.satellite_result = {

                "pre":
                    pre_img,

                "post":
                    compare,

                "damage_map":
                    damage_map,

                "stats":
                    stats,

                "success":
                    success,

                "matches":
                    matches
            }

    result = st.session_state.satellite_result

    if result:

        st.markdown(
            '<div class="section-title">🛰️ Analysis Output</div>',
            unsafe_allow_html=True
        )

        a, b, c = st.columns(3)

        with a:

            st.image(
                cv_to_rgb(result["pre"]),
                caption="PRE-DISASTER",
                width="stretch"
            )

        with b:

            st.image(
                cv_to_rgb(result["post"]),
                caption="POST-DISASTER",
                width="stretch"
            )

        with c:

            st.image(
                cv_to_rgb(result["damage_map"]),
                caption="DAMAGE CLASSIFICATION",
                width="stretch"
            )

        if result["success"]:

            st.success(
                f"Image alignment successful — "
                f"{result['matches']} feature matches"
            )

        else:

            st.warning(
                "Automatic image alignment was not reliable."
            )

        stats = result["stats"]

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

                st.markdown(f"""
                <div class="kpi-card">

                    <div class="kpi-label">
                        {label}
                    </div>

                    <div class="kpi-value">
                        {stats[label]}%
                    </div>

                </div>
                """, unsafe_allow_html=True)

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
        '<div class="section-caption">'
        'Capture a disaster scene and run AI object detection'
        '</div>',
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
                caption="CAPTURED SCENE",
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
                    caption="AI DETECTION RESULT",
                    width="stretch"
                )

                if detections:

                    st.dataframe(
                        pd.DataFrame(detections),
                        width="stretch",
                        hide_index=True
                    )

                    persons = sum(
                        1
                        for d in detections
                        if d["category"]
                        ==
                        "Possible Survivor"
                    )

                    if persons:

                        st.error(
                            f"🚨 {persons} possible "
                            "survivor/person detection(s)"
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
# TAB 4 — ALERT HISTORY
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">🔔 Mission Alert History</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-caption">'
        'Persistent AI-generated survivor/person alerts'
        '</div>',
        unsafe_allow_html=True
    )

    notifications = load_notifications()

    st.session_state.notifications = notifications

    if notifications:

        latest = notifications[-1]

        st.markdown(f"""
        <div class="survivor-alert">

            <div class="alert-title">
                🚨 LATEST SURVIVOR ALERT
            </div>

            <div class="alert-text">
                Person detected by AI
            </div>

            <div class="alert-data">

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
        """, unsafe_allow_html=True)

        rows = []

        for alert in reversed(
            notifications
        ):

            rows.append({

                "Time":
                    alert.get(
                        "timestamp"
                    ),

                "Alert":
                    alert.get(
                        "message"
                    ),

                "Latitude":
                    alert.get(
                        "latitude"
                    ),

                "Longitude":
                    alert.get(
                        "longitude"
                    ),

                "Altitude":
                    alert.get(
                        "altitude"
                    ),

                "Confidence":
                    f'{alert.get("confidence", 0)}%',

                "Scan":
                    alert.get(
                        "scan_id"
                    )
            })

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True
        )

        st.markdown(
            '<div class="section-title">📍 Alert Locations</div>',
            unsafe_allow_html=True
        )

        alert_points = []

        for alert in notifications:

            if (
                alert.get("latitude")
                is not None
                and
                alert.get("longitude")
                is not None
            ):

                alert_points.append({

                    "latitude":
                        alert["latitude"],

                    "longitude":
                        alert["longitude"]
                })

        if alert_points:

            st.map(
                pd.DataFrame(alert_points),
                height=430
            )

    else:

        st.markdown("""
        <div class="info-card">

            <div class="info-card-title">
                🔔 NO ALERTS
            </div>

            <div class="info-card-text">
                No AI survivor alerts have been generated yet.
            </div>

        </div>
        """, unsafe_allow_html=True)


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">

    DISASTER RESPONSE COMMAND CENTER
    &nbsp;•&nbsp;
    AI DAMAGE INTELLIGENCE
    &nbsp;•&nbsp;
    GPS RESPONSE TRACKING

    <br><br>

    Research & demonstration prototype
    •
    AI person detection requires human verification

</div>
""", unsafe_allow_html=True)