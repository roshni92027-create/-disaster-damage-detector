
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path
import json
import hashlib
from datetime import datetime

# Optional AI detector. The app still works for satellite analysis if Ultralytics is unavailable.
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Disaster Damage Detector",
    page_icon="🚨",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background: #07111f; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; }
.hero { text-align:center; padding:10px 0 24px; }
.hero h1 { color:white; font-size:42px; margin-bottom:4px; }
.hero p { color:#9caec4; font-size:16px; }
.section-title { color:white; font-size:25px; font-weight:700; margin:25px 0 12px; }
.metric { background:#101c2d; border:1px solid #26374d; border-radius:14px; padding:15px; text-align:center; }
.metric-title { color:#9caec4; font-size:12px; }
.metric-value { color:white; font-size:25px; font-weight:700; }
.note { color:#9caec4; font-size:13px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
<h1>🚨 Disaster Damage Detector</h1>
<p>Satellite change detection + AI camera assessment + responder mapping</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "satellite_result" not in st.session_state:
    st.session_state.satellite_result = None

if "camera_result" not in st.session_state:
    st.session_state.camera_result = None

if "drone_history" not in st.session_state:
    st.session_state.drone_history = None

if "drone_last_hash" not in st.session_state:
    st.session_state.drone_last_hash = None

if "drone_current_analysis" not in st.session_state:
    st.session_state.drone_current_analysis = None


# ============================================================
# HELPERS
# ============================================================

def uploaded_to_cv(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def cv_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize_images(img1, img2):
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    return (
        cv2.resize(img1, (w, h)),
        cv2.resize(img2, (w, h)),
    )


# ============================================================
# DRONE SCAN HISTORY / PERSISTENCE
# ============================================================

DRONE_ROOT = Path("drone_history")
DRONE_SCANS = DRONE_ROOT / "scans"
DRONE_RESULTS = DRONE_ROOT / "results"
DRONE_SCANS.mkdir(parents=True, exist_ok=True)
DRONE_RESULTS.mkdir(parents=True, exist_ok=True)


def drone_image_hash(uploaded_file):
    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()


def load_drone_history():
    records = []
    for path in sorted(DRONE_RESULTS.glob("scan_*.json"), key=lambda p: p.stat().st_mtime):
        try:
            records.append(json.loads(path.read_text()))
        except Exception:
            continue
    return records


def save_drone_image(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_drone_record(record):
    path = DRONE_RESULTS / f"{record['scan_id']}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def drone_damage_summary(percentages):
    affected = percentages["Low Damage"] + percentages["Moderate Damage"] + percentages["Critical"]
    return {
        "Safe": round(float(percentages["Safe"]), 2),
        "Low Damage": round(float(percentages["Low Damage"]), 2),
        "Moderate Damage": round(float(percentages["Moderate Damage"]), 2),
        "Critical": round(float(percentages["Critical"]), 2),
        "Affected Area": round(float(affected), 2),
    }


def analyze_drone_scan(current_image, previous_record, low, moderate, critical, minimum_area, responder_points, show_responders):
    current = current_image.copy()

    if previous_record is None:
        difference = np.zeros(current.shape[:2], dtype=np.uint8)
        damage_map = np.zeros_like(current)
        damage_map[:, :] = (0, 170, 0)
        percentages = {"Safe": 100.0, "Low Damage": 0.0, "Moderate Damage": 0.0, "Critical": 0.0}
        return {"current": current, "previous": None, "difference": difference, "damage_map": damage_map, "percentages": percentages, "aligned": False, "matches": 0, "baseline": True}

    previous_path = Path(previous_record["image_path"])
    if not previous_path.exists():
        return analyze_drone_scan(current, None, low, moderate, critical, minimum_area, responder_points, show_responders)

    previous = cv2.imread(str(previous_path))
    if previous is None:
        return analyze_drone_scan(current, None, low, moderate, critical, minimum_area, responder_points, show_responders)

    previous, current = resize_images(previous, current)
    aligned_current, aligned, matches = align_images(previous, current)
    difference = calculate_difference(previous, aligned_current)
    damage_map = create_damage_map(difference, low, moderate, critical, minimum_area)
    if show_responders:
        damage_map = add_responder_markers(damage_map, responder_points)
    percentages = damage_percentages(difference, low, moderate, critical)
    return {"current": current, "previous": aligned_current, "difference": difference, "damage_map": damage_map, "percentages": percentages, "aligned": aligned, "matches": matches, "baseline": False}


# ============================================================
# SATELLITE ALIGNMENT
# ============================================================

def align_images(pre, post):
    gray_pre = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)

    kp1, des1 = orb.detectAndCompute(gray_pre, None)
    kp2, des2 = orb.detectAndCompute(gray_post, None)

    if des1 is None or des2 is None:
        return post, False, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(des2, des1, k=2)

    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 8:
        return post, False, len(good)

    src_pts = np.float32(
        [kp2[m.queryIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    dst_pts = np.float32(
        [kp1[m.trainIdx].pt for m in good]
    ).reshape(-1, 1, 2)

    H, _ = cv2.findHomography(
        src_pts, dst_pts, cv2.RANSAC, 5.0
    )

    if H is None:
        return post, False, len(good)

    h, w = pre.shape[:2]
    aligned = cv2.warpPerspective(post, H, (w, h))

    return aligned, True, len(good)


def calculate_difference(pre, post):
    pre_gray = cv2.cvtColor(pre, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(post, cv2.COLOR_BGR2GRAY)

    pre_gray = cv2.GaussianBlur(pre_gray, (7, 7), 0)
    post_gray = cv2.GaussianBlur(post_gray, (7, 7), 0)

    difference = cv2.absdiff(pre_gray, post_gray)

    difference = cv2.normalize(
        difference, None, 0, 255, cv2.NORM_MINMAX
    )

    kernel = np.ones((5, 5), np.uint8)
    difference = cv2.morphologyEx(
        difference, cv2.MORPH_OPEN, kernel
    )
    difference = cv2.morphologyEx(
        difference, cv2.MORPH_CLOSE, kernel
    )

    return difference


def create_damage_map(
    difference,
    low,
    moderate,
    critical,
    minimum_area,
):
    h, w = difference.shape

    # Safe = green
    result = np.zeros((h, w, 3), dtype=np.uint8)
    result[:, :] = (0, 170, 0)

    low_mask = (
        (difference >= low) &
        (difference < moderate)
    )
    moderate_mask = (
        (difference >= moderate) &
        (difference < critical)
    )
    critical_mask = difference >= critical

    # BGR
    result[low_mask] = (0, 255, 255)       # yellow
    result[moderate_mask] = (0, 140, 255)  # orange
    result[critical_mask] = (0, 0, 255)   # red

    binary = np.zeros_like(difference)
    binary[difference >= low] = 255

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    critical_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue

        mask = np.zeros_like(difference)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        values = difference[mask == 255]

        if len(values) == 0:
            continue

        mean_value = float(np.mean(values))

        if mean_value >= critical:
            cv2.drawContours(
                result, [contour], -1, (0, 0, 255), -1
            )
            critical_contours.append(contour)

        elif mean_value >= moderate:
            cv2.drawContours(
                result, [contour], -1, (0, 140, 255), -1
            )

        else:
            cv2.drawContours(
                result, [contour], -1, (0, 255, 255), -1
            )

    # Filled critical circles as requested
    for contour in critical_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = max(int(radius), 15)

        cv2.circle(
            result, center, radius, (0, 0, 255), -1
        )
        cv2.circle(
            result, center, radius, (0, 0, 120), 3
        )

    return result


def damage_percentages(difference, low, moderate, critical):
    total = difference.size

    safe = np.sum(difference < low)
    low_damage = np.sum(
        (difference >= low) & (difference < moderate)
    )
    moderate_damage = np.sum(
        (difference >= moderate) & (difference < critical)
    )
    critical_damage = np.sum(difference >= critical)

    return {
        "Safe": safe / total * 100,
        "Low Damage": low_damage / total * 100,
        "Moderate Damage": moderate_damage / total * 100,
        "Critical": critical_damage / total * 100,
    }


def add_responder_markers(image, points):
    output = image.copy()
    h, w = output.shape[:2]

    for number, (xp, yp) in enumerate(points, 1):
        x = int(w * xp / 100)
        y = int(h * yp / 100)

        cv2.circle(output, (x, y), 32, (255, 255, 255), -1)
        cv2.circle(output, (x, y), 32, (0, 0, 0), 2)

        # Helmet
        cv2.ellipse(
            output, (x, y - 7), (15, 10),
            0, 180, 360, (0, 165, 255), -1
        )
        cv2.rectangle(
            output, (x - 15, y - 7),
            (x + 15, y + 3), (0, 165, 255), -1
        )

        # Medical plus
        cv2.line(
            output, (x - 8, y + 13),
            (x + 8, y + 13), (0, 0, 255), 3
        )
        cv2.line(
            output, (x, y + 5),
            (x, y + 21), (0, 0, 255), 3
        )

        cv2.putText(
            output,
            f"Responder {number}",
            (x + 38, y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# ============================================================
# OPENCV CAMERA HEURISTICS
# ============================================================

def opencv_scene_analysis(image):
    output = image.copy()

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # -------- crack-like thin structures --------
    edges = cv2.Canny(gray, 80, 160)
    edges = cv2.dilate(
        edges, np.ones((3, 3), np.uint8), iterations=1
    )

    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crack_like = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, False)

        if area < 20 and perimeter > 80:
            x, y, cw, ch = cv2.boundingRect(contour)

            if cw > 10 and ch > 10:
                cv2.rectangle(
                    output,
                    (x, y),
                    (x + cw, y + ch),
                    (0, 0, 255),
                    2,
                )
                crack_like += 1

    # -------- water-like regions --------
    lower_water = np.array([80, 40, 40])
    upper_water = np.array([135, 255, 255])

    water_mask = cv2.inRange(
        hsv, lower_water, upper_water
    )

    water_pixels = np.sum(water_mask > 0)
    water_percentage = water_pixels / (h * w) * 100

    water_contours, _ = cv2.findContours(
        water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in water_contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(
                output, [contour], -1, (255, 0, 0), 3
            )

    # -------- dark obstruction/debris indicator --------
    dark_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 65]),
    )

    dark_percentage = np.sum(dark_mask > 0) / (h * w) * 100
    debris_possible = dark_percentage > 15

    return {
        "image": output,
        "crack_like": crack_like,
        "water_percentage": water_percentage,
        "debris_possible": debris_possible,
    }


# ============================================================
# YOLO CAMERA AI
# ============================================================

@st.cache_resource
def load_yolo_model(model_path):
    if not YOLO_AVAILABLE:
        return None

    return YOLO(model_path)


def find_model_path():
    """
    Priority:
    1. custom disaster model: models/disaster_best.pt
    2. local YOLO26n model: models/yolo26n.pt
    3. yolo26n.pt (Ultralytics downloads it automatically)
    """

    custom = Path("models/disaster_best.pt")
    local = Path("models/yolo26n.pt")

    if custom.exists():
        return str(custom), True

    if local.exists():
        return str(local), False

    return "yolo26n.pt", False


def classify_detection(name):
    """
    Maps model class names to project categories.
    This supports both COCO classes and future custom disaster classes.
    """

    n = name.lower().replace("_", " ").replace("-", " ")

    if any(x in n for x in [
        "fire", "flame", "smoke"
    ]):
        return "Fire / Smoke"

    if any(x in n for x in [
        "flood", "water"
    ]):
        return "Flood / Water"

    if any(x in n for x in [
        "debris", "rubble", "wreckage", "rubble pile"
    ]):
        return "Debris / Rubble"

    if any(x in n for x in [
        "collapsed building", "damaged building",
        "destroyed building", "building damage"
    ]):
        return "Building Damage"

    if any(x in n for x in [
        "crack", "structural crack"
    ]):
        return "Structural Crack"

    if any(x in n for x in [
        "person", "people", "human", "rescuer",
        "responder", "worker"
    ]):
        return "Person / Responder"

    if any(x in n for x in [
        "car", "truck", "bus", "motorcycle",
        "bicycle", "vehicle"
    ]):
        return "Vehicle"

    if any(x in n for x in [
        "road", "bridge"
    ]):
        return "Infrastructure"

    return "Other"


def run_yolo(image, model_path, conf):
    model = load_yolo_model(model_path)

    if model is None:
        return image.copy(), [], "YOLO unavailable"

    results = model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        verbose=False,
    )

    if not results:
        return image.copy(), [], "No results"

    result = results[0]
    plotted = result.plot()

    detections = []

    if result.boxes is not None:
        names = result.names

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            raw_name = names.get(
                cls_id, str(cls_id)
            )

            category = classify_detection(
                raw_name
            )

            xyxy = box.xyxy[0].cpu().numpy().astype(int)

            detections.append({
                "class": raw_name,
                "category": category,
                "confidence": confidence,
                "box": xyxy.tolist(),
            })

    return plotted, detections, "OK"


def combine_camera_results(
    image,
    yolo_image,
    cv_result,
    detections,
):
    """
    Adds project-specific warning banners and a simple
    risk score. This is a screening score, not a safety certification.
    """

    output = yolo_image.copy()

    risk = 0
    reasons = []

    # AI categories
    category_counts = {}

    for d in detections:
        cat = d["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    if category_counts.get("Fire / Smoke", 0) > 0:
        risk += 40
        reasons.append("fire/smoke detected")

    if category_counts.get("Flood / Water", 0) > 0:
        risk += 35
        reasons.append("water/flood indicator detected")

    if category_counts.get("Debris / Rubble", 0) > 0:
        risk += 30
        reasons.append("debris/rubble detected")

    if category_counts.get("Building Damage", 0) > 0:
        risk += 45
        reasons.append("building-damage class detected")

    if category_counts.get("Structural Crack", 0) > 0:
        risk += 35
        reasons.append("crack class detected")

    # OpenCV visual screening
    if cv_result["water_percentage"] > 8:
        risk += 20
        reasons.append("water-like pixels detected")

    if cv_result["crack_like"] > 10:
        risk += 15
        reasons.append("many edge/crack-like structures")

    if cv_result["debris_possible"]:
        risk += 10
        reasons.append("dark obstruction/debris indicator")

    risk = min(risk, 100)

    if risk >= 70:
        status = "🔴 HIGH VISUAL RISK"
    elif risk >= 40:
        status = "🟠 MODERATE VISUAL RISK"
    elif risk >= 15:
        status = "🟡 POSSIBLE DAMAGE"
    else:
        status = "🟢 LOW VISUAL RISK"

    # Top banner
    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 52),
        (15, 25, 45),
        -1,
    )

    cv2.putText(
        output,
        f"{status} | Risk Score {risk}/100",
        (18, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output, risk, status, reasons, category_counts


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown("## ⚙️ Detection Settings")

low_threshold = st.sidebar.slider(
    "Low Damage Threshold", 10, 150, 35
)

moderate_threshold = st.sidebar.slider(
    "Moderate Damage Threshold", 30, 200, 75
)

critical_threshold = st.sidebar.slider(
    "Critical Damage Threshold", 60, 255, 120
)

minimum_area = st.sidebar.slider(
    "Minimum Damage Area", 50, 5000, 300
)

st.sidebar.markdown("---")
st.sidebar.markdown("## ⛑️ Responder Mapping")

show_responders = st.sidebar.checkbox(
    "Show Responder Markers", True
)

responder_points = []

if show_responders:
    count = st.sidebar.number_input(
        "Number of Responder Zones",
        min_value=0,
        max_value=10,
        value=1,
    )

    for i in range(int(count)):
        x = st.sidebar.slider(
            f"Responder {i+1} X",
            0, 100, 50,
            key=f"resp_x_{i}",
        )
        y = st.sidebar.slider(
            f"Responder {i+1} Y",
            0, 100, 50,
            key=f"resp_y_{i}",
        )
        responder_points.append((x, y))

st.sidebar.markdown("---")
st.sidebar.markdown("## 🤖 Camera AI")

camera_conf = st.sidebar.slider(
    "YOLO Confidence",
    0.10,
    0.90,
    0.25,
    0.05,
)

model_path, is_custom = find_model_path()

if is_custom:
    st.sidebar.success(
        f"Custom disaster model loaded:\n{model_path}"
    )
else:
    st.sidebar.info(
        f"Pretrained model:\n{model_path}\n\n"
        "For disaster-specific classes, add "
        "`models/disaster_best.pt`."
    )


# ============================================================
# SATELLITE UI
# ============================================================

st.markdown(
    '<div class="section-title">🛰️ Satellite Damage Analysis</div>',
    unsafe_allow_html=True,
)

sat1, sat2 = st.columns(2)

with sat1:
    st.markdown("### Pre-Disaster")
    pre_file = st.file_uploader(
        "Upload pre-disaster satellite image",
        type=["jpg", "jpeg", "png"],
        key="pre_upload",
    )

with sat2:
    st.markdown("### Post-Disaster")
    post_file = st.file_uploader(
        "Upload post-disaster satellite image",
        type=["jpg", "jpeg", "png"],
        key="post_upload",
    )


if pre_file is not None and post_file is not None:
    if st.button(
        "🚨 ANALYZE DISASTER DAMAGE",
        width="stretch",
    ):
        with st.spinner("Running OpenCV satellite analysis..."):
            try:
                pre = uploaded_to_cv(pre_file)
                post = uploaded_to_cv(post_file)

                pre, post = resize_images(pre, post)

                aligned_post, aligned, matches = align_images(
                    pre, post
                )

                difference = calculate_difference(
                    pre, aligned_post
                )

                damage_map = create_damage_map(
                    difference,
                    low_threshold,
                    moderate_threshold,
                    critical_threshold,
                    minimum_area,
                )

                if show_responders:
                    damage_map = add_responder_markers(
                        damage_map,
                        responder_points,
                    )

                percentages = damage_percentages(
                    difference,
                    low_threshold,
                    moderate_threshold,
                    critical_threshold,
                )

                st.session_state.satellite_result = {
                    "pre": pre,
                    "post": aligned_post,
                    "difference": difference,
                    "damage_map": damage_map,
                    "percentages": percentages,
                    "aligned": aligned,
                    "matches": matches,
                }

            except Exception as e:
                st.error(f"Satellite analysis failed: {e}")


# ============================================================
# SATELLITE RESULTS
# ============================================================

if st.session_state.satellite_result is not None:
    result = st.session_state.satellite_result

    st.markdown("---")

    if result["aligned"]:
        st.success(
            f"✅ OpenCV ORB alignment successful — "
            f"{result['matches']} good feature matches."
        )
    else:
        st.warning(
            f"⚠️ Reliable automatic alignment was not found "
            f"({result['matches']} good matches). "
            "The comparison continued after resizing."
        )

    st.markdown(
        '<div class="section-title">🗺️ Damage Intensity Map</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "🔴 Critical &nbsp;&nbsp; 🟠 Moderate &nbsp;&nbsp; "
        "🟡 Low &nbsp;&nbsp; 🟢 Safe &nbsp;&nbsp; ⛑️ Responder"
    )

    a, b, c = st.columns(3)

    with a:
        st.markdown("### BEFORE")
        st.image(
            cv_to_rgb(result["pre"]),
            width="stretch",
        )

    with b:
        st.markdown("### AFTER")
        st.image(
            cv_to_rgb(result["post"]),
            width="stretch",
        )

    with c:
        st.markdown("### DAMAGE MAP")
        st.image(
            cv_to_rgb(result["damage_map"]),
            width="stretch",
        )

    p = result["percentages"]

    st.markdown("---")
    st.markdown("### 📊 Damage Statistics")

    cols = st.columns(4)
    values = [
        ("🟢 SAFE", p["Safe"]),
        ("🟡 LOW", p["Low Damage"]),
        ("🟠 MODERATE", p["Moderate Damage"]),
        ("🔴 CRITICAL", p["Critical"]),
    ]

    for col, (label, value) in zip(cols, values):
        with col:
            st.markdown(
                f"""
                <div class="metric">
                    <div class="metric-title">{label}</div>
                    <div class="metric-value">{value:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    graph = pd.DataFrame({
        "Zone": [
            "Safe",
            "Low Damage",
            "Moderate Damage",
            "Critical",
        ],
        "Percentage": [
            p["Safe"],
            p["Low Damage"],
            p["Moderate Damage"],
            p["Critical"],
        ],
    })

    st.markdown("### 📈 Disaster Intensity Distribution")
    st.bar_chart(
        graph.set_index("Zone"),
        height=320,
    )

    result_rgb = cv_to_rgb(result["damage_map"])
    result_pil = Image.fromarray(result_rgb)

    buffer = BytesIO()
    result_pil.save(buffer, format="PNG")

    st.download_button(
        "⬇️ Download Damage Map",
        buffer.getvalue(),
        "disaster_damage_map.png",
        "image/png",
        width="stretch",
    )


# ============================================================
# DRONE SCAN HISTORY UI
# ============================================================

st.markdown("---")
st.markdown('<div class="section-title">🚁 Drone Change Detection & Scan History</div>', unsafe_allow_html=True)
st.markdown("""
<div class="note">
Prototype mode: upload one new drone image for each scan. The first scan is stored as the baseline. Every later scan is automatically compared with the previous saved scan, and the image, damage map and statistics are retained locally.
</div>
""", unsafe_allow_html=True)

drone_file = st.file_uploader("📡 Upload simulated drone scan", type=["jpg", "jpeg", "png"], key="drone_scan_upload")

d1, d2 = st.columns(2)
with d1:
    process_drone = st.button("🚁 REGISTER NEW DRONE SCAN", width="stretch")
with d2:
    if st.button("🔄 Refresh Scan History", width="stretch"):
        st.session_state.drone_history = load_drone_history()
        st.rerun()

if st.session_state.drone_history is None:
    st.session_state.drone_history = load_drone_history()

if drone_file is not None and process_drone:
    current_hash = drone_image_hash(drone_file)
    if current_hash == st.session_state.drone_last_hash:
        st.info("This drone image is already registered. Upload a new image for the next scan.")
    else:
        with st.spinner("Registering drone scan and running change detection..."):
            try:
                current_image = uploaded_to_cv(drone_file)
                history = load_drone_history()
                previous_record = history[-1] if history else None
                analysis = analyze_drone_scan(current_image, previous_record, low_threshold, moderate_threshold, critical_threshold, minimum_area, responder_points, show_responders)
                scan_number = len(history) + 1
                scan_id = f"scan_{scan_number:03d}"
                timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
                image_path = DRONE_SCANS / f"{scan_id}.png"
                map_path = DRONE_SCANS / f"{scan_id}_damage_map.png"
                save_drone_image(analysis["current"], image_path)
                save_drone_image(analysis["damage_map"], map_path)
                stats = drone_damage_summary(analysis["percentages"])
                previous_stats = previous_record.get("statistics", {}) if previous_record else {}
                record = {
                    "scan_id": scan_id,
                    "timestamp": timestamp,
                    "image_path": str(image_path),
                    "damage_map_path": str(map_path),
                    "compared_with": previous_record["scan_id"] if previous_record else None,
                    "baseline": not bool(previous_record),
                    "alignment_successful": bool(analysis["aligned"]),
                    "feature_matches": int(analysis["matches"]),
                    "statistics": stats,
                    "previous_statistics": previous_stats,
                    "image_hash": current_hash,
                }
                save_drone_record(record)
                st.session_state.drone_last_hash = current_hash
                st.session_state.drone_history = load_drone_history()
                st.session_state.drone_current_analysis = {**analysis, "record": record}
                if record["baseline"]:
                    st.success(f"✅ {scan_id} saved. This scan is now the baseline.")
                else:
                    st.success(f"✅ {scan_id} saved and automatically compared with {record['compared_with']}.")
            except Exception as e:
                st.error(f"Drone scan failed: {e}")

drone_history = st.session_state.drone_history or []
if drone_history:
    latest = drone_history[-1]
    st.markdown("### 📡 Latest Drone Scan")
    latest_analysis = st.session_state.drone_current_analysis
    if latest_analysis is not None and latest_analysis.get("record", {}).get("scan_id") == latest.get("scan_id"):
        analysis = latest_analysis
        a, b, c = st.columns(3)
        with a:
            st.image(cv_to_rgb(analysis["previous"] if analysis["previous"] is not None else analysis["current"]), caption=f"Previous: {latest['compared_with']}" if latest["compared_with"] else "Baseline drone scan", width="stretch")
        with b:
            st.image(cv_to_rgb(analysis["current"]), caption=f"Current: {latest['scan_id']}", width="stretch")
        with c:
            st.image(cv_to_rgb(analysis["damage_map"]), caption="Automatic change / damage map", width="stretch")
        if latest["baseline"]:
            st.info("🟢 Baseline created. The next drone scan will be automatically compared with this saved scan.")
        elif latest["alignment_successful"]:
            st.success(f"✅ Compared with {latest['compared_with']} using ORB alignment ({latest['feature_matches']} good matches).")
        else:
            st.warning("⚠️ Reliable automatic alignment was not found; comparison continued after resizing.")
        p = latest["statistics"]
        st.markdown("#### 📊 Current Scan Damage Statistics")
        stat_cols = st.columns(5)
        stat_values = [("🟢 SAFE", p["Safe"]), ("🟡 LOW", p["Low Damage"]), ("🟠 MODERATE", p["Moderate Damage"]), ("🔴 CRITICAL", p["Critical"]), ("⚠️ AFFECTED", p["Affected Area"])]
        for col, (label, value) in zip(stat_cols, stat_values):
            with col:
                st.markdown(f'<div class="metric"><div class="metric-title">{label}</div><div class="metric-value">{value:.1f}%</div></div>', unsafe_allow_html=True)
        if not latest["baseline"] and latest.get("previous_statistics"):
            prev = latest["previous_statistics"]
            st.markdown("#### 📈 Change Since Previous Scan")
            c1, c2, c3 = st.columns(3)
            c1.metric("Affected Area", f"{p['Affected Area']:.1f}%", f"{p['Affected Area']-prev.get('Affected Area',0):+.1f} pp")
            c2.metric("Critical", f"{p['Critical']:.1f}%", f"{p['Critical']-prev.get('Critical',0):+.1f} pp")
            c3.metric("Moderate", f"{p['Moderate Damage']:.1f}%", f"{p['Moderate Damage']-prev.get('Moderate Damage',0):+.1f} pp")
    st.markdown("### 🗂️ Drone Scan History")
    rows=[]
    for r in drone_history:
        s=r["statistics"]
        rows.append({"Scan":r["scan_id"],"Time":r["timestamp"],"Compared With":r["compared_with"] or "Baseline","Affected":f"{s['Affected Area']:.1f}%","Critical":f"{s['Critical']:.1f}%","Moderate":f"{s['Moderate Damage']:.1f}%","Low":f"{s['Low Damage']:.1f}%","Safe":f"{s['Safe']:.1f}%"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.download_button("⬇️ Download Drone Scan History", json.dumps(drone_history, indent=2), "drone_scan_history.json", "application/json", width="stretch")
    st.markdown('<div class="note">Prototype note: scan files are stored in <code>drone_history/</code>. For permanent production history on cloud hosting, use a database or object storage.</div>', unsafe_allow_html=True)
else:
    st.info("No drone scans saved yet. Upload your first simulated drone image — it will automatically become the baseline.")


# ============================================================
# CAMERA UI
# ============================================================

st.markdown("---")
st.markdown(
    '<div class="section-title">📷 AI Camera Disaster Assessment</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="note">
    Capture an affected-area photo. The system combines YOLO object
    detection with OpenCV visual screening. Generic pretrained YOLO
    classes are useful for people and vehicles; disaster-specific
    classes require a custom model.
    </div>
    """,
    unsafe_allow_html=True,
)

camera_image = st.camera_input(
    "📷 Capture affected area",
    key="disaster_camera",
)

if camera_image is not None:
    if st.button(
        "🔍 ANALYZE CAMERA IMAGE",
        width="stretch",
    ):
        with st.spinner("Running AI + OpenCV camera analysis..."):
            try:
                camera_cv = uploaded_to_cv(camera_image)

                cv_result = opencv_scene_analysis(
                    camera_cv
                )

                yolo_image, detections, yolo_status = run_yolo(
                    camera_cv,
                    model_path,
                    camera_conf,
                )

                final_image, risk, status, reasons, category_counts = (
                    combine_camera_results(
                        camera_cv,
                        yolo_image,
                        cv_result,
                        detections,
                    )
                )

                st.session_state.camera_result = {
                    "image": final_image,
                    "risk": risk,
                    "status": status,
                    "reasons": reasons,
                    "detections": detections,
                    "category_counts": category_counts,
                    "opencv": cv_result,
                    "yolo_status": yolo_status,
                }

            except Exception as e:
                st.error(f"Camera analysis failed: {e}")


# ============================================================
# CAMERA RESULTS
# ============================================================

if st.session_state.camera_result is not None:
    cam = st.session_state.camera_result

    st.markdown("---")
    st.markdown(
        '<div class="section-title">📷 AI Camera Results</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.3, 1])

    with left:
        st.image(
            cv_to_rgb(cam["image"]),
            caption="YOLO + OpenCV detection overlay",
            width="stretch",
        )

    with right:
        st.markdown(
            f"""
            <div class="metric">
                <div class="metric-title">OVERALL SCREENING</div>
                <div class="metric-value">{cam["status"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.progress(cam["risk"] / 100)
        st.write(f"**Visual risk score:** {cam['risk']}/100")

        if cam["reasons"]:
            st.markdown("#### ⚠️ Indicators")
            for reason in cam["reasons"]:
                st.write(f"• {reason}")
        else:
            st.success("No major visual indicators detected.")

        cv_result = cam["opencv"]

        st.metric(
            "Crack-like edge regions",
            cv_result["crack_like"],
        )

        st.metric(
            "Water-like area",
            f"{cv_result['water_percentage']:.1f}%",
        )

        st.metric(
            "Debris/obstruction indicator",
            "Possible"
            if cv_result["debris_possible"]
            else "Not prominent",
        )

    st.markdown("### 🤖 AI Detections")

    if cam["detections"]:
        rows = []

        for d in cam["detections"]:
            rows.append({
                "Detected": d["class"],
                "Project Category": d["category"],
                "Confidence": f"{d['confidence'] * 100:.1f}%",
            })

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(
            "No YOLO objects passed the confidence threshold."
        )

    st.markdown(
        """
        ⚠️ **Prototype note:** this is an emergency-screening
        demonstration, not a structural-safety certification.
        OpenCV color/edge heuristics can produce false positives.
        Disaster-specific YOLO classes require a custom annotated dataset.
        """,
    )
