import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
from pathlib import Path

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
<p>Satellite change detection + AI camera assessment + line/crack analysis + responder mapping</p>
</div>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "satellite_result" not in st.session_state:
    st.session_state.satellite_result = None

if "camera_result" not in st.session_state:
    st.session_state.camera_result = None


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
    result[low_mask] = (0, 255, 255)
    result[moderate_mask] = (0, 140, 255)
    result[critical_mask] = (0, 0, 255)

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

    # Filled critical circles
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
# OPENCV CAMERA: LINE + CRACK DETECTION
# ============================================================

def detect_lines_and_cracks(image):
    """
    OpenCV visual screening:
    1. Canny detects edges.
    2. HoughLinesP detects strong straight/structural lines.
    3. Morphological processing highlights thin irregular crack-like structures.
    4. Results are drawn on a dedicated overlay.

    This is a visual screening method, not a validated structural-crack detector.
    """

    output = image.copy()
    line_overlay = image.copy()

    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve contrast and suppress small camera noise.
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # ---------------- CANNY EDGES ----------------
    edges = cv2.Canny(gray, 60, 150)

    # Close small gaps so crack-like structures become continuous.
    crack_kernel = np.ones((3, 3), np.uint8)
    crack_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        crack_kernel,
        iterations=1,
    )

    # ---------------- HOUGH LINE DETECTION ----------------
    min_line_length = max(30, int(min(h, w) * 0.08))

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=min_line_length,
        maxLineGap=12,
    )

    line_count = 0
    line_lengths = []

    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)

            length = float(
                np.hypot(x2 - x1, y2 - y1)
            )

            # Ignore tiny/noisy lines.
            if length < min_line_length:
                continue

            cv2.line(
                line_overlay,
                (x1, y1),
                (x2, y2),
                (255, 0, 0),
                2,
                cv2.LINE_AA,
            )

            line_count += 1
            line_lengths.append(length)

    # ---------------- CRACK-LIKE DETECTION ----------------
    # Thin edge structures are emphasized using a smaller kernel.
    thin_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )

    contours, _ = cv2.findContours(
        thin_edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    crack_like = 0
    crack_boxes = []

    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, False)

        if perimeter <= 0:
            continue

        x, y, cw, ch = cv2.boundingRect(contour)

        # Thin/elongated edge candidate.
        aspect = max(cw, ch) / max(1, min(cw, ch))
        density = perimeter / max(1.0, area)

        is_candidate = (
            perimeter > 90
            and max(cw, ch) > 25
            and aspect > 2.0
            and density > 0.12
        )

        if is_candidate:
            # Avoid treating huge image borders as cracks.
            if cw < 0.55 * w and ch < 0.55 * h:
                crack_like += 1
                crack_boxes.append((x, y, cw, ch))

                cv2.rectangle(
                    output,
                    (x, y),
                    (x + cw, y + ch),
                    (0, 0, 255),
                    2,
                )

                cv2.putText(
                    output,
                    "CRACK-LIKE",
                    (x, max(18, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

    # Blend detected structural lines into the main image.
    output = cv2.addWeighted(
        output,
        0.78,
        line_overlay,
        0.55,
        0,
    )

    # Make a clean black/white edge view for the UI.
    edge_view = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Highlight crack candidates in the edge view.
    for x, y, cw, ch in crack_boxes:
        cv2.rectangle(
            edge_view,
            (x, y),
            (x + cw, y + ch),
            (0, 0, 255),
            2,
        )

    # ---------------- WATER-LIKE REGIONS ----------------
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_water = np.array([80, 40, 40])
    upper_water = np.array([135, 255, 255])

    water_mask = cv2.inRange(
        hsv,
        lower_water,
        upper_water,
    )

    water_pixels = np.sum(water_mask > 0)
    water_percentage = water_pixels / (h * w) * 100

    water_contours, _ = cv2.findContours(
        water_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    water_overlay = output.copy()

    for contour in water_contours:
        if cv2.contourArea(contour) > 500:
            cv2.drawContours(
                water_overlay,
                [contour],
                -1,
                (255, 0, 0),
                3,
            )

    output = cv2.addWeighted(
        output,
        0.85,
        water_overlay,
        0.30,
        0,
    )

    # ---------------- DARK OBSTRUCTION / DEBRIS INDICATOR ----------------
    dark_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 0]),
        np.array([180, 255, 65]),
    )

    dark_percentage = np.sum(dark_mask > 0) / (h * w) * 100
    debris_possible = dark_percentage > 15

    # ---------------- SUMMARY BANNER ----------------
    avg_line_length = (
        float(np.mean(line_lengths))
        if line_lengths
        else 0.0
    )

    cv2.rectangle(
        output,
        (0, 0),
        (w, 40),
        (7, 17, 31),
        -1,
    )

    cv2.putText(
        output,
        f"LINES: {line_count} | CRACK-LIKE: {crack_like}",
        (14, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return {
        "image": output,
        "edge_view": edge_view,
        "line_overlay": line_overlay,
        "crack_like": crack_like,
        "line_count": line_count,
        "avg_line_length": avg_line_length,
        "water_percentage": water_percentage,
        "debris_possible": debris_possible,
        "edges": edges,
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
    Supports both COCO classes and future custom disaster classes.
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


# ============================================================
# COMBINE CAMERA RESULTS
# ============================================================

def combine_camera_results(
    image,
    yolo_image,
    cv_result,
    detections,
):
    """
    Combines:
    - YOLO object boxes
    - OpenCV Hough line detection
    - OpenCV crack-like screening
    - water/debris heuristics

    Risk score is only a visual screening indicator.
    """

    # Start from YOLO result so its boxes remain visible.
    output = yolo_image.copy()

    # Add Hough lines on top.
    line_overlay = cv_result["line_overlay"]
    output = cv2.addWeighted(
        output,
        0.78,
        line_overlay,
        0.38,
        0,
    )

    # Draw crack-like regions clearly.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 150)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )

    h, w = image.shape[:2]

    for contour in contours:
        perimeter = cv2.arcLength(contour, False)
        x, y, cw, ch = cv2.boundingRect(contour)

        if perimeter <= 90:
            continue

        aspect = max(cw, ch) / max(1, min(cw, ch))
        if (
            max(cw, ch) > 25
            and aspect > 2.0
            and cw < 0.55 * w
            and ch < 0.55 * h
        ):
            cv2.rectangle(
                output,
                (x, y),
                (x + cw, y + ch),
                (0, 0, 255),
                2,
            )

    risk = 0
    reasons = []

    category_counts = {}

    for d in detections:
        cat = d["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # YOLO indicators
    if category_counts.get("Fire / Smoke", 0) > 0:
        risk += 40
        reasons.append("fire/smoke detected by YOLO")

    if category_counts.get("Flood / Water", 0) > 0:
        risk += 35
        reasons.append("water/flood indicator detected by YOLO")

    if category_counts.get("Debris / Rubble", 0) > 0:
        risk += 30
        reasons.append("debris/rubble detected by YOLO")

    if category_counts.get("Building Damage", 0) > 0:
        risk += 45
        reasons.append("building-damage class detected by YOLO")

    if category_counts.get("Structural Crack", 0) > 0:
        risk += 35
        reasons.append("crack class detected by YOLO")

    # OpenCV indicators
    if cv_result["water_percentage"] > 8:
        risk += 20
        reasons.append("water-like pixels detected")

    if cv_result["crack_like"] > 10:
        risk += 15
        reasons.append("multiple crack-like edge structures")

    if cv_result["line_count"] > 20:
        risk += 5
        reasons.append("multiple structural/edge lines detected")

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

st.sidebar.markdown(
    "**Camera pipeline:**\n"
    "Canny → Hough Lines → Crack-like screening → YOLO"
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
        use_container_width=True,
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
            use_container_width=True,
        )

    with b:
        st.markdown("### AFTER")
        st.image(
            cv_to_rgb(result["post"]),
            use_container_width=True,
        )

    with c:
        st.markdown("### DAMAGE MAP")
        st.image(
            cv_to_rgb(result["damage_map"]),
            use_container_width=True,
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
        use_container_width=True,
    )


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
    detection with OpenCV line detection, crack-like screening,
    water analysis and obstruction screening.
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
        use_container_width=True,
    ):
        with st.spinner("Running YOLO + OpenCV line/crack analysis..."):
            try:
                camera_cv = uploaded_to_cv(camera_image)

                cv_result = detect_lines_and_cracks(
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

    left, right = st.columns([1.35, 1])

    with left:
        st.image(
            cv_to_rgb(cam["image"]),
            caption="YOLO boxes + Hough lines + crack-like regions",
            use_container_width=True,
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

    # ---------------- OPENCV VISUAL ANALYSIS ----------------
    cv_result = cam["opencv"]

    st.markdown("### 🔬 OpenCV Line & Crack Analysis")

    metric_cols = st.columns(4)

    with metric_cols[0]:
        st.metric(
            "Detected Lines",
            cv_result["line_count"],
        )

    with metric_cols[1]:
        st.metric(
            "Crack-like Regions",
            cv_result["crack_like"],
        )

    with metric_cols[2]:
        st.metric(
            "Avg Line Length",
            f"{cv_result['avg_line_length']:.0f}px",
        )

    with metric_cols[3]:
        st.metric(
            "Water-like Area",
            f"{cv_result['water_percentage']:.1f}%",
        )

    line_col, edge_col = st.columns(2)

    with line_col:
        st.image(
            cv_to_rgb(cv_result["line_overlay"]),
            caption="Hough line detection",
            use_container_width=True,
        )

    with edge_col:
        st.image(
            cv_to_rgb(cv_result["edge_view"]),
            caption="Canny edges + crack-like regions",
            use_container_width=True,
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
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No YOLO objects passed the confidence threshold."
        )

    st.markdown(
        """
        ⚠️ **Prototype note:** line/crack detection is an OpenCV
        visual screening method and can produce false positives from
        edges, shadows, textures and image noise. YOLO disaster classes
        such as fire, flood, debris, building damage and structural
        cracks require a custom annotated model.
        """,
    )
