import cv2
import numpy as np


def align_images(pre_image, post_image):
    """
    Aligns post-disaster image with pre-disaster image
    using ORB feature matching + Homography.
    """
    height, width = pre_image.shape[:2]
    post_image = cv2.resize(post_image, (width, height))

    gray_pre = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
    gray_post = cv2.cvtColor(post_image, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=5000)

    keypoints_pre, descriptors_pre = orb.detectAndCompute(gray_pre, None)
    keypoints_post, descriptors_post = orb.detectAndCompute(gray_post, None)

    if descriptors_pre is None or descriptors_post is None:
        return post_image, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors_pre, descriptors_post)
    matches = sorted(matches, key=lambda x: x.distance)

    good_matches = matches[:100]
    match_count = len(good_matches)

    if match_count < 10:
        return post_image, match_count

    src_points = np.float32(
        [keypoints_post[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    dst_points = np.float32(
        [keypoints_pre[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    matrix, _ = cv2.findHomography(
        src_points, dst_points, cv2.RANSAC, 5.0
    )

    if matrix is None:
        return post_image, match_count

    aligned_image = cv2.warpPerspective(
        post_image, matrix, (width, height)
    )

    return aligned_image, match_count


def draw_responder_helmet_badge(img, center, size=24, label="RESPONDERS ON-SITE"):
    """
    Renders an emergency rescue helmet with plus sign (⛑️ badge) at the given center coordinates.
    """
    x, y = int(center[0]), int(center[1])
    r = int(size)

    # 1. High-contrast outer dropshadow
    cv2.circle(img, (x, y), r + 5, (10, 10, 10), -1)

    # 2. White outer circular ring
    cv2.circle(img, (x, y), r + 2, (255, 255, 255), -1)

    # 3. Vivid Rescue Red Helmet Base Circle
    helmet_red = (30, 30, 220)  # BGR Red
    cv2.circle(img, (x, y), r, helmet_red, -1)

    # 4. Reflective Safety Brim Line (Gold/Yellow)
    brim_y = y + int(r * 0.22)
    brim_w = int(r * 0.75)
    cv2.line(
        img,
        (x - brim_w, brim_y),
        (x + brim_w, brim_y),
        (0, 215, 255),
        max(2, int(r * 0.14))
    )

    # 5. Helmet Dome 3D Highlight Curvature
    axes = (int(r * 0.65), int(r * 0.52))
    cv2.ellipse(
        img,
        (x, y - int(r * 0.08)),
        axes,
        0,
        190,
        350,
        (255, 255, 255),
        1
    )

    # 6. Bold White Rescue Cross (+) inside helmet
    cross_color = (255, 255, 255)
    pw = max(2, int(r * 0.24))
    pl = int(r * 0.56)
    cy = y - int(r * 0.12)

    # Vertical bar
    cv2.rectangle(
        img,
        (x - pw // 2, cy - pl // 2),
        (x + pw // 2 + (pw % 2), cy + pl // 2),
        cross_color,
        -1
    )
    # Horizontal bar
    cv2.rectangle(
        img,
        (x - pl // 2, cy - pw // 2),
        (x + pl // 2, cy + pw // 2 + (pw % 2)),
        cross_color,
        -1
    )

    # 7. Descriptive Badge Tag below helmet
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.35, min(0.5, r / 50.0))
        thick = 1
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
        tag_y = y + r + th + 8

        # Tag background container
        pad_x = 6
        pad_y = 4
        cv2.rectangle(
            img,
            (x - tw // 2 - pad_x, tag_y - th - pad_y),
            (x + tw // 2 + pad_x, tag_y + pad_y),
            (15, 15, 15),
            -1
        )
        cv2.rectangle(
            img,
            (x - tw // 2 - pad_x, tag_y - th - pad_y),
            (x + tw // 2 + pad_x, tag_y + pad_y),
            (0, 230, 0),
            1
        )
        cv2.putText(
            img,
            label,
            (x - tw // 2, tag_y - 1),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA
        )


def segment_damage_zones(damage_mask, difference, min_zone_area=250):
    """
    Clusters pixel-level damage into distinct tactical zones and categorizes them:
    - Critical (Red, filled circle)
    - Moderate (Orange, filled circle)
    - Minor (Yellow, filled circle)
    """
    total_pixels = damage_mask.size
    effective_min_area = max(min_zone_area, int(total_pixels * 0.0004))

    # Morphological dilation to group adjacent damage sites into coherent zones
    cluster_kernel = np.ones((25, 25), np.uint8)
    grouped_mask = cv2.dilate(damage_mask, cluster_kernel, iterations=2)

    contours, _ = cv2.findContours(
        grouped_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    raw_zones = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < effective_min_area:
            continue

        (cx, cy), radius = cv2.minEnclosingCircle(cnt)
        cx, cy, radius = int(cx), int(cy), max(25, int(radius))

        # Compute zone damage statistics
        zone_mask = np.zeros(damage_mask.shape, dtype=np.uint8)
        cv2.circle(zone_mask, (cx, cy), radius, 255, -1)

        damaged_pixels = cv2.countNonZero(
            cv2.bitwise_and(damage_mask, damage_mask, mask=zone_mask)
        )
        mean_diff = cv2.mean(difference, mask=zone_mask)[0]

        # Severity score combines damaged area and change intensity
        circle_area = np.pi * (radius ** 2)
        density = (damaged_pixels / circle_area) if circle_area > 0 else 0
        score = damaged_pixels * (mean_diff / 255.0) * (1.0 + density)

        raw_zones.append({
            "center": (cx, cy),
            "radius": radius,
            "damaged_pixels": damaged_pixels,
            "mean_intensity": round(mean_diff, 1),
            "density": round(density * 100, 1),
            "score": round(score, 2),
            "area": int(area)
        })

    # Sort descending by damage score
    raw_zones.sort(key=lambda z: z["score"], reverse=True)

    # Palette definitions (BGR):
    # Critical -> Vivid Red
    # Moderate -> Vivid Orange
    # Minor -> Vivid Yellow
    colors = {
        "Critical": (25, 25, 230),    # Red
        "Moderate": (0, 135, 255),   # Orange
        "Minor": (0, 220, 255)       # Yellow
    }

    num_zones = len(raw_zones)
    categorized_zones = []

    for i, z in enumerate(raw_zones):
        zone_id = f"Zone {i + 1}"
        if num_zones == 1:
            severity = "Critical" if z["score"] > 500 else "Moderate"
        elif num_zones == 2:
            severity = "Critical" if i == 0 else "Moderate"
        elif num_zones == 3:
            severity = ["Critical", "Moderate", "Minor"][i]
        else:
            # 4 or more zones: distribute into 3 tiers
            c_thresh = max(1, num_zones // 3)
            m_thresh = max(c_thresh + 1, (2 * num_zones) // 3)
            if i < c_thresh:
                severity = "Critical"
            elif i < m_thresh:
                severity = "Moderate"
            else:
                severity = "Minor"

        z_data = {
            "id": zone_id,
            "severity": severity,
            "color_bgr": colors[severity],
            "center": z["center"],
            "radius": z["radius"],
            "score": z["score"],
            "damaged_pixels": z["damaged_pixels"],
            "mean_intensity": z["mean_intensity"],
            "density": z["density"],
            "responders_reached": (i == 0)  # Default first zone as reached
        }
        categorized_zones.append(z_data)

    return categorized_zones


def render_tactical_damage_map(
    base_image,
    zones,
    damage_mask,
    responders_status=None,
    safe_area_opacity=0.35,
    zone_fill_opacity=0.48,
    show_safe_area=True
):
    """
    Renders the tactical map with:
    1. Safe Area: Entire unaffected region tinted Green ("Safe area pura green rahega")
    2. Critical Zones: Red circle with filled Red color (not just outline)
    3. Moderate Zones: Orange circle with filled Orange color
    4. Minor Zones: Yellow circle with filled Yellow color
    5. Responders Reached: Emergency helmet symbol with plus badge (⛑️)
    """
    result = base_image.copy()
    h, w = result.shape[:2]

    if responders_status is None:
        responders_status = {}

    # -------------------------------------------------------------
    # 1. SAFE AREA: PURA GREEN
    # -------------------------------------------------------------
    if show_safe_area:
        # Build mask of all damage circles
        danger_mask = np.zeros((h, w), dtype=np.uint8)
        for z in zones:
            cv2.circle(danger_mask, z["center"], z["radius"], 255, -1)

        # Combine with raw damage mask to ensure no damaged spots are treated as safe
        danger_mask = cv2.bitwise_or(danger_mask, damage_mask)
        safe_mask = cv2.bitwise_not(danger_mask)

        # Create emerald green layer (BGR: 40, 185, 45)
        green_layer = np.full_like(result, (40, 185, 45))
        blended_safe = cv2.addWeighted(
            result,
            1.0 - safe_area_opacity,
            green_layer,
            safe_area_opacity,
            0
        )
        result[safe_mask > 0] = blended_safe[safe_mask > 0]

    # -------------------------------------------------------------
    # 2. DAMAGE ZONES: FILLED CIRCLES (RED, ORANGE, YELLOW)
    # -------------------------------------------------------------
    fill_layer = result.copy()
    for z in zones:
        cv2.circle(
            fill_layer,
            z["center"],
            z["radius"],
            z["color_bgr"],
            -1  # Filled circle!
        )

    # Alpha-blend the filled circles so underlying imagery remains clearly visible
    result = cv2.addWeighted(
        result,
        1.0 - zone_fill_opacity,
        fill_layer,
        zone_fill_opacity,
        0
    )

    # -------------------------------------------------------------
    # 3. ZONE BORDERS, LABELS & ⛑️ RESPONDER HELMET BADGES
    # -------------------------------------------------------------
    for z in zones:
        cx, cy = z["center"]
        r = z["radius"]
        color = z["color_bgr"]
        zid = z["id"]
        sev = z["severity"]

        # Solid high-visibility perimeter ring
        cv2.circle(result, (cx, cy), r, color, 3)
        cv2.circle(result, (cx, cy), r + 2, (15, 15, 15), 1)

        # Zone Header Tag at top of circle
        lbl = f"{zid}: {sev}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.45
        thick = 1
        (tw, th), _ = cv2.getTextSize(lbl, font, scale, thick)
        tag_y = max(th + 12, cy - r - 8)

        cv2.rectangle(
            result,
            (cx - tw // 2 - 6, tag_y - th - 5),
            (cx + tw // 2 + 6, tag_y + 4),
            (15, 15, 15),
            -1
        )
        cv2.rectangle(
            result,
            (cx - tw // 2 - 6, tag_y - th - 5),
            (cx + tw // 2 + 6, tag_y + 4),
            color,
            1
        )
        cv2.putText(
            result,
            lbl,
            (cx - tw // 2, tag_y - 2),
            font,
            scale,
            (255, 255, 255),
            thick,
            cv2.LINE_AA
        )

        # Check responder status
        is_reached = responders_status.get(zid, z.get("responders_reached", False))

        if is_reached:
            # Render ⛑️ Rescue Helmet with Plus Symbol at zone center
            badge_size = max(20, min(36, int(r * 0.26)))
            draw_responder_helmet_badge(
                result,
                (cx, cy),
                size=badge_size,
                label="RESPONDERS ON-SITE"
            )
        else:
            # Draw target crosshair / status marker for pending zones
            cv2.circle(result, (cx, cy), 5, (255, 255, 255), -1)
            cv2.circle(result, (cx, cy), 6, (15, 15, 15), 1)
            pending_lbl = "Awaiting Responders"
            font_p = cv2.FONT_HERSHEY_SIMPLEX
            (ptw, pth), _ = cv2.getTextSize(pending_lbl, font_p, 0.35, 1)
            py = cy + 16
            cv2.rectangle(
                result,
                (cx - ptw // 2 - 4, py - pth - 3),
                (cx + ptw // 2 + 4, py + 3),
                (20, 20, 20),
                -1
            )
            cv2.putText(
                result,
                pending_lbl,
                (cx - ptw // 2, py - 1),
                font_p,
                0.35,
                (220, 220, 220),
                1,
                cv2.LINE_AA
            )

    return result


def detect_damage_from_single_image(image, responders_status=None):
    """
    Detects structural damage, fractures, debris, and hazard zones
    from a single camera capture or live video frame without requiring a pre-disaster image.
    Categorizes the scene into Critical (Red), Moderate (Orange), Minor (Yellow),
    and Safe Ground (Green) with ⛑️ responder tracking.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Structural edge & crack detection
    edges = cv2.Canny(gray, 40, 140)

    # 2. Local texture gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    # 3. High-frequency variance
    blur = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = cv2.absdiff(gray, blur)

    # Combined hazard score
    hazard_score = (0.45 * (edges / 255.0) + 0.30 * (morph_grad / 255.0) + 0.25 * (diff / 255.0))
    hazard_uint8 = np.uint8(np.clip(hazard_score * 255 * 1.6, 0, 255))

    thresh_val = int(np.mean(hazard_uint8) + 0.7 * np.std(hazard_uint8))
    _, hazard_mask = cv2.threshold(hazard_uint8, thresh_val, 255, cv2.THRESH_BINARY)

    # Morphological cleaning
    clean_k = np.ones((5, 5), np.uint8)
    hazard_mask = cv2.morphologyEx(hazard_mask, cv2.MORPH_OPEN, clean_k)
    hazard_mask = cv2.morphologyEx(hazard_mask, cv2.MORPH_CLOSE, clean_k)

    # Cluster hazards
    cluster_k = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(hazard_mask, cluster_k, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = max(300, int(h * w * 0.002))
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), max(28, int(r))

        c_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(c_mask, (cx, cy), r, 255, -1)
        cnt_px = cv2.countNonZero(cv2.bitwise_and(hazard_mask, hazard_mask, mask=c_mask))
        mean_haz = cv2.mean(hazard_uint8, mask=c_mask)[0]
        score = cnt_px * (mean_haz / 255.0)

        candidates.append({
            "center": (cx, cy),
            "radius": r,
            "score": round(score, 2),
            "damaged_pixels": cnt_px,
            "mean_intensity": round(mean_haz, 1),
            "density": round((cnt_px / (np.pi * r * r)) * 100, 1),
            "area": int(area)
        })

    candidates.sort(key=lambda x: x["score"], reverse=True)
    top_candidates = candidates[:3]

    # If camera image is very uniform, create a representative focal zone
    if not top_candidates:
        top_candidates.append({
            "center": (w // 2, h // 2),
            "radius": min(w, h) // 4,
            "score": 100.0,
            "damaged_pixels": 800,
            "mean_intensity": 45.0,
            "density": 15.0,
            "area": 4000
        })

    colors = {
        "Critical": (25, 25, 230),    # Red
        "Moderate": (0, 135, 255),   # Orange
        "Minor": (0, 220, 255)       # Yellow
    }

    categorized_zones = []
    for i, z in enumerate(top_candidates):
        sev = ["Critical", "Moderate", "Minor"][min(i, 2)]
        categorized_zones.append({
            "id": f"Zone {i + 1}",
            "severity": sev,
            "color_bgr": colors[sev],
            "center": z["center"],
            "radius": z["radius"],
            "score": z["score"],
            "damaged_pixels": z["damaged_pixels"],
            "mean_intensity": z["mean_intensity"],
            "density": z["density"],
            "responders_reached": (i == 0)
        })

    total_pixels = hazard_mask.size
    damaged_pixels = np.count_nonzero(hazard_mask)
    damage_percentage = (damaged_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    if damage_percentage < 5:
        intensity = "Low"
    elif damage_percentage < 20:
        intensity = "Moderate"
    elif damage_percentage < 40:
        intensity = "High"
    else:
        intensity = "Severe"

    heatmap = cv2.applyColorMap(hazard_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.65, heatmap, 0.35, 0)

    if responders_status is None:
        responders_status = {z["id"]: z["responders_reached"] for z in categorized_zones}

    tactical_map = render_tactical_damage_map(
        image,
        categorized_zones,
        hazard_mask,
        responders_status=responders_status
    )

    return {
        "aligned": image,
        "difference": diff,
        "mask": hazard_mask,
        "heatmap": heatmap,
        "overlay": overlay,
        "tactical_map": tactical_map,
        "zones": categorized_zones,
        "responders_status": responders_status,
        "damage_percentage": damage_percentage,
        "intensity": intensity,
        "match_count": 0
    }


def detect_damage(pre_image, post_image, responders_status=None):
    """
    Detects changed/damaged regions between pre-disaster and post-disaster images.
    Returns aligned image, damage mask, heatmap, tactical zone map, and zone metadata.
    """
    # 1. ALIGN IMAGES
    aligned_post, match_count = align_images(pre_image, post_image)

    # 2. GRAYSCALE
    pre_gray = cv2.cvtColor(pre_image, cv2.COLOR_BGR2GRAY)
    post_gray = cv2.cvtColor(aligned_post, cv2.COLOR_BGR2GRAY)

    # 3. REDUCE NOISE
    pre_gray = cv2.GaussianBlur(pre_gray, (5, 5), 0)
    post_gray = cv2.GaussianBlur(post_gray, (5, 5), 0)

    # 4. PIXEL DIFFERENCE
    difference = cv2.absdiff(pre_gray, post_gray)

    # 5. THRESHOLD
    _, damage_mask = cv2.threshold(difference, 35, 255, cv2.THRESH_BINARY)

    # 6. MORPHOLOGICAL CLEANING
    kernel = np.ones((5, 5), np.uint8)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_OPEN, kernel)
    damage_mask = cv2.morphologyEx(damage_mask, cv2.MORPH_CLOSE, kernel)

    # 7. OVERALL DAMAGE PERCENTAGE
    total_pixels = damage_mask.size
    damaged_pixels = np.count_nonzero(damage_mask)
    damage_percentage = (damaged_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    # 8. DAMAGE INTENSITY
    if damage_percentage < 5:
        intensity = "Low"
    elif damage_percentage < 20:
        intensity = "Moderate"
    elif damage_percentage < 40:
        intensity = "High"
    else:
        intensity = "Severe"

    # 9. HEATMAP & STANDARD OVERLAY
    heatmap = cv2.applyColorMap(damage_mask, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(aligned_post, 0.65, heatmap, 0.35, 0)

    # 10. TACTICAL ZONES SEGMENTATION & RENDERING
    zones = segment_damage_zones(damage_mask, difference)

    # If caller provided initial responder states, apply them
    if responders_status is None:
        responders_status = {z["id"]: z["responders_reached"] for z in zones}

    tactical_map = render_tactical_damage_map(
        aligned_post,
        zones,
        damage_mask,
        responders_status=responders_status
    )

    return {
        "aligned": aligned_post,
        "difference": difference,
        "mask": damage_mask,
        "heatmap": heatmap,
        "overlay": overlay,
        "tactical_map": tactical_map,
        "zones": zones,
        "responders_status": responders_status,
        "damage_percentage": damage_percentage,
        "intensity": intensity,
        "match_count": match_count
    }