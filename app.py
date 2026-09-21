

from pathlib import Path
from datetime import datetime
import json
import math
import time
import hashlib
import threading
import queue
import requests

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

try:
    import folium
    from folium.plugins import (
        AntPath,
        Fullscreen,
        LocateControl,
        MeasureControl,
        MousePosition,
        Geocoder,
    )
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except Exception:
    FOLIUM_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except Exception:
    MAVLINK_AVAILABLE = False

try:
    import av
    from streamlit_webrtc import webrtc_streamer, WebRtcMode
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False

st.set_page_config(
    page_title="Disaster Management Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------- LOGIN --------------------
LOGIN_USERNAME = "admin"
LOGIN_PASSWORD = "disaster123"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

def login_page():
    st.markdown(
        """
        <div style="
            max-width:500px;
            margin:90px auto 20px auto;
            padding:35px;
            background:#081522;
            border:1px solid #173650;
            border-radius:18px;
            text-align:center;
            box-shadow:0 10px 40px rgba(0,0,0,.35);
        ">
            <div style="font-size:3rem;">🛡️</div>
            <h1 style="margin-bottom:5px;">Disaster Response</h1>
            <p style="color:#6e91ad;">COMMAND CENTER LOGIN</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("🔐 LOGIN", width="stretch", type="primary"):
            if username == LOGIN_USERNAME and password == LOGIN_PASSWORD:
                st.session_state.logged_in = True
                st.rerun()
            else:
                st.error("Invalid username or password")

if not st.session_state.logged_in:
    login_page()
    st.stop()

# -------------------- END LOGIN --------------------


st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:#030914;color:#eaf4ff}
.block-container{padding:2.6rem 1rem 1.2rem;max-width:1900px}
[data-testid="stSidebar"]{background:#06111e;border-right:1px solid #17314b}
h1,h2,h3,h4{color:#f4f8ff!important} p,span,label{color:#a9bdd1}
.topbar{background:linear-gradient(90deg,#071626,#081d31);border:1px solid #173c5b;border-radius:16px;padding:14px 18px;margin-bottom:12px;box-shadow:0 8px 30px rgba(0,0,0,.25)}
.brand{font-size:1.45rem;font-weight:900;letter-spacing:.05em;color:#f6fbff}.subbrand{font-size:.72rem;letter-spacing:.13em;color:#6e91ad;margin-top:2px}
.status-pill{display:inline-flex;gap:7px;align-items:center;background:#06251f;border:1px solid #0d6b57;color:#53e3ae;border-radius:999px;padding:7px 12px;font-weight:800;font-size:.78rem}
.status-dot{width:8px;height:8px;border-radius:50%;background:#41e29e;box-shadow:0 0 10px #41e29e;display:inline-block}
.panel{background:linear-gradient(180deg,#081522,#06111c);border:1px solid #173650;border-radius:15px;padding:12px;margin-bottom:12px;box-shadow:0 8px 24px rgba(0,0,0,.20)}
.panel-head{display:flex;justify-content:space-between;align-items:center;font-weight:850;letter-spacing:.05em;color:#dcecff;margin-bottom:9px}.eyebrow{font-size:.68rem;color:#5e829e;letter-spacing:.13em;font-weight:800}
.live-badge{font-size:.68rem;color:#5df0b4;border:1px solid #16634f;background:#06251e;border-radius:999px;padding:4px 8px}
.danger-badge{font-size:.68rem;color:#ff7b80;border:1px solid #71333b;background:#291018;border-radius:999px;padding:4px 8px}
.stat-card{background:#071523;border:1px solid #17354f;border-radius:13px;padding:10px 12px;min-height:88px}.stat-label{font-size:.66rem;letter-spacing:.1em;color:#6e8ba2}.stat-value{font-size:1.35rem;font-weight:900;color:#eff8ff;margin-top:4px}.stat-note{font-size:.68rem;color:#78a0ba}
.alert-row{border:1px solid #18364e;background:#071624;border-radius:11px;padding:9px;margin:7px 0}.alert-title{font-weight:800;color:#eaf5ff;font-size:.82rem}.alert-detail{font-size:.7rem;color:#7f9cb3}.coord{font-family:monospace;color:#bce7ff;font-size:.72rem}
.kpi-green{color:#55e5aa}.kpi-red{color:#ff6d73}.kpi-orange{color:#ffb24e}.kpi-cyan{color:#52dfff}
[data-testid="stMetric"]{background:#071522;border:1px solid #173650;border-radius:12px}
.stButton>button{background:#091c2c;border:1px solid #214862;color:#eaf5ff;border-radius:9px;min-height:36px;font-weight:700}.stButton>button:hover{border-color:#31c9ff;background:#0d2940}
[data-testid="stFileUploader"]{background:#071522;border:1px dashed #28506c;border-radius:10px}
[data-testid="stTabs"] button{color:#9fb8ca!important}.small-note{font-size:.68rem;color:#63829a}.map-chip{display:inline-block;padding:5px 8px;border-radius:7px;background:#0b2031;border:1px solid #1a415b;font-size:.68rem;color:#b9d8eb;margin-right:5px}
.progress-wrap{height:8px;background:#10283a;border-radius:99px;overflow:hidden}.progress-fill{height:100%;background:linear-gradient(90deg,#13c8ff,#42e5ae);border-radius:99px}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dashboard_data"
IMG_DIR = DATA_DIR / "images"
FOOTAGE_DIR = DATA_DIR / "footage"
MAP_DIR = DATA_DIR / "maps"
LOG_DIR = DATA_DIR / "logs"
PREPOST_DIR = DATA_DIR / "prepost"
for d in (DATA_DIR, IMG_DIR, FOOTAGE_DIR, MAP_DIR, LOG_DIR, PREPOST_DIR):
    d.mkdir(parents=True, exist_ok=True)

ALERTS_FILE = LOG_DIR / "alerts.json"
EVENTS_FILE = LOG_DIR / "events.json"
AID_FILE = LOG_DIR / "aid_drops.json"
TELEMETRY_FILE = LOG_DIR / "telemetry.json"
SCANS_FILE = LOG_DIR / "scans.json"

DEFAULT_GPS = {"lat":23.2599,"lon":77.4126,"altitude":80.0,"heading":45.0,"source":"Demo Simulator","timestamp":None}

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def uid(prefix):
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{int(time.time()*1000)%1000}"

def load_json(path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        pass
    return default

def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

def append_json(path, item, limit=1000):
    data = load_json(path, [])
    data.append(item)
    save_json(path, data[-limit:])

def nearest_pre_scan(lat, lon, radius_km=2.0):
    scans = load_json(SCANS_FILE, [])
    candidates = []
    for scan in scans:
        if scan.get("type") != "Pre-Disaster" or not scan.get("image_path"):
            continue
        if not Path(scan["image_path"]).exists():
            continue
        d = haversine(lat, lon, scan["lat"], scan["lon"])
        if d <= radius_km:
            candidates.append((d, scan))
    return min(candidates, key=lambda x: x[0]) if candidates else None

def cv_image(upload):
    return cv2.cvtColor(np.array(Image.open(upload).convert("RGB")), cv2.COLOR_RGB2BGR)

def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None

def resize_pair(a,b,width=1000):
    if a is None or b is None: return None,None
    if a.shape[1] > width:
        a=cv2.resize(a,(width,int(a.shape[0]*width/a.shape[1])),interpolation=cv2.INTER_AREA)
    if b.shape[1] > width:
        b=cv2.resize(b,(width,int(b.shape[0]*width/b.shape[1])),interpolation=cv2.INTER_AREA)
    h=min(a.shape[0],b.shape[0]); w=min(a.shape[1],b.shape[1])
    return cv2.resize(a,(w,h)),cv2.resize(b,(w,h))

def haversine(lat1,lon1,lat2,lon2):
    r=6371.0
    p1,p2=math.radians(lat1),math.radians(lat2)
    a=math.sin(math.radians(lat2-lat1)/2)**2+math.cos(p1)*math.cos(p2)*math.sin(math.radians(lon2-lon1)/2)**2
    return r*2*math.asin(math.sqrt(a))

def generate_gps():
    t=time.time()
    return {"lat":round(23.2599+math.sin(t/18)*.003,6),
            "lon":round(77.4126+math.cos(t/20)*.003,6),
            "altitude":round(75+math.sin(t/11)*12,1),
            "heading":round((t*7)%360,1),"source":"Demo Simulator","timestamp":now()}

def get_gps(source, lat, lon, alt, heading, endpoint=None, baud=57600):
    if source=="Demo Simulator": return generate_gps(),None
    if source=="MAVLink":
        if not MAVLINK_AVAILABLE: return None,"pymavlink is not installed."
        try:
            c=mavutil.mavlink_connection(endpoint,baud=baud)
            m=c.recv_match(type="GLOBAL_POSITION_INT",blocking=True,timeout=3)
            if m is None: return None,"No GPS telemetry received."
            return {"lat":round(m.lat/1e7,6),"lon":round(m.lon/1e7,6),
                    "altitude":round(m.relative_alt/1000,2),"heading":round(m.hdg/100,2),
                    "source":"MAVLink","timestamp":now()},None
        except Exception as e: return None,str(e)
    return {"lat":float(lat),"lon":float(lon),"altitude":float(alt),
            "heading":float(heading),"source":"Manual","timestamp":now()},None

def align_images(ref,cur):
    g1=cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY); g2=cv2.cvtColor(cur,cv2.COLOR_BGR2GRAY)
    orb=cv2.ORB_create(nfeatures=3000)
    k1,d1=orb.detectAndCompute(g1,None); k2,d2=orb.detectAndCompute(g2,None)
    if d1 is None or d2 is None: return cur,False
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2,d1,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.75*n.distance]
    if len(good)<8: return cur,False
    src=np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst=np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    H,_=cv2.findHomography(src,dst,cv2.RANSAC,5)
    if H is None:return cur,False
    h,w=ref.shape[:2]
    return cv2.warpPerspective(cur,H,(w,h)),True

def damage_compare(pre,post,critical=100,moderate=65,low=35):
    pre,post=resize_pair(pre,post)
    if pre is None:return None,None,None
    aligned,_=align_images(pre,post)
    a=cv2.GaussianBlur(cv2.cvtColor(pre,cv2.COLOR_BGR2GRAY),(5,5),0)
    b=cv2.GaussianBlur(cv2.cvtColor(aligned,cv2.COLOR_BGR2GRAY),(5,5),0)
    diff=cv2.normalize(cv2.absdiff(a,b),None,0,255,cv2.NORM_MINMAX)
    mask=(diff>=low).astype(np.uint8)*255
    k=np.ones((5,5),np.uint8)
    mask=cv2.morphologyEx(cv2.morphologyEx(mask,cv2.MORPH_OPEN,k),cv2.MORPH_CLOSE,k)
    cmap=np.zeros_like(aligned)
    crit=diff>=critical; mod=(diff>=moderate)&~crit; lowm=(diff>=low)&~crit&~mod; safe=~(crit|mod|lowm)
    cmap[crit]=[0,0,255]; cmap[mod]=[0,165,255]; cmap[lowm]=[0,255,255]; cmap[safe]=aligned[safe]//4
    stats={}
    total=max(diff.size,1)
    for name,m in [("Critical",crit),("Moderate",mod),("Low",lowm),("Safe",safe)]:
        stats[name]=round(float(np.sum(m))/total*100,2)
    return cmap,diff,stats

@st.cache_resource(show_spinner=False)
def load_model():
    if not YOLO_AVAILABLE:return None
    for p in [BASE_DIR/"models"/"disaster_best.pt",BASE_DIR/"models"/"best.pt",BASE_DIR/"best.pt"]:
        if p.exists():
            try:return YOLO(str(p))
            except Exception:pass
    for name in ["yolo11n.pt","yolov8n.pt"]:
        try:return YOLO(name)
        except Exception:pass
    return None

def run_yolo(img,conf=.35):
    model=load_model()
    if model is None:return img,[]
    try:
        r=model.predict(img,conf=conf,verbose=False)[0]
        det=[]
        for b in r.boxes:
            xy=b.xyxy[0].cpu().numpy().tolist()
            det.append({"Class":str(r.names[int(b.cls[0])]),"Confidence":round(float(b.conf[0]),3),
                        "X1":round(xy[0]),"Y1":round(xy[1]),"X2":round(xy[2]),"Y2":round(xy[3])})
        return r.plot(),det
    except Exception:
        return img,[]

def estimate_position(gps,bbox,w,h,hfov=70,vfov=50):
    x1,y1,x2,y2=bbox
    dx=((x1+x2)/2/w-.5)*hfov
    dy=((y1+y2)/2/h-.5)*vfov
    alt=max(float(gps["altitude"]),1)
    east=alt*math.tan(math.radians(dx)); north=alt*math.tan(math.radians(-dy))
    hd=math.radians(float(gps["heading"]))
    e=east*math.sin(hd)+north*math.cos(hd)
    n=east*math.cos(hd)-north*math.sin(hd)
    lat=gps["lat"]+n/111320
    lon=gps["lon"]+e/(111320*math.cos(math.radians(gps["lat"])))
    return round(lat,6),round(lon,6)

def make_alert(det,gps,scan_id,w,h,hfov=70,vfov=50):
    lat,lon=estimate_position(gps,(det["X1"],det["Y1"],det["X2"],det["Y2"]),w,h,hfov,vfov)
    return {"alert_id":uid("ALERT"),"type":"Possible Survivor","scan_id":scan_id,
            "latitude":lat,"longitude":lon,"confidence":det["Confidence"],
            "timestamp":now(),"gps_source":gps["source"],"image_path":None}

def parse_map_coordinates(text):
    """Parse a latitude,longitude search string."""
    try:
        parts = [x.strip() for x in text.split(",")]
        if len(parts) != 2:
            return None
        lat = float(parts[0])
        lon = float(parts[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return [lat, lon]
    except Exception:
        pass
    return None


def geocode_location(query):
    """Online place-name search using Nominatim; coordinate search works offline."""
    try:
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "limit": 1},
            headers={"User-Agent": "disaster-damage-detector/1.0"},
            timeout=5,
        )
        response.raise_for_status()
        results = response.json()
        if results:
            return [float(results[0]["lat"]), float(results[0]["lon"])]
    except Exception:
        pass
    return None


def create_interactive_operation_map(
    gps,
    telemetry,
    responders=None,
    survivor_points=None,
    zones=None,
    offline_image=None,
):
    """Build the Google-Maps-style interactive disaster operation map."""
    if not FOLIUM_AVAILABLE:
        return None

    responders = responders or []
    survivor_points = survivor_points or []
    zones = zones or []

    lat = float(gps.get("lat", DEFAULT_GPS["lat"]))
    lon = float(gps.get("lon", DEFAULT_GPS["lon"]))

    if "map_center" not in st.session_state:
        st.session_state.map_center = [lat, lon]
    if "map_zoom" not in st.session_state:
        st.session_state.map_zoom = 15

    center = st.session_state.map_center
    zoom = int(st.session_state.map_zoom)

    m = folium.Map(
        location=center,
        zoom_start=zoom,
        tiles=None,
        control_scale=True,
        prefer_canvas=True,
    )

    # Normal street map.
    folium.TileLayer(
        tiles="OpenStreetMap",
        name="Street Map",
        overlay=False,
        control=True,
    ).add_to(m)

    # Satellite imagery layer. This is an online layer; the offline image
    # overlay below remains available when a local orthophoto is uploaded.
    folium.TileLayer(
        tiles=(
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        ),
        attr="Esri World Imagery",
        name="Satellite",
        overlay=False,
        control=True,
    ).add_to(m)

    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Terrain",
        overlay=False,
        control=True,
    ).add_to(m)

    # Google-Maps-style controls.
    Fullscreen(position="topleft").add_to(m)
    LocateControl(auto_start=False, position="topleft").add_to(m)
    MeasureControl(
        position="topleft",
        primary_length_unit="kilometers",
        secondary_length_unit="meters",
    ).add_to(m)
    MousePosition(
        position="bottomright",
        separator=" | ",
        prefix="GPS:",
        num_digits=6,
    ).add_to(m)
    Geocoder(
        collapsed=True,
        position="topright",
        add_marker=True,
    ).add_to(m)

    # -------------------------
    # Drone
    # -------------------------
    drone_html = """
    <div style="font-size:28px;transform:translate(-50%,-50%);text-shadow:0 0 7px #00e5ff;">
        🛸
    </div>
    """

    drone_marker = folium.Marker(
        [lat, lon],
        tooltip="LIVE DRONE D1",
        popup=folium.Popup(
            f"""
            <b>DRONE D1</b><br>
            Latitude: {lat:.6f}<br>
            Longitude: {lon:.6f}<br>
            Altitude: {float(gps.get('altitude', 0)):.1f} m<br>
            Heading: {float(gps.get('heading', 0)):.1f}°<br>
            Source: {gps.get('source', '--')}
            """,
            max_width=280,
        ),
        icon=folium.DivIcon(html=drone_html),
    )
    drone_marker.add_to(m)

    # -------------------------
    # Flight trail
    # -------------------------
    path = []
    for point in telemetry[-100:]:
        try:
            path.append([float(point["lat"]), float(point["lon"])])
        except Exception:
            continue

    if not path:
        path = [[lat, lon]]
    elif path[-1] != [lat, lon]:
        path.append([lat, lon])

    if len(path) >= 2:
        AntPath(
            locations=path,
            color="#00d9ff",
            pulse_color="#ffffff",
            weight=4,
            opacity=0.9,
            delay=700,
        ).add_to(m)
        trail_line = folium.PolyLine(
            path,
            color="#00d9ff",
            weight=2,
            opacity=0.45,
            tooltip="Drone flight trail",
        )
        trail_line.add_to(m)

        # Smooth client-side drone animation. The marker and live trail move
        # inside the Leaflet iframe, so Streamlit does NOT rerun/rebuild the
        # map every frame. This keeps the map stable and removes flicker.
        if str(gps.get("source", "")) == "Demo Simulator":
            import json as _json
            animation_path = _json.dumps(path)
            drone_js = drone_marker.get_name()
            trail_js = trail_line.get_name()
            animation_script = f"""
            <script>
            (function() {{
                const marker = {drone_js};
                const trail = {trail_js};
                const route = {animation_path};
                if (!marker || !trail || route.length < 2) return;

                let segment = 0;
                let progress = 0;
                let last = performance.now();
                const speed = 0.00055;

                function animate(now) {{
                    const dt = Math.min((now - last) / 1000, 0.08);
                    last = now;
                    progress += dt * speed;

                    while (progress >= 1) {{
                        progress -= 1;
                        segment += 1;
                        if (segment >= route.length - 1) segment = 0;
                    }}

                    const a = route[segment];
                    const b = route[segment + 1];
                    const lat = a[0] + (b[0] - a[0]) * progress;
                    const lon = a[1] + (b[1] - a[1]) * progress;

                    marker.setLatLng([lat, lon]);

                    // Growing trail: keep the travelled section plus the
                    // current interpolated position.
                    const travelled = route.slice(0, segment + 1);
                    travelled.push([lat, lon]);
                    trail.setLatLngs(travelled);

                    requestAnimationFrame(animate);
                }}

                requestAnimationFrame(animate);
            }})();
            </script>
            """
            from branca.element import Element
            m.get_root().html.add_child(Element(animation_script))

    # -------------------------
    # Responders
    # -------------------------
    responder_group = folium.FeatureGroup(name="Responders", show=True)
    for i, responder in enumerate(responders, 1):
        try:
            r_lat = float(responder["lat"])
            r_lon = float(responder["lon"])
        except Exception:
            continue

        responder_html = f"""
        <div style="
            background:#101b2a;
            border:2px solid #22c55e;
            border-radius:50%;
            width:32px;height:32px;
            display:flex;align-items:center;justify-content:center;
            font-size:17px;
            box-shadow:0 0 10px rgba(34,197,94,.8);
        ">🪖</div>
        """

        folium.Marker(
            [r_lat, r_lon],
            tooltip=f"RESPONDER R{i}",
            popup=f"<b>RESPONDER R{i}</b><br>Lat: {r_lat:.6f}<br>Lon: {r_lon:.6f}",
            icon=folium.DivIcon(html=responder_html),
        ).add_to(responder_group)
    responder_group.add_to(m)

    # -------------------------
    # Survivor locations
    # -------------------------
    survivor_group = folium.FeatureGroup(name="Survivor Alerts", show=True)
    for i, point in enumerate(survivor_points, 1):
        try:
            s_lat = float(point["lat"])
            s_lon = float(point["lon"])
        except Exception:
            continue

        folium.Marker(
            [s_lat, s_lon],
            tooltip=f"SURVIVOR {i}",
            popup=f"<b>🚨 SURVIVOR DETECTED</b><br>Lat: {s_lat:.6f}<br>Lon: {s_lon:.6f}",
            icon=folium.Icon(color="red", icon="plus", prefix="fa"),
        ).add_to(survivor_group)
    survivor_group.add_to(m)

    # -------------------------
    # Damage zones
    # -------------------------
    zone_group = folium.FeatureGroup(name="Damage Zones", show=True)
    zone_colors = {
        "critical": "#ef4444",
        "moderate": "#f59e0b",
        "low": "#facc15",
        "safe": "#22c55e",
    }

    for zone in zones:
        try:
            zone_points = zone["points"]
            kind = str(zone.get("kind", "moderate")).lower()
            label = zone.get("label", kind.upper())
            color = zone_colors.get(kind, "#f59e0b")
            folium.Polygon(
                locations=[[float(a), float(b)] for a, b in zone_points],
                color=color,
                weight=2,
                fill=True,
                fill_color=color,
                fill_opacity=0.28,
                tooltip=label,
                popup=f"<b>{label}</b>",
            ).add_to(zone_group)
        except Exception:
            continue
    zone_group.add_to(m)

    # -------------------------
    # Approximate offline orthophoto overlay
    # -------------------------
    if offline_image is not None:
        try:
            radius_km = 2.0
            lat_delta = radius_km / 111.32
            lon_delta = radius_km / (
                111.32 * max(math.cos(math.radians(lat)), 0.1)
            )
            bounds = [
                [lat - lat_delta, lon - lon_delta],
                [lat + lat_delta, lon + lon_delta],
            ]
            folium.raster_layers.ImageOverlay(
                image=cv2.cvtColor(offline_image, cv2.COLOR_BGR2RGB),
                bounds=bounds,
                opacity=0.55,
                name="Offline Orthophoto",
                interactive=True,
            ).add_to(m)
        except Exception:
            pass

    folium.LayerControl(position="topright", collapsed=False).add_to(m)
    return m

# Thread-shared live state. Streamlit UI never writes from the WebRTC callback.
# The WebRTC callback owns camera capture, person detection, survivor snapshots,
# movement tracking, coordinate updates, and segmented video recording.
LIVE_LOCK=threading.Lock()
LIVE_FRAME=None
LIVE_DETECTIONS=[]
LIVE_EVENTS=queue.Queue()
LIVE_LAST_INFER=0.0
LIVE_VIDEO_WRITER=None
LIVE_VIDEO_PATH=None
LIVE_VIDEO_STARTED=0.0
LIVE_VIDEO_FRAMES=0
LIVE_VIDEO_SEGMENT_SECONDS=60
LIVE_GPS=DEFAULT_GPS.copy()

# Survivor tracking:
# - A newly detected person gets one captured image + coordinate immediately.
# - A stationary person gets no repeated coordinate alerts.
# - A moving person gets a coordinate update at most every 5 seconds.
LIVE_TRACKS={}
LIVE_NEXT_TRACK_ID=1
LIVE_TRACK_MATCH_METERS=25.0
LIVE_MOVEMENT_THRESHOLD_METERS=5.0
LIVE_COORD_UPDATE_SECONDS=5.0
LIVE_TRACK_TIMEOUT_SECONDS=30.0


def _open_live_video_writer(width, height):
    """Open a new 60-second MJPG/AVI segment for reliable server-side recording."""
    global LIVE_VIDEO_WRITER, LIVE_VIDEO_PATH, LIVE_VIDEO_STARTED, LIVE_VIDEO_FRAMES
    try:
        if LIVE_VIDEO_WRITER is not None:
            LIVE_VIDEO_WRITER.release()
    except Exception:
        pass
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    path=FOOTAGE_DIR/f"rgb_live_{stamp}.avi"
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"MJPG"),20.0,(int(width),int(height)))
    if not writer.isOpened():
        path=FOOTAGE_DIR/f"rgb_live_{stamp}.mp4"
        writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),20.0,(int(width),int(height)))
    if writer.isOpened():
        LIVE_VIDEO_WRITER=writer
        LIVE_VIDEO_PATH=path
        LIVE_VIDEO_STARTED=time.time()
        LIVE_VIDEO_FRAMES=0
        return True
    LIVE_VIDEO_WRITER=None
    LIVE_VIDEO_PATH=None
    return False


def _record_live_frame(image):
    """Record every incoming RGB frame and rotate files every 60 seconds."""
    global LIVE_VIDEO_FRAMES
    h,w=image.shape[:2]
    if LIVE_VIDEO_WRITER is None or time.time()-LIVE_VIDEO_STARTED >= LIVE_VIDEO_SEGMENT_SECONDS:
        _open_live_video_writer(w,h)
    if LIVE_VIDEO_WRITER is not None:
        try:
            LIVE_VIDEO_WRITER.write(image)
            LIVE_VIDEO_FRAMES += 1
        except Exception:
            pass


def _match_survivor_track(lat,lon,used_tracks):
    """Match a detection to the nearest active survivor track."""
    best_id=None
    best_distance=float("inf")
    now_ts=time.time()
    for track_id,track in LIVE_TRACKS.items():
        if track_id in used_tracks:
            continue
        if now_ts-track["last_seen"] > LIVE_TRACK_TIMEOUT_SECONDS:
            continue
        distance=haversine(track["lat"],track["lon"],lat,lon)*1000.0
        if distance <= LIVE_TRACK_MATCH_METERS and distance < best_distance:
            best_id=track_id
            best_distance=distance
    return best_id


def _register_survivor_detection(det,gps,image_width,image_height,used_tracks=None):
    """Return tracking/report information for one person detection."""
    global LIVE_NEXT_TRACK_ID

    lat,lon=estimate_position(
        gps,
        (det["X1"],det["Y1"],det["X2"],det["Y2"]),
        image_width,
        image_height,
    )
    current_time=time.time()
    if used_tracks is None:
        used_tracks=set()

    track_id=_match_survivor_track(lat,lon,used_tracks)
    if track_id is None:
        track_id=f"S{LIVE_NEXT_TRACK_ID}"
        LIVE_NEXT_TRACK_ID+=1
        LIVE_TRACKS[track_id]={
            "lat":lat,
            "lon":lon,
            "last_seen":current_time,
            "last_report":0.0,
            "last_report_lat":lat,
            "last_report_lon":lon,
            "ever_reported":False,
        }
        used_tracks.add(track_id)
        return track_id,lat,lon,True,False

    track=LIVE_TRACKS[track_id]
    movement_from_last=haversine(track["lat"],track["lon"],lat,lon)*1000.0
    movement_from_report=haversine(track["last_report_lat"],track["last_report_lon"],lat,lon)*1000.0

    track["lat"]=lat
    track["lon"]=lon
    track["last_seen"]=current_time
    used_tracks.add(track_id)

    first_report=not track["ever_reported"]
    moving=movement_from_report >= LIVE_MOVEMENT_THRESHOLD_METERS

    report_now=False
    if first_report:
        report_now=True
    elif moving and current_time-track["last_report"] >= LIVE_COORD_UPDATE_SECONDS:
        report_now=True

    return track_id,lat,lon,report_now,moving


def _save_survivor_snapshot_and_alert(det,gps,annotated,image_shape,track_id,lat,lon,is_location_update=False):
    """Save a survivor image only for detection events and log its coordinates."""
    event_id=uid("LIVE")
    image_path=None

    # Only the initial human detection creates a new captured image.
    # Movement updates send coordinates without creating duplicate images.
    if not is_location_update:
        image_path=FOOTAGE_DIR/f"{event_id}.jpg"
        cv2.imwrite(str(image_path),annotated)

    alert={
        "alert_id":event_id,
        "type":"Survivor location update" if is_location_update else "Possible Survivor",
        "scan_id":event_id,
        "track_id":track_id,
        "latitude":lat,
        "longitude":lon,
        "confidence":det["Confidence"],
        "timestamp":now(),
        "gps_source":gps["source"],
        "image_path":str(image_path) if image_path else None,
        "is_location_update":is_location_update,
    }
    append_json(ALERTS_FILE,alert)
    append_json(EVENTS_FILE,{
        "event_id":event_id,
        "type":"Survivor location update" if is_location_update else "Survivor detected",
        "track_id":track_id,
        "latitude":lat,
        "longitude":lon,
        "confidence":det["Confidence"],
        "timestamp":alert["timestamp"],
        "image_path":alert["image_path"],
    },500)
    return alert


def live_callback(frame):
    global LIVE_FRAME,LIVE_DETECTIONS,LIVE_LAST_INFER

    image=frame.to_ndarray(format="bgr24")
    annotated=image.copy()

    # Continuous video recording remains unchanged.
    with LIVE_LOCK:
        _record_live_frame(image)
        LIVE_FRAME=image.copy()

    # Run AI inference periodically so the live video remains smooth.
    if time.time()-LIVE_LAST_INFER < 0.45:
        return av.VideoFrame.from_ndarray(annotated,format="bgr24")

    LIVE_LAST_INFER=time.time()
    det=[]
    model=load_model()

    if model is not None:
        try:
            result=model.predict(image,conf=.35,verbose=False)[0]
            for b in result.boxes:
                xy=b.xyxy[0].cpu().numpy().tolist()
                label=str(result.names[int(b.cls[0])])
                score=float(b.conf[0])

                # RGB live AI remains survivor-only.
                if label.lower() != "person":
                    continue

                item={
                    "Class":label,
                    "Confidence":round(score,3),
                    "X1":round(xy[0]),
                    "Y1":round(xy[1]),
                    "X2":round(xy[2]),
                    "Y2":round(xy[3]),
                }
                det.append(item)

                x1,y1,x2,y2=map(int,xy)
                cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,80,255),2)
                cv2.putText(
                    annotated,
                    f"SURVIVOR {score:.0%}",
                    (x1,max(20,y1-8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    .55,
                    (0,80,255),
                    2,
                )
        except Exception:
            det=[]

    with LIVE_LOCK:
        LIVE_FRAME=image.copy()
        LIVE_DETECTIONS=det
        gps=LIVE_GPS.copy()

    # Track every detected person independently.
    used_tracks=set()
    for person in det:
        try:
            track_id,lat,lon,report_now,moving=_register_survivor_detection(
                person,
                gps,
                image.shape[1],
                image.shape[0],
                used_tracks,
            )
            track=LIVE_TRACKS[track_id]

            if report_now:
                is_update=bool(track["ever_reported"])
                alert=_save_survivor_snapshot_and_alert(
                    person,
                    gps,
                    annotated,
                    image.shape,
                    track_id,
                    lat,
                    lon,
                    is_location_update=is_update,
                )

                track["last_report"]=time.time()
                track["last_report_lat"]=lat
                track["last_report_lon"]=lon
                track["ever_reported"]=True

                LIVE_EVENTS.put({
                    "event_id":alert["alert_id"],
                    "alerts":[alert],
                    "moving":moving,
                })
        except Exception:
            # One problematic detection should never stop the live video.
            continue

    # Remove stale survivor tracks so old people do not get matched forever.
    current_time=time.time()
    stale=[tid for tid,t in LIVE_TRACKS.items()
           if current_time-t["last_seen"] > LIVE_TRACK_TIMEOUT_SECONDS]
    for tid in stale:
        LIVE_TRACKS.pop(tid,None)

    return av.VideoFrame.from_ndarray(annotated,format="bgr24")


def responder_state(gps):
    """Demo responder telemetry; replace with live responder GPS when connected."""
    t=time.time()
    specs=[
        ("R-01", .0010,-.0012,"Sector A", "MOVING",78),
        ("R-02",-.0014,.0009,"Sector B", "ON SITE",64),
        ("R-03", .0004,.0018,"Sector C", "MOVING",83),
        ("R-04",-.0008,-.0020,"Sector D", "STANDBY",91),
        ("R-05", .0018,.0002,"Support", "RETURNING",57),
    ]
    out=[]
    for i,(rid,dy,dx,sector,status,batt) in enumerate(specs):
        phase=t/(9+i*2)
        move_lat=math.sin(phase)*.00035 if status in ("MOVING","RETURNING") else 0
        move_lon=math.cos(phase)*.00035 if status in ("MOVING","RETURNING") else 0
        out.append({"id":rid,"lat":gps["lat"]+dy+move_lat,"lon":gps["lon"]+dx+move_lon,
                    "sector":sector,"status":status,"battery":max(15,min(99,batt+int(4*math.sin(t/17+i))))})
    return out


def damage_zones_from_diff(diff,gps,min_area_ratio=.002,threshold=35):
    """Convert connected high-difference regions into approximate map zones.
    Geometry is based on the uploaded image and drone FOV/altitude, so it is an
    estimate until the image has calibrated georeferencing/orthorectification.
    """
    if diff is None:return []
    h,w=diff.shape[:2]
    mask=(diff>=threshold).astype(np.uint8)*255
    kernel=np.ones((7,7),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    n,labels,stats,cent=cv2.connectedComponentsWithStats(mask,8)
    zones=[]
    min_area=max(80,int(h*w*min_area_ratio))
    gps_lat=float(gps["lat"]); gps_lon=float(gps["lon"]); alt=max(float(gps.get("altitude",80)),1)
    hfov=70; vfov=50
    hd=math.radians(float(gps.get("heading",0)))
    for idx in range(1,n):
        x,y,ww,hh,area=stats[idx]
        if area<min_area: continue
        ratio=area/(h*w)
        if ratio>=.08: kind,label="critical","CRITICAL DAMAGE"
        elif ratio>=.03: kind,label="moderate","MODERATE DAMAGE"
        else: kind,label="low","LOW DAMAGE"
        pts=[]
        for px,py in [(x,y),(x+ww,y),(x+ww,y+hh),(x,y+hh)]:
            dx=((px/w)-.5)*hfov; dy=((py/h)-.5)*vfov
            east=alt*math.tan(math.radians(dx)); north=alt*math.tan(math.radians(-dy))
            e=east*math.sin(hd)+north*math.cos(hd); nn=east*math.cos(hd)-north*math.sin(hd)
            pts.append((gps_lat+nn/111320,gps_lon+e/(111320*math.cos(math.radians(gps_lat)))))
        zones.append({"kind":kind,"label":f"{label} • {ratio*100:.1f}% pixels","points":pts,"area_percent":round(ratio*100,2)})
    return sorted(zones,key=lambda z:z["area_percent"],reverse=True)[:8]


def mission_stats(alerts,zones,responders,aid_records,scanned_area=4.2):
    critical=sum(z.get("area_percent",0) for z in zones if z.get("kind")=="critical")
    survivor_alerts=[a for a in alerts if not a.get("is_location_update")]
    progress=min(99,int(35+len(zones)*6+len(survivor_alerts)*4+len(aid_records)*3))
    active=sum(r.get("status") in ("MOVING","ON SITE") for r in responders)
    return {"progress":progress,"survivors":len(survivor_alerts),"responders_active":active,
            "responders_total":len(responders),"zones":len(zones),"critical_pct":critical,
            "area_scanned":scanned_area,"aid_drops":len(aid_records)}


# -------------------- SESSION STATE --------------------
if "gps" not in st.session_state: st.session_state.gps=DEFAULT_GPS.copy()
if "pre" not in st.session_state: st.session_state.pre=None
if "post" not in st.session_state: st.session_state.post=None
if "last_result" not in st.session_state: st.session_state.last_result=None
if "damage_zones" not in st.session_state: st.session_state.damage_zones=[]
if "active_panel" not in st.session_state: st.session_state.active_panel=None
if "selected_survivor" not in st.session_state: st.session_state.selected_survivor=None
if "selected_map_coord" not in st.session_state: st.session_state.selected_map_coord=None
if "offline_map" not in st.session_state: st.session_state.offline_map=None
if "offline_map_upload_hash" not in st.session_state: st.session_state.offline_map_upload_hash=None
if "map_version" not in st.session_state: st.session_state.map_version=0
if "toast_seen" not in st.session_state: st.session_state.toast_seen=set()
if "post_source" not in st.session_state: st.session_state.post_source="Upload image"
if "post_capture_path" not in st.session_state: st.session_state.post_capture_path=None

# -------------------- CONTROL SIDEBAR --------------------
with st.sidebar:
    st.markdown("## 🛰️ RESPONSE CONTROL")
    source=st.selectbox("GPS source",["Demo Simulator","Manual","MAVLink"])
    if source=="Manual":
        lat=st.number_input("Latitude",value=float(st.session_state.gps["lat"]),format="%.6f")
        lon=st.number_input("Longitude",value=float(st.session_state.gps["lon"]),format="%.6f")
        alt=st.number_input("Altitude (m)",min_value=1.0,value=float(st.session_state.gps["altitude"])); heading=st.number_input("Heading (°)",0.0,360.0,float(st.session_state.gps["heading"]))
    else: lat,lon,alt,heading=st.session_state.gps["lat"],st.session_state.gps["lon"],st.session_state.gps["altitude"],st.session_state.gps["heading"]
    endpoint=st.text_input("MAVLink endpoint","/dev/tty.usbmodem") if source=="MAVLink" else None
    if st.button("UPDATE TELEMETRY",width="stretch"):
        g,e=get_gps(source,lat,lon,alt,heading,endpoint)
        if g: st.session_state.gps=g; append_json(TELEMETRY_FILE,g,1000); st.rerun()
        else: st.error(e)
    st.divider(); st.markdown("### AI / DAMAGE")
    yolo_conf=st.slider("YOLO confidence",.10,.90,.35,.05)
    critical=st.slider("Critical threshold",40,180,100,5); moderate=st.slider("Moderate threshold",20,120,65,5); low=st.slider("Low threshold",5,80,35,5)
    st.caption("Damage zones are computed from pixel differences between the uploaded pre/post images. Map geometry is approximate without calibrated georeferencing.")

if source=="Demo Simulator":
    st.session_state.gps=generate_gps(); append_json(TELEMETRY_FILE,st.session_state.gps,1000)
gps=st.session_state.gps
with LIVE_LOCK: LIVE_GPS=gps.copy()
alerts=load_json(ALERTS_FILE,[]); aid_records=load_json(AID_FILE,[]); telemetry=load_json(TELEMETRY_FILE,[]) or [gps]
responders=responder_state(gps)
stats=mission_stats(alerts,st.session_state.damage_zones,responders,aid_records)

# -------------------- HEADER --------------------
header1,header2,header3,header4=st.columns([4.2,1.25,1.15,1.1],gap="small")
with header1:
    st.markdown('<div class="topbar"><div class="brand">🛡️ DISASTER RESPONSE COMMAND CENTER</div><div class="subbrand">AI SEARCH & RESCUE • DRONE SURVEILLANCE • DAMAGE INTELLIGENCE • FIELD COORDINATION</div></div>',unsafe_allow_html=True)
with header2: st.markdown('<div class="topbar"><span class="status-pill"><span class="status-dot"></span>SYSTEM ONLINE</span><div class="small-note">DRONE D1 • LIVE</div></div>',unsafe_allow_html=True)
with header3: st.markdown(f'<div class="topbar"><div class="eyebrow">MISSION CLOCK</div><div class="stat-value" style="font-size:1.0rem">{now().split()[1]}</div><div class="small-note">UTC+05:30</div></div>',unsafe_allow_html=True)
with header4: st.markdown('<div class="topbar"><div class="eyebrow">MISSION MODE</div><div class="danger-badge">● ACTIVE</div><div class="small-note">SAVE LIVES</div></div>',unsafe_allow_html=True)

# -------------------- KPI STRIP --------------------
k1,k2,k3,k4,k5,k6=st.columns(6,gap="small")
for col,label,val,note,cls in [
    (k1,"MISSION PROGRESS",f"{stats['progress']}%","Search & Rescue","kpi-cyan"),
    (k2,"SURVIVORS",str(stats['survivors']),"AI detections","kpi-red"),
    (k3,"DAMAGE ZONES",str(stats['zones']),"From PRE ↔ POST","kpi-orange"),
    (k4,"RESPONDERS",f"{stats['responders_active']}/{stats['responders_total']}","Active / total","kpi-green"),
    (k5,"AREA SCANNED",f"{stats['area_scanned']:.1f} km²","Mission estimate","kpi-cyan"),
    (k6,"AID DROPS",str(stats['aid_drops']),"Commands logged","kpi-green")]:
    with col: st.markdown(f'<div class="stat-card"><div class="stat-label">{label}</div><div class="stat-value {cls}">{val}</div><div class="stat-note">{note}</div></div>',unsafe_allow_html=True)

left,center,right=st.columns([1.35,3.25,1.45],gap="small")

# -------------------- LEFT: LIVE INTELLIGENCE --------------------
with left:
    st.markdown('<div class="panel"><div class="panel-head"><span>📹 RGB LIVE FEED • DRONE D1</span><span class="live-badge">● LIVE</span></div>',unsafe_allow_html=True)
    if WEBRTC_AVAILABLE:
        webrtc_result=webrtc_streamer(
            key="live-disaster-camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=live_callback,
            media_stream_constraints={"video":{"width":{"ideal":1280},"height":{"ideal":720},"facingMode":"environment"},"audio":False},
            rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            async_processing=False,
        )
        st.caption("▶ START the live drone camera feed. Recording begins automatically; a photo is saved only when AI detects a person/survivor.")
        with LIVE_LOCK:
            recording_path=str(LIVE_VIDEO_PATH) if LIVE_VIDEO_PATH else None
            recording_frames=LIVE_VIDEO_FRAMES
        if recording_path:
            st.markdown(f'<div class="small-note">● RECORDING • {recording_frames:,} frames • segment: {Path(recording_path).name}</div>',unsafe_allow_html=True)
    else:
        cam=st.camera_input("Camera snapshot",key="rgb_camera")
        if cam:
            frame=cv_image(cam)
            with LIVE_LOCK:
                LIVE_FRAME=frame.copy()
                LIVE_DETECTIONS=[]
            st.image(rgb(frame),width="stretch")
        st.caption("Live WebRTC is unavailable. Use the camera snapshot fallback.")
    with LIVE_LOCK:
        lf=None if LIVE_FRAME is None else LIVE_FRAME.copy(); live_det=list(LIVE_DETECTIONS)
    if lf is not None and not WEBRTC_AVAILABLE:
        st.image(rgb(lf),width="stretch",caption=f"RGB snapshot: {len(live_det)} detections")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-head"><span>🌡️ THERMAL VIEW</span><span class="live-badge">SIMULATED</span></div>',unsafe_allow_html=True)
    if lf is not None:
        gray=cv2.cvtColor(lf,cv2.COLOR_BGR2GRAY); thermal=cv2.applyColorMap(gray,cv2.COLORMAP_INFERNO); st.image(rgb(thermal),width="stretch")
    else: st.caption("Waiting for live frame…")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-head"><span>🛰️ DAMAGE INTELLIGENCE</span><span class="danger-badge">PRE ↔ POST</span></div>',unsafe_allow_html=True)

    # PRE-DISASTER IMAGE
    pre_up=st.file_uploader("Pre-disaster image",["jpg","jpeg","png","webp"],key="pre_left")
    if pre_up:
        st.session_state.pre=cv_image(pre_up)

    if st.session_state.pre is not None:
        st.image(
            rgb(st.session_state.pre),
            caption="PRE-DISASTER IMAGE",
            width="stretch",
        )

    if pre_up and st.button("💾 INDEX PRE-DISASTER IMAGE",width="stretch"):
        p=PREPOST_DIR/f"pre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(p),st.session_state.pre)
        append_json(
            SCANS_FILE,
            {
                "scan_id":uid("SCAN"),
                "type":"Pre-Disaster",
                "lat":gps["lat"],
                "lon":gps["lon"],
                "timestamp":now(),
                "image_path":str(p),
            },
            500,
        )
        st.success("Pre-disaster reference indexed.")

    # POST-DISASTER IMAGE: upload OR choose an already captured survivor image.
    post_source=st.selectbox(
        "Post-disaster image source",
        ["Upload image","Captured images"],
        key="post_source",
    )

    post_up=None
    selected_capture=None

    if post_source=="Upload image":
        post_up=st.file_uploader(
            "Post-disaster image",
            ["jpg","jpeg","png","webp"],
            key="post_left",
        )
        if post_up:
            st.session_state.post=cv_image(post_up)
            st.session_state.post_capture_path=None
    else:
        captured_paths=sorted(
            [p for p in FOOTAGE_DIR.glob("LIVE_*.jpg") if p.exists()],
            key=lambda p:p.stat().st_mtime,
            reverse=True,
        )
        if captured_paths:
            capture_labels=[p.name for p in captured_paths]
            selected_name=st.selectbox(
                "Select captured image",
                capture_labels,
                key="post_capture_select",
            )
            selected_capture=next(
                (p for p in captured_paths if p.name==selected_name),
                None,
            )
            if selected_capture is not None:
                selected_image=cv2.imread(str(selected_capture))
                if selected_image is not None:
                    st.session_state.post=selected_image
                    st.session_state.post_capture_path=str(selected_capture)
        else:
            st.info("No captured survivor images available yet. Detect a person first.")

    if st.session_state.post is not None:
        caption="POST-DISASTER IMAGE"
        if st.session_state.post_capture_path:
            caption=f"POST-DISASTER • {Path(st.session_state.post_capture_path).name}"
        st.image(
            rgb(st.session_state.post),
            caption=caption,
            width="stretch",
        )

    # Preserve the existing nearby-reference behavior when a post image is uploaded
    # without a pre image in the current session.
    if post_up is not None and pre_up is None:
        matched=nearest_pre_scan(gps["lat"],gps["lon"],2.0)
        if matched:
            d,scan=matched
            st.info(f"Matched reference: {d:.3f} km • {scan['scan_id']}")
            st.session_state.pre=cv2.imread(scan["image_path"])

    if st.session_state.pre is not None and st.session_state.post is not None and st.button("⚡ RUN REAL DAMAGE COMPARISON",width="stretch"):
        cmap,diff,dstats=damage_compare(st.session_state.pre,st.session_state.post,critical,moderate,low)
        p=PREPOST_DIR/f"post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(str(p),st.session_state.post)
        append_json(
            SCANS_FILE,
            {
                "scan_id":uid("SCAN"),
                "type":"Post-Disaster",
                "lat":gps["lat"],
                "lon":gps["lon"],
                "timestamp":now(),
                "image_path":str(p),
                "damage_statistics":dstats,
            },
            500,
        )
        append_json(
            EVENTS_FILE,
            {
                "event_id":uid("COMPARE"),
                "type":"Pre/Post comparison",
                "lat":gps["lat"],
                "lon":gps["lon"],
                "timestamp":now(),
                "damage_statistics":dstats,
            },
            500,
        )
        st.session_state.last_result={"damage_map":cmap,"stats":dstats,"diff":diff}
        st.session_state.damage_zones=damage_zones_from_diff(diff,gps,threshold=low)
        st.rerun()

    if st.session_state.last_result:
        r=st.session_state.last_result
        st.image(
            rgb(r["damage_map"]),
            caption="Detected change / damage map",
            width="stretch",
        )
        st.dataframe(
            pd.DataFrame(
                {
                    "Class":list(r["stats"]),
                    "Area %":list(r["stats"].values()),
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Detected map zones: {len(st.session_state.damage_zones)}")

    st.markdown('</div>',unsafe_allow_html=True)

# -------------------- CENTER: MAP --------------------
with center:
    st.markdown('<div class="panel"><div class="panel-head"><span>🗺️ LIVE OPERATION MAP</span><span class="map-chip">SATELLITE</span><span class="map-chip">STREET</span><span class="map-chip">RESPONDERS</span></div>',unsafe_allow_html=True)
    if FOLIUM_AVAILABLE:
        c1,c2,c3=st.columns([4.3,1.0,1.2])
        with c1: map_search=st.text_input("Search",placeholder="Place or latitude, longitude",label_visibility="collapsed",key="map_search")
        with c2: search_clicked=st.button("🔍 Search",use_container_width=True,key="map_search_button")
        with c3: center_drone=st.button("🎯 Drone",use_container_width=True,key="center_drone")
        if search_clicked and map_search.strip():
            target=parse_map_coordinates(map_search.strip()) or geocode_location(map_search.strip())
            if target:
                st.session_state.map_center=target
                st.session_state.map_zoom=16
                st.session_state.map_version += 1
            else: st.warning("Location not found.")
        if center_drone:
            st.session_state.map_center=[float(gps["lat"]),float(gps["lon"])]
            st.session_state.map_zoom=18
            st.session_state.map_version += 1

        path=telemetry[-100:] if telemetry else [{"lat":gps["lat"],"lon":gps["lon"]}]
        if len(path)<2: path=[{"lat":gps["lat"]-.001,"lon":gps["lon"]-.001},gps]
        operation_map=create_interactive_operation_map(gps,telemetry,responder_state(gps),[{"lat":a.get("latitude"),"lon":a.get("longitude")} for a in alerts[-10:] if a.get("latitude") is not None],st.session_state.damage_zones,st.session_state.get("offline_map"))
        map_state=st_folium(operation_map,width=1200,height=680,returned_objects=["last_clicked","last_object_clicked"],key=f"operation_map_{st.session_state.map_version}")
        if map_state:
            if map_state.get("last_clicked"):
                c=map_state["last_clicked"]; st.session_state.selected_map_coord=[c["lat"],c["lng"]]
            obj=map_state.get("last_object_clicked")
            if obj and isinstance(obj,dict):
                if obj.get("lat") is not None and obj.get("lng") is not None: st.session_state.selected_map_coord=[obj["lat"],obj["lng"]]
        if st.session_state.selected_map_coord: st.markdown(f'<div class="map-chip">📍 SELECTED {st.session_state.selected_map_coord[0]:.6f}, {st.session_state.selected_map_coord[1]:.6f}</div>',unsafe_allow_html=True)
        st.caption(f"🛸 D1 {gps['lat']:.6f}, {gps['lon']:.6f} • Alt {gps['altitude']:.1f} m • Heading {gps['heading']:.0f}°")

        # Offline map upload is placed directly below the Live Operation Map.
        st.markdown("#### 🗺️ OFFLINE MAP")
        offline_up=st.file_uploader(
            "Upload offline map",
            type=["jpg","jpeg","png","webp"],
            key="live_operation_offline_map",
        )
        if offline_up is not None:
            try:
                raw=offline_up.getvalue()
                upload_hash=hashlib.sha256(raw).hexdigest()
                if upload_hash != st.session_state.offline_map_upload_hash:
                    arr=np.frombuffer(raw,dtype=np.uint8)
                    offline_image=cv2.imdecode(arr,cv2.IMREAD_COLOR)
                    if offline_image is None:
                        st.error("Unable to read the offline map image.")
                    else:
                        st.session_state.offline_map=offline_image
                        st.session_state.offline_map_upload_hash=upload_hash
                        st.session_state.map_version += 1
                if st.session_state.offline_map is not None:
                    st.image(
                        cv2.cvtColor(st.session_state.offline_map,cv2.COLOR_BGR2RGB),
                        caption="Offline Map",
                        width="stretch",
                    )
            except Exception as e:
                st.error(f"Offline map error: {e}")
        elif st.session_state.offline_map is not None:
            st.image(
                cv2.cvtColor(st.session_state.offline_map,cv2.COLOR_BGR2RGB),
                caption="Offline Map",
                width="stretch",
            )
    else: st.error("Install folium and streamlit-folium.")
    st.markdown('</div>',unsafe_allow_html=True)

# -------------------- RIGHT: ALERTS + RESPONDERS --------------------
with right:
    st.markdown('<div class="panel"><div class="panel-head"><span>🚨 AI ALERTS</span><span class="danger-badge">LIVE</span></div>',unsafe_allow_html=True)
    if alerts:
        for a in reversed(alerts[-5:]):
            alert_title="📍 Survivor location update" if a.get("is_location_update") else "🚨 Survivor detected"
            st.markdown(f'<div class="alert-row"><div class="alert-title">{alert_title}</div><div class="alert-detail">Confidence {float(a.get("confidence",0))*100:.0f}% • {a.get("timestamp","")}</div><div class="coord">{float(a.get("latitude",0)):.6f}, {float(a.get("longitude",0)):.6f}</div></div>',unsafe_allow_html=True)
            cc1,cc2=st.columns(2)
            with cc1:
                if st.button("📍 CENTER",key=f"center_{a.get('alert_id')}",width="stretch"):
                    st.session_state.map_center=[a["latitude"],a["longitude"]]; st.session_state.map_zoom=18; st.session_state.selected_survivor=a.get("alert_id"); st.session_state.map_version += 1; st.rerun()
            with cc2:
                if st.button("🗑️ DELETE",key=f"del_{a.get('alert_id')}",width="stretch"):
                    alerts=[x for x in alerts if x.get("alert_id")!=a.get("alert_id")]; save_json(ALERTS_FILE,alerts)
                    p=a.get("image_path");
                    if p and Path(p).exists(): Path(p).unlink()
                    st.rerun()
    else: st.info("No survivor detections yet.")
    st.markdown('</div>',unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-head"><span>🪖 RESPONDER STATUS</span><span class="live-badge">LIVE</span></div>',unsafe_allow_html=True)
    for r in responders:
        status_cls="kpi-green" if r["status"] in ("MOVING","ON SITE") else "kpi-orange"
        st.markdown(f'<div class="alert-row"><div class="alert-title">🪖 {r["id"]} <span class="{status_cls}">● {r["status"]}</span></div><div class="alert-detail">{r["sector"]} • Battery {r["battery"]}%</div><div class="coord">{r["lat"]:.6f}, {r["lon"]:.6f}</div></div>',unsafe_allow_html=True)
    st.markdown('</div>',unsafe_allow_html=True)

# -------------------- BOTTOM OPERATIONS --------------------
b1,b2,b3=st.columns([1.4,1.6,1.3],gap="small")
with b1:
    st.markdown('<div class="panel"><div class="panel-head"><span>📊 MISSION STATISTICS</span></div>',unsafe_allow_html=True)
    st.progress(stats["progress"]/100,text=f"Mission progress • {stats['progress']}%")
    st.markdown(f"**Survivors:** {stats['survivors']}  \
**Damage zones:** {stats['zones']}  \
**Responders active:** {stats['responders_active']}/{stats['responders_total']}  \
**Aid kits dropped:** {stats['aid_drops']}")
    st.markdown('</div>',unsafe_allow_html=True)
with b2:
    st.markdown('<div class="panel"><div class="panel-head"><span>🧰 AID KIT MANAGEMENT</span></div>',unsafe_allow_html=True)
    options=[a for a in alerts[-50:] if a.get("latitude") is not None and not a.get("is_location_update")]
    if options:
        labels=[f"{a.get('alert_id','SURVIVOR')} • {a['latitude']:.6f}, {a['longitude']:.6f}" for a in options]
        idx=st.selectbox("Target survivor",range(len(labels)),format_func=lambda i:labels[i],key="aid_target")
        target=options[idx]; kit=st.selectbox("Aid kit",["Medical","Water","Food","Emergency Pack"],key="aid_kit")
        if st.button("🚁 CONFIRM DROP AID KIT",type="primary",width="stretch"):
            drop={"drop_id":uid("AID"),"kit":kit,"latitude":target["latitude"],"longitude":target["longitude"],"alert_id":target.get("alert_id"),"timestamp":now(),"status":"COMMAND LOGGED"}
            append_json(AID_FILE,drop,500); st.success(f"{kit} kit command logged for {target['latitude']:.6f}, {target['longitude']:.6f}."); st.rerun()
    else: st.info("Detect a survivor first to unlock a targeted aid drop.")
    st.markdown('</div>',unsafe_allow_html=True)
with b3:
    st.markdown('<div class="panel"><div class="panel-head"><span>🛰️ LATEST DAMAGE</span></div>',unsafe_allow_html=True)
    if st.session_state.last_result:
        ds=st.session_state.last_result["stats"]
        for name,val in ds.items(): st.markdown(f"**{name}**  \
{val:.1f}%")
    else: st.caption("Upload matching pre/post images and run comparison.")
    st.markdown('</div>',unsafe_allow_html=True)

# -------------------- COMMAND / STORAGE --------------------
st.markdown("### COMMAND / STORAGE")
cmds=["🗺️ Offline Maps","📍 Coordinates","🖼️ Captured Images","📋 Flight Logs","➕ Drop Aid Kit"]
cc=st.columns(len(cmds))
for i,label in enumerate(cmds):
    with cc[i]:
        if st.button(label,key=f"cmd_{i}",width="stretch"): st.session_state.active_panel=i
if st.session_state.active_panel==0:
    up=st.file_uploader("Offline orthophoto",["jpg","jpeg","png","webp"],key="offline_orthophoto_upload")
    if up:
        st.session_state.offline_map=cv_image(up); st.success("Offline map loaded.")
elif st.session_state.active_panel==1:
    rows=[{"Asset":"DRONE D1","Latitude":gps["lat"],"Longitude":gps["lon"],"Altitude m":gps["altitude"],"Heading":gps["heading"]}]+[{"Asset":r["id"],"Latitude":r["lat"],"Longitude":r["lon"],"Status":r["status"],"Battery":r["battery"]} for r in responders]
    st.dataframe(pd.DataFrame(rows),hide_index=True,width="stretch")
elif st.session_state.active_panel==2:
    caps=sorted(FOOTAGE_DIR.glob("*.jpg"),reverse=True)
    if caps:
        for p in caps[:12]:
            c1,c2=st.columns([4,1]); c1.image(str(p),caption=p.stem,width="stretch")
            with c2:
                if st.button("🗑️ Delete",key=f"capture_delete_{p.name}"): p.unlink(missing_ok=True); st.rerun()
    else: st.info("No captures yet.")
elif st.session_state.active_panel==3:
    st.dataframe(pd.DataFrame(telemetry[-100:]),hide_index=True,width="stretch")
elif st.session_state.active_panel==4:
    st.info("Use the Aid Kit Management panel above for targeted drops.")

st.divider(); st.caption("Prototype command center • Damage zones are computed from image change detection; survivor coordinates are estimated from camera geometry/GPS. Validate with calibrated geospatial data and trained responders before operational use.")

