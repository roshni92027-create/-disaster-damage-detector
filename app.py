
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

st.markdown("""
<style>
.stApp,[data-testid="stAppViewContainer"],[data-testid="stHeader"]{background:#050b14;color:#e8eef7}
.block-container{padding:1rem 1.2rem 1rem;max-width:1800px}
[data-testid="stSidebar"]{background:#07111d;border-right:1px solid #1d344b}
h1,h2,h3,h4{color:#f5f8fb!important}
p,span,label{color:#b8c8d9}
.dash-card{background:#0a1522;border:1px solid #1c344a;border-radius:14px;padding:12px;margin-bottom:10px}
.section-title{font-size:.78rem;font-weight:800;letter-spacing:.12em;color:#7892aa;text-transform:uppercase}
.live{color:#56e39a;font-weight:800}.danger{color:#ff5d62;font-weight:800}
.warn{color:#ffad4d;font-weight:800}.safe{color:#5be38e;font-weight:800}
.coord{font-family:monospace;color:#d9e8f7}
[data-testid="stMetric"]{background:#091522;border:1px solid #1c344a;border-radius:12px}
.stButton>button{background:#0c1d2d;border:1px solid #29465e;color:#e8f2fb;border-radius:9px;min-height:38px}
.stButton>button:hover{border-color:#55a7e6;background:#10283d}
[data-testid="stFileUploader"]{background:#091522;border:1px dashed #29465e;border-radius:10px}
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

    folium.Marker(
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
    ).add_to(m)

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
        folium.PolyLine(
            path,
            color="#00d9ff",
            weight=2,
            opacity=0.45,
            tooltip="Drone flight trail",
        ).add_to(m)

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
LIVE_LOCK=threading.Lock()
LIVE_FRAME=None
LIVE_DETECTIONS=[]
LIVE_EVENTS=queue.Queue()
LIVE_LAST_ALERT=0.0
LIVE_LAST_INFER=0.0
LIVE_GPS=DEFAULT_GPS.copy()

def live_callback(frame):
    global LIVE_FRAME,LIVE_DETECTIONS,LIVE_LAST_ALERT,LIVE_LAST_INFER
    image=frame.to_ndarray(format="bgr24")
    annotated=image.copy()
    with LIVE_LOCK:
        LIVE_FRAME=image.copy()
    if time.time() - LIVE_LAST_INFER < 0.45:
        return av.VideoFrame.from_ndarray(annotated,format="bgr24")
    LIVE_LAST_INFER=time.time()
    det=[]
    model=load_model()
    if model is not None:
        try:
            result=model.predict(image,conf=.35,verbose=False)[0]
            for b in result.boxes:
                xy=b.xyxy[0].cpu().numpy().tolist()
                label=str(result.names[int(b.cls[0])]); score=float(b.conf[0])
                item={"Class":label,"Confidence":round(score,3),"X1":round(xy[0]),"Y1":round(xy[1]),"X2":round(xy[2]),"Y2":round(xy[3])}
                det.append(item)
                if label.lower()=="person":
                    x1,y1,x2,y2=map(int,xy)
                    cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,80,255),2)
                    cv2.putText(annotated,f"SURVIVOR {score:.0%}",(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,80,255),2)
        except Exception:
            det=[]
    with LIVE_LOCK:
        LIVE_FRAME=image.copy()
        LIVE_DETECTIONS=det
    persons=[x for x in det if x["Class"].lower()=="person"]
    if persons and time.time()-LIVE_LAST_ALERT>10:
        LIVE_LAST_ALERT=time.time()
        with LIVE_LOCK: gps=LIVE_GPS.copy()
        event_id=uid("LIVE")
        alerts=[]
        for p in persons:
            alert=make_alert(p,gps,event_id,image.shape[1],image.shape[0])
            cap_path=FOOTAGE_DIR/f"{event_id}.jpg"
            cv2.imwrite(str(cap_path),annotated)
            alert["image_path"]=str(cap_path)
            append_json(ALERTS_FILE,alert)
            alerts.append(alert)
        LIVE_EVENTS.put({"event_id":event_id,"alerts":alerts})
    return av.VideoFrame.from_ndarray(annotated,format="bgr24")

if "gps" not in st.session_state: st.session_state.gps=DEFAULT_GPS.copy()
if "pre" not in st.session_state: st.session_state.pre=None
if "post" not in st.session_state: st.session_state.post=None
if "last_result" not in st.session_state: st.session_state.last_result=None
if "active_panel" not in st.session_state: st.session_state.active_panel=None
if "toast_seen" not in st.session_state: st.session_state.toast_seen=set()

with st.sidebar:
    st.title("⚙️ Response Control")
    source=st.selectbox("GPS source",["Demo Simulator","Manual","MAVLink"])
    if source=="Manual":
        lat=st.number_input("Latitude",value=float(st.session_state.gps["lat"]),format="%.6f")
        lon=st.number_input("Longitude",value=float(st.session_state.gps["lon"]),format="%.6f")
        alt=st.number_input("Altitude (m)",min_value=1.0,value=float(st.session_state.gps["altitude"]))
        heading=st.number_input("Heading (°)",min_value=0.0,max_value=360.0,value=float(st.session_state.gps["heading"]))
    else:
        lat,lon,alt,heading=st.session_state.gps["lat"],st.session_state.gps["lon"],st.session_state.gps["altitude"],st.session_state.gps["heading"]
    endpoint=st.text_input("MAVLink endpoint","/dev/tty.usbmodem") if source=="MAVLink" else None
    if st.button("Update telemetry",width="stretch"):
        g,e=get_gps(source,lat,lon,alt,heading,endpoint)
        if g:
            st.session_state.gps=g
            append_json(TELEMETRY_FILE,g,limit=1000)
        else: st.error(e)
    st.divider()
    st.subheader("AI / Damage")
    yolo_conf=st.slider("YOLO confidence",.10,.90,.35,.05)
    critical=st.slider("Critical threshold",40,180,100,5)
    moderate=st.slider("Moderate threshold",20,120,65,5)
    low=st.slider("Low threshold",5,80,35,5)
    st.caption("Priority-zone colors are prototype visualization; they are not a live population dataset.")

if source=="Demo Simulator":
    st.session_state.gps=generate_gps()
    append_json(TELEMETRY_FILE,st.session_state.gps,limit=1000)
gps=st.session_state.gps
with LIVE_LOCK: LIVE_GPS=gps.copy()

alerts=load_json(ALERTS_FILE,[])
scan_records=load_json(EVENTS_FILE,[])
aid_records=load_json(AID_FILE, [])
telemetry=load_json(TELEMETRY_FILE,[])
if not telemetry:
    telemetry=[gps]

# Dashboard notification feed shown alongside live AI alerts.
def dashboard_events(current_gps, alert_list):
    events=[]
    for a in alert_list[-8:]:
        events.append({
            "time": str(a.get("timestamp", "--"))[-8:-3],
            "icon": "🚨",
            "title": "Survivor Detected",
            "detail": f"Lat: {a.get('latitude','--')}, Lon: {a.get('longitude','--')}",
            "kind": "danger",
            "image_path": a.get("image_path")
        })
    events += [
        {"time":"14:24","icon":"🪖","title":"Responder Message (Unit 2)","detail":"Reaching location in 5 minutes.","kind":"info"},
        {"time":"14:20","icon":"🔋","title":"Low Battery Warning","detail":"Battery at 20% • Return-to-base check.","kind":"warn"},
        {"time":"14:12","icon":"✅","title":"Area Surveyed — No Damage","detail":"Sector B3 completed.","kind":"safe"},
        {"time":"14:05","icon":"🪖","title":"Responder Message (Unit 3)","detail":"Proceeding to marked location.","kind":"info"},
        {"time":"13:58","icon":"⚠️","title":"New Zone Identified","detail":"Moderate damage area marked.","kind":"warn"},
        {"time":"13:45","icon":"ℹ️","title":"System","detail":"Drone D1 started autonomous survey.","kind":"info"},
    ]
    return events

header1, header2, header3 = st.columns([3.4,1.2,1.4], gap="small")
with header1:
    st.markdown("# 🛸 DISASTER MANAGEMENT DASHBOARD")
    st.caption("AERIAL INTELLIGENCE  |  FASTER RESPONSE  |  SAFER TOMORROW")
with header2:
    st.success("● SYSTEM ONLINE")
    st.caption("DRONE D1 • LIVE")
with header3:
    st.metric("MISSION TIME", now().split()[1])
    st.caption("SAVE LIVES • BUILD A SAFER TOMORROW")
st.divider()

left,center,right=st.columns([0.95,3.6,1.0],gap="small")

with left:
    st.markdown("### 📹 RGB LIVE FEED")
    if WEBRTC_AVAILABLE:
        webrtc_streamer(
            key="live-disaster-camera",
            mode=WebRtcMode.SENDRECV,
            video_frame_callback=live_callback,
            media_stream_constraints={"video":True,"audio":False},
            frontend_rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            server_rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
            async_processing=True,
        )
    else:
        st.info("Install streamlit-webrtc for continuous browser camera.")
        cam=st.camera_input("Camera fallback")
        if cam:
            frame=cv_image(cam)
            with LIVE_LOCK: LIVE_FRAME=frame.copy()
            st.image(rgb(frame),width="stretch")
    st.markdown("### 🌡️ THERMAL VIEW")
    with LIVE_LOCK:
        lf=None if LIVE_FRAME is None else LIVE_FRAME.copy()
    if lf is not None:
        gray=cv2.cvtColor(lf,cv2.COLOR_BGR2GRAY)
        thermal=cv2.applyColorMap(gray,cv2.COLORMAP_INFERNO)
        st.image(rgb(thermal),width="stretch")
    else:
        st.caption("Waiting for live frame…")
    st.markdown("### 🛰️ PRE ↔ POST")
    pc1,pc2=st.columns(2)
    with pc1:
        pre_up=st.file_uploader("Pre",["jpg","jpeg","png","webp"],key="pre_left")
    with pc2:
        post_up=st.file_uploader("Post",["jpg","jpeg","png","webp"],key="post_left")
    if pre_up:
        st.session_state.pre=cv_image(pre_up)
        if st.button("💾 SAVE PRE-DISASTER SCAN",width="stretch"):
            pre_path=PREPOST_DIR/f"pre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(pre_path),st.session_state.pre)
            append_json(SCANS_FILE,{"scan_id":uid("SCAN"),"type":"Pre-Disaster",
                                    "lat":gps["lat"],"lon":gps["lon"],"timestamp":now(),
                                    "image_path":str(pre_path)},limit=500)
            st.success("Pre-disaster image indexed by current GPS.")
    if post_up:
        st.session_state.post=cv_image(post_up)
    if post_up and not pre_up:
        matched=nearest_pre_scan(gps["lat"],gps["lon"],2.0)
        if matched:
            distance,scan=matched
            st.info(f"GPS-matched pre-disaster scan: {distance:.3f} km away • {scan['scan_id']}")
            st.session_state.pre=cv2.imread(scan["image_path"])
        else:
            st.warning("No pre-disaster scan found within 2 km of the current GPS.")
    if st.session_state.pre is not None and st.session_state.post is not None:
        if st.button("COMPARE DAMAGE",width="stretch"):
            cmap,diff,stats=damage_compare(st.session_state.pre,st.session_state.post,critical,moderate,low)
            post_path=PREPOST_DIR/f"post_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(post_path),st.session_state.post)
            append_json(SCANS_FILE,{"scan_id":uid("SCAN"),"type":"Post-Disaster",
                                    "lat":gps["lat"],"lon":gps["lon"],"timestamp":now(),
                                    "image_path":str(post_path),
                                    "damage_statistics":stats},limit=500)
            append_json(EVENTS_FILE,{"event_id":uid("COMPARE"),"type":"Pre/Post comparison",
                                     "lat":gps["lat"],"lon":gps["lon"],"timestamp":now(),
                                     "damage_statistics":stats},limit=500)
            st.session_state.last_result={"damage_map":cmap,"stats":stats,"diff":diff}
    if st.session_state.last_result:
        r=st.session_state.last_result
        st.caption("Damage classification")
        st.dataframe(pd.DataFrame({"Class":list(r["stats"]),"Area %":list(r["stats"].values())}),hide_index=True,width="stretch")
        if r["damage_map"] is not None: st.image(rgb(r["damage_map"]),width="stretch")

with center:
    st.markdown("### 🗺️ LIVE OPERATION MAP")

    if not FOLIUM_AVAILABLE:
        st.error(
            "Interactive map dependencies are missing. "
            "Install folium and streamlit-folium from requirements.txt."
        )
    else:
        # -------------------------
        # Google-Maps-style search bar
        # -------------------------
        search_col, search_btn_col, follow_col, drone_col = st.columns(
            [4.2, 1.0, 1.45, 1.2], gap="small"
        )

        with search_col:
            map_search = st.text_input(
                "Map search",
                placeholder="Search place or enter latitude, longitude",
                label_visibility="collapsed",
                key="map_search",
            )

        with search_btn_col:
            search_clicked = st.button(
                "🔍 Search",
                use_container_width=True,
                key="map_search_button",
            )

        with follow_col:
            follow_drone = st.checkbox(
                "Follow Drone",
                value=False,
                key="follow_drone",
            )

        with drone_col:
            center_drone = st.button(
                "🎯 Drone",
                use_container_width=True,
                key="center_drone",
            )

        # -------------------------
        # Search handling
        # -------------------------
        if search_clicked and map_search.strip():
            target = parse_map_coordinates(map_search.strip())
            if target is None:
                target = geocode_location(map_search.strip())

            if target:
                st.session_state.map_center = target
                st.session_state.map_zoom = 16
                st.success(
                    f"Map centered at {target[0]:.6f}, {target[1]:.6f}"
                )
            else:
                st.warning(
                    "Location not found. Try a place name or "
                    "latitude,longitude."
                )

        if center_drone or follow_drone:
            st.session_state.map_center = [
                float(gps["lat"]),
                float(gps["lon"]),
            ]
            st.session_state.map_zoom = 17

        # -------------------------
        # Map data
        # -------------------------
        base = st.session_state.get("offline_map")
        if base is None and st.session_state.pre is not None:
            base = st.session_state.pre

        path = telemetry[-100:] if telemetry else [
            {"lat": gps["lat"], "lon": gps["lon"]}
        ]
        if len(path) < 2:
            path = [
                {"lat": gps["lat"] + .001, "lon": gps["lon"] - .001},
                {"lat": gps["lat"], "lon": gps["lon"]},
            ]

        responders = [
            {
                "lat": gps["lat"] + .0012,
                "lon": gps["lon"] - .0015,
            },
            {
                "lat": gps["lat"] - .0015,
                "lon": gps["lon"] + .001,
            },
            {
                "lat": gps["lat"] + .0005,
                "lon": gps["lon"] + .002,
            },
        ]

        zones = [
            {
                "kind": "critical",
                "label": "CRITICAL / HIGH SURVIVOR",
                "points": [
                    (gps["lat"] + .0002, gps["lon"] - .0025),
                    (gps["lat"] + .0022, gps["lon"] - .0020),
                    (gps["lat"] + .0017, gps["lon"] - .0003),
                    (gps["lat"] - .0002, gps["lon"] - .0007),
                ],
            },
            {
                "kind": "moderate",
                "label": "MODERATE / HIGH PRIORITY",
                "points": [
                    (gps["lat"] + .0015, gps["lon"] + .0002),
                    (gps["lat"] + .0026, gps["lon"] + .0018),
                    (gps["lat"] + .0005, gps["lon"] + .0028),
                    (gps["lat"] - .0001, gps["lon"] + .0010),
                ],
            },
            {
                "kind": "low",
                "label": "LOW",
                "points": [
                    (gps["lat"] - .0003, gps["lon"] - .0030),
                    (gps["lat"] - .0017, gps["lon"] - .0022),
                    (gps["lat"] - .0022, gps["lon"] - .0002),
                    (gps["lat"] - .0005, gps["lon"] + .0001),
                ],
            },
            {
                "kind": "safe",
                "label": "SURVEYED / SAFE",
                "points": [
                    (gps["lat"] - .0018, gps["lon"] + .0005),
                    (gps["lat"] - .0010, gps["lon"] + .0028),
                    (gps["lat"] - .0026, gps["lon"] + .0032),
                    (gps["lat"] - .0030, gps["lon"] + .0010),
                ],
            },
        ]

        recent = alerts[-10:]
        survivor_points = [
            {
                "lat": a.get("latitude"),
                "lon": a.get("longitude"),
            }
            for a in recent
            if a.get("latitude") is not None
            and a.get("longitude") is not None
        ]

        operation_map = create_interactive_operation_map(
            gps=gps,
            telemetry=telemetry,
            responders=responders,
            survivor_points=survivor_points,
            zones=zones,
            offline_image=base,
        )

        # IMPORTANT: only listen for explicit map clicks.
        # Listening to center/zoom causes Streamlit to rerun the whole
        # page on every pan/zoom event, which makes the iframe flicker.
        map_state = st_folium(
            operation_map,
            width=1200,
            height=760,
            returned_objects=["last_clicked"],
            key="operation_map",
        )

        # A click is an intentional interaction, so a single rerun here is
        # acceptable. Pan/zoom now stay entirely inside the Leaflet iframe.
        if map_state and map_state.get("last_clicked"):
            clicked = map_state["last_clicked"]
            st.session_state.selected_map_coord = [
                clicked["lat"],
                clicked["lng"],
            ]

        if st.session_state.get("selected_map_coord"):
            selected = st.session_state.selected_map_coord
            st.caption(
                f"📍 Selected Coordinates: "
                f"{selected[0]:.6f}, {selected[1]:.6f}"
            )

        legend1, legend2, legend3, legend4 = st.columns(4)
        legend1.markdown("🔴 **CRITICAL**")
        legend2.markdown("🟠 **MODERATE**")
        legend3.markdown("🟡 **LOW**")
        legend4.markdown("🟢 **SURVEYED**")
        st.caption(
            f"🛸 Drone D1: {gps['lat']:.6f}, {gps['lon']:.6f}"
            f"  •  Alt {gps['altitude']:.1f} m"
            f"  •  Heading {gps['heading']:.0f}°"
        )
        if survivor_points:
            st.caption(
                "🚨 Latest survivor coordinates are displayed on the map "
                "and persisted in the alert log."
            )

with right:
    @st.fragment(run_every="2s")
    def notification_panel():
        live_alerts=load_json(ALERTS_FILE,[])
        feed=dashboard_events(gps, live_alerts)
        st.markdown("### 🔔 NOTIFICATIONS & ALERTS")
        tabs=st.tabs(["All","Survivors","Responders","System"])
        with tabs[0]:
            if not feed:
                st.info("No alerts yet.")
            else:
                for item in feed[:9]:
                    cls=item["kind"]
                    st.markdown(f"**{item['time']}  {item['icon']}  {item['title']}**")
                    st.caption(item["detail"])
                    if item.get("image_path") and Path(item["image_path"]).exists():
                        st.image(item["image_path"], width="stretch")
                    st.divider()
        with tabs[1]:
            survivors=[x for x in feed if x["kind"]=="danger"]
            if survivors:
                for x in survivors: st.error(f"{x['icon']} {x['title']}\n\n{x['detail']}")
            else: st.info("No survivor alerts.")
        with tabs[2]:
            for x in feed:
                if x["title"].startswith("Responder"):
                    st.info(f"{x['icon']} {x['title']}\n\n{x['detail']}")
        with tabs[3]:
            for x in feed:
                if x["kind"] in ("warn","safe","info"):
                    st.caption(f"{x['time']} • {x['title']} — {x['detail']}")
        if live_alerts:
            latest=live_alerts[-1]
            latest_id=latest.get("alert_id")
            if latest_id and latest_id not in st.session_state.toast_seen:
                st.session_state.toast_seen.add(latest_id)
                st.toast(f"🚨 Survivor: {latest.get('latitude'):.6f}, {latest.get('longitude'):.6f}", icon="🚨")
    notification_panel()

st.divider()

st.markdown("### COMMAND / STORAGE")
buttons=[
    "🗺️ Offline Maps","📍 Coordinates","🛰️ Pre-Disaster Images","🖼️ Captured Images",
    "🎞️ Video Footage","📊 Damage Analysis","◔ Zone Statistics","📋 Flight Logs",
    "〽️ Sensor Data","➕ Drop Aid Kit","⚙️ Settings"
]
cols=st.columns(len(buttons))
for i,label in enumerate(buttons):
    with cols[i]:
        if st.button(label,key=f"cmd_{i}",width="stretch"):
            st.session_state.active_panel=i

panel=st.session_state.active_panel
if panel==0:
    with st.container(border=True):
        st.subheader("🗺️ Offline Maps")
        map_up=st.file_uploader("Load offline satellite / orthophoto",["jpg","jpeg","png","webp"],key="offline_map_up")
        if map_up:
            st.session_state.offline_map=cv_image(map_up)
            path=MAP_DIR/f"offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path),st.session_state.offline_map)
            st.success("Offline satellite map loaded.")
elif panel==1:
    with st.container(border=True):
        st.subheader("📍 Coordinates")
        st.dataframe(pd.DataFrame([{"Asset":"DRONE D1","Latitude":gps["lat"],"Longitude":gps["lon"],"Altitude m":gps["altitude"],"Heading":gps["heading"]}] + [{"Asset":f"RESPONDER {i+1}","Latitude":r["lat"],"Longitude":r["lon"]} for i,r in enumerate(responders)]), hide_index=True, width="stretch")
elif panel==2:
    with st.container(border=True):
        st.subheader("🛰️ Pre-Disaster Satellite Archive")
        files=sorted(PREPOST_DIR.glob("pre_*.jpg"),reverse=True)
        if files: st.image(str(files[0]),caption="Latest pre-disaster reference",width="stretch")
        else: st.info("Upload a pre-disaster image in PRE ↔ POST.")
elif panel==3:
    with st.container(border=True):
        st.subheader("🖼️ Captured Survivor Images")
        caps=sorted(FOOTAGE_DIR.glob("*.jpg"),reverse=True)
        if caps:
            cc=st.columns(min(4,len(caps)))
            for i,pth in enumerate(caps[:12]):
                with cc[i%len(cc)]: st.image(str(pth),caption=pth.stem,width="stretch")
        else: st.info("No survivor captures yet. Start the live RGB feed.")
elif panel==4:
    with st.container(border=True):
        st.subheader("🎞️ Video Footage / AI Events")
        events=load_json(EVENTS_FILE,[])
        if events: st.dataframe(pd.DataFrame(events[-50:]),hide_index=True,width="stretch")
        else: st.info("AI footage events will appear here.")
elif panel==5:
    with st.container(border=True):
        st.subheader("📊 Damage Analysis")
        if st.session_state.last_result:
            stats=st.session_state.last_result["stats"]
            st.bar_chart(pd.DataFrame({"Area %":stats}).T)
            st.dataframe(pd.DataFrame({"Damage Class":list(stats),"Area %":list(stats.values())}),hide_index=True,width="stretch")
        else: st.info("Run a PRE ↔ POST comparison first.")
elif panel==6:
    with st.container(border=True):
        st.subheader("◔ Zone Statistics")
        stats=st.session_state.last_result["stats"] if st.session_state.last_result else {"Critical":18,"Moderate":32,"Low":28,"Safe":22}
        st.metric("Critical / High Survivor Chance",f"{stats.get('Critical',0):.1f}%")
        st.metric("Moderate / High Population",f"{stats.get('Moderate',0):.1f}%")
        st.metric("Low Damage",f"{stats.get('Low',0):.1f}%")
        st.metric("Surveyed / Safe",f"{stats.get('Safe',0):.1f}%")
elif panel==7:
    with st.container(border=True):
        st.subheader("📋 Flight Logs")
        st.dataframe(pd.DataFrame(telemetry[-100:]),hide_index=True,width="stretch")
elif panel==8:
    with st.container(border=True):
        st.subheader("〽️ Sensor Data")
        c1,c2,c3=st.columns(3)
        c1.metric("GPS Source",gps["source"])
        c2.metric("Altitude",f"{gps['altitude']:.1f} m")
        c3.metric("Heading",f"{gps['heading']:.0f}°")
        st.caption("Thermal panel is a thermal-style visualization when a true thermal sensor stream is not connected.")
elif panel==9:
    with st.container(border=True):
        st.subheader("➕ DROP AID KIT")
        options=[a for a in alerts[-20:] if a.get("latitude") is not None]
        if not options:
            st.info("Detect a person/survivor first. The target coordinates will appear here automatically.")
        else:
            labels=[f"{a['alert_id']} • {a['latitude']:.6f}, {a['longitude']:.6f} • {a['confidence']*100:.0f}%" for a in options]
            idx=st.selectbox("Select survivor target",range(len(labels)),format_func=lambda i:labels[i])
            target=options[idx]
            st.success(f"TARGET LOCKED • {target['latitude']:.6f}, {target['longitude']:.6f}")
            kit=st.selectbox("Aid kit type",["Medical","Water","Food","Emergency Pack"])
            if st.button("🚁 CONFIRM AID DROP",type="primary",width="stretch"):
                drop={"drop_id":uid("AID"),"kit":kit,"latitude":target["latitude"],"longitude":target["longitude"],"alert_id":target["alert_id"],"timestamp":now(),"status":"COMMAND LOGGED"}
                append_json(AID_FILE,drop)
                st.success(f"Aid drop command logged for {target['latitude']:.6f}, {target['longitude']:.6f}.")
elif panel==10:
    with st.container(border=True):
        st.subheader("⚙️ Settings")
        st.write("Use the Response Control sidebar for GPS source, AI confidence, damage thresholds, and MAVLink settings.")
        st.checkbox("Show prototype zone overlays",value=True)
        st.checkbox("Enable survivor alerts",value=True)

st.divider()
st.caption("Prototype command center • AI damage classification and survivor coordinates are estimates. Validate with trained responders, calibrated sensors, and appropriate geospatial/field instrumentation before operational use.")
st.caption(f"System time: {now()}  •  Alerts: {len(alerts)}  •  Aid drops: {len(aid_records)}")
