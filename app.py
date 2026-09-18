
from pathlib import Path
from datetime import datetime
import json
import math
import time
import hashlib
import threading
import queue

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

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

def create_map(base,gps,path,responders,zones):
    W,H=1200,620
    if base is not None:
        im=Image.fromarray(cv2.cvtColor(base,cv2.COLOR_BGR2RGB)).convert("RGB").resize((W,H))
        im=im.copy()
    else:
        im=Image.new("RGB",(W,H),(16,27,36)); d=ImageDraw.Draw(im)
        for x in range(0,W,50): d.line((x,0,x,H),fill=(29,45,56),width=1)
        for y in range(0,H,50): d.line((0,y,W,y),fill=(29,45,56),width=1)
        for x,y,r in [(180,120,90),(500,360,130),(900,170,110),(930,480,150)]:
            d.ellipse((x-r,y-r,x+r,y+r),fill=(22,39,48),outline=(42,67,77),width=2)
        d.text((20,20),"OFFLINE SATELLITE / TACTICAL MAP",fill=(170,194,210))
    d=ImageDraw.Draw(im)
    def xy(lat,lon):
        scale=.006
        return (int(W/2+(lon-gps["lon"])/scale*W/2),int(H/2-(lat-gps["lat"])/scale*H/2))
    zone_colors={"critical":"#ff4048","moderate":"#ff9f43","low":"#ffd447","safe":"#48d597"}
    for z in zones:
        pts=[xy(a,b) for a,b in z["points"]]
        d.polygon(pts,fill=zone_colors[z["kind"]],outline=zone_colors[z["kind"]])
        cx=sum(p[0] for p in pts)//len(pts); cy=sum(p[1] for p in pts)//len(pts)
        d.text((cx-45,cy-8),z["label"],fill="white")
    if len(path)>1:
        pts=[xy(p["lat"],p["lon"]) for p in path[-50:]]
        for a,b in zip(pts[:-1],pts[1:]):
            d.line((a,b),fill="#c9e7ff",width=2)
    for i,r in enumerate(responders,1):
        x,y=xy(r["lat"],r["lon"]); d.ellipse((x-10,y-10,x+10,y+10),fill="#5bc0ff",outline="white",width=2)
        d.text((x+12,y-10),f"R{i}",fill="white")
    dx,dy=xy(gps["lat"],gps["lon"])
    d.polygon([(dx,dy-16),(dx-12,dy+10),(dx,dy+5),(dx+12,dy+10)],fill="#56e39a",outline="white")
    d.text((dx+14,dy-14),"DRONE",fill="#56e39a")
    return im

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
aid_records=load_json(AID_FILE)
telemetry=load_json(TELEMETRY_FILE,[])
if not telemetry:
    telemetry=[gps]

st.markdown("## 🛰️ DISASTER MANAGEMENT DASHBOARD")
h1,h2,h3,h4=st.columns([1.2,1,1,1])
h1.markdown("**COMMAND CENTER**  •  <span class='live'>● SYSTEM ONLINE</span>",unsafe_allow_html=True)
h2.metric("DRONE","LIVE",gps["source"])
h3.metric("COORDINATES",f"{gps['lat']:.4f}",f"{gps['lon']:.4f}")
h4.markdown(f"**{now().split()[1]}**  \n<span class='small-muted'>SAVE LIVES • AI RESPONSE</span>",unsafe_allow_html=True)
st.divider()

left,center,right=st.columns([1.15,2.2,1.05],gap="small")

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
    st.markdown("### 🗺️ OFFLINE SATELLITE OPERATION MAP")
    base=st.session_state.get("offline_map")
    if base is None and st.session_state.pre is not None: base=st.session_state.pre
    path=telemetry[-40:] if telemetry else [{"lat":gps["lat"],"lon":gps["lon"]}]
    if len(path) < 2:
        path=[{"lat":gps["lat"]+.001,"lon":gps["lon"]-.001},{"lat":gps["lat"],"lon":gps["lon"]}]
    responders=[{"lat":gps["lat"]+.0012,"lon":gps["lon"]-.0015},{"lat":gps["lat"]-.0015,"lon":gps["lon"]+.001},{"lat":gps["lat"]+.0005,"lon":gps["lon"]+.002}]
    zones=[
        {"kind":"critical","label":"CRITICAL / HIGH SURVIVOR","points":[(gps["lat"]+.0002,gps["lon"]-.0025),(gps["lat"]+.0022,gps["lon"]-.0020),(gps["lat"]+.0017,gps["lon"]-.0003),(gps["lat"]-.0002,gps["lon"]-.0007)]},
        {"kind":"moderate","label":"MODERATE / HIGH PRIORITY","points":[(gps["lat"]+.0015,gps["lon"]+.0002),(gps["lat"]+.0026,gps["lon"]+.0018),(gps["lat"]+.0005,gps["lon"]+.0028),(gps["lat"]-.0001,gps["lon"]+.001)]},
        {"kind":"low","label":"LOW","points":[(gps["lat"]-.0003,gps["lon"]-.003),(gps["lat"]-.0017,gps["lon"]-.0022),(gps["lat"]-.0022,gps["lon"]-.0002),(gps["lat"]-.0005,gps["lon"]+.0001)]},
        {"kind":"safe","label":"SURVEYED / SAFE","points":[(gps["lat"]-.0018,gps["lon"]+.0005),(gps["lat"]-.001,gps["lon"]+.0028),(gps["lat"]-.0026,gps["lon"]+.0032),(gps["lat"]-.003,gps["lon"]+.001)]},
    ]
    st.image(create_map(base,gps,path,responders,zones),width="stretch")
    m1,m2,m3,m4=st.columns(4)
    m1.markdown("🔴 **CRITICAL**")
    m2.markdown("🟠 **MODERATE**")
    m3.markdown("🟡 **LOW**")
    m4.markdown("🟢 **SURVEYED**")
    st.caption(f"Drone: {gps['lat']:.6f}, {gps['lon']:.6f}  •  Alt {gps['altitude']:.1f} m  •  Heading {gps['heading']:.0f}°")
    if alerts:
        recent=alerts[-5:]
        survivor_points=[{"lat":a["latitude"],"lon":a["longitude"]} for a in recent if a.get("latitude") is not None]
        if survivor_points:
            st.caption("Latest survivor coordinates are persisted in the alert log and can be used for responder routing.")

with right:
    @st.fragment(run_every="2s")
    def notification_panel():
        live_alerts=load_json(ALERTS_FILE,[])
        st.markdown("### 🔔 NOTIFICATIONS")
        if not live_alerts:
            st.info("No active alerts.")
        else:
            latest=live_alerts[-1]
            latest_id=latest.get("alert_id")
            if latest_id and latest_id not in st.session_state.toast_seen:
                st.session_state.toast_seen.add(latest_id)
                st.toast(
                    f"🚨 Survivor detected at {latest.get('latitude'):.6f}, {latest.get('longitude'):.6f}",
                    icon="🚨",
                )
            st.error(f"🚨 SURVIVOR DETECTED  •  {latest.get('confidence',0)*100:.0f}%")
            st.markdown(f"**📍 {latest.get('latitude'):.6f}, {latest.get('longitude'):.6f}**")
            st.caption(f"{latest.get('timestamp')}  •  {latest.get('scan_id')}")
            if latest.get("image_path") and Path(latest["image_path"]).exists():
                st.image(latest["image_path"],caption="Captured survivor frame",width="stretch")
            st.divider()
            for a in reversed(live_alerts[-6:]):
                st.markdown(f"**🚨 Survivor**  \n`{a.get('latitude'):.6f}, {a.get('longitude'):.6f}`  \n{a.get('confidence',0)*100:.0f}% • {a.get('timestamp')}")
        if gps["altitude"]<20:
            st.warning("🔋 LOW ALTITUDE / RETURN CHECK")
    notification_panel()

st.divider()

st.markdown("### COMMAND / STORAGE")
buttons=["🗺️ Offline Maps","📍 Coordinates","🛰️ Pre-Disaster","🖼️ Captured Images","🎞️ Footage","📊 Graphs","➕ Drop Aid Kit"]
cols=st.columns(len(buttons))
for i,label in enumerate(buttons):
    with cols[i]:
        if st.button(label,key=f"cmd_{i}",width="stretch"):
            st.session_state.active_panel=i

panel=st.session_state.active_panel
if panel==0:
    with st.container(border=True):
        st.subheader("Offline Map Storage")
        map_up=st.file_uploader("Load offline satellite / orthophoto",["jpg","jpeg","png","webp"],key="offline_map_up")
        if map_up:
            st.session_state.offline_map=cv_image(map_up)
            path=MAP_DIR/f"offline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(path),st.session_state.offline_map)
            st.success("Offline map loaded into the command center.")
elif panel==1:
    with st.container(border=True):
        st.subheader("Coordinates & Responder Routing")
        st.dataframe(pd.DataFrame([{"Asset":"DRONE","Latitude":gps["lat"],"Longitude":gps["lon"],"Altitude m":gps["altitude"],"Heading":gps["heading"]}]+
                                  [{"Asset":f"RESPONDER {i+1}","Latitude":r["lat"],"Longitude":r["lon"]} for i,r in enumerate(responders)]),
                     hide_index=True,width="stretch")
        st.caption("Responder coordinates in this prototype are configurable demo markers.")
elif panel==2:
    with st.container(border=True):
        st.subheader("Pre-Disaster Satellite Archive")
        files=sorted(PREPOST_DIR.glob("*"),reverse=True)
        if files:
            st.image(str(files[0]),width="stretch")
        else: st.info("Upload a pre-disaster image in the left PRE ↔ POST panel to create the archive.")
elif panel==3:
    with st.container(border=True):
        st.subheader("Captured Survivor Images")
        caps=sorted(FOOTAGE_DIR.glob("*.jpg"),reverse=True)
        if caps:
            cc=st.columns(min(3,len(caps)))
            for i,p in enumerate(caps[:9]):
                with cc[i%len(cc)]: st.image(str(p),caption=p.stem,width="stretch")
        else: st.info("No captured survivor frames yet.")
elif panel==4:
    with st.container(border=True):
        st.subheader("Footage / AI Events")
        events=load_json(EVENTS_FILE,[])
        if events: st.dataframe(pd.DataFrame(events[-50:]),hide_index=True,width="stretch")
        else: st.info("Live AI events will appear here.")
elif panel==5:
    with st.container(border=True):
        st.subheader("Mission Graphs")
        if st.session_state.last_result:
            stats=st.session_state.last_result["stats"]
            st.bar_chart(pd.DataFrame({"Area %":stats}).T)
        hist=load_json(EVENTS_FILE,[])
        st.metric("Recorded mission events",len(hist))
        st.metric("Survivor alerts",len(alerts))
elif panel==6:
    with st.container(border=True):
        st.subheader("➕ DROP AID KIT")
        options=[a for a in alerts[-20:] if a.get("latitude") is not None]
        if not options:
            st.info("First detect a survivor/person. The selected target coordinate will appear here.")
        else:
            labels=[f"{a['alert_id']} • {a['latitude']:.6f}, {a['longitude']:.6f} • {a['confidence']*100:.0f}%" for a in options]
            idx=st.selectbox("Select survivor target",range(len(labels)),format_func=lambda i:labels[i])
            target=options[idx]
            st.success(f"TARGET LOCKED: {target['latitude']:.6f}, {target['longitude']:.6f}")
            kit=st.selectbox("Aid kit type",["Medical","Water","Food","Emergency Pack"])
            if st.button("🚁 CONFIRM AID DROP",type="primary",width="stretch"):
                drop={"drop_id":uid("AID"),"kit":kit,"latitude":target["latitude"],"longitude":target["longitude"],
                      "alert_id":target["alert_id"],"timestamp":now(),"status":"COMMAND LOGGED"}
                append_json(AID_FILE,drop)
                st.success(f"Aid drop logged for {target['latitude']:.6f}, {target['longitude']:.6f}.")

st.divider()
st.caption("Prototype command center • AI damage classification and survivor coordinates are estimates. Validate with trained responders, calibrated sensors, and appropriate geospatial/field instrumentation before operational use.")
st.caption(f"System time: {now()}  •  Alerts: {len(alerts)}  •  Aid drops: {len(aid_records)}")
