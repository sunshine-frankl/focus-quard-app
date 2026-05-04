import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import time
from collections import deque
import plotly.graph_objects as go
import threading
import queue
import io
import hashlib
import uuid

TELEGRAM_BOT_TOKEN = "8702324957:AAE45czlrbs5nt9q7uxxwgukArUpNjoZ-j0"
TELEGRAM_CHAT_ID   = "-1003964944926"

EAR_THRESHOLD      = 0.20
EAR_CONSEC_FRAMES  = 3
GAZE_THRESHOLD     = 0.12
MAX_BLINK_RATE     = 25
YOLO_MODEL         = "yolov8n.pt"
YOLO_CONF          = 0.45
SUSPICIOUS_OBJECTS = {"cell phone", "book", "remote", "laptop", "tv"}
VIOLATION_COOLDOWN = 15.0
GAZE_GRACE_SEC     = 2.5
ABSENCE_GRACE_SEC  = 3.0

L_EAR_IDX  = [33,  160, 158, 133, 153, 144]
R_EAR_IDX  = [362, 385, 387, 263, 373, 380]
L_IRIS_IDX = 468
R_IRIS_IDX = 473
L_EYE_LEFT  = 33;  L_EYE_RIGHT  = 133
R_EYE_LEFT  = 362; R_EYE_RIGHT  = 263


def ear(lm, indices, w, h):
    pts = np.array([(lm[i].x * w, lm[i].y * h) for i in indices])
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    return (A + B) / (2.0 * C)


def iris_ratio(lm, iris_idx, left_idx, right_idx, w, h):
    ix   = lm[iris_idx].x * w
    ex_l = lm[left_idx].x  * w
    ex_r = lm[right_idx].x * w
    width = ex_r - ex_l
    return 0.5 if abs(width) < 1 else (ix - ex_l) / width


@st.cache_resource
def get_face_mesh():
    return mp.solutions.face_mesh.FaceMesh(
        max_num_faces=4, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

@st.cache_resource
def get_yolo():
    try:
        from ultralytics import YOLO
        return YOLO(YOLO_MODEL)
    except Exception:
        return None

try:
    import requests as _req
except ImportError:
    _req = None

@st.cache_resource
def get_notifier():
    class _N:
        def __init__(self):
            self._q = queue.Queue(maxsize=20)
            threading.Thread(target=self._loop, daemon=True).start()
        def send(self, img, cap):
            if not (_req and TELEGRAM_BOT_TOKEN): return
            try: self._q.put_nowait((img.copy(), cap))
            except queue.Full: pass
        def _loop(self):
            while True:
                img, cap = self._q.get()
                try:
                    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ok:
                        _req.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto",
                            data={"chat_id": TELEGRAM_CHAT_ID, "caption": cap, "parse_mode": "Markdown"},
                            files={"photo": ("v.jpg", io.BytesIO(buf.tobytes()), "image/jpeg")},
                            timeout=15)
                except Exception: pass
                finally: self._q.task_done()
    return _N()


def make_state():
    return {
        "session_start":    time.time(),
        "total_blinks":     0,
        "frame_counter":    0,
        "last_blink_time":  time.time(),
        "focus_scores":     [],
        "violations_log":   [],
        "vio_first":        {},
        "vio_sent":         {},
        "gaze_buf":         [],
        "focus_score":      0,
        "gaze_ui":          "—",
        "blink_rate":       0.0,
        "session_time":     0,
        "status":           "⏸ Waiting",
        "color":            "#555555",
        "active_violations": [],
    }


def process_frame(img_bytes, state, settings):
    """Decode bytes → process → return (annotated_rgb, updated_state)."""
    try:
        arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return None, state

        h, w = img.shape[:2]
        face_mesh = get_face_mesh()
        rgb       = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results   = face_mesh.process(rgb)
        ann       = img.copy()

        faces_count   = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        person_absent = faces_count == 0
        gaze_cv, gaze_ui = ("No person", "🚫 None") if person_absent else ("Center", "👀 Center")

        if results.multi_face_landmarks:
            for face_lm in results.multi_face_landmarks:
                lm = face_lm.landmark

                # Bounding box
                xs = [int(l.x * w) for l in lm]
                ys = [int(l.y * h) for l in lm]
                cv2.rectangle(ann, (max(0,min(xs)-8), max(0,min(ys)-8)),
                              (min(w,max(xs)+8), min(h,max(ys)+8)), (0,255,120), 2)

                # Eye dots
                for idx in L_EAR_IDX + R_EAR_IDX:
                    cv2.circle(ann, (int(lm[idx].x*w), int(lm[idx].y*h)), 2, (0,255,255), -1)
                for ii in [L_IRIS_IDX, R_IRIS_IDX]:
                    cv2.circle(ann, (int(lm[ii].x*w), int(lm[ii].y*h)), 5, (255,80,80), -1)

                # EAR / blink
                avg_ear = (ear(lm,L_EAR_IDX,w,h) + ear(lm,R_EAR_IDX,w,h)) / 2.0
                if avg_ear < EAR_THRESHOLD:
                    state["frame_counter"] += 1
                    if (state["frame_counter"] >= EAR_CONSEC_FRAMES
                            and time.time() - state["last_blink_time"] > 0.4):
                        state["total_blinks"] += 1
                        state["last_blink_time"] = time.time()
                else:
                    state["frame_counter"] = 0

                # Gaze
                l_r = iris_ratio(lm, L_IRIS_IDX, L_EYE_LEFT, L_EYE_RIGHT, w, h)
                r_r = iris_ratio(lm, R_IRIS_IDX, R_EYE_LEFT, R_EYE_RIGHT, w, h)
                buf = state["gaze_buf"]
                buf.append((l_r + r_r) / 2.0)
                if len(buf) > 6: buf.pop(0)
                smooth = sum(buf) / len(buf)

                if   smooth < 0.5 - GAZE_THRESHOLD: gaze_cv, gaze_ui = "Left",  "👈 Left"
                elif smooth > 0.5 + GAZE_THRESHOLD: gaze_cv, gaze_ui = "Right", "👉 Right"
                else:
                    dev = abs(smooth - 0.5)
                    if dev > GAZE_THRESHOLD * 0.6:
                        side  = "Left" if smooth < 0.5 else "Right"
                        arrow = "👈"   if smooth < 0.5 else "👉"
                        gaze_cv, gaze_ui = f"Slight {side}", f"{arrow} Slight {side}"
                    else:
                        gaze_cv, gaze_ui = "Center", "👀 Center"
        else:
            state["gaze_buf"] = []

        # YOLO
        yolo_objects = []
        yolo = get_yolo()
        if settings.get("enable_yolo") and yolo:
            try:
                res = yolo.predict(img, imgsz=416, conf=YOLO_CONF, verbose=False)
                if res and res[0].boxes is not None:
                    for box, cf, cid in zip(res[0].boxes.xyxy.cpu().numpy(),
                                             res[0].boxes.conf.cpu().numpy(),
                                             res[0].boxes.cls.cpu().numpy().astype(int)):
                        name = yolo.names.get(int(cid), str(cid))
                        if name in SUSPICIOUS_OBJECTS:
                            bx1,by1,bx2,by2 = box.astype(int)
                            yolo_objects.append({"class":name,"conf":float(cf),
                                                 "box":(int(bx1),int(by1),int(bx2),int(by2))})
            except Exception: pass
        for obj in yolo_objects:
            bx1,by1,bx2,by2 = obj["box"]
            cv2.rectangle(ann,(bx1,by1),(bx2,by2),(0,0,255),2)
            cv2.putText(ann,f"{obj['class']} {obj['conf']:.2f}",(bx1+2,by1-6),
                        cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)

        # Score
        session_time = max(1, time.time() - state["session_start"])
        blink_rate   = (state["total_blinks"] / session_time) * 60
        score = max(15, min(100,
            92 - (77 if person_absent else 0)
               - (35 if not person_absent and gaze_cv not in ("Center",) else 0)
               - max(0, (blink_rate - MAX_BLINK_RATE) * 0.8)
               - (40 if faces_count > 1 else 0)
               - len(yolo_objects) * 25))

        state["focus_scores"].append(score)
        if len(state["focus_scores"]) > 400:
            state["focus_scores"] = state["focus_scores"][-400:]

        # Violations
        now = time.time()
        active = []
        if settings.get("track_absence") and person_absent:
            active.append(("person_absent", "🚫 Person absent"))
        if settings.get("track_gaze") and not person_absent and "Center" not in gaze_cv:
            active.append(("gaze_away", gaze_ui))
        if settings.get("track_extra") and faces_count > 1:
            active.append(("extra_face", f"👥 {faces_count} faces"))
        for obj in yolo_objects:
            cls = obj["class"]
            if settings.get("track_phone") and cls in ("cell phone","remote"):
                active.append(("phone", f"📱 Phone ({obj['conf']:.2f})"))
            elif settings.get("track_book") and cls == "book":
                active.append(("book", f"📚 Book ({obj['conf']:.2f})"))
            elif settings.get("track_objects") and cls in ("laptop","tv"):
                active.append((cls, f"💻 {cls.capitalize()} ({obj['conf']:.2f})"))

        active_types = {v[0] for v in active}
        for t in list(state["vio_first"]):
            if t not in active_types: del state["vio_first"][t]

        grace = {"person_absent": ABSENCE_GRACE_SEC, "gaze_away": GAZE_GRACE_SEC, "extra_face": 1.0}
        notifier = get_notifier()
        for vtype, vtext in active:
            if vtype not in state["vio_first"]:
                state["vio_first"][vtype] = now; continue
            if now - state["vio_first"][vtype] < grace.get(vtype, 0.6): continue
            if now - state["vio_sent"].get(vtype, 0) < VIOLATION_COOLDOWN: continue
            state["vio_sent"][vtype] = now
            ts = time.strftime("%H:%M:%S")
            state["violations_log"].insert(0, f"[{ts}] {vtext}")
            if len(state["violations_log"]) > 20:
                state["violations_log"] = state["violations_log"][:20]
            if settings.get("enable_telegram"):
                notifier.send(ann,
                    f"🚨 *Violation*\n👤 {settings.get('student_name','?')}\n"
                    f"⏰ {ts}\n📋 {vtext}\n📉 Focus: {int(score)}%")

        # Status
        if   person_absent:       status, color = "🔴 No person",   "#ff4444"
        elif active:              status, color = "🔴 Violation",   "#ff4444"
        elif score > 78:          status, color = "🟢 Focused",     "#00ff9d"
        elif score > 55:          status, color = "🟡 Drifting",    "#ffcc00"
        else:                     status, color = "🔴 Not focused", "#ff4444"

        # Overlay on ann
        font = cv2.FONT_HERSHEY_SIMPLEX
        sc   = (80,255,140) if score > 78 else ((0,200,255) if score > 55 else (80,80,255))
        for text, y, col in [(f"Focus {int(score)}%",28,sc),
                              (f"Gaze  {gaze_cv}",   52,(220,220,220)),
                              (f"Faces {faces_count}",76,(220,220,220))]:
            cv2.putText(ann, text, (12,y), font, 0.55, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(ann, text, (12,y), font, 0.55, col,     1, cv2.LINE_AA)

        state.update({
            "focus_score":       score,
            "gaze_ui":           gaze_ui,
            "blink_rate":        blink_rate,
            "session_time":      session_time,
            "status":            status,
            "color":             color,
            "active_violations": [t for _, t in active],
        })

        ann_rgb = cv2.cvtColor(ann, cv2.COLOR_BGR2RGB)
        return ann_rgb, state

    except Exception as e:
        state["status"] = f"⚠️ Error: {str(e)[:40]}"
        return None, state


# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE
# ══════════════════════════════════════════════════════════════════════════════
def _hash(pwd): return hashlib.sha256(pwd.encode()).hexdigest()

@st.cache_resource
def get_db():
    return {
        "users": {
            "admin":   {"name":"Administrator","password":_hash("admin"),  "role":"admin"},
            "teacher": {"name":"Teacher",       "password":_hash("teacher"),"role":"teacher"},
            "student": {"name":"Student",       "password":_hash("student"),"role":"student"},
        },
        "exams": {},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="Focus Guard", page_icon="🧠", layout="wide")
st.markdown("""
<style>
.stApp{background-color:#0e1117}
[data-testid="stMetric"]{background:#1c2333;border-radius:10px;padding:12px 16px;border:1px solid #2a3550}
.vrow{background:#1f1318;border-left:3px solid #ff4444;border-radius:0 6px 6px 0;padding:7px 12px;margin:4px 0;color:#ffaaaa;font-size:.88rem}
.exam-card{background:#1c2333;border-radius:10px;padding:16px 20px;border:1px solid #2a3550;margin-bottom:10px}
.badge-pending{background:#2a2000;color:#ffd60a;border-radius:4px;padding:2px 8px;font-size:.78rem}
.badge-active{background:#002a0a;color:#00ff9d;border-radius:4px;padding:2px 8px;font-size:.78rem}
.badge-submitted{background:#00152a;color:#00b4ff;border-radius:4px;padding:2px 8px;font-size:.78rem}
</style>""", unsafe_allow_html=True)

ROLE_ICON = {"admin":"🛡️","teacher":"👨‍🏫","student":"🎓"}


# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ══════════════════════════════════════════════════════════════════════════════
def login_page():
    _, col, _ = st.columns([1,1.2,1])
    with col:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## 🧠 Focus Guard")
        st.caption("AI Proctoring System · Please sign in")
        st.divider()
        st.markdown("**Quick login:**")
        c1,c2,c3 = st.columns(3)
        db = get_db()
        for btn, un in [(c1,"admin"),(c2,"teacher"),(c3,"student")]:
            lbl = {"admin":"🛡️ Admin","teacher":"👨‍🏫 Teacher","student":"🎓 Student"}[un]
            if btn.button(lbl, use_container_width=True):
                u = db["users"][un]
                st.session_state.update({"authenticated":True,"username":un,
                                         "display_name":u["name"],"role":u["role"]}); st.rerun()
        st.divider()
        st.markdown("**Or sign in manually:**")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Sign in", use_container_width=True, type="primary"):
            u = db["users"].get(username)
            if u and u["password"] == _hash(password):
                st.session_state.update({"authenticated":True,"username":username,
                                         "display_name":u["name"],"role":u["role"]}); st.rerun()
            else:
                st.error("Invalid username or password")

if not st.session_state.get("authenticated"):
    login_page(); st.stop()

role    = st.session_state["role"]
uname   = st.session_state["username"]
display = st.session_state["display_name"]

with st.sidebar:
    st.markdown(f"### {ROLE_ICON.get(role,'👤')} {display}")
    st.caption(f"Role: **{role}** · `{uname}`")
    if st.button("🚪 Sign out", use_container_width=True):
        st.session_state.clear(); st.rerun()
    st.divider()


# ══════════════════════════════════════════════════════════════════════════════
#  ADMIN
# ══════════════════════════════════════════════════════════════════════════════
def admin_page():
    db = get_db()
    st.title("🛡️ Admin Panel"); st.divider()
    tab_u, tab_e = st.tabs(["👥 Users","📋 All Exams"])

    with tab_u:
        st.subheader("Current users")
        for un, info in list(db["users"].items()):
            c1,c2,c3,c4 = st.columns([2,2,1.5,1])
            c1.markdown(f"**{info['name']}**"); c2.markdown(f"`{un}`")
            c3.markdown(f"{ROLE_ICON.get(info['role'],'')} {info['role']}")
            if un != "admin" and c4.button("🗑️", key=f"del_{un}"):
                del db["users"][un]; st.rerun()
        st.divider(); st.subheader("➕ Add new user")
        c1,c2,c3,c4 = st.columns([2,2,2,1.5])
        nn=c1.text_input("Full name",key="nu_n"); nu=c2.text_input("Username",key="nu_u")
        np_=c3.text_input("Password",key="nu_p",type="password"); nr=c4.selectbox("Role",["teacher","student"],key="nu_r")
        if st.button("Add user",type="primary"):
            if nn and nu and np_:
                if nu in db["users"]: st.error("Username exists")
                else:
                    db["users"][nu]={"name":nn,"password":_hash(np_),"role":nr}
                    st.success(f"User **{nu}** created"); st.rerun()
            else: st.warning("Fill all fields")

    with tab_e:
        if not db["exams"]: st.info("No exams yet")
        else:
            for eid,ex in db["exams"].items():
                st.markdown(f"""<div class="exam-card"><b>{ex['title']}</b> &nbsp;
                    <span class="badge-{ex['status']}">{ex['status'].upper()}</span><br>
                    <small>Teacher:{ex['teacher']} · Student:{ex['student']} · {ex['created_at']}</small>
                    </div>""", unsafe_allow_html=True)
                if ex["status"]=="submitted" and ex.get("result"):
                    r=ex["result"]; m1,m2,m3=st.columns(3)
                    m1.metric("Avg Focus",f"{r['avg_focus']:.0f}%")
                    m2.metric("Blinks/min",f"{r['blink_rate']:.1f}")
                    m3.metric("Violations",r['violations'])


# ══════════════════════════════════════════════════════════════════════════════
#  TEACHER
# ══════════════════════════════════════════════════════════════════════════════
def teacher_page():
    db = get_db()
    st.title("👨‍🏫 Teacher Panel"); st.divider()
    tab_c, tab_r = st.tabs(["➕ Create Exam","📊 Results"])

    with tab_c:
        st.subheader("Create new exam session")
        students = {u:i["name"] for u,i in db["users"].items() if i["role"]=="student"}
        if not students: st.warning("No students registered yet.")
        else:
            title=st.text_input("Exam title",placeholder="e.g. Midterm Exam — Math")
            s_un=st.selectbox("Assign to student",list(students.keys()),
                              format_func=lambda u:f"{students[u]} ({u})")
            tg=st.checkbox("📨 Send violations to Telegram",value=True)
            if st.button("📋 Create exam",type="primary",use_container_width=True):
                if not title: st.warning("Enter title")
                else:
                    eid=str(uuid.uuid4())[:8]
                    db["exams"][eid]={"title":title,"teacher":uname,"student":s_un,
                        "created_at":time.strftime("%Y-%m-%d %H:%M"),
                        "status":"pending","telegram":tg,"result":None}
                    st.success(f"✅ Exam **{title}** created"); st.rerun()

    with tab_r:
        my={eid:ex for eid,ex in db["exams"].items() if ex["teacher"]==uname}
        if not my: st.info("No exams yet")
        else:
            for eid,ex in my.items():
                st.markdown(f"""<div class="exam-card"><b>{ex['title']}</b> &nbsp;
                    <span class="badge-{ex['status']}">{ex['status'].upper()}</span><br>
                    <small>Student:<b>{ex['student']}</b> · {ex['created_at']}</small></div>""",
                    unsafe_allow_html=True)
                if ex["status"]=="submitted" and ex.get("result"):
                    r=ex["result"]; m1,m2,m3,m4=st.columns(4)
                    m1.metric("Avg Focus",f"{r['avg_focus']:.0f}%")
                    m2.metric("Min Focus",f"{r['min_focus']:.0f}%")
                    m3.metric("Blinks/min",f"{r['blink_rate']:.1f}")
                    m4.metric("Violations",r['violations'])
                    if r.get("focus_scores"):
                        fig=go.Figure()
                        fig.add_trace(go.Scatter(y=r["focus_scores"],mode="lines",
                            line=dict(color="#00ff9d",width=2),
                            fill="tozeroy",fillcolor="rgba(0,255,157,0.08)"))
                        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                            height=160,showlegend=False,margin=dict(l=0,r=0,t=4,b=0),
                            yaxis=dict(range=[0,100],ticksuffix="%",
                                       gridcolor="rgba(255,255,255,0.05)",tickfont=dict(color="#aaa")),
                            xaxis=dict(showgrid=False,showticklabels=False))
                        st.plotly_chart(fig,use_container_width=True,key=f"r_{eid}")
                    st.divider()
                elif ex["status"]=="pending": st.caption("⏳ Waiting for student")
                elif ex["status"]=="active":  st.caption("🟢 Exam in progress...")


# ══════════════════════════════════════════════════════════════════════════════
#  STUDENT
# ══════════════════════════════════════════════════════════════════════════════
def student_page():
    db = get_db()
    my_exams = {eid:ex for eid,ex in db["exams"].items()
                if ex["student"]==uname and ex["status"] in ("pending","active")}

    st.title("🎓 Student Panel"); st.divider()

    if not my_exams:
        st.info("📭 No active exams assigned to you.")
        done = {eid:ex for eid,ex in db["exams"].items()
                if ex["student"]==uname and ex["status"]=="submitted"}
        if done:
            st.subheader("✅ Completed exams")
            for eid,ex in done.items():
                st.markdown(f"""<div class="exam-card"><b>{ex['title']}</b> &nbsp;
                    <span class="badge-submitted">SUBMITTED</span><br>
                    <small>Teacher:{ex['teacher']} · {ex['created_at']}</small></div>""",
                    unsafe_allow_html=True)
        return

    eid, exam = next(iter(my_exams.items()))
    st.subheader(f"📋 {exam['title']}")
    c1,c2,c3=st.columns(3)
    c1.metric("Teacher",exam["teacher"])
    c2.metric("Started",exam["created_at"])
    c3.metric("Status", exam["status"].upper())
    st.divider()

    if exam["status"]=="pending":
        db["exams"][eid]["status"]="active"; st.rerun()

    settings = dict(student_name=display, enable_telegram=exam.get("telegram",True),
                    enable_yolo=True, track_absence=True, track_gaze=True,
                    track_extra=True, track_phone=True, track_book=True, track_objects=True)

    # Session state init
    sk = f"st_{eid}"
    if sk not in st.session_state:
        st.session_state[sk] = make_state()

    # ── Layout ────────────────────────────────────────────────────────────────
    col_cam, col_side = st.columns([2.2, 1])

    with col_side:
        st.subheader("📊 Metrics")
        ph_img    = col_cam.empty()   # annotated frame placeholder
        c1,c2=st.columns(2)
        ph_focus  = c1.empty()
        ph_time   = c2.empty()
        c3,c4=st.columns(2)
        ph_blink  = c3.empty()
        ph_gaze   = c4.empty()
        st.divider()
        ph_status = st.empty()
        st.divider()
        st.subheader("🚨 Violations")
        ph_viol   = st.empty()
        st.divider()
        st.warning("⚠️ Do not close this tab until you submit!")
        submit = st.button("✅ Submit exam", type="primary", use_container_width=True)

    with col_cam:
        st.subheader("🎥 Camera")
        st.caption("📸 Click **Take Photo** to capture a frame for analysis")
        camera_image = st.camera_input("", key=f"cam_{eid}", label_visibility="collapsed")

    # ── Process new frame ─────────────────────────────────────────────────────
    if camera_image is not None:
        img_bytes = camera_image.getvalue()
        # Only process if it's a new frame (compare size as quick check)
        last_size = st.session_state.get(f"last_img_size_{eid}", -1)
        if len(img_bytes) != last_size:
            st.session_state[f"last_img_size_{eid}"] = len(img_bytes)
            ann_rgb, new_state = process_frame(img_bytes, st.session_state[sk], settings)
            st.session_state[sk] = new_state
            if ann_rgb is not None:
                st.session_state[f"last_ann_{eid}"] = ann_rgb

    # ── Show last annotated frame ─────────────────────────────────────────────
    if f"last_ann_{eid}" in st.session_state:
        ph_img.image(st.session_state[f"last_ann_{eid}"], use_container_width=True)

    # ── Show metrics ──────────────────────────────────────────────────────────
    s = st.session_state[sk]
    ph_focus.metric("🎯 Focus",       f"{int(s['focus_score'])}%")
    ph_time.metric( "⏱ Session",     f"{int(s['session_time'])} s")
    ph_blink.metric("👁 Blinks/min",  f"{s['blink_rate']:.1f}")
    ph_gaze.metric( "👀 Gaze",        s["gaze_ui"])
    ph_status.markdown(
        f"<h3 style='color:{s['color']};margin:0'>{s['status']}</h3>",
        unsafe_allow_html=True)

    vlog = s["violations_log"]
    if vlog:
        ph_viol.markdown("".join(f'<div class="vrow">{v}</div>' for v in vlog[:8]),
                         unsafe_allow_html=True)
    elif s["active_violations"]:
        ph_viol.markdown("".join(f'<div class="vrow">{v}</div>' for v in s["active_violations"][:8]),
                         unsafe_allow_html=True)
    else:
        ph_viol.success("No violations ✅")

    # Focus chart
    fs = s["focus_scores"]
    if len(fs) > 2:
        fig=go.Figure()
        fig.add_trace(go.Scatter(y=fs,mode="lines",
            line=dict(color="#00ff9d",width=2.5),
            fill="tozeroy",fillcolor="rgba(0,255,157,0.08)"))
        fig.add_hline(y=78,line_color="rgba(0,255,157,0.3)",line_dash="dot")
        fig.add_hline(y=55,line_color="rgba(255,204,0,0.3)", line_dash="dot")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
            height=200,showlegend=False,margin=dict(l=0,r=0,t=8,b=0),
            yaxis=dict(range=[0,100],ticksuffix="%",
                       gridcolor="rgba(255,255,255,0.05)",tickfont=dict(color="#aaa")),
            xaxis=dict(showgrid=False,showticklabels=False))
        col_cam.plotly_chart(fig,use_container_width=True,key=f"ch_{eid}")

    # ── Submit ────────────────────────────────────────────────────────────────
    if submit:
        fs = s["focus_scores"]
        vlog = s["violations_log"]
        db["exams"][eid]["result"] = {
            "avg_focus":    sum(fs)/len(fs) if fs else 0,
            "min_focus":    min(fs) if fs else 0,
            "blink_rate":   s["blink_rate"],
            "violations":   len(vlog),
            "focus_scores": fs[-100:],
            "submitted_at": time.strftime("%H:%M:%S"),
        }
        db["exams"][eid]["status"] = "submitted"
        st.success("✅ Exam submitted!"); st.rerun()

    # ── Auto-rerun to keep session_time ticking ───────────────────────────────
    time.sleep(1)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTER
# ══════════════════════════════════════════════════════════════════════════════
if   role == "admin":   admin_page()
elif role == "teacher": teacher_page()
elif role == "student": student_page()
else: st.error("Unknown role")

st.caption("Focus Guard · MediaPipe FaceMesh + YOLOv8 + Telegram")
