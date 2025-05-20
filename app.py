import cv2
import streamlit as st
import numpy as np
import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import threading
import pandas as pd
import yt_dlp
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
from functools import lru_cache

# Cache model download to avoid reloading on every rerun
@lru_cache(maxsize=None)
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# Audio functions using pydub/simpleaudio
def play_alarm():
    try:
        audio = AudioSegment.from_file("alarm system.mp3")
        play(audio)
    except Exception as e:
        st.error(f"Gagal memutar alarm: {str(e)}")

def stop_alarm():
    pass  # pydub playback can't be stopped, but will auto-complete

# YouTube stream handling
def get_youtube_stream(url):
    try:
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'quiet': True,
            'noplaylist': True
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"Error getting YouTube stream: {str(e)}")
        return None

# Detection function
def detect_suspicious_activity(frame, model, conf_threshold, heatmap, aois,
                              activity_logs, max_repeated_movements, alarm_triggered,
                              heatmap_history):
    results = model(frame)
    current_activities = defaultdict(int)
    suspicious = False

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            label = result.names[int(box.cls[0])]

            if label == "person" and conf > conf_threshold:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                heatmap[y1:y2, x1:x2] += 1

                for idx, (ax1, ay1, ax2, ay2) in enumerate(aois):
                    if x1 >= ax1 and y1 >= ay1 and x2 <= ax2 and y2 <= ay2:
                        current_activities[idx] += 1

    for idx, count in current_activities.items():
        if count > 0:
            activity_logs[idx].append(datetime.datetime.now())

        cutoff = datetime.datetime.now() - datetime.timedelta(seconds=10)
        activity_logs[idx] = [t for t in activity_logs[idx] if t > cutoff]

        if len(activity_logs[idx]) > max_repeated_movements:
            suspicious = True
            if not alarm_triggered[0]:
                alarm_triggered[0] = True
                st.warning(f"\U0001F6A8 *ALARM*: Gerakan mencurigakan di Zona {idx+1}!")
                threading.Thread(target=play_alarm, daemon=True).start()

    heatmap_max = int(np.max(heatmap))
    timestamp = datetime.datetime.now().strftime("%H:%M:%S")

    heatmap_history.append({"time": timestamp, "activity": heatmap_max})

    if heatmap_max > 1:
        suspicious = True
        if not alarm_triggered[0]:
            alarm_triggered[0] = True
            st.warning("\U0001F6A8 *ALARM*: Aktivitas mencurigakan terdeteksi!")
            threading.Thread(target=play_alarm, daemon=True).start()

    elif heatmap_max < 1:
        if alarm_triggered[0]:
            alarm_triggered[0] = False
            stop_alarm()
            st.info("\u2705 Tidak ada aktivitas mencurigakan. Alarm dimatikan.")

    return frame

# Streamlit UI Configuration
st.set_page_config(page_title="Smart Security System", layout="wide")
st.title("\U0001F512 Smart Security System")

# Session state initialization
if 'alarm_triggered' not in st.session_state:
    st.session_state.alarm_triggered = False
if 'cap' not in st.session_state:
    st.session_state.cap = None

with st.sidebar:
    st.header("\u2699\ufe0f Pengaturan Sistem")
    conf_threshold = st.slider("*Tingkat Kepercayaan Deteksi*", 0.0, 1.0, 0.5, 0.01)
    max_reps = st.number_input("*Batas Gerakan untuk Alarm*", 1, 50, 5)
    youtube_url = st.text_input("Masukkan URL YouTube Live", placeholder="https://www.youtube.com/...")

    st.subheader("\U0001F4DC Zona Pengawasan (AOI)")
    num_aois = st.number_input("Jumlah Zona", 0, 5, 1)
    aois = []
    for i in range(num_aois):
        with st.expander(f"Pengaturan Zona {i+1}"):
            x1 = st.slider(f"X1 (Kiri)", 0, 1920, 200, key=f"x1_{i}")
            y1 = st.slider(f"Y1 (Atas)", 0, 1080, 200, key=f"y1_{i}")
            x2 = st.slider(f"X2 (Kanan)", 0, 1920, 800, key=f"x2_{i}")
            y2 = st.slider(f"Y2 (Bawah)", 0, 1080, 600, key=f"y2_{i}")
            aois.append((x1, y1, x2, y2))

# Main columns
col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F3A5 Live Camera Feed")
    camera_placeholder = st.empty()
with col2:
    st.subheader("\U0001F4CA Grafik Aktivitas")
    heatmap_placeholder = st.empty()

# Control buttons
if st.sidebar.button("\U0001F3A5 Mulai Streaming YouTube") and youtube_url:
    if "youtube.com" not in youtube_url:
        st.error("URL YouTube tidak valid!")
    else:
        stream_url = get_youtube_stream(youtube_url)
        if stream_url:
            st.session_state.cap = cv2.VideoCapture(stream_url)
            st.success("Streaming dimulai!")

if st.sidebar.button("\u23F9 Hentikan Streaming"):
    if st.session_state.cap:
        st.session_state.cap.release()
        st.session_state.cap = None
        stop_alarm()
    st.rerun()

# Main processing loop
if st.session_state.cap and st.session_state.cap.isOpened():
    heatmap = np.zeros((1080, 1920), dtype=np.uint8)
    activity_logs = defaultdict(list)
    heatmap_history = deque(maxlen=100)
    
    try:
        while st.session_state.cap.isOpened():
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.error("❌ Gagal membaca frame.")
                break
            
            heatmap = (heatmap * 0.95).astype(np.uint8)
            frame = cv2.resize(frame, (1920, 1080))
            
            processed_frame = detect_suspicious_activity(
                frame, model, conf_threshold, heatmap, aois,
                activity_logs, max_reps, [st.session_state.alarm_triggered],
                heatmap_history
            )
            
            camera_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), 
                                    channels="RGB", use_container_width=True)
            
            if heatmap_history:
                df_heat = pd.DataFrame(heatmap_history)
                heatmap_placeholder.line_chart(df_heat.set_index("time"))

    except Exception as e:
        st.error(f"Terjadi error: {str(e)}")
        st.session_state.cap.release()

else:
    st.info("ℹ Silakan masukkan URL YouTube Live dan klik 'Mulai Streaming'")
