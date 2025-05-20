import cv2
from pytube import YouTube
import streamlit as st
import numpy as np
import datetime
from collections import defaultdict, deque
from ultralytics import YOLO
import pygame
import threading
import pandas as pd
import yt_dlp

# Load model YOLOv8
model = YOLO("yolov8n.pt")

# Function to play the alarm
def play_alarm():
    try:
        pygame.mixer.init()
        pygame.mixer.music.load("alarm system.mp3")
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"Gagal memutar alarm: {e}")

# Function to stop the alarm
def stop_alarm():
    try:
        if pygame.mixer.get_init():
            pygame.mixer.music.stop()
    except Exception as e:
        print(f"Gagal menghentikan alarm: {e}")

# Function to get YouTube stream URL
def get_youtube_stream(url):
    ydl_opts = {
        'format': 'best[ext=mp4]',
        'quiet': True,
        'noplaylist': True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info['url']

# Main detection function
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

# Streamlit configuration
st.set_page_config(page_title="Smart Security System", layout="wide")
st.title("\U0001F512 Smart Security System")

with st.sidebar:
    st.header("\u2699\ufe0f Pengaturan Sistem")
    video_source = st.radio("*Sumber Video*", ["Webcam", "CCTV (HDMI via Capture Card)", "YouTube Live"], index=0)
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

col1, col2 = st.columns(2)
with col1:
    st.subheader("\U0001F3A5 Live Camera Feed")
    camera_placeholder = st.empty()
with col2:
    st.subheader("\U0001F4CA Grafik Aktivitas")
    heatmap_placeholder = st.empty()

status_text = st.empty()
status_text.info("\U0001F7E2 *Sistem aktif*. Menunggu deteksi...")

cap = None
if video_source == "Webcam":
    if st.sidebar.button("\U0001F3A5 Mulai Streaming Webcam"):
        cap = cv2.VideoCapture(0)

elif video_source == "CCTV (HDMI via Capture Card)":
    cam_idx = st.sidebar.number_input("Indeks Kamera CCTV", 0, 10, 0)
    if st.sidebar.button("\U0001F517 Sambungkan ke CCTV"):
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

elif video_source == "YouTube Live":
    if st.sidebar.button("\U0001F3A5 Mulai Streaming YouTube"):
        if youtube_url:
            stream_url = get_youtube_stream(youtube_url)
            if stream_url:
                cap = cv2.VideoCapture(stream_url)

if cap and cap.isOpened():
    heatmap = np.zeros((1080, 1920), dtype=np.uint8)
    activity_logs = defaultdict(list)
    alarm_triggered = [False]
    heatmap_history = deque(maxlen=100)

    while True:
        ret, frame = cap.read()
        if not ret:
            status_text.error("❌ Gagal membaca frame.")
            break
        
        heatmap = (heatmap * 0.95).astype(np.uint8)

        frame = cv2.resize(frame, (1920, 1080))
        frame = detect_suspicious_activity(
            frame, model, conf_threshold, heatmap, aois,
            activity_logs, max_reps, alarm_triggered, heatmap_history
        )

        camera_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        df_heat = pd.DataFrame(heatmap_history)
        if not df_heat.empty:
            heatmap_placeholder.line_chart(df_heat.set_index("time"))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
else:
    status_text.warning("⚠ Silakan pilih sumber video dan pastikan kamera terhubung.")
