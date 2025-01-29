import streamlit as st
from PIL import Image
import cv2
import tempfile
import numpy as np
import torch
from ultralytics import YOLO
from twilio.rest import Client
import random

# Load YOLO model
model = YOLO('yolov8_trained_model.pt')

# Safety gear classes
safety_gear_classes = [
    'Gloves', 'Hardhat', 'Mask', 'NO-Gloves', 'NO-Mask', 'NO-Safety Boot', 
    'NO-Safety Vest', 'Person', 'Safety Boot', 'Safety Vest'
]

# Twilio setup for sending WhatsApp alerts
def send_alert(missing_items):
    account_sid = 'AC8ee95b70767f5f8741c8edbdf66fa124'
    auth_token = 'a9ac877af27c4a5400889f1f806e9e8b'
    client = Client(account_sid, auth_token)

    # Generate a random employer ID for this alert
    employer_id = random.randint(1000, 9999)
    missing_items_str = ', '.join(missing_items)
    message_body = f"Alert: Missing safety gear detected for Employer ID {employer_id}. Missing items: {missing_items_str}"

    client.messages.create(
        from_='whatsapp:+14155238886',
        body=message_body,
        to='whatsapp:+919159759423'
    )

    st.write(f"Alert sent to supervisor! Employer ID: {employer_id}, Missing items: {missing_items_str}")

# App UI configuration
st.title("🛠️ Smart Safety Gear Detection System")
st.sidebar.title("Settings")
st.sidebar.write("Configure detection and alert settings here.")

# Create detection mode buttons
mode = st.selectbox("Select Detection Mode", ["Image Detection", "Video Detection", "Live Camera"])

# Confidence threshold for detection
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.40)

# Function to handle image processing
def process_image(image):
    if image.mode == "RGBA":
        image = image.convert("RGB")
    image = np.array(image)

    # Perform inference using YOLO model
    results = model(image)
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
    missing_items = [item for item in safety_gear_classes if item not in detected_classes]
    
    if missing_items:
        send_alert(missing_items)
    
    st.image(results[0].plot(), caption="Processed Image", use_container_width=True)

# Function to handle video processing
def process_video(video_file):
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)

    alert_sent = False  # Flag to track if alert has already been sent during the video

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        missing_items = [item for item in safety_gear_classes if item not in detected_classes]

        # Send alert only once per video if missing items are detected
        if missing_items and not alert_sent:
            send_alert(missing_items)
            alert_sent = True  # Set flag to True to prevent sending multiple alerts

        st.image(results[0].plot(), caption="Processed Frame", use_container_width=True)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()


# Function to handle live camera detection
def process_live_camera():
    cap = cv2.VideoCapture(0)
    st.text("Opening live camera feed...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to open camera.")
            break
        results = model(frame)
        detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
        missing_items = [item for item in safety_gear_classes if item not in detected_classes]

        if missing_items:
            send_alert(missing_items)
        
        st.image(results[0].plot(), caption="Live Camera Frame", use_container_width=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()

# Execute chosen detection mode
if mode == "Image Detection":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        process_image(image)

elif mode == "Video Detection":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi"])
    if uploaded_video is not None:
        st.video(uploaded_video)
        process_video(uploaded_video)

elif mode == "Live Camera":
    if st.button("Start Live Camera Detection"):
        process_live_camera()
