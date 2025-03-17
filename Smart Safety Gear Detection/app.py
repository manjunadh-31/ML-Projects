import streamlit as st
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLO
from twilio.rest import Client
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import tempfile
import random



# Configure YOLO model path
MODEL_PATH = r"C:\Users\manju\Downloads\best.pt"
model = YOLO(MODEL_PATH)

# Twilio and Email configuration
EMAIL_ADDRESS = "manjunadhxxxx@gmail.com"
EMAIL_PASSWORD = "phjr zzxf xxxx nohn"
RECIPIENT_EMAIL = "manjunadhgxxxx@gmail.com"
TWILIO_SID = "ACfc99751c99b2f90a33xxxx49fa25fc9f"
TWILIO_AUTH_TOKEN = "e1796dcc3c5d39xxxx31d08b68a8f1b6"
TWILIO_WHATSAPP_NUMBER = "whatsapp:+14155xxxx86"
RECIPIENT_WHATSAPP_NUMBER = "whatsapp:+9199596xxxx5"

# Safety gear classes
safety_gear_classes = ['Gloves', 'Hardhat', 'Mask', 'NO-Gloves', 'NO-Hardhat', 'NO-Mask',
                       'NO-Safety Boot', 'NO-Safety Vest', 'Person', 'Safety Boot', 'Safety Vest']
colors = {'safe': (0, 255, 0), 'unsafe': (0, 0, 255)}

# Streamlit app setup
st.set_page_config(page_title="Smart Safety Gear Detection", page_icon="🛠️", layout="wide")
st.markdown("<h1 style='text-align: center;'>🛠️ Smart Safety Gear Detection System 🛠️</h1>", unsafe_allow_html=True)

# Sidebar for settings
st.sidebar.header("Settings")
alerts_enabled = st.sidebar.checkbox("Enable Alerts", value=True)
enable_email = st.sidebar.checkbox("Enable Email Alerts", value=True)
enable_whatsapp = st.sidebar.checkbox("Enable WhatsApp Alerts", value=True)
# Sidebar for settings
st.sidebar.header("Settings")
# Confidence Threshold
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 1.0, 0.5, help="Minimum confidence score for detections."
)
# IoU Threshold
iou_threshold = st.sidebar.slider(
    "IoU Threshold", 0.1, 1.0, 0.5, help="Intersection over Union (IoU) threshold for filtering overlapping detections."
)
# Maximum Detections
max_detections = st.sidebar.slider(
    "Maximum Detections", 1, 100, 50, help="Maximum number of objects to detect per image/frame."
)
# Resize Image
resize_width = st.sidebar.slider(
    "Resize Width", 256, 1920, 640, step=64, help="Resize image width for faster processing."
)
resize_height = st.sidebar.slider(
    "Resize Height", 256, 1920, 640, step=64, help="Resize image height for faster processing."
)

# Select Classes
selected_classes = st.sidebar.multiselect(
    "Classes to Detect",
    safety_gear_classes,
    default=['Gloves', 'Hardhat', 'Mask', 'NO-Gloves', 'NO-Hardhat', 'NO-Mask',
           'NO-Safety Boot', 'NO-Safety Vest', 'Person', 'Safety Boot', 'Safety Vest'],
    help="Select specific classes to detect (e.g., Hardhat, Safety Vest)."
)
detect_all_classes = st.sidebar.checkbox("Detect All Classes", value=True)

# Non-Maximum Suppression (NMS)
nms_threshold = st.sidebar.slider(
    "NMS Threshold", 0.1, 1.0, 0.5, help="Threshold for removing overlapping bounding boxes."
)


# Alert functions
def send_email_alert(missing_items):
    # Format the missing items into a readable string
    missing_items_str = ", ".join(missing_items)
    
    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = "Safety Gear Alert 🚨"
    
    # Include the missing items in the email body
    msg.attach(MIMEText(f"⚠ Missing safety gear detected! Immediate attention required. Missing items: {missing_items_str}", 'plain'))
    
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        st.sidebar.success("🚀 Email Alert Sent Successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Email Alert Error: {e}")


def send_whatsapp_alert(missing_items):
    # Format the missing items into a readable string
    missing_items_str = ", ".join(missing_items)
    
    client = Client(TWILIO_SID, TWILIO_AUTH_TOKEN)
    try:
        employer_id = random.randint(1000, 9999)
        # Include the missing items in the WhatsApp message body
        message_body = f"Alert: Missing safety gear detected for Employer ID {employer_id}. Missing items: {missing_items_str}"
        
        client.messages.create(
            from_=TWILIO_WHATSAPP_NUMBER,
            body=message_body,
            to=RECIPIENT_WHATSAPP_NUMBER
        )
        st.sidebar.success(f"📲 WhatsApp Alert Sent Successfully! Employer ID: {employer_id}")
    except Exception as e:
        st.sidebar.error(f"❌ WhatsApp Alert Error: {e}")


# Detection function
def detect_safety_gear(image, is_video=False):
    results = model(image, conf=confidence_threshold)
    missing_items = []
    detection_flag = False

    for result in results:
        boxes = result.boxes
        if len(boxes.xyxy) > 0:
            detected_classes = [result.names[int(cls)] for cls in boxes.cls.cpu().numpy()]
            missing_items = [item for item in safety_gear_classes if item not in detected_classes]
            for box, label in zip(boxes.xyxy, detected_classes):
                x1, y1, x2, y2 = map(int, box)
                color = colors['unsafe'] if label.startswith('NO') else colors['safe']
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if label.startswith('NO'):
                    detection_flag = True

    if missing_items and alerts_enabled:
        if enable_email:
            send_email_alert(missing_items)
        if enable_whatsapp:
            send_whatsapp_alert(missing_items)

    if not is_video:
        return image, detection_flag, missing_items
    return detection_flag, missing_items


# Image detection
upload_choice = st.radio("Select Input Type:", ("Image Detection", "Video Detection", "Live Camera"))

if upload_choice == "Image Detection":
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        img_array = np.array(image)
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        with st.spinner("Processing Image..."):
            annotated_image, detection_flag, missing_items = detect_safety_gear(img)
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            st.image(annotated_image, caption="Processed Image", use_column_width=True)

            if detection_flag:
                st.error("⚠ Missing Safety Gear Detected!")
                st.write(f"Missing Items: {', '.join(missing_items)}")
            else:
                st.success("✅ All Safety Gear Present")


elif upload_choice == "Video Detection":
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        with st.spinner("Processing Video..."):
            video_placeholder = st.empty()
            detection_flag = False
            all_missing_items = set()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                detection_flag_frame, missing_items_frame = detect_safety_gear(frame, is_video=True)
                detection_flag |= detection_flag_frame
                all_missing_items.update(missing_items_frame)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            cap.release()

            if detection_flag:
                st.error("⚠ Missing Safety Gear Detected!")
                st.write(f"Missing Items: {', '.join(all_missing_items)}")
            else:
                st.success("✅ All Safety Gear Present")


elif upload_choice == "Live Camera":
    # Initialize the session state for live camera control
    if "live_camera_active" not in st.session_state:
        st.session_state["live_camera_active"] = False

    # Start button for live detection
    if st.button("Start Live Camera Detection"):
        st.session_state["live_camera_active"] = True
        cap = cv2.VideoCapture(0)

        with st.spinner("Starting Live Camera..."):
            while cap.isOpened() and st.session_state["live_camera_active"]:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to open camera.")
                    break

                detection_flag = detect_safety_gear(frame, is_video=True)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Live Camera Frame", use_column_width=True)

                if detection_flag:
                    st.warning("⚠ Missing Safety Gear Detected!")
                else:
                    st.success("✅ All Safety Gear Present")

            cap.release()
            st.session_state["live_camera_active"] = False  # Reset state when the camera is stopped

    # Stop button for live detection
    if st.button("Stop Live Camera Detection"):
        st.session_state["live_camera_active"] = False
        st.info("Live Camera Detection Stopped.")