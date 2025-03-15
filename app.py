import streamlit as st
import cv2
import torch
import pyttsx3
import speech_recognition as sr
from geopy.distance import geodesic
import json
import time
import difflib
import math
import numpy as np
import geocoder
from PIL import Image

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.setProperty("volume", 1.0)
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)

# Load YOLO model for object detection
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load university locations from GeoJSON
with open("university_map.geojson", "r", encoding="utf-8") as file:
    university_map = json.load(file)

# Extract locations into a dictionary
locations = {
    feature["properties"]["name"].strip().lower(): tuple(reversed(feature["geometry"]["coordinates"]))
    for feature in university_map["features"]
}

# Function to get real-time GPS location
def get_current_location():
    try:
        g = geocoder.ip('me')  # Get location from IP
        return tuple(g.latlng) if g.latlng else None  # Returns (latitude, longitude)
    except Exception as e:
        st.warning(f"⚠️ GPS Error: {e}")
        return None  # GPS failed

# Function to calculate bearing angle
def calculate_bearing(start, end):
    lat1, lon1 = math.radians(start[0]), math.radians(start[1])
    lat2, lon2 = math.radians(end[0]), math.radians(end[1])
    
    delta_lon = lon2 - lon1
    x = math.sin(delta_lon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    bearing = math.degrees(math.atan2(x, y))

    return (bearing + 360) % 360  # Normalize

# Function to determine movement direction
def get_direction(previous_location, current_location, destination):
    if not previous_location or not current_location:
        return "Calculating route..."

    bearing_to_dest = calculate_bearing(current_location, destination)
    movement_bearing = calculate_bearing(previous_location, current_location)
    bearing_difference = (bearing_to_dest - movement_bearing + 360) % 360

    if 0 <= bearing_difference < 30 or bearing_difference > 330:
        return "Move Forward"
    elif 30 <= bearing_difference < 150:
        return "Turn Right"
    elif 150 <= bearing_difference < 210:
        return "Move Backward"
    elif 210 <= bearing_difference < 330:
        return "Turn Left"
    else:
        return "Recalculating..."

# Predefined real-world widths of objects (in cm)
KNOWN_WIDTHS = {
    "person": 40,
    "chair": 50,
    "car": 180,
    "bottle": 7,
    "cup": 8
}

FOCAL_LENGTH = 615  # Adjust based on testing

# Function to estimate distance based on object width
def estimate_distance(object_width_pixels, real_width_cm):
    return (FOCAL_LENGTH * real_width_cm) / object_width_pixels if object_width_pixels > 0 else -1  

# Function to detect objects
def detect_objects(frame):
    if frame is None:
        return [], frame

    results = model(frame)
    detected_objects = []

    for result in results.xyxy[0]:  
        x1, y1, x2, y2, confidence, class_id = result.tolist()
        label = model.names[int(class_id)]
        object_width_pixels = x2 - x1
        real_width_cm = KNOWN_WIDTHS.get(label, 30)
        distance = estimate_distance(object_width_pixels, real_width_cm)

        if confidence > 0.5:
            detected_objects.append(f"{label} {int(distance)} cm ahead")
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2, int(y2))), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} {int(distance)} cm", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return detected_objects, frame

# Streamlit UI
st.title("👁‍🗨 Blind Navigation System")
st.write("Helping visually impaired individuals navigate using AI and GPS.")

# Select Destination
destination = st.selectbox("📍 Select Destination:", list(locations.keys()))

if st.button("Start Navigation 🚀"):
    if destination:
        target_location = locations[destination]
        st.success(f"Navigating to **{destination}**...")

        last_location = get_current_location()
        
        if last_location:
            distance = geodesic(last_location, target_location).meters
            st.write(f"📏 **Distance:** {distance:.2f} meters")

            # Update Movement Directions
            direction = get_direction(last_location, get_current_location(), target_location)
            st.write(f"🛤 **Direction:** {direction}")

            # Capture Camera
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()

            if ret:
                detected_objects, processed_frame = detect_objects(frame)
                
                # Convert OpenCV image to PIL for Streamlit
                processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(processed_frame)
                st.image(image, caption="Object Detection", use_column_width=True)

                if detected_objects:
                    st.warning(f"⚠️ Caution: {', '.join(detected_objects)}")
                    engine.say(f"Caution! {', '.join(detected_objects)}")
                    engine.runAndWait()

            cap.release()
    
        else:
            st.error("⚠️ Could not retrieve real-time GPS location.")

