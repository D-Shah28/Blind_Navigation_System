import os
import streamlit as st

# Install missing system dependencies
os.system("apt-get update")
os.system("apt-get install -y libgl1-mesa-glx")

import cv2  # Now import OpenCV after installing dependencies
import torch
import speech_recognition as sr
import numpy as np
import requests
import json
import geopy.distance
from gtts import gTTS
import time
import threading

# Initialize Streamlit app
st.title("Blind Navigation System")
st.write("🎤 Speak your destination...")

# Load YOLO model (Upgraded to YOLOv5m for better accuracy)
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5m")

model = load_model()

# Load university locations from GeoJSON
with open("university_map.geojson", "r", encoding="utf-8") as file:
    university_map = json.load(file)

# Function to get coordinates of a destination
def get_coordinates(destination):
    for feature in university_map["features"]:
        if feature["properties"]["name"].lower() == destination.lower():
            return feature["geometry"]["coordinates"]
    return None

# Function to get current GPS location (Placeholder: Replace with real GPS data)
def get_current_location():
    return [28.6139, 77.2090]  # Example coordinates for testing

# Function to calculate distance
def calculate_distance(coord1, coord2):
    return geopy.distance.geodesic(coord1, coord2).m

# Text-to-Speech function
def speak(text):
    tts = gTTS(text=text, lang="en")
    tts.save("output.mp3")
    os.system("mpg321 output.mp3")  # Works on local, may not work in Streamlit Cloud

# Function to recognize voice input using Google Web Speech API
def recognize_voice():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        st.write("🎤 Listening for destination...")
        try:
            audio = recognizer.listen(source)
            destination = recognizer.recognize_google(audio)
            st.write(f"✅ Recognized: {destination}")
            return destination.lower()
        except sr.UnknownValueError:
            st.write("❌ Could not understand audio.")
            return None
        except sr.RequestError:
            st.write("⚠️ Speech recognition service unavailable.")
            return None

# Function to provide navigation guidance
def start_navigation(destination):
    target_coords = get_coordinates(destination)
    if target_coords is None:
        st.write("⚠️ Destination not found in university map.")
        return

    current_coords = get_current_location()
    distance = calculate_distance(current_coords, target_coords)

    st.write(f"📍 Navigating to {destination} ({distance:.2f} meters away)")
    speak(f"Navigating to {destination}")

    while distance > 5:  # Stop when close to destination
        current_coords = get_current_location()  # Update real-time location
        distance = calculate_distance(current_coords, target_coords)

        if distance > 50:
            st.write(f"➡️ Move Forward | {distance:.2f}m left")
            speak(f"Move forward, {distance:.2f} meters left")
        elif distance <= 50 and distance > 5:
            st.write("🛑 Approaching destination...")
            speak("You are close to your destination.")
        else:
            st.write("✅ You have arrived!")
            speak("You have arrived at your destination.")
            break

        time.sleep(5)  # Wait before updating

# Function to detect objects
def detect_objects():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for det in results.xyxy[0]:  # xyxy format
            x1, y1, x2, y2, conf, cls = det
            label = model.names[int(cls)]
            obj_distance = round((y2 - y1) / 10, 2)  # Approximate distance measure

            if obj_distance < 100:  # Notify only for nearby objects
                st.write(f"⚠️ Caution: {label} {obj_distance} cm ahead")
                speak(f"Caution, {label} {obj_distance} centimeters ahead")

        time.sleep(2)

    cap.release()
    cv2.destroyAllWindows()

# Main execution
destination = recognize_voice()
if destination:
    navigation_thread = threading.Thread(target=start_navigation, args=(destination,))
    object_detection_thread = threading.Thread(target=detect_objects)

    navigation_thread.start()
    object_detection_thread.start()

    navigation_thread.join()
    object_detection_thread.join()

st.write("🔄 Restart the app to enter a new destination.")
