import streamlit as st
import torch
import json
import geocoder
from geopy.distance import geodesic
import difflib

# Load YOLO model for object detection
model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# Load university locations from GeoJSON
with open("university_map.geojson", "r", encoding="utf-8") as file:
    university_map = json.load(file)

# Extract locations into a dictionary
locations = {feature["properties"].get("name", "").strip().lower(): tuple(reversed(feature["geometry"]["coordinates"])) for feature in university_map["features"]}

# Function to get real-time GPS location
def get_current_location():
    try:
        g = geocoder.ip('me')  # Get location from IP
        if g.latlng:
            return tuple(g.latlng)  # Returns (latitude, longitude)
    except Exception as e:
        st.error(f"⚠️ GPS Error: {e}")
    return None  # GPS failed

# Navigation function
def blind_navigation(destination):
    if destination not in locations:
        st.error("Invalid destination. Please try again.")
        return

    target_location = locations[destination]
    st.success(f"✅ Navigating to {destination} → 📍 {target_location}")

    current_location = get_current_location()
    if current_location:
        distance = geodesic(current_location, target_location).meters
        st.info(f"📍 Distance to destination: {distance:.2f} meters")

    st.write("Real-time navigation is not supported on Streamlit Cloud, but this would work on a local setup.")

# Streamlit UI
st.title("🚀 Blind Navigation System")
st.write("This system helps visually impaired individuals navigate using AI and GPS.")

# User input for destination
destination = st.text_input("Enter your destination (e.g., 'Library', 'NC Hostel'):")
if st.button("Start Navigation"):
    if destination:
        matched_location = difflib.get_close_matches(destination.lower(), locations.keys(), n=1, cutoff=0.5)
        if matched_location:
            blind_navigation(matched_location[0])
        else:
            st.error("Destination not recognized. Try again.")
