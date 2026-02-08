# 🎓 AI-Based Student Attendance & Duration Tracker  
### Using YOLOv9 + OpenCV for Online Classes

## 📌 Project Overview
This project is an **AI-powered student attendance tracking system** designed for **online classes**.  
It uses **YOLOv9** for real-time person detection and **OpenCV** to track students and calculate how long each student stays present during a live session.

The system automatically:
- Detects students from live video / recorded sessions
- Tracks their presence continuously
- Calculates **attendance duration**
- Marks students as **Present / Absent** based on time thresholds

This removes the need for manual attendance and provides **accurate, time-based participation analysis**.

---

## 🚀 Features
- ✅ Real-time **student detection** using YOLOv9  
- 🎯 Accurate **person tracking** with unique IDs  
- ⏱️ **Attendance duration calculation** for each student  
- 📊 Automatic **Present / Absent classification**  
- 🎥 Works with **online class recordings or live webcam feed**  
- 🧠 Reduces proxy attendance and human error  

---

## 🛠️ Tech Stack
- **YOLOv9** – Object detection (Person class)
- **OpenCV** – Video processing & tracking
- **Python** – Core implementation
- **NumPy** – Numerical operations
- **CV2 Tracker / Custom Tracking Logic**

---

## ⚙️ How It Works
1. Video stream is captured from:
   - Online class recording **OR**
   - Live webcam feed
2. YOLOv9 detects all **persons (students)** in each frame
3. Each student is assigned a **unique tracking ID**
4. Presence time is recorded frame-by-frame
5. Total duration is calculated for every student
6. Students are marked:
   - **Present** → If duration ≥ threshold
   - **Absent** → If duration < threshold
