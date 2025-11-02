import cv2
import mediapipe as mp
import numpy as np

# กำหนดตัวแปรสำหรับวาดเส้นและจุด
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 1. ย้ายฟังก์ชันมาไว้ด้านบน
# ----------------------------------------------------
def calculate_angle(a,b,c):
    """
    คำนวณองศาระหว่าง 3 จุด (เช่น ไหล่, ศอก, ข้อมือ)
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle 
# ----------------------------------------------------


cap = cv2.VideoCapture(0)

# Curl counter variables
counter = 0 
# 2. แก้ไข: กำหนดค่าเริ่มต้นเป็น "down" เพื่อป้องกัน Error
stage = "down" 

## Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # 3. เพิ่ม: ตรวจสอบว่ากล้องอ่านเฟรมได้หรือไม่
        if not ret:
            print("Ignoring empty camera frame.")
            break

        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # Calculate angle
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # Visualize angle
            cv2.putText(image, str(int(angle)), # แสดงเป็นเลขจำนวนเต็ม
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Curl counter logic
            if angle > 160: # เมื่องอแขนจนสุด (ค่าองศามาก)
                stage = "down"
            if angle < 30 and stage =='down': # เมื่อยกแขนขึ้น (ค่าองศาน้อย) และก่อนหน้านี้อยู่ท่า "down"
                stage="up"
                counter +=1
                print(counter)
            
            # 4. ย้ายเข้ามา: ต้องวาดเส้นหลังจากที่แน่ใจแล้วว่าเจอ 'landmarks'
            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                    ) 
                        
        except:
            # หากตรวจไม่พบร่างกาย (landmarks) ก็ให้ข้ามไป (pass)
            pass
        
        # Render curl counter
        # Setup status box
        cv2.rectangle(image, (0,0), (300,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (150,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, stage, 
                    (145,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()