import cv2
from ultralytics import YOLO

# 1. YOLOv8 ka ek chota aur fast model load kar rahe hain
model = YOLO("yolov8n.pt")

# 2. Laptop ka webcam shuru karne ke liye (0 matlab default webcam)
cap = cv2.VideoCapture(0)

print("Webcam shuru ho raha hai... Band karne ke liye keyboard par 'q' dabayein.")

while cap.isOpened():
    success, frame = cap.read()
    
    if not success:
        print("Webcam se frame nahi mil raha hai.")
        break

    # 3. AI Model se objects ko detect karwa rahe hain
    # persist=True se object track hota rehta hai (ki wo kahan ja raha hai)
    results = model.track(frame, persist=True, verbose=False)

    # 4. Jo bhi detect hua, use screen par draw (boxes) kar rahe hain
    annotated_frame = results[0].plot()

    # 5. Live video window screen par dikhana
    cv2.imshow("YOLOv8 Real-Time Object Detection & Tracking", annotated_frame)

    # Agar keyboard par 'q' dabayenge, toh video band ho jayegi
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sab kuch sahi se band karna
cap.release()
cv2.destroyAllWindows()
print("Webcam successfully band ho gaya hai.")