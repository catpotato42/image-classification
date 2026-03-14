import cv2
import time
import os

# --- CONFIG ---
OUTPUT_FOLDER = "my_dataset/thinking_class"
TARGET_FPS = 12
# --------------

#delay based on fps
INTERVAL_SECONDS = 1.0 / TARGET_FPS

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print(f"Background capture started at {TARGET_FPS} FPS.")
print(f"Saving images to: {OUTPUT_FOLDER}")
print("[q] = quit, [p] = pause")

captured_count = 0
is_paused = True

try:
    while True:
        loop_start = time.time()
        
        #capture frame
        ret, frame = cap.read()
        
        if not ret:
            print("dropped frame/camera disc")
            time.sleep(1)
            continue
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'): #quit
            break
        elif key == ord('p'): #toggle pause
            time.sleep(1)
            is_paused = not is_paused
            print("Paused" if is_paused else "Resumed")

        status_text = f"COUNT: {captured_count} | {'PAUSED' if is_paused else 'RECORDING'}"
        status_color = (0, 0, 255) if is_paused else (0, 255, 0) #Red if paused, Green if recording        

        if not is_paused:
            timestamp = time.strftime("%m-%d_%H-%M-%S")
            raw_path = os.path.join(OUTPUT_FOLDER, f"face_{timestamp}_{captured_count:05d}.jpg")
            filename = os.path.abspath(raw_path)
            if not cv2.imwrite(filename, frame):
                print(f"save {captured_count} failed")
                time.sleep(0.5)
                continue 
            captured_count += 1

        cv2.putText(frame, status_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        cv2.imshow("Live Capture Feed", frame)
        
        time_taken = time.time() - loop_start
        sleep_time = INTERVAL_SECONDS - time_taken
        
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    #Ctrl+C, Ctrl+\
    print("Capture interrupted\nw")

finally:
    cap.release()
    print(f"Successfully saved {captured_count} images.")