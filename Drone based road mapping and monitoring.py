import cv2
from ultralytics import YOLO
import tkinter as tk
from PIL import Image, ImageTk

def process_frame():
    global paused
    if paused:
        root.after(10, process_frame)
        return
    ret, frame = cap.read()
    if ret:
        results = tuned_model.predict(frame)

        processed_frame = results[0].plot(line_width=1)

        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        processed_frame_img = Image.fromarray(processed_frame_rgb)
        processed_frame_tk = ImageTk.PhotoImage(processed_frame_img)

        label.config(image=processed_frame_tk)
        label.image = processed_frame_tk

        out.write(processed_frame)

        root.after(10, process_frame)
    else:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

def toggle_pause():
    global paused
    paused = not paused

tuned_model = YOLO('best.pt')

cap = cv2.VideoCapture('Sample7.mp4')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('Processed_sample.avi', fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

root = tk.Tk()
root.title("Real-time Video Analysis")

label = tk.Label(root)
label.pack()

paused = False
pause_button = tk.Button(root, text="Pause", command=toggle_pause)
pause_button.pack()

process_frame()

root.mainloop()
