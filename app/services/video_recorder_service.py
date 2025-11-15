# app/video_recorder.py
import cv2 # type: ignore
import threading
from app.core.config import settings

class VideoRecorder:
    def __init__(self, output_file="output.avi"):
        self.stream_url = settings.CAMERA_SERVER_URL
        self.output_file = output_file
        self.recording = False
        self.thread = None

    def start(self):
        if self.recording:
            print("Already recording.")
            return
        self.recording = True
        self.thread = threading.Thread(target=self._record)
        self.thread.start()

    def _record(self):
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            print("Could not open stream.")
            self.recording = False
            return

        # Pobierz rozmiar obrazu
        fps = 20.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_file, fourcc, fps, (width, height))

        print(f"Recording started -> {self.output_file}")
        while self.recording and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)

        cap.release()
        out.release()
        print("Recording stopped.")

    def stop(self):
        self.recording = False
        if self.thread:
            self.thread.join()
