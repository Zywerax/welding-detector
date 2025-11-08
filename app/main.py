import cv2

def test_camera(camera_index=1):
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Nie można otworzyć kamery {camera_index}")
        return

    print("Kamera uruchomiona. Wciśnij 's' aby zapisać klatkę lub 'q' aby zakończyć.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Nie można odczytać klatki.")
            break

        cv2.imshow("Podgląd kamery", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite("frame.jpg", frame)
            print("Zapisano klatkę jako frame.jpg")
        elif key == ord('q'):
            print("Zakończono podgląd.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_camera()