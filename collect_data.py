import os
import time
import cv2

def Collect_img():
    # Create directories to store data
    classes = ['Correct', 'Incorrect']
    for class_name in classes:
        os.makedirs(os.path.join("dataset", class_name), exist_ok=True)

    # Open camera or video feed
    cap = cv2.VideoCapture('C:/Users/Nima/PycharmProjects/pythonProject3/video/تمرين رفع الساق والاستلقاء على الأرض Lying floor leg raise.mp4')
    # Change to video file path if using a video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow("Collect_img", frame)
        key = cv2.waitKey(20)

        # Press '1' to collect Correct data, '0' to collect Incorrect data
        if key == ord('1'):
            cv2.imwrite(os.path.join('dataset', 'Correct', f'correct_{time.time()}.jpg'), frame)
        elif key == ord('0'):
            cv2.imwrite(os.path.join('dataset', 'Incorrect', f'incorrect_{time.time()}.jpg'), frame)
        elif key == 27:  # Press 'Esc' to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Collect_img()
