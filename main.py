import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class ExerciseDetectorGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Exercise Detector")
        self.root.geometry("400x200")

        self.current_page = 1

        # Page 1 - Welcome Page
        self.label_welcome = tk.Label(self.root, text="Hello,\nAre you ready to determine \nIf you are doing the leg lift exercise \ncorrectly or not?")
        self.label_welcome.pack(pady=20)
        self.button_yes = tk.Button(self.root, text="Yes", command=self.go_to_page_2, bg="green", fg="white")
        self.button_yes.pack(pady=(0, 10))
        self.button_no = tk.Button(self.root, text="No", command=self.root.destroy, bg="red", fg="white")
        self.button_no.pack(pady=(0, 10))

        # Page 2 - Enter Weight Page
        self.label_weight = tk.Label(self.root, text="Enter your weight:")
        self.entry_weight = tk.Entry(self.root)
        self.label_height = tk.Label(self.root, text="Enter your height:")
        self.entry_height = tk.Entry(self.root)
        self.button_next = tk.Button(self.root, text="Next", command=self.calculate_bmi, bg="green", fg="white")

        # Page 3 - Result Page
        self.label_result = tk.Label(self.root, text="")
        self.button_go = tk.Button(self.root, text="Go", bg="green", fg="white", command=self.start_video)

        # Load trained model
        self.model = None

    def go_to_page_2(self):
        self.current_page = 2
        self.label_welcome.pack_forget()
        self.button_yes.pack_forget()
        self.button_no.pack_forget()
        self.show_page_2()

    def show_page_2(self):
        self.label_weight.pack()
        self.entry_weight.pack()
        self.label_height.pack()
        self.entry_height.pack()
        self.button_next.pack(pady=20)

    def calculate_bmi(self):
        try:
            weight = float(self.entry_weight.get())
            height = float(self.entry_height.get())
            height_last_two_digits = int(str(int(height))[-2:])  # Extract last two digits of height
            weight_last_two_digits = int(str(int(weight))[-2:])  # Extract last two digits of weight

            # Check if height falls within the specified range [120, 199]
            if 120 <= height <= 199:
                # Calculate the range for weight based on the last two digits of height
                weight_range_lower = height_last_two_digits - 5
                weight_range_upper = height_last_two_digits + 10

                # Check if weight falls within the specified range [height_last_two_digits - 5, height_last_two_digits + 10]
                if weight_range_lower <= weight <= weight_range_upper:
                    self.label_result.config(text="You are fit, let's start!")
                else:
                    self.label_result.config(text="You may need to focus on your health. Let's start exercising!")
            else:
                self.label_result.config(text="Please enter a height between 120 and 199.")

            self.current_page = 3
            self.show_page_3()
        except ValueError:
            messagebox.showerror("Error", "Please enter valid weight and height.")

    def show_page_3(self):
        self.label_weight.pack_forget()
        self.entry_weight.pack_forget()
        self.label_height.pack_forget()
        self.entry_height.pack_forget()
        self.button_next.pack_forget()
        self.label_result.pack()
        self.button_go.pack(pady=20)

    def start_video(self):
        self.root.destroy()
        if self.model is None:
            self.model = load_model("exercise_model.h5")
        self.run_video()

    def run_video(self):
        # Open camera or video feed
        cap = cv2.VideoCapture('C:/Users/Nima/PycharmProjects/pythonProject3/video/تمرين رفع الساق والاستلقاء على الأرض Lying floor leg raise.mp4')
        # Change to video file path if using a video

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Preprocess frame
            image = cv2.resize(frame, (100, 100))  # Resize to match input shape
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            image = image / 255.0  # Normalize the pixel values

            # Predict class
            class_prob = self.model.predict(image)
            class_index = np.argmax(class_prob)

            # Display result
            label = "Correct" if class_index == 0 else "Incorrect"
            color = (0, 255, 0) if class_index == 0 else (0, 0, 255)
            cv2.putText(frame, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.imshow("Gym_exercise", frame)

            if cv2.waitKey(1) == 27:  # Press 'Esc' to exit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    ExerciseDetectorGUI().root.mainloop()
