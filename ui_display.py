import cv2


class UIDisplay:
    def __init__(self):
        self.window_name = "Hand Gesture Recognition"

    def show(self, frame, results, gestures):
        for result in results:
            frame = result.plot()

        for gesture in gestures:
            if gesture == "choose":
                cv2.putText(frame, "Choose Detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(self.window_name, frame)

    def check_exit(self):
        return cv2.waitKey(1) & 0xFF == ord("q")

    def close(self):
        cv2.destroyAllWindows()
