import cv2
import os
import time


def open_stream(rtsp_url):
    # Force FFmpeg to use UDP transport for RTSP
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
    cap = cv2.VideoCapture(rtsp_url)
    time.sleep(2)  # Give the stream time to initialize
    return cap


def main():
    rtsp_url = "rtsp://user1:1234abcd@115.79.213.124:10554/streaming/channels/501"
    cap = open_stream(rtsp_url)

    if not cap.isOpened():
        print("Error: Unable to open the video stream initially.")

    print("Press 'q' to exit the stream.")

    while True:
        ret, frame = cap.read()

        # If no frame is received, attempt to reconnect
        if not ret:
            print("No frame received. Attempting to reconnect...")
            cap.release()
            time.sleep(1)
            cap = open_stream(rtsp_url)
            continue

        cv2.imshow("RTSP Stream", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exiting the stream...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()