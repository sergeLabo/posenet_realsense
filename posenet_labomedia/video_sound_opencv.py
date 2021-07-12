import cv2
import numpy as np

from ffpyplayer.player import MediaPlayer

video_path="/media/data/3D/dwhelper/Marcel Duchamp - L'art du possible ARTE.mp4"

def PlayVideo(video_path):
    video=cv2.VideoCapture(video_path)
    player = MediaPlayer(video_path)
    while True:
        grabbed, frame = video.read()
        audio_frame, val = player.get_frame()
        print(audio_frame, val)
        if not grabbed:
            print("End of video")
            break
        if cv2.waitKey(28) & 0xFF == ord("q"):
            break
        cv2.imshow("Video", frame)
        if val != 'eof' and audio_frame is not None:
            #audio
            img, t = audio_frame
    video.release()
    cv2.destroyAllWindows()

PlayVideo(video_path)
