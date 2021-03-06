# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Modification by La Labomedia July 2021


import argparse
import collections
import time

import cv2

from pose_engine import PoseEngine, KeypointType

EDGES = (
    (KeypointType.NOSE, KeypointType.LEFT_EYE),
    (KeypointType.NOSE, KeypointType.RIGHT_EYE),
    (KeypointType.NOSE, KeypointType.LEFT_EAR),
    (KeypointType.NOSE, KeypointType.RIGHT_EAR),
    (KeypointType.LEFT_EAR, KeypointType.LEFT_EYE),
    (KeypointType.RIGHT_EAR, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_EYE, KeypointType.RIGHT_EYE),
    (KeypointType.LEFT_SHOULDER, KeypointType.RIGHT_SHOULDER),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_ELBOW),
    (KeypointType.LEFT_SHOULDER, KeypointType.LEFT_HIP),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_ELBOW),
    (KeypointType.RIGHT_SHOULDER, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_ELBOW, KeypointType.LEFT_WRIST),
    (KeypointType.RIGHT_ELBOW, KeypointType.RIGHT_WRIST),
    (KeypointType.LEFT_HIP, KeypointType.RIGHT_HIP),
    (KeypointType.LEFT_HIP, KeypointType.LEFT_KNEE),
    (KeypointType.RIGHT_HIP, KeypointType.RIGHT_KNEE),
    (KeypointType.LEFT_KNEE, KeypointType.LEFT_ANKLE),
    (KeypointType.RIGHT_KNEE, KeypointType.RIGHT_ANKLE),
)


def shadow_text(img, x, y, text, font_size=16):
    cv2.putText(img, text,(x+1, y+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, text,(x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_pose(img, pose, src_size, appsink_size, color=(0, 255, 255), threshold=0.2):
    """
    pose = Pose(keypoints={
    <KeypointType.NOSE: 0>: Keypoint(point=Point(x=357.3, y=218.4), score=0.99),
    src_size = (640, 480)
    appsink_size = (640, 480)
    """

    # Calcul des scales: si src_size = '480x360' --> appsink_size = (480, 360)
    box_x, box_y = src_size[0], src_size[0]  # 641, 480
    scale_x = src_size[0] / appsink_size[0]
    scale_y = src_size[1] / appsink_size[1]

    xys = {}
    for label, keypoint in pose.keypoints.items():
        # keypoint = Keypoint(point=Point(x=445.2, y=403.1), score=0.309)
        if keypoint.score > threshold:
            # Offset and scale to source coordinate space.
            # Suppression de Offset, pourquoi Offset ? pour 480x360 ?
            kp_x = int((keypoint.point[0]) * scale_x)
            kp_y = int((keypoint.point[1]) * scale_y)

            xys[label] = (kp_x, kp_y)
            cv2.circle(img, (kp_x, kp_y), 5, color=(209, 156, 0), thickness=-1) #cyan
            cv2.circle(img, (kp_x, kp_y), 6, color=color, thickness=1)

    for a, b in EDGES:
        if a not in xys or b not in xys: continue
        ax, ay = xys[a]
        bx, by = xys[b]
        cv2.line(img, (ax, ay), (bx, by), color, 2)

def avg_fps_counter(window_size):
    window = collections.deque(maxlen=window_size)
    prev = time.monotonic()
    yield 0.0  # First fps value.

    while True:
        curr = time.monotonic()
        window.append(curr - prev)
        prev = curr
        yield len(window) / sum(window)

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mirror', help='flip video horizontally', action='store_true')
    parser.add_argument('--model', help='.tflite model path.', required=False)
    parser.add_argument('--res', help='Resolution', default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc', help='Which video source to use', default='/dev/video0')
    parser.add_argument('--h264', help='Use video/x-h264 input', action='store_true')
    parser.add_argument('--jpeg', help='Use image/jpeg input', action='store_true')
    args = parser.parse_args()

    args.res = '640x480'  #
    default_model = 'models/mobilenet/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'

    if args.res == '480x360':
        src_size = (640, 480)
        appsink_size = (480, 360)
        model = args.model or default_model % (353, 481)
    elif args.res == '640x480':
        src_size = (640, 480)
        appsink_size = (640, 480)
        model = args.model or default_model % (481, 641)
    elif args.res == '1280x720':
        src_size = (1280, 720)
        appsink_size = (1280, 720)
        model = args.model or default_model % (721, 1281)

    print('Loading model: ', model)
    engine = PoseEngine(model, mirror=args.mirror)
    input_shape = engine.get_input_tensor_shape()
    inference_size = (input_shape[2], input_shape[1])

    n = 0
    sum_process_time = 0
    sum_inference_time = 0
    fps_counter  = avg_fps_counter(30)

    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        img = cv2.resize(frame,
                         dsize=appsink_size,
                         interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        start_time = time.monotonic()
        outputs, inference_time = engine.DetectPosesInImage(img)
        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time

        avg_inference_time = sum_inference_time / n
        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f Nposes %d' % (
            avg_inference_time, 1000 / avg_inference_time, next(fps_counter), len(outputs)
        )

        if args.mirror:
            frame = cv2.flip(frame, 1)
        shadow_text(frame, 10, 20, text_line)
        for pose in outputs:
            draw_pose(frame, pose, src_size, appsink_size)

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
