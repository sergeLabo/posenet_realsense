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

import argparse
import collections
import time

import cv2
import pyrealsense2 as rs
import numpy as np

from pose_engine import PoseEngine

EDGES = (
    ('nose', 'left eye'),
    ('nose', 'right eye'),
    ('nose', 'left ear'),
    ('nose', 'right ear'),
    ('left ear', 'left eye'),
    ('right ear', 'right eye'),
    ('left eye', 'right eye'),
    ('left shoulder', 'right shoulder'),
    ('left shoulder', 'left elbow'),
    ('left shoulder', 'left hip'),
    ('right shoulder', 'right elbow'),
    ('right shoulder', 'right hip'),
    ('left elbow', 'left wrist'),
    ('right elbow', 'right wrist'),
    ('left hip', 'right hip'),
    ('left hip', 'left knee'),
    ('right hip', 'right knee'),
    ('left knee', 'left ankle'),
    ('right knee', 'right ankle'),
)


def shadow_text(img, x, y, text, font_size=16):
    cv2.putText(img, text,(x+1, y+1),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(img, text,(x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def draw_pose(img, pose, src_size, appsink_size, color=(0, 255, 255), threshold=0.2):
    scale_x = src_size[0] / appsink_size[0]
    scale_y = src_size[1] / appsink_size[1]
    xys = {}
    for label, keypoint in pose.keypoints.items():
        if keypoint.score < threshold: continue
        # Offset and scale to source coordinate space.
        kp_y = int(scale_y*keypoint.yx[0])
        kp_x = int(scale_x*keypoint.yx[1])
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
    parser = argparse.ArgumentParser(formatter_class=\
                                     argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--mirror',
                        help='flip video horizontally',
                        action='store_true')
    parser.add_argument('--model',
                        help='.tflite model path.',
                        required=False)
    parser.add_argument('--res',
                        help='Resolution',
                        default='640x480',
                        choices=['480x360', '640x480', '1280x720'])
    parser.add_argument('--videosrc',
                        help='Which video source to use',
                        default='/dev/video0')
    parser.add_argument('--h264',
                        help='Use video/x-h264 input',
                        action='store_true')
    parser.add_argument('--jpeg',
                        help='Use image/jpeg input',
                        action='store_true')

    args = parser.parse_args()

    args.res = '640x480'  # '1280x720'  #
    default_model = 'models/posenet_mobilenet_v1_075_%d_%d_quant_decoder_edgetpu.tflite'

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

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, src_size[0], src_size[1], rs.format.z16, 30)
    config.enable_stream(rs.stream.color, src_size[0], src_size[1], rs.format.rgb8, 30)
    pipeline.start(config)

    c = rs.colorizer()
    align_to = rs.stream.color
    align = rs.align(align_to)
    while True:
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        if not depth: continue

        color_data = color.as_frame().get_data()
        color_image_rgb = np.asanyarray(color_data)
        color_image = cv2.cvtColor(color_image_rgb, cv2.COLOR_RGB2BGR)

        depth_colormap = c.colorize(depth)
        depth_data = depth_colormap.as_frame().get_data()
        depth_image = np.asanyarray(depth_data)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR)

        if (color_image_rgb.shape[0] != appsink_size[1] or
            color_image_rgb.shape[1] != appsink_size[0]) :
            color_image_rgb = cv2.resize(color_image_rgb,
                                  dsize=appsink_size,
                                  interpolation=cv2.INTER_NEAREST)

        start_time = time.monotonic()
        outputs, inference_time = engine.DetectPosesInImage(color_image_rgb)
        end_time = time.monotonic()
        n += 1
        sum_process_time += 1000 * (end_time - start_time)
        sum_inference_time += inference_time

        avg_inference_time = sum_inference_time / n
        text_line = 'PoseNet: %.1fms (%.2f fps) TrueFPS: %.2f Nposes %d' % (
            avg_inference_time, 1000 / avg_inference_time, next(fps_counter), len(outputs)
        )

        if args.mirror:
            color_image = cv2.flip(color_image, 1)
            depth_image = cv2.flip(depth_image, 1)
        shadow_text(color_image, 10, 20, text_line)
        shadow_text(depth_image, 10, 20, text_line)
        for pose in outputs:
            draw_pose(color_image, pose, src_size, appsink_size)
            draw_pose(depth_image, pose, src_size, appsink_size)

        cv2.imshow('color', color_image)
        cv2.imshow('depth', depth_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
