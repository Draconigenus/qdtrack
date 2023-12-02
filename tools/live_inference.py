import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import cv2
import mmcv
from select import select
import sys
from pynput import keyboard
import time

from qdtrack.apis import inference_model, init_model

slowdown = False
cycle = False
cycle_start = 0

def on_press(key):
    global slowdown
    global cycle
    global cycle_start
    if (key == keyboard.Key.space):
        cycle = not cycle
        cycle_start = time.time()
    else:
        slowdown = not slowdown
        cycle = False

def main():
    global slowdown
    global cycle_start
    parser = ArgumentParser()
    parser.add_argument('config', help='config file')
    # parser.add_argument('--input', help='input video file or folder')
    # parser.add_argument(
    #     '--output', help='output video file (mp4 format) or folder')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.0,
        help='The threshold of score to filter bboxes.')
    parser.add_argument(
        '--device', default='cuda:0', help='device used for inference')
    parser.add_argument(
        '--backend',
        choices=['cv2', 'plt'],
        default='cv2',
        help='the backend to visualize the results')
    parser.add_argument('--fps', help='FPS of the output video')
    parser.add_argument("--slowdown", type=float, default=2.0)
    args = parser.parse_args()

    # if args.show or OUT_VIDEO:
    #     if fps is None and IN_VIDEO:
    #         fps = imgs.fps
    #     if not fps:
    #         raise ValueError('Please set the FPS for the output video.')
    #     fps = int(fps)

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    cam = cv2.VideoCapture(-1)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_BACKLIGHT, 0)
    cam.set(cv2.CAP_PROP_ZOOM, 0)
    # test and show/save the images
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    cycle_start = time.time()
    i = 0
    while True:
        result, img = cam.read()
        if isinstance(img, str):
            img = osp.join(args.input, img)
        start_time = time.time()
        result = inference_model(model, img, frame_id=i)
        end_time = time.time()
        frame_time = end_time - start_time
        fps = 1.0 / frame_time
        if slowdown:
            fps = fps / args.slowdown
        additional_delay = ((1.0 / fps) - frame_time)
        img = model.show_result(
            img,
            result,
            score_thr=args.score_thr,
            show=False,
            wait_time=1,
            out_file=None,
            backend=args.backend)
        # print(f"Displaying Frame {i}")
        cv2.putText(img, f"{'CPU' if slowdown else 'FPGA Accelerator'} FPS: {int(fps)}", (0,25), cv2.FONT_HERSHEY_COMPLEX, 1.0, color=(0,255,0))
        mmcv.imshow(img, wait_time=1)
        if slowdown:
            time.sleep(additional_delay)

        if (cycle and (time.time() - cycle_start) > 5):
            slowdown = not slowdown
            cycle_start = time.time()
        i += 1

    # if args.output and OUT_VIDEO:
    #     print(f'making the output video at {args.output} with a FPS of {fps}')
    #     mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
    #     out_dir.cleanup()


if __name__ == '__main__':
    main()
