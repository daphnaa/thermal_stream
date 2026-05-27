#!/usr/bin/env python3
# =======================================================
# LINUX:
# python3 record_infiray_thermal.py \
#   --device /dev/video0 \
#   --out-dir Documents/thermal_recordings \
#   --session-name field_test \
#   --width 256 \
#   --height 384 \
#   --fps 25 \
#   --save-packed-npy \
#   --display

# python3 record_infiray_thermal.py \
#   --device /dev/video0 \
#   --out-dir Documents/thermal_recordings \
#   --session-name field_test \
#   --width 256 \
#   --height 384 \
#   --fps 25 \
#   --save-packed-npy

# WINDOWS:
# python record_infiray_thermal.py --device 0 --display --save-packed-npy


# Ctrl+C  stop from terminal
# q       stop from OpenCV display window
# SPACE   pause/resume from OpenCV display window

# =======================================================
import argparse
import csv
import signal
import time
from pathlib import Path

import platform

import cv2
import numpy as np
import yaml


RUNNING = True
PAUSED = False


def handle_sigint(sig, frame):
    global RUNNING
    RUNNING = False


def make_stamp() -> str:
    return time.strftime("%Y_%m_%d___%H_%M_%S")


def build_output_dir(base_dir: Path, session_name: str, stamp: str) -> Path:
    safe_name = session_name.strip().replace(" ", "_")
    if safe_name:
        return base_dir / f"{safe_name}_{stamp}"
    return base_dir / stamp


def ensure_dirs(out_dir: Path, stamp: str, timestamp_subdirs: bool = True):
    suffix = f"_{stamp}" if timestamp_subdirs else ""

    dirs = {
        "top_y": out_dir / f"top_y{suffix}",
        "bottom_y": out_dir / f"bottom_y{suffix}",
        "preview": out_dir / f"preview{suffix}",
        "packed": out_dir / f"packed_npy{suffix}",
    }

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    return dirs


def decode_yuyv(frame: np.ndarray) -> np.ndarray:
    """
    Decode YUYV 4:2:2 frame and return the Y/luma channel.

    Expected packed layout per two pixels:
        Y0 U0 Y1 V0

    With OpenCV CAP_PROP_CONVERT_RGB=0, this is often represented as:
        H x W x 2 uint8

    The Y channel is the useful grayscale/thermal-like image for this camera.
    """

    if frame is None:
        raise ValueError("Empty frame")

    # Common OpenCV raw YUYV form: H x W x 2
    if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 2:
        return frame[:, :, 0]

    # Sometimes OpenCV gives a flat/raw buffer.
    if frame.dtype == np.uint8 and frame.ndim == 2:
        h, packed_w = frame.shape

        # If packed_w is width * 2, reshape to H x W x 2.
        if packed_w % 2 == 0:
            w = packed_w // 2
            packed = frame.reshape(h, w, 2)
            return decode_yuyv(packed)

        return frame

    # Sometimes OpenCV silently converts to BGR despite settings.
    if frame.dtype == np.uint8 and frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    raise ValueError(f"Unsupported frame format: shape={frame.shape}, dtype={frame.dtype}")


def split_top_bottom(img: np.ndarray):
    h, _ = img.shape[:2]

    if h % 2 != 0:
        return None, None

    half = h // 2
    top = img[:half, :]
    bottom = img[half:, :]

    return top, bottom


def parse_device(device: str):
    """
    OpenCV on Linux can use '/dev/video0'.
    OpenCV on Windows usually needs integer camera index: 0, 1, 2...
    """
    if isinstance(device, str) and device.isdigit():
        return int(device)
    return device


def open_camera(device: str, width: int, height: int, fps: float):
    system = platform.system().lower()
    dev = parse_device(device)

    if system == "windows":
        # DirectShow is usually better than MSMF for weird USB cameras.
        cap = cv2.VideoCapture(dev, cv2.CAP_DSHOW)
    elif system == "linux":
        cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    else:
        cap = cv2.VideoCapture(dev)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera: {device}")

    # On Linux this may expose raw YUYV.
    # On Windows OpenCV may still convert internally depending on driver.
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)

    # Try YUYV. On Windows this may or may not be accepted.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    if width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)

    if height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    return cap


def write_config(out_dir: Path, dirs: dict, args, stamp: str, actual_width, actual_height, actual_fps):
    config = {
        "camera": "infiray_p2_pro_or_similar",
        "recording_stamp": stamp,
        "device": args.device,
        "v4l2_format": "YUYV",
        "requested_width": args.width,
        "requested_height": args.height,
        "requested_fps": args.fps,
        "actual_width": actual_width,
        "actual_height": actual_height,
        "actual_fps": actual_fps,
        "duration_sec": args.duration_sec,
        "max_frames": args.max_frames,
        "save_packed_npy": args.save_packed_npy,
        "output_dirs": {k: str(v) for k, v in dirs.items()},
        "controls": {
            "terminal_stop": "Ctrl+C",
            "display_stop": "q",
            "display_pause_resume": "space",
            "file_stop": "touch STOP inside the output session directory",
        },
        "notes": (
            "V4L2 exposes YUYV, not Y16. "
            "Use top_y or bottom_y as the main grayscale dataset after checking which half is best. "
            "packed_npy preserves the exact OpenCV frame for later re-decoding."
        ),
    }

    with open(out_dir / "capture_config.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def make_mp4_writer(mp4_path: Path, preview: np.ndarray, fps: float):
    h, w = preview.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, (w, h), isColor=True)

    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {mp4_path}")

    return writer


def main():
    global RUNNING, PAUSED
    parser = argparse.ArgumentParser(description="Record InfiRay/P2 Pro YUYV stream into timestamped folders.")

    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument(
        "--out-dir",
        default=str(Path.home() / "Documents/thermal_recordings"),
        help="Base output directory. A timestamped session folder is created inside it.",
    )
    parser.add_argument("--session-name", default="thermal_field")

    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=float, default=25.0)

    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--duration-sec", type=float, default=0.0)
    parser.add_argument("--save-preview-every", type=int, default=10)
    parser.add_argument("--save-packed-npy", action="store_true")
    parser.add_argument(
        "--flat-subdirs",
        action="store_true",
        help="Use top_y/bottom_y/preview/packed_npy instead of timestamped subdir names.",
    )

    args = parser.parse_args()

    signal.signal(signal.SIGINT, handle_sigint)

    stamp = make_stamp()
    base_dir = Path(args.out_dir).expanduser()
    out_dir = build_output_dir(base_dir, args.session_name, stamp)
    out_dir.mkdir(parents=True, exist_ok=True)

    dirs = ensure_dirs(out_dir, stamp, timestamp_subdirs=not args.flat_subdirs)

    cap = open_camera(args.device, args.width, args.height, args.fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))

    write_config(out_dir, dirs, args, stamp, actual_width, actual_height, actual_fps)

    timestamps_path = out_dir / f"timestamps_{stamp}.csv"
    mp4_writer = None
    mp4_path = out_dir / f"preview_{stamp}.mp4"
    stop_file = out_dir / "STOP"

    frame_id = 0
    dropped = 0
    start_wall = time.time()
    start_mono = time.monotonic()

    print(f"[INFO] Output session: {out_dir}")
    print(f"[INFO] Camera: {args.device}")
    print(f"[INFO] Actual size: {actual_width}x{actual_height}, fps={actual_fps}")
    print("[INFO] Controls:")
    print("       Ctrl+C  -> stop from terminal")
    print("       q       -> stop if display window is open")
    print("       SPACE   -> pause/resume if display window is open")
    print(f"       touch {stop_file} -> stop from another terminal")
    print()

    with open(timestamps_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "frame_id",
            "timestamp_ns",
            "timestamp_sec",
            "monotonic_sec",
            "top_y_file",
            "bottom_y_file",
            "packed_file",
            "y_min",
            "y_max",
            "y_mean",
            "frame_shape",
            "frame_dtype",
            "recording_stamp",
        ])

        while RUNNING:
            if stop_file.exists():
                print("[INFO] STOP file detected. Stopping.")
                break

            elapsed_mono = time.monotonic() - start_mono

            if args.duration_sec > 0 and elapsed_mono >= args.duration_sec:
                print("[INFO] Duration limit reached. Stopping.")
                break

            if args.max_frames > 0 and frame_id >= args.max_frames:
                print("[INFO] Max frame limit reached. Stopping.")
                break

            ok, frame = cap.read()
            timestamp_ns = time.time_ns()
            timestamp_sec = timestamp_ns / 1e9
            monotonic_sec = time.monotonic()

            if not ok or frame is None:
                dropped += 1
                print(f"[WARN] Dropped frame read #{dropped}")
                time.sleep(0.01)
                continue

            try:
                y = decode_yuyv(frame)
            except Exception as e:
                print(f"[ERROR] Decode failed: {e}")
                print(f"[ERROR] frame shape={getattr(frame, 'shape', None)}, dtype={getattr(frame, 'dtype', None)}")
                break

            top_y, bottom_y = split_top_bottom(y)

            name = f"{frame_id:06d}_{stamp}.png"
            top_y_file = ""
            bottom_y_file = ""

            if top_y is not None and bottom_y is not None:
                top_y_path = dirs["top_y"] / name
                bottom_y_path = dirs["bottom_y"] / name

                cv2.imwrite(str(top_y_path), top_y)
                cv2.imwrite(str(bottom_y_path), bottom_y)

                top_y_file = str(top_y_path.relative_to(out_dir))
                bottom_y_file = str(bottom_y_path.relative_to(out_dir))
            else:
                # In 256x192 mode there is no vertical split; save full Y in top_y dir.
                top_y_path = dirs["top_y"] / name
                cv2.imwrite(str(top_y_path), y)
                top_y_file = str(top_y_path.relative_to(out_dir))

            packed_file = ""

            if args.save_packed_npy:
                packed_name = f"{frame_id:06d}_{stamp}.npy"
                packed_path = dirs["packed"] / packed_name
                np.save(str(packed_path), frame)
                packed_file = str(packed_path.relative_to(out_dir))

            preview = y

            if args.save_preview_every > 0 and frame_id % args.save_preview_every == 0:
                preview_path = dirs["preview"] / name
                cv2.imwrite(str(preview_path), preview)

            if mp4_writer is None:
                mp4_fps = actual_fps if actual_fps > 1.0 else args.fps
                mp4_writer = make_mp4_writer(mp4_path, preview, mp4_fps)

            preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
            mp4_writer.write(preview_bgr)

            y_min = int(np.min(y))
            y_max = int(np.max(y))
            y_mean = float(np.mean(y))

            writer.writerow([
                frame_id,
                timestamp_ns,
                f"{timestamp_sec:.9f}",
                f"{monotonic_sec:.9f}",
                top_y_file,
                bottom_y_file,
                packed_file,
                y_min,
                y_max,
                f"{y_mean:.3f}",
                str(frame.shape),
                str(frame.dtype),
                stamp,
            ])

            if args.display:
                global PAUSED
                cv2.imshow("Y channel - q stop, SPACE pause/resume", preview)

                # if top_y is not None and bottom_y is not None:
                #     cv2.imshow("top Y", top_y)
                #     cv2.imshow("bottom Y", bottom_y)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[INFO] q pressed. Stopping.")
                    break
                if key == ord(" "):
                    PAUSED = not PAUSED
                    print("[INFO] Paused." if PAUSED else "[INFO] Resumed.")

                while PAUSED and RUNNING:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord("q"):
                        RUNNING = False
                        break
                    if key == ord(" "):
                        PAUSED = False
                        print("[INFO] Resumed.")
                        break
                    if stop_file.exists():
                        RUNNING = False
                        break

            if frame_id % 50 == 0:
                elapsed_wall = time.time() - start_wall
                hz = frame_id / elapsed_wall if elapsed_wall > 0 else 0.0
                print(
                    f"[INFO] frame={frame_id} hz={hz:.2f} "
                    f"Y min={y_min} max={y_max} mean={y_mean:.1f} "
                    f"shape={frame.shape} dtype={frame.dtype}"
                )

            frame_id += 1

    cap.release()

    if mp4_writer is not None:
        mp4_writer.release()

    if args.display:
        cv2.destroyAllWindows()

    elapsed_wall = time.time() - start_wall
    hz = frame_id / elapsed_wall if elapsed_wall > 0 else 0.0

    print("[DONE]")
    print(f"[DONE] Frames saved: {frame_id}")
    print(f"[DONE] Dropped reads: {dropped}")
    print(f"[DONE] Average rate: {hz:.2f} Hz")
    print(f"[DONE] Output folder: {out_dir}")
    print(f"[DONE] Preview video: {mp4_path}")


if __name__ == "__main__":
    main()


