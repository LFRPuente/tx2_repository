"""Open an AXIS camera live stream with OpenCV.

Credentials can be passed with flags, environment variables, or prompted
interactively:

    python tools/axis_camera_viewer.py --ip 10.14.115.74

Environment variables:
    AXIS_USER
    AXIS_PASSWORD
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys
from urllib.parse import quote

import cv2


def build_rtsp_url(ip: str, username: str, password: str, profile: str) -> str:
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    return f"rtsp://{user}:{pwd}@{ip}/axis-media/media.amp?videocodec={profile}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View an AXIS camera RTSP stream.")
    parser.add_argument("--ip", required=True, help="Camera IP, for example 10.14.115.74")
    parser.add_argument("--user", default=os.getenv("AXIS_USER"), help="AXIS username")
    parser.add_argument(
        "--password",
        default=os.getenv("AXIS_PASSWORD"),
        help="AXIS password. If omitted, the script prompts without echo.",
    )
    parser.add_argument(
        "--codec",
        choices=("h264", "jpeg"),
        default="h264",
        help="Requested stream codec.",
    )
    parser.add_argument("--window", default="AXIS camera", help="Window title")
    parser.add_argument(
        "--snapshot",
        help="Optional path to save one frame, then continue showing live video.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    username = args.user or input("AXIS user: ").strip()
    password = args.password
    if password is None:
        password = getpass.getpass("AXIS password: ")

    url = build_rtsp_url(args.ip, username, password, args.codec)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"Could not open RTSP stream for {args.ip}. Check credentials/network.")
        return 2

    saved_snapshot = False
    print("Live view open. Press q or Esc to close.")
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Stream ended or frame read failed.")
            break

        if args.snapshot and not saved_snapshot:
            cv2.imwrite(args.snapshot, frame)
            print(f"Saved snapshot: {args.snapshot}")
            saved_snapshot = True

        cv2.imshow(args.window, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
