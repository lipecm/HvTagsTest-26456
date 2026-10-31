#!/usr/bin/env python3
"""AprilTag detector -> the desktop simulator dashboard.

Your laptop webcam (or a video file) becomes the robot camera: tags are detected
(tag36h11, the FTC family), drawn on a live preview AND streamed into the dashboard's
Camera panel, with distance/bearing/roll posted to http://localhost:8080.

    python webcam_tags.py                             # auto-picks the camera that sees tags
    python webcam_tags.py --camera 1                  # force a specific camera
    python webcam_tags.py --source "C:\\clips\\f.mp4" # video file (loops)
    python webcam_tags.py --tag-size 2.0              # smaller print tags

LOOP BUDGET
    The capture/detect loop performs no network I/O at all. Config polling, posting
    detections to the simulator and JPEG-encoding the preview stream each run on their
    own thread, so a frame costs only: grab + decode (+ one rescue pass if nothing was
    found). Loop time is printed on the HUD and every 5 s in the log.
"""
import argparse
import json
import math
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import cv2.aruco as aruco

DICT = aruco.getPredefinedDictionary(aruco.DICT_APRILTAG_36h11)


# --------------------------------------------------------------------------- #
# background workers - keep the hot loop free of sockets and JPEG encoding      #
# --------------------------------------------------------------------------- #

class MJPEGStream:
    """Serves the latest annotated frame as multipart MJPEG, for the dashboard's Camera panel."""

    def __init__(self, port):
        self.frame_id = 0
        self.jpg = None
        self._pending = None
        holder = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path != "/stream.mjpg":
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                last_id = -1
                try:
                    while True:
                        if holder.jpg is None or holder.frame_id == last_id:
                            time.sleep(0.02)
                            continue
                        last_id = holder.frame_id
                        jpg = holder.jpg
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                            + str(len(jpg)).encode()
                            + b"\r\n\r\n" + jpg + b"\r\n"
                        )
                except Exception:
                    pass  # browser went away

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        threading.Thread(target=self._encode_loop, daemon=True).start()

    def submit(self, frame):
        """Non-blocking: hand the frame to the encoder thread (each cap.read() yields a
        fresh array, so the worker never races the next iteration)."""
        self._pending = frame

    def _encode_loop(self):
        while True:
            frame = self._pending
            if frame is None:
                time.sleep(0.004)
                continue
            self._pending = None
            try:
                ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ok:
                    self.jpg = jpg.tobytes()
                    self.frame_id += 1
            except Exception:
                pass


class IngestPoster:
    """POSTs detections to the simulator without ever blocking the capture loop."""

    def __init__(self, url):
        self.url = url
        self.ok = None
        self._body = None
        self._event = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, body):
        self._body = body
        self._event.set()

    def _run(self):
        while True:
            self._event.wait(0.04)
            self._event.clear()
            body = self._body
            if body is None:
                continue
            try:
                req = urllib.request.Request(
                    self.url, data=body, headers={"Content-Type": "application/json"}
                )
                urllib.request.urlopen(req, timeout=0.4).read()
                self.ok = True
            except Exception as e:
                if self.ok is not False:
                    print(f"sim not reachable ({type(e).__name__}) - preview still runs")
                self.ok = False


class ConfigClient:
    """Polls the dashboard's tunables in the background; the loop just reads .cfg."""

    def __init__(self, url, period=0.4):
        self.url = url
        self.cfg = None
        self._stop = False
        threading.Thread(target=self._run, daemon=True).start()
        # NOTE: period is the refresh rate; the loop never waits on this thread.

    def stop(self):
        self._stop = True

    def _run(self):
        while not self._stop:
            try:
                self.cfg = json.loads(urllib.request.urlopen(self.url, timeout=0.3).read())
            except Exception:
                pass
            time.sleep(0.4)


# --------------------------------------------------------------------------- #
# camera helpers                                                               #
# --------------------------------------------------------------------------- #

def try_open_camera(idx):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)  # DirectShow is the reliable one on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(idx)
    return cap if cap.isOpened() else None


def probe_cameras(indices, tries=8):
    """Pick the camera that actually sees AprilTags (the default index is often a
    different device). Returns (best_index, {index: (max_tags, distinct_ids, frames)})."""
    results = {}
    for idx in indices:
        cap = try_open_camera(idx)
        if cap is None:
            results[idx] = None
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        max_tags, seen, frames = 0, set(), 0
        for i in range(tries):
            ok, frame = cap.read()
            if not ok:
                continue
            frames += 1
            if i < 2:
                continue  # first frames are still settling exposure
            ids = detect_ids(frame, PRIMARY, RESCUE)
            if ids is not None and len(ids):
                max_tags = max(max_tags, len(ids))
                seen.update(int(x) for x in ids)
        cap.release()
        results[idx] = (max_tags, len(seen), frames)

    best, score = None, (-1, -1)
    for idx, r in results.items():
        if r is None:
            continue
        if (r[0], r[1]) > score:
            best, score = idx, (r[0], r[1])
    return best, results


def waiting_frame(w, h, text):
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(frame, text, (24, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


def hud_text(img, text, org, scale=0.6, thick=2):
    """Monochrome HUD label: white glyphs + black outline, readable on any scene."""
    x, y = org
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thick, cv2.LINE_AA)


# --------------------------------------------------------------------------- #
# detection                                                                    #
# --------------------------------------------------------------------------- #

def make_detector(tuned):
    p = aruco.DetectorParameters()
    p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    p.cornerRefinementWinSize = 5
    p.cornerRefinementMaxIterations = 20
    p.cornerRefinementMinAccuracy = 0.02
    if tuned:
        p.minMarkerPerimeterRate = 0.01
        p.adaptiveThreshWinSizeMax = 45
        p.adaptiveThreshWinSizeStep = 5
        p.polygonalApproxAccuracyRate = 0.05
        p.errorCorrectionRate = 0.8
    return aruco.ArucoDetector(DICT, p)


# Primary: fast, proven 8/8 on this camera. Rescue: looser, only run when the primary
# pass comes back empty (so the common frame costs exactly one pass).
PRIMARY = make_detector(False)
RESCUE = make_detector(True)


def detect_ids(frame, primary=PRIMARY, rescue=RESCUE):
    corners, ids, _ = primary.detectMarkers(frame)
    if ids is None or not len(ids):
        corners, ids, _ = rescue.detectMarkers(frame)
    return ids


def detect_corners_ids(frame, primary=PRIMARY, rescue=RESCUE):
    corners, ids, _ = primary.detectMarkers(frame)
    if ids is None or not len(ids):
        corners, ids, _ = rescue.detectMarkers(frame)
    if ids is None or not len(ids):
        return [], None
    return list(corners), ids.reshape(-1)


# --------------------------------------------------------------------------- #
# temporal stability - kills flicker, but keeps duplicate-ID tags separate      #
# --------------------------------------------------------------------------- #

class TrackStability:
    """A tag must be seen in `confirm` consecutive frames before it is emitted, and stays
    emitted for `hold` seconds through dropouts.

    Tracks are keyed by (id, centre position) - NOT by id alone - because this tag set
    legitimately contains two copies of each id (8 physical tags, 4 unique ids).
    """

    def __init__(self, confirm=2, hold=0.25, match_px=90.0):
        self.confirm = confirm
        self.hold = hold
        self.match_px = match_px
        self.tracks = []

    def update(self, items):
        """items: list of (cx, cy, payload). Returns the payloads that should be emitted."""
        now = time.time()
        matched = set()

        for track in self.tracks:
            best, best_d = None, self.match_px
            for i, (cx, cy, _) in enumerate(items):
                if i in matched or track["id"] != items[i][2]["id"]:
                    continue
                d = math.hypot(track["cx"] - cx, track["cy"] - cy)
                if d < best_d:
                    best, best_d = i, d
            if best is not None:
                matched.add(best)
                cx, cy, payload = items[best]
                track["cx"], track["cy"] = cx, cy
                track["seen"] = now
                track["hits"] += 1
                track["payload"] = payload
            # else: keep the track alive until `hold` expires

        for i, (cx, cy, payload) in enumerate(items):
            if i in matched:
                continue
            self.tracks.append({
                "id": payload["id"], "cx": cx, "cy": cy,
                "seen": now, "hits": 1, "payload": payload,
            })

        self.tracks = [t for t in self.tracks if now - t["seen"] <= self.hold]
        return [t["payload"] for t in self.tracks if t["hits"] >= self.confirm]


# --------------------------------------------------------------------------- #
# pose                                                                         #
# --------------------------------------------------------------------------- #

def intrinsics(width, height, fov_deg):
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array([[fx, 0, width / 2.0],
                     [0, fx, height / 2.0],
                     [0, 0, 1.0]], dtype=np.float64)


def pose_from_pnp(pts, tag_size, K):
    """IPPE square solve -> FTC frame (x right, y up, z forward). None on failure."""
    s = tag_size / 2.0
    # OpenCV SOLVEPNP_IPPE_SQUARE wants TL, TR, BR, BL with this exact object layout.
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float64)
    flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    try:
        ok, rvec, tvec = cv2.solvePnP(obj, pts.astype(np.float64), K, None, flags=flag)
    except Exception:
        return None
    if not ok:
        return None
    x, y, z = float(tvec[0][0]), -float(tvec[1][0]), float(tvec[2][0])  # flip cam-y to up
    rng = math.sqrt(x * x + y * y + z * z)
    if not (0.0 < rng < 1e5) or z <= 0:
        return None
    return {
        "range": rng,
        "bearing": math.degrees(math.atan2(x, z)),
        "elevation": math.degrees(math.asin(max(-1.0, min(1.0, y / rng)))),
        "x": x, "y": y, "z": z,
    }


def detect_frame(display, work, expand, width, height, tag_size, fov_deg):
    """Detect on `work` (maybe downscaled), report/draw in full-res coords."""
    corners, ids = detect_corners_ids(work)
    if ids is None:
        return []

    full = np.array([(c.reshape(4, 2) * expand).reshape(1, 4, 2) for c in corners],
                    dtype=np.float32)
    aruco.drawDetectedMarkers(display, full, ids.reshape(-1, 1), (255, 255, 255))

    K = intrinsics(width, height, fov_deg)
    detections = []

    for corner, marker_id in zip(full, ids):
        pts = corner.reshape(4, 2).astype(float)  # TL, TR, BR, BL
        tl, tr, br, bl = pts
        cx, cy = pts.mean(axis=0)

        side = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)
                + np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 4.0
        if side < 8.0:
            continue  # degenerate mark - never let it become a target
        span = max(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)), 1e-6)
        if side / span > 0.75:
            continue  # not a ringed marker (filled blob), would corrupt pose

        pose = pose_from_pnp(pts, tag_size, K)
        if pose is None:
            size_px = max(side, 1e-6)
            distance = tag_size * K[0][0] / size_px
            bearing = math.degrees(math.atan2(cx - width / 2.0, K[0][0]))
            elevation = math.degrees(math.atan2(-(cy - height / 2.0), K[0][0]))
            pose = {
                "range": distance, "bearing": bearing, "elevation": elevation,
                "x": distance * math.sin(math.radians(bearing)),
                "y": distance * math.sin(math.radians(elevation)),
                "z": distance,
            }

        distance = pose["range"]
        bearing = pose["bearing"]
        elevation = pose["elevation"]
        roll = math.degrees(math.atan2(tr[1] - tl[1], tr[0] - tl[0]))

        payload = {
            "kind": "single",
            "name": None,
            "id": int(marker_id),
            "meta": False,
            "percent": 100,
            "margin": 100,
            "hamming": 0,
            "range": round(distance, 1),
            "bearing": round(bearing, 1),
            "elevation": round(elevation, 1),
            "x": round(pose["x"], 1),
            "y": round(pose["y"], 1),
            "z": round(pose["z"], 1),
            "pitch": round(-elevation, 1),
            "roll": round(roll, 1),
            "yaw": round(bearing, 1),
        }
        detections.append((cx, cy, payload))

        hud_text(display, f"ID {int(marker_id)}  {distance:.0f}in away",
                 (int(tl[0]), max(int(tl[1]) - 10, 16)), 0.6, 2)
    return detections


# --------------------------------------------------------------------------- #
# camera controls                                                              #
# --------------------------------------------------------------------------- #

def set_exposure(cap, exposure_us):
    if exposure_us <= 0:
        ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # DirectShow: 0.75 = auto
        if not ok:
            ok = cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # MSMF: 3 = auto
        return "auto" if ok else "auto (prop unsupported)"

    # Most UVC cams on Windows take exposure in log2(seconds); clamp to the usual range.
    t = max(1e-6, exposure_us / 1e6)
    native = int(max(-13, min(-1, round(math.log2(t)))))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # DirectShow: 0.25 = manual
    ok = cap.set(cv2.CAP_PROP_EXPOSURE, native)
    shown = cap.get(cv2.CAP_PROP_EXPOSURE)
    return f"{native} ({exposure_us / 1000:.1f} ms, cam says {shown:g})" if ok \
        else f"unsupported (wanted {native})"


def apply_camera_controls(cap, cfg, state, is_video):
    """Push dashboard exposure/gain/resolution into the actual webcam. True = size changed."""
    if not cfg:
        return False
    resized = False

    exposure = int(cfg.get("exposureUs", -1))
    if exposure != state.get("exposure"):
        state["exposure"] = exposure
        state["exposureText"] = set_exposure(cap, exposure)
        print(f"exposure -> {state['exposureText']}")

    gain = int(cfg.get("gain", -1))
    if gain != state.get("gain"):
        state["gain"] = gain
        if gain < 0:
            state["gainText"] = "unchanged"
        else:
            state["gainText"] = str(gain) if cap.set(cv2.CAP_PROP_GAIN, gain) else "unsupported"
        print(f"gain -> {state['gainText']}")

    resolution = int(cfg.get("resolution", 0))
    if resolution != state.get("resolution") and not is_video:
        state["resolution"] = resolution
        w, h = [(640, 480), (1280, 720)][resolution if resolution in (0, 1) else 0]
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        print(f"resolution -> {w}x{h}")
        resized = True

    return resized


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", default="auto",
                    help="'auto' (default) probes for the camera that sees tags, a camera "
                         "index, or a video file path")
    ap.add_argument("--camera", type=int, default=None,
                    help="shortcut for a camera index (overrides --source)")
    ap.add_argument("--probe", default="0,1,2,3",
                    help="camera indices tried in auto mode (default 0,1,2,3)")
    ap.add_argument("--tag-size", type=float, default=5.0,
                    help="tag side length in inches (FTC 2024+ decode tags are 5)")
    ap.add_argument("--fov", type=float, default=65.0, help="horizontal FOV, degrees")
    ap.add_argument("--ingest", default="http://127.0.0.1:8081/sim/detections")
    ap.add_argument("--mirror", action="store_true", help="flip preview horizontally")
    ap.add_argument("--no-preview", action="store_true", help="no preview window")
    ap.add_argument("--no-loop", action="store_true", help="stop at end of video instead of looping")
    ap.add_argument("--stream-port", type=int, default=8090,
                    help="MJPEG stream for the dashboard's Camera panel")
    ap.add_argument("--no-stream", action="store_true", help="don't serve the dashboard stream")
    ap.add_argument("--camera-settle", type=float, default=1.0,
                    help="seconds to let auto-exposure settle before trusting detection")
    args = ap.parse_args()

    # ---------------- source selection ----------------
    forced = args.camera is not None
    if forced:
        source, mode = str(args.camera), "cam"
    elif args.source == "auto":
        source, mode = None, "probe"
    elif args.source.isdigit():
        source, mode = args.source, "cam"
    else:
        source, mode = args.source, "video"

    is_video = mode == "video"
    config_url = args.ingest.rsplit("/", 1)[0] + "/config"  # :8081/sim/config
    cap = None

    if mode == "video":
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise SystemExit(f"cannot open video '{source}'")
    elif mode == "probe":
        indices = [int(x) for x in args.probe.split(",") if x.strip().isdigit()]
        print(f"probing cameras {indices} for AprilTags...")
        best, results = probe_cameras(indices)
        for idx in indices:
            r = results.get(idx)
            print(f"  camera {idx}: " + ("no device" if r is None
                                         else f"max {r[0]} tags, {r[1]} distinct id(s), {r[2]} frames"))
        if best is None:
            print("no camera saw a tag - defaulting to camera 0 (use --camera N to force)")
            best = 0
        source, mode = str(best), "cam"
        print(f"-> using camera {source}")
        cap = try_open_camera(int(source))
    else:
        cap = try_open_camera(int(source))

    if mode == "cam" and cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if cap else 640
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if cap else 480
    tag_size = args.tag_size
    fov = args.fov

    stream = None
    if not args.no_stream:
        try:
            stream = MJPEGStream(args.stream_port)
            print(f"dashboard video: http://localhost:{args.stream_port}/stream.mjpg")
        except OSError as e:
            print(f"stream port {args.stream_port} unavailable ({e}) - dashboard video off")

    print(f"tag size {tag_size} in, FOV {fov} deg, posting to {args.ingest}")
    if mode == "cam" and cap is None:
        print(f"waiting for camera {source} - plug the USB webcam in and it starts by itself")

    title = "AprilTag - " + (source if mode == "cam" else source)
    frames = tag_frames = all8_frames = 0
    last_report = time.time()
    last_post_ok = None
    loop_ms_avg = loop_ms_max = 0.0
    det_ms_avg = 0.0

    poster = IngestPoster(args.ingest)
    cfg_client = ConfigClient(config_url)
    cfg = None
    ctrl = {}
    stability = TrackStability()
    settle_until = time.time() + args.camera_settle

    while True:
        t_frame = time.perf_counter()

        # ---- dashboard tunables (read only; polling happens on ConfigClient's thread) ----
        cfg = cfg_client.cfg if cfg_client.cfg is not None else cfg
        if cap is not None and not is_video and cfg is not None:
            if apply_camera_controls(cap, cfg, ctrl, is_video):
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                settle_until = time.perf_counter() + args.camera_settle
                print(f"camera now {width}x{height}")
        if cfg:
            tag_size = float(cfg.get("tagSizeIn", tag_size) or tag_size)
            fov = float(cfg.get("fovDeg", fov) or fov)

        # ---- get a frame (video loops; camera waits for the USB stick) ----
        if is_video:
            ok, frame = cap.read()
            if not ok:
                if args.no_loop:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            if not (0 < src_fps < 120):
                src_fps = 30.0
        else:
            if cap is None:
                frame = waiting_frame(640, 480, f"waiting for camera {source}...")
                if stream:
                    stream.submit(frame)
                if not args.no_preview:
                    cv2.imshow(title, frame)
                    if cv2.waitKey(100) & 0xFF == ord("q"):
                        break
                time.sleep(1.0)
                cap = try_open_camera(int(source))
                if cap is not None:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    ctrl.clear()
                    settle_until = time.perf_counter() + args.camera_settle
                    print(f"camera attached: {width}x{height}")
                continue
            ok, frame = cap.read()
            if not ok:
                cap.release()
                cap = None
                print("camera lost - waiting for it to come back")
                continue
            src_fps = 30.0

        if args.mirror:
            frame = cv2.flip(frame, 1)

        # ---- decimation: detect on a downscaled copy (SDK parity) ----
        dec = float((cfg or {}).get("decimation", 1) or 1)
        dec = max(1.0, min(4.0, dec))
        if dec > 1.01:
            work = cv2.resize(frame, (max(1, int(width / dec)), max(1, int(height / dec))),
                              interpolation=cv2.INTER_AREA)
        else:
            work = frame
        expand = width / work.shape[1]

        t_det = time.perf_counter()
        candidates = detect_frame(frame, work, expand, width, height, tag_size, fov)
        det_ms = (time.perf_counter() - t_det) * 1000.0

        # ---- temporal stability ----
        settling = time.perf_counter() < settle_until
        if settling:
            stability.update(candidates)
            payloads = [p for _, _, p in candidates]
        else:
            payloads = list(stability.update(candidates))

        poster.submit(json.dumps({"detections": payloads}).encode("utf-8"))
        if poster.ok is not None:
            last_post_ok = poster.ok

        frames += 1
        if payloads:
            tag_frames += 1
        if len(payloads) >= 8:
            all8_frames += 1

        loop_ms = (time.perf_counter() - t_frame) * 1000.0
        loop_ms_avg += (loop_ms - loop_ms_avg) / min(frames, 50)
        loop_ms_max = max(loop_ms_max, loop_ms)
        det_ms_avg += (det_ms - det_ms_avg) / min(frames, 50)

        hud = f"{len(payloads)} tag(s) | cam {source} | loop {loop_ms:.1f} ms | " \
              f"sim:{'ok' if last_post_ok else 'off'}"
        hud_text(frame, hud, (12, height - 16), 0.6, 2)
        ctrl_hud = f"exp {ctrl.get('exposureText', '?')} | gain {ctrl.get('gainText', '?')} | " \
                   f"dec {dec:g} | det {det_ms:.1f} ms"
        hud_text(frame, ctrl_hud, (12, height - 44), 0.55, 2)
        if settling:
            hud_text(frame, "exposure settling...", (12, height - 68), 0.6, 2)

        if stream is not None:
            stream.submit(frame)

        if time.time() - last_report > 5:
            pct = 100.0 * tag_frames / max(frames, 1)
            print(f"{frames} frames | tags in {pct:.0f}% | all-8 in "
                  f"{100.0 * all8_frames / max(frames, 1):.0f}% | loop avg "
                  f"{loop_ms_avg:.1f} ms max {loop_ms_max:.1f} ms | detect avg {det_ms_avg:.1f} ms")
            last_report = time.time()
            frames = tag_frames = all8_frames = 0
            loop_ms_max = 0.0

        if not args.no_preview:
            cv2.imshow(title, frame)
            delay = 1 if is_video else max(1, int(1000 / src_fps))
            if cv2.waitKey(delay) & 0xFF == ord("q"):
                break

    cfg_client.stop()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
