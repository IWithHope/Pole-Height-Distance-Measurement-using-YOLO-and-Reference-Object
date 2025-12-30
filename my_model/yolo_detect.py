import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# ============================================================
# GLOBALS FOR MOUSE MEASUREMENT
# ============================================================
# Stores click points in ORIGINAL (frame) coordinates
clicked_points = []             # list of (x,y) in original frame coords; grouped into pairs (1-2, 3-4, ...)
measurements = []               # list of dicts: {'p1':(x,y),'p2':(x,y),'pixel_dist':v,'meters':v or None}
real_frame_for_drawing = None   # original-sized frame that we draw into
current_display_scale = 1.0     # scale factor applied to show the frame in window
meters_per_pixel = None         # determined from cones (0.7 m)

# UI / window
WINDOW_NAME = "YOLO detection results"

# ============================================================
# Convert Display Coordinates → Real Frame Coordinates
# ============================================================
def display_to_real(x, y):
    """Map a coordinate from displayed (scaled) window coords back to original frame coords."""
    global current_display_scale
    if current_display_scale == 0 or current_display_scale is None:
        return int(x), int(y)
    rx = int(round(x / current_display_scale))
    ry = int(round(y / current_display_scale))
    return rx, ry

# ============================================================
# Refresh Scaled Window After Drawing
# ============================================================
def refresh_scaled_display():
    """Scale real_frame_for_drawing to fit a reasonable window size and show it."""
    global real_frame_for_drawing, current_display_scale
    if real_frame_for_drawing is None:
        return
    h, w = real_frame_for_drawing.shape[:2]

    # Keep the whole image visible by scaling (no cropping).
    # If image smaller than max dims, keep 1:1.
    max_w, max_h = 1200, 800
    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h, 1.0)  # never upscale beyond 1.0
    current_display_scale = scale

    disp_w = int(round(w * scale))
    disp_h = int(round(h * scale))

    # Use INTER_AREA for shrinking
    disp = cv2.resize(real_frame_for_drawing, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW_NAME, disp)


# ============================================================
# Mouse Callback — MULTIPLE POLES SUPPORTED (pairs: 1-2, 3-4, ...)
# ============================================================
def mouse_callback(event, x, y, flags, param):
    """Left click to mark a point. Points are stored in original-frame coords.
       When the click count becomes even (2,4,6...), an independent measurement
       is computed for the last pair."""
    global clicked_points, real_frame_for_drawing, meters_per_pixel, measurements

    if real_frame_for_drawing is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert display coords -> original frame coords
        rx, ry = display_to_real(x, y)
        clicked_points.append((rx, ry))

        # Draw a small marker on the real frame (we draw on the original-sized frame)
        cv2.circle(real_frame_for_drawing, (rx, ry), 6, (0, 0, 255), -1)

        # Only compute measurement when we have completed a pair (even number of points)
        if len(clicked_points) % 2 == 0:
            p1 = clicked_points[-2]
            p2 = clicked_points[-1]

            # pixel distance (vertical distance used for height if you want vertical only,
            # but we calculate Euclidean distance so user can click top and bottom not strictly vertical)
            pixel_dist = float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))

            # compute meters if we have meters_per_pixel
            meters = None
            if meters_per_pixel is not None:
                meters = pixel_dist * meters_per_pixel

            # Save measurement independent of other pairs
            measurements.append({
                'p1': p1,
                'p2': p2,
                'pixel_dist': pixel_dist,
                'meters': meters
            })

            # Draw the arrow/line and label on the real frame
            draw_measurement_on_frame(real_frame_for_drawing, measurements[-1])

        # Refresh displayed image
        refresh_scaled_display()

    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click = clear all clicks & measurements
        clicked_points = []
        measurements = []
        # Optionally also clear meters_per_pixel (we keep it)
        # meters_per_pixel = None
        # Repaint frame by re-running detection loop (main loop will overwrite real_frame_for_drawing)
        refresh_scaled_display()


# ============================================================
# Draw a measurement (arrow + label) on a frame (in original coords)
# ============================================================
def draw_measurement_on_frame(frame, meas):
    p1 = meas['p1']
    p2 = meas['p2']
    pixel_dist = meas['pixel_dist']
    meters = meas.get('meters', None)

    # draw line and arrow (arrowedLine draws arrow from p1->p2; use both directions for clear arrowheads)
    cv2.line(frame, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.arrowedLine(frame, p1, p2, (255, 0, 0), 2, tipLength=0.05)
    cv2.arrowedLine(frame, p2, p1, (255, 0, 0), 2, tipLength=0.05)

    # Put text near mid point
    mid_x = int((p1[0] + p2[0]) / 2)
    mid_y = int((p1[1] + p2[1]) / 2)
    if meters is not None:
        label = f"{meters:.2f} m"
    else:
        label = f"{int(pixel_dist)} px"
    # Draw a filled rectangle behind label for readability
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    rect_tl = (mid_x + 8, mid_y - text_h - 6)
    rect_br = (mid_x + 8 + text_w + 6, mid_y + 6)
    cv2.rectangle(frame, rect_tl, rect_br, (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, label, (mid_x + 10, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


# ============================================================
# Draw Cone Height Arrow (0.7m reference)
# ============================================================
def draw_height_arrow(frame, box, label_text="0.7 m"):
    x1, y1, x2, y2 = map(int, box)
    mid_x = int((x1 + x2) / 2)
    top_y = y1
    bottom_y = y2

    cv2.line(frame, (mid_x, top_y), (mid_x, bottom_y), (0, 255, 0), 2)
    cv2.arrowedLine(frame, (mid_x, bottom_y), (mid_x, top_y + 20), (0, 255, 0), 2, tipLength=0.07)
    cv2.arrowedLine(frame, (mid_x, top_y), (mid_x, bottom_y - 20), (0, 255, 0), 2, tipLength=0.07)

    text_y = int((top_y + bottom_y) / 2)
    cv2.putText(frame, label_text, (mid_x + 12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame


# ============================================================
# ARGUMENT PARSER
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, help='Path to YOLO model file (e.g. runs/detect/train/weights/best.pt)')
parser.add_argument('--source', required=True, help='Image file, folder, or video.')
parser.add_argument('--thresh', default=0.5, help='Confidence threshold')
parser.add_argument('--resolution', default=None, help='(optional) WxH display resolution - not used for cropping')
parser.add_argument('--record', action='store_true', help='(optional) write output video')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

# ============================================================
# CHECK MODEL FILE
# ============================================================
if not os.path.exists(model_path):
    print('ERROR: Model file not found:', model_path)
    sys.exit(1)

# Load YOLO model
model = YOLO(model_path, task='detect')
labels = model.names

# ============================================================
# DETERMINE SOURCE TYPE
# ============================================================
img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.JPG', '.JPEG', '.PNG', '.BMP']
vid_exts = ['.avi', '.mov', '.mp4', '.mkv', '.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    ext = os.path.splitext(img_source)[1]
    source_type = 'image' if ext.lower() in img_exts else 'video'
else:
    print("Invalid source:", img_source)
    sys.exit(1)

# Load images list if image/folder
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(os.path.join(img_source, "*")) if os.path.splitext(f)[1].lower() in img_exts]
    imgs_list.sort()

# Video capture (if needed)
cap = None
if source_type == 'video':
    cap = cv2.VideoCapture(img_source)
    # Optionally set resolution (if user supplied) – commented out so we don't crop/resize incorrectly
    # if user_res:
    #     resW, resH = map(int, user_res.split('x'))
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

# ============================================================
# Prepare window & callback
# ============================================================
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

# Optional: prepare video writer if requested (size must match original frame size when starting)
recorder = None

# ============================================================
# MAIN LOOP
# ============================================================
img_index = 0
video_paused = False

while True:
    start_t = time.perf_counter()

    # load frame
    if source_type in ['image', 'folder']:
        if img_index >= len(imgs_list):
            print("Processed all images.")
            break
        frame = cv2.imread(imgs_list[img_index])
        if frame is None:
            print("Failed to read:", imgs_list[img_index])
            img_index += 1
            continue
        # For images: we wait for user key to move to next
    else:  # video
        if not video_paused:
            ret, frame = cap.read()
            if not ret:
                print("End of video.")
                break
        else:
            # when paused, keep last frame (frame variable remains)
            pass

    # keep an editable copy of original-sized image to draw into
    real_frame_for_drawing = frame.copy()

    # ---------- YOLO detection ----------
    results = model(frame, verbose=False)
    detections = results[0].boxes

    # Build a list of cone pixel heights (we average them to get better reference)
    cone_pixel_heights = []
    for det in detections:
        conf = det.conf.item()
        classidx = int(det.cls.item())
        classname = labels[classidx].lower()
        if conf < min_thresh:
            continue
        xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
        xmin, ymin, xmax, ymax = xyxy

        # draw bounding box & label
        color = (255, 100, 0)
        cv2.rectangle(real_frame_for_drawing, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(real_frame_for_drawing, f"{classname}: {int(conf*100)}%",
                    (xmin, max(0, ymin - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # If cone — measure pixel height and draw arrow labelled 0.7 m
        if "cone" in classname or classidx == 0:
            cone_pixel_heights.append(abs(ymax - ymin))
            draw_height_arrow(real_frame_for_drawing, [xmin, ymin, xmax, ymax], "0.7 m")

    # compute meters_per_pixel using average cone height if present
    if len(cone_pixel_heights) > 0:
        avg_cone_px = float(np.mean(cone_pixel_heights))
        meters_per_pixel = 0.7 / avg_cone_px
    else:
        meters_per_pixel = None

    # Redraw any previous user measurements (so they persist across frames)
    for m in measurements:
        draw_measurement_on_frame(real_frame_for_drawing, m)

    # If user has started a new pair but not yet finished (odd number of clicked_points),
    # show the single point marker so user sees it (we keep the click markers in original coords)
    for pt in clicked_points:
        cv2.circle(real_frame_for_drawing, pt, 6, (0, 255, 255), -1)

    # Overlay helpful text (top-left)
    overlay_y = 20
    cv2.putText(real_frame_for_drawing, f"Detections: {len(detections)}", (10, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    overlay_y += 26
    if meters_per_pixel is not None:
        cv2.putText(real_frame_for_drawing, f"Estimated px/m: {1/meters_per_pixel:.1f} px/m   (ref 0.7m cones)",
                    (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    else:
        cv2.putText(real_frame_for_drawing, "No cone detected → scale unknown (measurements in px)",
                    (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    overlay_y += 22
    cv2.putText(real_frame_for_drawing, "Left-click: add point; Right-click: clear all; q: quit; c: clear measurements; p: save",
                (10, overlay_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # If a pair just completed (the last measurement may have meters==None but we can update it now if cone found),
    # ensure measurements entry uses meters_per_pixel if available
    for m in measurements:
        if m['meters'] is None and meters_per_pixel is not None:
            m['meters'] = m['pixel_dist'] * meters_per_pixel

    # show scaled display
    refresh_scaled_display()

    # write to recorder if requested - must initialize writer on first frame (match frame size)
    if record and recorder is None:
        h, w = frame.shape[:2]
        record_fname = 'demo_out.avi'
        recorder = cv2.VideoWriter(record_fname, cv2.VideoWriter_fourcc(*'MJPG'), 20, (w, h))

    if record and recorder is not None:
        # write original sized frame with drawings (not scaled)
        recorder.write(real_frame_for_drawing)

    # Key handling
    if source_type in ['image', 'folder']:
        key = cv2.waitKey(0) & 0xFF
    else:
        key = cv2.waitKey(5) & 0xFF

    if key == ord('q') or key == 27:  # q or ESC
        break
    elif key == ord('c'):
        # clear measurements and clicks but keep cones & scale
        clicked_points = []
        measurements = []
        # repaint will happen next loop
    elif key == ord('p'):
        # save capture of drawn original-sized frame
        fname = f"capture_{int(time.time())}.png"
        cv2.imwrite(fname, real_frame_for_drawing)
        print("Saved", fname)
    elif key == ord('n'):
        # next image in folder
        if source_type in ['image', 'folder']:
            img_index += 0  # we already incremented when reading; to force next, simply continue
            # nothing special required because loop reads next image at top
    elif key == ord('s'):
        if source_type in ['video']:
            video_paused = not video_paused
            if video_paused:
                print("Video paused. Click points to measure poles.")
            else:
                print("Resuming video.")
    # else: ignore other keys

    # small sleep optional (not necessary)
    # time.sleep(0.001)

    # end of loop timing
    end_t = time.perf_counter()
    # fps = 1.0 / (end_t - start_t) if (end_t - start_t) > 0 else 0.0
    # print("Loop FPS:", fps)

    # If processing images, go to next after user presses any key (we already waitKey(0))
    if source_type in ['image'] and key in [ord('q'), 27]:
        break
    if source_type in ['image', 'folder'] and key != ord('q') and key != 27:
        # continue to next image on any non-q key (this preserves previous behavior: wait for key then continue)
        img_index += 0  # already advanced when reading; nothing else needed

# cleanup
cv2.destroyAllWindows()
if cap is not None:
    cap.release()
if recorder is not None:
    recorder.release()
