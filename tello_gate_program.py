import cv2
import numpy as np
from djitellopy import Tello
import time

#Probar con 1 para el dron
tello = Tello()
tello.connect()
print(f"Battery: {tello.get_battery()}%")

tello.streamon()
frame_read = tello.get_frame_read()

def p_controller(error, kp):
    return kp * error

KP_X = 0.15
KP_Y = 0.12
DEADBAND_X = 20
DEADBAND_Y = 20
MAX_CONTROL = 20
CENTER_X_OFFSET = 5
CENTER_Y_OFFSET_GATE = 50  # Adjust this value to shift the vertical center point if needed
CENTER_Y_OFFSET = -50  # Adjust this value to shift the vertical center point if needed
PARTIAL_EDGE_MARGIN = 10
PARTIAL_GATE_ASPECT = 1.0 # Width to height ration of gate

# Frame counters for alignment detection
X_ZERO_CONFIRM_FRAMES = 4 # Number of consecutive frames with x error within deadband before considering it zero
x_zero_frames = 0
MAX_X_CHANGE_PER_FRAME = 2
last_left_right = 0

Y_HOLD_FRAMES = 5
Y_EXIT_TOLERANCE = DEADBAND_Y + 10

y_hold_count = 0
y_is_aligned = False


HOVER_Y_MIN = -260
HOVER_Y_MAX = -210
HOVER_CONFIRM_FRAMES = 5

hover_frame_count = 0
hover_mode = False

# Gate approach and crossing settings.
APPROACH_START_RATIO = 0.25
CROSS_RATIO = 0.45 #Smaller value means commits to crossing gate earlier
APPROACH_SPEED = 15
CROSS_SPEED = 15
CROSS_DURATION = 9.5
CENTER_CONFIRM_FRAMES = 3
CROSS_CONFIRM_FRAMES = 2
CENTER_SMOOTHING = 0.15

smoothed_center_x = None
smoothed_center_y = None

#Prevent one bad frame from stopping forward movement variables
APPROACH_GRACE_SECONDS = 0.20
last_centered_time = None


airborne = False
takeoff_requested = False

AXIS_PERIOD_SECONDS = 0.17
active_axis = "x"
last_axis_switch = time.monotonic()

flight_mode = "ALIGN"
cross_start_time = None
center_hold_count = 0
cross_ready_frames = 0

try:
    takeoff_requested = True
    tello.takeoff()
    airborne = True
    time.sleep(3) # Wait for the drone to stabilize after takeoff
    while True:
        # DJITelloPy supplies frames in RGB order; convert once so the OpenCV
        # masking, drawing, and display code below consistently uses BGR.
        frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_RGB2BGR)
        #Reset for the frame
        best_candidate = None
        best_area = 0
        partial_gate = False
        #Default no correction
        control_x = 0
        control_y = 0
        # setting forward and yaw to 0 for now, will be updated later
        forward = 0
        is_centered = False
        gate_size_ratio = None
        yaw = 0
        #Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        #Define color range for blue
        lower = np.array([100,80,70])
        upper = np.array([120,255,255])
        #Thresholding
        mask = cv2.inRange(hsv,lower,upper)
        # Bridge small gaps in the blue tape caused by video compression,
        # lighting, or an incomplete HSV mask.
        close_kernel = np.ones((9, 9), np.uint8)
        clean_mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            close_kernel
        )

        # Use the repaired mask for gate contour detection.
        contours, hierarchy = cv2.findContours(
            clean_mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        image_center_x = frame.shape[1] / 2 - CENTER_X_OFFSET
        image_center_y = frame.shape[0] / 2 + CENTER_Y_OFFSET

        # findCountours returns None when no contours were found.
        if hierarchy is not None:
            hierarchy = hierarchy[0] # Shape becomes: one [next, previous, child, parent]

            for outer_index, outer_contour in enumerate(contours):
                # Hierarchy entry format:
                # [next contour, previous contour, first child contour, parent contour]
                child_index = hierarchy[outer_index][2]
                parent_index = hierarchy[outer_index][3]

                # A gate should be an outer blue boundary, so it has no parent,
                # and it must enclose at least one child contour (the hole in the gate).
                if parent_index != -1 or child_index == -1:
                    continue

                # An outer contour can contain several children. Use the largest one
                # because the gate hole should be the dominant enclosed region.
                child_indeces = []
                while child_index != -1:
                    child_indeces.append(child_index)
                    child_index = hierarchy[child_index][0] # move to the next sibling

                inner_index = max(
                    child_indeces,
                    key=lambda index: cv2.contourArea(contours[index])
                )
                inner_contour = contours[inner_index]

                # Measure the outside and inside of the the candidate frame.
                outer_area = cv2.contourArea(outer_contour)
                inner_area = cv2.contourArea(inner_contour)
                if outer_area == 0:
                    continue

                # The opening should occupy a significant portion of the gate.
                # This rejects solid blue blobs and other non-gate shapes.
                opening_ratio = inner_area / outer_area

                # Get the outside rectangle dimensions.
                x, y, w, h = cv2.boundingRect(outer_contour)
                aspect_ratio = w / h
                image_area = frame.shape[0] * frame.shape[1]
                size_ratio = (w * h) / image_area

                # Approximate the outside and inside boundaries as polygons
                outer_epsilon = 0.02 * cv2.arcLength(outer_contour, True)
                inner_epsilon = 0.02 * cv2.arcLength(inner_contour, True)

                outer_approx = cv2.approxPolyDP(outer_contour, outer_epsilon, True)
                inner_approx = cv2.approxPolyDP(inner_contour, inner_epsilon, True)

                outer_vertices = len(outer_approx)
                inner_vertices = len(inner_approx)

                # A valid gate should:
                # - be large enought to matter,
                # - have a reasonable aspect ratio,
                # - occupy a significant portion of the image,
                # - have a large enough opening,
                # - have 4 vertices on the outside and inside.
                is_gate = (
                    outer_area > 500 and
                    0.5 < aspect_ratio < 3.0 and
                    size_ratio > 0.02 and
                    0.25 < opening_ratio < 0.95 and
                    outer_vertices == 4 and
                    inner_vertices == 4
                )

                if not is_gate:
                    continue

                # This remains the outer contour center for now.
                # Later, option 2 can replace this with the intersection of the
                # inner-opening diagonals, which is more accurate for navigation.
                moments = cv2.moments(outer_contour)
                if moments["m00"] == 0:
                    continue
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])

                # Keep the largest valid gate candidate for this frame.
                if outer_area > best_area:
                    best_area = outer_area
                    best_candidate = {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "center_x": center_x,
                        "center_y": center_y,
                        "area": outer_area,
                        "vertices": outer_vertices,
                        "approx": outer_approx,
                        "inner_approx": inner_approx,
                        "opening_ratio": opening_ratio,
                        "size_ratio": size_ratio
                    }
        # If the strict detector did not find a complete rectangle,
        # look for a large blue partial gate touching the top edge.
        if best_candidate is None:
            image_area = frame.shape[0] * frame.shape[1]

            num_labels, labels, stats, centroids = \
                cv2.connectedComponentsWithStats(clean_mask, 8)

            for i in range(1, num_labels):
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                area = stats[i, cv2.CC_STAT_AREA]

                touches_top = y <= PARTIAL_EDGE_MARGIN
                aspect_ratio = w / max(h, 1)
                # Estimate the complete gate area when its top is clipped.
                # Using only the visible height would make the ratio shrink
                # exactly when the drone is close enough to cross.
                estimated_full_height = max(h, w / PARTIAL_GATE_ASPECT)
                size_ratio = (w * estimated_full_height) / image_area
                extent = area / max(w * h, 1)

                # A clipped gate should be large, tall, blue, and touch the top edge.
                if (
                    touches_top and
                    area > 500 and
                    0.5 < aspect_ratio < 3.0 and
                    size_ratio > 0.02 and
                    extent < 0.45
                ):
                    component_mask = np.uint8(labels == i)
                    component_contours, _ = cv2.findContours(
                        component_mask,
                        cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE
                    )

                    if not component_contours:
                        continue

                    contour = max(component_contours, key=cv2.contourArea)
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)

                    # Estimate the full gate height from its known width.
                    estimated_full_height = w / PARTIAL_GATE_ASPECT

                    # The bottom edge is still visible, so estimate the missing top.
                    partial_center_x = x + w // 2
                    partial_center_y = int((y + h) - estimated_full_height / 2)

                    # Keep the estimated center inside the image.
                    partial_center_y = int(np.clip(
                        partial_center_y,
                        0,
                        frame.shape[0] - 1
                    ))

                    best_candidate = {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "center_x": partial_center_x,
                        "center_y": partial_center_y,
                        "area": area,
                        "vertices": len(approx),
                        "approx": approx,
                        "inner_approx": None,
                        "opening_ratio": 0,
                        "size_ratio": size_ratio
                    }

                    partial_gate = True
                    break
        if best_candidate is None:
            print("No gate detected", flush=True)
            # The stable-center requirement must be consecutive frames with
            # a valid gate; do not carry the count across a lost detection.
            center_hold_count = 0

        if best_candidate is not None:
            #Get the best candidate

            #Apply gate center smoothing
            raw_center_x = best_candidate["center_x"]
            raw_center_y = best_candidate["center_y"] - CENTER_Y_OFFSET_GATE

            if smoothed_center_x is None:
                smoothed_center_x = raw_center_x
                smoothed_center_y = raw_center_y
            else:
                smoothed_center_x = (
                    CENTER_SMOOTHING * raw_center_x +
                    (1 - CENTER_SMOOTHING) * smoothed_center_x
                )
                smoothed_center_y = (
                    CENTER_SMOOTHING * raw_center_y +
                    (1 - CENTER_SMOOTHING) * smoothed_center_y
                )

            center_x = int(smoothed_center_x)
            center_y = int(smoothed_center_y)
            #Calculate error and control signals
            error_x = center_x - image_center_x
            error_y = center_y - image_center_y
            in_hover_range = HOVER_Y_MIN <= error_y <= HOVER_Y_MAX
            gate_size_ratio = best_candidate["size_ratio"]
            # Checking if error y is centered within the gate
            if HOVER_Y_MIN <= error_y <= HOVER_Y_MAX:
                hover_frame_count += 1

                if hover_frame_count >= HOVER_CONFIRM_FRAMES:
                    hover_mode = True
            else:
                hover_frame_count = 0
                hover_mode = False
            # Confirm that the drone has remained at the correct height.
            if y_is_aligned:
                # Leave the hold state only if the error becomes clearly too large.
                if abs(error_y) > Y_EXIT_TOLERANCE:
                    y_is_aligned = False
                    y_hold_count = 0
            else:
                if abs(error_y) <= DEADBAND_Y:
                    y_hold_count += 1

                    if y_hold_count >= Y_HOLD_FRAMES:
                        y_is_aligned = True
                else:
                    y_hold_count = 0

            # Treat the calibrated hover range as an acceptable Y position.
            is_centered = (
                abs(error_x) < DEADBAND_X and
                (y_is_aligned or hover_mode or in_hover_range)
            )
            if is_centered:
                last_centered_time = time.monotonic()

            # Require both axes to remain acceptable together before
            # allowing the forward approach to start.
            if is_centered:
                center_hold_count += 1
            else:
                center_hold_count = 0

            if abs(error_x) < DEADBAND_X: #If the x error is within the deadband, set control signal to 0
                control_x = 0
            else:
                control_x = p_controller(error_x, KP_X)
            if abs(error_y) < DEADBAND_Y: #If the y error is within the deadband, set control signal to 0
                control_y = 0
            else:
                control_y = p_controller(error_y, KP_Y)
            control_x = np.clip(control_x, -MAX_CONTROL, MAX_CONTROL)
            control_y = np.clip(control_y, -MAX_CONTROL, MAX_CONTROL)
            # The drone is already at the acceptable height.
            # Stop vertical movement immediately.

            if in_hover_range:
                control_y = 0

            if abs(control_x) < 0.5:
                x_zero_frames += 1
            else:
                x_zero_frames = 0

            x_is_aligned = x_zero_frames >= X_ZERO_CONFIRM_FRAMES
            # The vertical center is unreliable when the top is clipped.
            if partial_gate:
                control_y = np.clip(control_y, -1, 1)  # small vertical correction while focusing on horizontal alignment

            print(f"mode={flight_mode} "
                f"size_ratio={gate_size_ratio:.3f} "
                f"partial={partial_gate} "
                f"error=({error_x:.0f}, {error_y:.0f}) "
                f"control=({control_x:.1f}, {control_y:.1f})",
                flush=True)
            #Get other best candidate information to display
            x = best_candidate["x"]
            y = best_candidate["y"]
            w = best_candidate["w"]
            h = best_candidate["h"]
            area = best_candidate["area"]
            vertices = best_candidate["vertices"]
            approx = best_candidate["approx"]
            inner_approx = best_candidate["inner_approx"]
            opening_ratio = best_candidate["opening_ratio"]
            size_ratio = best_candidate["size_ratio"]

            #Draw bounding box
            cv2.rectangle(
                frame,
                (x,y),
                (x+w, y+h),
                (0,255,0),
                2
            )
            cv2.rectangle(
                mask_display,
                (x,y),
                (x+w, y+h),
                (0,255,0),
                2
            )
            #Draw contour
            cv2.drawContours(
                frame,
                [approx],
                0,
                (255,0,0),
                2
            )
            #Draw centroid
            cv2.circle(
                frame,
                (int(center_x), int(center_y)),
                5,
                (0,0,255),
                -1
            )
            cv2.circle(
                mask_display,
                (int(center_x), int(center_y)),
                5,
                (0,0,255),
                -1
            )
            #Display image center
            cv2.circle(
                frame,
                (int(image_center_x), int(image_center_y)),
                5,
                (255, 255, 0),
                -1
            )
            #Draw line from image center to centroid
            cv2.line(
                frame,
                (int(image_center_x), int(image_center_y)),
                (int(center_x), int(center_y)),
                (255, 255, 0),
                2
            )
            #Display information
            cv2.putText(
                frame,
                f"Area: {area}",
                (x, y-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1
            )
            cv2.putText(
                mask_display,
                f"Area: {area}",
                (x, y-30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1
            )

            cv2.putText(
                frame,
                "GATE",
                (x,y-35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1
            )
            cv2.putText(
                mask_display,
                "GATE",
                (x,y-35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1
            )
            cv2.putText(
                frame,
                f"Vertices: {vertices}",
                (x, y-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1
            )
            cv2.putText(
                mask_display,
                f"Vertices: {vertices}",
                (x, y-15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,255,255),
                1
            )
        recently_centered = (
            flight_mode == "APPROACH" and
            best_candidate is not None and
            last_centered_time is not None and
            time.monotonic() - last_centered_time <= APPROACH_GRACE_SECONDS
        )

        # A close, centered gate is ready to cross.  Partial-gate mode is
        # allowed here because the gate can leave the top of the camera view
        # as the drone approaches it.
        cross_ready = (
            best_candidate is not None and
            is_centered and
            gate_size_ratio is not None and
            gate_size_ratio >= CROSS_RATIO
        )
        if cross_ready:
            cross_ready_frames += 1
        else:
            cross_ready_frames = 0

        # If the drone starts close to the gate, commit directly from ALIGN.
        if (
            flight_mode == "ALIGN" and
            center_hold_count >= CENTER_CONFIRM_FRAMES and
            cross_ready_frames >= CROSS_CONFIRM_FRAMES
        ):
            flight_mode = "CROSS"
            cross_start_time = time.monotonic()
            print(f"ALIGN -> CROSS (size_ratio={gate_size_ratio:.3f})", flush=True)

        # Start approaching once the gate is centered and still far enough away.
        if (
            flight_mode == "ALIGN" and
            center_hold_count >= CENTER_CONFIRM_FRAMES and
            gate_size_ratio is not None and
            gate_size_ratio <= APPROACH_START_RATIO
        ):
            flight_mode = "APPROACH"

        # Commit after the size/centering condition has been stable.  The
        # candidate may be partial because the gate can leave the camera view.
        if (
            flight_mode == "APPROACH" and
            cross_ready_frames >= CROSS_CONFIRM_FRAMES
        ):
            flight_mode = "CROSS"
            cross_start_time = time.monotonic()
            print(f"APPROACH -> CROSS (size_ratio={gate_size_ratio:.3f})", flush=True)

        # Switch between horizontal and vertical correction phases.
        if time.monotonic() - last_axis_switch >= AXIS_PERIOD_SECONDS:
            active_axis = "y" if active_axis == "x" else "x"
            last_axis_switch = time.monotonic()

        # Crossing ignores vision corrections and moves forward for a fixed
        # duration, because the gate may leave the camera view.
        if flight_mode == "CROSS":
            forward = CROSS_SPEED
            left_right = 0
            up_down = 0

            if time.monotonic() - cross_start_time >= CROSS_DURATION:
                flight_mode = "DONE"

        # Approach moves forward only when the gate is centered.
        elif recently_centered:
            forward = APPROACH_SPEED
            left_right = 0
            up_down = 0

        # Done mode stops all commanded motion.
        elif flight_mode == "DONE":
            forward = 0
            left_right = 0
            up_down = 0

        # Align mode or an off-center approach uses the existing X/Y controller.
        else:
            forward = 0

            if best_candidate is None:
                left_right = 0
                up_down = 0
            elif x_is_aligned:
                left_right = 0
                up_down = int(np.clip(control_y, -MAX_CONTROL, MAX_CONTROL))
            elif y_is_aligned or hover_mode:
                # Y is stable, so keep the full X correction active.
                left_right = int(control_x)
                up_down = 0
            elif active_axis == "x":
                left_right = int(control_x)
                up_down = int(np.clip(control_y * 0.25, -5, 5))
            else:
                left_right = int(np.clip(control_x * 0.25, -7, 7))
                up_down = int(control_y)

        tello.send_rc_control(left_right, forward, up_down, 0)

        #Show original
        cv2.imshow("frame", frame)
        #show hsv
        #cv2.imshow("HSV", hsv)
        #Show annotated masked image and raw mask
        #cv2.imshow("annotated mask", mask_display)
        #cv2.imshow("mask", mask)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    tello.send_rc_control(0, 0, 0, 0)
    if airborne or takeoff_requested:
        tello.land()
    tello.streamoff()
    tello.end()
    cv2.destroyAllWindows()
