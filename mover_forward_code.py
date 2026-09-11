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

KP_X = 0.1
KP_Y = 0.1
DEADBAND_X = 20
DEADBAND_Y = 20
MAX_CONTROL = 20
CENTER_Y_OFFSET = 80  # Adjust this value to shift the vertical center point if needed
PARTIAL_EDGE_MARGIN = 10

# Constants for gate approach and crossing
APPROACH_START_RATIO = 0.25
CROSS_RATIO = 0.45

APPROACH_SPEED = 8
CROSS_SPEED = 5
CROSS_DURATION = 1.0

airborne = False
takeoff_requested = False

AXIS_PERIOD_SECONDS = 0.10
active_axis = "x"
last_axis_switch = time.monotonic()

flight_mode = "ALIGN"
cross_start_time = None

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
        image_center_x = frame.shape[1] / 2
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
                size_ratio = (w * h) / image_area
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

                    best_candidate = {
                        "x": x,
                        "y": y,
                        "w": w,
                        "h": h,
                        "center_x": x + w // 2,
                        "center_y": y + h // 2,
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

        if best_candidate is not None:
            #Get the best candidate
            center_x = best_candidate["center_x"]
            center_y = best_candidate["center_y"]
            #Calculate error
            error_x = center_x - image_center_x
            error_y = center_y - image_center_y
            # get the size ratio of the detected gate relative to the image size
            gate_size_ratio = best_candidate["size_ratio"]

            is_centered = (
                abs(error_x) < DEADBAND_X and
                abs(error_y) < DEADBAND_Y
            )

            # Calculate control signals based on the error and proportional gain
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

            print(f"error=({error_x:.0f}, {error_y:.0f}) "
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

        if(
            flight_mode == "ALIGN" and
            is_centered and
            gate_size_ratio is not None and
            gate_size_ratio < APPROACH_START_RATIO
        ):
            flight_mode = "APPROACH"
        # Change flight mode to CROSS if the gate is centered, not partial, and the size ratio indicates it's time to cross
        if (
            flight_mode == "APPROACH" and
            is_centered and
            not partial_gate and
            gate_size_ratio is not None and
            gate_size_ratio >= CROSS_RATIO
        ):
            flight_mode = "CROSS"
            cross_start_time = time.monotonic()

        # Hover-only diagnostic: keep vision running without letting the
        # detector command any lateral or vertical movement.
        # Switch between horizontal and vertical correction every 0.25 seconds.
        if time.monotonic() - last_axis_switch >= AXIS_PERIOD_SECONDS:
            active_axis = "y" if active_axis == "x" else "x"
            last_axis_switch = time.monotonic()

        # Cross mode: move forward for a fixed duration.
        if flight_mode == "CROSS":
            forward = CROSS_SPEED
            left_right = 0
            up_down = 0
            if time.monotonic() - cross_start_time >= CROSS_DURATION:
                flight_mode = "DONE"

        # Approach mode: move forward while correcting for lateral and vertical error.
        elif flight_mode == "APPROACH" and is_centered:
            forward = APPROACH_SPEED
            left_right = 0
            up_down = 0

        # Done mode: Stop all movement and hover in place.
        elif flight_mode == "DONE":
            forward = 0
            left_right = 0
            up_down = 0

        # Align mode or off-center: correct for lateral and vertical error without moving forward.
        else:
            forward = 0

            if best_candidate is None:
                left_right = 0
                up_down = 0
            elif active_axis == "x":
                left_right = int(np.clip(control_x, -MAX_CONTROL, MAX_CONTROL))
                up_down = int(np.clip(control_y*0.25, -1, 1))
            else:  # Reduce vertical correction to avoid oscillation
                left_right = int(np.clip(control_x*0.25, -3, 3))
                up_down = int(np.clip(control_y, -MAX_CONTROL, MAX_CONTROL))

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
