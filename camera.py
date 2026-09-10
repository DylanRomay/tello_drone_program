import cv2
import numpy as np

#Probar con 1 para el dron
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise RuntimeError("Could not open camera 0")

def p_controller(error, kp):
    return kp * error

KP_X = 0.2
KP_Y = 0.2
DEADBAND_X = 20
DEADBAND_Y = 20

while True:
    ret, frame = cam.read()
    if not ret:
        print("Could not read a frame from camera", flush=True)
        break
    #Reset for the frame
    best_candidate = None
    best_area = 0
    #Default no correction
    control_x = 0
    control_y = 0
    #Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Define color range for blue
    lower = np.array([100,80,70])
    upper = np.array([120,255,255])
    #Thresholding
    mask = cv2.inRange(hsv,lower,upper)
    mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    #Sacar centroides del imagen
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    #Get image center
    image_center_x = frame.shape[1] / 2
    image_center_y = frame.shape[0] / 2
    # Analyze objects
    for i in range(1, num_labels):
        #Bounding box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        #Area
        area = stats[i, cv2.CC_STAT_AREA]
        #Centroid
        center_x, center_y = centroids[i]
        #Avoid division by 0
        if h > 0:
            #shape descriptor
            aspect_ratio = w/h
            #Bounding box area
            bounding_area = w*h
            #Extent
            extent = area/bounding_area
            #Calculate image area
            image_area = frame.shape[0]*frame.shape[1]
            #Gate size relative to image size
            size_ratio = bounding_area/image_area
            # Object filtering
            if area > 500 and 0.5 < aspect_ratio < 3.0 and size_ratio > 0.02:

                #Crear mask for specific object
                component_mask = np.uint8(labels == i)

                #Find contours
                contours, _ = cv2.findContours(
                    component_mask,
                    cv2.RETR_EXTERNAL,
                    cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    contour = contours[0]

                    #Approximate contour
                    epsilon = 0.02 * cv2.arcLength(
                        contour,
                        True
                    )

                    approx = cv2.approxPolyDP(
                        contour,
                        epsilon,
                        True
                    )

                    #Number of vertices
                    vertices = len(approx)

                    #Gate candidate
                    if vertices ==4 and area > best_area:
                        best_area = area
                        best_candidate = {
                            "x": x,
                            "y": y,
                            "w": w,
                            "h": h,
                            "center_x": center_x,
                            "center_y": center_y,
                            "area": area,
                            "vertices": vertices,
                            "approx": approx
                        }
    if best_candidate is not None:
        #Get the best candidate
        center_x = best_candidate["center_x"]
        center_y = best_candidate["center_y"]
        #Calculate error and control signals
        error_x = center_x - image_center_x
        error_y = center_y - image_center_y
        if abs(error_x) < DEADBAND_X: #If the x error is within the deadband, set control signal to 0
            control_x = 0
        else:
            control_x = p_controller(error_x, KP_X)
        if abs(error_y) < DEADBAND_Y: #If the y error is within the deadband, set control signal to 0
            control_y = 0
        else:
            control_y = p_controller(error_y, KP_Y)
        control_x = np.clip(control_x, -100, 100)
        control_y = np.clip(control_y, -100, 100)
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
    # Show original
    cv2.imshow("frame", frame)
    #show hsv
    cv2.imshow("HSV", hsv)
    #Show annotated masked image and raw mask
    cv2.imshow("annotated mask", mask_display)
    cv2.imshow("mask", mask)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
