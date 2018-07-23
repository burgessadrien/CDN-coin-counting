import numpy as np
import cv2

def run_main():
    cap = cv2.VideoCapture("/dev/video1") # check this
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    while(True):
        ret, frame = cap.read()
        roi = frame[0:1000, 0:1000]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (15, 15), 0)
        thresh = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 17, 2)

        kernel = np.ones((3, 3), np.uint8)
        dilation = cv2.dilate(thresh, kernel, iterations=1)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations=4)
        
        cont_img = closing.copy()
        _, contours, hierarchy = cv2.findContours(cont_img, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        money = 0
        amount = ""
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 4500 or area > 45000:
                continue
            if len(cnt) < 5:
                continue                         
                
            hull = cv2.convexHull(cnt)
            hullArea = cv2.contourArea(hull)            
            
            if hullArea < 20000:
                money += 0.1
            elif hullArea >= 20000 and hullArea < 24500:
                money += 0.05
            elif hullArea >= 24500 and hullArea < 35000:
                money += 0.25
            elif hullArea > 35000:
                money += 2.00    
                
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(roi, ellipse, (0,255,0), 2)

        #cv2.imshow("Grayscale", gray)
        #cv2.imshow("Gaussian Blur", gray_blur)
        #cv2.imshow("Adaptive Thresholding", thresh)
        #cv2.imshow("Morphological Dilation", dilation)
        #cv2.imshow("Morphological Closing", closing)
        
        amount = "$" + str("{:.2f}".format(money))                
        cv2.putText(roi, "Total value", (580,550), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (32,32,32), 5, cv2.LINE_AA)
        cv2.putText(roi, "of coins:", (580,610), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (32,32,32), 5, cv2.LINE_AA)
        cv2.putText(roi, amount, (760,690), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (32,32,32), 5, cv2.LINE_AA)
        cv2.imshow('Contours', roi)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("testing shit")
    run_main()
