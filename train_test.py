import cv2
import numpy as np

## Training
samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

# In OpenCV version 3 or higher, cv2.KNearest() is replaced by cv2.ml.KNearest_create() 
model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE, responses)

## Testing
im = cv2.imread('Images/Test/validate_2.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

student_id = []

for cont in contours:
    if cv2.contourArea(cont)>50:
        [x,y,w,h] = cv2.boundingRect(cont)
        if h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            string = str(int((results[0][0])))
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))
            student_id.append(string)
        
# Removing the mistaken numbers (J, D), reversing the list to the correct sequence
# Joining the list to create a string instead of list        
student_id = student_id[0:10]
student_id.reverse()
student_id = "".join(student_id)

print(student_id)
