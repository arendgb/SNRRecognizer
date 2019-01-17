import cv2
import numpy as np

# GUI module Tkinter
import tkinter as tk
from tkinter import filedialog as fd
from tkinter import *

global file_paths

def selectImages():
    "Gives the ability to select images on the Operating System"
    global file_paths
    file_paths= fd.askopenfilenames(parent=root, title='Choose a file')
    # set label to selected files len.
    label_info.config(text="Selected Images : " +str(len(file_paths)) )
    label_info.config(font=("", 22))

def getId():
    "Retrieves the Student ID by using the KNearest model and our previously generated .data files. Using the KNearest and .data files, it will create a Trained model to predict digits in an image"
    print((len(file_paths)))
    # Training
    samples = np.loadtxt('Trained-data/generalsamples.data',np.float32)
    responses = np.loadtxt('Trained-data/generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))

    # In OpenCV version 3 or higher, cv2.KNearest() is replaced by cv2.ml.KNearest_create()
    model = cv2.ml.KNearest_create()
    model.train(samples,cv2.ml.ROW_SAMPLE, responses)

    ## Testing
    # loop to the files selected to get their id's
    for i in range(0,len(file_paths)):
        im = cv2.imread(file_paths[i])
        out = np.zeros(im.shape,np.uint8)
        gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)

        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        student_id = []

        for contour in contours:
            if cv2.contourArea(contour)>50:
                [x,y,w,h] = cv2.boundingRect(contour)
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

        # Removing the mistaken numbers (J, D) by slicing the student_id, reversing the list to the correct sequence
        # Joining the list to create a string instead of list
        student_id.reverse()
        student_id = student_id[3:]
        student_id = "".join(student_id)

        # Inserting id's to textbox on GUI
        txtOutput.insert(tk.END, student_id+'\n')

        # print(student_id )


# SETTING UP GUI.
root = tk.Tk()
root.geometry("600x450")
root.title('SNRRecognizer')

select_img_btn= tk.Button(root, text="Select Images", command=selectImages)
select_img_btn.pack()
select_img_btn.place(x=10,y=400)
select_img_btn.config(font=("",16))

get_id_btn = tk.Button(root, text=" Get ID ", command=getId)
get_id_btn.pack()
get_id_btn.place(x=200,y=400)
get_id_btn.config(font=("",16))

label_info = Label(root, text="" )
label_info.pack()
label_info.place(x=150,y=300)

# Id no box
# add a frame and put a text area into it
txtFrame = Frame(root, borderwidth=1, relief="sunken")
txtOutput = Text(txtFrame, wrap = NONE, height = 10, width = 45, borderwidth=0,font=("",16))
vscroll = Scrollbar(txtFrame, orient=VERTICAL, command=txtOutput.yview)
txtOutput['yscroll'] = vscroll.set

vscroll.pack(side="right", fill="y")
txtOutput.pack(side="left", fill="both", expand=True)

txtFrame.place(x=10, y=15)

root.mainloop()


