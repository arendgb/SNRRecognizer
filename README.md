# SNRRecognizer #

## Introduction ## 

SNR Recognizer was created as an assignment involving 1 or modules. SNR Recognizer is a new way for reading in Student IDs and being able to use the extracted Student Number (SNR) as a way to track Attendance or give access to specific areas (such as Libary or Workplace). 

SNR Recognizer is built upon Python and uses OpenCV in combination with the K-Nearest feature to track and identify the numbers within the image provided. It has a GUI interface for easy usage where it is possible to select one or more images of Student IDs. 

# Basic Usage #
** Train the classifier **
```
python Classifier_train.py
```
A new windows will pop up. Type the corresponding digits on your keyboard. So if in the Window the the digit "9" has a contour around it, then you need to type the digit 9 on your keyboard. After following through from 9 to 0, the program mistakes D and J as a digit, with these characters be sure to press "q" on your keyboard to provide it with a non-digit.

# Run the actual SNR Recognizer GUI#
Now that we have the Data files created, we can run the Classifier itself.
The program is incoperated into a GUI to make the use easier.

```
python SNRRecognizer.py
```
A new window will pop up showing a GUI program. Please note that it currently doesn't show the labels in the GUI on MacOS, but does work in Windows and Linux. The left button let's you select one or more images, when these are selected you can run the program by clicking on the right button.

The Student IDs will be shown on the screen. We have included validation images in the folder /Resources/Images/Test-images to perform the model on these images. With these images the model is 100% accurate, it currently isn't accurate with other student id photos.

Resources used for the project:

OCR Digit Recognition with OpenCV and Python
https://stackoverflow.com/questions/9413216/simple-digit-recognition-ocr-in-opencv-python
https://medium.com/@gsari/digit-recognition-with-opencv-and-python-cbf962f7e2d0

OpenCV documentation/tutorials
https://docs.opencv.org/3.0.0/d4/d73/tutorial_py_contours_begin.html (Contours)
https://media.readthedocs.org/pdf/opencv-python-tutroals/latest/opencv-python-tutroals.pdf (page 99)

Tkinter
https://www.python-course.eu/python_tkinter.php
https://tkdocs.com/tutorial/ 
