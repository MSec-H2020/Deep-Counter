# Deep-Counter
# Overview
The amount of garbage increases gradually year and year, and we cannot miss looking at it because we have problems reducing CO2.
However, it is difficult to measure it currently. This reason is that the garbage is collected by a garbage truck, and it is processed immediately when through in the garbage truck by workers.  Moreover, garbage is collected by a bag, so we cannot measure its weight.
Therefore, people cannot measure the amount of garbage automatically.
  
We aimed at this research problem and developed a system to measure it automatically.
The application is the "Deep-Counter".
A Deep-Counter can count the garbage bags automatically by using a deep learning technology (YOLO algorithm).

The application image shows the following.  
<img width="246" alt="DeepCounter01" src="https://user-images.githubusercontent.com/13267712/140925742-3ed69fc1-165a-4a82-9722-e70bcd58e6f1.png">


This application detects the bag by using a camera of the garbage truck and counting it.  


## System Architecture
This application can perform everywhere.  
If you want to do this, you download this application and execute the "run" command only.  
Currently, we uploaded the weight file of the Japanese garbage bag (in Fujisawa City, Kanagawa).  
Therefore, when you want to detect the garbage bag of your city, you must train this application by using your dataset.  
<img width="756" alt="deepcounter" src="https://user-images.githubusercontent.com/13267712/140927272-6c531b5b-d65f-434d-9748-6493aa8b3ef3.png">



## develop Environment
language : Python3
using model : YOLOv5

## Detection Program.
counter.py is the counting program.  
you use this when you want to detect garbage bags.

## About learning models
train/test/eval.py is used to learn and test to predict models.  
The weight file is put on the 'weights' folder.
