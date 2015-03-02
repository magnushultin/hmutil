hmutil
======

Detects circular targets in video files from scene cameras of headmounted eye trackers and outputs target position as x,y.
Also has the option to track small reference images and make a simple perspective transform to limit the search area.

#Settings
Setting which tracker type the video file is of is required.
  
--tobii  
--smi_helm  
--smi_glass  
--ps_tracker

Videos are put in the /vids folder

##Optional settings
--showresult  
Shows a window with the binarization and frames bwith circle drawn.  

--record  
Saves a recording of the detection and binary image.  

--saveframes  
Saves all the frames in the video.  

--findcorners  
Takes the ref images of corners and tries to match them.  

--crop  
Makes a perspective transform and crops the image to the screencorners before finding circle. Findcorners option is needed for this to work.  

-h, --help  
Shows help message and quits.