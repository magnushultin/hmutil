#!/usr/bin/python
import numpy as np
import cv2
import cv2.cv as cv
import sys
import glob
import os.path

def main():
	
	# take command line arguments
	show_res, record, save_frames = arguments()
	
	# find the paths for all the videos in folder vids
	# change this to the folder with the videos in it
	paths = glob.glob("./vids/*")
	
	# go through all the video files
	for path in paths:
		print path
		cap = load_vid(path)
		file_name = os.path.basename(path)

		# write a textfile
		txt_file = open('output/data/%s.txt' %file_name, 'w')
		txt_file.write('x  y  frame  time(s)\n')
		frame_count = 0;
		total_frames = cap.get(cv.CV_CAP_PROP_FRAME_COUNT)

		# setup to make a recording of the results
		if record: 
			fourcc = cv.CV_FOURCC(*'XVID')
			w=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH ))
			h=int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT ))

			if not os.path.isdir("./output/videos/"):
				os.mkdir("./output/videos/")
			video_writer = cv2.VideoWriter("./output/videos/output_%s.avi" % file_name, fourcc, 25, (w, h))
			video_writer_b = cv2.VideoWriter("output/videos/output_%s_b.avi" % file_name, fourcc, 25, (w, h))
		
		while(1):
			# load frame
			frame, ret = get_frame(cap)
			if not ret:
				break
			# increase frame counter
			frame_count += 1
			# get time for frame in seconds
			time_s = cap.get(cv.CV_CAP_PROP_POS_MSEC)/1000
			# print progress of the movie
			sys.stdout.write("frame progress %d/%d  \r" % (frame_count,total_frames) )
			sys.stdout.flush()
			
			# Different ways of applying threshold for binarization
			# adaptation filter, works best with hough transformation
			thresh = get_binary_adapt(frame)

			# otsu threshold, a variable threshold. Works best with shape detection
			#thresh = get_binary_otsu(frame)

			# a set threshold value of 180. Works with shape detection when 
			# the environment lightning is known
			#thresh = get_binary(frame,240)

			# cleanup the binary image
			thresh = clean_binary(thresh)

			# clean binary for otsu
			#thresh = clean_binary_otsu(thresh)
			
			# HOUGH circle detection
			circles = find_circles(thresh)
			draw_circles(circles,frame)
			txt_file = write_txt(txt_file,circles,frame_count,time_s)
			
			# SHAPE detection for circle
			#centers = check_roundness(thresh)
			#mark_centers(centers,frame)
			#write_txt2(txt_file,centers,frame_count,time_s)
			
			# save the frame to a recording
			if record:
				video_writer.write(frame)
				video_writer_b.write(cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB))

			# save the frame to image
			if save_frames:
				if not os.path.isdir("./output/frames/"+file_name+"/"):
					os.mkdir("./output/frames/"+file_name+"/")
				cv2.imwrite("./output/frames/%s/frame_%d.jpg" % (file_name,frame_count), frame)
				#cv.WriteFrame(writer, image)

			# show the results in a window
			if show_res:
				show_result(frame,thresh)
			
			# if at the end of video stop
			if frame_count == total_frames:
				break
			
			k = cv2.waitKey(1) & 0xff
			if k == 27:
				break
		
		txt_file.close()
		cap.release()
		if record:
			video_writer.release()
			video_writer_b.release()
		cv2.destroyAllWindows()
	
def load_vid(path):
	cap = cv2.VideoCapture(path)
	return cap

def get_frame(cap):
	ret, frame = cap.read()
	if not ret:
		print('')
		print 'Could not load frame'
	return frame ,ret
	
def get_binary_adapt(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.GaussianBlur(gray,(7,7),0)
	thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
									cv2.THRESH_BINARY_INV,5, 2)
	return thresh
	
def get_binary(frame,treshold):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray,3)
	thresh = cv2.threshold(blur, treshold, 255, cv2.THRESH_BINARY)[1]
	return thresh

def get_binary_otsu(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray,3)
	ret2,thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	return thresh

def clean_binary(thresh):
	kernel = np.ones((3,3),np.uint8)
	height, width = thresh.shape
	if height > 300:	# destroys too much with lower resultions
		thresh = cv2.erode(thresh,kernel,iterations = 1)
		thresh = cv2.dilate(thresh,kernel,iterations = 2) 
		thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)	
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	return thresh

def clean_binary_otsu(thresh):
	kernel = np.ones((3,3),np.uint8)
	thresh = cv2.erode(thresh,kernel,iterations = 1)
	thresh = cv2.dilate(thresh,kernel,iterations = 2) 
	#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)	
	#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	return thresh

def find_circles(thresh):
	# find circle using Hough transformation
	circles = cv2.HoughCircles(thresh,cv2.cv.CV_HOUGH_GRADIENT,1,10,
								param1=100,param2=20,minRadius=3,maxRadius=20)
	return circles

def draw_circles(circles, frame):
	if not circles is None:
		circles = np.uint16(np.around(circles))
		#for i in circles[0,:]:
		i = circles[0][0] # take the first and best circle
			# draw the outer circle
		#cv2.circle(frame,(i[[0]],i[[1]]),i[[2]],(0,255,0),2)
			#draw the center of the circle
		cv2.circle(frame,(i[[0]],i[[1]]),2,(0,0,255),3)
		
def resize_check(frame):
	height, width, depth = frame.shape
	if height > 500 or width > 500:
		return True
	else:
		return False

def resize_window(frame):
	return cv2.resize(frame, (0,0), fx=0.7, fy=0.7)
	
def write_txt(txt_file,circles,frame_count,time_s):
	if not circles is None:
		circles = np.uint16(np.around(circles))
		i = circles[0][0] # take the first and best circle
		txt_file.write('%d %d %d %f\n' %(i[[0]],i[[1]],frame_count,time_s))
		return txt_file
	else:
		return txt_file

def write_txt2(txt_file,centers,frame_count,time_s):
	if not centers == None:
		if not len(centers) == 0:
			center = centers[-1] # take the last and best circle
			txt_file.write('%d %d %d %f\n' %(center[0],center[1],frame_count,time_s))
			return txt_file
	else:
		return txt_file
		
def show_result(frame,thresh):
	if resize_check(frame): # if height or width > 500
		resized_frame = resize_window(frame)
		resized_thresh = resize_window(thresh)
		cv2.imshow('Binary image',resized_thresh)
		cv2.imshow('Frame',resized_frame)
	else:
		cv2.imshow('Binary image',thresh)
		cv2.imshow('Frame',frame)
		
def check_roundness(thresh):
	binary = thresh.copy()
	contours, hierarchy = cv2.findContours(binary, cv2.cv.CV_RETR_LIST,
											cv2.cv.CV_CHAIN_APPROX_SIMPLE)
	centers = []
	radii = []
	my_max = 0.0
	for contour in contours:
		area = cv2.contourArea(contour)
		if area == 0.0:
			continue
			
		if area > 300:
			continue
		
		parameter = cv2.arcLength(contour,True)
		roundness = (4*area*(np.pi))/(np.square(parameter))
		my_max = np.maximum(my_max,roundness)
		if roundness == my_max:
			if roundness > 0.9: # round object if greater than 0.9
			 	br = cv2.boundingRect(contour)
				radii.append(br[2])
				m = cv2.moments(contour)
				center = (int(m['m10'] / m['m00']), int(m['m01'] / m['m00']))
				centers.append(center)
	return centers
	
def mark_centers(centers, frame):
	if not centers == None:
		if not len(centers) == 0:
			center = centers[-1]
			cv2.circle(frame,(center[0],center[1]),2,(0,0,255),3)	

def arguments():
	args = sys.argv[1:]
	show_result = False
	record = False
	save_frames = False
	for arg in args:
		if arg == '--showresult':
			show_result = True
		elif arg == '--record':
			record = True
		elif arg == '--saveframes':
			save_frames = True
		elif arg == '--help' or arg == '-h':
			print'Usage: [-h --help] [--showresult] [--record] [--saveframes]'
			print 'Description: Script for finding the x,y coordinates, frame and ' \
			'time for circle in videos stored in folder.'
			print 'optional arguments:'
			print '-h, --help	show this help message and quit'
			print '--showresult	shows a window with the binarization and frames' \
						'with circle drawn.'
			print '--record	saves a recording of the detection and binary image'
			print '--saveframes saves all the frames in the video'
			sys.exit(0)
		else:
			print 'error: wrong command. type --help or -h for help'
			sys.exit(1)
	return show_result, record, save_frames
	
		
if __name__ == '__main__':
	main()