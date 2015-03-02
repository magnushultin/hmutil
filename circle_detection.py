#!/usr/bin/python
import numpy as np
import cv2
import cv2.cv as cv
import sys
import glob
import os.path
import settings

# globals for saving references when clicking main window
ix,iy = -1,-1
ex,ey = -1,-1
corners = []
draw = False
new_corner = False

def main():
	
	# take command line arguments
	show_res, record, save_frames, find_corners, crop, setting = arguments()

	# find the paths for all the videos in folder vids
	# change this to the folder with the videos in it
	paths = glob.glob("./vids/*")

	# corner images for template matching
	if find_corners:
		if setting:
			corner1 = cv2.imread("./ref/"+setting+"/corner_1.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
			corner2 = cv2.imread("./ref/"+setting+"/corner_2.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
			corner3 = cv2.imread("./ref/"+setting+"/corner_3.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
			corner4 = cv2.imread("./ref/"+setting+"/corner_4.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
		else:
			corner1 = cv2.imread("./output/ref/corner_1.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
			corner2 = cv2.imread("./output/ref/corner_2.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
			corner3 = cv2.imread("./output/ref/corner_3.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
			corner4 = cv2.imread("./output/ref/corner_4.png",cv2.CV_LOAD_IMAGE_GRAYSCALE)
	
	# go through all the video files
	for path in paths:
		print path
		cap = load_vid(path)
		file_name = os.path.basename(path)
		# write a textfile
		if not os.path.isdir("./output/"):
				os.mkdir("./output/")
		if not os.path.isdir("./output/data/"):
				os.mkdir("./output/data/")
		txt_file = open('output/data/%s.txt' %file_name, 'w')
		txt_file.write('x  y  frame  time(s) x_c1 y_c1 x_c2 y_c2 x_c3 y_c3 x_c4 y_c4\n')
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

		# setup window for viewing
		if show_res:
			cv2.namedWindow('Frame')
			cv2.namedWindow('Binary image')
			cv2.namedWindow('Transformed image')
			cv2.namedWindow('Controls',cv.CV_WINDOW_NORMAL)
			
			# setup slider for options	
			cv2.createTrackbar('Guassian kernel size','Controls',11,15,nothing)
			cv2.createTrackbar('param1','Controls',100,150,nothing)
			cv2.createTrackbar('param2','Controls',10,50,nothing)
			cv2.createTrackbar('minRadius','Controls',7,20,nothing)
			cv2.createTrackbar('maxRadius','Controls',14,50,nothing)
			cv2.createTrackbar('minArea','Controls',30,200,nothing)
			cv2.createTrackbar('maxArea','Controls',100,300,nothing)
			cv2.createTrackbar('fixTresh','Controls',170,255,nothing)
			cv2.createTrackbar('roundStat','Controls',70,90,nothing)

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

			frame_cpy = frame.copy()
			frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

			# find corners to track
			if len(corners) < 5:
				cv.SetMouseCallback('Frame',mouse_handler,None)
				save_corner(frame_bw)
				draw_corner(frame)

			if find_corners:
				height, width, depth = frame.shape
				half_width = round(width/2)
				half_height = round(height/2)
				x1,y1 = template_match(corner1,frame_bw[0:half_height,0:half_width],frame[0:half_height,0:half_width])
				x2,y2 = template_match(corner2,frame_bw[0:half_height,half_width:width],frame[0:half_height,half_width:width])
				x3,y3 = template_match(corner3,frame_bw[half_height:height,0:half_width],frame[half_height:height,0:half_width])
				x4,y4 = template_match(corner4,frame_bw[half_height:height,half_width:width],frame[half_height:height,half_width:width])
				x2=x2+half_width
				y3=y3+half_height
				x4=x4+half_width
				y4=y4+half_height
				
			if crop:
				if setting:
					# adjust to corners of template image
					x1 = x1 + settings.size_corr(setting)
					x2 = x2 - settings.size_corr(setting)
					x3 = x3 + settings.size_corr(setting)
					x4 = x4 - settings.size_corr(setting)
					y1 = y1 + settings.size_corr(setting)
					y2 = y2 + settings.size_corr(setting)
					y3 = y3 - settings.size_corr(setting)
					y4 = y4 - settings.size_corr(setting)

				src = np.array([[x1,y1],[x2,y2],[x4,y4],[x3,y3]],np.float32)
				if x4 > x2:
					if y4 > y3:
						dst = np.array([[0,0],[x4-x1,0],[x4-x1,y4-y1],[0,y4-y1]],np.float32)
						M = cv2.getPerspectiveTransform(src,dst)
						timg = cv2.warpPerspective(frame_cpy,M,(int(x4)-int(x1),int(y4)-int(y1)))
					else:
						dst = np.array([[0,0],[x4-x1,0],[x4-x1,y3-y1],[0,y3-y1]],np.float32)
						M = cv2.getPerspectiveTransform(src,dst)
						timg = cv2.warpPerspective(frame_cpy,M,(int(x4)-int(x1),int(y3)-int(y1)))
				else:
					if y4 > y3:
						dst = np.array([[0,0],[x2-x1,0],[x2-x1,y4-y1],[0,y4-y1]],np.float32)
						M = cv2.getPerspectiveTransform(src,dst)
						timg = cv2.warpPerspective(frame_cpy,M,(int(x2)-int(x1),int(y4)-int(y1)))
					else:
						dst = np.array([[0,0],[x2-x1,0],[x2-x1,y3-y1],[0,y3-y1]],np.float32)
						M = cv2.getPerspectiveTransform(src,dst)
						timg = cv2.warpPerspective(frame_cpy,M,(int(x2)-int(x1),int(y3)-int(y1)))
				cv2.imshow('Transformed image',timg)
			
			# Different ways of applying threshold for binarization
			# adaptation filter, works best with hough transformation
			if crop:
				#thresh = get_binary_adapt(timg,setting)
				# otsu threshold, a variable threshold. Works best with shape detection
				#thresh = get_binary_otsu(timg)
				if setting == 'smi_glass' or setting == 'smi_helm':
					thresh = get_binary_otsu(timg)
				elif setting == 'ps_tracker':
					thresh = get_binary(timg,setting)
					#thresh = clean_binary(thresh)
				elif setting:
					thresh = get_binary(timg,setting)
				else:
					#thresh = get_binary_otsu(timg)
					thresh = get_binary(timg,cv2.getTrackbarPos('fixTresh','Controls'))
			else:
				#thresh = get_binary_adapt(frame_cpy)
				# otsu threshold, a variable threshold. Works best with shape detection
				#thresh = get_binary_otsu(frame_cpy)
				if setting == 'smi_glass' or setting == 'smi_helm':
					thresh = get_binary_otsu(frame_cpy)
				elif setting:
					thresh = get_binary(frame_cpy,setting)
				else:
					#thresh = get_binary_otsu(frame_cpy)
					thresh = get_binary(frame_cpy,None)


			# a set threshold value of 180. Works with shape detection when 
			# the environment lightning is known
			# thresh = get_binary(frame,240)

			# cleanup the binary image
			#thresh = clean_binary(thresh)

			# clean binary for otsu
			#thresh = clean_binary_otsu(thresh)
				
			# HOUGH circle detection
			#circle = find_circles(thresh,setting)

			# SHAPE detection for circle
			circles = check_roundness(thresh,setting)

			# invert the transformed coordinate of circle
			if crop:
				circles = inv_coord(M,circles)
			
			# draw the circle center on frame
			draw_circles(circles,frame)
			
			# write circle center and corner coordinates to txt file
			if find_corners:
				txt_file = write_txt(txt_file,circles,frame_count,time_s,x1,y1,x2,y2,x3,y3,x4,y4)
			else:
				txt_file = write_txt(txt_file,circles,frame_count,time_s,None,None,None,None,None,None,None,None)
				
			#mark_centers(centers,frame)
			#write_txt2(txt_file,centers,frame_count,time_s)

			# save the frame to a recording
			if record:
				video_writer.write(frame)
				video_writer_b.write(cv2.cvtColor(thresh,cv2.COLOR_GRAY2RGB))

			# save the frame to image
			if save_frames:
				if not os.path.isdir("./output/frames/"):
					os.mkdir("./output/frames/")
				if not os.path.isdir("./output/frames/"+file_name+"/color/"):
					os.mkdir("./output/frames/"+file_name)
					os.mkdir("./output/frames/"+file_name+"/color/")
				if not os.path.isdir("./output/frames/"+file_name+"/binary/"):
					os.mkdir("./output/frames/"+file_name+"/binary/")
				cv2.imwrite("./output/frames/%s/color/frame_%d.jpg" % (file_name,frame_count), frame)
				cv2.imwrite("./output/frames/%s/binary/frame_%d.jpg" % (file_name,frame_count), thresh)
			
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
	
def get_binary_adapt(frame,setting):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#kernel_size = cv2.getTrackbarPos('Guassian kernel size','Controls')
	kernel_size = settings.gauss(setting)
	# should make sure that kernel size is odd and positive
	if (kernel_size%2) < 1:
		kernel_size += 1
 	blur = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
	thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
									cv2.THRESH_BINARY_INV,5, 2)
	return thresh
	
def get_binary(frame,setting):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	if setting == 'ps_tracker':
		kernel_size = settings.gauss(setting)
		blur = cv2.GaussianBlur(gray,(kernel_size,kernel_size),0)
	else:
		blur = cv2.medianBlur(gray,3)
	thresh = cv2.threshold(blur, settings.fix_treshold(setting), 255, cv2.THRESH_BINARY)[1]
	return thresh

def get_binary_otsu(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	blur = cv2.medianBlur(gray,3)
	ret2,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	#thresh = clean_binary(thresh)
	return thresh

def clean_binary(thresh):
	kernel = np.ones((3,3),np.uint8)
	#kernel = np.ones((3,3),np.uint8)
	height, width = thresh.shape
	#if height > 300:	# destroys too much with lower resultions
	thresh = cv2.erode(thresh,kernel,iterations = 0) #1
	thresh = cv2.dilate(thresh,kernel,iterations = 0) 
		#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)	
	#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	return thresh

def clean_binary_otsu(thresh):
	kernel = np.ones((3,3),np.uint8)
	thresh = cv2.erode(thresh,kernel,iterations = 1)
	thresh = cv2.dilate(thresh,kernel,iterations = 2) 
	#thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)	
	#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
	return thresh

def find_circles(thresh, setting):
	# find circle using Hough transformation
	if setting:
		parameter1 = settings.param1(setting)
		parameter2 = settings.param2(setting)
		min_rad = settings.min_radius(setting)
		max_rad = settings.max_radius(setting)
	else:
		parameter1 = cv2.getTrackbarPos('param1','Controls')
		parameter2 = cv2.getTrackbarPos('param2','Controls')
		min_rad = cv2.getTrackbarPos('minRadius','Controls')
		max_rad = cv2.getTrackbarPos('maxRadius','Controls')

	circles = cv2.HoughCircles(thresh,cv2.cv.CV_HOUGH_GRADIENT,1,10,
								param1=parameter1,param2=parameter2,minRadius=min_rad,maxRadius=max_rad)
	if not circles is None:
		circle = circles[0][0]
		return circle
	else:
		return None

def draw_circles(circles, frame):
	if not len(circles) == 0:
		for circle in circles:
			circle = np.uint16(np.around(circle))
			#for i in circles[0,:]:
			i = circle # take the first and best circle
				# draw the outer circle
			#cv2.circle(frame,(i[[0]],i[[1]]),i[[2]],(0,255,0),2)
				# draw the center of the circle
			cv2.circle(frame,(i[[0]],i[[1]]),2,(0,0,255),3)

def inv_coord(M,circles):
	if len(circles) == 0:
		return circles
	n_circles = []
	for circle in circles:
		Minv = np.linalg.inv(M)
		#i = circles[0][0]
		x = circle[0]
		y = circle[1]
		#
		#(M11x + M12y + M13) / (M31x + M32y + M33) , M21x + M22y + M23 / M31x + M32y + M33
		#
		xy = [((Minv[0][0]*x)+(Minv[0][1]*y)+(Minv[0][2]))/((Minv[2][0]*x)+(Minv[2][1]*y)+Minv[2][2]),((Minv[1][0]*x)+(Minv[1][1]*y)+Minv[1][2])/((Minv[2][0]*x)+(Minv[2][1]*y)+Minv[2][2]), 1]
		circle = xy
		n_circles.append(circle)
	return n_circles
		
def resize_check(frame):
	height, width, depth = frame.shape
	if height > 500 or width > 500:
		return False # true
	else:
		return False

def resize_window(frame):
	return cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
	
def write_txt(txt_file,circles,frame_count,time_s,x1,y1,x2,y2,x3,y3,x4,y4):
	if not len(circles) == 0:
		for circle in circles:
			circle = np.uint16(np.around(circle))
			i = circle
			if not x1 is None:
				txt_file.write('%d %d %d %f %d %d %d %d %d %d %d %d\n' %(i[[0]],i[[1]],frame_count,time_s,x1,y1,x2,y2,x3,y3,x4,y4))
			else:
				txt_file.write('%d %d %d %f\n' %(i[[0]],i[[1]],frame_count,time_s))
		return txt_file
	else:
		if not x1 is None:
				txt_file.write('nan nan %d %f %d %d %d %d %d %d %d %d\n' %(frame_count,time_s,x1,y1,x2,y2,x3,y3,x4,y4))
		else:
				txt_file.write('nan nan %d %f\n' %(frame_count,time_s))
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
	#if resize_check(frame): # if height or width > 500
	#resized_frame = resize_window(frame)
	#resized_thresh = resize_window(thresh)
	#cv2.imshow('Binary image',resized_thresh)
	#cv2.imshow('Frame',resized_frame)
	#else:
	cv2.imshow('Binary image',thresh)
	cv2.imshow('Frame',frame)
		
def check_roundness(thresh,setting):
	binary = thresh.copy()
	contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
											cv2.cv.CV_CHAIN_APPROX_SIMPLE) #cv2.cv.CV_RETR_TREE
	#cv2.cv.CV_RETR_LIST
	centers = []
	radii = []
	my_max = 0.0
	if setting:
		min_area = settings.min_area(setting)
		max_area = settings.max_area(setting)
	else:
		min_area = cv2.getTrackbarPos('minArea','Controls')
		max_area = cv2.getTrackbarPos('maxArea','Controls')

	for contour in contours:
		#cv2.drawContours(thresh, [contour],-1,(255,255,255),thickness=cv.CV_FILLED)#,maxLevel=2
		area = cv2.contourArea(contour)
		# cv2.drawContours(thresh, contours,-1,(255,255,255),thickness=cv.CV_FILLED,maxLevel=1)#,maxLevel=2
		if area == 0.0:
			continue
			
		if area > max_area:
			continue

		if area < min_area:
			continue

		#approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
		parameter = cv2.arcLength(contour,True)
		roundness = (4*area*(np.pi))/(np.square(parameter))
		#my_max = np.maximum(my_max,roundness)
		#if roundness == my_max:
		if setting:
			roundness_circle = settings.roundness(setting)
		else:
			roundness_circle = cv2.getTrackbarPos('roundStat','Controls')/100.00

		if roundness > roundness_circle: # round object if greater than 0.9
			br = cv2.boundingRect(contour)
			radii.append(br[2])
			m = cv2.moments(contour)
			center = (int(m['m10'] / m['m00']), (int(m['m01'] / m['m00'])))
			centers.append(center)
	#if not len(centers) == 0:
	#	return centers[-1]
	#else:
	#	return None
	return centers
	
def mark_centers(centers, frame):
	if not centers == None:
		if not len(centers) == 0:
			center = centers[-1]
			cv2.circle(frame,(center[0],center[1]),2,(0,0,255),3)


def mouse_handler(event,x,y,flags,param):
	global ix,iy,ex,ey
	global corners
	global draw
	global new_corner

	if event == cv2.EVENT_LBUTTONDOWN:
		draw = True
		ix,iy = x-16,y-16
		ex,ey = x+16,y+16
	if event == cv2.EVENT_LBUTTONUP:
		corners.append((ix,iy,ex,ey))
		new_corner = True
		draw = False

def save_corner(frame):
	global new_corner
	if new_corner:
		if not os.path.isdir("./output/ref/"):
			os.mkdir("./output/ref/")
		cv2.imwrite('./output/ref/corner_%d.png' %(len(corners)),frame[iy:ey,ix:ex])
		new_corner = False

def draw_corner(frame):
	if draw:
		cv2.rectangle(frame,(ix,iy),(ex,ey),(0,255,0),1)

def template_match(corner,bw,frame):
	rows,cols = corner.shape[:2]
	result = cv2.matchTemplate(bw,corner,cv2.TM_CCOEFF_NORMED) # TM_CCOR_NORMED
	cv2.normalize(result,result,0,255,cv2.NORM_MINMAX)
	(min_val,max_val,(min_x,min_y),(max_x,max_y)) = cv2.minMaxLoc(result)
	cv2.rectangle(frame,(max_x,max_y),(max_x+cols,max_y+rows),(0,255,0),2)
	return round(max_x+cols/2),round(max_y+rows/2)

def arguments():
	args = sys.argv[1:]
	show_result = False
	record = False
	save_frames = False
	find_corners = False
	crop = False
	setting = None
	for arg in args:
		if arg == '--tobii':
			setting = 'tobii'
		elif arg == '--smi_helm':
			setting = 'smi_helm'
		elif arg == '--smi_glass':
			setting = 'smi_glass'
		elif arg == '--asl':
			setting = 'asl'
		elif arg == '--ps_tracker':
			setting = 'ps_tracker'
		elif arg == '--showresult':
			show_result = True
		elif arg == '--record':
			record = True
		elif arg == '--saveframes':
			save_frames = True
		elif arg == '--findcorners':
			find_corners = True
		elif arg == '--crop':
			if find_corners == True:
				crop = True
			else:
				print 'error: wrong command. Need to set findcorners to true before crop.' \
						' Type --help or -h for help.'
				sys.exit(1)
		elif arg == '--help' or arg == '-h':
			print'Usage: [-h --help] [--tobii --smi_helm --smi_glass --asl --ps_tracker] [--showresult] [--record] [--saveframes] [--findcorners] [--crop]'
			print 'Description: Script for finding the x,y coordinates, frame and ' \
			'time for circle in videos stored in folder.'
			print 'required argument:'
			print '--tobii, --smi_helm, --smi_glass, --asl, --ps_tracker settings for different trackers'
			print 'optional arguments:'
			print '-h, --help	show this help message and quit'
			print '--showresult	shows a window with the binarization and frames' \
						'with circle drawn.'
			print '--record	saves a recording of the detection and binary image'
			print '--saveframes saves all the frames in the video'
			print '--findcorners takes the ref images of corners and tries to match them'
			print '--crop crops the image to the screencorners before finding circle. findcorners option is needed.'
			sys.exit(0)
		else:
			print 'error: wrong command. type --help or -h for help'
			sys.exit(1)
	return show_result, record, save_frames, find_corners, crop, setting

def nothing(x):
	pass

		
if __name__ == '__main__':
	main()