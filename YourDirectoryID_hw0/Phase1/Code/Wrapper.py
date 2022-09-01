#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2


def main():


	#Grey Scale
	original = cv2.imread('../BSDS500/Images/1.jpg')
	print(original)


	R = original[:,:,0] #all the first values (R)
	G = original[:,:,1] #all the second values (G)
	B = original[:,:,2] #all the third values (B)

	grayScale = (0.299 * R) + (0.587 * G) + (0.114 *B)


	print(grayScale)


	#cv2.waitKey(3000)

	#cv2.filter2D

	"""
	Generate Difference of Gaussian Filter Bank: (DoG) removing any noise first
	Display all the filters in this filter bank and save image as DoG.png, (What is the DoG.png?)
	use command "cv2.imwrite(...)"


	1) Convolve image with Gaussian Filter with a certain sigma (standard deviation) 
	convultion: f an h are an image and filter convolved image is (h * f)

	d/dx(h * f) = (d/dx(h)) * f
	- > output an image where noise is removed until certain extent
	2) Compute the derivative of the image and look at it's peak then you get the edge locations




	steps
	1) create a sobel filter 3 x 3 Gx and Gy
	2) create a gaussian kernel  - At any sigma value you think is good for your problem
	3) Convolve Both (means operating sobel on Gaussian)
	4) You get a DoG at one orientation


	7 by 7 kernel add the 3 x 3 sobel filter on each of these

	subtract it 

	source: https://medium.com/jun94-devpblog/cv-3-gradient-and-laplacian-filter-difference-of-gaussians-dog-7c22e4a9d6cc
	"""


	#sobel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] #Vertical Mask 
	#sobel = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]  #Horizontal Mask
	

	sobel = np.array([
  [1, 1, 2],
  [0, 0, 0],
  [-1, -2, -1]
])

	sobel2 = np.array([
  [0, 1, 2],
  [-1, 0, 1],
  [-2, -1, 0]
])




	#cv2.rotate()
	#numpy.convolve()


	kernel_size = 7
	#sigma = 2
	#K = 1



	#7 by 7 gaussian kernel
	def Gaussian(sigma): #what's the I in the equation, why is K squared in the second part

		kernel = [[0 for x in range(kernel_size)] for x in range(kernel_size)]
	
		
		total = 0

		sig = sigma * sigma *2 

		for i in range(-3, 4): #grid is 7 by 7
			for j in range(-3, 4):
				x = i
				y = j
				distance = (x * x) + (y *y) #distance away from mu or center reference https://www.geeksforgeeks.org/gaussian-filter-generation-c/
				
				value = 1 / (sig * np.pi)

				l = i + 3

				m = j + 3
			
				kernel[l][m] =  (np.exp(-( distance)/sig)) / (np.pi * sig) 

				total += kernel[l][m]


		#for normalising
		#for i in range(0,6):
			#for j in range(0,6):
				#kernel[i][j] = kernel[i][j] / total

	


		
		return kernel





	def Convolve(k, sobel):
		k = np.array(k)

		

		resulting_image = cv2.filter2D(k, -1, sobel)
		DoG = ((resulting_image - resulting_image.min()) * (1/(resulting_image.max() - resulting_image.min()) * 255)).astype('uint8')
		
					
		print("RESULTING IMAGE")
		print(DoG)

		return DoG



		#4 90 degrees 4 Side to side





	def rotate(images, num):


		allimg = []
		DoG_img = []
		#DoG_img2 = []

		

		for image in range(len(images)):
			DoG = images[image]

			DoG = cv2.resize(DoG, dsize = (0,0),
                    fx = 1, fy = 1)


			
			for i in range(4):
				DoG = cv2.rotate(DoG, cv2.ROTATE_90_CLOCKWISE)
				DoG_img.append(DoG)
	

		
		imgs = cv2.hconcat(DoG_img)
		


		if(num == 1):
			cv2.imwrite('firstrow.jpg', imgs)
		else:
			cv2.imwrite('secondrow.jpg', imgs)

	
	Dog1 = Convolve(Gaussian(1), sobel)
	Dog2 = Convolve(Gaussian(1), sobel2)

	Dog3 = Convolve(Gaussian(2), sobel)
	Dog4 = Convolve(Gaussian(2), sobel2)

	images1 = [Dog1, Dog2]
	images2 = [Dog3, Dog4]




	rotate(images1, 1)
	rotate(images2, 2)
	frow = cv2.imread('firstrow.jpg')
	srow = cv2.imread('secondrow.jpg')

	total = cv2.vconcat([frow, srow])

	cv2.imwrite('DoG.jpg', total)




	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"

	1) 
	"""

	sigma1 = [1, (np.sqrt(2)), 2, (2*np.sqrt(2))]
	sigma2 = [(np.sqrt(2)), 2, (2*np.sqrt(2)), 4]


	def Gaussian2nd(sigma):


		kernel = [[0 for x in range(7)] for x in range(7)]


		for i in range(-3, 4): #grid is 7 by 7
			for j in range(-3, 4):
				distance = (i*i) + (j*j)
				value = (-1  + ((i*i) / (sigma * sigma))) * ((np.exp(-(distance) / (2 * (sigma * sigma)))) / (2 * np.pi * (sigma **4)))
				kernel[i][j] = value

		return kernel




	def LM(sigma):

		sobel = np.array([
		  [1, 1, 2],
		  [0, 0, 0],
		  [-1, -2, -1]
		])

		sobel2 = np.array([
		  [0, 1, 2],
		  [-1, 0, 1],
		  [-2, -1, 0]
		])

		bank = []
		#First Derivative Gaussian first 3
		LM_1 = []
		for sig in range(3):
		
			kernel = Gaussian(sigma[sig])

			image1 = Convolve(kernel, sobel)
			image2 = Convolve(kernel, sobel2)
			allimg = rotationLM(image1, image2, 1)
			lm1 = cv2.imread('LM1.jpg')
	
			LM_1.append(lm1)


	
		img = cv2.vconcat(LM_1)
		cv2.imwrite('LMFirst.jpg', img)



		#Second derivative first 3
		LM_2 = []
		for sig in range(3):
			kernel = Gaussian2nd(sigma[sig])
			image1 = Convolve(kernel, sobel)
			image2 = Convolve(kernel, sobel2)
			allimg = rotationLM(image1, image2, 2)
			lm2 = cv2.imread('LM2.jpg')

		
			LM_2.append(lm2)

		img2 = cv2.vconcat(LM_2)
		cv2.imwrite('LMSecond.jpg', img2)



		first = cv2.imread('LMFirst.jpg')
		second = cv2.imread('LMSecond.jpg')
		finalLM = cv2.hconcat([first, second])
		cv2.imwrite('LM.jpg', finalLM)









	def rotationLM(image1, image2, num):

		imgs = []
		
		for i in range(4):
			image1 = cv2.rotate(image1, cv2.ROTATE_90_CLOCKWISE)
			imgs.append(image1)


		for i in range(2):
			image2 = cv2.rotate(image2, cv2.ROTATE_90_CLOCKWISE)
			imgs.append(image2)


		allimg = cv2.hconcat(imgs)

		if(num == 1):
			cv2.imwrite('LM1.jpg', allimg)
		else:
			cv2.imwrite('LM2.jpg', allimg)


		return allimg
		




		



		#cv2.waitKey(10000)   

	LM(sigma1)


	

	


	def LaplacianGaussian(sigma):

		sobel = np.array([
		[-1, -1, -1],
		[-1,4, -1],
		[-1, -1, -1]
		])


		kernel = [[0 for x in range(7)] for x in range(7)]


		for i in range(-3, 4): #grid is 7 by 7
			for j in range(-3, 4):
				distance = (i*i) + (j*j)
				val = (-1 * (1 / (np.pi *( sigma** 4)))) * (1 - (distance / (2 * (sigma **2)))) * np.exp(-(distance/(2*(sigma**2))))
				kernel[i][j] = val

		print("LM KERNEL")
		print(kernel)


		return Convolve(kernel, sobel)



	LG = []
	for sig in range(len(sigma1)):
		fil = LaplacianGaussian(sigma1[sig])
		LG.append(fil)

	for sig in range(len(sigma1)):
		fil = LaplacianGaussian(sigma1[sig])
		LG.append(fil * 3)

	allLG = cv2.hconcat(LG)
	cv2img = cv2.imshow('LG',allLG)
	cv2.imwrite('LG.jpg', allLG)
	cv2.waitKey(10000) 



	sobelGaus = np.array([
		[-1, -1, -1],
		[-1,8, -1],
		[-1, -1, -1]
		])

	gau = []
	for sig in sigma1:
			
		kernel = Gaussian(sig)
		image = Convolve(kernel, sobelGaus)
		gau.append(image)


	gau = cv2.hconcat(gau)

	cv2.imwrite('gau.jpg', gau)

	lg = cv2.imread('LG.jpg')
	gau = cv2.imread('gau.jpg')

	full = cv2.hconcat([lg, gau])
	cv2.imwrite('hconcat.jpg', full)

	finalLM = cv2.imread('LM.jpg')
	bottom = cv2.imread('hconcat.jpg')
	final = cv2.vconcat([finalLM, bottom])
	cv2.imwrite('LM.jpg', final)







		







	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""



	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""


	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Generate Color Map
	Perform color binning or clustering
	"""


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
    
if __name__ == '__main__':
    main()
 


