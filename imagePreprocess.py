import cv2
import glob
import argparse

def resizeImage(imagePath):
	
	img = cv2.imread(imagePath)
	print("original : ",img.shape)
	#if img.shape[0] <= 960 and img.shape[1] <= 960:
	#	return
	
	standard = min(960/img.shape[0], 960/img.shape[1])
	resizedImage = cv2.resize(img, dsize=(0,0), fx = standard, fy = standard, interpolation=cv2.INTER_CUBIC)
	cv2.imwrite(imagePath, resizedImage)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description = 'path')
	parser.add_argument("--img")
	parser.add_argument("--dir")
	args = parser.parse_args()
	if args.img:
		resizeImage(args.img)
	elif args.dir:
		DIR_PATH = args.dir+"/*"
		fileList = glob.glob(DIR_PATH)
		for filePath in fileList:
			resizeImage(filePath)
		print("complete")
	else:
		print("exit")
