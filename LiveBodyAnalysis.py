import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import argparse
import scipy.ndimage as sp
import SegmentAnalysis as SA
import glob
import os

# Ubuntu note for Faye: see all .png files in folder: find . -type f -regex ".*\.png"
# Also note: import pathlib --> pathlib.Path().absolute() gives you your current directory path

def LiveBodyAnalysisFunc():
    # Set up cmd arguments for the user
    parser = argparse.ArgumentParser(description = 'Enter absolute path and background coordinates')
    parser.add_argument('path', type = str, help = 'Absolute folder path of the head folder (ex. /mnt/c/Users/fhcle/Documents/GeorgiaTech/McGrath_Lab/Image_Annotation/Python')
    parser.add_argument('bkCoords', type = int, nargs = '+', help = 'Four ints representing coordinates of the image that are part of the background: (left_col, top_row, right_col, bottom_row)')
    parser.add_argument('output_name', type = str, help = 'Desired name of output file without .csv')
    args = parser.parse_args()
    bkCoords = args.bkCoords

    # Set up csv and headers for analysis
    o = open(args.output_name + '.csv','w')
    o.write('Folder,Name,Num,threshold,num_regions,est_len,cut_off,')
    for i in [2,9]:
        o.write('MeanInt-' + str(i) + ',avgWidth-' + str(i) + ',stdInt-' + str(i) + ',perct95-' + str(i) + ',medianInt-' + str(i) + ',sumInt-' + str(i) + ',')
    o.write('aspect_ratio,intRatio2-9,tipWidthRatio,')
    o.write('BodMeanInt,BodAvgWidth,BodStdInt,BodPerct95,BodMedianInt,BodSumInt,BKMeanInt\n')

    # Iterate through all folders (not files) in the given cmd head node directory.
    # Variable abs_directory is an absolute path.
    for abs_directory in [f.path for f in os.scandir(args.path) if f.is_dir()]:
        folder = abs_directory.split('/')[-1]

        # If there is a pycache folder in your directory (like mine), you want to skip it
        if abs_directory.endswith('__pycache__'):
            continue

        # Enter each directory
        try:
            os.chdir(abs_directory)
            print("Processing images in directory " + abs_directory)
        except:
            print("Could not access directory " + abs_directory)
            continue

        # Go through all .png files in the given directory
        img_num = 1
        for filename in glob.glob('*.png'):

            if img_num % 10 == 0:
                print("\tProcessing image #" + str(img_num))
            img_num += 1

            #####################################################################################
            # Step 1: read image
            #####################################################################################

            # Process a single image
            img = cv2.imread(filename, 0)

            # If image cannot be processed, continue to the next image in the folder
            if img is None:
                print("Could not read the image " + filename)
                continue

            # Write file name to output file only if its within the first 100 files
            name = filename.split('.')[0]
            num = ''.join([i for i in name.split('_')[-1] if i.isdigit()])
            # For the purpose of feature exploration, we only want the files that we
            # manually analyzed and which are in the ground truth data
            if int(num) > 100:
                continue
            o.write(folder + ',')
            o.write(name + ',')
            o.write(num + ',')

            # Select background region (safe region is rows 0-100, columns 0-2000)
            bk = img[bkCoords[1]:(bkCoords[1]+bkCoords[3]),bkCoords[0]:(bkCoords[0]+bkCoords[2])]

            #####################################################################################
            # Step 2: threshold image
            #####################################################################################

            # Otsu's thresholding to get the image to be binary black and white.
            #   Otsu's thresholding automatically calculates a threshold value
            #   from image histogram for a bimodal image. This works for our images
            #   because we have the worm (1) and the background (0).
            # cv2 function "threshold" uses Otsu's thresholding. If Otsu's thresholding
            # is NOT used, ret is the threshold value you entered (which would be 0 here).
            # Parameters: cv2.threshold(grayscale image, threshold, value given to pixels above
            #   threshold, threshold type)
            # Note for Otsu's thresholding, you cannot change the threshold input. ret will
            #   always be the threshold calculated and used.
            ret,th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # ret is type float representing the chosen threshold
            # th is a numpy array of 0s and 1s of your original image

            # Write threshold value to output
            o.write(str(ret) + ',')

            # Binary thresholding with an adjusted Otsu's threshold (last two numbers are block size
            # and param1)
            ret1,th1 = cv2.threshold(img, ret * 0.475, 255, cv2.THRESH_BINARY)

            # plt.subplot(2,1,1)
            # plt.imshow(img)
            # plt.title('Original Image')
            # plt.subplot(2,1,2)
            # plt.imshow(th1)
            # plt.title(str(ret1))
            # plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
            # plt.show()

            # Take Otsu's thresholded image, th2, and do dilating/morphological opening.
            # No function like matlab's imfill exists in cv2 - use scipy
            # Kernel size determines how much/little the image is dilated/opened.
            kernel_2 = np.ones((2,2), np.uint8)
            kernel_5 = np.ones((5,5), np.uint8)
            ImDia = cv2.dilate(th1, kernel_2, iterations = 1)
            ImOpen = cv2.morphologyEx(ImDia, cv2.MORPH_OPEN, kernel_5)
            filled = sp.morphology.binary_fill_holes(ImOpen)

            # optional: plot the results
            # plt.subplot(4,1,1)
            # plt.imshow(th1)
            # plt.title('Thresholded Image')
            # plt.subplot(4,1,2)
            # plt.imshow(ImDia)
            # plt.title('Dilated Image (kernel size = 2)')
            # plt.subplot(4,1,3)
            # plt.imshow(ImOpen)
            # plt.title('Image after Morphological Opening (kernel size = 5)')
            # plt.subplot(4,1,4)
            # plt.imshow(filled)
            # plt.title('filled')
            # plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
            # plt.show()

            #####################################################################################
            # Step 3: isolate worm
            #####################################################################################

            # Identify connected regions and select the largest, which we assume to be the worm.
            # labeled_array is the matrix of the image, containing integers corresponding to region
            #    0, region 1, region 2, etc.
            # num_regions is the number of regions found (excluding 0)
            labeled_array, num_regions = sp.label(filled)
            max_size = 0
            max_region = 0
            # Iterate through the regions identified by sp.label, excluding 0, the background
            for x in range(1, num_regions + 1):
                # Get the size of each region (count) and update the maximums
                count = np.count_nonzero(labeled_array == x)
                if (count > max_size):
                    max_size = count
                    max_region = x

            # Write number of regions to output
            o.write(str(num_regions) + ',')

            # Isolate the desired region by dividing each pixel by the number of the largest
            # region. If a pixel belongs to the largest region, this value will be one.
            # Generate a matrix of True and False, and turn it into 0s and 1s using .astype(int)
            justWorm = (np.divide(labeled_array, max_region) == 1).astype(int)

            # Multiply your binary mask by the original image to get just the worm body
            original = cv2.imread(filename, -1)
            ImSegGray = np.multiply(justWorm, original)

            # Plot resulting image
            # worm = np.divide(ImSegGray, 255)
            # plt.subplot(3,1,1)
            # plt.imshow(img)
            # plt.title('Original Image')
            # plt.subplot(3,1,2)
            # plt.imshow(justWorm)
            # plt.title('Worm Body Only')
            # plt.subplot(3,1,3)
            # plt.imshow(ImSegGray)
            # plt.title('Worm Body Only')
            # plt.subplots_adjust(wspace = 0.6, hspace = 0.6)
            # plt.show()

            #####################################################################################
            # Step 4: feature detection
            #####################################################################################

            # Is the image cut off? Is there a nonzero number in any of the outer rows/columns?
            cut_off = False
            row1 = ImSegGray[0]
            row2 = ImSegGray[ImSegGray.shape[0] - 1]
            for row in ImSegGray:
                if row[0] >= 1:
                    cut_off = True
                if row[ImSegGray.shape[1] - 1] >= 1:
                    cut_off = True
            for position in row1:
                if position >= 1:
                    cut_off = True
            for position in row2:
                if position >= 1:
                    cut_off = True
            border_sum = (np.sum(row1) + np.sum(row2) + np.sum(ImSegGray[0:(ImSegGray.shape[0]),0])
                + np.sum(ImSegGray[0:(ImSegGray.shape[0]),ImSegGray.shape[1] - 1]) - ImSegGray[0,0]
                - ImSegGray[0,ImSegGray.shape[1] - 1] - ImSegGray[ImSegGray.shape[0] - 1, 0]
                - ImSegGray[ImSegGray.shape[0] - 1, ImSegGray.shape[1] - 1])

            # Estimate worm length: np.nonzero() identifies the coordinates of all nonzero
            # elements
            [row,col] = np.nonzero(ImSegGray)

            # Set up min/max rows/columns
            mir = min(row)
            mar = max(row)
            mic = min(col)
            mac = max(col)

            # Estimated worm length
            est_len = mac - mic
            # Write est_len to output
            o.write(str(est_len) + ',')

            # Estimate the mean intensity of the worm and background
            bkMeanInt = np.mean(bk)
            wormMeanInt = np.mean(ImSegGray[np.nonzero(ImSegGray)])

            # H/T differentiation then re-orientation
            values = [[],[],[],[],[],[]]
            for i in range(10):
                # divide the worm body into 10 equal segments according to estimated length, est_len
                SecImage = ImSegGray[mir:mar,(mic+(i)*est_len//10):(mic+(i+1)*est_len//10)]
                [MeanInt, avgWidth, stdInt, perct95, medianInt, sumInt] = SA.stats(SecImage)
                values[0].append(MeanInt)
                values[1].append(avgWidth)
                values[2].append(stdInt)
                values[3].append(perct95)
                values[4].append(medianInt)
                values[5].append(sumInt)
                if (i == 2 or i == 9):
                    o.write(str(MeanInt) + ',' + str(avgWidth) + ',' + str(stdInt) + ',' + str(perct95) + ',' + str(medianInt) + ',' + str(sumInt) + ',')
           
            # Aspect ratio of worm
            asp_ratio = est_len / (np.max([values[1][0], values[1][-1]]))
            # Write aspect ratio to output
            o.write(str(asp_ratio) + ',')

            # Ratio of mean intensity of segment 2 vs segment 9 gives us a good indication
            # of where the nerve ring (high intensity, located toward the head of the animal)
            # is fixed in the image.
            seg2int = values[0][1]
            seg9int = values[0][8]
            # Write 2nd and 9th segment intensities to output
            o.write(str(seg9int/seg2int) + ',')

            # tipWidthRatio: Ratio
            tipWidthRatio = values[1][0]/values[1][9]
            o.write(str(tipWidthRatio) + ',')

            # Full worm body stats
            [BodMeanInt, BodAvgWidth, BodStdInt, BodPerct95, BodMedianInt, BodSumInt] = SA.stats(ImSegGray[min(row):max(row),min(col):max(col)])
            o.write(str(BodMeanInt) + ',' + str(BodAvgWidth) + ',' + str(BodStdInt) + ',')
            o.write(str(BodPerct95) + ',' + str(BodMedianInt) + ',' + str(BodSumInt) + ',')

            # Full background mean intensity
            o.write(str(SA.stats(bk)[0]) + '\n')

    o.close()

LiveBodyAnalysisFunc()