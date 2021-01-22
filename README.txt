1. Feature Extraction: LiveBodyAnalysis.py and SegmentAnalysis.py: LiveBodyAnalysis.py User Inputs
	a. Absolute path of the head folder
	   This assumes that you have one folder (ex. "My_Data") containing one folder
	   for each strain you are analyzing.
	   My_Data
	      StrainA
	      StrainB
	      ...
	b. Background coordinates
	   These coordinates will be the same for each image analyzed. Assume a specific
	   part of the image will always be background because the worm is designed
	   to be in the middle of the image. Suggested background coordinates for an image
	   of shape (512, 2048) is 0 0 2000 100.
	c. Desired output file name

	Output: csv file with all of extracted features and their associated folder and image names

	Example call: python3 LiveBodyAnalysis.py /mnt/c/Users/Documents/My_Data 0 0 2000 100 output
	For me: python3 LiveBodyAnalysis.py /mnt/c/Users/fhcle/Documents/GeorgiaTech/McGrath_Lab/Image_Annotation/Prefiltering_Manual 0 0 2000 100 extracted_features

2. Feature Exploration: FeatureExploration.py User Inputs
	a. Name of file produced by LiveBodyAnalysis.py, which must be located in same folder
	b. Name of ground truth file data, which must be located in same folder
	   File should have one sheet only and a column named 'Name' which denotes the name
	   of the image file exactly as it appears in your folder directory without the .png
	   or .jpg.
	   Name    Feature1    Feature 2    ...
	   Img1    1	       1            ...
	   Img2    0	       1            ...
	   Img3    0	       0            ...
	   ...

	Outputs: -Merged FeatureExtraction.py data and ground truth data *TRAINING*
		 -Two files for later testing *TESTING*
		 	i. Extracted features data
			ii. Ground truth data
	         -Correlation values excel file
		 -PCA values excel file

	Example call: python3 FeatureExploration.py output.csv gtruth.xlsx

3. Feature Selection: FeatureSelection.py User Inputs
	a. Name of merged file produced by FeatureExploration.py output #1, which must be located in same folder
	b. Name of testing extracted features data from FeatureExploration.py output #2i

	Outputs: -Different feature selection methods output excel files
		 -Predictions from the classifiers for input (b)

	Example call: python3 FeatureSelection.py output.xlsx test_extracted_features.xlsx
	
4. Model Accuracy: ModelAccuracy.py User Inputs
	a. Name of testing ground truth data from FeatureExploration.py output #2ii
	b. Name of file with predicted features from FeatureSelection.py output #2
	
	Outputs: Prints the accuracies of the classifiers from FeatureSelection.py

	Example call: python3 ModelAccuracy.py  predictions.xlsx test_gtruth.xlsx
	
	

Python modules to be installed: argparse, cv2, datetime, glob, matplotlib.pyplot, mlxtend, numpy, os,
pandas, scipy, sklearn, XlsxWriter