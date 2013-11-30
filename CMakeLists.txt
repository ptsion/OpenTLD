cmake_minimum_required(VERSION 2.8)
project( OpenTLD )
find_package( OpenCV REQUIRED )
add_executable( OpenTLD 
	BB.cpp
	BBPredict.cpp
	blob.cpp
	BlobContour.cpp
	BlobOperators.cpp
	BlobProperties.cpp
	BlobResult.cpp
	Clustering.cpp
	ComponentLabeling.cpp
	DetectionResult.cpp
	DetectorCascade.cpp
	EnsembleClassifier.cpp
	FBTrack.cpp
	ForegroundDetector.cpp
	Lk.cpp
	Median.cpp
	MedianFlowTracker.cpp
	NNClassifier.cpp
	tld_broken.cpp
	TLD.cpp
	TLDUtil.cpp
	Trajectory.cpp
	VarianceFilter.cpp )
target_link_libraries( OpenTLD ${OpenCV_LIBS} )
