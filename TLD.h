/*
 *   This file is part of OpenTLD.
 *
 *   OpenTLD is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   OpenTLD is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with OpenTLD.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TLD_H_
#define TLD_H_

#include <cv.h>

#include "MedianFlowTracker.h"
#include "DetectorCascade.h"

using namespace cv;

namespace tld
{

class TLD {
	void storeCurrentData();
	void fuseHypotheses();
	void learn();
	void initialLearning();
public:
	bool trackerEnabled;
	bool detectorEnabled;
	bool learningEnabled;
	bool alternating;

	MedianFlowTracker *medianFlowTracker;
	DetectorCascade *detectorCascade;
	NNClassifier *nnClassifier;
	bool valid;
	bool wasValid;
	Mat prevImg;
	Mat currImg;
	Rect *prevBB;
	Rect *currBB;
	float currConf;
	bool learning;

	TLD();
	virtual ~TLD();
	void release();
	void selectObject(const Mat &img, Rect *bb);
	void processImage(const Mat &img, bool showTrackResult = false);
   
private:
	float overlap_pth;
	float overlap_nth;
	float thetaFP_fuse;
	float thetaTP_fuse;
};

} /* namespace tld */
#endif /* TLD_H_ */
