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

#ifndef NNCLASSIFIER_H_
#define NNCLASSIFIER_H_

#include <vector>

#include <cv.h>

#include "NormalizedPatch.h"
#include "DetectionResult.h"

#define REDLIGHT 0	// REDLIGHT means P/N samples' size is not saturated
//#define YELLOLIGHT 1
#define GREENLIGHT 2	// GREENLIGHT means P/N samples' size is already saturated

namespace tld
{

class NNClassifier
{
	float ncc(float *f1, float *f2);
public:
	bool enabled;

	int *windows;
	float thetaFP_learn;
	float thetaTP_learn;
	DetectionResult *detectionResult;
	std::vector<NormalizedPatch>* falsePositives;
	std::vector<NormalizedPatch>* truePositives;

	NNClassifier();
	virtual ~NNClassifier();

	void release();
	float classifyPatch(NormalizedPatch *patch);
	float classifyBB(const cv::Mat &img, cv::Rect *bb);
	float classifyWindow(const cv::Mat &img, int windowIdx);
	void learn(std::vector<NormalizedPatch> patches);
	bool filter(const cv::Mat &img, int windowIdx);
	void deletePositives(std::vector<NormalizedPatch>* obj, std::vector<NormalizedPatch>* cmp, int pos); 

	int sample_status;

private:
	int maxTP;
	int maxFP;
	std::vector<std::vector <float>*>* corr_table;
};

} /* namespace tld */
#endif /* NNCLASSIFIER_H_ */
