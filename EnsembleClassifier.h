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

#ifndef ENSEMBLECLASSIFIER_H_
#define ENSEMBLECLASSIFIER_H_

#include <opencv/cv.h>

namespace tld
{

#define nt 10	// Some strange bugs happen on my Mac, have to do it this way...
#define nf 13

class EnsembleClassifier {
    const unsigned char *img;

    int calcFernFeature(int windowIdx, int treeIdx);
    void calcFeatureVector(int windowIdx, int *featureVector);
public:
    bool enabled;

    //Configurable members
    int numTrees;
    int numFeatures;

    int imgWidthStep;
    int numScales;
    cv::Size *scales;

    int *windowOffsets;
    int *featureOffsets;
    float *features;

    int numIndices;

    float *posteriors;
    int *positives;
    int *negatives;

    DetectionResult *detectionResult;

    EnsembleClassifier();
    virtual ~EnsembleClassifier();
    void init();
    void initFeatureLocations();
    void initFeatureOffsets();
    void initPosteriors();
    void release();
    void nextIteration(const cv::Mat &img);
    void classifyWindow(int windowIdx);
    void learn(int *boundary, int positive, int *featureVector);
    bool filter(int i);
private:
    float thetaFP_learn;
    float thetaTP_learn;
};

} /* namespace tld */
#endif /* ENSEMBLECLASSIFIER_H_ */
