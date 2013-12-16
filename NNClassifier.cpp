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

#include "NNClassifier.h"

#include "DetectorCascade.h"
#include "TLDUtil.h"

using namespace std;
using namespace cv;

namespace tld
{

NNClassifier::NNClassifier() {
	thetaFP_learn = 0.5;
	thetaTP_learn = 0.65;

	truePositives = new vector<NormalizedPatch>();
	falsePositives = new vector<NormalizedPatch>();

	maxTP = 100; // not a critical choice
	maxFP = 300;

	sample_status = REDLIGHT;

}

NNClassifier::~NNClassifier() {
    release();

    delete truePositives;
    delete falsePositives;
}

void NNClassifier::release()
{
    falsePositives->clear();
    truePositives->clear();
}

float NNClassifier::ncc(float *f1, float *f2)
{
    double corr = 0;
    double norm1 = 0;
    double norm2 = 0;

    int size = TLD_PATCH_SIZE * TLD_PATCH_SIZE;

    for(int i = 0; i < size; i++)
    {
        corr += f1[i] * f2[i];
        norm1 += f1[i] * f1[i];
        norm2 += f2[i] * f2[i];
    }

    // normalization to <0,1>

    return (corr / sqrt(norm1 * norm2) + 1) / 2.0;
}

float NNClassifier::classifyPatch(NormalizedPatch *patch)
{

    if(truePositives->empty())
    {
        return 0;
    }

    if(falsePositives->empty())
    {
        return 1;
    }

    float ccorr_max_p = 0;

    //Compare patch to positive patches
    for(size_t i = 0; i < truePositives->size(); i++)
    {
        float ccorr = ncc(truePositives->at(i).values, patch->values);

        if(ccorr > ccorr_max_p)
        {
            ccorr_max_p = ccorr;
        }
    }

    float ccorr_max_n = 0;

    //Compare patch to positive patches
    for(size_t i = 0; i < falsePositives->size(); i++)
    {
        float ccorr = ncc(falsePositives->at(i).values, patch->values);

        if(ccorr > ccorr_max_n)
        {
            ccorr_max_n = ccorr;
        }
    }

    float dN = 1 - ccorr_max_n;
    float dP = 1 - ccorr_max_p;

    float distance = dN / (dN + dP);
    //cout << "...distance to negative: " << dN << "\n";
    //cout << "...distance to positive: " << dP << "\n";
    return distance;
}

float NNClassifier::classifyBB(const Mat &img, Rect *bb)
{
    NormalizedPatch patch;

    tldExtractNormalizedPatchRect(img, bb, patch.values);
    return classifyPatch(&patch);

}

float NNClassifier::classifyWindow(const Mat &img, int windowIdx)
{
    NormalizedPatch patch;

    int *bbox = &windows[TLD_WINDOW_SIZE * windowIdx];
    tldExtractNormalizedPatchBB(img, bbox, patch.values);

    return classifyPatch(&patch);
}

bool NNClassifier::filter(const Mat &img, int windowIdx)
{
    if(!enabled) return true;

    float conf = classifyWindow(img, windowIdx);

    if(conf < thetaTP_learn) { // it seems 0.5 is ok
        return false;
    }

    return true;
}

void NNClassifier::deletePositives(vector<NormalizedPatch>* obj, vector<NormalizedPatch>* cmp, int pos) {
	if ( pos == 0 ) {
		cout << "...deleting false positives...\n";
	}
	float minDist = 1;
	vector<NormalizedPatch>::iterator victim;
	int i = 0;
	for ( vector<NormalizedPatch>::iterator it_obj = obj->begin(); it_obj != obj->end(); it_obj++ ) {
		if ( i >= pos ) {
			NormalizedPatch patch_obj = (*it_obj);
			for (vector<NormalizedPatch>::iterator it_cmp = cmp->begin(); it_cmp != cmp->end(); it_cmp++ ) {
				NormalizedPatch patch_cmp = (*it_cmp);
				float corr = ncc(patch_obj.values, patch_cmp.values);
				if ( 1-corr < minDist ) {
					minDist = 1-corr;
					victim = it_obj;
				}
			}
		}
		i++;
	}
	obj->erase(victim);
}

void NNClassifier::learn(vector<NormalizedPatch> patches) {
	cout<<"...NN Classifier learning with " << patches.size() << " patches...\n";
	//TODO: Randomization might be a good idea here
	for(size_t i = 0; i < patches.size(); i++) {
	
		NormalizedPatch patch = patches[i];
		float conf = classifyPatch(&patch);
		if(patch.positive && conf < thetaTP_learn) {
			if ( truePositives->size() < maxTP ) {
				truePositives->push_back(patch);
			}
		}

		if(!patch.positive && conf > thetaFP_learn) {	
			if ( falsePositives->size() < maxFP ) {
				falsePositives->push_back(patch);
			}
			
			if ( falsePositives->size() > maxFP ) {
				deletePositives( falsePositives, truePositives, 0 );
				sample_status = GREENLIGHT;
			}
			
			if ( falsePositives->size() > maxFP ) {
				falsePositives->erase(falsePositives->begin());
				sample_status = GREENLIGHT;
			}
		}
	}
	
	if ( truePositives->size() > maxTP ) {
		deletePositives( truePositives, falsePositives, maxTP/3 );
		sample_status = GREENLIGHT;
	}
	
	

	cout << "...true positive size: " << truePositives->size() << "\n";
	cout << "...false positive size: " << falsePositives->size() << "\n";
}


} /* namespace tld */
