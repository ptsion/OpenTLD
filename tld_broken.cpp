/*  Copyright 2011 AIT Austrian Institute of Technology
*
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

/**
  * @author Georg Nebehay
  */

#include <iostream>
#include <stdlib.h>
#include <cv.h>
#include <highgui.h>



#include "TLD.h"
#include "TLDUtil.h"

using namespace std;
using namespace cv;
using namespace tld;
bool selectObject;
Point origin;
Rect selection;

int trackObject = 0;
//bool initialBB = true;

int _frame_cols;
int _frame_rows;

const int showForeground = 1;
const float _threshold = 0.5;

#define VIDEOSOURCE 0
#define PROGRAM_EXIT -1
#define SUCCESS 0


TLD * tld_p;



static void onMouse( int event, int x, int y, int, void* )
{
    if( selectObject )
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
		selection &= Rect(0, 0, _frame_cols, _frame_rows);
	}

    switch( event )
    {
    case CV_EVENT_LBUTTONDOWN:
        origin = Point(x,y);
        selection = Rect(x,y,0,0);
        selectObject = true;
        break;
    case CV_EVENT_LBUTTONUP:
        selectObject = false;
        if( selection.width > 0 && selection.height > 0 )
            trackObject = -1;
        break;
    }
}


#define window_name "tld demo"

int main(int argc, char **argv)
{
	//	step 1: initialize image acquisition...
	VideoCapture capture = VideoCapture( VIDEOSOURCE );
	if ( !capture.isOpened() ) {
		cout << "Video capture failed!\n";
		return PROGRAM_EXIT;
	} else {
		cout <<"Video capture successed"<<endl;
	}
	
	Mat img;
	int frame_no = 0;
	for ( int i = 0; i < 5; i++ ) {
		capture >> img;
		frame_no++;
	}
	_frame_cols = img.cols;
	_frame_rows = img.rows;
	double fx = (double)640 / img.cols;
	double fy = fx;
	resize(img, img, Size(), fx, fy);
	
	Mat grey(img.rows, img.cols, CV_8UC1);
    cvtColor(cv::Mat(img), grey, CV_BGR2GRAY);
	
	tld_p = new TLD;
	tld_p->detectorCascade->imgWidth = grey.cols;
    tld_p->detectorCascade->imgHeight = grey.rows;
    tld_p->detectorCascade->imgWidthStep = grey.step;

	//	step 2: initialize bounding box...
	
	namedWindow( window_name, 0 );
	setMouseCallback( window_name, onMouse, 0 );
    
	Rect bb;
	//bool skipProcessingOnce = false;
    //bool reuseFrameOnce = false;
	while ( !img.empty() ) {

		double tic = cvGetTickCount();


		if( selectObject && selection.width > 0 && selection.height > 0 ) {
            		Mat roi(img, selection);
            		bitwise_not(roi, roi);
        	}




		if ( trackObject ) {
			if ( trackObject == -1 ) {
				bb = selection;
				trackObject = 1;
				printf( "Starting at %d %d %d %d\n", bb.x, bb.y, bb.width, bb.height);
			
				tld_p->selectObject(grey, &bb);
				//skipProcessingOnce = true;
				//reuseFrameOnce = true;

			}
		
			//TODO
			tld_p->processImage(img);

			int confident = (tld_p->currConf >= _threshold) ? 1 : 0;

			//if ( true ) //(showOutput || saveDir != NULL)
			{
				char string[128];

				char learningString[10] = "";

				if(tld_p->learning)
				{
					strcpy(learningString, "I AM Learning");
				}

				//sprintf(string, "#%d,Posterior %.2f; fps: %.2f, #numwindows:%d, %s", imAcq->currentFrame - 1, tld->currConf, fps, tld->detectorCascade->numWindows, learningString);
				printf("#%d, Posterior %.2f; #numwindows: %d, %s ", frame_no-1, tld_p->currConf, tld_p->detectorCascade->numWindows, learningString);
				CvScalar yellow = CV_RGB(255, 255, 0);
				CvScalar blue = CV_RGB(0, 0, 255);
				CvScalar black = CV_RGB(0, 0, 0);
				CvScalar white = CV_RGB(255, 255, 255);
	
				if(tld_p->currBB != NULL)
				{
					Scalar rectangleColor = (confident) ? Scalar(blue) : Scalar(yellow);
					rectangle(img, *tld_p->currBB, rectangleColor, 3, 8, 0);
				}

				if(showForeground)
				{

					for(size_t i = 0; i < tld_p->detectorCascade->detectionResult->fgList->size(); i++)
					{
						Rect r = tld_p->detectorCascade->detectionResult->fgList->at(i);
						rectangle(img, r, Scalar(white), 1, 8, 0);
					}
				}
			}
		}

		double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();
		toc = toc / 1000;	// toc in ms
		float fps = 1000 / toc;
		printf( "fps: %5.3f\n", fps );
		imshow( window_name, img );

		char c = (char)waitKey(MAX(5, (33-toc)));
		if( c == 27 ) {
			break;
		}
		capture >> img;
		resize(img, img, Size(), fx, fy);

	}
	
	return EXIT_SUCCESS;
}
