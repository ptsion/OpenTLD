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

#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string>
#include <cstring>

#include <cv.h>
#include <highgui.h>

#include "TLD.h"
#include "TLDUtil.h"

using namespace std;
using namespace cv;
using namespace tld;

bool capture_flag = true;

const int showForeground = 1;
const float _threshold = 0.5;

#define PROGRAM_EXIT -1
#define SUCCESS 0

const char * dir = "02_jumping";
const char * ext = "jpg";

void getFrame( Mat &dst, int frame_no, int flags = 1 ) {
	char filename[32];
	if ( frame_no < 10 ) {
		sprintf(filename, "%s/0000%d.%s", dir, frame_no, ext);
	} else if ( frame_no < 100 ) {
		sprintf(filename, "%s/000%d.%s", dir, frame_no, ext);
	} else if ( frame_no < 1000 ) {
		sprintf(filename, "%s/00%d.%s", dir, frame_no, ext);
	} else if ( frame_no < 10000 ) {
		sprintf(filename, "%s/0%d.%s", dir, frame_no, ext);
	} else {
		sprintf(filename, "%s/%d.%s", dir, frame_no, ext);
	}
	
	dst = imread( filename, flags );
}

Rect getInitBB() {
	char filename[32];
	sprintf(filename, "%s/init.txt", dir);
	ifstream ifs;
	ifs.open(filename);
	if ( !ifs.is_open() ) {
		cout << "can't open init.txt!\n";
		exit(1);
	}
	string line;
	getline(ifs, line);
	int x1 = 0, x2 = 0, y1 = 0, y2 = 0;
	sscanf(line.c_str(), "%d,%d,%d,%d", &x1, &y1, &x2, &y2);
	Rect bb = Rect(Point(x1, y1), Point(x2, y2));
	ifs.close();
	return bb;
}

ifstream gtfile;
void initGTFile() {
	char filename[32];
	sprintf(filename, "%s/gt.txt", dir);
	gtfile.open(filename);
	if ( !gtfile.is_open() ) {
		cout << "can't open gt.txt!\n";
		exit(1);
	}
}

Rect getGTBB() {
	string line;
	getline(gtfile, line);
	float x1 = 0, x2 = 0, y1 = 0, y2 = 0;
	sscanf(line.c_str(), "%f,%f,%f,%f", &x1, &y1, &x2, &y2);
	Rect bb = Rect(
		Point((int)(x1+0.5), (int)(y1+0.5)),
		Point((int)(x2+0.5), (int)(y2+0.5))
	);
	return bb;
}

double overlapRatio(Rect &r1, Rect &r2) {
	
	Rect r3 = r1 & r2;
	
	int a1 = r1.width * r1.height;
	int a2 = r2.width * r2.height;
	int a3 = r3.width * r3.height;
	
	return (double) a3/(a1+a2-a3);
}

void closeGTFile() {
	gtfile.close();
}

ofstream resultfile;
void initResultFile() {
	resultfile.open("TLD_result.txt");
	if ( !resultfile.is_open() ) {
		cout << "can't open result file!\n";
		exit(1);
	}
}

void writeResult(double overlap) {
	resultfile << overlap << "\n";
}

void closeResultFile() {
	resultfile.close();
}
CvScalar red = CV_RGB(255, 0, 0);
CvScalar green = CV_RGB(0, 255, 0);
CvScalar yellow = CV_RGB(255, 255, 0);
CvScalar blue = CV_RGB(0, 0, 255);
CvScalar black = CV_RGB(0, 0, 0);
CvScalar white = CV_RGB(255, 255, 255);

TLD * tld_p;
bool showTrackResult = false;

#define window_name "tld demo"

const int maxFrames = 600;

int main(int argc, char **argv)
{
	namedWindow( window_name, 0 );
	
	initGTFile();
	initResultFile();
	tld_p = new TLD;
	
	Mat img, grey;
	Rect gt;
	int frame_no = 1;

	getFrame(img, frame_no);
	gt = getGTBB();
	cvtColor(img, grey, CV_BGR2GRAY);
	
	tld_p->detectorCascade->imgWidth = grey.cols;
	tld_p->detectorCascade->imgHeight = grey.rows;
	tld_p->detectorCascade->imgWidthStep = grey.step;
	Rect bb = getInitBB();
	tld_p->selectObject(grey, &bb);
	
	rectangle( img, bb, Scalar(blue), 3, 8, 0 );
	imshow( window_name, img );
	waitKey( 33 );
	
	getFrame(img, frame_no);
	capture_flag = true;
	
	while ( !img.empty() && frame_no < maxFrames ) {
		int holdon = 0;
		double tic = cvGetTickCount();

			tld_p->processImage(img, showTrackResult);

			int confident = (tld_p->currConf >= _threshold) ? 1 : 0;

			{
				char string[128];

				char learningString[10] = "";

				if(tld_p->learning)
				{
					strcpy(learningString, "I AM Learning");
				}

			
				printf("#%d, Posterior %.2f; #numwindows: %d, %s ", frame_no-1, tld_p->currConf, tld_p->detectorCascade->numWindows, learningString);
	
				if (tld_p->currBB != NULL) {
					Scalar rectangleColor = (confident) ? Scalar(blue) : Scalar(yellow);
					rectangle(img, *(tld_p->currBB), rectangleColor, 3, 8, 0);
					Rect bb = *(tld_p->currBB);
					double overlap = overlapRatio(bb, gt);
					writeResult(overlap);
				}

				switch ( tld_p->detectorCascade->nnClassifier->sample_status ) {
				case REDLIGHT:
					circle(img, Point(10, 10), 5, Scalar(red), -1);
					break;
				case GREENLIGHT:
					circle(img, Point(10, 10), 5, Scalar(green), -1);
					break;
				default: 
					break;
				}
				if(showForeground) {
					for(size_t i = 0; i < tld_p->detectorCascade->detectionResult->fgList->size(); i++)
					{
						Rect r = tld_p->detectorCascade->detectionResult->fgList->at(i);
						rectangle(img, r, Scalar(white), 1, 8, 0);
					}
				}
			}

		double toc = (cvGetTickCount() - tic) / cvGetTickFrequency();
		toc = toc / 1000;	// toc in ms
		float fps = 1000 / toc;
		printf( "fps: %5.3f\n", MAX(fps, 30) );
		imshow( window_name, img );
	
		char c;
		if ( showTrackResult ) {
			c = waitKey( 0 );
		} else {
			c = waitKey( MAX(5,(33-toc)) );
		}
        	if( c == 27 ) {
			break;
		} else if ( c == 'p' ) {
			capture_flag = !capture_flag;
		} else if ( c == 't' ) {
			showTrackResult = true;
		} else {
			showTrackResult = false;
		}
		
		if ( capture_flag  ) {
			getFrame( img, frame_no);
			gt = getGTBB();
			frame_no++;
		} else if ( capture_flag && holdon ) {
			holdon--;
		}
		
	}
	
	closeGTFile();
	closeResultFile();
	
	return EXIT_SUCCESS;
}
