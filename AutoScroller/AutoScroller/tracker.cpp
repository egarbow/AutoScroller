#include <iostream>
#include <math.h>

#include <windows.h>
#include <winuser.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Point centers[10];
int numCenters = 1;
int yVals[5];
int timesNotDetected = 0;

void detectPupil(vector<Vec3f> circles, Mat& eye)
{
	Vec3f pupil;

	if (circles.size() > 0)
	{
		pupil = circles[0];

		for (int i = 0; i < circles.size(); i++)
		{
			int total = 0, prev = 1000, radius = 0, boundL, boundR, boundU, boundD, chan1 = 0, chan2 = 0, chan3 = 0;
			radius = cvRound(circles[i][2]);
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			total = 0;

			boundL = center.x - (sqrt((radius * radius) / 2) / 2);
			boundR = center.x + (sqrt((radius * radius) / 2) / 2);
			boundU = center.y - (sqrt((radius * radius) / 2) / 2);
			boundD = center.y + (sqrt((radius * radius) / 2) / 2);
			/*boundL = center.x - radius;
			boundR = center.x + radius;
			boundU = center.y - radius;
			boundD = center.y + radius;*/
			for (int x = boundL; x < boundR; x++) {
				for (int y = boundU; y < boundD; y++) {

					Vec3b intensity = eye.at<Vec3b>(y, x);
					chan1 += intensity.val[0];
					chan2 += intensity.val[1];
					chan3 += intensity.val[2];
				}
			}

			total = abs(chan1 - chan2) + abs(chan1 - chan3);
			//total = chan3;
			//total = chan1 + chan2 + chan3;

			if (total < prev)
				pupil = circles[i];

			prev = total;
			//if (pupil[2] > circles[i][2])
				//pupil = circles[i][2];
		}

		if (numCenters < 11)
		{
			Point center(cvRound(pupil[0]), cvRound(pupil[1]));
			centers[numCenters - 1] = center;
			numCenters++;
		}
		else
		{
			int totalX = 0, totalY = 0;
			for (int i = 0; i < 10; i++)
			{
				totalX += centers[i].x;
				totalY += centers[i].y;
			}
			Point newcenter(cvRound(pupil[0]), cvRound(pupil[1]));
			centers[numCenters % 10] = newcenter;
			numCenters++;


			Point center(totalX / 10, totalY / 10);

			int radius = cvRound(pupil[2]);
			// draw the circle center
			circle(eye, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			// draw the circle outline
			circle(eye, center, radius, Scalar(0, 0, 255), 3, 8, 0);

			//cout << center.y  << endl;
		}

		if (numCenters > 100000)
			numCenters = 0;
	}

	else
	{
		if (timesNotDetected > 4)
		{
			cout << "scrolling" << endl;
			//mouse_event(MOUSEEVENTF_WHEEL, 0, 0, DFCS_SCROLLDOWN, 0);
			INPUT in;
			in.type = INPUT_MOUSE;
			in.mi.dx = 0;
			in.mi.dy = 0;
			in.mi.dwFlags = MOUSEEVENTF_WHEEL;
			in.mi.time = 0;
			in.mi.dwExtraInfo = 0;
			in.mi.mouseData = -1.5 * WHEEL_DELTA;
			SendInput(1, &in, sizeof(in));
			timesNotDetected = 0;
		}
		else
			timesNotDetected++;
	}

	//return pupil;
}

void detectEyes(Mat& frame, CascadeClassifier& faceCascade, CascadeClassifier& eyeCascade)
{
	Mat grayscale, face, eye;
	cvtColor(frame, grayscale, COLOR_BGR2GRAY); // convert image to grayscale
	equalizeHist(grayscale, grayscale); // enhance image contrast 

	//filling faces vector with detected faces
	vector<Rect> faces;
	faceCascade.detectMultiScale(grayscale, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(150, 150));

	if (faces.size() == 0)
		return; // no face was detected

	//croping out the first face found
	face = frame(faces[0]);

	//looking within the face frame for eyes
	vector<Rect> eyes;
	eyeCascade.detectMultiScale(face, eyes, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 15));

	if (eyes.size() != 2)
		return; //both eyes not found

	//consistantly makeing the eye frame the "left" eye
	if (eyes[0].tl().x < eyes[1].tl().x)
		eye = face(eyes[0]);
	else
		eye = face(eyes[1]);


	vector<Vec3f> circles;
	Vec3f pupil;

	cvtColor(eye, grayscale, COLOR_BGR2GRAY); // convert image to grayscale
	equalizeHist(grayscale, grayscale); // enhance image contrast 

	HoughCircles(grayscale, circles, HOUGH_GRADIENT, 1, eye.cols / 8, 250, 15, eye.rows / 8, eye.rows / 3);

	detectPupil(circles, eye);

	rectangle(frame, faces[0].tl() + eyes[0].tl(), faces[0].tl() + eyes[0].br(), Scalar(0, 255, 0), 2);
	rectangle(frame, faces[0].tl() + eyes[1].tl(), faces[0].tl() + eyes[1].br(), Scalar(0, 255, 0), 2);

	rectangle(frame, faces[0].tl(), faces[0].br(), Scalar(255, 0, 0), 2);
}

int main(int argc, const char** argv)
{
	char trackingOption = '1';
	bool optionSelected = false;

	while (!optionSelected)
	{
		cout << "Please enter 1 for hand tracking or 2 for eye tracking:\n" << endl;
		cin >> trackingOption;

		if (trackingOption == '2')
		{
			//finds default camera on device and opens it
			VideoCapture cap(0);

			//opens the capture and errors if the operation fails
			if (!cap.isOpened())
			{
				cerr << "No Webcam Detected!" << endl;
				return -1;
			}

			optionSelected = true;

			CascadeClassifier faceCascade;
			CascadeClassifier eyeCascade;
#ifdef _DEBUG
			if (!faceCascade.load("./haarcascade_frontalface_alt.xml"))
			{
				cerr << "Could not load face detector." << endl;
				return -1;
			}
			if (!eyeCascade.load("./haarcascade_eye.xml"))
			{
				cerr << "Could not load eye detector." << endl;
				return -1;
			}
#else
			if (!faceCascade.load("ImageData/haarcascade_frontalface_alt.xml"))
			{
				cerr << "Could not load face detector." << endl;
				return -1;
			}
			if (!eyeCascade.load("ImageData/haarcascade_eye.xml"))
			{
				cerr << "Could not load eye detector." << endl;
				return -1;
			}
#endif
			Mat frame;

			//Each itteration of this while loop will deal with each frame captured by camera
			while (1)
			{
				//capturing a frame

				cap >> frame;

				if (frame.empty())
					break;

				//face = detectFace(frame, faceCascade);
				//eye = detectEye(frame, face, eyeCascade);

				detectEyes(frame, faceCascade, eyeCascade);

				imshow("Webcam", frame);

				if (waitKey(30) >= 0)
					break;

			}
		}
		else if (trackingOption == '1')
		{
			optionSelected = true;

			//finds default camera on device and opens it
			VideoCapture cap(0);

			//opens the capture and errors if the operation fails
			if (!cap.isOpened())
			{
				cerr << "No Webcam Detected!" << endl;
				return -1;
			}




		}
		else
		{
			cout << "\nInvlaid argument: '" << trackingOption << "' Please try again...\n" << endl;
		}
	}
}