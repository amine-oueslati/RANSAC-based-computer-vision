
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <opencv2\imgproc.hpp>

#include "MatrixReaderWriter.h"

using namespace cv;
using std::cout;
using std::endl;

#define SQRT2 1.41

struct normalizedData {
	Mat T1;
	Mat T2;
	vector<Point2f> newPts1;
	vector<Point2f> newPts2;
};


struct normalizedData NormalizeData(vector<pair<Point2f, Point2f>> pointPairs) {
	int ptsNum = pointPairs.size();

	//calculate means (they will be the center of coordinate systems)
	float mean1x = 0.0, mean1y = 0.0, mean2x = 0.0, mean2y = 0.0;
	for (int i = 0; i < ptsNum; i++) {
		mean1x += pointPairs[i].first.x;
		mean1y += pointPairs[i].first.y;

		mean2x += pointPairs[i].second.x;
		mean2y += pointPairs[i].second.y;
	}
	mean1x /= ptsNum;
	mean1y /= ptsNum;
	mean2x /= ptsNum;
	mean2y /= ptsNum;

	float spread1x = 0.0, spread1y = 0.0, spread2x = 0.0, spread2y = 0.0, spread2z = 0.0;

	for (int i = 0; i < ptsNum; i++) {
		//compute center of gravity for 2D and 3D
		spread1x += (pointPairs[i].first.x - mean1x) * (pointPairs[i].first.x - mean1x);
		spread1y += (pointPairs[i].first.y - mean1y) * (pointPairs[i].first.y - mean1y);
		spread2x += (pointPairs[i].second.x - mean2x) * (pointPairs[i].second.x - mean1x);
		spread2y += (pointPairs[i].second.y - mean2y) * (pointPairs[i].second.y - mean2y);
	}

	spread1x /= ptsNum;
	spread1y /= ptsNum;
	spread2x /= ptsNum;
	spread2y /= ptsNum;

	//Let'scompose affine transoformation
	//original moved to the center of gravity by translation, then sa
	Mat offs1 = Mat::eye(3, 3, CV_32F);
	Mat offs2 = Mat::eye(3, 3, CV_32F);
	Mat scale1 = Mat::eye(3, 3, CV_32F);
	Mat scale2 = Mat::eye(3, 3, CV_32F);

	offs1.at<float>(0, 2) = -mean1x;
	offs1.at<float>(1, 2) = -mean1y;

	offs2.at<float>(0, 2) = -mean2x;
	offs2.at<float>(1, 2) = -mean2y;

	scale1.at<float>(0, 0) = SQRT2 / sqrt(spread1x);
	scale1.at<float>(1, 1) = SQRT2 / sqrt(spread1y);

	scale2.at<float>(0, 0) = SQRT2 / sqrt(spread2x);
	scale2.at<float>(1, 1) = SQRT2 / sqrt(spread2y);


	struct normalizedData ret;
	ret.T1 = scale1 * offs1;
	ret.T2 = scale2 * offs2;

	//compute normalized points
	for (int i = 0; i < ptsNum; i++) {
		Point2f p1;
		Point2f p2;
		pair<Point2f, Point2f> currPts;

		p1.x = SQRT2 * (pointPairs[i].first.x - mean1x) / sqrt(spread1x);
		p1.y = SQRT2 * (pointPairs[i].first.y - mean1y) / sqrt(spread1y);

		p2.x = SQRT2 * (pointPairs[i].second.x - mean2x) / sqrt(spread2x);
		p2.y = SQRT2 * (pointPairs[i].second.y - mean2y) / sqrt(spread2y);

		ret.newPts1.push_back(p1);
		ret.newPts2.push_back(p2);
	}

	return ret;
}


vector<pair<Point2f, Point2f>> featureMatching(Mat img1, Mat img2, float ratio_thresh) {
	//-- Step 1: Initiate ORB detector
	Ptr<FeatureDetector> detector = ORB::create();

	//-- Step 2: Find the keypoints and descriptors with ORB
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	detector->detect(img1, keypoints1);
	detector->detect(img2, keypoints2);

	Ptr<DescriptorExtractor> extractor = ORB::create();
	extractor->compute(img1, keypoints1, descriptors1);
	extractor->compute(img2, keypoints2, descriptors2);

	//-- Step 3: Create BFMatcher object and match descriptors
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(DescriptorMatcher::BRUTEFORCE);
	std::vector< std::vector<DMatch> > knn_matches;
	matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

	//-- Step 4: Filter matches using the Lowe's ratio test
	std::vector<DMatch> good_matches;
	for (size_t i = 0; i < knn_matches.size(); i++)
	{
		if (knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance)
		{
			good_matches.push_back(knn_matches[i][0]);
		}
	}

	//-- Draw matches
	Mat img_matches;
	drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches);

	//-- Show detected matches
	//resize(img_matches, img_matches, cv::Size(img2.cols, img2.rows), 0, 0, CV_INTER_LINEAR);

	imshow("Good Matches", img_matches);


	//Convert the coordinates:
	int c = keypoints1.size();
	vector<pair<Point2f, Point2f>> pointPairs;
	for (int i = 0; i < c; i++) {
		pair<Point2f, Point2f> currPts;
		//c - column number
		currPts.first = Point2f((float)keypoints1[i].pt.x, (float)keypoints1[i].pt.y); 
		currPts.second = Point2f((float)keypoints2[i].pt.x, (float)keypoints2[i].pt.y);
		pointPairs.push_back(currPts);
	}

	return pointPairs;
}


Mat calcHomography(vector<pair<Point2f, Point2f> > pointPairs) {

	const int ptsNum = pointPairs.size();
	Mat A(2 * ptsNum, 9, CV_32F); //2*point_number
	for (int i = 0; i < ptsNum; i++) {
		float u1 = pointPairs[i].first.x; //index 1 - corrseponds to the 1st image
		float v1 = pointPairs[i].first.y;

		float u2 = pointPairs[i].second.x; //to the second image
		float v2 = pointPairs[i].second.y;

		A.at<float>(2 * i, 0) = u1;
		A.at<float>(2 * i, 1) = v1;
		A.at<float>(2 * i, 2) = 1.0f;
		A.at<float>(2 * i, 3) = 0.0f;
		A.at<float>(2 * i, 4) = 0.0f;
		A.at<float>(2 * i, 5) = 0.0f;
		A.at<float>(2 * i, 6) = -u2 * u1;
		A.at<float>(2 * i, 7) = -u2 * v1;
		A.at<float>(2 * i, 8) = -u2;

		A.at<float>(2 * i + 1, 0) = 0.0f;
		A.at<float>(2 * i + 1, 1) = 0.0f;
		A.at<float>(2 * i + 1, 2) = 0.0f;
		A.at<float>(2 * i + 1, 3) = u1;
		A.at<float>(2 * i + 1, 4) = v1;
		A.at<float>(2 * i + 1, 5) = 1.0f;
		A.at<float>(2 * i + 1, 6) = -v2 * u1;
		A.at<float>(2 * i + 1, 7) = -v2 * v1;
		A.at<float>(2 * i + 1, 8) = -v2;

	}

	Mat eVecs(9, 9, CV_32F), eVals(9, 9, CV_32F);

	eigen(A.t() * A, eVals, eVecs); //eigen OpenCV method computes the evinvalues and eigenvectors

	Mat H(3, 3, CV_32F); //homography 3*3 matrix
	for (int i = 0; i < 9; i++) H.at<float>(i / 3, i % 3) = eVecs.at<float>(8, i);

	//Normalize:
	H = H * (1.0 / H.at<float>(2, 2)); //the matrix is normalized
	

	return H;
}


double geometricDistance(pair<Point2f, Point2f> corr, Mat H) {
	Mat p(3, 1, CV_32F);
	p.at<float>(0, 0) = corr.first.x;
	p.at<float>(1, 0) = corr.first.y;
	p.at<float>(2, 0) = 1;

	Mat proj = H * p;
	proj = (1.0 / proj.at<float>(2, 0)) * proj;

	Mat p2(3, 1, CV_32F);
	p2.at<float>(0, 0) = corr.second.x;
	p2.at<float>(1, 0) = corr.second.y;
	p2.at<float>(2, 0) = 1;

	Mat error = p2 - proj;

	return cv::norm(error);
}


Mat RANSAC(struct normalizedData& data_, double threshold_, int maximum_iteration_number_)
{
	int iterationNumber = 0;
	vector<int> maxInliers;
	vector<int> inliers;
	Mat bestHomo(3, 3, CV_32F);
	constexpr int kSampleSize = 4;
	std::vector<int> sample(kSampleSize);

	while (iterationNumber++ < maximum_iteration_number_)
	{
		

		for (size_t sampleIdx = 0; sampleIdx < kSampleSize; sampleIdx++) {

			//Find 4 random points to calculate a homography
			do
			{
				// Generate a random index between [0, pointNumber]
				sample[sampleIdx] =
					round((data_.newPts1.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

				// If the first point is selected we don't have to check if
				// that particular index had already been selected.
				if (sampleIdx == 0)
					break;

				// If the second point is being generated,
				// it should be checked if the index had been selected beforehand. 
				if (sampleIdx == 1 &&
					sample[0] != sample[1])
					break;

				if (sampleIdx == 2 && sample[0] != sample[2] && sample[1] != sample[2])
					break;

				if (sampleIdx == 3 && sample[0] != sample[3] && sample[1] != sample[3] && sample[2] != sample[3])
					break;
			} while (true);
		}

		
		//Convert to the vector<pair<>> format:
		vector<pair<Point2f, Point2f>> normalized_pointPairs;
		for (int i = 0; i < data_.newPts1.size(); i++) {
			pair<Point2f, Point2f> currPts;
			currPts.first = data_.newPts1[i];
			currPts.second = data_.newPts2[i];
			normalized_pointPairs.push_back(currPts);
		}

		//Take those 4 points:
		vector<pair<Point2f, Point2f>> pointPairsForHomography;
		pointPairsForHomography.push_back(normalized_pointPairs[sample[0]]);
		pointPairsForHomography.push_back(normalized_pointPairs[sample[1]]);
		pointPairsForHomography.push_back(normalized_pointPairs[sample[2]]);
		pointPairsForHomography.push_back(normalized_pointPairs[sample[3]]);

		//Calculate the homography on these 4 points
		Mat H = calcHomography(pointPairsForHomography);

		//denormalize H
		H = data_.T2.inv() * H * data_.T1;
		
		//Calculate the number of inliers
		double distance = 0.0;
		inliers.clear();
		for (size_t i = 0; i < normalized_pointPairs.size(); i++) {
			distance = geometricDistance(normalized_pointPairs[i], H);
			if (distance < threshold_)
				inliers.push_back(i);
		}

		if (inliers.size() > maxInliers.size()) {
			maxInliers = inliers;
			bestHomo = H;
		}
	}

	return bestHomo;

}


void transformImage(Mat origImg, Mat& newImage, Mat tr, bool isPerspective) {
	Mat invTr = tr.inv();
	const int WIDTH = origImg.cols;
	const int HEIGHT = origImg.rows;

	const int newWIDTH = newImage.cols;
	const int newHEIGHT = newImage.rows;


	for (int x = 0; x < newWIDTH; x++) for (int y = 0; y < newHEIGHT; y++) {
		Mat pt(3, 1, CV_32F);
		pt.at<float>(0, 0) = x;
		pt.at<float>(1, 0) = y;
		pt.at<float>(2, 0) = 1.0;

		Mat ptTransformed = invTr * pt;
		if (isPerspective) ptTransformed = (1.0 / ptTransformed.at<float>(2, 0)) * ptTransformed;

		int newX = round(ptTransformed.at<float>(0, 0));
		int newY = round(ptTransformed.at<float>(1, 0));

		if ((newX >= 0) && (newX < WIDTH) && (newY >= 0) && (newY < HEIGHT))
			newImage.at<Vec3b>(y, x) = origImg.at<Vec3b>(newY, newX);
	}
}


int main(int argc, char** argv)
{
	Mat img1 = imread(argv[2]);
	Mat img2 = imread(argv[1]);

	vector<pair<Point2f, Point2f>> pointPairs = featureMatching(img1, img2, 0.8f);

	vector <Point2f> src_points, des_points;
	for (size_t i = 0; i < pointPairs.size(); i++)
	{
		src_points.push_back(pointPairs[i].first);
		des_points.push_back(pointPairs[i].second);
	}

	struct normalizedData normalized = NormalizeData(pointPairs);



	Mat bestH = RANSAC(normalized, 1, 100);
	
	Mat transformedImage = Mat::zeros(1.5 * img1.size().height, 2.0 * img1.size().width, img1.type());
	transformImage(img1, transformedImage, Mat::eye(3, 3, CV_32F), true);

	transformImage(img2, transformedImage, bestH, true); //first image transformed to the resulted


	imwrite("res.png", transformedImage);
	imshow("Display window", transformedImage);
	waitKey();
	
	
	return 0;
}      


