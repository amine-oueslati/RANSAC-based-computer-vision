
#include "MatrixReaderWriter.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include "stdafx.h"
#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <vector>
#include <time.h>

using namespace cv;
using namespace std;



MatrixReaderWriter* mrw;

void GenerateData(vector<Point3d>& points,
	MatrixReaderWriter* mrw
	);

void FitePlaneRANSAC(
	const vector<Point3d>& points,
	vector<int>& inliers,
	Mat& plane,
	double threshold,
	double confidence_,
	int iteration_number);

size_t GetIterationNumber(
	const double& inlierRatio_,
	const double& confidence_,
	const size_t& sampleSize_);

void FitPlaneLSQ(const vector<Point3d>* const points,
	vector<int>& inliers,
	Mat& plane);


////////////////////////////////////////

int main(int argc, const char** argv) {

	srand(time(NULL));

	if (argc != 2) { printf("Usage: file missing\n"); exit(0); }

	mrw = new MatrixReaderWriter(argv[1]);
	vector<Point3d> points;
	
	GenerateData(points, mrw);
	
	//Indices of points on the plane
	vector<int> inliers;

	//Parameters of the plane
	Mat bestPlane;
	
	FitePlaneRANSAC(
		points, // The generated 2D points
		inliers, // Output: the indices of the inliers
		bestPlane, // Output: the parameters of the found 2D line
		0.0005, // The inlier-outlier threshold
		0.999, // The confidence required in the results
		1000); // The number of iterations
	
	cout << "inliers " << inliers.size() << endl;
	cout << "best plane" << bestPlane << endl;

	ofstream ransacResult;
	ransacResult.open("ransacResult.xyz");
	for(int i = 0; i < inliers.size(); i++)
		ransacResult << points[inliers[i]].x << " " << points[inliers[i]].y << " " << points[inliers[i]].z <<"\n";
	ransacResult.close();

	const double& a1 = bestPlane.at<double>(0);
	const double& b1 = bestPlane.at<double>(1);
	const double& c1 = bestPlane.at<double>(2);
	const double& d1 = bestPlane.at<double>(3);
	
	// Calculate the error of RANSAC 
	double averageError1 = 0.0;
	for (const auto& inlierIdx : inliers){
		double distance = abs(a1 * points[inlierIdx].x + b1 * points[inlierIdx].y + c1 * points[inlierIdx].z + d1);
		averageError1 += distance;
	}

	averageError1 /= inliers.size();

	cout << "error 1 " << averageError1 << endl;
	
	
	
	FitPlaneLSQ(&points,
		inliers,
		bestPlane);

	const double& a2 = bestPlane.at<double>(0);
	const double& b2 = bestPlane.at<double>(1);
	const double& c2 = bestPlane.at<double>(2);
	const double& d2 = bestPlane.at<double>(3);

	
	// Calculate the error after optimization 
	double averageError2 = 0.0;
	for (const auto& inlierIdx : inliers){
		double distance = abs(a2 * points[inlierIdx].x + b2 * points[inlierIdx].y + c2 * points[inlierIdx].z + d2);
		averageError2 += distance;
	}
	averageError2 /= inliers.size();

	cout << "error 2 " << averageError2 << endl;
	
}

/////////////////////////////////////////

// Generating a vector from the xyz file
void GenerateData(vector<Point3d>& points,
	MatrixReaderWriter* mrw
) {
	int NUM = mrw->rowNum;

	for (int i = 0; i < NUM; i++) {
		Point3d point;
		point.x = mrw->data[i * 3];
		point.y = mrw->data[i * 3 + 1];
		point.z = mrw->data[i * 3 + 2];

		points.emplace_back(point);
	}
}

size_t GetIterationNumber(
	const double& inlierRatio_,
	const double& confidence_,
	const size_t& sampleSize_)
{
	double a =
		log(1.0 - confidence_);
	double b =
		log(1.0 - std::pow(inlierRatio_, sampleSize_));

	if (abs(b) < std::numeric_limits<double>::epsilon())
		return std::numeric_limits<size_t>::max();

	return a / b;
}


// Apply RANSAC to fit points to a plane
void FitePlaneRANSAC(
	const vector<Point3d>& points_,
	vector<int>& inliers_,
	Mat& plane_,
	double threshold_,
	double confidence_,
	int maximum_iteration_number_
)
{
	// The current number of iterations
	int iterationNumber = 0;
	// The number of inliers of the current best model
	int bestInlierNumber = 0;
	// The indices of the inliers of the current best model
	vector<int> bestInliers, inliers;
	bestInliers.reserve(points_.size());
	inliers.reserve(points_.size());
	// The parameters of the best plane
	Mat bestPlane(4, 1, CV_64F);
	// The sample size, i.e., 3 
	constexpr int kSampleSize = 3;
	// The current sample
	std::vector<int> sample(kSampleSize);

	size_t maximumIterations = maximum_iteration_number_;

	// RANSAC:
	// 1. Select a minimal sample, i.e., in this case, 3 random points.
	// 2. Fit a plane to the points.
	// 3. Count the number of inliers, i.e., the points closer than the threshold.
	// 4. Store the inlier number and the line parameters if it is better than the previous best. 

	while (iterationNumber++ < maximumIterations){
		for (size_t sampleIdx = 0; sampleIdx < kSampleSize; ++sampleIdx){
			do{
				// Generate a random index between [0, pointNumber]
				sample[sampleIdx] =
					round((points_.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

				if (sampleIdx == 0)
					break;
				// If the second point is being generated,
				// it should be checked if the index had been selected beforehand. 
				if (sampleIdx == 1 &&
					sample[0] != sample[1])
					break;
				//same thing for the third point
				if (sampleIdx == 2 &&
					sample[2] != sample[0] &&
					sample[2] != sample[1])
					break;
			} while (true);
		}


		// 2. Fit a plane to the points.
		const Point3d& p1 = points_[sample[0]]; // First point selected
		const Point3d& p2 = points_[sample[1]]; // Second point select		
		const Point3d& p3 = points_[sample[2]]; // third point select		

		//Take two vectors from the plane
		Point3d v1 = p2 - p1; // Vector 1
		Point3d v2 = p3 - p1; // Vector 2

		// cv::norm(v) = sqrt(v.x * v.x + v.y * v.y)
		v1 = v1 / cv::norm(v1);
		v2 = v2 / cv::norm(v2);
		// v1 * v2 
		Point3d n;
		n.x = v1.y * v2.z - v2.y * v1.z;
		n.y = v2.x * v1.z - v1.x * v2.z;
		n.z = v1.x * v2.y - v1.y * v2.x;
		// To get d use a point from the plane.
		double a = n.x;
		double b = n.y;
		double c = n.z;
		double d = -(a * p1.x + b * p1.y + c * p1.z);


		// - Distance of a plane and a point
		// - plane's implicit equations: a * x + b * y + c * z + d = 0
		// - a, b, c, d - parameters of the plane
		// - x, y, z - coordinates of a point on the plane
		// - n = [a, b, c] - the normal of the plane
		// - Distance(plane, point) = | a * x + b * y + c * z + d | / sqrt(a * a + b * b + c * c)
		// - If ||n||_2 = 1 then sqrt(a * a + b * b + c * c) = 1 and I don't have do the division.

		// 3. Count the number of inliers, i.e., the points closer than the threshold.
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx)
		{
			const Point3d& point = points_[pointIdx];
			const double distance =abs(a * point.x + b * point.y + c * point.z + d);
			if (distance < threshold_)
			{
				inliers.emplace_back(pointIdx);
			}
		}

		// 4. Store the inlier number and the line parameters if it is better than the previous best. 
		if (inliers.size() > bestInliers.size())
		{
			bestInliers.swap(inliers);
			inliers.clear();
			inliers.resize(0);

			bestPlane.at<double>(0) = a;
			bestPlane.at<double>(1) = b;
			bestPlane.at<double>(2) = c;
			bestPlane.at<double>(3) = d;

			// Update the maximum iteration number
			maximumIterations = GetIterationNumber(
				static_cast<double>(bestInliers.size()) / static_cast<double>(points_.size()),
				confidence_,
				kSampleSize);

			printf("Inlier number = %d\tMax iterations = %d\n", bestInliers.size(), maximumIterations);
		}
	}

	inliers_ = bestInliers;
	plane_ = bestPlane;
}



// Apply Least-Squares plane fitting.
void FitPlaneLSQ(const vector<Point3d>* const points,
	vector<int>& inliers,
	Mat& plane)
{
	vector<Point3d> normalizedPoints;
	normalizedPoints.reserve(inliers.size());

	// Calculating the mass point of the points
	Point3d masspoint(0, 0, 0);

	for (const auto& inlierIdx : inliers)
	{
		masspoint += points->at(inlierIdx);
		normalizedPoints.emplace_back(points->at(inlierIdx));
	}
	masspoint = masspoint * (1.0 / inliers.size());

	// Move the point cloud to have the origin in their mass point
	for (auto& point : normalizedPoints)
		point -= masspoint;

	// Calculating the average distance from the origin
	double averageDistance = 0.0;
	for (auto& point : normalizedPoints)
	{
		averageDistance += cv::norm(point);
	}

	averageDistance = averageDistance / normalizedPoints.size();
	
	const double ratio = sqrt(2) / averageDistance;

	// Making the average distance to be sqrt(2)
	for (auto& point : normalizedPoints)
		point *= ratio;

	// Now, we should solve the equation.
	cv::Mat A(normalizedPoints.size(), 3, CV_64F);

	// Building the coefficient matrix
	for (size_t pointIdx = 0; pointIdx < normalizedPoints.size(); ++pointIdx)
	{
		const size_t& rowIdx = pointIdx;

		A.at<double>(rowIdx, 0) = normalizedPoints[pointIdx].x;
		A.at<double>(rowIdx, 1) = normalizedPoints[pointIdx].y;
		A.at<double>(rowIdx, 2) = normalizedPoints[pointIdx].z;
	}

	cv::Mat evals, evecs;
	cv::eigen(A.t() * A, evals, evecs);
	cout << "eigen ----->" << evals << endl;
	const cv::Mat& normal = evecs.row(2);
	const double& a = normal.at<double>(0),
		& b = normal.at<double>(1),
		& c = normal.at<double>(2);
		const double d = -(a * masspoint.x + b * masspoint.y + c * masspoint.z);

	plane.at<double>(0) = a / 1000;
	plane.at<double>(1) = b / 1000;
	plane.at<double>(2) = c / 1000;
	plane.at<double>(3) = d / 1000;
}
