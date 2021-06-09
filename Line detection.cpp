#include "stdafx.h"

#include <iostream>
#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <vector>
#include <time.h>

using namespace cv;
using namespace std;


void DrawPoints(vector<Point2d>& points,
    Mat image);

void FitLineRANSAC(
    const vector<Point2d> &points,
	const std::vector<int>& mask_,
    vector<int>& inliers,
    Mat& line,
    double threshold,
    double confidence_,
    int iteration_number,
    Mat* image = nullptr);

size_t GetIterationNumber(
	const double& inlierRatio_,
	const double& confidence_,
	const size_t& sampleSize_);

void sequentialRANSAC(
	const vector<Point2d> &points,
	vector<vector<int>> &inliers,
	vector<Mat> &line,
	double threshold,
	double confidence,
	int iteration_number,
	int minimumInlierNumber,
	int   = std::numeric_limits<int>::max(),
	Mat* image = nullptr 
);



int main(int argc, char** argv) {
    //uploading the image
    cv::Mat image = cv::imread("C:/Users/ASUS/Desktop/test IMG/y.png");
    cv::Mat contours;

    cv::Canny(image, contours, 100, 200);
    
    vector<Point2d> points;

    Mat plot = Mat::zeros(image.rows, image.cols, CV_8UC3); // The image where we draw results. 


    //Generating the points from the contours image:
    uint8_t* myData = contours.data;
    int width = contours.cols;
    int height = contours.rows;
    int _stride = contours.step;//in case cols != strides
    for (int i = 0; i < height; i++){
        for (int j = 0; j < width; j++){
            double val = myData[i * _stride + j];
            if (val == 255) {
                Point2d point;
                point.x = j;
                point.y = i;
                points.emplace_back(point);
            }
        }
    }

	vector<vector<int>> inliers;
	// The parameters of the line
	vector<Mat> bestLines;
   
	sequentialRANSAC(
		points, // The generated 2D points
		inliers, // Output: the indices of the inliers
		bestLines, // Output: the parameters of the found 2D line
		1.0, // The inlier-outlier threshold
		0.99, // The confidence required in the results
		1000, // The number of iterations
		100); // minimum inliers
	
	for(int lineIdx = 0; lineIdx < bestLines.size(); ++lineIdx){
		// Draw the line from RANSAC
		const double& a1 = bestLines[lineIdx].at<double>(0);
		const double& b1 = bestLines[lineIdx].at<double>(1);
		const double& c1 = bestLines[lineIdx].at<double>(2);
		
		DrawPoints(points, plot);
		// Draw the 2D line
		cv::line(plot,
			Point2d(0, -c1 / b1),
			Point2d(image.cols, (-a1 * image.cols - c1) / b1),
			cv::Scalar(0, 0, 255),
			2);
		cv::line(image,
			Point2d(0, -c1 / b1),
			Point2d(image.cols, (-a1 * image.cols - c1) / b1),
			cv::Scalar(0, 0, 255),
			1);

		
	}
   imshow("contour", contours);
   imshow("image", image);
   imshow("Final result", plot);
   waitKey(0);

	return 0;
}



// Draw points to the image
void DrawPoints(vector<Point2d>& points,
    Mat image)
{
    for (int i = 0; i < points.size(); ++i)
    {
        // Draws a circle
        circle(image, // to this image 
            points[i], // at this location(center)
            1, // with this radius
            Scalar(255, 255, 255), // and this color
            -1); // The thickness of the circle's outline. -1 = filled circle
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
 
void sequentialRANSAC(
	const vector<Point2d>& points,
	vector<vector<int>>& inliers,
	vector<Mat>& lines,
	double threshold,
	double confidence,
	int iteration_number,
	int minimumInlierNumber,
	int lineNumber,
	Mat *image
) 
{
	

	std::vector<int> mask(points.size(), 0);
	for (int lineIdx = 0; lineIdx < lineNumber; ++lineIdx) {
		//new set of inliers
		inliers.emplace_back(std::vector<int>());
		//parameter of the new line
		lines.emplace_back(cv::Mat(3, 1, CV_64F));

		FitLineRANSAC(
			points, // The generated 2D points
			mask,
			inliers.back(), // Output: the indices of the inliers
			lines.back(), // Output: the parameters of the found 2D line
			threshold, // The inlier-outlier threshold
			confidence, // The confidence required in the results
			iteration_number // The number of iterations
		); // Optional: the image where we can draw results
		 
		if (inliers.back().size() < minimumInlierNumber) {
			inliers.resize(inliers.size() - 1);
			lines.resize(lines.size() - 1);
			break;
		}

		for (const auto& inlierIdx : inliers.back()) {
			mask[inlierIdx] = lineIdx + 1; 
		}
	}

	
}


void FitLineRANSAC(
	const vector<Point2d>& points_,
	const std::vector<int> &mask_,
	vector<int>& inliers_,
	Mat& line_,
	double threshold_,
	double confidence_,
	int maximum_iteration_number_,
	Mat* image_)
{
	// The current number of iterations
	int iterationNumber = 0;
	// The number of inliers of the current best model
	int bestInlierNumber = 0;
	// The indices of the inliers of the current best model
	vector<int> bestInliers, inliers;
	bestInliers.reserve(points_.size());
	inliers.reserve(points_.size());
	// The parameters of the best line
	Mat bestLine(3, 1, CV_64F);
	// Helpers to draw the line if needed
	Point2d bestPt1, bestPt2;
	// The sample size, i.e., 2 for 2D lines
	constexpr int kSampleSize = 2;
	// The current sample
	std::vector<int> sample(kSampleSize);

	bool shouldDraw = image_ != nullptr;
	cv::Mat tmp_image;
	size_t maximumIterations = maximum_iteration_number_;

	// RANSAC:
	// 1. Select a minimal sample, i.e., in this case, 2 random points.
	// 2. Fit a line to the points.
	// 3. Count the number of inliers, i.e., the points closer than the threshold.
	// 4. Store the inlier number and the line parameters if it is better than the previous best. 

	while (iterationNumber++ < maximumIterations)
	{
		// 1. Select a minimal sample, i.e., in this case, 2 random points.
		for (size_t sampleIdx = 0; sampleIdx < kSampleSize; ++sampleIdx)
		{
			do
			{
				// Generate a random index between [0, pointNumber]
				sample[sampleIdx] =
					round((points_.size() - 1) * static_cast<double>(rand()) / static_cast<double>(RAND_MAX));

				if (mask_[sample[sampleIdx]] != 0)
					continue;

				// If the first point is selected we don't have to check if
				// that particular index had already been selected.
				if (sampleIdx == 0)
					break;

				// If the second point is being generated,
				// it should be checked if the index had been selected beforehand. 
				if (sampleIdx == 1 &&
					sample[0] != sample[1])
					break;
			} while (true);
		}

		if (shouldDraw)
		{
			tmp_image = image_->clone();

			circle(tmp_image, // to this image 
				points_[sample[0]], // at this location
				5, // with this radius
				Scalar(0, 0, 255), // and this color
				-1); // The thickness of the circle's outline. -1 = filled circle

			circle(tmp_image, // to this image 
				points_[sample[1]], // at this location
				5, // with this radius
				Scalar(0, 0, 255), // and this color
				-1); // The thickness of the circle's outline. -1 = filled circle
		}

		// 2. Fit a line to the points.
		const Point2d& p1 = points_[sample[0]]; // First point selected
		const Point2d& p2 = points_[sample[1]]; // Second point select		
		Point2d v = p2 - p1; // Direction of the line
		// cv::norm(v) = sqrt(v.x * v.x + v.y * v.y)
		v = v / cv::norm(v);
		// Rotate v by 90° to get n.
		Point2d n;
		n.x = -v.y;
		n.y = v.x;
		// To get c use a point from the line.
		double a = n.x;
		double b = n.y;
		double c = -(a * p1.x + b * p1.y);

		// Draw the 2D line
		if (shouldDraw)
		{
			cv::line(tmp_image,
				Point2d(0, -c / b),
				Point2d(tmp_image.cols, (-a * tmp_image.cols - c) / b),
				cv::Scalar(0, 255, 0),
				2);
		}

		// - Distance of a line and a point
		// - Line's implicit equations: a * x + b * y + c = 0
		// - a, b, c - parameters of the line
		// - x, y - coordinates of a point on the line
		// - n = [a, b] - the normal of the line
		// - Distance(line, point) = | a * x + b * y + c | / sqrt(a * a + b * b)
		// - If ||n||_2 = 1 then sqrt(a * a + b * b) = 1 and I don't have do the division.

		// 3. Count the number of inliers, i.e., the points closer than the threshold.
		inliers.clear();
		for (size_t pointIdx = 0; pointIdx < points_.size(); ++pointIdx)
		{
			if (mask_[pointIdx] != 0)
				continue;

			const Point2d& point = points_[pointIdx];
			const double distance =
				abs(a * point.x + b * point.y + c);

			if (distance < threshold_)
			{
				inliers.emplace_back(pointIdx);

				if (shouldDraw)
				{
					circle(tmp_image, // to this image 
						points_[pointIdx], // at this location
						3, // with this radius
						Scalar(0, 255, 0), // and this color
						-1); // The thickness of the circle's outline. -1 = filled circle
				}
			}
		}

		// 4. Store the inlier number and the line parameters if it is better than the previous best. 
		if (inliers.size() > bestInliers.size())
		{
			bestInliers.swap(inliers);
			inliers.clear();
			inliers.resize(0);

			bestLine.at<double>(0) = a;
			bestLine.at<double>(1) = b;
			bestLine.at<double>(2) = c;

			// Update the maximum iteration number
			maximumIterations = GetIterationNumber(
				static_cast<double>(bestInliers.size()) / static_cast<double>(points_.size()),
				confidence_,
				kSampleSize);

			printf("Inlier number = %d\tMax iterations = %d\n", bestInliers.size(), maximumIterations);
		}

		if (shouldDraw)
		{
			cv::line(tmp_image,
				Point2d(0, -bestLine.at<double>(2) / bestLine.at<double>(1)),
				Point2d(tmp_image.cols, (-bestLine.at<double>(0) * tmp_image.cols - bestLine.at<double>(2)) / bestLine.at<double>(1)),
				cv::Scalar(255, 0, 0),
				2);

			cv::imshow("Image", tmp_image);
			cv::waitKey(0);
		}
	}

	inliers_ = bestInliers;
	line_ = bestLine;
}
