
#include <fstream>
#include <iostream>
#include <opencv\cv.hpp>
#include <opencv\highgui.h>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "MatrixReaderWriter.h"

// estimating the fundamental matrix 
void getFundamentalMatrixLSQ(
	const std::vector<cv::Point2d>& source_points_,
	const std::vector<cv::Point2d>& destination_points_,
	cv::Mat& fundamental_matrix_)
{
	const size_t pointNumber = source_points_.size();
	cv::Mat A(pointNumber, 9, CV_64F);

	for (size_t pointIdx = 0; pointIdx < pointNumber; ++pointIdx)
	{
		const double
			& x1 = source_points_[pointIdx].x,
			& y1 = source_points_[pointIdx].y,
			& x2 = destination_points_[pointIdx].x,
			& y2 = destination_points_[pointIdx].y;

		//2nd presentation page 27
		A.at<double>(pointIdx, 0) = x1 * x2;
		A.at<double>(pointIdx, 1) = x2 * y1;
		A.at<double>(pointIdx, 2) = x2;
		A.at<double>(pointIdx, 3) = y2 * x1;
		A.at<double>(pointIdx, 4) = y2 * y1;
		A.at<double>(pointIdx, 5) = y2;
		A.at<double>(pointIdx, 6) = x1;
		A.at<double>(pointIdx, 7) = y1;
		A.at<double>(pointIdx, 8) = 1;
	}

	cv::Mat evals, evecs;
	cv::Mat AtA = A.t() * A;
	cv::eigen(AtA, evals, evecs);
	
	cv::Mat x = evecs.row(evecs.rows - 1); // x = [f1 f2 f3 f4 f5 f6 f7 f8 f9]
	fundamental_matrix_.create(3, 3, CV_64F);
	memcpy(fundamental_matrix_.data, x.data, sizeof(double) * 9);
}

//estimating the 3D point coordinates from a point correspondences
// from the projection matrices of the two observing cameras.
void linearTriangulation(
	const cv::Mat& projection_1_,
	const cv::Mat& projection_2_,
	const cv::Mat& src_point_, // A point in the source image
	const cv::Mat& dst_point_, // A point in the destination image
	cv::Mat& point3d_)
{
	cv::Mat A(4, 3, CV_64F);
	cv::Mat b(4, 1, CV_64F);

	//PX = x, P - projection matrix, X - 3D coordinates of the points, x - 2D projection

	{
		const double
			& px = src_point_.at<double>(0),
			& py = src_point_.at<double>(1),
			& p1 = projection_1_.at<double>(0, 0),
			& p2 = projection_1_.at<double>(0, 1),
			& p3 = projection_1_.at<double>(0, 2),
			& p4 = projection_1_.at<double>(0, 3),
			& p5 = projection_1_.at<double>(1, 0),
			& p6 = projection_1_.at<double>(1, 1),
			& p7 = projection_1_.at<double>(1, 2),
			& p8 = projection_1_.at<double>(1, 3),
			& p9 = projection_1_.at<double>(2, 0),
			& p10 = projection_1_.at<double>(2, 1),
			& p11 = projection_1_.at<double>(2, 2),
			& p12 = projection_1_.at<double>(2, 3);
		//homogeneous 
		A.at<double>(0, 0) = px * p9 - p1;
		A.at<double>(0, 1) = px * p10 - p2;
		A.at<double>(0, 2) = px * p11 - p3;
		A.at<double>(1, 0) = py * p9 - p5;
		A.at<double>(1, 1) = py * p10 - p6;
		A.at<double>(1, 2) = py * p11 - p7;
		//inhomogeneous
		b.at<double>(0) = p4 - px * p12;
		b.at<double>(1) = p8 - py * p12;
	}

	{
		const double
			& px = dst_point_.at<double>(0),
			& py = dst_point_.at<double>(1),
			& p1 = projection_2_.at<double>(0, 0),
			& p2 = projection_2_.at<double>(0, 1),
			& p3 = projection_2_.at<double>(0, 2),
			& p4 = projection_2_.at<double>(0, 3),
			& p5 = projection_2_.at<double>(1, 0),
			& p6 = projection_2_.at<double>(1, 1),
			& p7 = projection_2_.at<double>(1, 2),
			& p8 = projection_2_.at<double>(1, 3),
			& p9 = projection_2_.at<double>(2, 0),
			& p10 = projection_2_.at<double>(2, 1),
			& p11 = projection_2_.at<double>(2, 2),
			& p12 = projection_2_.at<double>(2, 3);

		A.at<double>(2, 0) = px * p9 - p1;
		A.at<double>(2, 1) = px * p10 - p2;
		A.at<double>(2, 2) = px * p11 - p3;
		A.at<double>(3, 0) = py * p9 - p5;
		A.at<double>(3, 1) = py * p10 - p6;
		A.at<double>(3, 2) = py * p11 - p7;

		b.at<double>(2) = p4 - px * p12;
		b.at<double>(3) = p8 - py * p12;
	}

	//cv::Mat x = (A.t() * A).inv() * A.t() * b;
	point3d_ = A.inv(cv::DECOMP_SVD) * b;
}

// A function decomposing the essential matrix to the projection matrices
// of the two cameras.
void getProjectionMatrices(
	const cv::Mat& essential_matrix_,
	const cv::Mat& K1_,
	const cv::Mat& K2_,
	const cv::Mat& src_point_,
	const cv::Mat& dst_point_,
	cv::Mat& projection_1_,
	cv::Mat& projection_2_)
{
	// ****************************************************
	// Calculate the projection matrix of the first camera
	// ****************************************************
	projection_1_ = K1_ * cv::Mat::eye(3, 4, CV_64F); //assume that the 1st camera is in the origin

	// projection_1_.create(3, 4, CV_64F);
	// cv::Mat rotation_1 = cv::Mat::eye(3, 3, CV_64F);
	// cv::Mat translation_1 = cv::Mat::zeros(3, 1, CV_64F);

	// ****************************************************
	// Calculate the projection matrix of the second camera
	// ****************************************************

	// Decompose the essential matrix
	cv::Mat rotation_1, rotation_2, translation;

	cv::SVD svd(essential_matrix_, cv::SVD::FULL_UV);
	// It gives matrices U D Vt

	if (cv::determinant(svd.u) < 0)
		svd.u.col(2) *= -1;
	if (cv::determinant(svd.vt) < 0)
		svd.vt.row(2) *= -1;

	cv::Mat w = (cv::Mat_<double>(3, 3) << 0, -1, 0,
		1, 0, 0,
		0, 0, 1);

	rotation_1 = svd.u * w * svd.vt;
	rotation_2 = svd.u * w.t() * svd.vt;
	translation = svd.u.col(2) / cv::norm(svd.u.col(2));

	// The possible solutions:
	// (rotation_1, translation)
	// (rotation_2, translation)
	// (rotation_1, -translation)
	// (rotation_2, -translation)

	cv::Mat P21 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_1.at<double>(0, 0), rotation_1.at<double>(0, 1), rotation_1.at<double>(0, 2), translation.at<double>(0),
		rotation_1.at<double>(1, 0), rotation_1.at<double>(1, 1), rotation_1.at<double>(1, 2), translation.at<double>(1),
		rotation_1.at<double>(2, 0), rotation_1.at<double>(2, 1), rotation_1.at<double>(2, 2), translation.at<double>(2));
	cv::Mat P22 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_2.at<double>(0, 0), rotation_2.at<double>(0, 1), rotation_2.at<double>(0, 2), translation.at<double>(0),
		rotation_2.at<double>(1, 0), rotation_2.at<double>(1, 1), rotation_2.at<double>(1, 2), translation.at<double>(1),
		rotation_2.at<double>(2, 0), rotation_2.at<double>(2, 1), rotation_2.at<double>(2, 2), translation.at<double>(2));
	cv::Mat P23 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_1.at<double>(0, 0), rotation_1.at<double>(0, 1), rotation_1.at<double>(0, 2), -translation.at<double>(0),
		rotation_1.at<double>(1, 0), rotation_1.at<double>(1, 1), rotation_1.at<double>(1, 2), -translation.at<double>(1),
		rotation_1.at<double>(2, 0), rotation_1.at<double>(2, 1), rotation_1.at<double>(2, 2), -translation.at<double>(2));
	cv::Mat P24 = K2_ * (cv::Mat_<double>(3, 4) <<
		rotation_2.at<double>(0, 0), rotation_2.at<double>(0, 1), rotation_2.at<double>(0, 2), -translation.at<double>(0),
		rotation_2.at<double>(1, 0), rotation_2.at<double>(1, 1), rotation_2.at<double>(1, 2), -translation.at<double>(1),
		rotation_2.at<double>(2, 0), rotation_2.at<double>(2, 1), rotation_2.at<double>(2, 2), -translation.at<double>(2));

	//which one P is the correct one?
	std::vector< const cv::Mat* > Ps = { &P21, &P22, &P23, &P24 };
	double minDistance = std::numeric_limits<double>::max();

	for (const auto& P2ptr : Ps)
	{
		const cv::Mat& P1 = projection_1_;
		const cv::Mat& P2 = *P2ptr;

		// Estimate the 3D coordinates of a point correspondence
		cv::Mat point3d;
		linearTriangulation(P1,
			P2,
			src_point_,
			dst_point_,
			point3d);
		point3d.push_back(1.0);

		cv::Mat projection1 =
			P1 * point3d;
		cv::Mat projection2 =
			P2 * point3d;

		if (projection1.at<double>(2) < 0 ||
			projection2.at<double>(2) < 0)
			continue;

		projection1 = projection1 / projection1.at<double>(2);
		projection2 = projection2 / projection2.at<double>(2);

		// cv::norm(projection1 - src_point_)
		double dx1 = projection1.at<double>(0) - src_point_.at<double>(0);
		double dy1 = projection1.at<double>(1) - src_point_.at<double>(1);
		double squaredDist1 = dx1 * dx1 + dy1 * dy1;

		// cv::norm(projection2 - dst_point_)
		double dx2 = projection2.at<double>(0) - dst_point_.at<double>(0);
		double dy2 = projection2.at<double>(1) - dst_point_.at<double>(1);
		double squaredDist2 = dx2 * dx2 + dy2 * dy2;

		if (squaredDist1 + squaredDist2 < minDistance)
		{
			minDistance = squaredDist1 + squaredDist2;
			projection_2_ = P2.clone();
		}
	}
	//Extrinsic camera parameters - Rotation and Translation
}



int main(int argc, char** argv)
{
	// Load images
	cv::Mat image1 = cv::imread("1.png", cv::IMREAD_ANYCOLOR);
	cv::Mat image2 = cv::imread("2.png", cv::IMREAD_ANYCOLOR);
	
	
	const char* fileName("W_first_normal_general.mat");
	MatrixReaderWriter mtxrw(fileName);


	
	int r = mtxrw.rowNum;
	int c = mtxrw.columnNum;

	
	//Convert the coordinates:
	vector<pair<cv::Point2d, cv::Point2d> > pointPairs;
	vector<cv::Point2d> source_points;
	vector<cv::Point2d> destination_points;
	for (int i = 0; i < mtxrw.columnNum; i++) {
		pair<cv::Point2d, cv::Point2d> currPts;
		//c - column number
		currPts.first = cv::Point2d((double)mtxrw.data[i], (double)mtxrw.data[c + i]);
		currPts.second = cv::Point2d((double)mtxrw.data[2 * c + i], (double)mtxrw.data[3 * c + i]);
		source_points.push_back(currPts.first);
		destination_points.push_back(currPts.second);
		pointPairs.push_back(currPts);
	}
	

	//Fundamental matrix
	cv::Mat F(3, 3, CV_64F);
	getFundamentalMatrixLSQ(
		source_points,
		destination_points,
		F);
	

	//K - intrinsic camera parameters
	cv::Mat K = (cv::Mat_<double>(3, 3) << 1262.620252, 0, 934.611657,
		0, 1267.365350, 659.520995,
		0, 0, 1);
	
	// Essential matrix
	cv::Mat E = K.t() * F * K;
	cv::Mat P1, P2;

	// Decompose the essential matrix
	getProjectionMatrices(E,
		K,
		K,
		(cv::Mat)source_points[0], // A point in the source image
		(cv::Mat)destination_points[0], // A point in the destination image
		P1,
		P2);
	
	std::vector<cv::KeyPoint> src_inliers, dst_inliers;
	src_inliers.reserve(source_points.size());
	dst_inliers.reserve(source_points.size());
	std::ofstream out_file("result.xyz");
	for (auto inl_idx = 0; inl_idx < source_points.size(); inl_idx += 1)
	{
		const cv::Mat pt1 = static_cast<cv::Mat>(source_points[inl_idx]);
		const cv::Mat pt2 = static_cast<cv::Mat>(destination_points[inl_idx]);

		// Estimate the 3D coordinates of the current inlier correspondence
		cv::Mat point3d;
		linearTriangulation(P1,
			P2,
			pt1,
			pt2,
			point3d);

		
		out_file << point3d.at<double>(0) << " "
			<< point3d.at<double>(1) << " "
			<< point3d.at<double>(2) << "\n";
	}
	out_file.close();
	

	cout << r << endl;
}