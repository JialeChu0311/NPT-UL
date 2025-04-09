#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video.hpp"
#include "opencv2/videoio.hpp"
#include<vector>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include<algorithm>
#include <math.h>

using  namespace std;
using  namespace cv;

Mat simplestColorBalance(cv::Mat src, float percent);
double clip(double dd, double min, double max);

int num;

double B = 0.0, G = 0.0, R = 0.0;
double Bsum = 0.0, Gsum = 0.0, Rsum = 0.0;
double Bavg = 0.0, Gavg = 0.0, Ravg = 0.0;

double Bnewsum = 0.0, Gnewsum = 0.0, Rnewsum = 0.0;
double Bnewavg = 0.0, Gnewavg = 0.0, Rnewavg = 0.0;

double B_new = 0.0, G_new = 0.0, R_new = 0.0;
double gama = 0.9;

double a = 1.1;
double b = 1.3;

int i, j, l, p;
vector<Mat_<uchar>> channels;
vector<Mat_<uchar>> channel;

Mat dst, myImg, dst5;
Mat dst1, outImg_d1, outImg_e, dst4, outImg_g;
Mat outImg, outImg1;
Mat dst3, dst9, outImg2, outImg3;
double contrast11 = 1.2;
Mat channle(Mat myImg);
Mat RGB(Mat outImg);
Mat limite(Mat outImg1);
Mat gussfusing(Mat dst4);
void getHistogram(cv::Mat* channel, cv::Mat* hist);
Mat histStretch(Mat src, float percent, int direction);

Mat gamacol1(Mat image_roi, double gama1);

Mat histStretch(cv::Mat src, float percent, int direction);
void getHistogram(cv::Mat* channel, cv::Mat* hist);

vector<Mat> Gau_Pyr, lp_Pyr, Gau_Pyr2, lp_Pyr2;
vector<Mat> maskGaussianPyramid;
Mat output, dst_fusing;

Mat CLAHE1(Mat inputImag);
Mat CLAHE_image;
Mat outImg_f;
cv::Mat RGB1, LAB, chanBGR[3], chanLAB[3];
Scalar meanR, stddevR, meanG, stddevG, meanB, stddevB;
//画直方图
Mat showHist(Mat& img);

bool compare_natural(const string& a, const string& b) {
	size_t a_num = stoi(a.substr(a.find_last_of("/\\") + 1));
	size_t b_num = stoi(b.substr(b.find_last_of("/\\") + 1));
	return a_num < b_num;
}

int main()
{
	string path = "C:\\Users\\cjl\\Desktop\\image\\";//此处替换为自己的图片序列
	string write_path = "C:\\Users\\cjl\\Desktop\\output\\";//目标文件夹，需要提前建好  

	vector<String> src_name;
	glob(path, src_name, true);

	if (src_name.size() == 0)
	{
		cerr << "That's no file in " << path << endl;
		exit(1);
	}

	sort(src_name.begin(), src_name.end(), compare_natural);

	for (int f = 0; f < src_name.size(); ++f)
	{
		cout << src_name[f] << endl;
		myImg = imread(src_name[f]);
		if (myImg.empty())
		{
			cerr << "Read image " << src_name[f] << " failed!";
			exit(1);
		}

		num = myImg.cols * myImg.rows;
		Mat Hist_myImg = showHist(myImg);

		//通道衰减
		outImg = channle(myImg);
		Mat Hist_outImg = showHist(outImg);

		outImg1 = simplestColorBalance(outImg, contrast11);
		Mat Hist_outImg2 = showHist(outImg1);

		//白平衡
		outImg2 = RGB(outImg1);
		Mat Hist_outImg1 = showHist(outImg2);

		//限制
		outImg_d1 = limite(outImg2);
		Mat Hist_outImg_d1 = showHist(outImg_d1);

		//拉伸
		CLAHE_image = CLAHE1(outImg_d1);
		Mat Hist_CLAHE_image = showHist(CLAHE_image);

		//伽马
		gama = 1.2;
		dst4 = gamacol1(CLAHE_image, gama);
		Mat Hist_dst4 = showHist(dst4);

		//高通融合
		outImg_g = gussfusing(dst4);
		Mat Hist_outImg_g = showHist(outImg_g);

		string new_name = write_path + format("%d", f) + ".png";//控制输出为4位整数并在前面补0

		imshow("原图", myImg);
		//imshow("通道衰减", outImg);
		////imshow("灰度世界", dst);
		//imshow("颜色校正", outImg1);
		//imshow("白平衡", outImg2);
		////imshow("色调映射", outImg3);
		//imshow("限制", outImg_d1);
		////imshow("拉伸", outImg_e);
		//imshow("CLAHE拉伸", CLAHE_image);
		////imshow("直方图拉伸", outImg_f);
		//imshow("伽马", dst4);   
		imshow("高通融合", outImg_g);

		imwrite(new_name, outImg_g);
		waitKey(0);

		outImg = Mat(myImg.rows, myImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		outImg1 = Mat(myImg.rows, myImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		outImg_d1 = Mat(myImg.rows, myImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		outImg_e = Mat(myImg.rows, myImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		dst4 = Mat(myImg.rows, myImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		outImg_g = Mat(myImg.rows, myImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
		Bsum = 0;
		Gsum = 0;
		Rsum = 0;
		Gnewsum = 0;

		Gau_Pyr.clear();
		lp_Pyr.clear();
		Gau_Pyr2.clear();
		lp_Pyr2.clear();
		maskGaussianPyramid.clear();
	}
}


Mat simplestColorBalance(cv::Mat src, float percent) {			// Simplest Color Balance
	vector<Mat_<uchar>> channel;
	split(src, channel);
	cv::Mat flat, result[3];
	for (int i = 0; i < 3; i++) {
		flat = channel[i].clone();													// Clone the matrix of each color channel
		flat = flat.reshape(0, 1);													// Reshape the matrix to one column
		cv::sort(flat, flat, SORT_EVERY_ROW + SORT_ASCENDING);						// Sort values from low to high
		int min = flat.at<uchar>(0, floor(flat.cols * percent / 100.0));			// Minimum boundary
		int max = flat.at<uchar>(0, ceil(flat.cols * (1.0 - percent / 100.0)));		// Maximum boundary
		result[i] = (channel[i] - min) * 255.0 / (max - min);						// Pixel remapping				// CHANGE 255 to max
	}
	cv::Mat balanced;
	merge(result, 3, balanced);
	return balanced;
}


Mat gamacol1(Mat image_roi, double gama1)
{
	for (int q = 0; q < image_roi.rows; q++)
	{
		for (int w = 0; w < image_roi.cols; w++)
		{
			B = (double(image_roi.at<Vec3b>(q, w)[0])) / 255.2f;
			G = (double(image_roi.at<Vec3b>(q, w)[1])) / 255.2f;
			R = (double(image_roi.at<Vec3b>(q, w)[2])) / 255.2f;

			R_new = pow(R, gama);
			G_new = pow(G, gama);
			B_new = pow(B, gama);

			//cout << R_new << "   " << G_new << "  " << B_new << endl;

			image_roi.at<Vec3b>(q, w)[0] = B_new * 255;
			image_roi.at<Vec3b>(q, w)[1] = G_new * 255;
			image_roi.at<Vec3b>(q, w)[2] = R_new * 255;
			//cout << B <<"   "<< G<<"    "<< R << endl;
		}
	}
	return image_roi;
}


double clip(double dd, double min, double max)
{
	if (dd < min)
	{
		dd = min;
	}
	else {
		if (dd > max) {

			dd = max;
		}
		else {
			dd = dd;
		}
	}
	return dd;
}


Mat channle(Mat myImg)
{
	//通道衰减***************************************************************************************************
	outImg = Mat(myImg.rows, myImg.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	//int num = (myImg.rows) * (myImg.cols);

	for (i = 0; i < myImg.rows; i++)
	{
		for (j = 0; j < myImg.cols; j++)
		{
			// 读取每一点RGB
			B = (double(myImg.at<Vec3b>(i, j)[0])) / 255.2f;
			G = (double(myImg.at<Vec3b>(i, j)[1])) / 255.2f;
			R = (double(myImg.at<Vec3b>(i, j)[2])) / 255.2f;

			Bsum = Bsum + B;
			Gsum = Gsum + G;
			Rsum = Rsum + R;
		}
	}
	Bavg = Bsum / double(num);
	Gavg = Gsum / double(num);
	Ravg = Rsum / double(num);

	//cout << Bavg << "  " << Gavg << "  " << Ravg << "  " << num << endl;

	for (l = 0; l < myImg.rows; l++)
	{
		for (p = 0; p < myImg.cols; p++)
		{
			B = (double(myImg.at<Vec3b>(l, p)[0])) / 255.2f;
			G = (double(myImg.at<Vec3b>(l, p)[1])) / 255.2f;
			R = (double(myImg.at<Vec3b>(l, p)[2])) / 255.2f;

			//B
			if (Gavg > Bavg)
			{
				//a = (Gavg - Bavg)*10;
				B_new = B + (a * (1 - B) * (Gavg - Bavg) * G);
				G_new = G;
				Gnewsum = Gnewsum + G_new;
				outImg.at<Vec3b>(l, p)[0] = B_new * 255;
				outImg.at<Vec3b>(l, p)[1] = G_new * 255;
			}

			//G
			if (Bavg >= Gavg)
			{
				//a = (Bavg - Gavg) * 10;
				G_new = (G + (a * (1 - G) * (Bavg - Gavg) * B));
				B_new = B;
				Gnewsum = Gnewsum + G_new;

				B_new = B_new * 255;
				outImg.at<Vec3b>(l, p)[0] = B_new;
				G_new = G_new * 255;
				outImg.at<Vec3b>(l, p)[1] = G_new;
			}
		}
		//cout << "a=" << a << endl;
	}

	//cout << Gnewsum << endl;
	Gnewavg = Gnewsum / (double)num;
	//cout << Gnewavg << endl;

	for (int q = 0; q < myImg.rows; q++)
	{
		for (int w = 0; w < myImg.cols; w++)
		{
			B = (double(myImg.at<Vec3b>(q, w)[0])) / 255.2f;
			G = (double(myImg.at<Vec3b>(q, w)[1])) / 255.2f;
			R = (double(myImg.at<Vec3b>(q, w)[2])) / 255.2f;

			G_new = outImg.at<Vec3b>(q, w)[1];

			//R
			R_new = (R + (b * (1 - R) * (Gnewavg - Ravg) * G_new / 255.2f)) * 255;
			//cout << R_new << endl;
			outImg.at<Vec3b>(q, w)[2] = R_new;
			//cout << B_new << "   " << G_new << "    " << R_new << endl;
		}
	}

	Bsum = 0;
	Gsum = 0;
	Rsum = 0;
	Gnewsum = 0;
	return outImg;

}


Mat RGB(Mat outImg)
{
	//白平衡*******************************************************************************
	cv::Mat RGB, LAB, chanBGR[3], chanLAB[3];

	split(outImg, chanBGR);
	Scalar meanR, stddevR, meanG, stddevG, meanB, stddevB;
	meanStdDev(chanBGR[0], meanB, stddevB);
	meanStdDev(chanBGR[1], meanG, stddevG);
	meanStdDev(chanBGR[2], meanR, stddevR);

	cout << meanB / 255.2f << endl;
	cout << meanG / 255.2f << endl;
	cout << meanR / 255.2f << endl;
	double maxavg, maxsum;
	//double Rsum, Gsum, Bsum;
	if (meanB[0] > meanG[0])
	{
		maxavg = meanB[0];
	}
	else
	{
		maxavg = meanG[0];
	}

	if (meanR[0] > maxavg)
	{
		maxavg = meanR[0];
	}
	maxavg = maxavg / num / 255.2f;
	maxsum = maxavg * num;

	//cout << "num:  " << num << endl; 
	cout << "maxavg:  " << maxavg << endl;
	cout << "maxsum:  " << maxsum << endl;
	Rsum = meanR[0] * num / 255.2f;
	Gsum = meanG[0] * num / 255.2f;
	Bsum = meanB[0] * num / 255.2f;

	cout << "Rsum:  " << Rsum << endl;
	cout << "Gsum:  " << Gsum << endl;
	cout << "Bsum:  " << Bsum << endl;

	cout << stddevB * stddevB / 255.2f / 255.2f << endl;
	cout << stddevG * stddevG / 255.2f / 255.2f << endl;
	cout << stddevR * stddevR / 255.2f / 255.2f << endl;

	double stdavg = (stddevB[0] * stddevB[0] / 255.2f / 255.2f + stddevG[0] * stddevG[0] / 255.2f / 255.2f + stddevR[0] * stddevR[0] / 255.2f / 255.2f) / 3;
	cout << "stdavg:  " << stdavg << endl;
	Mat outImg1 = outImg.clone();

	if (stdavg <= 0.02)
	{
		cout << "白平衡完成" << endl;
		//double B, G, R;
		//double B_new, G_new, R_new;
		for (int i = 0; i < outImg.rows; i++)
		{
			for (int j = 0; j < outImg.cols; j++)
			{
				B = (double(outImg.at<Vec3b>(i, j)[0])) / 255.2f;
				G = (double(outImg.at<Vec3b>(i, j)[1])) / 255.2f;
				R = (double(outImg.at<Vec3b>(i, j)[2])) / 255.2f;
				//cout << "B:   " << B << endl;
				outImg1.at<Vec3b>(i, j)[0] = ((maxsum - Bsum) / (maxsum + Bsum) * maxavg + B) * 255;
				outImg1.at<Vec3b>(i, j)[1] = ((maxsum - Gsum) / (maxsum + Gsum) * maxavg + G) * 255;
				outImg1.at<Vec3b>(i, j)[2] = ((maxsum - Rsum) / (maxsum + Rsum) * maxavg + R) * 255;
				/*		cout << "B_new:   " << B_new / 255.2f << endl;*/
			}
		}
		cout << "白平衡已完成。" << endl;
	}

	return outImg1;
}

Mat limite(Mat outImg1)
{
	////对比度限制**********[1，2]*********************************************************************************************************
	dst1 = outImg1.clone();
	Mat outImg_d1(dst1.rows, dst1.cols, CV_8UC3, Scalar(0, 0, 0));

	int num = dst1.rows * dst1.cols; // 像素数量
	vector<double> arr_r(num), arr_g(num), arr_b(num); // 改用 vector 代替全局数组

	double d1 = 0.001; // 对比度裁剪参数

	// 1️⃣ 提取并归一化像素值
	for (int q = 0; q < dst1.rows; q++) {
		for (int w = 0; w < dst1.cols; w++) {
			int idx = q * dst1.cols + w; // 计算索引
			Vec3b pixel = dst1.at<Vec3b>(q, w);
			arr_b[idx] = pixel[0] / 255.0;
			arr_g[idx] = pixel[1] / 255.0;
			arr_r[idx] = pixel[2] / 255.0;
		}
	}

	// 2️⃣ 排序（从大到小）
	sort(arr_r.begin(), arr_r.end(), greater<double>());
	sort(arr_g.begin(), arr_g.end(), greater<double>());
	sort(arr_b.begin(), arr_b.end(), greater<double>());

	// 3️⃣ 计算裁剪阈值
	double r_min = arr_r[int(num * (1 - d1))];
	double r_max = arr_r[int(num * d1)];
	double g_min = arr_g[int(num * (1 - d1))];
	double g_max = arr_g[int(num * d1)];
	double b_min = arr_b[int(num * (1 - d1))];
	double b_max = arr_b[int(num * d1)];

	// 4️⃣ 重新调整图像像素
	for (int q = 0; q < dst1.rows; q++) {
		for (int w = 0; w < dst1.cols; w++) {
			Vec3b pixel = dst1.at<Vec3b>(q, w);
			double B = pixel[0] / 255.0;
			double G = pixel[1] / 255.0;
			double R = pixel[2] / 255.0;

			// 进行裁剪
			R = clip(R, r_min, r_max);
			G = clip(G, g_min, g_max);
			B = clip(B, b_min, b_max);

			outImg_d1.at<Vec3b>(q, w)[0] = static_cast<uchar>(B * 255);
			outImg_d1.at<Vec3b>(q, w)[1] = static_cast<uchar>(G * 255);
			outImg_d1.at<Vec3b>(q, w)[2] = static_cast<uchar>(R * 255);
		}
	}

	return outImg_d1;
}


Mat gussfusing(Mat dst4)
{
	////高通融合****[3，10]***************************************************************************************************
	/*dst4 = dst4.clone();*/
	Mat outImg_m = Mat(dst4.rows, dst4.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	Mat outImg_gauss = Mat(dst4.rows, dst4.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	Mat outImg_g = Mat(outImg_m.rows, outImg_m.cols, CV_8UC3, cv::Scalar(0, 0, 0));

	double g = 0.5;  //确定
	Mat gg = Mat(outImg_m.rows, outImg_m.cols, CV_8UC3, cv::Scalar(int(g * 255), int(g * 255), int(g * 255)));
	double B_m, G_m, R_m;
	int blur = 9;
	GaussianBlur(dst4, outImg_gauss, Size(blur, blur), 0, 0);

	outImg_m = dst4 - outImg_gauss + gg;
	//imshow("Gussa", outImg_m);

	for (int q = 0; q < outImg_m.rows; q++)
	{
		for (int w = 0; w < outImg_m.cols; w++)
		{
			B_m = (double(outImg_m.at<Vec3b>(q, w)[0])) / 255.2f;
			G_m = (double(outImg_m.at<Vec3b>(q, w)[1])) / 255.2f;
			R_m = (double(outImg_m.at<Vec3b>(q, w)[2])) / 255.2f;

			B = (double(dst4.at<Vec3b>(q, w)[0])) / 255.2f;
			G = (double(dst4.at<Vec3b>(q, w)[1])) / 255.2f;
			R = (double(dst4.at<Vec3b>(q, w)[2])) / 255.2f;

			if (B_m < g) {
				B_new = (B * B_m / g) * 255;
			}
			else {
				B_new = (1 - ((1 - B) * (1 - B_m) / g)) * 255;
			}

			if (G_m < g) {
				G_new = (G * G_m / g) * 255;
			}
			else {
				G_new = (1 - ((1 - G) * (1 - G_m) / g)) * 255;
			}

			if (R_m < g) {
				R_new = (R * R_m / g) * 255;
			}
			else {
				R_new = (1 - ((1 - R) * (1 - R_m) / g)) * 255;
			}

			//cout << B_new <<"   "<< G_new <<"    "<< R_new << endl;
			outImg_g.at<Vec3b>(q, w)[0] = B_new;
			outImg_g.at<Vec3b>(q, w)[1] = G_new;
			outImg_g.at<Vec3b>(q, w)[2] = R_new;
		}
	}
	return outImg_g;
}


cv::Mat histStretch(cv::Mat src, float percent, int direction) {
	cv::Mat histogram;
	getHistogram(&src, &histogram);
	float percent_sum = 0.0, channel_min = -1.0, channel_max = -1.0;
	float percent_min = percent / 100.0, percent_max = 1.0 - percent_min;
	int i = 0;

	while (percent_sum < percent_max * src.total()) {
		if (percent_sum < percent_min * src.total()) channel_min++;
		percent_sum += histogram.at<float>(i, 0);
		channel_max++;
		i++;
	}

	cv::Mat dst;
	if (direction == 0) dst = (src - channel_min) * (255.0 - channel_min) / (channel_max - channel_min) + channel_min;	// Stretches the channel towards the Upper side
	else if (direction == 2) dst = (src - channel_min) * channel_max / (channel_max - channel_min);						// Stretches the channel towards the Lower side
	else dst = (src - channel_min) * 255.0 / (channel_max - channel_min);												// Stretches the channel towards both sides
	dst.convertTo(dst, CV_8UC1);
	return dst;
}

void getHistogram(cv::Mat* channel, cv::Mat* hist) {								// Computes the histogram of a single channel
	int histSize = 256;
	float range[] = { 0, 256 };														// The histograms ranges from 0 to 255
	const float* histRange = { range };
	cv::calcHist(channel, 1, 0, Mat(), *hist, 1, &histSize, &histRange, true, false);
}


Mat CLAHE1(Mat inputImag)
{

	//if (inputImage.empty())
	//{
	//    std::cout << "Could not open or find the image!" << std::endl;
	//}

	// 将输入图像转换为Lab颜色空间
	Mat labImage;
	cvtColor(inputImag, labImage, COLOR_BGR2Lab);

	// 分离Lab图像的通道
	std::vector<Mat> labChannels(3);
	split(labImage, labChannels);

	// 对L通道应用CLAHE
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(1.2); // 对比度限制
	clahe->setTilesGridSize(Size(4, 4)); // 均衡化窗口大小
	clahe->apply(labChannels[0], labChannels[0]);

	// 合并Lab通道
	Mat outputLabImage;
	merge(labChannels, outputLabImage);

	// 将处理后的Lab图像转换回BGR颜色空间
	Mat outputImage;
	cvtColor(outputLabImage, outputImage, COLOR_Lab2BGR);

	resize(outputImage, outputImage, inputImag.size(), 0, 0, INTER_LINEAR);

	// 显示输入和输出图像
	//namedWindow("Input Image", WINDOW_NORMAL);
	//namedWindow("CLAHE Output", WINDOW_NORMAL);
	//imshow("Input Image", inputImag);
	//imshow("CLAHE Output", outputImage);

	//waitKey(0);
	return outputImage;
}


Mat showHist(Mat& img)
{
	//1、创建3个矩阵来处理每个通道输入图像通道。
	//我们用向量类型变量来存储每个通道，并用split函数将输入图像划分成3个通道。
	vector<Mat>bgr;
	split(img, bgr);

	//2、定义直方图的区间数
	int numbers = 256;

	//3、定义变量范围并创建3个矩阵来存储每个直方图
	float range[] = { 0,256 };
	const float* histRange = { range };
	Mat b_hist, g_hist, r_hist;

	//4、使用calcHist函数计算直方图
	int numbins = 256;
	calcHist(&bgr[0], 1, 0, Mat(), b_hist, 1, &numbins, &histRange);
	calcHist(&bgr[1], 1, 0, Mat(), g_hist, 1, &numbins, &histRange);
	calcHist(&bgr[2], 1, 0, Mat(), r_hist, 1, &numbins, &histRange);

	//5、创建一个512*300像素大小的彩色图像，用于绘制显示
	int width = 512;
	int height = 300;
	Mat histImage(height, width, CV_8UC3, Scalar(255, 255, 255));

	//6、将最小值与最大值标准化直方图矩阵
	normalize(b_hist, b_hist, 0, height, NORM_MINMAX);
	normalize(g_hist, g_hist, 0, height, NORM_MINMAX);
	normalize(r_hist, r_hist, 0, height, NORM_MINMAX);

	//7、使用彩色通道绘制直方图
	int binStep = cvRound((float)width / (float)numbins);  //通过将宽度除以区间数来计算binStep变量

	for (int i = 1; i < numbins; i++)
	{
		line(histImage,
			Point(binStep * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),

			Point(binStep * (i), height - cvRound(b_hist.at<float>(i))),
			Scalar(255, 0, 0)
		);

		line(histImage,
			Point(binStep * (i - 1), height - cvRound(b_hist.at<float>(i - 1))),

			Point(binStep * (i - 1), height),
			Scalar(255, 0, 0)
		);

		line(histImage,
			Point(binStep * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			Point(binStep * (i - 1), height - cvRound(g_hist.at<float>(i))),
			Scalar(0, 255, 0)
		);

		line(histImage,
			Point(binStep * (i - 1), height - cvRound(g_hist.at<float>(i - 1))),
			Point(binStep * (i), height),
			Scalar(0, 255, 0)
		);

		line(histImage,
			Point(binStep * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			Point(binStep * (i - 1), height - cvRound(r_hist.at<float>(i))),
			Scalar(0, 0, 255)
		);

		line(histImage,
			Point(binStep * (i - 1), height - cvRound(r_hist.at<float>(i - 1))),
			Point(binStep * (i - 1), height),
			Scalar(0, 0, 255)
		);

	}
	return histImage;
}