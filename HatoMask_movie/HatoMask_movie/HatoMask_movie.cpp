#include <iostream>
#include <opencv2/opencv.hpp>



using namespace std;
using namespace cv;

void overlayImage(Mat* src, Mat* overlay, const Point& location);

int main(int argc, char* argv[])
{
	//Mat dnnimg = imread("base7.png");
	//Mat dnnimg = imread("lena.jpg");
	string file_name = "Megamind.avi";
	if (argc > 1) {
		file_name = argv[1];
	}
	cv::VideoCapture cap;
	cap.open(file_name);
	if (cap.isOpened() == false) {
		std::cout << "movie file open failed\n";
		return -1;
	}

	int    fourcc, width, height;
	double fps;

	width = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);	
	height = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);	
	fps = cap.get(cv::CAP_PROP_FPS);					
	fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
	//std::cout << width << ", " << height << ", " << fps << ", " << fourcc << std::endl;

	string output_fname = file_name + string(".avi");
	cv::VideoWriter writer;
	writer.open(output_fname, fourcc, fps, cv::Size(width, height));
	if (writer.isOpened() == false) {
		std::cout << "movie output file open failed\n";
		return -1;
	}
	cv::Mat frame, dst;
	Mat kao = imread("kao.png", IMREAD_UNCHANGED);
	
	for (;;) {
		
		cap >> frame;
		if (frame.empty() == true) {
			break;
		}
		Mat dnnimg = frame;//imread(file_name);
		
		const std::string caffeConfigFile = "./deploy.prototxt";
		const std::string caffeWeightFile = "./res10_300x300_ssd_iter_140000_fp16.caffemodel";

		cv::dnn::Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);
		cv::Mat inputBlob = cv::dnn::blobFromImage(dnnimg, 1.0, cv::Size(300, 300));


		net.setInput(inputBlob, "data");
		cv::Mat detection = net.forward("detection_out");

		cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());
		float confidenceThreshold = 0.3;
		const double scale_factor = 0.4;
		cvtColor(dnnimg, dnnimg, COLOR_BGR2BGRA);
		for (int i = 0; i < detectionMat.rows; i++)
		{
			float confidence = detectionMat.at<float>(i, 2);
			if (confidence > confidenceThreshold)
			{
				int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * dnnimg.cols);
				int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * dnnimg.rows);
				int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * dnnimg.cols);
				int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * dnnimg.rows);
				int width = x2 - x1;
				int height = y2 - y1;
				//std::cout << x1 << ", " << y1 << ", " << x2 << ", " << y2 << std::endl;
				//cv::rectangle(dnnimg, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
				int cy = y1 - height * scale_factor;
				int ch = height + height * scale_factor;
				int cydiff1 = 0;
				int cydiff2 = 0;
				if (cy < 0) {
					cydiff1 = cy;
				}
				if (cy + ch > dnnimg.rows) {
					cydiff2 = dnnimg.rows - (cy + ch);
				}
				cy = std::max(0, cy);

				ch += cydiff1 + cydiff2;

				int cx = x1 - (width * scale_factor / 2);
				int ccx = x1 - (width * scale_factor);

				int cw = width + width * scale_factor;
				int cxdiff1 = 0;
				int cxdiff2 = 0;
				if (cx < 0) {
					cxdiff1 = cx;
				}
				if (cx + cw > dnnimg.cols) {
					cxdiff2 = dnnimg.cols - (cx + cw);
				}
				//std::cout << cxdiff1 << ", " << cxdiff2 << std::endl;
				cx = std::max(0, cx);

				cw += cxdiff1 + cxdiff2;


				//std::cout << cx << ", " << cy << ", " << cw << ", " << ch << std::endl;
				//std::cout << x1 << ", " << y1 << ", " << width << ", " << height << std::endl;
				Mat resized_kao;
				resize(kao, resized_kao, Size(cw, ch));
				//imshow("kao", resized_kao);
				Mat roi = dnnimg(Rect(cx, cy, resized_kao.cols, resized_kao.rows));
				overlayImage(&dnnimg, &resized_kao, Point(cx, cy));
			}
		}
		cvtColor(dnnimg, dnnimg, COLOR_BGRA2BGR);
		writer << dnnimg;
		imshow("mov", dnnimg);
		int key = waitKey(15);
		if (key == 27) {
			break;
		}
		//writer.write(dnnimg);
		
	}


	//imshow("dnnimg", dnnimg);
	//string output_fname = file_name + string(".jpg");
	//imwrite(output_fname, dnnimg);
	//waitKey(0);

	return 0;
}

void overlayImage(Mat * src, Mat * overlay, const Point & location)
{
	for (int y = max(location.y, 0); y < src->rows; ++y)
	{
		int fY = y - location.y;

		if (fY >= overlay->rows)
			break;

		for (int x = max(location.x, 0); x < src->cols; ++x)
		{
			int fX = x - location.x;

			if (fX >= overlay->cols)
				break;

			double opacity = ((double)overlay->data[fY * overlay->step + fX * overlay->channels() + 3]) / 255;

			for (int c = 0; opacity > 0 && c < src->channels(); ++c)
			{
				unsigned char overlayPx = overlay->data[fY * overlay->step + fX * overlay->channels() + c];
				unsigned char srcPx = src->data[y * src->step + x * src->channels() + c];
				src->data[y * src->step + src->channels() * x + c] = srcPx * (1. - opacity) + overlayPx * opacity;
			}
		}
	}
}