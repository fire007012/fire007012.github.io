#include<opencv2/opencv.hpp>
#include<iostream>
using namespace cv;
using namespace std;
class CascadeDetectorAdapter : public DetectionBasedTracker::IDetector
{
public:
	CascadeDetectorAdapter(cv::Ptr<cv::CascadeClassifier> detector) :
		IDetector(),
		Detector(detector)
	{
		CV_Assert(detector);
	}

	void detect(const cv::Mat& Image, std::vector<cv::Rect>& objects)
	{
		Detector->detectMultiScale(Image, objects, scaleFactor, minNeighbours, 0, minObjSize, maxObjSize);
	}

	virtual ~CascadeDetectorAdapter()
	{
	}

private:
	CascadeDetectorAdapter();
	cv::Ptr<cv::CascadeClassifier> Detector;
};

cv::Ptr<DetectionBasedTracker> tracker;

int main()
{
	string stdFileName = "D:/opencv/build/etc/haarcascades/haarcascade_frontalface_alt.xml";
	//创建一个主检测适配器
	cv::Ptr<CascadeDetectorAdapter> mainDetector = makePtr<CascadeDetectorAdapter>(
		makePtr<CascadeClassifier>(stdFileName));
	//创建一个跟踪检测适配器
	cv::Ptr<CascadeDetectorAdapter> trackingDetector = makePtr<CascadeDetectorAdapter>(
		makePtr<CascadeClassifier>(stdFileName));
	//创建跟踪器
	DetectionBasedTracker::Parameters DetectorParams;
	tracker = makePtr<DetectionBasedTracker>(mainDetector, trackingDetector, DetectorParams);
	tracker->run();

	VideoCapture capture;
	capture.open(0);
	if (!capture.isOpened())
	{
		cout << "opencv打开摄像头失败!\n" << endl;
		return -1;
	}
	Mat frame; //摄像头彩色图像
	Mat grayFrame; //摄像头灰度图像
	Mat equalizeFrame; //直方图
	while (true)
	{
		capture >> frame; //从capture中取数据，将画面输出到frame矩阵里面
		if (frame.empty())
		{
			cout << "读取摄像头数据失败!\n" << endl;
			return -1;
		}
		//imshow("摄像头", frame); //显示图像
		//灰度化处理
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY); //注意 : OpenCV中是BRG
		//imshow("灰度化", grayFrame); //显示图像
		//直方图均衡化，用来增强图像对比度，从而让轮廓更加明显
		equalizeHist(grayFrame, equalizeFrame);
		//imshow("直方图", equalizeFrame);

		std::vector<Rect>  faces;

		tracker->process(grayFrame);
		tracker->getObjects(faces);

		for (size_t i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(0, 0, 255));
		}

		imshow("摄像头", frame); //显示图像


		if (waitKey(30) == 27) //ESC键
		{
			break;
		}
	}
	tracker->stop();
	return 0;
}