//
// Created by lidongxuan on 2017/5/7.
//
#include <io.h> 
#include <iostream>
#include <fstream>
#include <sstream>
#include <opencv2/opencv.hpp>

#include "gender_train.h"
#include "nearsighted_glasses_train.h"
#include "hat_train.h"

using namespace std;
using namespace cv;

int image_height = 112;
int image_width = 92;

void get_image_files(string path, vector<string>& files);
void gender_detection(CvSVM& gender, Mat image, String& result);
void eyeglasses_detection(CascadeClassifier& sun_glasses, CascadeClassifier& sun_glasses_2, CascadeClassifier& sun_glasses_3, CvSVM& nearsighted_glasses_SVM, Ptr<FaceRecognizer>& nearsighted_glasses_LBP, Mat image, String& result, String& color);
void surgical_mask_detection(CascadeClassifier& nose, CascadeClassifier& mouth, Mat image, bool tryflip, String& result, String& color);
void hat_detection(Ptr<FaceRecognizer>& hat_PCA, Ptr<FaceRecognizer>& hat_LBP, Mat image, String& result, String& color);
void color_detection(Mat image, String target, String& color);

// ������
int main()
{
	// ѵ������ģ�ͣ��ڲ��Ե�ʱ����Ҫִ��
	// gender_train();
	// nearsighted_glasses_train();
	// hat_train();

	// �������ļ�
	cout << "��ʼ���У����Ժ�..." << endl;
	ofstream result_file;
	string result_path = "../���.txt";
	result_file.open(result_path);
	String gender, eyeglasses, eyeglasses_color, surgical_mask, surgical_mask_color, hat, hat_color;

	// �������ĵ�
	String test_folder_path = "../test_images";
	String test_image_path;
	vector<String> test_images;
	get_image_files(test_folder_path, test_images);
	Mat testimage, grayimage;
	int test_num = (int)test_images.size();

	// �Ա��жϳ�ʼ��
	CvSVM gender_classifier;
	gender_classifier.load("./sources/gender_model.txt");

	// �����жϳ�ʼ��
	CascadeClassifier surgical_mask_classifier_1, surgical_mask_classifier_2;
	surgical_mask_classifier_1.load("./sources/haarcascade_mcs_nose.xml");
	surgical_mask_classifier_2.load("./sources/haarcascade_mcs_mouth.xml");

	// �۾��жϳ�ʼ��
	CascadeClassifier sun_glasses_classifier;
	sun_glasses_classifier.load("./sources/haarcascade_eye.xml");
	CascadeClassifier sun_glasses_classifier_2;
	sun_glasses_classifier_2.load("./sources/haarcascade_mcs_eyepair_small.xml");
	CascadeClassifier sun_glasses_classifier_3;
	sun_glasses_classifier_3.load("./sources/haarcascade_eye_tree_eyeglasses.xml");
	CvSVM nearsighted_glasses_SVM;
	nearsighted_glasses_SVM.load("./sources/nearsighted_glasses_model.txt");
	Ptr<FaceRecognizer> nearsighted_glasses_LBP = createLBPHFaceRecognizer();
	nearsighted_glasses_LBP->load("./sources/nearsighted_glasses_model_LBP.xml");

	// ñ���жϳ�ʼ��
	Ptr<FaceRecognizer> hat_LBP = createLBPHFaceRecognizer();
	hat_LBP->load("./sources/hat_model_LBP.xml");
	Ptr<FaceRecognizer> hat_PCA = createEigenFaceRecognizer();
	hat_PCA->load("./sources/hat_model_PCA.xml");
	//    Ptr<FaceRecognizer> hat_Fisher = createFisherFaceRecognizer();
	//    hat_Fisher->load("./sources/hat_model_Fisher.xml");


	for (int i = 0; i<test_num; i++)
	{
		cout << "���ڲ��Ե�" << i + 1 << "��ͼƬ..." << endl;
		
		test_image_path = test_folder_path + "/" + test_images[i];
		testimage = imread(test_image_path.c_str());
		resize(testimage, testimage, Size(image_width, image_height));// resize���ٷ�Ҫ���112*92

		// �Ա��⣬������ص��ַ���gender
		gender_detection(gender_classifier, testimage, gender);

		// �۾���⣬��������۾���ɫ���ص�eyeglasses��eyeglasses_color
		eyeglasses_detection(sun_glasses_classifier, sun_glasses_classifier_2, sun_glasses_classifier_3, nearsighted_glasses_SVM, nearsighted_glasses_LBP, testimage, eyeglasses, eyeglasses_color);

		// ���ּ�⣬������������ɫ���ص�surgical_mask��surgical_mask_color
		surgical_mask_detection(surgical_mask_classifier_1, surgical_mask_classifier_2, testimage, 0, surgical_mask, surgical_mask_color);

		// ñ�Ӽ�⣬ֻ������ͼƬ���ϰ벿�ֽ��м��
		hat_detection(hat_PCA, hat_LBP, testimage, hat, hat_color);

		//cout<<test_images[i]<< " "
		//	<< gender << " "
		//	<< eyeglasses << " "
		//	<< eyeglasses_color << " "
		//	<< surgical_mask << " "
		//	<< surgical_mask_color << " "
		//	<< hat << " "
		//	<< hat_color << endl;

		result_file << test_images[i] << " "
			<< gender << " "
			<< eyeglasses << " "
			<< eyeglasses_color << " "
			<< surgical_mask << " "
			<< surgical_mask_color << " "
			<< hat << " "
			<< hat_color << endl;
	}
	result_file.close();
	cout << "�������!" << std::endl;
	cout << "���Խ���ѱ���������ִ�г���Ŀ¼�µġ����.txt����!" << std::endl;
	return 0;
}

// ��ȡ�����ļ����ڵ�����ͼƬ�ļ���
void get_image_files(string path, vector<string>& files)
{

	long   hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*.jpg").c_str(), &fileinfo)) != -1
		|| (hFile = _findfirst(p.assign(path).append("\\*.jpeg").c_str(), &fileinfo)) != -1)//�����ҳɹ��������
	{
		do
		{
			//������ļ�������Ŀ¼  
			if (!(fileinfo.attrib &  _A_SUBDIR))
			{
				files.push_back(fileinfo.name);
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

//�Ա���
void gender_detection(CvSVM& gender, Mat image, String& result)
{
	cvtColor(image, image, CV_BGR2GRAY);
	resize(image, image, Size(64, 64));
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16),cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> descriptor;
	hog->compute(image, descriptor, Size(1, 1), Size(0, 0));

	Mat testHog;
	testHog = Mat::zeros(1, descriptor.size(), CV_32FC1);
	int n = 0;
	for (vector<float>::iterator iter = descriptor.begin(); iter != descriptor.end(); iter++)
	{
		testHog.at<float>(0, n) = *iter;
		n++;
	}
	if ((int)gender.predict(testHog) == 2) result = "Ů";
	else result = "��";
}

// �۾����
void eyeglasses_detection(CascadeClassifier& sun_glasses,
	CascadeClassifier& sun_glasses_2,
	CascadeClassifier& sun_glasses_3,
	CvSVM& nearsighted_glasses_SVM, 
	Ptr<FaceRecognizer>& nearsighted_glasses_LBP, 
	Mat image, 
	String& result, String& color)
{
	Mat grayimage, grayimage_LBP;
	cvtColor(image, grayimage_LBP, CV_BGR2GRAY);
	cvtColor(image, grayimage, CV_BGR2GRAY);
	resize(grayimage, grayimage, Size(64, 64));
	HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16),
		cvSize(8, 8), cvSize(8, 8), 9);
	vector<float> descriptor;
	hog->compute(grayimage, descriptor, Size(1, 1), Size(0, 0));

	Mat testHog;
	testHog = Mat::zeros(1, (int)descriptor.size(), CV_32FC1);
	int n = 0;
	for (vector<float>::iterator iter = descriptor.begin(); iter != descriptor.end(); iter++)
	{
		testHog.at<float>(0, n) = *iter;
		n++;
	}
	int SVM_result = (int)nearsighted_glasses_SVM.predict(testHog);

	vector<Rect> eye_result;
	sun_glasses.detectMultiScale(grayimage_LBP, eye_result, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(45, 45));
	int eye_num = (int)eye_result.size();

	vector<Rect> eye_result_2;
	sun_glasses_2.detectMultiScale(grayimage_LBP, eye_result_2, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(90, 90));
	int eye_num_2 = (int)eye_result_2.size();

	vector<Rect> eye_result_3;
	sun_glasses_3.detectMultiScale(grayimage_LBP, eye_result_3, 1.1, 1, 0 | CV_HAAR_SCALE_IMAGE, Size(10, 10), Size(45, 45));
	int eye_num_3 = (int)eye_result_3.size();

	int LBP_result = nearsighted_glasses_LBP->predict(grayimage_LBP);

	// cout << eye_num << "," << eye_num_2 << "," << eye_num_3 << endl;

	if (eye_num > 0 || eye_num_2>0 || eye_num_3>0)
	{
		if (SVM_result == 1 && LBP_result == 1)
		{
			result = "��";
			color = "͸��ɫ";
		}
		else
		{
			result = "��";
			color = "��";
		}
	}
	else
	{
		result = "��";
		color_detection(image, "eyeglasses", color);
	}

}

// ���ּ��
void surgical_mask_detection(CascadeClassifier& nose, CascadeClassifier& mouth, Mat image, bool tryflip, String& result, String& color)
{
	Mat grayimage;
	cvtColor(image, grayimage, CV_BGR2GRAY);
	vector<Rect> nose_result, mouth_result;

	nose.detectMultiScale(grayimage, nose_result, 1.1, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(15, 15), Size(45, 45));
	mouth.detectMultiScale(grayimage, mouth_result, 1.1, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(15, 15), Size(45, 45));


	//Rect mark;
	//for (vector<Rect>::const_iterator nr = nose_result.begin(); nr != nose_result.end(); nr++)
	//   {
	//	mark.x = nr->x;
	//	mark.y = nr->y;
	//	mark.width = nr->width;
	//	mark.height = nr->height;
	//	rectangle(grayimage, mark, (0, 0, 255), 3, 8, 0);
	//   }

	//����ʵ�鷢�ּ������ʱ���⵽�۾��ϣ������Ӽ��Ľ�����׼ȷ��˵�������ӵĿɿ��Ը��ߣ�����֮�⻹Ҫ�������λ�ã�̫�ߵĻ����۾�
	result = "��";
	color = "��";
	if (nose_result.size() == 0 && mouth_result.size()==0)
	{
		result = "��";
		color_detection(image, "surgical_mask", color);
	}
	else if (nose_result.size() == 0 && mouth_result.size() != 0)
	{
		if (mouth_result[0].y < 56)
		{
			result = "��";
			color_detection(image, "surgical_mask", color);
		}
	}
}

// ñ�Ӽ��
void hat_detection(Ptr<FaceRecognizer>& hat_PCA, Ptr<FaceRecognizer>& hat_LBP, Mat image, String& result, String& color)
{
	Mat grayimage;
	cvtColor(image, grayimage, CV_BGR2GRAY);
	grayimage = grayimage(Rect(0, 0, 92, 52));

	if ((hat_PCA->predict(grayimage) == 1) && (hat_LBP->predict(grayimage) == 1))
	{
		result = "��";
		color_detection(image, "hat", color);
	}
	else
	{
		result = "��";
		color = "��";
	}
}

void color_detection(Mat image, String target, String& color)
{
	int x = 1, y = 1, width = 1, height = 1;

	if (target == "hat")
	{
		x = 26; y = 6; width = 40; height = 12;
	}
	else if (target == "surgical_mask")
	{
		x = 26; y = 80; width = 40; height = 16;
	}
	else if (target == "eyeglasses")
	{
		CascadeClassifier nose_detector;
		nose_detector.load("../sources/haarcascade_mcs_nose.xml");
		Mat grayimage;
		cvtColor(image, grayimage, CV_BGR2GRAY);
		vector<Rect> nose_result;
		nose_detector.detectMultiScale(grayimage, nose_result, 1.1, 2, 0 | CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_SCALE_IMAGE, Size(20, 20), Size(45, 45));
		if (nose_result.size() == 0)
		{
			x = 20; y = 52; width = 16; height = 8;
		}
		else
		{
			x = abs(nose_result[0].x - nose_result[0].width / 2);
			y = abs(nose_result[0].y - nose_result[0].height / 2);
			width = abs(nose_result[0].width - 3);
			height = nose_result[0].height / 2;
		}

		//Rect mark;
		//mark.x = x;
		//mark.y = y;
		//mark.width = width;
		//mark.height = height;
		//rectangle(image, mark, (0, 0, 255), 3, 8, 0);
		//imshow("result", image);
		//waitKey(50);

	}
	else
	{
		cout << "color_detection target error!" << endl;
	}

	if (image.channels() == 1)
	{
		color = "��ɫ";
	}
	else
	{
		Mat hsv;
		cvtColor(image, hsv, CV_RGB2HSV);

		vector<cv::Mat> mv;
		Mat hsv_h, hsv_s, hsv_v;
		split(hsv, mv);
		hsv_h = mv.at(0);
		hsv_s = mv.at(1);
		hsv_v = mv.at(2);

		Mat matS1, matS2, matS3;
		Mat element_Se, element_Sd;
		threshold(hsv_s, matS1, 90, 255, CV_THRESH_BINARY);
		erode(matS1, matS2, element_Se);
		dilate(matS2, matS3, element_Sd);

		Mat matV1;
		threshold(hsv_v, matV1, 248, 255, CV_THRESH_BINARY);

		Mat matAdd;
		add(matS3, matV1, matAdd);

		Mat matColorH, matColorS, matColorV;
		Rect rectROI(x, y, width, height);
		hsv_h(rectROI).convertTo(matColorH, matColorH.type(), 1, 0);
		hsv_s(rectROI).convertTo(matColorS, matColorS.type(), 1, 0);
		hsv_v(rectROI).convertTo(matColorV, matColorV.type(), 1, 0);

		Scalar mean_h = mean(matColorH);
		Scalar mean_s = mean(matColorS);
		Scalar mean_v = mean(matColorV);
		double dH = mean_h[0];
		double dS = mean_s[0];
		double dV = mean_v[0];

		if (dH>0 && dH<180 && dS>0 && dS<255 && dV>0 && dV<46) color = "��ɫ";
		if (dH>0 && dH<180 && dS>0 && dS<43 && dV>46 && dV<220) color = "��ɫ";
		if (dH>0 && dH<180 && dS>0 && dS<30 && dV>221 && dV<255) color = "��ɫ";
		if (dH>0 && dH<10 && dS>43 && dS<255 && dV>46 && dV<255) color = "��ɫ";
		if (dH>156 && dH<180 && dS>43 && dS<255 && dV>46 && dV<255) color = "��ɫ";
		if (dH>11 && dH<25 && dS>43 && dS<255 && dV>46 && dV<255) color = "��ɫ";
		if (dH>26 && dH<34 && dS>43 && dS<255 && dV>46 && dV<255) color = "��ɫ";
		if (dH>35 && dH<77 && dS>43 && dS<255 && dV>46 && dV<255) color = "��ɫ";
		if (dH>78 && dH<99 && dS>43 && dS<255 && dV>46 && dV<255) color = "��ɫ";
		if (dH>100 && dH<124 && dS>43 && dS<255 && dV>46 && dV<255) color = "��ɫ";
		if (dH>125 && dH<155 && dS>43 && dS<255 && dV>46 && dV<255) color = "��ɫ";

	}
}