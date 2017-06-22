//
// Created by lidongxuan on 2017/5/9.
//

#include "hat_train.h"

using namespace std;
using namespace cv;

void read_txt_hat(String &csvPath, vector<String> &trainPath, vector<int> &label, char separator = ';')
{
	string line, path, classLabel;
	ifstream file(csvPath.c_str(), ifstream::in);
	while (getline(file, line))
	{
		stringstream lines(line);
		getline(lines, path, separator);
		getline(lines, classLabel);
		if (!path.empty() && !classLabel.empty())
		{
			trainPath.push_back(path);
			label.push_back(atoi(classLabel.c_str()));
		}
	}
}

void read_txt_hat_2(string& fileName, vector<Mat>& images, vector<int>& labels, char separator = ';')
{
	ifstream file(fileName.c_str(), ifstream::in);    
	String line, path, label;
	while (getline(file, line))                     
	{
		stringstream lines(line);
		getline(lines, path, separator);       
		getline(lines, label);
		if (!path.empty() && !label.empty())       
		{
			images.push_back(imread(path, CV_LOAD_IMAGE_GRAYSCALE));    
			labels.push_back(atoi(label.c_str())); 
		}
	}
}

void hat_train()
{
	cout << "训练hatSVM中..." << endl;
	string trainCsvPath = "./data/hat_data/at.txt";
	vector<String> vecTrainPath;
	vector<int> vecTrainLabel;
	read_txt_hat(trainCsvPath, vecTrainPath, vecTrainLabel);

	int iNumTrain = (int)vecTrainLabel.size();
	Mat trainDataHog;
	Mat trainLabel;
	trainLabel = Mat::zeros(iNumTrain, 1, CV_32FC1);

	Mat imageSrc;
	for (int i = 0; i < iNumTrain; i++)
	{
		imageSrc = imread(vecTrainPath[i].c_str(), 0);
		resize(imageSrc, imageSrc, Size(128, 128));
		// 检测帽子时关注图片的上半部分即可
		imageSrc = imageSrc(Rect(4, 0, 112, 64));
		//输入参数分别是：窗口大小，block size，block stride，cell size，360度内分隔的梯度方向数
		HOGDescriptor *hog = new HOGDescriptor(cvSize(64, 64), cvSize(16, 16),
			cvSize(8, 8), cvSize(8, 8), 9);
		vector<float> descriptor;
		//imageSrc代表输入的图片，descriptors表示保存特征结果的Vector，Size(1,1)表示windows的步进，第四个为padding，用于填充图片以适应大小
		hog->compute(imageSrc, descriptor, Size(1, 1), Size(0, 0));

		if (i == 0)
		{
			trainDataHog = Mat::zeros(iNumTrain, descriptor.size(), CV_32FC1);
		}

		// 将特征存入trainDataHog
		int n = 0;
		for (vector<float>::iterator iter = descriptor.begin(); iter != descriptor.end(); iter++)
		{
			trainDataHog.at<float>(i, n) = *iter;
			n++;
		}
		//存入label
		trainLabel.at<float>(i, 0) = vecTrainLabel[i];
	}

	CvSVM svm;
	CvSVMParams param;
	CvTermCriteria criteria;
	criteria = cvTermCriteria(CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
	param = CvSVMParams(CvSVM::C_SVC, CvSVM::RBF,
		10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL, criteria);

	svm.train(trainDataHog, trainLabel, Mat(), Mat(), param);
	svm.save("./sources/hat_model_SVM.txt");
	svm.save("../Release/sources/hat_model_SVM.txt");

	//第二种方法
	String csvPath = "./data/hat_data/at.txt";
	vector<Mat> images;
	vector<int> labels;
	read_txt_hat_2(csvPath, images, labels);
	int iNumTrain_ = (int)labels.size();
	for (int i = 0; i < iNumTrain_; i++)
	{
		resize(images[i], images[i], Size(100, 100));
		images[i] = images[i](Rect(4, 0, 92, 52));
	}

	cout << "训练hatLBP中..." << endl;
	Ptr<FaceRecognizer> modelLBP = createLBPHFaceRecognizer();
	modelLBP->train(images, labels);
	modelLBP->save("./sources/hat_model_LBP.xml");
	modelLBP->save("../Release/sources/hat_model_LBP.xml");

	cout << "训练hatFisher中..." << endl;
	Ptr<FaceRecognizer> modelFisher = createFisherFaceRecognizer(); 
	modelFisher->train(images, labels);
	modelFisher->save("./sources/hat_model_Fisher.xml");
	modelFisher->save("../Release/sources/hat_model_Fisher.xml");

	cout << "训练hatPCA中..." << endl;
	Ptr<FaceRecognizer> modelPCA = createEigenFaceRecognizer();
	modelPCA->train(images, labels);
	modelPCA->save("./sources/hat_model_PCA.xml");
	modelPCA->save("../Release/sources/hat_model_PCA.xml");
}