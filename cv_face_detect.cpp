#include <highgui.hpp>
#include <imgproc.hpp>
#include <core.hpp>
#include <objdetect.hpp>
#include <cv.h>
#include <algorithm>
#include <iostream>
#include <fstream>

using namespace cv;
using namespace std;

const char* frontalPath = "haarcascades/haarcascade_frontalface_alt2.xml";  //XML for face detection

char* input_filename = NULL;
const char* output_filename = NULL;

int final_width = 25, final_height = 25;


void draw(Mat& img)
{
    imshow("test", img);
    waitKey();
}

void capture_realtime()
{
    Mat frame;
    VideoCapture cap(0);
    cap >> frame;
    cap.release();
    draw(frame);
}

bool cmp(Rect& l, Rect& r)
{
    return l.area() < r.area();
}

void save_mat(Mat& mat, const char* filename)
{
    char* temp = new char[mat.cols * mat.rows];
    int zero_count = 0;
    for(int i = 0; i < mat.rows; i ++)
    {
        for(int j = 0; j < mat.cols; j ++)
        {
            temp[i * mat.cols + j] = mat.at<uchar>(j, i);
            if(!temp[i * mat.cols + j])
                zero_count ++;
        }
    }
    FILE *fp = fopen(filename, "wb");
    fwrite(temp, sizeof(char), mat.cols * mat.rows, fp);
    fclose(fp);

    delete []temp;
    //printf("Zero Rate: %.2f%%\n", 100.0 * zero_count / mat.cols * mat.rows);
}

void image_convert(Mat& face)
{
    Mat sobels, grad_x, grad_y, agrad_x, agrad_y, finals, sobels2;
    cvtColor(face, sobels, CV_BGR2GRAY);
    Sobel(sobels, grad_x, CV_16S, 1, 0); //x-grad
    Sobel(sobels, grad_y, CV_16S, 0, 1); //y-grad
    convertScaleAbs(grad_x, agrad_x);
    convertScaleAbs(grad_y, agrad_y);

    //addWeighted(agrad_x, 0.5, agrad_y, 0.5, 0, sobels);
    Size final_rect(final_width, final_height);

    resize(sobels, finals, final_rect);

    //imshow("c0", finals);
    //waitKey();

    save_mat(finals, output_filename);
/*
    int count = 16;
    for(int i = 3; i < 7; i ++)
    {
        for(int j = 3; j < 7; j ++)
        {
            Rect rect;
            Mat tmp;
            rect.x = i * 10;
            rect.y = j * 10;
            rect.width = rect.height = 10;
            finals(rect).copyTo(tmp);
            char bmpName[256];
            sprintf(bmpName, "pieces\\%d.bmp", count);
            imwrite(bmpName, tmp);
            sprintf(bmpName, "pieces\\%d.dat", count);
            save_mat(tmp, bmpName);
            count ++;
        }
    }*/
    imwrite("result.bmp", finals);

}

Mat input_image()
{
    VideoCapture cap(input_filename);
    Mat frame;
    cap >> frame;
    cap.release();
    return frame;
}

Mat input_realtime_image()
{
    VideoCapture cap(0);
    Mat frame, gray, finals;

    CascadeClassifier cascade;
    cascade.load(frontalPath);
    vector<Rect> objs;
    cap >> frame;
    cvtColor(frame, gray, CV_BGR2GRAY);
    cascade.detectMultiScale(gray, objs);
    while(objs.size() == 0)
    {
        cap >> frame;
        cvtColor(frame, gray, CV_BGR2GRAY);
        cascade.detectMultiScale(gray, objs);
        imshow("capture", frame);
    }
    cap.release();
    sort(objs.begin(), objs.end(), cmp);
    for(int i = 0; i < objs.size(); i ++)
    {
        //rectangle(frame, objs[i], Scalar(0, 0, 100), 2);
        frame(objs[i]).copyTo(finals);
        printf("%d, %d, %d, %d\n", objs[i].x, objs[i].y, objs[i].width, objs[i].height);
    }
    if(objs.size() > 0)
        rectangle(frame, objs[objs.size() - 1], Scalar(0, 0, 100), 2);
    imwrite("detected_result.bmp", frame);

    return finals;
}

Mat detect_face(Mat& input)
{
    Mat gray, finals;
    cvtColor(input, gray, CV_BGR2GRAY);

    CascadeClassifier cascade;
    cascade.load(frontalPath);
    vector<Rect> objs;
    cascade.detectMultiScale(gray, objs);
    sort(objs.begin(), objs.end(), cmp);
    for(int i = 0; i < objs.size(); i ++)
    {
        //rectangle(input, objs[i], Scalar(0, 0, 100), 2);
        input(objs[i]).copyTo(finals);
        printf("%d, %d, %d, %d\n", objs[i].x, objs[i].y, objs[i].width, objs[i].height);
    }
    /*
    imshow("c", input);
    waitKey();
    */
    if(objs.size() == 0)
    {
        printf("no face detected.\n");
        input.copyTo(finals);
    }

    if(objs.size() > 0)
        rectangle(input, objs[objs.size() - 1], Scalar(0, 0, 100), 2);
    imwrite("detected_result.bmp", input);
    return finals;
}

void initialize()
{

}

int main(int argc, char* argv[])
{


    if(argc == 3)
    {
        input_filename = argv[1];
        output_filename = argv[2];

        /*

        input_filename = "1.bmp";
        output_filename = "1.dat";
    */
        initialize();
        Mat source = input_image();
        Mat face = detect_face(source);
        image_convert(face);

        return 0;

    }
    else if(argc == 2)
    {
        output_filename = argv[1];

        /*

        input_filename = "1.bmp";
        output_filename = "1.dat";
    */
        initialize();
        Mat face = input_realtime_image();
        image_convert(face);
        return 0;
    }

/*

        input_filename = "orl_dataset\\s2\\1.bmp";
        output_filename = "2.dat";

        initialize();
        Mat source = input_image();
        Mat face = detect_face(source);
        sobel_convert(face);
*/
    printf("用法: 1. preop.exe [input_image] [output_data]\n");
    printf("用法: 2. preop.exe [output_data] 实时采集\n");
    printf("本程序用于识别出图片中的人脸，并将脸部的sobel化数据保存为%d*%d的矩阵\n", final_width, final_height);
    printf("注意:本程序不能识别参数中的空格");

    return 0;
}
