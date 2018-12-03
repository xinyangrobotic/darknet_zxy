#include <iostream>
#include "darknet.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <fstream>
#include <boost/timer.hpp>

using namespace std;
using namespace cv;

list * g_options;
network * g_net;
float g_thresh = 0.5,g_hier_thresh = 0.5, g_nms = 0.45;


image color_mat_to_image(const cv::Mat &src)
{
    int h = src.rows;
    int w = src.cols;
    int c = src.channels();

    image out = make_image(w, h, c);
    IplImage src_tmp(src);

    unsigned char *data = (unsigned char *)src_tmp.imageData;
    h = src_tmp.height;
    w = src_tmp.width;
    c = src_tmp.nChannels;
    int step = src_tmp.widthStep;
    int i, j, k;

    for(i = 0; i < h; ++i){
        for(k= 0; k < c; ++k){
            for(j = 0; j < w; ++j){
                out.data[k*w*h + i*w + j] = data[i*step + j*c + k]/255.;
            }
        }
    }
    return out;
}


void draw_detections(Mat& org, image im, detection *dets, int num, float thresh, const vector<string>& names, Rect& rect,  int classes)
{
    int i,j;
    cout<<"num: "<<num<<endl;
    for(i = 0; i < num; ++i){
        int classes =80;
//        cout<<"class"<<classes<<endl;
        for(j = 0; j < classes; ++j){

            if (dets[i].prob[j] > thresh){


                box b = dets[i].bbox;
                int left  = (b.x-b.w/2.)*im.w;
                int right = (b.x+b.w/2.)*im.w;
                int top   = (b.y-b.h/2.)*im.h;
                int bot   = (b.y+b.h/2.)*im.h;

                if(left < 0) left = 0;
                if(right > im.w-1) right = im.w-1;
                if(top < 0) top = 0;
                if(bot > im.h-1) bot = im.h-1;
                rect.x = left;
                rect.y = top;
                rect.width = right - left;
                rect.height = bot - top;
                rectangle(org, Point(rect.x, rect.y),
                          Point(rect.x + rect.width, rect.y + rect.height), Scalar(255, 178, 150), 3);
//                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
                cout<<"object: = "<<names[j]<<"   "<<"scores: = "<<dets[i].prob[j]*100<<"%"<<endl;
                string label = format("%.2f", dets[i].prob[j]*100);
                label = names[j] + "  "+ label;
                int baseLine;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                top = max(top, labelSize.height);
                rectangle(org, Point(rect.x, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width),
                        top + baseLine), Scalar(255, 255, 255), FILLED);
                putText(org, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);



            }
        }

    }

}

int main(int argc, char** argv) {
    std::cout << "Hello, World!" << std::endl;
    char* file1 = argv[1];
    char* file2 = argv[2];

    g_net = load_network(file1, file2,0 );
    set_batch_network(g_net,1);
    srand(2222222);

    VideoCapture cap("/home/zxy/project/yolo3/video/run.mp4");
//    VideoCapture cap(1);
    string classesFile = "/home/zxy/project/darknet-master/data/coco.names";

    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);


    Mat frame;
    // Create a window
    static const string kWinName = "object detection";
    namedWindow(kWinName, WINDOW_NORMAL);

    while(waitKey(1) < 0){
        boost::timer timer;
        cap >> frame;

        image im = color_mat_to_image(frame);
//        image im = load_image()
        image sized = letterbox_image(im,g_net->w, g_net->h);
        layer l = g_net->layers[g_net->n - 1];

        float *X = sized.data;
        network_predict(g_net, X);
        int nboxes = 0;
        detection *dets = get_network_boxes(g_net, im.w, im.h, g_thresh, g_hier_thresh, 0, 1, &nboxes);
        if (g_nms) do_nms_sort(dets, nboxes, l.classes, g_nms);//nboxes 得到的方框数, l.classes 一共的种类数

        int max_idx=-1;
        float max_prob=-1.0;
        Rect rect;
        draw_detections(frame, im, dets, nboxes, g_thresh, classes, rect, l.classes);
//        for (int i=0;i<nboxes;i++)
//        {
//            if (max_prob<dets[i].prob[dets[i].sort_class])
//            {
//                max_prob=dets[i].prob[dets[i].sort_class];
//                max_idx=i;
//            }
//        }
        float FPS = 1 / timer.elapsed();
        char dest[20];
        sprintf(dest, "%f", FPS);
        string label = dest;
        label = "FPS: " + label;
        putText(frame, label, Point(50, 50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(255,255,255),1);

        imshow(kWinName, frame);
        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
    }
    cap.release();
    free_network(g_net);

    return 0;
}