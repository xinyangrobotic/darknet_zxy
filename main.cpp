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
float g_thresh = 0.3,g_hier_thresh = 0.3, g_nms = 0.45;


typedef struct target {
    Rect rect;
    int classes;
    float score;
} target;

static bool lId(target d1, target d2){
    if(d1.classes < d2.classes)
        return true;
//        if(d1.rect.x < d2.rect.x){
//            if(d1.rect.y < d2.rect.y){
//                if(d1.score > d2.score)
//                    return true;
//                else
//                    return false;
//            } else
//                return false;
//        } else
//            return false;

    if(d1.classes == d2.classes && d1.rect.x < d2.rect.x)
        return true;
    if(d1.classes == d2.classes && d1.rect.x < d2.rect.x && d1.rect.y < d2.rect.y)
        return true;
    if(d1.classes == d2.classes && d1.rect.x < d2.rect.x && d1.rect.y < d2.rect.y && d1.score > d2.score)
        return true;
    return false;

}

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

void eraseWrong(vector<target>& AllTarget){
    float rthresh = 0.2;
    for(auto iter = AllTarget.begin(); iter != AllTarget.end(); ){
        if (iter == AllTarget.begin()){
                iter++;
        } else{
            if( iter->classes == (iter - 1)->classes){
                if( (abs(iter->rect.x - (iter - 1)->rect.x) / iter->rect.width) < rthresh
                 && (abs(iter->rect.y - (iter - 1)->rect.y) / iter->rect.height) < rthresh){
                    if(iter->score < (iter-1)->score)
                        AllTarget.erase(iter);
                    else
                        AllTarget.erase((iter - 1));
                }
                else
                    ++iter;
            } else
                ++iter;

        }
        cout<<"classes: = "<<iter->classes<<"  "<<"rect(x,y): "<<iter->rect.tl()<<"  "
            <<"rect.size: "<<iter->rect.size()<<"  "<<"score: "<<iter->score<<endl;
    }
}

void OptimalObject(image im, detection *dets, int num, float thresh, vector<target>& AllTarget){
    int i,j;
    for(i = 0; i < num; ++i) {
        int classes = 80;
        Rect rect;
        for (j = 0; j < classes; ++j) {

            if (dets[i].prob[j] > thresh) {
                box b = dets[i].bbox;
                int left = (b.x - b.w / 2.) * im.w;
                int right = (b.x + b.w / 2.) * im.w;
                int top = (b.y - b.h / 2.) * im.h;
                int bot = (b.y + b.h / 2.) * im.h;

                if (left < 0) left = 0;
                if (right > im.w - 1) right = im.w - 1;
                if (top < 0) top = 0;
                if (bot > im.h - 1) bot = im.h - 1;
                rect.x = left;
                rect.y = top;
                rect.width = right - left;
                rect.height = bot - top;
                target temp;
                temp.rect = rect;
                temp.classes = j;
                temp.score = dets[i].prob[j];
                AllTarget.push_back(temp);
            }
        }
    }
    for(i =0; i < 4; i++) {
        sort(AllTarget.begin(), AllTarget.end(), lId);
    }
    eraseWrong(AllTarget);
}

void draw_target( Mat& org, const vector<target>& AllTarget,const vector<string>& names){
    for(auto i = 0; i < AllTarget.size(); i++){
        rectangle(org, AllTarget[i].rect, Scalar(255, 178, 150), 3);
                cout<<"object: = "<<names[AllTarget[i].classes]<<"   "<<"scores: = "<<AllTarget[i].score*100<<"%"<<
                "   "<<"center: = "<<AllTarget[i].rect.tl()<<endl;
                string label = format("%.2f", AllTarget[i].score*100);
                label = names[AllTarget[i].classes] + "  "+ label;
                int baseLine;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                int top = max(AllTarget[i].rect.y , labelSize.height);
                rectangle(org, Point(AllTarget[i].rect.x, top - round(1.5*labelSize.height)),
                        Point(AllTarget[i].rect.x + round(1.5*labelSize.width), top + baseLine), Scalar(255, 255, 255), FILLED);
                putText(org, label, Point(AllTarget[i].rect.x, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
    }
}
//void draw_detections(Mat& org, image im, detection *dets, int num, float thresh, const vector<string>& names, Rect& rect,  int classes)
//{
//    int i,j;
//    cout<<"num: "<<num<<endl;
//    sort(dets, dets+num,)
//    for(i = 0; i < num; ++i){
//        int classes =80;
////        cout<<"class"<<classes<<endl;
//        for(j = 0; j < classes; ++j){
//
//            if (dets[i].prob[j] > thresh){
//
//
//                box b = dets[i].bbox;
//                int left  = (b.x-b.w/2.)*im.w;
//                int right = (b.x+b.w/2.)*im.w;
//                int top   = (b.y-b.h/2.)*im.h;
//                int bot   = (b.y+b.h/2.)*im.h;
//
//                if(left < 0) left = 0;
//                if(right > im.w-1) right = im.w-1;
//                if(top < 0) top = 0;
//                if(bot > im.h-1) bot = im.h-1;
//                rect.x = left;
//                rect.y = top;
//                rect.width = right - left;
//                rect.height = bot - top;
//                rectangle(org, rect, Scalar(255, 178, 150), 3);
////                printf("%s: %.0f%%\n", names[j], dets[i].prob[j]*100);
//                cout<<"object: = "<<names[j]<<"   "<<"scores: = "<<dets[i].prob[j]*100<<"%"<<
//                "   "<<"center: = "<<rect.tl()<<endl;
//                string label = format("%.2f", dets[i].prob[j]*100);
//                label = names[j] + "  "+ label;
//                int baseLine;
//                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//                top = max(top, labelSize.height);
//                rectangle(org, Point(rect.x, top - round(1.5*labelSize.height)), Point(left + round(1.5*labelSize.width),
//                        top + baseLine), Scalar(255, 255, 255), FILLED);
//                putText(org, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,0),1);
//
//
//
//            }
//        }
//
//    }
//
//}

int main(int argc, char** argv) {
    std::cout << "Hello, World!" << std::endl;
    char* file1 = argv[1];//cfg文件
    char* file2 = argv[2];//weights文件

    g_net = load_network(file1, file2,0 );
    set_batch_network(g_net,1);
    srand(2222222);

//    VideoCapture cap("/home/zxy/project/yolo3/video/road.flv");
    VideoCapture cap(1);
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

//        int max_idx=-1;
//        float max_prob=-1.0;
        Rect rect;
        vector<target> AllTarget;
//        draw_detections(frame, im, dets, nboxes, g_thresh, classes, rect, l.classes);
        OptimalObject(im, dets, nboxes, g_thresh, AllTarget);
        cout<<"*******************************************"<<endl;
        draw_target(frame, AllTarget, classes);
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