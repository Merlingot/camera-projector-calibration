#include <opencv2/highgui.hpp>

using namespace cv;


int main ( int argc, char **argv )
{
    // define dimension of the main display
    int width_first  = 2880;
    int height_first = 1800;

    // define dimension of the second display
    int width_second  = 1280;
    int height_second = 720;

    // move the window to the second display
    // (assuming the two displays are top aligned)
    namedWindow("My Window", CV_WINDOW_NORMAL);
    moveWindow("My Window", width_first, height_first);
    setWindowProperty("My Window", CV_WND_PROP_FULLSCREEN, CV_WINDOW_FULLSCREEN);

    // create target image
    Mat img = Mat(Size(width_second, height_second), CV_8UC1);

    // show the image
    imshow("My Window", img);
    waitKey(0);

    return 0;
}
