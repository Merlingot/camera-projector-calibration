
#include <opencv2/highgui.hpp>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/structured_light.hpp>
#include <opencv2/phase_unwrapping.hpp>
using namespace cv;
using namespace std;


int main(int argc, char **argv)
{

    structured_light::SinusoidalPattern::Params params;
    phase_unwrapping::HistogramPhaseUnwrapping::Params paramsUnwrapping;
    // Retrieve parameters written in the command line
    params.width = 800;
    params.height = 600;
    params.nbrOfPeriods = 5;
    params.setMarkers = true;
    params.horizontal = true;
    params.methodId = 1;
    params.shiftValue = static_cast<float>(2 * CV_PI / 3);
    params.nbrOfPixelsBetweenMarkers = 70;

    String outputCapturePath = "output/captures/";
    String outputPatternPath = "output/patterns/";
    String outputWrappedPhasePath = "output/wrapped/";
    String outputUnwrappedPhasePath = "output/unwrapped/";
    String reliabilitiesPath = "output/reliabilities/";


    Ptr<structured_light::SinusoidalPattern> sinus =
            structured_light::SinusoidalPattern::create(makePtr<structured_light::SinusoidalPattern::Params>(params));
    Ptr<phase_unwrapping::HistogramPhaseUnwrapping> phaseUnwrapping;
    vector<Mat> patterns;
    Mat shadowMask;
    Mat unwrappedPhaseMap, unwrappedPhaseMap8;
    Mat unwrappedProj;
    Mat wrappedPhaseMap, wrappedPhaseMap8;
    Mat wrappedProj;
    Mat matches;
    //Generate sinusoidal patterns
    sinus->generate(patterns);

    // Camera ----------------
    cout << "Opening Camera" << endl;
    VideoCapture cap(1);
    if( !cap.isOpened() )
    {
        cout << "Camera could not be opened" << endl;
        return -1;
    }
    cap.set(CAP_PROP_PVAPI_PIXELFORMAT, CAP_PVAPI_PIXELFORMAT_MONO8);

    // Arranger les window pour le projecteur ----------------
    cout << "Opening window in main monitor" << endl;
    namedWindow("pattern", WINDOW_NORMAL);
    moveWindow("pattern", 0, 0);
    imshow("pattern", patterns[0]);
    // move the window to the second display
    cout << "Move window to monitor" << endl;
    int width_first  = 1440;
    int height_first = 900;
    // waitKey(0);
    moveWindow("pattern", 0, -height_first);
    cout << "Press on the window then press any key to make window full screen and start" << endl;
    waitKey(0);
    setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    // -----------------------

    //Test caméra
    // Prendre une photo sans rien
    int allo = 1;
    vector<Mat> imgw(allo);
    cap >> imgw[0];
    imwrite(outputUnwrappedPhasePath + "image_sans_frange" + ".png", imgw);

    int nbrOfImages = 3;
    int count = 0;
    vector<Mat> img(nbrOfImages);
    Size camSize(-1, -1);
    Size projSize(800, 600);
    while( count < nbrOfImages )
    {
        for(int i = 0; i < (int)patterns.size(); ++i )
        {
            Mat colorimg;
            imshow("pattern", patterns[i]);
            waitKey(30);
            cap >> colorimg;
            cvtColor(colorimg, img[count], COLOR_BGR2GRAY);
            count += 1;
        }
    }
    cap.release();
    destroyAllWindows();

    cout << "Starting unwraping" << endl;

    switch(params.methodId)
    {
        case structured_light::FTP:
            for( int i = 0; i < nbrOfImages; ++i )
            {
                /*We need three images to compute the shadow mask, as described in the reference paper
                 * even if the phase map is computed from one pattern only
                */
                vector<Mat> captures;
                if( i == nbrOfImages - 2 )
                {
                    captures.push_back(img[i]);
                    captures.push_back(img[i-1]);
                    captures.push_back(img[i+1]);
                }
                else if( i == nbrOfImages - 1 )
                {
                    captures.push_back(img[i]);
                    captures.push_back(img[i-1]);
                    captures.push_back(img[i-2]);
                }
                else
                {
                    captures.push_back(img[i]);
                    captures.push_back(img[i+1]);
                    captures.push_back(img[i+2]);
                }

                sinus->computePhaseMap(captures, wrappedPhaseMap, shadowMask);

                if( camSize.height == -1 )
                {
                    camSize.height = img[i].rows;
                    camSize.width = img[i].cols;
                    paramsUnwrapping.height = camSize.height;
                    paramsUnwrapping.width = camSize.width;
                    phaseUnwrapping =
                    phase_unwrapping::HistogramPhaseUnwrapping::create(paramsUnwrapping);
                }
                sinus->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, camSize, shadowMask);
                phaseUnwrapping->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, shadowMask);
                Mat reliabilities, reliabilities8;
                phaseUnwrapping->getInverseReliabilityMap(reliabilities);
                reliabilities.convertTo(reliabilities8, CV_8U, 255,128);
                ostringstream tt;
                tt << i;
                imwrite(reliabilitiesPath + tt.str() + ".png", reliabilities8);
                unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);
                wrappedPhaseMap.convertTo(wrappedPhaseMap8, CV_8U, 255, 128);
                if( !outputUnwrappedPhasePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    imwrite(outputUnwrappedPhasePath + "_FTP_" + name.str() + ".png", unwrappedPhaseMap8);
                }
                if( !outputWrappedPhasePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    imwrite(outputWrappedPhasePath + "_FTP_" + name.str() + ".png", wrappedPhaseMap8);
                }
            }
            break;
        case structured_light::PSP:
        case structured_light::FAPS:
            for( int i = 0; i < nbrOfImages - 2; ++i )
            {
                vector<Mat> captures;
                captures.push_back(img[i]);
                captures.push_back(img[i+1]);
                captures.push_back(img[i+2]);
                sinus->computePhaseMap(captures, wrappedPhaseMap, shadowMask);

                if( camSize.height == -1 )
                {
                    camSize.height = img[i].rows;
                    camSize.width = img[i].cols;
                    paramsUnwrapping.height = camSize.height;
                    paramsUnwrapping.width = camSize.width;
                    phaseUnwrapping =
                    phase_unwrapping::HistogramPhaseUnwrapping::create(paramsUnwrapping);
                }
                sinus->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, camSize, shadowMask);


                cout << "matches" << endl;
                sinus->computePhaseMap(patterns, wrappedProj);
                sinus->unwrapPhaseMap(wrappedProj, unwrappedProj, projSize);
                sinus->findProCamMatches(patterns, unwrappedPhaseMap, matches);
                imwrite(outputPatternPath + "match" + ".png", matches);
                

                unwrappedPhaseMap.convertTo(unwrappedPhaseMap8, CV_8U, 1, 128);
                wrappedPhaseMap.convertTo(wrappedPhaseMap8, CV_8U, 255, 128);
                phaseUnwrapping->unwrapPhaseMap(wrappedPhaseMap, unwrappedPhaseMap, shadowMask);
                Mat reliabilities, reliabilities8;
                phaseUnwrapping->getInverseReliabilityMap(reliabilities);
                reliabilities.convertTo(reliabilities8, CV_8U, 255,128);
                ostringstream tt;
                tt << i;
                imwrite(reliabilitiesPath + tt.str() + ".png", reliabilities8);
                if( !outputUnwrappedPhasePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    if( params.methodId == structured_light::PSP )
                        imwrite(outputUnwrappedPhasePath + "_PSP_" + name.str() + ".png", unwrappedPhaseMap8);
                    else
                        imwrite(outputUnwrappedPhasePath + "_FAPS_" + name.str() + ".png", unwrappedPhaseMap8);
                }
                if( !outputWrappedPhasePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    if( params.methodId == structured_light::PSP )
                        imwrite(outputWrappedPhasePath + "_PSP_" + name.str() + ".png", wrappedPhaseMap8);
                    else
                        imwrite(outputWrappedPhasePath + "_FAPS_" + name.str() + ".png", wrappedPhaseMap8);
                }
                if( !outputCapturePath.empty() )
                {
                    ostringstream name;
                    name << i;
                    if( params.methodId == structured_light::PSP )
                        imwrite(outputCapturePath + "_PSP_" + name.str() + ".png", img[i]);
                    else
                        imwrite(outputCapturePath + "_FAPS_" + name.str() + ".png", img[i]);
                    if( i == nbrOfImages - 3 )
                    {
                        if( params.methodId == structured_light::PSP )
                        {
                            ostringstream nameBis;
                            nameBis << i+1;
                            ostringstream nameTer;
                            nameTer << i+2;
                            imwrite(outputCapturePath + "_PSP_" + nameBis.str() + ".png", img[i+1]);
                            imwrite(outputCapturePath + "_PSP_" + nameTer.str() + ".png", img[i+2]);
                        }
                        else
                        {
                            ostringstream nameBis;
                            nameBis << i+1;
                            ostringstream nameTer;
                            nameTer << i+2;
                            imwrite(outputCapturePath + "_FAPS_" + nameBis.str() + ".png", img[i+1]);
                            imwrite(outputCapturePath + "_FAPS_" + nameTer.str() + ".png", img[i+2]);
                        }
                    }
                }
            }
            break;
        default:
            cout << "error" << endl;
    }
    cout << "done" << endl;
    if( !outputPatternPath.empty() )
    {
        for( int i = 0; i < 3; ++ i )
        {
            ostringstream name;
            name << i + 1;
            imwrite(outputPatternPath + name.str() + ".png", patterns[i]);
        }
    }

    return 0;
}
