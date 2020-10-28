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
static const char* keys =
{
    "{@width | | Projector width}"
    "{@height | | Projector height}"
    "{@periods | | Number of periods}"
    "{@setMarkers | | Patterns with or without markers}"
    "{@horizontal | | Patterns are horizontal}"
    "{@methodId | | Method to be used}"
    "{@outputPatternPath | | Path to save patterns}"
    "{@outputWrappedPhasePath | | Path to save wrapped phase map}"
    "{@outputUnwrappedPhasePath | | Path to save unwrapped phase map}"
    "{@outputCapturePath | | Path to save the captures}"
    "{@reliabilitiesPath | | Path to save reliabilities}"
};
static void help()
{
    cout << "\nThis example generates sinusoidal patterns" << endl;
    cout << "To call: ./example_structured_light_createsinuspattern <width> <height>"
            " <number_of_period> <set_marker>(bool) <horizontal_patterns>(bool) <method_id>"
            " <output_captures_path> <output_pattern_path>(optional) <output_wrapped_phase_path> (optional)"
            " <output_unwrapped_phase_path>" << endl;
}
int main(int argc, char **argv)
{
    if( argc < 2 )
    {
        help();
        return -1;
    }
    structured_light::SinusoidalPattern::Params params;
    phase_unwrapping::HistogramPhaseUnwrapping::Params paramsUnwrapping;
    // Retrieve parameters written in the command line
    CommandLineParser parser(argc, argv, keys);
    params.width = parser.get<int>(0);
    params.height = parser.get<int>(1);
    params.nbrOfPeriods = parser.get<int>(2);
    params.setMarkers = parser.get<bool>(3);
    params.horizontal = parser.get<bool>(4);
    params.methodId = parser.get<int>(5);
    String outputCapturePath = parser.get<String>(6);
    params.shiftValue = static_cast<float>(2 * CV_PI / 3);
    params.nbrOfPixelsBetweenMarkers = 70;
    String outputPatternPath = parser.get<String>(7);
    String outputWrappedPhasePath = parser.get<String>(8);
    String outputUnwrappedPhasePath = parser.get<String>(9);
    String reliabilitiesPath = parser.get<String>(10);
    Ptr<structured_light::SinusoidalPattern> sinus =
            structured_light::SinusoidalPattern::create(makePtr<structured_light::SinusoidalPattern::Params>(params));
    Ptr<phase_unwrapping::HistogramPhaseUnwrapping> phaseUnwrapping;
    vector<Mat> patterns;
    Mat shadowMask;
    Mat unwrappedPhaseMap, unwrappedPhaseMap8;
    Mat wrappedPhaseMap, wrappedPhaseMap8;
    //Generate sinusoidal patterns
    sinus->generate(patterns);


    VideoCapture cap(1);
    if( !cap.isOpened() )
    {
        cout << "Camera could not be opened" << endl;
        return -1;
    }
    cap.set(CAP_PROP_PVAPI_PIXELFORMAT, CAP_PVAPI_PIXELFORMAT_MONO8);
    namedWindow("pattern", WINDOW_NORMAL);
    setWindowProperty("pattern", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    imshow("pattern", patterns[0]);
    cout << "Press any key when ready" << endl;
    waitKey(0);
    int nbrOfImages = 30;
    int count = 0;
    vector<Mat> img(nbrOfImages);
    Size camSize(-1, -1);
    while( count < nbrOfImages )
    {
        for(int i = 0; i < (int)patterns.size(); ++i )
        {
            Mat colorimg;
            imshow("pattern", patterns[i]);
            waitKey(300);
            cap >> colorimg;
            cvtColor(colorimg, img[count], COLOR_BGR2GRAY);
            count += 1;
        }
    }
    destroyWindow("pattern");
    cout << "press enter when ready" << endl;

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
