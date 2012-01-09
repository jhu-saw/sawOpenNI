/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id$

  Author(s):  Balazs Vagvolgyi
  Created on: 2011

  (C) Copyright 2006-2011 Johns Hopkins University (JHU), All Rights
  Reserved.

--- begin cisst license - do not edit ---

This software is provided "as is" under an open source license, with
no warranty.  The complete license can be found in license.txt and
http://www.cisst.org/cisst/license.txt.

--- end cisst license ---
*/

#include <cisstStereoVision.h>
#include <sawOpenNI/svlFilterSourceKinect.h>
#include <sawOpenNI/osaOpenNI.h>
#include <cisstStereoVision/svlFilterInput.h>
#include <cisstStereoVision/svlFilterOutput.h>
#include <cisstCommon/cmnGetChar.h>
#include <cisstCommon/cmnPath.h>

#include "oniPlaneSegmentation.h"


class CFilterPlaneSegmentation : public svlFilterBase
{
public:

    CFilterPlaneSegmentation() :
        svlFilterBase()
    {
        AddInput("rgb", true);
        AddInputType("rgb", svlTypeImageRGB);
        AddOutput("rgb", true);
        SetOutputType("rgb", svlTypeImageRGB);

        PointCloudInput = AddInput("pointcloud", false);
        AddInputType("pointcloud", svlTypeImage3DMap);

        PlaneDistances = new svlSampleImageMono16;
        PlaneObjects   = new svlSampleBlobs;

        PlaneDistancesOutput = AddOutput("planedistances",  false);
        PlaneObjectsOutput   = AddOutput("planeobjects",    false);
        SetOutputType("planedistances", svlTypeImageMono16);
        SetOutputType("planeobjects",   svlTypeBlobs);
    }

    ~CFilterPlaneSegmentation()
    {
        Release();

        if (PlaneDistances) delete PlaneDistances;
        if (PlaneObjects)   delete PlaneObjects;
    }

protected:

    int Initialize(svlSample* syncInput, svlSample* &syncOutput)
    {
        syncOutput = syncInput;

        PlaneDistances->SetSize(syncInput);
        PlaneObjects->SetChannelCount(1);
        PlaneObjects->SetBufferSize(1000);

        PlaneDistancesOutput->SetupSample(PlaneDistances);
        PlaneObjectsOutput->SetupSample(PlaneObjects);

        return SVL_OK;
    }

    int Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput)
    {
        syncOutput = syncInput;

        _OnSingleThread(procInfo)
        {
            // Pull point cloud from async input
            svlSampleImage3DMap* pointcloud_sample = dynamic_cast<svlSampleImage3DMap*>(PointCloudInput->PullSample(true));
            if (!pointcloud_sample) return SVL_FAIL;

            Segmentation.Process(dynamic_cast<svlSampleImageRGB*>(syncInput),
                                 pointcloud_sample,
                                 PlaneDistances,
                                 PlaneObjects);

            // Push samples to async outputs
            PlaneDistances->SetTimestamp(syncInput->GetTimestamp());
            PlaneObjects->SetTimestamp(syncInput->GetTimestamp());

            PlaneDistancesOutput->PushSample(PlaneDistances);
            PlaneObjectsOutput->PushSample(PlaneObjects);
        }

        return SVL_OK;
    }

private:
    svlFilterInput*  PointCloudInput;
    svlFilterOutput* PlaneDistancesOutput;
    svlFilterOutput* PlaneObjectsOutput;

    svlSampleImageMono16* PlaneDistances;
    svlSampleBlobs*       PlaneObjects;

public:
    oniPlaneSegmentation Segmentation;
};


int main()
{
    svlInitialize();

    svlStreamManager stream;

    svlFilterSourceKinect kinect;
    svlFilterImageOverlay overlay;
    svlFilterImageWindow window1, window2, window3;
    svlFilterStreamTypeConverter to_rgb1(svlTypeImageMono16, svlTypeImageRGB);
    svlFilterStreamTypeConverter to_rgb2(svlTypeImageMono16, svlTypeImageRGB);

    CFilterPlaneSegmentation segmentation;

    // Set Kinect configuration file
    cmnPath path;
    path.Add(".");
    std::string configFile = path.Find("SamplesConfig.xml");
    if (configFile == "") {
        std::cerr << "can't find file \"SamplesConfig.xml\" in path: " << path << std::endl;
        exit (-1);
    }
    kinect.SetKinectConfigFile(configFile);

    // Setup Mono16 to RGB converter
    to_rgb1.SetMono16ShiftDown(4);
    to_rgb2.SetMono16ShiftDown(8);

    overlay.AddInputBlobs("blobs");
    svlOverlayBlobs ovrl_blobs1(0, true, "blobs", 0);
    overlay.AddOverlay(ovrl_blobs1);
    overlay.AddQueuedItems();

    // Setup windows
    window1.SetTitle("Segmented Image");
    window2.SetTitle("Depth Image");
    window3.SetTitle("Elevation Map");

    // Chain filters to trunk
    stream.SetSourceFilter(&kinect);

    kinect.GetOutput("rgb")->Connect(segmentation.GetInput("rgb"));
    kinect.GetOutput("pointcloud")->Connect(segmentation.GetInput("pointcloud"));
    kinect.GetOutput("depth")->Connect(to_rgb1.GetInput());

    segmentation.GetOutput("rgb")->Connect(overlay.GetInput());
    segmentation.GetOutput("planedistances")->Connect(to_rgb2.GetInput());
    segmentation.GetOutput("planeobjects")->Connect(overlay.GetInput("blobs"));

    overlay.GetOutput()->Connect(window1.GetInput());
    to_rgb1.GetOutput()->Connect(window2.GetInput());
    to_rgb2.GetOutput()->Connect(window3.GetInput());

    // Initialize and start stream
    if (stream.Play() == SVL_OK) {
        int ch;
        unsigned int keycounter = 0;

        do {
            if ((keycounter % 20) == 0) {
                std::cerr << std::endl << "------------------------------------------" << std::endl;
                std::cerr << "GradientRadius [pixels]     = " << segmentation.Segmentation.GetGradientRadius() << std::endl;
                std::cerr << "GradientHistogramThreshold  = " << (int)segmentation.Segmentation.GetGradientHistogramThreshold() << std::endl;
                std::cerr << "PlaneDistanceThreshold [mm] = " << segmentation.Segmentation.GetPlaneDistanceThreshold() << std::endl;
                std::cerr << "MinObjectArea [pixels]      = " << segmentation.Segmentation.GetMinObjectArea() << std::endl << std::endl;
                std::cerr << "Keyboard commands in command window:" << std::endl;
                std::cerr << "  '1'   - GradientRadius --" << std::endl;
                std::cerr << "  '2'   - GradientRadius ++" << std::endl;
                std::cerr << "  '3'   - GradientHistogramThreshold --" << std::endl;
                std::cerr << "  '4'   - GradientHistogramThreshold ++" << std::endl;
                std::cerr << "  '5'   - PlaneDistanceThreshold --" << std::endl;
                std::cerr << "  '6'   - PlaneDistanceThreshold ++" << std::endl;
                std::cerr << "  '7'   - MinObjectArea --" << std::endl;
                std::cerr << "  '8'   - MinObjectArea ++" << std::endl;
                std::cerr << "  'q'   - Quit" << std::endl;
                std::cerr << "------------------------------------------" << std::endl << std::endl;
            }
            keycounter ++;

            ch = cmnGetChar();
            switch (ch) {
                case '1':
                    segmentation.Segmentation.SetGradientRadius(segmentation.Segmentation.GetGradientRadius() - 1);
                    std::cerr << "  GradientRadius [pixels]     = " << segmentation.Segmentation.GetGradientRadius() << std::endl;
                break;

                case '2':
                    segmentation.Segmentation.SetGradientRadius(segmentation.Segmentation.GetGradientRadius() + 1);
                    std::cerr << "  GradientRadius [pixels]     = " << segmentation.Segmentation.GetGradientRadius() << std::endl;
                break;

                case '3':
                    segmentation.Segmentation.SetGradientHistogramThreshold(segmentation.Segmentation.GetGradientHistogramThreshold() - 1);
                    std::cerr << "  GradientHistogramThreshold  = " << (int)segmentation.Segmentation.GetGradientHistogramThreshold() << std::endl;
                break;

                case '4':
                    segmentation.Segmentation.SetGradientHistogramThreshold(segmentation.Segmentation.GetGradientHistogramThreshold() + 1);
                    std::cerr << "  GradientHistogramThreshold  = " << (int)segmentation.Segmentation.GetGradientHistogramThreshold() << std::endl;
                break;

                case '5':
                    segmentation.Segmentation.SetPlaneDistanceThreshold(segmentation.Segmentation.GetPlaneDistanceThreshold() - 1);
                    std::cerr << "  PlaneDistanceThreshold [mm] = " << segmentation.Segmentation.GetPlaneDistanceThreshold() << std::endl;
                break;

                case '6':
                    segmentation.Segmentation.SetPlaneDistanceThreshold(segmentation.Segmentation.GetPlaneDistanceThreshold() + 1);
                    std::cerr << "  PlaneDistanceThreshold [mm] = " << segmentation.Segmentation.GetPlaneDistanceThreshold() << std::endl;
                break;

                case '7':
                    segmentation.Segmentation.SetMinObjectArea(segmentation.Segmentation.GetMinObjectArea() - 1);
                    std::cerr << "  MinObjectArea [pixels]      = " << segmentation.Segmentation.GetMinObjectArea() << std::endl;
                break;

                case '8':
                    segmentation.Segmentation.SetMinObjectArea(segmentation.Segmentation.GetMinObjectArea() + 1);
                    std::cerr << "  MinObjectArea [pixels]      = " << segmentation.Segmentation.GetMinObjectArea() << std::endl;
                break;
            }
        } while (ch != 'q' && ch != 'Q');

        std::cout << "Quitting." << std::endl;
    }
    else {
        std::cout << "Error... Quitting." << std::endl;
    }

    // Safely stopping and deconstructing stream before de-allocation
    stream.Release();
    kinect.GetOutput("depth")->Disconnect();
    kinect.GetOutput("pointcloud")->Disconnect();
    segmentation.GetOutput("planedistances")->Disconnect();
    segmentation.GetOutput("planeobjects")->Disconnect();

    return 1;
}

