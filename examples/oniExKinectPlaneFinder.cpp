/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*
  $Id: $

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

#include "planefit.h"


class svlFilterKinectGradientLabeler : public svlFilterBase
{
public:

    svlFilterKinectGradientLabeler() :
        svlFilterBase(),
        ThresholdMM(20),
        HistogramThreshold(20),
        MinObjectArea(200),
        Radius(25)
    {
        AddInput("rgb", true);
        AddInputType("rgb", svlTypeImageRGB);
        AddOutput("rgb", true);
        SetOutputType("rgb", svlTypeImageRGB);

        DepthInput = AddInput("depth", false);
        AddInputType("depth", svlTypeImageMono16);
        DepthOutput = AddOutput("depth", false);
        SetOutputType("depth", svlTypeImageMono16);

        PointCloudInput = AddInput("pointcloud", false);
        AddInputType("pointcloud", svlTypeImage3DMap);

        SegmentsInput = AddInput("segments", false);
        AddInputType("segments", svlTypeBlobs);

        LabelsSample = new svlSampleImageMono8;

        HistogramImage  = new svlSampleImageMono8;
        HistogramLabels = new svlSampleImageMono32;
        HistogramBlobs  = new svlSampleBlobs;

        const unsigned int histogram_size = 256;
        HistogramImage->SetSize(histogram_size, histogram_size);
        HistogramLabels->SetSize(HistogramImage);
        HistogramBlobs->SetChannelCount(1);
        HistogramBlobs->SetBufferSize(10);

        TempHistogram = new svlSampleImageMono32;
        TempHistogram->SetSize(histogram_size, histogram_size);

        Blobs             = new svlSampleBlobs;
        BlobLabels        = new svlSampleImageMono32;
        PlaneDistances    = new svlSampleImageMono16;
        PlaneObjectMask   = new svlSampleImageMono8;
        PlaneObjectLabels = new svlSampleImageMono32;
        PlaneObjects      = new svlSampleBlobs;

        PlaneDistancesOutput = AddOutput("planedistances",  false);
        PlaneObjectsOutput   = AddOutput("planeobjects",    false);
        SetOutputType("planedistances", svlTypeImageMono16);
        SetOutputType("planeobjects",   svlTypeBlobs);
    }

    virtual ~svlFilterKinectGradientLabeler()
    {
        Release();

        if (LabelsSample)      delete LabelsSample;
        if (HistogramImage)    delete HistogramImage;
        if (HistogramLabels)   delete HistogramLabels;
        if (HistogramBlobs)    delete HistogramBlobs;
        if (TempHistogram)     delete TempHistogram;
        if (Blobs)             delete Blobs;
        if (BlobLabels)        delete BlobLabels;
        if (PlaneDistances)    delete PlaneDistances;
        if (PlaneObjectMask)   delete PlaneObjectMask;
        if (PlaneObjectLabels) delete PlaneObjectLabels;
        if (PlaneObjects)      delete PlaneObjects;
    }

    void SetRadius(unsigned int radius)
    {
        if (radius > 0) Radius = radius;
    }

    unsigned int GetRadius()
    {
        return Radius;
    }

protected:

    virtual int Initialize(svlSample* syncInput, svlSample* &syncOutput)
    {
        syncOutput = syncInput;

        // Setup depth output with sample prototype
        svlSampleImageMono16 depth_sample;
        depth_sample.SetSize(syncInput);
        DepthOutput->SetupSample(&depth_sample);

        // Setup labels output with sample prototype
        LabelsSample->SetSize(syncInput);

        GradX.SetSize(depth_sample.GetHeight(), depth_sample.GetWidth());
        GradY.SetSize(depth_sample.GetHeight(), depth_sample.GetWidth());

        Blobs->SetChannelCount(1);
        Blobs->SetBufferSize(1000);
        BlobLabels->SetSize(syncInput);

        PlaneObjects->SetChannelCount(1);
        PlaneObjects->SetBufferSize(1000);
        PlaneDistances->SetSize(syncInput);
        PlaneObjectMask->SetSize(syncInput);
        PlaneObjectLabels->SetSize(syncInput);
        PlaneDistancesOutput->SetupSample(PlaneDistances);
        PlaneObjectsOutput->SetupSample(PlaneObjects);

        return SVL_OK;
    }

    virtual int Process(svlProcInfo* procInfo, svlSample* syncInput, svlSample* &syncOutput)
    {
        syncOutput = syncInput;

        _OnSingleThread(procInfo)
        {
            // Pull depth image from async input
            svlSampleImageMono16* depth_sample = dynamic_cast<svlSampleImageMono16*>(DepthInput->PullSample(true));
            if (!depth_sample) return SVL_FAIL;

            // Pull point cloud from async input
            svlSampleImage3DMap* pointcloud_sample = dynamic_cast<svlSampleImage3DMap*>(PointCloudInput->PullSample(true));
            if (!pointcloud_sample) return SVL_FAIL;

            ComputeDepthGradientHistogram(pointcloud_sample->GetMatrixRef(), TempHistogram->GetMatrixRef());

            vctDynamicMatrix<double> kernel(3, 3);
            kernel.SetAll(1.0 / 9.0);
            svlSampleImageMono32 thist;
            thist.SetSize(TempHistogram);
            svlImageProcessing::Convolution(TempHistogram, 0, &thist, 0, kernel);
            svlImageProcessing::Convolution(&thist, 0, TempHistogram, 0, kernel);

            NormalizeGradientHistogram(TempHistogram->GetMatrixRef(), HistogramImage->GetMatrixRef());
            FindHistogramBlobs(HistogramImage, HistogramLabels, HistogramBlobs);

            LabelImage(HistogramBlobs);

            svlImageProcessing::LabelBlobs(LabelsSample, BlobLabels, BlobDetectorInternals);
            svlImageProcessing::GetBlobsFromLabels(LabelsSample, BlobLabels, Blobs, BlobDetectorInternals,
                                                   0, 0,
                                                   0.0, 0.0);

            FindLargestSegment(BlobLabels, Blobs);
            FitPlane(BlobLabels, PlaneDistances, pointcloud_sample);
            LabelObjects(PlaneDistances, PlaneObjectMask);

            svlImageProcessing::LabelBlobs(PlaneObjectMask, PlaneObjectLabels, PlaneBlobInternals);
            svlImageProcessing::GetBlobsFromLabels(PlaneObjectMask, PlaneObjectLabels, PlaneObjects, PlaneBlobInternals,
                                                   MinObjectArea, 0,
                                                   0.0, 0.0);

            VisualizePlaneObjects(dynamic_cast<svlSampleImageRGB*>(syncInput), PlaneDistances, PlaneObjectLabels, PlaneObjects);

            // Push samples to async outputs
            HistogramImage->SetTimestamp(syncInput->GetTimestamp());
            PlaneDistances->SetTimestamp(syncInput->GetTimestamp());
            PlaneObjects->SetTimestamp(syncInput->GetTimestamp());

            DepthOutput->PushSample(depth_sample);
            PlaneDistancesOutput->PushSample(PlaneDistances);
            PlaneObjectsOutput->PushSample(PlaneObjects);
        }

        return SVL_OK;
    }

    virtual int Release()
    {
        return SVL_OK;
    }

    void ComputeDepthGradientHistogram(vctDynamicMatrixRef<float> pointcloud, vctDynamicMatrixRef<unsigned int> histogram)
    {
        const int radius = Radius;
        const int width = (pointcloud.cols() / 3);
        const int height = pointcloud.rows();
        const int hist_size = histogram.cols();
        const int hist_center = hist_size / 2;
        const int hist_min = -hist_center;
        const int hist_max = hist_center - 1;

        float* pf;
        int i, j, rad_l, rad_r, rad_t, rad_b, d, dx, dy, x, y, lx, ly, lz, rx, ry, rz, tx, ty, tz, bx, by, bz;

        GradX.SetAll(0);
        GradY.SetAll(0);
        histogram.SetAll(0);

        for (j = 0; j < height; j ++) {
            if (j >= radius) rad_t = radius;
            else rad_t = j;
            if (j < (height - radius)) rad_b = radius;
            else rad_b = height - j - 1;

            for (i = 0; i < width; i ++) {
                if (i >= radius) rad_l = radius;
                else rad_l = i;
                if (i < (width - radius)) rad_r = radius;
                else rad_r = width - i - 1;

                pf = pointcloud.Pointer(j, (i - rad_l) * 3);
                lx = pf[0] * 1000; ly = pf[1] * 1000; lz = pf[2] * 1000;

                pf = pointcloud.Pointer(j, (i + rad_r) * 3);
                rx = pf[0] * 1000; ry = pf[1] * 1000; rz = pf[2] * 1000;

                pf = pointcloud.Pointer(j - rad_t, i * 3);
                tx = pf[0] * 1000; ty = pf[1] * 1000; tz = pf[2] * 1000;

                pf = pointcloud.Pointer(j + rad_b, i * 3);
                bx = pf[0] * 1000; by = pf[1] * 1000; bz = pf[2] * 1000;

                if (lz == 0 || rz == 0 || tz == 0 || bz == 0 || lx == rx || ty == by) {
                    // Black pixels are invalid
                    continue;
                }

                x = rx - lx; y = ry - ly;
                d = sqrt_uint32(x * x + y * y);
                dx = (rz - lz) * 20 / d;

                x = bx - tx; y = by - ty;
                d = sqrt_uint32(x * x + y * y);
                dy = (bz - tz) * 20 / d;

                if (dx == 0 && dy == 0) {
                    // These are ignored
                    continue;
                }

                GradX.Element(j, i) = dx;
                GradY.Element(j, i) = dy;

                if (dx < hist_min) dx = hist_min;
                else if (dx > hist_max) dx = hist_max;
                if (dy < hist_min) dy = hist_min;
                else if (dy > hist_max) dy = hist_max;

                histogram.Element(dy + hist_center, dx + hist_center) ++;
            }
        }
    }

    void NormalizeGradientHistogram(vctDynamicMatrixRef<unsigned int> histogram, vctDynamicMatrixRef<unsigned char> norm_histogram)
    {
        const unsigned int hist_size = histogram.cols();
        unsigned int i, j, ival, imax = 0;

        for (j = 0; j < hist_size; j ++) {
            for (i = 0; i < hist_size; i ++) {
                ival = histogram.Element(j, i);
                if (ival > imax) imax = ival;
            }
        }
        if (imax > 0) {
            for (j = 0; j < hist_size; j ++) {
                for (i = 0; i < hist_size; i ++) {
                    ival = histogram.Element(j, i);
                    norm_histogram.Element(j, i) = (ival * 255) / imax;
                }
            }
        }
    }

    void FindHistogramBlobs(svlSampleImageMono8* image, svlSampleImageMono32* labels, svlSampleBlobs* blobs)
    {
        // Thresholding
        const unsigned int pixel_count = image->GetWidth() * image->GetHeight();
        unsigned char* phist = image->GetUCharPointer();
        for (unsigned int i = 0; i < pixel_count; i ++) {
            if (*phist > HistogramThreshold) *phist = HistogramThreshold;
            else *phist = 0;
            phist ++;
        }

        svlImageProcessing::LabelBlobs(image, labels, HistogramBlobInternals);
        svlImageProcessing::GetBlobsFromLabels(image, labels, blobs, HistogramBlobInternals,
                                               0, 0,
                                               0.0, 0.0);
    }

    void LabelImage(svlSampleBlobs* segments)
    {
        const unsigned int segment_count = std::min(segments->GetBufferUsed(), 3u);
        const unsigned int pixel_count = LabelsSample->GetWidth() * LabelsSample->GetHeight();
        const int hist_center = TempHistogram->GetWidth() / 2;

        unsigned char *img = LabelsSample->GetUCharPointer();
        short *gradx = GradX.Pointer();
        short *grady = GradY.Pointer();

        short gx, gy, left, right, top, bottom;
        svlBlob blob;

        memset(img, 0, pixel_count);

        for (unsigned int j = 1; j <= segment_count; j ++) {
            segments->GetBlob(j - 1, blob);
            left   = blob.left - hist_center;
            right  = blob.right - hist_center;
            top    = blob.top - hist_center;
            bottom = blob.bottom - hist_center;

            for (unsigned int i = 0; i < pixel_count; i ++) {
                gx = gradx[i];
                gy = grady[i];

                if ((gx != 0 || gy != 0) &&
                    gx >= left && gx <= right &&
                    gy >= top  && gy <= bottom) {
                    img[i] = j;
                }
            }
        }
    }

    inline unsigned int sqrt_uint32(unsigned int value)
    {
        unsigned int a, g = 0;
        unsigned int bshft = 15;
        unsigned int b = 1 << bshft;

        do {
            a = (g + g + b) << bshft;
            if (value >= a) {
                g += b;
                value -= a;
            }
            b >>= 1;
        } while (bshft --);

        return g;
    }

    void FindLargestSegment(svlSampleImageMono32* labels, svlSampleBlobs* segments)
    {
        const unsigned int segment_count = segments->GetBufferUsed();
        const unsigned int pixel_count = labels->GetWidth() * labels->GetHeight();

        svlBlob blob;
        int largest_size = -1, label = -1, id = -1;
        for (unsigned int i = 0; i < segment_count; i ++) {
            segments->GetBlob(i, blob);
            if (blob.used && static_cast<int>(blob.area) > largest_size) {
                largest_size = blob.area;
                label = blob.label;
                id = i + 1;
            }
        }

        unsigned int* plabels = labels->GetPointer();

        if (id < 1) {
            memset(plabels, 0, pixel_count * 4);
        }
        else {
            for (unsigned int i = 0; i < pixel_count; i ++) {
                if (static_cast<int>(*plabels) != id) *plabels = 0;
                else *plabels = 255;
                plabels ++;
            }
        }
    }

    void FitPlane(svlSampleImageMono32* labels, svlSampleImageMono16* distances, svlSampleImage3DMap* points)
    {
        const unsigned int width  = labels->GetWidth();
        const unsigned int height = labels->GetHeight();

        if (PlaneFitPoints.size() < (width * height)) PlaneFitPoints.SetSize(width * height);

        vctDynamicMatrixRef<float> pointcloud(points->GetMatrixRef());
        unsigned int* plabels = labels->GetPointer();
        unsigned int i, j, c = 0;
        float *pf1, *pf2;
        double cx = 0, cy = 0, cz = 0;

        for (j = 0; j < height; j ++) {
            for (i = 0; i < width; i ++) {

                if (*plabels == 255) {

                    pf1 = pointcloud.Pointer(j, i * 3);
                    pf2 = &(PlaneFitPoints[c][0]);
                    pf2[0] = pf1[0];
                    pf2[1] = pf1[1];
                    pf2[2] = pf1[2];

                    cx += pf1[0];
                    cy += pf1[1];
                    cz += pf1[2];
                    c ++;
                }

                plabels ++;
            }
        }

        if (c > 2) {
            vctDynamicVectorRef<vctFloat3> point_vec(PlaneFitPoints, 0, c);
            PlaneFit<float> fitter;

            fitter.Calculate(point_vec, PlaneFitWeights, Plane);

            // Check distance of points from plane
            float a = Plane[0], b = Plane[1], c = Plane[2], d = Plane[3];
            float normlen = sqrt(a * a + b * b + c * c);
            float dist;
            int ival;
            unsigned short* pdistances = distances->GetPointer();

            for (j = 0; j < height; j ++) {
                for (i = 0; i < width; i ++) {
                    pf1 = pointcloud.Pointer(j, i * 3);

                    dist = (pf1[0] * a + pf1[1] * b + pf1[2] * c + d) / normlen;
                    if (b < 0.0f) dist = -dist;

                    ival = dist * 100000 + 32768;
                    if (ival < 0) ival = 0;
                    else if (ival > 65535) ival = 65535;
                    *pdistances = ival;

                    pdistances ++;
                }
            }
        }
    }

    void LabelObjects(svlSampleImageMono16* distances, svlSampleImageMono8* labels)
    {
        unsigned short* pdistances = distances->GetPointer();
        unsigned char* plabels = labels->GetPointer();
        const unsigned int width  = labels->GetWidth();
        const unsigned int height = labels->GetHeight();
        unsigned int i, j;
        unsigned int dist;

        const unsigned int threshold = ThresholdMM * 100;

        for (j = 0; j < height; j ++) {
            for (i = 0; i < width; i ++) {
                dist = *pdistances;
                if (dist >= 32768) {
                    if ((dist - 32768) >= threshold) *plabels = 255;
                    else *plabels = 0;
                }
                else {
                    if ((32768 - dist) >= threshold) *plabels = 128;
                    else *plabels = 0;
                }

                pdistances ++;
                plabels ++;
            }
        }
    }

    void VisualizePlaneObjects(svlSampleImageRGB* image, svlSampleImageMono16* distances, svlSampleImageMono32* labels, svlSampleBlobs* blobs)
    {
        unsigned short* pdistances = distances->GetPointer();
        unsigned int* plabels = labels->GetPointer();
        unsigned char* pimage = image->GetUCharPointer();
        svlBlob* pblobs = blobs->GetBlobsPointer();

        const unsigned int blob_count = blobs->GetBufferUsed();
        const unsigned int width  = image->GetWidth();
        const unsigned int height = image->GetHeight();
        const int right  = width - 1;
        const int bottom = height - 1;
        unsigned int i, j;
        unsigned int dist;

        const unsigned int threshold = ThresholdMM * 100;

        // Remove blobs that are at the image border
        for ( i = 0; i < blob_count; i ++) {
            if (pblobs[i].used &&
                (pblobs[i].left   <= 0  ||
                 pblobs[i].right  >= right ||
                 pblobs[i].top    <= 0   ||
                 pblobs[i].bottom >= bottom)) {
                 pblobs[i].used = false;
            }
        }

        for (j = 0; j < height; j ++) {
            for (i = 0; i < width; i ++) {

                dist = *pdistances;
                if (dist >= 32768) {
                    if ((dist - 32768) < threshold) {
                        *pimage = 128; pimage ++;
                        *pimage = 0;   pimage ++;
                        *pimage = 0;   pimage ++;
                    }
                    else {
                        if (*plabels > 0 && !pblobs[*plabels - 1].used) {
                            *pimage = 0; pimage ++;
                            *pimage = 0; pimage ++;
                            *pimage = 0; pimage ++;
                        }
                        else {
                            pimage += 3;
                        }
                    }
                }
                else {
                    if ((32768 - dist) < threshold) {
                        *pimage = 128; pimage ++;
                        *pimage = 0;   pimage ++;
                        *pimage = 0;   pimage ++;
                    }
                    else {
                        if (*plabels > 0 && !pblobs[*plabels - 1].used) {
                            *pimage = 0; pimage ++;
                            *pimage = 0; pimage ++;
                            *pimage = 0; pimage ++;
                        }
                        else {
                            pimage += 3;
                        }
                    }
                }

                pdistances ++;
                plabels ++;
            }
        }
    }


public:
    unsigned int ThresholdMM;
    unsigned char HistogramThreshold;
    unsigned int MinObjectArea;

private:
    svlFilterInput* DepthInput;
    svlFilterOutput* DepthOutput;
    svlFilterInput* PointCloudInput;
    svlSampleImageMono8* LabelsSample;
    svlSampleImageMono8* HistogramImage;
    svlSampleImageMono32* HistogramLabels;
    svlSampleBlobs* HistogramBlobs;
    svlFilterInput* SegmentsInput;
    vctDynamicMatrix<short> GradX, GradY;
    unsigned int Radius;

    svlSampleImageMono32* TempHistogram;

    svlSampleBlobs* Blobs;
    svlSampleImageMono32* BlobLabels;
    svlImageProcessing::Internals BlobDetectorInternals;
    svlImageProcessing::Internals HistogramBlobInternals;
    svlImageProcessing::Internals PlaneBlobInternals;

    svlSampleImageMono16* PlaneDistances;
    svlSampleImageMono8* PlaneObjectMask;
    svlSampleImageMono32* PlaneObjectLabels;
    svlSampleBlobs* PlaneObjects;
    svlFilterOutput* PlaneDistancesOutput;
    svlFilterOutput* PlaneObjectsOutput;

    vctDynamicVector<vctFloat3> PlaneFitPoints;
    vctDynamicVector<float> PlaneFitWeights;
    vctFloat4 Plane;
};


int main()
{
    svlInitialize();

    svlStreamManager stream;

    svlFilterSourceKinect kinect;
    svlFilterImageOverlay overlay1;
    svlFilterImageBlobDetector blobdetector2;
    svlFilterImageWindow window1, window2, window3, window5;
    svlFilterStreamTypeConverter to_rgb1(svlTypeImageMono16, svlTypeImageRGB);
    svlFilterStreamTypeConverter to_rgb2(svlTypeImageMono8, svlTypeImageRGB);
    svlFilterStreamTypeConverter to_rgb3(svlTypeImageMono32, svlTypeImageRGB);
    svlFilterStreamTypeConverter to_rgb4(svlTypeImageMono32, svlTypeImageRGB);
    svlFilterStreamTypeConverter to_rgb5(svlTypeImageMono16, svlTypeImageRGB);

    svlFilterKinectGradientLabeler gradient_labeler;

    kinect.SetKinectConfigFile("/Users/vagvoba/Code/cisst/source/trunk/saw/components/sawOpenNI/examples/SamplesConfig.xml");

    // Setup Mono16 to RGB converter
    to_rgb1.SetMono16ShiftDown(4);
    to_rgb5.SetMono16ShiftDown(8);

    overlay1.AddInputBlobs("blobs");
    svlOverlayBlobs ovrl_blobs1(0, true, "blobs", 0);
    overlay1.AddOverlay(ovrl_blobs1);
    overlay1.AddQueuedItems();

    // Setup windows
    window1.SetTitle("RGB Image");
    window2.SetTitle("Depth Image");
    window3.SetTitle("Labels");
    window5.SetTitle("Plane Objects");

    // Chain filters to trunk
    stream.SetSourceFilter(&kinect);

    kinect.GetOutput("rgb")->Connect(gradient_labeler.GetInput("rgb"));
    kinect.GetOutput("depth")->Connect(gradient_labeler.GetInput("depth"));
    kinect.GetOutput("pointcloud")->Connect(gradient_labeler.GetInput("pointcloud"));

    gradient_labeler.GetOutput("rgb")->Connect(overlay1.GetInput());
    gradient_labeler.GetOutput("depth")->Connect(to_rgb1.GetInput());
    gradient_labeler.GetOutput("planedistances")->Connect(to_rgb5.GetInput());
    gradient_labeler.GetOutput("planeobjects")->Connect(overlay1.GetInput("blobs"));

    overlay1.GetOutput()->Connect(window1.GetInput());
    to_rgb1.GetOutput()->Connect(window2.GetInput());
    to_rgb4.GetOutput()->Connect(window3.GetInput());
    to_rgb5.GetOutput()->Connect(window5.GetInput());

    // Initialize and start stream
    if (stream.Play() == SVL_OK) {
        std::cout << "Press 'q' to stop stream..." << std::endl;

        std::cerr << "  radius = " << gradient_labeler.GetRadius() << std::endl;
        std::cerr << "  threshold = " << (int)gradient_labeler.HistogramThreshold << std::endl;
        std::cerr << "  elevation threshold [mm] = " << gradient_labeler.ThresholdMM << std::endl;
        std::cerr << "  minimum object area [pixels] = " << gradient_labeler.MinObjectArea << std::endl;

        int ch;
        do {
            ch = cmnGetChar();
            switch (ch) {
                case '-':
                    gradient_labeler.SetRadius(gradient_labeler.GetRadius() - 1);
                    std::cerr << "  radius = " << gradient_labeler.GetRadius() << std::endl;
                break;

                case '=':
                    gradient_labeler.SetRadius(gradient_labeler.GetRadius() + 1);
                    std::cerr << "  radius = " << gradient_labeler.GetRadius() << std::endl;
                break;

                case '9':
                    gradient_labeler.HistogramThreshold --;
                    std::cerr << "  threshold = " << (int)gradient_labeler.HistogramThreshold << std::endl;
                break;

                case '0':
                    gradient_labeler.HistogramThreshold ++;
                    std::cerr << "  threshold = " << (int)gradient_labeler.HistogramThreshold << std::endl;
                break;

                case '7':
                    gradient_labeler.MinObjectArea --;
                    std::cerr << "  minimum object area [pixels] = " << (int)gradient_labeler.MinObjectArea << std::endl;
                break;

                case '8':
                    gradient_labeler.MinObjectArea ++;
                    std::cerr << "  minimum object area [pixels] = " << (int)gradient_labeler.MinObjectArea << std::endl;
                break;

                case '1':
                    gradient_labeler.ThresholdMM --;
                    std::cerr << "  elevation threshold [mm] = " << gradient_labeler.ThresholdMM << std::endl;
                break;

                case '2':
                    gradient_labeler.ThresholdMM ++;
                    std::cerr << "  elevation threshold [mm] = " << gradient_labeler.ThresholdMM << std::endl;
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
    gradient_labeler.GetOutput("depth")->Disconnect();
    gradient_labeler.GetOutput("planedistances")->Disconnect();
    gradient_labeler.GetOutput("planeobjects")->Disconnect();

    return 1;
}

