/* -*- Mode: C++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-    */
/* ex: set filetype=cpp softtabstop=4 shiftwidth=4 tabstop=4 cindent expandtab: */

/*Frame
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

#include "oniPlaneSegmentation.h"
#include "planefit.h"

#define _oniPlane_VERBOSE   0


/**********************/
/*** oniPlane class ***/
/**********************/

oniPlane::oniPlane() :
    ID(-1),
    Frame(-1),
    First(-1),
    Area(0),
    Rect(-1, -1, -1, -1),
    Mask(0),
    Dist(0),
    Hist(0)
{
    CoW.SetAll(-1);
    Plane.SetAll(0.0);
}

oniPlane::~oniPlane()
{
    Release();
}

void oniPlane::Allocate(unsigned int width, unsigned int height)
{
    Release();

    Mask = new svlSampleImageMono8;
    Dist = new svlSampleImageMono16;
    Hist = new svlSampleImageMono32;
    Mask->SetSize(width, height);
    Dist->SetSize(width, height);
    Hist->SetSize(256, 256);
}

void oniPlane::Release()
{
    delete Mask;
    delete Dist;
    delete Hist;
}

void oniPlane::CopyOf(const oniPlane & other)
{
    ID    = other.ID;
    Frame = other.Frame;
    First = other.First;
    CoW.Assign(other.CoW);
    Area  = other.Area;
    Rect.Assign(other.Rect);
    Mask->CopyOf(other.Mask);
    Dist->CopyOf(other.Dist);
    Hist->CopyOf(other.Hist);
    Plane.Assign(other.Plane);
}


/**********************************/
/*** oniPlaneSegmentation class ***/
/**********************************/

oniPlaneSegmentation::oniPlaneSegmentation() :
    Initialized(false),
    FrameCounter(0),
    PlaneIDCounter(0),
    PlaneID(0),
    PlaneDistanceThreshold(700),
    ColorMatchWeight(0),
    GradientHistogramThreshold(20),
    MinObjectArea(120),
    GradientRadius(25),
    MaxPlaneMatchError(30)
{
    GradientLabels    = new svlSampleImageMono8;
    HistogramImage    = new svlSampleImageMono8;
    HistogramLabels   = new svlSampleImageMono32;
    HistogramBlobs    = new svlSampleBlobs;
    TempHistogram     = new svlSampleImageMono32;
    TempHistogram2    = new svlSampleImageMono32;
    Blobs             = new svlSampleBlobs;
    BlobLabels        = new svlSampleImageMono32;
    TempMask          = new svlSampleImageMono8;
    PlaneObjectLabels = new svlSampleImageMono32;
    YUVImage          = new svlSampleImageRGB;
    UVHistogramImage  = new svlSampleImageRGB;

    PlaneCache.SetSize(3); // Maximum number of planes detected on a frame
    VisiblePlanePositions.SetSize(PlaneCache.size());
    PlaneHistory.SetSize(20); // Number of recent planes the algorithm remembers

    HistogramImage->SetSize(256, 256);
    HistogramLabels->SetSize(HistogramImage);
    TempHistogram->SetSize(HistogramImage);
    TempHistogram2->SetSize(HistogramImage);
    HistogramBlobs->SetChannelCount(1);
    HistogramBlobs->SetBufferSize(10);
    Blobs->SetChannelCount(1);
    Blobs->SetBufferSize(1000);
    UVHistogramImage->SetSize(256, 256);
}

oniPlaneSegmentation::~oniPlaneSegmentation()
{
    if (GradientLabels)    delete GradientLabels;
    if (HistogramImage)    delete HistogramImage;
    if (HistogramLabels)   delete HistogramLabels;
    if (HistogramBlobs)    delete HistogramBlobs;
    if (TempHistogram)     delete TempHistogram;
    if (TempHistogram2)    delete TempHistogram2;
    if (Blobs)             delete Blobs;
    if (BlobLabels)        delete BlobLabels;
    if (TempMask)          delete TempMask;
    if (PlaneObjectLabels) delete PlaneObjectLabels;
    if (YUVImage)          delete YUVImage;
    if (UVHistogramImage)  delete UVHistogramImage;
}

void oniPlaneSegmentation::SetPlaneID(unsigned int planeid)
{
    if (planeid >= PlaneCache.size()) planeid = 0;
    PlaneID = planeid;
}

unsigned int oniPlaneSegmentation::GetPlaneID() const
{
    return PlaneID;
}

void oniPlaneSegmentation::SetPlaneDistanceThreshold(double threshold)
{
    if (threshold >= 0.0) PlaneDistanceThreshold = static_cast<unsigned int>((threshold + 0.005) * 100.0);
}

double oniPlaneSegmentation::GetPlaneDistanceThreshold() const
{
    return 0.01 * PlaneDistanceThreshold;
}

void oniPlaneSegmentation::SetColorMatchWeight(double weight)
{
    if (weight >= 0.0) ColorMatchWeight = static_cast<unsigned int>((weight + 0.0005) * 1000.0);
}

double oniPlaneSegmentation::GetColorMatchWeight() const
{
    return 0.001 * ColorMatchWeight;
}

void oniPlaneSegmentation::SetGradientHistogramThreshold(unsigned char threshold)
{
    GradientHistogramThreshold = threshold;
}

unsigned char oniPlaneSegmentation::GetGradientHistogramThreshold() const
{
    return GradientHistogramThreshold;
}

void oniPlaneSegmentation::SetMinObjectArea(unsigned int minarea)
{
    MinObjectArea = minarea;
}

unsigned int oniPlaneSegmentation::GetMinObjectArea() const
{
    return MinObjectArea;
}

void oniPlaneSegmentation::SetGradientRadius(unsigned int radius)
{
    if (radius > 0) GradientRadius = radius;
}

unsigned int oniPlaneSegmentation::GetGradientRadius() const
{
    return GradientRadius;
}

bool oniPlaneSegmentation::Process(svlSampleImageRGB* rgb, svlSampleImage3DMap* pointcloud, svlSampleImageRGB* visualized, svlSampleImageMono16* planedistance, svlSampleBlobs* planeobjects)
{
    if (!Initialized) {
        if (!Initialize(rgb, pointcloud, visualized, planedistance, planeobjects)) return false;
    }

    // Calculate YUV image
    svlConverter::RGB24toYUV444(rgb->GetUCharPointer(), YUVImage->GetUCharPointer(), YUVImage->GetWidth() * YUVImage->GetHeight());

    // Compute gradient histogram
    ComputeDepthGradientHistogram(pointcloud->GetMatrixRef(), TempHistogram->GetMatrixRef());

    // Blur gradient histogram to smooth clusters
    vctDynamicMatrix<double> kernel(3, 3);
    kernel.SetAll(1.0 / 9.0);
    svlImageProcessing::Convolution(TempHistogram, 0, TempHistogram2, 0, kernel);
    svlImageProcessing::Convolution(TempHistogram2, 0, TempHistogram, 0, kernel);

    // Normalize gradient histogram
    NormalizeGradientHistogram(TempHistogram->GetMatrixRef(), HistogramImage->GetMatrixRef());

    // Apply thresholding to gradient histogram
    ThresholdHistogram(HistogramImage);

    // Find blobs on thresholded gradient histogram
    svlImageProcessing::LabelBlobs(HistogramImage, HistogramLabels, HistogramBlobInternals);
    svlImageProcessing::GetBlobsFromLabels(HistogramImage, HistogramLabels, HistogramBlobs, HistogramBlobInternals,
                                           0, 0,
                                           0.0, 0.0);

    // Label image pixels according to blobs on the gradient histogram
    LabelImage(HistogramBlobs, GradientLabels);

    // Find blobs on the labeled image
    svlImageProcessing::LabelBlobs(GradientLabels, BlobLabels, BlobDetectorInternals);
    svlImageProcessing::GetBlobsFromLabels(GradientLabels, BlobLabels, Blobs, BlobDetectorInternals,
                                           0, 0,
                                           0.0, 0.0);

    // Find largest blobs on labeled image, those are the largest planar surfaces
    PlaneCacheSize = FindLargestSegments(BlobLabels, PlaneCache, Blobs);

    // Go through all planes found
    unsigned int planeid;
    for (planeid = 0; planeid < PlaneCacheSize; planeid ++)
    {
        // Fit 3D plane on the points of the planar surface and
        // calculate distance of every pixel from the plane
        FitPlane(planeid, pointcloud);

        // Label pixels according to their distance from the plane:
        // - pixels on the plane        = 0
        // - pixels under the plane     = 128
        // - pixels on top of the plane = 255
        LabelObjects(planeid);

        // Calculate UV histogram for plane
        CalculateColorHistogram(planeid, YUVImage);

        if (ColorMatchWeight) {
            // Refine plane definition using color similarity measures
            FilterLabels(planeid, YUVImage);

            // Close pixel size holes on the plane
            svlImageProcessing::Erode(PlaneCache[planeid].Mask, 0, TempMask, 0, 1);
            svlImageProcessing::Dilate(TempMask, 0, PlaneCache[planeid].Mask, 0, 1);
        }

        // Update plane statistics
        ComputePlaneStats(planeid);

        PlaneCache[planeid].Frame = static_cast<int>(FrameCounter);
    }

    // Plane tracking (match plane cache with previously detected planes)
    TrackPlanes();

    // Draw planes in uniques color on visualization image
    DrawPlanes(visualized);

    planeid = PlaneID;
    if (planeid < PlaneCacheSize) {
/*
        // Find blobs on the image labeled according to distance of pixels from plane
        svlImageProcessing::LabelBlobs(PlaneCache[planeid].Mask, PlaneObjectLabels, PlaneBlobInternals);
        svlImageProcessing::GetBlobsFromLabels(PlaneCache[planeid].Mask, PlaneObjectLabels, planeobjects, PlaneBlobInternals,
                                               MinObjectArea, 0,
                                               0.0, 0.0);

        // Visualize results:
        // - plane pixels are colored blue
        // - blobs adjacent to the image border are colored black
        // - blobs under or on top of the plane are drawn in their original RGB color
        VisualizePlaneObjects(rgb, visualized, PlaneCache[planeid].Mask, PlaneObjectLabels, planeobjects);
*/
        // Draw UV histogram
        CreateColorHistogramImage(PlaneCache[planeid].Hist, UVHistogramImage);

        planedistance->CopyOf(PlaneCache[planeid].Dist);
    }

    FrameCounter ++;

    return true;
}

bool oniPlaneSegmentation::GetUVHistogram(svlSampleImageRGB* uvhistogramimage)
{
    if (!uvhistogramimage) return false;
    vctDynamicMatrix<double> kernel(3, 3);
    kernel.SetAll(1.1 / 9.0);
    svlImageProcessing::Convolution(UVHistogramImage, 0, uvhistogramimage, 0, kernel);
    svlImageProcessing::Convolution(uvhistogramimage, 0, UVHistogramImage, 0, kernel);
    svlImageProcessing::Convolution(UVHistogramImage, 0, uvhistogramimage, 0, kernel);
    return true;
}

bool oniPlaneSegmentation::Initialize(svlSampleImageRGB* rgb, svlSampleImage3DMap* pointcloud, svlSampleImageRGB* visualized, svlSampleImageMono16* planedistance, svlSampleBlobs* planeobjects)
{
    if (!rgb || !pointcloud || !visualized || !planedistance || !planeobjects) return false;

    const unsigned int width  = rgb->GetWidth();
    const unsigned int height = rgb->GetHeight();
    if (width < 1 || height < 1) return false;

    if (pointcloud->GetWidth() != width || pointcloud->GetHeight() != height) return false;

    // Resize output samples if needed
    planedistance->SetSize(width, height);
    visualized->SetSize(width, height);
    planeobjects->SetChannelCount(1);
    planeobjects->SetBufferSize(1000);

    // Resize buffers if needed
    GradientLabels->SetSize(width, height);
    BlobLabels->SetSize(width, height);
    TempMask->SetSize(width, height);
    PlaneObjectLabels->SetSize(width, height);
    YUVImage->SetSize(width, height);
    GradX.SetSize(height, width);
    GradY.SetSize(height, width);

    for (unsigned int i = 0; i < PlaneCache.size(); i ++) {
        PlaneCache[i].Allocate(width, height);
    }

    for (unsigned int i = 0; i < PlaneHistory.size(); i ++) {
        PlaneHistory[i].Allocate(width, height);
    }

    VisiblePlanePositions.SetAll(-1);
    NumberOfVisiblePlanes = 0;
    PlaneHistoryPosition  = -1;
    FrameCounter          = 0;

    Initialized = true;
    return true;
}

void oniPlaneSegmentation::ComputeDepthGradientHistogram(vctDynamicMatrixRef<float> pointcloud, vctDynamicMatrixRef<unsigned int> histogram)
{
    const int radius = GradientRadius;
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

void oniPlaneSegmentation::NormalizeGradientHistogram(vctDynamicMatrixRef<unsigned int> histogram, vctDynamicMatrixRef<unsigned char> norm_histogram)
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

void oniPlaneSegmentation::ThresholdHistogram(svlSampleImageMono8* image)
{
    const unsigned int pixel_count = image->GetWidth() * image->GetHeight();
    unsigned char* phist = image->GetUCharPointer();
    for (unsigned int i = 0; i < pixel_count; i ++) {
        if (*phist > GradientHistogramThreshold) *phist = GradientHistogramThreshold;
        else *phist = 0;
        phist ++;
    }
}

void oniPlaneSegmentation::LabelImage(svlSampleBlobs* segments, svlSampleImageMono8* gradlabels)
{
    const unsigned int segment_count = std::min(segments->GetBufferUsed(), 3u);
    const unsigned int pixel_count = gradlabels->GetWidth() * gradlabels->GetHeight();
    const int hist_center = TempHistogram->GetWidth() / 2;

    unsigned char *img = gradlabels->GetUCharPointer();
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

unsigned int oniPlaneSegmentation::sqrt_uint32(unsigned int value)
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

unsigned int oniPlaneSegmentation::FindLargestSegments(svlSampleImageMono32* labels, vctDynamicVector<oniPlane>& planes, svlSampleBlobs* segments)
{
    const unsigned int segments_to_find = planes.size();
    if (segments_to_find < 1) return 0;

    const unsigned int segment_count = segments->GetBufferUsed();
    const unsigned int pixel_count = labels->GetWidth() * labels->GetHeight();
    unsigned char *pplnlabels;
    unsigned int *plabels;

    int prev_largest = 10000000, largest_size, label, ID;
    unsigned int N;
    svlBlob blob;

    for (N = 0; N < segments_to_find; N ++) {

        // Find Nth largest blob
        largest_size = -1; label = -1; ID = -1;
        for (unsigned int i = 0; i < segment_count; i ++) {

            segments->GetBlob(i, blob);
            if (blob.used &&
                static_cast<int>(blob.area) > largest_size &&
                static_cast<int>(blob.area) < prev_largest) {
                largest_size = blob.area;
                label = blob.label;
                ID = i + 1;
            }
        }
        prev_largest = largest_size;
        if (ID < 1) break;

        // Mark blob on labels image
        plabels    = labels->GetPointer();
        pplnlabels = planes[N].Mask->GetPointer();
        for (unsigned int i = 0;i < pixel_count; i ++, plabels ++, pplnlabels ++) {
            if (static_cast<int>(*plabels) != ID) *pplnlabels = 0;
            else *pplnlabels = 255;
        }
    }

    // Zero the unused plane label maps
    for (; N < segments_to_find; N ++) memset(planes[N].Mask->GetPointer(), 0, pixel_count * 4);

    return N;
}

void oniPlaneSegmentation::FitPlane(unsigned int planeid, svlSampleImage3DMap* points)
{
    svlSampleImageMono8*  labels    = PlaneCache[planeid].Mask;
    svlSampleImageMono16* distances = PlaneCache[planeid].Dist;
    const unsigned int width  = labels->GetWidth();
    const unsigned int height = labels->GetHeight();

    if (PlaneFitPoints.size() < (width * height)) PlaneFitPoints.SetSize(width * height);

    vctDynamicMatrixRef<float> pointcloud(points->GetMatrixRef());
    unsigned char* plabels = labels->GetPointer();
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

        fitter.Calculate(point_vec, PlaneFitWeights, PlaneCache[planeid].Plane);

        // Check distance of points from plane
        const float a = PlaneCache[planeid].Plane[0];
        const float b = PlaneCache[planeid].Plane[1];
        const float c = PlaneCache[planeid].Plane[2];
        const float d = PlaneCache[planeid].Plane[3];
        const float normlen = sqrt(a * a + b * b + c * c);
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

void oniPlaneSegmentation::LabelObjects(unsigned int planeid)
{
    svlSampleImageMono8*  planelabels = PlaneCache[planeid].Mask;
    svlSampleImageMono16* distances   = PlaneCache[planeid].Dist;

    const unsigned int threshold   = PlaneDistanceThreshold;
    const unsigned int pixel_count = distances->GetWidth() * distances->GetHeight();

    unsigned short* pdistances = distances->GetPointer();
    unsigned char*  plab       = planelabels->GetPointer();

    unsigned int dist;

    for (unsigned int i = 0; i < pixel_count; i ++, pdistances ++, plab ++) {
        dist = *pdistances;
        if (dist >= 32768) {
            if ((dist - 32768) >= threshold) *plab = 255;
            else *plab = 0;
        }
        else {
            if ((32768 - dist) >= threshold) *plab = 128;
            else *plab = 0;
        }
    }
}

void oniPlaneSegmentation::CalculateColorHistogram(unsigned int planeid, svlSampleImageRGB* yuvimage)
{
    svlSampleImageMono8*  planelabels = PlaneCache[planeid].Mask;
    svlSampleImageMono32* uvhistogram = PlaneCache[planeid].Hist;

    const unsigned int pixel_count = yuvimage->GetWidth() * yuvimage->GetHeight();

    unsigned char* pyuv  = yuvimage->GetUCharPointer();
    unsigned char* plab  = planelabels->GetUCharPointer();
    unsigned int*  phist = uvhistogram->GetPointer();

    unsigned int size = 0;

    memset(uvhistogram->GetUCharPointer(), 0, uvhistogram->GetDataSize());

    // Calculate histogram
    for (unsigned int i = 0; i < pixel_count; i ++, pyuv += 3, plab ++) {
        if (*plab == 0) {
            phist[256 * pyuv[2] + pyuv[1]] ++;
            size ++;
        }
    }

    if (size > 0) {
        // Normalize histogram to volume = 100000
        unsigned int histsize = 256 * 256;
        for (unsigned int i = 0; i < histsize; i ++, phist ++) {
            *phist = ((*phist) * 100000) / size;
        }
    }
}

void oniPlaneSegmentation::FilterLabels(unsigned int planeid, svlSampleImageRGB* yuvimage)
{
    svlSampleImageMono8*  planelabels = PlaneCache[planeid].Mask;
    svlSampleImageMono16* distances   = PlaneCache[planeid].Dist;
    svlSampleImageMono32* uvhistogram = PlaneCache[planeid].Hist;

    const unsigned int dist_threshold = PlaneDistanceThreshold; // [100th  of a mm distance]
    const unsigned int col_weight     = ColorMatchWeight;
    const unsigned int pixel_count    = yuvimage->GetWidth() * yuvimage->GetHeight();

    unsigned short* pdistances = distances->GetPointer();
    unsigned char*  plab       = planelabels->GetPointer();
    unsigned char*  pyuv       = yuvimage->GetUCharPointer();
    unsigned int*   phist      = uvhistogram->GetPointer();

    unsigned int dist, colmatch;
    unsigned char y, u, v;

    for (unsigned int i = 0; i < pixel_count; i ++, plab ++, pdistances ++) {
        if (*plab == 255) { // When pixel is on top of the surface
            y = *pyuv; pyuv ++;
            u = *pyuv; pyuv ++;
            v = *pyuv; pyuv ++;

            colmatch = col_weight * phist[256 * v + u] / 1000;

            dist = *pdistances;
            if (dist >= 32768) dist -= 32768;
            else dist = 32768 - dist;

            if (colmatch < dist) dist -= colmatch;
            else dist = 0;

            if (dist <= dist_threshold) *plab = 0;
        }
        else pyuv += 3;
    }
}

void oniPlaneSegmentation::ComputePlaneStats(unsigned int planeid)
{
    const int width  = PlaneCache[planeid].Mask->GetWidth();
    const int height = PlaneCache[planeid].Mask->GetHeight();

    unsigned char* pplnlab = PlaneCache[planeid].Mask->GetPointer();
    int i, j, wx = 0, wy = 0, left = 10000, right = -1, top = 10000, bottom = -1;
    unsigned int area = 0;

    for (j = 0; j < height; j ++) {
        for (i = 0; i < width; i ++, pplnlab ++) {
            if (*pplnlab == 0) {
                // Pixel belongs to plane
                area ++;
                wx += i;
                wy += j;
                if (left > i)   left = i;
                if (right < i)  right = i;
                if (top > i)    top = i;
                if (bottom < i) bottom = i;
            }
        }
    }

    if (area > 0) {
        PlaneCache[planeid].CoW[0] = wx / area;
        PlaneCache[planeid].CoW[1] = wy / area;
        PlaneCache[planeid].Area   = area;
        PlaneCache[planeid].Rect.Assign(left, top, right, bottom);
    }
    else {
        PlaneCache[planeid].CoW[0] = -1;
        PlaneCache[planeid].CoW[1] = -1;
        PlaneCache[planeid].Area   = 0;
        PlaneCache[planeid].Rect.Assign(-1, -1, -1, -1);
    }
}

void oniPlaneSegmentation::TrackPlanes()
{
    if (PlaneCacheSize < 1) {
        // Lost tracking of all planes
        // Update plane history
        // Mark all previously seen planes as non-visible
        VisiblePlanePositions.SetAll(-1);
        NumberOfVisiblePlanes = 0;
        return;
    }

    unsigned int i, j, v;
    int dx, dy, ac, ah, dist, areadiff, error, bestmatch, minerror;

    // Match previously visible planes to plane cache
    for (j = 0; j < PlaneCacheSize; j ++) {
        bestmatch = -1;
        minerror  = 10000000;

        for (i = 0; i < NumberOfVisiblePlanes; i ++) {
            v = VisiblePlanePositions[i];

            // calculate distance between center-of-weights
            dx = PlaneCache[j].CoW[0] - PlaneHistory[v].CoW[0];
            dy = PlaneCache[j].CoW[1] - PlaneHistory[v].CoW[1];
            dist = sqrt_uint32(dx * dx + dy * dy);

            // calculate change in area size [percentages]
            ac = static_cast<int>(PlaneCache[j].Area);
            ah = static_cast<int>(PlaneHistory[v].Area);
            areadiff = std::abs(ac - ah) * 100 / ah;

            error = dist + areadiff;
            if (error < minerror) {
                minerror = error;
                bestmatch = v;
            }
        }

        if (bestmatch >= 0 && minerror <= static_cast<int>(MaxPlaneMatchError)) {
            // Match found
            PlaneCache[j].ID = PlaneHistory[bestmatch].ID;

#if _oniPlane_VERBOSE > 0
            std::cerr << ". Matched cache item " << j << " with history item at position " << bestmatch
                      << "(ID=" << PlaneHistory[bestmatch].ID << ", error=" << minerror << ")" << std::endl;
#endif // _oniPlane_VERBOSE
        }
        else {
            // Generate new plane ID
            PlaneCache[j].ID    = PlaneIDCounter ++;
            PlaneCache[j].First = FrameCounter;

#if _oniPlane_VERBOSE > 0
            std::cerr << "+ Add cache item " << j << " to history at position " << PlaneHistoryPosition
                      << "(ID=" << PlaneCache[j].ID << ", minerror=" << minerror << ")" << std::endl;
#endif // _oniPlane_VERBOSE
        }

        // Increment history position
        PlaneHistoryPosition ++;
        // Roll over if needed
        if (PlaneHistoryPosition >= static_cast<int>(PlaneHistory.size())) PlaneHistoryPosition = 0;

        // Copy plane data to history
        PlaneHistory[PlaneHistoryPosition].CopyOf(PlaneCache[j]);

        VisiblePlanePositions[j] = PlaneHistoryPosition;
    }

    NumberOfVisiblePlanes = PlaneCacheSize;
}

void oniPlaneSegmentation::DrawPlanes(svlSampleImageRGB* visualized)
{
    const unsigned int pixel_count = visualized->GetWidth() * visualized->GetHeight();

    unsigned char *pplnlab, *pvis;
    std::stringstream strstr;
    unsigned int i, j;
    svlRGB rgb;
    int pos;

    memset(visualized->GetUCharPointer(), 0, visualized->GetDataSize());

    for (j = 0; j < NumberOfVisiblePlanes; j ++) {
        if ((PlaneCache[j].Frame - PlaneCache[j].First) < 10) continue;

        pplnlab = PlaneCache[j].Mask->GetPointer();
        pvis    = visualized->GetUCharPointer();
        pos     = VisiblePlanePositions[j];
        rgb     = GetUniqueColorForNumber(PlaneHistory[pos].ID);

        for (i = 0; i < pixel_count; i ++, pplnlab ++) {
            if (*pplnlab == 0) {
                // Pixel belongs to plane
                *pvis = rgb.r; pvis ++;
                *pvis = rgb.g; pvis ++;
                *pvis = rgb.b; pvis ++;
            }
            else pvis += 3;
        }
    }

    for (j = 0; j < NumberOfVisiblePlanes; j ++) {
        if ((PlaneCache[j].Frame - PlaneCache[j].First) < 10) continue;

        pos = VisiblePlanePositions[j];
        strstr.str("");
        strstr << PlaneHistory[pos].ID;
        svlDraw::Text(visualized, 0,
                      svlPoint2D(PlaneHistory[pos].CoW[0], PlaneHistory[pos].CoW[1]),
                      strstr.str(),
                      14,
                      svlRGB(255, 255, 255));
    }
}

void oniPlaneSegmentation::VisualizePlaneObjects(svlSampleImageRGB* image, svlSampleImageRGB* visualized, svlSampleImageMono8* planelabels, svlSampleImageMono32* objectlabels, svlSampleBlobs* blobs)
{
    const unsigned int width       = image->GetWidth();
    const unsigned int height      = image->GetHeight();
    const unsigned int pixel_count = width * height;
    const unsigned int blob_count  = blobs->GetBufferUsed();
    const int          right       = width  - 1;
    const int          bottom      = height - 1;

    unsigned char* pplnlab = planelabels->GetPointer();
    unsigned int*  pobjlab = objectlabels->GetPointer();
    unsigned char* pimage  = image->GetUCharPointer();
    unsigned char* pvis    = visualized->GetUCharPointer();
    svlBlob*       pblobs  = blobs->GetBlobsPointer();

    // Remove blobs that are at the image border
    for (unsigned int i = 0; i < blob_count; i ++) {
        if (pblobs[i].used &&
            (pblobs[i].left   <= 0  ||
             pblobs[i].right  >= right ||
             pblobs[i].top    <= 0   ||
             pblobs[i].bottom >= bottom)) {
             pblobs[i].used = false;
        }
    }

    for (unsigned int i = 0; i < pixel_count; i ++, pimage += 3, pplnlab ++, pobjlab ++) {
        if (*pplnlab == 0) {
            // Pixel belongs to plane
            *pvis = 128; pvis ++;
            *pvis = 0;   pvis ++;
            *pvis = 0;   pvis ++;
        }
        else {
            if (*pplnlab == 255) {
                // Pixel is on top of the surface
                if (*pobjlab > 0) {
                    // Pixel belongs to an object on top of the surface
                    if (pblobs[*pobjlab - 1].used) {
                        // Object is valid
                        *pvis = pimage[0]; pvis ++;
                        *pvis = pimage[1]; pvis ++;
                        *pvis = pimage[2]; pvis ++;
                    }
                    else {
                        // Object has been filtered out
                        *pvis = 0; pvis ++;
                        *pvis = 0; pvis ++;
                        *pvis = 0; pvis ++;
                    }
                }
                else {
                    // Pixel doesn't belong to an object on top of the surface
                    *pvis = 0; pvis ++;
                    *pvis = 0; pvis ++;
                    *pvis = 0; pvis ++;
                }
            }
            else {
                // Pixel is under the surface
                *pvis = 32; pvis ++;
                *pvis = 0; pvis ++;
                *pvis = 0; pvis ++;
            }
        }
    }
}

void oniPlaneSegmentation::CreateColorHistogramImage(svlSampleImageMono32* uvhistogram, svlSampleImageRGB* uvhistogramimage)
{
    const unsigned int histsize = 256 * 256;

    unsigned int*  phist    = uvhistogram->GetPointer();
    unsigned char* phistimg = uvhistogramimage->GetPointer();

    unsigned int i, largest = 0;
    unsigned char prob;

    // Find max. value
    for (i = 0; i < histsize; i ++, phist ++) {
        if (*phist > largest) largest = *phist;
    }

    if (largest > 0) {
        // Normalize and draw histogram in YUV color space
        for (phist = uvhistogram->GetPointer(), i = 0; i < histsize; i ++, phist ++) {
            prob = (*phist * 255) / largest;
            *phistimg = prob; phistimg ++;
            *phistimg = prob; phistimg ++;
            *phistimg = prob; phistimg ++;
        }
    }
    else {
        memset(phistimg, 0, uvhistogramimage->GetDataSize());
    }
}

svlRGB oniPlaneSegmentation::GetUniqueColorForNumber(unsigned int number)
{
    svlRGB rgb;
    const unsigned char intensity = 128;

    switch (number % 6) {
        case 0:  rgb.Assign(intensity, 0, 0);                 break;
        case 1:  rgb.Assign(0, intensity, 0);                 break;
        case 2:  rgb.Assign(0, 0, intensity);                 break;
        case 3:  rgb.Assign(intensity, intensity, 0);         break;
        case 4:  rgb.Assign(0, intensity, intensity);         break;
        case 5:  rgb.Assign(intensity, 0, intensity);         break;
        default: rgb.Assign(intensity, intensity, intensity); break; // should not occur
    }

    return rgb;
}

