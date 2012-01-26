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

#ifndef _oniPlaneSegmentation_H
#define _oniPlaneSegmentation_H

#include <cisstStereoVision.h>


class oniPlane
{
public:
    oniPlane();
    virtual ~oniPlane();

    void Allocate(unsigned int width, unsigned int height);
    void Release();
    void CopyOf(const oniPlane & other);

    int                   ID;    // unique identifier
    int                   Frame; // ID of video frame where plane was detected most recently
    int                   First; // ID of video frame where plane was first detected
    vctInt2               CoW;   // center of weight
    unsigned int          Area;  // number of pixels covered
    svlRect               Rect;  // bounding rectangle
    svlSampleImageMono8*  Mask;  // pixel mask
    svlSampleImageMono16* Dist;  // distance image (pixel value indicates distance from plane)
    svlSampleImageMono32* Hist;  // UV histogram
    vctFloat4             Plane; // 3D plane equation
};


class oniPlaneSegmentation
{
public:
    oniPlaneSegmentation();
    ~oniPlaneSegmentation();

    void SetPlaneID(unsigned int planeid);
    void SetPlaneDistanceThreshold(double threshold);
    void SetColorMatchWeight(double weight);
    void SetGradientHistogramThreshold(unsigned char threshold);
    void SetMinObjectArea(unsigned int minarea);
    void SetGradientRadius(unsigned int radius);
    unsigned int  GetPlaneID() const;
    double        GetPlaneDistanceThreshold() const;
    double        GetColorMatchWeight() const;
    unsigned char GetGradientHistogramThreshold() const;
    unsigned int  GetMinObjectArea() const;
    unsigned int  GetGradientRadius() const;

    bool Process(svlSampleImageRGB*    rgb,
                 svlSampleImage3DMap*  pointcloud,
                 svlSampleImageRGB*    visualized,
                 svlSampleImageMono16* planedistance,
                 svlSampleBlobs*       planeobjects);

    bool GetUVHistogram(svlSampleImageRGB* uvhistogramimage);

private:
    bool Initialize(svlSampleImageRGB*    rgb,
                    svlSampleImage3DMap*  pointcloud,
                    svlSampleImageRGB*    visualized,
                    svlSampleImageMono16* planedistance,
                    svlSampleBlobs*       planeobjects);

    void ComputeDepthGradientHistogram(vctDynamicMatrixRef<float>        pointcloud,
                                       vctDynamicMatrixRef<unsigned int> histogram);

    void NormalizeGradientHistogram(vctDynamicMatrixRef<unsigned int>  histogram,
                                    vctDynamicMatrixRef<unsigned char> norm_histogram);

    void ThresholdHistogram(svlSampleImageMono8* image);

    void LabelImage(svlSampleBlobs* segments, svlSampleImageMono8* gradlabels);

    unsigned int FindLargestSegments(svlSampleImageMono32*       labels,
                                     vctDynamicVector<oniPlane>& planes,
                                     svlSampleBlobs*             segments);

    void FitPlane(unsigned int planeid, svlSampleImage3DMap* points);

    void LabelObjects(unsigned int planeid);

    void CalculateColorHistogram(unsigned int planeid, svlSampleImageRGB* yuvimage);

    void FilterLabels(unsigned int planeid, svlSampleImageRGB* yuvimage);

    void ComputePlaneStats(unsigned int planeid);

    void TrackPlanes();

    void DrawPlanes(svlSampleImageRGB* visualized);

    void VisualizePlaneObjects(svlSampleImageRGB*    image,
                               svlSampleImageRGB*    visualized,
                               svlSampleImageMono8*  planelabels,
                               svlSampleImageMono32* objectlabels,
                               svlSampleBlobs*       blobs);

    void CreateColorHistogramImage(svlSampleImageMono32* uvhistogram,
                                   svlSampleImageRGB*    uvhistogramimage);

    svlRGB GetUniqueColorForNumber(unsigned int number);

    inline unsigned int sqrt_uint32(unsigned int value);

private:
    bool Initialized;

    unsigned int  FrameCounter; // Zero based frame index
    unsigned int  PlaneIDCounter; // Zero based plane ID
    unsigned int  PlaneID;      // ID of plane to be visualized
    unsigned int  PlaneDistanceThreshold;
    unsigned int  ColorMatchWeight;
    unsigned char GradientHistogramThreshold;
    unsigned int  MinObjectArea;
    unsigned int  GradientRadius;
    unsigned int  MaxPlaneMatchError;

    svlSampleBlobs*       Blobs;
    svlSampleBlobs*       HistogramBlobs;
    svlSampleImageMono8*  GradientLabels;
    svlSampleImageMono8*  HistogramImage;
    svlSampleImageMono8*  TempMask;
    svlSampleImageMono32* HistogramLabels;
    svlSampleImageMono32* TempHistogram;
    svlSampleImageMono32* TempHistogram2;
    svlSampleImageMono32* BlobLabels;
    svlSampleImageMono32* PlaneObjectLabels;
    svlSampleImageRGB*    YUVImage;
    svlSampleImageRGB*    UVHistogramImage;

    vctDynamicMatrix<short> GradX, GradY;
    vctDynamicVector<vctFloat3> PlaneFitPoints;
    vctDynamicVector<float> PlaneFitWeights;

    svlImageProcessing::Internals BlobDetectorInternals;
    svlImageProcessing::Internals HistogramBlobInternals;
    svlImageProcessing::Internals PlaneBlobInternals;

    // Plane cache used as work buffer
    vctDynamicVector<oniPlane> PlaneCache;
    unsigned int PlaneCacheSize;

    // Plane history
    vctDynamicVector<oniPlane> PlaneHistory;
    vctDynamicVector<int> VisiblePlanePositions;
    unsigned int NumberOfVisiblePlanes;
    int PlaneHistoryPosition;
};

#endif // _oniPlaneSegmentation_H

