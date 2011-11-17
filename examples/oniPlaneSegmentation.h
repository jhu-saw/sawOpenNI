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


class oniPlaneSegmentation
{
public:
    oniPlaneSegmentation();
    ~oniPlaneSegmentation();

    void SetPlaneDistanceThreshold(unsigned int threshold);
    void SetGradientHistogramThreshold(unsigned char threshold);
    void SetMinObjectArea(unsigned int minarea);
    void SetGradientRadius(unsigned int radius);
    unsigned int  GetPlaneDistanceThreshold() const;
    unsigned char GetGradientHistogramThreshold() const;
    unsigned int  GetMinObjectArea() const;
    unsigned int  GetGradientRadius() const;

    bool Process(svlSampleImageRGB*    rgb,
                 svlSampleImage3DMap*  pointcloud,
                 svlSampleImageMono16* planedistance,
                 svlSampleBlobs*       planeobjects);

private:
    bool Initialize(svlSampleImageRGB*    rgb,
                    svlSampleImage3DMap*  pointcloud,
                    svlSampleImageMono16* planedistance,
                    svlSampleBlobs*       planeobjects);

    void ComputeDepthGradientHistogram(vctDynamicMatrixRef<float>        pointcloud,
                                       vctDynamicMatrixRef<unsigned int> histogram);

    void NormalizeGradientHistogram(vctDynamicMatrixRef<unsigned int>  histogram,
                                    vctDynamicMatrixRef<unsigned char> norm_histogram);

    void ThresholdHistogram(svlSampleImageMono8*  image);

    void LabelImage(svlSampleBlobs* segments);

    void FindLargestSegment(svlSampleImageMono32* labels,
                            svlSampleBlobs*       segments);

    void FitPlane(svlSampleImageMono32* labels,
                  svlSampleImageMono16* distances,
                  svlSampleImage3DMap*  points);

    void LabelObjects(svlSampleImageMono16* distances,
                      svlSampleImageMono8*  labels);

    void VisualizePlaneObjects(svlSampleImageRGB*    image,
                               svlSampleImageMono16* distances,
                               svlSampleImageMono32* labels,
                               svlSampleBlobs*       blobs);

    inline unsigned int sqrt_uint32(unsigned int value);

private:
    unsigned int  PlaneDistanceThreshold;
    unsigned char GradientHistogramThreshold;
    unsigned int  MinObjectArea;
    unsigned int  GradientRadius;

    svlSampleBlobs*       Blobs;
    svlSampleBlobs*       HistogramBlobs;
    svlSampleImageMono8*  LabelsSample;
    svlSampleImageMono8*  HistogramImage;
    svlSampleImageMono8*  PlaneObjectMask;
    svlSampleImageMono32* HistogramLabels;
    svlSampleImageMono32* TempHistogram;
    svlSampleImageMono32* TempHistogram2;
    svlSampleImageMono32* BlobLabels;
    svlSampleImageMono32* PlaneObjectLabels;

    vctDynamicMatrix<short> GradX, GradY;

    vctDynamicVector<vctFloat3> PlaneFitPoints;
    vctDynamicVector<float> PlaneFitWeights;
    vctFloat4 Plane;

    svlImageProcessing::Internals BlobDetectorInternals;
    svlImageProcessing::Internals HistogramBlobInternals;
    svlImageProcessing::Internals PlaneBlobInternals;
};

#endif // _oniPlaneSegmentation_H

