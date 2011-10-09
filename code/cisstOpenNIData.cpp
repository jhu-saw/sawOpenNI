#include <cisstOpenNI/cisstOpenNI.h>
#include "cisstOpenNIData.h"

//----------------------------------------------------------------------------/
// Callbacks
//----------------------------------------------------------------------------/

// Callback: New user was detected
void XN_CALLBACK_TYPE User_NewUser(xn::UserGenerator& generator, 
                                   XnUserID nId, 
                                   void* pCookie)
{
    printf("New User %d\n", nId);
    
    cisstOpenNIData* openNIDataObject = reinterpret_cast<cisstOpenNIData*>(pCookie);

    openNIDataObject->NewUserCallback(generator,nId);
    openNIDataObject->usrState = CNI_USR_NEW;
}

// Callback: An existing user was lost
void XN_CALLBACK_TYPE User_LostUser(xn::UserGenerator& generator, 
                                    XnUserID nId, 
                                    void* pCookie)
{
    printf("Lost user %d\n", nId);

    cisstOpenNIData* openNIDataObject = reinterpret_cast<cisstOpenNIData*>(pCookie);
    openNIDataObject->usrState = CNI_USR_LOST;
    openNIDataObject->usrCalState = CNI_USR_IDLE;
}


// Callback: Detected a pose
void XN_CALLBACK_TYPE UserPose_PoseDetected(xn::PoseDetectionCapability& capability, 
                                            const XnChar* strPose, 
                                            XnUserID nId, 
                                            void* pCookie)
{
    printf("Pose %s detected for user %d\n", strPose, nId);

    cisstOpenNIData* openNIDataObject = reinterpret_cast<cisstOpenNIData*>(pCookie);

    openNIDataObject->UserPoseDetectedCallback(capability,strPose,nId);
    openNIDataObject->usrState = CNI_USR_POSE;
}

// Callback: Started calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationStart(xn::SkeletonCapability& capability, 
                                                       XnUserID nId, 
                                                       void* pCookie)
{
    printf("Calibration started for user %d\n", nId);
    cisstOpenNIData* openNIDataObject = reinterpret_cast<cisstOpenNIData*>(pCookie);
    openNIDataObject->usrState = CNI_USR_CAL_START;
}

// Callback: Finished calibration
void XN_CALLBACK_TYPE UserCalibration_CalibrationEnd(xn::SkeletonCapability& capability, 
                                                     XnUserID nId, 
                                                     XnBool bSuccess, 
                                                     void* pCookie)
{
    cisstOpenNIData* openNIDataObject = reinterpret_cast<cisstOpenNIData*>(pCookie);

    openNIDataObject->UserCalibrationEndCallback(capability,bSuccess,nId);
    openNIDataObject->usrState = CNI_USR_CAL_END;
}

/// Methods ------------------------------------------------------------------------/

void cisstOpenNIData::NewUserCallback(  xn::UserGenerator& generator, 
                                    XnUserID nId)
{
    if (this->needPose)
    {
        usergenerator.GetPoseDetectionCap().StartPoseDetection(this->strPose, nId);
    }
    else
    {
        usergenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
    }
}

void cisstOpenNIData::UserPoseDetectedCallback( xn::PoseDetectionCapability& capability,
                                            const XnChar* strPose,
                                            XnUserID nId)
{ 
	usergenerator.GetPoseDetectionCap().StopPoseDetection(nId);
	usergenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
}

void cisstOpenNIData::UserCalibrationEndCallback(   xn::SkeletonCapability& capability,
                                                XnBool bSuccess,  
                                                XnUserID nId)
{ 
	if (bSuccess)
	{
		// Calibration succeeded
		printf("Calibration complete, start tracking user %d\n", nId);
		usergenerator.GetSkeletonCap().StartTracking(nId);
        usrCalState = CNI_USR_SUCCESS;
	}
	else
	{
		// Calibration failed
		printf("Calibration failed for user %d\n", nId);
        usrCalState = CNI_USR_FAIL;
		if (this->needPose)
		{
			usergenerator.GetPoseDetectionCap().StartPoseDetection(this->strPose, nId);
            usrCalState = CNI_USR_WAIT;
		}
		else
		{
			usergenerator.GetSkeletonCap().RequestCalibration(nId, TRUE);
		}
	}
}

