# Run vesel and material instance predicion on all images in input folder
###############################################################################################################################

#--------------------------------------Running pramters-------------------------------------------------------------------------------
UseGPU=True # run on GPU (true) or CPU (false) # Note this system is slow on GPU and very very slow on CPU
FreezeBatchNorm_EvalON=True # Freeze the upddating of bath normalization statitics -->  use net.eval()

VesIOUthresh=0.4 #7 # IOU quality threshold for predicted vessel instance to be accepted
NumVessCycles=1 # Number of attempts to search for vessel instance, increase the probability to find vessel but also running time
UseIsVessel=False #True # Only If the vessel instance net was trained with COCO  it can predict whether the instance belongs to a vessel  which can help to remove a false segment
IsVesThresh=0.5

#...........................................Trained net Paths.......................................................................
SemanticNetTrainedModelPath="utils/Semantic/logs/1000000_Semantic_withCOCO_AllSets.torch"
InstanceVesselNetTrainedModelPath="utils/InstanceVesselWithCOCO/logs/Vessel_Coco_610000_Trained_on_All_Sets.torch"
#...............................Imports..................................................................

import os
import torch
import numpy as np
import utils.Semantic.FCN_NetModel as SemanticNet
import utils.Semantic.CategoryDictionary as CatDic
import utils.InstanceVessel.FCN_NetModel as VesselInstNet

import cv2



#---------------------------Sub output dirs-------------------------------------------------
#OutVizIns=OutDir+"/InstanceVisual/" # instance annotations overlay on image for visuallization


#----make ouput dirs--------
#if not os.path.exists(OutDir): os.makedirs(OutDir)
#if not os.path.exists(OutVizIns): os.makedirs(OutVizIns)
#print("Loading nets")

#=========================Load Semantic net====================================================================================================================
#print("Load semantic net")
SemNet=SemanticNet.Net(CatDic.CatNum) # Create net and load pretrained encoder path
if UseGPU:
     print("USING GPU")
     SemNet.load_state_dict(torch.load(SemanticNetTrainedModelPath))#180000.torch"))
else:
     print("USING CPU")
     SemNet.load_state_dict(torch.load(SemanticNetTrainedModelPath, map_location=torch.device('cpu')))  # 180000.torch"))

# SemNet.cuda()
# SemNet.half()
# #SemNet.eval()

# #=======================Load vessel Instance net======================================================================================================================
#print("Load vessel instance  net")
VesNet=VesselInstNet.Net(NumClasses=2) # Create net and load pretrained
VesNet.AddEvaluationClassificationLayers(NumClass=1)
if UseGPU:
      VesNet.load_state_dict(torch.load(InstanceVesselNetTrainedModelPath))
else:
      VesNet.load_state_dict(torch.load(InstanceVesselNetTrainedModelPath, map_location=torch.device('cpu')))


print("Finished loading nets")

############################################################################################################################################################################################################################################################
############################################################Split vessel region to vessel instances#########################################################################################################################################################
def FindVesselInstances(Img,VesselsMask): # Split the VesselMask into vessel instances using GES net for instances

            H,W=VesselsMask.shape
            InsList = np.zeros([0, H,W]) # list of vessels instances
            InstRank = [] # Score predicted for the instace
            InstMap = np.zeros([H,W],int) # map of instances that were already discovered
            NInst=0 # Number of instances
            OccupyMask=np.zeros([H,W],int) # Region that have been segmented
            ROIMask=VesselsMask.copy() # Region to be segmented
            NumPoints= int(340000 * 10/(H*W)) # Num instace points to guess per experiment
#===============Generate instance map========================================================================================
            for cycle in range(NumVessCycles):
                # ........................Generate input for the instance net,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,
                PointerMask=np.zeros([NumPoints,H,W],dtype=float) # Pointer mask
                ROI = np.ones([NumPoints, H, W], dtype=float)
                ImgList= np.ones([NumPoints, H, W,3], dtype=float)

                for i in range(NumPoints): # Generate pointer mask
                        while(True):
                            px = np.random.randint(W)
                            py = np.random.randint(H)
                            if (VesselsMask[py, px]) == 1: break
                        PointerMask[i,py,px]=1
                        ImgList[i]=Img
                        #ROI[i]=VesselsMask
                        # --------------------------------------
                # for f in range(1):#NumPoints):
                #     ImgList[f, :, :, 1] *= VesselsMask.astype(np.uint8)
                #     misc.imshow(Imgs[f])
                #     misc.imshow((ROI[f] + ROI[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
    #=====================================Run Net predict instance region their IOU score and wether they correspond to vessels or other objects============================================================================================================================
            ####    VesNet.train()#*******************************************
                with torch.autograd.no_grad():
                    Prob, Lb, PredIOU, PredIsVessel = VesNet.forward(Images=ImgList, Pointer=PointerMask,ROI=ROI,TrainMode=False,UseGPU=UseGPU, FreezeBatchNorm_EvalON=FreezeBatchNorm_EvalON)
    #======================================s========================================================================================================
                Masks = Lb.data.cpu().numpy().astype(float)
                IOU = PredIOU.data.cpu().numpy().astype(float)
                IsVessel = PredIsVessel.data.cpu().numpy().astype(float)[:,1]

    ##################################Filter overlapping and low score segment############################################################################
                Accept=np.ones([NumPoints])
                for f in range(NumPoints):
                    SumMask=Masks[f].sum()
                    if IOU[f]<VesIOUthresh-cycle*0.05 or ((Masks[f]*OccupyMask).sum()/SumMask)>0.08:
                           Accept[f]=0
                           continue
                    for i in range(NumPoints):
                        if i==f: continue
                        if IOU[f] > IOU[i] or Accept[i]==0: continue
                        fr=(Masks[i]*Masks[f]).sum()/SumMask
                        if  (fr>0.05):
                                    Accept[f]=0
                                    break

    #===================================================Remove  predictions that over lap previous prediction========================================================================================================================
                for f in range(NumPoints):
                    if Accept[f]==0: continue
                    OverLap = Masks[f] * OccupyMask
                    if (OverLap.sum() > 0):
                        Masks[f][OverLap>0] = 0
                    for i in range(NumPoints):
                        if Accept[i] == 0 or i==f or  IOU[f]>IOU[i]: continue
                        OverLap=Masks[i]*Masks[f]
                        fr=(OverLap).sum()
                        if  (fr>0):  Masks[f][OverLap>0]=0
    #=============================Add selected mask to final segmentatiomn map and instance list=======================================================================================================================================
                for f in range(NumPoints):
                        if Accept[f]==0: continue
                        if (IsVessel[f] > IsVesThresh or not UseIsVessel):
                            NInst+=1
                            InsList = np.concatenate([InsList,np.expand_dims(Masks[f],0)],axis=0)
                            InstRank.append(IOU[f])
                            InstMap[Masks[f]>0]=NInst
                        OccupyMask[Masks[f]>0]=1
    #=============================================================================================================================================================================================
                #print("cycle"+str(cycle))
                # for i in range(NInst):
                #     print(InstRank[i])
                #     Img2=Img.copy()
                #     Img2[:, :, 1] *= 1 - Masks[i].astype(np.uint8)
                #     Img2[:, :, 0] *= 1 - Masks[i].astype(np.uint8)
                #
                #     misc.imshow(cv2.resize(np.concatenate([Img,Img2],axis=1),(1000,500)))


    #===============================================Update ROI mask==============================================================================================================================================
                ROIMask[OccupyMask>0]=0
                if (ROIMask.sum()/ VesselsMask.sum())<0.03: break

            return InsList,InstRank,InstMap, OccupyMask,NInst
###########################################################################################################################################################################################################################################

CatDic={}

def get_frame_OutAnnMap(Im):
    h0,w0,d=Im.shape
    r=np.max([h0,w0])
    #print(Im.shape)
    if r>840:
        fr=840/r
        Im=cv2.resize(Im,(int(w0*fr),int(h0*fr)))
    # if r<200:
    #     fr=200/r
    #     Im=cv2.resize(Im,(int(w0*fr),int(h0*fr)))
    Imgs=np.expand_dims(Im,axis=0)
# =====================================================================Semantic============================================================================================================================================================================
    #print("Applying semantic segmentation")
    with torch.autograd.no_grad():
          OutProbDict,OutLbDict=SemNet.forward(Images=Imgs,TrainMode=False,UseGPU=UseGPU, FreezeBatchNormStatistics=FreezeBatchNorm_EvalON) # Run semntic net inference and get prediction


#####################################################Instance segmentation#####################################################################################################
#####################################################Instance segmentation#####################################################################################################
 #------------------------Find Vessel instance take the vessel region find in the semantic segmentation and split it into individual vessel instances--------------------------------------------------------------------------------------------------------------------
    print("Applying Vessel instance segmentation")
    NumVess=0
    OutAnnMap = np.zeros([Imgs[0].shape[0], Imgs[0].shape[1]], dtype=np.uint8)

    VesselRegion=OutLbDict['Vessel'].data.cpu().numpy()[0].astype(float)
    if VesselRegion.mean()<0.001:
               VesInsList=[]
               InstRank=[]
               InstMapVes=[]
               OccupyMask=np.zeros([Imgs[0].shape[0], Imgs[0].shape[1]])
               NInst=0
    else:
               VesInsList,InstRank,InstMapVes, OccupyMask,NInst=FindVesselInstances(Imgs[0],VesselRegion)


#######################################
    for ff,VesIns in enumerate(VesInsList): # go over all vessel instances and find the material instances inside this vessels
       NumVess+=1
       OutAnnMap[:,:][VesIns>0]=NumVess
    
    h,w = np.shape(OutAnnMap)
    hest = np.zeros([256],dtype = np.int32)
    for row in range(h):
        for col in range(w):
            pv = OutAnnMap[row,col]
            if pv != 0:
                hest[pv] += 1
    max_vessl_pixel = np.argmax(hest)
    if max_vessl_pixel != 0:
        max_vessl_mask = OutAnnMap.copy()
        for row in range(h):
            for col in range(w):
                pv = OutAnnMap[row,col]
                if pv == max_vessl_pixel:
                    max_vessl_mask[row,col] = 255
                else:
                    max_vessl_mask[row,col] = 0
        return True,max_vessl_mask
    else:
        max_vessl_mask = OutAnnMap.copy()
        return False,max_vessl_mask
    #cv2.imwrite(OutVizIns+"/"+"test.png", VizInstImg)
    #print("-----------------------------------------------Finish image-------------------------------------------------")
