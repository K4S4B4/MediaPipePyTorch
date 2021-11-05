import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from blazebase import BlazeDetector, BlazeBlock, FinalBlazeBlock


class BlazeFace(BlazeDetector):
    """The BlazeFace face detection model from MediaPipe.
    
    The version from MediaPipe is simpler than the one in the paper; 
    it does not use the "double" BlazeBlocks.

    Because we won't be training this model, it doesn't need to have
    batchnorm layers. These have already been "folded" into the conv 
    weights by TFLite.

    The conversion to PyTorch is fairly straightforward, but there are 
    some small differences between TFLite and PyTorch in how they handle
    padding on conv layers with stride 2.

    This version works on batches, while the MediaPipe version can only
    handle a single image at a time.

    Based on code from https://github.com/tkat0/PyTorch_BlazeFace/ and
    https://github.com/hollance/BlazeFace-PyTorch and
    https://github.com/google/mediapipe/

    """
    def __init__(self, back_model=False):
        super(BlazeFace, self).__init__()

        # These are the settings from the MediaPipe example graph
        # mediapipe/graphs/face_detection/face_detection_mobile_gpu.pbtxt
        self.num_classes = 1
        self.num_anchors = 896
        self.num_coords = 16
        self.score_clipping_thresh = 100.0
        self.back_model = back_model
        if back_model:
            self.x_scale = 256.0
            self.y_scale = 256.0
            self.h_scale = 256.0
            self.w_scale = 256.0
            self.min_score_thresh = 0.65
        else:
            self.x_scale = 128.0
            self.y_scale = 128.0
            self.h_scale = 128.0
            self.w_scale = 128.0
            self.min_score_thresh = 0.75
        self.min_suppression_threshold = 0.3
        self.num_keypoints = 6

        # These settings are for converting detections to ROIs which can then
        # be extracted and feed into the landmark network
        # use mediapipe/calculators/util/detections_to_rects_calculator.cc
        self.detection2roi_method = 'box'
        # mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
        self.kp1 = 1
        self.kp2 = 0
        self.theta0 = 0.
        self.dscale = 1.5
        self.dy = 0.

        self._define_layers()

    def _define_layers(self):
        if self.back_model:
            self.backbone = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),

                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24, stride=2),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 24),
                BlazeBlock(24, 48, stride=2),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 48),
                BlazeBlock(48, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )
            self.final = FinalBlazeBlock(96)
            self.classifier_8 = nn.Conv2d(96, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

            self.regressor_8 = nn.Conv2d(96, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        else:
            self.backbone1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2, padding=0, bias=True),
                nn.ReLU(inplace=True),

                BlazeBlock(24, 24),
                BlazeBlock(24, 28),
                BlazeBlock(28, 32, stride=2),
                BlazeBlock(32, 36),
                BlazeBlock(36, 42),
                BlazeBlock(42, 48, stride=2),
                BlazeBlock(48, 56),
                BlazeBlock(56, 64),
                BlazeBlock(64, 72),
                BlazeBlock(72, 80),
                BlazeBlock(80, 88),
            )
        
            self.backbone2 = nn.Sequential(
                BlazeBlock(88, 96, stride=2),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
                BlazeBlock(96, 96),
            )

            self.classifier_8 = nn.Conv2d(88, 2, 1, bias=True)
            self.classifier_16 = nn.Conv2d(96, 6, 1, bias=True)

            self.regressor_8 = nn.Conv2d(88, 32, 1, bias=True)
            self.regressor_16 = nn.Conv2d(96, 96, 1, bias=True)
        
    def forward(self, x):
        ##########################################
        x = x[:,:,:,[2, 1, 0]] # BRG to RGB
        x = x.permute(0,3,1,2).float() / 255.
        ##########################################

        # TFLite uses slightly different padding on the first conv layer
        # than PyTorch, so do it manually.
        x = F.pad(x, (1, 2, 1, 2), "constant", 0)
        
        b = x.shape[0]      # batch size, needed for reshaping later

        if self.back_model:
            x = self.backbone(x)           # (b, 16, 16, 96)
            h = self.final(x)              # (b, 8, 8, 96)
        else:
            x = self.backbone1(x)           # (b, 88, 16, 16)
            h = self.backbone2(x)           # (b, 96, 8, 8)
        
        # Note: Because PyTorch is NCHW but TFLite is NHWC, we need to
        # permute the output from the conv layers before reshaping it.
        
        c1 = self.classifier_8(x)       # (b, 2, 16, 16)
        c1 = c1.permute(0, 2, 3, 1)     # (b, 16, 16, 2)
        c1 = c1.reshape(b, -1, 1)       # (b, 512, 1)

        c2 = self.classifier_16(h)      # (b, 6, 8, 8)
        c2 = c2.permute(0, 2, 3, 1)     # (b, 8, 8, 6)
        c2 = c2.reshape(b, -1, 1)       # (b, 384, 1)

        raw_score_tensor = torch.cat((c1, c2), dim=1)  # (b, 896, 1)

        r1 = self.regressor_8(x)        # (b, 32, 16, 16)
        r1 = r1.permute(0, 2, 3, 1)     # (b, 16, 16, 32)
        r1 = r1.reshape(b, -1, 16)      # (b, 512, 16)
        #r1 = r1.reshape(b, -1, 8, 2)      # (b, 512, 8, 2)

        r2 = self.regressor_16(h)       # (b, 96, 8, 8)
        r2 = r2.permute(0, 2, 3, 1)     # (b, 8, 8, 96)
        r2 = r2.reshape(b, -1, 16)      # (b, 384, 16)
        #r2 = r2.reshape(b, -1, 8, 2)      # (b, 384, 8, 2)

        raw_box_tensor = torch.cat((r1, r2), dim=1)  # (b, 896, 16)
        #raw_box_tensor = torch.cat((r1, r2), dim=1)  # (b, 896, 8, 2)
        #return [r, c]

        #index = torch.argmax(raw_score_tensor, 1).squeeze()
        max = torch.max(raw_score_tensor, 1)
        index = max[1]

        #raw_box_tensor = raw_box_tensor[torch.arange(b), index]
        #raw_score_tensor = raw_score_tensor[torch.arange(b), index]

        # batch=1を前提にする
        raw_box_tensor = raw_box_tensor[0][index]
        raw_score_tensor = raw_score_tensor[0][index]

        #self.anchors = self.anchors.reshape(896, 2, 2)
        #raw_box_tensor = raw_box_tensor / self.x_scale * self.anchors[index, 1] + self.anchors[index, 0]

        ## 6,7
        #raw_box_tensor[:,:,4+2*self.kp1  ] = raw_box_tensor[:,:,4+2*self.kp1  ] / self.x_scale * self.anchors[index, 2] + self.anchors[index, 0]
        #raw_box_tensor[:,:,4+2*self.kp1+1] = raw_box_tensor[:,:,4+2*self.kp1+1] / self.y_scale * self.anchors[index, 3] + self.anchors[index, 1]

        ## 4,5
        #raw_box_tensor[:,:,4+2*self.kp2  ] = raw_box_tensor[:,:,4+2*self.kp2  ] / self.x_scale * self.anchors[index, 2] + self.anchors[index, 0]
        #raw_box_tensor[:,:,4+2*self.kp2+1] = raw_box_tensor[:,:,4+2*self.kp2+1] / self.y_scale * self.anchors[index, 3] + self.anchors[index, 1]

        for k in range(self.num_keypoints):
            offset = 4 + k*2
            raw_box_tensor[:,:,offset    ] = raw_box_tensor[:,:,offset    ] / self.x_scale * self.anchors[index, 2] + self.anchors[index, 0]
            raw_box_tensor[:,:,offset + 1] = raw_box_tensor[:,:,offset + 1] / self.y_scale * self.anchors[index, 3] + self.anchors[index, 1]


        return raw_box_tensor, raw_score_tensor

