import sfm.sfmpipeline
import sfm.methods
import os
sfm.sfmpipeline.SFMPipeline('.', 40).evaluate(sfm.methods.keypoints_matches)