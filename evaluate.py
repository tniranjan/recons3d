import sfm.sfmpipeline
import sfm.methods
import os

kp_methods = ["ALIKED", "SuperPoint", "DISK", "SIFT", "LOFTR","DoGHardNet"]#["D2Net, LOFTR"]
for kp_method in kp_methods:
    sfm.sfmpipeline.SFMPipeline('.', 150).evaluate(kp_method)