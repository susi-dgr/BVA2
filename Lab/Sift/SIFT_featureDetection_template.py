import cv2
import numpy as np
import matplotlib.pyplot as plt

inPath = "C:\\Users\\p21702\\Desktop\\BVA2\\ue_2021\\ex_Localization\\openCV_SIFT\\insightImg.jpg"


def unpackSIFTOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @created by Silencer at 2018.01.23 11:12:30 CST
    @brief Unpack Sift Keypoint by Silencer
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave & 0xFF
    layer = (_octave >> 8) & 0xFF
    if octave >= 128:
        octave |= -128
    if octave >= 0:
        scale = float(1 / (1 << octave))
    else:
        scale = float(1 << -octave)
    return (octave, layer, scale)