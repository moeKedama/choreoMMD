import os
import numpy as np
import json
import wave
import librosa
import fbx
from fbx import *
from samples import FbxCommon


if __name__ == '__main__':

    manager, scene = FbxCommon.InitializeSdkObjects()
    filename = "0071@000.fbx"
    result = FbxCommon.LoadScene(manager, scene, filename)
    dir(scene)
    # root_node = scene.GetRootNode()
    pNode = scene.GetRootNode()
    dir(pNode)
    camera = pNode.GetCamera()
    dir(camera)

    # 就一个class
    for i in range(scene.GetSrcObjectCount(FbxCriteria.ObjectType(FbxAnimStack.ClassId))):
        # Take 001 遍历 take
        lAnimStack = scene.GetSrcObject(FbxCriteria.ObjectType(FbxAnimStack.ClassId), i)
        print("Take: %s" % lAnimStack.GetName())

        lAnimLayer = lAnimStack.GetSrcObject(FbxCriteria.ObjectType(FbxAnimLayer.ClassId), i)

        lAnimCurve = pNode.LclRotation.GetCurve(lAnimLayer, "X")
        # if (lAnimCurve is not None):
        #     lKeyCount = lAnimCurve.KeyGetCount()
        #     for lCount in range(lKeyCount):
        #         lKeyValue = lAnimCurve.KeyGetValue(lCount)
        #         lKeyTime = lAnimCurve.KeyGetTime(lCount).GetSecondDouble()
        #
        # lAnimCurve = pNode.LclRotation.GetCurve(lAnimLayer, "Y")
        # if (lAnimCurve is not None):
        #     lKeyCount = lAnimCurve.KeyGetCount()
        #     for lCount in range(lKeyCount):
        #         lKeyValue = lAnimCurve.KeyGetValue(lCount)
        #         lKeyTime = lAnimCurve.KeyGetTime(lCount).GetSecondDouble()
        #
        # lAnimCurve = pNode.LclRotation.GetCurve(lAnimLayer, "Z")
        # if (lAnimCurve is not None):
        #     lKeyCount = lAnimCurve.KeyGetCount()
        #     for lCount in range(lKeyCount):
        #         lKeyValue = lAnimCurve.KeyGetValue(lCount)
        #         lKeyTime = lAnimCurve.KeyGetTime(lCount).GetSecondDouble()
