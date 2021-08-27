import torch
from quantized import BinarizeLinear
from torch import nn

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 *
                                                     (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()

def exportThreshold(model):
    print("Export thresholds")

    #clear File
    f = open("../export/thresholds.txt", "w")
    f.write("")
    f.close()

    #write thresholds
    f = open("../export/thresholds.txt", "a")
    for layer in model.modules():
        if(type(layer) == type(nn.BatchNorm1d(1024))):
            layerThreshold = "["
            for node in layer.weight:
                layerThreshold += (str(node.item())+",")
            layerThreshold = layerThreshold[:-1]
            layerThreshold += "]"
            f.write(layerThreshold)
    f.close()


def export(model):
    print("Starting export...")

    #clear File
    f = open("../export/weights.txt", "w")
    f.write("")
    f.close()

    #write weigths to file
    f = open("../export/weights.txt", "a")
    layerCount = 1
    torch.set_printoptions(profile="full")
    for layer in model.modules():
        if(type(layer) == type(BinarizeLinear(2048, 2048))):
            layerWeights = ""
            print("----Exporting layer " + str(layerCount) + "----")
            layerCount += 1
            finishedNodes = 0
            totalLayerNodes = layer.weight.size()[0]
            printProgressBar(0, totalLayerNodes,
                             prefix='Progress:', suffix='Complete', length=50)
            for node in layer.weight:
                for edge in enumerate(node):
                    layerWeights += str(max(int(edge[1].item()), 0))
                finishedNodes += 1
                printProgressBar(finishedNodes, totalLayerNodes,
                                 prefix='Progress:', suffix='Complete', length=50)
            f.write(layerWeights[:-1])
    f.close()
