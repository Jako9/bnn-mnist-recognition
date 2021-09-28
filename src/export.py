import torch
import math 
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
    """Calculate and export the activation thresholds

    Args:
        model (NeuralNetwork): The model containing the BatchNorm layers to calculate thresholds
    """
    print("Export thresholds")
    #clear File
    f = open("../export/thresholds.txt", "w")
    f.write("")
    f.close()

    #prepare file for appandance of thresholds
    f = open("../export/thresholds.txt", "a")
    for layer in model.modules():
        if(type(layer) == type(nn.BatchNorm1d(1024))):
            layerThreshold = "["
            
            #Calculate the threshold and store it in a helper var for the current layer
            for ind,node in enumerate(layer.weight):
                #mean_x - (standard-derr_x/gamma_x)*beta_x           rounded down by int conversion
                layerThreshold += str(int(layer.running_mean[ind].item() - ( math.sqrt(layer.running_var[ind].item()) / layer.weight[ind].item() )*layer.bias[ind].item())) + ","
                
            #remove tailing ","
            layerThreshold = layerThreshold[:-1]            
            layerThreshold += "]"
            f.write(layerThreshold)
    f.close()


def export(model):
    """Export the weights into a file

    Args:
        model (NeuralNetowrk): The NeuralNetwork from which the weights should be exportet
    """
    print("Starting export...")
    #clear File
    f = open("../export/weights.txt", "w")
    f.write("")
    f.close()

    # prepare writing weights to the file
    f = open("../export/weights.txt", "a")
    layerCount = 1
    torch.set_printoptions(profile="full")
    # loop though th fully-connected layers
    for layer in model.modules():
        if(type(layer) == type(BinarizeLinear(2048, 2048))):
            layerWeights = ""
            print("----Exporting layer " + str(layerCount) + "----")
            layerCount += 1
            finishedNodes = 0
            totalLayerNodes = layer.weight.size()[0]
            printProgressBar(0, totalLayerNodes,
                             prefix='Progress:', suffix='Complete', length=50)
            # loop though each node in a layer                 
            for node in layer.weight:
                #write each weight for each neuron in a var for later export to reduce IO-costs
                for edge in enumerate(node):
                    layerWeights += str(max(int(edge[1].item()), 0))
                finishedNodes += 1
                printProgressBar(finishedNodes, totalLayerNodes,
                                 prefix='Progress:', suffix='Complete', length=50)
            # after a layer is finished, write its weights to the file
            f.write(layerWeights[:-1])
    f.close()
