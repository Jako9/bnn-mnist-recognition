##TRAINING
BATCH_SIZE = 200
LEARNING_RATE = 1.5
STEP_SIZE = 25
EPOCHS = 3
#-------Important for when using ProbabilityTransform-------
#for a fair comparison between ThresholdTransform and ProbabilityTransform REPETITIONS * EPOCHS should be constant
REPETITIONS = 50

USE_PROBABILITY_TRANSFORM = True

#Threshold for a 1 when not using ProbabilityTransform
THRESHOLD = 150

#To compare different methods of image-binarization
#-------IF TRUE BNN IS NOT PROPERLY TRAINING-------
SHOW_PROCESSED_NUMBERS = False
SELECTED_NUMBER_INDEX = 2 #index 2 is a good looking 4 ;)

#How often should the training be done with the same parameter
MEASUREMENT_RUNS = 3
