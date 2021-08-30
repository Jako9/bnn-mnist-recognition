##TRAINING
BATCH_SIZE = 200
LEARNING_RATE = 1.5
STEP_SIZE = 25
EPOCHS = 1
#-------Important for when using ProbabilityTransform-------
#for a fair comparison between ThresholdTransform and ProbabilityTransform REPETITIONS * EPOCHS should be constant
REPETITIONS = 1

USE_PROBABILITY_TRANSFORM = True

#To compare different methods of image-binarization
#-------IF TRUE BNN IS NOT PROPERLY TRAINING-------
SHOW_PROCESSED_NUMBERS = False
SELECTED_NUMBER_INDEX = 21

#How often should the training be done with the same parameter
MEASUREMENT_RUNS = 1
