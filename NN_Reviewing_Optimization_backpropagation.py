




# In[1]:

import numpy as np
from simple_MLP_fixed_size import *

# Parameter definitions, to be replaced with user inputs
alpha = 0.9            # parameter governing steepness of sigmoid transfer function
summedInput = 1
maxNumIterations = 1000000    # You can adjust this parameter; 10,000 typically gives good results when training.
eta = 0.9              # training rate

# Establish some parameters just before we start training
epsilon = 0.2 # epsilon determines when we are done training; 
              # for each presentation of a training data set, we get a new value for the summed squared error (SSE)
              # and we will terminate the run when any ONE of these SSEs is < epsilon;
              # Note that this is a very crude stopping criterion, we can refine it in later versions. 
              # must b
iteration = 0 # This counts the number of iterations that we've made through the training cycle.
SSE_InitialTotal = 0.0 # We initially set the SSE to be zero before any training pass; we accumulate inputs
                       # into the SSE once we have a set of weights, and push the input data through the network,
                       # generating a set of outputs. 
                       # We compare the generated outputs (actuals) with the desired, obtain errors, square them, 
                       # and sum across all the outputs. This gives our SSE for that particular data set, for that 
                       # particular training pass. 
                       # If the SSE is low enough (< epsilon), we stop training. 

# Set default values for debug parameters
# We are setting the debug parameters to be "Off" (debugxxxOff = True)
debugCallInitializeOff = True
debugInitializeOff = True    

# Right now, for simplicity, we're going to hard-code the numbers of layers that we have in our 
# multilayer Perceptron (MLP) neural network. 
# We will have an input layer (I), an output layer (O), and a single hidden layer (H). 

# Obtain the array sizes (arraySizeList) by calling the appropriate function
[inputArrayLength, 
 hiddenArrayLength, 
 outputArrayLength] = arraySizeList = obtainNeuralNetworkSizeSpecs ()

# In addition to connection weights, we also use bias weights:
#   - one bias term for each of the hidden nodes
#   - one bias term for each of the output nodes
#   This means that a 1-D array of bias weights for the hidden nodes will have the same dimension as 
#      the hidden array length, and
#   also a 1-D array of bias weights for the output nodes will have the same dimension as
#      the output array length. 
biasHiddenWeightArraySize = hiddenArrayLength
biasOutputWeightArraySize = outputArrayLength   

####################################################################################################
# Initialize the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
####################################################################################################                

#
# The wWeightArray is for Input-to-Hidden
# The vWeightArray is for Hidden-to-Output
# The wWeightArray is for Input-to-Hidden
# The vWeightArray is for Hidden-to-Output

# We have a single function to initialize weights in a connection weight matrix (2-D array).
#   This function needs to know the sizes (lengths) of the lower and the upper sets of nodes.
#   These form the [row, column] size specifications for the returned weight matrices (2-D arrays). 
#   We will store these sizes in each of two different lists. 

# Specify the sizes for the input-to-hidden connection weight matrix (2-D array)
wWeightArraySizeList = (inputArrayLength, hiddenArrayLength)

# Specify the sizes for the hidden-to-output connection weight matrix (2-D array)    
vWeightArraySizeList = (hiddenArrayLength, outputArrayLength)

####################################################################################################
# Next step - Get an initial value for the Total Summed Squared Error (Total_SSE)
#   The function will return an array of SSE values, SSE_Array[0] ... SSE_Array[3] are the initial SSEs
#   for training sets 0..3; SSE_Array[4] is the sum of the SSEs. 
####################################################################################################                

# Before starting the training run, compute the initial SSE Total 
#   (sum across SSEs for each training data set) 
debugSSE_InitialComputationOff = True

# Initialize an array of SSE values
# The first four SSE values are the SSE's for specific input/output pairs; 
#   the fifth is the sum of all the SSE's.

SSE_Array = [0] * inputArrayLength + [0] * outputArrayLength + [0] # [0,0,0,0,0]


# Compute the random weights and bias values ########## may need to move this to outside the while loop

# Obtain the actual (randomly-initialized) values for the input-to-hidden connection weight matrix.
#wWeightArray = initializeWeightArray (wWeightArraySizeList, debugInitializeOff)
wWeightArray = np.array([(0.15,  0.23), (0.20, 0.30)])
# Obtain the actual (randomly-initialized) values for the hidden-to-output connection weight matrix.
#vWeightArray = initializeWeightArray (vWeightArraySizeList, debugInitializeOff)
vWeightArray = np.array([(0.40,  0.50), (0.45, 0.55)])
# Now, we similarly need to obtain randomly-initialized values for the two sets of bias weights.
#    Each set of bias weights is stored in its respective 1-D array
#    Recall that we have previously initialized the SIZE for each of these 1-D arrays.
#biasHiddenWeightArray = initializeBiasWeightArray (biasHiddenWeightArraySize)
#biasOutputWeightArray = initializeBiasWeightArray (biasOutputWeightArraySize)
biasHiddenWeightArray = np.array((0.20,  0.86))
biasOutputWeightArray = np.array((0.28,  0.41))

print ' '
print 'About to enter the while loop for ', maxNumIterations, ' iterations'
print ' '

while iteration < maxNumIterations: 
           
####################################################################################################
# Next step - Obtain a single set of input values for the X-OR problem; two integers - can be 0 or 1
####################################################################################################                

# Randomly select one of four training sets; the inputs will be randomly assigned to 0 or 1
    trainingDataList = obtainRandomXORTrainingValues () 
    input0 = trainingDataList[0]
    input1 = trainingDataList[1] 
    desiredOutput0 = trainingDataList[2]
    desiredOutput1 = trainingDataList[3]
    setNumber = trainingDataList[4] # obtain the number (0 ... 3) of the training data set.       
    print ' '
    print 'Iteration number ', iteration
    print ' '
    print 'Randomly selected training data set number ', trainingDataList[4] 
    #print 'The inputs and desired outputs for the X-OR problem from this data set are:'
    #print '          Input0 = ', input0,         '            Input1 = ', input1   
    #print ' Desired Output0 = ', desiredOutput0, '   Desired Output1 = ', desiredOutput1    
    #print ' '
    
    
####################################################################################################
# Compute a single feed-forward pass
####################################################################################################                

# Initialize the error list
    errorList = (0,0)

# Initialize the actualOutput list
#    Remember, we've hard-coded the number of hidden nodes and output nodes in this version;
#      numHiddenNodes = 2; numOutputNodes = 2. 
#    We want to see the ACTUAL VALUES ("activations") of both the hidden AND the output nodes;
#      this is just to satisfy our own interest. 
#    In just a few lines down, we will use the function "ComputeSingleFeedforwardPass" to get us all 
#      of those activations. 
    actualAllNodesOutputList = (0,0,0,0)     

# Create the inputData list      
    inputDataList = (input0, input1)         

# Compute a single feed-forward pass and obtain the Actual Outputs
    debugComputeSingleFeedforwardPassOff = True
    actualAllNodesOutputList = ComputeSingleFeedforwardPass (alpha, inputDataList, 
    wWeightArray, vWeightArray, biasHiddenWeightArray, biasOutputWeightArray,debugComputeSingleFeedforwardPassOff)

# Assign the hidden and output values to specific different variables
    actualHiddenOutput0 = actualAllNodesOutputList [0] 
    actualHiddenOutput1 = actualAllNodesOutputList [1] 
    actualOutput0 = actualAllNodesOutputList [2]
    actualOutput1 = actualAllNodesOutputList [3] 

# Determine the error between actual and desired outputs
    error0 = desiredOutput0 - actualOutput0
    error1 = desiredOutput1 - actualOutput1
    errorList = (error0, error1)

# Compute the Summed Squared Error, or SSE
    SSEInitial = error0**2 + error1**2

    
# Perform first part of the backpropagation of weight changes    
    newVWeightArray = BackpropagateOutputToHidden (alpha, eta, errorList, actualAllNodesOutputList, 
    vWeightArray)

    newBiasOutputWeightArray = BackpropagateBiasOutputWeights (alpha, eta, errorList, actualAllNodesOutputList, 
    biasOutputWeightArray)
    newBiasOutputWeight0 = newBiasOutputWeightArray[0]
    newBiasOutputWeight1 = newBiasOutputWeightArray[1]

    newWWeightArray = BackpropagateHiddenToInput (alpha, eta, errorList, actualAllNodesOutputList, 
    inputDataList, vWeightArray, wWeightArray, biasHiddenWeightArray, biasOutputWeightArray)

    newBiasHiddenWeightArray = BackpropagateBiasHiddenWeights (alpha, eta, errorList, actualAllNodesOutputList, 
    vWeightArray, biasHiddenWeightArray, biasOutputWeightArray)
    newBiasHiddenWeight0 = newBiasHiddenWeightArray[0]
    newBiasHiddenWeight1 = newBiasHiddenWeightArray[1]        

    newBiasWeightArray = [[newBiasOutputWeight0, newBiasOutputWeight1], [newBiasHiddenWeight0, newBiasHiddenWeight1]] 
           
# Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
    vWeightArray = newVWeightArray[:]

# Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
    wWeightArray = newWWeightArray[:]

# Run the computeSingleFeedforwardPass again, to compare the results after just adjusting the hidden-to-output weights
    newAllNodesOutputList = ComputeSingleFeedforwardPass (alpha, inputDataList, wWeightArray, vWeightArray,
    biasHiddenWeightArray, biasOutputWeightArray, debugComputeSingleFeedforwardPassOff)         
    newOutput0 = newAllNodesOutputList [2]
    newOutput1 = newAllNodesOutputList [3] 

# Determine the new error between actual and desired outputs
    newError0 = desiredOutput0 - newOutput0
    newError1 = desiredOutput1 - newOutput1
    newErrorList = (newError0, newError1)

# Compute the new Summed Squared Error, or SSE
    SSE0 = newError0**2
    SSE1 = newError1**2
    newSSE = SSE0 + SSE1

# Assign the SSE to the SSE for the appropriate training set
    SSE_Array[setNumber] = newSSE

# Obtain the previous SSE Total from the SSE array
    previousSSE_Total = SSE_Array[4]
    print ' ' 
    print '  The previous SSE Total was %.4f' % previousSSE_Total

# Compute the new sum of SSEs (across all the different training sets)
#   ... this will be different because we've changed one of the SSE's
    newSSE_Total = SSE_Array[0] + SSE_Array[1] +SSE_Array[2] + SSE_Array[3]
    #print '  The new SSE Total was %.4f' % newSSE_Total
    #print '    For node 0: Desired Output = ',desiredOutput0,  ' New Output = %.4f' % newOutput0 
    #print '    For node 1: Desired Output = ',desiredOutput1,  ' New Output = %.4f' % newOutput1  
    #print '    Error(0) = %.4f,   Error(1) = %.4f' %(newError0, newError1)
    #print '    SSE0(0) =   %.4f,   SSE(1) =   %.4f' % (SSE0, SSE1)                
# Assign the new SSE to the final place in the SSE array
    SSE_Array[4] = newSSE_Total
    deltaSSE = previousSSE_Total - newSSE_Total
    print '  Delta in the SSEs is %.4f' % deltaSSE 
    if deltaSSE > 0:
        print 'SSE improvement'
        
        bestSSE_Total = newSSE_Total
        best_wWeightArray = wWeightArray[:]
        best_vWeightArray = vWeightArray[:]
        best_biasHiddenWeightArray = biasHiddenWeightArray[:]
        best_biasOutputWeightArray = biasOutputWeightArray[:]
        best_desiredOutput0 = desiredOutput0
        best_newOutput0 = newOutput0
        best_desiredOutput1 = desiredOutput1
        best_newOutput1 = newOutput1
        
    else: print 'NO improvement'     


# Assign the new errors to the error list             
    errorList = newErrorList[:]


    print ' '
    print 'Iteration number ', iteration
    iteration = iteration + 1

    if newSSE_Total < epsilon:
        print 'reached epsilon target'

        break
        
print '------------------------------------------------------------------------------'
print ' '
print 'Out of while loop after ', maxNumIterations, ' iterations'   
print ' '
print 'The best sum of these squared errors (SSE) for training set was %.4f' %bestSSE_Total
print 'The wWeightArray needs to be'
print ' '
print best_wWeightArray
print ' '
print 'The vWeightArray needs to be'
print ' '
print best_vWeightArray
print ' '
print 'The biasHiddenWeightArray needs to be'
print ' '
print best_biasHiddenWeightArray
print ' '
print 'The biasOutputWeightArray needs to be'
print ' '
print best_biasOutputWeightArray
print ' '
print 'the desired output', best_desiredOutput0, ' the actual output ', best_newOutput0
print 'the desired output', best_desiredOutput1, ' the actual output ', best_newOutput1