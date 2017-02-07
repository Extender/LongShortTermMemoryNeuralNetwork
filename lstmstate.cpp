#include "lstmstate.h"

LSTMState::LSTMState(uint32_t _inputCount, uint32_t _outputCount, LSTMState *copyFrom)
{
    inputCount=_inputCount;
    outputCount=_outputCount;

    uint32_t inputAndOutputCount=inputCount+outputCount;

    uint32_t outputBasedDoubleArraySize=outputCount*sizeof(double);
    uint32_t outputBasedDoublePointerArraySize=outputCount*sizeof(double*);
    uint32_t inputAndOutputBasedDoubleArraySize=inputAndOutputCount*sizeof(double);
    forgetGateWeights=(double**)malloc(outputBasedDoublePointerArraySize);
    inputGateWeights=(double**)malloc(outputBasedDoublePointerArraySize);
    outputGateWeights=(double**)malloc(outputBasedDoublePointerArraySize);
    candidateGateWeights=(double**)malloc(outputBasedDoublePointerArraySize);
    input=(double*)malloc(inputCount*sizeof(double));
    output=(double*)malloc(outputBasedDoubleArraySize);
    desiredOutput=(double*)malloc(outputBasedDoubleArraySize);
    cellStates=(double*)malloc(outputBasedDoubleArraySize);
    forgetGateValues=(double*)malloc(outputBasedDoubleArraySize);
    inputGateValues=(double*)malloc(outputBasedDoubleArraySize);
    outputGateValues=(double*)malloc(outputBasedDoubleArraySize);
    candidateGateValues=(double*)malloc(outputBasedDoubleArraySize);
    forgetGateValueSumBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    inputGateValueSumBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    outputGateValueSumBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    candidateGateValueSumBiasWeights=(double*)malloc(outputBasedDoubleArraySize);
    // The derivatives do not need to be initialized.
    bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastCellStates=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_s
    bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastOutputs=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_h
    bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToInputs=(double*)malloc(outputBasedDoubleArraySize); // bottom_diff_x
    if(copyFrom==0)
    {
        srand(time(NULL));
        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            forgetGateValueSumBiasWeights[cell]=0.0;
            inputGateValueSumBiasWeights[cell]=0.0;
            outputGateValueSumBiasWeights[cell]=0.0;
            candidateGateValueSumBiasWeights[cell]=0.0;

            forgetGateWeights[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            inputGateWeights[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            outputGateWeights[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            candidateGateWeights[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            for(uint32_t i=0;i<inputAndOutputCount;i++)
            {
                forgetGateWeights[cell][i]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                inputGateWeights[cell][i]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                outputGateWeights[cell][i]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
                candidateGateWeights[cell][i]=-0.1+0.2*((double)rand()/(double)RAND_MAX);
            }
        }
    }
    else
    {
        // Copy the one-dimensional bias weight arrays using memcpy:
        memcpy(forgetGateValueSumBiasWeights,copyFrom->forgetGateValueSumBiasWeights,outputBasedDoubleArraySize);
        memcpy(inputGateValueSumBiasWeights,copyFrom->inputGateValueSumBiasWeights,outputBasedDoubleArraySize);
        memcpy(outputGateValueSumBiasWeights,copyFrom->outputGateValueSumBiasWeights,outputBasedDoubleArraySize);
        memcpy(candidateGateValueSumBiasWeights,copyFrom->candidateGateValueSumBiasWeights,outputBasedDoubleArraySize);
        // Create deep copies of the two-dimensional weight arrays:
        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            forgetGateWeights[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            inputGateWeights[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            outputGateWeights[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            candidateGateWeights[cell]=(double*)malloc(inputAndOutputBasedDoubleArraySize);
            memcpy(forgetGateWeights[cell],copyFrom->forgetGateWeights[cell],inputAndOutputBasedDoubleArraySize);
            memcpy(inputGateWeights[cell],copyFrom->inputGateWeights[cell],inputAndOutputBasedDoubleArraySize);
            memcpy(outputGateWeights[cell],copyFrom->outputGateWeights[cell],inputAndOutputBasedDoubleArraySize);
            memcpy(candidateGateWeights[cell],copyFrom->candidateGateWeights[cell],inputAndOutputBasedDoubleArraySize);
        }
    }
}

void LSTMState::freeMemory()
{
    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        free(forgetGateWeights[cell]);
        free(inputGateWeights[cell]);
        free(outputGateWeights[cell]);
        free(candidateGateWeights[cell]);
    }
    free(input);
    free(output);
    free(desiredOutput);
    free(cellStates);
    free(forgetGateWeights);
    free(inputGateWeights);
    free(outputGateWeights);
    free(candidateGateWeights);
    free(forgetGateValueSumBiasWeights);
    free(inputGateValueSumBiasWeights);
    free(outputGateValueSumBiasWeights);
    free(candidateGateValueSumBiasWeights);
    free(forgetGateValues);
    free(inputGateValues);
    free(outputGateValues);
    free(candidateGateValues);
    free(bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastCellStates);
    free(bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToInputs);
    free(bottomDerivativesOfLossesFromThisStateUpwardsWithRespectToLastOutputs);
}

LSTMState::~LSTMState()
{
    freeMemory();
}
