#ifndef LSTMSTATE_H
#define LSTMSTATE_H

#include <stdlib.h>
#include <stdint.h>
#include <memory.h>
#include <math.h>
#include <time.h>

class LSTMState
{
public:
    double **forgetGateWeights;
    double **inputGateWeights;
    double **outputGateWeights;
    double **candidateGateWeights;
    double *forgetGateBiasWeights;
    double *inputGateBiasWeights;
    double *outputGateBiasWeights;
    double *candidateGateBiasWeights;

    double *forgetGateValues;
    double *inputGateValues;
    double *outputGateValues;
    double *candidateGateValues;
    double *input;
    double *output;
    double *desiredOutput;
    double *cellStates;

    uint32_t inputCount;
    uint32_t outputCount;

    double *bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToCellStates; // bottom_diff_s
    double *bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToOutputs; // bottom_diff_h
    double *bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToInputs; // bottom_diff_x

    LSTMState(uint32_t _inputCount,uint32_t _outputCount,LSTMState *copyFrom=0);
    void freeMemory();
    ~LSTMState();
};

#endif // LSTMSTATE_H
