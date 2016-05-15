#include "lstm.h"

double LSTM::sig(double input)
{
    // Derivative: sig(input)*(1.0-sig(input))
    return 1.0/(1.0+pow(M_E,-input));
}

double LSTM::tanh(double input)
{
    // Derivative: 1.0-pow(tanh(input),2.0)
    return (1.0-pow(M_E,-2.0*input))/(1.0+pow(M_E,-2.0*input));
}

double LSTM::euclideanLossFunct(double desiredOutput, double output)
{
    return pow(desiredOutput*output,2);
}

double *LSTM::cloneDoubleArray(double *array, uint32_t size)
{
    size_t arraySize=size*sizeof(double);
    double *out=(double*)malloc(arraySize);
    memcpy(out,array,arraySize);
    return out;
}

double *LSTM::mergeDoubleArrays(double *array1, uint32_t size1, double *array2, uint32_t size2)
{
    size_t array1Size=size1*sizeof(double);
    size_t array2Size=size2*sizeof(double);
    size_t arraySize=size1+size2;
    double *out=(double*)malloc(arraySize);
    memcpy(out,array1,array1Size);
    memcpy(out+array1Size,array2,array2Size);
    return out;
}

double *LSTM::multiplyDoubleArrayByDoubleArray(double *array1, uint32_t size, double *array2)
{
    double *out=cloneDoubleArray(array1,size);
    for(uint32_t i=0;i<size;i++)
        out[i]*=array2[i];
    return out;
}

double *LSTM::multiplyDoubleArray(double *array, uint32_t size, double factor)
{
    double *out=cloneDoubleArray(array,size);
    for(uint32_t i=0;i<size;i++)
        out[i]*=factor;
    return out;
}

double *LSTM::addToDoubleArray(double *array, uint32_t size, double summand)
{
    double *out=cloneDoubleArray(array,size);
    for(uint32_t i=0;i<size;i++)
        out[i]+=summand;
    return out;
}

double LSTM::sumDoubleArray(double *array, uint32_t size)
{
    double out=0.0;
    for(uint32_t i=0;i<size;i++)
        out+=array[i];
    return out;
}

void LSTM::directlyMultiplyDoubleArrayByDoubleArray(double *array1, uint32_t size, double *array2)
{
    for(uint32_t i=0;i<size;i++)
        array1[i]*=array2[i];
}

void LSTM::directlyMultiplyDoubleArray(double *array, uint32_t size, double factor)
{
    for(uint32_t i=0;i<size;i++)
        array[i]*=factor;
}

void LSTM::directlyAddToDoubleArray(double *array, uint32_t size, double summand)
{
    for(uint32_t i=0;i<size;i++)
        array[i]+=summand;
}

void LSTM::fillDoubleArray(double *array, uint32_t size, double value)
{
    for(uint32_t i=0;i<size;i++)
        array[i]=value;
}

void LSTM::fillDoubleArrayWithRandomValues(double *array, uint32_t size, double from, double to)
{
    srand(time(NULL));
    for(uint32_t i=0;i<size;i++)
        array[i]=from+((double)rand()/(double)RAND_MAX)*(to-from);
}

LSTMState *LSTM::pushState()
{
    // This works as follows: the buffer is larger (usually 2 times larger) than the required size, allowing us to avoid having to move memory
    // every time a new state is pushed. Once the buffer is filled, the needed elements in the front are moved back, overriding the old states
    // that aren't needed anymore, and creating room for new states to be pushed.

    if(stateArrayPos==0xffffffff)
        stateArrayPos=0; // Do not increment the position the first time pushState() is called.
    else
    {
        if(stateArrayPos==stateArraySize-1)
        {
            // Overwrite old states that aren't needed anymore, and set the new position:
            // Note that the current state will be a backpropagation state after the new state is pushed to the array.
            delete states[stateArrayPos-backpropagationSteps]; // Delete unneeded state
            memcpy(states,states+(stateArraySize-backpropagationSteps),backpropagationSteps*sizeof(LSTMState*));
            stateArrayPos=backpropagationSteps-1;
        }
        stateArrayPos++;
    }
    // Copy values from previous state, if such a state exists:
    LSTMState *newState=stateArrayPos>0/*Has previous state?*/?new LSTMState(inputCount,outputCount,getState(1)):new LSTMState(inputCount,outputCount);
    states[stateArrayPos]=newState;
    if(stateArrayPos>backpropagationSteps)
    {
        // Free memory occupied by the now unneeded state (each time a new state is pushed, the memory occupied by the oldest state, which is
        // not needed anymore from that point on, is freed):
        delete states[stateArrayPos-backpropagationSteps-1];
    }
    return states[stateArrayPos];
}

LSTMState *LSTM::getCurrentState()
{
    return states[stateArrayPos];
}

bool LSTM::hasState(uint32_t stepsBack)
{
    return stateArrayPos!=0xffffffff&&stepsBack<=__min(backpropagationSteps,stateArrayPos);
}

uint32_t LSTM::getAvailableStepsBack()
{
    return stateArrayPos!=0xffffffff?__min(backpropagationSteps,stateArrayPos):0;
}

LSTMState *LSTM::getState(uint32_t stepsBack)
{
    return states[stateArrayPos-stepsBack];
}

LSTM::LSTM(uint32_t _inputCount, uint32_t _outputCount, uint32_t _backpropagationSteps, double _learningRate)
{
    inputCount=_inputCount;
    outputCount=_outputCount;
    backpropagationSteps=_backpropagationSteps;
    learningRate=_learningRate;

    stateArraySize=2*backpropagationSteps+1 /*One for the current state.*/;
    stateArrayPos=0xffffffff;
    states=(LSTMState**)malloc(stateArraySize*sizeof(LSTMState*));
}

LSTM::~LSTM()
{
    for(uint32_t layer=stateArrayPos-backpropagationSteps;layer<=stateArrayPos;layer++)
        delete states[layer];
    free(states);
}

double *LSTM::process(double *input)
{
    LSTMState *l=pushState();
    memcpy(l->input,input,inputCount*sizeof(double)); // Store for backpropagation
    bool hasPreviousState=hasState(1);
    LSTMState *previousState=hasPreviousState?getState(1):0;
    double *output=(double*)malloc(outputCount*sizeof(double));
    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        // Calculate forget gate value

        double forgetGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            forgetGateValueSum+=l->forgetGateWeights[cell][i]*input[i];
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                forgetGateValueSum+=l->forgetGateWeights[cell][inputCount+i]*previousState->output[i];
        }
        l->forgetGateValues[cell]=sig(forgetGateValueSum+l->forgetGateBiasWeights[cell]);

        // Calculate input gate value

        double inputGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            inputGateValueSum+=l->inputGateWeights[cell][i]*input[i];
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                inputGateValueSum+=l->inputGateWeights[cell][inputCount+i]*previousState->output[i];
        }
        l->inputGateValues[cell]=sig(inputGateValueSum+l->inputGateBiasWeights[cell]);

        // Calculate output gate value

        double outputGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            outputGateValueSum+=l->outputGateWeights[cell][i]*input[i];
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                outputGateValueSum+=l->outputGateWeights[cell][inputCount+i]*previousState->output[i];
        }
        l->outputGateValues[cell]=sig(outputGateValueSum+l->outputGateBiasWeights[cell]);

        // Calculate candidate assessment gate value

        double candidateGateValueSum=0.0;
        for(uint32_t i=0;i<inputCount;i++)
            candidateGateValueSum+=l->candidateGateWeights[cell][i]*input[i];
        if(hasPreviousState) // Else each product simply yields 0, eliminating the need to add it to the sum.
        {
            for(uint32_t i=0;i<outputCount;i++)
                candidateGateValueSum+=l->candidateGateWeights[cell][inputCount+i]*previousState->output[i];
        }
        l->candidateGateValues[cell]=tanh(candidateGateValueSum+l->candidateGateBiasWeights[cell]);

        // Calculate new cell state

        l->cellStates[cell]=(hasPreviousState?l->forgetGateValues[cell]*previousState->cellStates[cell]/*Old cell state*/:0.0)+l->inputGateValues[cell]*l->candidateGateValues[cell]; // Store for backpropagation

        // Calculate new output value

        // colah's version has a tanh function around the cell state: output[cell]=l->outputGateValues[cell]*tanh(l->cellStates[cell]);
        // Maybe add the tanh?
        output[cell]=l->outputGateValues[cell]*l->cellStates[cell];
        l->output[cell]=output[cell]; // Store for backpropagation
    }
    return output;
}

void LSTM::learn(double **desiredOutputs)
{
    uint32_t availableStepsBack=getAvailableStepsBack();
    // Note that we sum this over all steps, so we do not need the extra time dimension.
    double **wi_diff=(double**)malloc(outputCount*sizeof(double*));
    double **wf_diff=(double**)malloc(outputCount*sizeof(double*));
    double **wo_diff=(double**)malloc(outputCount*sizeof(double*));
    double **wg_diff=(double**)malloc(outputCount*sizeof(double*));
    double *bi_diff=(double*)malloc(outputCount*sizeof(double));
    double *bf_diff=(double*)malloc(outputCount*sizeof(double));
    double *bo_diff=(double*)malloc(outputCount*sizeof(double));
    double *bg_diff=(double*)malloc(outputCount*sizeof(double));
    bool weightsAllocated=false;
    uint32_t inputAndOutputCount=inputCount+outputCount;

    LSTMState *latestState=getCurrentState();

    // This will cycle totalStepCount times, but we need to go backwards, so we use "stepsBack" in combination with "getState(stepsBack)".
    for(uint32_t stepsBack=0;stepsBack<=availableStepsBack;stepsBack++)
    {
        // 0 = current state
        LSTMState *thisState=getState(stepsBack);
        bool hasDeeperState=stepsBack<availableStepsBack; // Or hasNext? Should we use the next value instead?
        bool hasHigherState=stepsBack>0; // Or hasNext? Should we use the next value instead?
        LSTMState *deeperState=hasDeeperState?getState(stepsBack+1):0;
        LSTMState *higherState=hasHigherState?getState(stepsBack-1):0;
        double *_ds=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the cell states
        double *_do=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the output gate values
        double *_di=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the input gate values
        double *_dg=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the candidate gate values
        double *_df=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the forget gate values
        double *_di_input=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the values inside the activation function calls of the input gates (e.g. tanh(x) <- x)
        double *_df_input=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the values inside the activation function calls of the forget gates (e.g. tanh(x) <- x)
        double *_do_input=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the values inside the activation function calls of the output gates (e.g. tanh(x) <- x)
        double *_dg_input=(double*)malloc(outputCount*sizeof(double)); // Derivative of the loss function w.r.t. the values inside the activation function calls of the candidate gates (e.g. tanh(x) <- x)
        // top_diff_is: diff_h = s->bottom_diff_h
        // top_diff_is: diff_s = higherState->bottom_diff_s (topmost: 0)

        // dxc: transpose operation: array of cells=>weights becomes an array of weights=>cells:
        // e.g.:
        // cell1: [weight1_1,weight1_2,weight1_3]
        // cell2: [weight2_1,weight2_2,weight2_3]
        // becomes:
        // weight1: [cell1,cell2]
        // weight2: [cell1,cell2]
        // weight3: [cell1,cell2
        //
        // Here, we have np.dot(self.param.wi.T, di_input)
        // That means:
        // dxc represents all weights
        // Each weight i in dxc has as its value: sum over cells(sum of the four weights of a cell that have the index i (i,f,o,g), each multiplied by their cell's and weight group's (i,f,o, or g) respective derivative w.r.t. the input))

        double *dxc=(double*)malloc((inputAndOutputCount)*sizeof(double));
        bool dxcWeightsSet=false;

        for(uint32_t cell=0;cell<outputCount;cell++)
        {
            // For each cell
            double diff_s=hasHigherState?higherState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToCellStates[cell]:0.0; // Cell state diff: backpropagation through time
            double diff_h=2.0*(thisState->output[cell]-desiredOutputs[availableStepsBack-stepsBack][cell]); //
            if(hasHigherState)
                diff_h+=higherState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToOutputs[cell];

            _ds[cell]=thisState->outputGateValues[cell]*diff_h+diff_s;
            _do[cell]=thisState->cellStates[cell]*diff_h;
            _di[cell]=thisState->candidateGateValues[cell]*_ds[cell];
            _dg[cell]=thisState->inputGateValues[cell]*_ds[cell];
            _df[cell]=(hasDeeperState?deeperState->cellStates[cell]:0.0)*_ds[cell];
            _di_input[cell]=(1.0-thisState->inputGateValues[cell])*thisState->inputGateValues[cell]*_di[cell];
            _df_input[cell]=(1.0-thisState->forgetGateValues[cell])*thisState->forgetGateValues[cell]*_df[cell];
            _do_input[cell]=(1.0-thisState->outputGateValues[cell])*thisState->outputGateValues[cell]*_do[cell];
            _dg_input[cell]=(1.0-pow(thisState->candidateGateValues[cell],2.0))*_dg[cell];

            if(!weightsAllocated)
            {
                bi_diff[cell]=0.0;
                bf_diff[cell]=0.0;
                bo_diff[cell]=0.0;
                bg_diff[cell]=0.0;
            }

            bi_diff[cell]+=_di_input[cell];
            bf_diff[cell]+=_df_input[cell];
            bo_diff[cell]+=_do_input[cell];
            bg_diff[cell]+=_dg_input[cell];

            if(!weightsAllocated)
            {
                wi_diff[cell]=(double*)malloc(inputAndOutputCount*sizeof(double));
                wf_diff[cell]=(double*)malloc(inputAndOutputCount*sizeof(double));
                wo_diff[cell]=(double*)malloc(inputAndOutputCount*sizeof(double));
                wg_diff[cell]=(double*)malloc(inputAndOutputCount*sizeof(double));
            }

            // For each weight
            for(uint32_t weightInput=0;weightInput<inputCount;weightInput++)
            {
                if(!weightsAllocated)
                {
                    wi_diff[cell][weightInput]=0.0;
                    wf_diff[cell][weightInput]=0.0;
                    wo_diff[cell][weightInput]=0.0;
                    wg_diff[cell][weightInput]=0.0;
                }
                // thisState, not latestState
                wi_diff[cell][weightInput]+=_di_input[cell]*thisState->input[weightInput];
                wf_diff[cell][weightInput]+=_df_input[cell]*thisState->input[weightInput];
                wo_diff[cell][weightInput]+=_do_input[cell]*thisState->input[weightInput];
                wg_diff[cell][weightInput]+=_dg_input[cell]*thisState->input[weightInput];
                if(!dxcWeightsSet)
                    dxc[weightInput]=0.0;
                dxc[weightInput]+=latestState->inputGateWeights[cell][weightInput]*_di_input[cell]+latestState->forgetGateWeights[cell][weightInput]*_df_input[cell]+latestState->outputGateWeights[cell][weightInput]*_do_input[cell]+latestState->candidateGateWeights[cell][weightInput]*_dg_input[cell];
            }
            for(uint32_t weightOutput=0;weightOutput<outputCount;weightOutput++)
            {
                if(!weightsAllocated)
                {
                    wi_diff[cell][inputCount+weightOutput]=0.0;
                    wf_diff[cell][inputCount+weightOutput]=0.0;
                    wo_diff[cell][inputCount+weightOutput]=0.0;
                    wg_diff[cell][inputCount+weightOutput]=0.0;
                }
                // deeperState, not stateBeforeLatestState.
                if(hasDeeperState)
                {
                    wi_diff[cell][inputCount+weightOutput]+=_di_input[cell]*deeperState->output[weightOutput];
                    wf_diff[cell][inputCount+weightOutput]+=_df_input[cell]*deeperState->output[weightOutput];
                    wo_diff[cell][inputCount+weightOutput]+=_do_input[cell]*deeperState->output[weightOutput];
                    wg_diff[cell][inputCount+weightOutput]+=_dg_input[cell]*deeperState->output[weightOutput];
                }
                if(!dxcWeightsSet)
                    dxc[inputCount+weightOutput]=0.0;
                dxc[inputCount+weightOutput]+=latestState->inputGateWeights[cell][inputCount+weightOutput]*_di_input[cell]+latestState->forgetGateWeights[cell][inputCount+weightOutput]*_df_input[cell]+latestState->outputGateWeights[cell][inputCount+weightOutput]*_do_input[cell]+latestState->candidateGateWeights[cell][inputCount+weightOutput]*_dg_input[cell];
            }
            // bottom_diff_s:
            thisState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToCellStates[cell]=_ds[cell]*thisState->forgetGateValues[cell];
            if(!dxcWeightsSet)
                dxcWeightsSet=true;
        }

        // bottom_diff_x:
        memcpy(thisState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToInputs,dxc,inputCount*sizeof(double));
        // bottom_diff_h:
        memcpy(thisState->bottomDerivativesOfLossesFromThisStepOnwardsWithRespectToOutputs,dxc+inputCount,outputCount*sizeof(double));

        free(dxc);
        free(_ds);
        free(_do);
        free(_di);
        free(_dg);
        free(_df);
        free(_di_input);
        free(_df_input);
        free(_do_input);
        free(_dg_input);
        if(!weightsAllocated)
            weightsAllocated=true;
    }

    // Now that we have cycled through all states, apply all changes:
    for(uint32_t cell=0;cell<outputCount;cell++)
    {
        for(uint32_t inputWeight=0;inputWeight<inputCount;inputWeight++)
        {
            latestState->inputGateWeights[cell][inputWeight]-=learningRate*wi_diff[cell][inputWeight];
            latestState->forgetGateWeights[cell][inputWeight]-=learningRate*wf_diff[cell][inputWeight];
            latestState->outputGateWeights[cell][inputWeight]-=learningRate*wo_diff[cell][inputWeight];
            latestState->candidateGateWeights[cell][inputWeight]-=learningRate*wg_diff[cell][inputWeight];
        }
        for(uint32_t outputWeight=0;outputWeight<outputCount;outputWeight++)
        {
            latestState->inputGateWeights[cell][inputCount+outputWeight]-=learningRate*wi_diff[cell][inputCount+outputWeight];
            latestState->forgetGateWeights[cell][inputCount+outputWeight]-=learningRate*wf_diff[cell][inputCount+outputWeight];
            latestState->outputGateWeights[cell][inputCount+outputWeight]-=learningRate*wo_diff[cell][inputCount+outputWeight];
            latestState->candidateGateWeights[cell][inputCount+outputWeight]-=learningRate*wg_diff[cell][inputCount+outputWeight];
        }
        latestState->inputGateBiasWeights[cell]-=learningRate*bi_diff[cell];
        latestState->forgetGateBiasWeights[cell]-=learningRate*bf_diff[cell];
        latestState->outputGateBiasWeights[cell]-=learningRate*bo_diff[cell];
        latestState->candidateGateBiasWeights[cell]-=learningRate*bg_diff[cell];
        free(wi_diff[cell]);
        free(wf_diff[cell]);
        free(wo_diff[cell]);
        free(wg_diff[cell]);
    }
    free(wi_diff);
    free(wf_diff);
    free(wo_diff);
    free(wg_diff);
    free(bi_diff);
    free(bf_diff);
    free(bo_diff);
    free(bg_diff);
}
