#include "../../include/Optimizer.hpp"
#include <iostream>

class AdaGrad : public Optimizer
{
private:
    double epsilon;

public:
    AdaGrad(double learningRate, double decay, double epsilon)
    {
        this->learningRate = learningRate;
        this->currentLearningRate = learningRate;
        this->decay = decay;
        this->iterations = 0;
        this->epsilon = epsilon;
    }

    void updateLayer(Layer *layer)
    {
        layer->optiWeightHelper.array() += layer->dWeights.array().pow(2);
        layer->optiBiasHelper.array() += layer->dBiases.array().pow(2);

        layer->weights.array() += (-this->currentLearningRate * layer->dWeights).array() / (layer->optiWeightHelper.array().sqrt() + this->epsilon).array();
        layer->biases.array() += (-this->currentLearningRate * layer->dBiases).array() / (layer->optiBiasHelper.array().sqrt() + this->epsilon).array();
    }
};