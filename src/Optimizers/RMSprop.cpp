#include "../../include/Optimizer.hpp"

class RMSprop : public Optimizer
{
private:
    double epsilon;
    double rho;

public:
    RMSprop(double learningRate, double decay, double epsilon, double rho)
    {
        this->learningRate = learningRate;
        this->currentLearningRate = learningRate;
        this->decay = decay;
        this->iterations = 0;
        this->epsilon = epsilon;
        this->rho = rho;
    }

    void updateLayer(Layer *layer)
    {
        layer->optiWeightHelper = (this->rho * layer->optiWeightHelper).array() + (1 - this->rho) * layer->dWeights.array().pow(2);
        layer->optiBiasHelper = (this->rho * layer->optiBiasHelper).array() + (1 - this->rho) * layer->dBiases.array().pow(2);

        layer->weights.array() += (-this->currentLearningRate * layer->dWeights).array() / (layer->optiWeightHelper.array().sqrt() + this->epsilon).array();
        layer->biases.array() += (-this->currentLearningRate * layer->dBiases).array() / (layer->optiBiasHelper.array().sqrt() + this->epsilon).array();
    }
};