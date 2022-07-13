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

    void preUpdateLayer()
    {
        if (this->decay != 0)
            this->currentLearningRate = this->learningRate * (1 / (1 + this->decay * this->iterations));
    }

    void updateLayer(Layer *layer)
    {
        // layer->optiWeightHelper = this->rho * layer->optiWeightHelper + (1 - this->rho) * layer->dWeights.array().pow(2);
        // layer->optiBiasHelper = this->rho * layer->optiBiasHelper + (1 - this->rho) * layer->dBiases.array().pow(2);

        layer->weights.array() += (-this->currentLearningRate * layer->dWeights).array() / (layer->optiWeightHelper.array().sqrt() + this->epsilon).array();
        layer->weights.array() += (-this->currentLearningRate * layer->dBiases).array() / (layer->optiBiasHelper.array().sqrt() + this->epsilon).array();
    }

    void postUpdateLayer()
    {
        this->iterations++;
    }
};