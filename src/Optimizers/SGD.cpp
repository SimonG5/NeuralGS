#include "../../include/Optimizer.hpp"

class SGD : public Optimizer
{
private:
    double momentum;

public:
    SGD(double learningRate, double decay, double momentum)
    {
        this->learningRate = learningRate;
        this->currentLearningRate = learningRate;
        this->decay = decay;
        this->iterations = 0;
        this->momentum = momentum;
    }

    void updateLayer(Layer *layer)
    {
        if (this->momentum != 0)
        {
            layer->weights += this->momentum * layer->optiWeightHelper - this->currentLearningRate * layer->dWeights;
            layer->biases += this->momentum * layer->optiBiasHelper - this->currentLearningRate * layer->dBiases;
        }
        else
        {
            layer->weights += -this->currentLearningRate * layer->dWeights;
            layer->biases += -this->currentLearningRate * layer->dBiases;
        }
    }
};