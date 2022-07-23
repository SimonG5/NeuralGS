#pragma once

#include "Layer.hpp"

class Optimizer
{
protected:
    int iterations;
    double decay;
    double learningRate;
    double currentLearningRate;

public:
    virtual ~Optimizer() = default;
    virtual void updateLayer(Layer *layer) = 0;

    void preUpdateLayer()
    {
        if (this->decay != 0)
            this->currentLearningRate = this->learningRate * (1 / (1 + this->decay * this->iterations));
    }

    void postUpdateLayer()
    {
        this->iterations++;
    }

    double getLearningRate()
    {
        return currentLearningRate;
    }
};
