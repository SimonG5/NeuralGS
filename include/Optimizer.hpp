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
    virtual void preUpdateLayer() = 0;
    virtual void updateLayer(Layer *layer) = 0;
    virtual void postUpdateLayer() = 0;

    double getLearningRate()
    {
        return currentLearningRate;
    }
};
