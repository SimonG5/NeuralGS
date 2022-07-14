#include "../../include/Optimizer.hpp"

class Adam : public Optimizer
{
private:
    double epsilon;
    double betaOne;
    double betaTwo;

public:
    Adam(double learningRate, double decay, double epsilon, double betaOne, double betaTwo)
    {
        this->learningRate = learningRate;
        this->currentLearningRate = learningRate;
        this->decay = decay;
        this->iterations = 0;
        this->epsilon = epsilon;
        this->betaOne = betaOne;
        this->betaTwo = betaTwo;
    }

    void updateLayer(Layer *layer)
    {
        layer->optiWeightHelper = (this->betaOne * layer->optiWeightHelper).array() + (1 - this->betaOne) * layer->dWeights.array();
        layer->optiBiasHelper = (this->betaOne * layer->optiBiasHelper).array() + (1 - this->betaOne) * layer->dBiases.array();

        Eigen::MatrixXd weightMomentumCorrected = layer->optiWeightHelper.array() / (1 - pow(this->betaOne, this->iterations + 1));
        Eigen::VectorXd biasMomentumCorrected = layer->optiBiasHelper.array() / (1 - pow(this->betaOne, this->iterations + 1));

        layer->weightCache = this->betaTwo * layer->weightCache.array() + (1 - this->betaTwo) * layer->dWeights.array().pow(2);
        layer->biasCache = this->betaTwo * layer->biasCache.array() + (1 - this->betaTwo) * layer->dBiases.array().pow(2);

        Eigen::MatrixXd weightCacheCorrected = layer->weightCache.array() / (1 - pow(this->betaTwo, this->iterations + 1));
        Eigen::VectorXd biasCacheCorrected = layer->biasCache.array() / (1 - pow(this->betaTwo, this->iterations + 1));

        layer->weights.array() += (-this->currentLearningRate * weightMomentumCorrected).array() / (weightCacheCorrected.array().sqrt() + this->epsilon).array();
        layer->biases.array() += (-this->currentLearningRate * biasMomentumCorrected).array() / (biasCacheCorrected.array().sqrt() + this->epsilon).array();
    }
};