#include "../include/Model.hpp"
#include <chrono>
#include <fstream>

Model::Model()
{
    this->accuarcyTotal = 0;
    this->testBatches = 0;
    this->lossTotal = 0;
}

Model::~Model()
{
    delete (this->loss);
    delete (this->optimizer);
    for (size_t i = 0; i < this->layers.size(); i++)
    {
        delete (this->layers[i]);
        delete (this->activators[i]);
    }
}

void Model::train(const Eigen::MatrixXd &x, const Eigen::VectorXi &y, const int &iterations, const int &printGap)
{
    for (int i = 0; i < iterations + 1; i++)
    {
        Eigen::MatrixXd output = forward(x);

        double correctValues = 0;

        for (int r = 0; r < output.rows(); r++)
        {
            double highestValue = 0;
            int highestIndex = 0;
            for (int c = 0; c < output.cols(); c++)
            {
                if (output(r, c) > highestValue)
                {
                    highestValue = output(r, c);
                    highestIndex = c;
                }
            }
            if (highestIndex == y(r))
            {
                correctValues++;
            }
        }

        double accuary = correctValues / y.size();
        double lossVal = this->loss->forward(output, y);

        backward(output, y);

        this->optimizer->preUpdateLayer();
        for (size_t i = 0; i < this->layers.size(); i++)
        {
            this->optimizer->updateLayer(layers[i]);
        }
        this->optimizer->postUpdateLayer();

        if (i % printGap == 0)
        {
            std::cout << "Epoch: " << i << " Accuary: " << accuary << " Loss: " << lossVal << " Learning rate: " << this->optimizer->getLearningRate() << std::endl;
            saveModel();
        }
    }
}

void Model::test(const Eigen::MatrixXd &x, const Eigen::VectorXi &y)
{
    Eigen::MatrixXd output = forward(x);

    double correctValues = 0;

    for (int r = 0; r < output.rows(); r++)
    {
        double highestValue = 0;
        int highestIndex = 0;
        for (int c = 0; c < output.cols(); c++)
        {
            if (output(r, c) > highestValue)
            {
                highestValue = output(r, c);
                highestIndex = c;
            }
        }
        if (highestIndex == y(r))
        {
            correctValues++;
        }
    }

    double accuary = correctValues / y.size();
    double lossVal = this->loss->forward(output, y);
    this->accuarcyTotal += accuary;
    this->lossTotal += lossVal;
    this->testBatches++;

    std::cout << "Test set batch: " << testBatches << ", average accuary: " << (this->accuarcyTotal / testBatches) << ", average loss: " << (this->lossTotal / testBatches) << std::endl;
}

void Model::setLoss(Loss *loss)
{
    this->loss = loss;
}

void Model::setOptimizer(Optimizer *optimizer)
{
    this->optimizer = optimizer;
}

void Model::appendLayer(Layer *layer, Activator *activator)
{
    this->layers.push_back(layer);
    this->activators.push_back(activator);
}

Eigen::MatrixXd Model::forward(const Eigen::MatrixXd &input)
{
    this->layers[0]->forward(input);
    this->activators[0]->forward(this->layers[0]->getOutput());

    for (size_t i = 1; i < this->layers.size(); i++)
    {
        this->layers[i]->forward(this->activators[i - 1]->getOutput());
        this->activators[i]->forward(this->layers[i]->getOutput());
    }

    return this->activators[activators.size() - 1]->getOutput();
}

void Model::backward(const Eigen::MatrixXd &output, const Eigen::VectorXi &y)
{
    this->loss->backward(output, y);

    for (int i = this->layers.size() - 1; i >= 0; i--)
    {
        if (i == static_cast<int>(this->layers.size()) - 1)
        {
            this->activators[i]->backward(loss->getDInputs());
        }
        else
        {
            this->activators[i]->backward(this->layers[i + 1]->dInputs);
        }
        this->layers[i]->backward(this->activators[i]->getDInputs());
    }
}

void Model::saveModel()
{
    std::ofstream o("model.json");
    json output;
    for (size_t i = 0; i < this->layers.size(); i++)
    {
        Eigen::MatrixXd weights = this->layers[i]->getWeights();
        Eigen::RowVectorXd biases = this->layers[i]->getBiases();
        for (int r = 0; r < weights.rows(); r++)
        {
            for (int c = 0; c < weights.cols(); c++)
            {
                output["Layer " + std::to_string(i + 1) + " - " + std::to_string(r + 1) + ":" + std::to_string(c + 1)] = std::to_string(weights(r, c));
            }
        }
        for (int j = 0; j < biases.size(); j++)
        {
            output["Biases " + std::to_string(i + 1) + ":" + std::to_string(j + 1)] = std::to_string(biases(j));
        }
    }
    std::ofstream file("model.json");
    file << output;
}

void Model::loadModel(json model)
{
    for (size_t i = 0; i < this->layers.size(); i++)
    {
        Eigen::MatrixXd currentLayer = this->layers[i]->getWeights();
        Eigen::RowVectorXd currentBias = this->layers[i]->getBiases();
        for (int r = 0; r < currentLayer.rows(); r++)
        {
            for (int c = 0; c < currentLayer.cols(); c++)
            {
                std::string weight = model["Layer " + std::to_string(i + 1) + " - " + std::to_string(r + 1) + ":" + std::to_string(c + 1)];
                currentLayer(r, c) = std::stod(weight);
            }
        }
        for (int r = 0; r < currentBias.size(); r++)
        {
            std::string bias = model["Biases " + std::to_string(i + 1) + ":" + std::to_string(r + 1)];
            currentBias(r) = std::stod(bias);
        }
        this->layers[i]->setWeights(currentLayer);
        this->layers[i]->setBiases(currentBias);
    }
}