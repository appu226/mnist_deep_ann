#pragma once

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

namespace mnist_deep_ann
{
    // Diagnostics from a single neuron
    struct NeuronDiagnostics {

        // current weight to each input
        std::vector<double> inputWeights;

        // last computed output
        double previousOutput;

        // last computed derivative of error function to each weight
        std::vector<double> errorSensitivitiesToWeights;

        // last computed derivative of error function to neuron output
        double errorSensitivityToOutput;


        NeuronDiagnostics(size_t numInputs):
            inputWeights(numInputs, 0.0),
            previousOutput(0.0),
            errorSensitivitiesToWeights(numInputs, 0.0),
            errorSensitivityToOutput(0.0)
        { }
    };







    // represents an input connection into a neuron
    struct NeuronInput {
        
        // name of neuron
        std::string neuronName;

        // which input of the neuron
        size_t inputIndex;

    };





    // full network diagnostics
    struct NetworkDiagnostics {

        // name of this network
        std::string networkName;
        
        // map from neuron name to neuron diagnostics
        std::unordered_map<std::string, NeuronDiagnostics> neurons;

        // list of connections for each input to the network
        // inputConnections[i][j].{neuronName, inputIndex} tells us that:
        // - network input `i`
        // - is connected to neuron with name `neuronName`
        // - at input index `inputIndex` of that neuron
        std::vector<std::vector<NeuronInput> > inputConnections;

        // list of output connecitons
        // outputConnection[i] gives the name of the neuron that feeds
        //     its output to the `i`-th network output
        std::vector<std::string> outputConnections;

        // list of inter neuron connections
        // neuralConnections[i],{first, second.{neuronName, inputIndex}} tells us that:
        // - output of neuron with name `first`
        // - is connected to input of neuron with name `neuronName`
        // - at input index `i`
        std::vector<std::pair<std::string, NeuronInput> > neuronConnections;

        using Ptr = std::shared_ptr<NetworkDiagnostics>;
    };


} // end namespace mnist_deep_ann
