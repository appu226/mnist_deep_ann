#pragma once

#include <memory>
#include <optional>
#include <vector>


#include "ann_diagnostics.h"
#include "ann_export.h"


namespace mnist_deep_ann
{
    using RVec = std::vector<double>;

    class ANN_EXPORT IActivationFunction {
       public:
       using CPtr = std::shared_ptr<IActivationFunction const>;
       virtual ~IActivationFunction() {}
       virtual double compute(double weightedSumOfInputs) const = 0;
       virtual double derivative(double weightedSumOfInputs) const = 0;
    };

    struct NeuronIndex        { size_t v; };
    struct ConnectionIndex    { size_t v; };
    struct NeuronInputIndex   { size_t v; };
    struct NetworkInputIndex  { size_t v; };
    struct NetworkOutputIndex { size_t v; };

    struct Example {
        RVec inputs;
        RVec outputs;
    };


    class ANN_EXPORT INetwork {
        public:
        using Ptr = std::shared_ptr<INetwork>;
        virtual ~INetwork() {}

        virtual NeuronIndex addNeuron(IActivationFunction::CPtr const& activationFunction, size_t numInputs) = 0;
        virtual ConnectionIndex connectNeurons(NeuronIndex fromNeuron, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) = 0;
        virtual NetworkInputIndex addInput() = 0;
        virtual NetworkOutputIndex addOutput(NeuronIndex fromNeuron) = 0;
        virtual void connectInputToNeuron(NetworkInputIndex fromInput, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) = 0;

        virtual std::optional<std::string> getValidationError() const = 0;

        virtual double propagateExamples(const std::vector<Example>& examples, double stepSize) = 0;

        virtual void perturbWeights(double maxAbsShift) = 0;

        virtual RVec evaluate(const RVec& inputs) = 0;

        virtual NetworkDiagnostics::Ptr getDiagnostics() const = 0;
        virtual void updateDiagnostics(NetworkDiagnostics & diagnostics) const = 0;

        static Ptr create();

    };

}
