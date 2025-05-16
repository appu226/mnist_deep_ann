#pragma once

#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


#include "ann_export.h"


#define ANN_ASSERT(expr, message) \
	{ \
		if (!(expr)) \
		{ \
			std::stringstream msg_ss;\
			msg_ss << message; \
			throw std::runtime_error(msg_ss.str()); \
		} \
	}


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

    struct WeightIndex        { size_t v; };
    struct NeuronIndex        { size_t v; };
    struct ConnectionIndex    { size_t v; };
    struct NeuronInputIndex   { size_t v; };
    struct NetworkInputIndex  { size_t v; };
    struct NetworkOutputIndex { size_t v; };

    struct Example {
        RVec inputs;
        RVec outputs;
        using Vec = std::vector<Example>;
        using VecPtr = std::shared_ptr<Vec>;
        using VecCPtr = std::shared_ptr<Vec const>;
    };


    class ANN_EXPORT INetwork {
        public:
        using Ptr = std::shared_ptr<INetwork>;
        virtual ~INetwork() {}

        virtual WeightIndex addWeight() = 0;
        virtual NeuronIndex addNeuron(IActivationFunction::CPtr const& activationFunction, size_t numInputs) = 0;
        virtual void connectNeurons(NeuronIndex fromNeuron, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) = 0;
        virtual NetworkInputIndex addInput() = 0;
        virtual NetworkOutputIndex addOutput(NeuronIndex fromNeuron) = 0;
        virtual void connectInputToNeuron(NetworkInputIndex fromInput, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) = 0;
        virtual void connectWeightToNeuron(WeightIndex fromWeight, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) = 0;

        virtual std::optional<std::string> getValidationError() const = 0;

        virtual double propagateExamples(const std::vector<Example>& examples, double stepSize) = 0;

        virtual void perturbWeights(double maxAbsShift) = 0;

        virtual RVec evaluate(const RVec& inputs) = 0;

        virtual void adjust(double stepSize) = 0;

        static Ptr create();

        virtual double getNeuronOutput(NeuronIndex neuron) const = 0;
        virtual double getErrorSensitivityToNeuronOutput(NeuronIndex neuron) const = 0;
        virtual double getWeight(WeightIndex weight) const = 0;
        virtual double getErrorSensitivityToWeight(WeightIndex weight) const = 0;

        virtual void diagnose(std::ostream& out) const = 0;

    };

}
