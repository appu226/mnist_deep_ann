#include "ann_interface.h"

#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using namespace mnist_deep_ann;



namespace {



#define ANN_ASSERT(expr, message) \
	{ \
		if (!(expr)) \
		{ \
			std::stringstream msg_ss;\
			msg_ss << message; \
			throw std::runtime_error(msg_ss.str()); \
		} \
	}




	// Element forward declaration
	// Base class to all types of objects in the network:
	//   Neurons, Weights, 
	//   Network Inputs and Outputs
	//   Even the wires that connect stuff
	struct Element;
	using ElementWPtr = std::weak_ptr<Element>;
	using ElementWPtrVec = std::vector<ElementWPtr>;
	using ElementPtr = std::shared_ptr<Element>;
	using ElementSet = std::unordered_set<Element const*>;
	using ElementPtrVec = std::vector<ElementPtr>;



	// Element definition
	// Base class to all types of objects in the network:
	//   Neurons, Weights, 
	//   Network Inputs and Outputs
	//   Even the wires that connect stuff
	struct Element
	{
		virtual ~Element() {}

		ElementWPtrVec inputElements;
		ElementWPtrVec outputElements;

		// the output value of this element
		double output;
		// the derivative of error to the output of this element
		double errorSensitivityToOutput;

		virtual void forwardPropagate() = 0;
		virtual void backPropagate() = 0;
	};



	// Network Input element definition
	struct NetworkInput : public Element
	{
		void forwardPropagate() override {}
		void backPropagate() override {}
	};



	// Network Output Error
	//   returns the squared error from expected output
	struct NetworkOutputError : public Element
	{
		double expectedOutput;

		NetworkOutputError(ElementWPtr inputElement)
		{
			inputElements.push_back(std::move(inputElement));
		}
		void forwardPropagate() override {
			ANN_ASSERT(inputElements.size() == 1, "Network output should be connected to one and only one element.");
			output = expectedOutput - inputElements[0].lock()->output;
			output *= output;
		}

		void backPropagate() override
		{
			ANN_ASSERT(inputElements.size() == 1, "Network output should be connected to one and only one element.");
			errorSensitivityToOutput = 1.0;
			auto input = inputElements[0].lock();
			input->errorSensitivityToOutput += 2 * (input->output - expectedOutput);
		}
	};




	// Weight definition
	struct Weight : public Element
	{
		void forwardPropagate() override {}
		void backPropagate() override {}

		void adjust(double stepSize)
		{
			output -= errorSensitivityToOutput * stepSize;
		}
	};


	
	// Neuron definition
	// The input connections are a sequence of pairs
	//    of weight and input value.
	struct Neuron : public Element
	{

		double cachedWeightedSumOfInputs;
		IActivationFunction::CPtr activationFunction;

		Neuron(size_t v_numInputs, IActivationFunction::CPtr v_activationFunction) :
			cachedWeightedSumOfInputs(0),
			activationFunction(v_activationFunction)
		{
			inputElements.resize(v_numInputs * 2);
		}


		size_t numInputs() const { return inputElements.size() / 2; }
		double weightedSumOfInputs() const
		{
			double result = 0;
			double w, i;
			for (size_t idxInput = 0; idxInput + 1 < inputElements.size(); idxInput += 2)
			{
				w = inputElements[idxInput].lock()->output;
				i = inputElements[idxInput + 1].lock()->output;
				result += w * i;
			}
			return result;
		}

		void forwardPropagate() override
		{
			cachedWeightedSumOfInputs = weightedSumOfInputs();
			output = activationFunction->compute(cachedWeightedSumOfInputs);
		}

		void backPropagate() override
		{
			double activationFunctionDeriv = activationFunction->derivative(cachedWeightedSumOfInputs);
			double errorSensitivityToActivationFunction = errorSensitivityToOutput * activationFunctionDeriv;
			for (size_t idxInput = 0; idxInput + 1 < inputElements.size(); idxInput += 2)
			{
				auto w = inputElements[idxInput].lock();
				auto i = inputElements[idxInput + 1].lock();
				w->errorSensitivityToOutput += errorSensitivityToActivationFunction * i->output;
				i->errorSensitivityToOutput += errorSensitivityToActivationFunction * w->output;
			}
		}

	};



	// Implementation for an INetwork
	class NetworkImpl : public INetwork {
	public:

		// strongly typed element pointers
		std::vector<std::shared_ptr<Weight> > m_weights;
		std::vector<std::shared_ptr<Neuron> > m_neurons;
		std::vector<std::shared_ptr<NetworkInput> > m_inputs;
		std::vector<std::shared_ptr<NetworkOutputError> > m_outputErrors;
		
		// elements in reverse topologically sorted order
		std::vector<std::shared_ptr<Element> > m_reverseTopSort;
		// whether reverseTopSort has been properly populated or not
		//   this is needed to ensure we don't trigger top sort all the time
		bool m_initialized = false;
			

		// set m_initialized to false in constructor
		NetworkImpl()
		{
			m_initialized = false;
		}


		WeightIndex addWeight() override
		{
			m_initialized = false;
			size_t result = m_weights.size();
			m_weights.emplace_back(new Weight);
			return { result };
		}


		NeuronIndex addNeuron(IActivationFunction::CPtr const& activationFunction, size_t numInputs) override
		{
			m_initialized = false;
			size_t result = m_neurons.size();
			m_neurons.emplace_back(new Neuron(numInputs, activationFunction));
			return { result };
		}


		void connectNeurons(NeuronIndex fromNeuron, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) override
		{
			m_initialized = false;
			ANN_ASSERT(m_neurons.size() > fromNeuron.v, "Invalid neuron index " << fromNeuron.v);
			ANN_ASSERT(m_neurons.size() > toNeuron.v, "Invalid neuron index " << toNeuron.v);
			auto & from = m_neurons[fromNeuron.v];
			auto & to = m_neurons[toNeuron.v];
			ANN_ASSERT(to->numInputs() > toNeuronInput.v, "Invalid input index " << toNeuronInput.v << " on neuron " << toNeuron.v);
			ANN_ASSERT(to->inputElements[toNeuronInput.v * 2 + 1].expired(), "Neuron " << toNeuron.v << " already has an input at index " << toNeuronInput.v);
			from->outputElements.push_back(to);
			to->inputElements[toNeuronInput.v * 2 + 1] = from;
		}

		NetworkInputIndex addInput() override
		{
			m_initialized = false;
			size_t result = m_inputs.size();
			m_inputs.emplace_back(new NetworkInput);
			return { result };
		}


		NetworkOutputIndex addOutput(NeuronIndex fromNeuron) override
		{
			m_initialized = false;
			size_t result = m_outputErrors.size();
			ANN_ASSERT(m_neurons.size() > fromNeuron.v, "Invalid neuron index " << fromNeuron.v);
			auto & from = m_neurons[fromNeuron.v];
			m_outputErrors.emplace_back(new NetworkOutputError(from));
			from->outputElements.emplace_back(m_outputErrors.back());
			return { result };
		}


		void connectInputToNeuron(NetworkInputIndex fromInput, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) override
		{
			m_initialized = false;
			ANN_ASSERT(m_inputs.size() > fromInput.v, "Invalid input index " << fromInput.v);
			auto & from = m_inputs[fromInput.v];
			ANN_ASSERT(m_neurons.size() > toNeuron.v, "Invalid neuron index " << toNeuron.v);
			auto & to = m_neurons[toNeuron.v];
			ANN_ASSERT(to->numInputs() > toNeuronInput.v, "Invalid input index " << toNeuronInput.v << " to neuron " << toNeuron.v);
			ANN_ASSERT(to->inputElements[toNeuronInput.v * 2 + 1].expired(), "Neuron " << toNeuron.v << " already has an input at index " << toNeuronInput.v);
			from->outputElements.push_back(to);
			to->inputElements[toNeuronInput.v * 2 + 1] = from;
		}



		void connectWeightToNeuron(WeightIndex fromWeight, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) override
		{
			m_initialized = false;
			ANN_ASSERT(m_weights.size() > fromWeight.v, "Invalid weight index " << fromWeight.v);
			auto& from = m_weights[fromWeight.v];
			ANN_ASSERT(m_neurons.size() > toNeuron.v, "Invalid neuron index " << toNeuron.v);
			auto& to = m_neurons[toNeuron.v];
			ANN_ASSERT(to->numInputs() > toNeuronInput.v, "Invalid neuron input " << toNeuronInput.v);
			ANN_ASSERT(to->inputElements[2 * toNeuronInput.v].expired(), "Neuron " << toNeuron.v << " already has an input at index " << toNeuronInput.v);
			from->outputElements.emplace_back(to);
			to->inputElements[2 * toNeuronInput.v] = from;
		}




		std::optional<std::string> getValidationError() const override
		{
#define NETWORKIMPL_VALIDATE(cond, msg) \
if (!(cond)) \
{ \
    std::stringstream msg_ss; \
    msg_ss << msg; \
    return msg_ss.str(); \
}

			std::unordered_map<Element const*, std::unordered_set<Element const*> > outputCheck, inputCheck;
			auto recordInputsAndOutputs = [&inputCheck, &outputCheck](Element const& element)
				{
					for (auto& inputElement : element.inputElements)
						inputCheck[&element].insert(inputElement.lock().get());
					for (auto& outputElement : element.outputElements)
						outputCheck[&element].insert(outputElement.lock().get());
				};

			for (auto& element : m_inputs)
				recordInputsAndOutputs(*element);
			for (auto& element : m_weights)
				recordInputsAndOutputs(*element);
			for (auto& element : m_neurons)
				recordInputsAndOutputs(*element);
			for (auto& element : m_outputErrors)
				recordInputsAndOutputs(*element);

			for (size_t in = 0; in < m_neurons.size(); ++in)
			{
				auto n = m_neurons[in].get();
				NETWORKIMPL_VALIDATE(
					n->activationFunction, 
					"Neuron " << in << " does not have a valid activation function"
				);

				for (size_t ii = 0; ii < n->numInputs(); ++ii)
				{
					NETWORKIMPL_VALIDATE(
						!n->inputElements[ii * 2].expired(),
						"Neuron " << in << " does not have a weight at index " << ii
					);
					auto w = n->inputElements[ii * 2].lock().get();
					NETWORKIMPL_VALIDATE(
						outputCheck[w].count(n) > 0,
						"Neuron weight " << ii << " for neuron " << in << " has not registered neuron " << in << " as an output."
					);

					NETWORKIMPL_VALIDATE(
						!n->inputElements[ii * 2 + 1].expired(),
						"Neuron " << in << " does not have an input connection at index " << ii
					);
					auto i = n->inputElements[ii * 2 + 1].lock().get();
					NETWORKIMPL_VALIDATE(
						outputCheck[w].count(n) != 0,
						"Neuron input " << ii << " for neuron " << in << " has not registered neuron " << in << " as an output."
					);
				}
				for (auto& no : n->outputElements)
				{
					NETWORKIMPL_VALIDATE(
						!no.expired(),
						"Neuron " << in << " has a stale output"
					);
					NETWORKIMPL_VALIDATE(
						inputCheck[no.lock().get()].count(n) > 0,
						"Neuron " << in << " has not been registered as an input for some output element"
					);
				}
			}

			for (size_t iw = 0; iw < m_weights.size(); ++iw)
			{
				NETWORKIMPL_VALIDATE(
					m_weights[iw],
					"Stale weight at index " << iw
				);
				auto w = m_weights[iw].get();
				NETWORKIMPL_VALIDATE(
					w->inputElements.empty(), 
					"Weight " << iw << " should not have input elements"
				);
				for (auto& o : w->outputElements)
				{
					NETWORKIMPL_VALIDATE(
						!o.expired(), 
						"Weight " << iw << " has a stale output"
					);
					NETWORKIMPL_VALIDATE(
						inputCheck[o.lock().get()].count(w) > 0,
						"Weight " << iw << " is not registered as an input for some output element."
					);
				}
			}

			for (size_t ini = 0; ini < m_inputs.size(); ++ini)
			{
				NETWORKIMPL_VALIDATE(
					m_inputs[ini],
					"Stale input at index " << ini
				);
				auto ni = m_inputs[ini].get();
				NETWORKIMPL_VALIDATE(
					ni->inputElements.empty(),
					"Input " << ini << " should not have any inputs."
				);
				for (auto& nio : ni->outputElements)
				{
					NETWORKIMPL_VALIDATE(
						!nio.expired(),
						"Input " << ini << " has a stale output."
					);
					NETWORKIMPL_VALIDATE(
						inputCheck[nio.lock().get()].count(ni) > 0,
						"Input " << ini << " is not registered as an input for some output element."
					);
				}
			}

			for (size_t ioe = 0; ioe < m_outputErrors.size(); ++ioe)
			{
				NETWORKIMPL_VALIDATE(
					m_outputErrors[ioe],
					"Output " << ioe << " not properly set up."
				);
				auto oe = m_outputErrors[ioe].get();
				NETWORKIMPL_VALIDATE(
					oe->outputElements.size() == 0,
					"Output " << ioe << " should not have outputs connected to it."
				);
				NETWORKIMPL_VALIDATE(
					oe->inputElements.size() == 1 && !oe->inputElements.front().expired(),
					"Output " << ioe << " should have exactly one input."
				);
				auto oei = oe->inputElements.front().lock().get();
				NETWORKIMPL_VALIDATE(
					outputCheck[oei].count(oe) > 0,
					"Output " << ioe << " not properly registered as an output."
				);
			}
			return {};
#undef NETWORKIMPL_VALIDATE
		}



		// depth first visit of an element
		//   needed for topological sorting
		static void visit(ElementPtr const& element, ElementSet& visited, ElementSet& visiting, ElementPtrVec& reverseTopSort)
		{
			if (visited.count(element.get()) > 0)
				return;
			if (visiting.count(element.get()) > 0)
				throw std::runtime_error("Cyclic neural networks not yet supported.");
			visiting.insert(element.get());
			for (auto& oe : element->outputElements)
				visit(oe.lock(), visited, visiting, reverseTopSort);
			visiting.erase(element.get());
			visited.insert(element.get());
			reverseTopSort.push_back(element);
		}




		// check for errors
		//   topologically sort
		void initialize()
		{
			if (m_initialized)
				return;
			auto err = getValidationError();
			if (err.has_value())
				throw std::runtime_error(err.value());


			m_reverseTopSort.clear();

			ElementSet visited;
			ElementSet visiting;
			for (auto& element : m_outputErrors)
				visit(element, visited, visiting, m_reverseTopSort);
			for (auto& element : m_neurons)
				visit(element, visited, visiting, m_reverseTopSort);
			for (auto& element : m_inputs)
				visit(element, visited, visiting, m_reverseTopSort);
			for (auto& element : m_weights)
				visit(element, visited, visiting, m_reverseTopSort);

			m_initialized = true;
		}




		double propagateExamples(const std::vector<Example>& examples, double stepSize) override
		{
			initialize();
			double error = 0;
				
			for (auto& elem : m_reverseTopSort)
				elem->errorSensitivityToOutput = false;
			for (auto const& example : examples)
			{
				ANN_ASSERT(example.inputs.size() == m_inputs.size(), "Incorrect number of inputs in example");
				for (size_t ii = 0; ii < m_inputs.size(); ++ii)
					m_inputs[ii]->output = example.inputs[ii];
				ANN_ASSERT(example.outputs.size() == m_outputErrors.size(), "Incorrect number of outputs in example");
				for (size_t io = 0; io < m_outputErrors.size(); ++io)
					m_outputErrors[io]->expectedOutput = example.outputs[io];

				for (auto elemIt = m_reverseTopSort.rbegin(); elemIt != m_reverseTopSort.rend(); ++elemIt)
					(*elemIt)->forwardPropagate();

				for (auto output : m_outputErrors)
					error += output->output;

				for (auto& elem : m_reverseTopSort)
					elem->backPropagate();
			}

			for (auto& w : m_weights)
				w->adjust(stepSize);

			return error;
		}


		void perturbWeights(double maxAbsShift) override
		{
			initialize();
			if (maxAbsShift == 0)
				return;
			if (maxAbsShift < 0)
				maxAbsShift *= -1;
			std::default_random_engine generator;
			std::uniform_real_distribution<double> distribution(-maxAbsShift, maxAbsShift);
			for (auto& w : m_weights)
				w->output += distribution(generator);
		}




		RVec evaluate(const RVec& inputs) override
		{
			initialize();
			ANN_ASSERT(inputs.size() == m_inputs.size(), "Incorrect number of inputs.");
			for (size_t ii = 0; ii < inputs.size(); ++ii)
				m_inputs[ii]->output = inputs[ii];
			for (auto elemIt = m_reverseTopSort.rbegin(); elemIt != m_reverseTopSort.rend(); ++elemIt)
				(*elemIt)->forwardPropagate();
			RVec result;
			result.reserve(m_outputErrors.size());
			for (auto& oe : m_outputErrors)
			{
				ANN_ASSERT(oe->inputElements.size() == 1 && oe->inputElements[0].lock(), "Output not connected to any inputs");
				result.push_back(oe->inputElements[0].lock()->output);
			}
			return result;
		}


		double getNeuronOutput(NeuronIndex neuron) const override
		{
			ANN_ASSERT(m_neurons.size() > neuron.v, "Invalid neuron index " << neuron.v);
			return m_neurons[neuron.v]->output;
		}

		double getErrorSensitivityToNeuronOutput(NeuronIndex neuron) const override
		{
			ANN_ASSERT(m_neurons.size() > neuron.v, "Invalid neuron index " << neuron.v);
			return m_neurons[neuron.v]->errorSensitivityToOutput;
		}

		double getWeight(WeightIndex weight) const override
		{
			ANN_ASSERT(m_weights.size() > weight.v, "Invalid weight index " << weight.v);
			return m_weights[weight.v]->output;
		}

		double getErrorSensitivityToWeight(WeightIndex weight) const override
		{
			ANN_ASSERT(m_weights.size() > weight.v, "Invalid weight index " << weight.v);
			return m_weights[weight.v]->errorSensitivityToOutput;
		}
			
	}; // end NetworkImpl

} // end anonymous namespace


namespace mnist_deep_ann {

	INetwork::Ptr INetwork::create()
	{
		return std::make_shared<NetworkImpl>();
	}

} // end namespace mnist_deep_ann