#include "ann_interface.h"

#include <stdexcept>
#include <unordered_set>

using namespace mnist_deep_ann;

namespace {

inline std::string s(size_t t) { return std::to_string(t); }

// class forward declarations
struct Neuron;
struct Connection;


// typedefs
using NeuronPtr = std::shared_ptr<Neuron>;
using ConnectionPtr = std::shared_ptr<Connection>;
using NeuronWPtr = std::weak_ptr<Neuron>;

// function declarations
double dotProduct(RVec const& x, RVec const& y);


// class declarations
struct Neuron {
    Neuron(size_t numInputs, IActivationFunction::CPtr const& activationFunction);

    IActivationFunction::CPtr activationFunction;
    RVec weights;
    std::vector<ConnectionPtr> inputConnections;
    std::vector<ConnectionPtr> outputConnections;
    double deriv;
    RVec accumulatedWeightAdjustment;

};

struct Connection {
    Connection(NeuronWPtr const& incomingFromNeuron,
               NeuronWPtr const& outgoingToNeuron,
               NeuronInputIndex outgoingToNeuronInputIndex);
    
    NeuronWPtr incomingFromNeuron;
    NeuronWPtr outgoingToNeuron;
    NeuronInputIndex outgoingToNeuronInputIndex;
    double output;
    double errorSensitivityToOutput;
};

class NetworkImpl: public INetwork
{
    public:
    NetworkImpl();
    NeuronIndex addNeuron(IActivationFunction::CPtr const& activationFunction, size_t numInputs) override;
    ConnectionIndex connectNeurons(NeuronIndex fromNeuron, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) override;
    NetworkInputIndex addInput() override;
    NetworkOutputIndex addOutput(NeuronIndex fromNeuron) override;

    void connectInputToNeuron(NetworkInputIndex fromInput, NeuronIndex toNeuron, NeuronInputIndex toNeuronInput) override;

    std::optional<std::string> getValidationError() const override;
    double propagateExamples(const std::vector<Example>& examples, double stepSize) override;
    RVec evaluate(RVec const& input) override;

    private:
    std::vector<NeuronPtr> m_neurons;
    std::vector<ConnectionPtr> m_neuronConnections;
    std::vector<std::vector<ConnectionPtr> > m_inputConnections;
    std::vector<ConnectionPtr> m_outputConnections;
    std::vector<NeuronPtr> m_inputToOutputNeuronOrdering;
    bool m_isPreProcessed;

    NeuronPtr const& safeGetNeuron(NeuronIndex idx) const;
    void validateNeuronInputIndex(const Neuron& neuron, NeuronIndex nidx, NeuronInputIndex iidx) const;
    void validateNetworkInput(NetworkInputIndex idx) const;
    void preprocess();
};

struct NeuronVisitingState
{
    NeuronPtr nptr;
    size_t next_input_to_visit;
    NeuronVisitingState(NeuronPtr const& v_nptr, size_t v_next_input_to_visit):
        nptr(v_nptr),
        next_input_to_visit(v_next_input_to_visit)
    { }
};





// function definitions
double dotProduct(RVec const& x, RVec const& y)
{
    auto xit = x.cbegin(), xend = x.cend(), yit = y.cbegin(), yend = y.cend();
    double result = 0;
    for (; xit < xend && yit < yend; ++xit, ++yit)
        result += (*xit) * (*yit);
    return result;
}




// Neuron function definitions
Neuron::Neuron(size_t numInputs, IActivationFunction::CPtr const& v_af) :
    activationFunction(v_af),
    weights(numInputs, 0.0),
    inputConnections(numInputs, nullptr),
    outputConnections(),
    deriv(0.0),
    accumulatedWeightAdjustment(numInputs, 0.0)
{ }




// Connection function definitions
Connection::Connection(
    NeuronWPtr const& v_incomingFromNeuron,
    NeuronWPtr const& v_outgoingToNeuron,
    NeuronInputIndex v_outgoingToNeuronInputIndex) :
    incomingFromNeuron(v_incomingFromNeuron),
    outgoingToNeuron(v_outgoingToNeuron),
    outgoingToNeuronInputIndex(v_outgoingToNeuronInputIndex),
    output(0.0),
    errorSensitivityToOutput(0.0)
{ }









// NetworkImpl function definitions
NetworkImpl::NetworkImpl():
    m_neurons(),
    m_neuronConnections(),
    m_inputConnections(),
    m_outputConnections(),
    m_inputToOutputNeuronOrdering(),
    m_isPreProcessed(false)
{

}
NeuronIndex NetworkImpl::addNeuron(IActivationFunction::CPtr const& activationFunction, size_t numInputs) {
    m_isPreProcessed = false;
    m_neurons.emplace_back(new Neuron(numInputs, activationFunction));
    return NeuronIndex{m_neurons.size() - 1};
}

ConnectionIndex NetworkImpl::connectNeurons(
    NeuronIndex fromNeuron,
    NeuronIndex toNeuron,
    NeuronInputIndex toNeuronInput
)
{
    m_isPreProcessed = false;
    auto const& fromNeuronPtr = safeGetNeuron(fromNeuron);
    auto const& toNeuronPtr = safeGetNeuron(toNeuron);
    validateNeuronInputIndex(*toNeuronPtr, toNeuron, toNeuronInput);
    if (toNeuronPtr->inputConnections[toNeuronInput.v] != nullptr)
        throw std::runtime_error(
            "Neuron " + std::to_string(toNeuron.v) + " already has input at idx "
            + std::to_string(toNeuronInput.v)
        );
    auto connection = std::make_shared<Connection>(
        fromNeuronPtr,
        toNeuronPtr,
        toNeuronInput
    );
    fromNeuronPtr->outputConnections.push_back(connection);
    m_neuronConnections.push_back(connection);
    toNeuronPtr->inputConnections[toNeuronInput.v] = connection;
    m_neuronConnections.push_back(connection);
    return ConnectionIndex{m_neuronConnections.size() - 1};
}


NetworkInputIndex NetworkImpl::addInput()
{
    m_isPreProcessed = false;
    m_inputConnections.emplace_back();
    return {m_inputConnections.size() - 1};
}


NetworkOutputIndex NetworkImpl::addOutput(NeuronIndex fromNeuron)
{
    m_isPreProcessed = false;
    auto const& fromNeuronPtr = safeGetNeuron(fromNeuron);
    m_outputConnections.emplace_back(new Connection(fromNeuronPtr, {}, {0}));
    fromNeuronPtr->outputConnections.push_back(m_outputConnections.back());
    return {m_outputConnections.size() - 1};
}



void NetworkImpl::connectInputToNeuron(
    NetworkInputIndex fromNetworkInput,
    NeuronIndex toNeuron,
    NeuronInputIndex toNeuronInput)
{
    m_isPreProcessed = false;
    auto const& toNeuronPtr = safeGetNeuron(toNeuron);
    validateNeuronInputIndex(*toNeuronPtr, toNeuron, toNeuronInput);
    validateNetworkInput(fromNetworkInput);
    m_inputConnections[fromNetworkInput.v].emplace_back(new Connection({}, toNeuronPtr, toNeuronInput));
    toNeuronPtr->inputConnections[toNeuronInput.v] = m_inputConnections[fromNetworkInput.v].back();
}

std::optional<std::string> NetworkImpl::getValidationError() const
{
    // collect set of registered neurons
    std::unordered_set<Neuron const*> registered_neurons;
    for (const auto& n: m_neurons) registered_neurons.insert(n.get());

    // collect set of registered connections
    std::unordered_set<Connection const*> registered_connections;
    for (const auto& c: m_neuronConnections) registered_connections.insert(c.get());
    for (const auto& cvec: m_inputConnections)
        for (const auto& c: cvec)
            registered_connections.insert(c.get());
    for (const auto& c: m_outputConnections) registered_connections.insert(c.get());

    // we must have some neurons
    if (m_neurons.empty())
        return "No neurons, nothing to train";
    // check neurons
    for (size_t in = 0, nn = m_neurons.size(); in < nn; ++in)
    {
        auto const& nptr = m_neurons[in];
        auto const n_name = "m_neurons[" + s(in) + "]";
        // neuron must not be null
        if (!nptr) return "Null neuron at " + n_name;
        // num weights must match num inputs
        if (nptr->weights.size() != nptr->inputConnections.size())
            return n_name + " has inconsistent number of weights and input connections";

        // neuron must have input connections
        if (nptr->inputConnections.empty())
            return n_name + " has no input connections";
        // check neuron input connections
        for (size_t iic = 0, nic = nptr->inputConnections.size(); iic < nic; ++iic)
        {
            auto const& cptr = nptr->inputConnections[iic];
            auto c_name = n_name + "->inputConnections[" + s(iic) + ']';
            // neuron input connection must not be null
            if (!cptr)
                return "Null connection at " + c_name;
            // neuron input connection must be registered
            if (registered_connections.count(cptr.get()) == 0)
                return "Unregistered connection at " + c_name;
            
            // neuron input connection must point back to neuron
            if (cptr->outgoingToNeuron.lock() != nptr)
                return c_name + "->outgoingToNeuron does not point back to " + n_name;
            if (cptr->outgoingToNeuronInputIndex.v != iic)
                return c_name + "->outgoingToNeuronInputIndex is incorrect";
        }

        // neuron must have output connections
        if (nptr->outputConnections.empty())
            return n_name + "has no output connections";
        // check all neuron output connections
        for (size_t ioc = 0, noc = nptr->outputConnections.size(); ioc < noc; ++ioc)
        {
            auto const& cptr = nptr->outputConnections[ioc];
            auto c_name = n_name + "->outputConnections[" + s(ioc) + ']';
            // neuron output connection must not be null
            if (!cptr) return "Null connection at " + c_name;
            // neuron output connection must be registered
            if (registered_connections.count(cptr.get()) == 0)
                return "Unregistered connection at " + c_name;
            
            // neuron output connection must point back to neuron
            if (cptr->incomingFromNeuron.lock() != nptr)
                return c_name + "->incomingFromNeuron does not point back " + n_name;
        }
    }
    
    // check neuron to neuron connections
    for (size_t ic = 0; ic < m_neuronConnections.size(); ++ic)
    {
        auto const& cptr = m_neuronConnections[ic];
        auto const c_name = "m_neuronConnections[" + s(ic) + "]";
        // connection must not be null
        if (!cptr) return "Null connection at " + c_name;
        
        // check incoming neuron
        auto const cinptr = cptr->incomingFromNeuron.lock();
        auto cin_name = c_name + "->incomingFromNeuron";
        // incoming neuron must not be null
        if (!cinptr) return "Null neuron at " + cin_name;
        // incoming neuron must be registered
        if (registered_neurons.count(cinptr.get()) == 0)
            return "Unregistered neuron at " + cin_name;
        // incoming neuron must pointback to connection
        bool points_back = false;
        for (const auto& cinoc: cinptr->outputConnections)
            if (cinoc == cptr)
            {
                points_back = true;
                break;
            }
        if (!points_back)
            return cin_name + " does not point back to " + c_name;

        // check outgoing neuron
        auto const& conptr = cptr->outgoingToNeuron.lock();
        auto con_name = c_name + "->outgingToNeuron";
        auto con_idx = cptr->outgoingToNeuronInputIndex.v;
        // outgoing neuron must not be null
        if (!conptr) return "Null neuron at " + con_name;
        // outgoing neuron must be registered
        if (registered_neurons.count(conptr.get()) == 0)
            return "Unregistered neuron at " + con_name;
        // outgoing index must be in range
        if (con_idx >= conptr->inputConnections.size())
            return c_name + "->outgointToNeuronIndex does not fit into " 
            + con_name + "->inputConnections";
        // outgoing neuron must point back to connection
        if (conptr->inputConnections[con_idx] != cptr)
            return con_name + " does not point back to " + c_name;
    }

    // must have some inputs
    if (m_inputConnections.empty())
        return "No input connections, nothing to train";
    // check inputs
    for (size_t ii = 0, ni = m_inputConnections.size(); ii < ni; ++ii)
    {
        // input connection must connect to some neuron
        if (m_inputConnections[ii].empty())
            return "Input " + s(ii) + " is not connected to any neurons";
        for (size_t iic = 0, nic = m_inputConnections[ii].size(); iic < nic; ++iic)
        {
            // check connection to input
            const auto& cptr = m_inputConnections[ii][iic];
            auto c_name = "m_inputConnections[" + s(ii) + "][" + s(iic) + ']';
            // connection must not be null
            if (!cptr) return "Null connection at " + c_name;

            // connection must not have any incoming neuron
            if (cptr->incomingFromNeuron.lock())
                return c_name + " must not have input neuron.";
            
            // check connection output
            auto const& conptr = cptr->outgoingToNeuron.lock();
            auto const con_name = c_name + "->outgingNeuron" ;
            // output neuron should not be null
            if (!conptr) return "Null neuron at " + con_name;
            // output neuron must be registered
            if (registered_neurons.count(conptr.get()) == 0)
                return "Unregistered neuron at " + con_name;
            // output neuronInputIndex should be valid
            if (cptr->outgoingToNeuronInputIndex.v >= conptr->inputConnections.size())
                return c_name + "->outgoingToNeuronInputIndex is invalid";
            // connection output neuron should have the same connection as incoming
            if (conptr->inputConnections[cptr->outgoingToNeuronInputIndex.v] != cptr)
                return con_name + " is not pointing back to " + c_name;
        }
    }

    // must have some output neurons
    if (m_outputConnections.empty())
        return "No output connections, nothing to train";
    // check all output connections
    for (size_t ioc = 0, noc = m_outputConnections.size(); ioc < noc; ++ioc)
    {
        auto const& ocptr = m_outputConnections[ioc];
        auto oc_name = "m_outputConnections[" + s(ioc) + ']';
        // check that output connection is not null
        if (!ocptr) return "Null connection at " + oc_name;

        // check incoming neuron to output connection
        auto const& ocinptr = ocptr->incomingFromNeuron.lock();
        auto const ocin_name = oc_name + "->incomingFromNeuron";
        // check incoming neuron is not null
        if (!ocinptr) return "Null neuron at " + ocin_name;
        // incoming neuron must be registered
        if (registered_neurons.count(ocinptr.get()) == 0)
            return "Unregistered neuron at " + ocin_name;
        // check incoming neuron has connection as output
        bool has_as_output = false;
        for (const auto& ocinoc: ocinptr->outputConnections)
            if (ocinoc == ocptr)
            {
                has_as_output = true;
                break;
            }
        if (!has_as_output)
            return ocin_name + " does not have " + oc_name + " as output connection";

        // check that outgoing neuron is null
        if (ocptr->outgoingToNeuron.lock())
            return oc_name + " must not have output neuron";
    }

    return {};
}

double NetworkImpl::propagateExamples(const std::vector<Example>& examples, double stepSize)
{
    preprocess();
    double totalError = 0;

    // clear accumulated weight adjustments
    for (auto& neuron: m_neurons)
        neuron->accumulatedWeightAdjustment = RVec(neuron->accumulatedWeightAdjustment.size(), 0.0);

    // for each example
    for (size_t iex = 0, nex = examples.size(); iex < nex; ++iex)
    {
        const auto& example = examples[iex];

        // verifications
        if (example.inputs.size() != m_inputConnections.size())
            throw std::runtime_error(
                "Invalid number of inputs in example " + s(iex) + " (expected: "
                + s(m_inputConnections.size()) + ", found: " + s(example.inputs.size())
            );
        if (example.outputs.size() != m_outputConnections.size())
            throw std::runtime_error(
                "Invalid number of outputs in example " + s(iex) + " (expected: "
                + s(m_outputConnections.size()) + ", found: " + s(example.outputs.size())
            );

        // set inputs
        for (size_t ii = 0, ni = example.inputs.size(); ii < ni; ++ii)
        {
            for (auto& conn: m_inputConnections[ii])
                conn->output = example.inputs[ii];
        }

        // forward propagate
        for (auto& neuron: m_inputToOutputNeuronOrdering)
        {
            RVec inputs;
            inputs.reserve(neuron->inputConnections.size());
            for (auto const& conn: neuron->inputConnections)
                inputs.push_back(conn->output);
            double weightedSum = dotProduct(inputs, neuron->weights);
            double output = neuron->activationFunction->compute(weightedSum);
            neuron->deriv = neuron->activationFunction->derivative(weightedSum);
            for (auto& conn: neuron->outputConnections)
                conn->output = output;
        }

        // compute error, and sensitivity of error to outputs
        for (size_t io = 0, no = m_outputConnections.size(); io < no; ++io)
        {
            auto& conn = m_outputConnections[io];
            double diff = conn->output - example.outputs[io];
            totalError += diff*diff;
            conn->errorSensitivityToOutput = 2*diff;
        }

        // backward propagate
        for (size_t in_plus_one = m_inputToOutputNeuronOrdering.size(); in_plus_one > 0; --in_plus_one)
        {
            auto& neuron = m_inputToOutputNeuronOrdering[in_plus_one - 1];
            double errorSensitivityToNeuronOutput = 0;
            for (auto const& conn: neuron->outputConnections)
                errorSensitivityToNeuronOutput += conn->errorSensitivityToOutput;
            for (size_t iic = 0, nic = neuron->inputConnections.size(); iic < nic; ++iic)
            {
                auto& input = neuron->inputConnections[iic];
                input->errorSensitivityToOutput = errorSensitivityToNeuronOutput * neuron->deriv * neuron->weights[iic];
                neuron->accumulatedWeightAdjustment[iic] += errorSensitivityToNeuronOutput * neuron->deriv * input->output;
            }
        }
    }

    // adjust weights
    for (auto& n: m_neurons)
        for (size_t iic = 0, nic = n->inputConnections.size(); iic < nic; ++iic)
            n->weights[iic] -= stepSize * n->accumulatedWeightAdjustment[iic];

    return totalError;

}


RVec NetworkImpl::evaluate(RVec const& input)
{
    preprocess();

    // verify input size
    if (input.size() != m_inputConnections.size())
        throw std::runtime_error("Invalid number of inputs, expected: " + s(m_inputConnections.size())
            + ", found: " + s(input.size()));
    
    // set inputs
    for (size_t ii = 0, ni = input.size(); ii < ni; ++ii)
        for (auto const& ic: m_inputConnections[ii])
            ic->output = input[ii];
    
    // forward propagate
    for (auto const& n: m_inputToOutputNeuronOrdering)
    {
        double wsi = 0;
        for (size_t inic = 0, nnic = n->inputConnections.size(); inic < nnic; ++inic)
            wsi += n->inputConnections[inic]->output * n->weights[inic];
        double value = n->activationFunction->compute(wsi);
        for (auto const& oc: n->outputConnections)
            oc->output = value;
    }

    // return output
    RVec result;
    result.reserve(m_outputConnections.size());
    for (auto const& oc: m_outputConnections)
        result.push_back(oc->output);
    return result;
}


NeuronPtr const& NetworkImpl::safeGetNeuron(NeuronIndex idx) const
{
    if (idx.v >= m_neurons.size())
        throw std::runtime_error("Invalid neuron index " + s(idx.v));
    if (!m_neurons[idx.v])
        throw std::runtime_error("Null neuron at index " + s(idx.v));
    return m_neurons[idx.v];
}


void NetworkImpl::validateNeuronInputIndex(Neuron const& neuron, NeuronIndex nidx, NeuronInputIndex iidx) const
{
    if (iidx.v >= neuron.inputConnections.size())
        throw std::runtime_error(
            "Invalid neuron input index " + s(iidx.v) + " while accessing neuron "
            + s(nidx.v)
        );
}

void NetworkImpl::validateNetworkInput(NetworkInputIndex idx) const
{
    if (idx.v >= m_inputConnections.size())
        throw std::runtime_error("Invalid network input index " + s(idx.v));
}

void NetworkImpl::preprocess()
{
    if (m_isPreProcessed) return;
    auto validationError = getValidationError();
    if (validationError.has_value())
        throw std::runtime_error("Invalid network: " + *validationError);
    std::unordered_set<Neuron *> visited_neurons;

    m_isPreProcessed = true;
    std::vector<NeuronVisitingState> stack;
    stack.reserve(m_neurons.size());
    for (auto const& conn: m_outputConnections)
    {
        auto nptr = conn->incomingFromNeuron.lock();
        if (!nptr || visited_neurons.count(nptr.get()) > 0)
            continue;
        visited_neurons.insert(nptr.get());
        stack.emplace_back(nptr, 0);
    }

    while(!stack.empty())
    {
        auto& top = stack.back();
        if (top.next_input_to_visit >= top.nptr->inputConnections.size())
        {
            m_inputToOutputNeuronOrdering.push_back(top.nptr);
            stack.pop_back();
            continue;
        }
        auto next_ptr = top.nptr->inputConnections[top.next_input_to_visit]->incomingFromNeuron.lock();
        ++(top.next_input_to_visit);
        if (!next_ptr || visited_neurons.count(next_ptr.get()) > 0)
            continue;
        visited_neurons.insert(next_ptr.get());
        stack.emplace_back(next_ptr, 0);
    }
}


} // end anonymous namespace





// header function implementations
namespace mnist_deep_ann
{

    INetwork::Ptr INetwork::create() {
        return std::make_shared<NetworkImpl>();
    }

} // end namespace mnist_deep_ann

