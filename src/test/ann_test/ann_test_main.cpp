#include <iostream>
#include <random>
#include <stdexcept>


#include <ann/ann_interface.h>
#include <ann/simple_activation.h>


namespace mnist_deep_ann {

template<typename TestType>
void runTest(const std::string& testName, TestType& test)
{
    try {
        test();
        std::cout << "[INFO] test " << testName << ": OK" << std::endl;
    }
    catch (std::bad_alloc const&)
    {
        std::cout << "[CRASH] bad_alloc in test " << testName << std::endl;
        throw;
    }
    catch (std::exception const& e)
    {
        std::cout << "[ERROR] Exception in test " << testName << ": " << e.what() << std::endl;
    }
    catch (...)
    {
        std::cout << "[ERROR] Unexpected error in test " << testName << std::endl;
    }
}

bool isAbsolutelyClose(double x, double y, double eps)
{
    return x < y + eps && x > y - eps;
}

#define ASSERT(x) if (!(x)) { throw std::runtime_error(#x); }


void simpleActivationTest()
{
    auto func = SimpleActivation::create();
    double fx, dfdx;
    for (double x = -50; x < 50; x += .5)
    {
        if (x < 0)
            fx = x/(1-x);
        else
            fx = x/(1+x);
        ASSERT(isAbsolutelyClose(func->compute(x), fx, 1e-12));
        dfdx = (func->compute(x+1e-10) - func->compute(x-1e-10)) / 2e-10;
        ASSERT(isAbsolutelyClose(func->derivative(x), dfdx, 1e-5));
        ASSERT(-1 < fx && fx < 1);
        ASSERT(0 < dfdx && dfdx <= 1);
    }
}

void singleNeuronTest()
{
    auto const NUM_INPUTS = 5;
    auto const SEED = 38561;
    auto const NUM_ITERATIONS = 100;
    auto const NUM_TESTS = 30;

    auto network = INetwork::create();
    auto func = SimpleActivation::create();
    auto neuron = network->addNeuron(func, NUM_INPUTS);
    for (size_t i = 0; i < NUM_INPUTS; ++i)
    {
        auto input = network->addInput();
        network->connectInputToNeuron(input, neuron, {i});
    }
    network->addOutput(neuron);

    std::default_random_engine gen(SEED);
    std::uniform_real_distribution<double> sampler(0.1, 0.9);
    std::vector<double> expectedWeights;
    for (size_t i = 0; i < NUM_INPUTS; ++i)
        expectedWeights.push_back(sampler(gen));

    std::vector<Example> examples;
    for (size_t i = 0; i < NUM_INPUTS; ++i)
    {
        examples.emplace_back();
        auto & ex = examples.back();

        ex.inputs.resize(NUM_INPUTS, 0.0);
        ex.inputs[i] = 1.0;
        ex.outputs.push_back(func->compute(expectedWeights[i]));
    }


    ASSERT(!network->getValidationError().has_value());
    double training_error;
    for (size_t i = 0; i < NUM_ITERATIONS; ++i)
    {
        training_error = network->propagateExamples(examples, 1.0);
    }
    ASSERT(training_error < 1e-12);

    RVec test_input(NUM_INPUTS, 0.0);
    double wsi, expected_output;
    RVec actual_output;
    for (size_t itest = 0; itest < NUM_TESTS; ++itest)
    {
        wsi = 0;
        for (size_t i_input = 0; i_input < NUM_INPUTS; ++i_input)
        {
            test_input[i_input] = sampler(gen);
            wsi += test_input[i_input] * expectedWeights[i_input];
        }
        expected_output = func->compute(wsi);
        actual_output = network->evaluate(test_input);
        ASSERT(actual_output.size() == 1 && abs(actual_output[0] - expected_output) < 1e-12);
    }    

}

} // end namespace mnist_deep_ann






int main()
{
    using namespace mnist_deep_ann;
    runTest("simpleActivationTest", simpleActivationTest);
    runTest("singleNeuronTest", singleNeuronTest);
    std::cout << "[INFO] ann_test finished." << std::endl;
    return 0;
}