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

double sq(double x)
{
    return x * x;
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
        auto weight = network->addWeight();
        network->connectInputToNeuron(input, neuron, {i});
        network->connectWeightToNeuron(weight, neuron, {i});
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
        ASSERT(actual_output.size() == 1 && abs(actual_output[0] - expected_output) < 1e-8);
    }    

}


void multiNeuronTest()
{
    ////std::cout << "Creating network" << std::endl;
    //auto f = SimpleActivation::create();
    //auto network = INetwork::create();

    //auto i0 = network->addInput();
    //auto i1 = network->addInput();
    //auto n00 = network->addNeuron(f, 2);
    //network->connectInputToNeuron(i0, n00, { 0 });
    //network->connectInputToNeuron(i1, n00, { 1 });
    //
    //auto n01 = network->addNeuron(f, 2);
    //network->connectInputToNeuron(i0, n01, { 0 });
    //network->connectInputToNeuron(i1, n01, { 1 });

    //auto n10 = network->addNeuron(f, 2);
    //network->connectNeurons(n00, n10, { 0 });
    //network->connectNeurons(n01, n10, { 1 });
    //
    //auto n11 = network->addNeuron(f, 2);
    //network->connectNeurons(n00, n11, { 0 });
    //network->connectNeurons(n01, n11, { 1 });

    //auto o0 = network->addOutput(n10);
    //auto o1 = network->addOutput(n11);

    //network->perturbWeights(1.0);
    //auto diag = network->getDiagnostics();

    //std::vector<RVec> w;
    //{
    //    auto const& neurons = diag->neurons;
    //    w = {
    //        { neurons.at("Neuron_0").inputWeights[0], neurons.at("Neuron_0").inputWeights[1] },
    //        { neurons.at("Neuron_1").inputWeights[0], neurons.at("Neuron_1").inputWeights[1] },
    //        { neurons.at("Neuron_2").inputWeights[0], neurons.at("Neuron_2").inputWeights[1] },
    //        { neurons.at("Neuron_3").inputWeights[0], neurons.at("Neuron_3").inputWeights[1] }
    //    };
    //}

    ////std::cout << "Verifying outputs" << std::endl;
    //RVec inputs = { 0.123, 0.456 };
    //RVec outputs = network->evaluate(inputs);

    //auto n0_act = w[0][0] * inputs[0] + w[0][1] * inputs[1];
    //auto n0v = f->compute(n0_act);
    //auto n1_act = w[1][0] * inputs[0] + w[1][1] * inputs[1];
    //auto n1v = f->compute(n1_act);

    //auto n2_act = w[2][0] * n0v + w[2][1] * n1v;
    //auto n2v = f->compute(n2_act);
    //auto n3_act = w[3][0] * n0v + w[3][1] * n1v;
    //auto n3v = f->compute(n3_act);

    //double const tol = 1e-12;
    //ASSERT(isAbsolutelyClose(outputs[0], n2v, tol));
    //ASSERT(isAbsolutelyClose(outputs[1], n3v, tol));


    ////std::cout << "Verifying error sensitivity to layer 2 outputs" << std::endl;
    //Example e{ inputs, { outputs[0] + 1, outputs[1] + 1 }  };
    //network->propagateExamples({ e }, 0);
    //network->updateDiagnostics(*diag);
    //
    //std::vector<RVec> errorSensitivityToWeights;
    //RVec errorSensitivityToNeurons;
    //{
    //    auto const& neurons = diag->neurons;
    //    errorSensitivityToWeights = {
    //        neurons.at("Neuron_0").errorSensitivitiesToWeights,
    //        neurons.at("Neuron_1").errorSensitivitiesToWeights,
    //        neurons.at("Neuron_2").errorSensitivitiesToWeights,
    //        neurons.at("Neuron_3").errorSensitivitiesToWeights
    //    };
    //    errorSensitivityToNeurons = {
    //        neurons.at("Neuron_0").errorSensitivityToOutput,
    //        neurons.at("Neuron_1").errorSensitivityToOutput,
    //        neurons.at("Neuron_2").errorSensitivityToOutput,
    //        neurons.at("Neuron_3").errorSensitivityToOutput
    //    };
    //}

    //auto n2d = 2 * (n2v - e.outputs[0]);
    //ASSERT(isAbsolutelyClose(n2d, errorSensitivityToNeurons[2], tol));
    //auto n3d = 2 * (n2v - e.outputs[0]);
    //ASSERT(isAbsolutelyClose(n3d, errorSensitivityToNeurons[3], tol));
    ////std::cout << "Verifying error sensitivity to layer 2 weights" << std::endl;

    //auto n2fd = n2d * f->derivative(n2_act);
    //auto n2w0d = n2fd * n0v;
    //ASSERT(isAbsolutelyClose(n2w0d, errorSensitivityToWeights[2][0], tol));
    //auto n2w1d = n2fd * n1v;
    //ASSERT(isAbsolutelyClose(n2w1d, errorSensitivityToWeights[2][1], tol));

    //double const x_delta = 1e-8;
    //{
    //    auto new_n2w0 = w[2][0] + x_delta;
    //    auto new_n2_act = new_n2w0 * n0v + w[2][1] * n1v;
    //    auto new_n2v = f->compute(new_n2_act);
    //    auto new_error = (new_n2v - e.outputs[0]) * (new_n2v - e.outputs[0]);
    //    auto old_error = (outputs[0] - e.outputs[0]) * (outputs[0] - e.outputs[0]);
    //    auto y_delta = new_error - old_error;
    //    auto real_slope = y_delta / x_delta;
    //    auto relative_error_in_slope = (real_slope - n2w0d) / real_slope;
    //    ASSERT(abs(relative_error_in_slope) < 1e-4);
    //}

    ////std::cout << "Verifying error sensitivity to layer 1 outputs" << std::endl;

    //auto n2i0d = n2fd * w[2][0];
    //auto n3i0d = n3d * f->derivative(n3_act) * w[3][0];
    //auto n0d = n2i0d + n3i0d;
    //ASSERT(isAbsolutelyClose(n0d, errorSensitivityToNeurons[0], tol));
    //{
    //    auto new_n0v = n0v + x_delta;
    //    auto new_n2_act = new_n0v * w[2][0] + n1v * w[2][1];
    //    auto new_n2v = f->compute(new_n2_act);
    //    auto new_n3_act = new_n0v * w[3][0] + n1v * w[3][1];
    //    auto new_n3v = f->compute(new_n3_act);
    //    auto new_error = sq(new_n2v - e.outputs[0]) + sq(new_n3v - e.outputs[1]);
    //    auto old_error = sq(n2v - e.outputs[0]) + sq(n3v - e.outputs[1]);
    //    auto slope = (new_error - old_error) / x_delta;
    //    auto relative_error_in_slope = (slope - n0d) / slope;
    //    ASSERT(abs(relative_error_in_slope) < 1e-4);
    //}

    ////std::cout << "Verifying error sensitivity to layer 1 weight" << std::endl;

    //auto n0w0d = n0d * f->derivative(n0_act) * inputs[0];
    //ASSERT(isAbsolutelyClose(n0w0d, errorSensitivityToWeights[0][0], tol));
    //{
    //    auto new_n0w0v = w[0][0] + x_delta;
    //    auto new_n0_act = new_n0w0v * inputs[0] + w[0][1] * inputs[1];
    //    auto new_n0v = f->compute(new_n0_act);
    //    auto new_n2_act = new_n0v * w[2][0] + n1v * w[2][1];
    //    auto new_n2v = f->compute(new_n2_act);
    //    auto new_n3_act = new_n0v * w[3][0] + n1v * w[3][1];
    //    auto new_n3v = f->compute(new_n3_act);
    //    auto new_error = sq(new_n2v - e.outputs[0]) + sq(new_n3v - e.outputs[1]);
    //    auto old_error = sq(n2v - e.outputs[0]) + sq(n3v - e.outputs[1]);
    //    auto slope = (new_error - old_error) / x_delta;
    //    auto relative_error_in_slope = (slope - n0w0d) / slope;
    //    ASSERT(abs(relative_error_in_slope) < 1e-4);
    //}
    ////std::cout << "All verifications successful" << std::endl;

}

} // end namespace mnist_deep_ann






int main()
{
    using namespace mnist_deep_ann;
    runTest("simpleActivationTest", simpleActivationTest);
    runTest("singleNeuronTest", singleNeuronTest);
    runTest("multiNeuronTest", multiNeuronTest);
    std::cout << "[INFO] ann_test finished." << std::endl;
    return 0;
}
