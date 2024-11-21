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

class EdgeDetector {
    private:
    enum class PixelType { LOW, MEDIUM, HIGH };

    public:

    EdgeDetector(size_t rows, size_t cols):
        m_rows(rows),
        m_cols(cols),
        m_normalized(rows, std::vector<PixelType>(cols, PixelType::LOW)),
        m_rough(rows, std::vector<PixelType>(cols, PixelType::LOW)),
        m_dfsGround(rows, std::vector<bool>(cols, false))
        { }

    double isEdge(RVec const& pixels);

    private:
    size_t m_rows, m_cols;
    std::vector<std::vector<PixelType> > m_normalized, m_rough;
    std::vector<std::vector<bool> > m_dfsGround;
    size_t count(PixelType pt, size_t& x, size_t& y) const;
    void dfs(PixelType pt, size_t x, size_t y);
    bool isContiguous(PixelType pt, size_t seed_x, size_t seed_y);
    bool coversEnoughGround(PixelType pt);
};

double EdgeDetector::isEdge(RVec const& pixels)
{
    // check size
    if (pixels.size() != m_rows * m_cols) return 0.0;
    
    // normalize to LOW-MEDIUM-HIGH
    for (size_t r = 0; r < m_rows; ++r)
        for (size_t c = 0; c < m_cols; ++c)
        {
            double p = pixels[r*m_cols + c];
            if (p < .333)
                m_normalized[r][c] = PixelType::LOW;
            else if (p < .667)
                m_normalized[r][c] = PixelType::MEDIUM;
            else
                m_normalized[r][c] = PixelType::HIGH;

        }
    
    
    size_t low_x = 0, low_y = 0, high_x = 0, high_y = 0;
    size_t medium_count = count(PixelType::MEDIUM, low_x, low_y);
    size_t low_count = count(PixelType::LOW, low_x, low_y);
    size_t high_count = count(PixelType::HIGH, high_x, high_y);
    if (medium_count > 0)
    {
        if (low_count == 0 || high_count == 0)
            return 0.0;
        // fill MEDIUM with closest non-medium value
        m_rough = m_normalized;
        for (size_t r = 0; r < m_rows; ++r)
        {
            for (size_t c = 0; c < m_cols; ++c)
            {
                if (m_normalized[r][c] != PixelType::MEDIUM)
                    continue;
                size_t closest_distance = m_rows * m_rows + m_cols * m_cols + 1;
                PixelType closest_filled_pixel = PixelType::LOW;
                for (size_t r2 = 0; r2 < m_rows; ++r2)
                {
                    for (size_t c2 = 0; c2 < m_cols; ++c2)
                    {
                        if (m_rough[r2][c2] == PixelType::MEDIUM)
                            continue;
                        size_t distance = (r-r2)*(r-r2) + (c-c2)*(c-c2);
                        if (distance < closest_distance)
                        {
                            closest_distance = distance;
                            closest_filled_pixel = m_rough[r2][c2];
                        }
                    }
                }
                m_normalized[r][c] = closest_filled_pixel;
            }
        }
        low_count = count(PixelType::LOW, low_x, low_y);
        high_count = count(PixelType::HIGH, high_x, high_y);
    }
    double threshold = m_rows * m_cols * (0.5 - 1.0/(m_rows + m_cols));
    if (low_count > threshold      // at lest 
        && high_count > threshold 
        && isContiguous(PixelType::LOW, low_x, low_y)
        && isContiguous(PixelType::HIGH, high_x, high_y)
    )
        return 1.0;
    else
        return 0.0;
}

bool EdgeDetector::isContiguous(PixelType pt, size_t seed_x, size_t seed_y)
{
    // reset dfs ground
    for (auto& row: m_dfsGround)
        for (auto it = row.begin(); it < row.end(); ++it)
            *it = false;
    

    // visit all connected pixels that match pt
    dfs(pt, seed_x, seed_y);
    for (size_t irow = 0; irow < m_rows; ++irow)
        for (size_t icol = 0; icol < m_cols; ++icol)
            if (m_normalized[irow][icol] == pt && !m_dfsGround[irow][icol]) 
            // found pt pixel that was not visited, 
            // and hence, was disconnected from seed
            {
                return false;
            }

    return true;
        
}

size_t EdgeDetector::count(PixelType pt, size_t& x, size_t& y) const
{
    size_t res = 0;
    for (size_t irow = 0; irow < m_rows; ++irow)
    {
        for (size_t icol = 0; icol < m_cols; ++icol)
        {
            if (m_normalized[irow][icol] == pt)
            {
                ++res;
                x = irow;
                y = icol;
            }
        }
    }
    return res;
}


void EdgeDetector::dfs(PixelType pt, size_t seed_x, size_t seed_y)
{
    if (m_dfsGround[seed_x][seed_y] || m_normalized[seed_x][seed_y] != pt)
        return;
    m_dfsGround[seed_x][seed_y] = true;
    if (seed_x > 0) dfs(pt,seed_x - 1, seed_y);
    if (seed_y > 0) dfs(pt,seed_x, seed_y - 1);
    if (seed_y + 1 < m_cols) dfs(pt,seed_x, seed_y + 1);
    if (seed_x + 1 < m_rows) dfs(pt,seed_x + 1, seed_y);
}



void edgeDetectorTest()
{
    std::vector<std::pair<RVec, double> > examples {
        {
            {
                1.0, 1.0, 1.0,
                1.0, 0.0, 0.0,
                1.0, 0.0, 0.0
            }, 
            1.0
        },
        {
            {
                1.0, 0.0, 1.0,
                1.0, 0.0, 1.0,
                0.0, 0.0, 0.0
            },
            0.0
        },
        {
            {
                0.0, 0.0, 0.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0
            },
            1.0
        },
        {
            {
                0.5, 0.0, 0.0,
                0.0, 0.0, 1.0,
                1.0, 1.0, 1.0
            },
            1.0
        },
        {
            {
                1.0, 0.0, 1.0,
                0.0, 1.0, 0.0,
                1.0, 0.0, 1.0
            },
            0.0
        },
        {
            {
                1.0, 1.0, 1.0,
                1.0, 1.0, 0.0,
                1.0, 1.0, 1.0
            },
            0.0
        },
        {
            {
                1.0, 1.0, 0.5,
                0.5, 0.5, 0.5,
                0.5, 0.0, 0.0
            },
            1.0
        }
    };
    EdgeDetector ed(3, 3);
    for (auto const& example: examples)
    {
        auto const& input = example.first;
        double expectedOutput = example.second;
        double actualOutput = ed.isEdge(input);
        ASSERT(isAbsolutelyClose(expectedOutput, actualOutput, 1e-5));
    }
}



void edgeDetectorNeuralNetworkTest()
{
    size_t const GRID_ROWS = 5;
    size_t const GRID_COLS = 5;
    size_t const GRID_SIZE = GRID_ROWS * GRID_COLS;
    size_t const NUM_LAYERS = 10;
    size_t const NUM_RANDOM_TRAINING_EXAMPLES = GRID_SIZE*3;
    size_t const SEED = 3478;
    size_t const NUM_ITERATIONS = 5;
    auto func = SimpleActivation::create();
    auto network = INetwork::create();
    std::vector<NetworkInputIndex> networkInputs;
    for (size_t i = 0; i < GRID_SIZE; ++i)
        networkInputs.push_back(network->addInput());
    std::vector<std::vector<NeuronIndex> > neurons;
    for (size_t ilayer = 0; ilayer < NUM_LAYERS; ++ilayer)
    {
        neurons.emplace_back();
        auto & layer = neurons.back();
        layer.reserve(GRID_SIZE);
        for (size_t ineuron = 0; ineuron < GRID_SIZE; ++ineuron)
        {
            auto neuron = network->addNeuron(func, GRID_SIZE);
            layer.push_back(neuron);
            if (ilayer == 0)
            {
                for (size_t iinput = 0; iinput < networkInputs.size(); ++iinput)
                    network->connectInputToNeuron(networkInputs[iinput], neuron, {iinput});
            }
            else
            {
                auto const& previousLayer = neurons[neurons.size() - 2];
                for (size_t ipn = 0; ipn < previousLayer.size(); ++ipn)
                    network->connectNeurons(previousLayer[ipn], neuron, {ipn});
            }
        }
    }
    auto const& lastLayer = neurons.back();
    auto outputNeuron = network->addNeuron(func, lastLayer.size());
    for (size_t iinput = 0; iinput < lastLayer.size(); ++iinput)
        network->connectNeurons(lastLayer[iinput], outputNeuron, {iinput});
    network->addOutput(outputNeuron);

    auto pictureToInputs = [&network, GRID_ROWS, GRID_COLS](std::vector<std::vector<double> > const& picture) -> RVec {
        ASSERT(picture.size() == GRID_ROWS);
        RVec result;
        result.reserve(GRID_COLS * GRID_ROWS);
        for (auto const& row: picture)
        {
            ASSERT(row.size() == GRID_COLS);
            result.insert(result.end(), row.cbegin(), row.cend());
        }
        return result;
    };

    auto createElemental = [GRID_ROWS, GRID_COLS](std::vector<std::vector<double> > & picture, size_t r0, size_t c0, size_t r1, size_t c1) -> void {
        for (size_t r = 0; r < GRID_ROWS; ++r)
        {
            for (size_t c = 0; c < GRID_COLS; ++c)
            {
                size_t dist0 = (r0-r)*(r0-r) + (c0-c)*(c0-c);
                size_t dist1 = (r1-r)*(r1-r) + (c1-c)*(c1-c);
                picture[r][c] = ((dist0 < dist1) ? 0.0 : 1.0);
            }
        }
    };

    // elemental examples
    std::vector<Example> examples;
    std::vector<RVec> elementals;
    std::vector<std::vector<double> > picture (GRID_ROWS, std::vector<double>(GRID_COLS, 0.0));
    for (size_t seed_r = 0; seed_r < GRID_ROWS; ++seed_r)
    {
        createElemental(picture, seed_r, 0, GRID_ROWS - seed_r - 1, GRID_COLS - 1);
        elementals.push_back(pictureToInputs(picture));
        examples.push_back({elementals.back(), RVec(1, 1.0)});

        createElemental(picture, seed_r, GRID_COLS - 1, GRID_ROWS - seed_r - 1, 0);
        elementals.push_back(pictureToInputs(picture));
        examples.push_back({elementals.back(), RVec(1, 1.0)});
    }

    for (size_t seed_c = 0; seed_c < GRID_COLS; ++seed_c)
    {
        createElemental(picture, 0, seed_c, GRID_ROWS - 1, GRID_COLS - seed_c - 1);
        elementals.push_back(pictureToInputs(picture));
        examples.push_back({elementals.back(), RVec(1, 1.0)});

        createElemental(picture, GRID_ROWS - 1, seed_c, 0, GRID_COLS - seed_c - 1);
        elementals.push_back(pictureToInputs(picture));
        examples.push_back({elementals.back(), RVec(1, 1.0)});
    }

    // all 1-s and all 0-s not allowed
    for (size_t r = 0; r < GRID_ROWS; ++r) for (size_t c = 0; c < GRID_COLS; ++c) picture[r][c] = 0;
    examples.push_back({pictureToInputs(picture), RVec(1, 0.0)});
    for (size_t r = 0; r < GRID_ROWS; ++r) for (size_t c = 0; c < GRID_COLS; ++c) picture[r][c] = 1;
    examples.push_back({pictureToInputs(picture), RVec(1, 0.0)});

    // some random examples
    std::default_random_engine gen(SEED);
    std::uniform_real_distribution<double> sampler(0.001, 0.999);
    EdgeDetector ed(GRID_ROWS, GRID_COLS);
    RVec flat_inputs(GRID_SIZE, 0.0);
    for (size_t rex = 0; rex < NUM_RANDOM_TRAINING_EXAMPLES; ++rex)
    {
        for (size_t i = 0; i < flat_inputs.size(); ++i) flat_inputs[i] = sampler(gen);
        examples.push_back({flat_inputs, RVec(1, ed.isEdge(flat_inputs))});
    }

    // some perturbations on elementals
    std::vector<double> const SHOCK_SIZE_SCALE_FACTORS{0.01, 0.25, 0.5, 0.75, 0.99};
    for (double SSSF: SHOCK_SIZE_SCALE_FACTORS)
    {
        for (auto const& elem: elementals)
        {
            flat_inputs = elem;
            for (auto& pixel: flat_inputs) pixel += (sampler(gen) - 0.5) * SSSF;
            examples.push_back({flat_inputs, RVec(1, ed.isEdge(flat_inputs))});
        }
    }

    // for (auto const& ex: examples)
    // {
    //     for (size_t r = 0; r < GRID_ROWS; ++r)
    //     {
    //         for (size_t c = 0; c < GRID_COLS; ++c)
    //         {
    //             double p = ex.inputs[r * GRID_COLS + c];
    //             if (p < .333333)
    //                 std::cout << "  ";
    //             else if (p >= .666666)
    //                 std::cout << "**";
    //             else
    //                 std::cout << "--";
    //         }
    //         std::cout << std::endl;
    //     }
    //     std::cout << ex.outputs[0] << "\n----------" << std::endl;
    // }

    // double err;
    for (size_t i = 0; i < NUM_ITERATIONS; ++i)
    {
        // std::cout << "========== iteration " << i << std::endl;
        for (const auto& e: examples)
        {
            // err =
            network->propagateExamples({e}, 1000);
            // std::cout << err << "\n";
        }
    }
            

}

} // end namespace mnist_deep_ann






int main()
{
    using namespace mnist_deep_ann;
    runTest("simpleActivationTest", simpleActivationTest);
    runTest("singleNeuronTest", singleNeuronTest);
    runTest("edgeDetectorTest", edgeDetectorTest);
    runTest("edgeDetectorNeuralNetworkTest", edgeDetectorNeuralNetworkTest);
    std::cout << "[INFO] ann_test finished." << std::endl;
    return 0;
}