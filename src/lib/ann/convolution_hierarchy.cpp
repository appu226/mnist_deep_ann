#include "convolution_hierarchy.h"

using namespace mnist_deep_ann;



namespace {

     
    struct Classifiers
    {
        std::vector<NeuronIndex> classifiers;
    };

    struct ClassifierWeights
    {
        std::vector<std::vector<WeightIndex> > weightWindow;
    };

    struct LayerDimensions
    {
        size_t numClassifiers;
        size_t resultingImageWidth;
        size_t resultingImageHeight;
        size_t slidingWindowWidth;
        size_t slidingWindowHeight;
    };

    struct ClassificationLayer
    {
        ClassificationLayer(LayerDimensions v_dimensions) :
            classifierWeights(),
            resultingImage(),
            dimensions(std::move(v_dimensions))
        {
            

        }
        std::vector<ClassifierWeights> classifierWeights;
        std::vector<std::vector<Classifiers> > resultingImage;
        LayerDimensions dimensions;
    };
    using ClassificationLayerPtr = std::shared_ptr<ClassificationLayer>;


    class ConvolutionHierarchyImpl : public IConvolutionHierarchy
    { 
    public:

        size_t m_imageWidth;
        size_t m_imageHeight;
        IActivationFunction::CPtr m_activationFunction;
        INetwork::Ptr m_network;
        std::vector<ClassificationLayerPtr> m_layers;
        std::vector<std::vector<NetworkInputIndex> > m_image;
        bool m_isFinalized;

        //TODO:
        // initialize m_image in constructor with m_imageWidth X m_imageHeight network inputs


        // Add a layer of 
        void addLayer(size_t newWindowSizeX, size_t newWindowSizeY, size_t numCategories) override
        {
            ANN_ASSERT(numCategories > 0, "numCategories cannot be zero.");
            
            // figure out input image dims
            size_t inputImageWidth, inputImageHeight, inputImageNumCategories;
            if (m_layers.empty())
            {
                inputImageWidth = m_imageWidth;
                inputImageHeight = m_imageHeight;
                inputImageNumCategories = 1;
            }
            else
            {
                auto const& lld = m_layers.back()->dimensions;
                inputImageWidth = lld.resultingImageWidth;
                inputImageHeight = lld.resultingImageHeight;
                inputImageNumCategories = lld.numClassifiers;
            }

            // get new layer dimensions
            ANN_ASSERT(newWindowSizeX <= inputImageWidth, "newWindowSizeX should be less than " << inputImageWidth << " given: " << newWindowSizeX);
            ANN_ASSERT(newWindowSizeY <= inputImageHeight, "newWindowSizeY should be less than " << inputImageHeight << " given: " << newWindowSizeY);
            size_t resultingImageWidth = inputImageWidth - newWindowSizeX + 1;
            size_t resultingImageHeight = inputImageHeight - newWindowSizeY + 1;
            LayerDimensions ld{ numCategories, resultingImageWidth, resultingImageHeight, newWindowSizeX, newWindowSizeY };
            
            // set layer weights
            auto cl = std::make_shared<ClassificationLayer>(ld);
            auto& wts = cl->classifierWeights;
            wts.reserve(inputImageNumCategories);
            for (size_t cat = 0; cat < inputImageNumCategories; ++cat)
            {
                wts.emplace_back();
                auto & cwts = wts.back();
                cwts.weightWindow.reserve(inputImageWidth);
                for (size_t x = 0; x < inputImageWidth; ++x)
                {
                    cwts.weightWindow.emplace_back();
                    auto& cwtsCol = cwts.weightWindow.back();
                    cwtsCol.reserve(inputImageHeight);
                    for (size_t y = 0; y < inputImageHeight; ++y)
                        cwtsCol.push_back(m_network->addWeight());
                }
            }

            // set layer neurons
            auto& neurons = cl->resultingImage;
            neurons.reserve(resultingImageWidth);
            size_t numNeuronInputs = inputImageWidth * inputImageHeight * inputImageNumCategories;
            for (size_t onx = 0; onx < resultingImageWidth; ++onx)
            {
                neurons.emplace_back();
                auto& neuronCol = neurons.back();
                neuronCol.reserve(resultingImageHeight);
                for (size_t ony = 0; ony < resultingImageHeight; ++ony)
                {
                    neuronCol.emplace_back();
                    auto& classifiers = neuronCol.back().classifiers;
                    classifiers.reserve(numCategories);
                    for (size_t ocat = 0; ocat < numCategories; ++ocat)
                    {
                        classifiers.push_back(m_network->addNeuron(m_activationFunction, numNeuronInputs));
                        auto neuron = classifiers.back();
                        size_t nin = 0;
                        for (size_t inx = 0; inx < inputImageWidth; ++inx)
                        {
                            for (size_t iny = 0; iny < inputImageHeight; ++iny)
                            {
                                for (size_t icat = 0; icat < inputImageNumCategories; ++icat)
                                {
                                    m_network->connectWeightToNeuron(cl->classifierWeights[icat].weightWindow[inx][iny], neuron, { nin });
                                    if (m_layers.empty())
                                    {
                                        m_network->connectInputToNeuron(m_image[inx][iny], neuron, { nin });
                                    }
                                    else
                                    {
                                        auto& pl = m_layers.back()->resultingImage;
                                        m_network->connectNeurons(pl[inx][iny].classifiers[icat], neuron, { nin });
                                    }
                                    ++nin;
                                }
                            }
                        }
                    }
                }
            }

            m_layers.push_back(cl);


        }


        void addFinalLayer(size_t numClassifiers) override
        {
            // TODO
            ANN_ASSERT(!m_isFinalized, "ConvolutionHierarchy already finalized.");
            if (m_layers.empty())
            {
                addLayer(m_imageWidth, m_imageHeight, numClassifiers);
            }
            else
            {
                auto const& ld = m_layers.back()->dimensions;
                addLayer(ld.resultingImageWidth, ld.resultingImageHeight, numClassifiers);
            }
            m_isFinalized = true;
        }



        double propagateExamples(std::vector<Example> const& examples, double stepSize) override
        {
            return m_network->propagateExamples(examples, stepSize);
        }



        size_t categorize(RVec const& input) const override
        {
            ANN_ASSERT(
                input.size() == m_imageWidth * m_imageHeight, 
                "Unexpected image size (" << input.size() << " px), expected: (" 
                    << m_imageWidth << " x " << m_imageHeight << ").");
            RVec resultVector = m_network->evaluate(input);
            ANN_ASSERT(resultVector.empty(), "Classifier not implemented for 0 categories.");
            size_t resultIdx = 0;
            for (size_t ir = 0; ir < resultVector.size(); ++ir)
            {
                if (resultVector[ir] > resultVector[resultIdx])
                    resultIdx = ir;
            }
            return resultIdx;
        }
    };


} // end anonymous namespace