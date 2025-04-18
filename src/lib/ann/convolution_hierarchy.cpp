#include "convolution_hierarchy.h"

using namespace mnist_deep_ann;



namespace {

    struct LayerDimensions
    {
        size_t numClassifiers;
        size_t resultingImageWidth;
        size_t resultingImageHeight;
    };

    struct ClassificationLayer
    {
        ClassificationLayer(LayerDimensions v_dimensions) :
            classifierWeights(),
            resultingImage(),
            dimensions(std::move(v_dimensions))
        { }

        // OC x IC x WX x WY
        std::vector<std::vector<std::vector<std::vector<WeightIndex> > > > classifierWeights;

        // OC x ONX x ONY
        std::vector<std::vector<std::vector<NeuronIndex> > > resultingImage;
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

        ConvolutionHierarchyImpl::ConvolutionHierarchyImpl(
            size_t imageWidth,
            size_t imageHeight,
            IActivationFunction::CPtr const& activationFunction
        ) :
            m_imageWidth(imageWidth),
            m_imageHeight(imageHeight),
            m_activationFunction(activationFunction),
            m_network(INetwork::create()),
            m_layers(),
            m_image(),
            m_isFinalized(false)
        {
            m_image.reserve(imageWidth);
            for (size_t x = 0; x < imageWidth; ++x)
            {
                m_image.emplace_back();
                auto& imageRow = m_image.back();
                imageRow.reserve(imageHeight);
                for (size_t y = 0; y < imageHeight; ++y)
                {
                    imageRow.push_back(m_network->addInput());
                }
            }
        }


        void addLayer(size_t WX, size_t WY, size_t OC) override
        {
            // We wish to create a the layer with windows of size
            //    WX * WY
            // and OC output categories.
            ANN_ASSERT(OC > 0, "numCategories cannot be zero.");
            
            // Suppose the previous layer has:
            //    INX * INY input neurons
            //    IC categories
            size_t INX, INY, IC;
            if (m_layers.empty())
            {
                INX = m_imageWidth;
                INY = m_imageHeight;
                IC = 1;
            }
            else
            {
                auto const& lld = m_layers.back()->dimensions;
                INX = lld.resultingImageWidth;
                INY = lld.resultingImageHeight;
                IC = lld.numClassifiers;
            }

            // Then, we will have these many output neurons:
            //    OC * (INX-WX+1) * (INY-WY+1) 
            //  = OC *    ONX     *    ONY
            ANN_ASSERT(WX <= INX, "newWindowSizeX should be less than " << INX << " given: " << WX);
            ANN_ASSERT(WY <= INY, "WY should be less than " << INY << " given: " << WY);
            size_t ONX = INX - WX + 1;
            size_t ONY = INY - WY + 1;
            LayerDimensions ld{ OC, ONX, ONY };
            auto cl = std::make_shared<ClassificationLayer>(ld);
            auto& nrns = cl->resultingImage;
            nrns.resize(OC);
            for (size_t oc = 0; oc < OC; ++oc)
            {
                nrns[oc].resize(ONX);
                for (size_t onx = 0; onx < ONX; ++onx)
                {
                    nrns[oc][onx].reserve(ONY);
                    for (size_t ony = 0; ony < ONY; ++ony)
                    {
                        // The number of inputs for each output neuron will be:
                        //    IC * WX * WY
                        nrns[oc][onx].emplace_back(m_network->addNeuron(m_activationFunction, IC * WX * WY));
                    }
                }
            }

            // The number of weights will be:
            //    OC * IC * WX * WY
            auto& wts = cl->classifierWeights;
            wts.resize(OC);
            for (size_t oc = 0; oc < OC; ++oc)
            {
                wts[oc].resize(IC);
                for (size_t ic = 0; ic < IC; ++ic)
                {
                    wts[oc][ic].resize(WX);
                    for (size_t wx = 0; wx < WX; ++wx)
                    {
                        wts[oc][ic][wx].reserve(WY);
                        for (size_t wy = 0; wy < WY; ++wy)
                        {
                            wts[oc][ic][wx].emplace_back(m_network->addWeight());
                        }
                    }
                }
            }
            
            
            // The output neuron at this coordinate:
            //    (oc, onx, ony)
            // will use the input neurons at coordinate:
            //    (ic, inx+onx, iny+ony)
            // with weight
            //    (oc, ic, inx, iny)
            // such that:
            //    ic  \elem [0, IC)
            //    inx \elem [0, WX)
            //    iny \elem [0, WY)

            for (size_t oc = 0; oc < OC; ++oc)
            {
                for (size_t onx = 0; onx < ONX; ++onx)
                {
                    for (size_t ony = 0; ony < ONY; ++ony)
                    {
                        size_t inputIndex = 0;
                        for (size_t ic = 0; ic < IC; ++ic)
                        {
                            for (size_t inx = 0; inx < WX; ++inx)
                            {
                                for (size_t iny = 0; iny < WY; ++iny)
                                {
                                    m_network->connectWeightToNeuron(
                                        wts[oc][ic][inx][iny],
                                        nrns[oc][onx][ony],
                                        { inputIndex }
                                    );
                                    if (m_layers.empty())
                                    {
                                        m_network->connectInputToNeuron(
                                            m_image[inx+onx][iny+ony],
                                            nrns[oc][onx][ony],
                                            { inputIndex }
                                        );
                                    }
                                    else
                                    {
                                        m_network->connectNeurons(
                                            m_layers.back()->resultingImage[ic][inx+onx][iny+ony],
                                            nrns[oc][onx][ony],
                                            { inputIndex }
                                        );
                                    }
                                    ++inputIndex;
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
            for (size_t cat = 0; cat < numClassifiers; ++cat)
            {
                m_network->addOutput(m_layers.back()->resultingImage[cat][0][0]);
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


namespace mnist_deep_ann
{
    IConvolutionHierarchy::Ptr IConvolutionHierarchy::create(size_t imageWidth, size_t imageHeight, IActivationFunction::CPtr const& activationFunction)
    {
        return std::make_shared<ConvolutionHierarchyImpl>(imageWidth, imageHeight, activationFunction);
    }
}