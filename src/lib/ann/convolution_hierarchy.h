#pragma once

#include "ann_interface.h"


namespace mnist_deep_ann {

    class ANN_EXPORT IConvolutionHierarchy {

    public :
        
        using Ptr = std::shared_ptr<IConvolutionHierarchy>;

        virtual ~IConvolutionHierarchy() {}

        // create a convolution hierarchy for parsing an image
        static Ptr create(size_t imageWidth, size_t imageHeight, IActivationFunction::CPtr const& activationFunction);


        // `addLayer`: Add a new layer of classifiers.
        // Each classifier works on a small window of pixels, and gives back a signal.
        //   The window is slid across the entire image, to get back a new smaller image of signals.
        //   This smaller image then becomes the base for the next layer.
        //   Note that each pixel is a vector of classifier signals 
        //     from all the classifiers of the previous layer.
        virtual void addLayer(size_t windowWidth, size_t windowHeight, size_t numClassifiers) = 0;

        // Add a final layer
        //   with `numClassifiers` output neurons.
        // The output neuron with the highest value is the classification.
        virtual void addFinalLayer(size_t numClassifiers) = 0;


        // Propagate forward and backward using a sequence of examples
        //   and return the sum of squared classification errors.
        // Each example is the row-major-flattened image as input,
        //   and 0-1 category vector (only the correct category is 1, all others are 0) as output.
        virtual double propagateExamples(std::vector<Example> const& examples, double stepSize) = 0;

        // Adjust all weights based on previously computed sensitivity to error
        virtual void adjust(double stepSize) = 0;

        // Use a trained vector to classify an example image
        virtual size_t categorize(RVec const& input) const = 0;

        // randomly perturb weights
        virtual void perturbWeights(double maxAbsShift) = 0;

        // print internal diagnostics
        virtual void diagnose(std::ostream& out) const = 0;

        // Evaluate examples and compute error
        virtual double getClassificationError(std::vector<Example> const& examples) const = 0;
    };


} // end namespace mnist_deep_ann