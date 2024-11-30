#include "ann_interface.h"

namespace mnist_deep_ann {

    class ANN_EXPORT SimpleActivation: public IActivationFunction
    {
        public:
        double compute(double weightedSumOfInputs) const override;
        double derivative(double weightedSumOfInputs) const override;
        static CPtr create();
    };

} // end namespace mnist_deep_ann