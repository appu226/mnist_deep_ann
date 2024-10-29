#include "simple_activation.h"


namespace mnist_deep_ann {

    double SimpleActivation::compute(double x) const
    {
        if (x < 0)
            return x/(1-x);
        else
            return x/(1+x);
    }

    double SimpleActivation::derivative(double x) const
    {
        if (x < 0)
            return 1/(1-x)/(1-x);
        else
            return 1/(1+x)/(1+x);
    }

    SimpleActivation::CPtr SimpleActivation::create()
    {
        return std::make_shared<SimpleActivation>();
    }

} // end namespace mnist_deep_ann