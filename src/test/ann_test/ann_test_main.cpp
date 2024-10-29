#include <iostream>
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


void simpleTest()
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

} // end namespace mnist_deep_ann






int main()
{
    using namespace mnist_deep_ann;
    runTest("simpleTest", simpleTest);
    std::cout << "[INFO] ann_test finished." << std::endl;
    return 0;
}