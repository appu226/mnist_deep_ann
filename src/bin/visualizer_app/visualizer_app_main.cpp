#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "clo.hpp"
#include "mnist_data.h"




using namespace mnist_deep_ann;



namespace {
    struct Settings {
        std::string mnistDataPath;
        std::string mnistLabelsPath;
        
        using UPtr = std::unique_ptr<Settings>;
        using UCPtr = std::unique_ptr<Settings const>;
        static UCPtr parse(int argc, char const * const * const argv);
    }; // end struct Settings

} // end anonymous namespace






int main(int const argc, char const * const * const argv)
{
    auto appSettings = Settings::parse(argc, argv);

    auto data = MnistDataInstance::parseDataFromFile(appSettings->mnistDataPath);
    auto labels = MnistDataInstance::parseLabelsFromFile(appSettings->mnistLabelsPath);

    std::cout << "Parsed " << data->size() << " data instances and " << labels->size() << " labels." << std::endl;

    std::cout << "DONE" << std::endl;
    return 0;
}



namespace {
    Settings::UCPtr Settings::parse(int argc, char const * const * const argv)
    {
        auto mnistDataPathOption = CommandLineOption<std::string>::make("--mnist_data_path", "Path to MNIST data file", true);
        auto mnistLabelsPathOption = CommandLineOption<std::string>::make("--mnist_labels_path", "Path to MNIST labels file", true);
        mnist_deep_ann::parse({mnistDataPathOption, mnistLabelsPathOption}, argc, argv);
        auto result = std::make_unique<Settings>();
        result->mnistDataPath = mnistDataPathOption->value.value();
        result->mnistLabelsPath = mnistLabelsPathOption->value.value();
        return result;
    } // end Settings::parse
} // end anonymous namespace