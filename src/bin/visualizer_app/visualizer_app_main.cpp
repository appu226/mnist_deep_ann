#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "clo.hpp"


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

    std::cout << "DONE" << std::endl;
    return 0;
}



namespace {
    Settings::UCPtr Settings::parse(int argc, char const * const * const argv)
    {
        using CloInterfacePtr = std::shared_ptr<CloInterface>;
        auto mnistDataPathOption = CommandLineOption<std::string>::make("--mnist_data_path", "Path to MNIST data file", true);
        auto mnistLabelsPathOption = CommandLineOption<std::string>::make("--mnist_labels_path", "Path to MNIST labels file", true);
        mnist_deep_ann::parse({mnistDataPathOption, mnistLabelsPathOption}, argc, argv);
        auto result = std::make_unique<Settings>();
        result->mnistDataPath = mnistDataPathOption->value.value();
        result->mnistLabelsPath = mnistLabelsPathOption->value.value();
        return result;
    }
}