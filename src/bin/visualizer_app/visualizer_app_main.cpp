#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include "clo.hpp"
#include "mnist_data.h"




using namespace mnist_deep_ann;



namespace {
    struct Settings {
        std::string mnistDataPath;
        std::string mnistLabelsPath;
        std::optional<size_t> indexToDisplay;
        
        using UPtr = std::unique_ptr<Settings>;
        using UCPtr = std::unique_ptr<Settings const>;
        static UCPtr parse(int argc, char const * const * const argv);
    }; // end struct Settings

    void displayDataInstance(MnistDataInstance const& dataInstance);

} // end anonymous namespace






int main(int const argc, char const * const * const argv)
{
    auto appSettings = Settings::parse(argc, argv);

    auto data = MnistDataInstance::parseDataFromFile(appSettings->mnistDataPath);
    auto labels = MnistDataInstance::parseLabelsFromFile(appSettings->mnistLabelsPath);
    std::cout << "Parsed " << data->size() << " data instances and " << labels->size() << " labels." << std::endl;

    if (appSettings->indexToDisplay.has_value())
    {
        size_t indexToDisplay = appSettings->indexToDisplay.value();
        std::cout << "Displaying data at index " << indexToDisplay
            << "Label : " << int(labels->at(indexToDisplay)) << '\n';
        displayDataInstance(*data->at(indexToDisplay));
    }

    std::cout << "DONE" << std::endl;
    return 0;
}



namespace {
    Settings::UCPtr Settings::parse(int argc, char const * const * const argv)
    {
        auto mnistDataPathOption = CommandLineOption<std::string>::make("--mnist_data_path", "Path to MNIST data file", true);
        auto mnistLabelsPathOption = CommandLineOption<std::string>::make("--mnist_labels_path", "Path to MNIST labels file", true);
        auto indexToDisplayOption = CommandLineOption<size_t>::make("--index_to_display", "Index of example to display", false, std::optional<size_t>());
        mnist_deep_ann::parse({mnistDataPathOption, mnistLabelsPathOption, indexToDisplayOption}, argc, argv);
        auto result = std::make_unique<Settings>();
        result->mnistDataPath = mnistDataPathOption->value.value();
        result->mnistLabelsPath = mnistLabelsPathOption->value.value();
        result->indexToDisplay = indexToDisplayOption->value;
        return result;
    } // end Settings::parse



    void displayDataInstance(MnistDataInstance const& dataInstance)
    {
        std::cout << "Size = " << dataInstance.rows << 'x' << dataInstance.cols << "\n ";
        for (size_t c = 0; c < dataInstance.cols; ++c) 
            std::cout << "--";
        std::cout << '\n';
        for (size_t r = 0; r < dataInstance.rows; ++r)
        {
            std::cout << '|';
            for (size_t c = 0; c < dataInstance.cols; ++c)
            {
                unsigned char v = dataInstance.getPixel(r, c);
                if (v < 86)
                    std::cout << "  ";
                else if (v < 172)
                    std::cout << "--";
                else
                    std::cout << "**";
            }
            std::cout << "|\n";
        }
        std::cout << ' ';
        for (size_t c = 0; c < dataInstance.cols; ++c) 
            std::cout << "--";
        std::cout << '\n';
    }// end displayDataInstance


} // end anonymous namespace