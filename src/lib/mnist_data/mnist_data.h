#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include "mnist_data_export.h"

namespace mnist_deep_ann
{

    struct MnistDataInstance
    {
        using Ptr = std::shared_ptr<MnistDataInstance>;
        using CPtr = std::shared_ptr<MnistDataInstance const>;
        using DataVecPtr = std::shared_ptr<std::vector<CPtr> >;
        using LabelVecPtr = std::shared_ptr<std::vector<unsigned char> >;
        size_t rows;
        size_t cols;
        std::vector<unsigned char> raw_data;
        MNIST_DATA_EXPORT unsigned char getPixel(size_t row, size_t col) const;

        MNIST_DATA_EXPORT static DataVecPtr parseData(std::ifstream& data_file_stream);
        MNIST_DATA_EXPORT static DataVecPtr parseDataFromFile(std::string const& data_file_path);
        MNIST_DATA_EXPORT static LabelVecPtr parseLabels(std::ifstream& label_file_stream);
        MNIST_DATA_EXPORT static LabelVecPtr parseLabelsFromFile(std::string const& label_file_path);
    };

} // end namespace mnist_deep_ann