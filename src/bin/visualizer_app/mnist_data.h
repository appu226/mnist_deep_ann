#pragma once

#include <fstream>
#include <vector>
#include <memory>

namespace mnist_deep_ann
{

    struct MnistDataInstance
    {
        using Ptr = std::shared_ptr<MnistDataInstance>;
        using CPtr = std::shared_ptr<MnistDataInstance const>;
        using DataVecPtr = std::shared_ptr<std::vector<CPtr> >;
        using LabelVecPtr = std::shared_ptr<std::vector<char> >;
        size_t rows;
        size_t cols;
        std::vector<char> raw_data;
        char getPixel(size_t row, size_t col) const;

        static DataVecPtr parseData(std::ifstream& data_file_stream);
        static DataVecPtr parseDataFromFile(std::string const& data_file_path);
        static LabelVecPtr parseLabels(std::ifstream& label_file_stream);
        static LabelVecPtr parseLabelsFromFile(std::string const& label_file_path);
    };

} // end namespace mnist_deep_ann