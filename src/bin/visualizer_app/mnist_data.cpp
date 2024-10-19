#include "mnist_data.h"

#include <stdexcept>

namespace {

    int readInt(std::ifstream& fin)
    {
        unsigned char a, b, c, d;
        fin.read(reinterpret_cast<char*>(&a), 1);
        fin.read(reinterpret_cast<char*>(&b), 1);
        fin.read(reinterpret_cast<char*>(&c), 1);
        fin.read(reinterpret_cast<char*>(&d), 1);
        return (int(a)<<24) + (int(b)<<16) + (int(c)<<8) + int(d);
    }

}// end anonymous namespace



namespace mnist_deep_ann
{

    char MnistDataInstance::getPixel(size_t row, size_t col) const
    {
        if (row >= rows && col >=cols)
            throw std::runtime_error(
                "MnistDataInstance::getPixel: invalid coordinates (" 
                + std::to_string(row) + ", " + std::to_string(col)
                + ") on data of size " + std::to_string(rows) + "x"
                + std::to_string(cols));
        if (raw_data.size() != rows*cols)
            throw std::runtime_error(
                "MnistDataInstance::getPixel: improperly initialized raw_data size "
                + std::to_string(raw_data.size()) + ", expected "
                + std::to_string(rows*cols));
        return raw_data[row*cols + col];
    }

    MnistDataInstance::DataVecPtr MnistDataInstance::parseData(std::ifstream& data_file_stream)
    {
        int magic, num_instances, num_rows, num_cols;
        magic = readInt(data_file_stream);
        if (magic != 2051)
            throw std::runtime_error(
                "Invalid file format. Expected magic number 2051, found "
                + std::to_string(magic));
        
        num_instances = readInt(data_file_stream);
        num_rows = readInt(data_file_stream);
        num_cols = readInt(data_file_stream);
        if (num_instances < 0 || num_rows < 0 || num_cols < 0)
            throw std::runtime_error(
                "Invalid file format. Expected positive num of instances, rows, cols. Found: "
                + std::to_string(num_instances) + ", " + std::to_string(num_rows) + ", "
                + std::to_string(num_cols) 
            );
        auto result = std::make_shared<std::vector<CPtr>>();
        while (num_instances--)
        {
            auto dataInstance = std::make_shared<MnistDataInstance>();
            dataInstance->rows = num_rows;
            dataInstance->cols = num_cols;
            dataInstance->raw_data.resize(num_rows * num_cols, 0);
            data_file_stream.read(dataInstance->raw_data.data(), num_rows * num_cols);
            result->push_back(dataInstance);
        }
        return result;
    }


    MnistDataInstance::DataVecPtr MnistDataInstance::parseDataFromFile(std::string const& data_file_path)
    {
        std::ifstream data_stream(data_file_path, std::ios_base::binary);
        return parseData(data_stream);
    }


    MnistDataInstance::LabelVecPtr MnistDataInstance::parseLabels(std::ifstream& labels_stream)
    {
        int magic = readInt(labels_stream);
        if (magic != 2049)
            throw std::runtime_error(
                "Invalid file format. Expected magic number 2049, found "
                + std::to_string(magic));
        int num_labels = readInt(labels_stream);
        if (num_labels < 0)
            throw std::runtime_error(
                "Invalid file format. Expected positive number of labels, found " + std::to_string(num_labels)
            );
        auto result = std::make_shared<std::vector<char> >();
        result->resize(num_labels);
        labels_stream.read(result->data(), num_labels);
        return result;
    }

    MnistDataInstance::LabelVecPtr MnistDataInstance::parseLabelsFromFile(std::string const& labels_file_path)
    {
        std::ifstream labels_stream(labels_file_path, std::ios_base::binary);
        return parseLabels(labels_stream);
    }

} // end namespace mnist_deep_ann