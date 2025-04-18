#include <iostream>
#include <stdexcept>
#include <string>

#include <ann/convolution_hierarchy.h>
#include <ann/simple_activation.h>
#include <command_line_options/clo.hpp>
#include <mnist_data/mnist_data.h>



using namespace mnist_deep_ann;



namespace {
	struct Settings {
		std::string trainingDataPath;
		std::string trainingLabelsPath;
		std::string testDataPath;
		std::string testLabelsPath;

		using UPtr = std::unique_ptr<Settings>;
		using UCPtr = std::unique_ptr<Settings const>;
		static UCPtr parse(int argc, char const* const* const argv);
	};

	IConvolutionHierarchy::Ptr createConvolutionHierarchy(size_t imageWidth, size_t imageHeight);
	void train_network(
		IConvolutionHierarchy::Ptr const& network, 
		MnistDataInstance::DataVecPtr const& trainingData,
		MnistDataInstance::LabelVecPtr const& traiingLabels);
	Example::VecCPtr convertToMnistExamples(
		MnistDataInstance::DataVecPtr const& mnistData,
		MnistDataInstance::LabelVecPtr const& mnistLabels
	);

} // end anonymous namespace



int main(int argc, char const* const* const argv)
{
	std::string step;

	try {

		step = "Parsing training and test data...";
		auto appSettings = Settings::parse(argc, argv);
		auto trainingData = MnistDataInstance::parseDataFromFile(appSettings->trainingDataPath);
		auto trainingLabels = MnistDataInstance::parseLabelsFromFile(appSettings->trainingLabelsPath);
		auto testData = MnistDataInstance::parseDataFromFile(appSettings->testDataPath);
		auto testLabels = MnistDataInstance::parseLabelsFromFile(appSettings->testLabelsPath);

		step = "Creating network";
		auto network = createConvolutionHierarchy(trainingData->at(0)->rows, trainingData->at(0)->cols);

		step = "Training";
		train_network(network, trainingData, trainingLabels);
	}
	catch (std::bad_alloc const&)
	{
		std::cout << "bad_alloc while " << step << std::endl;
		throw;
	}
	catch (std::exception const& e)
	{
		std::cout << "Exception <" << e.what()
			<< "> while " << step << std::endl;
		throw;
	}
	catch (...)
	{
		std::cout << "Unknown exception while " << step << std::endl;
		throw;
	}

	std::cout << "DONE" << std::endl;
	return 0;
} // end main




namespace {
	Settings::UCPtr Settings::parse(int argc, char const* const* argv)
	{
		auto trainingDataPath = CommandLineOption<std::string>::make("--training_data_path", "Path to MNIST data file with training data", true);
		auto trainingLabelsPath = CommandLineOption<std::string>::make("--training_labels_path", "Path to MNIST labels file with training labels", true);
		auto testDataPath = CommandLineOption<std::string>::make("--test_data_path", "Path to MNIST data file with test data", true);
		auto testLabelsPath = CommandLineOption<std::string>::make("--test_labels_path", "Path to MNIST labels file with test labels", true);
		mnist_deep_ann::parse(
			{ trainingDataPath, trainingLabelsPath, testDataPath, testLabelsPath },
			argc, argv
		);
		auto result = std::make_unique<Settings>();
		result->trainingDataPath = trainingDataPath->value.value();
		result->trainingLabelsPath = trainingLabelsPath->value.value();
		result->testDataPath = testDataPath->value.value();
		result->testLabelsPath = testLabelsPath->value.value();
		return result;
	}



	IConvolutionHierarchy::Ptr createConvolutionHierarchy(size_t imageWidth, size_t imageHeight)
	{
		auto result = IConvolutionHierarchy::create(imageWidth, imageHeight, SimpleActivation::create());
		result->addLayer(5, 5, 20);
		//result->addLayer(5, 5, 40);
		result->addFinalLayer(10);
		return result;
	}

	void train_network(
		IConvolutionHierarchy::Ptr const& network,
		MnistDataInstance::DataVecPtr const& trainingData,
		MnistDataInstance::LabelVecPtr const& trainingLabels)
	{
		const size_t NUM_ITERS = 100;
		auto trainingExamples = convertToMnistExamples(trainingData, trainingLabels);
		Example::Vec subset(trainingExamples->cbegin(), trainingExamples->cbegin() + 60);
		for (size_t iter = 0; iter < NUM_ITERS; ++iter)
		{
			//double err = network->propagateExamples(*trainingExamples, 1);
			double err = network->propagateExamples(subset, 100);
			std::cout << "iter " << iter << " error " << err << std::endl;
		}
	}

	Example::VecCPtr convertToMnistExamples(
		MnistDataInstance::DataVecPtr const& mnistData,
		MnistDataInstance::LabelVecPtr const& mnistLabels
	)
	{
		Example::VecPtr result(new Example::Vec);
		result->reserve(mnistData->size());
		for (size_t i = 0; i < mnistData->size() && i < mnistLabels->size(); ++i)
		{
			auto const& data = mnistData->at(i);
			auto const label = mnistLabels->at(i);
			result->emplace_back();
			auto& example = result->back();
			example.inputs.reserve(data->raw_data.size());
			for (auto const pixel : data->raw_data)
				example.inputs.push_back(static_cast<double>(pixel) / 255.0);
			example.outputs.resize(10, 0.0);
			if (label > 10)
				throw std::runtime_error("Label must be in [0, 9], found: " + std::to_string(label));
			example.outputs[label] = 1.0;
		}
		return result;
	}
} // end anonymous namespace