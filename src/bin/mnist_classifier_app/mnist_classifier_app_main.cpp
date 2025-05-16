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

		//std::ofstream fout("temp\\diags.json");
		//network->diagnose(fout);

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
		result->addLayer(5, 5, 5);
		result->addLayer(10, 10, 20);
		result->addLayer(10, 10, 20);
		result->addFinalLayer(10);
		return result;
	}






	void train_network(
		IConvolutionHierarchy::Ptr const& network,
		MnistDataInstance::DataVecPtr const& trainingData,
		MnistDataInstance::LabelVecPtr const& trainingLabels)
	{
		size_t const NUM_ITERS = 1000;
		double const STEP_LO = 0.0;
		double const STEP_HI = 0.001;
		double const STEP_TERM = .000001;
		double const GOL_RAT = 1.618034;
		size_t eval_id = 0;
		double current_step;
		auto trainingExamples = convertToMnistExamples(trainingData, trainingLabels);
		network->perturbWeights(1);
		Example::Vec subset(trainingExamples->cbegin(), trainingExamples->cbegin() + 20);

		auto evaluate = [&network, &eval_id, &current_step, &subset](double const next_step) -> double {
			network->adjust(next_step - current_step);
			current_step = next_step;
			double result = network->getClassificationError(subset);
			std::cout << "eval " << (++eval_id) << " error " << result << std::endl;
			return result;
		};
		auto move_if_min = [&network, &current_step, &subset](double possible_min_step, double possible_min_err, double compare1, double compare2, double compare3) -> bool {
			if (possible_min_err <= compare1 && possible_min_err <= compare2 && possible_min_err <= compare3)
			{
				network->adjust(possible_min_step - current_step);
				current_step = possible_min_step;
				std::cout << "moving to new local minima " << possible_min_err << std::endl;
				return true;
			}
			else
			{
				return false;
			}
		};
		auto move_to_min =
			[&network, &current_step, &move_if_min](
				double step_a, double err_a,
				double step_b, double err_b,
				double step_c, double err_c,
				double step_d, double err_d)
			{
				if (move_if_min(step_a, err_a, err_b, err_c, err_d))
					return;
				else if (move_if_min(step_b, err_b, err_a, err_c, err_d))
					return;
				else if (move_if_min(step_c, err_c, err_a, err_b, err_d))
					return;
				else if (move_if_min(step_d, err_d, err_a, err_b, err_c))
					return;
			};

		for (size_t iter = 0; iter < NUM_ITERS; ++iter)
		{
			//   *----------*-------*----------*
			//   A          B       C          D
			//
			//   1. AC/AB = golden_ratio
			//   2. AB = CD <=> AC = CD
			// 
			//   therefore:
			//   3. AD = AB + BD = AB + AC 
			//      => AD/AC = AC/AB = golden_ratio
			//   

			double err_a = network->propagateExamples(subset, 0);
			std::cout << "eval " << (++eval_id) << " error " << err_a << std::endl;
			current_step = 0;
			double step_a = STEP_LO;

			double step_d = STEP_HI;
			double err_d = evaluate(step_d);

			double step_c = step_a + (step_d - step_a) / GOL_RAT; // AC = AD / golden_ratio
			double err_c = evaluate(step_c);

			double step_b = step_a + (step_c - step_a) / GOL_RAT; // AB = AC / golden_ratio
			double err_b = evaluate(step_b);

			while (step_c - step_b > STEP_TERM)
			{
				if (err_b > err_c)
				{
					// drop A
					//
					// *-------------*------*-----------*
					// A             B      C           D
					//       
					//               v      v           v
					//
					//               *------*----*------*
					//               A      B    C      D
					// A <- B
					// B <- C
					// AC = AD / golden_ratio
					step_a = step_b;
					err_a = err_b;
					step_b = step_c;
					err_b = err_c;
					step_c = step_a + (step_d - step_a) / GOL_RAT;
					err_c = evaluate(step_c);
				}
				else
				{
					// drop D
					//
					// *-------------*------*-----------*
					// A             B      C           D
					//       
					// v             v      v
					//
					// *------*------*------*
					// A      B      C      D
					// D <- C
					// C <- B
					// AB = AC / golden_ratio
					step_d = step_c;
					err_d = err_c;
					step_c = step_b;
					err_c = err_b;
					step_b = step_a + (step_c - step_a) / GOL_RAT;
					err_b = evaluate(step_b);
				}
			}
			
			move_to_min(step_a, err_a, step_b, err_b, step_c, err_c, step_d, err_d);

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
				example.inputs.push_back(static_cast<double>(pixel) / 255.0 - .5);
			example.outputs.resize(10, 0.0);
			if (label > 10)
				throw std::runtime_error("Label must be in [0, 9], found: " + std::to_string(label));
			example.outputs[label] = 1.0;
		}
		return result;
	}
} // end anonymous namespace