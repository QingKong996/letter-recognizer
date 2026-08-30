#include <assert.h>
#include <cstdint>
#include <iostream>
#include <array>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <vector>
#include <bit>


#define HEIGHT 28
#define WIDTH 28
#define SAMPLE_SIZE 124800 
#define TEST_ROUND 20800 
#define CLASSES 26

template<typename T>
using Layer = std::array<std::array<T, HEIGHT>, WIDTH>;
template<typename T>
using Model = std::array<Layer<T>, CLASSES>;


std::array<float, CLASSES> forward(Layer<uint8_t> const &input, Model<float> const &weight){
    std::array<float, CLASSES> res = {};

    for (int cls = 0; cls < CLASSES; ++cls) {
        for(int x = 0; x < WIDTH; ++x){
            for(int y = 0; y < HEIGHT; ++y){
                res[cls] += input[x][y] * weight[cls][x][y];
            }
        }
    }
    return res;
}


using Data = std::vector<Layer<uint8_t>>;


void print_layer(Layer<uint8_t> const &layer){
    for(int x = 0; x < WIDTH; ++x){
        for(int y = 0;y < HEIGHT; ++y){
            int value = layer[x][y];
            std::cout << "\033[48;2;" << value << ";" << value << ";" << value << "m  ";
        }
        std::cout << "\033[0m\n";
    }
}

void print_model(Model<float> const &model){
    for(int cls = 0; cls < CLASSES; ++cls){
        float max = std::numeric_limits<float>::lowest();
        float min = std::numeric_limits<float>::max();

        for(int x = 0; x < WIDTH; ++x){
            for(int y = 0;y < HEIGHT; ++y){
                if(model[cls][x][y] > max) max = model[cls][x][y];
                if(model[cls][x][y] < min) min = model[cls][x][y];
            }
        }

        for(int x = 0; x < WIDTH; ++x){
            for(int y = 0;y < HEIGHT; ++y){
                float t = (model[cls][x][y] - min) / (max - min);
                int value = static_cast<int>(t * 255.0f);
                
                std::cout << "\033[48;2;" << value << ";" << value << ";" << value << "m  ";
            }
            std::cout << "\033[0m\n";
        }
    }
}

void load_image_data(Data &data, std::filesystem::path path, int dataCount){
    std::ifstream file(path, std::ios::binary);

    assert(file && "Failed to open training data file");

    uint32_t details[4]; // Magic Number, Data Count, rows, cols
    for(int i = 0; i < 4; ++i){
        file.read(reinterpret_cast<char*>(&details[i]), 4);
        details[i] = std::byteswap(details[i]);
        // std::cout << details[i] << "\n";
    }

    assert(details[2] == HEIGHT && "rows not equal to HEIGHT");
    assert(details[3] == WIDTH && "cols not equal to WIDTH");

    assert(details[1] >= dataCount && "Data Count less than dataCount");

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    data.resize(dataCount);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * WIDTH * HEIGHT);
    for(Layer<uint8_t> &layer : data){
        for(int x = 0; x < WIDTH; ++x){
            for(int y = x + 1; y < HEIGHT; ++y){
                std::swap(layer[x][y], layer[y][x]);
            }
        }
        // print_layer(layer);
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Success load " << dataCount << " layers\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";
}

void load_label_data(std::vector<uint8_t> &data, std::filesystem::path path, int dataCount){
    std::ifstream file(path, std::ios::binary);

    assert(file && "Failed to open training data file");

    uint32_t details[2]; // Magic Number, Data Count, rows, cols
    for(int i = 0; i < 2; ++i){
        file.read(reinterpret_cast<char*>(&details[i]), 4);
        details[i] = std::byteswap(details[i]);
        // std::cout << details[i] << "\n";
    }

    assert(details[1] >= dataCount && "Data Count less than dataCount");

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    data.resize(dataCount);
    file.read(reinterpret_cast<char*>(data.data()), data.size());
    // std::cout << static_cast<char>(data.back() + 'a' - 1) << ' ';

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Success load " << dataCount << " lables\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";
}

void apply_patch(Layer<float> &weight, Layer<uint8_t> const &data, bool negative){
    for(int x = 0; x < WIDTH; ++x){
        for(int y = 0; y < HEIGHT; ++y){
            float patch = data[x][y] / 255.0;
            if(negative) patch *= -1;
            weight[x][y] += patch;
        }
    }
}

void train(Data const &trainingData, std::vector<uint8_t> const &trainingLable, Model<float> &model){
    assert(trainingData.size() >= SAMPLE_SIZE && "Training data less than SAMPLE_SIZE");
    assert(trainingLable.size() >= SAMPLE_SIZE && "Training lable less than SAMPLE_SIZE");

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    for(int trainingRound = 0; trainingRound < SAMPLE_SIZE; ++trainingRound) {
        std::array<float, CLASSES> res = forward(trainingData[trainingRound], model);
        auto it = std::max_element(res.begin(), res.end());
        std::size_t index = std::distance(res.begin(), it);
        if(index + 1 == trainingLable[trainingRound]){
            // std::cout << "expected: " << static_cast<char>(trainingLable[trainingRound]+ 'a' - 1) << " get: " << static_cast<char>(index + 'a') << '\n';
        }else{
            // std::cout << "expected: " << static_cast<char>(trainingLable[trainingRound]+ 'a' - 1) << " bug get: " << static_cast<char>(index + 'a') << '\n';
            apply_patch(model[index], trainingData[trainingRound] ,true);
            apply_patch(model[trainingLable[trainingRound] - 1], trainingData[trainingRound] ,false);
        }
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Success train " << SAMPLE_SIZE << " rounds\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";

}

void test(Data const &testingData, std::vector<uint8_t> const &testingLable, Model<float> const &model, int testRound){
    assert(testingData.size() >= testRound && "Testing data less than testRound");
    assert(testingLable.size() >= testRound && "Testing lable less than testRound");

    int errCount = 0;

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    for(int testingRound = 0; testingRound < testRound; ++testingRound){
        std::array<float, CLASSES> res = forward(testingData[testingRound], model);
        auto it = std::max_element(res.begin(), res.end());
        std::size_t index = std::distance(res.begin(), it);
        if(index + 1 == testingLable[testingRound]){
        }else{
            errCount++;
        }
    }
    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    

    std::cout << "Success test " << testRound << " rounds\n";
    std::cout << "Correct: " << testRound - errCount << " Error:" << errCount << " \n";
    std::cout << "Correct rate: " << static_cast<float>((testRound - errCount) / (testRound * 1.0f)) << '\n';
    std::cout << "Time taken: " << duration.count() << " seconds\n";
}

int main() {

    // Layer input = {};
    // auto ans = forward(input, weight);
    // for(auto t : ans){
    //     std::cout << t << '\n';
    // }
    Model<float> weight = {};
    Data trainingData;
    std::vector<uint8_t> trainingLable;

    load_image_data(trainingData, "./dataset/emnist-letters-train-images-idx3-ubyte/emnist-letters-train-images-idx3-ubyte", SAMPLE_SIZE);
    load_label_data(trainingLable, "./dataset/emnist-letters-train-labels-idx1-ubyte/emnist-letters-train-labels-idx1-ubyte", SAMPLE_SIZE);

    Data testingData;
    std::vector<uint8_t> testingLable;

    load_image_data(testingData, "./dataset/emnist-letters-test-images-idx3-ubyte/emnist-letters-test-images-idx3-ubyte", TEST_ROUND);
    load_label_data(testingLable, "./dataset/emnist-letters-test-labels-idx1-ubyte/emnist-letters-test-labels-idx1-ubyte", TEST_ROUND);

    for(int epoch = 0; epoch < 10; ++epoch){
        std::cout << "--------------------" << "EPOCH:" << epoch << "--------------------" << '\n';
        train(trainingData, trainingLable, weight);
        // print_model(weight);
        std::cout << "--------------------" << "TRAINING_DATASET" << "--------------------" << '\n';
        test(trainingData, trainingLable, weight, SAMPLE_SIZE);
        std::cout << "--------------------" << "TESTING_DATASET" << "--------------------" << '\n';
        test(testingData, testingLable, weight, TEST_ROUND);
    }



}
