#include <assert.h>
#include <cstdint>
#include <iostream>
#include <array>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <vector>
#include <bit>
#include <algorithm>
#include <limits>

#define SAMPLE_SIZE 124800 
#define TEST_ROUND  20800 
#define CLASSES     26

constexpr size_t IMAGE_HEIGHT = 28;
constexpr size_t IMAGE_WIDTH  = 28;

// template<typename T, size_t H, size_t W>
// using Matrix = std::array<std::array<T, H>, W>;

template<typename T, size_t Rows, size_t Cols>
struct Matrix {
    std::array<T, Rows * Cols> data = {};

    constexpr static size_t rows = Rows;
    constexpr static size_t cols = Cols;

    constexpr T& operator()(size_t row, size_t col){
        return data[row * Cols + col];
    }
    constexpr const T& operator()(size_t row, size_t col) const {
        return data[row * Cols + col];
    }
    
    constexpr Matrix operator*(Matrix matrix) const {
        assert(matrix.rows == cols && "Left matrix cols must equal right matrix rows");


    }

};


using Data = std::vector<Matrix<uint8_t, 28, 28>>;

template<typename T, size_t Classes, size_t Height, size_t Width>
struct Linear {
    std::array<Matrix<T, Height, Width>, Classes> weight;
    std::array<T, Classes> bias;


    template<size_t Input_H, size_t Input_W, size_t Output_H, size_t Output_W>
    std::array<float, CLASSES> forward(Matrix<uint8_t, Input_H, Input_W> const &input){
        std::array<float, CLASSES> res = {};

        assert(false && "TODO: implement new forward");

        for (int cls = 0; cls < CLASSES; ++cls) {
            res[cls] += bias[cls];
            for(int x = 0; x < H; ++x){
                for(int y = 0; y < W; ++y){
                    res[cls] += input[x][y] * (weight[cls][x][y] / 255.0f);
                }
            }
        }
        return res;
    }

    template<size_t H, size_t W>
    void print(){
        for(int cls = 0; cls < CLASSES; ++cls){
            float max = std::numeric_limits<float>::lowest();
            float min = std::numeric_limits<float>::max();

            for(int x = 0; x < W; ++x){
                for(int y = 0;y < H; ++y){
                    if(weight[cls][x][y] > max) max = weight[cls][x][y];
                    if(weight[cls][x][y] < min) min = weight[cls][x][y];
                }
            }

            for(int x = 0; x < W; ++x){
                for(int y = 0;y < H; ++y){
                    float t = (weight[cls][x][y] - min) / (max - min);
                    int value = static_cast<int>(t * 255.0f);
                    
                    std::cout << "\033[48;2;" << value << ";" << value << ";" << value << "m  ";
                }
                std::cout << "\033[0m\n";
            }
        }
    }


};

template<typename T>
struct Model {
    Linear<float, 26, IMAGE_HEIGHT, IMAGE_WIDTH> linear_1;

    void train(){

    }

    void test(){

    }
};





template<size_t H, size_t W>
void print_matrix(Matrix<uint8_t, H, W> const &matrix){
    for(int x = 0; x < W; ++x){
        for(int y = 0;y < H; ++y){
            int value = matrix[x][y];
            std::cout << "\033[48;2;" << value << ";" << value << ";" << value << "m  ";
        }
        std::cout << "\033[0m\n";
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

    assert(details[2] == IMAGE_HEIGHT && "rows not equal to IMAGE_HEIGHT");
    assert(details[3] == IMAGE_WIDTH && "cols not equal to IMAGE_WIDTH");

    assert(details[1] >= dataCount && "Data Count less than dataCount");

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    data.resize(dataCount);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * IMAGE_WIDTH * IMAGE_HEIGHT);
    for(Matrix<uint8_t, IMAGE_HEIGHT, IMAGE_WIDTH> &layer : data){
        for(int x = 0; x < IMAGE_WIDTH; ++x){
            for(int y = x + 1; y < IMAGE_HEIGHT; ++y){
                std::swap(layer[x][y], layer[y][x]);
            }
        }
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Success load " << dataCount << " layers\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";
}

void load_label_data(std::vector<uint8_t> &data, std::filesystem::path path, int dataCount){
    std::ifstream file(path, std::ios::binary);

    assert(file && "Failed to open training data file");

    uint32_t details[2]; // Magic Number, Data Count
    for(int i = 0; i < 2; ++i){
        file.read(reinterpret_cast<char*>(&details[i]), 4);
        details[i] = std::byteswap(details[i]);
    }

    assert(details[1] >= dataCount && "Data Count less than dataCount");

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    data.resize(dataCount);
    file.read(reinterpret_cast<char*>(data.data()), data.size());

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Success load " << dataCount << " lables\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";
}







int main() {

    Model<float> weight = {};
    Model<float> averageWeight = {};
    Data trainingData;
    std::vector<uint8_t> trainingLable;

    load_image_data(trainingData, "./dataset/emnist-letters-train-images-idx3-ubyte/emnist-letters-train-images-idx3-ubyte", SAMPLE_SIZE);
    load_label_data(trainingLable, "./dataset/emnist-letters-train-labels-idx1-ubyte/emnist-letters-train-labels-idx1-ubyte", SAMPLE_SIZE);

    Data testingData;
    std::vector<uint8_t> testingLable;

    load_image_data(testingData, "./dataset/emnist-letters-test-images-idx3-ubyte/emnist-letters-test-images-idx3-ubyte", TEST_ROUND);
    load_label_data(testingLable, "./dataset/emnist-letters-test-labels-idx1-ubyte/emnist-letters-test-labels-idx1-ubyte", TEST_ROUND);

    // for(int epoch = 0; epoch < 10; ++epoch){
        // std::cout << "--------------------" << "EPOCH:" << epoch << "--------------------" << '\n';

        averageWeight = train(trainingData, trainingLable, weight);
        // print_model(weight);
        std::cout << "--------------------" << "TRAINING_DATASET NORMAL_MODEL" << "--------------------" << '\n';
        test(trainingData, trainingLable, weight, SAMPLE_SIZE);
        std::cout << "--------------------" << "TESTING_DATASET NORMAL_MODEL" << "--------------------" << '\n';
        test(testingData, testingLable, weight, TEST_ROUND);
        std::cout << "--------------------" << "TRAINING_DATASET AVERAGE_MODEL" << "--------------------" << '\n';
        test(trainingData, trainingLable, averageWeight, SAMPLE_SIZE);
        std::cout << "--------------------" << "TESTING_DATASET AVERAGE_MODEL" << "--------------------" << '\n';
        test(testingData, testingLable, averageWeight, TEST_ROUND);
    // }

    

}
