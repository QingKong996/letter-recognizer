#include <assert.h>
#include <cstdint>
#include <iostream>
#include <array>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <bit>
#include <algorithm>
#include <limits>

#define CLASSES 26

constexpr size_t IMAGE_HEIGHT = 28;
constexpr size_t IMAGE_WIDTH  = 28;

constexpr float LEARNING_RATE = 1.0f;

template<typename T, size_t Rows, size_t Cols>
struct Matrix {
    std::array<T, Rows * Cols> data {};

    constexpr static size_t rows = Rows;
    constexpr static size_t cols = Cols;

    template<typename Func>
    void apply_func_to_elements(Func func){
        for(size_t row = 0; row < Rows; ++row){
            for(size_t col = 0; col < Cols; ++col){
                (*this)(row, col) = func((*this)(row, col));
            }
        }
    }
    constexpr T& operator()(size_t row, size_t col){
        return data[row * Cols + col];
    }
    constexpr const T& operator()(size_t row, size_t col) const {
        return data[row * Cols + col];
    }

    constexpr Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& matrix) const {
        Matrix<T, Rows, Cols> res {};
        for(size_t row = 0; row < Rows; ++row){
            for(size_t col = 0; col < Cols; ++col){
                res(row, col) = (*this)(row, col) + matrix(row, col);
            }
        }
        return res;
    }
    constexpr Matrix<T, Rows, Cols> operator-(const Matrix<T, Rows, Cols>& matrix) const {
        Matrix<T, Rows, Cols> res {};
        for(size_t row = 0; row < Rows; ++row){
            for(size_t col = 0; col < Cols; ++col){
                res(row, col) = (*this)(row, col) - matrix(row, col);
            }
        }
        return res;
    }
    
    template<size_t OtherCols>
    constexpr Matrix<T, Rows, OtherCols> operator*(const Matrix<T, Cols, OtherCols>& matrix) const {

        Matrix<T, Rows, OtherCols> res {};

        for(size_t row = 0; row < Rows; ++row){
            for(size_t col = 0; col < OtherCols; ++col){
                for(size_t i = 0; i < Cols; ++i){
                    res(row, col) += (*this)(row, i) * matrix(i, col);
                }
            }
        }
        return res;
    }

    template<typename Scalar>
    constexpr Matrix<T, Rows, Cols> operator*(Scalar scalar){
        for(size_t row = 0; row < Rows; ++row){
            for(size_t col = 0; col < Cols; ++col){
                (*this)(col, row) *= scalar;
            }
        }
    }
    template<typename Scalar>
    constexpr Matrix<T, Rows, Cols> operator/(Scalar scalar){
        for(size_t row = 0; row < Rows; ++row){
            for(size_t col = 0; col < Cols; ++col){
                (*this)(col, row) /= scalar;
            }
        }
    }

    constexpr Matrix<T, Cols, Rows> transpose() const {
        Matrix<T, Cols, Rows> res;
        for(size_t row = 0; row < Rows; ++row){
            for(size_t col = 0; col < Cols; ++col){
                res(col, row) = (*this)(row, col);
            }
        }
        return res;
    }

    constexpr Matrix<T, Cols, Rows> hadamard(Matrix<T, Cols, Rows> matrix) const {
        Matrix<T, Cols, Rows> res;
        for(size_t row = 0; row < Rows; ++row){
            for(size_t col = 0; col < Cols; ++col){
                res(row, col) = (*this)(row, col) * matrix(row, col);
            }
        }
        return res;
    }


};



template<typename T, size_t Input_size, size_t Output_size>
struct Linear {
    Matrix<T, Input_size, Output_size> weight;
    Matrix<T, 1, Output_size> bias;


    Matrix<T, 1, Output_size> forward(Matrix<uint8_t, 1, Input_size> const& input){
        return input * weight + bias;
    }
    void print(){
        float max = std::numeric_limits<float>::lowest();
        float min = std::numeric_limits<float>::max();

        for(int row = 0; row < Input_size; ++row){
            for(int col = 0; col < Output_size; ++col){
                if(weight(row, col) > max) max = weight(row, col);
                if(weight(row, col) < min) min = weight(row, col);
            }
        }


        for(int row = 0; row < Input_size; ++row){
            for(int col = 0; col < Output_size; ++col){
                float t = (weight(row, col) - min) / (max - min);
                int value = static_cast<int>(t * 255.0f);
                std::cout << "\033[48;2;" << value << ";" << value << ";" << value << "m  ";
            }
            std::cout << "\033[0m\n";
        }
    }
};


template<size_t size>
void load_image_data(std::array<Matrix<uint8_t, 1, IMAGE_WIDTH * IMAGE_HEIGHT>, size>& data, std::filesystem::path path){
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

    assert(details[1] >= size && "Data size less than size");

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), data.size() * IMAGE_WIDTH * IMAGE_HEIGHT);
    for(Matrix<uint8_t, IMAGE_HEIGHT, IMAGE_WIDTH> &matrix: data){
        for(int x = 0; x < IMAGE_WIDTH; ++x){
            for(int y = x + 1; y < IMAGE_HEIGHT; ++y){
                std::swap(matrix(x, y), matrix(y, x));
            }
        }
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Success load " << size << " layers\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";
}

template<size_t size>
void load_label_data(std::array<uint8_t, size>& data, std::filesystem::path path){
    std::ifstream file(path, std::ios::binary);

    assert(file && "Failed to open training data file");

    uint32_t details[2]; // Magic Number, Data Count
    for(int i = 0; i < 2; ++i){
        file.read(reinterpret_cast<char*>(&details[i]), 4);
        details[i] = std::byteswap(details[i]);
    }

    assert(details[1] >= size && "Data Count less than dataCount");

    std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

    data.resize(size);
    file.read(reinterpret_cast<char*>(data.data()), data.size());

    std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    std::cout << "Success load " << size << " lables\n";
    std::cout << "Time taken: " << duration.count() << " seconds\n";
}


template<typename T>
T ReLU(T input){
    return input > 0 ? input : 0;
}

template<typename T>
T sigmoid(T input){
    return 1 / (1 + std::exp(input));
}

template<typename T, size_t size>
Matrix<T, 1, size> softmax(Matrix<T, 1, size> matrix){
    Matrix<T, 1, size> res {};
    float sum = 0;
    for(int col = 0; col < size; ++col){
        sum += std::exp(matrix(0, col));
    }
    for(int col = 0; col < size; ++col){
        res(0, col) = std::exp(matrix(0, col)) / sum;
    }
    return res;
}

template<typename T>
struct Model {
    Linear<float, IMAGE_HEIGHT * IMAGE_WIDTH, IMAGE_HEIGHT * IMAGE_WIDTH> linear_1;
    
    Linear<float, IMAGE_HEIGHT * IMAGE_WIDTH, CLASSES> linear_2;
    constexpr static int SAMPLE_SIZE = 124800;
    constexpr static int TEST_ROUND = 20800;

    void train(){
        std::array<Matrix<uint8_t, 1, IMAGE_WIDTH * IMAGE_HEIGHT>, SAMPLE_SIZE> trainingData;
        std::array<uint8_t, SAMPLE_SIZE> trainingLable;

        load_image_data(trainingData, "./dataset/emnist-letters-train-images-idx3-ubyte/emnist-letters-train-images-idx3-ubyte");
        load_label_data(trainingLable, "./dataset/emnist-letters-train-labels-idx1-ubyte/emnist-letters-train-labels-idx1-ubyte");

        for(int round = 0; round < SAMPLE_SIZE; ++round){
            Matrix<float, 1, IMAGE_WIDTH * IMAGE_HEIGHT> res_1 = linear_1.forward(trainingData[round]);

            res_1.apply_func_to_elements(ReLU<float>);

            Matrix<float, 1, CLASSES> res_2 = linear_2.forward(trainingData[round]);


            Matrix<float, 1, CLASSES> res_3 = softmax(res_2);
            float loss = -1.0f * std::log(res_3(0, trainingLable[round] - 1));


        }

    }

    void test(){

        std::array<Matrix<uint8_t, 1, IMAGE_WIDTH * IMAGE_HEIGHT>, TEST_ROUND> testingData;
        std::array<uint8_t, TEST_ROUND> testingLable;

        load_image_data(testingData, "./dataset/emnist-letters-test-images-idx3-ubyte/emnist-letters-test-images-idx3-ubyte");
        load_label_data(testingLable, "./dataset/emnist-letters-test-labels-idx1-ubyte/emnist-letters-test-labels-idx1-ubyte");
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








int main() {


}
