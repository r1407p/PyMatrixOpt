#include "Matrix.hpp"
#include <stdexcept>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include <iostream>


namespace py=pybind11;

Matrix::Matrix(){
    this->m_nrow = 0;
    this->m_ncol = 0;
    this->m_buffer = nullptr;
}

Matrix::Matrix(size_t nrow, size_t ncol){
    this->m_nrow = nrow;
    this->m_ncol = ncol;
    this->m_buffer = new double[nrow * ncol];
    for(size_t i = 0; i < nrow * ncol; i++){
        this->m_buffer[i] = 0;
    }
}

Matrix::Matrix(size_t row, size_t col, double val){
    this->m_nrow = row;
    this->m_ncol = col;
    this->m_buffer = new double[row * col];
    for(size_t i = 0; i < row * col; i++){
        this->m_buffer[i] = val;
    }
}

Matrix::Matrix(size_t row, size_t col,const std::vector<double> &v){
    this->m_nrow = row;
    this->m_ncol = col;
    this->m_buffer = new double[row * col];
    if(v.size() != row * col){
        throw std::invalid_argument("size of vector does not match matrix size");
    }
    for(size_t i = 0; i < row * col; i++){
        this->m_buffer[i] = v[i];
    }
}
Matrix::Matrix(const Matrix &m){
    this->m_nrow = m.m_nrow;
    this->m_ncol = m.m_ncol;
    this->m_buffer = new double[m.m_nrow * m.m_ncol];
    for(size_t i = 0; i < m.m_nrow * m.m_ncol; i++){
        this->m_buffer[i] = m.m_buffer[i];
    }
} 

size_t Matrix::index(size_t i, size_t j) const{
    return i * m_ncol + j;
}
size_t Matrix::nrow() const{
    return m_nrow;
}
size_t Matrix::ncol() const{
    return m_ncol;
}

double* Matrix::get_buffer() const{
    return m_buffer;
}
    
Matrix::~Matrix() {
    delete[] m_buffer;
}

double Matrix::operator() (size_t row, size_t col) const{
    if (row < 0 || row >= m_nrow || col < 0 || col > m_ncol){
        throw std::out_of_range("index out of range");
    }
    return m_buffer[index(row, col)];
}
double &Matrix::operator() (size_t row, size_t col){
    if (row < 0 || row >= m_nrow || col < 0 || col > m_ncol){
        throw std::out_of_range("index out of range");
    }
    return m_buffer[index(row, col)];
}
bool Matrix::operator==(const Matrix &m){
    if(this->m_nrow != m.m_nrow || this->m_ncol != m.m_ncol){
        return false;
    }
    for(size_t i = 0; i < this->m_nrow * this->m_ncol; i++){
        if(this->m_buffer[i] != m.m_buffer[i]){
            return false;
        }
    }
    return true;
}

Matrix Matrix::operator+(const Matrix &m){
    if(this->m_nrow != m.m_nrow || this->m_ncol != m.m_ncol){
        throw std::invalid_argument("matrix size does not match");
    }
    Matrix result(m_nrow, m_ncol);
    for(size_t i = 0; i < m_nrow * m_ncol; i++){
        result.m_buffer[i] = this->m_buffer[i] + m.m_buffer[i];
    }
    return result;
}
Matrix Matrix::operator-(const Matrix &m){
    if(this->m_nrow != m.m_nrow || this->m_ncol != m.m_ncol){
        throw std::invalid_argument("matrix size does not match");
    }
    Matrix result(m_nrow, m_ncol);
    for(size_t i = 0; i < m_nrow * m_ncol; i++){
        result.m_buffer[i] = this->m_buffer[i] - m.m_buffer[i];
    }
    return result;

}
Matrix& Matrix::operator=(const Matrix &m){
    if(this->m_nrow != m.m_nrow || this->m_ncol != m.m_ncol){
        throw std::invalid_argument("matrix size does not match");
    }
    for(size_t i = 0; i < m_nrow * m_ncol; i++){
        this->m_buffer[i] = m.m_buffer[i];
    }
    return *this;
}

Matrix matrix_multiply_naive(Matrix const &m1, Matrix const &m2){
    if(m1.ncol() != m2.nrow()){
        throw std::invalid_argument("matrix size does not match");
    }
    Matrix result(m1.nrow(), m2.ncol());
    for(size_t i = 0; i < m1.nrow(); i++){
        for(size_t j = 0; j < m2.ncol(); j++){
            for(size_t k = 0; k < m1.ncol(); k++){
                result(i, j) += m1(i, k) * m2(k, j);
            }
        }
    }
    return result;
}

// need to implement
Matrix matrix_multiply_strassen(Matrix const &m1, Matrix const &m2){
    if(m1.ncol() != m2.nrow()){
        throw std::invalid_argument("matrix size does not match");
    }

    //the first version, check if the matrix is square
    if(m1.nrow() != m1.ncol() || m2.nrow() != m2.ncol()){
        throw std::invalid_argument("matrix size does not match");
    }
    size_t n = m1.nrow();
    Matrix result(n, n);

    if (n == 1) {
        result(0, 0) = m1(0, 0) * m2(0, 0);
        return result;
    }

    if(n % 2 != 0){
        return matrix_multiply_naive(m1, m2);
    }
    size_t half_n = n / 2;

    Matrix a11(half_n, half_n), a12(half_n, half_n), a21(half_n, half_n), a22(half_n, half_n);
    Matrix b11(half_n, half_n), b12(half_n, half_n), b21(half_n, half_n), b22(half_n, half_n);

    for(size_t i = 0; i < half_n; i++){
        for(size_t j = 0; j < half_n; j++){
            a11(i, j) = m1(i, j);
            a12(i, j) = m1(i, j + half_n);
            a21(i, j) = m1(i + half_n, j);
            a22(i, j) = m1(i + half_n, j + half_n);
            b11(i, j) = m2(i, j);
            b12(i, j) = m2(i, j + half_n);
            b21(i, j) = m2(i + half_n, j);
            b22(i, j) = m2(i + half_n, j + half_n);
        }
    }

    Matrix p1 = matrix_multiply_strassen(a11 + a22, b11 + b22);
    Matrix p2 = matrix_multiply_strassen(a21 + a22, b11);
    Matrix p3 = matrix_multiply_strassen(a11, b12 - b22);
    Matrix p4 = matrix_multiply_strassen(a22, b21 - b11);
    Matrix p5 = matrix_multiply_strassen(a11 + a12, b22);
    Matrix p6 = matrix_multiply_strassen(a21 - a11, b11 + b12);
    Matrix p7 = matrix_multiply_strassen(a12 - a22, b21 + b22);

    Matrix c11 = p1 + p4 - p5 + p7;
    Matrix c12 = p3 + p5;
    Matrix c21 = p2 + p4;
    Matrix c22 = p1 - p2 + p3 + p6;
    for (size_t i = 0; i < half_n; ++i) {
        for (size_t j = 0; j < half_n; ++j) {
            result(i, j) = c11(i, j);
            result(i, j + half_n) = c12(i, j);
            result(i + half_n, j) = c21(i, j);
            result(i + half_n, j + half_n) = c22(i, j);
        }
    }
    return result;
}

// need to implement
Matrix matrix_multiply_coppersmith_winograd(Matrix const &m1, Matrix const &m2){
    
}

PYBIND11_MODULE(Matrix, m) {
    py::class_<Matrix>(m, "Matrix")
    .def(py::init<>())
    .def(py::init<size_t, size_t>())
    .def(py::init<size_t, size_t, double>())
    .def(py::init<size_t, size_t, const std::vector<double> &>())
    .def(py::init<const Matrix &>())
    .def("__getitem__", [](Matrix &m, std::vector<std::size_t> idx){ 	 
        return m(idx[0],idx[1]);       
    })
    .def("__setitem__",[](Matrix &m, std::vector<std::size_t> idx, int val){
        m(idx[0],idx[1]) = val;
    })
    .def_property_readonly("nrow", &Matrix::nrow)
    .def_property_readonly("ncol", &Matrix::ncol)
    .def("__eq__", &Matrix::operator ==);

    m.def("matrix_multiply_naive", &matrix_multiply_naive, "");
    m.def("matrix_multiply_strassen", &matrix_multiply_strassen, "");
    m.def("matrix_multiply_coppersmith_winograd", &matrix_multiply_coppersmith_winograd, "");

}