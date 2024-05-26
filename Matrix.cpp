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

Matrix Matrix::transpose() const {
    Matrix transposed(m_ncol, m_nrow);
    for (size_t i = 0; i < m_nrow; ++i) {
        for (size_t j = 0; j < m_ncol; ++j) {
            transposed(j, i) = (*this)(i, j);
        }
    }
    return transposed;
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

Matrix matrix_multiply_naive_tile(Matrix const &m1, Matrix const &m2, std::size_t size){
    if(m1.ncol() != m2.nrow()){
        throw std::invalid_argument("matrix size does not match");
    }
    Matrix result(m1.nrow(), m2.ncol());
    for(size_t i = 0; i < m1.nrow(); i += size){
        for(size_t j = 0; j < m2.ncol(); j += size){
            for(size_t k = 0; k < m1.ncol(); k += size){
                for(size_t ii = i; ii < std::min(i + size, m1.nrow()); ii++){
                    for(size_t jj = j; jj < std::min(j + size, m2.ncol()); jj++){
                        for(size_t kk = k; kk < std::min(k + size, m1.ncol()); kk++){
                            result(ii, jj) += m1(ii, kk) * m2(kk, jj);
                        }
                    }
                }
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
    Matrix S1(half_n, half_n), S2(half_n, half_n), S3(half_n, half_n), S4(half_n, half_n);
    Matrix T1(half_n, half_n), T2(half_n, half_n), T3(half_n, half_n), T4(half_n, half_n);
    S1 = a21 + a22;
    S2 = S1 - a11;
    S3 = a11 - a21;
    S4 = a12 - S2;
    T1 = b12 - b11;
    T2 = b22 - T1;
    T3 = b22 - b12;
    T4 = T2 - b21;

    Matrix M1(half_n, half_n), M2(half_n, half_n), M3(half_n, half_n), M4(half_n, half_n), M5(half_n, half_n), M6(half_n, half_n), M7(half_n, half_n);
    Matrix U1(half_n, half_n), U2(half_n, half_n), U3(half_n, half_n), U4(half_n, half_n), U5(half_n, half_n), U6(half_n, half_n), U7(half_n, half_n);
    M1 = matrix_multiply_coppersmith_winograd(a11, b11);
    M2 = matrix_multiply_coppersmith_winograd(a12, b21);
    M3 = matrix_multiply_coppersmith_winograd(S4, b22);
    M4 = matrix_multiply_coppersmith_winograd(a22, T4);
    M5 = matrix_multiply_coppersmith_winograd(S1, T1);
    M6 = matrix_multiply_coppersmith_winograd(S2, T2);
    M7 = matrix_multiply_coppersmith_winograd(S3, T3);

    U1 = M1 + M2;
    U2 = M1 + M6;
    U3 = U2 + M7;
    U4 = U2 + M5;
    U5 = U4 + M3;
    U6 = U3 - M4;
    U7 = U3 + M5;

    for (size_t i = 0; i < half_n; ++i) {
        for (size_t j = 0; j < half_n; ++j) {
            result(i, j) = U1(i, j);
            result(i, j + half_n) = U5(i, j);
            result(i + half_n, j) = U6(i, j);
            result(i + half_n, j + half_n) = U7(i, j);
        }
    }
    return result;
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
    m.def("matrix_multiply_naive_tile", &matrix_multiply_naive_tile, "");
    m.def("matrix_multiply_strassen", &matrix_multiply_strassen, "");
    m.def("matrix_multiply_coppersmith_winograd", &matrix_multiply_coppersmith_winograd, "");
}