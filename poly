#include <iostream>
#include <vector>
#include <complex>


// 计算复数多项式的系数
std::vector<std::complex<double>> poly(const std::vector<std::complex<double>>& roots) {
    int n = roots.size();
    std::vector<std::complex<double>> coefficients(n + 1, 0.0);
    coefficients[0] = 1.0;

    for (int i = 0; i < n; i++) {
        for (int j = n; j >= 1; j--) {
            coefficients[j] = coefficients[j] - roots[i] * coefficients[j - 1];
        }
    }

    return coefficients;
}

int main() {
    std::vector<std::complex<double>> roots = {-1, -1};

    std::vector<std::complex<double>> coefficients = poly(roots);

    std::cout << "多项式系数：";
    for (const auto& coeff : coefficients) {
        std::cout << coeff << " ";
    }
    std::cout << std::endl;

    return 0;
}
