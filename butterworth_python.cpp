#include <iostream>
#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

#define M_PI 3.141592653589793


struct Complex {
    double real;
    double imag;
};

// Function to calculate the relative degree
int relativeDegree(const std::vector<std::complex<double>>& z, const std::vector<std::complex<double>>& p) {
    int degree = static_cast<int>(p.size()) - static_cast<int>(z.size());
    return degree;
}

// Function to compute poles and zeros for an analog prototype of the Nth-order Butterworth filter
void buttap(int N, std::vector<std::complex<double>>& poles, std::vector<std::complex<double>>& zeros, double& k) {

    zeros.clear();
    poles.clear();

    for (int m = -N + 1; m < N; m += 2) {
        // Middle value is 0 to ensure an exactly real pole
        double realPart = -std::exp(0) * std::cos(M_PI * m / (2 * N));
        double imagPart = -std::exp(0) * std::sin(M_PI * m / (2 * N));
        poles.push_back(std::complex<double>(realPart, imagPart));
    }

    k = 1.0;
}

// Function to transform analog low-pass poles and zeros to desired frequency
// Function to transform lowpass filter to another lowpass filter with a different cutoff frequency
void lp2lp_zpk(const std::vector<std::complex<double>>& z, const std::vector<std::complex<double>>& p, double k,
    std::vector<std::complex<double>>& z_lp, std::vector<std::complex<double>>& p_lp, double& k_lp, double wo = 1.0) {
    int degree = relativeDegree(z, p);

    z_lp.clear();
    p_lp.clear();

    for (const auto& zero : z) {
        z_lp.push_back(wo * zero);
    }

    for (const auto& pole : p) {
        p_lp.push_back(wo * pole);
    }

    k_lp = k * std::pow(wo, degree);
}

// Function to compute the product of a vector of complex numbers
std::complex<double> complexVectorProduct(const std::vector<std::complex<double>>& vec) {
    std::complex<double> result(1.0, 0.0);
    for (const auto& val : vec) {
        result *= val;
    }
    return result;
}

// Function to perform bilinear transformation from analog to digital domain
void bilinear_zpk(const std::vector<std::complex<double>>& z, const std::vector<std::complex<double>>& p, double k, double fs,
    std::vector<std::complex<double>>& z_z, std::vector<std::complex<double>>& p_z, double& k_z) {
    int degree = relativeDegree(z, p);

    z_z.clear();
    p_z.clear();

    double fs2 = 2.0 * fs;

    for (const auto& zero : z) {
        z_z.push_back((fs2 + zero) / (fs2 - zero));
    }

    for (const auto& pole : p) {
        p_z.push_back((fs2 + pole) / (fs2 - pole));
    }

    // Append zeros at Nyquist frequency
    for (int i = 0; i < degree; ++i) {
        z_z.push_back(-1.0);
    }

    // Compensate for gain change
    k_z = k * std::real(complexVectorProduct(fs2 - z) / complexVectorProduct(fs2 - p));
}

// Function to convert poles, zeros, and gain to transfer function coefficients
// Function to convert zeros, poles, and gain to polynomial coefficients
void zpk2tf(const std::vector<std::complex<double>>& z, const std::vector<std::complex<double>>& p, double k,
    std::vector<double>& b, std::vector<double>& a) {
    std::vector<double> b_temp;

    for (const auto& zero : z) {
        std::vector<double> temp = { -zero.real(), 1.0 };  // Coefficients for (s - zero)
        b_temp.insert(b_temp.end(), temp.begin(), temp.end());
    }

    if (b_temp.empty()) {
        b.push_back(k);
    }
    else {
        for (const auto& coeff : b_temp) {
            b.push_back(k * coeff);
        }
    }

    for (const auto& pole : p) {
        std::vector<double> temp = { -pole.real(), 1.0 };  // Coefficients for (s - pole)
        a.insert(a.end(), temp.begin(), temp.end());
    }
}

std::vector<double> iirfilter(int N, double Wn, std::string btype = "band", bool analog = false, std::string ftype = "butter", std::string output = "ba", double fs = 0.0) {
    std::vector<Complex> z;
    std::vector<Complex> p;
    double k = 0.0;

    buttap(N, p, z);

    // Pre-warp frequencies for digital filter design
    double warped = Wn;
    if (!analog) {
        fs = 2.0;
        warped = 2 * fs * std::tan(M_PI * Wn / fs);
    }

    lp2lp_zpk(p, z, warped);

    // Find discrete equivalent if necessary
    if (!analog) {
        bilinear_zpk(p, z, fs);
    }

    std::vector<double> num;
    std::vector<double> den;
    zpk2tf(p, z, k, num, den);

    // Placeholder return for demonstration purposes
    return num;
}



std::vector<double> butter(int N, double Wn, std::string btype = "low", bool analog = false, std::string output = "ba", double fs = 0.0) {
    return iirfilter(N, Wn, btype, analog, output, "butter", fs);
}

// Apply forward and backward filtering using a filter coefficients array
std::vector<double> filtfilt(const std::vector<double>& b, const std::vector<double>& a, const std::vector<double>& input) {
    std::vector<double> output(input.size(), 0.0);
    std::vector<double> state(b.size() - 1, 0.0);

    // Forward filtering
    for (std::size_t n = 0; n < input.size(); ++n) {
        output[n] = b[0] * input[n] + state[0];
        for (std::size_t i = 1; i < b.size(); ++i) {
            if (n >= i) {
                output[n] += b[i] * input[n - i] - a[i] * output[n - i];
            }
        }
        state.insert(state.begin(), output[n]);
        state.pop_back();
    }

    // Backward filtering
    std::reverse(output.begin(), output.end());
    state.assign(a.size() - 1, 0.0);
    for (std::size_t n = 0; n < output.size(); ++n) {
        double temp = output[n];
        output[n] = b[0] * temp + state[0];
        for (std::size_t i = 1; i < b.size(); ++i) {
            if (n >= i) {
                output[n] += b[i] * output[n - i] - a[i] * temp;
            }
        }
        state.insert(state.begin(), output[n]);
        state.pop_back();
    }

    std::reverse(output.begin(), output.end());
    return output;
}

int main() {
    int N = 4;
    std::vector<std::complex<double>> p;

    /*for (int m = -N + 1; m < N; m += 2) {
        double angle = -M_PI * m / (2 * N);
        p.push_back(std::polar(std::exp(1.0), angle));
    }*/
    p.push_back(std::complex<double>(0, 1));

    std::cout << "Poles: ";
    for (const auto& pole : p) {
        std::cout << pole << " ";
    }
    std::cout << std::endl;

    return 0;
}