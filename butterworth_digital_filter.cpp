#include <iostream>
#include <vector>
#include <DigitalFilters/DigitalFilters.h>  // Include the Digital Filters library

//Ê¹ÓÃ C++ µÄ Digital Filters ¿â

int main() {
    // Define the filter parameters
    int order = 4;
    double cutoff = 0.2;
    double samplingFreq = 100.0;

    // Create a low-pass Butterworth filter
    DigitalFilters::FilterCoeffs coeffs = DigitalFilters::Design::Butterworth::LowPass(order, cutoff, samplingFreq);
    DigitalFilters::Filter filter(coeffs);

    // Input signal
    std::vector<double> inputSignal = { 1.0, 2.0, 3.0, 4.0, 5.0 };

    // Apply forward and backward filtering
    std::vector<double> filteredSignal = filter.Apply(inputSignal);
    std::reverse(filteredSignal.begin(), filteredSignal.end());
    filteredSignal = filter.Apply(filteredSignal);
    std::reverse(filteredSignal.begin(), filteredSignal.end());

    // Print filtered signal
    std::cout << "Filtered signal: ";
    for (const auto& value : filteredSignal) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}
