#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

/**
 * Performs 1D K-Means clustering on grayscale pixel intensities.
 * Input: grayscale image (CV_8U)
 * Output: segmented grayscale image where each pixel is replaced by its cluster mean.
 */
Mat kmeans1D(const Mat &gray, int K, int maxIters, double epsilon) {
    int rows = gray.rows;
    int cols = gray.cols;
    int N = rows * cols;

    // Flatten pixels into vector<double>
    vector<double> pixels;
    pixels.reserve(N);
    for (int r = 0; r < rows; ++r) {
        const uchar* ptr = gray.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c) {
            pixels.push_back(static_cast<double>(ptr[c]));
        }
    }

    // Initialize means: pick K pixels evenly spaced in sorted range
    double minVal, maxVal;
    minMaxLoc(gray, &minVal, &maxVal);
    vector<double> means(K);
    for (int k = 0; k < K; ++k) {
        means[k] = minVal + (maxVal - minVal) * (k + 0.5) / K;
    }

    vector<int> labels(N, 0);
    vector<double> sums(K);
    vector<int> counts(K);

    for (int iter = 0; iter < maxIters; ++iter) {
        // Assignment step
        for (int i = 0; i < N; ++i) {
            double bestDist = numeric_limits<double>::max();
            int bestK = 0;
            for (int k = 0; k < K; ++k) {
                double d = abs(pixels[i] - means[k]);
                if (d < bestDist) {
                    bestDist = d;
                    bestK = k;
                }
            }
            labels[i] = bestK;
        }

        // Initialize sums & counts
        fill(sums.begin(), sums.end(), 0.0);
        fill(counts.begin(), counts.end(), 0);

        // Update step: accumulate
        for (int i = 0; i < N; ++i) {
            int k = labels[i];
            sums[k] += pixels[i];
            counts[k]++;
        }

        // Compute new means and check convergence
        double maxShift = 0.0;
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                double newMean = sums[k] / counts[k];
                maxShift = max(maxShift, abs(newMean - means[k]));
                means[k] = newMean;
            }
        }
        if (maxShift < epsilon) {
            cout << "Converged in " << iter + 1 << " iterations.\n";
            break;
        }
    }

    // Reconstruct segmented image
    Mat seg(rows, cols, CV_8U);
    int idx = 0;
    for (int r = 0; r < rows; ++r) {
        uchar* ptr = seg.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c) {
            ptr[c] = static_cast<uchar>(round(means[labels[idx++]]));
        }
    }

    return seg;
}

int main() {
    string inputPath = "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/test.jpg";
    int K = 3;
    string outputPath ="D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/out_test.jpg";
    int maxIters = 100;
    double epsilon = 1e-4;

    // Read and convert to grayscale
    Mat img = imread(inputPath);
    if (img.empty()) {
        cerr << "Error: Could not open or find the image.\n";
        return -1;
    }
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Run K-Means segmentation
    Mat segmented = kmeans1D(gray, K, maxIters, epsilon);

    // Save result
    imwrite(outputPath, segmented);
    cout << "Segmented image saved to " << outputPath << endl;

    return 0;
}
