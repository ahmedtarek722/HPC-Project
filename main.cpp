#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>

using namespace cv;
using namespace std;
using Clock = chrono::high_resolution_clock;


void convertToGray(const Mat& img, Mat& gray, vector<double>& pixels) {
    int rows = img.rows, cols = img.cols;
    gray.create(rows, cols, CV_8U);
    pixels.clear();
    pixels.reserve(rows * cols);
    for (int r = 0; r < rows; ++r) {
        // pointer to BGR triples
        const Vec3b* bgrPtr = img.ptr<Vec3b>(r);
        // pointer to output gray row
        uchar* grayPtr = gray.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c) {
            // standard_luminance_formula
            double lum = 0.299 * bgrPtr[c][2]+ 0.587 * bgrPtr[c][1]+ 0.114 * bgrPtr[c][0];
            grayPtr[c] = static_cast<uchar>(round(lum));
            pixels.push_back(lum);
        }
    }
}


void findMinMax(const vector<double>& pixels, double& minVal, double& maxVal) {
    minVal= numeric_limits<double>::max();
    maxVal= numeric_limits<double>::lowest();
    for (double v : pixels) {
        if(v<minVal)minVal= v;
        if(v>maxVal)maxVal= v;
    }
}

Mat kmeans1D(int K,const vector<double>& pixels_in,int rows, int cols,int maxIters, double epsilon) {
    int N = rows * cols;
    
    vector<double> means(K);
    // Initialize means evenly in the [minVal, maxVal] range
    double minVal, maxVal;
    findMinMax(pixels_in, minVal, maxVal);
    for (int k = 0; k < K; ++k) {
        means[k] = minVal + (maxVal - minVal) * (k + 0.5) / K;
    }

    vector<int> labels(N);
    vector<double> sums(K);
    vector<int> counts(K);
    // K‐Means loop
    for (int iter = 0; iter < maxIters; ++iter) {
        // Assignment step -> to find the shortest distance to get the optimum cluster
        for (int i = 0; i < N; ++i) {
            double bestDist = numeric_limits<double>::max();
            int bestK = 0;
            for (int k = 0; k < K; ++k) {
                double d = fabs(pixels_in[i] - means[k]);
                if (d < bestDist) {
                    bestDist = d;
                    bestK = k;
                }
            }
            labels[i] = bestK;
        }

        // Zero‐out accumulators
        fill(sums.begin(), sums.end(), 0.0);
        fill(counts.begin(), counts.end(), 0);

        // Update step: accumulate sums and counts
        for (int i = 0; i < N; ++i) {
            int k = labels[i];
            sums[k] += pixels_in[i];
            counts[k]++;
        }

        // Recompute means and check convergence
        double maxShift = 0.0;
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                double newMean = sums[k] / counts[k];
                maxShift = max(maxShift, fabs(newMean - means[k]));
                means[k] = newMean;
            }
        }
        if (maxShift < epsilon) {
            cout << "Converged in " << iter+1 << " iterations.\n";
            break;
        }
    }

    // Build output segmented image
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
    string inputPath  = "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/test.jpg";
    string outputPath = "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/out_test_seq.jpg";
    int K = 3;
    int maxIters = 100;
    double epsilon = 1e-4; //equal to 0.0001

    Mat img = imread(inputPath);
    if (img.empty()) {
        cerr << "Error: Could not open or find the image.\n";
        return -1;
    }

    // Manual grayscale conversion + flatten
    Mat gray;
    vector<double> pixels;
    convertToGray(img, gray, pixels);
    // Run K-Means
    auto t0 = Clock::now();
    Mat segmented = kmeans1D(K,pixels,gray.rows, gray.cols,maxIters, epsilon);
    auto t1 = Clock::now();
    chrono::duration<double> elapsed = t1 - t0;
    cout << "Segmentation took " << elapsed.count() << " seconds.\n";

    imwrite(outputPath, segmented);
    cout << "Segmented image saved to " << outputPath << endl;
    return 0;
}
