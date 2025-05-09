#include <opencv2/opencv.hpp>
#include <omp.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

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


Mat kmeans1D_OpenMP(vector<double> &pixels,int rows,int cols,int K, int maxIters = 100, double epsilon = 1e-4) {
    int N= rows * cols;

    // Initialize means evenly over intensity range
    double minVal, maxVal;
    findMinMax(pixels, minVal, maxVal);
    vector<double> means(K);
    for (int k= 0; k < K; ++k) {
        means[k] = minVal + (maxVal - minVal) * (k + 0.5) / K;
    }

    vector<int> labels(N);
    vector<double> sums(K);
    vector<int> counts(K);

    for (int iter = 0; iter < maxIters; ++iter) {
        // Assignment step (parallel)
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
           // int tid = omp_get_thread_num();
           // cout << "hi " << tid <<endl;
            double bestDist = numeric_limits<double>::max();
            int bestK = 0;
            for (int k = 0; k < K; ++k) {
                double d = fabs(pixels[i] - means[k]);
                if (d < bestDist) {
                    bestDist = d;
                    bestK = k;
                }
            }
            labels[i] = bestK;
        }

        // Reset global sums & counts
        fill(sums.begin(), sums.end(), 0.0);
        fill(counts.begin(), counts.end(), 0);

        // Local accumulation with reduction via critical
        #pragma omp parallel
        {
            vector<double> local_sums(K, 0.0);
            vector<int> local_counts(K, 0);
            #pragma omp for schedule(static)
            for (int i = 0; i < N; ++i) {
                int k = labels[i];
                local_sums[k] += pixels[i];
                local_counts[k]++;
            }
            #pragma omp critical
            {
                for (int k = 0; k < K; ++k) {
                    sums[k] += local_sums[k];
                    counts[k] += local_counts[k];
                }
            }
        }

        // Update means and check convergence (serial)
        double maxShift = 0.0;
        for (int k = 0; k < K; ++k) {
            if (counts[k] > 0) {
                double newMean = sums[k] / counts[k];
                maxShift = max(maxShift, fabs(newMean - means[k]));
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
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        uchar* ptr = seg.ptr<uchar>(r);
        int base = r * cols;
        for (int c = 0; c < cols; ++c) {
            ptr[c] = static_cast<uchar>(round(means[labels[base + c]]));
        }
    }
    return seg;
}

int main() {
    string inputPath= "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/test.jpg";
    int K = 3;
    string outputPath= "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/out_test_openMP.jpg";

    Mat img = imread(inputPath);
    if (img.empty()) {
        cerr << "Error: Could not open or find the image.\n";
        return -1;
    }
    Mat gray; 
    vector<double> pixels;
    convertToGray(img, gray, pixels);

    //Start timer0 to calculate computational time 
    auto t0 = Clock::now();

    // Run OpenMP K-Means
    Mat segmented = kmeans1D_OpenMP(pixels,gray.rows,gray.cols,K);

    //Start timer1 to calculate computational time 
    auto t1 = Clock::now();
    chrono::duration<double> elapsed = t1 - t0;
    cout << "Segmentation took " 
         << elapsed.count() << " seconds.\n";

    imwrite(outputPath, segmented);
    cout << "Segmented image saved to " << outputPath << endl;
    return 0;
}
