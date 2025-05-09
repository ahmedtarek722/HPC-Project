#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

using namespace cv;
using namespace std;

// Convert a BGR image to gray + fill a double‐precision pixel array
void convertToGray(const Mat& img, Mat& gray, vector<double>& pixels) {
    int rows = img.rows, cols = img.cols;
    gray.create(rows, cols, CV_8U);
    pixels.clear();
    pixels.reserve(rows * cols);
    for (int r = 0; r < rows; ++r) {
        const Vec3b* bgrPtr = img.ptr<Vec3b>(r);
        uchar* gPtr   = gray.ptr<uchar>(r);
        for (int c = 0; c < cols; ++c) {
            double lum = 0.299 * bgrPtr[c][2]+ 0.587 * bgrPtr[c][1]+ 0.114 * bgrPtr[c][0];
            gPtr[c] = static_cast<uchar>(round(lum));
            pixels.push_back(lum);
        }
    }
}

// Simple linear scan to find min/max in a double array
void findMinMax(const vector<double>& pixels, double& minVal, double& maxVal) {
    minVal = numeric_limits<double>::max();
    maxVal = numeric_limits<double>::lowest();
    for (double v : pixels) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
}

vector<uchar> kmeans1D_mpi(const vector<uchar>& localPixels,int K, int maxIters, double epsilon,MPI_Comm comm, int rank)
{
    int localCount = int(localPixels.size());
    // convert uchar→double
    vector<double> pixels(localCount);
    for (int i = 0; i < localCount; ++i)
        pixels[i] = static_cast<double>(localPixels[i]);

    // --- INITIALIZE MEANS ---
    vector<double> means(K);
    if (rank == 0) {
        // use findMinMax instead of std::min_element / max_element
        double minVal, maxVal;
        findMinMax(pixels, minVal, maxVal);
        for (int k = 0; k < K; ++k)
            means[k] = minVal + (maxVal - minVal) * (k + 0.5) / K;
    }
    MPI_Bcast(means.data(), K, MPI_DOUBLE, 0, comm);

    // buffers for clustering
    vector<int> labels(localCount);
    vector<double> localSums(K), globalSums(K);
    vector<int> localCounts(K), globalCounts(K);

    for (int iter = 0; iter < maxIters; ++iter) {
        // 1) ASSIGNMENT
        for (int i = 0; i < localCount; ++i) {
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

        // 2) LOCAL ACCUMULATION
        fill(localSums.begin(), localSums.end(), 0.0);
        fill(localCounts.begin(), localCounts.end(), 0);
        for (int i = 0; i < localCount; ++i) {
            int c= labels[i];
            localSums[c] += pixels[i];
            localCounts[c]++;
        }

        // 3) GLOBAL REDUCTION
        MPI_Allreduce(localSums.data(),globalSums.data(),K, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(localCounts.data(), globalCounts.data(), K, MPI_INT,MPI_SUM, comm);

        // 4) UPDATE & TRACK SHIFT
        double localMaxShift = 0.0;
        for (int k = 0; k < K; ++k) {
            if (globalCounts[k]> 0) {
                double newMean = globalSums[k] / globalCounts[k];
                localMaxShift = max(localMaxShift, fabs(newMean - means[k]));
                means[k] = newMean;
            }
        }

        // 5) CHECK CONVERGENCE
        double globalMaxShift;
        MPI_Allreduce(&localMaxShift, &globalMaxShift, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (globalMaxShift < epsilon) {
            if (rank == 0)
                cout << "Converged in " << iter+1 << " iterations.\n";
            break;
        }
    }

    // 6) RECONSTRUCT LOCAL SEGMENT
    vector<uchar> localSeg(localCount);
    for (int i = 0; i < localCount; ++i)
        localSeg[i] = static_cast<uchar>(round(means[labels[i]]));

    return localSeg;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    string inputPath= "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/test.jpg";
    int K = 3;
    string outputPath= "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/out_test_openMP.jpg";
    const int maxIters= 100;
    const double epsilon= 1e-4;

    int rows = 0, cols = 0;
    vector<uchar> allPixels;

    if (rank == 0) {
        // load & convert to gray using your function
        Mat img = imread(inputPath);
        if (img.empty()) {
            cerr << "Error: cannot read " << inputPath << endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        Mat gray;
        vector<double> dummy;   // we don't need the double‐pixel buffer here
        convertToGray(img, gray, dummy);

        rows = gray.rows;
        cols = gray.cols;
        allPixels.resize(rows * cols);
        for (int r = 0; r < rows; ++r)
            memcpy(allPixels.data() + r*cols,
                   gray.ptr<uchar>(r),
                   cols * sizeof(uchar));
    }

    // broadcast dims
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // prepare scatter
    vector<int> sendCounts(size), displs(size);
    if (rank == 0) {
        int base = rows / size, rem = rows % size, offset = 0;
        for (int i = 0; i < size; ++i) {
            int r = base + (i < rem ? 1 : 0);
            sendCounts[i] = r * cols;
            displs[i] = offset;
            offset+= sendCounts[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // scatter
    int localCount;
    MPI_Scatter(sendCounts.data(), 1, MPI_INT,&localCount,1, MPI_INT,0, MPI_COMM_WORLD);

    vector<uchar> localPixels(localCount);
    MPI_Scatterv(allPixels.data(), sendCounts.data(), displs.data(),MPI_UNSIGNED_CHAR,localPixels.data(), localCount,MPI_UNSIGNED_CHAR,0, MPI_COMM_WORLD);

    // k-means
    auto localSeg = kmeans1D_mpi(localPixels, K, maxIters, epsilon,MPI_COMM_WORLD, rank);

    // gather
    vector<uchar> fullSeg;
    if (rank == 0)
        fullSeg.resize(rows * cols);

    MPI_Gatherv(localSeg.data(),localCount, MPI_UNSIGNED_CHAR,fullSeg.data(),sendCounts.data(),displs.data(),MPI_UNSIGNED_CHAR,0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        Mat out(rows, cols, CV_8U, fullSeg.data());
        imwrite(outputPath, out);
        cout << "Saved: " << outputPath << "\n"<< "Total MPI time: " << (t1 - t0) << " s\n";
    }

    MPI_Finalize();
    return 0;
}
