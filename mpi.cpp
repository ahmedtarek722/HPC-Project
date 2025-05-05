#include <opencv2/opencv.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace cv;
using namespace std;

/**
 * Performs 1D K-Means on a local slice of the image.
 * All processes jointly compute the global means via MPI_Allreduce.
 */
Mat kmeans1D_mpi(const Mat& localGray, int K,
                 int maxIters = 100, double epsilon = 1e-4)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int localRows = localGray.rows;
    int cols      = localGray.cols;
    int localN    = localRows * cols;

    // Flatten local pixels into a vector<double>
    vector<double> pixels(localN);
    for(int r=0; r<localRows; ++r)
        for(int c=0; c<cols; ++c)
            pixels[r*cols + c] = localGray.at<uchar>(r,c);

    // Step 1: initialize means on rank 0, then broadcast
    vector<double> means(K);
    if(rank == 0) {
        double minVal, maxVal;
        minMaxLoc(localGray, &minVal, &maxVal);
        // Find global min/max across all ranks
        double gMin, gMax;
        MPI_Allreduce(&minVal, &gMin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&maxVal, &gMax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        for(int k=0; k<K; ++k)
            means[k] = gMin + (gMax - gMin) * (k + 0.5) / K;
    }
    MPI_Bcast(means.data(), K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    vector<int>    labels(localN);
    vector<double> localSums(K);
    vector<int>    localCounts(K);

    // Main K-Means loop
    for(int iter=0; iter<maxIters; ++iter) {
        // — Assignment step (local only)
        for(int i=0; i<localN; ++i) {
            double bestD = numeric_limits<double>::max();
            int bestK = 0;
            for(int k=0; k<K; ++k) {
                double d = fabs(pixels[i] - means[k]);
                if(d < bestD) { bestD = d; bestK = k; }
            }
            labels[i] = bestK;
        }

        // — Compute local sums & counts
        fill(localSums.begin(), localSums.end(), 0.0);
        fill(localCounts.begin(), localCounts.end(), 0);
        for(int i=0; i<localN; ++i) {
            int c = labels[i];
            localSums[c]   += pixels[i];
            localCounts[c] += 1;
        }

        // — Reduce to global sums & counts
        vector<double> globalSums(K);
        vector<int>    globalCounts(K);
        MPI_Allreduce(localSums.data(),   globalSums.data(),   K, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(localCounts.data(), globalCounts.data(), K, MPI_INT,    MPI_SUM, MPI_COMM_WORLD);

        // — Update means and check convergence
        double maxShift = 0.0;
        for(int k=0; k<K; ++k) {
            if(globalCounts[k] > 0) {
                double newMean = globalSums[k] / globalCounts[k];
                maxShift = max(maxShift, fabs(newMean - means[k]));
                means[k] = newMean;
            }
        }
        if(maxShift < epsilon) {
            if(rank==0) 
                cout << "Converged in " << iter+1 << " iterations.\n";
            break;
        }
    }

    // Reconstruct local segmented image
    Mat localSeg(localRows, cols, CV_8U);
    for(int r=0; r<localRows; ++r) {
        for(int c=0; c<cols; ++c) {
            int idx = r*cols + c;
            localSeg.at<uchar>(r,c) = 
                static_cast<uchar>(round(means[ labels[idx] ]));
        }
    }
    return localSeg;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Adjust these paths or pass as arguments
    const string inputPath  = "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/test.jpg";
    const string outputPath = "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/out_mpi_test.jpg";
    const int    K          = 3;
    const int    maxIters   = 100;
    const double epsilon    = 1e-4;

    Mat gray;
    int rows, cols;

    // Rank 0 reads and broadcasts dimensions
    if(rank == 0) {
        Mat img = imread(inputPath);
        if(img.empty()) {
            cerr << "Error: could not open " << inputPath << "\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        cvtColor(img, gray, COLOR_BGR2GRAY);
        rows = gray.rows;
        cols = gray.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute how many rows each rank gets
    vector<int> sendcounts(size), displs(size);
    int base = rows / size, rem = rows % size, offset = 0;
    for(int i=0; i<size; ++i) {
        int r = base + (i < rem ? 1 : 0);
        sendcounts[i] = r * cols;
        displs[i]      = offset * cols;
        offset        += r;
    }
    int localCount = sendcounts[rank];

    // Scatter the grayscale pixels
    vector<uchar> localBuf(localCount);
    if(rank == 0) {
        vector<uchar> flat(rows * cols);
        for(int r=0; r<rows; ++r)
            for(int c=0; c<cols; ++c)
                flat[r*cols + c] = gray.at<uchar>(r,c);

        MPI_Scatterv(flat.data(), sendcounts.data(), displs.data(),
                     MPI_UNSIGNED_CHAR,
                     localBuf.data(), localCount, MPI_UNSIGNED_CHAR,
                     0, MPI_COMM_WORLD);
    } else {
        MPI_Scatterv(nullptr, nullptr, nullptr, MPI_UNSIGNED_CHAR,
                     localBuf.data(), localCount, MPI_UNSIGNED_CHAR,
                     0, MPI_COMM_WORLD);
    }

    // Build local Mat and run parallel k-means
    int localRows = localCount / cols;
    Mat localGray(localRows, cols, CV_8U, localBuf.data());
    Mat localSeg = kmeans1D_mpi(localGray, K, maxIters, epsilon);

    // Gather segmented chunks back to rank 0
    vector<uchar> localSegBuf(localCount);
    memcpy(localSegBuf.data(), localSeg.data, localCount);
    vector<uchar> fullSeg;
    if(rank == 0)
        fullSeg.resize(rows * cols);

    MPI_Gatherv(localSegBuf.data(), localCount, MPI_UNSIGNED_CHAR,
                rank == 0 ? fullSeg.data() : nullptr,
                sendcounts.data(), displs.data(),
                MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

    // Rank 0 writes the final image
    if(rank == 0) {
        Mat seg(rows, cols, CV_8U, fullSeg.data());
        imwrite(outputPath, seg);
        cout << "Segmented image saved to " << outputPath << "\n";
    }

    MPI_Finalize();
    return 0;
}
