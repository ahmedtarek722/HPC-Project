#include <mpi.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>

using namespace cv;
using namespace std;

/**
 * Perform K-Means on each process’s local pixel chunk, synchronizing
 * cluster means across all ranks via collective operations.
 *
 * @param localPixels   flattened grayscale intensities owned by this rank
 * @param K             number of clusters
 * @param maxIters      maximum iterations
 * @param epsilon       convergence threshold
 * @param comm          MPI communicator
 * @return              a vector<uchar> of the same size as localPixels,
 *                      containing the segmented intensities
 */
vector<uchar> kmeans1D_mpi(const vector<uchar>& localPixels,
                           int                          K,
                           int                          maxIters,
                           double                       epsilon,
                           MPI_Comm                     comm)
{
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int localCount = (int)localPixels.size();
    // Convert to double
    vector<double> pixels(localCount);
    for (int i = 0; i < localCount; ++i)
        pixels[i] = static_cast<double>(localPixels[i]);

    // Initialize means on rank 0 and broadcast
    vector<double> means(K);
    if (rank == 0) {
        // find global min/max by gathering from rank 0’s full image if needed
        double minVal = *min_element(pixels.begin(), pixels.end());
        double maxVal = *max_element(pixels.begin(), pixels.end());
        for (int k = 0; k < K; ++k)
            means[k] = minVal + (maxVal - minVal) * (k + 0.5) / K;
    }
    MPI_Bcast(means.data(), K, MPI_DOUBLE, 0, comm);

    vector<int>    labels(localCount);
    vector<double> localSums(K);
    vector<int>    localCounts(K);
    vector<double> globalSums(K);
    vector<int>    globalCounts(K);

    for (int iter = 0; iter < maxIters; ++iter) {
        // 1) Assignment
        for (int i = 0; i < localCount; ++i) {
            double bestDist = numeric_limits<double>::max();
            int    bestK    = 0;
            for (int k = 0; k < K; ++k) {
                double d = fabs(pixels[i] - means[k]);
                if (d < bestDist) {
                    bestDist = d;
                    bestK    = k;
                }
            }
            labels[i] = bestK;
        }

        // 2) Local accumulate
        fill(localSums.begin(),   localSums.end(),   0.0);
        fill(localCounts.begin(), localCounts.end(), 0);
        for (int i = 0; i < localCount; ++i) {
            int c = labels[i];
            localSums[c]   += pixels[i];
            localCounts[c] += 1;
        }

        // 3) Global reduction
        MPI_Allreduce(localSums.data(),   globalSums.data(),   K, MPI_DOUBLE, MPI_SUM, comm);
        MPI_Allreduce(localCounts.data(), globalCounts.data(), K, MPI_INT,    MPI_SUM, comm);

        // 4) Update means & track maximum shift
        double localMaxShift = 0.0;
        for (int k = 0; k < K; ++k) {
            if (globalCounts[k] > 0) {
                double newMean = globalSums[k] / globalCounts[k];
                localMaxShift  = max(localMaxShift, fabs(newMean - means[k]));
                means[k]       = newMean;
            }
        }

        // 5) Check convergence across all ranks
        double globalMaxShift;
        MPI_Allreduce(&localMaxShift, &globalMaxShift, 1, MPI_DOUBLE, MPI_MAX, comm);
        if (globalMaxShift < epsilon) {
            if (rank == 0) cout << "Converged in " << iter+1 << " iterations.\n";
            break;
        }
    }

    // Reconstruct segmented slice
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

    // Paths and parameters
    string inputPath  = "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/test.jpg";
    string outputPath = "D:/AINSHAMS_SEMESTERS/semester 10/High performance computing/project/out_test_MPI.jpg";
    int    K          = 3;
    int    maxIters   = 100;
    double epsilon    = 1e-4;

    // Only rank 0 loads and flattens the image
    int rows = 0, cols = 0;
    vector<uchar> allPixels;
    if (rank == 0) {
        Mat img = imread(inputPath);
        Mat gray;
        cvtColor(img, gray, COLOR_BGR2GRAY);
        rows = gray.rows;
        cols = gray.cols;

        allPixels.resize(rows * cols);
        for (int r = 0; r < rows; ++r) {
            memcpy(allPixels.data() + r*cols, gray.ptr<uchar>(r), cols);
        }
    }

    // Broadcast image dimensions to all ranks
    MPI_Bcast(&rows, 1, MPI_INT,  0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT,  0, MPI_COMM_WORLD);

    // Compute scatter counts & displacements on rank 0
    vector<int> sendCounts(size), displs(size);
    if (rank == 0) {
        int base = rows / size;
        int rem  = rows % size;
        int offset = 0;
        for (int i = 0; i < size; ++i) {
            int r = base + (i < rem ? 1 : 0);
            sendCounts[i] = r * cols;
            displs[i]     = offset;
            offset       += sendCounts[i];
        }
    }

    // Synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // Scatter the row‐wise chunks
    int localCount;
    MPI_Scatter(sendCounts.data(), 1, MPI_INT,
                &localCount,      1, MPI_INT,
                0, MPI_COMM_WORLD);

    vector<uchar> localPixels(localCount);
    MPI_Scatterv(allPixels.data(), sendCounts.data(), displs.data(),
                 MPI_UNSIGNED_CHAR,
                 localPixels.data(), localCount, MPI_UNSIGNED_CHAR,
                 0, MPI_COMM_WORLD);

    // Perform the distributed K-Means
    auto localSeg = kmeans1D_mpi(localPixels, K, maxIters, epsilon, MPI_COMM_WORLD);

    // Gather the segmented chunks back
    vector<uchar> fullSeg;
    if (rank == 0) fullSeg.resize(rows * cols);

    MPI_Gatherv(localSeg.data(),    localCount, MPI_UNSIGNED_CHAR,
                fullSeg.data(),      sendCounts.data(), displs.data(),
                MPI_UNSIGNED_CHAR,   0, MPI_COMM_WORLD);

    // Synchronize and stop timing
    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    if (rank == 0) {
        // Reconstruct and save the segmented image
        Mat out(rows, cols, CV_8U, fullSeg.data());
        imwrite(outputPath, out);
        cout << "Segmented image saved to " << outputPath << "\n";

        // Report total parallel runtime
        cout << "Total MPI segmentation time: "
             << (t_end - t_start) << " seconds." << endl;
    }

    MPI_Finalize();
    return 0;
}
