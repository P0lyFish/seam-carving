#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>


using namespace cv;
using namespace std;


const Mat kx = (Mat_<double>(3, 3) << 0, 0, 0, -1, 0, 1, 0, 0, 0);
const Mat ky = (Mat_<double>(3, 3) << 0, -1, 0, 0, 0, 0, 0, 1, 0);
const int IMG_MAX_SIZE = 5000;

enum Orientation {
    VERTICAL,
    HORIZONTAL
};


int f[IMG_MAX_SIZE][IMG_MAX_SIZE], T[IMG_MAX_SIZE][IMG_MAX_SIZE];
mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());


string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}


Mat find_energy(const Mat &img) {
    Mat channels[3], channels_32f[3];
    split(img, channels);
    for (int i = 0; i < 3; ++i)
        channels[i].convertTo(channels_32f[i], CV_32F);

    Mat img_gray = channels_32f[0] + channels_32f[1] + channels_32f[2];
    Mat Fx, Fy;

    filter2D(img_gray, Fx, -1, kx, Point(-1, -1), 0, BORDER_REPLICATE);
    filter2D(img_gray, Fy, -1, ky, Point(-1, -1), 0, BORDER_REPLICATE);

    Mat energy = abs(Fx) + abs(Fy);
    energy.convertTo(energy, CV_32S);

    return energy;
}


int find_seam(const Mat &energy_org, vector<int> &path, const Orientation &ori) {
    Mat energy;

    if (ori == HORIZONTAL)
        energy = energy_org.t();
    else
        energy = energy_org;

    int source_h = energy.size[0], source_w = energy.size[1];

    for (int i = 0; i < source_w; ++i) {
        f[0][i] = energy.at<int>(0, i);
    }

    for (int i = 1; i < source_h; ++i) {
        for (int j = 0; j < source_w; ++j) {
            int cost = energy.at<int>(i, j);
            // cerr << cost << '\n';
            f[i][j] =f[i - 1][j] + cost;
            T[i][j] = 0;
            if (j > 0 && f[i - 1][j - 1] + cost < f[i][j]) {
                f[i][j] = f[i - 1][j - 1] + cost;
                T[i][j] = -1;
            }
            if (j + 1 < source_w && f[i - 1][j + 1] + cost < f[i][j]) {
                f[i][j] = f[i - 1][j + 1] + cost;
                T[i][j] = 1;
            }
        }
    }

    int cur_x = source_h - 1, cur_y = -1;
    for (int i = 0; i < source_w; ++i) {
        if (cur_y == -1 || f[cur_x][cur_y] > f[cur_x][i])
            cur_y = i;
    }

    int optimal_path_energy = f[cur_x][cur_y];

    path.push_back(cur_y);
    for (int i = 1; i < source_h; ++i) {
        cur_y += T[cur_x][cur_y];
        --cur_x;
        path.push_back(cur_y);
    }
    reverse(path.begin(), path.end());

    return optimal_path_energy;
}


Mat remove_seam(const Mat &img_org, const vector<int> &seam, const Orientation &ori) {
    Mat img;
    if (ori == VERTICAL)
        img = img_org;
    else
        img = img_org.t();

    int source_h = img.size[0], source_w = img.size[1];
    // cerr << source_h << ' ' << source_w << '\n';
    Mat img_reduced(source_h, source_w - 1, img.type());

    assert(source_h == seam.size());

    for (int i = 0; i < source_h; ++i) {
        int removing_idx = seam[i];

        if (removing_idx > 0) {
            Rect left_rect(0, i, removing_idx, 1);
            img(left_rect).copyTo(img_reduced(left_rect));
        }

        if (removing_idx + 1 < source_w) {
            // cerr << removing_idx + 1 << ' ' << source_w - removing_idx - 1 << '\n';
            Rect src_right_rect(removing_idx + 1, i, source_w - removing_idx - 1, 1);
            Rect dst_right_rect(removing_idx, i, source_w - removing_idx - 1, 1);
            img(src_right_rect).copyTo(img_reduced(dst_right_rect));
            // cout << removing_idx << '\n';
        }
    }

    if (ori == HORIZONTAL)
        return img_reduced.t();
    return img_reduced;
}


Mat add_seam(const Mat &img_org, const vector<int> &seam, const Orientation &ori) {
    Mat img;
    if (ori == VERTICAL)
        img = img_org;
    else
        img = img_org.t();

    int source_h = img.size[0], source_w = img.size[1];
    // cerr << source_h << ' ' << source_w << '\n';
    Mat img_increased(source_h, source_w + 1, img.type());

    assert(source_h == seam.size());

    for (int i = 0; i < source_h; ++i) {
        int adding_idx = seam[i];
        if (adding_idx == source_w - 1)
            --adding_idx;

        Rect left_rect(0, i, adding_idx + 1, 1);
        img(left_rect).copyTo(img_increased(left_rect));

        Rect src_right_rect(adding_idx + 1, i, source_w - adding_idx - 1, 1);
        Rect dst_right_rect(adding_idx + 2, i, source_w - adding_idx - 1, 1);
        img(src_right_rect).copyTo(img_increased(dst_right_rect));

        Vec3b left_pixel = img.at<Vec3b>(i, adding_idx);
        Vec3b right_pixel = img.at<Vec3b>(i, adding_idx + 1);

        vector<uchar> avg(3);

        for (int i = 0; i < 3; ++i) {
            avg[i] = char(((int)left_pixel[i] + (int)right_pixel[i]) / 2);
        }

        img_increased.at<Vec3b>(i, adding_idx + 1) = Vec3b(avg[0], avg[1], avg[2]);
    }

    if (ori == HORIZONTAL)
        return img_increased.t();
    return img_increased;
}


Mat reduce_size_by_k(const Mat &img_org, int k, const Orientation &ori) {
    Mat img_reduced = img_org.clone();
    vector<int> seam;

    while (k--) {
        seam.clear();

        Mat energy = find_energy(img_reduced);

        find_seam(energy, seam, ori);
        img_reduced = remove_seam(img_reduced, seam, ori);
    }

    return img_reduced;
}


Mat increase_size_by_k(const Mat &img_org, int k, const Orientation &ori) {
    Mat img_increased = img_org.clone();
    vector<int> seam;

    while (k--) {
        seam.clear();

        Mat energy = find_energy(img_increased);

        find_seam(energy, seam, ori);
        img_increased = add_seam(img_increased, seam, ori);
    }

    return img_increased;
}


Mat reduce_both_size(const Mat &img_org, int delta_r, int delta_c) {
    int source_h = img_org.size[0], source_w = img_org.size[1];

    assert(source_h > delta_r && source_w > delta_c);

    Mat img_reduced = img_org.clone(), energy;

    // vector<int> order;
    // for (int i = 0; i < delta_r; ++i)
    //     order.push_back(0);
    // for (int i = 0; i < delta_c; ++i)
    //     order.push_back(1);

    // shuffle(order.begin(), order.end(), rng);

    // for (auto c : order) {
    //     if (c == 0)
    //         img_reduced = reduce_size_by_k(img_reduced, 1, VERTICAL);
    //     else
    //         img_reduced = reduce_size_by_k(img_reduced, 1, HORIZONTAL);
    // }
    
    vector<int> vertical_seam, horizontal_seam;
    
    while (delta_r || delta_c) {
        if (delta_r == 0) {
             img_reduced = reduce_size_by_k(img_reduced, 1, HORIZONTAL);
             --delta_c;
        }
        else if (delta_c == 0) {
             img_reduced = reduce_size_by_k(img_reduced, 1, VERTICAL);
             --delta_r;
        }
        else {
            vertical_seam.clear();
            horizontal_seam.clear();
            energy = find_energy(img_reduced);
            int vertical_cost = find_seam(energy, vertical_seam, VERTICAL);
            int horizontal_cost = find_seam(energy, horizontal_seam, HORIZONTAL);

            if (vertical_cost < horizontal_cost) {
                // cerr << vertical_cost << ' ' << horizontal_cost << '\n';
                img_reduced = remove_seam(img_reduced, vertical_seam, VERTICAL);
                --delta_r;
            }
            else {
                img_reduced = remove_seam(img_reduced, horizontal_seam, HORIZONTAL);
                --delta_c;
            }
        }
    }

    cerr << img_reduced.size[0] << ' ' << img_reduced.size[1] << '\n';
    return img_reduced;
}


int main(int argc, char** argv) {
    Mat img = imread("../data/test03.jpg");
    if (img.empty()) {
        cout << "Image not found!\n";
    }
    else {
        imwrite("test03_ours.png", reduce_both_size(img, 700, 200));
        waitKey(0);
    }

    return 0;
}
