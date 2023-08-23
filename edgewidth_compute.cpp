#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include <algorithm>

using namespace std;
using namespace cv;

#define M_PI 3.14159265358979323846


//�ж�һ����ά�������Ƿ��з���ֵ
bool any(const std::vector<std::vector<float>>& matrix) {
    for (const auto& row : matrix) {
        for (float value : row) {
            if (value != 0.0) {
                return true;
            }
        }
    }
    return false;
}


//x���������ݶȽǶ�
std::vector<std::vector<float>> quantizeAngles_x(const std::vector<std::vector<float>>& edge_angles) {
    std::vector<std::vector<float>> quantized_angles(edge_angles.size(), std::vector<float>(edge_angles[0].size(), 0.0));

    for (int i = 0; i < edge_angles.size(); ++i) {
        for (int j = 0; j < edge_angles[i].size(); ++j) {
            quantized_angles[i][j] = 180.0 * std::round(edge_angles[i][j] / 180.0);
        }
    }

    return quantized_angles;
}


//y���������ݶȽǶ�
std::vector<std::vector<float>> quantizeAngles_y(const std::vector<std::vector<float>>& edge_angles) {
    // ����һ���� edge_angles ��ͬ��С�Ķ�ά����
    std::vector<std::vector<float>> quantized_angles(edge_angles.size(), std::vector<float>(edge_angles[0].size()));

    for (int i = 0; i < edge_angles.size(); ++i) {
        for (int j = 0; j < edge_angles[i].size(); ++j) {
            quantized_angles[i][j] = 180.0 * std::round((edge_angles[i][j] - 90.0) / 180.0) + 90.0;
        }
    }

    return quantized_angles;
}


vector<vector<float>> marziliano_method_h(const vector<vector<int>>& edges, const vector<vector<int>>& image) {
    // ���ڴ洢��Ե��ȵľ���
    vector<vector<float>> edge_widths(image.size(), vector<float>(image[0].size(), 0.0));

    // ����ͼ����ݶ�
    vector<vector<float>> gradient_y(image.size(), vector<float>(image[0].size(), 0.0));
    vector<vector<float>> gradient_x(image.size(), vector<float>(image[0].size(), 0.0));
    for (int row = 1; row < image.size() - 1; row++) {
        for (int col = 1; col < image[0].size() - 1; col++) {
            gradient_y[row][col] = image[row + 1][col] - image[row - 1][col];
            gradient_x[row][col] = image[row][col + 1] - image[row][col - 1];
        }
    }

    // �����Ե�ĽǶ�
    vector<vector<float>> edge_angles(image.size(), vector<float>(image[0].size(), 0.0));
    for (int row = 0; row < image.size(); row++) {
        for (int col = 0; col < image[0].size(); col++) {
            if (gradient_x[row][col] != 0.0) {
                edge_angles[row][col] = atan2(gradient_y[row][col], gradient_x[row][col]) * (180 / M_PI);
            }
            else if (gradient_x[row][col] == 0.0 && gradient_y[row][col] == 0.0) {
                edge_angles[row][col] = 0.0;
            }
            else if (gradient_x[row][col] == 0.0 && gradient_y[row][col] == M_PI / 2) {
                edge_angles[row][col] = 90.0;
            }
        }
    }

    // �����Ե���
    float edge_ignore = 0.1;
    float finetune_number = 0.0;
    int finetune = 0;
    int margin = 0;

    if (any(edge_angles)) {
        std::vector<std::vector<float>> quantized_angles = quantizeAngles_x(edge_angles);
        for (int row = 1; row < image.size() - 1; row++) {
            for (int col = edge_ignore * image[0].size(); col < (1 - edge_ignore) * (image[0].size() - 1); col++) {
                if (edges[row][col] == 1) {
                    if (quantized_angles[row][col] == 180.0 || quantized_angles[row][col] == -180.0) {
                        for (int margin = 0; margin <= 100; margin++) {
                            int inner_border = col - 1 - margin;
                            int outer_border = col - 2 - margin;
                            if (outer_border < 0 || (image[row][outer_border] - image[row][inner_border]) <= -1) {
                                outer_border += 1;
                                for (int finetune = 0; finetune <= margin; finetune++) {
                                    inner_border += 1;
                                    if ((image[row][outer_border] - image[row][inner_border]) > finetune_number) {
                                        break;
                                    }
                                }
                                margin = margin - finetune;
                                break;
                            }
                        }

                        float width_left = margin + 1;

                        for (int margin = 0; margin <= 100; margin++) {
                            int inner_border = col + 1 + margin;
                            int outer_border = col + 2 + margin;
                            if (outer_border >= image[0].size() || (image[row][outer_border] - image[row][inner_border]) >= 1) {
                                outer_border -= 1;
                                for (int finetune = 0; finetune <= margin; finetune++) {
                                    inner_border -= 1;
                                    if ((image[row][inner_border] - image[row][outer_border]) > finetune_number) {
                                        break;
                                    }
                                }
                                margin = margin - finetune;
                                break;
                            }
                        }

                        float width_right = margin + 1;

                        edge_widths[row][col] = width_left + width_right;
                    }

                    if (quantized_angles[row][col] == 0.0) {
                        for (int margin = 0; margin <= 100; margin++) {
                            int inner_border = col - 1 - margin;
                            int outer_border = col - 2 - margin;
                            if (outer_border < 0 || (image[row][outer_border] - image[row][inner_border]) >= 1) {
                                outer_border += 1;
                                for (int finetune = 0; finetune <= margin; finetune++) {
                                    inner_border += 1;
                                    if ((image[row][inner_border] - image[row][outer_border]) > finetune_number) {
                                        break;
                                    }
                                }
                                margin = margin - finetune;
                                break;
                            }
                        }

                        float width_left = margin + 1;

                        for (int margin = 0; margin <= 100; margin++) {
                            int inner_border = col + 1 + margin;
                            int outer_border = col + 2 + margin;
                            if (outer_border >= image[0].size() || (image[row][outer_border] - image[row][inner_border]) <= -1) {
                                outer_border -= 1;
                                for (int finetune = 0; finetune <= margin; finetune++) {
                                    inner_border -= 1;
                                    if ((image[row][outer_border] - image[row][inner_border]) > finetune_number) {
                                        break;
                                    }
                                }
                                margin = margin - finetune;
                                break;
                            }
                        }

                        float width_right = margin + 1;

                        edge_widths[row][col] = width_right + width_left;
                    }
                }
            }
        }
    }

    return edge_widths;
}


vector<vector<float>> marziliano_method_v(const vector<vector<int>>& edges, const vector<vector<int>>& image) {
    // ���ڴ洢��Ե��ȵľ���
    vector<vector<float>> edge_widths(image.size(), vector<float>(image[0].size(), 0.0));

    // ����ͼ����ݶ�
    vector<vector<float>> gradient_y(image.size(), vector<float>(image[0].size(), 0.0));
    vector<vector<float>> gradient_x(image.size(), vector<float>(image[0].size(), 0.0));
    for (int row = 1; row < image.size() - 1; row++) {
        for (int col = 1; col < image[0].size() - 1; col++) {
            gradient_y[row][col] = image[row + 1][col] - image[row - 1][col];
            gradient_x[row][col] = image[row][col + 1] - image[row][col - 1];
        }
    }

    // �����Ե�ĽǶ�
    vector<vector<float>> edge_angles(image.size(), vector<float>(image[0].size(), 0.0));
    for (int row = 0; row < image.size(); row++) {
        for (int col = 0; col < image[0].size(); col++) {
            if (gradient_x[row][col] != 0.0) {
                edge_angles[row][col] = atan2(gradient_y[row][col], gradient_x[row][col]) * (180 / M_PI);
            }
            else if (gradient_x[row][col] == 0.0 && gradient_y[row][col] == 0.0) {
                edge_angles[row][col] = 0.0;
            }
            else if (gradient_x[row][col] == 0.0 && gradient_y[row][col] == M_PI / 2) {
                edge_angles[row][col] = 90.0;
            }
        }
    }

    // �����Ե���
    float edge_ignore = 0.1;
    float finetune_number = 0.0;
    int finetune = 0;
    int margin = 0;

    if (any(edge_angles)) {
        std::vector<std::vector<float>> quantized_angles = quantizeAngles_y(edge_angles);
        for (int row = edge_ignore * image.size(); row < (1 - edge_ignore) * (image.size() - 1); row++) {
            for (int col = 1; col < image[0].size() - 1; col++) {
                if (edges[row][col] == 1) {
                    if (quantized_angles[row][col] == -90.0) {
                        for (int margin = 0; margin <= 100; margin++) {
                            int inner_border = row - 1 - margin;
                            int outer_border = row - 2 - margin;
                            if (outer_border < 0 || (image[outer_border][col] - image[inner_border][col]) <= -1) {
                                outer_border += 1;
                                for (int finetune = 0; finetune <= margin; finetune++) {
                                    inner_border += 1;
                                    if ((image[outer_border][col] - image[inner_border][col]) > finetune_number) {
                                        break;
                                    }
                                }
                                margin = margin - finetune;
                                break;
                            }
                        }

                        float width_up = margin + 1;

                        for (int margin = 0; margin <= 100; margin++) {
                            int inner_border = row + 1 + margin;
                            int outer_border = row + 2 + margin;
                            if (outer_border >= image.size() || (image[outer_border][col] - image[inner_border][col]) >= 1) {
                                outer_border -= 1;
                                for (int finetune = 0; finetune <= margin; finetune++) {
                                    inner_border -= 1;
                                    if ((image[inner_border][col] - image[outer_border][col]) > finetune_number) {
                                        break;
                                    }
                                }
                                margin = margin - finetune;
                                break;
                            }
                        }

                        float width_down = margin + 1;

                        edge_widths[row][col] = width_up + width_down;
                    }

                    if (quantized_angles[row][col] == 90.0) {
                        for (int margin = 0; margin <= 100; margin++) {
                            int inner_border = row - 1 - margin;
                            int outer_border = row - 2 - margin;
                            if (outer_border < 0 || (image[outer_border][col] - image[inner_border][col]) >= 1) {
                                outer_border += 1;
                                for (int finetune = 0; finetune <= margin; finetune++) {
                                    inner_border += 1;
                                    if ((image[inner_border][col] - image[outer_border][col]) > finetune_number) {
                                        break;
                                    }
                                }
                                margin = margin - finetune;
                                break;
                            }
                        }

                        int width_up = margin + 1;

                        for (int margin = 0; margin <= 100; margin++) {
                            int inner_border = row + 1 + margin;
                            int outer_border = row + 2 + margin;
                            if (outer_border >= image.size() || (image[outer_border][col] - image[inner_border][col]) <= -1) {
                                outer_border -= 1;
                                for (int finetune = 0; finetune <= margin; finetune++) {
                                    inner_border -= 1;
                                    if ((image[outer_border][col] - image[inner_border][col]) > finetune_number) {
                                        break;
                                    }
                                }
                                margin = margin - finetune;
                                break;
                            }
                        }

                        float width_down = margin + 1;

                        edge_widths[row][col] = width_up + width_down;
                    }
                }
            }
        }
    }

    return edge_widths;
}


//cv::Matתstd::vector
std::vector<std::vector<float>> mat2vector(const cv::Mat& mat) {
    int rows = mat.rows;
    int cols = mat.cols;

    std::vector<std::vector<float>> vec(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            vec[i][j] = mat.at<float>(i, j);
        }
    }

    return vec;
}

// int ����cv::Matתstd::vector
std::vector<std::vector<int>> mat2vector_int(const cv::Mat& mat) {
    int rows = mat.rows;
    int cols = mat.cols;

    std::vector<std::vector<int>> vec(rows, std::vector<int>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            vec[i][j] = mat.at<int>(i, j);
        }
    }

    return vec;
}

// bool ����cv::Matתstd::vector
std::vector<std::vector<bool>> mat2vector_bool(const cv::Mat& mat) {
    int rows = mat.rows;
    int cols = mat.cols;

    std::vector<std::vector<bool>> vec(rows, std::vector<bool>(cols));

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            vec[i][j] = mat.at<int>(i, j) ? true : false;
        }
    }

    return vec;
}

// std::vectorתcv::Mat
cv::Mat vector2mat(const std::vector<std::vector<float>>& vec) {
    int rows = vec.size();
    int cols = vec[0].size();

    cv::Mat mat(rows, cols, CV_32F);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat.at<float>(i, j) = vec[i][j];
        }
    }

    return mat;
}

// int ����std::vectorתcv::Mat
cv::Mat vector2mat_int(const std::vector<std::vector<int>>& vec) {
    int rows = vec.size();
    int cols = vec[0].size();

    cv::Mat mat(rows, cols, CV_32S);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat.at<int>(i, j) = vec[i][j];
        }
    }

    return mat;
}

// bool ����std::vectorתcv::Mat
cv::Mat vector2mat_bool(const std::vector<std::vector<bool>>& vec) {
    int rows = vec.size();
    int cols = vec[0].size();

    cv::Mat mat(rows, cols, CV_8U);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat.at<uchar>(i, j) = vec[i][j] ? 1 : 0;
        }
    }

    return mat;
}


std::vector<std::vector<bool>> _simple_thinning(const std::vector<std::vector<float>>& strength) {
    int num_rows = strength.size();
    int num_cols = strength[0].size();

    std::vector<std::vector<bool>> result(num_rows, std::vector<bool>(num_cols, false));

    std::vector<float> zero_column(num_rows, 0.0f);
    std::vector<float> zero_row(num_cols, 0.0f);

    for (int i = 0; i < num_rows; ++i) {
        for (int j = 0; j < num_cols; ++j) {
            // ��һ��������ҵ�sobelֵ������Ϊ1������Ϊ0
            bool x_condition = (j == 0 || strength[i][j] > strength[i][j - 1]) &&
                (j == num_cols - 1 || strength[i][j] > strength[i][j + 1]);

            // ��һ��������µ�sobelֵ������Ϊ1������Ϊ0
            bool y_condition = (i == 0 || strength[i][j] > strength[i - 1][j]) &&
                (i == num_rows - 1 || strength[i][j] > strength[i + 1][j]);

            result[i][j] = x_condition || y_condition;
        }
    }

    return result;
}


std::vector<std::vector<bool>> sobel_h(const cv::Mat& image) {
    // sobel����
    cv::Mat h1 = (cv::Mat_<float>(3, 3) << 0.25, 0.5, 0.25,
                                            0, 0, 0,
                                            -0.25, -0.5, -0.25);

    // ȡ�������ע���Ի��5x5�ĺ�
    /*
    cv::Mat h1 = (cv::Mat_<float>(5,5) << 1, 2, 0, -2, -1,
                                         4, 8, 0, -8, -4,
                                         6, 12, 0, -12, -6,
                                         4, 8, 0, -8, -4,
                                         1, 2, 0, -2, -1);
    */
    // ��һ��h1
    h1 = h1 / cv::sum(cv::abs(h1))[0];

    cv::Mat convoluted;
    cv::filter2D(image, convoluted, -1, h1.t());

    cv::Mat strength2;
    cv::pow(convoluted, 2, strength2);

    // ����Ӧ��ֵ
    cv::Mat strength1 = strength2.reshape(1, 1);
    cv::sort(strength1, strength1, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
    float thresh2 = strength1.at<float>(0, static_cast<int>(strength1.cols * 0.99));

    cv::threshold(strength2, strength2, thresh2, 0, cv::THRESH_TOZERO);

    return _simple_thinning(mat2vector(strength2));  // ��������C++�����ж������������
}


std::vector<std::vector<bool>> sobel_v(const cv::Mat& image) {
    // sobel����
    cv::Mat h1 = (cv::Mat_<float>(3, 3) << 0.25, 0, -0.25,
        0.5, 0, -0.5,
        0.25, 0, -0.25);

    // ȡ�������ע���Ի��5x5�ĺ�
    /*
    cv::Mat h1 = (cv::Mat_<float>(5,5) << 1, 2, 0, -2, -1,
                                         4, 8, 0, -8, -4,
                                         6, 12, 0, -12, -6,
                                         4, 8, 0, -8, -4,
                                         1, 2, 0, -2, -1);
    */
    // ��һ��h1
    h1 = h1 / cv::sum(cv::abs(h1))[0];

    cv::Mat convoluted;
    cv::filter2D(image, convoluted, -1, h1.t());

    cv::Mat strength2;
    cv::pow(convoluted, 2, strength2);

    // ����Ӧ��ֵ
    cv::Mat strength1 = strength2.reshape(1, 1);
    cv::sort(strength1, strength1, cv::SORT_EVERY_ROW + cv::SORT_ASCENDING);
    float thresh2 = strength1.at<float>(0, static_cast<int>(strength1.cols * 0.99));

    cv::threshold(strength2, strength2, thresh2, 0, cv::THRESH_TOZERO);

    return _simple_thinning(mat2vector(strength2));  // ��������C++�����ж������������
}


//��������֮��ľ���
//float distance(const cv::Point2f& p1, const cv::Point2f& p2) {
//    return cv::norm(p1 - p2);
//}


//��Զ�����
std::pair<std::vector<int>, cv::Mat> FPS(const cv::Mat& sample, int num) {
    int n = sample.rows;
    cv::Mat center;
    cv::reduce(sample, center, 0, cv::REDUCE_AVG); // �����������
    std::vector<int> selected;
    std::vector<float> min_distance(n);

    for (int i = 0; i < n; i++) {
        min_distance[i] = cv::norm(sample.row(i) - center);
    }

    int p0 = std::max_element(min_distance.begin(), min_distance.end()) - min_distance.begin();
    selected.push_back(p0); // ѡ����������Զ��p0

    for (int i = 0; i < n; i++) {
        min_distance[i] = cv::norm(sample.row(p0) - sample.row(i));
    }

    selected.push_back(std::max_element(min_distance.begin(), min_distance.end()) - min_distance.begin());

    for (int i = 2; i < num; i++) {
        for (int p = 0; p < n; p++) {
            float d = cv::norm(sample.row(selected.back()) - sample.row(p));
            if (d <= min_distance[p]) {
                min_distance[p] = d;
            }
        }
        selected.push_back(std::max_element(min_distance.begin(), min_distance.end()) - min_distance.begin());
    }

    cv::Mat selected_points(num, sample.cols, sample.type());
    for (int i = 0; i < num; i++) {
        sample.row(selected[i]).copyTo(selected_points.row(i));
    }

    return std::make_pair(selected, selected_points);
}


// ִ�е�ͨ�˲��ĺ���
std::vector<int> lowPassFilter(const std::vector<int>& signal, float cutoff) {
    std::vector<float> b(9, 0.0), a(9, 0.0); // �˲���ϵ��
    int order = 8; // �˲�������
    float theta = 2.0 * M_PI * cutoff;

    // ʹ��Butterworth�˲�����ʽ�����˲���ϵ��
    for (int i = 0; i <= order; i++) {
        float val = ((i == 0 || i == order) ? 1.0 : 2.0) * std::pow(std::sin((order - i) * theta / 2.0), order) / std::pow(2.0, order - 1);
        if (i % 2 == 0)
            b[i] = val;
        else
            a[i] = val;
    }

    std::vector<float> filteredSignal(signal.size(), 0.0); // ���˺���ź�

    // ���źŽ���ǰ��-�����˲�
    for (int i = 0; i < signal.size(); i++) {
        float y = 0.0;

        // ǰ���˲�
        for (int j = 0; j <= order; j++) {
            if (i - j >= 0)
                y += b[j] * signal[i - j];
        }

        // �����˲�
        for (int j = 1; j <= order; j++) {
            if (i + j < signal.size())
                y -= a[j] * filteredSignal[i + j];
        }

        filteredSignal[i] = y;
    }

    // �����˺���ź�ת��Ϊ�������ü�ֵ
    std::vector<int> filteredIntSignal(filteredSignal.size());
    for (int i = 0; i < filteredSignal.size(); i++) {
        // �ֶ�����ֵ�Ĳü�
        if (filteredSignal[i] > 255)
            filteredIntSignal[i] = 255;
        else if (filteredSignal[i] < 0)
            filteredIntSignal[i] = 0;
        else
            filteredIntSignal[i] = static_cast<int>(filteredSignal[i]);
    }

    return filteredIntSignal;

}


std::pair<int, std::pair<int, float>> compute_blur(std::string image_path, int rotation_angle) {
    Mat image_color = imread(image_path);

    //��ȡ�Ҷ�ͼ
    Mat image;
    cvtColor(image_color, image, COLOR_BGR2GRAY);

    Mat image1_rotated;
    Mat image_rotated;

    if (rotation_angle != 0) {
        Mat image_color_rotated;
        Mat M = getRotationMatrix2D(Point2f(image.cols / 2, image.rows / 2), rotation_angle, 1);
        warpAffine(image_color, image_color_rotated, M, image_color.size());

        Mat image1 = Mat::zeros(image.size(), CV_8UC3);

        int width = image1.cols;
        int height = image1.rows;

        rectangle(image1, Point2f(0, 0), Point2f(3, height), Scalar(255, 255, 255), cv::FILLED);
        rectangle(image1, Point2f(width - 4, 0), Point2f(width - 1, height), Scalar(255, 255, 255), cv::FILLED);
        rectangle(image1, Point2f(0, 0), Point2f(width, 3), Scalar(255, 255, 255), cv::FILLED);
        rectangle(image1, Point2f(0, height - 4), Point2f(width, height - 1), Scalar(255, 255, 255), cv::FILLED);

        warpAffine(image1, image1_rotated, M, image1.size());
        cvtColor(image1_rotated, image1_rotated, COLOR_BGR2GRAY);

        cvtColor(image_color_rotated, image_rotated, COLOR_BGR2GRAY);
        image = image_rotated.clone();
    }

    Mat img = image.clone();

    int height = img.rows;
    int width = img.cols;
    int b_w = 320; // ͼ���Ŀ��
    int b_h = b_w; // ͼ���ĸ߶�
    int L = b_w * b_h; // ͼ���ĳߴ�
    std::vector<float> ratiolisth; // ��¼��ı�Ե���ر������б�
    std::vector<float> ratiolistv; // ��¼��Ĵ�ֱ�����Ե���ص��б�
    float ignore_ratio = 0.1; // ���鲻��������ı�Ե���صı���

    std::vector<int> indexlist_h; // ��¼���ѡ���index
    std::vector<int> selected_index_h; // ��¼ѡ�п��index
    cv::Mat coordinate_list_h; // ��¼���ѡ�������
    std::vector<int> indexlist_v;
    std::vector<int> selected_index_v;
    cv::Mat coordinate_list_v;
    int block_num_h_ori = 20; // ˮƽ�������ѡ�����Ŀ
    int block_num_v_ori = block_num_h_ori; // ��ֱ�������ѡ�����Ŀ
    int blocknum_h = 9; // ˮƽ������Ҫ�Ŀ����Ŀ
    int blocknum_v = blocknum_h; // ��ֱ������Ҫ�Ŀ���

    int blocknum_final = 5; // ������������ģ���ȵĿ�ĸ���

    std::vector<std::vector<int>> bbox_list_h; // ��ѡ����ˮƽ��
    std::vector<std::vector<int>> bbox_list_v; // ѡ���Ĵ�ֱ��
    std::vector<std::vector<int>> bbox_list; // ������
    int bbox_num = 0; // ��ĸ���

    int bbox_num1 = bbox_num;
    int bbox_num2 = bbox_num;

    int blockrow = floor(height / b_h); // ͼ�������
    int blockcol = floor(width / b_w); // ͼ�������

    std::pair<std::vector<int>, cv::Mat> pair_FPS_h;
    std::pair<std::vector<int>, cv::Mat> pair_FPS_v;

    std::vector<std::vector<bool>> sobel_h_edges = sobel_h(img);
    std::vector<std::vector<bool>> sobel_v_edges = sobel_v(img);

    //��ṹ�壬���һά��������ά����
    struct Block {
        int index;
        std::pair<int, int> coordinates;
    };

    // ��ͼ���ÿһ��Ӧ�õ�ͨ�˲���
    std::vector<std::vector<int>> img_hfilted(height, std::vector<int>(width));
    for (int i = 0; i < height; i++) {
        img_hfilted[i] = lowPassFilter(std::vector<int>(img.row(i).begin<int>(), img.row(i).end<int>()), 0.1);
    }

    // ��ͼ���ÿһ��Ӧ�õ�ͨ�˲���
    std::vector<std::vector<int>> img_vfilted(height, std::vector<int>(width));
    for (int i = 0; i < width; i++) {
        std::vector<int> columnData(height);
        for (int j = 0; j < height; j++) {
            columnData[j] = img.at<int>(j,i);
        }
        std::vector<int> filteredColumn = lowPassFilter(columnData, 0.1);
        for (int j = 0; j < height; j++) {
            img_vfilted[j][i] = filteredColumn[j];
        }
    }

    //ȥ��������ת�ĺڱߵ��µı�Ե��
    if (rotation_angle != 0) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                if (image1_rotated.at<int>(i,j) > 0) {
                    sobel_h_edges[i][j] = false;
                    sobel_v_edges[i][j] = false;
                }
                if (img.at<int>(i,j) == 0) {
                    sobel_h_edges[i][j] = false;
                    sobel_v_edges[i][j] = false;
                }
            }
        }
    }

    // �����¼��Ե��Ⱥͱ�׼��ľ���
    cv::Mat edgewidth_h(selected_index_h.size(), blockcol, CV_32FC1);      // ��¼����ˮƽ��Ե���
    cv::Mat edgewidth_v(selected_index_h.size(), blockcol, CV_32FC1);      // ��¼���鴹ֱ��Ե���
    cv::Mat edgewidth_h_std(selected_index_h.size(), blockcol, CV_32FC1);  // ��¼����ˮƽ��Ե��ȵı�׼��
    cv::Mat edgewidth_v_std(selected_index_h.size(), blockcol, CV_32FC1);  // ��¼���鴹ֱ��Ե��ȵı�׼��

    // ѭ�������Ե��Ⱥͱ�׼��
    for (int i = 0; i < blockrow; i++) {
        for (int j = 0; j < blockcol; j++) {
            std::vector<std::vector<int>> edges_h_temp(b_h, std::vector<int>(b_w));
            std::vector<std::vector<int>> edges_v_temp(b_h, std::vector<int>(b_w));

            // ��ȡ��ǰ��ı�Ե����
            for (int m = 0; m < b_h; m++) {
                for (int n = 0; n < b_w; n++) {
                    edges_h_temp[m][n] = sobel_h_edges[b_h * i + m][b_w * j + n];
                    edges_v_temp[m][n] = sobel_v_edges[b_h * i + m][b_w * j + n];
                }
            }

            // �߽�������0������������
            int ignore_pixels = static_cast<int>(ignore_ratio * b_w);
            for (int m = 0; m < b_h; m++) {
                for (int n = 0; n < ignore_pixels; n++) {
                    edges_h_temp[m][n] = 0;
                }
                for (int n = b_w - ignore_pixels; n < b_w; n++) {
                    edges_h_temp[m][n] = 0;
                }
            }

            for (int n = 0; n < b_w; n++) {
                for (int m = 0; m < ignore_pixels; m++) {
                    edges_v_temp[m][n] = 0;
                }
                for (int m = b_h - ignore_pixels; m < b_h; m++) {
                    edges_v_temp[m][n] = 0;
                }
            }

            // �����Ե���ر���
            float ratio_temp_h = 0.0;
            float ratio_temp_v = 0.0;

            for (int m = 0; m < b_h; m++) {
                for (int n = 0; n < b_w; n++) {
                    ratio_temp_h += static_cast<float>(edges_h_temp[m][n]);
                    ratio_temp_v += static_cast<float>(edges_v_temp[m][n]);
                }
            }

            ratio_temp_h /= L;
            ratio_temp_v /= L;

            // �洢��Ե���ر���
            ratiolisth.push_back(ratio_temp_h);
            ratiolistv.push_back(ratio_temp_v);
        }
    }




    // x�������
    std::vector<Block> blocks;

    // ����blocks���洢���ǵ�index��coordinates
    for (int i = 0; i < blockrow; ++i) {
        for (int j = 0; j < blockcol; ++j) {
            Block block;
            block.index = i * blockcol + j;
            block.coordinates = std::make_pair(i + 1, j + 1);
            blocks.push_back(block);
        }
    }

    // ����ratio��ֵ��С��������blocks
    std::sort(blocks.begin(), blocks.end(), [&](const Block& b1, const Block& b2) {
        return ratiolisth[b1.index] < ratiolisth[b2.index];
        });

    int indexflag = 1;

    while (block_num_h_ori) {
        int indexnow = blocks.back().index;
        int i = (indexnow / blockcol) + 1;
        int j = (indexnow % blockcol) + 1;
        indexlist_h.push_back(indexnow);
        coordinate_list_h.at<int>(indexflag - 1, 0) = i;
        coordinate_list_h.at<int>(indexflag - 1, 1) = j;
        indexflag++;
        block_num_h_ori--;

        if (indexflag >= blockrow * blockcol) {
            break;
        }
    }
    
    //����Զ�����ѡ���
    pair_FPS_h = FPS(coordinate_list_h, blocknum_h);
    std::vector<int> index;
    cv::Mat selected_block;
    index = pair_FPS_h.first;
    selected_block = pair_FPS_h.second;

    for(int i = 0; i < index.size(); ++i) {
        selected_index_h.push_back(indexlist_h[index[i]]);
    }

    for (int i = 0; i < selected_index_h.size(); ++i) {
        int indexnow = selected_index_h[i];
        int row = (indexnow / blockcol) + 1;
        int col = (indexnow % blockcol) + 1;
        std::vector<int> bbox = { b_h * (col - 1), b_w * (row - 1), b_h * col, b_w * row };
        bbox_list_h.push_back(bbox);
        bbox_num1++;

        cv::Mat Block_filted_temp = vector2mat_int(img_hfilted)(cv::Range(b_h * (row - 1), b_h * row), cv::Range(b_w * (col - 1), b_w * col));
        cv::Mat edges_h_temp = vector2mat_bool(sobel_h_edges)(cv::Range(b_h * (row - 1), b_h * row), cv::Range(b_w * (col - 1), b_w * col));
        std::vector<std::vector<float>> edgewidth_h_temp_vector = marziliano_method_h(edges_h_temp, Block_filted_temp);
        int total_edges_h_temp = 0;
        for (const auto& row : edgewidth_h_temp_vector) {
            for (const auto& value : row) {
                if (value) {
                    total_edges_h_temp++;
                }
            }
        }

        cv::Mat edgewidth_h_temp = vector2mat(edgewidth_h_temp_vector);

        if (total_edges_h_temp > 0) {
            float average = static_cast<float>(cv::sum(edgewidth_h_temp)[0]) / total_edges_h_temp;
            edgewidth_h.at<float>(row - 1, col - 1) = average;

            cv::Scalar stddev;
            cv::meanStdDev(edgewidth_h_temp, cv::noArray(), stddev);
            edgewidth_h_std.at<float>(row - 1, col - 1) = static_cast<float>(stddev.val[0]);
        }

        // ȥ��3_sigma֮��ĵ����¼����ֵ
        cv::threshold(edgewidth_h_temp, edgewidth_h_temp,
            edgewidth_h.at<float>(row - 1, col - 1) - (3 * edgewidth_h_std.at<float>(row - 1, col - 1)),
            0, cv::THRESH_TOZERO);
        cv::threshold(edgewidth_h_temp, edgewidth_h_temp,
            edgewidth_h.at<float>(row - 1, col - 1) + (3 * edgewidth_h_std.at<float>(row - 1, col - 1)),
            0, cv::THRESH_TOZERO_INV);
        total_edges_h_temp = cv::countNonZero(edgewidth_h_temp);

        if (total_edges_h_temp > 0) {
            float average = static_cast<float>(cv::sum(edgewidth_h_temp)[0]) / total_edges_h_temp;
            edgewidth_h.at<float>(row - 1, col - 1) = average;

            cv::Scalar stddev;
            cv::meanStdDev(edgewidth_h_temp, cv::noArray(), stddev);
            edgewidth_h_std.at<float>(row - 1, col - 1) = static_cast<float>(stddev.val[0]);
        }
    }

    cv::Mat edgewidth_h_std_nonzero_index;
    cv::findNonZero(edgewidth_h_std, edgewidth_h_std_nonzero_index); // ģ���ȷ��Ϊ��Ŀ������

    cv::Mat edgewidth_h_std_nonzero;
    for (int i = 0; i < edgewidth_h_std_nonzero_index.total(); i++) {
        int row = edgewidth_h_std_nonzero_index.at<cv::Point>(i).y;
        int col = edgewidth_h_std_nonzero_index.at<cv::Point>(i).x;
        edgewidth_h_std_nonzero.push_back(edgewidth_h_std.at<float>(row, col));
    }

    cv::Mat edgewidth_h_nonzero_index;
    cv::findNonZero(edgewidth_h, edgewidth_h_nonzero_index); // �õ�edgewidth_h�з���Ԫ�ص�����ֵ

    cv::Mat edgewidth_h_nonzero;
    for (int i = 0; i < edgewidth_h_nonzero_index.total(); i++) {
        int row = edgewidth_h_nonzero_index.at<cv::Point>(i).y;
        int col = edgewidth_h_nonzero_index.at<cv::Point>(i).x;
        edgewidth_h_nonzero.push_back(edgewidth_h.at<float>(row, col));
    }

    cv::Mat std_index;
    cv::sortIdx(edgewidth_h_std_nonzero, std_index, cv::SORT_ASCENDING); // �������С�������������ֵ

    cv::Mat edgewidth_h_finalblk;
    for (int i = 0; i < blocknum_final; i++) {
        int index = std_index.at<int>(i);
        edgewidth_h_finalblk.push_back(edgewidth_h_nonzero.at<float>(index));
    }




    // y�������
    std::vector<Block> blocks_v;

    // ����blocks���洢���ǵ�index��coordinates
    for (int i = 0; i < blockrow; ++i) {
        for (int j = 0; j < blockcol; ++j) {
            Block block;
            block.index = i * blockcol + j;
            block.coordinates = std::make_pair(i + 1, j + 1);
            blocks_v.push_back(block);
        }
    }

    // ����ratio��ֵ��С��������blocks
    std::sort(blocks_v.begin(), blocks_v.end(), [&](const Block& b1, const Block& b2) {
        return ratiolistv[b1.index] < ratiolistv[b2.index];
        });

    indexflag = 1;

    while (block_num_v_ori) {
        int indexnow = blocks_v.back().index;
        int i = (indexnow / blockcol) + 1;
        int j = (indexnow % blockcol) + 1;
        indexlist_v.push_back(indexnow);
        coordinate_list_v.at<int>(indexflag - 1, 0) = i;
        coordinate_list_v.at<int>(indexflag - 1, 1) = j;
        indexflag++;
        block_num_v_ori--;

        if (indexflag >= blockrow * blockcol) {
            break;
        }
    }

    //����Զ�����ѡ���
    pair_FPS_v = FPS(coordinate_list_v, blocknum_v);
    index = pair_FPS_v.first;
    selected_block = pair_FPS_v.second;

    for (int i = 0; i < index.size(); ++i) {
        selected_index_v.push_back(indexlist_v[index[i]]);
    }

    for (int i = 0; i < selected_index_v.size(); ++i) {
        int indexnow = selected_index_v[i];
        int row = (indexnow / blockcol) + 1;
        int col = (indexnow % blockcol) + 1;
        std::vector<int> bbox = { b_h * (col - 1), b_w * (row - 1), b_h * col, b_w * row };
        bbox_list_v.push_back(bbox);
        bbox_num2++;

        cv::Mat Block_filted_temp = vector2mat_int(img_vfilted)(cv::Range(b_h * (row - 1), b_h * row), cv::Range(b_w * (col - 1), b_w * col));
        cv::Mat edges_v_temp = vector2mat_bool(sobel_v_edges)(cv::Range(b_h * (row - 1), b_h * row), cv::Range(b_w * (col - 1), b_w * col));
        std::vector<std::vector<float>> edgewidth_v_temp_vector = marziliano_method_h(edges_v_temp, Block_filted_temp);
        int total_edges_v_temp = 0;
        for (const auto& row : edgewidth_v_temp_vector) {
            for (const auto& value : row) {
                if (value) {
                    total_edges_v_temp++;
                }
            }
        }

        cv::Mat edgewidth_v_temp = vector2mat(edgewidth_v_temp_vector);

        if (total_edges_v_temp > 0) {
            float average = static_cast<float>(cv::sum(edgewidth_v_temp)[0]) / total_edges_v_temp;
            edgewidth_v.at<float>(row - 1, col - 1) = average;

            cv::Scalar stddev;
            cv::meanStdDev(edgewidth_v_temp, cv::noArray(), stddev);
            edgewidth_v_std.at<float>(row - 1, col - 1) = static_cast<float>(stddev.val[0]);
        }

        // ȥ��3_sigma֮��ĵ����¼����ֵ
        cv::threshold(edgewidth_v_temp, edgewidth_v_temp,
            edgewidth_v.at<float>(row - 1, col - 1) - (3 * edgewidth_v_std.at<float>(row - 1, col - 1)),
            0, cv::THRESH_TOZERO);
        cv::threshold(edgewidth_v_temp, edgewidth_v_temp,
            edgewidth_v.at<float>(row - 1, col - 1) + (3 * edgewidth_v_std.at<float>(row - 1, col - 1)),
            0, cv::THRESH_TOZERO_INV);
        total_edges_v_temp = cv::countNonZero(edgewidth_v_temp);

        if (total_edges_v_temp > 0) {
            float average = static_cast<float>(cv::sum(edgewidth_v_temp)[0]) / total_edges_v_temp;
            edgewidth_v.at<float>(row - 1, col - 1) = average;

            cv::Scalar stddev;
            cv::meanStdDev(edgewidth_v_temp, cv::noArray(), stddev);
            edgewidth_v_std.at<float>(row - 1, col - 1) = static_cast<float>(stddev.val[0]);
        }
    }

    cv::Mat edgewidth_v_std_nonzero_index;
    cv::findNonZero(edgewidth_v_std, edgewidth_v_std_nonzero_index); // ģ���ȷ��Ϊ��Ŀ������

    cv::Mat edgewidth_v_std_nonzero;
    for (int i = 0; i < edgewidth_v_std_nonzero_index.total(); i++) {
        int row = edgewidth_v_std_nonzero_index.at<cv::Point>(i).y;
        int col = edgewidth_v_std_nonzero_index.at<cv::Point>(i).x;
        edgewidth_v_std_nonzero.push_back(edgewidth_v_std.at<float>(row, col));
    }

    cv::Mat edgewidth_v_nonzero_index;
    cv::findNonZero(edgewidth_v, edgewidth_v_nonzero_index); // �õ�edgewidth_h�з���Ԫ�ص�����ֵ

    cv::Mat edgewidth_v_nonzero;
    for (int i = 0; i < edgewidth_v_nonzero_index.total(); i++) {
        int row = edgewidth_v_nonzero_index.at<cv::Point>(i).y;
        int col = edgewidth_v_nonzero_index.at<cv::Point>(i).x;
        edgewidth_v_nonzero.push_back(edgewidth_v.at<float>(row, col));
    }

    cv::Mat std_index_v;
    cv::sortIdx(edgewidth_v_std_nonzero, std_index_v, cv::SORT_ASCENDING); // �������С�������������ֵ

    cv::Mat edgewidth_v_finalblk;
    for (int i = 0; i < blocknum_final; i++) {
        int index = std_index_v.at<int>(i);
        edgewidth_v_finalblk.push_back(edgewidth_v_nonzero.at<float>(index));
    }


    return std::make_pair(1, std::make_pair(2, 3.3));
}


int main() {
    vector<vector<int>> edges = {
        {0, 1, 0, 1, 0},
        {0, 1, 1, 1, 0},
        {0, 0, 1, 0, 0},
        {0, 1, 1, 1, 0},
        {0, 1, 0, 1, 0}
    };

    vector<vector<float>> image = {
        {100, 100, 200, 100, 100},
        {100, 100, 100, 100, 100},
        {150, 150, 150, 150, 150},
        {100, 100, 100, 100, 100},
        {100, 100, 200, 100, 100}
    };

    //vector<vector<float>> edge_widths = marziliano_method_h(edges, image);

    
    //vector<vector<bool>> edges_h = sobel_h(vector2mat(image));

    /*for (int row = 0; row < edge_widths.size(); row++) {
        for (int col = 0; col < edge_widths[0].size(); col++) {
            cout << edge_widths[row][col] << " ";
        }
        cout << endl;
    }*/

    cv::Mat edgewidth_h_nonzero;
    edgewidth_h_nonzero.push_back(0);

    /*std::cout << edgewidth_h_nonzero << '\n';*/

    return 0;
}
