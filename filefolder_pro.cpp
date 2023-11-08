// 第一种

#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    std::string folder_path = "/path/to/your/folder"; // 替换为要遍历的文件夹路径
    std::vector<std::string> file_names;

    if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                file_names.push_back(entry.path().filename().string());
            }
        }

        // 对文件名进行排序
        std::sort(file_names.begin(), file_names.end());

        // 打印排序后的文件名
        for (const std::string& file_name : file_names) {
            std::cout << file_name << std::endl;
        }
    } else {
        std::cout << "指定的路径不是文件夹或路径不存在。" << std::endl;
    }

    return 0;
}





//第二种

#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

void listFiles(const fs::path& dir, std::vector<fs::path>& files) {
    if (fs::exists(dir) && fs::is_directory(dir)) {
        for (const auto& entry : fs::directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                files.push_back(entry.path());
            } else if (entry.is_directory()) {
                listFiles(entry.path(), files);
            }
        }
    }
}

int main() {
    std::string folder_path = "/path/to/your/folder"; // 替换为要遍历的文件夹路径
    std::vector<fs::path> file_paths;

    listFiles(folder_path, file_paths);

    // 对文件路径进行排序
    std::sort(file_paths.begin(), file_paths.end());

    // 打印排序后的文件路径
    for (const fs::path& file_path : file_paths) {
        std::cout << file_path.filename() << std::endl;
    }

    return 0;
}






//第三个，对于文件夹中类似于a_1_b_2_c_3.jpg格式的文件名，遍历文件夹，并根据下划线分割文件名，根据a后的数字排序
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <string>
#include <regex>

namespace fs = std::filesystem;

// 自定义比较函数，用于对文件名进行排序
bool customSort(const fs::path& path1, const fs::path& path2) {
    // 使用正则表达式从文件名中提取"a"后的数字部分
    std::string filename1 = path1.filename().string();
    std::string filename2 = path2.filename().string();
    std::regex pattern("a_(\\d+)");
    std::smatch match1, match2;

    if (std::regex_search(filename1, match1, pattern) && std::regex_search(filename2, match2, pattern)) {
        int number1 = std::stoi(match1[1].str());
        int number2 = std::stoi(match2[1].str());
        return number1 < number2;
    }

    return false;
}

int main() {
    std::string folder_path = "/path/to/your/folder"; // 替换为要遍历的文件夹路径
    std::vector<fs::path> file_paths;

    if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                file_paths.push_back(entry.path());
            }
        }

        // 使用自定义比较函数对文件路径进行排序
        std::sort(file_paths.begin(), file_paths.end(), customSort);

        // 打印排序后的文件路径
        for (const fs::path& file_path : file_paths) {
            std::cout << file_path.filename() << std::endl;
        }
    } else {
        std::cout << "指定的路径不是文件夹或路径不存在。" << std::endl;
    }

    return 0;
}




//第四个
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <string>
#include <regex>

namespace fs = std::filesystem;

// 自定义比较函数，用于对文件名进行排序
bool customSort(const fs::path& path1, const fs::path& path2) {
    // 使用正则表达式从文件名中提取"a"后的数字部分
    std::string filename1 = path1.filename().string();
    std::string filename2 = path2.filename().string();
    std::regex pattern("a_(\\d+)_b_(\\d+)_c_(\\d+)\\.jpg");
    std::smatch match1, match2;

    if (std::regex_search(filename1, match1, pattern) && std::regex_search(filename2, match2, pattern)) {
        int a_number1 = std::stoi(match1[1].str());
        int a_number2 = std::stoi(match2[1].str());
        
        if (a_number1 < a_number2) {
            return true;
        } else if (a_number1 == a_number2) {
            int b_number1 = std::stoi(match1[2].str());
            int b_number2 = std::stoi(match2[2].str());
            return b_number1 < b_number2;
        }
    }

    return false;
}

int main() {
    std::string folder_path = "/path/to/your/folder"; // 替换为要遍历的文件夹路径
    std::vector<fs::path> file_paths;

    if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                file_paths.push_back(entry.path());
            }
        }

        // 使用自定义比较函数对文件路径进行排序
        std::sort(file_paths.begin(), file_paths.end(), customSort);

        // 打印排序后的文件路径
        for (const fs::path& file_path : file_paths) {
            std::cout << file_path.filename() << std::endl;
        }
    } else {
        std::cout << "指定的路径不是文件夹或路径不存在。" << std::endl;
    }

    return 0;
}




//分割文件名
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string filename = "file_part1_part2_part3.txt"; // 替换为你的文件名

    // 使用std::string的find和substr函数来分割文件名
    std::vector<std::string> parts;
    size_t startPos = 0;
    size_t foundPos;

    while ((foundPos = filename.find('_', startPos)) != std::string::npos) {
        std::string part = filename.substr(startPos, foundPos - startPos);
        parts.push_back(part);
        startPos = foundPos + 1; // 移动起始位置到下一个部分的开始
    }

    // 处理最后一个部分
    std::string lastPart = filename.substr(startPos);
    parts.push_back(lastPart);

    // 打印分割后的部分
    for (const std::string& part : parts) {
        std::cout << "Part: " << part << std::endl;
    }

    return 0;
}



//C++17之前的标准库版本
#include <iostream>
#include <vector>
#include <string>
#include <experimental/filesystem> // 使用C++17之前的标准库版本

namespace fs = std::experimental::filesystem;

void listFiles(const std::string& folder_path, std::vector<std::string>& file_names) {
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            file_names.push_back(entry.path().filename().string());
        } else if (entry.is_directory()) {
            listFiles(entry.path(), file_names); // 递归进入子文件夹
        }
    }
}

int main() {
    std::string folder_path = "/path/to/your/folder"; // 替换为要遍历的文件夹路径
    std::vector<std::string> file_names;

    if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
        listFiles(folder_path, file_names);

        // 打印文件名
        for (const std::string& file_name : file_names) {
            std::cout << file_name << std::endl;
        }
    } else {
        std::cout << "指定的路径不是文件夹或路径不存在。" << std::endl;
    }

    return 0;
}
