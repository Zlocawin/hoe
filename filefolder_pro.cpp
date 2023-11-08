// ��һ��

#include <iostream>
#include <filesystem>
#include <vector>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    std::string folder_path = "/path/to/your/folder"; // �滻ΪҪ�������ļ���·��
    std::vector<std::string> file_names;

    if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                file_names.push_back(entry.path().filename().string());
            }
        }

        // ���ļ�����������
        std::sort(file_names.begin(), file_names.end());

        // ��ӡ�������ļ���
        for (const std::string& file_name : file_names) {
            std::cout << file_name << std::endl;
        }
    } else {
        std::cout << "ָ����·�������ļ��л�·�������ڡ�" << std::endl;
    }

    return 0;
}





//�ڶ���

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
    std::string folder_path = "/path/to/your/folder"; // �滻ΪҪ�������ļ���·��
    std::vector<fs::path> file_paths;

    listFiles(folder_path, file_paths);

    // ���ļ�·����������
    std::sort(file_paths.begin(), file_paths.end());

    // ��ӡ�������ļ�·��
    for (const fs::path& file_path : file_paths) {
        std::cout << file_path.filename() << std::endl;
    }

    return 0;
}






//�������������ļ�����������a_1_b_2_c_3.jpg��ʽ���ļ����������ļ��У��������»��߷ָ��ļ���������a�����������
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <string>
#include <regex>

namespace fs = std::filesystem;

// �Զ���ȽϺ��������ڶ��ļ�����������
bool customSort(const fs::path& path1, const fs::path& path2) {
    // ʹ��������ʽ���ļ�������ȡ"a"������ֲ���
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
    std::string folder_path = "/path/to/your/folder"; // �滻ΪҪ�������ļ���·��
    std::vector<fs::path> file_paths;

    if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                file_paths.push_back(entry.path());
            }
        }

        // ʹ���Զ���ȽϺ������ļ�·����������
        std::sort(file_paths.begin(), file_paths.end(), customSort);

        // ��ӡ�������ļ�·��
        for (const fs::path& file_path : file_paths) {
            std::cout << file_path.filename() << std::endl;
        }
    } else {
        std::cout << "ָ����·�������ļ��л�·�������ڡ�" << std::endl;
    }

    return 0;
}




//���ĸ�
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <string>
#include <regex>

namespace fs = std::filesystem;

// �Զ���ȽϺ��������ڶ��ļ�����������
bool customSort(const fs::path& path1, const fs::path& path2) {
    // ʹ��������ʽ���ļ�������ȡ"a"������ֲ���
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
    std::string folder_path = "/path/to/your/folder"; // �滻ΪҪ�������ļ���·��
    std::vector<fs::path> file_paths;

    if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file()) {
                file_paths.push_back(entry.path());
            }
        }

        // ʹ���Զ���ȽϺ������ļ�·����������
        std::sort(file_paths.begin(), file_paths.end(), customSort);

        // ��ӡ�������ļ�·��
        for (const fs::path& file_path : file_paths) {
            std::cout << file_path.filename() << std::endl;
        }
    } else {
        std::cout << "ָ����·�������ļ��л�·�������ڡ�" << std::endl;
    }

    return 0;
}




//�ָ��ļ���
#include <iostream>
#include <string>
#include <vector>

int main() {
    std::string filename = "file_part1_part2_part3.txt"; // �滻Ϊ����ļ���

    // ʹ��std::string��find��substr�������ָ��ļ���
    std::vector<std::string> parts;
    size_t startPos = 0;
    size_t foundPos;

    while ((foundPos = filename.find('_', startPos)) != std::string::npos) {
        std::string part = filename.substr(startPos, foundPos - startPos);
        parts.push_back(part);
        startPos = foundPos + 1; // �ƶ���ʼλ�õ���һ�����ֵĿ�ʼ
    }

    // �������һ������
    std::string lastPart = filename.substr(startPos);
    parts.push_back(lastPart);

    // ��ӡ�ָ��Ĳ���
    for (const std::string& part : parts) {
        std::cout << "Part: " << part << std::endl;
    }

    return 0;
}



//C++17֮ǰ�ı�׼��汾
#include <iostream>
#include <vector>
#include <string>
#include <experimental/filesystem> // ʹ��C++17֮ǰ�ı�׼��汾

namespace fs = std::experimental::filesystem;

void listFiles(const std::string& folder_path, std::vector<std::string>& file_names) {
    for (const auto& entry : fs::directory_iterator(folder_path)) {
        if (entry.is_regular_file()) {
            file_names.push_back(entry.path().filename().string());
        } else if (entry.is_directory()) {
            listFiles(entry.path(), file_names); // �ݹ�������ļ���
        }
    }
}

int main() {
    std::string folder_path = "/path/to/your/folder"; // �滻ΪҪ�������ļ���·��
    std::vector<std::string> file_names;

    if (fs::exists(folder_path) && fs::is_directory(folder_path)) {
        listFiles(folder_path, file_names);

        // ��ӡ�ļ���
        for (const std::string& file_name : file_names) {
            std::cout << file_name << std::endl;
        }
    } else {
        std::cout << "ָ����·�������ļ��л�·�������ڡ�" << std::endl;
    }

    return 0;
}
