#include <iostream>
#include <map>
#include <string>
#include <vector>

int main() {
    std::map<std::string, std::vector<double>> data; 
    data["x"]  = {1.0, 3.0, 5.0};
    data["y"] = {2.5, 8.9, 2.0};
    data["z"] = {3.14, 7.9, 3.7};

    bool first = true;
    for (const auto &kv : data) {
        if (!first) {
            std::cout << "\n";
        }
        first = false;
        std::cout << kv.first << ": [";
        const auto &vec = kv.second;
        for (size_t i = 0; i < vec.size(); ++i) {
            std::cout << vec[i];
            if (i < vec.size() - 1)
                std::cout << ", ";
        }
        std::cout << "]";
    }

    std::cout << std::endl;
    return 0;
}
