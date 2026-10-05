#include "../include/document.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

DocumentLoader::DocumentLoader() {}

std::vector<Document> DocumentLoader::load_from_directory(const std::string& dir_path) {
    (void)dir_path;
    std::vector<Document> docs;
    return docs;
}

std::vector<Document> DocumentLoader::load_from_manifest(const std::string& manifest_path) {
    std::vector<Document> docs;
    std::ifstream infile(manifest_path);
    if (!infile.is_open()) {
        std::cerr << "[Error] Could not open manifest file: " << manifest_path << std::endl;
        return docs;
    }

    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) continue;
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string id = line.substr(0, tab_pos);
            std::string content = line.substr(tab_pos + 1);
            docs.push_back({id, content, {}});
        } else {
            // Treat whole line as document content with auto id
            std::string id = "doc_" + std::to_string(docs.size());
            docs.push_back({id, line, {}});
        }
    }
    return docs;
}
