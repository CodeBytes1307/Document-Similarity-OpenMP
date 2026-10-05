#ifndef DOCUMENT_HPP
#define DOCUMENT_HPP

#include <string>
#include <vector>

struct Document {
    std::string id;
    std::string raw_text;
    std::vector<std::string> tokens;
};

class DocumentLoader {
public:
    DocumentLoader();
    static std::vector<Document> load_from_manifest(const std::string& manifest_path);
    static std::vector<Document> load_from_directory(const std::string& dir_path);
};

#endif // DOCUMENT_HPP
