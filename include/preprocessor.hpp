#ifndef PREPROCESSOR_HPP
#define PREPROCESSOR_HPP

#include <string>
#include <vector>
#include <unordered_set>
#include "document.hpp"

class TextPreprocessor {
private:
    std::unordered_set<std::string> stop_words;

public:
    TextPreprocessor();
    explicit TextPreprocessor(const std::unordered_set<std::string>& custom_stopwords);

    std::vector<std::string> tokenize(const std::string& text) const;
    void preprocess_document(Document& doc) const;
    void preprocess_documents_sequential(std::vector<Document>& docs) const;
    void preprocess_documents_parallel(std::vector<Document>& docs, int num_threads = 0) const;
};

#endif // PREPROCESSOR_HPP
