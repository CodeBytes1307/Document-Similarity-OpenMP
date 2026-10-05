#include "../include/preprocessor.hpp"
#include <algorithm>
#include <cctype>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

TextPreprocessor::TextPreprocessor() {
    stop_words = {
        "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
        "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
        "below", "between", "both", "but", "by", "can", "cannot", "could", "couldn't",
        "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
        "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
        "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
        "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
        "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
        "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
        "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
        "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
        "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
        "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
        "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
        "they've", "this", "those", "through", "to", "too", "under", "until", "up",
        "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
        "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
        "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
        "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
        "yourself", "yourselves"
    };
}

TextPreprocessor::TextPreprocessor(const std::unordered_set<std::string>& custom_stopwords)
    : stop_words(custom_stopwords) {}

std::vector<std::string> TextPreprocessor::tokenize(const std::string& text) const {
    std::vector<std::string> tokens;
    std::string current_token;
    
    for (char ch : text) {
        if (std::isalnum(static_cast<unsigned char>(ch))) {
            current_token += static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
        } else {
            if (!current_token.empty()) {
                if (stop_words.find(current_token) == stop_words.end()) {
                    tokens.push_back(current_token);
                }
                current_token.clear();
            }
        }
    }
    
    if (!current_token.empty()) {
        if (stop_words.find(current_token) == stop_words.end()) {
            tokens.push_back(current_token);
        }
    }
    
    return tokens;
}

void TextPreprocessor::preprocess_document(Document& doc) const {
    doc.tokens = tokenize(doc.raw_text);
}

void TextPreprocessor::preprocess_documents_sequential(std::vector<Document>& docs) const {
    for (size_t i = 0; i < docs.size(); ++i) {
        docs[i].tokens = tokenize(docs[i].raw_text);
    }
}

void TextPreprocessor::preprocess_documents_parallel(std::vector<Document>& docs, int num_threads) const {
    int n = static_cast<int>(docs.size());
    #pragma omp parallel for schedule(static) num_threads(num_threads > 0 ? num_threads : omp_get_max_threads())
    for (int i = 0; i < n; ++i) {
        docs[i].tokens = tokenize(docs[i].raw_text);
    }
}
