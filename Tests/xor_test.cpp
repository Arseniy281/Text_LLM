#include <iostream>
#include <fstream>
#include <string>
#include <regex>
#include <cctype>

bool IsServiceLine(const std::string& line) {
    if (line.find("Sync & corrections") != std::string::npos) {
        return true;
    }

    if (line.find("www.") != std::string::npos) {
        return true;
    }

    if (line.find("http://") != std::string::npos) {
        return true;
    }

    if (line.find("https://") != std::string::npos) {
        return true;
    }

    return false;
}

std::string RemoveTags(const std::string& line) {
    return std::regex_replace(
        line,
        std::regex("<[^>]*>"),
        ""
    );
}

std::string Trim(const std::string& line) {
    size_t start = line.find_first_not_of(" \t\r\n");

    if (start == std::string::npos) {
        return "";
    }

    size_t end = line.find_last_not_of(" \t\r\n");

    return line.substr(start, end - start + 1);
}

bool IsOnlyNoise(const std::string& line) {
    if (line.empty()) {
        return true;
    }

    bool has_text = false;

    for (char c : line) {
        unsigned char ch = static_cast<unsigned char>(c);

        if (std::isalpha(ch) || std::isdigit(ch)) {
            has_text = true;
            break;
        }
    }

    return !has_text;
}

bool IsSoundEffect(const std::string& line) {
    if (line.size() < 3) {
        return false;
    }

    if (line.front() != '(' || line.back() != ')') {
        return false;
    }

    std::string text = line.substr(1, line.size() - 2);

    for (char& c : text) {
        c = static_cast<char>(
            std::tolower(static_cast<unsigned char>(c))
        );
    }

    const std::string sounds[] = {
        "laugh",
        "laughing",
        "sigh",
        "sighs",
        "cry",
        "crying",
        "whimper",
        "whimpers",
        "snarl",
        "snarling",
        "growl",
        "growling",
        "groan",
        "groaning",
        "moan",
        "moaning",
        "gasp",
        "gasping",
        "breath",
        "breathing",
        "cough",
        "coughing",
        "scream",
        "screaming",
        "shout",
        "shouting",
        "door",
        "footsteps",
        "gunshot",
        "gunshots",
        "explosion",
        "explosions",
        "phone buzzing",
        "cell phone buzzing",
        "music",
        "applause",
        "mouthing"
    };

    for (const std::string& sound : sounds) {
        if (text.find(sound) != std::string::npos) {
            return true;
        }
    }

    return false;
}

std::string RemoveUnderscores(const std::string& line) {
    std::string result;

    for (char c : line) {
        if (c != '_') {
            result += c;
        }
    }

    return result;
}

int main() {
    const std::string input_path =
        "../Data/english_clean.txt";

    const std::string output_path =
        "../Data/english_final.txt";

    std::ifstream input(input_path);
    std::ofstream output(output_path);

    if (!input.is_open()) {
        std::cerr << "Cannot open english_clean.txt\n";
        return 1;
    }

    if (!output.is_open()) {
        std::cerr << "Cannot create english_final.txt\n";
        return 1;
    }

    std::string line;

    size_t total_lines = 0;
    size_t saved_lines = 0;
    size_t removed_lines = 0;

    while (std::getline(input, line)) {
        total_lines++;

        line = Trim(line);

        if (line.empty()) {
            continue;
        }

        if (IsServiceLine(line)) {
            removed_lines++;
            continue;
        }

        line = RemoveTags(line);
        line = RemoveUnderscores(line);
        line = Trim(line);

        if (line.empty()) {
            removed_lines++;
            continue;
        }

        if (IsOnlyNoise(line)) {
            removed_lines++;
            continue;
        }

        if (IsSoundEffect(line)) {
            removed_lines++;
            continue;
        }

        output << line << '\n';
        saved_lines++;
    }

    input.close();
    output.close();

    std::cout << "Done!\n";
    std::cout << "Total lines:   " << total_lines << '\n';
    std::cout << "Saved lines:   " << saved_lines << '\n';
    std::cout << "Removed lines: " << removed_lines << '\n';
    std::cout << "Output:        " << output_path << '\n';

    return 0;
}