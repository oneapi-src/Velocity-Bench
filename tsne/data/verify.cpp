#include "verify.hpp"

int verify(std::string file_golden, std::string file_output, float epsilon, float mismatch_allowed) {

    std::ifstream fg(file_golden);
    std::ifstream fo(file_output);

    if (!fg.is_open() || !fo.is_open()) {
        std::cout << "Error: Unable to open file(s)" << std::endl;
        return 1;
    }

    std::string line_g, line_o;
    std::getline(fg, line_g);
    std::getline(fo, line_o);

    std::istringstream iss_g(line_g);
    std::istringstream iss_o(line_o);

    int Ng, Mg, No, Mo;
    iss_g >> Ng >> Mg;
    iss_o >> No >> Mo;

    if (Ng != No || Mg != Mo) {
        std::cout << "Error: first line of " << file_golden << " and " << file_output << " are different" << std::endl;
        return 2;
    }

    if (Ng < 1) {
        std::cout << "Error: number of points can't be less than 1" << std::endl;
        return 3;
    }

    int mismatch_count = 0;
    while (std::getline(fg, line_g) && std::getline(fo, line_o)) {
        std::istringstream iss_g(line_g);
        std::istringstream iss_o(line_o);

        float xg, yg, xo, yo;
        iss_g >> xg >> yg;
        iss_o >> xo >> yo;

        if (std::abs(xg - xo) / std::abs(xg) > epsilon || std::abs(yg - yo) / std::abs(yg) > epsilon) {
            mismatch_count++;
        }
    }

    float mismatch_percent = static_cast<float>(mismatch_count) / Ng * 100;
    if (mismatch_percent > mismatch_allowed) {
        std::cout << "Error: mismatch: " << mismatch_percent << "% but should be less than " << mismatch_allowed << "%" << std::endl;
        return 4;
    }

    return 0;
}
