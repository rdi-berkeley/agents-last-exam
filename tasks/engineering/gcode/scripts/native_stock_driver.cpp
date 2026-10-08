#include <camotics/sim/CutSim.h>
#include <camotics/sim/Simulation.h>
#include <gcode/ToolPath.h>
#include <gcode/ToolTable.h>
#include <cbang/Exception.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

struct Component {
    unsigned number;
    double offset;
};

double readNumber() {
    double value;
    if (!(std::cin >> value) || !std::isfinite(value))
        throw std::runtime_error("Invalid finite numeric input");
    return value;
}

int main(int argc, char **argv) {
    try {
        if (argc != 2) return 2;
        cb::Vector3D lower, upper;
        for (unsigned axis = 0; axis < 3; ++axis) lower[axis] = readNumber();
        for (unsigned axis = 0; axis < 3; ++axis) {
            upper[axis] = readNumber();
            if (upper[axis] <= lower[axis]) return 2;
        }
        double resolution = readNumber();
        unsigned toolCount;
        if (!(std::cin >> toolCount) || toolCount > 1000 || resolution <= 0) return 2;
        GCode::ToolTable table;
        std::map<unsigned, std::vector<Component>> components;
        unsigned nextNumber = 1;
        for (unsigned toolIndex = 0; toolIndex < toolCount; ++toolIndex) {
            unsigned original, count;
            if (!(std::cin >> original >> count) || count == 0 || count > 1000 || components.count(original)) return 2;
            for (unsigned part = 0; part < count; ++part) {
                char kind;
                if (!(std::cin >> kind)) return 2;
                double length = readNumber(), topRadius = readNumber(), bottomRadius = readNumber(), offset = readNumber();
                if (length <= 0 || topRadius <= 0 || bottomRadius < 0 || offset < 0) return 2;
                GCode::Tool tool(nextNumber);
                if (kind == 'B') tool.setShape(GCode::ToolShape::TS_BALLNOSE);
                else if (kind == 'N') tool.setShape(GCode::ToolShape::TS_SNUBNOSE);
                else return 2;
                tool.setLength(length);
                tool.setDiameter(2 * topRadius);
                tool.setSnubDiameter(2 * bottomRadius);
                table.set(tool);
                components[original].push_back({nextNumber++, offset});
            }
        }
        unsigned motionCount;
        if (!(std::cin >> motionCount) || motionCount > 1000000) return 2;
        cb::SmartPointer<GCode::ToolPath> path = new GCode::ToolPath(table);
        cb::SmartPointer<std::string> filename = new std::string("native-canonical-motion");
        unsigned feedCount = 0;
        for (unsigned index = 0; index < motionCount; ++index) {
            unsigned original, rapid;
            if (!(std::cin >> original >> rapid) || rapid > 1) return 2;
            cb::Vector3D start, end;
            for (unsigned axis = 0; axis < 3; ++axis) start[axis] = readNumber();
            for (unsigned axis = 0; axis < 3; ++axis) end[axis] = readNumber();
            if (rapid) continue;
            if (!components.count(original)) return 2;
            ++feedCount;
            for (const auto &component : components.at(original)) {
                GCode::Axes nativeStart, nativeEnd;
                for (unsigned axis = 0; axis < 3; ++axis) {
                    nativeStart.setIndex(axis, start[axis] + (axis == 2 ? component.offset : 0));
                    nativeEnd.setIndex(axis, end[axis] + (axis == 2 ? component.offset : 0));
                }
                GCode::Move move(GCode::MoveType::MOVE_CUTTING, nativeStart, nativeEnd,
                    path->size(), component.number, 1, 10000, index + 1, filename, 1);
                path->move(move);
            }
        }
        std::string unexpected;
        if (std::cin >> unexpected || !std::cin.eof() || path->empty()) return 2;
        CAMotics::Simulation simulation(path, 0, 0, cb::Rectangle3D(lower, upper), resolution,
            std::numeric_limits<double>::max(), CAMotics::RenderMode(), 1);
        CAMotics::CutSim simulator;
        auto surface = simulator.computeSurface(simulation);
        std::ofstream output(argv[1], std::ios::binary);
        if (!output) return 2;
        surface->writeSTL(output, true, "Native CAMotics stock", simulation.computeHash());
        output.close();
        if (!output) return 2;
        std::cout << "{\"motion_count\":" << motionCount << ",\"feed_count\":" << feedCount
                  << ",\"component_motion_count\":" << path->size() << "}\n";
        return 0;
    } catch (const cb::Exception &error) {
        std::cerr << error << '\n';
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
    }
    return 2;
}
