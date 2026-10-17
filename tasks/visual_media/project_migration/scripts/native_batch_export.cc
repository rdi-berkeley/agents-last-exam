#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "ardour/export_pointers.h"
#include "pbd/xml++.h"

using namespace ARDOUR;

static void* export_factory = nullptr;

template<typename Function>
Function resolve(const char* symbol) {
    auto address = dlsym(RTLD_NEXT, symbol);
    if (!address) {
        std::cerr << "Missing native API symbol: " << symbol << std::endl;
        std::exit(71);
    }
    return reinterpret_cast<Function>(address);
}

extern "C" ExportChannelConfigPtr capture_factory(void* factory)
    asm("_ZN6ARDOUR20ExportElementFactory18add_channel_configEv");

extern "C" ExportChannelConfigPtr capture_factory(void* factory) {
    export_factory = factory;
    return resolve<ExportChannelConfigPtr(*)(void*)>(
        "_ZN6ARDOUR20ExportElementFactory18add_channel_configEv")(factory);
}

extern "C" bool queue_exports(void* handler, ExportTimespanPtr timespan,
    ExportChannelConfigPtr channels, ExportFormatSpecPtr format,
    ExportFilenamePtr filename, BroadcastInfoPtr broadcast)
    asm("_ZN6ARDOUR13ExportHandler17add_export_configENS_19ComparableSharedPtrINS_14ExportTimespanEEEN5boost10shared_ptrINS_26ExportChannelConfigurationEEENS5_INS_25ExportFormatSpecificationEEENS5_INS_14ExportFilenameEEENS5_IN12AudioGrapher13BroadcastInfoEEE");

extern "C" bool queue_exports(void* handler, ExportTimespanPtr timespan,
    ExportChannelConfigPtr channels, ExportFormatSpecPtr format,
    ExportFilenamePtr filename, BroadcastInfoPtr broadcast) {
    using Queue = bool(*)(void*, ExportTimespanPtr, ExportChannelConfigPtr,
                         ExportFormatSpecPtr, ExportFilenamePtr, BroadcastInfoPtr);
    auto queue = resolve<Queue>(
        "_ZN6ARDOUR13ExportHandler17add_export_configENS_19ComparableSharedPtrINS_14ExportTimespanEEEN5boost10shared_ptrINS_26ExportChannelConfigurationEEENS5_INS_25ExportFormatSpecificationEEENS5_INS_14ExportFilenameEEENS5_IN12AudioGrapher13BroadcastInfoEEE");
    const char* manifest = std::getenv("ARDOUR_BATCH_MANIFEST");
    if (!manifest) return queue(handler, timespan, channels, format, filename, broadcast);
    if (!export_factory) std::exit(72);
    auto create_channels = resolve<ExportChannelConfigPtr(*)(void*)>(
        "_ZN6ARDOUR20ExportElementFactory18add_channel_configEv");
    auto copy_filename = resolve<ExportFilenamePtr(*)(void*, ExportFilenamePtr)>(
        "_ZN6ARDOUR20ExportElementFactory17add_filename_copyEN5boost10shared_ptrINS_14ExportFilenameEEE");
    auto set_channels = resolve<int(*)(void*, const XMLNode&)>(
        "_ZN6ARDOUR26ExportChannelConfiguration9set_stateERK7XMLNode");
    auto valid_channels = resolve<bool(*)(const void*)>(
        "_ZNK6ARDOUR26ExportChannelConfiguration23all_channels_have_portsEv");
    auto set_folder = resolve<bool(*)(void*, std::string)>(
        "_ZN6ARDOUR14ExportFilename10set_folderENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE");
    std::ifstream stream(manifest);
    std::string line;
    unsigned count = 0;
    while (std::getline(stream, line)) {
        std::istringstream row(line);
        std::string config_path, output_folder;
        if (!std::getline(row, config_path, '\t') || !std::getline(row, output_folder))
            std::exit(73);
        XMLTree config(config_path);
        auto branch_channels = create_channels(export_factory);
        auto branch_filename = copy_filename(export_factory, filename);
        if (!config.root() || set_channels(branch_channels.get(), *config.root()) != 0
            || !valid_channels(branch_channels.get())
            || !set_folder(branch_filename.get(), output_folder)) std::exit(74);
        if (!queue(handler, timespan, branch_channels, format, branch_filename, broadcast))
            std::exit(75);
        ++count;
        std::cout << "BATCH_CONFIG " << config_path << " " << output_folder << std::endl;
    }
    if (count < 1) std::exit(76);
    std::cout << "BATCH_SHARED_TIMESPAN_CONFIGS " << count << std::endl;
    return true;
}
