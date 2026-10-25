const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const showdown =
        b.option(bool, "showdown", "Enable Pokemon Showdown compatibility mode") orelse false;

    const exe = b.addExecutable(.{
        .name = if (showdown) "oracle-showdown" else "oracle",
        .root_module = b.createModule(.{
            .root_source_file = b.path("oracle_script.zig"),
            .optimize = optimize,
            .target = target,
            // Shipped to the task VM as an opaque oracle: no symbols, no
            // debug info, statically linked so it needs nothing from the image.
            .strip = b.option(bool, "strip", "Strip debug symbols") orelse false,
        }),
    });
    const pkmn = b.dependency("pkmn", .{ .showdown = showdown, .log = true });
    exe.root_module.addImport("pkmn", pkmn.module("pkmn"));
    b.installArtifact(exe);
}
