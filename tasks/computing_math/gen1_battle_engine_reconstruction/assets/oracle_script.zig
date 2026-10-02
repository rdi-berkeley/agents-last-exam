//! Lane-agnostic Gen I battle oracle.
//!
//! Runs one battle to completion from a pair of seeds and writes an observable
//! transcript: the protocol log emitted by each update, then the terminal
//! result and the full serialised battle state. Output is the ground truth the
//! benchmarked agent must reproduce; it deliberately contains no rules.
const pkmn = @import("pkmn");
const std = @import("std");

const gen1 = pkmn.gen1.helpers;

/// Length of the roll tape. Large enough that no battle in the shipped corpus
/// exhausts it; the engine panics rather than wrapping if it ever does.
const TAPE = 1 << 10;

/// Tape generator. Deliberately ours and not the engine's, so the tape is a
/// published input to the task rather than a thing the agent must reverse.
fn fillTape(tape: []u8, seed: u64) void {
    // splitmix64 finalizer first: `seed | 1` alone collides every even seed with
    // its odd successor, which silently halves the number of distinct tapes.
    var z = seed +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    z = z ^ (z >> 31);
    var x = z | 1;
    for (tape) |*b| {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        b.* = @truncate(x >> 24);
    }
}

/// Lead index (1-6) from a "<lead>:<moves>" side spec.
fn parseLead(spec: []const u8) !u8 {
    const i = std.mem.indexOfScalar(u8, spec, ':') orelse return error.MissingLead;
    const v = try std.fmt.parseUnsigned(u8, spec[0..i], 10);
    if (v < 1 or v > 6) return error.LeadOutOfRange;
    return v - 1;
}

fn afterColon(spec: []const u8) []const u8 {
    const i = std.mem.indexOfScalar(u8, spec, ':') orelse return spec;
    return spec[i + 1 ..];
}

/// Parse "1,3,4" into move indices. Returns how many were parsed.
fn parseScript(text: []const u8, out: []u6) !usize {
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, text, ',');
    while (it.next()) |tok| {
        if (tok.len == 0) continue;
        if (n == out.len) return error.TooManyMoves;
        const v = try std.fmt.parseUnsigned(u8, tok, 10);
        if (v < 1 or v > 4) return error.MoveOutOfRange;
        out[n] = @intCast(v);
        n += 1;
    }
    if (n == 0) return error.EmptyScript;
    return n;
}

/// Prefer the scripted move; fall back to the first legal option when it is not
/// available, which happens on a forced switch after a faint.
fn pick(opts: []const pkmn.Choice, want: u6) pkmn.Choice {
    for (opts) |c| if (c.type == .Move and c.data == want) return c;
    return opts[0];
}

fn hex(w: *std.Io.Writer, bytes: []const u8) !void {
    for (bytes) |b| try w.print("{x:0>2}", .{b});
}

const TEAM1 = [_]gen1.Pokemon{
            .{ .species = .Electrode, .moves = &.{ .Wrap, .Thrash, .Dig, .PinMissile } },
            .{ .species = .Tauros, .moves = &.{ .DoubleKick, .Twineedle, .Slash, .HornDrill } },
            .{ .species = .Gengar, .moves = &.{ .Swift, .JumpKick, .DoubleEdge, .MegaDrain } },
            .{ .species = .Scyther, .moves = &.{ .DreamEater, .Explosion, .Substitute, .Reflect } },
            .{ .species = .Charizard, .moves = &.{ .LightScreen, .Haze, .Mist, .Disable } },
            .{ .species = .Jynx, .moves = &.{ .Mimic, .Conversion, .SeismicToss, .SuperFang } },
};

const TEAM2 = [_]gen1.Pokemon{
            .{ .species = .Dragonite, .moves = &.{ .FocusEnergy, .Bide, .Rage, .LeechSeed } },
            .{ .species = .Exeggutor, .moves = &.{ .SleepPowder, .ThunderWave, .Toxic, .ConfuseRay } },
            .{ .species = .Machamp, .moves = &.{ .Blizzard, .FireBlast, .BodySlam, .Bite } },
            .{ .species = .Chansey, .moves = &.{ .Psybeam, .Smog, .AuroraBeam, .Acid } },
            .{ .species = .Rhydon, .moves = &.{ .BubbleBeam, .Psychic, .Recover, .HyperBeam } },
            .{ .species = .Slowbro, .moves = &.{ .SwordsDance, .Amnesia, .Agility, .Screech } },
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.arena.allocator();
    const args = try init.minimal.args.toSlice(allocator);

    var err = std.Io.File.stderr().writer(init.io, &.{});
    if (args.len != 4 and args.len != 5) {
        try err.interface.print(
            "Usage: {s} <tape-seed|tape-file> <p1-spec> <p2-spec> [max-updates]\n" ++
            "  a side spec is <lead>:<moves>, e.g. 3:1,4  (lead 1-6, moves 1-4)\n",
            .{args[0]},
        );
        std.process.exit(1);
    }
    // A decimal argument seeds the tape generator, which is how the corpus is
    // built. Anything else is a path to a hex tape. Grading always uses the file
    // form: an agent must never have to reverse-engineer the tape generator,
    // which is an irrelevant puzzle with nothing to do with the battle rules.
    const tape_seed: ?u64 = std.fmt.parseUnsigned(u64, args[1], 10) catch null;
    var s1: [64]u6 = undefined;
    var s2: [64]u6 = undefined;
    const lead1 = parseLead(args[2]) catch {
        try err.interface.print("Invalid p1-spec: {s}\n", .{args[2]});
        std.process.exit(1);
    };
    const lead2 = parseLead(args[3]) catch {
        try err.interface.print("Invalid p2-spec: {s}\n", .{args[3]});
        std.process.exit(1);
    };
    const n1s = parseScript(afterColon(args[2]), &s1) catch {
        try err.interface.print("Invalid p1-moves: {s}\n", .{args[2]});
        std.process.exit(1);
    };
    const n2s = parseScript(afterColon(args[3]), &s2) catch {
        try err.interface.print("Invalid p2-moves: {s}\n", .{args[3]});
        std.process.exit(1);
    };

    var obuf: [1 << 16]u8 = undefined;
    var out = std.Io.File.stdout().writer(init.io, &obuf);
    const w = &out.interface;

    // Fixed teams. Drawing them from the RNG would confound any comparison
    // between builds: the two rule sets use different RNGs, so random teams
    // differ before a single move is made.
    // One Pokemon per side. With nobody to switch in, a faint ends the battle
    // instead of handing the move script to the next member, which is what keeps
    // a scenario family isolated to the mechanic it targets.
    const t1 = [_]gen1.Pokemon{TEAM1[lead1]};
    const t2 = [_]gen1.Pokemon{TEAM2[lead2]};
    var battle = gen1.Battle.fixed([_]u8{0} ** TAPE, &t1, &t2);

    if (tape_seed) |sd| {
        fillTape(&battle.rng.rolls, sd);
    } else {
        const text = std.Io.Dir.cwd().readFileAlloc(
            init.io, args[1], allocator, .limited(TAPE * 2 + 64),
        ) catch {
            try err.interface.print("Cannot read tape file: {s}\n", .{args[1]});
            std.process.exit(1);
        };
        var n: usize = 0;
        var i: usize = 0;
        while (n < TAPE and i + 1 < text.len) : (i += 2) {
            while (i < text.len and (text[i] == '\n' or text[i] == ' ')) i += 1;
            if (i + 1 >= text.len) break;
            const hi = std.fmt.charToDigit(text[i], 16) catch break;
            const lo = std.fmt.charToDigit(text[i + 1], 16) catch break;
            battle.rng.rolls[n] = hi * 16 + lo;
            n += 1;
        }
        if (n < TAPE) {
            try err.interface.print("Tape too short: {d} of {d} rolls\n", .{ n, TAPE });
            std.process.exit(1);
        }
    }

    // The initial state is observable: the agent is told the teams it must
    // battle with, otherwise the task would be underdetermined.
    const STATE = @offsetOf(@TypeOf(battle), "rng");
    try w.writeAll("init ");
    try hex(w, std.mem.toBytes(battle)[0..STATE]);
    try w.writeAll("\n");

    // Scripted scenarios can stalemate: two sides both picking a non-damaging
    // move never end. Cap the run so a scenario is always bounded, and mark the
    // transcript so the corpus builder can drop or keep it deliberately.
    const max_updates: usize = if (args.len == 5)
        std.fmt.parseUnsigned(usize, args[4], 10) catch 512
    else
        512;

    var choices: [pkmn.CHOICES_SIZE]pkmn.Choice = undefined;
    var k1: usize = 0;
    var k2: usize = 0;

    var buf: [pkmn.LOGS_SIZE]u8 = undefined;
    var writer: pkmn.protocol.Writer = .{ .buffer = &buf };
    var options = pkmn.battle.options(
        pkmn.protocol.FixedLog{ .writer = &writer },
        pkmn.gen1.chance.NULL,
        pkmn.gen1.calc.NULL,
    );

    var c1: pkmn.Choice = .{};
    var c2: pkmn.Choice = .{};
    var n: usize = 0;

    var result = try battle.update(c1, c2, &options);
    while (result.type == .None and n < max_updates) : (result = try battle.update(c1, c2, &options)) {
        try w.print("u{d} {d} {d} {d} {d} ", .{ n, @intFromEnum(c1.type), c1.data, @intFromEnum(c2.type), c2.data });
        try hex(w, buf[0..writer.pos]);
        try w.writeAll("\n");
        n += 1;

        const m1 = battle.choices(.P1, result.p1, &choices);
        c1 = pick(choices[0..m1], s1[k1 % n1s]);
        k1 += 1;
        const m2 = battle.choices(.P2, result.p2, &choices);
        c2 = pick(choices[0..m2], s2[k2 % n2s]);
        k2 += 1;
        writer.reset();
    }
    try w.print("u{d} {d} {d} {d} {d} ", .{ n, @intFromEnum(c1.type), c1.data, @intFromEnum(c2.type), c2.data });
    try hex(w, buf[0..writer.pos]);
    try w.writeAll("\n");

    try w.print("result {d}{s}\n",
        .{ @intFromEnum(result.type), if (n >= max_updates) " truncated" else "" });
    try w.writeAll("state ");
    try hex(w, std.mem.toBytes(battle)[0..STATE]);
    try w.writeAll("\n");
    try w.flush();
}
