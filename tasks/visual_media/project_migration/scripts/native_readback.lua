local session = load_session(arg[1], arg[2])
assert(session)
sleep(6)
local output = assert(io.open(arg[3], "w"))
local routes = Session:get_routes()
for route in routes:iter() do
    local index = 0
    while true do
        local processor = route:nth_processor(index)
        if processor:isnil() then break end
        if processor:active() and not processor:to_unknownprocessor():isnil() then
            output:write("unavailable\t" .. route:name() .. "\t" .. processor:name() .. "\n")
        end
        index = index + 1
    end
    local track = route:to_track()
    if not track:isnil() then
        local playlist = track:playlist()
        local regions = playlist:region_list()
        for region in regions:iter() do
            local midi = region:to_midiregion()
            assert(not midi:isnil())
            local count = 0
            local model = midi:model()
            local notes = ARDOUR.LuaAPI.note_list(model)
            for note in notes:iter() do
                output:write(string.format("note\t%s\t%s\t%.0f\t%.0f\t%d\t%d\t%d\t%d\n", route:name(), region:name(), note:time():to_double() * 1920, (note:time():to_double() + note:length():to_double()) * 1920, note:channel(), note:note(), note:velocity(), note:off_velocity()))
                count = count + 1
            end
            output:write(string.format("region\t%s\t%s\t%d\t%d\t%d\t%s\n", route:name(), region:name(), count, region:position(), region:length(), tostring(region:muted())))
        end
    end
end
output:close()
if arg[4] ~= "read-only" then
    assert(Session:save_state(arg[4], false, false, false) == 0)
end
Session:close()
print("MIGRATION_NATIVE_REOPEN_OK")
