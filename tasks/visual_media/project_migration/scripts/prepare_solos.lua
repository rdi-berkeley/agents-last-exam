local session = load_session(arg[1], arg[2])
assert(session)
sleep(6)
local routes = Session:get_routes()
local by_name = {}
for route in routes:iter() do
    by_name[route:name()] = route
    route:solo_control():set_value(0, PBD.GroupControlDisposition.NoGroup)
end
for line in io.lines(arg[3]) do
    local snapshot, name = line:match("([^\t]+)\t(.+)")
    assert(by_name[name], name)
    by_name[name]:solo_control():set_value(1, PBD.GroupControlDisposition.NoGroup)
    assert(Session:save_state(snapshot, false, false, false) == 0)
    by_name[name]:solo_control():set_value(0, PBD.GroupControlDisposition.NoGroup)
end
by_name = nil
routes = nil
collectgarbage()
Session:close()
