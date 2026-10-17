local session = load_session(arg[1], arg[2])
assert(session)
sleep(6)
local routes = Session:get_routes()
local by_name = {}
for route in routes:iter() do
    by_name[route:name()] = route
    route:solo_control():set_value(0, PBD.GroupControlDisposition.NoGroup)
end
for name in io.lines(arg[3]) do
    assert(by_name[name], name)
    by_name[name]:solo_control():set_value(1, PBD.GroupControlDisposition.NoGroup)
end
assert(Session:save_state(arg[2], false, false, false) == 0)
by_name = nil
routes = nil
collectgarbage()
Session:close()
