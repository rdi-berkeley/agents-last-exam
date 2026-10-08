from functools import wraps


def install_tool_binding():
    import FreeCAD
    from Path.Dressup import Utils as Dressup
    from Path.Main.Gui import Simulator

    from tool_library import set_simulator_tool

    original = Simulator.PathSimulation.SetupOperation
    if getattr(original, "original_tool_binding", False):
        return

    @wraps(original)
    def setup_operation(simulation, operation_index):
        result = original(simulation, operation_index)
        if simulation.tool is None:
            return result
        controller = Dressup.toolController(simulation.operation)
        if hasattr(controller, "OriginalAssembly"):
            shapes = set_simulator_tool(simulation.voxSim, controller, 0.05 * simulation.accuracy)
            simulation.cutTool.Shape = shapes["Cutter"]
        return result

    setup_operation.original_tool_binding = True
    Simulator.PathSimulation.SetupOperation = setup_operation
    if FreeCAD.Version()[:3] == ["1", "1", "3"]:
        FreeCAD.Console.PrintWarning(
            "The bundled voxel preview is inaccurate on sloped cuts. "
            "Use preview_stock.py for a checked stock model.\n"
        )
