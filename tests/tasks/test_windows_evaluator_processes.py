import base64
import json
from unittest.mock import AsyncMock
from types import SimpleNamespace

import pytest

from tasks.engineering.humanoid_velocity_tracking_policy.scripts.simulation_process import (
    SimulationError,
    run_simulation,
    validate_rollout_results,
)
from tasks.visual_media.human_mesh_animation_reproduction.main import _blender_can_open_blend


class SimulationSession:
    def __init__(self, *, status=None, result=None, alive=True, launch_code=0, prerequisites=True):
        self.status = status
        self.result = result
        self.alive = alive
        self.launch_code = launch_code
        self.prerequisites = prerequisites
        self.commands = []
        self.write_file = AsyncMock()

    async def file_exists(self, path):
        if path.endswith('.status.json'):
            return self.status is not None
        if path == 'results.json':
            return self.result is not None
        return self.prerequisites

    async def read_bytes(self, path):
        content = self.status if path.endswith('.status.json') else self.result
        return json.dumps(content).encode('utf-8-sig')

    async def run_command(self, command, check=False):
        script = command
        if '-EncodedCommand ' in command:
            script = base64.b64decode(command.split()[-1]).decode('utf-16-le')
        self.commands.append(script)
        if '$process.Id' in script:
            return {'return_code': self.launch_code, 'stdout': '1234'}
        if 'Get-Process' in script:
            return {'return_code': 0 if self.alive else 3, 'stdout': ''}
        return {'return_code': 0, 'stdout': ''}


async def simulate(session, timeout=0.05):
    return await run_simulation(
        session, executable=r'C:\Isaac Lab\python.exe',
        arguments=['-u', r'C:\evaluation files\run_eval.py'],
        working_directory=r'C:\evaluation files', results_path='results.json',
        timeout=timeout, poll_interval=0.001,
    )


@pytest.mark.asyncio
async def test_simulation_requires_successful_exit_and_complete_results():
    result = {'lin_vel_rmse_xy': .2, 'ang_vel_mae_z': .1, 'fall_rate': 0}
    result['per_seed'] = {key: [value] * 10 for key, value in result.items()}
    result['per_seed']['seeds'] = list(range(10))
    session = SimulationSession(status={'exit_code': 0}, result=result)
    assert await simulate(session)
    script = session.write_file.call_args.args[1]
    assert 'OMNI_KIT_ACCEPT_EULA' in script
    assert '-RedirectStandardOutput' in script
    assert '-RedirectStandardError' in script
    assert '"C:\\evaluation files\\run_eval.py"' in script


@pytest.mark.asyncio
@pytest.mark.parametrize('kwargs,match', [
    ({'prerequisites': False}, 'prerequisite is missing'),
    ({'launch_code': 1}, 'launch failed'),
    ({'status': {'exit_code': 7}}, 'exited unsuccessfully'),
    ({'status': {'exit_code': 0}}, 'without a results file'),
    ({'status': {'exit_code': 0}, 'result': {}}, 'incomplete rollout results'),
    ({'alive': False}, 'without a completion record'),
])
async def test_simulation_failures_raise_instead_of_returning_false(kwargs, match):
    with pytest.raises(SimulationError, match=match):
        await simulate(SimulationSession(**kwargs))


@pytest.mark.asyncio
async def test_timeout_terminates_entire_process_tree():
    session = SimulationSession()
    with pytest.raises(SimulationError, match='incomplete evaluation, no score'):
        await simulate(session, timeout=0)
    assert 'taskkill /PID 1234 /T /F' in session.commands


@pytest.mark.asyncio
async def test_evaluator_does_not_turn_simulator_failure_into_zero(monkeypatch):
    from tasks.engineering.humanoid_velocity_tracking_policy import main as task

    session = AsyncMock()
    session.file_exists.return_value = True
    session.read_bytes.return_value = b'fixture'
    meta = task.HumanoidVelocityTrackingConfig(VARIANT_NAME='base').to_metadata()
    monkeypatch.setattr(task, 'validate_submission', lambda output: {'score': None})
    monkeypatch.setattr(task, '_maybe_run_full_sim', AsyncMock(side_effect=SimulationError('simulator failed')))
    with pytest.raises(SimulationError, match='simulator failed'):
        await task.evaluate(SimpleNamespace(metadata=meta), session)


@pytest.mark.asyncio
async def test_invalid_policy_is_zero_without_launching_simulation(monkeypatch):
    from tasks.engineering.humanoid_velocity_tracking_policy import main as task

    session = AsyncMock()
    session.file_exists.return_value = True
    session.read_bytes.return_value = b'invalid policy syntax!'
    meta = task.HumanoidVelocityTrackingConfig(VARIANT_NAME='base').to_metadata()
    launch = AsyncMock()
    monkeypatch.setattr(task, '_maybe_run_full_sim', launch)
    assert await task.evaluate(SimpleNamespace(metadata=meta), session) == [0.0]
    launch.assert_not_called()


@pytest.mark.parametrize('per_seed', [{}, {'seeds': list(range(10))}, {'seeds': [0, 1]}])
def test_incomplete_seed_data_is_not_a_solver_failure(per_seed):
    with pytest.raises(SimulationError, match='incomplete rollout results'):
        validate_rollout_results({'lin_vel_rmse_xy': .2, 'ang_vel_mae_z': .1, 'fall_rate': 0, 'per_seed': per_seed})


@pytest.mark.asyncio
async def test_blender_setup_failure_is_not_a_solver_zero():
    session = AsyncMock()
    session.run_command.return_value = {'return_code': 125, 'stdout': 'BLEND_CHECK_ERROR: access denied', 'stderr': ''}
    with pytest.raises(RuntimeError, match='Blender validation could not run'):
        await _blender_can_open_blend(session, blender_path='blender.exe', blend_path='scene.blend')


@pytest.mark.asyncio
@pytest.mark.parametrize('return_code,output,expected', [
    (0, 'Read blend: scene.blend\nBLEND_OPEN_CHECK_OK', True),
    (1, 'Read blend: scene.blend\nBLEND_OPEN_CHECK_OK', False),
    (0, 'BLEND_OPEN_CHECK_OK', False),
    (0, 'Read blend: scene.blend\nERROR Cannot read file BLEND_OPEN_CHECK_OK', False),
])
async def test_blender_exit_status_and_marker_are_both_required(return_code, output, expected):
    session = AsyncMock()
    session.run_command.return_value = {'return_code': return_code, 'stdout': output, 'stderr': ''}
    result = await _blender_can_open_blend(session, blender_path=r'C:\Program Files\Blender\blender.exe', blend_path=r"E:\submission\artist's scene.blend")
    assert result[0] is expected
    command = session.run_command.call_args.args[0]
    script = base64.b64decode(command.split()[-1]).decode('utf-16-le')
    assert "artist''s scene.blend" in script
    assert '--python-expr' not in script
    assert '--python $script' in script
    assert 'exit $code' in script
