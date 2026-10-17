import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from jsonschema import Draft202012Validator, ValidationError

from tasks.business_finance.sec_10k_financial_parsing.scripts.score_outputs import _qa_score

TASK_DIR = Path(__file__).resolve().parents[2] / "tasks/business_finance/sec_10k_financial_parsing"
ASSETS_DIR = TASK_DIR / "assets"


@pytest.fixture
def qa_example():
    return json.loads((ASSETS_DIR / "qa_answers_example.json").read_text())


@pytest.fixture
def qa_validator():
    schema = json.loads((ASSETS_DIR / "qa_answers_schema.json").read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def test_example_and_question_formats_agree_with_public_schema(qa_example, qa_validator):
    qa_validator.validate(qa_example)
    questions = json.loads((ASSETS_DIR / "questions.json").read_text())
    assert {question["id"] for question in questions} == {"Q1", "Q2", "Q3"}
    for question in questions:
        answer_schema = qa_validator.schema["properties"][question["id"]]
        assert set(question["answer_format"]) == set(answer_schema["required"])
        assert set(qa_example[question["id"]]) == set(question["answer_format"])
        assert f"key {question['id']}" in question["answer_location"]
    assert qa_validator.schema["properties"]["Q1"]["properties"]["pct_increase"]["type"] == "number"
    assert qa_validator.schema["properties"]["Q2"]["properties"]["fiscal_year"]["type"] == "integer"
    assert qa_validator.schema["properties"]["Q3"]["properties"]["ranking"]["items"] == {
        "type": "string"
    }


@pytest.mark.parametrize("mistake", ["list", "wrapper", "ranking_list", "missing_question", "year"])
def test_schema_explains_previously_implicit_requirements(qa_example, qa_validator, mistake):
    if mistake == "list":
        qa_example = [{"id": key, "answer": value} for key, value in qa_example.items()]
    elif mistake == "wrapper":
        qa_example = {key: {"answer": value} for key, value in qa_example.items()}
    elif mistake == "ranking_list":
        qa_example["Q3"] = qa_example["Q3"]["ranking"]
    elif mistake == "missing_question":
        del qa_example["Q2"]
    else:
        qa_example["Q2"]["fiscal_year"] = "2019"
    with pytest.raises(ValidationError):
        qa_validator.validate(qa_example)


def test_scorer_accepts_documented_shape_without_relaxing_list_parser(tmp_path, qa_example):
    reference = tmp_path / "reference.json"
    prediction = tmp_path / "prediction.json"
    reference.write_text(json.dumps(qa_example))
    prediction.write_text(json.dumps(qa_example))
    assert _qa_score(prediction, reference) == 1.0
    prediction.write_text(
        json.dumps([{"id": key, "answer": value} for key, value in qa_example.items()])
    )
    assert _qa_score(prediction, reference) == 0.0
    qa_example["Q1"]["ticker"] = "WRONG"
    prediction.write_text(json.dumps(qa_example))
    assert _qa_score(prediction, reference) == pytest.approx(2 / 3)


def test_runtime_and_task_card_publish_identical_answer_contract(qa_example):
    from tasks.business_finance.sec_10k_financial_parsing import main

    config = main.Sec10KFinancialParsingConfig()
    card = json.loads((TASK_DIR / "task_card.json").read_text())
    runtime_contract = config.task_description.split("Analytical answer format:\n")[1].split(
        "Important requirements:\n"
    )[0]
    card_contract = (
        card["taskPrompt"]
        .split("Analytical answer format:\n")[1]
        .split("Important requirements:\n")[0]
    )
    assert runtime_contract == card_contract
    assert "top-level JSON object (dictionary)" in runtime_contract
    example_text = runtime_contract.split("```json\n")[1].split("```", 1)[0]
    assert json.loads(example_text) == qa_example
    metadata = config.to_metadata()
    assert metadata["qa_schema_file"].endswith("/input/schema/qa_answers_schema.json")
    assert metadata["qa_example_file"].endswith("/input/schema/qa_answers_example.json")


@pytest.mark.asyncio
async def test_setup_refreshes_baked_question_contract_and_stages_schema(monkeypatch):
    from tasks.business_finance.sec_10k_financial_parsing import main

    metadata = main.Sec10KFinancialParsingConfig().to_metadata()
    task = SimpleNamespace(metadata=metadata)
    files = {metadata["questions_file"]: "outdated baked question list"}
    setup = AsyncMock()
    monkeypatch.setattr(main, "_setup", setup)

    async def write_file(path, content):
        setup.assert_awaited_once()
        files[path] = content

    session = SimpleNamespace(
        write_file=AsyncMock(side_effect=write_file),
        read_file=AsyncMock(side_effect=lambda path: files[path]),
    )
    await main.start(task, session)
    assert session.write_file.await_count == 3
    assert session.read_file.await_count == 3
    for filename, key in [
        ("questions.json", "questions_file"),
        ("qa_answers_schema.json", "qa_schema_file"),
        ("qa_answers_example.json", "qa_example_file"),
    ]:
        assert files[metadata[key]] == (ASSETS_DIR / filename).read_text()
    assert set(files) == {
        metadata["questions_file"],
        metadata["qa_schema_file"],
        metadata["qa_example_file"],
    }


@pytest.mark.asyncio
async def test_setup_rejects_stale_baked_readback(monkeypatch):
    from tasks.business_finance.sec_10k_financial_parsing import main

    task = SimpleNamespace(metadata=main.Sec10KFinancialParsingConfig().to_metadata())
    monkeypatch.setattr(main, "_setup", AsyncMock())
    session = SimpleNamespace(
        write_file=AsyncMock(),
        read_file=AsyncMock(return_value="old baked input was not replaced"),
    )
    with pytest.raises(RuntimeError, match="contract readback mismatch"):
        await main.start(task, session)
    session.write_file.assert_awaited_once()
    session.read_file.assert_awaited_once_with(task.metadata["questions_file"])
