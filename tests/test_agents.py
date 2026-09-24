import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch, MagicMock
from agents.lead_researcher import LeadResearcher
from agents.web_scout import WebScout
from agents.base_agent import BaseAgent


class FakeLLM:
    def __init__(self, response: str):
        self.response = response

    def chat(self, system, user, max_tokens=1500):
        return self.response


def test_lead_researcher_parses_json_plan():
    fake = FakeLLM('["sub-task one", "sub-task two", "sub-task three"]')
    agent = LeadResearcher(llm=fake)
    sub_tasks = agent.plan("What is retrieval-augmented generation?")
    assert sub_tasks == ["sub-task one", "sub-task two", "sub-task three"]


def test_lead_researcher_falls_back_on_bad_json():
    fake = FakeLLM("not json at all, just prose response")
    agent = LeadResearcher(llm=fake)
    sub_tasks = agent.plan("some query")
    assert isinstance(sub_tasks, list) and len(sub_tasks) >= 1


def test_base_agent_parse_json_handles_fenced_block():
    raw = '```json\n["a", "b"]\n```'
    assert BaseAgent.parse_json(raw) == ["a", "b"]


def test_base_agent_parse_json_handles_plain_object():
    raw = 'Here is the result: {"claim": "x", "verdict": "SUPPORTED"}'
    parsed = BaseAgent.parse_json(raw)
    assert parsed["verdict"] == "SUPPORTED"


@patch("agents.web_scout.web_search")
@patch("agents.web_scout.fetch_page_text")
def test_web_scout_execute_calls_search_and_extracts_claims(mock_fetch, mock_search):
    mock_search.return_value = [
        {"title": "Doc A", "url": "http://a.com", "snippet": "Fact snippet A"},
        {"title": "Doc B", "url": "http://b.com", "snippet": "Fact snippet B"},
    ]
    mock_fetch.return_value = "Expanded page text with more detail."
    fake_llm = FakeLLM('["Claim 1 extracted", "Claim 2 extracted"]')

    agent = WebScout(llm=fake_llm)
    result = agent.execute("test sub-task", max_results=2, fetch_top=1)

    mock_search.assert_called_once_with("test sub-task", max_results=2)
    assert result["sub_task"] == "test sub-task"
    assert result["claims"] == ["Claim 1 extracted", "Claim 2 extracted"]
    assert len(result["sources"]) == 2
