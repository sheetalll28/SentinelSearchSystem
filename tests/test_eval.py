import sys, os, shutil
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eval.judge import judge_report, EvalResult, save_results


class FakeJudgeLLM:
    def __init__(self, response):
        self.response = response

    def chat(self, system, user, max_tokens=500):
        return self.response


def test_judge_report_parses_clean_json():
    fake = FakeJudgeLLM('{"factual_accuracy": 4, "completeness": 5, "readability": 4, "justification": "good"}')
    scores = judge_report("q", "# report", judge_llm=fake)
    assert scores["factual_accuracy"] == 4
    assert scores["completeness"] == 5


def test_judge_report_handles_unparseable_output_gracefully():
    fake = FakeJudgeLLM("I refuse to output JSON today")
    scores = judge_report("q", "# report", judge_llm=fake)
    assert scores["factual_accuracy"] is None
    assert "unparseable" in scores["justification"]


def test_save_results_writes_json_and_csv_and_computes_averages():
    results = [
        EvalResult("q1", 4, 5, 4, "good", 500, 12.3),
        EvalResult("q2", 2, 3, 3, "weak", 400, 9.1),
    ]
    out_dir = "./eval/_test_results_tmp"
    try:
        json_path, csv_path = save_results(results, out_dir=out_dir)
        assert os.path.exists(json_path)
        assert os.path.exists(csv_path)
    finally:
        shutil.rmtree(out_dir, ignore_errors=True)
