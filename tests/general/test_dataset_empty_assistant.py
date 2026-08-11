from datasets import Dataset

from swift.dataset.loader import DatasetLoader


def test_empty_assistant_filter_is_default_off():
    dataset = Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]},
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
        ]
    )

    filtered = DatasetLoader()._filter_empty_assistant_responses(dataset)

    assert len(filtered) == 2


def test_empty_assistant_filter_drops_only_blank_assistant_content():
    dataset = Dataset.from_list(
        [
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": ""}]},
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "  "}]},
            {"messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": "a"}]},
        ]
    )

    filtered = DatasetLoader(drop_empty_assistant_response=True)._filter_empty_assistant_responses(dataset)

    assert len(filtered) == 1
    assert filtered[0]["messages"][-1]["content"] == "a"
