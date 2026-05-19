"""Regression checks for path/query splitting in parse_http_string."""

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import parse_http_string


def test_path_with_query_splits():
    row = parse_http_string("/api/workspaces/insky/user-favorites/?all=true")
    assert row["path"] == "/api/workspaces/insky/user-favorites/"
    assert row["query"] == "all=true"


def test_bare_query_stays_query():
    row = parse_http_string("id=1&name=test")
    assert row["path"] == ""
    assert row["query"] == "id=1&name=test"


def test_path_only():
    row = parse_http_string("/api/workspaces/insky/modules/")
    assert row["path"] == "/api/workspaces/insky/modules/"
    assert row["query"] == ""


if __name__ == "__main__":
    test_path_with_query_splits()
    test_bare_query_stays_query()
    test_path_only()
    print("ok")
