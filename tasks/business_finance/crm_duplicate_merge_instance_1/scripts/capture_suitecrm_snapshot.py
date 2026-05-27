#!/usr/bin/env python
"""Capture the live SuiteCRM state needed for benchmark scoring via V4.1 REST API."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

REST_ENDPOINT = "/service/v4_1/rest.php"


def _rest_call(
    base_url: str, method: str, rest_data: dict[str, Any]
) -> dict[str, Any]:
    data = urllib.parse.urlencode(
        {
            "method": method,
            "input_type": "JSON",
            "response_type": "JSON",
            "rest_data": json.dumps(rest_data),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{REST_ENDPOINT}",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_expected(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _normalise_name(first_name: Any, last_name: Any) -> str:
    return f"{str(first_name or '').strip().lower()} {str(last_name or '').strip().lower()}".strip()


def _escape_sql(value: str) -> str:
    return value.replace("'", "''")


def _login(base_url: str, username: str, password: str) -> str:
    md5_password = hashlib.md5(password.encode("utf-8")).hexdigest()
    result = _rest_call(
        base_url,
        "login",
        {
            "user_auth": {"user_name": username, "password": md5_password},
            "application_name": "agenthle_eval",
        },
    )
    session_id = result.get("id")
    if not session_id:
        raise RuntimeError("V4.1 login failed: " + json.dumps(result))
    return str(session_id)


def _search_contacts_by_name(
    base_url: str,
    session: str,
    first_name: str,
    last_name: str,
) -> tuple[list[dict[str, Any]], int]:
    query = (
        f"contacts.first_name='{_escape_sql(first_name)}' "
        f"AND contacts.last_name='{_escape_sql(last_name)}' "
        f"AND contacts.deleted=0"
    )
    select_fields = [
        "first_name",
        "last_name",
        "email1",
        "phone_work",
        "phone_mobile",
        "primary_address_street",
        "primary_address_city",
        "primary_address_postalcode",
        "primary_address_country",
        "lead_source",
        "email_opt_out",
    ]
    result = _rest_call(
        base_url,
        "get_entry_list",
        {
            "session": session,
            "module_name": "Contacts",
            "query": query,
            "order_by": "",
            "offset": 0,
            "select_fields": select_fields,
            "link_name_to_fields_array": [],
            "max_results": 10,
            "deleted": 0,
        },
    )
    entries = result.get("entry_list", [])
    contacts = []
    for entry in entries:
        nvl = entry.get("name_value_list", {})
        contact: dict[str, Any] = {}
        for field_name, field_data in nvl.items():
            if isinstance(field_data, dict):
                contact[field_name] = field_data.get("value")
            else:
                contact[field_name] = field_data
        contacts.append(contact)
    return contacts, len(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--expected", required=True)
    args = parser.parse_args()

    payload: dict[str, Any] = {
        "ok": False,
        "reason": "",
        "source": "suitecrm_rest_v4_1_snapshot",
        "contacts": [],
        "count_by_name": {},
    }

    try:
        expected_payload = _load_expected(args.expected)
        base_url = args.url.rstrip("/")
        session = _login(base_url, args.user, args.password)

        for pair_id, expected_entry in expected_payload.items():
            if pair_id.startswith("_") or not isinstance(expected_entry, dict):
                continue
            expected_fields = expected_entry.get("fields", {})
            if not isinstance(expected_fields, dict):
                continue
            first_name = str(expected_fields.get("first_name", ""))
            last_name = str(expected_fields.get("last_name", ""))
            contacts, count = _search_contacts_by_name(
                base_url, session, first_name, last_name
            )
            if contacts:
                payload["contacts"].append(contacts[0])
            payload["count_by_name"][_normalise_name(first_name, last_name)] = count

        payload["ok"] = True
        payload["reason"] = "ok"
    except Exception as exc:
        payload["reason"] = f"{type(exc).__name__}:{exc}"

    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
