"""
Make Outbound Plivo Call
========================
Usage:
    python scripts/make_call.py --to +91XXXXXXXXXX --ngrok https://xxxx.ngrok-free.app --language hi
    python scripts/make_call.py --to +91XXXXXXXXXX --ngrok https://xxxx.ngrok-free.app --language hi --port 8001
"""

import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))


def make_call(from_number, to_number, answer_url, auth_id, auth_token):
    resp = httpx.post(
        f"https://api.plivo.com/v1/Account/{auth_id}/Call/",
        auth=(auth_id, auth_token),
        json={
            "from": from_number,
            "to": to_number,
            "answer_url": answer_url,
            "answer_method": "POST",
        },
    )
    return resp.status_code, resp.json()


def get_plivo_numbers(auth_id, auth_token):
    resp = httpx.get(
        f"https://api.plivo.com/v1/Account/{auth_id}/Number/",
        auth=(auth_id, auth_token),
        params={"limit": 5},
    )
    if resp.status_code == 200:
        return [n["number"] for n in resp.json().get("objects", [])]
    return None


def main():
    parser = argparse.ArgumentParser(description="Make outbound Plivo call")
    parser.add_argument("--to", required=True, help="Destination phone number")
    parser.add_argument("--from-number", help="Plivo number (auto-detected)")
    parser.add_argument("--ngrok", required=True, help="Ngrok HTTPS URL")
    parser.add_argument("--language", "-l", default="hi")
    parser.add_argument("--port", type=int, default=8000, help="Server port (8000=cascaded, 8001=native)")
    args = parser.parse_args()

    auth_id = os.getenv("PLIVO_AUTH_ID", "")
    auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
    if not auth_id or not auth_token:
        print("ERROR: PLIVO_AUTH_ID/PLIVO_AUTH_TOKEN not set")
        sys.exit(1)

    from_number = args.from_number
    if not from_number:
        numbers = get_plivo_numbers(auth_id, auth_token)
        if numbers:
            from_number = numbers[0]
            print(f"Using: {from_number}")
        else:
            print("ERROR: No Plivo numbers found")
            sys.exit(1)

    answer_url = f"{args.ngrok}/?language={args.language}"

    print(f"Calling {args.to} from {from_number}")
    print(f"Answer URL: {answer_url}")
    print(f"Mode: {'cascaded' if args.port == 8000 else 'native'}")

    status, resp = make_call(from_number, args.to, answer_url, auth_id, auth_token)
    if status in (200, 201):
        print(f"Call fired! UUID: {resp.get('request_uuid', '?')}")
    else:
        print(f"Failed: {status} {json.dumps(resp)}")


if __name__ == "__main__":
    main()
