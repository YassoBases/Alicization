"""Review CLI: python -m review <command> [...] --run-dir runs/<id>

Commands:
  list                       pending queue (filter: --status --type)
  show <id>                  full proposal + evidence refs (source BLINDED
                             until status=evaluated)
  approve <id> [--note]      -> status=approved + experiment ticket emitted
  reject <id> [--note]
  postpone <id> [--note]
  partial <id> [--note]      -> partially_approved + ticket
  modify <id>                step 1: writes <id>.edit.json for hand-editing
  modify <id> --apply        step 2: records the human diff, applies
  rate <id> <1-5> [--note]   usefulness rating

Blind review: `list`/`show` never display source before evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys

from review.queue import ReviewQueue, blind_view


def _print_row(view: dict) -> None:
    print(f"{view['id']}  [{view['status']:>18}]  {view['type']:<18} "
          f"conf={view['confidence']:.2f}  tick={view['created_tick']}  "
          f"src={view['source']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m review", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list")
    p_list.add_argument("--status", default=None)
    p_list.add_argument("--type", dest="ptype", default=None)

    for name in ("show", "approve", "reject", "postpone", "partial"):
        p = sub.add_parser(name)
        p.add_argument("id")
        if name != "show":
            p.add_argument("--note", default="")

    p_mod = sub.add_parser("modify")
    p_mod.add_argument("id")
    p_mod.add_argument("--apply", action="store_true")

    p_rate = sub.add_parser("rate")
    p_rate.add_argument("id")
    p_rate.add_argument("rating", type=int)
    p_rate.add_argument("--note", default="")

    args = parser.parse_args(argv)
    queue = ReviewQueue(args.run_dir)

    if args.command == "list":
        rows = queue.proposals(status=args.status, ptype=args.ptype)
        if not rows:
            print("(queue empty)")
        for p in rows:
            _print_row(blind_view(p))
    elif args.command == "show":
        print(json.dumps(blind_view(queue.get(args.id)), indent=2))
    elif args.command in ("approve", "reject", "postpone", "partial"):
        p = queue.decide(args.id, args.command, note=args.note)
        print(f"{p.id} -> {p.status}")
        if p.status in ("approved", "partially_approved"):
            print(f"ticket: experiments/tickets/{p.id}.md  (execute it BY HAND)")
    elif args.command == "modify":
        if args.apply:
            p = queue.modify_apply(args.id)
            print(f"{p.id} -> modified (human diff recorded)")
        else:
            path = queue.modify_start(args.id)
            print(f"edit {path} with your editor, then: modify {args.id} --apply")
    elif args.command == "rate":
        p = queue.rate(args.id, args.rating, note=args.note)
        print(f"{p.id} rated {args.rating}/5")
    return 0


if __name__ == "__main__":
    sys.exit(main())
