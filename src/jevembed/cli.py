import argparse
import json
from pathlib import Path
import sys

from .client import JevEmbed
from .config import ModelConfig
from .errors import JevEmbedError


def main(argv=None):
    parser = argparse.ArgumentParser(description="JevEmbed local embedding decisions")
    parser.add_argument("--config", action="append", required=True, help="YAML config; repeat to register multiple models")
    parser.add_argument("--input", default="-", help="Request JSON or - for stdin")
    parser.add_argument("--output", default="-", help="Result JSON or - for stdout")
    parser.add_argument("--explain", action="store_true", help="Compile only, without loading weights")
    parser.add_argument("--trace", action="store_true", help="Return response and diagnostic trace")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    try:
        client = JevEmbed()
        for path in args.config:
            client.register(ModelConfig.load(path))
        if args.serve:
            import uvicorn
            from .server import create_app
            uvicorn.run(create_app(client), host=args.host, port=args.port)
            return 0
        request = json.loads(sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8"))
        result = client.explain(request) if args.explain else (
            client.evaluate_with_trace(request) if args.trace else client.evaluate(request))
        output = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output == "-":
            sys.stdout.write(output)
        else:
            Path(args.output).write_text(output, encoding="utf-8")
        return 0
    except (JevEmbedError, ValueError, OSError, ImportError) as exc:
        print(f"jevembed: {exc}", file=sys.stderr)
        return 2
