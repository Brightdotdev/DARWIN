from src.mcp.Server import _serve_stdio, _serve_sse, _init_ok, parser

if __name__ == "__main__":
    args = parser.parse_args()
    if not _init_ok:
        import sys

        sys.exit(1)
    if args.sse:
        _serve_sse(args.port)
    else:
        _serve_stdio()
