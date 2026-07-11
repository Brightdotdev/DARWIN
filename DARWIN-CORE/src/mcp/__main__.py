from src.mcp.Server import parser, run_stdio, run_sse

if __name__ == "__main__":
    args = parser().parse_args()
    if args.sse:
        run_sse(host=args.host, port=args.port)
    else:
        run_stdio()
