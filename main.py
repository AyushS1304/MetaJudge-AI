from __future__ import annotations

try:
    from pipeline import main as pipeline_main
except ModuleNotFoundError as exc:
    missing_dependency = exc.name

    def pipeline_main() -> int:
        print(
            f"Missing dependency: {missing_dependency}. "
            "Install the project requirements or run the app with .\\.venv\\Scripts\\python.exe."
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(pipeline_main())
