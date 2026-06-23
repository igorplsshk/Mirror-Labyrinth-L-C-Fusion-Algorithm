"""
run_analysis.py — L-C Fusion Algorithm entry point
====================================================
Reproduces the fusion dynamics plots from the Mirror Labyrinth paper
using the empirical data stored in data/.

Usage
-----
    # Plot Gemini case study from stored empirical data (no GPU required):
    python run_analysis.py --case gemini

    # Plot DeepSeek case study:
    python run_analysis.py --case deepseek

    # Run live analysis on your own dialogue logs (requires GPU recommended):
    python run_analysis.py --case live --user-log path/to/user.txt --model-log path/to/model.txt

    # Save to file instead of displaying:
    python run_analysis.py --case gemini --save output/gemini_fusion.png
"""

import argparse
import sys
from pathlib import Path


def run_empirical(case: str, save_path: str | None) -> None:
    """Reproduce paper figures from stored empirical data."""
    from lc_fusion.visualize import plot_fusion_dynamics

    if case == "gemini":
        from data.gemini_case_study import GEMINI_ANOMALOUS, GEMINI_STAGES  # type: ignore

        similarity = [row[1] for row in GEMINI_ANOMALOUS]
        relevance  = [row[2] for row in GEMINI_ANOMALOUS]

        plot_fusion_dynamics(
            similarity_dynamics=similarity,
            relevance_dynamics=relevance,
            model_name="Gemini 2.5 Pro",
            stages=GEMINI_STAGES,
            save_path=save_path,
            show=save_path is None,
        )
        print("Gemini 2.5 Pro — peak similarity:", max(similarity))

    elif case == "deepseek":
        from data.deepseek_case_study import DEEPSEEK_ANOMALOUS, DEEPSEEK_STAGES

        similarity = [row[1] for row in DEEPSEEK_ANOMALOUS]

        plot_fusion_dynamics(
            similarity_dynamics=similarity,
            model_name="DeepSeek-V3.2 Exp",
            stages=DEEPSEEK_STAGES,
            save_path=save_path,
            show=save_path is None,
        )
        print("DeepSeek-V3.2 — peak similarity:", max(similarity))

    else:
        print(f"Unknown case '{case}'. Choose 'gemini' or 'deepseek'.", file=sys.stderr)
        sys.exit(1)


def run_live(user_log: str, model_log: str, save_path: str | None) -> None:
    """Run the full algorithm on raw dialogue text files (one message per line)."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    from lc_fusion import analyze_dialogue_fusion
    from lc_fusion.visualize import plot_fusion_dynamics

    user_msgs  = Path(user_log).read_text(encoding="utf-8").strip().splitlines()
    model_msgs = Path(model_log).read_text(encoding="utf-8").strip().splitlines()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "intfloat/multilingual-e5-large"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    scores = analyze_dialogue_fusion(user_msgs, model_msgs, tokenizer, model, device=device)

    plot_fusion_dynamics(
        similarity_dynamics=scores,
        model_name="Live Analysis",
        save_path=save_path,
        show=save_path is None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="L-C Fusion Algorithm — Mirror Labyrinth analysis runner"
    )
    parser.add_argument(
        "--case",
        choices=["gemini", "deepseek", "live"],
        default="gemini",
        help="Which analysis to run (default: gemini)",
    )
    parser.add_argument(
        "--user-log",
        default=None,
        help="Path to user messages file (one message per line). Required for --case live.",
    )
    parser.add_argument(
        "--model-log",
        default=None,
        help="Path to model responses file (one message per line). Required for --case live.",
    )
    parser.add_argument(
        "--save",
        default=None,
        metavar="PATH",
        help="Save figure to this path instead of displaying it.",
    )

    args = parser.parse_args()

    if args.case == "live":
        if not args.user_log or not args.model_log:
            parser.error("--case live requires --user-log and --model-log")
        run_live(args.user_log, args.model_log, args.save)
    else:
        run_empirical(args.case, args.save)


if __name__ == "__main__":
    main()
