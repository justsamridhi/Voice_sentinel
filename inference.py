import argparse
import logging
import sys

from src.utils.logging import setup_logging
from src.inference.pipeline import InferencePipeline

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceSentinel Audio Deepfake Inference CLI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--audio", type=str, required=True, help="Path to target audio file (.wav, .flac)")
    parser.add_argument("--device", type=str, default="cpu", help="Computing device (cpu or cuda)")
    args = parser.parse_args()

    # Configure logging to console only
    setup_logging()

    try:
        pipeline = InferencePipeline(checkpoint_path=args.checkpoint, device=args.device)
        result = pipeline.predict(args.audio)
    except Exception as e:
        logger.error(f"Inference execution failed: {e}")
        sys.exit(1)

    logger.info("=== Prediction Results ===")
    logger.info(f"Target Audio : {args.audio}")
    logger.info(f"Prediction   : {result['label'].upper()}")
    logger.info(f"Confidence   : {result['confidence'] * 100:.2f}%")
    logger.info(f"Spoof Prob   : {result['spoof_probability'] * 100:.2f}%")
    logger.info(f"Bonafide Prob: {result['bonafide_probability'] * 100:.2f}%")


if __name__ == "__main__":
    main()
