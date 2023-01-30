import argparse
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from wasr_t.data.folder import FolderDataset
from wasr_t.data.transforms import PytorchHubNormalization
from wasr_t.inference import Predictor
from wasr_t.wasr_t import wasr_temporal_resnet101
from wasr_t.utils import load_weights, Option

# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

OUTPUT_DIR = 'output/predictions'
HIST_LEN = 5
RESIZE = (512,384)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network Sequential Inference")
    parser.add_argument("--sequence-dir", type=str, required=False,
                        help="Path to the directory containing frames of the input sequence.")
    parser.add_argument("--hist-len", default=HIST_LEN, type=int,
                        help="Number of past frames to be considered in addition to the target frame (context length). Must match the value used in training.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Model weights file.")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Directory where the predictions will be stored.")
    parser.add_argument("--resize", type=Option(int), default=RESIZE, nargs='+',
                        help="Resize input images to a specified size. Use `none` for no resizing.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--gpus", default=-1,
                        help="Number of gpus (or GPU ids) used for training.")
    return parser.parse_args()

def export_predictions(probs, batch, output_dir):
    features, metadata = batch

    # Class prediction
    out_class = probs.argmax(1).astype(np.uint8)

    for i, pred_mask in enumerate(out_class):
        pred_mask = SEGMENTATION_COLORS[pred_mask]
        mask_img = Image.fromarray(pred_mask)

        out_path = output_dir / Path(metadata['image_path'][i]).with_suffix('.png')
        if not out_path.parent.exists():
            out_path.parent.mkdir(parents=True, exist_ok=True)

        mask_img.save(str(out_path))

def predict_sequence(predictor, sequence_dir, output_dir, size):
    """Runs inference on a sequence of images. The frames are processed sequentially (stateful). The state is cleared at the start of the sequence."""
    predictor.model.clear_state()

    dataset = FolderDataset(sequence_dir, normalize_t=PytorchHubNormalization(), resize=size)
    dl = DataLoader(dataset, batch_size=1, num_workers=1) # NOTE: Batch size must be 1 in sequential mode.

    for batch in tqdm(dl, desc='Processing frames'):
        features, metadata = batch
        probs = predictor.predict_batch(features)
        export_predictions(probs, batch, output_dir=output_dir)


def run_inference(args):
    model = wasr_temporal_resnet101(pretrained=False, hist_len=args.hist_len)
    state_dict = load_weights(args.weights)
    model.load_state_dict(state_dict)
    model = model.sequential() # Enable sequential mode

    predictor = Predictor(model, half_precision=args.fp16)
    output_dir = Path(args.output_dir)

    size = None
    if args.resize[0] is not None:
        size = args.resize

    predict_sequence(predictor, args.sequence_dir, output_dir, size=size)

def main():
    args = get_arguments()
    print(args)

    run_inference(args)


if __name__ == '__main__':
    main()
