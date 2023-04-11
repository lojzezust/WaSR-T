import argparse
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode, functional as TF
import cv2
import time

from wasr_t.data.transforms import PytorchHubNormalization
from wasr_t.mobile_wasr_t import  wasr_temporal_lraspp_mobilenetv3
from wasr_t.utils import load_weights

width = 256
height = 192
fps = int(30)

# Colors corresponding to each segmentation class
SEGMENTATION_COLORS = np.array([
    [247, 195, 37],
    [41, 167, 224],
    [90, 75, 164]
], np.uint8)

HIST_LEN = 5

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="WaSR Network Sequential Inference")
    parser.add_argument("--hist-len", default=HIST_LEN, type=int,
                        help="Number of past frames to be considered in addition to the target frame (context length). Must match the value used in training.")
    parser.add_argument("--weights", type=str, required=True,
                        help="Model weights file.")
    parser.add_argument("--fp16", action='store_true',
                        help="Use half precision for inference.")
    parser.add_argument("--gpus", default=-1,type=int,
                        help="Number of gpus (or GPU ids) used for training.")
    return parser.parse_args()

def get_gstream_input() -> cv2.VideoCapture:

    # pipeline from webcam
    pipeline = f"v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480,framerate={fps}/1 ! videoconvert ! videoscale ! video/x-raw,format=BGR,width={width},height={height} ! appsink drop=true"

    # pipeline from local video
    # pipeline = f"filesrc location=MaSTr1325/images/wasrt_mobilenetv3_input.webm ! matroskademux ! vp9dec ! videoconvert ! videoscale ! video/x-raw,format=BGR,width={width},height={height} ! appsink drop=true"

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap

def get_gstream_output() -> cv2.VideoWriter:
    # pipeline_s = "appsrc ! videoconvert ! autovideosink sync=false"
    pipeline_s = "appsrc ! videoconvert ! x264enc ! flvmux ! filesink location=out.flv"
    out = cv2.VideoWriter(pipeline_s,cv2.CAP_GSTREAMER, 0, fps, (width, height), True) 
    return out

def get_model(args):
    model = wasr_temporal_lraspp_mobilenetv3(pretrained=False, hist_len=args.hist_len, sequential=True)
    state_dict = load_weights(args.weights)

    # if PyTorch 2.0's torch.compile() function generated these weights, then we need to remove
    # the _orig_mod label from each parameter.
    state_dict = {key.replace("_orig_mod.", "") : value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.sequential()
    model = model.eval()

    if args.fp16:
        model = model.half()

    device = torch.device('cpu') if args.gpus == 0 else torch.device('cuda')
    model = model.to(device)
    model.device = device

    model.backbone = torch.jit.optimize_for_inference(torch.jit.script(model.backbone))
    model.decoder.arm1 = torch.jit.optimize_for_inference(torch.jit.script(model.decoder.arm1))
    model.decoder.arm2 = torch.jit.optimize_for_inference(torch.jit.script(model.decoder.arm2))
    model.decoder.ffm = torch.jit.optimize_for_inference(torch.jit.script(model.decoder.ffm))
    model.decoder.aspp = torch.jit.optimize_for_inference(torch.jit.script(model.decoder.aspp))

    return model

class Inferencer:

    def __init__(self, model):
        self.model = model

        if any(p.dtype is torch.float16 for p in self.model.parameters()):
            self.dtype = torch.float16
        else:
            self.dtype = torch.float32

    def process_frame(self, frame : np.ndarray):

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tf = PytorchHubNormalization()
        frame = tf(frame)

        frame = torch.Tensor(frame).to(self.model.device).to(self.dtype)
        frame = frame.unsqueeze(0)

        with torch.inference_mode():
            probs = self.model({'image': frame})['out']
        
        probs = TF.resize(probs, (height, width), interpolation=InterpolationMode.BILINEAR)
        out_class = probs.argmax(1).to(torch.uint8).squeeze().detach().cpu().numpy()
        pred_mask = SEGMENTATION_COLORS[out_class]
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_RGB2BGR)
        return pred_mask

def main():

    args = get_arguments()
    print(f"Got arguments: {args}")

    print("Initializing GStreamer input.")
    cap = get_gstream_input()

    print("Initializing GStreamer output.")
    out = get_gstream_output()

    print("Instantiating and compiling model.")
    model = get_model(args)
    inferencer = Inferencer(model)

    print("Beginning inference.")

    tic = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = inferencer.process_frame(frame)
            out.write(frame)
            toc = time.time()
            print(f"\rInstantaneous FPS {(1.0 / (toc - tic)) :.2f}.", end='')
            tic = toc
        cv2.waitKey(1)
    
    print("Video capture is closed.")

    # Release everything if job is finished
    cap.release()
    out.release()

if __name__ == '__main__':
    main()
